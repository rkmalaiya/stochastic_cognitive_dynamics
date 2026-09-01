import os
import gc
import socket
import json
from datetime import datetime
# Original fixed-GPU selection retained for reference. SLURM sets
# CUDA_VISIBLE_DEVICES separately for every GPU task before Python starts.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0" "1"

#from pyexpat import model
from attr import dataclass
#from dataclasses import dataclass, fields
import cme.decision_models.confidence_accumulation as ca
import cme.decision_models.diffusion_discrete as dd
import cme.decision_models.quantum_discrete as qd
import cme.utils.post_process_model as ppm
import jax.numpy as npx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import arviz as az
import numpy as np
import pickle
import time
import itertools as iter
from joblib import Parallel, delayed
from cme.utils import common_logging as cl
log = cl.get_logger("fit_model")
import jax


def _get_slurm_process_partition():
    """Return this process's rank and the process count for an srun step."""
    process_id = os.environ.get("SLURM_PROCID")

    # A normal local run, or a batch script that did not use srun, is one
    # process even when the surrounding allocation contains several tasks.
    if process_id is None:
        return 0, 1

    process_count = os.environ.get(
        "SLURM_STEP_NUM_TASKS",
        os.environ.get("SLURM_NTASKS", "1"),
    )

    try:
        process_id = int(process_id)
        process_count = int(process_count)
    except ValueError:
        log.warning(
            "Invalid SLURM process information: SLURM_PROCID=%r, "
            "process count=%r. Running as one process.",
            process_id,
            process_count,
        )
        return 0, 1

    if process_count < 1 or not 0 <= process_id < process_count:
        log.warning(
            "Inconsistent SLURM process information: rank=%s, count=%s. "
            "Running as one process.",
            process_id,
            process_count,
        )
        return 0, 1

    return process_id, process_count


def _partition_work_for_slurm(work, process_id, process_count):
    """Assign independent work items to this SLURM process."""
    return work[process_id::process_count]


def _read_text_if_available(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except (OSError, UnicodeError):
        return None


def _format_kib(value):
    try:
        return f"{int(value.split()[0]) / (1024 ** 2):.2f} GiB"
    except (AttributeError, TypeError, ValueError):
        return "unavailable"


def _format_bytes(value):
    if value in (None, "", "max"):
        return value or "unavailable"
    try:
        return f"{int(value) / (1024 ** 3):.2f} GiB"
    except ValueError:
        return "unavailable"


def _get_memory_diagnostics():
    details = []

    status = _read_text_if_available("/proc/self/status")
    if status:
        status_values = {}
        for line in status.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                status_values[key] = value.strip()

        details.extend([
            f"process_rss={_format_kib(status_values.get('VmRSS'))}",
            f"process_peak_rss={_format_kib(status_values.get('VmHWM'))}",
            f"process_virtual={_format_kib(status_values.get('VmSize'))}",
            f"process_peak_virtual={_format_kib(status_values.get('VmPeak'))}",
            f"process_threads={status_values.get('Threads', 'unavailable')}",
        ])

    meminfo = _read_text_if_available("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                details.append(
                    f"node_memory_available={_format_kib(line.partition(':')[2].strip())}"
                )
                break

    cgroup = _read_text_if_available("/proc/self/cgroup")
    if cgroup:
        cgroup_base = None
        current_name = None
        limit_name = None

        for line in cgroup.splitlines():
            hierarchy, controllers, relative_path = line.split(":", 2)
            if hierarchy == "0":  # cgroup v2
                cgroup_base = os.path.join(
                    "/sys/fs/cgroup", relative_path.lstrip("/")
                )
                current_name = "memory.current"
                limit_name = "memory.max"
                break
            if "memory" in controllers.split(","):  # cgroup v1
                cgroup_base = os.path.join(
                    "/sys/fs/cgroup/memory", relative_path.lstrip("/")
                )
                current_name = "memory.usage_in_bytes"
                limit_name = "memory.limit_in_bytes"
                break

        if cgroup_base:
            current = _read_text_if_available(
                os.path.join(cgroup_base, current_name)
            )
            limit = _read_text_if_available(
                os.path.join(cgroup_base, limit_name)
            )
            details.append(
                f"job_cgroup_memory={_format_bytes(current)}/{_format_bytes(limit)}"
            )

    return ", ".join(details) if details else "memory diagnostics unavailable"


#file_loc_X= "data/ad_X_"
#file_loc_RT= "data/ad_rt_"



#n_states, start_width, delta, measurement_prob = 11, 4, 0.1, 0.8 
#n_states, start_width, delta, measurement_prob = 11, 5, 0.01, 0.8
#n_states, start_width, delta, measurement_prob = 11, 5, 0.001, 0.8 
#n_states, start_width, delta, measurement_prob = 51, 25, 0.01, 0.8
#folder, file_pre, file_posts, version, n_states, start_width, delta, measurement_prob, params_type, model_type, transition_type, likelihood_type, sampling_type = "data", "ad", ["init_close_HPE", "init_far_HPE", "init_close_LPE", "init_far_LPE"], 11, 5, 0.01, 0.8, "Centralized", "Quantum", "TIMESTEP", "SINGLE", "GEN" 

@dataclass
class ModelDetails:

    folder:str = "data"
    file_pre:str = ""
    data:dict = {} #key,(RT,X) 
    version:float = 0.1
    n_states:int = 11
    start_width:int = None # None value will be automatically calculated.
    response_width:int = 1
    delta:float = 1
    measurement_prob:float = 0.2
    num_warmup: int = 20 
    samples_n: int = 50
    num_chains: int = 4
    max_tree_depth: int = 10
    predictive_n: int = None
    batch_size: int = None
    params_type:str = "Centralized|NonCentralized"
    model_type:list = ["Markov","Quantum"]
    transition_type:str = "RT|TIMESTEP"
    likelihood_type:str = "SINGLE" #|JOINT
    sampling_type:str = "GEN"
    estimation_type:str = "MCMC|VI"
    execution_type:str = "Both" #Posterior|Predictive|Both
    scale: str = None #"None|Log|SQRT"
    conf_scale: list = [None,None] #"None|(add_scale, mul_scale)"
    csv_header:bool = False
    is_test:bool = False
    is_parallel:bool=False


_FIT_CONFIGURATION_COLUMNS = [
    "data",
    "file_pre",
    "version",
    "model_type",
    "n_states",
    "start_width",
    "response_width",
    "delta",
    "measurement_prob",
    "params_type",
    "transition_type",
    "likelihood_type",
    "sampling_type",
    "estimation_type",
    "execution_type",
    "num_warmup",
    "samples_n",
    "num_chains",
    "max_tree_depth",
    "predictive_n",
    "batch_size",
]

def _configuration_value(value):
    if isinstance(value, dict):
        value = list(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, default=str)
    if value is None:
        return ""
    return str(value)


def _write_fit_configuration_csv(model):
    process_id, _ = _get_slurm_process_partition()
    if process_id != 0:
        return

    configuration = {
        column: _configuration_value(getattr(model, column))
        for column in _FIT_CONFIGURATION_COLUMNS
    }

    os.makedirs("export", exist_ok=True)
    csv_path = os.path.join("export", "fit_model_configurations.csv")
    columns = ["created_date", *_FIT_CONFIGURATION_COLUMNS]

    if os.path.exists(csv_path):
        configurations = pd.read_csv(
            csv_path, dtype=str, keep_default_na=False
        )
        matching = pd.Series(True, index=configurations.index)
        for column, value in configuration.items():
            matching &= configurations[column].eq(value)

        if matching.any():
            return
    else:
        configurations = pd.DataFrame(columns=columns)

    configuration["created_date"] = datetime.now().isoformat(
        timespec="seconds"
    )
    configurations = pd.concat(
        [configurations, pd.DataFrame([configuration])],
        ignore_index=True,
    )
    configurations[columns].to_csv(csv_path, index=False)


def _add_prior_to_arviz_data(arviz_data, prior_samples, prior_pd_samples, RT, coords, dims):
    prior_rt = np.asarray([samples["Samples"]["RT"].values for samples in prior_pd_samples]) # predictive_n x I x J when reshaped below
    prior_rt = prior_rt.reshape((-1,*RT.shape)) # predictive_n x I x J
    prior_samples_arviz = {key:np.asarray(samples)[None,...] for key, samples in prior_samples.items()} # 1 x predictive_n x ... (chain x draw x ...)
    # Previous version-specific construction retained for reference:
    # prior_idata = az.from_dict(prior=prior_samples_arviz,
    #                            prior_predictive={"RT":prior_rt[None,...]}, # 1 x predictive_n x I x J (chain x draw x participant x trial)
    #                            coords=coords, dims=dims)
    # arviz_data["prior"] = prior_idata["prior"]
    # arviz_data["prior_predictive"] = prior_idata["prior_predictive"]

    # Previous calls with implicit sample dimensions retained for reference:
    # arviz_data["prior"] = az.dict_to_dataset(prior_samples_arviz, coords=coords, dims=dims) # 1 x predictive_n x ... (chain x draw x ...)
    # arviz_data["prior_predictive"] = az.dict_to_dataset({"RT":prior_rt[None,...]}, coords=coords, dims=dims) # 1 x predictive_n x I x J (chain x draw x participant x trial)
    arviz_data["prior"] = az.dict_to_dataset(prior_samples_arviz, sample_dims=["chain", "draw"], coords=coords, dims=dims) # 1 x predictive_n x ... (chain x draw x ...)
    arviz_data["prior_predictive"] = az.dict_to_dataset({"RT":prior_rt[None,...]}, sample_dims=["chain", "draw"], coords=coords, dims=dims) # 1 x predictive_n x I x J (chain x draw x participant x trial)
    return arviz_data



#folder, file_pre, file_posts, version, n_states, start_width, delta, measurement_prob, params_type, model_type, transition_type, likelihood_type, sampling_type
def fit_model(model: ModelDetails):
    process_id, process_count = _get_slurm_process_partition()
    _write_fit_configuration_csv(model)
    compute_backend = jax.default_backend()
    compute_devices = jax.devices()
    # Original compute-device logging retained for reference:
    # log.info(f"Compute devices: {jax.default_backend()}, {jax.devices()}")
    log.info(
        "Compute process %s/%s: backend=%s, devices=%s, "
        "CUDA_VISIBLE_DEVICES=%s",
        process_id,
        process_count,
        compute_backend,
        compute_devices,
        os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
    )
    file_loc = f"{model.folder}/{model.file_pre}"
    
    #file_post = 
    #version = 0.5
    #len(model.file_posts)
    model_jobs = list(iter.product(model.data, zip(model.model_type, model.n_states, model.response_width, model.conf_scale)))

    if model.estimation_type == "VI" and process_count > 1:
        assigned_model_jobs = _partition_work_for_slurm(model_jobs, process_id, process_count)
    else:
        assigned_model_jobs = model_jobs

    # Original dataset/model-level parallelism retained for reference:
    # n_jobs = min(3, len(model.data) + len(model.model_type)) if not model.is_test and model.is_parallel and jax.default_backend() != "gpu" else 1
    if (
        model.estimation_type == "VI"
        # Original single-process check retained for reference:
        # and process_count == 1
        and os.environ.get("SLURM_PROCID") is None
        and not model.is_test
        and model.is_parallel
        and compute_backend != "gpu"
    ):
        n_jobs = max(1, min(3, len(assigned_model_jobs)))
    else:
        n_jobs = 1

    # Original dataset/model job logging retained for reference:
    # log.info(f"Received request for {n_jobs} files to be executed in parallel for {model.model_type}_version:{model.version}_states:{model.n_states}_resp_width:{model.response_width}!!")
    log.info(
        "Process %s/%s received %s of %s dataset/model jobs; running %s "
        "at a time for estimation type %s and model(s) %s_version:%s_"
        "states:%s_resp_width:%s",
        process_id,
        process_count,
        len(assigned_model_jobs),
        len(model_jobs),
        n_jobs,
        model.estimation_type,
        model.model_type,
        model.version,
        model.n_states,
        model.response_width,
    )
    
    Parallel(n_jobs=n_jobs, prefer="processes", backend = "loky")(delayed(_run_model)(
                                    
                                    file_loc, data, model.version, 
                                    n_states, model.start_width, response_width, model.delta, model.measurement_prob, 
                                    model.params_type, model_type, model.transition_type, model.likelihood_type, 
                                    model.sampling_type, model.estimation_type, model.execution_type,
                                    model.num_warmup, model.samples_n, model.num_chains, model.max_tree_depth,
                                    model.predictive_n, model.batch_size, model.is_test,
                                    model.scale, conf_scale, model.csv_header, model.is_parallel) 
                                                
                                    # Original unpartitioned dataset/model product retained for reference:
                                    # for data, (model_type, n_states, response_width, conf_scale) in iter.product(model.data, zip(model.model_type, model.n_states, model.response_width, model.conf_scale)))
                                    for data, (model_type, n_states, response_width, conf_scale) in assigned_model_jobs)
    # Original unconditional overall success message retained for reference:
    # log.info(f"All jobs successfully completed for {model.model_type}_{model.version}!!!!")
    log.info(f"All assigned dataset/model job loops completed for {model.model_type}_{model.version}!!!!")


def _run_model(file_loc, data, version, 
            n_states, start_width, response_width, delta, measurement_prob, 
            params_type, model_type, transition_type, likelihood_type, sampling_type, estimation_type,execution_type,
            num_warmup, samples_n, num_chains, max_tree_depth, predictive_n, batch_size, is_test, scale, conf_scale, csv_header, is_parallel):
    
    start_width1 = (n_states-2*response_width)//2
    if start_width == None or start_width == 0:
        start_width = start_width1
    elif start_width > start_width1:
        raise Warning(f"start_width larger than ideal value of {start_width}. {model_type} model may have unexpected results")
    
    #if model_type == "Quantum":
    #    start_width = start_width//2

    if isinstance(data, str):
        RT_file = f"{file_loc}{data}_rt.csv"
        X_file = f"{file_loc}{data}_ra.csv"
        name = data
        
        df_X = pd.read_csv(X_file, header="infer" if csv_header else None)
        df_RT = pd.read_csv(RT_file, header="infer" if csv_header else None)
    else:
        name, (RT, X) = data
        df_X = pd.DataFrame(X)
        df_RT = pd.DataFrame(RT)


    if "id" in df_X.columns:
        df_ID = df_X[["id"]]
    else:
        df_ID = df_X.index

    df_X = df_X.drop("id", axis=1) if "id" in df_X.columns else df_X 
    df_RT = df_RT.drop("id", axis=1) if "id" in df_RT.columns else df_RT

    df_X = df_X.drop("Unnamed: 0", axis=1).dropna() if "Unnamed: 0" in df_X.columns else df_X 
    df_RT = df_RT.drop("Unnamed: 0", axis=1).dropna() if "Unnamed: 0" in df_RT.columns else df_RT 

    Xs = df_X.values
    RTs = df_RT.values
    IDs = df_ID.values
    
    log.info(f"Received participant size: {RTs.shape} for data {data if isinstance(data, str) else data[0]} and {model_type}")

    if scale == "Log":
        RTs = np.log(RTs)
    elif scale=="SQRT":
        RTs = np.sqrt(RTs)

    if predictive_n is None:
        predictive_n = RTs.shape[0]

    if batch_size is None:
        batch_size = RTs.shape[0]

    X_split = np.split(Xs, npx.arange(batch_size, Xs.shape[0], batch_size), axis=0)
    RT_split = np.split(RTs, npx.arange(batch_size, RTs.shape[0], batch_size), axis=0)
    ID_split = np.split(IDs, npx.arange(batch_size, IDs.shape[0], batch_size), axis=0)
    
    def run_half_model(i, X, RT, ID):
        
        #min_RT_sec = np.clip(RT.mean() - 3*RT.std(), a_min=delta, a_max=None)
        #max_RT_sec = RT.mean() + 3*RT.std()
        min_RT_sec = np.clip(RT.min(), a_min=delta*2, a_max=None)
        max_RT_sec = RT.max()
        log.info(f"Starting Prior Predictive Sampling_{name}_{model_type}_{version}_{i} for {min_RT_sec} to {max_RT_sec} secs")
        prior_samples, prior_pd_samples = ca.sample_prior_pred_params(n_states=n_states,start_width=start_width, response_width=response_width,
                                                        delta=delta, data_samples=RT.shape, min_RT_sec = min_RT_sec, max_RT_sec = max_RT_sec,
                                                        measurement_prob=measurement_prob, X=X, RT=None, n_samples=predictive_n,
                                                        params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                        likelihood_type=likelihood_type, sampling_type=sampling_type, 
                                                    )
        df_prior_pred_all = pd.concat([samples["Samples"] for samples in prior_pd_samples])
        df_prior_pred_all.to_csv(f"export/prior_predictive_{name}_{model_type}_{version}_{i}.csv")

        if execution_type == "Prior":
            return None

        start_time_sampling = time.perf_counter()
        log.info(f"Starting Posterior Sampling_{name}_{model_type}_v:{version}_n:{n_states}_s:{start_width}_r:{response_width}_{i}")
        
        if estimation_type == "MCMC":
            # Previous fixed four-chain call retained for reference:
            # post_chain = ca.sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, response_width=response_width,
            #                                         delta=delta,measurement_prob=measurement_prob,
            #                                         num_warmup=num_warmup, samples_n=samples_n,
            #                                         params_type=params_type, model_type=model_type, transition_type=transition_type,
            #                                         likelihood_type=likelihood_type, num_chains=4
            #                                         )

            post_chain = ca.sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, response_width=response_width,
                                                    delta=delta,measurement_prob=measurement_prob,
                                                    num_warmup=num_warmup, samples_n=samples_n,
                                                    params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                    likelihood_type=likelihood_type, num_chains=num_chains,
                                                    max_tree_depth=max_tree_depth
                                                    )
            
            post_samples = post_chain.get_samples()

            # post_samples_chain = post_chain.get_samples(group_by_chain=True)

            # for k in ["m", "s", "m_si", "s_si", "mu", "mu_r", "sigma", "sigma_r", "sigma_final"]:
            #     if k in post_samples_chain:
            #         print("******",k,": " ,post_samples_chain[k].shape)

            coords = {
                        "part_id": ID.reshape(-1),#.squeeze(),
                    }
            dims = {
                        "mu_r": ["part_id"],
                        "sigma_r": ["part_id"],
                        "mu": ["part_id"],
                        "sigma_final": ["part_id"],
                        "phi_0": ["part_id"],
                        "likl_rt": ["part_id"],
                        "phi_conc": ["part_id"],
                        "sigma": ["part_id"],
                        "phi_init": ["part_id"],
                        "likelihood": ["part_id"],
                        "RT":["part_id"]
                    }
            arviz_data = az.from_numpyro(post_chain,
                                        coords=coords,
                                        dims=dims, log_likelihood=True)
            arviz_data = _add_prior_to_arviz_data(arviz_data, prior_samples, prior_pd_samples, RT, coords, dims)
            
            # Previous call with implicit observed-data dimensions retained for reference:
            # obs_idata = az.from_dict({"observed_data": {"RT": RT}}, coords=coords, dims=dims)
            obs_idata = az.from_dict({"observed_data": {"RT": RT}}, sample_dims=[], coords=coords, dims=dims) # I x J (participant x trial)
            arviz_data["observed_data"] = obs_idata["observed_data"]
            # Previous display-oriented summary call retained for reference:
            # df_summary = az.summary(arviz_data, var_names=["mu", "phi_init", "sigma_final"]) #"sigma_final", "likl_rt", using phi_init instead of phi_0 because phi_0 is padded with zeros for response states. If unpadded, the likelihood function gives a high likelihood for even 0 (or delta) response times.
            df_summary = az.summary(arviz_data, var_names=["mu", "phi_init", "sigma_final"], round_to="none") # Keep raw numeric values for downstream calculations; phi_init is used because phi_0 is padded with zeros for response states.
            
            #df_summary = (df_summary.reset_index(names="params")
                            #.assign(param_name = lambda df: df.params.str.split("[",expand=True)[0])
                            #.assign(part_id = lambda df: df.params.str.split("[",expand=True)[1].str.split(",",expand=True)[0])
                            #.assign(dims = lambda df:df.params.str.split("[", expand=True)[1].str.removesuffix("]")) 
            #        )
            
        elif estimation_type == "VI": 
            post_samples = ca.sample_posterior_params_VI(RT, X, n_states=n_states, start_width=start_width, response_width=response_width,
                                                    delta=delta,measurement_prob=measurement_prob,
                                                    num_warmup=num_warmup, samples_n=samples_n,
                                                    params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                    likelihood_type=likelihood_type 
                                                    )
            keys = ["mu", "sigma_final", "phi_0"]
            df_summary = []
            for k in keys:
                d = post_samples[k]
                d = d.mean(axis=(0))
                ranges = [range(s) for s in d.shape]
                names=[]
                for r in iter.product(*ranges):
                    names.append(k + "[" + ",".join(str(n)  for n in r) + "]")
                df_summary.append(pd.DataFrame(dict(
                    params = names,
                    mean = d.flatten(),
                )))
            df_summary = pd.concat(df_summary).set_index("params")
        else:
            raise Exception(f"Please select one of {estimation_type}")
        df_summary_csv = (df_summary
                            .reset_index(names="params")
                            .assign(param_name = lambda df: df.params.str.split("[",expand=True)[0])
                            .assign(part_id = lambda df: df.params.str.split("[",expand=True)[1].str.split(",",expand=True)[0])
                            .assign(dims = lambda df:df.params.str.split("[", expand=True)[1].str.removesuffix("]")) 
                    )
        df_summary_csv.to_csv(f"export/posterior_summary_{name}_{model_type}_{version}_{i}.csv")
        total_samples = post_samples["mu"].shape[0] # Numpyro merges the chain and draws dimensions
        pred_idx = np.random.default_rng().choice(total_samples, predictive_n, replace=False)
        log.info(f"Ending Posterior Sampling_{name}_{model_type}_{version}_{i} after {((time.perf_counter() - start_time_sampling)/60):.2f} mins")
        
        df_phi = df_summary.filter(like="phi_init",axis=0)[["mean"]].reset_index(names="idx")
        try:
            df_t = df_phi.idx.str.split("[", expand=True).loc[:,1].str.split(",", expand=True)
        except:
            print(df_phi.idx.str.split("[", expand=True))

        df_phi[["part_id", "phi_0"]] = df_t[[0,3]]#.astype(int)
        df_phi = df_phi.pivot(index="part_id", columns="phi_0", values="mean")
        
        #df_init_state_all = pd.concat([pd.DataFrame(i_s.squeeze()).reset_index().rename(columns={"index":"part_id"}).melt(id_vars="part_id", var_name="state", value_name="value").assign(param_id = i)
        #        for i, i_s in enumerate(post_samples["phi_0"][pred_idx,...])
        #        ]).astype({"param_id":"category"})
        #df_init_state_all.to_csv(f"export/initial_states_{name}_{model_type}_{version}_all.csv", index=None)

        df_phi.to_csv(f"export/initial_states_{name}_{model_type}_{version}_{i}.csv")
        pd.DataFrame(ID).to_csv(f"export/participants_id_{name}_{model_type}_{version}_{i}.csv")


        log.info(f"Starting Mean Confidence_{name}_{model_type}_{version}_{i}")
        drift_rate_est = post_samples["mu"].mean(axis=0)
        diffusion_rate_est = post_samples["sigma_final"].mean(axis=0)
        phi_0_est = post_samples["phi_0"].mean(axis=0) #posterior mean

        mean_init_conf, mean_final_conf, mean_resp_conf = ppm.get_mean_confidence(n_states, response_width, measurement_prob, delta, 
                                                                                  X, RT, drift_rate_est, diffusion_rate_est, phi_0_est, 
                                                                                  conf_scale, model_type, transition_type, likelihood_type)

        intensity_matrix = ca.get_intensity_matrix(n_states, drift_rate_est, diffusion_rate_est, model_type)
        Mc, Mw, Mn = ca._get_measurement_matrix(n_states, response_width, prob=measurement_prob, model_type = model_type)

        phi_t = ca.perform_state_transition(intensity_matrix, RT_s = RT, RA_s = X, Mc=Mc, Mw=Mw, Mn=Mn, phi_0=phi_0_est, delta=delta,
                                            transition_type=transition_type, likelihood_type=likelihood_type)

        pd.DataFrame(mean_init_conf[...,0]).reset_index(names="part_id").to_csv(f"export/mean_init_conf_{name}_{model_type}_{version}_{i}.csv")
        pd.DataFrame(mean_final_conf[...,0,0]).reset_index(names="part_id").to_csv(f"export/mean_final_conf_{name}_{model_type}_{version}_{i}.csv")
        pd.DataFrame(mean_resp_conf[...,0,0]).reset_index(names="part_id").to_csv(f"export/mean_resp_conf_{name}_{model_type}_{version}_{i}.csv")
        
        if execution_type == "Posterior":
            arviz_data.to_netcdf(f"export/arviz_inferencedata_{name}_{model_type}_{version}_{i}.nc")
            return None

        log.info(f"Starting Posterior Predictive Sampling_{name}_{model_type}_{version}_{i}")

        drift_rate_samples = post_samples["mu"][pred_idx, ...]
        diffusion_rate_samples = post_samples["sigma_final"][pred_idx,...]
        phi_0_samples = post_samples["phi_0"][pred_idx,...]

        post_pd_samples = ca.sample_post_pred_params(n_states=n_states, response_width=response_width, 
                                                     delta=delta,
                                                     measurement_prob=measurement_prob,
                                                    X=X, 
                                                    #X=None,
                                                    data_samples=RT.shape, 
                                                    min_RT_sec = min_RT_sec, max_RT_sec = max_RT_sec,
                                                    drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, 
                                                    phi_0_samples=phi_0_samples,
                                                    #RT=RT,
                                                    RT=None,
                                                    params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                    likelihood_type=likelihood_type, sampling_type=sampling_type,
                                                    is_parallel=False
                                                    )
        df_post_pred_all = pd.concat([samples["Samples"] for samples in post_pd_samples])
        df_post_pred_all.to_csv(f"export/posterior_predictive_{name}_{model_type}_{version}_{i}.csv")
        pp_rt = np.array([s["Samples"]["RT"].values for s in post_pd_samples])
        
        # Previous call with implicit sample dimensions retained for reference:
        # pp_idata = az.from_dict({"posterior_predictive": {"RT": pp_rt.reshape((-1,*RT.shape))[np.newaxis, ...]}}, coords=coords, dims=dims)
        pp_idata = az.from_dict({"posterior_predictive": {"RT": pp_rt.reshape((-1,*RT.shape))[np.newaxis, ...]}}, sample_dims=["chain", "draw"], coords=coords, dims=dims) # 1 x posterior_draws x I x J (chain x draw x participant x trial)
        arviz_data["posterior_predictive"] = pp_idata["posterior_predictive"]
        arviz_data.to_netcdf(f"export/arviz_inferencedata_{name}_{model_type}_{version}_{i}.nc")

        with open(f'export/mcmc_samples_{name}_{model_type}_{version}_{i}.pkl', 'wb') as outp:
            pickle.dump(dict(
                            n_states = n_states,
                            post_samples = post_samples, 
                            mean_init_conf = mean_init_conf,
                            mean_final_conf = mean_final_conf, 
                            phi_t = phi_t, 
                            phi_0 = df_phi,
                            prior_pd_samples = prior_pd_samples, 
                            post_pd_samples = post_pd_samples, 
                            RT = RT, 
                            X = X), outp, pickle.HIGHEST_PROTOCOL)

    fn = []

    process_id, process_count = _get_slurm_process_partition()
    all_batches = list(enumerate(zip(X_split, RT_split, ID_split)))

    if estimation_type == "MCMC" and process_count > 1:
        assigned_batches = _partition_work_for_slurm(all_batches, process_id, process_count)
    else:
        assigned_batches = all_batches

    def run_half_model_safely(i, X, RT, ID):
        participant_ids = np.asarray(ID).reshape(-1).tolist()

        log.info(
            "PARTICIPANT_MEMORY_START: batch=%s, participant_ids=%s, "
            "host=%s, pid=%s; memory: %s",
            i,
            participant_ids,
            socket.gethostname(),
            os.getpid(),
            _get_memory_diagnostics(),
        )

        try:
            run_half_model(i, X, RT, ID)
            return True
        except Exception as error:
            log.exception(
                "PARTICIPANT_FAILED_CONTINUING: batch=%s, participant_ids=%s, "
                "dataset=%s, model=%s, version=%s, error_type=%s, error=%s, "
                "host=%s, pid=%s, SLURM_JOB_ID=%s, SLURM_STEP_ID=%s, "
                "SLURM_PROCID=%s/%s, SLURM_CPUS_PER_TASK=%s, "
                "SLURM_MEM_PER_NODE=%s; memory_before_cleanup: %s",
                i,
                participant_ids,
                name,
                model_type,
                version,
                type(error).__name__,
                error,
                socket.gethostname(),
                os.getpid(),
                os.environ.get("SLURM_JOB_ID", "not set"),
                os.environ.get("SLURM_STEP_ID", "not set"),
                process_id,
                process_count,
                os.environ.get("SLURM_CPUS_PER_TASK", "not set"),
                os.environ.get("SLURM_MEM_PER_NODE", "not set"),
                _get_memory_diagnostics(),
            )

            try:
                jax.clear_caches()
                gc.collect()
            except Exception:
                pass

            log.error(
                "PARTICIPANT_SKIPPED: batch=%s, participant_ids=%s; "
                "continuing with the next assigned participant; "
                "memory_after_cleanup: %s",
                i,
                participant_ids,
                _get_memory_diagnostics(),
            )
            return False

    # Original unpartitioned batch scheduling retained for reference:
    # batch_n = 0
    # for i, (X, RT, ID) in enumerate(zip(X_split, RT_split, ID_split)):
    #     fn.append(delayed(run_half_model)(i, X, RT, ID))
    #     batch_n = batch_n + 1
    for i, (X, RT, ID) in assigned_batches:
        # Original unguarded participant execution retained for reference:
        # fn.append(delayed(run_half_model)(i, X, RT, ID))
        fn.append(delayed(run_half_model_safely)(i, X, RT, ID))

    batch_n = len(fn)
        #run_half_model()
        #if is_test:
        #    break
    
    start_time = time.perf_counter()
    # Original batch-level parallelism retained for reference:
    # n_jobs1=min(3,batch_n) if not is_test and is_parallel and jax.default_backend() != "gpu"  else 1
    # Original single-worker SLURM behavior retained for reference:
    # if process_count > 1:
    #     n_jobs1 = 1
    # Original multi-process SLURM check retained for reference:
    # if process_count > 1:
    if os.environ.get("SLURM_PROCID") is not None:
        # Original cores-per-batch SLURM scheduling retained for reference:
        # cores_per_batch = int(os.environ.get("CME_CORES_PER_BATCH", "10"))
        # allocated_cores = int(os.environ.get(
        #     "SLURM_CPUS_PER_TASK",
        #     os.cpu_count() or 1,
        # ))
        # local_worker_limit = max(1, allocated_cores // cores_per_batch)
        # n_jobs1 = max(1, min(local_worker_limit, batch_n))

        # SLURM already distributes batches between nodes. Run one JAX
        # process per node and let JAX/native math use the allocated cores.
        n_jobs1 = 1
    elif estimation_type == "MCMC":
        n_jobs1 = min(3, batch_n) if not is_test and is_parallel and jax.default_backend() != "gpu" else 1
    else:
        n_jobs1 = 1

    # Original participant-batch job logging retained for reference:
    # log.info(f"Starting {n_jobs1} jobs for sub-batch of participants in {data if isinstance(data, str) else data[0]} and {model_type}")
    log.info(
        "Process %s/%s received participant batches %s for %s and %s; "
        "running %s at a time",
        process_id,
        process_count,
        [i for i, _ in assigned_batches],
        data if isinstance(data, str) else data[0],
        model_type,
        n_jobs1,
    )
    # Original result-discarding execution retained for reference:
    # Parallel(n_jobs=n_jobs1, prefer="processes", backend = "loky")(f for f in fn)
    participant_results = Parallel(n_jobs=n_jobs1, prefer="processes", backend = "loky")(f for f in fn)
    failed_batches = participant_results.count(False)
    log.info(
        "Participant loop finished for %s, %s, %s: completed=%s, failed=%s",
        name,
        model_type,
        version,
        len(participant_results) - failed_batches,
        failed_batches,
    )
    
    # Original unconditional success message retained for reference:
    # log.info(f"Job successfully completed for {name}, {model_type}, {version} after {((time.perf_counter() - start_time)/60):.2f} mins")
    log.info(f"Job loop completed for {name}, {model_type}, {version} after {((time.perf_counter() - start_time)/60):.2f} mins")
