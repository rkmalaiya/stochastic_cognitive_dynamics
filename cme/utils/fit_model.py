import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0" "1"

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



#folder, file_pre, file_posts, version, n_states, start_width, delta, measurement_prob, params_type, model_type, transition_type, likelihood_type, sampling_type
def fit_model(model: ModelDetails):
    
    log.info(f"Compute devices: {jax.default_backend()}, {jax.devices()}")
    file_loc = f"{model.folder}/{model.file_pre}"
    
    #file_post = 
    #version = 0.5
    #len(model.file_posts)
    n_jobs = min(3, len(model.data) + len(model.model_type)) if not model.is_test and model.is_parallel and jax.default_backend() != "gpu" else 1
    log.info(f"Received request for {n_jobs} files to be executed in parallel for {model.model_type}_version:{model.version}_states:{model.n_states}_resp_width:{model.response_width}!!")
    
    Parallel(n_jobs=n_jobs, prefer="processes", backend = "loky")(delayed(_run_model)(
                                    
                                    file_loc, data, model.version, 
                                    n_states, model.start_width, response_width, model.delta, model.measurement_prob, 
                                    model.params_type, model_type, model.transition_type, model.likelihood_type, 
                                    model.sampling_type, model.estimation_type, model.execution_type,
                                    model.num_warmup, model.samples_n, model.predictive_n, model.batch_size, model.is_test, 
                                    model.scale, conf_scale, model.csv_header, model.is_parallel) 
                                                
                                    for data, (model_type, n_states, response_width, conf_scale) in iter.product(model.data, zip(model.model_type, model.n_states, model.response_width, model.conf_scale)))
    log.info(f"All jobs successfully completed for {model.model_type}_{model.version}!!!!")


def _run_model(file_loc, data, version, 
            n_states, start_width, response_width, delta, measurement_prob, 
            params_type, model_type, transition_type, likelihood_type, sampling_type, estimation_type,execution_type,
            num_warmup, samples_n, predictive_n, batch_size, is_test, scale, conf_scale, csv_header, is_parallel):
    
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
    
    log.info(f"Received participant size: {RTs.shape}")

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
        
        min_RT_sec = np.clip(RT.mean() - 3*RT.std(), a_min=delta, a_max=None)
        max_RT_sec = RT.mean() + 3*RT.std()
        log.info(f"Starting Prior Predictive Sampling_{name}_{model_type}_{version}_{i} for {min_RT_sec} to {max_RT_sec} secs")
        prior_pd_samples = ca.sample_prior_pred_params(n_states=n_states,start_width=start_width, response_width=response_width,
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
            post_chain = ca.sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, response_width=response_width,
                                                    delta=delta,measurement_prob=measurement_prob,
                                                    num_warmup=num_warmup, samples_n=samples_n,
                                                    params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                    likelihood_type=likelihood_type, num_chains=4 
                                                    )
            
            post_samples = post_chain.get_samples()
            coords = {
                        "part_id": ID.squeeze(),
                    }
            dims = {
                        "mu_r": ["part_id"],
                        "sigma_r": ["part_id"],
                        "phi_0": ["part_id"],
                    }
            arviz_data = az.from_numpyro(post_chain,
                                        coords=coords,
                                        dims=dims)
            arviz_data.to_netcdf(f"export/arviz_inferencedata_{name}_{model_type}_{version}_{i}.nc")
            df_summary = az.summary(arviz_data, var_names=["mu", "phi_0", "sigma_final"]) #"sigma_final", "likl_rt", 
            
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
        total_samples = post_samples["mu"].shape[0]
        pred_idx = np.random.default_rng().choice(total_samples, predictive_n, replace=False)
        log.info(f"Ending Posterior Sampling_{name}_{model_type}_{version}_{i} after {((time.perf_counter() - start_time_sampling)/60):.2f} mins")
        
        df_phi = df_summary.filter(like="phi_0",axis=0)[["mean"]].reset_index(names="idx")
        df_t = df_phi.idx.str.split("[", expand=True)[1].str.split(",", expand=True)
        df_phi[["part_id", "phi_0"]] = df_t[[0,2]].astype(int)
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
        #arviz_data.to_netcdf(f"export/arviz_inferencedata_{name}_{model_type}_{version}_{i}.nc")

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

    batch_n = 0
    for i, (X, RT, ID) in enumerate(zip(X_split, RT_split, ID_split)):
        fn.append(delayed(run_half_model)(i, X, RT, ID))
        batch_n = batch_n + 1
        #run_half_model()
        #if is_test:
        #    break
    
    start_time = time.perf_counter()
    n_jobs1=min(3,batch_n) if not is_test and is_parallel and jax.default_backend() != "gpu"  else 1
    log.info(f"Starting {n_jobs1} jobs for sub-batch of participants")
    Parallel(n_jobs=n_jobs1, prefer="processes", backend = "loky")(f for f in fn)
    
    log.info(f"Job successfully completed for {name}, {model_type}, {version} after {((time.perf_counter() - start_time)/60):.2f} mins")
