from tracemalloc import start
from attr import dataclass
#from dataclasses import dataclass, fields
import cme.decision_models.confidence_accumulation as ca
import cme.decision_models.quantum_discrete as qd
import jax.numpy as npx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import arviz as az
import numpy as np
import pickle
import time
from joblib import Parallel, delayed
from cme.utils import common_logging as cl
log = cl.get_logger("fit_model")
import jax
log.info(f"JAX devices: {jax.default_backend()}")
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
    file_posts:list = []
    version:float = 0.1
    n_states:int = 11
    start_width:int = None # None value will be automatically calculated.
    response_width:int = 1
    delta:float = 0.01
    measurement_prob:float = 0.8
    num_warmup: int = 1400 
    samples_n: int = 1700
    predictive_n: int = 100
    batch_size: int = 100
    params_type:str = "Centralized|NonCentralized"
    model_type:str = "Markov|Quantum"
    transition_type:str = "RT|TIMESTEP"
    likelihood_type:str = "SINGLE|JOINT"
    sampling_type:str = "MCMC|GEN"
    scale: str = "None|Log|SQRT"
    conf_scale: str = "None|(add_scale, mul_scale)"
    csv_header:bool = False
    is_test:bool = False
    is_parallel:bool=True


#folder, file_pre, file_posts, version, n_states, start_width, delta, measurement_prob, params_type, model_type, transition_type, likelihood_type, sampling_type
def fit_model(model: ModelDetails):

    file_loc = f"{model.folder}/{model.file_pre}"
    start_width = (model.n_states-2*model.response_width)
    if model.start_width == None or model.start_width == 0:
        model.start_width = start_width
    elif model.start_width > start_width:
        raise Warning(f"start_width larger than ideal value of {start_width}. {model.model_type} model may have unexpected results")
    
    #file_post = 
    #version = 0.5
    #len(model.file_posts)
    n_jobs = min(4, len(model.file_posts)) if not model.is_test and model.is_parallel else 1
    log.info(f"Received request for {n_jobs} files to be executed in parallel for {model.model_type}_{model.version}!!")
    log.info(f"Received configuration: {model}")
    Parallel(n_jobs=n_jobs)(delayed(_run_model)(
                                    
                                    f"{file_loc}{name}_rt.csv", f"{file_loc}{name}_ra.csv", name, model.version, 
                                    model.n_states, model.start_width, model.response_width, model.delta, model.measurement_prob, 
                                    model.params_type, model.model_type, model.transition_type, model.likelihood_type, model.sampling_type,
                                    model.num_warmup, model.samples_n, model.predictive_n, model.batch_size, model.is_test, 
                                    model.scale, model.conf_scale, model.csv_header, model.is_parallel) 
                                                
                                    for name in model.file_posts)
    log.info(f"All jobs successfully completed for {model.model_type}_{model.version}!!!!")


def _run_model(RT_file, X_file, name, version, 
            n_states, start_width, response_width, delta, measurement_prob, 
            params_type, model_type, transition_type, likelihood_type, sampling_type,
            num_warmup, samples_n, predictive_n, batch_size, is_test, scale, conf_scale, csv_header, is_parallel):
    
    df_X = pd.read_csv(X_file, header="infer" if csv_header else None)
    df_RT = pd.read_csv(RT_file, header="infer" if csv_header else None)

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
    
    if scale == "Log":
        RTs = np.log(RTs)
    elif scale=="SQRT":
        RTs = np.sqrt(RTs)

    X_split = np.split(Xs, npx.arange(batch_size, Xs.shape[0], batch_size), axis=0)
    RT_split = np.split(RTs, npx.arange(batch_size, RTs.shape[0], batch_size), axis=0)
    ID_split = np.split(IDs, npx.arange(batch_size, IDs.shape[0], batch_size), axis=0)
    
    def run_half_model(i, X, RT, ID):
        
        q_Mc, q_Mw, q_Mn = ca._get_measurement_matrix(n_states, response_width, prob=measurement_prob, model_type = model_type)

        log.info(f"Starting Prior Predictive Sampling_{name}_{model_type}_{version}_{i}")
        prior_pd_samples = ca.sample_prior_pred_params(n_states=n_states,start_width=start_width, response_width=response_width,
                                                        delta=delta, data_samples=RT.shape,
                                                        measurement_prob=measurement_prob, X=X, RT=RT, n_samples=predictive_n,
                                                        params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                        likelihood_type=likelihood_type, sampling_type=sampling_type, 
                                                    )
        start_time_sampling = time.perf_counter()
        log.info(f"Starting Posterior Sampling_{name}_{model_type}_{version}_{i}")
        post_chain = ca.sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, response_width=response_width,
                                                delta=delta,measurement_prob=measurement_prob,
                                                num_warmup=num_warmup, samples_n=samples_n,
                                                params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                likelihood_type=likelihood_type 
                                                )
        log.info(f"Ending Posterior Sampling_{name}_{model_type}_{version}_{i} after {((time.perf_counter() - start_time_sampling)/60):.2f} mins")
        post_samples = post_chain.get_samples()
        total_samples = post_samples["mu"].shape[0]
        pred_idx = np.random.default_rng().choice(total_samples, predictive_n)

        df_summary = az.summary(az.from_numpyro(post_chain), var_names=["mu", "phi_0", "likl_rt"]) #"sigma_final",
        df_summary_csv = (df_summary
                            .reset_index(names="params")
                            .assign(param_name = lambda df: df.params.str.split("[",expand=True)[0])
                            .assign(part_id = lambda df: df.params.str.split("[",expand=True)[1].str.split(",",expand=True)[0])
                            .assign(dims = lambda df:df.params.str.split("[", expand=True)[1].str.removesuffix("]")) 
                    )
        df_phi = df_summary.filter(like="phi_0",axis=0)[["mean"]].reset_index(names="idx")
        df_t = df_phi.idx.str.split("[", expand=True)[1].str.split(",", expand=True)
        df_phi[["part_id", "phi_0"]] = df_t[[0,2]].astype(int)
        df_phi = df_phi.pivot(index="part_id", columns="phi_0", values="mean")
        
        #df_init_state_all = pd.concat([pd.DataFrame(i_s.squeeze()).reset_index().rename(columns={"index":"part_id"}).melt(id_vars="part_id", var_name="state", value_name="value").assign(param_id = i)
        #        for i, i_s in enumerate(post_samples["phi_0"][pred_idx,...])
        #        ]).astype({"param_id":"category"})
        #df_init_state_all.to_csv(f"export/initial_states_{name}_{model_type}_{version}_all.csv", index=None)


        log.info(f"Starting Mean Confidence_{name}_{model_type}_{version}_{i}")
        drift_rate_est = post_samples["mu"].mean(axis=0)
        diffusion_rate_est = post_samples["sigma_final"].mean(axis=0)
        phi_0_est = post_samples["phi_0"].mean(axis=0) #posterior mean

        intensity_matrix_quantum = qd._buildH(n_states, drift_rate_est, diffusion_rate_est)

        mean_init_conf = ca.get_mean_init_confidence(n_states=n_states, phi_0 = phi_0_est, model_type=model_type)
        mean_final_conf = ca.get_mean_confidence(n_states=n_states, intensity_matrix=intensity_matrix_quantum,phi_0=phi_0_est,
                            delta= delta, Mc = q_Mc, Mw=q_Mw, Mn=q_Mn, t=RT,x=X, conf_scale=conf_scale,
                            model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type)
        phi_t = ca.perform_state_transition(intensity_matrix_quantum, RT_s = RT, RA_s = X, Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, phi_0=phi_0_est, delta=delta,
                                            transition_type=transition_type, likelihood_type=likelihood_type)

        
        drift_rate_samples = post_samples["mu"][pred_idx, ...]
        diffusion_rate_samples = post_samples["sigma_final"][pred_idx,...]
        phi_0_samples = post_samples["phi_0"][pred_idx,...]

        log.info(f"Starting Posterior Predictive Sampling_{name}_{model_type}_{version}_{i}")
        
        post_pd_samples = ca.sample_post_pred_params(n_states=n_states, start_width=start_width, delta=delta,measurement_prob=measurement_prob,
                                                    X=X, data_samples=RT.shape,
                                                    drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, 
                                                    phi_0_samples=phi_0_samples,
                                                    RT=RT,
                                                    params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                    likelihood_type=likelihood_type, sampling_type=sampling_type
                                                    )

        with open(f'export/mcmc_samples_{name}_{model_type}_{version}_{i}.pkl', 'wb') as outp:
            pickle.dump(dict(post_samples = post_samples, 
                            mean_init_conf = mean_init_conf,
                            mean_final_conf = mean_final_conf, 
                            phi_t = phi_t, 
                            phi_0 = df_phi,
                            prior_pd_samples = prior_pd_samples, 
                            post_pd_samples = post_pd_samples, 
                            RT = RT, 
                            X = X), outp, pickle.HIGHEST_PROTOCOL)

        df_summary_csv.to_csv(f"export/posterior_summary_{name}_{model_type}_{version}_{i}.csv")
        df_phi.to_csv(f"export/initial_states_{name}_{model_type}_{version}_{i}.csv")
        pd.DataFrame(ID).to_csv(f"export/participants_id_{name}_{model_type}_{version}_{i}.csv")
        
        df_prior_pred_all = pd.concat([samples["Samples"] for samples in prior_pd_samples])
        df_prior_pred_all.to_csv(f"export/prior_predictive_{name}_{model_type}_{version}_{i}.csv")

        df_post_pred_all = pd.concat([samples["Samples"] for samples in post_pd_samples])
        df_post_pred_all.to_csv(f"export/posterior_predictive_{name}_{model_type}_{version}_{i}.csv")

    fn = []

    batch_n = 0
    for i, (X, RT, ID) in enumerate(zip(X_split, RT_split, ID_split)):
        fn.append(delayed(run_half_model)(i, X, RT, ID))
        batch_n = batch_n + 1
        #run_half_model()
        #if is_test:
        #    break
    
    start_time = time.perf_counter()
    n_jobs1=min(3,batch_n) if not is_test and is_parallel else 1
    log.info(f"Starting {n_jobs1} jobs for sub-batch of participants at time")
    Parallel(n_jobs=n_jobs1, prefer="processes", backend = "loky")(f for f in fn)
    
    log.info(f"Job successfully completed for {name}, {model_type}, {version} after {((time.perf_counter() - start_time)/60):.2f} mins")
