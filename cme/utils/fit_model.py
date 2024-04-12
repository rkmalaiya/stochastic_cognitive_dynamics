from attr import dataclass
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
from joblib import Parallel, delayed

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
    start_width:int = 5
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

#folder, file_pre, file_posts, version, n_states, start_width, delta, measurement_prob, params_type, model_type, transition_type, likelihood_type, sampling_type
def fit_model(model: ModelDetails):

    file_loc = f"{model.folder}/{model.file_pre}"
    
    #file_post = 
    #version = 0.5
    #len(model.file_posts)
    n_jobs = min(4, len(model.file_posts))
    print(f"Received request for {n_jobs} files to be executed in parallel for {model.model_type}_{model.version}!!")
    Parallel(n_jobs=n_jobs)(delayed(_run_model)(
                                    
                                    f"{file_loc}{name}_rt.csv", f"{file_loc}{name}_ra.csv", name, model.version, 
                                    model.n_states, model.start_width, model.response_width, model.delta, model.measurement_prob, 
                                    model.params_type, model.model_type, model.transition_type, model.likelihood_type, model.sampling_type,
                                    model.num_warmup, model.samples_n, model.predictive_n, model.batch_size) 
                                                
                                    for name in model.file_posts)
    print(f"All jobs successfully completed for {model.model_type}_{model.version}!!!!")


def _run_model(RT_file, X_file, name, version, 
            n_states, start_width, response_width, delta, measurement_prob, 
            params_type, model_type, transition_type, likelihood_type, sampling_type,
            num_warmup, samples_n, predictive_n = 100, batch_size=10):
    
    df_X = pd.read_csv(X_file)
    df_RT = pd.read_csv(RT_file)

    df_X = df_X.drop("id", axis=1) if "id" in df_X.columns else df_X 
    df_RT = df_RT.drop("id", axis=1) if "id" in df_RT.columns else df_RT 

    Xs = df_X.values
    RTs = df_RT.values

    X_split = np.split(Xs, npx.arange(batch_size, Xs.shape[0], batch_size), axis=0)
    RT_split = np.split(RTs, npx.arange(batch_size, RTs.shape[0], batch_size), axis=0)


    for i, (X, RT) in enumerate(zip(X_split, RT_split)):
    #for i in range(1):
        q_Mc, q_Mw, q_Mn = ca._get_measurement_matrix(n_states, response_width, prob=measurement_prob, model_type = model_type)

        prior_pd_samples = ca.sample_prior_pred_params(n_states=n_states,start_width=start_width,delta=delta,
                                                    measurement_prob=measurement_prob, X=X, RT=RT, n_samples=predictive_n,
                                                    params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                    likelihood_type=likelihood_type, sampling_type=sampling_type, 
                                                    )
        
        post_chain = ca.sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, delta=delta,measurement_prob=measurement_prob,
                                            num_warmup=num_warmup, samples_n=samples_n,
                                            params_type=params_type, model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type 
                            )
        post_samples = post_chain.get_samples()
        df_summary = az.summary(az.from_numpyro(post_chain), var_names=["mu", "sigma_final","phi_0", "likl_rt"])
        df_phi = df_summary.filter(like="phi_0",axis=0)[["mean"]].reset_index(names="idx")
        df_t = df_phi.idx.str.split("[", expand=True)[1].str.split(",", expand=True)
        df_phi[["part_id", "phi_0"]] = df_t[[0,2]].astype(int)
        df_phi = df_phi.pivot(index="part_id", columns="phi_0", values="mean")
        
        df_summary.to_csv(f"export/posterior_summary_{name}_{model_type}_{version}_{i}.csv")
        df_phi.to_csv(f"export/initial_states_{name}_{model_type}_{version}_{i}.csv")

        #df_init_state_all = pd.concat([pd.DataFrame(i_s.squeeze()).reset_index().rename(columns={"index":"part_id"}).melt(id_vars="part_id", var_name="state", value_name="value").assign(param_id = i)
        #        for i, i_s in enumerate(post_samples["phi_0"])
        #        ]).astype({"param_id":"category"})
        #df_init_state_all.to_csv(f"export/initial_states_{name}_{model_type}_{version}_all.csv", index=None)

        drift_rate_est = post_samples["mu"].mean(axis=0)
        diffusion_rate_est = post_samples["sigma_final"].mean(axis=0)
        phi_0_est = post_samples["phi_0"].mean(axis=0)

        intensity_matrix_quantum = qd._buildH(n_states, drift_rate_est, diffusion_rate_est)

        mean_conf = ca.get_mean_confidence(n_states=n_states, intensity_matrix=intensity_matrix_quantum,phi_0=phi_0_est,
                            delta= delta, Mc = q_Mc, Mw=q_Mw, Mn=q_Mn, t=RT,x=X,
                            model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type)
        phi_t = ca.perform_state_transition(intensity_matrix_quantum, RT_s = RT, RA_s = X, Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, phi_0=phi_0_est, delta=delta,
                                            transition_type=transition_type, likelihood_type=likelihood_type)

        total_samples = samples_n * 4
        drift_rate_samples = post_samples["mu"][np.random.default_rng().choice(total_samples, 500),...]
        diffusion_rate_samples = post_samples["sigma_final"][np.random.default_rng().choice(total_samples, 500),...]
        phi_0_samples = post_samples["phi_0"][np.random.default_rng().choice(total_samples, 500),...]

        post_pd_samples = ca.sample_post_pred_params(n_states=n_states, start_width=start_width, delta=delta,measurement_prob=measurement_prob,
                                                    X=X, 
                                                    drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, 
                                                    phi_0_samples=phi_0_samples,
                                                    RT=RT,
                                                    params_type=params_type, model_type=model_type, transition_type=transition_type, 
                                                    likelihood_type=likelihood_type, sampling_type=sampling_type
                                                    )

        with open(f'export/mcmc_samples_{name}_{model_type}_{version}_{i}.pkl', 'wb') as outp:
            pickle.dump([post_samples, mean_conf, phi_t, prior_pd_samples, post_pd_samples], outp, pickle.HIGHEST_PROTOCOL)

        
        df_prior_pred_all = pd.concat([samples["Samples"] for samples in prior_pd_samples])
        df_prior_pred_all.to_csv(f"export/prior_predictive_{name}_{model_type}_{version}_{i}.csv")

        df_post_pred_all = pd.concat([samples["Samples"] for samples in post_pd_samples])
        df_post_pred_all.to_csv(f"export/posterior_predictive_{name}_{model_type}_{version}_{i}.csv")


    
    print(f"Job successfully completed for {name}_{model_type}, {version}")
