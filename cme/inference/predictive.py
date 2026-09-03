"""
Prior and posterior predictive sampling.

Two sampling routes share these entry points:
  "MCMC" runs NUTS over RT with the parameters held fixed (predictive_mcmc_fn)
  "GEN"/"SIM" draw RT through the model's own generator (ca.get_RT)
"""

import arviz as az
import scipy.stats as stats
from joblib import Parallel, delayed
from numpyro.infer import MCMC, NUTS, Predictive

import cme.decision_models.confidence_accumulation as ca
from cme.decision_models.confidence_accumulation import get_RT
from cme.inference.numpyro_model import model, transformed_likelihood
from cme.utils import common_logging as cl
from cme.utils import common_utils as cu

log = cl.get_logger("inference.predictive")


def predictive_model(RT_pred, n_states, response_width, delta, measurement_prob, phi_0, RA, 
                     drift_rate, diffusion_rate, 
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
    Mc, Mw, Mn = ca._get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)
    
    if model_type == "Markov":
        intensity_matrix = ca.diffusion_buildK(n_states, drift_rate, diffusion_rate)

    elif model_type == "Quantum":
        intensity_matrix = ca.quantum_buildH(n_states, drift_rate, diffusion_rate)
    else:
        raise Exception(f"Please select one of {model_type}")
    
    likl = transformed_likelihood(intensity_matrix, phi_0, delta, RT_pred, RA, Mc, Mw, Mn, 
                      transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    return likl.sum()

def predictive_mcmc_fn(n_states, response_width, delta, measurement_prob, X, 
                       drift_rate, diffusion_rate, phi_0,
                       params_type, model_type, transition_type, likelihood_type):
    
    if likelihood_type == "SINGLE":
        kernel = NUTS(potential_fn= lambda RT_pred: 
                                    predictive_model(RT_pred, n_states, response_width, delta, measurement_prob, phi_0, 
                                                    X, drift_rate, diffusion_rate,
                                                    transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type,))
        pred_shape = 4, *X.shape
        predictive_mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=4)
        predictive_mcmc.run(cu.get_rng(), init_params=stats.lognorm(s=1).rvs((pred_shape)),
                        extra_fields=('potential_energy',)
                        )
        

    elif likelihood_type == "JOINT":
        kernel = NUTS(potential_fn= lambda RT_pred_s: 
                        predictive_model(RT_pred_s, n_states, response_width, delta, measurement_prob, phi_0, 
                                        X, drift_rate, diffusion_rate,
                                        transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type,))
        pred_shape = 4, 2, *X[0].shape
        predictive_mcmc = MCMC(kernel, num_warmup=30, num_samples=20, num_chains=4)
        predictive_mcmc.run(cu.get_rng(), init_params=stats.lognorm(s=1).rvs((pred_shape)),
                    extra_fields=('potential_energy',)
                    )

    return {"drift_rate":drift_rate, "diffusion_rate":diffusion_rate, "predictive_chain":az.from_numpyro(predictive_mcmc)} #predictive_samples

def sample_prior_pred_params(n_states, start_width, response_width, delta, measurement_prob, X, RT=None,  
                        n_samples=10, data_samples=(1,10), min_RT_sec = 0, max_RT_sec = 10,
                        params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", 
                        transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", sampling_type = "MCMC|GEN", n_jobs=1, key=None):

    prior_predictive = Predictive(model, num_samples=n_samples, parallel=True)    
    if X is None:
        #X = np.ones(data_samples)
        raise Exception("X cannot be missing")
    prior_samples = prior_predictive(cu.get_rng() if key is None else key, n_states, start_width, response_width, delta, X, None, measurement_prob,
                                    params_type = params_type, transition_type=transition_type, 
                                    likelihood_type=likelihood_type, model_type=model_type)
    
    drift_rate_samples = prior_samples["mu"]
    diffusion_rate_samples = prior_samples["sigma_final"]
    phi_0_samples = prior_samples["phi_0"]
    predictive_samples = []
    parallel = Parallel(n_jobs=n_jobs)#drift_rate_samples.shape[0])

    if sampling_type == "MCMC":
        predictive_samples = parallel(delayed(predictive_mcmc_fn)(n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0,
                                                params_type = params_type, model_type = model_type, transition_type = transition_type, likelihood_type = likelihood_type)
                                    for drift_rate, diffusion_rate, phi_0 in zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples)
                                    )
    elif sampling_type == "GEN" or sampling_type == "SIM":
            predictive_samples = parallel(delayed(get_RT)(RT, n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0, min_RT_sec = min_RT_sec,  max_RT_sec = max_RT_sec,
                                                param_sample_id = param_sample_id,
                                                model_type = model_type, transition_type = transition_type, 
                                                likelihood_type = likelihood_type, data_samples = data_samples,
                                                sampling_type=sampling_type)
                                    for param_sample_id, (drift_rate, diffusion_rate, phi_0) in 
                                    enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples))
                                    )
    else:
        predictive_samples = dict(drift_rate = drift_rate_samples, diffusion_rate = diffusion_rate_samples, phi_0 = phi_0_samples)
    #return predictive_samples
    return prior_samples, predictive_samples

def sample_post_pred_params(n_states, response_width, delta, measurement_prob, X,
                            drift_rate_samples, diffusion_rate_samples, phi_0_samples, RT=None, 
                            data_samples=(1,10), min_RT_sec = 0, max_RT_sec=10,
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", 
                            transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", 
                            sampling_type = "MCMC|GEN", is_parallel=True):
    if X is None:
        raise Exception("X cannot be missing")
    predictive_samples = []
    n_jobs = drift_rate_samples.shape[0] if drift_rate_samples.shape[0] < 60 else 60
    parallel = Parallel(n_jobs=1 if not is_parallel else n_jobs)
    if sampling_type == "MCMC":
        predictive_samples = parallel(delayed(predictive_mcmc_fn)(n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0,
                                                params_type, model_type, transition_type, likelihood_type)
                                    for drift_rate, diffusion_rate, phi_0 in zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples)
                                    )
    elif sampling_type == "GEN" or sampling_type == "SIM":
            predictive_samples = parallel(delayed(get_RT)(RT, n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0, min_RT_sec = min_RT_sec, max_RT_sec=max_RT_sec,
                                                param_sample_id = param_sample_id,
                                                model_type = model_type, transition_type = transition_type, 
                                                likelihood_type = likelihood_type, data_samples = data_samples, 
                                                sampling_type=sampling_type)
                                    for param_sample_id, (drift_rate, diffusion_rate, phi_0) in 
                                    enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples))
                                    )
    else:
        predictive_samples = dict(drift_rate = drift_rate_samples, diffusion_rate = diffusion_rate_samples, phi_0 = phi_0_samples)
    
    return predictive_samples
