"""
Posterior sampling by variational inference (SVI).
"""

import numpyro.infer.autoguide as ag
from numpyro.infer import Predictive, SVI, Trace_ELBO
from optax import adam, chain, clip

from cme.inference.numpyro_model import model, get_original_params
from cme.utils import common_logging as cl
from cme.utils import common_utils as cu

log = cl.get_logger("inference.vi")


def sample_posterior_params_VI(DT, X, n_states, start_width, response_width, delta, measurement_prob,
                            num_warmup=100, samples_n=500, num_chains=4, batch_size=2,  
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    #guide = ag.AutoNormal(model)
    #guide = ag.AutoDiagonalNormal(model)
    #guide = ag.AutoMultivariateNormal(model)
    guide = ag.AutoLowRankMultivariateNormal(model)
    #guide = ag.AutoDAIS(model)
    #guide = ag.AutoDelta(model)

    #optimizer = Adam(step_size=0.5)
    #svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi = SVI(model, guide, chain(clip(10.0), adam(1e-3)), loss=Trace_ELBO())
    svi_result = svi.run(cu.get_rng(), num_warmup + samples_n, n_states, start_width, 
                        response_width, delta, X, DT, measurement_prob, 
                        params_type = params_type, transition_type=transition_type, 
                        likelihood_type=likelihood_type, model_type=model_type, stable_update=True)

    predictive = Predictive(guide, params=svi_result.params, num_samples=samples_n, parallel=True)
    posterior_samples = predictive(cu.get_rng(),n_states, start_width, response_width, delta, X, DT, measurement_prob, 
                   params_type = params_type, transition_type=transition_type, 
                   likelihood_type=likelihood_type, model_type=model_type)
    
    posterior_samples = get_original_params(posterior_samples, response_width, params_type, model_type)
    return posterior_samples
