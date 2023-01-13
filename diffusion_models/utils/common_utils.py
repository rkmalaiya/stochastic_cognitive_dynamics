import pymc as pm
import arviz as az
import pymc.sampling.jax as jx
import numpy as np
import pandas as pd
import diffusion_models.utils.common_logging as cl
log = cl.get_logger("Common-Utils")

_cores=4


def sample_posterior(model, samples_n, chains, tune, sampler, acceptance_rate):
    if sampler == "PYMC":
        return _sample_posterior_PyMC(model, samples_n, chains, tune, acceptance_rate)
    elif sampler == "JAX":
        return _sample_posterior_JAX(model, samples_n, chains, tune, acceptance_rate)
    elif sampler == "SMC":
        return _sample_posterior_SMC(model, samples_n, chains)
    
def _sample_posterior_PyMC(model, samples_n, chains, tune, acceptance_rate):
    with model:
        log.debug(model.free_RVs)
        posterior = pm.sample(samples_n, tune = tune, #step = pm.Metropolis(),
        return_inferencedata=True, chains=chains,
        #target_accept=acceptance_rate, 
        cores=_cores, progressbar=True)
        #posterior = pm.sample(10000, step = pm.Metropolis(), return_inferencedata=True, cores=4)

    return posterior 

def _sample_posterior_JAX(model, samples_n, chains, tune, acceptance_rate):
    with model:
        log.debug(model.free_RVs)
        posterior = jx.sample_numpyro_nuts(samples_n, tune = tune, chains=chains, target_accept=acceptance_rate)
        
    return posterior 
    
def _sample_posterior_SMC(model, samples_n, chains):
    with model:
        posterior = pm.sample_smc(samples_n,chains=chains)
    return posterior

def sample_post_pred(model, posterior, samples_n, cores=8):
    with model:
        posterior_pred = pm.sample_posterior_predictive(posterior.sel(chain=[0]))
    return posterior_pred

def sample_prior(model, samples_n=100):
    with model:
        prior_chain = pm.sample_prior_predictive(samples=samples_n)
    return prior_chain

def get_gradient(model, vars):
    return model.compile_dlogp(vars)

def get_gradient2(model, vars):
    return model.compile_d2logp(vars)

def OPG(state, grad, posterior_chain, K, J):
    #df_posterior = az.summary(posterior_chain)
    #state = get_state(df_posterior, K, J)
    grad = grad(state)
    opg_cov = np.outer(grad,grad)
    return grad, opg_cov

def hessian(state, grad2, posterior_chain, K, J):
    #df_posterior = az.summary(posterior_chain)
    #state = get_state(df_posterior, K, J)
    hess_cov = grad2(state)
    return hess_cov

def extract_var(posterior_chains, var="", axis=-2):
    var_mat_c, var_mat_ic = None, None
    for p_c, p_ic in posterior_chains:
        if var_mat_c is not None:
            var_mat_c = np.append(var_mat_c, p_c.posterior[var].values, axis=axis)
        else:
            var_mat_c = p_c.posterior[var].values
        
        if var_mat_ic is not None:
            var_mat_ic = np.append(var_mat_ic, p_ic.posterior[var].values, axis=axis)
        else:
            var_mat_ic = p_ic.posterior[var].values
    
    return var_mat_c, var_mat_ic

def get_rhat(posterior_chain):
    return az.summary(posterior_chain).loc[:,["r_hat"]].T

def get_chains_for_param(posterior_chain, param):
    t = posterior_chain.posterior[param].to_numpy()
    return pd.DataFrame(t.reshape((t.shape[0],-1 )).T)

def get_individuals_for_param(posterior_chain, param):
    t = posterior_chain.posterior[param].to_numpy()
    chain, samples, I, J = t.shape
    return pd.DataFrame(t.reshape((-1,I)))