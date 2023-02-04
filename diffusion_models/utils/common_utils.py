import pymc as pm
import arviz as az
import pymc.sampling.jax as jx
import numpy as np
import pandas as pd
import diffusion_models.utils.common_logging as cl
log = cl.get_logger("Common-Utils")

_cores=4


def sample_posterior(model, samples_n, chains, tune, sampler, acceptance_rate, likelihood=True):
    if sampler == "PYMC":
        return _sample_posterior_PyMC(model, samples_n, chains, tune, acceptance_rate, likelihood)
    elif sampler == "JAX":
        return _sample_posterior_JAX(model, samples_n, chains, tune, acceptance_rate)
    elif sampler == "SMC":
        return _sample_posterior_SMC(model, samples_n, chains)
    
def calculate_r_star(df_posterior, group_var, group_std_name, var_name, idx_name):
    
    df_stn  = df_posterior.query(f"{group_var} == {group_std_name} & var_name=='{var_name}'")[[group_var,"mean", idx_name]].pivot(columns=group_var, values="mean", index=idx_name).sort_index()
    df_sd  = df_posterior.query(f"{group_var} == {group_std_name} & var_name=='{var_name}'")[[group_var,"sd", idx_name]].pivot(columns=group_var, values="sd", index=idx_name).sort_index()
    
    df_pert  = df_posterior.query(f"{group_var} != {group_std_name} & var_name=='{var_name}'")[[group_var,"mean", idx_name]].pivot(columns=group_var, values="mean", index=idx_name).sort_index()
    
    df_rstar = ((df_pert-  df_stn.values)**2) / df_sd.values
    return df_rstar


def relative_model_fit(posterior_chain, method="WAIC|LOO", **kwargs):
    if method == "WAIC":
        var_name = kwargs["var_name"]
        return _calculate_waic(posterior_chain, var_name)

def _calculate_waic(posterior_chain, var_name):
    w = az.waic(posterior_chain, var_name=var_name,scale='negative_log')
    return w


def _sample_posterior_PyMC(model, samples_n, chains, tune, acceptance_rate, likelihood):
    with model:
        log.debug(model.free_RVs)
        posterior = pm.sample(samples_n, tune = tune, #step = pm.Metropolis(),
        return_inferencedata=True, chains=chains,
        target_accept=acceptance_rate, 
        cores=_cores, progressbar=True)

    if likelihood:
        posterior = _calculate_likelihood(posterior, model)
        #posterior = pm.sample(10000, step = pm.Metropolis(), return_inferencedata=True, cores=4)

    return posterior 

def _sample_posterior_JAX(model, samples_n, chains, tune, acceptance_rate):
    with model:
        log.debug(model.free_RVs)
        posterior = jx.sample_numpyro_nuts(samples_n, tune = tune, chains=chains, target_accept=acceptance_rate, chain_method="vectorized")
        
    return posterior 

def _sample_posterior_SMC(model, samples_n, chains):
    with model:
        posterior = pm.sample_smc(samples_n,chains=chains)
    return posterior

def _calculate_likelihood(posterior,model):
    posterior_chain = pm.compute_log_likelihood(posterior, model=model)
    return posterior_chain

def sample_post_pred(model, posterior, extend=True):
    with model:
        posterior_pred = pm.sample_posterior_predictive(posterior, extend_inferencedata=extend)
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

def get_summary(posterior_chain):
    df = az.summary(posterior_chain)
    df.loc[:,["var_name", "var_idx"]] = df.loc[:,["r_hat"]].reset_index().loc[:,"index"].str.split("[",expand=True).values
    
    return df.reset_index(drop=True)

def get_rhat(df_posterior):
    return df_posterior.loc[:,["r_hat"]].T

def get_chains_for_param(posterior_chain, param):
    t = posterior_chain.posterior[param].to_numpy()
    return pd.DataFrame(t.reshape((t.shape[0],-1 )).T)

def get_individuals_for_param(posterior_chain, param):
    t = posterior_chain.posterior[param].to_numpy()
    chain, samples, I, J = t.shape
    return pd.DataFrame(t.reshape((-1,I)))