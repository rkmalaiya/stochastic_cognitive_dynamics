#%%
from pytensor import config
config.floatX = "float32"
config.openmp = True
config.openmp_elemwise_minsize=0

import pymc as pm
import numpy as np
from cme.utils import common_utils as ut
import cme.utils.common_logging as cl
log = cl.get_logger("diffusion")

name = "qdiffusion"

vars_transf_item = ["a_i_interval__",
"v_i_interval__"
]
vars_item = ["a_i",
"v_i"
]

vars_transf_pers = [
"a_p_log__",
"v_p_log__",
"t_er_log__"
]
vars_pers = [
"a_p",
"v_p",
"t_er"
]

def get_model(I,J, RT = None, X=None, a_p_mu = 0, full=False):

    log.debug(f"Getting model for prior: {a_p_mu}")
    with pm.Model() as qdiffusion:
        a_i = pm.Uniform("a_i",0,0.5, shape = (1,J))
        v_i = pm.Uniform("v_i",0,0.5, shape = (1,J))
        a_p = pm.Lognormal("a_p",a_p_mu,1,shape = (I,1))
        v_p = pm.Lognormal("v_p",0,1,shape = (I,1))
        
        t_er = pm.Lognormal("t_er",0,1,shape = (I,1))

        if full:
            delta = pm.Normal("delta")
            v_p = v_p - v_i + delta

        mu_kj, sigma_kj, p_kj = q_diffusion(a_i, v_i, a_p, v_p, t_er)

        RT_kj = pm.Normal("RT_kj",mu_kj, sigma_kj, shape = (I,J), observed = np.log(RT) if RT is not None else None)
        X_kj = pm.Bernoulli("X_kj", pm.invlogit(p_kj), shape = (I,J), observed = X)

    return qdiffusion #, (a_i, v_i, a_p, v_p, t_er, RT_kj, X_kj)


def q_diffusion(a_i, v_i, a_p, v_p, t_er, grad=False):

    p_jk = np.dot((v_p * a_p), (1 / (v_i * a_i)))
    h_jk = -p_jk 
    
    E_RT_kj = (1/2) * (
        np.dot((a_p * (1/v_p)),
        (v_i * (1/a_i)))
    ) * (
        (1 - np.exp(h_jk) ) / (1+np.exp(h_jk))
    ) + t_er

    V_RT_kj = (1/2) * (
        ( np.dot(a_p , (1/a_i))) * 
        np.power(np.dot((1/v_p), v_i),3) * (
            ( 2 * h_jk * np.exp(h_jk) - np.exp(2*h_jk) + 1) / (
                np.power((np.exp(h_jk) + 1),2)
            )
        )
    )

    mu_kj = np.log( E_RT_kj) - (1/2) * (
        1 + (V_RT_kj / np.power(E_RT_kj,2))
    )

    sigma_kj = np.sqrt(np.log(
        1 + (V_RT_kj / np.power(E_RT_kj,2))
    ))
    return mu_kj, sigma_kj, p_jk

def get_state(df_posterior, K, J):

    state = {}
    for (v_t, v) in zip(vars_transf_item, vars_item):
        state.update({
            v_t:
            df_posterior.filter(like=v,axis=0).loc[:,"mean"].to_numpy().squeeze().reshape(1,J)
            if type == 3 else
            df_posterior.filter(like=v,axis=0).loc[:,"mean"].to_numpy().squeeze()
        })

    for (v_t, v) in zip(vars_transf_pers, vars_pers):
        state.update({
            v_t:df_posterior.filter(like=v,axis=0).loc[:,"mean"].to_numpy().squeeze().reshape(K,1)
            if type != 1 else
            df_posterior.filter(like=v,axis=0).loc[:,"mean"].to_numpy().squeeze()
        })
    
    #state.update({"RT_kj_missing":df_posterior.filter(like="RT_kj_missing",axis=0).loc[:,"mean"].to_numpy()})

    return state

def gen_sample_data(model):
    prior_data = ut.sample_prior(model)
    prior_data_rt = prior_data.prior["RT_kj"].values.squeeze().mean(axis=0)
    prior_data_ra = prior_data.prior["X_kj"].values.squeeze()[1,:,:]
    return(prior_data_rt, prior_data_ra)

def sample_posterior_params(RT, X, samples_n, chains, tune, sampler="PYMC", acceptance_rate=0.90, likelihood=False, full=False, **kwargs):
    a_p_mu = 0
    if ("a_p_mu" in kwargs):
        a_p_mu = kwargs.get("a_p_mu")
    model = get_model(*X.shape, X = X, RT=RT, a_p_mu = a_p_mu,full=full)
    posterior_chain = ut.sample_posterior(model, samples_n, chains, tune, sampler, acceptance_rate, likelihood=likelihood,**kwargs)
    return posterior_chain, model

def sample_predictive_dist(model, posterior_chain=None, mode="Prior|Posterior|Both"):
    pred_chain = None
    
    if(mode == "Prior" or mode=="Both"):
        pred_chain = ut.sample_prior_pred(model)
    if((mode == "Posterior" or mode=="Both") and posterior_chain is not None):
        pred_chain = ut.sample_post_pred(model, posterior_chain)

    if(pred_chain is None):
        raise Exception("Either mode or posterior not provided")
    return pred_chain

def post_process_posterior(posterior_chain, method = "None|WAIC|LOO", **kwargs):
    summ = ut.get_summary(posterior_chain)
    if method is not None:
        w = ut.relative_model_fit(posterior_chain, method, **kwargs)
        print("***********", "w.elpd_waic", w.elpd_waic)
        summ = summ.assign(elpd_waic=w.elpd_waic).assign(elpd_se = w.se).assign(p_waic = w.p_waic)
    return summ

def _quick_test():
    X = np.random.randint(0,2,(70,5))
    RT = np.random.uniform(0,4,(70,5))

    log.debug(f"Starting Diffusion test")
    model = get_model(*X.shape, X = X, RT=RT,full=True)
    posterior_chain = pm.sample(model=model, draws=10, chains=2,tune=10)
    #with model:
    #    posterior_jax = jx.sample_numpyro_nuts(1000, tune = 500, chains=4, chain_method="parallel")
    log.debug(f"Posterior model v_correct {posterior_chain.posterior.v_p.shape}")
    assert posterior_chain.posterior.v_p.shape == (2,10,70,1)
    #assert posterior_jax.posterior.v_p.shape == (2,10,70,1)


if __name__ == "__main__":
    
    log.debug("Starting test")
    _quick_test()