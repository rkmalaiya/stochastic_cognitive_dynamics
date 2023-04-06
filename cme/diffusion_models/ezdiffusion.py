#%%
import pymc as pm
import numpy as np
from cme.utils import common_utils as ut
import cme.utils.common_logging as cl
import pymc.sampling.jax as jx
log = cl.get_logger("diffusion")

type = 3

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

a_p_mu=0
v_p_mu=0


def get_model(I,J, RT = None, X=None):
    
    with pm.Model() as qdiffusion:

        a_p = pm.Lognormal("a_p",a_p_mu,1,shape = (I,1))
        v_p = pm.Lognormal("v_p",v_p_mu,1,shape = (I,1))
        #v_p = pm.Gamma("v_p",mu,2,shape = (K,1))
        
        t_er = pm.Lognormal("t_er",0,1,shape = (I,1))

        mu_kj, sigma_kj, p_kj = _ez_diffusion(a_p, v_p, t_er)

        RT_kj = pm.Normal("RT_kj",mu_kj, sigma_kj, shape = (I,J), observed = np.log(RT))
        X_kj = pm.Bernoulli("X_kj", pm.invlogit(p_kj), shape = (I,J), observed = X)

    return qdiffusion #, (a_i, v_i, a_p, v_p, t_er, RT_kj, X_kj)

def _ez_diffusion(a_p, v_p, t_er, grad=False):

    p_jk = v_p * a_p
    h_jk = -p_jk 
    
    E_RT_kj = ((1/2) * (a_p * (1/v_p))
    ) * (
        (1 - np.exp(h_jk) ) / (1+np.exp(h_jk))
    ) + t_er

    V_RT_kj = (1/2) * (
        a_p * np.power(1/v_p,3) * (
            ( 2 * h_jk * np.exp(h_jk) - np.exp(2*h_jk) + 1) / (
                np.power((np.exp(h_jk) + 1),2)
            )
        )
    )

    mu_kj = np.log( E_RT_kj) - (1/2) * (
        1 + (V_RT_kj / np.power(E_RT_kj,2))
    )

    sigma_kj = np.log(
        1 + (V_RT_kj / np.power(E_RT_kj,2))
    )
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

def sample_posterior_params(RT, X, samples_n, chains, tune, sampler="PYMC", acceptance_rate=0.90):
    model = get_model(*X.shape, X = X, RT=RT)
    posterior_chain = ut.sample_posterior(model, samples_n, chains, tune, sampler, acceptance_rate)
    return posterior_chain, model

def _quick_test():
    X = np.random.randint(0,2,(70,5))
    RT = np.random.uniform(0,4,(70,5))

    log.debug(f"Starting Diffusion test")
    model = get_model(*X.shape, X = X, RT=RT)
    posterior_chain = pm.sample(model=model, draws=10, chains=2,tune=10)
    with model:
        posterior_jax = jx.sample_numpyro_nuts(1000, tune = 500, chains=4, chain_method="parallel")
    log.debug(f"Posterior model v_correct {posterior_chain.posterior.v_p.shape}")
    assert posterior_chain.posterior.v_p.shape == (2,10,70,1)
    assert posterior_jax.posterior.v_p.shape == (2,10,70,1)


if __name__ == "__main__":
    
    log.debug("Starting test")
    _quick_test()