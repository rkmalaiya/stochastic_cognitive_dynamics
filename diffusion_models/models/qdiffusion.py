#%%
import pymc as pm
import numpy as np
from diffusion_models.utils import common_utils as ut


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



def q_diffusion_model_1_1(K,J, mu, RT = None, X=None):
    
    with pm.Model() as qdiffusion:
        a_i = pm.Uniform("a_i",0,0.5)
        v_i = pm.Uniform("v_i",0,100)

        a_p = pm.Lognormal("a_p",a_p_mu,1)
        v_p = pm.Lognormal("v_p",v_p_mu,1)

        t_er = pm.Lognormal("t_er",0,1)

        mu_kj, sigma_kj, p_kj = q_diffusion(a_i, v_i, a_p, v_p, t_er)

        X_kj = pm.Bernoulli("X_kj", pm.invlogit(p_kj), shape = (K,J), observed = X)
        RT_kj = pm.Normal("RT_kj",mu_kj, sigma_kj, shape = (K,J), observed = RT)
        

    return qdiffusion, (a_i, v_i, a_p, v_p, t_er, RT_kj, X_kj)



def q_diffusion_model_K_1(K,J, RT = None, X=None):
    
    with pm.Model() as qdiffusion:
        a_i = pm.Uniform("a_i",0,0.5)
        v_i = pm.Uniform("v_i",0,100)

        a_p = pm.Lognormal("a_p",a_p_mu,1, shape = (K,1))
        v_p = pm.Lognormal("v_p",v_p_mu,1, shape = (K,1))

        t_er = pm.Lognormal("t_er",0,1, shape = (K,1))

        mu_kj, sigma_kj, p_kj = q_diffusion(a_i, v_i, a_p, v_p, t_er)

        RT_kj = pm.Normal("RT_kj",mu_kj, sigma_kj, shape = (K,J), observed = RT)
        X_kj = pm.Bernoulli("X_kj", pm.invlogit(p_kj), shape = (K,J), observed = X)

    return qdiffusion, (a_i, v_i, a_p, v_p, t_er, RT_kj, X_kj)

def q_diffusion_model(K,J, mu, RT = None, X=None):
    
    with pm.Model() as qdiffusion:
        a_i = pm.Uniform("a_i",0,0.5, shape = (1,J))
        v_i = pm.Uniform("v_i",0,0.5, shape = (1,J))

        a_p = pm.Lognormal("a_p",a_p_mu,1,shape = (K,1))
        v_p = pm.Lognormal("v_p",v_p_mu,1,shape = (K,1))
        #v_p = pm.Gamma("v_p",mu,2,shape = (K,1))
        
        t_er = pm.Lognormal("t_er",0,1,shape = (K,1))

        mu_kj, sigma_kj, p_kj = q_diffusion(a_i, v_i, a_p, v_p, t_er)

        RT_kj = pm.Normal("RT_kj",mu_kj, sigma_kj, shape = (K,J), observed = RT)
        X_kj = pm.Bernoulli("X_kj", pm.invlogit(p_kj), shape = (K,J), observed = X)

    return qdiffusion, (a_i, v_i, a_p, v_p, t_er, RT_kj, X_kj)

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
            ( 1* h_jk * np.exp(h_jk) - np.exp(2*h_jk) + 1) / (
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


def get_model_vars(K,J, mu=0, data_rt=None, data_ra=None, type=type):
    if type == 1:
        return q_diffusion_model_1_1(K,J, mu, data_rt, data_ra)
    #elif type == 2:
    #    return q_diffusion_model_K_1(K,J, mu, data_rt, data_ra)
    elif type == 3:
        return q_diffusion_model(K,J, mu, data_rt, data_ra)


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

def sample_posterior(model, samples_n, chains):
    return ut.sample_posterior(model, samples_n=samples_n, chains=chains)

