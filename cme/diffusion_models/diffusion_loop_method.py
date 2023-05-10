#%%
import pymc as pm
import numpy as np
import pytensor as ae
from pytensor import tensor as at
import scipy as sp
import cme.utils.common_logging as cl
from cme.utils import common_utils as ut
import pandas as pd
import os
from pytensor.scan.utils import until

#import pymc.sampling.jax as jx

# enable on-the-fly graph computations
#ae.config.compute_test_value = 'warn'

log = cl.get_logger("diffusion")
eps = 0.01 # for numerical stability
err = 10e-10
max_k = 40
err2 = 10e-29


def _diffusion_sum_inf(K, prob_rt_std, t, w, a):
    
    #a = a.squeeze()
    prob_rt_std_new =  (K * at.exp( - ((K*np.pi)**2 * t/(2*a*a)) ) * at.sin( K * np.pi * w )) #exp becomes zero for large tt, hence adding eps
        
    return [K+1, prob_rt_std + prob_rt_std_new], until(at.le(at.abs(prob_rt_std - prob_rt_std_new), err2)) #or t==0

def _get_count_l(tt):

    K = at.sqrt((-2) * at.log(np.pi * tt * err) / ((np.pi**2) * tt) )
    K = at.switch(at.gt(K, 1/(np.pi * at.sqrt(tt)) ), K, (1 / (np.pi * at.sqrt(tt)) ) ) # based on RWiener package in rlang
    return K

def _get_count_s(tt):
    K = at.sqrt( -2 * tt * at.log( 2 * err* at.sqrt( 2 * np.pi * tt ) ) )
    K = at.switch(at.gt(K, at.sqrt(tt)+1), K, at.sqrt(tt) + 1)
    return K

def _get_lambda(tt):
    
    return 2 + _get_count_s(tt) - _get_count_l(tt)

def _diffusion_01w_s(tt, w):

    K_m = _get_count_s(tt)
    K_n = at.max(at.floor(K_m))
    K_n = at.switch(at.gt(K_n,max_k), max_k, K_n)
    K_n = at.switch(at.isnan(K_n), 1, K_n)

    K=at.arange( -at.floor((K_n-1)/2), at.ceil((K_n-1)/2) + 1 )[:,np.newaxis, np.newaxis]

    prob_rt_std = ((w + 2*K) * at.exp( ((w+2*K)*(w+2*K)) /(2*tt)) ).sum(axis=0)

    prob_rt_std = prob_rt_std * 1/at.sqrt(2*np.pi*tt*tt*tt)

    return prob_rt_std

def _diffusion_01w_l(tt, w):

    K_m = _get_count_l(tt)
    K_n = at.max(at.floor(K_m))
    K_n = at.switch(at.gt(K_n, max_k), max_k, K_n)
    K_n = at.switch(at.isnan(K_n), 1, K_n)

    K=at.arange(1,K_n+1)[:,np.newaxis, np.newaxis]

    prob_rt_std = np.pi * (K * at.exp( - ((K*np.pi)**2 * tt/2) ) * at.sin( K * np.pi * w )).sum(axis=0) #exp becomes zero for large tt, hence adding eps
    return  prob_rt_std

def _diffusion_01std(t,a,w):

    tt = t/(a**2)
    lmda = _get_lambda(tt)
    st = _diffusion_01w_s(tt, w)
    lt = _diffusion_01w_l(tt, w)
    prob_rt_std = at.switch(at.lt(lmda, 0), st, lt)

    # For Stability
    prob_rt_final = at.switch(at.lt(prob_rt_std,0), 0, prob_rt_std) 
    prob_rt_final = at.switch(at.isnan(prob_rt_final), 0, prob_rt_final)
    prob_rt_final = at.switch(at.isinf(prob_rt_final), 0, prob_rt_final)

    return prob_rt_final #should return a scaler



def _calculate_parti_item_logp(dt, x, v, a, z):
    # All parameters received here are scaler.
 
    #print("dt *******",dt.ndim)
    #print("x *******",x.ndim)
    #print("v *******",v.ndim)
    #print("a *******",a.ndim)
    #print("z *******",z.ndim)

    v_i = at.switch(at.eq(x,1),-v,v)
    z_i = at.switch(at.eq(x,1),1-z,z)
    
    #[_, prob_rt_std],_ = ae.scan(_diffusion_sum_inf,
    #       outputs_info = [at.constant(1), at.constant( 0.0, dtype="float64")],
    #        non_sequences=[dt, z_i, a],
    #        n_steps=50
    #       )
    
    #prob_rt_std = prob_rt_std[-1,...]
    
    prob_rt_std = _diffusion_01std(dt, a, z_i)
    prob_rt = np.pi / a*a * at.exp( (-z_i*a*v_i) - (v_i*v_i * dt)/2 ) * prob_rt_std

    return prob_rt

def _calculate_participant_logp(dt, x, v, a, z):
    prob_rt_t,_ = ae.scan(_calculate_parti_item_logp, 
            sequences=[dt,x],
            non_sequences=[v, a, z]
            )
    
    prob_rt = prob_rt_t.sum()

    return prob_rt

def _diffusion_RT_logp(RT, obs_X, v, a, z, ter):    

    DT = at.switch(at.le(RT, ter), 0, RT-ter) 
    prob_rt,_ = ae.scan(_calculate_participant_logp,
                        sequences=[DT, obs_X, v, a, z]
    )

    # eps is used to stabilished log
    prob_rt = at.switch(at.le(prob_rt,0), err, prob_rt) 
    prob_rt = at.switch(at.isinf(prob_rt), err, prob_rt) 
    prob_rt = at.switch(at.isnan(prob_rt), err, prob_rt) 

    total_p = prob_rt.sum()

    total_logp = at.log(total_p) 

    return total_logp 


def _diffusion_default_priors(I):

    with pm.Model() as model:

        m = pm.Normal("m",0,1,shape=4)
        s = pm.Normal("s",0,0.2,shape=4)

        v_pr = pm.Normal("v_pr",0,1,shape=(I,)) # Drift Rate
        a_pr = pm.Normal("a_pr",0,1,shape=(I,)) # Boundary
        z_pr = pm.Normal("z_pr",0,1,shape=(I,)) # Bias
        ter_pr = pm.Normal("t_er_pr",0,1,shape=(I,1)) # Non-Decision Time. Kept 1 dimension here to support DT = RT-ter

        v = m[0] + s[0] * v_pr
        a = at.exp(m[1] + s[1] * a_pr)
        z = pm.math.sigmoid(m[2] + s[2] * z_pr)
        ter = pm.math.sigmoid(m[3] + s[3] * ter_pr)
        
    return model, v, a, z, ter

def get_model(I, obs_X, obs_RT=None):
    
    model, v, a, z, t_er = _diffusion_default_priors(I)
  
    vars_RT = obs_X, v, a, z, t_er

    with model:
        pm.DensityDist(
            "RT",
            *vars_RT,
            logp=_diffusion_RT_logp,
            #random=_draw_RT,
            observed=obs_RT,
            #size = obs_X.shape
        )
        
    return model

def sample_prior_data(I, J, samples_n):
    X=at.zeros(shape=(I,J)).eval()
    model = get_model(I = I, obs_X = X, obs_RT=None)
    return ut.sample_prior(model, samples_n)


def sample_posterior_params(RT, X, samples_n, chains, tune, sampler="PYMC", acceptance_rate=0.90, **kwargs):
    log.debug(f"Data-  I:{RT.shape[0]}, J:{RT.shape[1]}")
    model = get_model(I = X.shape[0], obs_X = X, obs_RT=RT)
    posterior_chain = ut.sample_posterior(model,samples_n, chains,tune, sampler=sampler, acceptance_rate= acceptance_rate, **kwargs)
    return posterior_chain, model

def sample_post_pred_data(posterior_chain, model, samples_n = 100):
    return ut.sample_post_pred(model, posterior_chain, samples_n)

def _test_likelihood_using_prior(I, J):
    """
    Testing likelihood function
    """

    log.debug(f"Starting Diffusion test")
    #v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic = pm.draw([v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic])

    X = at.as_tensor(np.random.randint(0,2,(I, J)))
    _, v, a, z, t_er = _diffusion_default_priors(X.shape[0])
    RT = at.as_tensor(np.random.uniform(0,4,(I,J)))

    lp = _diffusion_RT_logp(RT, X, v, a, z,t_er)

    lp_v = lp.eval()

    log.debug(f"lp: {lp_v}")
    #log.debug(f"priors:{prior_sample}")
    log.debug(f"X:{X.eval()}")
    log.debug(f"RT:{RT.eval()}")
    #log.debug(f"data shape: {RT.eval().shape}")
    
    #assert (lp_v[0] >= 0)#.all()
    log.info("Test Successful!")
    return lp_v, v.eval(), a.eval(), z.eval(), t_er.eval(),  X.eval(), RT.eval() 

#%%
def _quick_test():
    X = np.random.randint(0,2,(9,5))
    RT = np.random.uniform(0,4,(9,5))

    log.debug(f"Starting Diffusion test")
    prior_chain = None #sample_prior_data(I=8,samples_n =5)
    model = get_model(I = X.shape[0], obs_X = X, obs_RT=RT)
    posterior_chain = pm.sample(model=model, draws=12, chains=2,tune=10)
    post_pred_chain = None #pm.sample_posterior_predictive(posterior_chain,model)
    #with model:
    #    posterior_jax = jx.sample_numpyro_nuts(13, tune = 20, chains=2)
    #log.debug(f"Posterior model v_correct {posterior_chain.posterior.v.shape}")
    
    return prior_chain, posterior_chain #, posterior_jax, post_pred_chain


#%%
if __name__ == "__main__":
    
    log.debug("Starting test")
    #_test_diffusion_01w()
    lp = _test_likelihood_using_prior(10,4)
    
    prior_chain, posterior_chain, posterior_jax, post_pred_chain = _quick_test()

    #assert prior_chain.prior.v.shape == (1,10,100,2)
    assert posterior_chain.posterior.v.shape == (2,12,70,2)
    #assert posterior_jax.posterior.v.shape == (2,13,70,2)
    #assert post_pred_chain.posterior_predictive.RT.shape == (2,12,70,5)

    
# %%
