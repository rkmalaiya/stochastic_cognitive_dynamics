#%%
import pymc as pm
import numpy as np
import pytensor as ae
from pytensor import tensor as at
import scipy as sp
import diffusion_models.utils.common_logging as cl
from diffusion_models.utils import common_utils as ut
import pandas as pd
import os
import pymc.sampling.jax as jx

# enable on-the-fly graph computations
# ae.config.compute_test_value = 'warn'

log = cl.get_logger("diffusion")
eps = 0.0001 # for numerical stability
err = 0.001

def _get_count(t):
    return at.sqrt((-2) * at.log(np.pi * t * err) / ((np.pi**2) * t) )

def _diffusion_01w(t, a, w):

    K_n = 10 #at.max(_get_count(t))
    
    K=at.arange(1,K_n)[:,np.newaxis, np.newaxis]
    tt=t/(a**2)
    prob_rt_std = np.pi * K * (at.exp( - ((K*np.pi)**2 * tt/2) )) * at.sin( K * np.pi * w ) #exp becomes zero for large tt, hence adding eps
    x_printed_8 = ae.printing.Print('K')(K )
    
    x_printed_7 = ae.printing.Print('tt')(tt )
    x_printed_6 = ae.printing.Print('t')(t )
    x_printed_5 = ae.printing.Print('w')(w )
    x_printed_8 = ae.printing.Print('prob_rt_std for all Ks')(prob_rt_std )
    x_printed_9 = ae.printing.Print('prob_rt_final for all Ks')(prob_rt_std.sum(axis=0) )
    
    prob_rt_final = prob_rt_std #at.switch(at.le(prob_rt_std,0),0,prob_rt_std) #at.switch(at.le(t,0),0, prob_rt_std )
    return prob_rt_final.sum(axis=0) #should return a scaler

def _diffusion_X_logp(X, v, a, z):
    w = z/a
    prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
    return prob_X

def _diffusion_RT_logp(RT, v, a, z, t_er):
    
    #X = at.as_tensor(X)
        
    w = z #z/a  
    #w = at.switch(at.ge(w,1), 0.99,w) # to avoid instability during intial evaluation.

    t = RT-t_er
    t = at.switch(at.le(t,0),0,t)
    x_printed_2 = ae.printing.Print('RT-t_re')(t)

    #prob_rt = _diffusion(t, v, w, a)
    
    prob_rt_std = _diffusion_01w(t,a,w)
    
    x_printed_3 = ae.printing.Print('prob_rt_std all trial')(prob_rt_std)

    #prob_rt = (1 / a**2) * at.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std
    prob_rt = (1 / a*a) * at.exp( (-w*a*v) - (v*v * t)/2 ) * prob_rt_std
    
    
    
    #all though care has been taken to not process pdf for -ve time that results in -ve pdf, some -ve pdf are still creeping up
    prob_rt = at.switch(at.le(t,0),0,prob_rt) #Removing pdfs for t <= 0 because t <=0 is not supported

    x_printed_13 = ae.printing.Print('per individual all trial logp')(prob_rt)

    total_logp = prob_rt.sum(axis=1)

    x_printed_12 = ae.printing.Print('***all individual final sum logp')(prob_rt)
    return total_logp 


def _diffusion_default_priors(I, X):
        
    with pm.Model() as model:
        
        v_c = pm.LogNormal("v_c",0,1,shape=(I,1)) #v = ae.tensor.tile(v, (1,J))
        a_c = pm.Gamma("a_c",2,2,shape=(I,1))
        z_c = pm.Beta("z_c", 1,1,shape=(I,1)) # z ranges from 0 to a
        t_er_c = pm.HalfNormal("t_er_c",2,shape=(I,1))
        
        v_ic = pm.LogNormal("v_ic",0,1,shape=(I,1)) #v = ae.tensor.tile(v, (1,J))
        a_ic = pm.Gamma("a_ic",2,2,shape=(I,1))
        z_ic = pm.Beta("z_ic", 1,1,shape=(I,1)) # z ranges from 0 to a
        t_er_ic = pm.HalfNormal("t_er_ic",2,shape=(I,1))

        v = at.switch(at.eq(X,1), -v_c, v_ic)
        a = at.switch(at.eq(X,1), a_c, a_ic)
        z = at.switch(at.eq(X,1), z_c, z_ic)
        t_er = at.switch(at.eq(X,1), t_er_c, t_er_ic)
        
        x_printed_12 = ae.printing.Print('v')(v)
        x_printed_14 = ae.printing.Print('a')(a)
        x_printed_15 = ae.printing.Print('z')(z)
        x_printed_16 = ae.printing.Print('t_er')(t_er)

    return model, v, a, z, t_er

def get_model(I, obs_X = None, obs_RT=None):
    
    model, v, a, z, t_er = _diffusion_default_priors(I, obs_X)
    vars_RT = v, a, z, t_er
    vars_X = v, a, z

    with model:
        pm.DensityDist(
            "X_pdf",
            *vars_X,
            logp=_diffusion_X_logp,
            observed=obs_X
        )

    with model:
        pm.DensityDist(
            "RT_pdf",
            *vars_RT,
            logp=_diffusion_RT_logp,
            observed=obs_RT
        )
        
    return model

def sample_prior_data(samples_n = 100):
    pass

def sample_posterior_params(RT, X, samples_n, chains, tune, sampler="PYMC", acceptance_rate=0.90):

    model = get_model(I = X.shape[0], obs_X = X, obs_RT=RT)
    posterior_chain = ut.sample_posterior(model,samples_n, chains,tune, sampler=sampler, acceptance_rate= acceptance_rate)
    return posterior_chain, model

def sample_post_pred_data(posterior_chain, model, samples_n = 100):
    pass

def _test_diffusion_01w():

    log.debug(f"Starting Diffusion test")
    t = at.as_tensor(2.01281869)
    a = at.as_tensor(0.10357323)
    z = at.as_tensor(0.01953739)
    w = z/a

    logp_per_trial = _diffusion_01w(t,a,w).eval()

    log.debug(f"Log per Trial: {np.round(logp_per_trial, decimals=5)}")
    assert np.round(logp_per_trial, decimals=5) == 0.00

def _test_likelihood_using_error():
    v = at.as_tensor([[-4.7455195,  3.5246989, -4.7455195,  3.5246989, -4.7455195]])
    a = at.as_tensor([[0.10100832, 0.48082409, 0.10100832, 0.48082409, 0.10100832]])
    z = at.as_tensor([[0.03027856, 0.33936817, 0.03027856, 0.33936817, 0.03027856]])
    t_er = at.as_tensor([[2.17066536, 0.17037338, 2.17066536, 0.17037338, 2.17066536]])

    RT = at.as_tensor([[0.36327581, 0.77126385, 1.02809235, 3.05141406, 0.37536871]])
    X = at.as_tensor([[1, 0, 1, 0, 1]])

    rt_logp = _diffusion_RT_logp(RT, v, a, z, t_er)
    log.debug(f"rt_logp:{rt_logp}")

    assert rt_logp >= 0
    log.info("Test Successful!")

def _test_likelihood_using_prior(parti_n):
    """
    Testing likelihood function
    """

    log.debug(f"Starting Diffusion test")
    #v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic = pm.draw([v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic])

    X = at.as_tensor(np.random.randint(0,2,(parti_n,5)))
    model, v, a, z, t_er = _diffusion_default_priors(X.shape[0], X)
    RT = at.as_tensor(np.random.uniform(0,4,(parti_n,5)))

    lp = _diffusion_RT_logp(RT, v, a, z, t_er)

    lp_v = lp.eval()

    log.debug(f"lp: {lp_v}")
    #log.debug(f"priors:{prior_sample}")
    log.debug(f"X:{X.eval()}")
    log.debug(f"RT:{RT.eval()}")
    #log.debug(f"data shape: {RT.eval().shape}")
    
    assert (lp_v[0] >= 0)#.all()
    log.info("Test Successful!")

def _quick_test():
    X = np.random.randint(0,2,(70,5))
    RT = np.random.uniform(0,4,(70,5))

    log.debug(f"Starting Diffusion test")
    model = get_model(I = X.shape[0], obs_X = X, obs_RT=RT)
    posterior_chain = pm.sample(model=model, draws=10, chains=2,tune=10)
    with model:
        posterior_jax = jx.sample_numpyro_nuts(10, tune = 20, chains=2)
    log.debug(f"Posterior model v_correct {posterior_chain.posterior.v_c.shape}")
    assert posterior_chain.posterior.v_c.shape == (2,10,70,1)
    assert posterior_jax.posterior.v_c.shape == (2,10,70,1)


#%%
if __name__ == "__main__":
    
    log.debug("Starting test")
    _test_diffusion_01w()
    _test_likelihood_using_prior(10)
    _quick_test()
    
# %%
