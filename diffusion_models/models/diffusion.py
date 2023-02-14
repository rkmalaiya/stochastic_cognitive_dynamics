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
#ae.config.compute_test_value = 'warn'

log = cl.get_logger("diffusion")
eps = 0.05 # for numerical stability
err = 10e-10
max_k = 20

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

def _diffusion_01w_s(tt, a, w):

    K_m = _get_count_s(tt)
    K_n = at.max(at.floor(K_m))
    K_n = at.switch(at.gt(K_n,max_k), max_k, K_n)

    x_printed_8 = ae.printing.Print('s K_n')(K_n)

    K=at.arange( -at.floor((K_n-1)/2), at.ceil((K_n-1)/2) + 1 )[:,np.newaxis, np.newaxis]
    x_printed_8 = ae.printing.Print('s K_m')(K_m)
    x_printed_7 = ae.printing.Print('s tt')(tt )

    prob_rt_std = at.log(w + 2*K) + (( (w+2*K)*(w+2*K)/(2*tt) ))
    prob_rt_std = prob_rt_std * 1/(at.log(at.sqrt(2*np.pi*tt*tt*tt)))

    x_printed_8 = ae.printing.Print('s prob_rt_std for each Ks')(prob_rt_std )
    return prob_rt_std.sum(axis=0)

def _diffusion_01w_l(tt, a, w):

    K_m = _get_count_l(tt)
    K_n = at.max(at.floor(K_m))
    K_n = at.switch(at.gt(K_n,max_k), max_k, K_n)

    x_printed_8 = ae.printing.Print('l K')(K_n)

    K=at.arange(1,K_n+1)[:,np.newaxis, np.newaxis]
    x_printed_8 = ae.printing.Print('l K_m')(K_m)
    x_printed_7 = ae.printing.Print('l tt')(tt )

    prob_rt_std_sin = at.log(at.sin( K * np.pi * w ).sum(axis=0))
    prob_rt_std = at.log(K).sum(axis=0) + (( - ((K*np.pi)**2 * tt/2) )).sum(axis=0) + prob_rt_std_sin #exp becomes zero for large tt, hence adding eps
    x_printed_8 = ae.printing.Print('l prob_rt_std for each Ks')(prob_rt_std )
    return  at.log(np.pi) + prob_rt_std

def _diffusion_01w(t,a,w):

    tt = t/(a**2)
    tt= at.switch(tt <= eps, eps, tt)
    
    #prob_rt_std = _diffusion_01w_l(tt, a, w)
    prob_rt_std = at.switch(at.lt(_get_lambda(tt), 0), _diffusion_01w_s(tt, a, w), _diffusion_01w_l(tt, a, w))

    x_printed_5 = ae.printing.Print('w')(w )
   
    
    prob_rt_final = prob_rt_std.sum(axis=0) #at.switch(at.le(prob_rt_std,0),0,prob_rt_std) #at.switch(at.le(t,0),0, prob_rt_std )
    x_printed_9 = ae.printing.Print('prob_rt_final summed over all Ks')(prob_rt_final )

    return prob_rt_final #should return a scaler



def _diffusion_X_logp(X, v, a, z):
    w = z#/a
    prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
    return prob_X

def _RT_logp(RT, obs_X, v, a, z, t_er):
    
    #X = at.as_tensor(X)

    V = at.switch(at.eq(obs_X,1), -v[:,[0]], v[:,[0]])
    A = at.switch(at.eq(obs_X,1), a[:,[0]], a[:,[0]])
    Z = at.switch(at.eq(obs_X,1), 1-z[:,[0]], z[:,[0]])
    T_er = at.switch(at.eq(obs_X,1), t_er[:,[0]], t_er[:,[0]])
    
    x_printed_12 = ae.printing.Print('v')(V)
    x_printed_14 = ae.printing.Print('a')(A)
    x_printed_15 = ae.printing.Print('z')(Z)
    x_printed_16 = ae.printing.Print('t_er')(T_er)


    W = Z/A #z/a  
    #w = at.switch(at.ge(w,1), 0.99,w) # to avoid instability during intial evaluation.

    DT = RT-T_er
    #DT = 
    x_printed_2 = ae.printing.Print('RT-t_re')(DT)

    #prob_rt = _diffusion(t, v, w, a)
    
    prob_rt_std = at.switch(at.le(DT,0),0, _diffusion_01w(DT,A,W))
    
    x_printed_3 = ae.printing.Print(f'prob_rt_std all {obs_X.shape}')(prob_rt_std)

    #prob_rt = (1 / a**2) * at.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std
    prob_rt = at.log(1 / A*A) + ( (-W*A*V) - (V*V * DT)/2 ) * prob_rt_std
    
    
    #prob_X = _diffusion_X_logp(obs_X, V, A, Z)
    #all though care has been taken to not process pdf for -ve time that results in -ve pdf, some -ve pdf are still creeping up
    prob_rt = at.switch(at.le(DT,0),0,prob_rt) #Removing pdfs for t <= 0 because t <=0 is not supported

    x_printed_13 = ae.printing.Print('per individual all trial logp')(prob_rt)

    return prob_rt

def _diffusion_RT_logp(RT, obs_X, v, a, z, t_er):    

    prob_rt = _RT_logp(RT, obs_X, v, a, z, t_er)

    total_logp = prob_rt.sum(axis=1) 

    x_printed_12 = ae.printing.Print('***all individual final sum logp')(prob_rt)
    return total_logp 



def _diffusion_default_priors(I, X):

    with pm.Model() as model:

        v_m = pm.Normal("v_m", 2,3)
        v_s = pm.HalfNormal("v_s",2)
        v = pm.Normal("v", v_m, v_s**2, shape=(I,1)) #v = ae.tensor.tile(v, (1,J))
        #v = pm.Normal("v", 1,1, shape=(I,2)) #v = ae.tensor.tile(v, (1,J))

        #a_m = pm.Gamma("a_m",1.5, 0.75, shape=(1,2))
        a_m = pm.Gamma("a_m",1.5,0.75)
        #a_s = pm.HalfNormal("a_s",0.1)
        #a_s = pm.TruncatedNormal("a_s",5,1)
        a = pm.Gamma("a",a_m,1,shape=(I,1))
        #a = pm.Gamma("a",2,2,shape=(I,2))

        z_m = pm.Normal("z_m",0.5,0.5)
        z_s = pm.HalfNormal("z_s",0.05)
        z = pm.LogitNormal("z",z_m,z_s**2,shape=(I,1)) # z ranges from 0 to a
        #z = pm.Normal("z",1,1,shape=(I,2)) # z ranges from 0 to a

        #z = pm.invlogit(z)

        #ter_m = pm.Gamma("ter_m",0.4, 0.2)
        #ter_s = pm.HalfNormal("ter_s",1)
        t_er = pm.LogNormal("t_er",0.1,1,shape=(I,1))
        #t_er = pm.LogNormal("t_er",ter_m,ter_s,shape=(I,1))
        #t_er = pm.Normal("t_er",1,1,shape=(I,2))
        
        #X = pm.Deterministic("X", X)

        #v_ic = pm.LogNormal("v_ic",0,1,shape=(I,1)) #v = ae.tensor.tile(v, (1,J))
        #a_ic = pm.Gamma("a_ic",2,2,shape=(I,1))
        #z_ic = pm.Beta("z_ic", 1,1,shape=(I,1)) # z ranges from 0 to a
        #t_er_ic = pm.HalfNormal("t_er_ic",2,shape=(I,1))

        #V = at.switch(at.eq(X,1), -v[:,[0]], v[:,[1]])
        #A = at.switch(at.eq(X,1), a[:,[0]], a[:,[1]])
        #Z = at.switch(at.eq(X,1), 1-z[:,[0]], z[:,[1]])
        #T_er = at.switch(at.eq(X,1), t_er[:,[0]], t_er[:,[1]])
        
        

    return model, v, a, z, t_er

def _draw_RT(*args, rng, size):
    obs_X, v, a, z, t_er = args
    I, J = obs_X.shape
    max_iter = 10000
    
    #RT_rand = rng.standard_normal(size)
    RT_rvs = np.empty(shape=(I,J))
    samples_rejection = J #100 * J

    for i_l in range(I):
        RT_arr = np.empty(shape=0)
       
        iter=0
        while (RT_arr.shape[0] < J):
            
            RT_rand = sp.stats.lognorm.rvs(1, 0,1, size=samples_rejection)
            u = sp.stats.uniform.rvs(0,1,size=samples_rejection)

            pdf_lognorm = sp.stats.lognorm.pdf(RT_rand,1,0,1)
            pdf_diffusion = _RT_logp(RT_rand,obs_X[[i_l],:],v[[i_l],:],a[[i_l],:],z[[i_l],:],t_er[[i_l],:]).eval()
            
            M = np.round(np.max(pdf_diffusion) + 1)
            

            #pdf_X = _diffusion_X_logp(obs_X,v,a,z).eval()
            #print("*****I_L",i_l)
            #print("*****M",M)
            #print("*****pdf_RT",pdf_diffusion)
            RT_rand_accept = RT_rand[np.where(u < pdf_diffusion[0,:] / (M*pdf_lognorm))]
            
            
            RT_arr = np.append(RT_arr, RT_rand_accept)
            
            iter += 1
            if iter > max_iter:
                raise Exception("Could not find samples")

        RT_rvs[i_l,:] = RT_arr[0:J] 
    print("*****RT", RT_rvs.shape)

    return RT_rvs

def get_model(I, obs_X, obs_RT=None):
    
    model, v, a, z, t_er = _diffusion_default_priors(I, obs_X)
       
    
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


def sample_posterior_params(RT, X, samples_n, chains, tune, sampler="PYMC", acceptance_rate=0.90):

    model = get_model(I = X.shape[0], obs_X = X, obs_RT=RT)
    posterior_chain = ut.sample_posterior(model,samples_n, chains,tune, sampler=sampler, acceptance_rate= acceptance_rate)
    return posterior_chain, model

def sample_post_pred_data(posterior_chain, model, samples_n = 100):
    return ut.sample_post_pred(model, posterior_chain, samples_n)

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

def _test_likelihood_using_prior(I, J):
    """
    Testing likelihood function
    """

    log.debug(f"Starting Diffusion test")
    #v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic = pm.draw([v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic])

    X = at.as_tensor(np.random.randint(0,2,(I, J)))
    model, v, a, z, t_er = _diffusion_default_priors(X.shape[0], X)
    RT = at.as_tensor(np.random.uniform(0,4,(I,J)))

    lp = _RT_logp(RT, X, v, a, z, t_er)

    lp_v = lp.eval()

    log.debug(f"lp: {lp_v}")
    #log.debug(f"priors:{prior_sample}")
    log.debug(f"X:{X.eval()}")
    log.debug(f"RT:{RT.eval()}")
    #log.debug(f"data shape: {RT.eval().shape}")
    
    #assert (lp_v[0] >= 0)#.all()
    log.info("Test Successful!")
    return v.eval(), a.eval(), z.eval(), t_er.eval(), lp_v, X.eval(), RT.eval() 

#%%
def _quick_test():
    X = np.random.randint(0,2,(9,5))
    RT = np.random.uniform(0,4,(9,5))

    log.debug(f"Starting Diffusion test")
    prior_chain = None #sample_prior_data(I=8,samples_n =5)
    model = get_model(I = X.shape[0], obs_X = X, obs_RT=RT)
    posterior_chain = pm.sample(model=model, draws=12, chains=2,tune=10)
    post_pred_chain = pm.sample_posterior_predictive(posterior_chain,model)
    with model:
        posterior_jax = jx.sample_numpyro_nuts(13, tune = 20, chains=2)
    log.debug(f"Posterior model v_correct {posterior_chain.posterior.v.shape}")
    
    return prior_chain, posterior_chain, posterior_jax, post_pred_chain


#%%
if __name__ == "__main__":
    
    log.debug("Starting test")
    #_test_diffusion_01w()
    #_test_likelihood_using_prior(10)
    
    prior_chain, posterior_chain, posterior_jax, post_pred_chain = _quick_test()

    #assert prior_chain.prior.v.shape == (1,10,100,2)
    assert posterior_chain.posterior.v.shape == (2,12,70,2)
    assert posterior_jax.posterior.v.shape == (2,13,70,2)
    assert post_pred_chain.posterior_predictive.RT.shape == (2,12,70,5)

    
# %%
