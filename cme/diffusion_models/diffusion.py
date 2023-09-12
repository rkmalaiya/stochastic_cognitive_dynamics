#%%
import pymc as pm
import numpy as np
from pytensor import tensor as at
import cme.utils.common_logging as cl
from cme.utils import common_utils as ut


# enable on-the-fly graph computations
#ae.config.compute_test_value = 'warn'

log = cl.get_logger("diffusion")
eps = 0.01 # for numerical stability
err = 10e-10
max_k = 100

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

    K=at.arange( -at.floor((K_n-1)/2), at.ceil((K_n-1)/2) + 1 )[:,np.newaxis, np.newaxis]
    #print("***********k",K.shape)
    prob_rt_std = ((w + 2*K) * at.exp(- ((w+2*K)*(w+2*K)) /(2*tt)) ).sum(axis=0)

    prob_rt_std = prob_rt_std * 1/at.sqrt(2*np.pi*tt*tt*tt)

    #print("***********prob_rt_std",prob_rt_std.shape)

    return prob_rt_std

def _diffusion_01w_l(tt, w):

    K_m = _get_count_l(tt)
    K_n = 200 #at.max(at.floor(K_m))
    #K_n = at.switch(at.gt(K_n, max_k), max_k, K_n)

    K=at.arange(1,K_n+1)[:,np.newaxis, np.newaxis]

    #print("***********k_l",K.shape)

    prob_rt_std = np.pi * (K * at.exp( - ((K*np.pi)**2 * tt/2) ) * at.sin( K * np.pi * w )).sum(axis=0) #exp becomes zero for large tt, hence adding eps
    
    #for k in K:
    #    prob_rt_std += np.pi * (k * at.exp( - ((k*np.pi)**2 * tt/2) ) * at.sin(k * np.pi * w ))

    #print("***********prob_rt_std",prob_rt_std.shape)
    
    return  prob_rt_std

def _diffusion_01std(t,a,w):

    tt = t/(a**2)
    #lmda = _get_lambda(tt)
    #st = _diffusion_01w_s(tt, w)
    lt = _diffusion_01w_l(tt, w)
    prob_rt_std = lt #at.switch(at.lt(lmda, 0), st, lt)

    # For Stability
    #prob_rt_final = at.switch(at.lt(prob_rt_std,0), 0, prob_rt_std) 
    #prob_rt_final = at.switch(at.isnan(prob_rt_final), 0, prob_rt_final)
    #prob_rt_final = at.switch(at.isinf(prob_rt_final), 0, prob_rt_final)

    prob_rt_final = prob_rt_std

    return prob_rt_final #should return a scaler

def _calculate_RT_logp(DT, V, A, Z):
    
    DT = at.switch(at.le(DT,0), 0, DT) 
 
    prob_rt_std = _diffusion_01std(DT,A,Z/A)
    
    prob_rt = (1 / (A*A)) * (at.exp( (-Z*A*V) - ((V*V) * DT)/2 )) * prob_rt_std
    prob_rt = prob_rt 

    return prob_rt
    

def _diffusion_RT_logp(RT, X, v, a, z, t_er):    

    V = at.switch(at.eq(X,1), -v, v) # -v,v
    A = at.switch(at.eq(X,1), a, a)
    Z = at.switch(at.eq(X,1), 1-z, z) # 1-z, z
    T_er = at.switch(at.eq(X,1), t_er, t_er)

    prob_rt = _calculate_RT_logp(RT-T_er, V, A, Z)
    prob_rt = at.switch(at.isinf(prob_rt), 0, prob_rt) 
    prob_rt = at.switch(at.isnan(prob_rt), 0, prob_rt)
    prob_rt = at.log(prob_rt.sum())

    # eps is used to stabilished log
    total_logp = at.switch(at.eq(prob_rt,0), 0, at.log(prob_rt)) 

    #total_logp = at.switch(at.isinf(total_logp), 0, total_logp) 
    #total_logp = at.switch(at.isnan(total_logp), 0, total_logp) 

    return total_logp
    


def _diffusion_default_priors_noncentral(I):

    with pm.Model() as model:

        m = pm.Normal("m",0,1,shape=4)
        s = pm.Normal("s",0,0.2,shape=4)

        v_pr = pm.Normal("v_pr",0,1,shape=(I,1)) # Drift Rate
        a_pr = pm.Normal("a_pr",0,1,shape=(I,1)) # Boundary
        z_pr = pm.Normal("z_pr",0,1,shape=(I,1)) # Bias
        t_er_pr = pm.Normal("t_er_pr",0,1,shape=(I,1)) # Non-Decision Time


        v = pm.Deterministic("v", m[0] + s[0] * v_pr)
        a = pm.Deterministic("a", at.exp(m[1] + s[1] * a_pr))
        z = pm.Deterministic("z", pm.math.sigmoid(m[2] + s[2] * z_pr))
        t_er = pm.Deterministic("t_er", pm.math.sigmoid(m[3] + s[3] * t_er_pr))
        


        #v = pm.Normal("v",0,1)
        #a = pm.Gamma("a",3,2)
        #z = pm.Beta("z", 1,1)
        #t_er = pm.Beta("ter", 1,1)

    return model, v, a, z, t_er

def _diffusion_default_priors_central(I):
        
    with pm.Model() as model:

        v_m = pm.Normal("v_m", 2,3)
        v_s = pm.HalfNormal("v_s", 2)
        v = pm.Normal("v", v_m, v_s, shape=(I,1)) #v = ae.tensor.tile(v, (1,J))
        #v = pm.Normal("v", 1,1, shape=(I,2)) #v = ae.tensor.tile(v, (1,J))

        #a_m = pm.Gamma("a_m",1.5, 0.75, shape=(1,2))
        a_m = pm.Gamma("a_m",0.1, 0.1)
        a_s = pm.HalfNormal("a_s",0.1)
        a = pm.Gamma("a",a_m,a_s,shape=(I,1))
        #a = pm.Gamma("a",2,2,shape=(I,2))

        z_m = pm.Normal("z_m",0.5,0.5)
        z_s = pm.HalfNormal("z_s",0.05)
        z = pm.LogitNormal("z",z_m,z_s,shape=(I,1)) # z ranges from 0 to a
        #z = pm.Normal("z",1,1,shape=(I,2)) # z ranges from 0 to a

        #z = pm.invlogit(z)

        ter_m = pm.Gamma("ter_m",0.4, 0.2)
        ter_s = pm.HalfNormal("ter_s",0.1)
        t_er = pm.LogNormal("t_er",ter_m,ter_s,shape=(I,1))
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

def _diffusion_default_priors(I, type="Central|NonCentral"):
    
    return _diffusion_default_priors_noncentral(I)

    #if type == "NonCentral":
    #    return _diffusion_default_priors_noncentral(I)
    #else:
    #    return _diffusion_default_priors_central(I)


def get_model(I, obs_X, obs_RT=None):
    
    model, v, a, z, t_er = _diffusion_default_priors(I, type="NonCentral")
       
    
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

    model = get_model(I = X.shape[0], obs_X = X, obs_RT=RT)
    posterior_chain = ut.sample_posterior(model,samples_n, chains,tune, sampler=sampler, acceptance_rate= acceptance_rate, **kwargs)
    return posterior_chain, model

def sample_post_pred_data(posterior_chain, model, samples_n = 100):
    return ut.sample_post_pred(model, posterior_chain, samples_n)

def _test_diffusion_01w():

    log.debug(f"Starting Diffusion test")
    t = at.as_tensor(2.01281869)
    a = at.as_tensor(0.10357323)
    z = at.as_tensor(0.01953739)
    w = z/a

    logp_per_trial = _diffusion_01std(t,a,w).eval()

    log.debug(f"Log per Trial: {np.round(logp_per_trial, decimals=5)}")
    assert np.round(logp_per_trial, decimals=5) == 0.00

def _test_likelihood_using_prior(I, J):
    """
    Testing likelihood function
    """

    log.debug(f"Starting Diffusion test")
    #v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic = pm.draw([v_c,a_c,z_c, t_er_c, v_ic, a_ic, z_ic, t_er_ic])

    X = at.as_tensor(np.random.randint(0,2,(I, J)))
    model, v, a, z, t_er = _diffusion_default_priors(I)
    RT = at.as_tensor(np.random.uniform(0,4,(I,J)))

    #lp = _calculate_RT_logp(RT - t_er, v, a, z)
    #lp_v = at.log(lp.sum()).eval()

    lp_v = _diffusion_RT_logp(RT, X, v, a, z, t_er).eval()

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
    with model:
        posterior_chain = pm.sample(draws=1000, chains=4,tune=500, step=pm.NUTS(max_treedepth=20))
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
