#%%
import pymc as pm
import numpy as np
from aesara import tensor as at
import scipy as sp
import diffusion_models.utils.common_logging as cl

log = cl.get_logger("diffusion")

def _diffusion_01w(t, a, w, K):
    tt=t/a**2
    prob_rt_std = np.pi * K * np.exp( -((K*np.pi)**2) * tt/2 ) * np.sin( K * np.pi * w )
    return prob_rt_std

def _diffusion_logp(RT, v, a, w, t_er):
    #t_max = np.max(RT)
    K = 1 #np.sqrt(-2 * np.log(np.pi * t_max * err) / (np.pi**2 * t_max) ) + 1
    
    t = RT-t_er
    #prob_rt_std = at.switch(pm.math.le(t,0), 0, _diffusion_01w(t, a, w, K))
    prob_rt_std = _diffusion_01w(t, a, w, K)
    prob_rt = (1 / a**2) * np.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std

    return prob_rt
    
def _diffusion_sv_logp(RT, v, a, w, t_er, sv):
    K=1
    t = RT-t_er
    #t[np.where(t <= 0)] = 0

    prob_rt_std = _diffusion_01w(t, a, w, K)
    prob_rt = np.exp(np.log(prob_rt_std) + ((a*w*a*sv)**2 - 2*a*v*w*a - (v**2)*t ) / (2*(sv**2)*t+2) )
    prob_rt = prob_rt / (np.sqrt( (sv**2)*t+1 )) / a**2
    return prob_rt


def _diffusion_default_priors(correct_resp):
    with pm.Model() as model:
        v = pm.Lognormal("v",0,1) #v = ae.tensor.tile(v, (1,J))
        a = pm.Lognormal("a",0,1)
        w = pm.Beta("w", 1,1)
        t_er = pm.Lognormal("t_er",0,1)
        if(correct_resp):
            v = -v
            w = 1-w

    return model, v,a,w, t_er

def _diffusion_sv_priors(correct_resp):
    model, v,a,w, t_er = _diffusion_default_priors(correct_resp)
    with model:
        sv = pm.HalfNormal("sv", 2)
    return model, v,a,w, t_er, sv


def _diffusion_draw(v,a,w, t_er, rng=None, size=None):
    sample_counter = 10000
    
    samples_rejection = np.max((100 * size[0], 100000))
    RT_rvs = np.empty(shape=size)
    #for i_l in zip(range(size[0])):

    RT_arr = np.empty(shape=0)

    while (RT_arr.shape[0] < size[0]):
        
        RT = sp.stats.lognorm.rvs(1, 0,1, size=samples_rejection) + t_er
        u = sp.stats.uniform.rvs(0,1,size=samples_rejection)

        pdf_lognorm = sp.stats.lognorm.pdf(RT,1,0,1)
        pdf_diffusion = _diffusion_logp(RT,v,a,w, t_er)
        
        M = np.round(np.max(pdf_diffusion) + 1)
        #log.debug(f"M: {M}:{np.max(pdf_diffusion)}")
        
        RT = RT[np.where(u < pdf_diffusion / (M*pdf_lognorm))]
        RT_arr = np.append(RT_arr, RT)
        sample_counter -= 1
        if(sample_counter <= 0):
            raise Exception(f"Could not sample for v:{v}, a:{a}, t_er:{t_er}, RT:{RT}")
    #RT_rvs[i_l,:] = RT_arr[0:size[1]] 

    return RT_arr[0:size[0]] #RT_rvs

def _diffusion_model(obs_X = None, obs_RT=None, sv = False, correct_resp = False):
    
    if(sv):
        model, v,a,w, t_er, sv = _diffusion_sv_priors(correct_resp)
        vars = v,a,w, t_er, sv
        logp = _diffusion_sv_logp
    else:
        model, v,a,w, t_er = _diffusion_default_priors(correct_resp)
        vars = v,a,w, t_er    
        logp = _diffusion_logp
    

    with model:
        pm.DensityDist(
            "RTs",
            *vars,
            logp=logp,
            observed = obs_RT,
            random=_diffusion_draw,
            size=(1,) if obs_RT is None else obs_RT.shape
        )
        
    return model

def diffusion_model_both(obs_X=None, obs_RT=None, sv = False):
    
    if obs_X is not None: 
        if len(obs_X.shape) > 1:
            raise Exception("Observed data can either be None or per participant vector")
    
        RT = obs_RT[obs_X>0]
        X = obs_X[obs_X>0]
        model_correct = _diffusion_model(X, RT, sv,correct_resp = True)

        RT = obs_RT[np.invert(obs_X>0)]
        X = obs_X[np.invert(obs_X>0)]
        model_incorrect = _diffusion_model(X, RT, sv,correct_resp = False)

    else:
        model_correct = _diffusion_model(correct_resp=True)
        model_incorrect = _diffusion_model(correct_resp=False)
    
    return model_correct, model_incorrect
        

#%%
if __name__ == "__main__":
    I,J = 5,(4,2,3,5,6)
    log.debug("Starting test")
    model,_ = diffusion_model_both()
    prior_chain = pm.sample_prior_predictive(model=model)
    log.debug(f"Prior RT {prior_chain.prior.RTs.shape}")#, " ***** min:", np.min(prior_chain.prior.RTs), " ***** max:", np.max(prior_chain.prior.RTs))

    for i,j in zip(range(1,I+1), J):
        X = np.random.randint(0,2,j)
        RT = np.random.uniform(0,4,j)

        log.debug(f"Starting Diffusion (X=1) (sv=False) for {i}, {j}")
        model_correct, _ = diffusion_model_both(X, RT, sv=False)
        posterior_chain = pm.sample(model=model_correct, draws=10, chains=2,tune=10)
        log.debug(f"Posterior model_correct {posterior_chain.posterior.v.shape}")

        log.debug(f"Starting Diffusion (X=0) (sv=False) for {i}, {j}")
        _, model_incorrect = diffusion_model_both(X, RT, sv=False)
        pm.sample(model=model_incorrect, draws=10, chains=2,tune=10)

        #log.debug(f"Starting Diffusion (X=1) (sv=True) for {i}, {j}")
        #model_correct, _ = diffusion_model_both(X, RT, sv=True)
        #pm.sample(model=model_correct, draws=10, chains=2,tune=10)

        #log.debug(f"Starting Diffusion (X=0) (sv=True) for {i}, {j}")
        #_, model_incorrect = diffusion_model_both(X, RT, sv=True)
        #pm.sample(model=model_incorrect, draws=10, chains=2,tune=10)
    
    with model_correct:
        postr_pred_chain = pm.sample_posterior_predictive(trace=posterior_chain)
    log.debug(f"Posterior RT {postr_pred_chain.posterior_predictive.RTs.shape}")


# %%
