#%%
import pymc as pm
import numpy as np
import aesara as ae
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
    #prob_x = (np.exp(-2*v*a) - np.exp(-2*v*w*a)) / (np.exp(-2*v*a) - 1)
    #prob_rt_std = np.pi * K * np.exp( -((K*np.pi)**2) * RT/2 ) * np.sin( K * np.pi * w )      
    #prob_rt = (1/prob_x) * (1 / a**2) * np.exp( (-w*a*v) - (v**2 * RT)/2 ) * prob_rt_std
    
    t = RT-t_er
    #ae.tensor.subtensor.set_subtensor(t[np.where(t <= 0)], 0)

    #if(t.eval() == 0):
    #    return 0

    prob_rt_std = _diffusion_01w(t, a, w, K)
    prob_rt = (1 / a**2) * np.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std
    
    #prob_rt[np.where(t <= 0)] = 0

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

    samples_rejection = 100 * size[1]
    RT_rvs = np.empty(shape=size)
    #for i_l, v,a,w, t_er in zip(range(size[0]), v,a,w, t_er, strict=True):
    for i_l in zip(range(size[0])):

        RT_arr = np.empty(shape=0)

        while (RT_arr.shape[0] < size[1]):
            
            RT = sp.stats.lognorm.rvs(1, 0,1, size=samples_rejection) + t_er
            u = sp.stats.uniform.rvs(0,1,size=samples_rejection)

            pdf_lognorm = sp.stats.lognorm.pdf(RT,1,0,1)
            pdf_diffusion = _diffusion_logp(RT,v,a,w, t_er)
            
            M = np.round(np.max(pdf_diffusion) + 1)
            
            RT = RT[np.where(u < pdf_diffusion / (M*pdf_lognorm))]
            RT_arr = np.append(RT_arr, RT)
            
        RT_rvs[i_l,:] = RT_arr[0:size[1]] 


    return RT_rvs

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
            "RT_logp",
            *vars,
            logp=logp,
            observed = obs_RT,
            #random=_diffusion_draw,
            #size=(1,1)
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

    #model,_ = diffusion_model_both()
    #prior_chain = pm.sample_prior_predictive(model=model)
    #print("******* Prior RT", prior_chain.prior.RT.shape, " ***** min:", np.min(prior_chain.prior.RT), " ***** max:", np.max(prior_chain.prior.RT))

    for i,j in zip(range(1,I+1), J):
        X = np.random.randint(0,2,j)
        RT = np.random.uniform(0,4,j)

        model_correct, _ = diffusion_model_both(X, RT, sv=False)
        pm.sample(model=model_correct, draws=10, chains=2,tune=10)

        _, model_incorrect = diffusion_model_both(X, RT, sv=False)
        pm.sample(model=model_incorrect, draws=10, chains=2,tune=10)

        #model_correct, _ = diffusion_model_both(X, RT, sv=True)
        #pm.sample(model=model_correct, draws=10, chains=2,tune=10)

        #_, model_incorrect = diffusion_model_both(X, RT, sv=True)
        #pm.sample(model=model_incorrect, draws=10, chains=2,tune=10)

#%%
def archive_fn():
    large_sample = 1000
    model, v,a,w, t_er = _diffusion_default_priors()
    with model: 
        v,a,w, t_er = pm.draw([v,a,w, t_er], draws=large_sample)

    rt_lower, rt_higher, rt_delta = 0.01, 3, 0.1
    #X = np.arange(rt_lower, rt_higher, rt_delta)
    X = np.random.uniform(rt_lower,rt_higher, size=large_sample).round(decimals=2)

    log_cdf = np.zeros(X.shape[0])
    for i_loop, X_i, v_i,a_i,w_i, t_er_i in zip(range(0, X.shape[0]), X, v,a,w, t_er):
        if i_loop == 0:
            log_cdf[i_loop] = _diffusion_logp(X_i, v_i,a_i,w_i, t_er_i)
        else:
            log_cdf[i_loop] = _diffusion_logp(X_i, v_i,a_i,w_i, t_er_i) + log_cdf[i_loop-1]

    log_cdf = log_cdf / log_cdf[X.shape[0] - 1]
    log_cdf

    samples = 100
    rts = np.zeros(samples)
    f = np.random.rand(samples)
    for i_loop in range(samples):
        idx = np.searchsorted(log_cdf, f[i_loop])
        rts[i_loop] = X[idx]
