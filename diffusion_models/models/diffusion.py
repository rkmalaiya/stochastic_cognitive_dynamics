#%%
import pymc as pm
import numpy as np
import pytensor as ae
from pytensor import tensor as at
import scipy as sp
import diffusion_models.utils.common_logging as cl
from diffusion_models.utils import common_utils as ut


log = cl.get_logger("diffusion")
eps = 0.001 # for numerical stability
err = 0.001
tune = 200

def _diffusion_01w(t, a, w):

    K = at.arange(10) #should be calculated using Navarro and Fuss 2009 paper.

    tt=t/(a**2)
    prob_rt_std = np.pi * K * (at.exp( - ((K*np.pi)**2 * tt/2) ) + eps) * at.sin( K * np.pi * w )
    
    return prob_rt_std.sum()

def _diffusion(t, v, a, w):
        
    prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
    prob_rt_std,_ = ae.scan(fn = _diffusion_01w, 
                            sequences=t, 
                            non_sequences=(a, w))
    
    #prob_rt = (1 / a**2) * at.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std
    prob_rt = (1 / a*a) * at.exp( (-w*a*v) - (v*v * t)/2 ) * prob_rt_std
    
    return prob_rt/prob_X

def _individual_logp(t, X, v, a, w):

    t_correct = t[X]
    t_incorrect = t[1-X]

    total_logp = _diffusion(t_correct, -v, a, 1-w) + _diffusion(t_incorrect, v, a, w)
    return total_logp

def _diffusion_RT_logp(RT, X, v, a, w, t_er):
    t = RT-t_er
    t = at.switch(at.le(t,0), eps,t)
    w = at.switch(at.ge(w,1), 0.99,w) # to avoid instability during intial evaluation.

    X = at.as_tensor(X)
    prob_rt, _ = ae.scan(fn = lambda t_l, X_l, v_l, a_l, w_l: at.switch(at.le(t, 0), eps, _individual_logp(t_l, X_l, v_l, a_l, w_l)), 
                        sequences=[t,X, v, a, w])
    
    return prob_rt.sum()


def _diffusion_default_priors(parti_n):
    with pm.Model() as model:
        v = pm.Normal("v",1,1,shape=(parti_n,1)) #v = ae.tensor.tile(v, (1,J))
        a = pm.Gamma("a",2,2,shape=(parti_n,1))
        z = pm.Uniform("z", 0,a,shape=(parti_n,1)) # z ranges from 0 to a
        w = z/a #pm.Deterministic("w", z/a)
        t_er = pm.HalfNormal("t_er",2,shape=(parti_n,1))
        
    return model, v,a,w, t_er

def _diffusion_draw(v,a,w, t_er, rng=None, size=None):
    sample_counter = 1000

    # To get how many response time samples are required for each participants.
    if len(size) > 1: 
        J = size[1]
    else:
        J = size[0]

    samples_rejection = np.max((100 * J, 100))
    RT_rvs = np.empty(shape=size)
    #for i_l in zip(range(size[0])):

    RT_arr = np.empty(shape=0)

    while (RT_arr.shape[0] < J):
        
        RT = sp.stats.lognorm.rvs(1,0,1, size=samples_rejection) + t_er
        u = sp.stats.uniform.rvs(0,1,size=samples_rejection)

        pdf_lognorm = sp.stats.lognorm.pdf(RT,1,0,1)
        pdf_diffusion = _diffusion_RT_logp(RT,v,a,w, t_er).eval()
        
        M = np.round(np.max(pdf_diffusion) + 1)
        #log.debug(f"M: {M}:{np.max(pdf_diffusion)}")
        idx = np.less_equal(u, pdf_diffusion / (M*pdf_lognorm))
        RT_filter = RT[idx]
        RT_arr = np.append(RT_arr, RT_filter)
        sample_counter -= 1
        if(sample_counter <= 0):
            raise Exception(f"Could not sample for v:{v}, a:{a}, w:{w}, t_er:{t_er}, RT:{RT}, pdf_diffusion:{pdf_diffusion}")
    #RT_rvs[i_l,:] = RT_arr[0:size[1]] 

    return RT_arr[0:J] #RT_rvs

def _diffusion_model(obs_X = None, obs_RT=None):
    
    model, v,a,w, t_er = _diffusion_default_priors(obs_X.shape[0])
    vars = obs_X, v,a,w, t_er    

    with model:
        pm.DensityDist(
            "RT",
            *vars,
            logp=_diffusion_RT_logp,
            observed=obs_RT
        )
        
    return model

def sample_prior_data(samples_n = 100):
    pass

def sample_posterior_params(RT, X, samples_n, chains, tune=tune, sampler="PYMC", acceptance_rate = 0.85):

    model = _diffusion_model(obs_X = X, obs_RT=RT)
    posterior_chain = ut.sample_posterior(model,samples_n, chains,tune, sampler=sampler)
    return posterior_chain, model

def sample_post_pred_data(posterior_chain, model, samples_n = 100):
    pass

def _test_edge_cases():

    #v:-1.2002378916969234, a:0.017329851659947146, w:0.8280975715451416, t_er:0.6342687555621264, RT:[1.15521305 3.05237406 1.16819656 ... 2.47025695 2.03251366 2.46823881]

    v = np.repeat([-0.9249241152935039], 4)[:,np.newaxis]
    a = np.repeat([0.023750197074163027], 4)[:,np.newaxis]
    w = np.repeat([0.44120144255887783] , 4)[:,np.newaxis]
    #t_er=0.7100510973761837], 4)
    t_er=np.repeat([0.07100510973761837], 4) [:,np.newaxis]
    RTs = [[0.09, 1.39385687, 1.55661403, 1.88891806, 2.23878542, 0.92574541, 3.56532722],
            [0.19, 1.59385687, 2.55661403, 3.88891806, 1.23878542, 1.92574541, 5.56532722]
    ]
    
    X = np.random.randint(0,2,(4,5))
    RT = np.random.uniform(0,4,(4,5))
    
    lp = _diffusion_RT_logp(RT, X, v,a,w,t_er)

    log.debug(f"lp: {lp.eval()}")


#%%
if __name__ == "__main__":
    J = (2,20)
    log.debug("Starting test")

    #_test_edge_cases()

    for j in J:
        X = np.random.randint(0,2,(700,j))
        RT = np.random.uniform(0,4,(700,j))

        log.debug(f"Starting Diffusion (sv=False) for {j} trials")
        model = _diffusion_model(obs_X = X, obs_RT=RT)
        posterior_chain = pm.sample(model=model, draws=10, chains=2,tune=10)
        with model:
            posterior = pm.sampling_jax.sample_numpyro_nuts(10, tune = 10, chains=2)
        log.debug(f"Posterior model_correct {posterior_chain.posterior.v.shape}")

