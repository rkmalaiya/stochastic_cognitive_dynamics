#%%
import pymc as pm
import numpy as np
import aesara as ae
from aesara import tensor as at
import scipy as sp
import diffusion_models.utils.common_logging as cl
from diffusion_models.utils import common_utils as ut
import arviz as az
import xarray as xa

log = cl.get_logger("diffusion")
eps = 0.001 # for numerical stability
err = 0.001
tune = 200

def _diffusion_01w(t, a, w, K):
    #K_n = at.sqrt(-2 * at.log(np.pi * t * err) / (np.pi**2 * t) ) + 1
    #K = np.asarray([1,2,3,4,5]) #at.arange(K_n)
    tt=t/(a**2)
    prob_rt_std = np.pi * K * (at.exp( - ((K*np.pi)**2 * tt/2) ) + eps) * at.sin( K * np.pi * w )
    
    return prob_rt_std#.sum()

def _individual_logp(t, X, v, a, w, sv):

    t_correct = t[X]
    t_incorrect = t[1-X]

    total_logp = _diffusion_sv(t_correct, -v, a, 1-w, sv) + _diffusion_sv(t_incorrect, v, a, w, sv)
    return total_logp

def _diffusion_sv_logp(RT, X, v, a, w, sv, t_er):
    t = RT-t_er
    t = at.switch(at.le(t,0), eps,t)
    w = at.switch(at.ge(w,1), 0.99,w) # to avoid instability during intial evaluation.

    X = at.as_tensor(X)
    prob_rt, _ = ae.scan(lambda t_l, X_l, v_l, a_l, w_l, sv_l: at.switch(at.le(t, 0), eps, _individual_logp(t_l, X_l, v_l, a_l, w_l, sv_l)), sequences=[t,X, v, a, w, sv])
    
    return prob_rt.sum()

def _diffusion_sv(t, v, a, w, sv):
    K = [1,2,3,4]
    
    prob_rt_std_all, _ = ae.scan(_diffusion_01w, sequences=t,non_sequences=(a, w, K))
    prob_rt_std = prob_rt_std_all.sum()
    prob_rt = np.exp(np.log(prob_rt_std) + ((a*w*a*sv)**2 - 2*a*v*w*a - (v**2)*t ) / (2*(sv**2)*t+2) + eps) #eps for numerical stability
    prob_rt = prob_rt / (np.sqrt( (sv**2)*t+1 )) / a**2

    return prob_rt

def _diffusion_default_priors(parti_n):
    with pm.Model() as model:
        v = pm.Normal("v",1,1,shape=(parti_n,1)) #v = ae.tensor.tile(v, (1,J))
        a = pm.Gamma("a",2,2,shape=(parti_n,1))
        z = pm.Uniform("z", 0,a,shape=(parti_n,1)) # z ranges from 0 to a
        w = pm.Deterministic("w", z/a)
        t_er = pm.HalfNormal("t_er",2,shape=(parti_n,1))
        sv = pm.HalfNormal("sv", 2,shape=(parti_n,1))
        
    return model, v,a,w,sv, t_er

def _diffusion_draw(v,a,w, t_er, rng=None, size=None):
    pass

def _diffusion_model(obs_X = None, obs_RT=None):
    
    model, v,a,w, t_er, sv = _diffusion_default_priors(obs_X.shape[0])
    vars = obs_X, v,a,w, t_er, sv
    logp = _diffusion_sv_logp

    with model:
        pm.DensityDist(
            "RT",
            *vars,
            logp=logp,
            observed=obs_RT,
            #random=_diffusion_draw,
            #size=(1,J) if obs_RT is None else obs_RT.shape
        )
        
    return model

def sample_prior_data(samples_n = 100):
    pass

def sample_posterior_params(RT, X, samples_n, chains):

    model = _diffusion_model(obs_X = X, obs_RT=RT)
    posterior_chain = ut.sample_posterior(model,samples_n, chains,tune)
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
    sv=np.repeat([-0.89], 4) [:,np.newaxis]
    RTs = [[0.09, 1.39385687, 1.55661403, 1.88891806, 2.23878542, 0.92574541, 3.56532722],
            [0.19, 1.59385687, 2.55661403, 3.88891806, 1.23878542, 1.92574541, 5.56532722]
    ]
    
    X = np.random.randint(0,2,(4,5))
    RT = np.random.uniform(0,4,(4,5))
    
    lp = _diffusion_sv_logp(RT, X, v,a,w,sv,t_er)

    print(f"lp: {lp.eval()}")


#%%
if __name__ == "__main__":
    I,J = 3,(4,2,3)
    log.debug("Starting test")

    #_test_edge_cases()

    for _,j in zip(range(1,I+1), J):
        X = np.random.randint(0,2,(4,j))
        RT = np.random.uniform(0,4,(4,j))

        log.debug(f"Starting Diffusion (sv=False) for {j}")
        model = _diffusion_model(obs_X = X, obs_RT=RT, sv = False)
        posterior_chain = pm.sample(model=model, draws=10, chains=2,tune=10)
        log.debug(f"Posterior model_correct {posterior_chain.posterior.v.shape}")
