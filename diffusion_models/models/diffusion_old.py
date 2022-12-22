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
tune = 300
def _diffusion_01w(t, a, w):
    #K_n = at.sqrt(-2 * at.log(np.pi * t * err) / (np.pi**2 * t) ) + 1
    K = np.asarray([1,2,3,4,5]) #at.arange(K_n)
    tt=t/(a**2)
    prob_rt_std = np.pi * K * (at.exp( - ((K*np.pi)**2 * tt/2) ) + eps) * at.sin( K * np.pi * w )
    
    return prob_rt_std.sum()

def _diffusion_logp(RT, v, a, w, t_er):
    t = RT-t_er
    t = at.switch(at.le(t,0), eps,t)
    w = at.switch(at.ge(w,1), 0.99,w) # to avoid instability during intial evaluation.

    prob_rt_std, _ = ae.scan(lambda t_l: at.switch(at.le(t, 0), eps, _diffusion_01w(t_l, a, w)), sequences=t)
    prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
    prob_rt = (1 / a**2) * at.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std.sum()
    prob_rt = prob_rt / prob_X

    return prob_rt
 
def _diffusion_sv_logp(RT, v, a, w, t_er, sv):
    t = RT-t_er
    t = at.switch(at.le(t,0), eps,t)
    w = at.switch(at.ge(w,1), 0.99,w)
    
    prob_rt_std_all, _ = ae.scan(lambda t_l: at.switch(at.le(t, 0), eps, _diffusion_01w(t_l, a, w)), sequences=t)
    prob_rt_std = prob_rt_std_all.sum()
    prob_rt = np.exp(np.log(prob_rt_std) + ((a*w*a*sv)**2 - 2*a*v*w*a - (v**2)*t ) / (2*(sv**2)*t+2) + eps) #eps for numerical stability
    prob_rt = prob_rt / (np.sqrt( (sv**2)*t+1 )) / a**2
    prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
    prob_rt = prob_rt / prob_X

    return prob_rt


def _diffusion_default_priors(correct_resp):
    with pm.Model() as model:
        v = pm.Normal("v",1,1) #v = ae.tensor.tile(v, (1,J))
        a = pm.Gamma("a",2,2)
        z = pm.Uniform("z", 0,a) # z ranges from 0 to a
        w = pm.Deterministic("w", z/a)
        t_er = pm.HalfNormal("t_er",2)
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
        pdf_diffusion = _diffusion_logp(RT,v,a,w, t_er).eval()
        
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

def _diffusion_model(obs_X = None, obs_RT=None, sv = False, correct_resp = False, J=1):
    
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
            "RT",
            *vars,
            logp=logp,
            observed=obs_RT,
            random=_diffusion_draw,
            size=(1,J) if obs_RT is None else obs_RT.shape
        )
        
    return model, vars

def _diffusion_model_both(obs_X=None, obs_RT=None, sv = False, J = 1):
    
    if obs_X is not None: 
        if obs_X.shape[0] > 1:
            raise Exception("Observed data can either be None or per participant vector")
    
        RT = obs_RT[obs_X>0]
        X = obs_X[obs_X>0]
        model_correct, vars_c = _diffusion_model(X, RT, sv,correct_resp = True)

        RT = obs_RT[np.invert(obs_X>0)]
        X = obs_X[np.invert(obs_X>0)]
        model_incorrect, vars_ic = _diffusion_model(X, RT, sv,correct_resp = False)

    else: #For prior sampling
        model_correct, vars_c = _diffusion_model(correct_resp=True, J=J)
        model_incorrect, vars_ic = _diffusion_model(correct_resp=False, J=J)
    
    return (model_correct, vars_c), (model_incorrect, vars_ic)

def get_models_n_vars(I = None, J = None, data_rt=None, data_ra=None, sv=False, correct_resp="1|0|Both"):

    """
    This function provides both correct and incorrect response models for each participant.
    """

    if(correct_resp == "1|0|Both"):
        correct_resp = "Both"

    if(I == None or J == None):
        if (data_rt is not None and len(data_rt.shape) > 0):
            I,J = data_rt.shape
        else:
            raise Exception("No way possible to determine the shape, please provide (I, J) or data")

    for i_l in range(I):

        if(data_ra is None or data_rt is None):
            (model_correct, vars_c), (model_incorrect,vars_ic) = _diffusion_model_both(sv=sv, J=J)
        else:    
            (model_correct, vars_c), (model_incorrect,vars_ic) = _diffusion_model_both(data_ra[[i_l],:], data_rt[[i_l],:], sv=sv, J=J)

        if(correct_resp == "1"):
            yield model_correct, vars_c
        elif(correct_resp == "0"):
            yield model_incorrect,vars_ic
        else:
            yield (model_correct, vars_c), (model_incorrect,vars_ic)

def accumulate_RT(RT_new, RT = None, axis=1):
    if(RT is None):
        RT = RT_new
    else:
        RT = np.append(RT, RT_new, axis=axis)
    return RT

def sample_prior_data(samples_n = 100, models=[]):
    
    RT_correct = None
    RT_incorrect = None

    if(len(models) == 0): # Default model creation (X=1)
        models = [[m_c, m_ic] for (m_c, _), (m_ic,_) in get_models_n_vars(I = 2, J = 4)] # 4 response times for correct responses of 2 participants

    for m_c, m_ic in models:
        prior_data_correct = ut.sample_prior(m_c, samples_n)
        prior_data_incorrect = ut.sample_prior(m_ic, samples_n)

        #log.debug(f"RT prior shape: {prior_data.prior.RT.values.shape}")
        RT_correct = accumulate_RT(prior_data_correct.prior.RT.values, RT_correct)
        RT_incorrect = accumulate_RT(prior_data_incorrect.prior.RT.values, RT_incorrect)

    return(RT_correct, RT_incorrect)


def sample_posterior_params(samples_n, chains, tune=tune, models=[], X=None, RT=None):

    """
    Arguments:
        models=[]: parameter should have tuples of correct and incorrect responses for each participant
    """
    posterior_chain = [] # per participant
    posterior_chain_correct = None
    posterior_chain_incorrect = None

    if(len(models) == 0): # Default model creation (X=1)
        models = [[m_c, m_ic] for (m_c, _), (m_ic,_)  in get_models_n_vars(data_rt=RT, data_ra=X)] # 4 response times for correct responses of 2 participants

    for model_correct, model_incorrect in models:
        
        p_c_c = ut.sample_posterior(model=model_correct, samples_n=samples_n,chains=chains,tune=tune)
        p_c_ic = ut.sample_posterior(model=model_incorrect, samples_n=samples_n,chains=chains,tune=tune)

        if(posterior_chain_correct != None):
            posterior_chain_correct.posterior = xa.concat((posterior_chain_correct.posterior, p_c_c.posterior), dim="particpant")
        else:
            posterior_chain_correct = p_c_c.copy()

        if(posterior_chain_incorrect != None):
            posterior_chain_incorrect.posterior = xa.concat((posterior_chain_incorrect.posterior, p_c_ic.posterior), dim="particpant")
        else:
            posterior_chain_incorrect = p_c_ic.copy()

        posterior_chain.append((p_c_c, p_c_ic))
        
    return (posterior_chain_correct, posterior_chain_incorrect), (posterior_chain, models)

def sample_post_pred_data(posteriors=[], samples_n = 100, models=[]):
    if len(models) == 0:
        raise Exception("Need to provide the models corresponding to the posterior trace")
    
    RT = []
    
    
    for (m_c, m_ic), (p_c, p_ic) in zip(models, posteriors):
        post_pred_correct = ut.sample_post_pred(m_c, posterior=p_c,samples_n=samples_n)
        post_pred_incorrect = ut.sample_post_pred(m_ic, posterior=p_ic,samples_n=samples_n)
        
        #RT_correct = accumulate_RT(post_pred_correct.posterior_predictive.RT.values, RT_correct, axis=-1)
        #RT_incorrect = accumulate_RT(post_pred_incorrect.posterior_predictive.RT.values, RT_incorrect, axis=-1)

        RT.append((post_pred_correct.posterior_predictive.RT.values, post_pred_incorrect.posterior_predictive.RT.values))

        #RT_correct_s = np.swapaxes(RT_correct, 0,2)
        #RT_incorrect_s = np.swapaxes(RT_incorrect, 0,2)

    return RT

def _test_edge_cases():

#v:-1.2002378916969234, a:0.017329851659947146, w:0.8280975715451416, t_er:0.6342687555621264, RT:[1.15521305 3.05237406 1.16819656 ... 2.47025695 2.03251366 2.46823881]

    v = -0.9249241152935039
    a = 0.023750197074163027
    w = 0.44120144255887783 
    #t_er=0.7100510973761837
    t_er=0.07100510973761837 
    RTs = [0.09, 1.39385687, 1.55661403, 1.88891806, 2.23878542, 0.92574541, 3.56532722]
    lp = _diffusion_logp(np.asarray(RTs), v,a,w,t_er)
    sample = _diffusion_draw(v,a,w,t_er,size=(1,5))
    
    #for RT in RTs:
    #    lp = _diffusion_logp(np.asarray(RT), v,a,w,t_er)
    log.debug(f"lp: {lp.eval()}")
    log.debug(f"samples: {sample}")

def _test_edge_cases_sc():
    v = -1.01
    a = -0.9 
    z = -1.51 
    t_er = -0.75
    sv = -0.89
    RTs = [2.81207854, 2.3956247 , 0.007444]
    #{'v': -0.97, 'a': -1.28, 'z': -1.47, 't_er': -1.01, 'sv': -2.04, 'RT': nan}
    lp = _diffusion_sv_logp(np.asarray(RTs),v,a,z/a,t_er,sv)
    log.debug(f"lp: {lp.eval()}")


#%%
if __name__ == "__main__":
    I,J = 3,(4,2,3)
    log.debug("Starting test")

    _test_edge_cases()
    _test_edge_cases_sc()

    #(model, _),_ = _diffusion_model_both()
    #prior_chain = pm.sample_prior_predictive(model=model)
    #log.debug(f"Prior RT {prior_chain.prior.RT.shape}")#, " ***** min:", np.min(prior_chain.prior.RTs), " ***** max:", np.max(prior_chain.prior.RTs))

    for _,j in zip(range(1,I+1), J):
        X = np.random.randint(0,2,(4,j))
        RT = np.random.uniform(0,4,(4,j))

        log.debug(f"Starting Diffusion (X=1) (sv=False) for {j}")
        (model_correct, _), _ = _diffusion_model_both(X, RT, sv=False)
        posterior_chain = pm.sample(model=model_correct, draws=10, chains=2,tune=10)
        log.debug(f"Posterior model_correct {posterior_chain.posterior.v.shape}")

        log.debug(f"Starting Diffusion (X=0) (sv=False) for {j}")
        _, (model_incorrect,_) = _diffusion_model_both(X, RT, sv=False)
        pm.sample(model=model_incorrect, draws=10, chains=2,tune=10)

        log.debug(f"Starting Diffusion (X=1) (sv=True) for {j}")
        (model_correct, _), _ = _diffusion_model_both(X, RT, sv=True)
        pm.sample(model=model_correct, draws=10, chains=2,tune=10)

        log.debug(f"Starting Diffusion (X=0) (sv=True) for {j}")
        _, (model_incorrect,_) = _diffusion_model_both(X, RT, sv=True)
        pm.sample(model=model_incorrect, draws=10, chains=2,tune=10)

    log.debug(f"Starting posterior prediction distribution for size {posterior_chain.posterior.v.shape}")
    with model_correct:
        postr_pred_chain = pm.sample_posterior_predictive(trace=posterior_chain.sel(chain =[0]))
    log.debug(f"Posterior RT {postr_pred_chain.posterior_predictive.RT.shape}")
    log.debug(f"Posterior max RT {np.max(postr_pred_chain.posterior_predictive.RT.values)}")
    assert postr_pred_chain.posterior_predictive.RT.shape[0:2] == (1,10) #chains=1, draw. Not RTs per participant because they may vary


# %%
"""X = np.random.randint(0,2,(2,3))
RT = np.random.uniform(0,4,(2,3))
model_correct, model_incorrect = [[m_c, m_ic] for (m_c, _), (m_ic, _) in get_models_n_vars(data_ra = X, data_rt = RT, sv=False)]
posterior_chain_0 = pm.sample(model=model_correct[0], draws=10, chains=2,tune=10)
posterior_chain_1 = pm.sample(model=model_correct[1], draws=10, chains=2,tune=10)
# %%
import xarray as xa
posterior_chain = posterior_chain_0
t = xa.concat((posterior_chain_0.posterior, posterior_chain_1.posterior), dim="participant")
#t = xa.concat((posterior_chain_0.posterior, posterior_chain_1.posterior), 
#dim=[v for v in posterior_chain_0.posterior.coords.keys()][2:3])
#[v for v in posterior_chain_0.posterior.coords.keys()][2:]
posterior_chain.posterior = t

# %%
with model_correct[0]:
    pm.sample_posterior_predictive(posterior_chain_0, var_names = ["RT"])
# %%
"""
