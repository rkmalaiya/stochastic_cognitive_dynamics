#from turtle import width
from turtle import pos
import jax.numpy as npx
import jax.scipy as sci
import numpyro as pyro
import cme.decision_models.quantum_discrete as qd
import cme.decision_models.diffusion_discrete as dd
import jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA, HMCECS, Predictive, SVI, Trace_ELBO
import numpyro.infer.autoguide as ag
from numpyro.optim import Adam
from jax import random
from jax import lax
import arviz as az 
from numpyro.distributions import constraints

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from joblib import Parallel, delayed
from joblib import parallel_config

from numpyro import enable_validation
enable_validation(True)

from cme.utils import common_logging as cl
from cme.utils import common_utils as cu
log = cl.get_logger("confidence_accumulation")

#pyro.set_platform("cpu")
pyro.set_host_device_count(64)
#pyro.enable_x64()
#from jax import config
#config.update("jax_enable_x64", False)

# To give a random starting seed to Numpyro MCMC. Split gurantees to generate a new random number for the same starting seed.
#_rng_key = random.key(1)#(0)
#_rng_key, _rng_key_ = random.split(_rng_key)

def centralized_parameters(I):
    """
    I: Number of participants
    
    """
    mu_m =  pyro.sample(f"mu_m", dist.Normal(0,1))
    mu_s =  pyro.sample(f"mu_s", dist.HalfNormal(2))
    with pyro.plate('I6', I, dim=-2):
        mu = pyro.sample("mu", dist.Normal(mu_m,mu_s)) # Drift Rate
        sigma = pyro.sample("sigma", dist.Normal(1,0.1)) # Diffusion Rate
    #sigma = pyro.deterministic("sigma", npx.ones((I,1)))    
    return mu, sigma

def non_centralized_parameters(I):
    """
    I: Number of participants
    
    """
    m = pyro.sample("m", dist.Normal(0,1))
    s = pyro.sample("s", dist.HalfNormal(1))

    with pyro.plate('I3', I, dim=-2):
        mu_r = pyro.sample("mu_r", dist.Normal(2,1)) # Drift Rate
        sigma_r = pyro.sample("sigma_r", dist.Normal(1,1)) # Diffusion Rate

        mu = pyro.deterministic("mu", m + s * mu_r)
        sigma = pyro.deterministic("sigma", (m + s * sigma_r)**2) 
    #sigma = pyro.deterministic("sigma", npx.ones((I,1)))
    
    return mu, sigma

def non_centralized_parameters_VI(I):
    """
    I: Number of participants
    
    """
    m_m = pyro.param("m_m", 0, constraint=constraints.positive)
    m_s = pyro.param("m_s", 1, constraint=constraints.positive)
    m = pyro.sample("m", dist.Normal(m_m,m_s))
    s = pyro.sample("s", dist.HalfNormal(m_s))

    with pyro.plate('I4', I, dim=-2):
        mu_r_m = pyro.param("m_m", 2, constraint=constraints.positive)
        mu_r_s = pyro.param("m_m", 1, constraint=constraints.positive)
        
        mu_r = pyro.sample("mu_r", dist.Normal(mu_r_m,mu_r_s)) # Drift Rate
        sigma_r = pyro.sample("sigma_r", dist.Normal(1,1)) # Diffusion Rate

        mu = pyro.deterministic("mu", m + s * mu_r)
        sigma = pyro.deterministic("sigma", m + s * sigma_r) 
    #sigma = pyro.deterministic("sigma", npx.ones((I,1)))
    
    return mu, sigma

def _get_initial_state_VI(n_states, start_width, I=1, prob=1,sub_sample_size=None):

    with pyro.plate('I2', I, dim=-4,subsample_size=sub_sample_size):
        with pyro.plate('S', n_states, dim=-1):
            conc = pyro.sample("phi_conc", dist.Beta(0.5,0.5))+0.01 #to avoid 0

    with pyro.plate('I3', I, dim=-3,subsample_size=sub_sample_size):
        p_0 = pyro.sample("phi_init", dist.Dirichlet(conc)) # Initial State
        

    #p_0 = npy.sample("phi_init", dist.Dirichlet((npx.ones(n_states))/n_states)) # Initial State

             
    p_0 = pyro.deterministic("phi_0", p_0.transpose(0,1,3,2)) #.transpose(0,1,3,2)

    return p_0 #s_0


def _timestep_transition_matrix_mat_pow(n, T_delta, Mn):

    T_i = []
    for n_i, T_delta_i in zip(n, T_delta):
        T_i_j = []
        for n_i_j in n_i:
            #T_delta_i_j = T_delta_i[j,...]
            T_nt = Mn @ T_delta_i[0,...]
            for _ in range(n_i_j.astype(int)-2):
                T_nt = T_nt @ Mn @ T_delta_i[0,...] # we need to vectorize this function
            T_i_j.append(T_nt)
        
        T_i.append(T_i_j)
    
    T_t = T_delta @ npx.asarray(T_i)
        
    return T_t


def _timestep_transition_matrix_callback(n, T_delta, Mn):

    n1 = n
    i1 = 0
    j1 = 0
    Mn1 = Mn

    def _take_step_j(static_params, params):
        
        #i, j = static_params["i"], static_params["j"]

        n_i_j = params["n"]
        #print(i1, j1)
        T_delta_i_j = static_params["T_delta"][0,...] #Here 0 is the jth dimension. For T_delta we don't need to have a value for each RT because they all will be same
        def oper(n_i_j, T_delta_i_j):
            return npx.linalg.matrix_power(Mn @ T_delta_i_j, n_i_j.astype(int).item() - 1)
        
        grad_oper = jax.grad(oper)
        a = jax.pure_callback(grad_oper, jax.ShapeDtypeStruct(T_delta_i_j.shape, T_delta_i_j.dtype), n_i_j, T_delta_i_j)

        T_i_j = jax.pure_callback(oper, jax.ShapeDtypeStruct(T_delta_i_j.shape, T_delta_i_j.dtype), n_i_j, T_delta_i_j)
         
        

        params["T_nt"] = T_i_j
        
        return (static_params, params)

    def _take_step_i(i, params):
        #nonlocal i1

        static_params = {"T_delta":params.pop("T_delta")}
        
        static_params, params = lax.scan(_take_step_j, static_params, params)
        params["T_delta"] = static_params["T_delta"]
        #i1 = i1 + 1
        #print(i1)
        return (i+1, params)
        
    T_nt = npx.empty(n.shape[:2] + T_delta.shape[-2:])
    #print(n.shape, T_delta.shape, T_nt.shape)
    params = {"T_delta":T_delta, "T_nt":T_nt, "n":n}

    i, params = lax.scan(_take_step_i, 0, params)
    T_nt = params["T_nt"]

    T_t = T_delta @ T_nt
        
    return T_t

def _timestep_transition_matrix_scan(n, T_delta, Mn):

    n1 = n
    i1 = 0
    Mn1 = Mn


    def _take_step_i(i, params):
        nonlocal i1
        j1 = 0
        
        def _take_step_j(static_params, params):
            nonlocal n1
            nonlocal j1
            n_i_j = n1[i1, j1]
            T_delta_i_j = static_params["T_delta_i"][0,...]
            
            T_i_j = npx.linalg.matrix_power(Mn1 @ T_delta_i_j, n_i_j.astype(int).item() - 1)
            params["T_nt"] = T_i_j
            j1 = j1 + 1
            return (static_params,params)

        
        T_delta_i = params.pop("T_delta")
        static_params = {"T_delta_i": T_delta_i}
        static_params, params = lax.scan(_take_step_j, static_params, params)
        i1 = i1 + 1
        params["T_delta"] = T_delta_i
        return (i, params)
        
    T_nt = npx.empty(n.shape[:2] + T_delta.shape[-2:])
    #print(n.shape, T_delta.shape, T_nt.shape)
    params = {"T_delta":T_delta, "T_nt":T_nt}

    i, params = lax.scan(_take_step_i, 0, params)
    T_nt = params["T_nt"]

    T_t = T_delta @ T_nt
        
    return T_t

def _timestep_transition_matrix_map(n, T_delta, Mn):

    #fn_params = []

    def take_step_j(T_delta_i, n_i):

        def mat_pow(n_i_j):
            T_nt = npx.linalg.matrix_power(Mn @ T_delta_i[0,...], n_i_j.astype(int).item() - 1)
            return T_nt
        
        return list(map(mat_pow, n_i))

    T_i = list(map(take_step_j, T_delta, n)) 
   
   # for n_i, T_delta_i in zip(n, T_delta):
   #     for n_i_j in n_i:
    #        fn_params.append(delayed(mat_pow)(n_i_j, T_delta_i))  
    #with parallel_config(backend='threading', n_jobs=2):
   #     T_i = Parallel()(fn for fn in fn_params)

    #T_i = []
    #for n_i, T_delta_i in zip(n, T_delta):
    #    T_i_j = []
    #    for n_i_j in n_i:
    #        T_nt = mat_pow(n_i_j, Mn, T_delta_i) # we need to vectorize this function
    #        T_i_j.append(T_nt)
    #    
    #    T_i.append(T_i_j)
    
    T_t = T_delta @ npx.asarray(T_i)#.reshape(n.shape[:2] + T_delta.shape[-2:])

    return T_t

def _timestep_transition_matrix(n, T_delta, Mn):

    T_i = []
    for n_i, T_delta_i in zip(n, T_delta):
        T_i_j = []
        for n_i_j in n_i:
            #T_delta_i_j = T_delta_i[j,...]
            T_nt = npx.linalg.matrix_power(Mn @ T_delta_i[0,...], n_i_j.astype(int).item() - 1) # we need to vectorize this function
            T_i_j.append(T_nt)
        
        T_i.append(T_i_j)
    
    T_t = T_delta @ npx.asarray(T_i) 
    #T_t = npx.asarray(T_i) # uncomment to include all response time
    return T_t


def _timestep_transition_matrix_failedd(n, T_delta, Mn):
    #Mn - n_states x n_states
    # T_delta - I x 1 x n_states x n_states
    
    def stack_i(i, params): 
        # return 1 x n_j x n_states x n_states
        T_delta_i = params.pop("T_delta") # Removed from params because it does not have the second J dimension, instead the second dimension is of lenght 1.
        #static_params = dict(T_delta = T_delta_i)

        T_t_i_0 = Mn @ T_delta_i[0,...] # 1 x 1 x n_states x n_states

        def stack_j(T_t_i_0, params):
            n_i_j = params["n"] 
            T_t_n_i_j = npx.tile(T_t_i_0, n_i_j-2) #n_i_j x 1 x 1 x n_states x n_states

            params2 = dict(T_t_n_i_j = T_t_n_i_j)
            
            def mat_power(T_t_i_0, params2):
                T_t_i_j = params2["T_t_n_i_j"]
                T_t_i_j = T_t_i_j @ T_t_i_0
                return (T_t_i_j, params2)
            
            T_t_i_j, params2 = lax.scan(mat_power, T_t_i_0, params2)
            params["T_t"] = T_t_i_j
            return (T_t_i_0, params)
        
        T_t_i_0, params = lax.scan(stack_j, T_t_i_0, params)
        params["T_delta"] = T_delta_i # adding back the removed value so that scan iterates over the first dimension of I length.
        return (i, params)
    
    T_t = npx.empty(n.shape[:2] + T_delta.shape[-2:])
    params = dict(n=n, T_delta = T_delta, T_t = T_t)

    _, params = lax.scan(stack_i, 0, params)
    T_t = params["T_t"]
    T_t = T_delta @ T_t

    return T_t

def _get_transition_matrix(intensity_matrix, RT, delta=None, Mn = None, transition_type="RT|TIMESTEP"):
   
    if transition_type == "RT":
        #T_t = sci.linalg.expm(intensity_matrix * ((RT[...,None,None]/delta) if not npx.isscalar(RT) else (RT/delta)))
        #T_t = sci.linalg.expm(intensity_matrix * ((RT[...,None,None]/delta) if not npx.isscalar(RT) else (RT/delta)))
        T_t = sci.linalg.expm(intensity_matrix * ((RT[...,None,None]) if not npx.isscalar(RT) else (RT)))
    elif transition_type == "TIMESTEP":
        ns=np.ceil(RT/delta) 
        #ns=np.floor(RT/delta) # uncomment to include all response time
        RT_remaining = RT - ns*delta
        T_delta = sci.linalg.expm(intensity_matrix * delta)

        # uncomment to include all response time
        # T_delta_remaining = sci.linalg.expm(intensity_matrix * ((RT_remaining[...,None,None]) if not npx.isscalar(RT_remaining) else (RT_remaining)))
        T_t = _timestep_transition_matrix(ns, T_delta, Mn)  #uncomment if fails
        
        # uncomment to include all response time
        #T_t = T_delta_remaining @ _timestep_transition_matrix(ns, T_delta, Mn)


    else:
        raise Exception(f"Please select one of {transition_type}")

    return T_t # I x J x n_state x n_state

def _get_measurement_matrix(n_states, response_width, prob=0.5, model_type = "Markov|Quantum"):

    if model_type == "Markov":
        Mc, Mw, Mn = dd._get_measurement_matrix(n_states, response_width, prob)
    elif model_type == "Quantum":
        Mc, Mw, Mn = qd._get_measurement_matrix(n_states, response_width, prob)
    else:
        raise Exception(f"Please select one of {model_type}")
    return Mc, Mw, Mn

def _get_initial_state(n_states, start_width, I = 1, prob=1, model_type = "Markov|Quantum", prior_type="Upper|Lower|Centered|Uniform|Model"):
    if prior_type == "Model":
        if model_type == "Markov":
            phi_0 = dd._get_initial_state(n_states, start_width, I, prob)   
        elif model_type == "Quantum":
            phi_0 = qd._get_initial_state(n_states, start_width, I, prob)
        else:
            raise Exception(f"Please select one of {model_type}")
    else:
        width = start_width #choose odd number
        if prior_type == "Upper":
            pad_width = (n_states-width,0)

        elif prior_type == "Lower":
            pad_width = (0,n_states-width)

        elif prior_type == "Centered":
            w_t = int(((n_states-width)/2))
            if(width % 2==0):
                pad_width = (w_t+1, w_t)
            else:    
                pad_width = (w_t, w_t) # will pad equally on left and right of array
            
        elif prior_type == "Uniform":
            pad_width = (0,0)
            width = n_states
        elif prior_type == "Opposite":
            pad_width = int((width+1)/2 )
            width = pad_width
            
            
            
        #conc = npx.ones(width)
        #p_0 = npx.pad(stats.dirichlet(conc).rvs(), ((0,0),pad_width)) # rvs are of shape (1,n_states)
        conc = npx.ones((1,width))
        if prior_type != "Opposite":
            p_0 = npx.pad(conc, ((0,0),pad_width)) 
        else:
            p_0 = npx.zeros((1, n_states))
            p_0 = p_0.at[:,:pad_width].set(conc)
            p_0 = p_0.at[:,-pad_width:].set(conc)

        p_0 = p_0 / npx.sum(p_0) # rvs are of shape (1,n_states)

        if model_type == "Markov":
            phi_0 = npx.tile(p_0.T[None, None,...], (I,1,1,1))
        elif model_type == "Quantum":
            phi_0 = npx.tile(p_0.T[None, None,...], (I,1,1,1))**(1/2)

    return phi_0   

def perform_state_transition(intensity_matrix, RT_s, RA_s, Mc, Mw, Mn, phi_0, delta, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
        
    if likelihood_type=="SINGLE":
        RT=RT_s
        T_t = _get_transition_matrix(intensity_matrix, RT=RT, delta=delta, Mn=Mn, transition_type=transition_type)
        
    elif likelihood_type=="JOINT":
        RT_1 = RT_s[0]
        T_t_1 = _get_transition_matrix(intensity_matrix, RT=RT_1, delta=delta, Mn=Mn, transition_type=transition_type)

        RT_2 = RT_s[1]
        T_t_2 = _get_transition_matrix(intensity_matrix, RT=RT_2, delta=delta, Mn=Mn, transition_type=transition_type)

        phi_t_1_c = T_t_2 @ Mc @ T_t_1 
        phi_t_1_w = T_t_2 @ Mw @ T_t_1 
        
        RA_1 = RA_s[0]
        T_t = npx.where(RA_1[..., None, None]==1, phi_t_1_c, phi_t_1_w)
        
    phi_t = T_t @ phi_0
    #if transition_type == "TIMESTEP":
    #    phi_t = phi_t / phi_t.sum(axis=-2, keepdims=True)
    return phi_t

def get_mean_init_confidence(n_states, phi_0, model_type = "Markov|Quantum"):
    if model_type == "Markov":
        P_0 = phi_0
    elif model_type == "Quantum":
        P_0 = npx.abs(phi_0)**2

    Mid = (n_states+1)//2
    mv = npx.arange(-(Mid-1), (Mid))

    mean_conf_init = mv @ P_0
    return mean_conf_init

def get_mean_confidence(n_states, intensity_matrix, phi_0, delta, Mc=None, Mw=None, Mn=None, t=None, x=None, 
                        conf_scale = None,
                        transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", 
                        model_type = "Markov|Quantum", return_type="Probability|MeanConfidence"):
    
    """
    n_states: int
    intensity_matrix: float|Complex[IxJ]
    t: float
    phi_0: float[Mx1]
    delta: int
    scale: "None|(add_scale, mul_scale)"
    """
    phi_t = perform_state_transition(intensity_matrix, RT_s = t, RA_s = x, Mc=Mc, Mw=Mw, Mn=Mn, phi_0=phi_0, delta=delta,
                                     transition_type=transition_type, likelihood_type=likelihood_type)
    #T_t = _get_transition_matrix(intensity_matrix, RT=t, delta=delta, Mn=Mn, transition_type=transition_type)

    if(return_type == "Probability"):
        phi_t_c = Mc @ phi_t
        phi_t_w = Mw @ phi_t

        if(likelihood_type == "SINGLE"):
            phi_t = phi_t_c if x==1 else phi_t_w
        elif(likelihood_type == "JOINT"):
            phi_t = phi_t_c if x[1]==1 else phi_t_w


    if model_type == "Markov":
        P_t = phi_t
    elif model_type == "Quantum":
        P_t = npx.abs(phi_t)**2
    
    Mid = (n_states+1)//2
    mv = npx.arange(-(Mid-1), (Mid))
    if conf_scale is not None:
        add_scale, mul_scale = conf_scale
        mv = cu.get_conf_scale(mv, add_scale, mul_scale)

    if return_type == "Probability":
        ret_val = P_t.sum()
    else: #if return_type == "MeanConfidence":
        ret_val = mv @ (P_t/P_t.sum())
    #else:
    #    raise Exception(f"Please provide one of {return_type}")

    return ret_val


def transformed_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):
    #if likelihood_type == "SINGLE":
    #    RT_s = npx.exp(RT_s)
    #elif likelihood_type == "JOINT":
    #    RT_s[0] = npx.exp(RT_s[0])
    #    RT_s[1] = npx.exp(RT_s[1])
    #else:
    #    raise Exception(f"Please select one of {likelihood_type} values for likelihood_type variable")

    return estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)

def estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):
    P_t = likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    P_t = npx.where(P_t <= 0, 0, P_t)
    return P_t

def likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):

    phi_t = perform_state_transition(intensity_matrix, RT_s, RA_s, Mc, Mw, Mn, phi_0, delta, 
                                     transition_type=transition_type, likelihood_type=likelihood_type)

    if likelihood_type == "SINGLE":
        RA = RA_s
        RT_cond = RT_s
    elif likelihood_type == "JOINT":
        RA = RA_s[1]
        RT_cond = RT_s[0] * RT_s[1] # So that even if a single RT is 0, the likelihood for that participant becomes 0
    else:
        raise Exception(f"Please select one of {likelihood_type} values for likelihood_type variable")


    P_t_c = (Mc @ phi_t)
    P_t_w = (Mw @ phi_t)
    P_t = npx.where(RA[...,None,None]==1, P_t_c, P_t_w)

    if model_type == "Markov":
        #P_t_c = (Mc @ phi_t).sum(axis=(-2,-1))
        #P_t_w = (Mw @ phi_t).sum(axis=(-2,-1))

        P_t = P_t.sum(axis=(-2,-1)) # Adding over states
        
        #P_t = npx.where(RA==1, P_t_c, P_t_w)
        
    elif model_type == "Quantum":
        #P_t_c = (npx.abs(Mc @ phi_t)**2).sum(axis=(-2,-1))
        #P_t_w = (npx.abs(Mw @ phi_t)**2).sum(axis=(-2,-1))

        P_t = (npx.abs(P_t)**2).sum(axis=(-2,-1)) #Adding over states
        

        #P_t = npx.where(RA==1, P_t_c, P_t_w)
        
    else:
        raise Exception(f"Please select one of {model_type}")
  
    
    #P_t = npx.where(RT_cond <= 0, 0, npx.log(P_t))

    return P_t #npx.log(npx.sum(P_t)) # summing over all participants and trials

def model(n_states, start_width, response_width, delta, RA_s, RT_s, measurement_prob, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
    if likelihood_type == "SINGLE":
        I, _ = RA_s.shape
    elif likelihood_type == "JOINT":
        if RA_s[0].shape != RA_s[1].shape:
            raise Exception("All Responses should be equal shaped")
        I, _ = RA_s[0].shape
    else:
        raise Exception(f"Please select one of {likelihood_type} for likelihood")

    if params_type == "Centralized":
        mu, sigma = centralized_parameters(I)
    elif params_type == "NonCentralized":
        mu, sigma = non_centralized_parameters(I)
    else:
        raise Exception(f"Please select one of {params_type}")

    if model_type == "Markov":
        sigma = pyro.deterministic("sigma_final",npx.abs(mu) + sigma**2) # Sigma needs to be larger than mu and Sigma cannot be negative
        intensity_matrix = dd._buildK(n_states, mu, sigma, delta)

    elif model_type == "Quantum":
        sigma = pyro.deterministic("sigma_final",sigma**2) # Sigma cannot be negative
        intensity_matrix = qd._buildH(n_states, mu, sigma, delta)
    else:
        raise Exception(f"Please select one of {model_type}")

    #phi_0 = pyro.deterministic("phi_0", _get_initial_state(n_states, start_width, I = I, prob=1, model_type = model_type))
    phi_0 = _get_initial_state(n_states, start_width, I = I, prob=1, model_type = model_type, prior_type="Model")
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)

    if RT_s is not None:
        likl = estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, 
                          transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
        #likl = npx.log(likl)
        pyro.deterministic("likl_rt", likl)
        pyro.factor("likelihood", likl) #.sum()

# def generate_RT(n_states, threshold, delta, measurement_prob, I, J, 
#                 drift_rate, diffusion_rate, phi_0, data_samples = 1, max_RT=40, max_samples=400,
#                 model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
#     random_ts = dist.Uniform(delta, max_RT/delta).sample(key=(rng := jax.random.split(rng)[0]), sample_shape=(I,max_samples))
#     intensity_matrix = get_intensity_matrix(n_states, drift_rate, diffusion_rate, model_type=model_type)
#     Mc, Mw, Mn = _get_measurement_matrix(n_states, threshold, prob=measurement_prob, model_type = model_type)
#     phi_t = perform_state_transition(intensity_matrix=intensity_matrix, RT_s = random_ts, RA_s = None, delta=delta, 
#                                         Mc = Mc, Mn = Mn, Mw = Mw, phi_0=phi_0, 
#                                         transition_type=transition_type, likelihood_type=likelihood_type)
#     states_t = dist.Multinomial(total_count=1, probs=phi_t[...,0]).sample(key=(rng:=jax.random.split(rng)[0]))
#     state_final = np.argwhere(states_t)
#     RA = np.select([
#             state_final[:,[-1]] < threshold - 1,
#             state_final[:,[-1]] >= n_states - threshold - 1
#         ], [0,1], default = np.nan)#[:,-1]
#     Response = np.hstack((state_final, RA, random_ts.flatten()[:,None]*delta))
#     df_res = pd.DataFrame(Response, columns=["I", "J", "state", "RA", "RT"])
#     return df_res
import time

def gen_RT(RT, n_states, response_width, delta, measurement_prob, RA, 
                     drift_rate, diffusion_rate, phi_0, data_samples = (1,10), 
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", 
                     key=None, max_RT_sec=50
                     ):
    
    key1 = cu.get_rng() if key is None else key
    part_I, part_J = data_samples
    max_samples = part_J * 50
    I, mu, sigma = part_I, drift_rate, diffusion_rate
    
        
    #print(f"{(time.perf_counter()/60):.3f}")
    random_ts = stats.uniform.rvs(delta, max_RT_sec/delta, (I,max_samples))     #dist.Uniform(delta, max_RT_sec/delta).sample(key=key1, sample_shape=(I,max_samples))
    
    intensity_matrix = get_intensity_matrix(n_states, mu, sigma, model_type=model_type)
    Mc, Mw, Mn = _get_measurement_matrix(n_states, response_width, prob=measurement_prob, model_type = model_type)
    
    phi_t = perform_state_transition(intensity_matrix=intensity_matrix, RT_s = random_ts, RA_s = None, delta=delta, 
                                        Mc = Mc, Mn = Mn, Mw = Mw, phi_0=phi_0, 
                                        transition_type=transition_type, likelihood_type=likelihood_type)
    
    states_t = dist.Multinomial(total_count=part_J, probs=(phi_t[...,0] / phi_t[...,0].sum(axis=-1, keepdims=True))).sample(key=key1) # output like one-hot encoding
    states_final = npx.argmax(states_t, axis=-1)

    corr_border_count = states_t[:,:,n_states - response_width:].sum(axis=-1)
    incorr_border_count = states_t[:,:,:response_width].sum(axis=-1)
    #RA = npx.tile(npx.nan, random_ts.shape)
    #RA.at[incorr_border_count > corr_border_count].set(0)
    #RA.at[incorr_border_count < corr_border_count].set(1)

    RA = npx.where(incorr_border_count > corr_border_count, 0, 
              npx.where(corr_border_count > incorr_border_count, 1, npx.nan))

    RT = npx.where(npx.isnan(RA), npx.nan, random_ts)
    
    return RT, RA, states_final

def get_RT(RT, n_states, response_width, delta, measurement_prob, RA, 
                     drift_rate, diffusion_rate, phi_0, data_samples = (1,10), param_sample_id=-1,
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", 
                     sampling_type = "GEN|SIM", is_test=False, key=None
                     ):
    
    def sim_RT():
        """
            This function calculates likelihood for one dataset of size I,J
        """
        #likl = simulate_likelihood(RT, n_states, start_width, delta, measurement_prob, phi_0, RA, 
        #            npx.asarray([mu]), npx.asarray([sigma]), 
        #            model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type)

        #return pd.DataFrame({"RT":RT.flatten(), "mu":mu[0], "sigma":sigma[0], "logp":likl.flatten()})

        #for mu, sigma, phi_0 in zip(drift_rate, diffusion_rate, phi_0):
            #for RT in np.arange(delta, RT_max, delta):
            #for t, x in zip(RT.flatten(), RA.flatten()):
            #sim_RT.append(delayed(get_RT))
        #    sim_RT.append(get_RT())
        
        #RT = np.random.default_rng().uniform(RT.min(), RT.max(), size=RT.shape)
        sim_RT = []
        likl = simulate_likelihood(RT, n_states, response_width, delta, measurement_prob, phi_0, RA, 
                        drift_rate, diffusion_rate, 
                        model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type)

        res_RT = sim_RT #Parallel(n_jobs=50)(fn() for fn in sim_RT)

        df_sim_RT = (pd.DataFrame(RT)
        .reset_index(names="part_id")
        .melt(id_vars="part_id", var_name="items", value_name="RT")
        .set_index(["part_id","items"])
        .join(pd.DataFrame(likl)
            .reset_index(names="part_id")
            .melt(id_vars="part_id", var_name="items", value_name="logp")
            .set_index(["part_id","items"]))
        )

        samples_arr = []
        #logp = np.absolute(df_sim_RT.logp)
        df_sim_RT = df_sim_RT.assign(logp = lambda df:np.absolute(df.logp), param_sample_id = param_sample_id)
        for i in range(RA.shape[0]): # weights="logp", 
            if df_sim_RT.loc[:,"logp"].values.sum() > 0:
                samples_arr.append(df_sim_RT.groupby("part_id").sample(frac=1,replace=True, weights="logp", random_state= np.random.default_rng()).assign(weighted_sample=i))
            else:
                samples_arr.append(df_sim_RT.groupby("part_id").sample(frac=1,replace=True, random_state= np.random.default_rng()).assign(weighted_sample=-i))

        #df_sim_RT = pd.concat(res_RT).astype(float)
        #df_sim_RT.loc[:,"logp"] = df_sim_RT.loc[:,"logp"]**2 #np.exp( - df_sim_RT.loc[:,"Likelihood"])


        #samples_arr = []
        #for key, df in df_sim_RT.groupby(["mu", "sigma"]):
        #    for i in range(n_samples):
        #        samples_arr.append(
        #           df.sample(frac=1, weights=npx.absolute(df.logp.values), replace=True).assign(weighted_sample=i)
                    #pd.DataFrame({"samples":df.sample(n=df.shape[0], weights=df.Likelihood), "weighted_sample":i, "key":key})
                    
        #            )

            #samples_arr.append(pd.DataFrame({"samples":df_sim_RT.RT.sample(n=df_sim_RT.shape[0], weights = df_sim_RT.Likelihood),
        #                   "sample_number":i}))
        df_samples = pd.concat(samples_arr)
        return df_samples, df_sim_RT

    if sampling_type == "SIM":
        df_samples, df_sim_RT = sim_RT()
    elif sampling_type == "GEN":
        RT, RA, states_final = gen_RT(RT, n_states, response_width, delta, measurement_prob, RA, 
                     drift_rate, diffusion_rate, phi_0, data_samples, param_sample_id,
                     model_type, transition_type, likelihood_type, 
                     sampling_type, is_test, key)
        df_samples = pd.DataFrame(dict(
            part_id = np.arange(RT.shape[0]),
            J = np.arange(RT.shape[1]),
            RT = RT.flatten(),
            RA = RA.flatten(),
            final_states = states_final.flatten()
        ))


    return {"drift_rate":drift_rate, "diffusion_rate":diffusion_rate, "initial_state":phi_0, "Likelihood":df_sim_RT, "Samples":df_samples}

def simulate_likelihood(RT_pred, n_states, response_width, delta, measurement_prob, phi_0, RA, 
                     drift_rate, diffusion_rate, 
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)
    
    if model_type == "Markov":
        intensity_matrix = dd._buildK(n_states, drift_rate, diffusion_rate, delta)

    elif model_type == "Quantum":
        intensity_matrix = qd._buildH(n_states, drift_rate, diffusion_rate, delta)
    else:
        raise Exception(f"Please select one of {model_type}")
    
    #I, J = RA.shape
    
    #with pyro.plate('I', I, dim=-2):
    #    with pyro.plate('J', J, dim=-1):
    #        RT_pred = pyro.sample("RT_pred", dist.LogNormal(0, 1.5))
    
    likl = transformed_likelihood(intensity_matrix, phi_0, delta, RT_pred, RA, Mc, Mw, Mn, 
                      transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    
    return likl

def predictive_model(RT_pred, n_states, response_width, delta, measurement_prob, phi_0, RA, 
                     drift_rate, diffusion_rate, 
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)
    
    if model_type == "Markov":
        intensity_matrix = dd._buildK(n_states, drift_rate, diffusion_rate)

    elif model_type == "Quantum":
        intensity_matrix = qd._buildH(n_states, drift_rate, diffusion_rate)
    else:
        raise Exception(f"Please select one of {model_type}")
    
    #I, J = RA.shape
    
    #with pyro.plate('I', I, dim=-2):
    #    with pyro.plate('J', J, dim=-1):
    #        RT_pred = pyro.sample("RT_pred", dist.LogNormal(0, 1.5))
    
    likl = transformed_likelihood(intensity_matrix, phi_0, delta, RT_pred, RA, Mc, Mw, Mn, 
                      transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    #pyro.deterministic("likl_prnt", likl)
    #pyro.factor("likelihood", likl)
    return likl.sum()

def guide(n_states, start_width, response_width, delta, RA_s, RT_s, measurement_prob, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    I, _ = RA_s.shape
    mu, sigma = non_centralized_parameters_VI(I)
    #intensity_matrix = dd._buildK(n_states, mu, sigma, delta)    
    if model_type == "Markov":
        sigma = pyro.deterministic("sigma_final",npx.abs(mu) + sigma) # Sigma needs to be larger than mu and Sigma cannot be negative
        intensity_matrix = dd._buildK(n_states, mu, sigma, delta)

    elif model_type == "Quantum":
        sigma = pyro.deterministic("sigma_final",sigma) # Sigma cannot be negative
        intensity_matrix = qd._buildH(n_states, mu, sigma, delta)
    else:
        raise Exception(f"Please select one of {model_type}")

    #phi_0 = pyro.deterministic("phi_0", _get_initial_state(n_states, start_width, I = I, prob=1, model_type = model_type))
    phi_0 = _get_initial_state(n_states, start_width, I = I, prob=1, model_type = model_type, prior_type="Model")

#def inbuilt_guide():
    #'m', 'mu_r', 'phi_conc', 'phi_init', 's', 'sigma_r'

def get_original_params(posterior_samples, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum"):
    if params_type == "Centralized":
        pass
    elif params_type == "NonCentralized":
        m = posterior_samples["m"]
        s = posterior_samples["s"]

        mu_r = posterior_samples["mu_r"]
        sigma_r = posterior_samples["sigma_r"]

        posterior_samples["mu"] = m + s * mu_r
        posterior_samples["sigma"] = (m + s * sigma_r)**2 
    
    p_0 = posterior_samples["phi_init"]

    if model_type == "Markov":
        p_0 = p_0.transpose(0, 1, 2, 4, 3)
        posterior_samples["sigma_final"] = npx.abs(posterior_samples["mu"]) + posterior_samples["sigma"] 

    elif model_type == "Quantum":
        p_0 = p_0.transpose(0, 1, 2, 4, 3)**(1/2)
        posterior_samples["sigma_final"] = posterior_samples["sigma"]

    posterior_samples["phi_0"] = p_0

    return posterior_samples


def sample_posterior_params_VI(DT, X, n_states, start_width, response_width, delta, measurement_prob,
                            num_warmup=100, samples_n=500, num_chains=4, batch_size=2,  
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    guide = ag.AutoNormal(model)
    #guide = ag.AutoMultivariateNormal(model)
    #guide = ag.AutoDAIS(model)

    optimizer = Adam(step_size=0.1)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(cu.get_rng(), num_warmup + samples_n, n_states, start_width, response_width, delta, X, DT, measurement_prob, 
                   params_type = params_type, transition_type=transition_type, 
                   likelihood_type=likelihood_type, model_type=model_type)

    predictive = Predictive(guide, params=svi_result.params, num_samples=1000)
    #posterior_samples = predictive(cu.get_rng(), data=None)
    posterior_samples = predictive(cu.get_rng(),n_states, start_width, response_width, delta, X, DT, measurement_prob, 
                   params_type = params_type, transition_type=transition_type, 
                   likelihood_type=likelihood_type, model_type=model_type)
    
    posterior_samples = get_original_params(posterior_samples, params_type, model_type)
    return posterior_samples


def sample_posterior_params(DT, X, n_states, start_width, response_width, delta, measurement_prob,
                            num_warmup=100, samples_n=500, num_chains=4, batch_size=2,  
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):

    #kernel = HMCECS(NUTS(model), num_blocks=10)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    #mcmc_chain.run(cu.get_rng(), n_states, start_width,  sigma, tau, DT, X, I, J, s_0, batch_size=batch_size, extra_fields=('hmc_state',))

    kernel = NUTS(model, forward_mode_differentiation=False)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)#, chain_method="vectorized")
    mcmc_chain.run(cu.get_rng(), n_states, start_width, response_width, delta, X, DT, measurement_prob, 
                   params_type = params_type, transition_type=transition_type, 
                   likelihood_type=likelihood_type, model_type=model_type,
                   extra_fields=('potential_energy',))

    #post_likl = mcmc_chain.get_extra_fields()['hmc_state'].potential_energy
    #post_likl = mcmc_chain.get_extra_fields()['potential_energy']
    return mcmc_chain#, post_likl

def predictive_mcmc_fn(n_states, response_width, delta, measurement_prob, X, 
                       drift_rate, diffusion_rate, phi_0,
                       params_type, model_type, transition_type, likelihood_type):
    
    #predictive_samples = []
    #for i, (drift_rate, diffusion_rate, phi_0), in enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples)):
    if likelihood_type == "SINGLE":
        kernel = NUTS(potential_fn= lambda RT_pred: 
                                    predictive_model(RT_pred, n_states, response_width, delta, measurement_prob, phi_0, 
                                                    X, drift_rate, diffusion_rate,
                                                    transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type,))
        pred_shape = 4, *X.shape
        predictive_mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=4)
        predictive_mcmc.run(cu.get_rng(), init_params=stats.lognorm(s=1).rvs((pred_shape)),
                        extra_fields=('potential_energy',)
                        )
        

    elif likelihood_type == "JOINT":
        kernel = NUTS(potential_fn= lambda RT_pred_s: 
                        predictive_model(RT_pred_s, n_states, response_width, delta, measurement_prob, phi_0, 
                                        X, drift_rate, diffusion_rate,
                                        transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type,))
        pred_shape = 4, 2, *X[0].shape
        predictive_mcmc = MCMC(kernel, num_warmup=30, num_samples=20, num_chains=4)
        predictive_mcmc.run(cu.get_rng(), init_params=stats.lognorm(s=1).rvs((pred_shape)),
                    extra_fields=('potential_energy',)
                    )
    
        #predictive_samples.append({"drift_rate":drift_rate, "diffusion_rate":diffusion_rate, "predictive_chain":az.from_numpyro(predictive_mcmc)})

    return {"drift_rate":drift_rate, "diffusion_rate":diffusion_rate, "predictive_chain":az.from_numpyro(predictive_mcmc)} #predictive_samples

def sample_prior_pred_params(n_states, start_width, response_width, delta, measurement_prob, X, RT=None,  
                        n_samples=10, data_samples=(1,10), #RT_max = 10,
                        params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", 
                        transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", sampling_type = "MCMC|GEN", n_jobs=1, key=None):

    prior_predictive = Predictive(model, num_samples=n_samples)    
    prior_samples = prior_predictive(cu.get_rng() if key is None else key, n_states, start_width, response_width, delta, X, None, measurement_prob,
                                    params_type = params_type, transition_type=transition_type, 
                                    likelihood_type=likelihood_type, model_type=model_type)
    
    drift_rate_samples = prior_samples["mu"]
    diffusion_rate_samples = prior_samples["sigma_final"]
    phi_0_samples = prior_samples["phi_0"]
    predictive_samples = []
    parallel = Parallel(n_jobs=n_jobs)#drift_rate_samples.shape[0])

    if sampling_type == "MCMC":
        predictive_samples = parallel(delayed(predictive_mcmc_fn)(n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0,
                                                params_type = params_type, model_type = model_type, transition_type = transition_type, likelihood_type = likelihood_type)
                                    for drift_rate, diffusion_rate, phi_0 in zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples)
                                    )
    elif sampling_type == "GEN" or sampling_type == "SIM":
            predictive_samples = parallel(delayed(get_RT)(RT, n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0, param_sample_id = param_sample_id,
                                                model_type = model_type, transition_type = transition_type, 
                                                likelihood_type = likelihood_type, data_samples = data_samples,
                                                sampling_type=sampling_type)
                                    for param_sample_id, (drift_rate, diffusion_rate, phi_0) in 
                                    enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples))
                                    )
    else:
        #raise Exception(f"Please select one of {sampling_type}")
        predictive_samples = dict(drift_rate = drift_rate_samples, diffusion_rate = diffusion_rate_samples, phi_0 = phi_0_samples)
    return predictive_samples

def sample_post_pred_params(n_states, response_width, delta, measurement_prob, X,
                            drift_rate_samples, diffusion_rate_samples, phi_0_samples, RT=None,
                            data_samples=(1,10), 
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", 
                            transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", sampling_type = "MCMC|GEN"):
    predictive_samples = []
    parallel = Parallel(n_jobs=drift_rate_samples.shape[0] if drift_rate_samples.shape[0] < 60 else 60)
    if sampling_type == "MCMC":
        predictive_samples = parallel(delayed(predictive_mcmc_fn)(n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0,
                                                params_type, model_type, transition_type, likelihood_type)
                                    for drift_rate, diffusion_rate, phi_0 in zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples)
                                    )
    elif sampling_type == "GEN" or sampling_type == "SIM":
            predictive_samples = parallel(delayed(get_RT)(RT, n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0, param_sample_id = param_sample_id,
                                                model_type = model_type, transition_type = transition_type, 
                                                likelihood_type = likelihood_type, data_samples = data_samples, 
                                                sampling_type=sampling_type)
                                    for param_sample_id, (drift_rate, diffusion_rate, phi_0) in 
                                    enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples))
                                    )
    else:
        predictive_samples = dict(drift_rate = drift_rate_samples, diffusion_rate = diffusion_rate_samples, phi_0 = phi_0_samples)
    #    raise Exception(f"Please select one of {sampling_type}")
    
    return predictive_samples

def get_arviz_model(mcmc_chain):
    return az.from_numpyro(mcmc_chain)

def get_intensity_matrix(n_states, mu, sigma, model_type="Markov|Quantum"):
    if model_type == "Markov":
        return dd._buildK(n_states, mu, sigma)
    elif model_type == "Quantum":
        return qd._buildH(n_states, mu, sigma)
    else:
        raise Exception(f"Please select one of {model_type}")

if __name__ == "__main__":

    #n_states, start_width, response_width, delta, measurement_prob, mu, sigma, I, J = 7, 4, 2, 1, 0.8, npx.asarray([[1]]), npx.asarray([[1]]), 10, 50
    n_states, start_width, response_width, delta, measurement_prob, mu, sigma, I, J = 51, None, 5, 1, 0.25, npx.asarray([[1]]), npx.asarray([[1]]), 10, 50
    start_width = (n_states-2*response_width)
    m_Mc, m_Mw, m_Mn = _get_measurement_matrix(n_states, 1, prob=measurement_prob, model_type = "Markov")
    q_Mc, q_Mw, q_Mn = _get_measurement_matrix(n_states, 1, prob=measurement_prob, model_type = "Quantum")
    
    X = stats.bernoulli(0.5).rvs(size=(I,J))
    RT = stats.lognorm(1,1).rvs(size=(I,J))
    post_chain = sample_posterior_params_VI(RT, X, n_states=n_states, start_width=start_width, response_width=response_width, 
                                         delta=delta,measurement_prob=measurement_prob,
                                         num_warmup=200, samples_n=200,
                                         params_type="NonCentralized", model_type="Markov", transition_type="TIMESTEP", likelihood_type="SINGLE" 
                            )
    print(post_chain.keys())

if False:

    log.debug("Constant Drift Rate - Mean Confidence 1")

    intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
    intensity_matrix_quantum = qd._buildH(n_states, mu, sigma)

    phi_0_markov = _get_initial_state(n_states, start_width,model_type="Markov", prior_type="Upper")
    phi_0_quantum = _get_initial_state(n_states, start_width,model_type="Quantum" , prior_type="Upper")

    mean_conf_quantum = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_quantum, 
                                            phi_0=phi_0_quantum, delta=1, Mn=q_Mn, t=npx.asarray([[10]]), transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Quantum")
    
    print(mean_conf_quantum)
    
    mean_conf_markov = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_markov, 
                                           phi_0=phi_0_markov, delta=delta, Mn=m_Mn, t=10, transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
    mean_conf_quantum = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_quantum, 
                                            phi_0=phi_0_quantum, delta=delta, Mn=q_Mn, t=10, transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")

    print(mean_conf_markov)
    print(mean_conf_quantum)

    log.debug("Constant Drift Rate - Mean Confidence 2")
    #import seaborn as sns
    #sns.set_context("paper", font_scale=1.5)
    mean_conf_markov_arr = []
    mean_conf_quantum_arr = []
    for t in np.arange(1, 15,0.001):

        mean_conf_markov = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_markov, 
                                           phi_0=phi_0_markov, delta=delta, Mn=m_Mn, t=t, transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
        mean_conf_markov_arr.append(mean_conf_markov.squeeze())
        mean_conf_quantum = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_quantum, 
                                            phi_0=phi_0_quantum, delta=delta, Mn=q_Mn, t=t, transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
        mean_conf_quantum_arr.append(mean_conf_quantum.squeeze())

    pd.Series(npx.asarray(mean_conf_markov_arr), name="Markov").plot()
    pd.Series(npx.asarray(mean_conf_quantum_arr), name="Quantum").plot()
    plt.xlabel("Response Time (in secs)")
    plt.ylabel("Evolution of Mean Confidence")
    plt.legend()
    plt.show()

    log.debug("Constant Drift Rate - Likelihood 1")

    likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                            RT_s=npx.asarray([[10, 20]]), RA_s=npx.asarray([[1, 0]]),  
                            Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                            transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Markov")
    
    likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                            RT_s=npx.asarray([[10, 30]]), RA_s=npx.asarray([[1, 0]]),  
                            Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                            transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Quantum")
    
    print(likl_markov)
    print(likl_quantum)

    log.debug("Constant Drift Rate - Likelihood 2")

    likl_markov_arr = []
    likl_quantum_arr = []

    for t in range(1,100):
        likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]),  
                                Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
        likl_markov_arr.append(likl_markov.squeeze())

        
        likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]),
                                Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
        likl_quantum_arr.append(likl_quantum.squeeze())
    
    pd.Series(npx.asarray(likl_markov_arr), name="Markov").plot()
    pd.Series(npx.asarray(likl_quantum_arr), name="Quantum").plot()
    plt.legend()
    plt.show()


    log.debug("Constant Drift Rate - Likelihood 3")

    likl_markov_arr = []
    likl_quantum_arr = []

    for t in range(1,100):
        likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[0]]),
                                Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
        likl_markov_arr.append(likl_markov.squeeze())

        
        likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[0]]), 
                                Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
        likl_quantum_arr.append(likl_quantum.squeeze())
    
    pd.Series(npx.asarray(likl_markov_arr), name="Markov").plot()
    pd.Series(npx.asarray(likl_quantum_arr), name="Quantum").plot()
    plt.legend()
    plt.show()

    log.debug("Constant Drift Rate - Likelihood 4")

    for mu, sigma in zip([npx.asarray([[1]]), npx.asarray([[0.5]]), npx.asarray([[10]]), npx.asarray([[-1]])],[npx.asarray([[1]]), npx.asarray([[10]]), npx.asarray([[0.05]]), npx.asarray([[1]])]):
        likl_markov_arr = []
        likl_quantum_arr = []
        intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
        intensity_matrix_quantum = qd._buildH(n_states, mu, sigma)

        for t in range(1,100):
            likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                    transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
            likl_markov_arr.append(likl_markov.squeeze())

            
            likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                    transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
            likl_quantum_arr.append(likl_quantum.squeeze())
        
        pd.Series(npx.asarray(likl_markov_arr), name=f"Markov:{mu}, {sigma}").plot()
        pd.Series(npx.asarray(likl_quantum_arr), name=f"Quantum:{mu}, {sigma}").plot()
        plt.legend()
        plt.show()

    log.debug("Constant Drift Rate - Likelihood 5")

    for mu, sigma in zip([npx.asarray([[1]]), npx.asarray([[0.5]]), npx.asarray([[10]]), npx.asarray([[-1]])],[npx.asarray([[1]]), npx.asarray([[10]]), npx.asarray([[0.05]]), npx.asarray([[1]])]):
        likl_markov_arr = []
        likl_quantum_arr = []
        intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
        intensity_matrix_quantum = qd._buildH(n_states, mu, sigma)

        for t in range(-10,100):
            likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                    transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Markov")
            likl_markov_arr.append(likl_markov.squeeze())

            
            likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                    transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Quantum")
            likl_quantum_arr.append(likl_quantum.squeeze())
        
        pd.Series(npx.asarray(likl_markov_arr), name=f"Markov:{mu}, {sigma}").plot()
        plt.legend()
        plt.show()
        pd.Series(npx.asarray(likl_quantum_arr), name=f"Quantum:{mu}, {sigma}").plot()
        plt.legend()
        plt.show()

    log.debug("Constant Drift Rate - Prior 1")

    X = stats.bernoulli(0.5).rvs(size=(I,J))
    RT = stats.lognorm(1,1).rvs(size=(I,J)) 

    predictive_samples = sample_prior_pred_params(n_states=n_states,start_width=start_width,response_width=response_width,
                                                  delta=delta,
                                                  measurement_prob=measurement_prob, X=X, RT=RT, 
                                                  n_samples=10,data_samples=X.shape,
                                                  params_type="Centralized", model_type="Quantum", transition_type="RT", 
                                                  likelihood_type="SINGLE", sampling_type="GEN", 
                                                 )
    # The predictive_samples contains posterior RT samples for each posterior parameter indexed by [0] below.
    #predictive_samples[0]["predictive_chain"]   
    #log.debug(az.summary(predictive_samples[0]["predictive_chain"]))
    df_samples = predictive_samples[0]["Samples"]
    df_sim_RT = predictive_samples[0]["Likelihood"]
    df_prior_all = pd.concat([samples["Samples"] for samples in predictive_samples])
    #sns.lineplot(df_prior_all, x="RT", y="logp", hue="param_sample_id")
    sns.kdeplot(df_prior_all, x="RT", hue="param_sample_id")
    #sns.histplot(df_prior_all, x="RT", hue="param_sample_id", multiple="dodge",element="bars")
    #plt.xlim(0,10) # because RT_max is set as 1000
    plt.show()

    log.debug("Constant Drift Rate - Prior 2")

    X = stats.bernoulli(0.5).rvs(size=(I,J))
    RT = stats.lognorm(1,1).rvs(size=(I,J)) 

    predictive_samples = sample_prior_pred_params(n_states=n_states,start_width=start_width,response_width=response_width,
                                                  delta=delta,
                                                  measurement_prob=measurement_prob, X=X, RT=RT, n_samples=2,
                                                  params_type="Centralized", model_type="Quantum", transition_type="RT", 
                                                  likelihood_type="SINGLE", sampling_type="MCMC"
                                                 )
    # The predictive_samples contains posterior RT samples for each posterior parameter indexed by [0] below.
    #predictive_samples[0]["predictive_chain"]   
    #log.debug(az.summary(predictive_samples[0]["predictive_chain"]))
    log.debug(f"Mean Rhat {az.rhat(predictive_samples[0]['predictive_chain'])['Param:0'].values.mean()}")     

    df_plot = pd.DataFrame()
    for i, prior_predictive_sample in enumerate(predictive_samples):
        #RT_pred = prior_predictive_sample["predictive_chain"]["posterior"]["Param:0"].values.reshape((-1, I, J))
        #mean_rt_pred_s = RT_pred.mean(axis=(0))
        #lp_s = predictive_samples[0]["predictive_chain"]["sample_stats"]["lp"].values
        #lp_s = predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values

        mean_rt_pred_s = prior_predictive_sample["predictive_chain"]["posterior"]["Param:0"].values.mean(axis=(-2,-1))
        lp_s = prior_predictive_sample["predictive_chain"]["sample_stats"]["lp"].values

    #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
        #sns.relplot(x=mean_rt_pred, y=lp, col=i)
        #sns.kdeplot(x=mean_rt_pred_s.flatten(), hue=i)
        df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), lp = lp_s.flatten(),
                                                        prior = i))])
    #sns.kdeplot(df_plot, x="mean_rt", hue="prior")
    sns.relplot(
        df_plot,
        x="mean_rt",
        y="lp",
        hue="prior"
        )
    plt.show()

    log.debug("Constant Drift Rate - Posterior Samples 1")

    X = stats.bernoulli(0.5).rvs(size=(I,J))
    RT = stats.lognorm(1,1).rvs(size=(I,J))
    post_chain = sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, response_width=response_width, 
                                         delta=delta,measurement_prob=measurement_prob,
                                         num_warmup=10, samples_n=10,
                                         params_type="Centralized", model_type="Quantum", transition_type="TIMESTEP", likelihood_type="SINGLE" 
                            )
    post_samples = post_chain.get_samples()
    #log.debug(az.summary(az.from_numpyro(post_chain)))

    log.debug("Constant Drift Rate - Post Predictive Samples 1")
    drift_rate_samples = post_samples["mu"][-5:,...]
    diffusion_rate_samples = post_samples["sigma_final"][-5:,...]
    phi_0_samples = post_samples["phi_0"][-5:,...]
    
    post_predictive_samples = sample_post_pred_params(n_states=n_states, start_width=start_width, 
                                                      response_width=response_width,
                                                      delta=delta,measurement_prob=measurement_prob,
                                                 X=X, 
                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
                                                 params_type="Centralized", model_type="Quantum", transition_type="RT", likelihood_type="SINGLE", sampling_type="MCMC"
                                                 )
    
    log.debug(f"Mean Rhat {az.rhat(post_predictive_samples[0]['predictive_chain'])['Param:0'].values.mean()}")  
    #log.debug(az.summary(post_predictive_samples[0]["predictive_chain"]))

    df_plot = pd.DataFrame()
    for i, post_pred_sample in enumerate(post_predictive_samples):  #Iterating over each posterior distribution
        #RT_pred = post_pred_sample["predictive_chain"]["posterior"]["Param:0"].values.reshape((-1, I, J))
        #mean_rt_pred_s = RT_pred.mean(axis=(0))
        mean_rt_pred_s = post_pred_sample["predictive_chain"]["posterior"]["Param:0"].values.mean(axis=(-2,-1))
        lp_s = post_pred_sample["predictive_chain"]["sample_stats"]["lp"].values
        #lp_s = post_predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values
    
        #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
        #    sns.relplot(x=mean_rt_pred, y=lp, col=i, kind="point")
        df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), lp = lp_s.flatten(),
                                                        posterior = i))])
    #sns.kdeplot(df_plot, x="mean_rt", hue="posterior")
    sns.relplot(
                df_plot,
                x="mean_rt",
                y="lp",
                hue="posterior"
                )
    plt.show()

    log.debug("Constant Drift Rate - Post Predictive Samples 2")
    drift_rate_samples = post_samples["mu"][-5:,...]
    diffusion_rate_samples = post_samples["sigma_final"][-5:,...]
    phi_0_samples = post_samples["phi_0"][-5:,...]
    
    post_predictive_samples = sample_post_pred_params(n_states=n_states, response_width=response_width, delta=delta,measurement_prob=measurement_prob,
                                                 X=X, 
                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
                                                 RT=RT,
                                                 params_type="Centralized", model_type="Quantum", transition_type="RT", 
                                                 likelihood_type="SINGLE", sampling_type="GEN"
                                                 )

    df_pred_all = pd.concat([samples["Samples"] for samples in post_predictive_samples])
    sns.lineplot(df_pred_all, x="RT", y="logp", hue="param_sample_id")
    sns.kdeplot(df_pred_all, x="RT", hue="param_sample_id")
    sns.histplot(df_pred_all, x="RT", hue="param_sample_id", multiple="dodge",element="bars")
    
    #df_samples = post_predictive_samples[0]["Samples"]
    #sns.kdeplot(df_samples.assign(hue = 
    #                              lambda df: df.mu.astype(str) + df.sigma.astype(str) + df.weighted_sample.astype(str)), 
    #            x="RT", hue="hue", legend=False)
    #plt.xlim(0,10) # because RT_max is set as 1000

if False:

    log.debug("Constant Drift Rate - Posterior Samples - Joint - 1")

    X_s = [stats.bernoulli(0.5).rvs(size=(I,J)), stats.bernoulli(0.5).rvs(size=(I,J))]
    RT_s = [stats.lognorm(1,1).rvs(size=(I,J)), stats.lognorm(1,1).rvs(size=(I,J))]
    post_chain_joint = sample_posterior_params(RT_s, X_s, n_states=n_states, start_width=start_width, response_width=response_width, 
                                               delta=delta,measurement_prob=measurement_prob,
                                                num_warmup=100, samples_n=100,
                                                params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="JOINT" 
                            )
    post_samples_joint = post_chain_joint.get_samples()


    log.debug("Constant Drift Rate - Post Predictive Samples - Joint - 1")
    drift_rate_samples = post_samples_joint["mu"][-2:,...]
    diffusion_rate_samples = post_samples_joint["sigma_final"][-2:,...]
    phi_0_samples = post_samples_joint["phi_0"][-2:,...]
    
    post_predictive_joint_samples = sample_post_pred_params(n_states=n_states, response_width=response_width, delta=delta,measurement_prob=measurement_prob,
                                                 X=X_s, 
                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
                                                 params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="JOINT"
                                                 )
    
    post_predictive_joint_samples[0]["predictive_chain"]  
    #log.debug(az.summary(post_predictive_samples[0]["predictive_chain"]))

    df_plot = pd.DataFrame()
    for i, post_pred_sample_joint in enumerate(post_predictive_joint_samples):  #Iterating over each posterior distribution
        RT_pred = post_pred_sample_joint["predictive_chain"]["posterior"]["Param:0"].values[:,:,0,...]
        RT_pred_1 = RT_pred.reshape((-1, I, J))
        RT_pred_2 = RT_pred.reshape((-1, I, J))
        mean_rt_pred_s = npx.asarray([RT_pred_1.mean(axis=(0)), RT_pred_2.mean(axis=(0))])
        #lp_s = post_predictive_samples[0]["predictive_chain"]["sample_stats"]["lp"].values
        #lp_s = post_predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values
    
        #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
        #    sns.relplot(x=mean_rt_pred, y=lp, col=i, kind="point")
        df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), 
                                                        posterior = i))])
    sns.kdeplot(df_plot, x="mean_rt", hue="posterior")
    plt.show()
    