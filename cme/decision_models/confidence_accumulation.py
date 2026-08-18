#from turtle import width
from turtle import pos
import jax.numpy as npx
import jax.scipy as sci
import numpyro as pyro
import numpyro as npy
import jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
import numpyro.infer.autoguide as ag
from jax import lax
import arviz as az 
from numpyro.distributions import constraints
from numpyro.infer.initialization import init_to_median

import numpy as np 
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed


from numpyro import enable_validation
enable_validation(True)

from cme.utils import common_logging as cl
from cme.utils import common_utils as cu
log = cl.get_logger("confidence_accumulation")

#pyro.set_platform("cpu")
pyro.set_host_device_count(64)
#pyro.enable_x64()

def diffusion_buildK(n_states, mu, sigma=1, delta=0.01, boundary_type = "External"): 
    mu = npx.asarray(mu) #Ix1
    n_part, n_mu = mu.shape #if len(npx.asarray(mu).shape) > 0 else 1
    K = npx.zeros((n_part, 1, n_states,n_states)) # participants, trials, transition states

    if n_mu == 1:
        mu=npx.repeat(mu,n_states,axis=1) # keeping mu constant over states

    b1 = 0.5 * (sigma - mu) #IxJ
    b2 = 0.5 * (sigma + mu) #IxJ
    a = -(b1+b2) #IxJ

    def _create(static_params, params):
        b1 = params["b1"] #scaler
        b2 = params["b2"] #scaler
        a = params["a"] #scaler
        K = static_params["K"] #n_states x n_states
        j = static_params["j"]
        K = K.at[0,[j-1,j,j+1],j].set([b1, a, b2])
        
        static_params = {"j":j+1, "K":K}
        return (static_params,params)
    
    def _create_i(i, params):
        params_j = {"b1":params["b1"], "b2":params["b2"], "a":params["a"]}
        static_params = {"j":0, "K":params["K"]}
        
        static_params, params_j = lax.scan(_create, static_params, params_j)#, unroll=True)
        params_j["K"] = static_params["K"]
        return (i, params_j)    

    
    params = {"b1":b1, "b2":b2, "a":a, "K":K}
    i, params = lax.scan(_create_i, 0, params)#, unroll=True)
    K = params["K"]

    K = K.at[:,0,:,0].set(0) 
    K = K.at[:,0,:,-1].set(0)

    if boundary_type == "External":
        K = K.at[:,0,[0,1],0].set(npx.asarray([a[:,0], -a[:,0]]).T)
        K = K.at[:,0,[-2,-1],-1].set(npx.asarray([-a[:,-1], a[:,-1]]).T)
        
    return K


def quantum_buildH(n_states, mu, sigma, delta=0.001, n_trials = None): 
    
    mu = npx.asarray(mu) #Ix1
    n_part, _ = mu.shape

    # build Hamiltonian  
    Mid = int((n_states+1)/2)
    mv = np.arange(-(Mid-1),(Mid)) #np.arange(0,n_states) #np.arange(-(Mid-1),(Mid))  # Basis vector
    b = (mu[...,None])*mv[None,None,:];  # I,1,n_states
    a = sigma#*np.ones((ns,1));  Ix1
    c=a

    H = npx.zeros((n_part,
                   1 if n_trials is None else n_trials,
                   n_states, n_states))

    def _create(i, params):
        a = params["a"]
        b = params["b"]
        c = params["c"]
        H = params["H"]
        
        rows_ = npx.arange(1,n_states)
        cols_ = npx.arange(0,n_states-1)
        diags_ = npx.arange(0, n_states)
        H = H.at[:,rows_,cols_].set(c[0])
        H = H.at[:,cols_, rows_].set(a[0])
        if n_trials is None:
            H = H.at[0,diags_, diags_].set(b[0,...])
        else:
            for n in range(n_trials):
                H = H.at[n,diags_, diags_].set(b[0,...])

        params["H"] = H
        return (i, params)    
    
    params = {"a":a, "b":b, "c":c, "H":H}

    i, params = lax.scan(_create, 0, params)#, unroll=True)

    return -1j * params["H"] # The -1j is being multiplied here to simplify the transaction multiplication operations.


def centralized_parameters(I):
    """
    I: Number of participants
    
    """
    mu_m =  pyro.sample(f"mu_m", dist.Normal(0,1))
    mu_s =  pyro.sample(f"mu_s", dist.HalfNormal(2))
    with pyro.plate('I6', I, dim=-2):
        mu = pyro.sample("mu", dist.Normal(mu_m,mu_s)) # Drift Rate
        sigma = pyro.sample("sigma", dist.Normal(1,0.1)) # Diffusion Rate
    return mu, sigma

# def non_centralized_parameters(model_type, I):
#     """
#     I: Number of participants
#     Model-specific priors for improved numerical stability
#     """
#     m = pyro.sample("m", dist.Normal(0.1,0.1))
#     #m = pyro.deterministic("m", 0.1)
#     s = pyro.sample("s", dist.HalfNormal(0.1))

#     # Quantum models need tighter prior control on sigma scale
#     if model_type == "Quantum":
#         m_si = pyro.sample("m_si", dist.Normal(0.5, 0.5))  # Shifted to ensure positive softplus output
#         s_si = pyro.sample("s_si", dist.HalfNormal(0.05))  # Tighter to prevent extreme values
#     else:  # Markov
#         m_si = pyro.sample("m_si", dist.Normal(0, 1))
#         s_si = pyro.sample("s_si", dist.HalfNormal(0.1))

#     with pyro.plate('I3', I, dim=-2):
#         if model_type == "Markov":
#             mu_r = pyro.sample("mu_r", dist.Normal(0.1,1)) # Drift Rate
#             mu = pyro.deterministic("mu", m + s * mu_r)
#         elif model_type == "Quantum":
#             mu_r = pyro.sample("mu_r", dist.Normal(0.1,1)) # Drift Rate
#             mu = pyro.deterministic("mu", jax.nn.softplus(m + s * mu_r))
       
#         if model_type == "Markov":
#             sigma_r = pyro.sample("sigma_r", dist.Normal(0,0.1)) # Diffusion Rate
#         elif model_type == "Quantum":
#             sigma_r = pyro.sample("sigma_r", dist.Normal(0,0.1)) # Diffusion Rate

        
#         sigma_base = jax.nn.softplus(m_si + s_si * sigma_r)
        
#         # Ensure minimum floor for numerical stability in Quantum likelihood
#         if model_type == "Quantum":
#             sigma = pyro.deterministic("sigma", npx.clip(sigma_base, 0.01, None))
#         else:
#             sigma = pyro.deterministic("sigma", sigma_base)
    
#     return mu, sigma


def non_centralized_parameters(model_type, I):
    # Fixed prior location/scale for drift
    m = pyro.deterministic("m", npx.asarray(0.1))
    s = pyro.deterministic("s", npx.asarray(0.1))

    # Fixed prior location/scale for sigma
    if model_type == "Quantum":
        m_si = pyro.deterministic("m_si", npx.asarray(0.5))
        s_si = pyro.deterministic("s_si", npx.asarray(0.05))
    else:  # Markov
        m_si = pyro.deterministic("m_si", npx.asarray(0.0))
        s_si = pyro.deterministic("s_si", npx.asarray(0.1))

    with pyro.plate("I3", I, dim=-2):

        # Drift
        if model_type == "Markov":
            mu_r = pyro.sample("mu_r", dist.Normal(0.1, 1.0))
            mu = pyro.deterministic("mu", m + s * mu_r)

        elif model_type == "Quantum":
            mu_r = pyro.sample("mu_r", dist.Normal(0.1, 1.0))
            mu = pyro.deterministic("mu", jax.nn.softplus(m + s * mu_r))

        else:
            raise Exception(f"Please select one of {model_type}")

        # Diffusion
        sigma_r = pyro.sample("sigma_r", dist.Normal(0.0, 0.1))

        sigma_base = jax.nn.softplus(m_si + s_si * sigma_r)

        if model_type == "Quantum":
            sigma = pyro.deterministic("sigma", npx.clip(sigma_base, 0.01, None))
        else:
            sigma = pyro.deterministic("sigma", sigma_base)

    return mu, sigma


def _timestep_transition_matrix(n, T_delta, Mn):
    """
    n: I x J
    T_delta: I x 1 x S x S
    Mn: S x S
    T_step: I x 1 x S x S
    T_i before trial selection: K x I x 1 x S x S
    T_i after trial selection: I x J x S x S
    """
    # T_i = []
    # for n_i, T_delta_i in zip(n, T_delta):
    #     T_i_j = []
    #     for n_i_j in n_i:
    #         #T_delta_i_j = T_delta_i[j,...]
    #         T_nt = npx.linalg.matrix_power(Mn @ T_delta_i[0,...], n_i_j.astype(int).item() - 1) # we need to vectorize this function
    #         T_i_j.append(T_nt)
    
    #     T_i.append(T_i_j)
    
    # T_t = T_delta @ npx.asarray(T_i)
    # #T_t = npx.asarray(T_i) # uncomment to include all response time
    # return T_t

    n = n.astype(int)
    if np.any(n < 1):
        raise ValueError("timestep counts must be at least one", n)

    T_step = Mn @ T_delta
    T_identity = npx.broadcast_to(npx.eye(T_step.shape[-1], dtype=T_step.dtype), T_step.shape)

    def _matrix_power(T_nt, _):
        T_nt = T_nt @ T_step
        return T_nt, T_nt

    _, T_i = lax.scan(_matrix_power, T_identity, None, length=int(n.max().item()) - 1)
    T_i = npx.concatenate((T_identity[None,...], T_i), axis=0)
    T_n = n - 1 # I x J, containing K-axis indices
    T_participant = npx.arange(n.shape[0])[:,None] # I x 1, containing I-axis indices
    T_participant = npx.broadcast_to(T_participant, n.shape) # I x J, containing I-axis indices
    T_i = T_i[T_n, T_participant, 0, ...] # I x J x S x S

    T_t = T_delta @ T_i
    #T_t = npx.asarray(T_i) # uncomment to include all response time
    return T_t

def _get_transition_matrix(intensity_matrix, RT, delta=None, Mn = None, transition_type="RT|TIMESTEP"):
   
    if transition_type == "RT":
        T_t = sci.linalg.expm(intensity_matrix * ((RT[...,None,None]) if not npx.isscalar(RT) else (RT)))
    elif transition_type == "TIMESTEP":
        ns=np.round(RT/delta) 
        T_delta = sci.linalg.expm(intensity_matrix * delta)
        T_t = _timestep_transition_matrix(ns, T_delta, Mn)  
    else:
        raise Exception(f"Please select one of {transition_type}")

    return T_t # I x J x n_state x n_state

def _get_measurement_matrix(n_states, response_width, prob=0.5, model_type = "Markov|Quantum"):

    if model_type == "Markov":
        Mcorr = npx.zeros(n_states)
        Mcorr = Mcorr.at[-response_width:].set(prob)
        Mcorr = npx.diag(Mcorr)

        Mincorr = npx.zeros(n_states)
        Mincorr = Mincorr.at[:response_width].set(prob)
        Mincorr = npx.diag(Mincorr)
        Mnoresp = npx.eye(n_states) - Mcorr - Mincorr
    elif model_type == "Quantum":
        Mcorr = npx.zeros(n_states)
        Mcorr = Mcorr.at[-response_width:].set(npx.sqrt(prob))
        Mcorr = npx.diag(Mcorr)

        Mincorr = npx.zeros(n_states)
        Mincorr = Mincorr.at[:response_width].set(npx.sqrt(prob))
        Mincorr = npx.diag(Mincorr)
        Mnoresp = npx.sqrt(npx.eye(n_states) - (Mcorr**2 + Mincorr**2))
    else:
        raise Exception(f"Please select one of {model_type}")
    return Mcorr, Mincorr, Mnoresp

def _get_initial_state(n_states, start_width, response_width, I = 1, prob=1, model_type = "Markov|Quantum", prior_type="Upper|Lower|Centered|All|Model"):
    if prior_type == "Model":
        if model_type == "Markov":
            with npy.plate('I1', I, dim=-4):
                with npy.plate('S', n_states - 2*response_width, dim=-1):
                    conc = npy.sample("phi_conc", dist.Beta(0.5,0.5))+0.01 #to avoid 0

            with npy.plate('I2', I, dim=-3):
                p_0 = npy.sample("phi_init", dist.Dirichlet(conc)) # Initial State
            p_0 = npx.pad(p_0, ((0,0),(0,0),(0,0),(response_width,response_width)))
            phi_0 = npy.deterministic("phi_0", p_0.transpose(0,1,3,2)) #.transpose(0,1,3,2)  
        elif model_type == "Quantum":
            with npy.plate('I1', I, dim=-4):
                with npy.plate('S', n_states - 2*response_width, dim=-1):
                    conc = npy.sample("phi_conc", dist.Beta(0.5,0.5))+0.01 #to avoid 0

            with npy.plate('I2', I, dim=-3):
                p_0 = npy.sample("phi_init", dist.Dirichlet(conc)) # Initial State
                
            p_0 = npx.pad(p_0, ((0,0),(0,0),(0,0),(response_width,response_width)))
            phi_0 = npy.deterministic("phi_0", p_0.transpose(0,1,3,2)**(1/2))
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
            
        elif prior_type == "All": 
            pad_width = (0,0)
            width = n_states
        elif prior_type == "Opposite":
            pad_width = int((width+1)/2 )
            width = pad_width
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
                        model_type = "Markov|Quantum", return_type="Probability|ResponseConfidence|MeanConfidence"):
    
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
 
    # if(return_type == "Probability"):
    #     phi_t_c = Mc @ phi_t
    #     phi_t_w = Mw @ phi_t

    #     # This is a bug and will not work
    #     if(likelihood_type == "SINGLE"):
    #         phi_t = phi_t_c if x==1 else phi_t_w
    #     elif(likelihood_type == "JOINT"):
    #         phi_t = phi_t_c if x[1]==1 else phi_t_w 

    if model_type == "Markov":
        P_t = phi_t
    elif model_type == "Quantum":
        P_t = npx.abs(phi_t)**2
    
    Mid = (n_states+1)//2
    mv = npx.arange(-(Mid-1), (Mid))
    if conf_scale is not None:
        add_scale, mul_scale = conf_scale
        mv = cu.get_conf_scale(mv, add_scale, mul_scale, n_states)

    if return_type == "Probability":
        ret_val = P_t.sum()

    elif return_type == "ResponseConfidence":
        phi_t_c = Mc @ phi_t
        phi_t_w = Mw @ phi_t
        if model_type == "Markov":
            P_t_c = phi_t_c
            P_t_w = phi_t_w
        elif model_type == "Quantum":
            P_t_c = npx.abs(phi_t_c)**2
            P_t_w = npx.abs(phi_t_w)**2

        #P_t = npx.where(x==1,P_t_c.sum(axis=(-1,-2)),P_t_w.sum(axis=(-1,-2)))
        ret_val_c = mv[None, None, None,:] @ P_t_c
        ret_val_w = mv[None, None, None,:] @ P_t_w
        ret_val = npx.where(x[...,None,None]==1,ret_val_c,ret_val_w)

    else: #if return_type == "MeanConfidence":
        ret_val = mv[None, None, None,:] @ P_t
    
    return ret_val


def transformed_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):
    
    return estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)

def estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):
    P_t = likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    #P_t = npx.where(P_t <= 0, 0.00001, npx.log(P_t))
    # P_t = npx.where(P_t <= 0, 10e-12, P_t)
    # P_t = npx.where(npx.isnan(P_t), 10e-12, P_t)
    # P_t = npx.log(P_t)
    # #pyro.deterministic("loglikl", P_t.sum(axis=-1)) #summing over trials
    # return P_t.sum(axis=-1)
    eps = 1e-15
    eps_nan = 1e-12 # having two different eps for debugging purposes.

    pyro.deterministic("P_min", npx.min(P_t))
    pyro.deterministic("P_max", npx.max(P_t))
    pyro.deterministic("P_clip_low", npx.mean(P_t <= eps))
    pyro.deterministic("P_clip_high", npx.mean(P_t >= 1.0))

    P_t = npx.where(npx.isnan(P_t), eps_nan, P_t)
    P_t = npx.clip(P_t, eps, 1.0)
    P_t = npx.log(P_t)
    #pyro.deterministic("loglikl", P_t.sum(axis=-1))
    return P_t#.sum(axis=-1) For LOO and other model comparison, I need trialvise log-likelihood.

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
    if (np.unique(RA).shape[0] <= 2):
        P_t = npx.where(RA[...,None,None]==1, P_t_c, P_t_w)
    elif (np.unique(RA).shape[0] == 3):
        P_t_n = (Mn @ phi_t)
        P_t = npx.where(RA[...,None,None]==1, P_t_c, npx.where(RA[...,None,None]==-1, P_t_w, P_t_n))
    else:
        raise Exception("Unique RA values unexpected: ", np.unique(RA))

    if model_type == "Markov":
        
        P_t = P_t.sum(axis=(-2,-1)) # Adding over states
        
    elif model_type == "Quantum":
        
        P_t = (npx.abs(P_t)**2).sum(axis=(-2,-1)) #Adding over states
        
    else:
        raise Exception(f"Please select one of {model_type}")
  
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
        mu, sigma = non_centralized_parameters(model_type, I)
    #elif params_type == "ParticipantLevel":
    #    mu, sigma = participant_parameters(model_type, I)
    else:
        raise Exception(f"Please select one of {params_type}")

    if model_type == "Markov":
        sigma = pyro.deterministic("sigma_final",npx.abs(mu) + sigma) # Sigma needs to be larger than mu and Sigma cannot be negative
                # removed sigma**2 to allow stability in parameter estimates. Negative values are avoided through softplus now
        intensity_matrix = diffusion_buildK(n_states, mu, sigma, delta)

    elif model_type == "Quantum":
        # For Quantum: ensure sigma > 0 and has numerical stability
        # Consider making sigma magnitude scale with mu for better parameter coupling
        sigma_quantum = npx.clip(npx.abs(mu) * 0.5 + sigma, 0.01, None)
        sigma = pyro.deterministic("sigma_final", sigma_quantum)
        intensity_matrix = quantum_buildH(n_states, mu, sigma, delta)
    else:
        raise Exception(f"Please select one of {model_type}")

    phi_0 = _get_initial_state(n_states, start_width, response_width, I = I, prob=1, model_type = model_type, prior_type="Model")
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)

    if RT_s is not None:
        likl = estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, 
                          transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
        #likl = npx.log(likl)
        pyro.deterministic("likl_rt", likl)
        pyro.factor("likelihood", likl) #.sum()

# def gen_RT(RT, n_states, response_width, delta, measurement_prob, RA, 
#                      drift_rate, diffusion_rate, phi_0, data_samples = (1,10), 
#                      model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", 
#                      key=None, max_RT_sec=50
#                      ):
#     #threshold = 0.85 # or 85
#     #key1 = cu.get_rng() if key is None else key
#     part_I, part_J = data_samples
#     max_samples = part_J * 10
#     I, mu, sigma = part_I, drift_rate, diffusion_rate

#     #random_ts = stats.uniform.rvs(delta, max_RT_sec/delta, (I,max_samples))     #dist.Uniform(delta, max_RT_sec/delta).sample(key=key1, sample_shape=(I,max_samples))
    
#     #intensity_matrix = get_intensity_matrix(n_states, mu, sigma, model_type=model_type)
#     #Mc, Mw, Mn = _get_measurement_matrix(n_states, response_width, prob=measurement_prob, model_type = model_type)
        
#     return RT, RA, None

def get_RT(RT, n_states, response_width, delta, measurement_prob, RA, 
                     drift_rate, diffusion_rate, phi_0, data_samples = (1,10), min_RT_sec = 0, max_RT_sec=10, param_sample_id=-1,
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", 
                     sampling_type = "GEN|SIM", is_test=False, key=None
                     ):
    
    def sim_RT():
        """
            This function calculates likelihood for one dataset of size I,J
        """
        sim_RT = []
        likl = simulate_likelihood(RT, n_states, response_width, delta, measurement_prob, phi_0, RA, 
                        drift_rate, diffusion_rate, 
                        model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type)

        res_RT = sim_RT #Parallel(n_jobs=50)(fn() for fn in sim_RT)

        df_sim_RT = (pd.DataFrame(RT).assign(drift_rate=drift_rate, diffusion_rate=diffusion_rate)
        .reset_index(names="part_id")
        .melt(id_vars=["part_id", "drift_rate", "diffusion_rate"], var_name="pseudo_item_id", value_name="RT")
        .assign(RA = RA.flatten())
        .set_index(["part_id","pseudo_item_id"])
        .join(pd.DataFrame(likl)
            .reset_index(names="part_id")
            .melt(id_vars="part_id", var_name="pseudo_item_id", value_name="logp")
            .set_index(["part_id","pseudo_item_id"]))
        )
        df_sim_RT = df_sim_RT.where(lambda df:~np.isnan(df),0)
        samples_arr = []
        df_sim_RT = df_sim_RT.assign(logp = lambda df:np.absolute(df.logp), param_sample_id = param_sample_id)
        try:
            df_samples = df_sim_RT.groupby(["part_id"]).sample(n=part_J,replace=True, weights="logp", random_state= np.random.default_rng()).assign(weighted_sample=True) #.assign(weighted_sample=i))
        except Exception as e:
            log.error(f"************Sampling failed: {e}, max drift rate: {drift_rate.max()}; likelihood sum:{df_sim_RT.loc[:,'logp'].values.sum():.2f}. Sampled without weights!***********")
            df_samples = df_sim_RT.groupby(["part_id"]).sample(n=part_J,replace=True, random_state= np.random.default_rng()).assign(weighted_sample=False) #.assign(weighted_sample=-i))

        return df_samples, df_sim_RT
    
    part_I, part_J = data_samples
    max_J = part_J if part_J > 100 else part_J if RA is not None else 100
    if sampling_type == "SIM":
        if RT is None:
            #rt = np.arange(0,max_RT_sec,delta)
            rt = np.linspace(min_RT_sec, max_RT_sec, max_J)
            RT = np.tile(rt, (part_I,1))
        df_samples, df_sim_RT = sim_RT()
    elif sampling_type == "GEN":
        
        if RT is None:
            #rt = np.arange(delta,max_RT_sec,delta)
            rt = np.linspace(min_RT_sec, max_RT_sec, max_J)
            RT = np.tile(rt, (part_I,1))
        
        
        if RA is None:
            RA_prob = stats.beta(1,1).rvs(size = max_J)
            RA = np.vstack([stats.bernoulli(RA_prob).rvs() for _ in range(part_I)])
        df_samples, df_sim_RT = sim_RT()

    return {"drift_rate":drift_rate, "diffusion_rate":diffusion_rate, "initial_state":phi_0, "Likelihood":df_sim_RT, "Samples":df_samples}

def simulate_likelihood(RT_pred, n_states, response_width, delta, measurement_prob, phi_0, RA, 
                     drift_rate, diffusion_rate, 
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)
    
    if model_type == "Markov":
        intensity_matrix = diffusion_buildK(n_states, drift_rate, diffusion_rate, delta)

    elif model_type == "Quantum":
        intensity_matrix = quantum_buildH(n_states, drift_rate, diffusion_rate, delta)
    else:
        raise Exception(f"Please select one of {model_type}")
        
    likl = likelihood(intensity_matrix, phi_0, delta, RT_pred, RA, Mc, Mw, Mn, 
                      transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    
    return likl

def predictive_model(RT_pred, n_states, response_width, delta, measurement_prob, phi_0, RA, 
                     drift_rate, diffusion_rate, 
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)
    
    if model_type == "Markov":
        intensity_matrix = diffusion_buildK(n_states, drift_rate, diffusion_rate)

    elif model_type == "Quantum":
        intensity_matrix = quantum_buildH(n_states, drift_rate, diffusion_rate)
    else:
        raise Exception(f"Please select one of {model_type}")
    
    likl = transformed_likelihood(intensity_matrix, phi_0, delta, RT_pred, RA, Mc, Mw, Mn, 
                      transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    return likl.sum()

# def guide(n_states, start_width, response_width, delta, RA_s, RT_s, measurement_prob, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
#     I, _ = RA_s.shape
#     mu, sigma = non_centralized_parameters_VI(I)
#     if model_type == "Markov":
#         sigma = pyro.deterministic("sigma_final",npx.abs(mu) + sigma) # Sigma needs to be larger than mu and Sigma cannot be negative
#         intensity_matrix = diffusion_buildK(n_states, mu, sigma, delta)

#     elif model_type == "Quantum":
#         sigma = pyro.deterministic("sigma_final",sigma) # Sigma cannot be negative
#         intensity_matrix = quantum_buildH(n_states, mu, sigma, delta)
#     else:
#         raise Exception(f"Please select one of {model_type}")

#     phi_0 = _get_initial_state(n_states, start_width, I = I, prob=1, model_type = model_type, prior_type="Model")


def get_original_params(posterior_samples, response_width, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum"):
    if params_type == "Centralized":
        pass
    elif params_type == "NonCentralized":
        m = posterior_samples["m"]
        s = posterior_samples["s"]

        m_si = posterior_samples["m_si"]
        s_si = posterior_samples["s_si"]


        mu_r = posterior_samples["mu_r"]
        sigma_r = posterior_samples["sigma_r"]

        posterior_samples["mu"] = m[:,None,None] + s[:,None,None] * mu_r
        posterior_samples["sigma"] = jax.nn.softplus(m_si[:,None,None] + s_si[:,None,None] * sigma_r) #(m[:,None,None] + s[:,None,None] * sigma_r)**2 
    
    p_0 = posterior_samples["phi_init"]

    if model_type == "Markov":
        p_0 = p_0.transpose(0, 1, 2, 4, 3)
        posterior_samples["sigma_final"] = npx.abs(posterior_samples["mu"]) + posterior_samples["sigma"] 

    elif model_type == "Quantum":
        p_0 = p_0.transpose(0, 1, 2, 4, 3)**(1/2)
        posterior_samples["sigma_final"] = posterior_samples["sigma"]

    posterior_samples["phi_0"] = npx.pad(p_0, ((0,0),(0,0),(0,0),(response_width,response_width),(0,0)))

    # Commenting these out so that point estimate is not calculated. That way Bayesian model checks can be used.
    # posterior_samples["mu"] = posterior_samples["mu"].mean(axis=0, keepdims=True)
    # posterior_samples["sigma_final"] = posterior_samples["sigma_final"].mean(axis=0, keepdims=True)
    # posterior_samples["phi_0"] = posterior_samples["phi_0"].mean(axis=0, keepdims=True)

    return posterior_samples

from optax import adam, chain, clip
def sample_posterior_params_VI(DT, X, n_states, start_width, response_width, delta, measurement_prob,
                            num_warmup=100, samples_n=500, num_chains=4, batch_size=2,  
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    #guide = ag.AutoNormal(model)
    #guide = ag.AutoDiagonalNormal(model)
    #guide = ag.AutoMultivariateNormal(model)
    guide = ag.AutoLowRankMultivariateNormal(model)
    #guide = ag.AutoDAIS(model)
    #guide = ag.AutoDelta(model)

    #optimizer = Adam(step_size=0.5)
    #svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi = SVI(model, guide, chain(clip(10.0), adam(1e-3)), loss=Trace_ELBO())
    svi_result = svi.run(cu.get_rng(), num_warmup + samples_n, n_states, start_width, 
                        response_width, delta, X, DT, measurement_prob, 
                        params_type = params_type, transition_type=transition_type, 
                        likelihood_type=likelihood_type, model_type=model_type, stable_update=True)

    predictive = Predictive(guide, params=svi_result.params, num_samples=samples_n, parallel=True)
    posterior_samples = predictive(cu.get_rng(),n_states, start_width, response_width, delta, X, DT, measurement_prob, 
                   params_type = params_type, transition_type=transition_type, 
                   likelihood_type=likelihood_type, model_type=model_type)
    
    posterior_samples = get_original_params(posterior_samples, response_width, params_type, model_type)
    return posterior_samples


def sample_posterior_params(DT, X, n_states, start_width, response_width, delta, measurement_prob,
                            num_warmup=100, samples_n=500, num_chains=4, batch_size=2,  
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):

    #kernel = HMCECS(NUTS(model), num_blocks=10)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    #mcmc_chain.run(cu.get_rng(), n_states, start_width,  sigma, tau, DT, X, I, J, s_0, batch_size=batch_size, extra_fields=('hmc_state',))

    # Adaptive mass matrix and increased target_accept_prob for better convergence with non-centered params
    kernel = NUTS(model, forward_mode_differentiation=False, adapt_mass_matrix=True, adapt_step_size = True, 
                  dense_mass=True, init_strategy=init_to_median(num_samples=20),
                  target_accept_prob=0.8 if model_type=="Quantum" else 0.9)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains, 
                      chain_method="vectorized" if jax.default_backend() == "gpu" else "parallel",
                      progress_bar=True, jit_model_args=False)
    mcmc_chain.run(cu.get_rng(), n_states, start_width, response_width, delta, X, DT, measurement_prob, 
                   params_type = params_type, transition_type=transition_type, 
                   likelihood_type=likelihood_type, model_type=model_type,
                   extra_fields=('potential_energy',))

    return mcmc_chain#, post_likl

def predictive_mcmc_fn(n_states, response_width, delta, measurement_prob, X, 
                       drift_rate, diffusion_rate, phi_0,
                       params_type, model_type, transition_type, likelihood_type):
    
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

    return {"drift_rate":drift_rate, "diffusion_rate":diffusion_rate, "predictive_chain":az.from_numpyro(predictive_mcmc)} #predictive_samples

def sample_prior_pred_params(n_states, start_width, response_width, delta, measurement_prob, X, RT=None,  
                        n_samples=10, data_samples=(1,10), min_RT_sec = 0, max_RT_sec = 10,
                        params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", 
                        transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", sampling_type = "MCMC|GEN", n_jobs=1, key=None):

    prior_predictive = Predictive(model, num_samples=n_samples, parallel=True)    
    if X is None:
        #X = np.ones(data_samples)
        raise Exception("X cannot be missing")
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
                                                drift_rate, diffusion_rate, phi_0, min_RT_sec = min_RT_sec,  max_RT_sec = max_RT_sec,
                                                param_sample_id = param_sample_id,
                                                model_type = model_type, transition_type = transition_type, 
                                                likelihood_type = likelihood_type, data_samples = data_samples,
                                                sampling_type=sampling_type)
                                    for param_sample_id, (drift_rate, diffusion_rate, phi_0) in 
                                    enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples))
                                    )
    else:
        predictive_samples = dict(drift_rate = drift_rate_samples, diffusion_rate = diffusion_rate_samples, phi_0 = phi_0_samples)
    return predictive_samples

def sample_post_pred_params(n_states, response_width, delta, measurement_prob, X,
                            drift_rate_samples, diffusion_rate_samples, phi_0_samples, RT=None, 
                            data_samples=(1,10), min_RT_sec = 0, max_RT_sec=10,
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", 
                            transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", 
                            sampling_type = "MCMC|GEN", is_parallel=True):
    if X is None:
        raise Exception("X cannot be missing")
    predictive_samples = []
    n_jobs = drift_rate_samples.shape[0] if drift_rate_samples.shape[0] < 60 else 60
    parallel = Parallel(n_jobs=1 if not is_parallel else n_jobs)
    if sampling_type == "MCMC":
        predictive_samples = parallel(delayed(predictive_mcmc_fn)(n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0,
                                                params_type, model_type, transition_type, likelihood_type)
                                    for drift_rate, diffusion_rate, phi_0 in zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples)
                                    )
    elif sampling_type == "GEN" or sampling_type == "SIM":
            predictive_samples = parallel(delayed(get_RT)(RT, n_states, response_width, delta, measurement_prob, X, 
                                                drift_rate, diffusion_rate, phi_0, min_RT_sec = min_RT_sec, max_RT_sec=max_RT_sec,
                                                param_sample_id = param_sample_id,
                                                model_type = model_type, transition_type = transition_type, 
                                                likelihood_type = likelihood_type, data_samples = data_samples, 
                                                sampling_type=sampling_type)
                                    for param_sample_id, (drift_rate, diffusion_rate, phi_0) in 
                                    enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples))
                                    )
    else:
        predictive_samples = dict(drift_rate = drift_rate_samples, diffusion_rate = diffusion_rate_samples, phi_0 = phi_0_samples)
    
    return predictive_samples

def get_arviz_model(mcmc_chain):
    return az.from_numpyro(mcmc_chain)

def get_intensity_matrix(n_states, mu, sigma, model_type="Markov|Quantum"):
    if model_type == "Markov":
        return diffusion_buildK(n_states, mu, sigma)
    elif model_type == "Quantum":
        return quantum_buildH(n_states, mu, sigma)
    else:
        raise Exception(f"Please select one of {model_type}")

