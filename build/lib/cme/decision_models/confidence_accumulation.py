#from turtle import width
import jax.numpy as npx
import jax.scipy as sci
from jax import lax

import numpy as np
import pandas as pd
import scipy.stats as stats

from cme.utils import common_logging as cl
from cme.utils import common_utils as cu
log = cl.get_logger("confidence_accumulation")

# This module is the model: it builds the intensity and measurement matrices,
# evolves the state, generates response data, and evaluates the likelihood.
# It holds no inference machinery and imports no numpyro - everything that
# estimates parameters from data lives in `cme.inference`.


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

def _timestep_transition_state(n, T_delta, Mn, phi_0):
    """
    n: I x J
    T_delta: I x 1 x S x S
    Mn: S x S
    phi_0: I x 1 x S x 1 or I x J x S x 1
    T_step: I x 1 x S x S
    phi_i before trial selection: K x I x 1 x S x 1 or K x I x J x S x 1
    phi_i after trial selection: I x J x S x 1
    phi_t: I x J x S x 1

    This calculates the same equation as _timestep_transition_matrix:
    T_delta @ matrix_power(Mn @ T_delta, n-1) @ phi_0.
    """
    n = n.astype(int)
    if np.any(n < 1):
        raise ValueError("timestep counts must be at least one", n)

    T_step = Mn @ T_delta # I x 1 x S x S
    phi_0 = phi_0.astype(npx.result_type(T_step, phi_0)) # I x 1 x S x 1 or I x J x S x 1

    def _state_transition(phi_i, _):
        phi_i = T_step @ phi_i # I x 1 x S x 1 or I x J x S x 1
        return phi_i, phi_i

    _, phi_i = lax.scan(_state_transition, phi_0, None, length=int(n.max().item()) - 1)
    phi_i = npx.concatenate((phi_0[None,...], phi_i), axis=0) # K x I x 1 x S x 1 or K x I x J x S x 1
    T_n = n - 1 # I x J, containing K-axis indices
    T_participant = npx.arange(n.shape[0])[:,None] # I x 1, containing I-axis indices
    T_participant = npx.broadcast_to(T_participant, n.shape) # I x J, containing I-axis indices
    T_trial = npx.zeros(n.shape, dtype=int) # I x J, containing the shared J-axis index
    if phi_0.shape[1] != 1:
        T_trial = npx.arange(n.shape[1])[None,:] # 1 x J, containing J-axis indices
        T_trial = npx.broadcast_to(T_trial, n.shape) # I x J, containing J-axis indices
    phi_i = phi_i[T_n, T_participant, T_trial, ...] # I x J x S x 1

    phi_t = T_delta @ phi_i # I x J x S x 1
    return phi_t

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
        raise Exception(
            'prior_type="Model" draws phi_0 as a latent variable and belongs to '
            "inference: use cme.inference.priors.sample_initial_state instead."
        )
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
    # Previous full transition-matrix calculation retained for reference:
    # if likelihood_type=="SINGLE":
    #     RT=RT_s
    #     T_t = _get_transition_matrix(intensity_matrix, RT=RT, delta=delta, Mn=Mn, transition_type=transition_type)
    #
    # elif likelihood_type=="JOINT":
    #     RT_1 = RT_s[0]
    #     T_t_1 = _get_transition_matrix(intensity_matrix, RT=RT_1, delta=delta, Mn=Mn, transition_type=transition_type)
    #
    #     RT_2 = RT_s[1]
    #     T_t_2 = _get_transition_matrix(intensity_matrix, RT=RT_2, delta=delta, Mn=Mn, transition_type=transition_type)
    #
    #     phi_t_1_c = T_t_2 @ Mc @ T_t_1
    #     phi_t_1_w = T_t_2 @ Mw @ T_t_1
    #
    #     RA_1 = RA_s[0]
    #     T_t = npx.where(RA_1[..., None, None]==1, phi_t_1_c, phi_t_1_w) # I x J x S x S
    #
    # phi_t = T_t @ phi_0 # I x J x S x 1

    if transition_type == "TIMESTEP":
        T_delta = sci.linalg.expm(intensity_matrix * delta) # I x 1 x S x S

        if likelihood_type=="SINGLE":
            RT=RT_s
            ns = np.round(RT/delta) # I x J
            phi_t = _timestep_transition_state(ns, T_delta, Mn, phi_0) # I x J x S x 1

        elif likelihood_type=="JOINT":
            RT_1 = RT_s[0]
            ns_1 = np.round(RT_1/delta) # I x J
            phi_t_1 = _timestep_transition_state(ns_1, T_delta, Mn, phi_0) # I x J x S x 1

            phi_t_1_c = Mc @ phi_t_1 # I x J x S x 1
            phi_t_1_w = Mw @ phi_t_1 # I x J x S x 1

            RA_1 = RA_s[0]
            phi_t_1 = npx.where(RA_1[..., None, None]==1, phi_t_1_c, phi_t_1_w) # I x J x S x 1

            RT_2 = RT_s[1]
            ns_2 = np.round(RT_2/delta) # I x J
            phi_t = _timestep_transition_state(ns_2, T_delta, Mn, phi_t_1) # I x J x S x 1
        else:
            raise Exception(f"Please select one of {likelihood_type}")

    elif transition_type == "RT":
        if likelihood_type=="SINGLE":
            RT=RT_s
            T_t = _get_transition_matrix(intensity_matrix, RT=RT, delta=delta, Mn=Mn, transition_type=transition_type) # I x J x S x S

        elif likelihood_type=="JOINT":
            RT_1 = RT_s[0]
            T_t_1 = _get_transition_matrix(intensity_matrix, RT=RT_1, delta=delta, Mn=Mn, transition_type=transition_type) # I x J x S x S

            RT_2 = RT_s[1]
            T_t_2 = _get_transition_matrix(intensity_matrix, RT=RT_2, delta=delta, Mn=Mn, transition_type=transition_type) # I x J x S x S

            phi_t_1_c = T_t_2 @ Mc @ T_t_1 # I x J x S x S
            phi_t_1_w = T_t_2 @ Mw @ T_t_1 # I x J x S x S

            RA_1 = RA_s[0]
            T_t = npx.where(RA_1[..., None, None]==1, phi_t_1_c, phi_t_1_w) # I x J x S x S
        else:
            raise Exception(f"Please select one of {likelihood_type}")

        phi_t = T_t @ phi_0 # I x J x S x 1
    else:
        raise Exception(f"Please select one of {transition_type}")

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

def get_intensity_matrix(n_states, mu, sigma, model_type="Markov|Quantum"):
    if model_type == "Markov":
        return diffusion_buildK(n_states, mu, sigma)
    elif model_type == "Quantum":
        return quantum_buildH(n_states, mu, sigma)
    else:
        raise Exception(f"Please select one of {model_type}")
