import jax.numpy as npx
import jax.scipy.linalg as ln
import jax.numpy.linalg as num_ln
import numpy as np
import pandas as pd
import numpyro as npy
import numpyro.distributions as dist
from jax import random
import seaborn as sns
import matplotlib.pyplot as plt
import cme.simulators.quantum_random_walk as drw
from joblib import Parallel, delayed
from numpyro.infer.util import log_density
from cme.utils import common_logging as cl
from numpyro.infer import MCMC, NUTS, SA, HMCECS, Predictive, MixedHMC, HMC
import scipy.stats as stats
from jax import lax

#from numpyro.contrib.tfp.mcmc import TFPKernel
#import tensorflow_probability as tfp

log = cl.get_logger("diffusion_discrete")

import jax
jax.config.update('jax_platforms', 'cpu')



npy.set_host_device_count(64)
_rng_key = random.PRNGKey(0)
_rng_key, _rng_key_ = random.split(_rng_key)

def _buildH(n_states, mu, sigma, delta=0.001, n_trials = None): 
    # H = buildH(a,b,c)
    # m = number of states  
    # a = off diag left  
    # b = diag  
    # c = off diag right
    
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
    #for i in npx.arange(n_part):
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

    return -1j * params["H"]


def _get_measurement_matrix(n_states, start_width, prob = 0.5):
    
    Mcorr = npx.zeros(n_states)
    Mcorr = Mcorr.at[-start_width:].set(npx.sqrt(prob))
    Mcorr = npx.diag(Mcorr)

    Mincorr = npx.zeros(n_states)
    Mincorr = Mincorr.at[:start_width].set(npx.sqrt(prob))
    Mincorr = npx.diag(Mincorr)
    
    #Mc = npx.zeros((n_states, n_states)) # correct response
    #for i in range(n_states):
    #    if i > int(n_states/2) + start_width:
    #        Mc = Mc.at[i,i].set(prob)

    #Mw = npx.zeros((n_states, n_states)) # incorrect response
    #for i in range(n_states):
    #    if i < int(n_states/2) - start_width:
    #        Mw = Mw.at[i,i].set(prob)

    Mnoresp = npx.sqrt(npx.eye(n_states) - (Mcorr**2 + Mincorr**2))
    #npx.sqrt(npx.eye(7) - Mc**2 - Mw**2)

    return Mcorr, Mincorr, Mnoresp

def _get_initial_state(n_states, start_width, I = 1, prob=1):

    #Mid = int((n_states+1)/2)
    #p_0 = npx.zeros((I,1,n_states,1)) 
    #p_0 = p_0.at[:,0,(Mid-start_width-1):(Mid+start_width),0].set(prob) # additional -1 because indexing starts from 0
    #p_0 = p_0.reshape(-1,1) # to get column vector

    
    #with npy.plate('I', I, dim=-3):
    #with npy.plate('S', n_states, dim=-2):
    #    p_0 = npy.sample("phi_init", dist.Dirichlet((npx.ones(n_states))/n_states)) # Initial State
    
    with npy.plate('S', n_states):
        conc = npy.sample("phi_conc", dist.Beta(0.5,0.5))+0.01 #to avoid 0
    with npy.plate('I', I, dim=-3):
        p_0 = npy.sample("phi_init", dist.Dirichlet(conc)) # Initial State
        

    #p_0 = npy.sample("phi_init", dist.Dirichlet((npx.ones(n_states))/n_states)) # Initial State

             
    p_0 = npy.deterministic("phi_0", p_0.transpose(0,1,3,2)**(1/2))
    #p_0 = p_0 / npx.sqrt(p_0.transpose(0,1,3,2) @ p_0)
    return p_0



""" def _get_measurement_matrix_2(n_states, start_width, prob = 0.5):
    
    Mcorr = npx.zeros(n_states)
    Mcorr = Mcorr.at[-start_width:].set(np.sqrt(prob))
    Mcorr = npx.diag(Mcorr)

    Mincorr = npx.zeros(n_states)
    Mincorr = Mincorr.at[:start_width].set(np.sqrt(prob))
    Mincorr = npx.diag(Mincorr)
    
    #Mc = npx.zeros((n_states, n_states)) # correct response
    #for i in range(n_states):
    #    if i > int(n_states/2) + start_width:
    #        Mc = Mc.at[i,i].set(prob)

    #Mw = npx.zeros((n_states, n_states)) # incorrect response
    #for i in range(n_states):
    #    if i < int(n_states/2) - start_width:
    #        Mw = Mw.at[i,i].set(prob)

    Mnoresp = npx.eye(n_states) - Mcorr - Mincorr

    return Mcorr, Mincorr, Mnoresp """

    
def sample_states_and_confidence(rt, phi_0, K, Mc, Mw, Mn, N, ra = None, delta = None, n_noresp = None,  noisy=False, has_intermediate = False):
    n_states = K.shape[-1] # picking the last dimension because the dimensions are I,J,K,K
    Mid = int((n_states+1)/2)
    mv = npx.arange(-(Mid-1),(Mid)).reshape(1,-1) # to get column vector

    #T_t=ln.expm(-1j*rt*K) # rt:(); K:m,m This is transaction matrix

    #S_t = T_t @ phi_0 # Probability of transition matrix at t time for each response time

    if has_intermediate: 
       # here the last response is not being sent because we don't want to multiple response measurement matrix to the last record.
       phi_t = _state_transition_with_intermediate(K, rt, ra, phi_0, n_noresp, delta, Mc, Mw, Mn, noisy=noisy)
       #ra = ra[-1]
       #rt = rt[-1]
       #phi_0 = phi_t
    else:
        phi_t = _state_transition(K, phi_0, rt, delta, n_noresp, Mn, noisy=noisy)

    P_t = npx.abs(phi_t)**2

    Mconf = mv @ (P_t)
    #print("Mconf",Mconf)
    
    if False:
        noresp_traj_arr = []
        noresp_state = Mn @ T_t
        noresp_traj_arr.append(noresp_state @ phi_0)

        for n in npx.arange(2,N): #arange goes upto N-1, hence we do not need to explicitly substract 1 here.
            noresp_state = noresp_state * noresp_state
            noresp_traj_arr.append(noresp_state @ phi_0)

        phi_t = noresp_state @ phi_0
        Mconf = mv @ phi_t

    return P_t, Mconf, phi_t 



def _state_transition(K, phi_0, rt, delta=None, n_noresp = None, Mn = None, noisy=False):
    if noisy:
        #delta = npx.expand_dims(delta, axis=2)
        #delta = npx.expand_dims(delta, axis=2)
        if n_noresp is None:
            n_noresp = rt/delta

        T_t=ln.expm(-1j*delta*K) # delta:I,1,1,1; K:I,J,m,m This is transaction matrix
        
        #n_noresp should be IxJ
        #if n_noresp.shape != T_t.shape[:2]:
        #    n_noresp = npx.broadcast_to(n_noresp, T_t.shape[:2])

        phi_noresp_arr_i = []

        T_t = npx.broadcast_to(T_t, rt.shape + T_t.shape[-2:]) # Because T_t is same for each trial. So if the dimensions are not broadcasted, below for loop with fail.
        phi_0 = npx.broadcast_to(phi_0, rt.shape + phi_0.shape[-2:])

        for T_t_i, n_i, phi_0_i in zip(T_t, n_noresp, phi_0): # Iterating on participant dimension
            phi_noresp_arr_i_j = []
            
            for T_t_i_j, n_i_j, phi_0_i_j in zip(T_t_i, n_i, phi_0_i): # Iterating on trial dimension

                Pt_i_j = T_t_i_j @ num_ln.matrix_power(Mn @ T_t_i_j, n_i_j.astype(int).item() - 1) @ phi_0_i_j
                #Pt_i_j = npx.abs(Pt_i_j)**2
                phi_noresp_arr_i_j.append(Pt_i_j) #n_state x 1
            
            phi_noresp_arr_i.append(npx.asarray(phi_noresp_arr_i_j))    
        phi_t = npx.asarray(phi_noresp_arr_i) # n_part x n_trials x n_state x 1

    else:
        if rt.ndim < K.ndim:
            phi_t = ln.expm(-1j * K * rt[...,None,None]) @ phi_0 # n_part x n_trials x n_state x 1
        else:    
            phi_t = ln.expm(-1j * K * rt) @ phi_0 # n_part x n_trials x n_state x 1
    
    return phi_t

def _state_transition_with_intermediate(K, rt, ra, phi_0, n_noresp, delta, Mc, Mw, Mn, noisy):
    phi_t = None
    phi_0_in = phi_0

    for i, (rt_in, ra_in) in enumerate(zip(rt,ra)):
            #n_noresp = rt_in/delta
            phi_t_in = _state_transition(K, phi_0_in, rt_in, delta, n_noresp, Mn, noisy=noisy)

            if i != (len(rt) - 1): 
            # avoid these steps for the last response
                phi_t_in_correct = (Mc @ phi_t_in)
                phi_t_in_incorrect = (Mw @ phi_t_in)

                phi_t = npx.where(ra_in[..., None, None]==0, phi_t_in_incorrect, phi_t_in_correct)
                
                #print(f"************** {phi_t.shape}")
            #if phi_t is None:
                #phi_t = phi_t_in
            #else:
            #    phi_t = phi_t @ phi_t_in  
            else: 
                #phi_t = phi_t_in

                phi_t_in_correct = (Mc @ phi_t_in)
                phi_t_in_incorrect = (Mw @ phi_t_in)

                phi_t = npx.where(ra_in[..., None, None]==0, phi_t_in_incorrect, phi_t_in_correct)
                
            
            phi_0_in = phi_t
    return phi_t

def likelihood(K, rt, ra, phi_0, n_noresp, delta, Mc, Mw, Mn, noisy=False, has_intermediate = False):
    #ic(rt.shape)
    #K= _buildK(n_states, mu=mu, sigma=sigma)
    
    #ic(n_noresp.shape)
    #if rt is not None:
    #    rt = npx.expand_dims(rt, axis=2)
    #    rt = npx.expand_dims(rt, axis=2)

    if has_intermediate: # should have two or more arrays of resposne time and response accuracy
       phi_t = _state_transition_with_intermediate(K, rt, ra, phi_0, n_noresp, delta, Mc, Mw, Mn, noisy=noisy)
       #ra = ra[-1]
       #rt = rt[-1]
    else:
        
        phi_t = _state_transition(K, phi_0, rt, delta, n_noresp, Mn, noisy=noisy)

    P_correct = (npx.abs(Mc @ phi_t)**2).sum(axis=(-2,-1)) # Mc @ phi_t is KxK @ IxJxKxK; *correct should be IxJ
    P_incorrect = (npx.abs(Mw @ phi_t)**2).sum(axis=(-2,-1))

    P_i_j = npx.where(ra==0, P_incorrect, P_correct)
    
    #ic(Pcorrect.shape)

    #Pcorrect = Pcorrect.sum(axis=(-2,-1)) #.squeeze() # adding up the probabilities over states for a given response
    
    likl = P_i_j.sum()

    return likl #likelihood hence adding up over all responses.


def get_likl_states_confidence(n_states, mu, sigma, delta, n_noresp, p_0, Mc1, Mw1, Mn1, ra, rt, noisy=False, has_intermediate=False):
    
    #if not has_intermediate:
    #    rt = npx.asarray([[rt]])
    #if n_noresp is None:
    #    n_noresp_1 = rt/delta
    #else: 
    #    delta = rt/n_noresp
    #    n_noresp_1 = n_noresp

        #print(p_0, "**********")

    if has_intermediate:
        I,J = rt[0].shape
        for i, (t,a) in enumerate(zip(rt, ra)):
            if(mu.shape[0] != t.shape[0]):
                rt[i] = npx.broadcast_to(t, [mu.shape[0]] + [t.shape[1]])
                ra[i] = npx.broadcast_to(a, [mu.shape[0]] + [a.shape[1]])
    else:
        I,J = rt.shape
        if(mu.shape[0] != rt.shape[0]):
                rt = npx.broadcast_to(rt, [mu.shape[0]] + [rt.shape[1]])
                #ra = npx.broadcast_to(ra, [mu.shape[0]] + [ra.shape[1]])

    K= _buildH(n_states, mu=mu, sigma=sigma, n_trials = None if noisy else J)
    n_noresp_1 = None
    likl = likelihood(K, rt, ra, p_0, n_noresp_1, delta, Mc1, Mw1, Mn1, noisy=noisy, has_intermediate=has_intermediate) #K,rt, ra, phi_0, delta, Mc, Mw, Mn
    Pt, Mconf, phi_t = sample_states_and_confidence(rt, p_0, K, Mc1, Mw1, Mn1, n_noresp_1, delta=delta, ra=ra, noisy=noisy, has_intermediate=has_intermediate)
    return delta, n_noresp_1, likl, Pt, Mconf, phi_t, rt

def perform_walk(n_states, start_width, mu, sigma, p_0 = None, max_timesteps=10, ra = npx.asarray([[1]]), delta=None, prob=0.5, n_noresp = None, dataset_id=1, noisy=False, has_intermediate=False, njobs=1):

    """This function performs walks for multiple participants. 
       But, response time and response accuracy is considered same between participants.
       Only, model parameters (e.g., drift rate) is allowed to change between participants.
    """

    avg_conf = [];  
    state_prob = []
    likl_prob = []
    n_noresp_arr = []
    phi_t_arr = []
    rt_arr = []
    #I,J = ra.shape
    if p_0 is None:
        p_0 = _get_initial_state(n_states, start_width, 1)

    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width, prob)
    if delta is not None:
        delta = npx.asarray(delta)
        #step = delta[0,0]
    elif False:
        n_noresp = npx.asarray(n_noresp)
        delta = max_timesteps/n_noresp
    
    step = delta[0,0]
    n_noresp = None
    #if n_noresp is not None:
    #    n_noresp = npx.asarray(n_noresp)
    rt_last_iter = []
    ra_last_iter = []

    # Because while walking, the joint probability will only be evaluated after the first response time is over
    has_inter_temp = False
    if has_intermediate:
        for rt_in, ra_in in zip(max_timesteps, ra):   
            #get_likl_states_confidence(n_states, mu, sigma, npx.asarray([[1]]), n_noresp, p_0, Mc, Mw, Mn, ra_last_iter + [ra_in], rt_last_iter + [step], has_intermediate=has_inter_temp)
            log.debug(f"(Multiple RTs) Starting Walk for ({step}:{rt_in}), {ra_in} steps with step size {step}; mu {mu.flatten()}")
            dely_get_likl_states_confidence = delayed(get_likl_states_confidence)
            # parallelized over timesteps
            res = Parallel(n_jobs=njobs)(dely_get_likl_states_confidence(n_states, mu, sigma, delta, n_noresp, p_0, Mc, Mw, Mn, ra_last_iter + [ra_in], rt_last_iter + [npx.asarray([[rt]])], noisy=noisy, has_intermediate=has_intermediate) 
                                        for rt in list(npx.arange(step, rt_in, step=step))
                                        #for rt in np.linspace(0,max_timesteps, 3000)
                                        )
            
            for delta, n_noresp_1, likl, Pt, Mconf, phi_t, rt in res:
                state_prob.append(np.asarray(Pt))
                likl_prob.append(np.asarray(likl))
                avg_conf.append(np.asarray(Mconf))
                phi_t_arr.append(np.asarray(phi_t))
                n_noresp_arr.append(n_noresp_1)
                rt_arr.append(np.asarray(rt).sum()) # add multiple time points

            
            
            rt_last_iter.append(npx.asarray([[rt_in]]))
            ra_last_iter.append(ra_in)
            #phi_0_c = Mc @ phi_t # updating initial state of next response with current state of last response time
            #phi_0_w = Mw @ phi_t

            # Here the last state is used as initial point for next walk.
            # It's important to note that this updated state now contains imaginary numbers 
            # (which is not the case of original starting point)
            #p_0 = np.where(ra_in == 1, phi_0_c, phi_0_w)
            has_inter_temp = True
    else:
        #get_likl_states_confidence(n_states, mu, sigma, npx.asarray([[1]]), n_noresp, p_0, Mc, Mw, Mn, ra, npx.asarray([[step]]))
        log.debug(f"(Single RT) Starting Walk for {max_timesteps} steps with step size {step}; mu {mu.shape}")
        dely_get_likl_states_confidence = delayed(get_likl_states_confidence)
        # parallelized over timesteps
        res = Parallel(n_jobs=njobs)(dely_get_likl_states_confidence(n_states, mu, sigma, delta, n_noresp, p_0, Mc, Mw, Mn, ra, npx.asarray([[rt]]), noisy=noisy, has_intermediate=has_intermediate) 
                                    for rt in list(npx.arange(step, max_timesteps, step=step))
                                    #for rt in np.linspace(0,max_timesteps, 3000)
                                    )


#for rt in np.linspace(1,max_timesteps, 50):
        for delta, n_noresp_1, likl, Pt, Mconf, phi_t, rt in res:
            state_prob.append(np.asarray(Pt))
            likl_prob.append(np.asarray(likl))
            avg_conf.append(np.asarray(Mconf))
            phi_t_arr.append(np.asarray(phi_t))
            n_noresp_arr.append(n_noresp_1)
            rt_arr.append(np.asarray(rt).squeeze())
        
    levels=list(zip(np.array(npx.mean(mu, axis=-1)), np.array(npx.mean(sigma, axis=-1))))
    col_names = pd.MultiIndex.from_tuples(levels)


    conf_arr = np.asarray(avg_conf).squeeze()
    #print("1) *********", conf_arr.ndim, conf_arr.shape)
    if conf_arr.ndim > 2:
        #print("in")
        conf_arr = conf_arr.mean(axis=1)
    #conf_arr = conf_arr.reshape(conf_arr.shape[0],-1)
    df_avg_conf = pd.DataFrame(conf_arr, columns=col_names) # timestep x n_part 
    df_avg_conf = df_avg_conf.reset_index().rename({"index":"time"}, axis=1)
    df_avg_conf.loc[:,"rt"] = pd.Series(rt_arr) #df_avg_conf["time"] * delta
    df_avg_conf = df_avg_conf.melt(id_vars=["rt","time"], value_name="avg_conf").rename(columns = {"variable_0":"drift_rate", "variable_1":"sigma"})
    df_avg_conf = df_avg_conf.assign(sample_id=dataset_id)

    df_likl = pd.DataFrame(np.asarray(likl_prob)).rename({0:"liklihood"}, axis=1) # likelihood already integrated over participants and within-trial drift rates
    df_likl = df_likl.reset_index().rename({"index":"time"}, axis=1).assign(drift_rate=np.mean(mu), sigma=np.mean(sigma))
    df_likl.loc[:,"rt"] = pd.Series(rt_arr) 
    df_likl = df_likl.assign(sample_id=dataset_id)
    
    # State Prob has a shape of timesteps x participants x states. Hence, creating array of dataframes for each participant
    df_st_arr = []
    for part_id in np.arange(np.asarray(state_prob).shape[1]):
        df_st = pd.DataFrame(np.asarray(state_prob)[:,part_id,:].squeeze())
        Mid = int((n_states+1)/2)
        mv = np.arange(-(Mid-1),(Mid))
        
        df_st.columns = mv
        #df_st.loc[:,"rt"] = pd.Series(rt_arr)
        
        df_st = (df_st.reset_index() 
                    .melt(id_vars = "index",var_name="state", value_name="probability") 
                    .rename({"index":"time"}, axis=1) 
                    .assign(mu=np.mean(mu, axis=-1)[part_id]) 
                    .assign(sigma=np.mean(sigma, axis=-1)[part_id])
                    .assign(part_id = part_id)
                )
        #df_st.loc[:,"state"] = df_st.state.astype("category")
        df_st.loc[:,"time"] = df_st.time.astype("category")
        df_st_arr.append(df_st)

    df_st = pd.concat(df_st_arr)
    df_st = df_st.assign(sample_id=dataset_id)

    return df_st, df_avg_conf, df_likl, pd.DataFrame(mu), pd.DataFrame(sigma)

def model_central(n_states, start_width, sigma, tau, rt, ra,I,J, s_0, batch_size=2, noisy=False, has_intermediate=False):
    
    #st = npy.sample("s_0", dist.Dirichlet(npx.ones((start_width))+4))
    
    #Mid_w = int((start_width+1)/2)
    #Mid = int((n_states+1)/2)
    #s_0 = npx.zeros((n_states,1)) 
    #s_0 = s_0.at[(Mid-Mid_w):(Mid+Mid_w),0].set(st)
    s_0 = _get_initial_state(n_states, start_width,I)
    s_0 = npy.deterministic("s_0", s_0)

    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width, 0.75)

    #mu_m =  npy.sample(f"mu_m", dist.Normal(2,3))
    #mu_s =  npy.sample(f"mu_s", dist.HalfNormal(2))
    delta = np.asarray(tau)
    
    #n_noresp = rt/delta if rt is not None else 10
    n_noresp = None
        
    with npy.plate('I', I, dim=-2) as ind: #, subsample_size=batch_size
        #mu =  npy.sample(f"mu", dist.Normal(mu_m,mu_s),sample_shape=(I,1))
        mu = npy.sample("mu", dist.Normal(0,2)) # Drift Rate
        sigma_t = npy.sample("sigma_t", dist.Normal(0,1)) # Diffusion Rate
        sigma = npy.deterministic("sigma",sigma_t**2 + 1) #Sigma cannot be negative
        #n_noresp = npy.sample("N", dist.Binomial(total_count=n_noresp_max, probs=0.5))
        #if(npx.count_nonzero(npx.array(npx.less_equal(sigma , 0))) > 0):
        #    log.debug(sigma)
        K = _buildH(n_states, mu, sigma, None if noisy else J)
    
    #with npy.plate('Obs', I, subsample_size=10) as ind: #,
        #mu =  npy.sample(f"mu", dist.Normal(0,5)) #,sample_shape=(I,)
    
        rt1 = rt #if rt is None else npx.asarray(rt)#[ind]
        ra1 = ra #if ra is None else npx.asarray(ra)#[ind]
        #_,lkl ,_ = likelihood(n_states, mu[ind], 1, rt1, ra1, s_0, Mr)
        if rt is not None:
        #    if not noisy:
        #        rt1 = rt1[...,None,None] # To multiply with Intensity Matrix
            lkl = likelihood(K, rt1, ra1, s_0, n_noresp, delta, Mc, Mw, Mn, noisy=noisy, has_intermediate=has_intermediate)
            #lkl = npx.where(npx.less_equal(K.at[1,1], npx.asarray(0)), npx.asarray(0), lkl)
            
            npy.factor(f"likelihood", lkl)

def model(n_states, start_width, sigma, tau, rt, ra,I,J, s_0, batch_size=2):
    
    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width, 0.5)

    #mu_m =  npy.sample(f"mu_m", dist.Normal(2,3))
    #mu_s =  npy.sample(f"mu_s", dist.HalfNormal(2))
    delta = np.asarray([[tau]])
    
    n_noresp = rt/delta if rt is not None else 10
    
    m = npy.sample("m", dist.Normal(0,1))
    s = npy.sample("s", dist.HalfNormal(1)) #s = pm.Normal("s",0,0.2,shape=4)


    with npy.plate('I', I, dim=-2) as ind: #, subsample_size=batch_size
        #mu =  npy.sample(f"mu", dist.Normal(mu_m,mu_s),sample_shape=(I,1))
        mu_r = npy.sample("mu_r", dist.Normal(2,1)) # Drift Rate
        sigma_r = npy.sample("sigma_r", dist.Normal(1,1)) # Diffusion Rate

        mu = npy.deterministic("mu", m + s * mu_r)
        sigma = npy.deterministic("sigma", (mu + s * sigma_r)**2) #Sigma cannot be negative and should be larger than mu

        K = _buildH(n_states, mu, sigma, delta=delta)
    
    #with npy.plate('Obs', I, subsample_size=10) as ind: #,
        #mu =  npy.sample(f"mu", dist.Normal(0,5)) #,sample_shape=(I,)
    
        rt1 = rt if rt is None else npx.asarray(rt)#[ind]
        ra1 = ra if ra is None else npx.asarray(ra)#[ind]
        #_,lkl ,_ = likelihood(n_states, mu[ind], 1, rt1, ra1, s_0, Mr)
        if rt is not None:
            lkl = likelihood(K, rt1, ra1, s_0, n_noresp, delta, Mc, Mw, Mn)
            npy.factor(f"likelihood", lkl)

def sample_posterior_params(DT, X, n_states, start_width, sigma, tau, I = None, num_warmup=100, samples_n=500, num_chains=4, batch_size=2, noisy=False, has_intermediate=False):

    s_0 = _get_initial_state(n_states, start_width,I)
    if has_intermediate:
        I,J = DT[0].shape if I is None else I, DT[0].shape[1]
    else:
        I,J = DT.shape if I is None else I, DT.shape[1]

    #kernel = HMCECS(NUTS(model), num_blocks=10)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    #mcmc_chain.run(_rng_key, n_states, start_width,  sigma, tau, DT, X, I, J, s_0, batch_size=batch_size, extra_fields=('hmc_state',))

    #kernel = NUTS(model)
    kernel = NUTS(model_central)
    #kernel = MixedHMC(HMC(model_central, trajectory_length=1.2), num_discrete_updates=20)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    mcmc_chain.run(_rng_key, n_states, start_width,  sigma, tau, DT, X, I, J, s_0, batch_size=batch_size, noisy=noisy, has_intermediate=has_intermediate, extra_fields=('potential_energy',))

    #kernel = TFPKernel[tfp.mcmc.NoUTurnSampler](model, step_size=1.)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    #mcmc_chain.run(_rng_key, n_states, start_width,  sigma, tau, DT, X, I, J, s_0, batch_size=batch_size, extra_fields=('hmc_state',))

    #post_likl = log_density(model, model_args=(n_states, start_width,  sigma, tau, DT, X, I, J, s_0), model_kwargs = None, params=mcmc_chain.get_samples())
    #post_likl = mcmc_chain.get_extra_fields()['hmc_state'].potential_energy
    post_likl = mcmc_chain.get_extra_fields()['potential_energy'] # This is negative log likelihood
    return mcmc_chain, post_likl

def sample_prior_pred_data(n_states, start_width, tau, sigma_s, I, J, samples_n=100, njobs=1, get_response=False, obs_response_range=None, ra_s = None, batch_size=2, noisy=False, has_intermediate=False):
    s_0 = _get_initial_state(n_states, start_width, I)

    #prior_predictive = Predictive(model, num_samples=samples_n)
    prior_predictive = Predictive(model_central, num_samples=samples_n)
    prior_predictions = prior_predictive(_rng_key, n_states, start_width, sigma_s[...,0] if sigma_s is not None else None, 
                                         tau, None, None, I, J, s_0, batch_size=batch_size)

    
    mu_s = prior_predictions["mu"]
    sigma_s = prior_predictions["sigma"]
    #N_s = prior_predictions["N"]
    s_0 = prior_predictions["s_0"][0] # picking the first sample because all samples will have same value 
    log.debug(f"mu: {mu_s.shape}")
    log.debug(f"sigma: {sigma_s.shape}")
    log.debug(f"s_0: {s_0.shape}")
    
    predictions = get_predictive_samples(n_states, start_width, mu_s, tau, sigma_s, J, p_0 = s_0, n_noresp_s=None, 
                                         samples_n=samples_n, njobs=njobs, get_response=get_response, ra_s = ra_s,
                                         obs_response_range=obs_response_range, 
                                         noisy=noisy, has_intermediate=has_intermediate)
    predictions.update(prior_predictions)

    return predictions

def get_predictive_samples(n_states, start_width, mu_s, tau, sigma_s, J, p_0=None, n_noresp_s=None, samples_n=100, njobs=1, get_response=False, obs_response_range=None, ra_s = None, noisy=False, has_intermediate=False):
    predictions = {}
    theta = int((n_states+1)/2)

    if False: #get_response:
        log.debug("Getting Response Times")
        RT, X, Steps = get_rt_sample(theta, 1.5, tau, sigma_s, mu_s, n_trials = J, njobs=njobs)

        predictions["RT"] = RT
        predictions["X"] = X
        predictions["Steps"] = Steps

    if ~get_response and obs_response_range is not None:
        log.debug(f"Confidence and Likelihood for parameter size {npx.asarray(mu_s).shape}")
        df_st, df_avg_conf, df_likl, df_mu, df_sigma = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        _, max_obs_resp = obs_response_range

        for d_id, (mu_d, sigma_d) in enumerate(zip(npx.asarray(mu_s), npx.asarray(sigma_s))): # Each sample. mu_p.shape=(part, within-trial variation)
            
            #print(mu.shape)
            #print("in loop muu", mu.shape)
            #print("sgimau", sigma.shape)
            #log.debug(f"Perform walk: {mu_p}")
            df_st_t, df_avg_conf_t, df_likl_t, df_mu_t, df_sigma_t = perform_walk(n_states, start_width, mu_d, 
                                                                                  sigma_d, p_0=p_0, n_noresp=None, 
                                                                                  max_timesteps=max_obs_resp, ra = ra_s,
                                                                                  delta=tau, njobs=njobs, 
                                                                                  noisy=noisy, has_intermediate=has_intermediate, 
                                                                                  dataset_id = d_id)

            #for (_,df_st_t), (_,df_avg_conf_t), mu, sigma in zip(df_st_m.iteritems(), df_avg_conf_m.iteritems(), mu_p, sigma_p):
            
            df_st = pd.concat([df_st, df_st_t])
            df_avg_conf = pd.concat([df_avg_conf, df_avg_conf_t])
            df_likl = pd.concat([df_likl, df_likl_t])
            df_mu = pd.concat([df_mu, df_mu_t])
            df_sigma = pd.concat([df_sigma, df_sigma_t])
    
        predictions["States"] = df_st
        predictions["Confidence"] = df_avg_conf
        predictions["Likelihood"] = df_likl
        predictions["For_mu"] = df_mu
        predictions["For_sigma"] = df_sigma


    #C = mu_s.shape[0]
    #Mid = int((n_states+1)/2)
    #for c in npx.arange(C):
    #    RT_t, X_t, steps_arr_t = get_rt_sample(theta, 1.5, tau, sigma, mu_s[c,...], I, J, samples_n)
    #    RT[c,...] = RT_t
    #    X[c,...] = X_t
    #    Steps[c,...] = steps_arr_t
    return predictions


def get_rt_sample(theta, alpha, tau, sigma, mu_s, n_trials, njobs=1):

    C, I, _ = mu_s.shape

    RT = np.empty((C, I, n_trials))
    X = np.empty((C, I, n_trials))
    steps_arr_c = []

    for c in npx.arange(C):
        steps_arr_i = []
        for i in npx.arange(I):

            RT_arr, X_arr, steps = drw.gen_rt_x(theta, alpha, tau, sigma[c,i,...], mu_s[c,i,...], samples=n_trials, process="Wiener", initial="Any", njobs=njobs)
            RT[c, i,:] = RT_arr
            X[c, i,:] = X_arr
            steps_arr_i.append(steps)
        steps_arr_c.append(steps_arr_i)
        
    Steps = pd.DataFrame(steps_arr_c)
    return RT, X, Steps


def sample_post_pred_data(n_states, start_width, tau, sigma_s, I, J, mcmc_samples, njobs=1, get_response=False, ra_s = None, obs_response_range=None, noisy=False, has_intermediate=False):
    mu_s = mcmc_samples["mu"]#.reshape((-1, mcmc_samples["mu"].shape[-1])) #[0:100,...] # reshaping is done to merge the samples and participants into a single dimension
    sigma_s = mcmc_samples["sigma"]#.reshape((-1, mcmc_samples["sigma"].shape[-1]))#[0:100,...]
    #N_s = mcmc_samples["N"]
    s_0 = mcmc_samples["s_0"][0]#.reshape((-1, mcmc_samples["s_0"].shape[-1]))#[0:100,...]

    predictions = get_predictive_samples(n_states, start_width, mu_s, tau, sigma_s, J, 
                                         p_0 = s_0, n_noresp_s=None, samples_n= 100, njobs= njobs,
                                           get_response=get_response, obs_response_range=obs_response_range, ra_s = ra_s,
                                           noisy=noisy, has_intermediate=has_intermediate)
    predictions.update({"mu": mu_s})
    predictions.update({"sigma": sigma_s})

    return predictions

    
if __name__ == "__main__":

    log.debug("Prior Prediction (noisy) - 1")
    n_states=7
    start_width=1
    I, J = 5,3
    tau = 1
    sigma=None

    #prior_predictions = sample_prior_pred_data(n_states, start_width, [[tau]], sigma,  I, J, samples_n=2, obs_response_range=(1,10), noisy=True)
    #RT = prior_predictions["RT"]
    #log.debug(RT.min(), RT.mean(), RT.max())

    log.debug("Prior Prediction (noisy, intermediate) - 1")
    I, J = 5,3
    ra_s = [npx.asarray([[0]]), npx.asarray([[1]])]
    prior_predictions1 = sample_prior_pred_data(n_states, start_width, [[tau]], sigma,  I, J, samples_n=1, obs_response_range=(1,[4,5]), ra_s = ra_s, noisy=True, has_intermediate=True)
    print(prior_predictions1["Confidence"]["avg_conf"])
    print("starting second")
    ra_s = [npx.asarray([[1]]), npx.asarray([[0]])]
    prior_predictions2 = sample_prior_pred_data(n_states, start_width, [[tau]], sigma,  I, J, samples_n=1, obs_response_range=(1,[4,5]), ra_s = ra_s, noisy=True, has_intermediate=True)
    print(prior_predictions2["Confidence"]["avg_conf"])

    mu_arr, sigma = npx.asarray([[0.01,1]]).T, npx.asarray([[5,10]]).T
    I=2

    log.debug(_buildH(7, mu=mu_arr, sigma=sigma))
    
    log.debug(_buildH(7, mu=[[1.5]], sigma=sigma))
    
    log.debug(_buildH(7, mu=[[0.5,1.5]], sigma=sigma)) # this didn't throw out of bounds exception because of JAX's behavior
    
    log.debug(_buildH(7, mu=[[1]], sigma=sigma).shape)
    
    log.debug(_buildH(7, mu=npx.asarray([[1,2,0.5]]), sigma=npx.asarray([[1]])).shape) # this didn't throw out of bounds exception because of JAX's behavior

    log.debug("Constant Drift Rate - 1")
    # Replicating the plots in Busemeyer 2010
    n_states=7
    start_width=4
    mu=[[1]] #drift rate
    sigma=npx.asarray([[1]]) #diffusion
    tau = 1
    delta=[[tau]]

    phi_0 = _get_initial_state(n_states, start_width,I)
    K = _buildH(n_states,mu,sigma)
    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width, 0.25)


    conf_arr = []
    P_t_arr = []
    likl_arr = []

    phi_0 = _get_initial_state(n_states, start_width,I=1)
    for r in np.arange(0,10):
        P_t, Mconf, noresp_traj_arr = sample_states_and_confidence(r, phi_0, K, Mc, Mw, Mn, 1)
        conf_arr.append(Mconf.squeeze())
        P_t_arr.append(np.asarray(P_t.squeeze()))

        likl = likelihood(K,npx.asarray([[r+1]]),npx.asarray([[1]]),phi_0,npx.asarray([[r+1]]),npx.asarray([[1]]),Mc,Mw,Mn) # if delta is 1, then n_noresp simply becomes the response time
        likl_arr.append(likl)

    conf = np.asarray(conf_arr)
    log.debug(conf.shape)
    pd.Series(conf).plot.line()

    df_st = pd.DataFrame(P_t_arr).reset_index(names="time").melt(id_vars="time", var_name="state")
    log.debug(df_st.shape)
    sns.relplot(df_st, x="time", y="state", size="value",sizes=(50, 300), color="black")
    plt.show()

    likl = np.asarray(likl_arr)
    log.debug(likl.shape)
    pd.Series(likl).plot.line()
    plt.show()
    
    log.debug("Constant Drift Rate - 2")
    # Replicating the plots in Busemeyer 2010
    n_states=101
    start_width=11
    mu=npx.asarray([[0.2]]) #drift rate
    sigma=npx.asarray([[2]]) #diffusion
    #mu=npx.asarray(mu)
    #sigma = npx.asarray(sigma)

    phi_0 = _get_initial_state(n_states, start_width, 1, 5)
    K = _buildH(n_states,mu,sigma)

    df_st, df_avg_conf, df_likl, df_mu, df_sigma = perform_walk(n_states=n_states, start_width=start_width, 
                                    mu=mu, sigma=sigma,max_timesteps=20, 
                                    delta=[[1]], prob=0.25, noisy=False, has_intermediate=False)#, n_noresp=npx.asarray([[1]]))
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf", kind="line")
    plt.show()
    sns.relplot(df_likl, x="time",y="liklihood", kind="line")
    plt.show()

    #log.debug("Constant Drift Rate - 3")
    #df_st, df_avg_conf, df_likl = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    #for N in range(100,200, 10):

    #    df_st_t, df_avg_conf_t, df_likl_t = perform_walk(n_states=n_states, start_width=start_width, 
    #                                    mu=mu, sigma=sigma,max_timesteps=200, n_noresp=[[N]],
    #                                    delta=None, prob=0.25)#, n_noresp=npx.asarray([[1]]))
    #    df_st = pd.concat([df_st, df_st_t.assign(N=N)])
    #    df_avg_conf = pd.concat([df_avg_conf, df_avg_conf_t.assign(N=N)])
    #    df_likl = pd.concat([df_likl, df_likl_t.assign(N=N)])    
    
    #sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    #sns.relplot(df_avg_conf, x="time",y="avg_conf", hue="N", kind="line")
    #plt.show()
    #sns.relplot(df_likl, x="time",y="liklihood", hue="N", kind="line")
    #plt.show()


    log.debug("Constant Drift Rate (with Intermediate) - 1")
    # Replicating the plots in Busemeyer 2010
    n_states=7
    start_width=1
    mu=npx.asarray([[0.2]]) #drift rate
    sigma=npx.asarray([[2]]) #diffusion
    #mu=npx.asarray(mu)
    #sigma = npx.asarray(sigma)

    phi_0 = _get_initial_state(n_states, start_width, 1, 5)
    K = _buildH(n_states,mu,sigma)
    ra_s = [npx.asarray([[1]]), npx.asarray([[1]])]
    df_st, df_avg_conf, df_likl, df_mu, df_sigma = perform_walk(n_states=n_states, start_width=start_width, 
                                    mu=mu, sigma=sigma,max_timesteps=[20,10], ra=ra_s,
                                    delta=[[1]], prob=0.25, noisy=False, has_intermediate=True)#, n_noresp=npx.asarray([[1]]))
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf", kind="line")
    plt.show()
    sns.relplot(df_likl, x="time",y="liklihood", kind="line")
    plt.show()

    log.debug("Constant Drift Rate (with Intermediate, Noisy) - 1")
    # Replicating the plots in Busemeyer 2010
    n_states=7
    start_width=1
    mu=npx.asarray([[0.2]]) #drift rate
    sigma=npx.asarray([[2]]) #diffusion
    #mu=npx.asarray(mu)
    #sigma = npx.asarray(sigma)

    phi_0 = _get_initial_state(n_states, start_width, 1, 5)
    K = _buildH(n_states,mu,sigma)
    ra_s = [npx.asarray([[1]]), npx.asarray([[1]])]
    df_st, df_avg_conf, df_likl, df_mu, df_sigma = perform_walk(n_states=n_states, start_width=start_width, 
                                    mu=mu, sigma=sigma,max_timesteps=[20,10], ra=ra_s,
                                    delta=[[1]], prob=0.25, noisy=True, has_intermediate=True)#, n_noresp=npx.asarray([[1]]))
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf", kind="line")
    plt.show()
    sns.relplot(df_likl, x="time",y="liklihood", kind="line")
    plt.show()

    log.debug("Prior Prediction - 1")
    I, J = 5,3
    sigma=None
    prior_predictions = sample_prior_pred_data(n_states, start_width, [[tau]], sigma,  I, J, samples_n=2, obs_response_range=(1,10), noisy=False)
    #RT = prior_predictions["RT"]
    #log.debug(RT.min(), RT.mean(), RT.max())

    log.debug("Prior Prediction (noisy) - 1")
    n_states=7
    start_width=1
    I, J = 5,3
    tau = 1
    sigma=None
    prior_predictions = sample_prior_pred_data(n_states, start_width, [[tau]], sigma,  I, J, samples_n=2, obs_response_range=(1,10), noisy=True)
    #RT = prior_predictions["RT"]
    #log.debug(RT.min(), RT.mean(), RT.max())

    log.debug("Prior Prediction (noisy, intermediate) - 1")
    I, J = 5,3
    ra_s = [npx.asarray([[0]]), npx.asarray([[1]])]
    prior_predictions = sample_prior_pred_data(n_states, start_width, [[tau]], sigma,  I, J, samples_n=2, obs_response_range=(1,[10,8]), ra_s = ra_s, noisy=True, has_intermediate=True)
    #RT = prior_predictions["RT"]
    #log.debug(RT.min(), RT.mean(), RT.max())

    log.debug("Posterior Sampling - 1")
    
    rt = np.random.uniform(0.1, 10, size=(I,J))
    x = np.random.randint(0,2, size=(I,J))
    n_states=7
    start_width=1
    

    mcmc_chain, post_likl = sample_posterior_params(DT=rt, X=x, sigma=sigma, tau=[[tau]], 
                                                    I=I, n_states=n_states, start_width=start_width, 
                                                    num_warmup=10, samples_n=4 )
    mcmc_chain.print_summary()
    mcmc_samples = mcmc_chain.get_samples()

    log.debug("Posterior Sampling (Intermediate) - 1")
    
    rts = [np.random.uniform(0.1, 10, size=(I,J)), np.random.uniform(0.1, 10, size=(I,J))]
    xs = [np.random.randint(0,2, size=(I,J)), np.random.randint(0,2, size=(I,J))]
    n_states=7
    start_width=1
    

    mcmc_chain, post_likl = sample_posterior_params(DT=rts, X=xs, sigma=sigma, tau=[[tau]], 
                                                    I=I, n_states=n_states, start_width=start_width, 
                                                    num_warmup=10, samples_n=4, has_intermediate=True )
    mcmc_chain.print_summary()
    mcmc_samples = mcmc_chain.get_samples()

    log.debug("Posterior Sampling (noisy) - 1")
    
    rt = np.random.uniform(0.1, 10, size=(I,J))
    x = np.random.randint(0,2, size=(I,J))
    n_states=7
    start_width=1
    

    mcmc_chain, post_likl = sample_posterior_params(DT=rt, X=x, sigma=sigma, tau=[[tau]], 
                                                    I=I, n_states=n_states, start_width=start_width, 
                                                    num_warmup=10, samples_n=4, noisy=True )
    mcmc_chain.print_summary()
    mcmc_samples = mcmc_chain.get_samples()

    log.debug("Posterior Sampling (Noisy, Intermediate) - 1")
    
    rts = [np.random.uniform(0.1, 10, size=(I,J)), np.random.uniform(0.1, 10, size=(I,J))]
    xs = [np.random.randint(0,2, size=(I,J)), np.random.randint(0,2, size=(I,J))]
    n_states=7
    start_width=1
    

    mcmc_chain, post_likl = sample_posterior_params(DT=rts, X=xs, sigma=sigma, tau=[[tau]], 
                                                    I=I, n_states=n_states, start_width=start_width, 
                                                    num_warmup=10, samples_n=4, has_intermediate=True, noisy=True )
    mcmc_chain.print_summary()
    mcmc_samples = mcmc_chain.get_samples()


    log.debug("Posterior Prediction - 1") # Test here
    predictions = sample_post_pred_data(n_states, start_width, [[tau]], sigma, I,J, mcmc_samples,obs_response_range=(1,100))
    #log.debug(RT.min(), RT.mean(), RT.max())


       



