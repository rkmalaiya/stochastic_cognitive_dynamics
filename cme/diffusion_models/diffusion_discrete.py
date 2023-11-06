import jax.numpy as npx
import jax.scipy.linalg as ln
import jax.numpy.linalg as num_ln
import numpy as np
import pandas as pd
import numpyro as npy
import numpyro.distributions as dist
from jax import random
from icecream import ic
import seaborn as sns
import matplotlib.pyplot as plt

ic.enable()
from numpyro.infer import MCMC, NUTS, SA, HMCECS, Predictive

npy.set_host_device_count(64)
_rng_key = random.PRNGKey(0)
_rng_key, _rng_key_ = random.split(_rng_key)

def _buildK(n_states, mu, sigma=1, delta=1): 
# m = number of states  
# a = off diag left  
# b = diag  
# c = off diag right
    mu = npx.asarray(mu)
    n_part, n_mu = mu.shape #if len(npx.asarray(mu).shape) > 0 else 1
    K = npx.zeros((n_part, 1, n_states,n_states)) # participants, trials, transition states

    if n_mu == 1:
        mu=npx.repeat(mu,n_states,axis=1) # keeping mu constant over states

    for i in range(n_part):
        #b1 = 0.5 * (((sigma**2)/delta**2) + mu[i,:]/delta)
        #b2 = 0.5 * (((sigma**2)/delta**2) - mu[i,:]/delta)
        b1 = 0.5 * (sigma - mu[i,:])
        b2 = 0.5 * (sigma + mu[i,:])
        a = -(b1+b2)

        for j in range(1,n_states-1):
            K = K.at[i,0,[j-1,j,j+1],j].set([b1[j], a[j], b2[j]])

        K = K.at[i,0,[0,1],0].set([a[0], -a[0]])
        K = K.at[i,0,[-2,-1],-1].set([-a[-1], a[-1]])
        
    return K

def _get_measurement_matrix(n_states, start_width):
    Mc = npx.zeros((n_states, n_states)) # correct response
    for i in range(n_states):
        if i > int(n_states/2) + start_width:
            Mc = Mc.at[i,i].set(0.5)

    Mw = npx.zeros((n_states, n_states)) # incorrect response
    for i in range(n_states):
        if i < int(n_states/2) - start_width:
            Mw = Mw.at[i,i].set(0.5)

    Mn = npx.eye(n_states) - Mc - Mw

    return Mc, Mw, Mn

def _get_initial_state(n_states, start_width):

    Mid = int((n_states+1)/2)
    p_0 = npx.ones((n_states,1)) 
    p_0 = p_0.at[(Mid-start_width-1):(Mid+start_width)].set(1) # additional -1 because indexing starts from 0
    p_0 = p_0.reshape(-1,1) # to get column vector
    p_0 = p_0 / npx.sum(p_0)
    return p_0
    
def sample_states_and_confidence(rt_delta, phi_0, K):
    n_states = K.shape[-1] # picking the last dimension because the dimensions are I,J,K,K
    Mid = int((n_states+1)/2)
    mv = npx.arange(-(Mid-1),(Mid)).reshape(1,-1) # to get column vector

    T_t=ln.expm(rt_delta*K) # rt:(); K:m,m This is transaction matrix
    phi_t = T_t @ phi_0 # Probability of transition matrix at t time for each response time
    Mconf = mv @ phi_t
    return phi_t, Mconf

def likelihood(K,rt, ra, phi_0, delta, Mc, Mw, Mn):
    #ic(rt.shape)
    #K= _buildK(n_states, mu=mu, sigma=sigma)
    n_noresp = rt/delta
    #ic(n_noresp.shape)
    rt = npx.expand_dims(rt, axis=2)
    rt = npx.expand_dims(rt, axis=2)

    T_t=ln.expm(delta*K) # rt:I,J,1,1; K:I,J,m,m This is transaction matrix
    phi_noresp_arr = []

    for T_t_i, n_i in zip(T_t, n_noresp):
        #ic(T_t_i.shape, n_i.shape)
        for T_t_i_j, n_i_j in zip(T_t_i, n_i):
            #ic(T_t_i_j.shape, n_i_j.shape)
            phi_noresp_arr.append(num_ln.matrix_power(Mn @ T_t_i_j, n_i_j.astype(int).item()) @ phi_0)

    phi_noresp = npx.asarray(phi_noresp_arr)

    Pcorrect = (Mc @ phi_noresp).sum(axis=(-2,-1))
    Pincorrect = (Mw @ phi_noresp).sum(axis=(-2,-1))

    #ic(Pcorrect.shape)

    #Pcorrect = Pcorrect.sum(axis=(-2,-1)) #.squeeze() # adding up the probabilities over states for a given response
    
    P_total = npx.where(ra==0, Pincorrect, Pcorrect)
    likl = P_total.sum(axis=-1)

    return likl #likelihood hence adding up over all responses.

def perform_walk(n_states, start_width, mu, sigma,max_timesteps=10, delta=0.1):

    avg_conf = [];  
    state_prob = []
    likl_prob = []
    p_0 = _get_initial_state(n_states, start_width)
    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width)
    ra = 1

    for rt in list(npx.arange(0,max_timesteps,step=delta)):
        
        #print(p_0, "**********")
        K= _buildK(n_states, mu=mu, sigma=sigma)
        likl = likelihood(K,npx.asarray([[rt]]), npx.asarray([[ra]]), p_0, delta, Mc, Mw, Mn) #K,rt, ra, phi_0, delta, Mc, Mw, Mn
        Pt, Mconf = sample_states_and_confidence(rt, p_0, K)
        state_prob.append(Pt)
        likl_prob.append(likl)
        avg_conf.append(Mconf)

    df_avg_conf = pd.DataFrame(np.asarray(avg_conf).squeeze()).rename({0:"avg_conf"}, axis=1)
    df_avg_conf = df_avg_conf.reset_index().rename({"index":"time"}, axis=1)

    df_likl = pd.DataFrame(np.asarray(likl_prob)).rename({0:"liklihood"}, axis=1)
    df_likl = df_likl.reset_index().rename({"index":"time"}, axis=1)

    df_st = pd.DataFrame(np.asarray(state_prob).squeeze())
    Mid = int((n_states+1)/2)
    mv = np.arange(-(Mid-1),(Mid))
    df_st.columns = mv
    df_st = df_st.reset_index() \
                .melt(id_vars = "index",var_name="state", value_name="probability") \
                .rename({"index":"time"}, axis=1)
    #df_st.loc[:,"state"] = df_st.state.astype("category")
    df_st.loc[:,"time"] = df_st.time.astype("category")

    return df_st, df_avg_conf, df_likl

def model(n_states, start_width, rt, ra,I,J, s_0, Mr):
    
    mu_m =  npy.sample(f"mu_m", dist.Normal(2,3))
    mu_s =  npy.sample(f"mu_s", dist.HalfNormal(2))
    

    with npy.plate('I', I) as ind:
        mu =  npy.sample(f"mu", dist.Normal(mu_m,mu_s)) #,sample_shape=(I,)
    
    with npy.plate('Obs', I, subsample_size=10) as ind: #,
        #mu =  npy.sample(f"mu", dist.Normal(0,5)) #,sample_shape=(I,)
    
        rt1 = rt if rt is None else npx.asarray(rt)[ind]
        ra1 = ra if ra is None else npx.asarray(ra)[ind]
        _,lkl ,_ = likelihood(n_states, mu[ind], 1, rt1, ra1, s_0, Mr)
        npy.factor(f"likelihood", lkl)

def sample_posterior_params(DT, X, I,J, num_warmup=100, samples_n=500, n_states=7, start_width=1, num_chains=4):

    s_0, Mr = _get_initial_state(n_states, start_width)

    kernel = HMCECS(NUTS(model), num_blocks=10)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    mcmc_chain.run(_rng_key, n_states, start_width, DT, X, I,J, s_0, Mr)

    #kernel = NUTS(model)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=4)
    #mcmc_chain.run(_rng_key, n_states, start_width, DT, X, I,J, s_0, Mr)

    return mcmc_chain

def sample_post_pred_data(model, var_name, samples_n=100):
    
    prior_predictive = Predictive(model, num_samples=samples_n)
    prior_predictions = prior_predictive(_rng_key, marriage=dset.MarriageScaled.values)[var_name]
    return prior_predictions

if __name__ == "__main__":
    
    mu_arr,sigma = npx.asarray([[0.01,0.01,0.01,0.01,0.01,0.01,5]]), npx.asarray([1])
    

    ic(_buildK(7, mu=mu_arr, sigma=sigma))
    
    ic(_buildK(7, mu=[[1.5]], sigma=1))
    
    ic(_buildK(7, mu=[[0.5,1.5]], sigma=1))
    
    ic(_buildK(7, mu=[[1]], sigma=1).shape)
    
    ic(_buildK(7, mu=npx.asarray([[1,2,0.5]]), sigma=1).shape)


    ic("Constant Drift Rate")
    # Replicating the plots in Busemeyer 2010
    n_states=7
    start_width=0
    mu=[[0.5]] #drift rate
    sigma=2 #diffusion

    phi_0 = _get_initial_state(n_states, start_width)
    K = _buildK(n_states,mu,sigma)

    conf_arr = []
    for r in np.arange(0,20,0.1):
        P_t, Mconf = sample_states_and_confidence(r, phi_0, K)
        conf_arr.append(Mconf.squeeze())

    conf = np.asarray(conf_arr)
    ic(conf.shape)
    pd.Series(conf).plot.line()
    plt.show()

    df_st, df_avg_conf, df_avg_corr = perform_walk(n_states=7, start_width=3, mu=[[0.5]], sigma=2,max_timesteps=20, delta=0.1)
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    plt.show()
    
    ic("Variable Drift Rate - 1")
    phi_0 = _get_initial_state(n_states, start_width)
    K = _buildK(7, mu=mu_arr, sigma=sigma)

    #conf_arr = []
    #for r in np.arange(0,20,0.1):
    #    P_t, Mconf = sample_states_and_confidence(r, phi_0, K)
    #    conf_arr.append(Mconf.squeeze())

    #conf = np.asarray(conf_arr)
    #pd.Series(conf).plot.line()
    #plt.show()


    df_st, df_avg_conf, df_likl = perform_walk(n_states=7, start_width=3, mu=mu_arr, sigma=2,max_timesteps=20, delta=0.1)
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf")
    plt.show()

    ic("Variable Drift Rate - 2")
    mu_arr = npx.asarray([np.repeat([0.01,0.01,0.01,5],26)[0:101]])

    phi_0 = _get_initial_state(n_states, start_width)
    K = _buildK(7, mu=mu_arr, sigma=sigma)
    #conf_arr = []
    
    #for r in np.arange(0,20,0.1):
    #    P_t, Mconf = sample_states_and_confidence(r, phi_0, K)
    #    conf_arr.append(Mconf.squeeze())

    #conf = np.asarray(conf_arr)
    #pd.Series(conf).plot.line()
    #plt.show()


    df_st, df_avg_conf, df_likl = perform_walk(n_states=101, start_width=11, mu=mu_arr, sigma=2,max_timesteps=800, delta=80)
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf")
    plt.show()
   



