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
import cme.simulators.diffusion_random_walk as drw

ic.enable()
from numpyro.infer import MCMC, NUTS, SA, HMCECS, Predictive

npy.set_host_device_count(64)
_rng_key = random.PRNGKey(0)
_rng_key, _rng_key_ = random.split(_rng_key)

def _buildK(n_states, mu, sigma=1, delta=0.001): 
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
        #b1 = 0.5 * (((sigma**2)/delta**2) - mu[i,:]/delta) # 9.765
        #b2 = 0.5 * (((sigma**2)/delta**2) + mu[i,:]/delta) # 10.325 
        b1 = 0.5 * (sigma - mu[i,:])
        b2 = 0.5 * (sigma + mu[i,:])
                
        a = -(b1+b2)
       
        for j in range(1,n_states-1):
            K = K.at[i,0,[j-1,j,j+1],j].set([b1[j], a[j], b2[j]])
            #K = K.at[i,0,[j-1,j,j+1],j].set([b1, a, b2])

        K = K.at[i,0,[0,1],0].set([a[0], -a[0]])
        K = K.at[i,0,[-2,-1],-1].set([-a[-1], a[-1]])

        #K = K.at[i,0,[0,1],0].set([a, -a])
        #K = K.at[i,0,[-2,-1],-1].set([-a, a])
        
    return K

def _get_measurement_matrix(n_states, start_width, prob = 0.5):
    
    Mcorr = npx.zeros(n_states)
    Mcorr = Mcorr.at[-start_width:].set(prob)
    Mcorr = npx.diag(Mcorr)

    Mincorr = npx.zeros(n_states)
    Mincorr = Mincorr.at[:start_width].set(prob)
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

    return Mcorr, Mincorr, Mnoresp

def _get_initial_state(n_states, start_width):

    Mid = int((n_states+1)/2)
    p_0 = npx.zeros((n_states,1)) 
    p_0 = p_0.at[(Mid-start_width-1):(Mid+start_width)].set(1) # additional -1 because indexing starts from 0
    p_0 = p_0.reshape(-1,1) # to get column vector
    p_0 = p_0 / npx.sum(p_0)
    return p_0
    
def sample_states_and_confidence(rt_delta, phi_0, K, Mn, N):
    n_states = K.shape[-1] # picking the last dimension because the dimensions are I,J,K,K
    Mid = int((n_states+1)/2)
    mv = npx.arange(-(Mid-1),(Mid)).reshape(1,-1) # to get column vector

    T_t=ln.expm(rt_delta*K) # rt:(); K:m,m This is transaction matrix
    phi_t = T_t @ phi_0 # Probability of transition matrix at t time for each response time
    Mconf = mv @ (phi_t)

    if False:
        noresp_traj_arr = []
        noresp_state = Mn @ T_t
        noresp_traj_arr.append(noresp_state @ phi_0)

        for n in npx.arange(2,N): #arange goes upto N-1, hence we do not need to explicitly substract 1 here.
            noresp_state = noresp_state * noresp_state
            noresp_traj_arr.append(noresp_state @ phi_0)

        phi_t = noresp_state @ phi_0
        Mconf = mv @ phi_t

    return phi_t, Mconf, npx.asarray(noresp_traj_arr) if False else None


def likelihood(K,rt, ra, phi_0, n_noresp, delta, Mc, Mw, Mn):
    #ic(rt.shape)
    #K= _buildK(n_states, mu=mu, sigma=sigma)
    
    #ic(n_noresp.shape)
    #if rt is not None:
    #    rt = npx.expand_dims(rt, axis=2)
    #    rt = npx.expand_dims(rt, axis=2)

    delta = npx.expand_dims(delta, axis=2)
    delta = npx.expand_dims(delta, axis=2)

    T_t=ln.expm(delta*K) # rt:I,J,1,1; K:I,J,m,m This is transaction matrix
    
    phi_noresp_arr_i = []
    for T_t_i, n_i in zip(T_t, n_noresp):
        phi_noresp_arr_i_j = []
        for T_t_i_j, n_i_j in zip(T_t_i, n_i):
            phi_noresp_arr_i_j.append(num_ln.matrix_power(Mn @ T_t_i_j, n_i_j.astype(int).item()-1) @ phi_0) #n_state x 1
        phi_noresp_arr_i.append(npx.asarray(phi_noresp_arr_i_j))    
    phi_noresp = npx.asarray(phi_noresp_arr_i) # n_part x n_trials x n_state x 1

    Pcorrect = (Mc @ phi_noresp).sum(axis=(-2,-1))
    Pincorrect = (Mw @ phi_noresp).sum(axis=(-2,-1))

    #ic(Pcorrect.shape)

    #Pcorrect = Pcorrect.sum(axis=(-2,-1)) #.squeeze() # adding up the probabilities over states for a given response
    
    P_total = npx.where(ra==0, Pincorrect, Pcorrect)
    likl = P_total.sum()

    return likl #likelihood hence adding up over all responses.

def perform_walk(n_states, start_width, mu, sigma, max_timesteps=10, delta=1, prob=0.25, n_noresp = None):

    avg_conf = [];  
    state_prob = []
    likl_prob = []
    n_noresp_arr = []
    noresp_traj_arr = []
    p_0 = _get_initial_state(n_states, start_width)
    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width, prob)
    ra = npx.asarray([[1]])
    if delta is not None:
        delta = npx.asarray(delta)

    if n_noresp is not None:
        n_noresp = npx.asarray(n_noresp)

    for rt in list(npx.arange(0.1,max_timesteps,step=0.1)):
    #for rt in np.linspace(1,max_timesteps, 50):
        rt = npx.asarray([[rt]])
        if n_noresp is None:
            n_noresp_1 = rt/delta
        else: 
            delta = rt/n_noresp

        #print(p_0, "**********")
        K= _buildK(n_states, mu=mu, sigma=sigma, delta=delta)
        

        likl = likelihood(K, rt, ra, p_0, n_noresp_1, delta, Mc, Mw, Mn) #K,rt, ra, phi_0, delta, Mc, Mw, Mn
        Pt, Mconf, noresp_traj = sample_states_and_confidence(rt, p_0, K.squeeze(), Mn, n_noresp_1.squeeze())
        state_prob.append(Pt)
        likl_prob.append(likl)
        avg_conf.append(Mconf)
        noresp_traj_arr.append(noresp_traj)
        n_noresp_arr.append(n_noresp_1)

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

def model(n_states, start_width, rt, ra,I,J, s_0):
    
    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width, 0.25)

    mu_m =  npy.sample(f"mu_m", dist.Normal(2,3))
    mu_s =  npy.sample(f"mu_s", dist.HalfNormal(2))
    delta = np.asarray([[0.001]])
    sigma=1
    n_noresp = rt/delta if rt is not None else 10
    
    with npy.plate('I', I, dim=-2) as ind:
        mu =  npy.sample(f"mu", dist.Normal(mu_m,mu_s),sample_shape=(I,1))

        K = _buildK(n_states,mu,sigma,delta=delta)
    
    #with npy.plate('Obs', I, subsample_size=10) as ind: #,
        #mu =  npy.sample(f"mu", dist.Normal(0,5)) #,sample_shape=(I,)
    
        #rt1 = rt if rt is None else npx.asarray(rt)[ind]
        #ra1 = ra if ra is None else npx.asarray(ra)[ind]
        #_,lkl ,_ = likelihood(n_states, mu[ind], 1, rt1, ra1, s_0, Mr)
        if rt is not None:
            lkl = likelihood(K, rt, ra, s_0, n_noresp, delta, Mc, Mw, Mn)
            npy.factor(f"likelihood", lkl)

def sample_posterior_params(DT, X, I,J, n_states, start_width, num_warmup=100, samples_n=500, num_chains=4):

    s_0 = _get_initial_state(n_states, start_width)
    
    #kernel = HMCECS(NUTS(model), num_blocks=10)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    #mcmc_chain.run(_rng_key, n_states, start_width, DT, X, I,J, s_0, Mr)

    kernel = NUTS(model)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    mcmc_chain.run(_rng_key, n_states, start_width, DT, X, I, J, s_0)

    return mcmc_chain

def sample_prior_pred_data(n_states, start_width, tau, sigma, I, J, samples_n=100):
    s_0 = _get_initial_state(n_states, start_width)
    prior_predictive = Predictive(model, num_samples=samples_n)
    prior_predictions = prior_predictive(_rng_key, n_states, start_width, None, None, I, J, s_0)

    theta = int((n_states+1)/2)
    mu_s = prior_predictions["mu"]
    
    RT, X, Steps = get_rt_sample(theta, 1.5, tau, sigma[0], mu_s, n_trials = J)

    prior_predictions["RT"] = RT
    prior_predictions["X"] = X
    prior_predictions["Steps"] = Steps

    #C = mu_s.shape[0]
    #Mid = int((n_states+1)/2)
    #for c in npx.arange(C):
    #    RT_t, X_t, steps_arr_t = get_rt_sample(theta, 1.5, tau, sigma, mu_s[c,...], I, J, samples_n)
    #    RT[c,...] = RT_t
    #    X[c,...] = X_t
    #    Steps[c,...] = steps_arr_t

    return prior_predictions

def get_rt_sample(theta, alpha, tau, sigma, mu_s, n_trials):

    C, I, _ = mu_s.shape

    RT = np.empty((C, I, n_trials))
    X = np.empty((C, I, n_trials))
    steps_arr_c = []

    for c in npx.arange(C):
        steps_arr_i = []
        for i in npx.arange(I):

            RT_arr, X_arr, steps = drw.gen_rt_x(theta, alpha, tau, sigma, mu_s[c,i,...], samples=n_trials, process="Wiener", initial="Any")
            RT[c, i,:] = RT_arr
            X[c, i,:] = X_arr
            steps_arr_i.append(steps)
        steps_arr_c.append(steps_arr_i)
        
    Steps = pd.DataFrame(steps_arr_c)
    return RT, X, Steps


def sample_post_pred_data(n_states, start_width, tau, sigma, mcmc_samples, n_trials):
    mu_s = mcmc_samples["mu"]

    theta = int((n_states+1)/2)

    RT, X, Steps = get_rt_sample(theta, 1.5, tau, sigma[0], mu_s, n_trials = n_trials)

    return RT, X, Steps

    

if __name__ == "__main__":
    
    mu_arr,sigma = npx.asarray([[0.01,0.01,0.01,0.01,0.01,1,1]]), npx.asarray([10])
    

    ic(_buildK(7, mu=mu_arr, sigma=sigma))
    
    ic(_buildK(7, mu=[[1.5]], sigma=10))
    
    ic(_buildK(7, mu=[[0.5,1.5]], sigma=10)) # this didn't throw out of bounds exception because of JAX's behavior
    
    ic(_buildK(7, mu=[[1]], sigma=10).shape)
    
    ic(_buildK(7, mu=npx.asarray([[1,2,0.5]]), sigma=1).shape) # this didn't throw out of bounds exception because of JAX's behavior

    ic("Posterior Sampling - 1")
    I, J = 5,6
    rt = np.random.uniform(0.1, 10, size=(I,J))
    x = np.random.randint(0,2, size=(I,J))
    n_states=7
    start_width=1
    tau = 0.001
    mcmc_chain = sample_posterior_params(DT=rt, X=x,I=I,J=J, n_states=n_states, start_width=start_width, num_warmup=10, samples_n=4 )
    mcmc_chain.print_summary()
    mcmc_samples = mcmc_chain.get_samples()

    ic("Prior Prediction - 1")
    prior_predictions = sample_prior_pred_data(n_states, start_width, tau, sigma,  I, J, samples_n=4)
    RT = prior_predictions["RT"]
    ic(RT.min(), RT.mean(), RT.max())

    ic("Posterior Prediction - 1") # Test here
    RT, X, steps_arr = sample_post_pred_data(n_states, start_width, tau, sigma, mcmc_samples, n_trials=J)
    ic(RT.min(), RT.mean(), RT.max())

    ic("Constant Drift Rate - 1")
    # Replicating the plots in Busemeyer 2010
    n_states=7
    start_width=3
    mu=[[0.5]] #drift rate
    sigma=2 #diffusion
    delta=[[tau]]

    phi_0 = _get_initial_state(n_states, start_width)
    K = _buildK(n_states,mu,sigma, delta)
    Mc, Mw, Mn = _get_measurement_matrix(n_states, start_width, 0.25)


    conf_arr = []
    
    for r in np.arange(0,20,0.1):
        P_t, Mconf, noresp_traj_arr = sample_states_and_confidence(r, phi_0, K, Mn, 1)
        conf_arr.append(Mconf.squeeze())

    conf = np.asarray(conf_arr)
    ic(conf.shape)
    pd.Series(conf).plot.line()
    plt.show()

    ic("Constant Drift Rate - 2")
    # Replicating the plots in Busemeyer 2010
    n_states=101
    start_width=4
    mu=[[0.56]] #drift rate
    sigma=20.09 #diffusion

    phi_0 = _get_initial_state(n_states, start_width)
    K = _buildK(n_states,mu,sigma,delta=delta)

    df_st, df_avg_conf, df_likl = perform_walk(n_states=n_states, start_width=start_width, mu=mu, sigma=sigma,max_timesteps=30, delta=delta, prob=0.25)#, n_noresp=npx.asarray([[1]]))
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf")
    sns.relplot(df_likl, x="time",y="liklihood")
    plt.show()


    ic("Variable Drift Rate - 1")
    phi_0 = _get_initial_state(7, start_width)
    #delta = 90/2
    K = _buildK(7, mu=mu_arr, sigma=10, delta=delta)
    
    Mc, Mw, Mn = _get_measurement_matrix(7, start_width, 0.25)

    Pt, Mconf, noresp_traj = sample_states_and_confidence(90, phi_0, K.squeeze(), Mn, npx.asarray([[2]]))

    df_st, df_avg_conf, df_likl = perform_walk(n_states=7, start_width=3, mu=mu_arr, sigma=10, max_timesteps=100, delta=delta)#, n_noresp=npx.asarray([[2]]))
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf")
    sns.relplot(df_likl, x="time",y="liklihood")
    plt.show()

    ic("Variable Drift Rate - 2")
    mu_arr = npx.asarray([np.repeat([0.01,0.01,0.01,2],26)[0:101]])

    phi_0 = _get_initial_state(n_states, start_width)
    K = _buildK(7, mu=mu_arr, sigma=sigma, delta=delta)

    df_st, df_avg_conf, df_likl = perform_walk(n_states=101, start_width=11, mu=mu_arr, sigma=10,max_timesteps=50, delta=delta)#, n_noresp=npx.asarray([[5]]))
    
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    sns.relplot(df_avg_conf, x="time",y="avg_conf")
    sns.relplot(df_likl, x="time",y="liklihood")
    plt.show()
   



