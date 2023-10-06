import jax.numpy as npx
import jax.scipy.linalg as ln
import numpy as np
import pandas as pd
import numpyro as npy
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS, SA, HMCECS, Predictive

npy.set_host_device_count(64)
_rng_key = random.PRNGKey(0)
_rng_key, _rng_key_ = random.split(_rng_key)

def _buildK(n_states, mu, sigma): 
# m = number of states  
# a = off diag left  
# b = diag  
# c = off diag right
    n_part = npx.asarray(mu).shape[0]
    K = npx.zeros((n_part, 1, n_states,n_states)) # participants, trials, transition states

    for j in range(n_part):
        mk = npx.ones((n_states,1))
        b = -sigma*mk
        a1 = 0.5 * (sigma-mu[j])*mk
        a2 = 0.5 * (sigma+mu[j])*mk
    
        for i in range(1, n_states-1):       
            K = K.at[j,0,i, [i-1,i,i+1]].set([a1[i][0], b[i][0], a2[i][0]])
        
    return K

def _get_initial_state(n_states, start_width):

    Mid = int((n_states+1)/2)
    s_0 = npx.zeros((n_states,1))
    s_0 = s_0.at[(Mid-start_width):(Mid+start_width)].set(1)
    s_0 = s_0 / npx.sum(s_0)
    Mr = npx.zeros((n_states, n_states))
    for i in range(n_states):
        if i == int(n_states/2):
            Mr = Mr.at[i,i].set(0.5)
        elif i > int(n_states/2):
            Mr = Mr.at[i,i].set(1)
    return s_0, Mr

def likelihood(n_states, mu, sigma,rt, ra, s_0, Mr):
    K= _buildK(n_states, mu=mu, sigma=sigma)

    Mid = int((n_states+1)/2)
    mv = npx.arange(-(Mid-1),(Mid))
    rt = npx.expand_dims(rt, axis=2)
    rt = npx.expand_dims(rt, axis=2)
    phi=ln.expm(rt*K) # rt:I,J,1,1; K:I,J,m,m This is transaction matrix
    Pt = npx.matmul(phi, s_0) # Probability of transition matrix at t time for each response time
    Mc = npx.matmul(mv,Pt)
    Pcorrect = npx.matmul(Mr,Pt)
    Pcorrect = Pcorrect.sum(axis=(-2,-1)) #.squeeze() # adding up the probabilities over states for a given response
    Pcorrect = npx.where(ra==0, 1-Pcorrect, Pcorrect)
    return Pt, Mc, Pcorrect.sum() #likelihood hence adding up over all responses.

def perform_walk(n_states, start_width, mu, sigma,timesteps=1.5, delta=0.01):

    avg_conf = [];  
    state_prob = []
    correct_prob = []
    s_0, Mr = _get_initial_state(n_states, start_width)

    for rt in list(np.arange(0,timesteps,step=delta)):
        
        Pt, Mc, Pcorrect = likelihood(n_states, mu, sigma,npx.asarray([[rt]]), npx.asarray([[1]]), s_0, Mr)

        state_prob.append(Pt)
        correct_prob.append(Pcorrect)
        avg_conf.append(Mc)

    df_avg_conf = pd.DataFrame(np.asarray(avg_conf).squeeze()).rename({0:"avg_conf"}, axis=1)
    df_avg_conf = df_avg_conf.reset_index().rename({"index":"time"}, axis=1)

    df_avg_corr = pd.DataFrame(np.asarray(correct_prob)).rename({0:"avg_conf"}, axis=1)
    df_avg_corr = df_avg_corr.reset_index().rename({"index":"time"}, axis=1)

    df_st = pd.DataFrame(np.asarray(state_prob).squeeze())
    Mid = int((n_states+1)/2)
    mv = np.arange(-(Mid-1),(Mid))
    df_st.columns = mv
    df_st = df_st.reset_index() \
                .melt(id_vars = "index",var_name="state", value_name="probability") \
                .rename({"index":"time"}, axis=1)
    #df_st.loc[:,"state"] = df_st.state.astype("category")
    df_st.loc[:,"time"] = df_st.time.astype("category")

    return df_st, df_avg_conf, df_avg_corr

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

    print(_buildK(7, mu=[1.5], sigma=1))
    print("*****************")
    print(_buildK(7, mu=[0.5,1.5], sigma=1))
    print("*****************")

    K = _buildK(7, mu=[1], sigma=1)
    print("test 1", K.shape)
    K = _buildK(7, mu=npx.asarray([1,2,0.5]), sigma=1)
    print("test 2", K.shape)
    df_st, df_avg_conf, df_avg_corr = perform_walk(n_states=7, start_width=3, mu=[0.5], sigma=2,timesteps=20, delta=0.1)
    print("test 3")
    #df_avg_conf.iloc[:,1].plot(xlabel="Number of Steps", ylabel="Average Confidence")
    #df_avg_corr.iloc[:,1].plot()
    

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    
    #rng_key = random.PRNGKey(0)

    #DT = random.normal(rng_key,shape=(50,20))**2

    #mcmc_chain = sample_posterior_params(DT, None, samples_n=500)
    #mcmc_chain.print_summary()

   

    rotation_RT = pd.read_csv("examples/data/final_project_rt.csv")
    rotation_RT_n = rotation_RT.loc[~rotation_RT.isna().any(axis=1),:].to_numpy()

    rotation_X = pd.read_csv("examples/data/final_project_ra.csv")
    rotation_X_n = rotation_X.loc[~rotation_RT.isna().any(axis=1),:].to_numpy()

    s_0, Mr = _get_initial_state(101, 11)
    Pcorrect_arr = []
    for mu in npx.linspace(0.1,1.5):
        Pt, Mc, Pcorrect = likelihood(101, [mu], 1,rotation_RT_n, rotation_X_n, s_0, Mr)
        Pcorrect_arr.append(Pcorrect)

    print("Pcorrect", np.asarray(Pcorrect_arr))
    print("Mc", Mc.shape)
    print("Pt", Pt.shape)

    s_0, Mr = _get_initial_state(103, 11)
    Pt, Mc, Pcorrect = likelihood(103, npx.linspace(0.1,1.5,rotation_RT_n.shape[0]), 1,rotation_RT_n, rotation_X_n, s_0, Mr)
    print("Pcorrect", Pcorrect)
    print("Mc", Mc.shape)
    print("Pt", Pt.shape)
    #plt.show()

    mcmc_chain = sample_posterior_params(rotation_RT_n, None, *rotation_RT_n.shape, num_warmup=10, samples_n=50, n_states=7, start_width=3)
    mcmc_chain.print_summary()



