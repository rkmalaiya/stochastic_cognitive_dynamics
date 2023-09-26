#%%
import torch as np
import torch.linalg as ln
import pandas as pd
import pyro as npy
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
#npy.set_host_device_count(4)

def _buildK(n_states, mu, sigma): 
# m = number of states  
# a = off diag left  
# b = diag  
# c = off diag right
    n_part = np.as_tensor(mu).shape[0]
    K = np.zeros((n_part, 1, n_states,n_states)) # participants, trials, transition states

    for j in range(n_part):
        mk = np.ones((n_states,1))
        b = -sigma*mk
        a1 = 0.5 * (sigma-mu[j])*mk
        a2 = 0.5 * (sigma+mu[j])*mk
    
        for i in range(1, n_states-1):       
            #K = K.at[j,0,i, [i-1,i,i+1]].set([a1[i][0], b[i][0], a2[i][0]])
            K[j,0,i, [i-1,i,i+1]] = np.as_tensor([a1[i][0], b[i][0], a2[i][0]])
        
    return K.double()

def _get_initial_state(n_states, start_width):

    Mid = int((n_states+1)/2)
    s_0 = np.zeros((n_states,1))
    s_0[(Mid-start_width):(Mid+start_width)] = 1
    s_0 = s_0 / np.sum(s_0)
    Mr = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == int(n_states/2):
            Mr[i,i] = 0.5
        elif i > int(n_states/2):
            Mr[i,i] = 1
    return s_0.double(), Mr.double()

def likelihood(n_states, start_width, mu, sigma,rt,s_0, Mr):
    K= _buildK(n_states, mu=mu, sigma=sigma)
    Mid = int((n_states+1)/2)
    mv = np.arange(-(Mid-1),(Mid)).double()
    rt = rt.unsqueeze(2).unsqueeze(3)
    phi=ln.matrix_exp(rt) # rt:I,J,1,1; K:I,J,m,m
    A = np.mul(rt,K)
    phi=ln.matrix_exp(A) # rt:I,J,1,1; K:I,J,m,m
    Pt = np.matmul(phi, s_0)
    Mc = np.matmul(mv,Pt)
    Pcorrect = np.matmul(Mr,Pt).sum(axis=-2) # adding up the probabilities over states for a given response

    return Pt, Mc, Pcorrect.sum() #likelihood hence adding up over all responses.

def perform_walk(n_states, start_width, mu, sigma,timesteps=1.5, delta=0.01):

    avg_conf = [];  
    state_prob = []
    correct_prob = []
    s_0, Mr = _get_initial_state(n_states, start_width)

    for rt in list(np.arange(0.1,timesteps,step=delta)):
        
        Pt, Mc, Pcorrect = likelihood(n_states, start_width, mu, sigma,np.as_tensor([[rt]]),s_0, Mr)

        state_prob.append(Pt)
        correct_prob.append(Pcorrect)
        avg_conf.append(Mc)

    df_avg_conf = pd.DataFrame(np.stack(avg_conf).squeeze()).rename({0:"avg_conf"}, axis=1)
    df_avg_conf = df_avg_conf.reset_index().rename({"index":"time"}, axis=1)

    df_avg_corr = pd.DataFrame(np.asarray(correct_prob)).rename({0:"avg_conf"}, axis=1)
    df_avg_corr = df_avg_corr.reset_index().rename({"index":"time"}, axis=1)

    df_st = pd.DataFrame(np.stack(state_prob).squeeze())
    Mid = int((n_states+1)/2)
    mv = np.arange(-(Mid-1),(Mid))
    df_st.columns = mv.numpy()
    df_st = df_st.reset_index() \
                .melt(id_vars = "index",var_name="state", value_name="probability") \
                .rename({"index":"time"}, axis=1)
    #df_st.loc[:,"state"] = df_st.state.astype("category")
    df_st.loc[:,"time"] = df_st.time.astype("category")

    return df_st, df_avg_conf, df_avg_corr

def model(n_states, start_width, rt, s_0, Mr):
    I,J = rt.shape
    #with npy.plate('I', I):
        
    mu =  npy.sample(f"mu", dist.Normal(0,5),sample_shape=(I,))
    _,lkl ,_ = likelihood(n_states, start_width, mu, 1, rt, s_0, Mr)
    
    npy.factor(f"likelihood", lkl)

def sample_posterior_params(DT, X, num_warmup=100, samples_n=500, n_states=7, start_width=1):

    s_0, Mr = _get_initial_state(n_states, start_width)

    kernel = NUTS(model)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=4)
    mcmc_chain.run(n_states, start_width, DT, s_0, Mr)
    return mcmc_chain

#%%
if __name__ == "__main__":

    print(_buildK(7, mu=[1.5], sigma=1))
    print("*****************")
    print(_buildK(7, mu=[0.5,1.5], sigma=1))
    print("*****************")

    K = _buildK(7, mu=[1], sigma=1)
    print("test 1", K.shape)
    K = _buildK(7, mu=np.as_tensor([1,2,0.5]), sigma=1)
    print("test 2", K.shape)
 
if __name__ == "__main__":
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

   
#%%  
import numpy as n
if __name__ == "__main__":
    rotation_RT = pd.read_csv("examples/data/final_project_rt.csv")
    rotation_RT_n = rotation_RT.loc[~rotation_RT.isna().any(axis=1),:].to_numpy()

    s_0, Mr = _get_initial_state(7, 1)
    Pcorrect_arr = []
    for mu in np.linspace(0.1,1.5,50):
        Pt, Mc, Pcorrect = likelihood(7, 1, [mu], 1 , np.from_numpy(n.ones_like(rotation_RT_n)), s_0, Mr)
        Pcorrect_arr.append(Pcorrect)

    print("Pcorrect", np.asarray(Pcorrect_arr))
    print("Mc", Mc.shape)
    print("Pt", Pt.shape)


    s_0, Mr = _get_initial_state(101, 11)
    Pcorrect_arr = []
    for mu in np.linspace(0.1,1.5,50):
        Pt, Mc, Pcorrect = likelihood(101, 11, [mu], 1,rotation_RT_n, s_0, Mr)
        Pcorrect_arr.append(Pcorrect)

    print("Pcorrect", np.asarray(Pcorrect_arr))
    print("Mc", Mc.shape)
    print("Pt", Pt.shape)

    s_0, Mr = _get_initial_state(103, 11)
    Pt, Mc, Pcorrect = likelihood(103, 11, np.linspace(0.1,1.5,rotation_RT_n.shape[0]), 1,rotation_RT_n, s_0, Mr)
    print("Pcorrect", Pcorrect)
    print("Mc", Mc.shape)
    print("Pt", Pt.shape)
    #plt.show()

    mcmc_chain = sample_posterior_params(rotation_RT_n, None, num_warmup=10, samples_n=50, n_states=7, start_width=3)
    mcmc_chain.print_summary()




# %%
