import numpy as np
from scipy.linalg import expm
import pandas as pd

def _buildH(m,a,b,c): 
# H = buildH(a,b,c)
# m = number of states  
# a = off diag left  
# b = diag  
# c = off diag right

    H = np.zeros((m, m))
    rows_ = np.arange(1,m)
    cols_ = np.arange(0,m-1)
    H[rows_,cols_] = c
    H[cols_, rows_] =  a

    cols_ = np.arange(0,m)
    H[cols_,cols_] = b
    return H

def get_initial_state(num_states, start_width, Mid, type="L|C|H"):

    S0 = np.zeros((num_states,1))  
    if type=="L":
        S0[:start_width] = 1
    elif type=="C":
        S0[(Mid-start_width):(Mid+start_width)] = 1;
    elif type=="H":
        S0[-start_width:] = 1
    else:
        raise Exception(f"Acceptable values are 'L|C|H', but {type} was provided")
    S0 = S0/np.sqrt(S0.T @ S0);  
    return S0

def perform_walk(num_states, start_width, mu, sigma=1,timesteps=200,type="C"):
    
    Mid = (num_states+1)//2
    
    # initial state
    S0 = get_initial_state(num_states,start_width, Mid,type)
    
    # build Hamiltonian  
    mv = np.arange(0,num_states) #np.arange(-(Mid-1),(Mid))  
    b = mu*mv;  
    a = sigma#*np.ones((ns,1));  
    H = _buildH(num_states,a,b,a); # function given below  

    tv = np.arange(0,timesteps//10,0.1) # no time steps 

    # time loop  
    avg_conf = [];  
    state_prob = []
    for t in tv: 
        
        U = expm(-1j*t*H) #-1i 
        
        St = U@S0  
        state_prob.append((np.abs(St)**2))
        Mc = mv@(np.abs(St)**2)  
        
        avg_conf.append(Mc)  
    
    df_avg_conf = pd.DataFrame(np.asarray(avg_conf)).rename({0:"avg_conf"}, axis=1)
    df_avg_conf = df_avg_conf.reset_index().rename({"index":"time"}, axis=1)

    df_st = pd.DataFrame(np.asarray(state_prob).squeeze())
    df_st.columns = mv
    df_st = df_st.reset_index() \
                .melt(id_vars = "index",var_name="state", value_name="probability") \
                .rename({"index":"time"}, axis=1)
    #df_st.loc[:,"state"] = df_st.state.astype("category")
    df_st.loc[:,"time"] = df_st.time.astype("category")
    df_st

    return df_st, df_avg_conf

if __name__ == "__main__":
    mu = 1
    sigma = 1
    num_states = 13 # choose a odd number
    start_width = 3
    timesteps=200
    df_st, df_avg_conf = perform_walk(num_states, start_width, mu, sigma,timesteps=timesteps)
    #print(df_st.shape)
    #print(df_avg_conf.shape)

    assert df_st.shape[0] == num_states * timesteps
    assert df_avg_conf.shape[0] == timesteps
    import matplotlib.pyplot as plt


    df_avg_conf.iloc[:,1].plot()

    import seaborn as sns
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    plt.show()
    
