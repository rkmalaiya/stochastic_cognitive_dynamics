import numpy as np
from scipy.linalg import expm
import pandas as pd


def _get_confidence_measurement_matrix(n_states, prob = 0.5):
    
    Mid = (num_states+1)//2

    Mcorr = np.zeros(n_states)
    Mcorr[-Mid+1:] = 1
    Mcorr[Mid] = 1/np.sqrt(2)
    Mcorr = np.diag(Mcorr)

    Mincorr = np.zeros(n_states)
    Mincorr[:Mid] = prob
    Mincorr[Mid] = 1/np.sqrt(2)
    Mincorr = np.diag(Mincorr)
    
    #Mc = npx.zeros((n_states, n_states)) # correct response
    #for i in range(n_states):
    #    if i > int(n_states/2) + start_width:
    #        Mc = Mc.at[i,i].set(prob)

    #Mw = npx.zeros((n_states, n_states)) # incorrect response
    #for i in range(n_states):
    #    if i < int(n_states/2) - start_width:
    #        Mw = Mw.at[i,i].set(prob)

    

    return Mcorr, Mincorr

def _get_measurement_matrix(n_states, start_width, prob = 0.5):
    
    Mcorr = np.zeros(n_states)
    Mcorr[-start_width:] = prob
    Mcorr = np.diag(Mcorr)

    Mincorr = np.zeros(n_states)
    Mincorr[:start_width] = prob
    Mincorr = np.diag(Mincorr)
    
    #Mc = npx.zeros((n_states, n_states)) # correct response
    #for i in range(n_states):
    #    if i > int(n_states/2) + start_width:
    #        Mc = Mc.at[i,i].set(prob)

    #Mw = npx.zeros((n_states, n_states)) # incorrect response
    #for i in range(n_states):
    #    if i < int(n_states/2) - start_width:
    #        Mw = Mw.at[i,i].set(prob)

    Mnoresp = np.eye(n_states) - Mcorr - Mincorr

    return Mcorr, Mincorr, Mnoresp



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
        S0[(Mid-start_width-1):(Mid+start_width)] = 1;
    elif type=="H":
        S0[-start_width:] = 1
    else:
        raise Exception(f"Acceptable values are 'L|C|H', but {type} was provided")
    S0 = S0/np.sqrt(S0.T @ S0);  
    return S0

def perform_walk(num_states, start_width, mu, sigma=1,rt=200,type="C"):
    
    Mcorr, Mincorr, Mnoresp = _get_measurement_matrix(num_states, start_width, prob = 0.5)

    Mid = (num_states+1)//2
    
    # initial state
    S0 = get_initial_state(num_states,start_width, Mid,type)
    
    # build Hamiltonian  
    mv = np.arange(-(Mid-1),(Mid))  # Basis vector np.arange(0,num_states) #
    b = mu*mv;  
    a = sigma#*np.ones((ns,1));  
    H = _buildH(num_states,a,b,a); # function given below  
    tv = np.arange(0,rt,0.1) # no of time steps 
    nt = tv.shape[0]
    # time loop  
    avg_conf = [];  
    state_prob = []
    likl_arr = []
    
    for n in range(1, nt):
        t = tv[n]
        U = expm(-1j*t*H) #-1i 
        St = U@S0  
        Pt = (np.abs(St)**2)
        Mc = mv@Pt #mean confidence
        likl = np.abs(Mcorr @ St)**2
        state_prob.append(Pt)
        likl_arr.append(likl.sum()) 
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
    
    return df_st, df_avg_conf, pd.Series(likl_arr)


def perform_walk_interfere(num_states, start_width, mu, sigma=1,rt=200,type="C", delta=1):
    
    Mcorr, Mincorr, Mnoresp = _get_measurement_matrix(num_states, start_width, prob = 0.5)

    Mid = (num_states+1)//2
    
    # initial state
    S0 = get_initial_state(num_states,start_width, Mid,type)
    
    # build Hamiltonian  
    mv = np.arange(0,num_states) #np.arange(-(Mid-1),(Mid))  # Basis vector
    b = mu*mv;  
    a = sigma#*np.ones((ns,1));  
    H = _buildH(num_states,a,b,a); # function given below  

    tv = np.arange(0,rt,0.1) # no of time steps 

    # time loop  
    avg_conf = [];  
    state_prob = []
    likl_arr = []
    
    for t in tv: 
        
        U = expm(-1j*delta*H) #-1i 
        N = int(t/delta)
        St = U @ np.linalg.matrix_power(Mnoresp @ U, N-1) @ S0  
        Pt = (np.abs(St)**2)
        Mc = mv@Pt #mean confidence
        likl = np.abs(Mcorr @ St)**2

        state_prob.append(Pt)
        likl_arr.append(likl.sum()) 
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
    
    return df_st, df_avg_conf, pd.DataFrame(pd.Series(likl_arr))




if __name__ == "__main__":
    mu = 1
    sigma = 1
    num_states = 7 # choose a odd number
    start_width = 3
    rt=20

    df_st, df_avg_conf, ds_likl = perform_walk(num_states, start_width, mu, sigma,rt=rt)

    #print(df_st.shape)
    #print(df_avg_conf.shape)

    #assert df_st.shape[0] == num_states * timesteps
    #assert df_avg_conf.shape[0] == timesteps
    import matplotlib.pyplot as plt


    df_avg_conf.iloc[:,1].plot()

    import seaborn as sns
    sns.relplot(df_st, x="time",y="state",size="probability",sizes=(50, 300), color="black")
    plt.show()
    ds_likl.plot()
    plt.show()

    df_st, df_avg_conf, ds_likl = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for delta in range(1, rt, 1):
        df_st_t, df_avg_conf_t, ds_likl_t = perform_walk_interfere(num_states, start_width, mu, sigma,rt=rt, delta=delta)
        df_st = pd.concat([df_st, df_st_t.assign(delta=delta)])
        df_avg_conf = pd.concat([df_avg_conf, df_avg_conf_t.assign(delta=delta)])
        ds_likl = pd.concat([ds_likl, ds_likl_t.assign(delta=delta)])
    df_avg_conf.groupby("delta")["avg_conf"].mean().plot.bar()

    sns.barplot(df_avg_conf.groupby("delta")["avg_conf"].mean().reset_index().assign(N = lambda df: np.round(rt/df["delta"])),
        x="N",
        y="avg_conf", color="Grey")