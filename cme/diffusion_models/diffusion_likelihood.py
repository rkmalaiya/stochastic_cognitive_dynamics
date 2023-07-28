#%%
import numpy as np
import cme.utils.common_logging as cl
from cme.utils import common_utils as ut


# enable on-the-fly graph computations
#ae.config.compute_test_value = 'warn'

log = cl.get_logger("diffusion")
eps = 0.01 # for numerical stability
err = 10e-10
max_k = 100

def _get_count_l(tt):

    K = np.sqrt((-2) * np.log(np.pi * tt * err) / ((np.pi**2) * tt) )

    K = np.vectorize(lambda K, tt: K if K > 1/(np.pi * np.sqrt(tt)) else 1 / (np.pi * np.sqrt(tt)))(K, tt)

    #K = np.switch(np.gt(K, 1/(np.pi * np.sqrt(tt)) ), K, (1 / (np.pi * np.sqrt(tt)) ) ) # based on RWiener package in rlang
    return K

def _get_count_s(tt):
    print("*****************")
    K = np.sqrt( -2 * tt * np.log( 2 * err* np.sqrt( 2 * np.pi * tt ) ) )
    print(K)
    K = np.vectorize(lambda K, tt: K if K > np.sqrt(tt)+1 else np.sqrt(tt)+1)(K, tt)
    print(K)
    #K = np.switch(np.gt(K, np.sqrt(tt)+1), K, np.sqrt(tt) + 1)
    return K

def _get_lambda(tt):
    
    return 2 + _get_count_s(tt) - _get_count_l(tt)

def _diffusion_01w_s(tt, w):

    K_m = _get_count_s(tt)
    K_n = np.max(np.floor(K_m))
    #K_n = np.switch(np.gt(K_n,max_k), max_k, K_n)
    print(K_m)
    K=np.arange( -np.floor((K_n-1)/2), np.ceil((K_n-1)/2) + 1 )[:,np.newaxis, np.newaxis]
    #print("***********k",K.shape)

    prob_rt_std = ((w + 2*K) * np.exp( - ((w+2*K)*(w+2*K)) /(2*tt)) ).sum(axis=0)

    prob_rt_std = prob_rt_std * 1/np.sqrt(2*np.pi*tt*tt*tt)

    #print("***********prob_rt_std",prob_rt_std.shape)

    return prob_rt_std

def _diffusion_01w_l(tt, w):

    K_m = _get_count_l(tt)
    K_n = np.max(np.floor(K_m))
    #K_n = np.switch(np.gt(K_n, max_k), max_k, K_n)

    K=np.arange(1,K_n+1)[:,np.newaxis, np.newaxis]

    #print("***********k_l",K.shape)

    prob_rt_std = np.pi * (K * np.exp( - ((K*np.pi)**2 * tt/2) ) * np.sin( K * np.pi * w )).sum(axis=0) #exp becomes zero for large tt, hence adding eps
    
    #for k in K:
    #    prob_rt_std += np.pi * (k * np.exp( - ((k*np.pi)**2 * tt/2) ) * np.sin(k * np.pi * w ))

    #print("***********prob_rt_std",prob_rt_std.shape)
    
    return  prob_rt_std

def _diffusion_01std(t,a,w):

    tt = t/(a**2)
    lmda = _get_lambda(tt)
    st = _diffusion_01w_s(tt, w)
    lt = _diffusion_01w_l(tt, w)
    prob_rt_std = np.vectorize(lambda lmda, st, lt: st if lmda < 0 else lt)(lmda, st, lt)
    #prob_rt_std = np.switch(np.lt(lmda, 0), st, lt)


    prob_rt_final = prob_rt_std

    return prob_rt_final #should return a scaler

def _calculate_RT_logp(DT, V, A, Z):
    
    prob_rt_std = _diffusion_01std(DT,A,Z/A)
    
    prob_rt = (1 / (A*A)) * (np.exp( (-Z*A*V) - ((V*V) * DT)/2 )) * prob_rt_std
    prob_rt = prob_rt 

    return prob_rt

#%%
#if __name__ == "main":
DT = np.linspace(0.1,2,10)
V = np.linspace(-2,2,10)
A = np.linspace(0.1,5,10)
Z = np.linspace(0,1,10)

prob_rt = _calculate_RT_logp(DT, V, A, Z)
print(prob_rt)


# %%
