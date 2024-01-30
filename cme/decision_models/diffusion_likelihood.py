#%%
import torch as np
import numpy as npy
import cme.utils.common_logging as cl
from cme.utils import common_utils as ut
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive


# enable on-the-fly graph computations
#ae.config.compute_test_value = 'warn'

log = cl.get_logger("diffusion")
eps = 0.01 # for numerical stability
err = 10e-10

loop_small = []
loop_big = []


def _get_count_l(tt):

    K = np.sqrt((-2) * np.log(np.pi * tt * err) / ((np.pi**2) * tt) )
    #K = np.vectorize(lambda K, tt: K if K > 1/(np.pi * np.sqrt(tt)) else 1 / (np.pi * np.sqrt(tt)))(K, tt)
    K = np.where(np.lt(K, 1/(np.pi * np.sqrt(tt))), K, 1 / (np.pi * np.sqrt(tt)))
    loop_big.append(K.max())
    #K = np.switch(np.gt(K, 1/(np.pi * np.sqrt(tt)) ), K, (1 / (np.pi * np.sqrt(tt)) ) ) # based on RWiener package in rlang
    return K

def _get_count_s(tt):
    K = np.sqrt( -2 * tt * np.log( 2 * err* np.sqrt( 2 * np.pi * tt ) ) )
    #K = np.vectorize(lambda K, tt: K if K > np.sqrt(tt)+1 else np.sqrt(tt)+1)(K, tt)
    K = np.where(K > np.sqrt(tt)+1,K, np.sqrt(tt)+1)
    #K = np.switch(np.gt(K, np.sqrt(tt)+1), K, np.sqrt(tt) + 1)
    loop_small.append(K.max())
    return K

def _get_lambda(tt):
    
    return 2 + _get_count_s(tt) - _get_count_l(tt)

def _diffusion_01w_s(tt, w):

    K_m = _get_count_s(tt)
    K_n = 200 #np.nanmax(np.ceil(K_m))
    #K_n = np.switch(np.gt(K_n,max_k), max_k, K_n)
    #print("small size - loop count", K_n.max(), np.nanmax(K_m))
    K=np.arange( -npy.floor((K_n-1)/2), npy.ceil((K_n-1)/2) + 1 )[:,npy.newaxis, npy.newaxis]
    #print("***********k",K.shape)
    prob_rt_std = 0
    #for K in K1:
    prob_rt_std = ((w + 2*K) * np.exp( - ((w+2*K)*(w+2*K)) /(2*tt)) ).sum(axis=0)

    prob_rt_std = prob_rt_std * 1/np.sqrt(2*np.pi*tt*tt*tt)

    #print("***********prob_rt_std",prob_rt_std.shape)

    return prob_rt_std

def _diffusion_01w_l(tt, w):

    K_m = _get_count_l(tt)
    K_n = 200 #np.nanmax(np.ceil(K_m))
    #K_n = np.switch(np.gt(K_n, max_k), max_k, K_n)
    #print("large size - loop count", K_n, K_m.shape)
    K=np.arange(1,K_n+1)[:,npy.newaxis, npy.newaxis]

    #print("***********k_l",K.shape)

    prob_rt_std = np.pi * (K * np.exp( - ((K*np.pi)**2 * tt/2) ) * np.sin( K * np.pi * w )).sum(axis=0) #exp becomes zero for large tt, hence adding eps
    
    #for k in K:
    #    prob_rt_std += np.pi * (k * np.exp( - ((k*np.pi)**2 * tt/2) ) * np.sin(k * np.pi * w ))

    #print("***********prob_rt_std",prob_rt_std.shape)
    
    return  prob_rt_std

def _diffusion_01std(t,a,w):
    
    tt = t/(a**2)
    lmda = _get_lambda(tt)

    prob_rt_std = _diffusion_01w_l(tt, w) #np.where(lmda < 0, _diffusion_01w_s(tt, w), _diffusion_01w_l(tt, w))
    
    #prob_rt_std = np.vectorize(lambda lmda, st, lt: st if lmda < 0 else lt)(lmda, st, lt)
    #prob_rt_std = np.switch(np.lt(lmda, 0), st, lt)


    prob_rt_final = prob_rt_std

    return prob_rt_final #should return a scaler

def _calculate_RT_logp(DT, V, A, Z):
    
    DT = np.where(DT <= 0, 0, DT)
    prob_rt_std = _diffusion_01std(DT,A,Z/A)
    
    prob_rt = (1 / (A*A)) * (np.exp( (-Z*A*V) - ((V*V) * DT)/2 )) * prob_rt_std
    prob_rt = np.log(prob_rt.sum()) 

    return prob_rt

def model(I, J, DT, X):
    m = pyro.sample("m", dist.Normal(0,1),sample_shape=(4,))
    s = pyro.sample("s", dist.Normal(0,0.2),sample_shape=(4,))

    v_pr = pyro.sample("v_pr", dist.Normal(0,1),sample_shape=(I,1)) # Drift Rate
    a_pr = pyro.sample("a_pr", dist.Normal(0,1),sample_shape=(I,1)) # Boundary
    z_pr = pyro.sample("z_pr", dist.Normal(0,1),sample_shape=(I,1)) # Bias
    t_er_pr = pyro.sample("t_er_pr", dist.Normal(0,1),sample_shape=(I,1)) # Non-Decision Time


    v = pyro.deterministic("v", m[0] + s[0] * v_pr)
    a = pyro.deterministic("a", np.exp(m[1] + s[1] * a_pr))
    z = pyro.deterministic("z", np.special.expit(m[2] + s[2] * z_pr))
    t_er = pyro.deterministic("t_er", np.special.expit(m[3] + s[3] * t_er_pr))

    
    V = np.where(X==1, -v, v)
    A = np.where(X==1, a, a)
    Z = np.where(X==1, 1-z, z)
    t_er = np.where(X==1, t_er, t_er)
    #print(DT-t_er)
    logp = _calculate_RT_logp(DT-t_er, V, A, Z)

    pyro.factor("obs", logp)

def sample_posterior_params(DT, X, num_warmup=100,samples_n=500,target_accept_prob=0.80):
    I,J = DT.shape
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=4)
    mcmc_chain.run(I=I, J=J, DT=DT, X=X)
    return mcmc_chain

#%%
if __name__ == "__main__":
    DT = np.linspace(0.1,2,1000)
    V = np.linspace(-2,2,1000)
    A = np.linspace(0.1,5,1000)
    Z = np.linspace(0,1,1000)

    print("*****************")

    prob_rt = _calculate_RT_logp(DT, V, A, Z)
    print(prob_rt)


    V = np.linspace(2,-2,1000)
    A = np.linspace(5,0.1,1000)
    Z = np.linspace(1,0,1000)

    print("*****************")

    prob_rt = _calculate_RT_logp(DT, V, A, Z)
    print(prob_rt)

    DT = np.linspace(10,-1,1000)
    V = np.linspace(2,-2,1000)
    A = np.linspace(5,0.1,1000)
    Z = np.linspace(1,0,1000)

    print("*****************")

    prob_rt = _calculate_RT_logp(DT, V, A, Z)
    print(prob_rt)

    DT_arr = []
    for i in range(0,50):
        DT = npy.linspace(i+1,0.1,100)
        #DT = jax.random.shuffle(jax.random.PRNGKey(0), DT)
        DT_arr.append(DT)

    DT = npy.array(DT_arr)
    X = npy.random.randint(0,2,(50,100))

    V = npy.linspace(npy.asarray([1]),-1,50)
    A = npy.linspace(npy.asarray([5]),0.1,50)
    Z = npy.linspace(npy.asarray([1]),0,50)


    #V = jax.random.shuffle(jax.random.PRNGKey(0), V)
    #A = jax.random.shuffle(jax.random.PRNGKey(0), A)
    #Z = jax.random.shuffle(jax.random.PRNGKey(0), Z)

    prob_rt = _calculate_RT_logp(DT, V, A, Z)
    print("*****************", prob_rt)
    #print(prob_rt)

    import pandas as pd
    #pd.DataFrame(dict(small = np.asarray(loop_small), 
    #                big = np.asarray(loop_big))).plot(ylabel="Steps")

    # %%

    mcmc = sample_posterior_params(DT, X)
    mcmc.print_summary()
# %%
