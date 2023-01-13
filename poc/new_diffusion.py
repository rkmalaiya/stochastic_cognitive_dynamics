import numpy as at
import os

err = 10e-10
eps=0.1
max_k=30

def _get_count_l(t):
    K = at.sqrt((-2) * at.log(at.pi * t * err) / ((at.pi**2) * t) )
    K = at.where(K == 1/(at.pi * at.sqrt(t)) , K, (1 / (at.pi * at.sqrt(t)) ) ) # based on RWiener package in rlang
    return K

def _get_count_s(tt):
    K = at.sqrt( -2 * tt * at.log( 2 * err* at.sqrt( 2 * at.pi * tt ) ) )
    K = at.where(K == at.sqrt(tt)+1, K, at.sqrt(tt) + 1)
    return K

def _get_lambda(tt):
    
    return 2 + _get_count_s(tt) - _get_count_l(tt)

def _diffusion_01w_s(tt, a, w):

    K_m = _get_count_s(tt)
    K_n = at.max(at.floor(K_m))
    K_n = at.where(K_n > max_k, max_k, K_n)

    K=at.arange( -at.floor((K_n-1)/2), at.ceil((K_n-1)/2) + 1 )[:,at.newaxis, at.newaxis]

    prob_rt_std = (w + 2*K) * (at.exp( (w+2*K)**2/(2*tt) ))
    prob_rt_std = prob_rt_std * 1/(at.sqrt(2*at.pi*tt*tt*tt))
    return prob_rt_std.sum(axis=0)

def _diffusion_01w_l(tt, a, w):

    K_m = _get_count_l(tt)
    K_n = at.max(at.floor(K_m))
    K_n = at.where(K_n > max_k, max_k, K_n)

    K=at.arange(1,K_n+1)[:,at.newaxis, at.newaxis]


    prob_rt_std = K * (at.exp( - ((K*at.pi)**2 * tt/2) )) * at.sin( K * at.pi * w ) #exp becomes zero for large tt, hence adding eps
    return prob_rt_std.sum(axis=0)

def _diffusion_01w(t,a,w):

    tt = t/(a**2)
    tt= at.where(tt <= eps, eps, tt)
    
    prob_rt_std = at.where(_get_lambda(tt) == 0, _diffusion_01w_s(tt, a, w), _diffusion_01w_l(tt, a, w))
 
    
    prob_rt_final = at.pi * prob_rt_std.sum(axis=0) #at.switch(at.le(prob_rt_std,0),0,prob_rt_std) #at.switch(at.le(t,0),0, prob_rt_std )

    return prob_rt_final #should return a scaler



def _diffusion_X_logp(X, v, a, z):
    w = z#/a
    prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
    return prob_X

def _RT_logp(RT, obs_X, v, a, z, t_er):
    
    #X = at.as_tensor(X)

    V = at.where(obs_X == 1, -v[:,[0]], v[:,[1]])
    A = at.where(obs_X==1, a[:,[0]], a[:,[1]])
    Z = at.where(obs_X==1, 1-z[:,[0]], z[:,[1]])
    T_er = at.where(obs_X==1, t_er[:,[0]], t_er[:,[1]])
    
    W = Z #z/a  

    DT = RT-T_er
    
    prob_rt_std = at.where(DT == 0,0, _diffusion_01w(DT,A,W))
    
    
    prob_rt = (1 / A*A) * at.exp( (-W*A*V) - (V*V * DT)/2 ) * prob_rt_std
    
    prob_rt = at.where(DT==0,0,prob_rt) #Removing pdfs for t <= 0 because t <=0 is not supported

   
    return at.log(prob_rt)

def _diffusion_RT_logp(RT, obs_X, v, a, z, t_er):    

    prob_rt = _RT_logp(RT, obs_X, v, a, z, t_er)

    total_logp = prob_rt.sum(axis=1) 

    return total_logp 

if __name__ == "__main__":
    #v = at.asarray([[-4.7455195,  3.5246989, -4.7455195,  3.5246989, -4.7455195]])
    #a = at.asarray([[0.10100832, 0.48082409, 0.10100832, 0.48082409, 0.10100832]])
    #z = at.asarray([[0.03027856, 0.33936817, 0.03027856, 0.33936817, 0.03027856]])
    #t_er = at.asarray([[2.17066536, 0.17037338, 2.17066536, 0.17037338, 2.17066536]])

    #RT = at.asarray([[0.36327581, 0.77126385, 1.02809235, 3.05141406, 0.37536871]])
    #X = at.asarray([[1, 0, 1, 0, 1]])
    print(os.getcwd())
    v = at.load("test/v.npy")
    a = at.load("test/a.npy")
    z = at.load("test/z.npy")
    t_er = at.load("test/t_er.npy")
    X = at.load("test/X.npy")
    RT = at.load("test/RT.npy")
    print("***********",RT.shape)
    
    rt_logp1 = _RT_logp(RT, X, v, a, z, t_er)
    print(rt_logp1)
    print("***************")
    rt_logp2 = _diffusion_RT_logp(RT, X, v, a, z, t_er)
    print(rt_logp2)

