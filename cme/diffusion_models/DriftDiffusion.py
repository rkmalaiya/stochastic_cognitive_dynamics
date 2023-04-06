#%%
import pymc as pm
import numpy as np
import pytensor as ae
from pytensor import tensor as at
import scipy as sp
from scipy.stats.sampling import SimpleRatioUniforms
import cme.utils.common_logging as cl
from cme.utils import common_utils as ut
import pandas as pd
import os
import pymc.sampling.jax as jx
from pytensor.tensor.var import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from typing import List, Tuple
from pymc.distributions.continuous import PositiveContinuous

class DriftDiffusionRV(RandomVariable):
    name: str = "Drift Diffusion RV"
    ndim_supp: int = 0
    ndims_params: List[int] = [0, 0, 0, 0]
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("Drift Diffusion RV", "\\operatorname{Drift Diffusion RV}")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        v: np.ndarray,
        a: np.ndarray,
        z: np.ndarray,
        t_er: np.ndarray,
        obs_X: np.ndarray,
        size: Tuple[int, ...],
    ) -> np.ndarray:
        v=at.as_tensor_variable(v)
        a=at.as_tensor_variable(a)
        z=at.as_tensor_variable(z)
        t_er=at.as_tensor_variable(t_er)
        obs_X=at.as_tensor_variable(obs_X)
        
        dd_logp = DriftDiffusionProb(v,a,z,t_er,obs_X)
        #ddm_gen = SimpleRatioUniforms(dd_logp, random_state=rng)
        #return ddm_gen.rvs(size)
        smpl = dd_logp._draw_RT(rng=rng, size=size)

class DriftDiffusion(PositiveContinuous):
    rv_op = DriftDiffusionRV()    

    @classmethod
    def dist(cls, v, a, z, t_er, obs_X, **kwargs):
        return super().dist([v, a, z, t_er, obs_X], **kwargs)

    def logp(RT, v,a,z,t_er):
        return DriftDiffusionProb(v,a,z,t_er).pdf(RT)


class DriftDiffusionProb():
    log = cl.get_logger("diffusion")
    eps = 0.1 # for numerical stability
    err = 10e-10
    max_k = 30
    

    def __init__(self,v, a, z, t_er, obs_X):
        self.v = v
        self.a = a
        self.z = z
        self.t_er = t_er
        self.obs_X = obs_X
    

    def pdf(self, RT):
        return self._diffusion_RT_p(RT).eval()
    
    def dpdf(self, RT):
        RT = at.as_tensor(RT)
        gd = at.grad(self._diffusion_RT_logp(RT), RT)
        gd_val = gd.eval() # [g.eval() for g in gd]
        return gd_val

    def _diffusion_RT_p(self, RT):
        prob_rt = self._RT_logp(RT, self.obs_X, self.v, self.a, self.z, self.t_er)
        total_p = prob_rt.sum()#axis=1) 
        return total_p

    def _diffusion_RT_logp(self, RT):
        total_p = self._diffusion_RT_p(RT)
        total_logp = at.log(total_p)

        x_printed_12 = ae.printing.Print('***per individual final sum logp')(total_logp)
        return total_logp

    
    def _get_count_l(self,tt):

        K = at.sqrt((-2) * at.log(np.pi * tt * self.err) / ((np.pi**2) * tt) )
        K = at.switch(at.gt(K, 1/(np.pi * at.sqrt(tt)) ), K, (1 / (np.pi * at.sqrt(tt)) ) ) # based on RWiener package in rlang
        return K

    def _get_count_s(self,tt):
        K = at.sqrt( -2 * tt * at.log( 2 * self.err* at.sqrt( 2 * np.pi * tt ) ) )
        K = at.switch(at.gt(K, at.sqrt(tt)+1), K, at.sqrt(tt) + 1)
        return K

    def _get_lambda(self,tt):
        
        return 2 + self._get_count_s(tt) - self._get_count_l(tt)

    def _diffusion_01w_s(self,tt, a, w):

        K_m = self._get_count_s(tt)
        K_n = at.max(at.floor(K_m))
        K_n = at.switch(at.gt(K_n,self.max_k), self.max_k, K_n)

        x_printed_8 = ae.printing.Print('s K_n')(K_n)

        K=at.arange( -at.floor((K_n-1)/2), at.ceil((K_n-1)/2) + 1 )[:,np.newaxis, np.newaxis]
        #K=at.arange( -10, 10 )[:,np.newaxis, np.newaxis]
        x_printed_8 = ae.printing.Print('s K_m')(K_m)
        x_printed_7 = ae.printing.Print('s tt')(tt )

        prob_rt_std = (w + 2*K) * (at.exp( - (w+2*K)*(w+2*K)/(2*tt) ))
        prob_rt_std = prob_rt_std * 1/(at.sqrt(2*np.pi*tt*tt*tt))
        x_printed_8 = ae.printing.Print('s prob_rt_std for each Ks')(prob_rt_std )
        return prob_rt_std.sum(axis=0)

    def _diffusion_01w_l(self,tt, a, w):

        K_m = self._get_count_l(tt)
        K_n = at.max(at.floor(K_m)) + 1
        K_n = at.switch(at.gt(K_n,self.max_k), self.max_k, K_n)

        x_printed_8 = ae.printing.Print('l K')(K_n)

        K=at.arange(1,K_n+1)[:,np.newaxis, np.newaxis]
        #K=at.arange(1,10)[:,np.newaxis, np.newaxis]
        x_printed_8 = ae.printing.Print('l K_m')(K_m)
        x_printed_7 = ae.printing.Print('l tt')(tt )


        prob_rt_std = K * (at.exp( - ((K*np.pi)**2 * tt/2) )) * at.sin( K * np.pi * w ) #exp becomes zero for large tt, hence adding eps
        x_printed_8 = ae.printing.Print('l prob_rt_std for each Ks')(prob_rt_std )
        return prob_rt_std.sum(axis=0)

    def _diffusion_01w(self,t,a,w):

        tt = t/(a**2)
        #tt = at.as_tensor(1.5/0.01**2)
        #tt= at.switch(tt <= eps, eps, tt)
        
        #prob_rt_std = _diffusion_01w_l(tt, a, w)
        prob_rt_std = at.switch(at.lt(self._get_lambda(tt), 0), self._diffusion_01w_s(tt, a, w), self._diffusion_01w_l(tt, a, w))

        x_printed_5 = ae.printing.Print('w')(w )
    
        
        prob_rt_final = np.pi * prob_rt_std.sum(axis=0) #at.switch(at.le(prob_rt_std,0),0,prob_rt_std) #at.switch(at.le(t,0),0, prob_rt_std )
        x_printed_9 = ae.printing.Print('prob_rt_final summed over all Ks')(prob_rt_final )

        return prob_rt_final #should return a scaler



    def _diffusion_X_logp(self,X, v, a, z):
        w = z#/a
        prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
        return prob_X

    def _RT_logp(self,RT, obs_X, V, A, Z, T_er):
        
        #X = at.as_tensor(X)

        #V = at.switch(at.eq(obs_X,1), -v[:,[0]], v[:,[1]])
        #V = at.switch(at.eq(obs_X,1), v[:,[0]], v[:,[1]])
        #A = at.switch(at.eq(obs_X,1), a[:,[0]], a[:,[1]])
        #Z = at.switch(at.eq(obs_X,1), 1-z[:,[0]], z[:,[1]])
        #Z = at.switch(at.eq(obs_X,1), z[:,[0]], z[:,[1]])
        #T_er = at.switch(at.eq(obs_X,1), t_er[:,[0]], t_er[:,[1]])
        
        x_printed_12 = ae.printing.Print('v')(V)
        x_printed_14 = ae.printing.Print('a')(A)
        x_printed_15 = ae.printing.Print('z')(Z)
        x_printed_16 = ae.printing.Print('t_er')(T_er)


        W = Z #z/a  
        #w = at.switch(at.ge(w,1), 0.99,w) # to avoid instability during intial evaluation.

        DT = at.switch(at.gt(RT, T_er), RT-T_er, self.eps)
        #DT = 
        x_printed_2 = ae.printing.Print('RT-t_re')(DT)

        #prob_rt = _diffusion(t, v, w, a)
        
        prob_rt_std = at.switch(at.le(DT,0),0, self._diffusion_01w(DT,A,W))
        
        x_printed_3 = ae.printing.Print(f'prob_rt_std all {obs_X.shape}')(prob_rt_std)

        #prob_rt = (1 / a**2) * at.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std
        prob_rt = (1 / A*A) * at.exp( (-W*A*V) - (V*V * DT)/2 ) * prob_rt_std
        
        
        #prob_X = _diffusion_X_logp(obs_X, V, A, Z)
        #all though care has been taken to not process pdf for -ve time that results in -ve pdf, some -ve pdf are still creeping up
        prob_rt = at.switch(at.le(DT,0),0,prob_rt) #Removing pdfs for t <= 0 because t <=0 is not supported

        x_printed_13 = ae.printing.Print('per individual all trial logp')(prob_rt)

        return prob_rt

    def _draw_RT(self, rng, size):
        obs_X, v, a, z, t_er = self.obs_X, self.v, self.a, self.z, self.t_er
        I, J = obs_X.shape
        max_iter = 10000
        
        #RT_rand = rng.standard_normal(size)
        RT_rvs = np.empty(shape=(I,J))
        samples_rejection = J #100 * J

        for i_l in range(I):
            RT_arr = np.empty(shape=0)
        
            iter=0
            while (RT_arr.shape[0] < J):
                
                RT_rand = sp.stats.lognorm.rvs(1, 0,1, size=samples_rejection)
                u = sp.stats.uniform.rvs(0,1,size=samples_rejection)

                pdf_lognorm = sp.stats.lognorm.pdf(RT_rand,1,0,1)
                pdf_diffusion = self._RT_logp(RT_rand,obs_X[[i_l],:],v[[i_l],:],a[[i_l],:],z[[i_l],:],t_er[[i_l],:]).eval()
                
                M = np.round(np.max(pdf_diffusion) + 1)
                

                #pdf_X = _diffusion_X_logp(obs_X,v,a,z).eval()
                #print("*****I_L",i_l)
                #print("*****M",M)
                #print("*****pdf_RT",pdf_diffusion)
                RT_rand_accept = RT_rand[np.where(u < pdf_diffusion[0,:] / (M*pdf_lognorm))]
                
                
                RT_arr = np.append(RT_arr, RT_rand_accept)
                
                iter += 1
                if iter > max_iter:
                    raise Exception("Could not find samples")

            RT_rvs[i_l,:] = RT_arr[0:J] 
        print("*****RT", RT_rvs.shape)

        return RT_rvs


 
#%%
if __name__ == "__main__":
    #rv = DriftDiffusionRV()
    #rv(0.1,0.2,1,0.01,[1],size=(1,1)).eval()
    v = at.as_tensor_variable(0.1)
    a = at.as_tensor_variable(0.4)
    z = at.as_tensor_variable(0.05)
    t_er = at.as_tensor_variable(0.01)
    obs_X = at.as_tensor_variable([1])
    RT = at.as_tensor_variable([1.2])
    
    p = DriftDiffusionProb(v,a,z,t_er,obs_X).pdf(RT)
    dp = DriftDiffusionProb(v,a,z,t_er,obs_X).dpdf(RT)
    print(p)
    print(dp)

    rv = DriftDiffusionRV()
    s = rv.rng_fn(np.random.default_rng(),0.1,0.2,1,0.01,[1],size=(1,1))
    print(s)
# %%

