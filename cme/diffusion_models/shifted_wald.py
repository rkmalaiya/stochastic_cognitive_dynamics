#%%
import matplotlib.pyplot as plt

import pytensor.tensor as jnp
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from pymc.variational.callbacks import CheckParametersConvergence

def _liklihood(r, a, v, t):
  
  r=r-t

  f1 = a / jnp.sqrt(2 * np.pi * (r)**3 )
  f2 = ((a - v*(r))**2) / 2*(r)
  likl = f1 * jnp.exp(-f2)
  
  likl = jnp.where(jnp.isinf(likl), 0, likl)

  likl = jnp.log(likl.sum())

  likl = jnp.where(jnp.isinf(likl), 0, likl)
  likl = jnp.where(jnp.isnan(likl), 0, likl)

  return likl

def _priors(I, model_mc):
  with model_mc:
    #a = pm.Normal("a", 0.5,2,shape=(I,1))
    #v = pm.LogNormal("v", 0,1, shape=(I,1))
    #t = pm.HalfNormal("t", 0.1, shape=(I,1))

    a = pm.Normal("a", 0.5, 2,shape=(I,1))
    v = pm.LogNormal("v", 0, 2, shape=(I,1)) #0,1
    t = pm.HalfNormal("t", 0.1, shape=(I,1))

  return a, v, t

def _priors_multi(I,model_mc):
  with model_mc:
    a_m = pm.Uniform("a_m", 0.1, 3)
    a_s = pm.Uniform("a_s", 1, 3)
    
    #t_m = pm.Uniform("t_m", 0, 0.82)
    t_s = pm.Uniform("t_s", 0.1, 0.5)
    
    v_m = pm.Uniform("v_m", 0, 1)
    v_s = pm.Uniform("v_s", 0.1, 1)

    a = pm.Normal("a", a_m, a_s,shape=(I,1), dims="partID")
    v = pm.LogNormal("v", v_m, v_s,shape=(I,1), dims="partID")
    t = pm.HalfNormal("t", t_s,shape=(I,1), dims="partID")
    
  return a, v, t
  
  
def model(r, I, coords=None):
  
  if coords is None:
    model_mc = pm.Model()
  else:
    model_mc = pm.Model(coords={"partID": coords})

  a, v, t = _priors_multi(I, model_mc)

  with model_mc:
    pm.CustomDist(
        'likl',
        a, v, t,
        logp=_liklihood,
        observed=r,
    )
  return model_mc
  
  
def fit_posterior_distribution(r):
  with model(r, r.shape[0]):
    mean_field = pm.fit(n=100000,
                        method="advi", 
                        callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")])
  return mean_field

def sample_prior_distribution(r):
  with model(r, 100,10):
    mcmc_prior = pm.sample_prior_predictive()
  return mcmc_prior

def sample_posterior_distribution(r, coords=None, tune=1500, samples = 2500):
  with model(r, r.shape[0], coords):
    mcmc = pm.sample(tune=tune, draws= samples, 
                     nuts_sampler="pymc",
                     nuts_sampler_kwargs={"target_accept":0.9}
                     ) #step=[pm.Metropolis]
    
  return mcmc

#%%
if __name__ == "__main__":

  likl_arr = []
  for r, a, v, t in zip(np.ones((258,128)), np.ones((258,128)), np.ones((258,128)), np.zeros((258,128))):
    likl_t = _liklihood(r, a, v, t)
    likl_arr.append(likl_t.eval())
  pd.Series(np.asarray(likl_arr)).plot.kde()
  # %%
