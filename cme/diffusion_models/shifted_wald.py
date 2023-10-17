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

def _random(*dist_params, rng=None, size=(100,50)):
  a, v, t = dist_params
  
  mu = a/v
  lam = a**2 

  return rng.wald(mu, lam, size=size) + t


def _priors(I, model_mc):
  with model_mc:
    #a = pm.Normal("a", 0.5,2,shape=(I,1))
    #v = pm.LogNormal("v", 0,1, shape=(I,1))
    #t = pm.HalfNormal("t", 0.1, shape=(I,1))

    a = pm.Normal("a", 0.5, 2, shape=(I,1), dims="partID")
    v = pm.LogNormal("v", 0, 2, shape=(I,1), dims="partID") #0,1
    t = pm.HalfNormal("t", 0.1, shape=(I,1), dims="partID")

  return a, v, t

def _priors_multi(I,model_mc):
  with model_mc:
    a_m = pm.Uniform("a_m", 0.1, 2)
    a_s = pm.Uniform("a_s", 0.1, 1)
    
    t_m = pm.Uniform("t_m", 0, 0.5)
    t_s = pm.Uniform("t_s", 0.01, 0.1)
    
    v_m = pm.Uniform("v_m", 0, 2)
    v_s = pm.Uniform("v_s", 0.1, 1)

    a = pm.LogNormal("a", a_m, a_s,shape=(I,1), dims="partID")
    v = pm.LogNormal("v", v_m, v_s,shape=(I,1), dims="partID")
    t = pm.LogNormal("t", t_m, t_s,shape=(I,1), dims="partID")
    
  return a, v, t
  
  
def model(r=None, I=None, coords=None):
  
  if coords is None:
    model_mc = pm.Model()
  else:
    model_mc = pm.Model(coords={"partID": coords})

  a, v, t = _priors(I, model_mc)

  with model_mc:
    pm.CustomDist(
        'RT',
        a, v, t,
        logp=_liklihood,
        random=_random,
        observed=r,
    )
  return model_mc
  
  
def fit_posterior_distribution(r):
  with model(r, r.shape[0]):
    mean_field = pm.fit(n=100000,
                        method="advi", 
                        callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")])
  return mean_field

def sample_prior_distribution(r = None, samples=None):
  with model(r, I = 1 if r is None else r.shape[0]):
    mcmc_prior = pm.sample_prior_predictive(samples=samples)
  return mcmc_prior

def sample_posterior_distribution(r, coords=None, tune=1500, samples = 2500):
  model_mc = model(r, r.shape[0], coords)
  with model_mc:
    mcmc = pm.sample(tune=tune, draws= samples, 
                     nuts_sampler="pymc",
                     nuts_sampler_kwargs={"target_accept":0.9}
                     ) #step=[pm.Metropolis]
    
  return mcmc, model_mc

def sample_posterior_predictive(model_mc, mcmc):
  with model_mc:
    mcmc = pm.sample_posterior_predictive(mcmc, extend_inferencedata=True)
  return mcmc

#%%
if __name__ == "__main__":

  mcmc_prior = sample_prior_distribution(samples=200)
  mcmc_prior = sample_prior_distribution(np.ones((258,128)), samples=200)
  

  #likl_arr = []
  #r, a, v, t = np.ones((258,128)), np.ones((258,128)), np.ones((258,128)), np.zeros((258,128))
  #likl_arr = _liklihood(r, a, v, t).eval()
  #pd.Series(np.asarray(likl_arr)).plot.kde()
  # %%
