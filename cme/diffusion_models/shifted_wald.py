#%%
import matplotlib.pyplot as plt

import pytensor.tensor as jnp
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from pymc.variational.callbacks import CheckParametersConvergence
from pytensor.tensor import TensorVariable
import pytensor.tensor as at

delta_t = 1
def _liklihood(r, a, v, t, x):
  
  r=r-(t*delta_t)

  f1 = a / jnp.sqrt(2 * np.pi * (r)**3 )
  f2 = ((a - v*(r))**2) / 2*(r)
  #likl = f1 * jnp.exp(-f2)
  likl = jnp.log(f1) - f2
  
  likl = jnp.where(jnp.isinf(likl), 0, likl)

  #likl = jnp.log(likl.sum())
  likl = likl.sum()

  likl = jnp.where(jnp.isinf(likl), 0, likl)
  likl = jnp.where(jnp.isnan(likl), 0, likl)
  likl = jnp.where(at.eq(x,0),0,likl)

  return likl

#def _random(a: TensorVariable,
#     v: TensorVariable,
#     t: TensorVariable, size: TensorVariable):

def _random(*dist_params, rng=None, size=None): 
  
  a, v, t, _ = dist_params

  mu = a/v + t*delta_t
  lam = a**2 

  return pm.Wald.dist(mu=mu, lam=lam,shape=size).eval()
  #return pm.Normal.dist(mu=(a/v) + (t*delta_t), sigma=jnp.sqrt(a/(v**3)),shape=size).eval()


def _priors(I, model_mc):
  with model_mc:
    #a = pm.Normal("a", 0.5,2,shape=(I,1))
    #v = pm.LogNormal("v", 0,1, shape=(I,1))
    #t = pm.HalfNormal("t", 0.1, shape=(I,1))

    #a = pm.LogNormal("a", 0, 1, shape=(I,1), dims="partID")
    #v = pm.LogNormal("v", 0, 0.5, shape=(I,1), dims="partID") #0,1
    #a = pm.HalfNormal("a", 1, shape=(I,1), dims="partID")
    #v = pm.HalfNormal("v", 0.5, shape=(I,1), dims="partID") #0,1
    a = pm.Gamma("a", mu=1, sigma=0.5, shape=(I,1), dims="partID")
    v = pm.Gamma("v", mu=1, sigma=0.5, shape=(I,1), dims="partID")
    t = pm.HalfNormal("t", 0.5, shape=(I,1), dims="partID")

  return a, v, t

def _priors_multi(I,model_mc):
  with model_mc:
    a_m = pm.Uniform("a_m", 0.5, 1.5)
    #a_s = pm.Uniform("a_s", 0.1, 0.5)
    
    #t_m = pm.Uniform("t_m", 0, 0.5)
    #t_s = pm.Uniform("t_s", 0.1, 0.9)
    
    v_m = pm.Uniform("v_m", 0.5, 1.5)
    #v_s = pm.Uniform("v_s", 0.1, 0.5)

    a = pm.Gamma("a", mu=a_m, sigma=0.1, shape=(I,1), dims="partID")
    v = pm.Gamma("v", mu=v_m, sigma=0.1, shape=(I,1), dims="partID")
    
    #a = pm.HalfNormal("a", a_s, shape=(I,1), dims="partID")
    #v = pm.HalfNormal("v", v_s, shape=(I,1), dims="partID")
    t = pm.HalfNormal("t", 0.5, shape=(I,1), dims="partID")
    
  return a, v, t

def _priors_noncentral(I,model_mc):
  with model_mc:
    a_m = pm.Uniform("a_m", 0.5, 1.5)
    a_s = 0.2**2 #pm.Uniform("a_s", 0.1, 0.5)
    
    #t_m = pm.Uniform("t_m", 0.05, 0.2)
    #t_s = pm.Uniform("t_s", 0.5, 1)
    
    v_m = pm.Uniform("v_m", 1.5, 2.5)
    v_s = 0.2**2 #pm.Uniform("v_s", 0.1, 0.5)

    #a_n = pm.Gamma("a_n",2,2, shape=(I,1), dims="partID")
    #v_n = pm.Gamma("v_n",2,2, shape=(I,1), dims="partID")
    #t_n = pm.Beta("t_n",2,5, shape=(I,1), dims="partID")

    a_n = pm.Normal("a_n", shape=(I,1), dims="partID")
    v_n = pm.Normal("v_n", shape=(I,1), dims="partID")

    a = pm.Deterministic("a", a_m + a_s*(a_n**2), dims="partID")
    v = pm.Deterministic("v", v_m + v_s*(v_n**2), dims="partID")
    #t = pm.Deterministic("t", t_m + t_s*t_n, dims="partID")

    #a = pm.Gamma("a", mu=a_m, sigma=a_s, shape=(I,1), dims="partID")
    #v = pm.Gamma("v", mu=v_m, sigma=v_s, shape=(I,1), dims="partID")
    
    #a = pm.HalfNormal("a", a_s, shape=(I,1), dims="partID")
    #v = pm.HalfNormal("v", v_s, shape=(I,1), dims="partID")
    t = pm.HalfNormal("t", 0.1, shape=(I,1), dims="partID")
    
  return a, v, t
  
  
def model(r=None, x = None, I=None, coords=None):
  
  if coords is None:
    model_mc = pm.Model()
  else:
    model_mc = pm.Model(coords={"partID": coords})

  a, v, t = _priors_noncentral(I, model_mc)
  x = at.as_tensor(x)
  with model_mc:
    pm.CustomDist(
        'RT',
        a, v, t, x,
        logp=_liklihood,
        random=_random,
        observed=r * delta_t,
    )
  return model_mc
  
  
def fit_posterior_distribution(r,x):
  with model(r, x, r.shape[0]):
    mean_field = pm.fit(n=100000,
                        method="advi", 
                        callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")])
  return mean_field

def sample_prior_distribution(r = None, x=None, samples=None):
  with model(r, x, I = 1 if r is None else r.shape[0]):
    mcmc_prior = pm.sample_prior_predictive(samples=samples)
  return mcmc_prior

def sample_posterior_distribution(r, x=None, coords=None, tune=1500, samples = 2500):
  model_mc = model(r, x, r.shape[0], coords)
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

  mcmc_prior = sample_prior_distribution(np.random.uniform(0.1,2,size=(258,128)), np.ones((258,128)), samples=20)
  print("test 1 ")
  mcmc, model_mc = sample_posterior_distribution(np.random.uniform(0.1,2,size=(258,128)), np.ones((258,128)), tune=10, samples=20)
  print("test 2")
  mcmc = sample_posterior_predictive(model_mc, mcmc)
  print("test 3")
  

  #likl_arr = []
  #r, a, v, t = np.ones((258,128)), np.ones((258,128)), np.ones((258,128)), np.zeros((258,128))
  #likl_arr = _liklihood(r, a, v, t).eval()
  #pd.Series(np.asarray(likl_arr)).plot.kde()
  # %%
