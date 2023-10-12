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


def _priors(I,J):
  with pm.Model() as model_mc:
    #a_m = pm.Uniform("a_m", 0, 2)
    #a_s = pm.Uniform("a_s", 0, 1)
    
    #t_m = pm.Uniform("t_m", 0, 0.82)
    #t_s = pm.Uniform("t_s", 0, 0.5)
    
    #v_m = pm.Uniform("v_m", 1, 5)
    #v_s = pm.Uniform("v_s", 0, 1)
    
    
    a = pm.Normal("a", 0.5,2,shape=(I,1))
    v = pm.Lognormal("v", 1,1, shape=(I,1))
    t = pm.HalfNormal("t", 0.1, shape=(I,1))

    #a = pm.Normal("a", a_m, a_s**2,shape=(I,1))
    #v = pm.Normal("v", v_m, v_s**2,shape=(I,1))
    #t = pm.HalfNormal("t", t_s**2,shape=(I,1))
    
  return a, v, t, model_mc
  
  
def model(r,I,J):
  
  a, v, t, model_mc = _priors(I,J)
  with model_mc:
    pm.CustomDist(
        'likl',
        a, v, t,
        logp=_liklihood,
        observed=r,
    )
  return model_mc
  
  
def fit_posterior_distribution(r):
  with model(r, *r.shape):
    mean_field = pm.fit(n=100000,
                        method="advi", 
                        callbacks=[pm.callbacks.CheckParametersConvergence(diff="absolute")])
  return mean_field

def sample_prior_distribution(r):
  with model(r, 100,10):
    mcmc_prior = pm.sample_prior_predictive()
  return mcmc_prior

# read the data

df = pd.read_csv("ES_Neu_Trials.csv").drop("Unnamed: 0", axis=1)
item_count = df.groupby("subjID").count().iloc[0,1]
df = df.assign(item_id = np.tile(np.arange(0,item_count), 258)) 

r = (df.drop("choice", axis=1)
    .pivot(columns="item_id",index="subjID", values="RT")
    .reset_index().drop(columns="subjID").to_numpy())
#r.shape

#%%
#_liklihood(1,1,1,1).eval()
_liklihood(1,0,1,1).eval()

#%%
mean_field = fit_posterior_distribution(r[0:100,:])
approx_sample = mean_field.sample(1000)

#%%
plt.plot(mean_field.hist);

#%%

def sample_posterior_distribution(r):
  with model(r, *r.shape):
    mcmc = pm.sample(tune=1000, draws= 1500, 
                     nuts_sampler="pymc",
                     nuts_sampler_kwargs={"target_accept":85}
                     ) #step=[pm.Metropolis]
    
  return mcmc

mcmc = sample_posterior_distribution(r)
summ = az.summary(mcmc)
summ.to_csv("neu_summ.csv")
print(summ)
mcmc.to_netcdf("neu_posterior.cdf")
mcmc.to_dataframe().to_csv("neu_posterior.csv")

az.plot_trace(mcmc)

#%%
  
#mcmc_prior = sample_prior_distribution(r)



# %%

likl_arr = []
for r, a, v, t in zip(r, np.ones((258,128)), np.ones((258,128)), np.zeros((258,128))):
  likl_t = _liklihood(r, a, v, t)
  likl_arr.append(likl_t.eval())
print(likl_arr)
pd.Series(np.asarray(likl_arr)).plot.kde()
# %%
