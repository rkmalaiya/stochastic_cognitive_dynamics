#%%
import pytensor.tensor as jnp
import pymc as pm
import pandas as pd
import numpy as np

def liklihood(r, a, v, t):
  
  #r=r-t

  f1 = a / jnp.sqrt(2 * np.pi * (r)**3 )
  f2 = ((a - v*(r))**2) / 2*(r)
  likl = f1 * jnp.exp(-f2)
  
  likl = jnp.where(jnp.isinf(likl), 0, likl)

  likl = jnp.log(likl.sum())

  likl = jnp.where(jnp.isinf(likl), 0, likl)

  return likl


def priors():
  with pm.Model() as model_mc:
    a_m = pm.Uniform("a_m", 0.67,2.35)
    a_s = pm.Uniform("a_s", 0,0.485)
    
    t_m = pm.Uniform("t_m", 0, 0.82)
    t_s = pm.Uniform("t_s", 0, 0.237)
    
    v_m = pm.Uniform("v_m", 0.85,7.43)
    v_s = pm.Uniform("v_s", 0,1.899)
    
    
    a = pm.Normal("a", a_m, a_s**2)
    v = pm.Lognormal("v", v_m, v_s**2)
    t = pm.Lognormal("t", t_m, t_s**2)
    
  return a, v, t, model_mc
  
  
def model(r):
  
  a, v, t, model_mc = priors()
  with model_mc:
    pm.CustomDist(
        'likl',
        a, v, t,
        logp=liklihood,
        observed=r,
    )
  return model_mc
  
  
def sample_posterior_distribution(r):
  with model(r):
    mcmc = pm.sample(step=[pm.Metropolis])
    
  return mcmc

# read the data

df = pd.read_csv("ES_Neu_Trials.csv").drop("Unnamed: 0", axis=1)
item_count = df.groupby("subjID").count().iloc[0,1]
df = df.assign(item_id = np.tile(np.arange(0,item_count), 258)) 

r = (df.drop("choice", axis=1)
    .pivot(columns="item_id",index="subjID", values="RT")
    .reset_index().drop(columns="subjID").to_numpy())
#r.shape

#%%
liklihood(1,1,1,1).eval()

#%%
mcmc = sample_posterior_distribution(r)
mcmc.summary()
  
  
  
  



# %%
