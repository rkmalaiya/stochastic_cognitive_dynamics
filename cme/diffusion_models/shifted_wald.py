#%%
import jax.numpy as jnp
import numpyro as py
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random
import pandas as pd
import numpy as np

def liklihood(r, a, v, t):
  
  f1 = a / jnp.sqrt(2 * jnp.pi * (r-t)**3 )
  
  f2 = ((a - v*(r-t))**2) / 2*(r-t)
  
  
  
  likl = f1 * jnp.exp(-f2)
  
  return jnp.log(likl).sum()


def priors():
  
  a_m = py.sample("a_m", dist.Uniform(0.67,2.35))
  a_s = py.sample("a_s", dist.Uniform(0,0.485))
  
  t_m = py.sample("t_m", dist.Uniform(0,0.82))
  t_s = py.sample("t_s", dist.Uniform(0,0.237))
  
  v_m = py.sample("v_m", dist.Uniform(0.85,7.43))
  v_s = py.sample("v_s", dist.Uniform(0,1.899))
  
  
  a = py.sample("a", dist.Normal(a_m, a_s**2))
  v = py.sample("v", dist.Normal(v_m, v_s**2))
  t = py.sample("t", dist.Normal(t_m, t_s**2))
  
  return a, v, t
  
  
def model(r):
  
  a, v, t = priors()
  py.factor("likl", likl(r, a, v, t))
  
def sample_posterior_distribution(model, r):
  rng_key = random.PRNGKey(0)
  rng_key, rng_key_ = random.split(rng_key)
  
  # Run NUTS.
  kernel = NUTS(model)
  num_samples = 200
  
  mcmc = MCMC(kernel, num_warmup=100, num_samples=num_samples)
  
  mcmc.run(
      rng_key_, r)
    
  return mcmc

# read the data

df = pd.read_csv("ES_Neu_Trials.csv").drop("Unnamed: 0", axis=1)
item_count = df.groupby("subjID").count().iloc[0,1]
df = df.assign(item_id = np.tile(np.arange(0,item_count), 258)) 

r = (df.drop("choice", axis=1)
    .pivot(columns="item_id",index="subjID", values="RT")
    .reset_index().drop(columns="subjID").to_numpy())
#r.shape
liklihood(1,1,1,1)
mcmc = sample_posterior_distribution(model, r)
mcmc.print_summary()
  
  
  
  


