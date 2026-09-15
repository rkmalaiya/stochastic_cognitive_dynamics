#%%
import jax
import jax.numpy as npx
import numpyro as pyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA, HMCECS, Predictive
from jax import random

#%%

a = npx.zeros((4,3,2,2))
b = npx.ones((4,3))

def fun(a1, b1):
    print("***",a1.shape)
    print("$$$",b1.shape)
    return npx.linalg.matrix_power(a1, b1)

#fun(a[...,0,0], b)
# %%

vfun1 = jax.vmap(fun, in_axes=(0,0))
vfun2 = jax.vmap(vfun1, in_axes=(0,0))


# %%
vfun2(a,b)
# %%


pyro.set_host_device_count(4)
I, J = 5, 3

def model(I,J):
    with pyro.plate('I', size=I, dim=-2):
            with pyro.plate('J', size=J, dim=-1):
                RT_pred = pyro.sample(f"RT_pred", dist.LogNormal(0, 1.5))

kernel = NUTS(model)
predictive_mcmc = MCMC(kernel, num_warmup=500, num_samples=10, num_chains=4)
predictive_mcmc.run(random.PRNGKey(0), I,J)
# %%
