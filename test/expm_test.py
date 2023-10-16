import jax.numpy as npx
import jax.scipy.linalg as ln
import numpy as np
import pandas as pd
import numpyro as npy
import numpyro.distributions as dist
from jax import random

from numpyro.infer import MCMC, NUTS, SA
npy.set_host_device_count(4)

def model():
    rt = 1.67
    mu =  npy.sample(f"mu", dist.Normal(0,5))

    K = npx.asarray([[[[ 0. ,   0. ,   0.  ,  0. ,   0.  ,  0.   , 0.  ],
    [-0.25, -1. ,   1.25 , 0.  ,  0. ,   0.  ,  0.  ],
    [ 0. ,  -0.25, -1. ,   1.25,  0.  ,  0.  ,  0.  ],
    [ 0. ,   0. ,  -0.25, -1.  ,  1.25 , 0. ,   0.  ],
    [ 0. ,   0. ,   0. ,  -0.25, -1. ,   1.25 , 0.  ],
    [ 0. ,   0. ,   0. ,   0. ,  -0.25 ,-1.  ,  1.25],
    [ 0. ,   0. ,   0.  ,  0. ,   0.  ,  0.  ,  0.  ]]]])
    phi=ln.expm(mu*rt*K)
    npy.factor(f"likelihood", phi.sum())

rng_key = random.PRNGKey(0)
kernel = NUTS(model)
mcmc_chain = MCMC(kernel, num_warmup=10, num_samples=20, num_chains=4)
mcmc_chain.run(rng_key)
mcmc_chain.print_summary()
