"""
Inference for the confidence accumulation model.

`cme.decision_models.confidence_accumulation` holds the model itself: it
generates response data and evaluates the likelihood, and knows nothing about
how parameters get estimated. Everything that fits that model lives here.

    priors          numpyro prior blocks (drift, diffusion, initial state)
    numpyro_model   the numpyro model wiring priors to the likelihood
    mcmc            NUTS posterior sampling
    vi              SVI / variational posterior sampling
    predictive      prior and posterior predictive sampling

Importing this package applies the numpyro global configuration below, so it
takes effect for anything that fits the model and for nothing that only
simulates from it.
"""

import numpyro as pyro
from numpyro import enable_validation

# Validation adds distribution argument checks into every model trace. Useful while
# developing the model, pure overhead once it is fixed. Flip back to True if a new
# prior or distribution misbehaves.
# Original always-on validation retained for reference:
# enable_validation(True)
enable_validation(False)

#pyro.set_platform("cpu")
# One XLA CPU device per MCMC chain. chain_method="parallel" maps one chain onto one
# device, so this must be >= num_chains or numpyro cannot run the chains in parallel.
# More devices than chains does nothing - the extra ones just sit idle. 64 devices on
# a 10-core SLURM allocation was the oversubscription; commenting it out left only 1
# device, which is what forced chain_method="vectorized".
# Original 64-device configuration retained for reference:
# pyro.set_host_device_count(64)
pyro.set_host_device_count(4)
#pyro.enable_x64()
