# %%
from diffusion_models.models import diffusion as dd
import pandas as pd
import numpy as np
from diffusion_models.utils import common_utils as ut
import arviz as az
import os

# %%
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
import importlib
importlib.reload(ut)


# %%
rotation_RT = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/rotation_rt.csv")
rotation_RT_n = rotation_RT.loc[~rotation_RT.isna().any(axis=1),:].to_numpy()#[0:1,:]

rotation_X = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/rotation_ra.csv")
rotation_X_n = rotation_X.loc[~rotation_RT.isna().any(axis=1),:].astype(int).to_numpy()#[0:1,:]

# %%
samples_n, tune, chains, acceptance_rate = 1000, 500, 4,0.90

posterior_chain,_ = dd.sample_posterior_params(rotation_RT_n, rotation_X_n, 
samples_n,chains=chains, sampler="PYMC", tune=tune, acceptance_rate = acceptance_rate)


# %%
r_hat = ut.get_rhat(posterior_chain)
r_hat

# %%
posterior_chain

# %%
az.summary(posterior_chain)

# %%
az.plot_trace(posterior_chain, compact=True)

# %%
v_c_ch = ut.get_chains_for_param(posterior_chain,"v_c")
v_c_ch

# %%
v_c_ch.plot(kind="density")

# %%
a_c_ch = ut.get_chains_for_param(posterior_chain,"a_c")
a_c_ch

# %%
a_c_ch.plot(kind="density")



# %%
