# %%
from cme.decision_models import diffusion_loop_method as dd
import pandas as pd
import numpy as np
from cme.utils import common_utils as ut
import arviz as az
import os

# %%
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# %%
rotation_RT = pd.read_csv(f"data/rotation_rt.csv")
#rotation_RT = pd.read_csv(f"data/sim_low_low_500_rt.csv")
rotation_RT_n = rotation_RT.loc[~rotation_RT.isna().any(axis=1),:].to_numpy()#[0:1,:]

rotation_X = pd.read_csv(f"data/rotation_ra.csv")
#rotation_X = pd.read_csv(f"data/sim_low_low_500_ra.csv")
rotation_X_n = rotation_X.loc[~rotation_RT.isna().any(axis=1),:].astype(int).to_numpy()#[0:1,:]

# %%
rotation_RT_n[rotation_RT_n < 0.9]

# %%
samples_n, tune, chains, acceptance_rate = 1000, 700, 4, 0.85

posterior_chain,_ = dd.sample_posterior_params(rotation_RT_n, rotation_X_n, 
samples_n,chains=chains, sampler="PYMC", tune=tune, acceptance_rate = acceptance_rate)


# %%
posterior_chain

# %%
df_posterior_chain = ut.get_summary(posterior_chain)
print(df_posterior_chain)

# %%
r_hat = ut.get_rhat(df_posterior_chain)
print(r_hat)

# %%
az.plot_trace(posterior_chain)

# %%
#posterior_chain = posterior_chain.sel(chain=[0,2,3])

# %%
az.summary(posterior_chain)

# %%
r_hat.T.plot(kind="kde")

# %%
#az.plot_trace(posterior_chain)

# %%
v_c_ch = ut.get_chains_for_param(posterior_chain,"v")
v_c_ch

# %%
v_c_ch.plot(kind="box")

# %%
a_c_ch = ut.get_chains_for_param(posterior_chain,"a")
a_c_ch

# %%
a_c_ch.fillna(0).plot(kind="box")

# %%
z_c_ch = ut.get_chains_for_param(posterior_chain,"z")
z_c_ch

# %%
z_c_ch.fillna(0).plot(kind="box")

# %%
ter_c_ch = ut.get_chains_for_param(posterior_chain,"t_er")
ter_c_ch

# %%
ter_c_ch.fillna(0).plot(kind="box")


