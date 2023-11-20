#%%
import cme.diffusion_models.diffusion_discrete as ddm_disc
import scipy.linalg as ln
K = ddm_disc._buildK(5, mu=[[0.01,0.05,5]], sigma=1)
K1 = ddm_disc._buildK(5, mu=[[0.05]], sigma=1)

#%%
K


#%%
K1

# %%
import numpy as np
Q = ln.expm(np.array(K.squeeze())*30).round(decimals=2)
Q
# %%

Q @ np.ones((5,1))/5
# %%
(0.46 + 0.2 + 0.04 + 0.01 + 0.) *0.2