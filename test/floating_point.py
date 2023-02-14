#%%
import numpy as np
import scipy.stats as st

for i in range(10000):
    sp = st.dirichlet(np.repeat(0.5,20), seed=1234).rvs()
    sm = np.sum(sp)
    if(sm>1):
        print(f"{sm} at iteration {i}")

# %%
st.dirichlet(np.repeat(0.5,20), seed=1234).rvs()
# %%
