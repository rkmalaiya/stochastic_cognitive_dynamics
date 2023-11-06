#%%
from numpy import *
from scipy import linalg as ln
import pandas as pd

def _buildK(a,b,c):
    m = a.shape[0]
    K = zeros((m,m))
    K[[0,1], 0] = asarray([b[0], -b[0]]).T
    K[[-2,-1], -1] = asarray([-b[-1], b[-1]]).T

    for k in range(1, m-1):
        v = [[k-1], [k], [k+1]]
        K[v,k] = asarray([a[k], b[k], c[k]])

    return K

ns = 7
ws = 3
tv = arange(0,20,0.1)
nt = tv.shape[0]
Mid = int((ns+1)/2)
mv = arange(-(Mid-1),(Mid))

mu=0.5
var=2
S0 = zeros((ns,1))
S0[(Mid-ws):(Mid+ws)] = 1
S0 = S0/sum(S0)

#%%
mk = ones((ns,1))
b = -var*mk
a1 = 0.5* (var-mu)*mk
a2 = 0.5* (var+mu)*mk
K = _buildK(a1,b,a2)

#%%
PM2 = []
for n in range(1,nt):
    t = tv[n]
    T = ln.expm(t*K)
    Pt = T @ S0
    Mc = mv @ Pt
    PM2.append(Mc)

pd.Series(asarray(PM2).squeeze()).plot.line()
# %%
