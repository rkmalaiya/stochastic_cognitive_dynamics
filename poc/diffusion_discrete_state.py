#%%
# This code is written to reproduce the Markov random walk related plots in Chap 8 of Busemeyer 2010
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
tv = arange(0,10,0.1)
nt = tv.shape[0]
Mid = int((ns+1)/2)
mv = arange(-(Mid-1),(Mid))

mu=0.5
var=2
S0 = zeros((ns,1))
S0[(Mid-ws):(Mid+ws-1)] = 1
S0 = S0/sum(S0)

mk = ones((ns,1))
b = -var*mk
b_m = 0.5* (var-mu)*mk
b_p = 0.5* (var+mu)*mk
K = _buildK(b_m,b,b_p)


PM2 = []
df_pt1 = []
for n in range(1,nt):
    t = tv[n]
    T = ln.expm(t*K)
    Pt = T @ S0
    df_pt1.append(pd.DataFrame(
        {"time":n, "states":arange(ns), "prob":Pt.squeeze()}
    ))
    Mconf = mv @ Pt
    PM2.append(Mconf)
df_pt1 = pd.concat(df_pt1)
pd.Series(asarray(PM2).squeeze()).plot.line()
# %%
ns = 11
ws = 4
mk = ones((ns,1))

b_m = 9.765 *mk
b_p = 10.325*mk
a = -(b_m + b_p)

Mid = int((ns+1)/2)
S0 = zeros((ns,1))
S0[(Mid-ws):(Mid+ws)] = 1
S0 = S0/sum(S0)

Mcorr = zeros(ns)
Mcorr[-10:] = 0.25
Mcorr = diag(Mcorr)

Mincorr = zeros(ns)
Mincorr[:10] = 0.25
Mincorr = diag(Mincorr)

Mnoresp = eye(ns)-Mcorr-Mincorr

delta_t = 1
RT = 300
n = int(RT/delta_t)


K = _buildK(b_m, a, b_p)

likl_arr = []
mconf_arr = []
ts_arr = []

Mid = int((ns+1)/2)
mv = arange(-(Mid-1),(Mid))
import pandas as pd
df_likl, df_conf = pd.DataFrame(), pd.DataFrame()
Pt_arr = []
for delta_t in range(10, RT, 10):
    likl_arr = []
    mconf_arr = []
    ts_arr = []
    
    for rt in range(delta_t,RT,delta_t):
        n = int(rt/delta_t)
        Pt = ln.expm(delta_t*K) @ ((t2 := linalg.matrix_power(Mnoresp @ ln.expm(delta_t*K), n-1)) @ S0)
        Pt_arr.append(pd.DataFrame({"delta":delta_t, "states":arange(ns), "time":rt, "prob":Pt.flatten()}))
        likl =  Mcorr @ Pt
        likl_arr.append(likl.sum())
        conf = mv @ Pt
        mconf_arr.append(conf)
        ts_arr.append(rt)
    df_likl = pd.concat([df_likl, pd.DataFrame(likl_arr).assign(delta=delta_t, rt = ts_arr)])
    df_conf = pd.concat([df_conf, pd.DataFrame(mconf_arr).assign(delta=delta_t, rt = ts_arr)])
df_pt = pd.concat(Pt_arr)
sns.relplot(df_pt1,
x="time", y="states", size="prob")
df_pt1.pivot(index="time", columns="states",values="prob")
# %%
import seaborn as sns
sns.kdeplot(df_likl, x=0, hue="delta")
df_conf.groupby("delta").mean()[0].plot.bar()
sns.lineplot(
    df_conf,
    x="rt",
    y=0,
    hue="delta"
)