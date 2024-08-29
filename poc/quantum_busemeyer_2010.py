import numpy as np
from scipy.stats import *
from scipy.linalg import *
import matplotlib.pyplot as plt

def _buildH(m,a,b,c): 
# H = buildH(a,b,c)
# m = number of states  
# a = off diag left  
# b = diag  
# c = off diag right


    H = np.zeros((m, m))
    H[0,0] = b[0]
    H[0,1] = c[0]
    H[m-1, m-2] = a[m-1]
    H[m-1, m-1] = b[m-1]

    for k in range(1,m-1):
        H[k,k-1] = a[k]
        H[k,k] = b[k]
        H[k,k+1] = c[k]

    return H

ns = 7
ws = 3
tv = np.arange(0,20,0.1)
nt = tv.shape[0]

Mid = (ns+1)//2
mv = np.arange(-(Mid-1), (Mid))
mu=1
ap=1

S0 = np.zeros((ns,1))
S0[Mid-ws-1:Mid+ws] = 1
S0 = S0/np.sqrt(S0.T @ S0)

b=mu*mv
a=ap*np.ones((ns,1))

H = _buildH(ns, a,b,a)
Mc_arr = []
Mc_arr_ts = []
Pt_arr = []

for n in range(1, nt):
    t = tv[n]
    U = expm(-1j*t*H)
    St = U@S0
    Pt = np.abs(St)**2
    Pt_arr.append(Pt.flatten())
    Mc = mv @ Pt
    Mc_arr.append(Mc)

import pandas as pd
ax = pd.Series(np.asarray(Mc_arr).flatten()).plot()
plt.show()

# For free time responses
Mc_arr = []
Mc_arr_ts = []
Pt_arr = []
rw=1

prob=0.25
Mr = np.zeros(ns)
Mr[-rw:] = np.sqrt(prob)
Mr = np.diag(Mr)

Mi = np.zeros(ns)
Mi[:rw] = np.sqrt(prob)
Mi = np.diag(Mi)

Mn = np.sqrt(np.eye(ns) - (Mr**2 + Mi**2))

print((Mr.T @ Mr + Mi.T @ Mi + Mn.T @ Mn)) # should be equal to 1

ws = 1

S0 = np.zeros((ns,1))
S0[Mid-ws-1:Mid+ws] = 1
S0 = S0/np.sqrt(S0.T @ S0)


for n in range(1,nt):
    #t = tv[n]
    U = expm(-1j*0.1*H)
    St = Mr @ U @ np.linalg.matrix_power(Mn @ U, n-1) @ S0
    Pt = np.abs(St)**2
    Pt_arr.append(Pt.sum())
    Mc = mv @ Pt
    Mc_arr.append(Mc)

import pandas as pd
pd.Series(np.asarray(Mc_arr).flatten()).plot()

