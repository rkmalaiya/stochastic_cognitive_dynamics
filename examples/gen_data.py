# %%
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_directory))

from cme.simulators import diffusion_random_walk as rw
from numpy import *
from scipy.stats import *
from pandas import *
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import numpy as np

data_folder = f"{current_directory}/data"
os.makedirs(data_folder, exist_ok=True)

# Here theta is the number of states being modeled.
theta, alpha, tau, sigma, I, J = 10, 1.5, 0.01, 1, 2, 5#(50,100,200)#,1000)
Is = (I//2,)
# RT_variability
v_p_s = {#"0_2":asarray([0.02]),
       #"0_5":asarray([0.05]),
       #"1":asarray([1]),
       "low_low":asarray([4]), # Low RT and Low Variability
       "high_low":asarray([0.01]), # High RT and Low Variability
       "low_medium":repeat([0.01, 4], (theta+2)/ 2)[0:theta+100], 
       "high_medium":repeat([0.01, 0.08], (theta+2)/ 2)[0:theta+100], 
       "low_high":repeat([0.01,0.02,2,4], (theta+2)/ 4)[0:theta+100], 
       "high_high":repeat([0.01,0.02,0.05,0.08], (theta+2)/ 4)[0:theta+100],  
       }

"""v_p_s = {
       "Fast RT, Single Stage":asarray([4]),
       "Fast RT, Multi Stage - Insightful":repeat([0.1,0.2,3,4], (theta+2)/ 4)[0:theta+100], 
       "Fast RT, Multi Stage - Continuous":repeat([1,2,3,4], (theta+2)/ 4)[0:theta+100],
       "Slow RT, Single Stage":asarray([0.1]),
       "Slow RT, Multi Stage - Insightful":repeat([0.1,0.2,0.3,4], (theta+2)/ 4)[0:theta+100], 
       "Slow RT, Multi Stage - Continuous":repeat([0.1,0.3,0.5,0.8], (theta+2)/ 4)[0:theta+100],  
       }"""

"""v_p_s = {
       
       "Fast RT, Single Stage":np.asarray([4]),
       "Fast RT, Multi Stage - Continuous":np.repeat([1,2,3,4], (theta+2)/ 4)[0:theta+100],
       "Fast RT, Multi Stage - Insightful":np.repeat([0.1,0.1, 4, 4], (theta+2)/ 4)[0:theta+100], 

       "Slow RT, Single Stage":np.asarray([0.1]),
       "Slow RT, Multi Stage - Continuous":np.repeat([0.1,0.3,0.5,0.8], (theta+2)/ 4)[0:theta+100],  
       "Slow RT, Multi Stage - Insightful":np.repeat([0.1,0.1, 2,2], (theta+2)/ 4)[0:theta+100], 

       }"""

lamb = 0.8
xs = np.linspace(0.05,5,theta+3)
ys = np.exp(lamb * xs) / 10

v_p_s = {
       
       "Single Stage - Fast RT":np.asarray([5]),
       "Single Stage - Fast RT-":-np.asarray([5]),
       "Single Stage - Slow RT":np.asarray([0.1]),
       "Single Stage - Slow RT-":-np.asarray([0.1]),
       #"Multi Stage - Insightful":np.repeat([0.5,0.5,0.5,5], ((theta+3)/ 4))[0:theta+100], 
       #"Multi Stage - Insightful-":-np.repeat([0.5,0.5,0.5,5], ((theta+3)/ 4))[0:theta+100], 
       #"Multi Stage - Continuous":ys, #np.repeat([0.5,1.5,3,5], ((theta+3)/ 4))[0:theta+100],
       #"Multi Stage - Continuous-":-ys #np.repeat([0.5,1.5,3,5], ((theta+3)/ 4))[0:theta+100],
       }


for k,v in v_p_s.items():
    print(k)

#v_i_s = asarray([0.5, 0.75, 1, 1.25]) # very easy to hard
v_i_s = asarray([1]) # No Effect

def sim_data(v_i_s, k, v_p, I):
    print(f"********* Starting process {k} for sample {I}")
    #RT, X, v_arr = rw.gen_RT_X_mat(theta, alpha, tau, sigma, v_p, v_i_s, I=I,J=10, process="Wiener", initial="Any", njobs=60)  
    RT, X, v_arr, tr_arr = rw.gen_RT_X_mat(theta, alpha, tau, sigma, v_p, I=I,J=J, process="Wiener", initial="Any", njobs=1)
    RT = RT + lognorm(0.01).rvs(1)[0]
    savetxt(f"{data_folder}/sim_{k}_{I}_rt.csv", RT, delimiter=",")
    savetxt(f"{data_folder}/sim_{k}_{I}_ra.csv", X, delimiter=",")
    if(len(asarray(v_arr).squeeze().shape)> 2):
        savetxt(f"{data_folder}/sim_{k}_{I}_drift.csv", asarray(v_arr).squeeze().mean(axis=-1), delimiter=",")
    else: 
        savetxt(f"{data_folder}/sim_{k}_{I}_drift.csv", atleast_1d(asarray(v_arr).squeeze()), delimiter=",")
    
    

# %%
#asyncio.run(sim_data(v_p, v_i_s, Is))
#Parallel(n_jobs=6)(delayed(sim_data)(v_i_s, k, v_p, I) for k,v_p in v_p_s.items() for I in Is)

[sim_data(v_i_s, k, v_p, I) for k,v_p in v_p_s.items() for I in Is]

# %%
import itertools
for I in Is:
       for k1,k2 in itertools.islice(itertools.pairwise(v_p_s.keys()),0, None, 2):
              rt1 = loadtxt(f"{data_folder}/sim_{k1}_{I}_rt.csv", delimiter=",", ndmin=2)
              rt2 = loadtxt(f"{data_folder}/sim_{k2}_{I}_rt.csv", delimiter=",", ndmin=2)
              rt = concatenate((rt1, rt2))
              savetxt(f"{data_folder}/sim_{k1}_{I+I}_rt.csv", rt, delimiter=",")
              print(rt.shape)

              x1 = loadtxt(f"{data_folder}/sim_{k1}_{I}_ra.csv", delimiter=",", ndmin=2)
              x2 = loadtxt(f"{data_folder}/sim_{k2}_{I}_ra.csv", delimiter=",", ndmin=2)
              x = concatenate((x1, x2))
              savetxt(f"{data_folder}/sim_{k1}_{I+I}_ra.csv", x, delimiter=",")
              print(x.shape)
       
       

# %%
