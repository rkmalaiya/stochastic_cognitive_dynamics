#%%
from diffusion_models.models import diffusion as dd
import diffusion_models.utils.common_logging as cl
import numpy as np
from diffusion_models.utils import common_utils as ut
import pandas as pd
import os
import pymc as pm
from pytensor import tensor as at


log = cl.get_logger("Test-Diffusion")

def test_diffusion_prior_predictive_default():
    log.debug("Starting test for Diffusion (Default)")
    RT, X = dd.sample_prior_data()
    log.debug(f"Response time: {RT.shape}")
    assert len(RT.shape) != 0
    log.info("Test passed!")

def test_diffusion_prior_predictive(I,J, samples_n):
    log.debug(f"Starting test for diffusion({I}, {J})")

    RT_correct, RT_incorrect = dd.sample_prior_data(samples_n)
    log.debug(f"Response time: {RT_correct.shape}")
    assert RT_correct.shape == (1,I*samples_n,J)
    assert RT_incorrect.shape == (1,I*samples_n,J)

    log.info("Test passed!")

def test_posterior_samples(X,RT,samples_n, tune=10, acceptance_rate = 0.85):
    chains = 4
    log.debug("Starting test for diffusion")
    
    posterior_chain,_ = dd.sample_posterior_params(RT, X, samples_n,chains=chains, sampler="PYMC", tune=tune, acceptance_rate = acceptance_rate)

    log.debug(f"drift rate shape {posterior_chain.posterior.v_c.shape}")
    assert posterior_chain.posterior.v_c.shape == (chains, samples_n, RT.shape[0]) #for all the trials / items per participant only 1 parameter is estimated

    log.info("Test passed!")

def test_convergence(X,RT,samples_n=200, tune=1500, acceptance_rate = 0.85):
    chains = 4
    log.debug("Starting test for diffusion")
    posterior_chain,_ = dd.sample_posterior_params(RT, X, samples_n,chains=chains, sampler="PYMC", tune=tune, acceptance_rate = acceptance_rate)
    log.debug("Getting r_hat")
    r_hat = ut.get_rhat(posterior_chain)
    log.debug(f"r_hat mean{r_hat.mean()}")
    #assert r_hat.mean() < 2 #soft convergence criterion




def test_case_1():
    test_diffusion_prior_predictive_default()
    test_diffusion_prior_predictive(10,4, 100)

def test_case_2():
    I,J = 600, 25
    for i,j in zip(range(2,I,100), range(3,J,4)): # Both ranges will produce 6 elements
        X = np.random.randint(0,2,(i,j))
        RT = np.random.uniform(0,4,(i,j))
        log.debug(f"Data generated of size {i}, {j}")
        test_posterior_samples(X, RT, 200)

def test_case_3():
    
    rotation_RT = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/rotation_rt.csv")
    rotation_RT_n = rotation_RT.loc[~rotation_RT.isna().any(axis=1),:].to_numpy()

    rotation_X = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/rotation_ra.csv")
    rotation_X_n = rotation_X.loc[~rotation_RT.isna().any(axis=1),:].astype(int).to_numpy()

    test_convergence(rotation_X_n[1:10,], rotation_RT_n[1:10,])

def test_case_4():
    for parti_n in range(1,50):
        log.debug(f"For participant count:{parti_n} ")
        dd._test_likelihood_using_prior(parti_n)


#%%
if __name__ == "__main__":   
    
    
    #test_case_2()
    test_case_3()
    test_case_4()
    

#%%

