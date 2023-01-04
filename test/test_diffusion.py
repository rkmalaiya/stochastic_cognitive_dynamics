#%%
from diffusion_models.models import diffusion as dd
import diffusion_models.utils.common_logging as cl
import numpy as np
from diffusion_models.utils import common_utils as ut

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

def test_posterior_samples(X,RT,samples_n):
    chains = 4
    log.debug("Starting test for diffusion")
    
    posterior_chain,_ = dd.sample_posterior_params(RT, X, samples_n,chains=chains, sampler="PYMC")

    assert posterior_chain.posterior.v.shape == (chains, samples_n, RT.shape[0], 1) #for all the trials / items per participant only 1 parameter is estimated

    log.info("Test passed!")




#%%
if __name__ == "__main__":   
    #test_diffusion_prior_predictive_default()
    #test_diffusion_prior_predictive(10,4, 100)
    I,J = 600, 25
    for i,j in zip(range(2,I,100), range(3,J,4)): # Both ranges will produce 6 elements
        X = np.random.randint(0,2,(i,j))
        RT = np.random.uniform(0,4,(i,j))
        log.debug(f"Data generated of size {i}, {j}")
        test_posterior_samples(X, RT, 200)
        
# %%
