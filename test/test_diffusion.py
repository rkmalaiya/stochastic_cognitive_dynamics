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

    models = [[m_c, m_ic] for (m_c,_), (m_ic,_) in dd.get_models_n_vars(I,J)]

    RT_correct, RT_incorrect = dd.sample_prior_data(samples_n, models)
    log.debug(f"Response time: {RT_correct.shape}")
    assert RT_correct.shape == (1,I*samples_n,J)
    assert RT_incorrect.shape == (1,I*samples_n,J)

    log.info("Test passed!")

def test_post_samples_n_post_pred(X,RT,samples_n):
    I,J = RT.shape
    chains = 4
    log.debug("Starting test for diffusion")
    models = [[m_c, m_ic] for (m_c, _), (m_ic,_)  in dd.get_models_n_vars(data_rt=RT, data_ra=X)]
    posteriors, models = dd.sample_posterior_params(samples_n,chains=chains,tune=100, models= models)
    v_c, v_ic = ut.extract_var(posterior_chains=posteriors, var="v")
    assert v_c.shape == (chains, samples_n, I, 1) #for all the trials / items per participant only 1 parameter is estimated
    assert v_ic.shape == (chains, samples_n, I, 1)

    #RT_correct, RT_incorrect = dd.sample_post_pred_data(posteriors=posteriors,samples_n=samples_n,models=models)

    #log.debug(f"RT_correct.shape {RT_correct.shape}")
    #assert RT_correct.shape == (1,I*samples_n,J)
    #assert RT_incorrect.shape == (1,I*samples_n,J)

    log.info("Test passed!")




#%%
if __name__ == "__main__":   
    #test_diffusion_prior_predictive_default()
    #test_diffusion_prior_predictive(10,4, 100)
    I,J = 3,4
    for i,j in zip(range(2,I), range(3,J)):
        X = np.random.randint(0,2,(i,j))
        RT = np.random.uniform(0,4,(i,j))
        log.debug(f"Data generated of size {i}, {j}")
        test_post_samples_n_post_pred(X, RT, 10)
        
# %%
