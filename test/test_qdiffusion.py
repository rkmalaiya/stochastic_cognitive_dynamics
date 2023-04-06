#%%
from cme.models import qdiffusion as qd
import cme.utils.common_logging as cl
#import sys
import numpy as np

log = cl.get_logger()


def test_qdiffusion_prior_predictive(I,J):
    log.debug("Starting test for Qdiffusion(", I, J, ")")
    model, _ = qd.get_model_vars(I,J)
    data_rt, data_ra = qd.gen_sample_data(model)
    assert data_rt.shape == (I,J)
    assert data_ra.shape == (I,J)
    log.info("Test passed!")

def test_qdiffusion_posterior(I,J, samples_n, chains):
    log.debug("Starting test for Qdiffusion(", I, J, ")")
    model, _ = qd.get_model_vars(I,J)
    data_rt, data_ra = qd.gen_sample_data(model)
    model, _ = qd.get_model_vars(I,J, data_rt=data_rt, data_ra=data_ra)
    posterior, posterior_pred = qd.sample_posterior(model, samples_n=samples_n, chains=chains)
    posterior_vp = posterior.posterior.v_p
    posterior_pred_RT = posterior_pred.posterior_predictive.RT_kj
    assert posterior_vp[0,0].shape == (I,1)
    assert posterior_pred_RT[0,0].shape == (I,J)
    log.info("Test passed!")


#%%
if __name__ == "__main__":   
      test_qdiffusion_prior_predictive(10,4)
      test_qdiffusion_posterior(10,4,100,4)  

# %%
