#%%
from diffusion_models.models import qdiffusion as qd
from diffusion_models.models import diffusion as dd

import diffusion_models.utils.common_utils as ut
import diffusion_models.utils.common_logging as cl
#import sys
import numpy as np


#log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#log = logging.getLogger("util")
#hdlr = logging.StreamHandler(sys.stdout)
#log.addHandler(hdlr)

log = cl.get_logger("util")

def get_posterior(model, samples_n = 10, chains=2):
    posterior, posterior_pred = ut.sample_posterior(model, samples_n=samples_n, chains=chains)
    return posterior, posterior_pred

def test_qdiffusion_prior_predictive(I,J):
    log.debug("Starting test for Qdiffusion(", I, J, ")")
    model, _ = qd.get_model_vars(I,J)
    data_rt, data_ra = qd.gen_sample_data(model)
    assert data_rt.shape == (I,J)
    assert data_ra.shape == (I,J)
    log.info("Test passed!")

def test_qdiffusion_posterior(I,J):
    log.debug("Starting test for Qdiffusion(", I, J, ")")
    model, _ = qd.get_model_vars(I,J)
    data_rt, data_ra = qd.gen_sample_data(model)
    model, _ = qd.get_model_vars(I,J, data_rt=data_rt, data_ra=data_ra)
    posterior, posterior_pred = get_posterior(model)
    posterior_vp = posterior.posterior.v_p
    posterior_pred_RT = posterior_pred.posterior_predictive.RT_kj
    assert posterior_vp[0,0].shape == (I,1)
    assert posterior_pred_RT[0,0].shape == (I,J)
    log.info("Test passed!")

def test_diffusion_posterior(I,J):
    I,J = 10,(4,2,3,5,6,2,7,8,4,5)

    for i,j in zip(range(1,I+1), J):
        X = np.random.randint(0,2,j)
        RT = np.random.uniform(0,4,j)

        model = dd.diffusion_model(RT[X>0])
        posterior, posterior_pred = get_posterior(model)
        
        model = dd.diffusion_model(RT[np.invert(X>0)])
        posterior, posterior_pred = get_posterior(model)


#%%
if __name__ == "__main__":   
      test_qdiffusion_prior_predictive(10,4)
      test_qdiffusion_posterior(10,4)  

# %%
log.debug("hi")
# %%
print("hi")