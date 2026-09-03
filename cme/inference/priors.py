"""
Prior blocks for the confidence accumulation model.

Each function opens numpyro sample sites and returns the parameters the model
needs. They are only meaningful inside a numpyro trace.
"""

import jax
import jax.numpy as npx
import numpyro as pyro
import numpyro as npy
import numpyro.distributions as dist

from cme.utils import common_logging as cl

log = cl.get_logger("inference.priors")


def centralized_parameters(I):
    """
    I: Number of participants
    
    """
    mu_m =  pyro.sample(f"mu_m", dist.Normal(0,1))
    mu_s =  pyro.sample(f"mu_s", dist.HalfNormal(2))
    with pyro.plate('I6', I, dim=-2):
        mu = pyro.sample("mu", dist.Normal(mu_m,mu_s)) # Drift Rate
        sigma = pyro.sample("sigma", dist.Normal(1,0.1)) # Diffusion Rate
    return mu, sigma

# def non_centralized_parameters(model_type, I):
#     """
#     I: Number of participants
#     Model-specific priors for improved numerical stability
#     """
#     m = pyro.sample("m", dist.Normal(0.1,0.1))
#     #m = pyro.deterministic("m", 0.1)
#     s = pyro.sample("s", dist.HalfNormal(0.1))

#     # Quantum models need tighter prior control on sigma scale
#     if model_type == "Quantum":
#         m_si = pyro.sample("m_si", dist.Normal(0.5, 0.5))  # Shifted to ensure positive softplus output
#         s_si = pyro.sample("s_si", dist.HalfNormal(0.05))  # Tighter to prevent extreme values
#     else:  # Markov
#         m_si = pyro.sample("m_si", dist.Normal(0, 1))
#         s_si = pyro.sample("s_si", dist.HalfNormal(0.1))

#     with pyro.plate('I3', I, dim=-2):
#         if model_type == "Markov":
#             mu_r = pyro.sample("mu_r", dist.Normal(0.1,1)) # Drift Rate
#             mu = pyro.deterministic("mu", m + s * mu_r)
#         elif model_type == "Quantum":
#             mu_r = pyro.sample("mu_r", dist.Normal(0.1,1)) # Drift Rate
#             mu = pyro.deterministic("mu", jax.nn.softplus(m + s * mu_r))
       
#         if model_type == "Markov":
#             sigma_r = pyro.sample("sigma_r", dist.Normal(0,0.1)) # Diffusion Rate
#         elif model_type == "Quantum":
#             sigma_r = pyro.sample("sigma_r", dist.Normal(0,0.1)) # Diffusion Rate

        
#         sigma_base = jax.nn.softplus(m_si + s_si * sigma_r)
        
#         # Ensure minimum floor for numerical stability in Quantum likelihood
#         if model_type == "Quantum":
#             sigma = pyro.deterministic("sigma", npx.clip(sigma_base, 0.01, None))
#         else:
#             sigma = pyro.deterministic("sigma", sigma_base)
    
#     return mu, sigma


def non_centralized_parameters(model_type, I):
    # Fixed prior location/scale for drift
    m = pyro.deterministic("m", npx.asarray(0.1))
    s = pyro.deterministic("s", npx.asarray(0.1))

    # Fixed prior location/scale for sigma
    if model_type == "Quantum":
        m_si = pyro.deterministic("m_si", npx.asarray(0.5))
        s_si = pyro.deterministic("s_si", npx.asarray(0.05))
    else:  # Markov
        m_si = pyro.deterministic("m_si", npx.asarray(0.0))
        s_si = pyro.deterministic("s_si", npx.asarray(0.1))

    with pyro.plate("I3", I, dim=-2):

        # Drift
        if model_type == "Markov":
            mu_r = pyro.sample("mu_r", dist.Normal(0.1, 1.0))
            mu = pyro.deterministic("mu", m + s * mu_r)

        elif model_type == "Quantum":
            mu_r = pyro.sample("mu_r", dist.Normal(0.1, 1.0))
            mu = pyro.deterministic("mu", jax.nn.softplus(m + s * mu_r))

        else:
            raise Exception(f"Please select one of {model_type}")

        # Diffusion
        sigma_r = pyro.sample("sigma_r", dist.Normal(0.0, 0.1))

        sigma_base = jax.nn.softplus(m_si + s_si * sigma_r)

        if model_type == "Quantum":
            sigma = pyro.deterministic("sigma", npx.clip(sigma_base, 0.01, None))
        else:
            sigma = pyro.deterministic("sigma", sigma_base)

    return mu, sigma


def sample_initial_state(n_states, response_width, I = 1, model_type = "Markov|Quantum"):
    """
    Initial state phi_0 as a latent variable, drawn per participant.

    This is the numpyro half of what used to be
    `confidence_accumulation._get_initial_state(..., prior_type="Model")`. The
    fixed-shape prior types ("Upper", "Lower", "Centered", "All", "Opposite")
    open no sample sites and stayed with the model.

    Sample sites: phi_conc, phi_init. Deterministic: phi_0.
    """
    if model_type == "Markov":
        with npy.plate('I1', I, dim=-4):
            with npy.plate('S', n_states - 2*response_width, dim=-1):
                conc = npy.sample("phi_conc", dist.Beta(0.5,0.5))+0.01 #to avoid 0

        with npy.plate('I2', I, dim=-3):
            p_0 = npy.sample("phi_init", dist.Dirichlet(conc)) # Initial State
        p_0 = npx.pad(p_0, ((0,0),(0,0),(0,0),(response_width,response_width)))
        phi_0 = npy.deterministic("phi_0", p_0.transpose(0,1,3,2)) #.transpose(0,1,3,2)  
    elif model_type == "Quantum":
        with npy.plate('I1', I, dim=-4):
            with npy.plate('S', n_states - 2*response_width, dim=-1):
                conc = npy.sample("phi_conc", dist.Beta(0.5,0.5))+0.01 #to avoid 0

        with npy.plate('I2', I, dim=-3):
            p_0 = npy.sample("phi_init", dist.Dirichlet(conc)) # Initial State
            
        p_0 = npx.pad(p_0, ((0,0),(0,0),(0,0),(response_width,response_width)))
        phi_0 = npy.deterministic("phi_0", p_0.transpose(0,1,3,2)**(1/2))
    else:
        raise Exception(f"Please select one of {model_type}")

    return phi_0
