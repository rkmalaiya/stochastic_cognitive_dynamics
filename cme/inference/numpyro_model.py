"""
The numpyro model: priors wired to the confidence accumulation likelihood.

`model` is what NUTS and SVI are pointed at. The likelihood it evaluates, and
the intensity/measurement matrices it builds, come from
`cme.decision_models.confidence_accumulation`; the priors come from
`cme.inference.priors`.

`estimation_likelihood` lives here rather than with the model because it is the
inference-facing wrapper: it records the clipping diagnostics (P_min, P_max,
P_clip_low, P_clip_high) as numpyro deterministic sites on its way to the log
scale that `pyro.factor` needs. The pure probability it wraps is
`ca.likelihood`.
"""

import jax
import jax.numpy as npx
import numpyro as pyro

import cme.decision_models.confidence_accumulation as ca
from cme.inference.priors import centralized_parameters, non_centralized_parameters, sample_initial_state
from cme.utils import common_logging as cl

log = cl.get_logger("inference.numpyro_model")


def transformed_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):
    
    return estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)

def estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):
    P_t = ca.likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    #P_t = npx.where(P_t <= 0, 0.00001, npx.log(P_t))
    # P_t = npx.where(P_t <= 0, 10e-12, P_t)
    # P_t = npx.where(npx.isnan(P_t), 10e-12, P_t)
    # P_t = npx.log(P_t)
    # #pyro.deterministic("loglikl", P_t.sum(axis=-1)) #summing over trials
    # return P_t.sum(axis=-1)
    eps = 1e-15
    eps_nan = 1e-12 # having two different eps for debugging purposes.

    pyro.deterministic("P_min", npx.min(P_t))
    pyro.deterministic("P_max", npx.max(P_t))
    pyro.deterministic("P_clip_low", npx.mean(P_t <= eps))
    pyro.deterministic("P_clip_high", npx.mean(P_t >= 1.0))

    P_t = npx.where(npx.isnan(P_t), eps_nan, P_t)
    P_t = npx.clip(P_t, eps, 1.0)
    P_t = npx.log(P_t)
    #pyro.deterministic("loglikl", P_t.sum(axis=-1))
    return P_t#.sum(axis=-1) For LOO and other model comparison, I need trialvise log-likelihood.

def model(n_states, start_width, response_width, delta, RA_s, RT_s, measurement_prob, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
    if likelihood_type == "SINGLE":
        I, _ = RA_s.shape
    elif likelihood_type == "JOINT":
        if RA_s[0].shape != RA_s[1].shape:
            raise Exception("All Responses should be equal shaped")
        I, _ = RA_s[0].shape
    else:
        raise Exception(f"Please select one of {likelihood_type} for likelihood")

    if params_type == "Centralized":
        mu, sigma = centralized_parameters(I)
    elif params_type == "NonCentralized":
        mu, sigma = non_centralized_parameters(model_type, I)
    #elif params_type == "ParticipantLevel":
    #    mu, sigma = participant_parameters(model_type, I)
    else:
        raise Exception(f"Please select one of {params_type}")

    if model_type == "Markov":
        sigma = pyro.deterministic("sigma_final",npx.abs(mu) + sigma) # Sigma needs to be larger than mu and Sigma cannot be negative
                # removed sigma**2 to allow stability in parameter estimates. Negative values are avoided through softplus now
        intensity_matrix = ca.diffusion_buildK(n_states, mu, sigma, delta)

    elif model_type == "Quantum":
        # For Quantum: ensure sigma > 0 and has numerical stability
        # Consider making sigma magnitude scale with mu for better parameter coupling
        sigma_quantum = npx.clip(npx.abs(mu) * 0.5 + sigma, 0.01, None)
        sigma = pyro.deterministic("sigma_final", sigma_quantum)
        intensity_matrix = ca.quantum_buildH(n_states, mu, sigma, delta)
    else:
        raise Exception(f"Please select one of {model_type}")

    phi_0 = sample_initial_state(n_states, response_width, I = I, model_type = model_type)
    Mc, Mw, Mn = ca._get_measurement_matrix(n_states = n_states, response_width=response_width, prob=measurement_prob, model_type = model_type)

    if RT_s is not None:
        likl = estimation_likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, 
                          transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
        #likl = npx.log(likl)
        pyro.deterministic("likl_rt", likl)
        pyro.factor("likelihood", likl) #.sum()

# def guide(n_states, start_width, response_width, delta, RA_s, RT_s, measurement_prob, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
#     I, _ = RA_s.shape
#     mu, sigma = non_centralized_parameters_VI(I)
#     if model_type == "Markov":
#         sigma = pyro.deterministic("sigma_final",npx.abs(mu) + sigma) # Sigma needs to be larger than mu and Sigma cannot be negative
#         intensity_matrix = diffusion_buildK(n_states, mu, sigma, delta)

#     elif model_type == "Quantum":
#         sigma = pyro.deterministic("sigma_final",sigma) # Sigma cannot be negative
#         intensity_matrix = quantum_buildH(n_states, mu, sigma, delta)
#     else:
#         raise Exception(f"Please select one of {model_type}")

#     phi_0 = _get_initial_state(n_states, start_width, I = I, prob=1, model_type = model_type, prior_type="Model")


def get_original_params(posterior_samples, response_width, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum"):
    if params_type == "Centralized":
        pass
    elif params_type == "NonCentralized":
        m = posterior_samples["m"]
        s = posterior_samples["s"]

        m_si = posterior_samples["m_si"]
        s_si = posterior_samples["s_si"]


        mu_r = posterior_samples["mu_r"]
        sigma_r = posterior_samples["sigma_r"]

        posterior_samples["mu"] = m[:,None,None] + s[:,None,None] * mu_r
        posterior_samples["sigma"] = jax.nn.softplus(m_si[:,None,None] + s_si[:,None,None] * sigma_r) #(m[:,None,None] + s[:,None,None] * sigma_r)**2 
    
    p_0 = posterior_samples["phi_init"]

    if model_type == "Markov":
        p_0 = p_0.transpose(0, 1, 2, 4, 3)
        posterior_samples["sigma_final"] = npx.abs(posterior_samples["mu"]) + posterior_samples["sigma"] 

    elif model_type == "Quantum":
        p_0 = p_0.transpose(0, 1, 2, 4, 3)**(1/2)
        posterior_samples["sigma_final"] = posterior_samples["sigma"]

    posterior_samples["phi_0"] = npx.pad(p_0, ((0,0),(0,0),(0,0),(response_width,response_width),(0,0)))

    # Commenting these out so that point estimate is not calculated. That way Bayesian model checks can be used.
    # posterior_samples["mu"] = posterior_samples["mu"].mean(axis=0, keepdims=True)
    # posterior_samples["sigma_final"] = posterior_samples["sigma_final"].mean(axis=0, keepdims=True)
    # posterior_samples["phi_0"] = posterior_samples["phi_0"].mean(axis=0, keepdims=True)

    return posterior_samples
