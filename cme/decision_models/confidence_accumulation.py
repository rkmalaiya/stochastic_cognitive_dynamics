import jax.numpy as npx
import jax.scipy as sci
import numpyro as pyro
import cme.decision_models.quantum_discrete as qd
import cme.decision_models.diffusion_discrete as dd
import jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SA, HMCECS, Predictive
from jax import random
import arviz as az 

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from cme.utils import common_logging as cl
log = cl.get_logger("confidence_accumulation")

pyro.set_platform("cpu")
pyro.set_host_device_count(64)
pyro.enable_x64()

_rng_key = random.PRNGKey(0)
_rng_key, _rng_key_ = random.split(_rng_key)

def centralized_parameters(I):
    """
    I: Number of participants
    
    """
    mu_m =  pyro.sample(f"mu_m", dist.Normal(0,1))
    mu_s =  pyro.sample(f"mu_s", dist.HalfNormal(2))
    with pyro.plate('I', I, dim=-2):

        mu = pyro.sample("mu", dist.Normal(mu_m,mu_s)) # Drift Rate
        sigma = pyro.sample("sigma", dist.Normal(1,2)) # Diffusion Rate
        
    return mu, sigma

def non_centralized_parameters(I):
    """
    I: Number of participants
    
    """
    m = pyro.sample("m", dist.Normal(0,1))
    s = pyro.sample("s", dist.HalfNormal(1))

    with pyro.plate('I', I, dim=-2):
        mu_r = pyro.sample("mu_r", dist.Normal(2,1)) # Drift Rate
        sigma_r = pyro.sample("sigma_r", dist.Normal(1,1)) # Diffusion Rate

        mu = pyro.deterministic("mu", m + s * mu_r)
        sigma = pyro.deterministic("sigma", m + s * sigma_r) 

    return mu, sigma

def _timestep_transition_matrix(n, T_delta, Mn):

    T_i = []
    for n_i, T_delta_i in zip(n, T_delta):
        T_i_j = []
        for n_i_j, T_delta_i_j in zip(n_i, T_delta_i):
            T_nt = npx.linalg.matrix_power(Mn @ T_delta_i_j, n_i_j.astype(int).item() - 1) # we need to vectorize this function
            T_i_j.append(T_nt)
        
        T_i.append(T_i_j)
    
    T_t = T_delta @ npx.asarray(T_i)
        
    return T_t

def _get_transition_matrix(intensity_matrix, RT, delta=None, Mn = None, transition_type="RT|TIMESTEP"):
   
    if transition_type == "RT":
        T_t = sci.linalg.expm(intensity_matrix * (RT[...,None,None] if not npx.isscalar(RT) else RT))
    elif transition_type == "TIMESTEP":
        ns=npx.ceil(RT/delta)
        T_delta = sci.linalg.expm(intensity_matrix * delta)
        T_t = _timestep_transition_matrix(ns, T_delta, Mn)
    else:
        raise Exception(f"Please select one of {transition_type}")

    return T_t

def _get_measurement_matrix(n_states, start_width, prob=0.5, model_type = "Markov|Quantum"):
    if model_type == "Markov":
        Mc, Mw, Mn = dd._get_measurement_matrix(n_states, start_width, prob)
    elif model_type == "Quantum":
        Mc, Mw, Mn = qd._get_measurement_matrix(n_states, start_width, prob)
    else:
        raise Exception(f"Please select one of {model_type}")
    return Mc, Mw, Mn

def _get_initial_state(n_states, start_width, I = 1, prob=1, model_type = "Markov|Quantum"):
    if model_type == "Markov":
        phi_0 = dd._get_initial_state(n_states, start_width, I, prob)   
    elif model_type == "Quantum":
        phi_0 = qd._get_initial_state(n_states, start_width, I, prob)
    else:
        raise Exception(f"Please select one of {model_type}")
    
    return phi_0   

def perform_state_transition(intensity_matrix, RT_s, RA_s, Mc, Mw, Mn, phi_0, delta, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
        
    if likelihood_type=="SINGLE":
        RT=RT_s
        T_t = _get_transition_matrix(intensity_matrix, RT=RT, delta=delta, Mn=Mn, transition_type=transition_type)
        
    elif likelihood_type=="JOINT":
        RT_1 = RT_s[0]
        T_t_1 = _get_transition_matrix(intensity_matrix, RT=RT_1, delta=delta, Mn=Mn, transition_type=transition_type)

        RT_2 = RT_s[1]
        T_t_2 = _get_transition_matrix(intensity_matrix, RT=RT_2, delta=delta, Mn=Mn, transition_type=transition_type)

        phi_t_1_c = T_t_2 @ Mc @ T_t_1 
        phi_t_1_w = T_t_2 @ Mw @ T_t_1 
        
        RA_1 = RA_s[0]
        T_t = npx.where(RA_1[...,None,None]==1, phi_t_1_c, phi_t_1_w)
        
    phi_t = T_t @ phi_0
    return phi_t

def get_mean_confidence(n_states, intensity_matrix, phi_0, delta, Mn, t, transition_type="RT|TIMESTEP"):
    
    """
    n_states: int
    intensity_matrix: float|Complex[IxJ]
    t: float
    phi_0: float[Mx1]
    delta: int
    """

    T_t = _get_transition_matrix(intensity_matrix, RT=t, delta=delta, Mn=Mn, transition_type=transition_type)

    Mid = (n_states+1)//2
    mv = npx.arange(-(Mid-1), (Mid))
    mean_confidence = mv @ T_t @ phi_0

    return mean_confidence

def likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT", model_type="Markov|Quantum"):

    phi_t = perform_state_transition(intensity_matrix, RT_s, RA_s, Mc, Mw, Mn, phi_0, delta, 
                                     transition_type=transition_type, likelihood_type=likelihood_type)

    if likelihood_type == "SINGLE":
        RA = RA_s
        RT = RT_s
    elif likelihood_type == "JOINT":
        RA = RA_s[1]
        RT = RT_s[0] * RT_s[1] # So that even if a single RT is 0, the likelihood for that participant becomes 0
    else:
        raise Exception(f"Please select one of {likelihood_type} values for likelihood_type variable")

    P_t_c = (Mc @ phi_t)
    P_t_w = (Mw @ phi_t)
    P_t = npx.where(RA[...,None,None]==1, P_t_c, P_t_w)

    if model_type == "Markov":
        #P_t_c = (Mc @ phi_t).sum(axis=(-2,-1))
        #P_t_w = (Mw @ phi_t).sum(axis=(-2,-1))

        P_t = P_t.sum(axis=(-2,-1)) # Adding over states
        
        #P_t = npx.where(RA==1, P_t_c, P_t_w)
        
    elif model_type == "Quantum":
        #P_t_c = (npx.abs(Mc @ phi_t)**2).sum(axis=(-2,-1))
        #P_t_w = (npx.abs(Mw @ phi_t)**2).sum(axis=(-2,-1))

        P_t = (npx.abs(P_t)**2).sum(axis=(-2,-1)) #Adding over states
        

        #P_t = npx.where(RA==1, P_t_c, P_t_w)
        
    else:
        raise Exception(f"Please select one of {model_type}")
    
        
    P_t = npx.where(RT <= 0, 0, P_t)

    return npx.log(npx.sum(P_t)) # summing over all participants and trials

def model(n_states, start_width, delta, RA_s, RT_s, measurement_prob, params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    if likelihood_type == "SINGLE":
        I, _ = RA_s.shape
    elif likelihood_type == "JOINT":
        if RA_s[0].shape != RA_s[1].shape:
            raise Exception("All Responses should be equal shaped")
        I, _ = RA_s[0].shape
    else:
        raise Exception(f"Please select one of {likelihood_type}")

    if params_type == "Centralized":
        mu, sigma = centralized_parameters(I)
    elif params_type == "NonCentralized":
        mu, sigma = non_centralized_parameters(I)
    else:
        raise Exception(f"Please select one of {params_type}")

    if model_type == "Markov":
        sigma = pyro.deterministic("sigma_final",mu + sigma**2) # Sigma needs to be larger than mu and Sigma cannot be negative
        intensity_matrix = dd._buildK(n_states, mu, sigma)

    elif model_type == "Quantum":
        sigma = pyro.deterministic("sigma_final",sigma**2) # Sigma cannot be negative
        intensity_matrix = -1j * qd._buildH(n_states, mu, sigma)
    else:
        raise Exception(f"Please select one of {model_type}")

    phi_0 = pyro.deterministic("phi_0", _get_initial_state(n_states, start_width, I = I, prob=1, model_type = model_type))
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, start_width=start_width, prob=measurement_prob, model_type = model_type)

    if RT_s is not None:
        likl = likelihood(intensity_matrix, phi_0, delta, RT_s, RA_s, Mc, Mw, Mn, 
                          transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
        pyro.factor(f"likelihood", likl)


def predictive_model(RT_pred, n_states, start_width, delta, measurement_prob, phi_0, RA, 
                     drift_rate, diffusion_rate, 
                     model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
    Mc, Mw, Mn = _get_measurement_matrix(n_states = n_states, start_width=start_width, prob=measurement_prob, model_type = model_type)
    
    if model_type == "Markov":
        intensity_matrix = dd._buildK(n_states, drift_rate, diffusion_rate)

    elif model_type == "Quantum":
        intensity_matrix = -1j * qd._buildH(n_states, drift_rate, diffusion_rate)
    else:
        raise Exception(f"Please select one of {model_type}")
    
    #I, J = RA.shape
    
    #with pyro.plate('I', I, dim=-2):
    #    with pyro.plate('J', J, dim=-1):
    #        RT_pred = pyro.sample("RT_pred", dist.LogNormal(0, 1.5))
    
    likl = likelihood(intensity_matrix, phi_0, delta, RT_pred, RA, Mc, Mw, Mn, 
                      transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type)
    #pyro.deterministic("likl_prnt", likl)
    #pyro.factor("likelihood", likl)
    return likl

def sample_posterior_params(DT, X, n_states, start_width, delta, measurement_prob,
                            num_warmup=100, samples_n=500, num_chains=4, batch_size=2,  
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):

    #kernel = HMCECS(NUTS(model), num_blocks=10)
    #mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    #mcmc_chain.run(_rng_key, n_states, start_width,  sigma, tau, DT, X, I, J, s_0, batch_size=batch_size, extra_fields=('hmc_state',))

    kernel = NUTS(model)
    mcmc_chain = MCMC(kernel, num_warmup=num_warmup, num_samples=samples_n, num_chains=num_chains)
    mcmc_chain.run(_rng_key, n_states, start_width,  delta, X, DT, measurement_prob, 
                   params_type = params_type, transition_type=transition_type, 
                   likelihood_type=likelihood_type, model_type=model_type,
                   extra_fields=('potential_energy',))

    #post_likl = mcmc_chain.get_extra_fields()['hmc_state'].potential_energy
    #post_likl = mcmc_chain.get_extra_fields()['potential_energy']
    return mcmc_chain#, post_likl

def predictive_mcmc_fn(n_states, start_width, delta, measurement_prob, X, 
                       drift_rate_samples, diffusion_rate_samples, phi_0_samples,
                       params_type, model_type, transition_type, likelihood_type):
    
    predictive_samples = []
    for i, (drift_rate, diffusion_rate, phi_0), in enumerate(zip(drift_rate_samples, diffusion_rate_samples, phi_0_samples)):
        if likelihood_type == "SINGLE":
            kernel = NUTS(potential_fn= lambda RT_pred: 
                                        predictive_model(RT_pred, n_states, start_width, delta, measurement_prob, phi_0, 
                                                        X, drift_rate, diffusion_rate,
                                                        transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type,))
            pred_shape = 4, *X.shape
            predictive_mcmc = MCMC(kernel, num_warmup=300, num_samples=100, num_chains=4)
            predictive_mcmc.run(_rng_key, init_params=stats.lognorm(s=1).rvs((pred_shape)),
                            extra_fields=('potential_energy',)
                            )
            

        elif likelihood_type == "JOINT":
            kernel = NUTS(potential_fn= lambda RT_pred_s: 
                            predictive_model(RT_pred_s, n_states, start_width, delta, measurement_prob, phi_0, 
                                            X, drift_rate, diffusion_rate,
                                            transition_type=transition_type, likelihood_type=likelihood_type, model_type=model_type,))
            pred_shape = 4, 2, *X[0].shape
            predictive_mcmc = MCMC(kernel, num_warmup=300, num_samples=200, num_chains=4)
            predictive_mcmc.run(_rng_key, init_params=stats.lognorm(s=1).rvs((pred_shape)),
                        extra_fields=('potential_energy',)
                        )
        
        predictive_samples.append({"drift_rate":drift_rate, "diffusion_rate":diffusion_rate, "predictive_chain":az.from_numpyro(predictive_mcmc)})

    return predictive_samples

def sample_prior_pred_params(n_states, start_width, delta, measurement_prob, X, 
                        n_samples=100, 
                        params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):
    
    prior_predictive = Predictive(model, num_samples=n_samples)    
    prior_samples = prior_predictive(_rng_key, n_states, start_width,  delta, X, None, measurement_prob,
                                    params_type = params_type, transition_type=transition_type, 
                                    likelihood_type=likelihood_type, model_type=model_type)
    
    drift_rate_samples = prior_samples["mu"]
    diffusion_rate_samples = prior_samples["sigma_final"]
    phi_0_samples = prior_samples["phi_0"]
    
    predictive_samples = predictive_mcmc_fn(n_states, start_width, delta, measurement_prob, X, 
                                            drift_rate_samples, diffusion_rate_samples, phi_0_samples,
                                            params_type, model_type, transition_type, likelihood_type)
    return predictive_samples

def sample_post_pred_params(n_states, start_width, delta, measurement_prob, X, 
                            drift_rate_samples, diffusion_rate_samples, phi_0_samples,
                            n_samples=50, 
                            params_type = "Centralized|NonCentralized", model_type="Markov|Quantum", transition_type="RT|TIMESTEP", likelihood_type="SINGLE|JOINT"):

    predictive_samples = predictive_mcmc_fn(n_states, start_width, delta, measurement_prob, X, 
                                            drift_rate_samples, diffusion_rate_samples, phi_0_samples,
                                            params_type, model_type, transition_type, likelihood_type)
    return predictive_samples


if __name__ == "__main__":

    n_states, start_width, delta, measurement_prob, mu, sigma, I, J = 7, 3, 1, 0.3, npx.asarray([[1]]), npx.asarray([[1]]), 5, 3
    m_Mc, m_Mw, m_Mn = _get_measurement_matrix(n_states, 1, prob=measurement_prob, model_type = "Markov")
    q_Mc, q_Mw, q_Mn = _get_measurement_matrix(n_states, 1, prob=measurement_prob, model_type = "Quantum")
    
    log.debug("Constant Drift Rate - Mean Confidence 1")

    intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
    intensity_matrix_quantum = -1j * qd._buildH(n_states, mu, sigma)

    phi_0_markov = _get_initial_state(n_states, start_width,model_type="Markov")
    phi_0_quantum = _get_initial_state(n_states, start_width,model_type="Quantum")

    mean_conf_markov = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_markov, 
                                           phi_0=phi_0_markov, delta=delta, Mn=m_Mn, t=10, transition_type="RT")
    mean_conf_quantum = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_markov, 
                                            phi_0=phi_0_quantum, delta=delta, Mn=q_Mn, t=10, transition_type="RT")

    print(mean_conf_markov)
    print(mean_conf_quantum)

    log.debug("Constant Drift Rate - Mean Confidence 2")
    mean_conf_markov_arr = []
    mean_conf_quantum_arr = []
    for t in range(1, 100):
        mean_conf_markov = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_markov, 
                                           phi_0=phi_0_markov, delta=delta, Mn=m_Mn, t=t, transition_type="RT")
        mean_conf_markov_arr.append(mean_conf_markov.squeeze())
        mean_conf_quantum = get_mean_confidence(n_states, intensity_matrix=intensity_matrix_quantum, 
                                            phi_0=phi_0_quantum, delta=delta, Mn=q_Mn, t=t, transition_type="RT")
        mean_conf_quantum_arr.append(mean_conf_quantum.squeeze())

    pd.Series(npx.asarray(mean_conf_markov_arr), name="Markov").plot()
    pd.Series(npx.asarray(mean_conf_quantum_arr), name="Quantum").plot()
    plt.legend()
    plt.show()

    log.debug("Constant Drift Rate - Likelihood 1")

    likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                            RT_s=npx.asarray([[10]]), RA_s=npx.asarray([[1]]),  
                            Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                            transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
    
    likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                            RT_s=npx.asarray([[10]]), RA_s=npx.asarray([[1]]),  
                            Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                            transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
    
    print(likl_markov)
    print(likl_quantum)

    log.debug("Constant Drift Rate - Likelihood 2")

    likl_markov_arr = []
    likl_quantum_arr = []

    for t in range(1,100):
        likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]),  
                                Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
        likl_markov_arr.append(likl_markov.squeeze())

        
        likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]),
                                Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
        likl_quantum_arr.append(likl_quantum.squeeze())
    
    pd.Series(npx.asarray(likl_markov_arr), name="Markov").plot()
    pd.Series(npx.asarray(likl_quantum_arr), name="Quantum").plot()
    plt.legend()
    plt.show()

    log.debug("Constant Drift Rate - Likelihood 3")

    likl_markov_arr = []
    likl_quantum_arr = []

    for t in range(1,100):
        likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[0]]),
                                Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
        likl_markov_arr.append(likl_markov.squeeze())

        
        likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[0]]), 
                                Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
        likl_quantum_arr.append(likl_quantum.squeeze())
    
    pd.Series(npx.asarray(likl_markov_arr), name="Markov").plot()
    pd.Series(npx.asarray(likl_quantum_arr), name="Quantum").plot()
    plt.legend()
    plt.show()

    log.debug("Constant Drift Rate - Likelihood 4")

    for mu, sigma in zip([npx.asarray([[1]]), npx.asarray([[0.5]]), npx.asarray([[10]]), npx.asarray([[-1]])],[npx.asarray([[1]]), npx.asarray([[10]]), npx.asarray([[0.05]]), npx.asarray([[1]])]):
        likl_markov_arr = []
        likl_quantum_arr = []
        intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
        intensity_matrix_quantum = -1j * qd._buildH(n_states, mu, sigma)

        for t in range(1,100):
            likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                    transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
            likl_markov_arr.append(likl_markov.squeeze())

            
            likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                    transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
            likl_quantum_arr.append(likl_quantum.squeeze())
        
        pd.Series(npx.asarray(likl_markov_arr), name=f"Markov:{mu}, {sigma}").plot()
        pd.Series(npx.asarray(likl_quantum_arr), name=f"Quantum:{mu}, {sigma}").plot()
        plt.legend()
        plt.show()

    log.debug("Constant Drift Rate - Likelihood 5")

    for mu, sigma in zip([npx.asarray([[1]]), npx.asarray([[0.5]]), npx.asarray([[10]]), npx.asarray([[-1]])],[npx.asarray([[1]]), npx.asarray([[10]]), npx.asarray([[0.05]]), npx.asarray([[1]])]):
        likl_markov_arr = []
        likl_quantum_arr = []
        intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
        intensity_matrix_quantum = -1j * qd._buildH(n_states, mu, sigma)

        for t in range(1,100):
            likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                    transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Markov")
            likl_markov_arr.append(likl_markov.squeeze())

            
            likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
                                    Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                    transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Quantum")
            likl_quantum_arr.append(likl_quantum.squeeze())
        
        pd.Series(npx.asarray(likl_markov_arr), name=f"Markov:{mu}, {sigma}").plot()
        plt.legend()
        plt.show()
        pd.Series(npx.asarray(likl_quantum_arr), name=f"Quantum:{mu}, {sigma}").plot()
        plt.legend()
        plt.show()

    log.debug("Constant Drift Rate - Prior 1")

    X = stats.bernoulli(0.5).rvs(size=(I,J))

    predictive_samples = sample_prior_pred_params(n_states=n_states,start_width=start_width,delta=delta,
                                                  measurement_prob=measurement_prob, X=X, n_samples=2,
                                                  params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="SINGLE" 
                                                 )
    # The predictive_samples contains posterior RT samples for each posterior parameter indexed by [0] below.
    #predictive_samples[0]["predictive_chain"]   
    #log.debug(az.summary(predictive_samples[0]["predictive_chain"]))
    
    df_plot = pd.DataFrame()
    for i, prior_predictive_sample in enumerate(predictive_samples):
        RT_pred = prior_predictive_sample["predictive_chain"]["posterior"]["Param:0"].values.reshape((-1, I, J))
        mean_rt_pred_s = RT_pred.mean(axis=(0))
        #lp_s = predictive_samples[0]["predictive_chain"]["sample_stats"]["lp"].values
        #lp_s = predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values
    #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
        #sns.relplot(x=mean_rt_pred, y=lp, col=i)
        #sns.kdeplot(x=mean_rt_pred_s.flatten(), hue=i)
        df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), 
                                                        prior = i))])
    sns.kdeplot(df_plot, x="mean_rt", hue="prior")
    plt.show()

    log.debug("Constant Drift Rate - Posterior Samples 1")


    X = stats.bernoulli(0.5).rvs(size=(I,J))
    RT = stats.lognorm(1,1).rvs(size=(I,J))
    post_chain = sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, delta=delta,measurement_prob=measurement_prob,
                                         num_warmup=100, samples_n=100,
                                         params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="SINGLE" 
                            )
    post_samples = post_chain.get_samples()
    #log.debug(az.summary(az.from_numpyro(post_chain)))

    log.debug("Constant Drift Rate - Posterior Samples - Joint - 1")


    X_s = [stats.bernoulli(0.5).rvs(size=(I,J)), stats.bernoulli(0.5).rvs(size=(I,J))]
    RT_s = [stats.lognorm(1,1).rvs(size=(I,J)), stats.lognorm(1,1).rvs(size=(I,J))]
    post_chain_joint = sample_posterior_params(RT_s, X_s, n_states=n_states, start_width=start_width, delta=delta,measurement_prob=measurement_prob,
                                                num_warmup=100, samples_n=100,
                                                params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="JOINT" 
                            )
    post_samples_joint = post_chain_joint.get_samples()


    log.debug("Constant Drift Rate - Post Predictive Samples 1")
    drift_rate_samples = post_samples["mu"][-2:,...]
    diffusion_rate_samples = post_samples["sigma_final"][-2:,...]
    phi_0_samples = post_samples["phi_0"][-2:,...]
    
    post_predictive_samples = sample_post_pred_params(n_states=n_states, start_width=start_width, delta=delta,measurement_prob=measurement_prob,
                                                 X=X, 
                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
                                                 params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="SINGLE"
                                                 )
    
    post_predictive_samples[0]["predictive_chain"]  
    #log.debug(az.summary(post_predictive_samples[0]["predictive_chain"]))

    df_plot = pd.DataFrame()
    for i, post_pred_sample in enumerate(post_predictive_samples):  #Iterating over each posterior distribution
        RT_pred = post_pred_sample["predictive_chain"]["posterior"]["Param:0"].values.reshape((-1, I, J))
        mean_rt_pred_s = RT_pred.mean(axis=(0))
        #lp_s = post_predictive_samples[0]["predictive_chain"]["sample_stats"]["lp"].values
        #lp_s = post_predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values
    
        #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
        #    sns.relplot(x=mean_rt_pred, y=lp, col=i, kind="point")
        df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), 
                                                        posterior = i))])
    sns.kdeplot(df_plot, x="mean_rt", hue="posterior")
    plt.show()


    log.debug("Constant Drift Rate - Post Predictive Samples - Joint - 1")
    drift_rate_samples = post_samples_joint["mu"][-2:,...]
    diffusion_rate_samples = post_samples_joint["sigma_final"][-2:,...]
    phi_0_samples = post_samples_joint["phi_0"][-2:,...]
    
    post_predictive_joint_samples = sample_post_pred_params(n_states=n_states, start_width=start_width, delta=delta,measurement_prob=measurement_prob,
                                                 X=X_s, 
                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
                                                 params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="JOINT"
                                                 )
    
    post_predictive_joint_samples[0]["predictive_chain"]  
    #log.debug(az.summary(post_predictive_samples[0]["predictive_chain"]))

    df_plot = pd.DataFrame()
    for i, post_pred_sample_joint in enumerate(post_predictive_joint_samples):  #Iterating over each posterior distribution
        RT_pred = post_pred_sample_joint["predictive_chain"]["posterior"]["Param:0"].values[:,:,0,...]
        RT_pred_1 = RT_pred.reshape((-1, I, J))
        RT_pred_2 = RT_pred.reshape((-1, I, J))
        mean_rt_pred_s = npx.asarray([RT_pred_1.mean(axis=(0)), RT_pred_2.mean(axis=(0))])
        #lp_s = post_predictive_samples[0]["predictive_chain"]["sample_stats"]["lp"].values
        #lp_s = post_predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values
    
        #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
        #    sns.relplot(x=mean_rt_pred, y=lp, col=i, kind="point")
        df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), 
                                                        posterior = i))])
    sns.kdeplot(df_plot, x="mean_rt", hue="posterior")
    plt.show()
    