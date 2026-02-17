import pytest
import cme.decision_models.confidence_accumulation as ca
import scipy.stats as stats
from collections import namedtuple
import jax.numpy as npx
from cme.utils import common_logging as cl
log = cl.get_logger("confidence_accumulation_test")

@pytest.fixture
def markov_constant_matrix():
    import numpy as np

    # Define your parameters
    # alpha is defined as -(beta_plus + beta_minus)
    beta_p = 0.6  # beta_plus
    beta_m = 0.4  # beta_minus
    a = -(beta_p + beta_m) # alpha

    K = np.array([
        [a,      beta_m, 0,      0,      0,      0,      0     ],
        [beta_p, a,      beta_m, 0,      0,      0,      0     ],
        [0,      beta_p, a,      beta_m, 0,      0,      0     ],
        [0,      0,      beta_p, a,      beta_m, 0,      0     ],
        [0,      0,      0,      beta_p, a,      beta_m, 0     ],
        [0,      0,      0,      0,      beta_p, a,      beta_p],
        [0,      0,      0,      0,      0,      beta_p, a     ]
    ])

    return(K)

@pytest.fixture
def quantum_constant_matrix():
    import numpy as np

    # Replace these with your actual numerical values
    m_3, m_2, m_1, m0, p1, p2, p3 = -3, -2, -1, 0, 1, 2, 3
    s2 = 0.5 # Example value for sigma squared

    H = np.array([
        [m_3, s2,  0,   0,   0,   0,   0  ],
        [s2,  m_2, s2,  0,   0,   0,   0  ],
        [0,   s2,  m_1, s2,  0,   0,   0  ],
        [0,   0,   s2,  m0,  s2,  0,   0  ],
        [0,   0,   0,   s2,  p1,  s2,  0  ],
        [0,   0,   0,   0,   s2,  p2,  s2 ],
        [0,   0,   0,   0,   0,   s2,  p3 ]
    ])

    return(H)



@pytest.fixture
def data_shape():
    return namedtuple("Data_Shape", ["I", "J"])(I=10, J=50)


@pytest.fixture
def model_constants():
    Model = namedtuple("Model", ["n_states", "start_width", "response_width", 
                                 "delta", "measurement_prob", "mu", "sigma",
                                 "m_Mc", "m_Mw", "m_Mn",
                                 "q_Mc", "q_Mw", "q_Mn"
                                 ])
    model = Model(
        n_states = 51, 
        start_width = None, 
        response_width = 5, 
        delta = 1, 
        measurement_prob = 0.25, 
        mu = npx.asarray([[1]]), 
        sigma = npx.asarray([[1]])
    )
    
    model.start_width = (model.n_states-2*model.response_width)
    model.m_Mc, model.m_Mw, model.m_Mn = ca._get_measurement_matrix(model.n_states, 1, prob=model.measurement_prob, model_type = "Markov")
    model.q_Mc, model.q_Mw, model.q_Mn = ca._get_measurement_matrix(model.n_states, 1, prob=model.measurement_prob, model_type = "Quantum")
    return model


@pytest.fixture
def get_markov_matrix(model_constants): 
    
    ws = model_constants.n_states//2
    tv = npx.arange(0,10,0.1)
    nt = tv.shape[0]
    Mid = int((model_constants.n_states+1)/2)
    mv = npx.arange(-(Mid-1),(Mid))

    #mu=0.5
    #var=2
    S0 = npx.zeros((model_constants.n_states,1))
    S0[(Mid-ws):(Mid+ws-1)] = 1
    S0 = S0/sum(S0)

    mk = npx.ones((model_constants.n_states,1))
    b = -model_constants.sigma*mk
    a = 0.5* (model_constants.sigma-model_constants.mu)*mk
    c = 0.5* (model_constants.sigma+model_constants.mu)*mk

    m = a.shape[0]
    K = npx.zeros((model_constants.n_states,model_constants.n_states))
    K[[0,1], 0] = npx.asarray([b[0], -b[0]]).T
    K[[-2,-1], -1] = npx.asarray([-b[-1], b[-1]]).T

    for k in range(1, m-1):
        v = [[k-1], [k], [k+1]]
        K[v,k] = npx.asarray([a[k], b[k], c[k]])

    return K


@pytest.fixture
def get_quantum_matrix(model_constants): 
# H = buildH(a,b,c)
# m = number of states  
# a = off diag left  
# b = diag  
# c = off diag right

    #ns = 7
    ws = model_constants.n_states//2 #3
    #tv = npx.arange(0,20,0.1)
    #nt = tv.shape[0]

    Mid = (model_constants.n_states+1)//2
    mv = npx.arange(-(Mid-1), (Mid))
    mu=1
    ap=1

    S0 = npx.zeros((model_constants.n_states,1))
    S0[Mid-ws-1:Mid+ws] = 1
    S0 = S0/npx.sqrt(S0.T @ S0)

    b=model_constants.mu*mv
    a=model_constants.sigma*npx.ones((model_constants.n_states,1))
    c=a

    H = npx.zeros((model_constants.n_states, model_constants.n_states))
    H[0,0] = b[0]
    H[0,1] = c[0]
    H[model_constants.n_states-1, model_constants.n_states-2] = a[model_constants.n_states-1]
    H[model_constants.n_states-1, model_constants.n_states-1] = b[model_constants.n_states-1]

    for k in range(1,model_constants.n_states-1):
        H[k,k-1] = a[k]
        H[k,k] = b[k]
        H[k,k+1] = c[k]

    return H



@pytest.fixture
def data_sim(data_shape):
    X = stats.bernoulli(0.5).rvs(size=(data_shape.I,data_shape.J))
    RT = stats.lognorm(1,1).rvs(size=(data_shape.I,data_shape.J))
    return namedtuple("Data", ["X", "RT"])(X=X, RT=RT)


# def test_VI(data_sim, model_constants):
#     post_chain = ca.sample_posterior_params_VI( data_sim.RT, data_sim.X, 
#                                                 n_states=model_constants.n_states, 
#                                                 start_width=model_constants.start_width, 
#                                                 response_width=model_constants.response_width, 
#                                                 delta=model_constants.delta,
#                                                 measurement_prob=model_constants.measurement_prob,
#                                                 num_warmup=200, samples_n=200,
#                                                 params_type="NonCentralized", 
#                                                 model_type="Markov", transition_type="TIMESTEP", 
#                                                 likelihood_type="SINGLE" 
#                         )
#     print(post_chain.keys())

class Test_Configuration:
    def test_transition_matrix_markov(model_constants, markov_constant_matrix):
        log.debug("Constant Drift Rate - Mean Confidence 1")

        intensity_matrix_markov = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Markov")
        assert markov_constant_matrix == intensity_matrix_markov
        
        
    def test_transition_matrix_quantum(model_constants, quantum_constant_matrix):
        log.debug("Constant Drift Rate - Mean Confidence 1")

        intensity_matrix_quantum = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Quantum")

    def test_initial_states_markov(model_constants):
        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Markov", prior_type="Upper")
        assert phi_0_markov.sum() == 1, "Markov States not summing up to 1"

    def test_initial_states_quantum(model_constants):
        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Quantum" , prior_type="Upper")
        assert phi_0_quantum.sum() == 1, "Quantum States not summing up to 1"

class Test_Confidence:
    def test_markov_confidence_for_internal(model_constants, get_markov_matrix):

        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Markov", prior_type="Upper")

        mean_conf_markov = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=get_markov_matrix, 
                                                    phi_0=phi_0_markov, delta=1, Mn=model_constants.q_Mn, 
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Markov")
        
        print(mean_conf_markov)


    def test_quantum_confidence_for_internal(model_constants, get_quantum_matrix):

        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Quantum" , prior_type="Upper")

        
        mean_conf_quantum = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=get_quantum_matrix, 
                                                    phi_0=phi_0_quantum, delta=1, Mn=model_constants.q_Mn, 
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Quantum")
        
        print(mean_conf_quantum)

    def test_integration_parameter_to_confidence_for_internal(model_constants):

        intensity_matrix_markov = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Markov")
        intensity_matrix_quantum = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Quantum")

        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Markov", prior_type="Upper")
        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Quantum" , prior_type="Upper")

        mean_conf_markov = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=intensity_matrix_markov, 
                                                    phi_0=phi_0_markov, delta=1, Mn=model_constants.q_Mn, 
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Markov")
        
        mean_conf_quantum = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=intensity_matrix_quantum, 
                                                    phi_0=phi_0_quantum, delta=1, Mn=model_constants.q_Mn, 
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Quantum")
        
        print(mean_conf_markov)
        print(mean_conf_quantum)

    def test_integration_parameter_to_confidence_for_internal(model_constants):

        intensity_matrix_markov = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Markov")
        intensity_matrix_quantum = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Quantum")

        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Markov", prior_type="Upper")
        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Quantum" , prior_type="Upper")

        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Markov", prior_type="Upper")
        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                            model_type="Quantum" , prior_type="Upper")

        mean_conf_markov = ca.get_mean_confidence(model_constants.n_states, intensity_matrix=intensity_matrix_markov, 
                                                phi_0=model_constants.phi_0_markov, delta=model_constants.delta, 
                                                Mn=model_constants.m_Mn, t=10, transition_type="RT", 
                                                likelihood_type="SINGLE", model_type="Markov")
        mean_conf_quantum = ca.get_mean_confidence(model_constants.n_states, intensity_matrix=intensity_matrix_quantum,
                                                phi_0=phi_0_quantum, delta=model_constants.delta, 
                                                Mn=model_constants.q_Mn, t=10, transition_type="RT", 
                                                likelihood_type="SINGLE", model_type="Quantum")

        print(mean_conf_markov)
        print(mean_conf_quantum)


# class Test_Likelihood:
#     log.debug("Constant Drift Rate - Likelihood 1")

#     likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
#                             RT_s=npx.asarray([[10, 20]]), RA_s=npx.asarray([[1, 0]]),  
#                             Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
#                             transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Markov")

#     likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
#                             RT_s=npx.asarray([[10, 30]]), RA_s=npx.asarray([[1, 0]]),  
#                             Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
#                             transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Quantum")

#     print(likl_markov)
#     print(likl_quantum)

#     log.debug("Constant Drift Rate - Likelihood 2")

#     likl_markov_arr = []
#     likl_quantum_arr = []

#     for t in range(1,100):
#         likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
#                                 RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]),  
#                                 Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
#                                 transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
#         likl_markov_arr.append(likl_markov.squeeze())

        
#         likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
#                                 RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]),
#                                 Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
#                                 transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
#         likl_quantum_arr.append(likl_quantum.squeeze())

#     pd.Series(npx.asarray(likl_markov_arr), name="Markov").plot()
#     pd.Series(npx.asarray(likl_quantum_arr), name="Quantum").plot()
#     plt.legend()
#     plt.show()


#     log.debug("Constant Drift Rate - Likelihood 3")

#     likl_markov_arr = []
#     likl_quantum_arr = []

#     for t in range(1,100):
#         likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
#                                 RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[0]]),
#                                 Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
#                                 transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
#         likl_markov_arr.append(likl_markov.squeeze())

        
#         likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
#                                 RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[0]]), 
#                                 Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
#                                 transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
#         likl_quantum_arr.append(likl_quantum.squeeze())

#     pd.Series(npx.asarray(likl_markov_arr), name="Markov").plot()
#     pd.Series(npx.asarray(likl_quantum_arr), name="Quantum").plot()
#     plt.legend()
#     plt.show()

#     log.debug("Constant Drift Rate - Likelihood 4")

#     for mu, sigma in zip([npx.asarray([[1]]), npx.asarray([[0.5]]), npx.asarray([[10]]), npx.asarray([[-1]])],[npx.asarray([[1]]), npx.asarray([[10]]), npx.asarray([[0.05]]), npx.asarray([[1]])]):
#         likl_markov_arr = []
#         likl_quantum_arr = []
#         intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
#         intensity_matrix_quantum = qd._buildH(n_states, mu, sigma)

#         for t in range(1,100):
#             likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
#                                     RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
#                                     Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
#                                     transition_type="RT", likelihood_type="SINGLE", model_type="Markov")
#             likl_markov_arr.append(likl_markov.squeeze())

            
#             likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
#                                     RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
#                                     Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
#                                     transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
#             likl_quantum_arr.append(likl_quantum.squeeze())
        
#         pd.Series(npx.asarray(likl_markov_arr), name=f"Markov:{mu}, {sigma}").plot()
#         pd.Series(npx.asarray(likl_quantum_arr), name=f"Quantum:{mu}, {sigma}").plot()
#         plt.legend()
#         plt.show()

#     log.debug("Constant Drift Rate - Likelihood 5")

#     for mu, sigma in zip([npx.asarray([[1]]), npx.asarray([[0.5]]), npx.asarray([[10]]), npx.asarray([[-1]])],[npx.asarray([[1]]), npx.asarray([[10]]), npx.asarray([[0.05]]), npx.asarray([[1]])]):
#         likl_markov_arr = []
#         likl_quantum_arr = []
#         intensity_matrix_markov = dd._buildK(n_states, mu, sigma)
#         intensity_matrix_quantum = qd._buildH(n_states, mu, sigma)

#         for t in range(-10,100):
#             likl_markov = likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
#                                     RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
#                                     Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
#                                     transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Markov")
#             likl_markov_arr.append(likl_markov.squeeze())

            
#             likl_quantum = likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
#                                     RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[1]]), 
#                                     Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
#                                     transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Quantum")
#             likl_quantum_arr.append(likl_quantum.squeeze())
        
#         pd.Series(npx.asarray(likl_markov_arr), name=f"Markov:{mu}, {sigma}").plot()
#         plt.legend()
#         plt.show()
#         pd.Series(npx.asarray(likl_quantum_arr), name=f"Quantum:{mu}, {sigma}").plot()
#         plt.legend()
#         plt.show()

# log.debug("Constant Drift Rate - Prior 1")

# X = stats.bernoulli(0.5).rvs(size=(I,J))
# RT = stats.lognorm(1,1).rvs(size=(I,J)) 

# predictive_samples = sample_prior_pred_params(n_states=n_states,start_width=start_width,response_width=response_width,
#                                                 delta=delta,
#                                                 measurement_prob=measurement_prob, X=X, RT=RT, 
#                                                 n_samples=10,data_samples=X.shape,
#                                                 params_type="Centralized", model_type="Quantum", transition_type="RT", 
#                                                 likelihood_type="SINGLE", sampling_type="GEN", 
#                                                 )
# df_samples = predictive_samples[0]["Samples"]
# df_sim_RT = predictive_samples[0]["Likelihood"]
# df_prior_all = pd.concat([samples["Samples"] for samples in predictive_samples])
# sns.kdeplot(df_prior_all, x="RT", hue="param_sample_id")
# plt.show()

# log.debug("Constant Drift Rate - Prior 2")

# X = stats.bernoulli(0.5).rvs(size=(I,J))
# RT = stats.lognorm(1,1).rvs(size=(I,J)) 

# predictive_samples = sample_prior_pred_params(n_states=n_states,start_width=start_width,response_width=response_width,
#                                                 delta=delta,
#                                                 measurement_prob=measurement_prob, X=X, RT=RT, n_samples=2,
#                                                 params_type="Centralized", model_type="Quantum", transition_type="RT", 
#                                                 likelihood_type="SINGLE", sampling_type="MCMC"
#                                                 )
# log.debug(f"Mean Rhat {az.rhat(predictive_samples[0]['predictive_chain'])['Param:0'].values.mean()}")     

# df_plot = pd.DataFrame()
# for i, prior_predictive_sample in enumerate(predictive_samples):

#     mean_rt_pred_s = prior_predictive_sample["predictive_chain"]["posterior"]["Param:0"].values.mean(axis=(-2,-1))
#     lp_s = prior_predictive_sample["predictive_chain"]["sample_stats"]["lp"].values

#     df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), lp = lp_s.flatten(),
#                                                     prior = i))])
# sns.relplot(
#     df_plot,
#     x="mean_rt",
#     y="lp",
#     hue="prior"
#     )
# plt.show()

# log.debug("Constant Drift Rate - Posterior Samples 1")

# X = stats.bernoulli(0.5).rvs(size=(I,J))
# RT = stats.lognorm(1,1).rvs(size=(I,J))
# post_chain = sample_posterior_params(RT, X, n_states=n_states, start_width=start_width, response_width=response_width, 
#                                         delta=delta,measurement_prob=measurement_prob,
#                                         num_warmup=10, samples_n=10,
#                                         params_type="Centralized", model_type="Quantum", transition_type="TIMESTEP", likelihood_type="SINGLE" 
#                         )
# post_samples = post_chain.get_samples()

# log.debug("Constant Drift Rate - Post Predictive Samples 1")
# drift_rate_samples = post_samples["mu"][-5:,...]
# diffusion_rate_samples = post_samples["sigma_final"][-5:,...]
# phi_0_samples = post_samples["phi_0"][-5:,...]

# post_predictive_samples = sample_post_pred_params(n_states=n_states, start_width=start_width, 
#                                                     response_width=response_width,
#                                                     delta=delta,measurement_prob=measurement_prob,
#                                                 X=X, 
#                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
#                                                 params_type="Centralized", model_type="Quantum", transition_type="RT", likelihood_type="SINGLE", sampling_type="MCMC"
#                                                 )

# log.debug(f"Mean Rhat {az.rhat(post_predictive_samples[0]['predictive_chain'])['Param:0'].values.mean()}")  
# #log.debug(az.summary(post_predictive_samples[0]["predictive_chain"]))

# df_plot = pd.DataFrame()
# for i, post_pred_sample in enumerate(post_predictive_samples):  #Iterating over each posterior distribution
#     #RT_pred = post_pred_sample["predictive_chain"]["posterior"]["Param:0"].values.reshape((-1, I, J))
#     #mean_rt_pred_s = RT_pred.mean(axis=(0))
#     mean_rt_pred_s = post_pred_sample["predictive_chain"]["posterior"]["Param:0"].values.mean(axis=(-2,-1))
#     lp_s = post_pred_sample["predictive_chain"]["sample_stats"]["lp"].values
#     #lp_s = post_predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values

#     #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
#     #    sns.relplot(x=mean_rt_pred, y=lp, col=i, kind="point")
#     df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), lp = lp_s.flatten(),
#                                                     posterior = i))])
# #sns.kdeplot(df_plot, x="mean_rt", hue="posterior")
# sns.relplot(
#             df_plot,
#             x="mean_rt",
#             y="lp",
#             hue="posterior"
#             )
# plt.show()

# log.debug("Constant Drift Rate - Post Predictive Samples 2")
# drift_rate_samples = post_samples["mu"][-5:,...]
# diffusion_rate_samples = post_samples["sigma_final"][-5:,...]
# phi_0_samples = post_samples["phi_0"][-5:,...]

# post_predictive_samples = sample_post_pred_params(n_states=n_states, response_width=response_width, delta=delta,measurement_prob=measurement_prob,
#                                                 X=X, 
#                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
#                                                 RT=RT,
#                                                 params_type="Centralized", model_type="Quantum", transition_type="RT", 
#                                                 likelihood_type="SINGLE", sampling_type="GEN"
#                                                 )

# df_pred_all = pd.concat([samples["Samples"] for samples in post_predictive_samples])
# sns.lineplot(df_pred_all, x="RT", y="logp", hue="param_sample_id")
# sns.kdeplot(df_pred_all, x="RT", hue="param_sample_id")
# sns.histplot(df_pred_all, x="RT", hue="param_sample_id", multiple="dodge",element="bars")

# #df_samples = post_predictive_samples[0]["Samples"]
# #sns.kdeplot(df_samples.assign(hue = 
# #                              lambda df: df.mu.astype(str) + df.sigma.astype(str) + df.weighted_sample.astype(str)), 
# #            x="RT", hue="hue", legend=False)
# #plt.xlim(0,10) # because RT_max is set as 1000

# log.debug("Constant Drift Rate - Posterior Samples - Joint - 1")

# X_s = [stats.bernoulli(0.5).rvs(size=(I,J)), stats.bernoulli(0.5).rvs(size=(I,J))]
# RT_s = [stats.lognorm(1,1).rvs(size=(I,J)), stats.lognorm(1,1).rvs(size=(I,J))]
# post_chain_joint = sample_posterior_params(RT_s, X_s, n_states=n_states, start_width=start_width, response_width=response_width, 
#                                             delta=delta,measurement_prob=measurement_prob,
#                                             num_warmup=100, samples_n=100,
#                                             params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="JOINT" 
#                         )
# post_samples_joint = post_chain_joint.get_samples()


# log.debug("Constant Drift Rate - Post Predictive Samples - Joint - 1")
# drift_rate_samples = post_samples_joint["mu"][-2:,...]
# diffusion_rate_samples = post_samples_joint["sigma_final"][-2:,...]
# phi_0_samples = post_samples_joint["phi_0"][-2:,...]

# post_predictive_joint_samples = sample_post_pred_params(n_states=n_states, response_width=response_width, delta=delta,measurement_prob=measurement_prob,
#                                                 X=X_s, 
#                                                 drift_rate_samples=drift_rate_samples, diffusion_rate_samples=diffusion_rate_samples, phi_0_samples=phi_0_samples,
#                                                 params_type="NonCentralized", model_type="Quantum", transition_type="RT", likelihood_type="JOINT"
#                                                 )

# post_predictive_joint_samples[0]["predictive_chain"]  
# #log.debug(az.summary(post_predictive_samples[0]["predictive_chain"]))

# df_plot = pd.DataFrame()
# for i, post_pred_sample_joint in enumerate(post_predictive_joint_samples):  #Iterating over each posterior distribution
#     RT_pred = post_pred_sample_joint["predictive_chain"]["posterior"]["Param:0"].values[:,:,0,...]
#     RT_pred_1 = RT_pred.reshape((-1, I, J))
#     RT_pred_2 = RT_pred.reshape((-1, I, J))
#     mean_rt_pred_s = npx.asarray([RT_pred_1.mean(axis=(0)), RT_pred_2.mean(axis=(0))])
#     #lp_s = post_predictive_samples[0]["predictive_chain"]["sample_stats"]["lp"].values
#     #lp_s = post_predictive_samples[0]["predictive_chain"]["posterior"]["likl_prnt"].values

#     #for i, (mean_rt_pred, lp) in enumerate(zip(mean_rt_pred_s, lp_s)):
#     #    sns.relplot(x=mean_rt_pred, y=lp, col=i, kind="point")
#     df_plot = pd.concat([df_plot, pd.DataFrame(dict(mean_rt=mean_rt_pred_s.flatten(), 
#                                                     posterior = i))])
# sns.kdeplot(df_plot, x="mean_rt", hue="posterior")
# plt.show()
