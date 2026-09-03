import pytest
import arviz as az
import cme.decision_models.confidence_accumulation as ca
import cme.inference.mcmc as inf_mcmc
import scipy.stats as stats
from collections import namedtuple
import jax
import jax.numpy as npx
import numpy as np
from cme.utils import common_logging as cl
log = cl.get_logger("confidence_accumulation_test")


@pytest.fixture
def model_constants():
    n_states = 7
    response_width = 2
    Model = namedtuple("Model", ["n_states", "start_width", "response_width", 
                                 "delta", "measurement_prob", "mu", "sigma",
                                 ])
    model = Model(
        n_states = n_states, 
        response_width = response_width, 
        start_width = (n_states-2*response_width),
        delta = 1, 
        measurement_prob = 0.25, 
        mu = npx.asarray([[2]]), 
        sigma = npx.asarray([[1]])
    )
        # model.start_width = (model.n_states-2*model.response_width)
    # model.m_Mc, model.m_Mw, model.m_Mn = ca._get_measurement_matrix(model.n_states, 1, prob=model.measurement_prob, model_type = "Markov")
    # model.q_Mc, model.q_Mw, model.q_Mn = ca._get_measurement_matrix(model.n_states, 1, prob=model.measurement_prob, model_type = "Quantum")

    return model

@pytest.fixture
def measurement_matrix_correct(model_constants):
    M = np.array([
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, model_constants.measurement_prob, 0 ],
        [0, 0, 0, 0, 0, 0, model_constants.measurement_prob ]
    ])
    return M

@pytest.fixture
def measurement_matrix_incorrect(model_constants):
    M = np.array([
        [model_constants.measurement_prob, 0, 0, 0, 0, 0, 0 ],
        [0, model_constants.measurement_prob, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0 ]
    ])
    return M

@pytest.fixture
def measurement_matrix_noresp(model_constants):
    M = np.array([
        [1-model_constants.measurement_prob, 0, 0, 0, 0, 0, 0 ],
        [0, 1-model_constants.measurement_prob, 0, 0, 0, 0, 0 ],
        [0, 0, 1, 0, 0, 0, 0 ],
        [0, 0, 0, 1, 0, 0, 0 ],
        [0, 0, 0, 0, 1, 0, 0 ],
        [0, 0, 0, 0, 0, 1-model_constants.measurement_prob, 0 ],
        [0, 0, 0, 0, 0, 0, 1-model_constants.measurement_prob ]
    ])
    return M

@pytest.fixture
def markov_constant_matrix():
    
    # Define your parameters
    # alpha is defined as -(beta_plus + beta_minus)
    beta_p = 1.5  # beta_plus
    beta_m = -0.5  # beta_minus
    a = (beta_p + beta_m) # alpha

    K = np.array([
        [-a,      beta_m, 0,      0,      0,      0,      0     ],
        [a,     -a,      beta_m,  0,      0,      0,      0     ],
        [0,      beta_p, -a,      beta_m, 0,      0,      0     ],
        [0,      0,      beta_p, -a,      beta_m, 0,      0     ],
        [0,      0,      0,      beta_p, -a,      beta_m, 0     ],
        [0,      0,      0,      0,      beta_p, -a,      a     ],
        [0,      0,      0,      0,      0,      beta_p,  -a    ]
    ])

    return(K)

@pytest.fixture
def quantum_constant_matrix():
    import numpy as np

    # Replace these with your actual numerical values
    m_3, m_2, m_1, m0, p1, p2, p3 = -6, -4, -2, 0, 2, 4, 6
    s2 = 1

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
    return namedtuple("Data_Shape", ["I", "J"])(I=2, J=5)


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
    X = stats.bernoulli(0.5).rvs(size=(data_shape.I,data_shape.J), random_state=1)
    RT = np.ceil(stats.lognorm(1,1).rvs(size=(data_shape.I,data_shape.J), random_state=1))
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

def test_MCMC(data_sim, model_constants):
    post_chain = inf_mcmc.sample_posterior_params( data_sim.RT, data_sim.X,
                                                n_states=model_constants.n_states,
                                                start_width=model_constants.start_width,
                                                response_width=model_constants.response_width,
                                                delta=model_constants.delta,
                                                measurement_prob=model_constants.measurement_prob,
                                                num_warmup=10, samples_n=10, num_chains=1,
                                                params_type="Centralized",
                                                model_type="Quantum", transition_type="TIMESTEP",
                                                likelihood_type="SINGLE"
                        )
    post_samples = post_chain.get_samples()
    mcmc_diagnostics = post_chain.get_extra_fields()
    arviz_data = az.from_numpyro(post_chain)
    assert npx.all(npx.asarray([key in post_samples for key in ["mu", "sigma_final", "phi_0"]])), "MCMC parameters missing"
    assert post_samples["mu"].shape == (10, data_sim.X.shape[0], 1), "MCMC samples not as expected"
    assert npx.all(npx.asarray([key in mcmc_diagnostics for key in ["num_steps", "accept_prob", "diverging", "adapt_state.step_size"]])), "MCMC diagnostics missing"
    assert npx.all(npx.asarray([key in arviz_data.sample_stats for key in ["n_steps", "acceptance_rate", "diverging", "step_size"]])), "MCMC diagnostics missing from ArviZ data"
    assert arviz_data.sample_stats["n_steps"].shape == (1,10), "ArviZ MCMC diagnostic shape not as expected" # num_chains x samples_n

class Test_Configuration:
    def test_timestep_transition_matrix(self):
        n = npx.asarray([[1, 2, 3, 1, 2, 5, 4, 1, 3],
                         [7, 8, 10, 15, 11, 12, 16, 10, 18]])
        T_delta = npx.asarray([[[[0.8, 0.2],
                                  [0.1, 0.9]]],
                               [[[0.7, 0.3],
                                  [0.2, 0.8]]]])
        Mn = npx.asarray([[0.9, 0.0],
                          [0.0, 0.8]])

        T_t = ca._timestep_transition_matrix(n, T_delta, Mn)
        T_expected = npx.asarray([[T_delta_i[0,...] @ np.linalg.matrix_power(Mn @ T_delta_i[0,...], n_i_j - 1)
                                   for n_i_j in n_i]
                                  for n_i, T_delta_i in zip(np.asarray(n), np.asarray(T_delta))])
        T_shared = T_delta[1,0,...] @ np.linalg.matrix_power(Mn @ T_delta[0,0,...], int(n[1,0].item()) - 1)

        assert npx.allclose(T_t, T_expected), "Timestep transition matrix not as expected"
        assert not npx.allclose(T_t[1,0,...], T_shared), "T_delta unexpectedly shared between participants"

    def test_timestep_transition_state(self):
        n = npx.asarray([[1, 2, 3, 1],
                         [2, 4, 1, 3]]) # I x J
        T_delta = npx.asarray([[[[0.8, 0.2],
                                  [0.1, 0.9]]],
                               [[[0.7, 0.3],
                                  [0.2, 0.8]]]]) # I x 1 x S x S
        Mn = npx.asarray([[0.9, 0.0],
                          [0.0, 0.8]]) # S x S
        phi_0 = npx.asarray([[[[0.7],
                                [0.3]]],
                             [[[0.2],
                                [0.8]]]]) # I x 1 x S x 1

        phi_t = ca._timestep_transition_state(n, T_delta, Mn, phi_0) # I x J x S x 1
        T_t = ca._timestep_transition_matrix(n, T_delta, Mn) # I x J x S x S
        phi_expected = T_t @ phi_0 # I x J x S x 1

        def _matrix_loss(T_delta):
            T_t = ca._timestep_transition_matrix(n, T_delta, Mn) # I x J x S x S
            phi_t = T_t @ phi_0 # I x J x S x 1
            return npx.sum(phi_t**2)

        def _state_loss(T_delta):
            phi_t = ca._timestep_transition_state(n, T_delta, Mn, phi_0) # I x J x S x 1
            return npx.sum(phi_t**2)

        matrix_grad = jax.grad(_matrix_loss)(T_delta) # I x 1 x S x S
        state_grad = jax.grad(_state_loss)(T_delta) # I x 1 x S x S

        assert npx.allclose(phi_t, phi_expected), "Timestep transition state not as expected"
        assert npx.allclose(state_grad, matrix_grad), "Timestep transition state gradients not as expected"

    def test_timestep_transition_state_quantum(self):
        n = npx.asarray([[1, 2, 4]]) # I x J
        T_delta = npx.asarray([[[[0.9+0.1j, 0.0+0.2j],
                                  [0.0+0.2j, 0.9-0.1j]]]]) # I x 1 x S x S
        Mn = npx.asarray([[0.9, 0.0],
                          [0.0, 0.8]]) # S x S
        phi_0 = npx.asarray([[[[npx.sqrt(0.6)+0.0j],
                                [npx.sqrt(0.4)+0.0j]]]]) # I x 1 x S x 1

        phi_t = ca._timestep_transition_state(n, T_delta, Mn, phi_0) # I x J x S x 1
        T_t = ca._timestep_transition_matrix(n, T_delta, Mn) # I x J x S x S
        phi_expected = T_t @ phi_0 # I x J x S x 1

        assert npx.allclose(phi_t, phi_expected), "Quantum timestep transition state not as expected"

    def test_timestep_transition_state_joint(self, model_constants, markov_constant_matrix, measurement_matrix_correct, measurement_matrix_incorrect, measurement_matrix_noresp):
        phi_0 = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                      model_constants.response_width,
                                      model_type="Markov", prior_type="Upper") # I x 1 x S x 1
        RT = npx.asarray([[[2, 3]],
                          [[4, 2]]]) # 2 x I x J
        RA = npx.asarray([[[1, 0]],
                          [[0, 1]]]) # 2 x I x J
        intensity_matrix = markov_constant_matrix[None, None, ...] # I x 1 x S x S

        phi_t = ca.perform_state_transition(intensity_matrix, RT_s=RT, RA_s=RA,
                                            Mc=measurement_matrix_correct, Mw=measurement_matrix_incorrect,
                                            Mn=measurement_matrix_noresp, phi_0=phi_0, delta=1,
                                            transition_type="TIMESTEP", likelihood_type="JOINT") # I x J x S x 1

        T_t_1 = ca._get_transition_matrix(intensity_matrix, RT=RT[0], delta=1,
                                          Mn=measurement_matrix_noresp, transition_type="TIMESTEP") # I x J x S x S
        T_t_2 = ca._get_transition_matrix(intensity_matrix, RT=RT[1], delta=1,
                                          Mn=measurement_matrix_noresp, transition_type="TIMESTEP") # I x J x S x S
        phi_t_1_c = T_t_2 @ measurement_matrix_correct @ T_t_1 # I x J x S x S
        phi_t_1_w = T_t_2 @ measurement_matrix_incorrect @ T_t_1 # I x J x S x S
        T_t = npx.where(RA[0,...,None,None]==1, phi_t_1_c, phi_t_1_w) # I x J x S x S
        phi_expected = T_t @ phi_0 # I x J x S x 1

        assert npx.allclose(phi_t, phi_expected), "Joint timestep transition state not as expected"

    def test_transition_matrix_markov(self, model_constants, markov_constant_matrix):
        log.debug("Constant Drift Rate - Mean Confidence 1")

        intensity_matrix_markov = ca.diffusion_buildK(model_constants.n_states, model_constants.mu, model_constants.sigma)
        assert npx.all(markov_constant_matrix == intensity_matrix_markov)
        
        
    def test_transition_matrix_quantum(self, model_constants, quantum_constant_matrix):
        log.debug("Constant Drift Rate - Mean Confidence 1")

        intensity_matrix_quantum = ca.quantum_buildH(model_constants.n_states, model_constants.mu, model_constants.sigma)
        intensity_matrix_quantum = -np.imag(intensity_matrix_quantum)
        assert npx.all(quantum_constant_matrix == intensity_matrix_quantum)

    def test_initial_states_markov(self, model_constants):
        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                             model_constants.response_width,
                                             model_type="Markov", prior_type="Upper")
        assert phi_0_markov.sum() == 1, "Markov States not summing up to 1"

    def test_initial_states_quantum(self, model_constants):
        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                              model_constants.response_width,
                                              model_type="Quantum" , prior_type="Upper")
        assert (phi_0_quantum**2).sum() > 0.99, "Quantum States not summing up to 1"

    def test_Markov_measurement_matrices(self, model_constants, measurement_matrix_correct, measurement_matrix_incorrect, measurement_matrix_noresp):
        m_Mc, m_Mw, m_Mn = ca._get_measurement_matrix(model_constants.n_states, model_constants.response_width, 
                                                      prob=model_constants.measurement_prob, model_type = "Markov")
        
        assert npx.all((m_Mc == measurement_matrix_correct) & (m_Mw == measurement_matrix_incorrect) & (m_Mn == measurement_matrix_noresp)), "Mismatch in Markov Response Measurement Matrix"

    def test_Quantum_measurement_matrices(self, model_constants, measurement_matrix_correct, measurement_matrix_incorrect, measurement_matrix_noresp):
        q_Mc, q_Mw, q_Mn = ca._get_measurement_matrix(model_constants.n_states, model_constants.response_width, 
                                                      prob=model_constants.measurement_prob, model_type = "Quantum")
        # assert npx.all((q_Mc**2 == measurement_matrix_correct) & (q_Mw**2 == measurement_matrix_incorrect) & (q_Mn**2 == measurement_matrix_noresp)), "Mismatch in Quantum Response Measurement Matrix"
        assert npx.allclose(q_Mc**2, measurement_matrix_correct) & npx.allclose(q_Mw**2, measurement_matrix_incorrect) & npx.allclose(q_Mn**2, measurement_matrix_noresp), "Mismatch in Quantum Response Measurement Matrix"

class Test_Confidence:

    def test_markov_confidence_for_internal(self, model_constants, markov_constant_matrix, measurement_matrix_noresp):

        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                             model_constants.response_width,
                                             model_type="Markov", prior_type="Upper")

        mean_conf_markov = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=markov_constant_matrix[None, None, ...], 
                                                    phi_0=phi_0_markov, delta=1, Mn=measurement_matrix_noresp,
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Markov")
        
        assert npx.allclose(mean_conf_markov, 0.14686, rtol = 1e-3), "Markov with internal transition expected mean incorrect"


    def test_quantum_confidence_for_internal(self, model_constants, quantum_constant_matrix, measurement_matrix_noresp):

        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                              model_constants.response_width,
                                              model_type="Quantum" , prior_type="Upper")

        quantum_intensity_matrix = -1j*quantum_constant_matrix[None, None, ...]
        mean_conf_quantum = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=quantum_intensity_matrix, 
                                                    phi_0=phi_0_quantum, delta=1, Mn=npx.sqrt(measurement_matrix_noresp), 
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Quantum")
        
        assert npx.allclose(mean_conf_quantum, 0.239, rtol = 1e-3), "Quantum with internal transition expected mean incorrect"
        

    def test_integration_parameter_to_confidence_for_internal(self, model_constants):

        intensity_matrix_markov = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Markov")
        intensity_matrix_quantum = ca.get_intensity_matrix(model_constants.n_states, model_constants.mu, 
                                                        model_constants.sigma, model_type="Quantum")

        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                             model_constants.response_width,
                                             model_type="Markov", prior_type="Upper")
        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                              model_constants.response_width,
                                             model_type="Quantum" , prior_type="Upper")

        m_Mc, m_Mw, m_Mn = ca._get_measurement_matrix(model_constants.n_states, model_constants.response_width, 
                                                      prob=model_constants.measurement_prob, model_type = "Markov")
        q_Mc, q_Mw, q_Mn = ca._get_measurement_matrix(model_constants.n_states, model_constants.response_width, 
                                                      prob=model_constants.measurement_prob, model_type = "Quantum")

        mean_conf_markov = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=intensity_matrix_markov, 
                                                    phi_0=phi_0_markov, delta=1, Mn=m_Mn, 
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Markov")
        
        mean_conf_quantum = ca.get_mean_confidence( model_constants.n_states, 
                                                    intensity_matrix=intensity_matrix_quantum, 
                                                    phi_0=phi_0_quantum, delta=1, Mn=q_Mn, 
                                                    t=npx.asarray([[10]]), transition_type="TIMESTEP", 
                                                    likelihood_type="SINGLE", model_type="Quantum")
        
        assert npx.allclose(mean_conf_markov, 0.14686, rtol = 1e-3) & npx.allclose(mean_conf_quantum, 0.239, rtol = 1e-3), "Integration Test Failed"


    def test_integration_initial_to_internal_likelihood_markov(self, model_constants, markov_constant_matrix, measurement_matrix_correct, measurement_matrix_incorrect, measurement_matrix_noresp):
        phi_0_markov = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                             model_constants.response_width,
                                             model_type="Markov", prior_type="Upper")
        
        likl_markov = ca.likelihood(intensity_matrix=markov_constant_matrix[None, None, ...], phi_0=phi_0_markov, delta=1,
                                    RT_s=npx.asarray([[10, 20]]), RA_s=npx.asarray([[1, 0]]),  
                                    Mc=measurement_matrix_correct, Mw=measurement_matrix_incorrect, Mn=measurement_matrix_noresp, 
                                    transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Markov")
        assert npx.allclose(likl_markov, npx.asarray([1.4769e-02, 2.0872e-06]), atol=1e-4), "Markov Likelihood not as expected"


    def test_integration_initial_to_internal_likelihood_quantum(self, model_constants, quantum_constant_matrix, measurement_matrix_correct, measurement_matrix_incorrect, measurement_matrix_noresp):
        phi_0_quantum = ca._get_initial_state(model_constants.n_states, model_constants.start_width,
                                              model_constants.response_width,
                                             model_type="Quantum" , prior_type="Upper")
        
        quantum_intensity_matrix = -1j*quantum_constant_matrix[None, None, ...]
        likl_quantum = ca.likelihood(intensity_matrix=quantum_intensity_matrix, 
                                     phi_0=phi_0_quantum, delta=1,
                                     RT_s=npx.asarray([[10, 30]]), RA_s=npx.asarray([[1, 0]]),  
                                     Mc=npx.sqrt(measurement_matrix_correct), Mw=npx.sqrt(measurement_matrix_incorrect),
                                     Mn=npx.sqrt(measurement_matrix_noresp), 
                                     transition_type="TIMESTEP", likelihood_type="SINGLE", model_type="Quantum")
        assert npx.allclose(likl_quantum, npx.asarray([0.0209, 0.000307]), atol=1e-4), "Quantum Likelihood not as expected"

