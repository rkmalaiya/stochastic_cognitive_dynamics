import cme.decision_models.confidence_accumulation as ca
import jax.numpy as npx
import scipy.stats as stats 


n_states, start_width, response_width, delta, measurement_prob = 11, None, 3, 0.1, 0.25
start_width = (n_states-2*response_width)
m_Mc, m_Mw, m_Mn = ca._get_measurement_matrix(n_states, 1, prob=measurement_prob, model_type = "Markov")
q_Mc, q_Mw, q_Mn = ca._get_measurement_matrix(n_states, 1, prob=measurement_prob, model_type = "Quantum")

mu, sigma = npx.asarray([[-0.65907584]]), npx.asarray([[0.34092416]])
t,a = 2.3053, 1

intensity_matrix_markov = ca.get_intensity_matrix(n_states, mu, sigma,model_type="Markov")
intensity_matrix_quantum = ca.get_intensity_matrix(n_states, mu, sigma,model_type="Quantum")

phi_0_markov = ca._get_initial_state(n_states, start_width,model_type="Markov", prior_type="Upper")
phi_0_quantum = ca._get_initial_state(n_states, start_width,model_type="Quantum" , prior_type="Upper")


phi_0_markov = npx.asarray([[[[1.31571211e-01],
         [6.54044253e-03],
         [2.57143046e-02],
         [4.86459476e-01],
         [7.21988146e-03],
         [2.20123214e-04],
         [3.27711275e-02],
         [6.83485613e-02],
         [1.84833910e-03],
         [2.38612483e-01],
         [6.94050041e-04]]]])

likl_markov = ca.likelihood(intensity_matrix=intensity_matrix_markov, phi_0=phi_0_markov, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[a]]), 
                                    Mc=m_Mc, Mw=m_Mw, Mn=m_Mn, 
                                    transition_type="RT", likelihood_type="SINGLE", model_type="Markov")


likl_quantum = ca.likelihood(intensity_matrix=intensity_matrix_quantum, phi_0=phi_0_quantum, delta=delta,
                                    RT_s=npx.asarray([[t]]), RA_s=npx.asarray([[a]]), 
                                    Mc=q_Mc, Mw=q_Mw, Mn=q_Mn, 
                                    transition_type="RT", likelihood_type="SINGLE", model_type="Quantum")
