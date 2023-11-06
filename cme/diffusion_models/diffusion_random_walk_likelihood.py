#%%
import pytensor.tensor as at
import pymc as pm
import cme.utils.common_logging as cl
from joblib import Parallel, delayed
import numpy as np
import pytensor.tensor.subtensor as st
import arviz as az

log = cl.get_logger("random-walk")


def _get_n_states(alpha, theta, tau, sigma):
    delta_state = alpha * sigma * at.sqrt(tau)
    n_states = round(theta/delta_state) #2 * round(theta/delta_state) + 1
    return n_states

def _create_Wiener_Q(n_states, alpha, tau, sigma, mu):
    
    Q = at.zeros((n_states, n_states))
    rows_ = at.arange(1,n_states)
    cols_ = at.arange(0,n_states-1)
    Q = st.set_subtensor(Q[rows_,cols_], 1/(2*alpha)*(1-mu*at.sqrt(tau)/sigma**2))
    Q = st.set_subtensor(Q[cols_, rows_],  1/(2*alpha)*(1+mu*at.sqrt(tau)/sigma**2))
    Q = st.set_subtensor(Q[at.arange(n_states), at.arange(n_states)], 1-(1/alpha))
    Q = st.set_subtensor(Q[0,1], 0)
    Q = st.set_subtensor(Q[-1,-2], 0)
    return Q

def _get_initial_state(n_states):
   p_0 = pm.draw(pm.Dirichlet.dist(np.repeat(0.5,n_states))) # To match the Q matrix (see Diederich 2003)
   p_0 = p_0.squeeze() #concatenate(([[0.0]], p_0, [[0.0]]), axis=1) #dirichlet.rvs(repeat(0.5,n_states))
   #z = at.sum(p_0)
   #while (z>1):
      #print("^^^^^^ ",z)
   #   p_0 = p_0 / z
   #   z = sum(p_0)
      #print("**** ",z)
   #s_0 = pm.draw(pm.Multinomial.dist(n=1,p=p_0))
   return p_0

def _liklihood(t, x, v):
   
   alpha=1.5
   tau = 0.01
   theta=5
   sigma = 1
   Delta = alpha * np.sqrt(sigma) * np.sqrt(tau)
   m = int(2 * np.round(theta/Delta) + 1)
   x = np.arange(-(m-1)/2, (m-1)/2)

   tm = _create_Wiener_Q(m, alpha, tau, sigma, v)
   Q = tm[1:-1,1:-1]
   R = tm[1:-1,[0,-1]]
   
   Z_m = m-2
   Z = _get_initial_state(Z_m)
   n = (t.eval() / tau) - 1

   #num1 = at.matmul(Z, at.nlinalg.matrix_power(Q, n-1))
   #num = at.matmul(num1, R)
   #den = at.matmul(at.matmul(Z, at.nlinalg.matrix_power((at.eye(m)-Q,-1))), R)

   t1 = at.nlinalg.matrix_power(Q, n)
   t2 = at.nlinalg.matrix_power(at.eye(Z_m) - Q,-1)
   
   num = (Z @ t1) @ R
   den = (Z @ t2) @ R

   Pr = num/den
   #print(Pr.eval().shape)
   likl = at.where(at.eq(x,0),Pr[0], Pr[1])
   #likl = likl.sum()
   return likl

#%%
if __name__ == "__main__":
   Pr = _liklihood(at.as_tensor([[1.5]]), at.as_tensor([[1]]), at.as_tensor([0.5]))
   print(Pr.eval().shape)
   
   with pm.Model() as model:
      v = pm.Normal("drift", 0,1,shape=(1,))
      likl = pm.CustomDist("rt",
      at.as_tensor([[1]]), v,
      logp = _liklihood,
      observed = at.as_tensor([[1.5]])
      )
      az.summary(pm.sample(draws=10, tune=5))
   
# %%
