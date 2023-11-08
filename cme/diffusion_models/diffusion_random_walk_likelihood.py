#%%
import jax.numpy as at
import cme.utils.common_logging as cl
import numpy as np
import arviz as az
import numpyro as ro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from jax import random
from numpyro.infer import MCMC, NUTS
import arviz as az


log = cl.get_logger("random-walk")


def get_n_states(alpha=1.5, theta=5, tau=0.01, sigma=1):
    delta_state = alpha * sigma * at.sqrt(tau)
    n_states = at.round(theta/delta_state) #2 * round(theta/delta_state) + 1
    return n_states.astype(int).item()

def _create_Wiener_Q(n_part, n_states, alpha, tau, sigma, v):
    Q = at.zeros((n_part, 1, n_states, n_states)) # n_part x n_trials x n_states x n_states
    rows_ = at.arange(1,n_states)
    cols_ = at.arange(0,n_states-1)
    for i in at.arange(n_part):
      b_m = (1/(2*alpha)*(1-v[i,:]*at.sqrt(tau)/sigma**2)).reshape(-1,1)
      b_p = (1/(2*alpha)*(1+v[i,:]*at.sqrt(tau)/sigma**2)).reshape(-1,1)
      Q = Q.at[i,:, rows_,cols_].set(b_m)
      Q = Q.at[i,:, cols_, rows_].set(b_p) # Because we don't need drift rates for boundary states
      Q = Q.at[i,:, at.arange(n_states), at.arange(n_states)].set(1-(1/alpha))
      Q = Q.at[i,:, 0,1].set(0)
      Q = Q.at[i,:, -1,-2].set(0)
    return Q

def init_model(n_part, v, n_states):

   #x = np.arange(-(n_states-1)/2, (n_states-1)/2)
   
   T_m = _create_Wiener_Q(n_part, n_states, alpha, tau, sigma, v)
   
   Z_m = n_states-2
   I = at.eye(Z_m)
   return T_m, I

def _liklihood(t, x, T_m, Z, I, N):
   
   #num1 = at.matmul(Z, at.nlinalg.matrix_power(Q, n-1))
   #num = at.matmul(num1, R)
   #den = at.matmul(at.matmul(Z, at.nlinalg.matrix_power((at.eye(m)-Q,-1))), R)

   n_part, n_trials = t.shape if t is not None else (1,1)

   Q = T_m[..., 1:-1,1:-1]
   R = T_m[..., 1:-1,[0,-1]]

   t1 = at.zeros((n_part, n_trials, Q.shape[-2], Q.shape[-1]))

   for i in range(n_part):
      for j in range(n_trials):
         a1 = Q[i,j]
         a2 = N[i,j]
         #a2 = a2.astype(int).item()
         t1 = t1.at[i,j,...].set(at.linalg.matrix_power(a1, a2.item()))

   t2 = at.linalg.matrix_power(I - Q, -1)
   
   num = (Z @ t1) @ R
   den = (Z @ t2) @ R

   Pr = num/den
   likl = at.where(x==0,Pr[...,0,0], Pr[...,0,1]) # now working but double check
   likl = likl.sum()
   return likl

def _model(n_states, n_within_trials=1, tau=0.01, t= None, x=None):

   n_part = t.shape[0] if t is not None else 1
   Z = ro.sample("start_state", dist.Dirichlet(np.repeat(0.5,n_states-2)), sample_shape=(n_part,)) # To match the Q matrix (see Diederich 2003)
   v = ro.sample("drift", dist.Normal(0,1),sample_shape=(n_part,n_within_trials))

   T_m, I = init_model(n_part, v, n_states)

   N = np.asarray((t / tau) - 1).astype(int)

   ro.factor("likl", _liklihood(t,x,T_m, Z, I, N))

   return Z, v

def sample_posterior(t, x, num_samples=200, num_warmup=100):
   rng_key = random.PRNGKey(0)
   rng_key, rng_key_ = random.split(rng_key)

   # Run NUTS.
   kernel = NUTS(_model)
   mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=4)
   mcmc.run(
      rng_key_, n_states = get_n_states(), t=t, x=x
   )

   mcmc_chain = az.from_numpyro(mcmc)
   return mcmc_chain

if __name__ == "__main__":
   alpha=1.5
   tau = 0.01
   theta=5
   sigma = 1

   n_states = get_n_states(alpha, theta, tau, sigma)
   Q = _create_Wiener_Q(n_part = 1, n_states=n_states, alpha=alpha, tau=tau, sigma=sigma, v = at.asarray([0.5]).reshape(1,-1) )

   Q = _create_Wiener_Q(n_part = 1, n_states=n_states, alpha=alpha, tau=tau, sigma=sigma, 
                        v = at.repeat(at.asarray([[0.5]]),n_states-1).reshape(1,-1) )

   rng = random.PRNGKey(0)

   n_part = 1
   alpha=1.5 
   tau = 0.01 
   theta=5
   sigma = 1
   n_states = get_n_states(alpha, theta, tau, sigma)

   #res = Predictive(get_prior(n_part, n_states, n_within_trial))(rng)
   Z = dist.Dirichlet(np.repeat(0.5,n_states-2)).sample(rng, sample_shape=(1,))
   v = dist.Normal(0,1).sample(rng, sample_shape=(n_part,1))


   T_m, I = init_model(n_part, v, n_states)

   t, x = at.asarray([[1.5]]), at.asarray([[1]]) 

   N = at.asarray((t / tau) - 1).astype(int)

   Pr = _liklihood(t, x, T_m, Z, I, N)
   print(Pr)

   n_part = 2
   alpha=1.5 
   tau = 0.01 
   theta=5
   sigma = 1
   n_states = get_n_states(alpha, theta, tau, sigma)
   #res = Predictive(get_prior(n_part, n_states, n_within_trial))(rng)
   Z = dist.Dirichlet(np.repeat(0.5,n_states-2)).sample(rng, sample_shape=(n_part,))
   v = dist.Normal(0,1).sample(rng, sample_shape=(n_part,1))

   T_m, I = init_model(n_part, v, n_states)
   n_within_trial = T_m.shape[0]

   t, x = np.random.random((n_part,4)) + np.random.random((n_part,4)), np.random.randint(0,2, size = (n_part,4))

   N = at.asarray((t / tau) - 1).astype(int)

   Pr = _liklihood(t, x, T_m, Z, I, N)
   print(Pr)
      
   n_part = 2
   alpha=1.5 
   tau = 0.01 
   theta=5
   sigma = 1
   n_states = get_n_states(alpha, theta, tau, sigma)
   #res = Predictive(get_prior(n_part, n_states, n_within_trial))(rng)
   Z = dist.Dirichlet(np.repeat(0.5,n_states-2)).sample(rng, sample_shape=(n_part,))
   v = dist.Normal(0,1).sample(rng, sample_shape=(n_part,n_states-1))

   T_m, I = init_model(n_part, v, n_states)
   n_within_trial = T_m.shape[0]

   t, x = np.random.random((n_part,4)) + np.random.random((n_part,4)), np.random.randint(0,2, size = (n_part,4))

   N = at.asarray((t / tau) - 1).astype(int)

   Pr = _liklihood(t, x, T_m, Z, I, N)
   print(Pr)
   print("sampling******")

   mcmc_chain = sample_posterior(t,x,50,10)
   print(az.summary(mcmc_chain))

