from numpy import *
from scipy.stats import *

def _get_n_states(alpha, theta, tau, sigma):
    delta_state = alpha * sigma * sqrt(tau)
    n_states = round(theta/delta_state) #2 * round(theta/delta_state) + 1
    return n_states

def _create_Wiener_Q(n_states, alpha, tau, sigma, mu):
    
    Q = zeros((n_states, n_states))
    rows_ = arange(1,n_states)
    cols_ = arange(0,n_states-1)
    Q[rows_,cols_] = 1/(2*alpha)*(1-mu*sqrt(tau)/sigma**2)
    Q[cols_, rows_] =  1/(2*alpha)*(1+mu*sqrt(tau)/sigma**2)
    Q[arange(n_states), arange(n_states)] = 1-(1/alpha)
    return Q

def _create_OU_Q(n_states, alpha, tau, sigma, delta, gamma):
    
    Q = zeros((n_states, n_states))
    rows_ = arange(1,n_states)
    cols_ = arange(0,n_states-1)
    x = arange(n_states)

    p1 = 1/(2*alpha) * ( 1 - (( delta-gamma * x ) * sqrt(tau)/sigma**2 ))
    p2 = 1/(2*alpha) * ( 1 + (( delta-gamma * x ) * sqrt(tau)/sigma**2 ))
    p3 = 1 - (1/alpha)
    

    Q[rows_,cols_] = p1[rows_]
    Q[cols_, rows_] = p2[cols_]
    Q[arange(n_states), arange(n_states)] = p3
    
    return Q

def _get_initial_state(n_states):
   p_0 = dirichlet.rvs(repeat(0.5,n_states-int(n_states/2)))
   p_0 = around(p_0.squeeze(), decimals=2) #concatenate(([[0.0]], p_0, [[0.0]]), axis=1) #dirichlet.rvs(repeat(0.5,n_states))
   z = sum(p_0)
   if (z>1):
      #print("^^^^^^ ",z)
      p_0 = p_0 / z
      #print("**** ",sum(p_0))
   s_0 = pad(multinomial.rvs(n=1,p=p_0), int(n_states/4))
   return s_0, p_0

def _random_walk_next_step(s_t, Q):
   ind_t = where(s_t)[0][0] #dot(Q, s_t) #select correct row
   p_t = Q[ind_t,:]
   p_t = around(p_t.squeeze(),decimals=2)
   z=sum(p_t)
   if(z>1):
      p_t = p_t/z
   
   s_t_1 = multinomial.rvs(n=1, p=p_t) 
   

      
   ind_t_1 = where(s_t_1)[0][0]
   #print(ind_t - ind_t_1, end=" ")
   if(ind_t - ind_t_1 < -1):
      return s_t, ind_t, p_t
   return s_t_1, ind_t_1, p_t

def _perform_walk(theta, alpha, tau,sigma, *params, process="Wiener|OU"):
   steps = []
   RT = -1

   n_states = _get_n_states(alpha, theta, tau, sigma)

   if (process=="Wiener"):
      mu, = params
      Q = _create_Wiener_Q(n_states,alpha,tau,sigma,mu)
   elif (process == "OU"):
      delta, gamma = params
      Q = _create_OU_Q(n_states,alpha,tau, sigma, delta, gamma)
   else:
      raise Exception("Only Wiener | OU process is allowed")
   
   s_t,_ = _get_initial_state(n_states)
   max_steps = 20000
   for i in range(max_steps): #n_walk):
      s_t_1, s_ind, p_t = _random_walk_next_step(s_t, Q)
      steps.append(s_ind)
      s_t = s_t_1
      
      if s_ind == n_states-1:#  or :
         RT = i*tau
         X = 1
         break
      elif s_ind == 1:
         RT = i*tau
         X = 0
   return(steps, RT, X)


def rt_samples(theta, alpha, tau, sigma, n_walk, *params, samples, process="Wiener|OU"):
    RT_arr = []
    X_arr = []
    steps_arr = []
    max_iter = samples*50
    for i in range(max_iter): 
        steps,RT,X = _perform_walk(theta, alpha, tau, sigma, *params, n_walk=n_walk, process=process)
        if RT > 0: 
            RT_arr.append(RT) 
            X_arr.append(X)
            steps_arr.append(steps)
        if size(RT_arr) >= samples:
            break 
    return RT_arr, X_arr, steps_arr