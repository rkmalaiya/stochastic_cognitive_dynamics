import numpy as np
import pandas as pd
import scipy.stats as stats
import cme.utils.common_logging as cl
from joblib import Parallel, delayed

log = cl.get_logger("random-walk")

def _scale_steps_by_boundary(steps, alpha, tau, sigma):
   
   #not sure about + _get_step_size(alpha, tau, sigma)
   return (((np.asarray(steps) ) - 1)/2) * _get_step_size(alpha, tau, sigma) + _get_step_size(alpha, tau, sigma)

def _get_step_size(alpha, tau, sigma):
   return alpha * sigma * np.sqrt(tau)

def _get_n_states(alpha, theta, tau, sigma):
    sigma = sigma#.squeeze()
    step_size = _get_step_size(alpha, tau, sigma)
    n_states = 2 * int(theta/step_size) + 1 #np.round(theta/delta_state) #2 * round(theta/delta_state) + 1 # #
    return n_states

def _create_Wiener_Q(n_states, alpha, tau, sigma, mu):
    Q = np.zeros((n_states, n_states))
    rows_ = np.arange(1,n_states)
    cols_ = np.arange(0,n_states-1)
    Q[rows_,cols_] = 1/(2*alpha)*(1-mu*np.sqrt(tau)/sigma**2)
    Q[cols_, rows_] =  1/(2*alpha)*(1+mu*np.sqrt(tau)/sigma**2)
    Q[np.arange(n_states), np.arange(n_states)] = 1-(1/alpha)
    Q[0,1] = 0
    Q[-1,-2] = 0
    return Q

def _create_pWiener_Q(n_states, alpha, tau, sigma, mu):
    
    Q = np.zeros((n_states, n_states))
    rows_ = np.arange(1,n_states)
    cols_ = np.arange(0,n_states-1)

    mu_t = mu*np.sqrt(tau)/sigma**2
    Q[rows_,cols_] = 1/(2*alpha)*(1-mu_t)[rows_]
    Q[cols_, rows_] =  1/(2*alpha)*(1+mu_t)[cols_]
    Q[np.arange(n_states), np.arange(n_states)] = 1-(1/alpha)
    Q[0,1] = 0
    Q[-1,-2] = 0
    return Q

def _create_OU_Q(n_states, alpha, tau, sigma, delta, gamma):
    
    Q = np.zeros((n_states, n_states))
    rows_ = np.arange(1,n_states)
    cols_ = np.arange(0,n_states-1)
    x = np.arange(n_states)

    p1 = 1/(2*alpha) * ( 1 - ( (delta - (gamma * x))  * np.sqrt(tau)/sigma**2 ))
    p2 = 1/(2*alpha) * ( 1 + ( (delta - (gamma * x))  * np.sqrt(tau)/sigma**2 ))
    p3 = 1 - (1/alpha)
    

    Q[rows_,cols_] = p1[rows_]
    Q[cols_, rows_] = p2[cols_]
    Q[np.arange(n_states), np.arange(n_states)] = p3
    Q[0,1] = 0
    Q[-1,-2] = 0

    return Q

def _get_initial_state(n_states):
   p_0 = stats.dirichlet(np.repeat(0.5,n_states-int(n_states/2))).rvs()
   p_0 = np.around(p_0.squeeze(), decimals=2) #concatenate(([[0.0]], p_0, [[0.0]]), axis=1) #dirichlet.rvs(repeat(0.5,n_states))
   z = np.sum(p_0)
   while (z>1):
      #print("^^^^^^ ",z)
      p_0 = p_0 / z
      z = np.sum(p_0)
      #print("**** ",z)
   s_0 = np.pad(stats.multinomial.rvs(n=1,p=p_0), int(n_states/4))
   return s_0, p_0

def _random_walk_next_step(s_t, Q):
   ind_t = np.where(s_t)[0][0] #dot(Q, s_t) #select correct row
   p_t = Q[ind_t,:]
   #p_t = around(p_t.squeeze(),decimals=2)
   z=sum(p_t)
   while(z>1):
      print(f"$$$$$$$$$$$ Floating point error $$$$$$$$$$$$$$$$$$$ {z}")

   s_t_1 = stats.multinomial(n=1, p=p_t).rvs().squeeze() #Not sure about adding squeeze here

   ind_t_1 = np.where(s_t_1)[0][0] 
   #print(ind_t - ind_t_1, end=" ")
   if(ind_t - ind_t_1 < -1):
      return s_t, ind_t, p_t
   return s_t_1, ind_t_1, p_t

def _perform_walk(theta, alpha, tau,sigma, *params, process="Wiener|OU", initial="EZ|Fixed|Any", bias=None):
   steps = []
   RT = -1
   X = -1
   max_steps = 100000
   
   n_states = _get_n_states(alpha, theta, tau, sigma)
   #log.debug(f"States: {n_states}")

   Q = get_transition_matrix(alpha, tau, sigma, *params, process=process, n_states=n_states)
   
   if(initial == "EZ"):
      s_t = np.zeros(n_states)
      s_t[int(round(n_states/2))] = 1
   elif(initial == "Fixed"):
      s_t = np.zeros(n_states)
      s_t[_get_n_states(alpha, bias, tau, sigma)] = 1
   else:
      s_t,_ = _get_initial_state(n_states)
   
   for i in range(max_steps): #n_walk):
      s_t_1, s_ind, p_t = _random_walk_next_step(s_t, Q)
      #print(s_ind)
      steps.append(s_ind)
      s_t = s_t_1
      
      if s_ind == n_states-1:#  or :
         RT = i*tau
         X = 1
         break
      elif s_ind == 0:# | s_ind == 1:
         RT = i*tau
         X = 0
         break
   #log.debug(f"*****{i}*****{tau}")
   return(steps, RT, X)

def get_transition_matrix(alpha, tau, sigma, *params, process, n_states):
    if (process=="Wiener"):
       mu, = params
       if(len(mu) == 1):
          Q = _create_Wiener_Q(n_states,alpha,tau,sigma,mu)
       else:
          Q = _create_pWiener_Q(n_states,alpha,tau,sigma,mu)

    elif (process == "OU"):
       delta, gamma = params
       Q = _create_OU_Q(n_states,alpha,tau, sigma, delta, gamma)
    else:
       raise Exception("Only Wiener | OU process is allowed")
    return Q


def gen_rt_x(theta, alpha, tau, sigma, *params, samples, process="Wiener|OU", initial="EZ|Fixed|Any", scale_steps = True, njobs=1,bias=None):
   # alpha is a constant
   # theta decides the number of states
   RT_arr = []
   X_arr = []
   steps_arr = []
   max_iter = 5000

   for mi in np.arange(max_iter):

      ret_ans = Parallel(n_jobs=njobs)(delayed(_perform_walk)(theta, alpha, tau, sigma,
                                                                 *params, process=process, initial=initial, bias=bias) 
                                                                 for _ in range(samples))
      for ans in ret_ans:
         steps, RT, X = ans
         if RT > 0: 
            RT_arr.append(RT) 
            X_arr.append(X)
            if scale_steps:
               steps = _scale_steps_by_boundary(steps, alpha, tau, sigma)
            steps_arr.append(steps)
            if np.size(RT_arr) >= samples:
               break

      # We need two breaks in case we need to iterate back and get more samples, 
      # so the internal for loop does not need to execute for full length of samples
      if (np.size(RT_arr) >= samples):
         break

   
      
         #if RT > 0: 
         #   RT_arr.append(RT) 
         #   X_arr.append(X)
         #   steps_arr.append(steps)
         #if size(RT_arr) >= samples:
         #   break 
   #)

   #for ans in ret_ans:
   #   steps, RT, X = ans
   #   if RT > 0: 
   #      RT_arr.append(RT) 
   #      X_arr.append(X)
   #      steps_arr.append(steps)
   #   if size(RT_arr) >= samples:
   #      break 

   if (np.size(RT_arr) < samples):

      raise Exception(f"Could not generate enough RTs: {np.size(RT_arr)}")
   
   return RT_arr, X_arr, steps_arr

def gen_RT_X_mat(theta, alpha, tau, sigma, *params, I,J, process="Wiener|OU|DiffusionIRT", initial="EZ|Any", njobs = 8):
   
   X = np.zeros((I,J))
   RT = np.zeros((I,J))
   v_arr = []
   tr_arr = []

   #print("Generating data for participant", end=",")

   for i in range(I):
      #print(f"{i}", end=",")
      if (len(params) == 1):
         v_s, = params   
         v_s = v_s + np.random.default_rng().normal(0,0.1**2)
         params = (v_s,)
      else:
         v_s, oths = params
         v_s = v_s + np.random.default_rng().normal(0,0.1**2)
         params = (v_s, oths)

      # To vary for each participant
      
      #params[0] = v_s, oths
      
      if process == "DiffusionIRT":
         v_p_s, v_i_s = params
         v_l = len(v_i_s)

         if(J % v_l > 0):
            raise Exception("Total number of items should be a multiple of the length of possible item drift rates")
            
         batch = J//v_l
         v_arr_ind = []
         for v_i in v_i_s:
            params_for_gen = [v_p_s / v_i,] #changed here
            v_arr_ind.append(params_for_gen)
            for ind in range(0,J, batch):
               rt, x, _ = gen_rt_x(theta, alpha, tau, sigma, *params_for_gen, samples=batch, process="Wiener", initial=initial,njobs=njobs)
               X[i,ind:ind+batch] = x
               RT[i,ind:ind+batch] = rt      
         v_arr.append(v_arr_ind)      
      else:
         v_arr.append(params[0])
         rt, x, tr = gen_rt_x(theta, alpha, tau, sigma, *params, samples=J, process=process, initial=initial,njobs=njobs)
         X[i,:] = x
         RT[i,:] = rt
         tr_arr.append(tr)

   #Parallel(n_jobs=njobs, require='sharedmem')(
   # delayed(par_gen_RT_X_mat)(theta, alpha, tau, sigma, params, J, process, initial, X, RT, v_arr, i) for i in range(I))
     
   return RT, X, v_arr, tr_arr
    
def store_randomwalk(RT, X, steps_arr,file_pre_name):
   df = []
   for i, steps in enumerate(steps_arr):
      df.append(pd.DataFrame({"rw_no":i,"steps":np.asarray(steps)}))

   pd.concat(df).to_csv(f"{file_pre_name}_steps.csv", index=False)
   np.savetxt(f"{file_pre_name}_RT.csv", RT)
   np.savetxt(f"{file_pre_name}_X.csv", X)

def load_randomwalk(file_pre_name):
   steps_arr = pd.read_csv(f"{file_pre_name}_steps.csv")
   RT = np.loadtxt(f"{file_pre_name}_RT.csv")
   X = np.loadtxt(f"{file_pre_name}_X.csv")
   return RT, X, steps_arr

if __name__ == "__main__":
    theta, alpha, tau, njobs = 100, 1.5, 0.01, 1

    mu, sigma = np.asarray([0.2]), np.asarray([1])
    get_transition_matrix(alpha, tau,sigma, mu, process="Wiener",n_states = theta)

    mu = np.repeat([0.01,0.02,0.05,0.08], (theta+2)/ 4)[0:theta+10]
    get_transition_matrix(alpha, tau,sigma, mu, process="Wiener",n_states = theta)
    log.debug("test successful0")



    mu=np.asarray([0.2])
    RT_arr, X_arr, steps_arr = gen_rt_x(theta, alpha, tau, sigma, mu,samples = 10, process="Wiener",njobs=1)
    assert len(RT_arr) == 10
    assert len(X_arr) == 10
    log.debug(X_arr)
    log.debug(RT_arr)
    log.debug("test successful1")

    delta, gamma = np.asarray([3]), np.asarray([0.01])
    RT_arr, X_arr, steps_arr = gen_rt_x(theta, alpha, tau, sigma, delta, gamma,samples = 10, process="OU",njobs=njobs)
    assert len(RT_arr) == 10
    assert len(X_arr) == 10
    log.debug(X_arr)
    log.debug(RT_arr)
    log.debug("test successful1-OU")


    mu=np.asarray([-0.2])
    RT_arr, X_arr, steps_arr = gen_rt_x(theta, alpha, tau, sigma, mu,samples = 10, process="Wiener",njobs=njobs)
    assert len(RT_arr) == 10
    assert len(X_arr) == 10
    log.debug(X_arr)
    log.debug(RT_arr)
    log.debug("test successful2")


    v_p=np.asarray([0.2])
    RT_mat, X_mat,v_arr,tr_arr = gen_RT_X_mat(theta, alpha, tau, sigma, v_p, I=10, J = 5, process="Wiener")
    log.debug(X_mat.shape)
    log.debug(RT_mat.shape)
    assert RT_mat.shape == (10,5)
    assert X_mat.shape == (10,5)
    assert len(v_arr) == 10
    log.debug("test successful3")
    

    v_p=np.asarray([0.2])
    #v_i=asarray([2,0.5])
    v_i = np.asarray([0.5, 0.75, 1, 1.25])
    RT_mat, X_mat,v_arr,tr_arr = gen_RT_X_mat(theta, alpha, tau, sigma, v_p, v_i,I=10, J = 8, process="DiffusionIRT")
    log.debug(X_mat.shape)
    log.debug(RT_mat.shape)
    assert RT_mat.shape == (10,8)
    assert X_mat.shape == (10,8)
    assert np.asarray(v_arr).squeeze().shape == (10,4)
    log.debug("test successful4")


    v_p=np.repeat([0.01,0.02,0.05,0.08], (theta+2)/ 4)[0:theta+10]
    #v_i=asarray([2,0.5])
    v_i = np.asarray([0.5, 0.75, 1, 1.25])
    RT_mat, X_mat,v_arr,tr_arr = gen_RT_X_mat(theta, alpha, tau, sigma, v_p, v_i,I=10, J = 8, process="DiffusionIRT", njobs=njobs)
    log.debug(X_mat.shape)
    log.debug(RT_mat.shape)
    assert RT_mat.shape == (10,8)
    assert X_mat.shape == (10,8)
    assert np.asarray(v_arr).squeeze().shape == (10,4,100)
    log.debug("test successful5")