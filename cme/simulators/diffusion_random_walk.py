from numpy import *
from pandas import *
from scipy.stats import *
import cme.utils.common_logging as cl
from joblib import Parallel, delayed

log = cl.get_logger("random-walk")


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

def _create_pWiener_Q(n_states, alpha, tau, sigma, mu):
    
    Q = zeros((n_states, n_states))
    rows_ = arange(1,n_states)
    cols_ = arange(0,n_states-1)

    mu_t = mu*sqrt(tau)/sigma**2
    Q[rows_,cols_] = 1/(2*alpha)*(1-mu_t)[rows_]
    Q[cols_, rows_] =  1/(2*alpha)*(1+mu_t)[cols_]
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
   p_0 = dirichlet(repeat(0.5,n_states-int(n_states/2))).rvs()
   p_0 = around(p_0.squeeze(), decimals=2) #concatenate(([[0.0]], p_0, [[0.0]]), axis=1) #dirichlet.rvs(repeat(0.5,n_states))
   z = sum(p_0)
   while (z>1):
      #print("^^^^^^ ",z)
      p_0 = p_0 / z
      z = sum(p_0)
      #print("**** ",z)
   s_0 = pad(multinomial.rvs(n=1,p=p_0), int(n_states/4))
   return s_0, p_0

def _random_walk_next_step(s_t, Q):
   ind_t = where(s_t)[0][0] #dot(Q, s_t) #select correct row
   p_t = Q[ind_t,:]
   p_t = around(p_t.squeeze(),decimals=2)
   z=sum(p_t)
   while(z>1):
      p_t = p_t/z
      z=sum(p_t+0.01)
      print(f"$$$$$$$$$$$ Floating point error $$$$$$$$$$$$$$$$$$$ {z}")
   
   s_t_1 = multinomial.rvs(n=1, p=p_t) 
   

      
   ind_t_1 = where(s_t_1)[0][0]
   #print(ind_t - ind_t_1, end=" ")
   if(ind_t - ind_t_1 < -1):
      return s_t, ind_t, p_t
   return s_t_1, ind_t_1, p_t

def _perform_walk(theta, alpha, tau,sigma, *params, process="Wiener|OU", initial="EZ|Any"):
   steps = []
   RT = -1
   X = -1
   max_steps = 40000
   
   n_states = theta #_get_n_states(alpha, theta, tau, sigma)

   Q = get_transition_matrix(alpha, tau, sigma, *params, process, n_states)
   
   if(initial == "EZ"):
      s_t = zeros(n_states)
      s_t[round(n_states/2)] = 1
   else:
      s_t,_ = _get_initial_state(n_states)
   
   for i in range(max_steps): #n_walk):
      s_t_1, s_ind, p_t = _random_walk_next_step(s_t, Q)
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
   return(steps, RT, X)

def get_transition_matrix(alpha, tau, sigma, params, process, n_states):
    if (process=="Wiener"):
       mu, = params
       if(not isinstance(mu, list)):
          Q = _create_Wiener_Q(n_states,alpha,tau,sigma,mu)
       elif(len(mu) > 1):
          Q = _create_pWiener_Q(n_states,alpha,tau,sigma,mu)
       else:
          raise Exception("Missing mu")
    elif (process == "OU"):
       delta, gamma = params
       Q = _create_OU_Q(n_states,alpha,tau, sigma, delta, gamma)
    else:
       raise Exception("Only Wiener | OU process is allowed")
    return Q


def gen_rt_x(theta, alpha, tau, sigma, *params, samples, process="Wiener|OU", initial="EZ|Any",njobs=8):
   RT_arr = []
   X_arr = []
   steps_arr = []
   max_iter = samples*50

   ret_ans = Parallel(n_jobs=njobs)(
       
     delayed(_perform_walk)(theta, alpha, tau, sigma, *params, process=process, initial=initial) for _ in range(max_iter)

      
         #if RT > 0: 
         #   RT_arr.append(RT) 
         #   X_arr.append(X)
         #   steps_arr.append(steps)
         #if size(RT_arr) >= samples:
         #   break 
   )

   for ans in ret_ans:
      steps, RT, X = ans
      if RT > 0: 
         RT_arr.append(RT) 
         X_arr.append(X)
         steps_arr.append(steps)
      if size(RT_arr) >= samples:
         break 


   
   return RT_arr, X_arr, steps_arr

def gen_RT_X_mat(theta, alpha, tau, sigma, *params, I,J, process="Wiener|OU|DiffusionIRT", initial="EZ|Any", njobs = 8):
   
   X = zeros((I,J))
   RT = zeros((I,J))
   v_arr = []
   tr_arr = []

   print("Generating data for participant")

   for i in range(I):
      print(f"{i}")
      if (len(params) == 1):
         v_s, = params   
         v_s = v_s + random.default_rng().normal(0,0.1**2)
         params = (v_s,)
      else:
         v_s, oths = params
         v_s = v_s + random.default_rng().normal(0,0.1**2)
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
            params_for_gen = [v_p_s / v_i]
            v_arr_ind.append(params_for_gen)
            for ind in range(0,J, batch):
               rt, x, _ = gen_rt_x(theta, alpha, tau, sigma, *params_for_gen, samples=batch, process="Wiener", initial=initial,njobs=njobs)
               X[i,ind:ind+batch] = x
               RT[i,ind:ind+batch] = rt      
         v_arr.append(v_arr_ind)      
      else:
         v_arr.append(params[0])
         rt, x, tr = gen_rt_x(theta, alpha, tau, sigma, *params, samples=J, process=process, initial=initial)
         X[i,:] = x
         RT[i,:] = rt
         tr_arr.append(tr)

   #Parallel(n_jobs=njobs, require='sharedmem')(
   # delayed(par_gen_RT_X_mat)(theta, alpha, tau, sigma, params, J, process, initial, X, RT, v_arr, i) for i in range(I))
     
   return RT, X, v_arr, tr_arr
    
def store_randomwalk(RT, X, steps_arr,file_pre_name):
   df = []
   for i, steps in enumerate(steps_arr):
      df.append(DataFrame({"rw_no":i,"steps":asarray(steps)}))

   concat(df).to_csv(f"{file_pre_name}_steps.csv", index=False)
   savetxt(f"{file_pre_name}_RT.csv", RT)
   savetxt(f"{file_pre_name}_X.csv", X)

def load_randomwalk(file_pre_name):
   steps_arr = read_csv(f"{file_pre_name}_steps.csv")
   RT = loadtxt(f"{file_pre_name}_RT.csv")
   X = loadtxt(f"{file_pre_name}_X.csv")
   return RT, X, steps_arr

if __name__ == "__main__":
    theta, alpha, tau, sigma = 100, 1.5, 0.01, 1

    mu = asarray([0.2])
    get_transition_matrix(alpha, tau,sigma=sigma, params=mu, process="Wiener",n_states = theta)


    mu=asarray([0.2])
    RT_arr, X_arr, steps_arr = gen_rt_x(theta, alpha, tau, sigma, mu,samples = 10, process="Wiener")
    assert len(RT_arr) == 10
    assert len(X_arr) == 10
    log.debug(X_arr)
    log.debug(RT_arr)
    log.debug("test successful")


    mu=asarray([-0.2])
    RT_arr, X_arr, steps_arr = gen_rt_x(theta, alpha, tau, sigma, mu,samples = 10, process="Wiener")
    assert len(RT_arr) == 10
    assert len(X_arr) == 10
    log.debug(X_arr)
    log.debug(RT_arr)
    log.debug("test successful")


    v_p=asarray([0.2])
    RT_mat, X_mat,v_arr,tr_arr = gen_RT_X_mat(theta, alpha, tau, sigma, v_p, I=10, J = 5, process="Wiener")
    log.debug(X_mat.shape)
    log.debug(RT_mat.shape)
    assert RT_mat.shape == (10,5)
    assert X_mat.shape == (10,5)
    assert len(v_arr) == 10
    log.debug("test successful")
    

    v_p=asarray([0.2])
    #v_i=asarray([2,0.5])
    v_i = asarray([0.5, 0.75, 1, 1.25])
    RT_mat, X_mat,v_arr,tr_arr = gen_RT_X_mat(theta, alpha, tau, sigma, v_p, v_i,I=10, J = 8, process="DiffusionIRT")
    log.debug(X_mat.shape)
    log.debug(RT_mat.shape)
    assert RT_mat.shape == (10,8)
    assert X_mat.shape == (10,8)
    assert asarray(v_arr).squeeze().shape == (10,4)
    log.debug("test successful")


    v_p=repeat([0.01,0.02,0.05,0.08], (theta+2)/ 4)[0:theta+10]
    #v_i=asarray([2,0.5])
    v_i = asarray([0.5, 0.75, 1, 1.25])
    RT_mat, X_mat,v_arr,tr_arr = gen_RT_X_mat(theta, alpha, tau, sigma, v_p, v_i,I=10, J = 8, process="DiffusionIRT")
    log.debug(X_mat.shape)
    log.debug(RT_mat.shape)
    assert RT_mat.shape == (10,8)
    assert X_mat.shape == (10,8)
    assert asarray(v_arr).squeeze().shape == (10,4,100)
    log.debug("test successful")