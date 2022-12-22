import pymc as pm
import numpy as np
import aesara as ae
from aesara import tensor as at

eps = 0.001 # for numerical stability
err = 0.001
tune = 300

def _diffusion_01w(t, a, w):
    #K_n = at.sqrt(-2 * at.log(np.pi * t * err) / (np.pi**2 * t) ) + 1
    K = np.asarray([1,2,3,4,5]) #at.arange(K_n)
    tt=t/(a**2)
    prob_rt_std = np.pi * K * (at.exp( - ((K*np.pi)**2 * tt/2) ) + eps) * at.sin( K * np.pi * w )
    
    return prob_rt_std.sum()

def _get_logp(t, v, a, w):

    #prob_X = ( at.exp(-2*v*a) - at.exp(-2*v*w*a) ) / (at.exp(-2*v*a) - 1)
    prob_rt_std = _diffusion_01w(t, a, w)
    prob_rt = (1 / a**2) * at.exp( (-w*a*v) - (v**2 * t)/2 ) * prob_rt_std.sum()
    prob_rt = prob_rt #/ prob_X
    return prob_rt

def _individual_logp(t, X, v, a, w):

    t_correct = t[X]
    t_incorrect = t[1-X]

    total_logp = _get_logp(t_correct, -v, a, 1-w) + _get_logp(t_incorrect, v, a, w)
    return total_logp

def _diffusion_logp(RT, X, v, a, w, t_er):
    t = RT-t_er
    t = at.switch(at.le(t,0), eps,t)
    w = at.switch(at.ge(w,1), 0.99,w) # to avoid instability during intial evaluation.

    X = at.as_tensor(X)
    prob_rt, _ = ae.scan(lambda X_l, t_l: at.switch(at.le(t, 0), eps, _individual_logp(t_l, X_l, v, a, w)), sequences=[X,t])
    
    return prob_rt.sum()


def _test_edge_cases():

#v:-1.2002378916969234, a:0.017329851659947146, w:0.8280975715451416, t_er:0.6342687555621264, RT:[1.15521305 3.05237406 1.16819656 ... 2.47025695 2.03251366 2.46823881]

    v = np.repeat([-0.9249241152935039], 4)[:,np.newaxis]
    a = np.repeat([0.023750197074163027], 4)[:,np.newaxis]
    w = np.repeat([0.44120144255887783] , 4)[:,np.newaxis]
    #t_er=0.7100510973761837], 4)
    t_er=np.repeat([0.07100510973761837], 4) [:,np.newaxis]
    RTs = [[0.09, 1.39385687, 1.55661403, 1.88891806, 2.23878542, 0.92574541, 3.56532722],
            [0.19, 1.59385687, 2.55661403, 3.88891806, 1.23878542, 1.92574541, 5.56532722]
    ]
    
    X = np.random.randint(0,2,(4,5))
    RT = np.random.uniform(0,4,(4,5))
    
    lp = _diffusion_logp(RT, X, v,a,w,t_er)

    print(f"lp: {lp.eval()}")
    

if __name__ == "__main__":
    _test_edge_cases()
