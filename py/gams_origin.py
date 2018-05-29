import numpy as np
from time import time
from tqdm import tqdm
from joblib import Parallel,delayed
import multiprocessing
import json

############################################################
# One-dimensional example

## dynamics:
## 
## X_{i+1} - X_i = - \mu \Delta t + \sqrt{2 \beta ^{-1} \Delta t} G_i
## X_0 = x_0 

## rare-events:
## A = ]-\infty, a[  ;   B = ]b,+\infty[
## p = \mathbb{P}(\tau_b < \tau_a)

## reaction coordinate:
## \xi = id

## To accelerate the calculations, we consider the update
## procedure as follows:
## X_{i+1} = X_i + NUM1 + NUM2 * G_i
############################################################

# reset RandomState: It is not pertinant to use system
# time as 

def reset_random_state():
    f = open("/dev/random","rb")
    rnd_str = f.read(4)
    rnd_int = int.from_bytes(rnd_str, byteorder = 'big')
    np.random.seed(rnd_int)

class particle:
    def __init__(self,\
            ind_sur,\
            inherited_traj,\
            parent_index,\
            ancestor_index,\
            num_settings
            ):
        """
        ind_sur:
        index of survival, a list of length <step> which contains 0,1 This is
        only needed for var estimation of <keep_survived> resampling strategy.

        inherited_traj: 
        the trajectory inherited from the parent

        parent_index: 
        index of parents

        ancestor_index: 
        index of parents at step 0

        num_settings:
        dictionary, numerical settings of dynamics

        xi: 
        reaction coordinate function
        (not implemented in this case as we choose xi = id)
        """
        self.inh_traj = inherited_traj
        self.parent = parent_index
        self.ancestor = ancestor_index
        self.traj = self.inh_traj
        self.trans_traj = self.inh_traj
        self.max_level = 0.
        self.ind_sur = ind_sur
        ## numerical settings
        mu = num_settings['mu']
        beta = num_settings['beta']
        dt = num_settings['dt']
        
        self.NUM1 = -mu*dt
        self.NUM2 = np.sqrt(dt * 2./beta)
        self.a = num_settings['a']

    def update_state(self, state):
        """
        return the updated state X_{i+1} given X_i
        NUM1 and NUM2 is pre-calculated to improve the
        performance
        """
        state_update = state + self.NUM1 +\
                self.NUM2*np.random.normal(0,1,1)[0]
        return state_update

    def update_free(self):
        """
        return the whole trajectory stopped by tau_A

        update max level
        """

        traj = self.inh_traj
        while(traj[-1] > self.a):
            traj = np.append(traj, self.update_state(state = traj[-1]))
        self.traj = traj 
        self.calculate_max_level()

    def update_cond(self, Z_q):
        """
        return the whole trajectory stopped by tau_A
        given X_{T(Z_q)} > Z_q

        update max level
        """

        traj = self.inh_traj
        # propose a X_{T(Z_q)}
        state_candidate = self.update_state(state = traj[-1])
        # reject if X_{T(Z_q)} <= Z_q
        while(state_candidate <= Z_q):
            state_candidate = self.update_state(state = traj[-1])
        # update X_{T(Z_q)}
        traj = np.append(traj,state_candidate)

        while(traj[-1] > self.a):
            traj = np.append(traj, self.update_state(state = traj[-1]))
        self.traj = traj 
        self.calculate_max_level()


    def calculate_max_level(self):
        self.max_level = np.max(self.traj)

    def calculate_tran_traj_origin(self, Z_q):
        """
        Z_q: current level at this step Calculate the
        transmissible trajectory from the beginning to
        the first moment under the current level
        """
        i = self.inh_traj.shape[0] - 1
        while(self.traj[i] < Z_q):
            i += 1 
        self.trans_traj = self.traj[0:i+1]

    def calculate_tran_traj_rejection(self, Z_q):
        """
        Z_q: current level at this step
        Calculate the transmissible trajectory from the beginning to the last
        moment under the current level

        We store until the **last state** before going beyond the
        current level.
        """
        i = self.inh_traj.shape[0] - 1
        while(self.traj[i] < Z_q):
            i += 1 
        self.trans_traj = self.traj[0:i]
    
def naive_trial(par,b):
    """
    one trial of naive MC, ingredient for parallel
    version of Naive MC.
    """
    par.update_free()
    return float(par.max_level >b)

def Naive_MC(N, num_settings, naive_trial = naive_trial):
    x_0 = num_settings['x_0']
    b = num_settings['b']
    par = particle(1,x_0,1,1,num_settings)
    num_cores = multiprocessing.cpu_count()
    t_0 = time()
    results =\
            Parallel(n_jobs=num_cores)(delayed(naive_trial)(par,b)\
                    for i in tqdm(range(N)))
    print('------------------------------------------------------------')
    print('Naive Monte Carlo:')
    print('reference mean: ', np.mean(results))
    print('reference variance (naive MC): ', np.var(results))
    print('time spent(parallel): ', time() - t_0, ' s')
    print('------------------------------------------------------------')


def calculate_level(layer, k):
    """
    return the current level given a layer of particles
    and k number of particles to kill
    """
    n_rep = len(layer)
    list_max_levels = [layer[i].max_level for i in range(n_rep)]
    #print(sorted(list_max_levels))
    return np.partition(list_max_levels,k)[k]


def GAMS_original(n_rep,\
    k,\
    num_settings,\
    selection_method = 'multinomial'):
    """
    Implementation of GAMS algorithm

    selection_method:
    multinomial
    or
    keep_survived

    num_settings:
    dict, containing
    a,b,z_max,x_0,beta,mu,dt
    """
    reset_random_state()
    ## numerical settings
    x_0 = num_settings['x_0']
    a = num_settings['a']
    b = num_settings['b']
    z_max = num_settings['z_max']
    mu = num_settings['mu']
    beta = num_settings['beta']
    dt = num_settings['dt']
    
    NUM1 = -mu*dt
    NUM2 = np.sqrt(dt * 2./beta)
    ## end of numerical settings

    ## Initiation (step 0)
    parsys = [[]]
    for i in range(n_rep):
        par = particle([],x_0,i,i,num_settings)
        #par.update_cond(Z_q = x_0)
        par.update_free()

        parsys[0] += [par]

    step  = 0 
    current_level = calculate_level(parsys[step],k)
    if(current_level <= x_0):
        ech = [float(parsys[step][i].max_level>=b) for i in range(n_rep)]
        E = np.mean(ech)
        V = E**2 - 1./(n_rep*(n_rep-1))*((E*n_rep)**2-\
            np.sum([ech[i]**2 for i in range(n_rep)]))

        return [E,V]
    ## Evolution 
    K = []
    while(current_level<z_max):
        I_on = []
        I_off = []
        # we ensure that we kill k particles at each step
        for i in range(n_rep):
            if parsys[step][i].max_level <= current_level:
                I_off += [i]
            else:
                I_on += [i]
        if len(I_off) == n_rep:
            # stop when distinction happens
            break

        K += [float(len(I_off))]
        # calculate transimissible trajectory
        # for the survived particles
        for i in I_on:
            parsys[step][i].calculate_tran_traj_origin(Z_q = current_level)

        parsys += [[]] # add an empty layer
        if selection_method == 'multinomial':
            for i in range(n_rep):
                parent_id = np.random.choice(I_on,size = 1)[0]
                parent = parsys[step][parent_id]
                
                par = particle([],\
                        parent.trans_traj,\
                        parent.parent,\
                        parent.ancestor,num_settings)
                par.update_free()
                # while(par.max_level < current_level):
                #     par.update_cond_cond(Z_q = current_level)
                parsys[step+1] += [par]

        elif selection_method == 'keep_survived':
            for i in range(n_rep):
                if i in I_off:
                    parent_id = np.random.choice(I_on,1)[0]
                    parent = parsys[step][parent_id]
                    par = particle(parent.ind_sur+[0],\
                            parent.trans_traj,\
                            parent.parent,\
                            parent.ancestor,num_settings)
                elif i in I_on:
                    parent_id = i
                    parent = parsys[step][parent_id]
                    par = particle(parent.ind_sur+[1],\
                            parent.trans_traj,\
                            parent.parent,\
                            parent.ancestor,num_settings)

                par.update_free()
                parsys[step+1] += [par]

        # update step number and calculate next level
        step += 1
        # print('level: ', current_level)
        current_level = calculate_level(parsys[step],k)
        
    p_n = np.mean([float(parsys[step][i].max_level>b) for i in range(n_rep)])
    gamma_1 = np.prod([1. - K[i]/float(n_rep) for i in range(step)])
    E = p_n * gamma_1

    if selection_method == 'multinomial':
        anc_list = [parsys[step][i].ancestor for i in range(n_rep)]
        anc_list = list(set(anc_list))
        V = E**2 -\
                gamma_1**2*(n_rep**(step-1)/(n_rep-1.)**(step+1))\
                *((p_n *n_rep)**2 -\
                        np.sum([np.sum([\
                                float(parsys[step][j].max_level>=b)\
                            for j in range(n_rep) if
                            parsys[step][j].ancestor == i\
                                    ])**2\
                                    for i in anc_list]))
    elif selection_method == 'keep_survived':

        layer = parsys[step]
        NUM3 = 0
        for i in range(n_rep):
            for j in range(n_rep):
                if layer[i].ancestor != layer[j].ancestor:

                    NUM5=1
                    for m in range(step):
                        if layer[i].ind_sur[m] + layer[j].ind_sur[m] <= 1:  
                            NUM5 *= ((n_rep-float(K[m]))/n_rep)**2
                        elif layer[i].ind_sur[m] + layer[j].ind_sur[m] == 2:  
                            NUM5 *=\
                            ((n_rep-float(K[m]))/n_rep)*((n_rep-float(K[m])-1)/(n_rep-1.))

                    NUM3 +=\
                    float(layer[i].max_level>b)*\
                    float(layer[j].max_level>b)*\
                    ((n_rep-1.)/n_rep)**\
                    np.sum(np.multiply(layer[i].ind_sur,layer[j].ind_sur))\
                    *NUM5

        # NUM4 = 0
        # for i in range(n_rep):
        #     NUM4 +=\
        #     float(layer[i].max_level>b)\
        #     *np.prod([1.-float(K[m])/n_rep for m in range(step) \
        #     if layer[i].ind_sur[m] >= 0 ])
        #E=NUM4/float(n_rep)
                    # print(step)
                    # print(layer[i].ind_sur)
                    # print('#')
                    # print(layer[j].ind_sur)
                    # print(np.sum(np.multiply(layer[i].ind_sur,layer[j].ind_sur)))

        #V = E**2-gamma_1**2*(n_rep**(step-1)/(n_rep-1.)**(step+1))*NUM3
        V =\
        E**2-(n_rep**(step-1)/(n_rep-1.)**(step+1))*NUM3

    return [E,V]



def GAMS_rejection(n_rep,\
        k,\
        num_settings,\
        selection_method = 'multinomial'):
    """
    Implementation of GAMS algorithm

    same input as GAMS_originalal
    Removing the extinction by resampling by rejection method.
    the number of killed particles at each step is
    exactly k.
    """
    reset_random_state() 
    ## numerical settings
    x_0 = num_settings['x_0']
    a = num_settings['a']
    b = num_settings['b']
    z_max = num_settings['z_max']
    mu = num_settings['mu']
    beta = num_settings['beta']
    dt = num_settings['dt']
    
    NUM1 = -mu*dt
    NUM2 = np.sqrt(dt * 2./beta)
    ## end of numerical settings

    ## Initiation (step 0)
    parsys = [[]]
    for i in range(n_rep):
        par = particle([1],x_0,i,i,num_settings)
        #par.update_cond(Z_q = x_0)
        par.update_free()

        parsys[0] += [par]

    step  = 0 
    current_level = calculate_level(parsys[step],k)
    # deal with the situation when stopped at step 0
    if current_level <= x_0:
        ech = [float(parsys[step][i].max_level>=b) for i in range(n_rep)]
        E = np.mean(ech)
        V = E**2 - 1./(n_rep*(n_rep-1))*((E*n_rep)**2-\
            np.sum([ech[i]**2 for i in range(n_rep)]))

        return [E,V]


    ## Evolution 
    while(current_level<z_max):
        parsys += [[]] # add nem empty layer
        I_on = []
        I_off = []
        # We ensure that we kill k particles at each
        # step. This could be done by resampling
        # conditionnally X_{T_{Z_q}}^{(i,q)} > Z_q.
        for i in range(n_rep):
            if parsys[step][i].max_level < current_level:
                I_off += [i]
            else:
                I_on += [i]
        # calculate transimissible trajectory
        # for the survived particles
        for i in I_on:
            parsys[step][i].calculate_tran_traj_rejection(Z_q = current_level)
            

        if selection_method == 'multinomial':
            for i in range(n_rep):
                parent_id = np.random.choice(I_on,size = 1)[0]
                parent = parsys[step][parent_id]
                
                par = particle(i,\
                        parent.trans_traj,\
                        parent.parent,\
                        parent.ancestor,num_settings)
                par.update_cond(Z_q = current_level)
                parsys[step+1] += [par]

        elif selection_method == 'keep_survived':
            for i in range(n_rep):
                if i in I_off:
                    parent_id = np.random.choice(I_on,1)[0]
                    parent = parsys[step][parent_id]
                    par = particle(parent.ind_sur+[0],\
                            parent.trans_traj,\
                            parent.parent,\
                            parent.ancestor,num_settings)
                elif i in I_on:
                    parent_id = i
                    parent = parsys[step][parent_id]
                    par = particle(parent.ind_sur+[1],\
                            parent.trans_traj,\
                            parent.parent,\
                            parent.ancestor,num_settings)

                par.update_cond(Z_q = current_level)
                # while(par.max_level < current_level):
                #     par.resample(Z_q = current_level)
                parsys[step+1] += [par]

        # update step number and calculate next level
        step += 1
        current_level = calculate_level(parsys[step],k)
    # calculation of approximation    
    p_n = np.mean([float(parsys[step][i].max_level>=b) for i in range(n_rep)])
    gamma_1 = (1. - float(k)/n_rep)**step
    E = p_n * gamma_1 

    if selection_method == 'multinomial':
        anc_list = [parsys[step][i].ancestor for i in range(n_rep)]
        anc_list = list(set(anc_list))
        V = E**2 -\
                (n_rep-float(k))**(2*step)/((n_rep*(n_rep-1.))**(step+1)) *\
                ((p_n *n_rep)**2 -\
                np.sum([np.sum([\
                float(parsys[step][j].max_level>=b)\
                for j in range(n_rep)\
                if parsys[step][j].ancestor == i])**2\
                for i in anc_list]))

    elif selection_method == 'keep_survived':

        layer = parsys[step]
        NUM3=0
        for i in range(n_rep):
            for j in range(n_rep):
                if layer[i].ancestor != layer[j].ancestor:
                    NUM3 +=\
                    float(layer[i].max_level>b)*\
                    float(layer[j].max_level>b)\
                    *((n_rep-1.)/n_rep)**\
                    np.sum(np.multiply(layer[i].ind_sur,layer[j].ind_sur))

        V =E**2-(n_rep-float(k))**(2*step)/((n_rep*(n_rep-1.))**(step+1))*NUM3
                



    return [E,V]


def parallel_simulation(framework,\
                        num_settings,\
                        n_rep,\
                        k_test,\
                        n_sim,\
                        log_file='results.log',\
                        json_file='results.json',\
                        log_print=False):

    ## numerical settings
    x_0 = num_settings['x_0']
    a = num_settings['a']
    b = num_settings['b']
    z_max = num_settings['z_max']
    mu = num_settings['mu']
    beta = num_settings['beta']
    dt = num_settings['dt']
    
    NUM1 = -mu*dt
    NUM2 = np.sqrt(dt * 2./beta)
    ## end of numerical settings

    num_cores = multiprocessing.cpu_count()
    if num_cores >300:
        num_cores -= 10
    t_0 = time()
    results =\
    Parallel(n_jobs=num_cores)(delayed(framework)\
    (n_rep = n_rep, k = k_test,\
    num_settings = num_settings,\
    selection_method=method_test)\
                    for i in tqdm(range(n_sim)))
    E_list = [results[i][0] for i in range(n_sim)]
    V_list = [results[i][1] for i in range(n_sim)]
    E_mean = np.mean(E_list)
    V_naive = np.var(E_list)
    V_mean = np.mean(V_list)
    delta_naive = 2*1.96*np.sqrt(V_naive/n_rep)
    delta_var_est = 2*1.96*np.sqrt(V_mean/n_rep)
    results_dict={\
        'E_list':E_list,\
        'V_list':V_list,\
        'E_mean':E_mean,\
        'V_naive':V_naive,\
        'V_mean':V_mean,\
        'delta_naive':delta_naive,\
        'delta_var_est':delta_var_est,\
        }
    if log_file: 
        file = open(log_file,'w')
        file.write('------------------------------------------------------------\n')
        file.write('GAMS: \n')
        file.write("number of CPUs: "+ str(num_cores)+'\n')
        file.write('a: '+str(num_settings['a'])+\
            '\t'+'b: '+str(num_settings['b'])+'\t'+'dt: '+str(num_settings['dt'])+'\n')
        file.write('n_rep: '+str(n_rep)+'\t'+'k: '+str(k_test)+'\t'+'n_sim: '+str(n_sim)+'\n')
        file.write('sampling method: '+ str(method_test)+'\n')
        file.write('------------------------------------------------------------'+'\n')
        file.write('mean: '+str(E_mean)+'\n')
        file.write('naive var estimator: '+str( V_naive)+'\n')
        file.write('mean of var estimator: '+str( V_mean)+'\n')
        file.write('var of variance estimator: '+str(np.var(V_list))+'\n')
        file.write('delta: '+str(delta_naive)+'\n')
        file.write('delta (by var estimator):'+str(delta_var_est)+'\n')
        file.write('time spent (parallel):  '+ str(time() - t_0)+' s'+'\n')
        file.write('------------------------------------------------------------\n')
        file.close()
    if json_file:
        with open(json_file, 'w') as f:
            json.dump(results_dict, f)

    if log_print: 
        print('------------------------------------------------------------')
        print('GAMS: ')
        print("number of CPUs: "+ str(num_cores))
        print('a: '+str(num_settings['a'])+\
            '\t'+'b: '+str(num_settings['b'])+'\t'+'dt: '+str(num_settings['dt']))
        print('n_rep: '+str(n_rep)+'\t'+'k: '+str(k_test)+'\t'+'n_sim: '+str(n_sim))
        print('sampling method: '+ str(method_test))
        print('------------------------------------------------------------')
        print('mean: '+str(E_mean))
        print('naive var estimator: '+str( V_naive))
        print('mean of var estimator: '+str( V_mean))
        print('var of variance estimator: '+str(np.var(V_list)))
        print('delta: '+str(delta_naive))
        print('delta (by var estimator):'+str(delta_var_est))
        print('time spent (parallel):  '+ str(time() - t_0)+' s')
        print('------------------------------------------------------------\n')
    #return results_dict


if __name__ == '__main__':


    ## numeric settings (same as in 1-dim example in GAMS.pdf):
    k_test_list = [1,5,30,50,70]
    for k in [50]:
        num_settings = {\
        'x_0' : np.array([1.]),\
        'a' : 0.1,\
        'b' : 4.9,\
        'z_max' : 4.9,\
        'mu' : 1.,\
        'beta':1.,\
        'dt' : 0.01,\
        }
        
        n_rep = 100
        k_test = k 
        n_sim = 500000
        
        ## resampling strategy
        
        method_test = 'keep_survived'
        #method_test = 'multinomial'
        
        ## GAMS settings
        GAMS_type = 'original' 
        #GAMS_type = 'rejection' 
        
        info = GAMS_type +'_'+ method_test +'_n_rep_'+str(n_rep)\
               +'_k_'+str(k_test) + '_n_sim_'+str(n_sim)\
               +'_a_'+str(num_settings['a'])+'_b_'+str(num_settings['b'])\
               +'_beta_'+str(num_settings['beta'])+'_dt_'+str(num_settings['dt'])\
               +'_mu_'+str(num_settings['mu'])
        parallel_simulation(framework = GAMS_original,\
                                  num_settings = num_settings,\
                                  n_rep = n_rep,\
                                  k_test = k_test,\
                                  n_sim = n_sim,\
                                  log_file='log/'+info+'.log',\
                                  json_file='json/'+info+'.json',\
                                  #log_file=False,\
                                  #json_file=False,\
                                  log_print=True)
    # read .json file
    # with open('json/'+info+'.json') as f:
    #       my_dict = json.load(f)

    # Naive MC reference
    # Naive_MC(10000000, num_settings)
