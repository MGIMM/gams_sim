import numpy as np
from time import time
from tqdm import tqdm


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

## numeric settings (same as in 1-dim example in GAMS.pdf):

x_0 = np.array([1.])
a = 0.8
b = 1.2
z_max = b
mu = 1.
beta1 = 8
beta2 = 24
dt = 0.01
n_rep = 200
k_test = 100
#N = 6e6

## To accelerate the calculations, we consider the update
## procedure as follows:
## X_{i+1} = X_i + NUM1 + NUM2 * G_i
## with
beta = beta1

NUM1 = -mu*dt
NUM2 = np.sqrt(dt * 2./beta)

####################

class particle:
    def __init__(self,\
            ind_sur,\
            inherited_traj,\
            parent_index,\
            ancestor_index,\
            ):
        """
        ind:
        index of particle at current step

        inherited_traj: 
        the trajectory inherited from the parent

        parent_index: 
        index of parents

        ancestor_index: 
        index of parents at step 0

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

    def update_state(self, state, NUM1 = NUM1, NUM2 = NUM2):
        """
        return the updated state X_{i+1} given X_i
        """
        state_update = np.sum([state, NUM1,\
                np.multiply(NUM2,np.random.normal(0,1,1)[0])]) 
        return state_update

    def update_free(self, a = a):
        """
        return the whole trajectory stopped by tau_A

        update max level
        """

        traj = self.inh_traj
        while(traj[-1] > a):
            traj = np.append(traj, self.update_state(state = traj[-1]))
        self.traj = traj 
        self.calculate_max_level()

    def update_cond(self, Z_q, a = a):
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

        while(traj[-1] > a):
            traj = np.append(traj, self.update_state(state = traj[-1]))
        self.traj = traj 
        self.calculate_max_level()


    def calculate_max_level(self):
        self.max_level = np.max(self.traj)

    def calculate_trans_traj(self, Z_q):
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

    
def calculate_level(layer, k = k_test):
    n_rep = len(layer)
    list_max_levels = [layer[i].max_level for i in range(n_rep)]
    #print(sorted(list_max_levels))
    return np.partition(list_max_levels,k)[k]


def sim_parsys(n_rep = n_rep,\
        k = k_test,\
        selection_method = 'multinomial'):
    """
    Implementation of GAMS algorithm

    selection_method:

    multinomial
    or
    keep_survived

    """

    ## Initiation (step 0)
    parsys = [[]]
    for i in range(n_rep):
        par = particle(0,x_0,i,i)
        #par.update_cond(Z_q = x_0)
        par.update_free()

        parsys[0] += [par]

    step  = 0 
    current_level = calculate_level(parsys[step])
    if current_level <= x_0:
        return [0,0]

    # ## (biased) remove the case where multiple particles have 
    # ## the same max level
    # while(current_level <= x_0):
    #     parsys = [[]]
    #     for i in range(n_rep):
    #         par = particle(i,x_0,i,i)
    #         #par.update_cond(Z_q = x_0)
    #         par.update_free()

    #         parsys[0] += [par]

    #     current_level = calculate_level(parsys[step])

    ## Evolution 
    # print(current_level)
    while(current_level<z_max):
        parsys += [[]] # add nem empty layer
        I_on = []
        I_off = []
        # we ensure that we kill k particles at each step
        for i in range(n_rep):
            if parsys[step][i].max_level < current_level:
                I_off += [i]
            else:
                I_on += [i]
        # calculate transimissible trajectory
        # for the survived particles
        for i in I_on:
            parsys[step][i].calculate_trans_traj(Z_q = current_level)
            parsys[step][i].ind_sur += 1

        if selection_method == 'multinomial':
            for i in range(n_rep):
                parent_id = np.random.choice(I_on,size = 1)[0]
                parent = parsys[step][parent_id]
                
                par = particle(i,\
                        parent.trans_traj,\
                        parent.parent,\
                        parent.ancestor)
                par.update_cond(Z_q = current_level)
                # while(par.max_level < current_level):
                #     par.update_cond_cond(Z_q = current_level)
                parsys[step+1] += [par]

        elif selection_method == 'keep_survived':
            for i in range(n_rep):
                if i in I_off:
                    parent_id = np.random.choice(I_on,1)[0]
                elif i in I_on:
                    parent_id = i

                parent = parsys[step][parent_id]
                par = particle(parent.ind_sur,\
                        parent.trans_traj,\
                        parent.parent,\
                        parent.ancestor)
                par.update_cond(Z_q = current_level)
                # while(par.max_level < current_level):
                #     par.resample(Z_q = current_level)
                parsys[step+1] += [par]

        # update step number and calculate next level
        step += 1
        # print('level: ', current_level)
        current_level = calculate_level(parsys[step])
        
    p_n = np.mean([float(parsys[step][i].max_level>=b) for i in range(n_rep)])
    p_n_ind_sur =\
    np.sum([float(parsys[step][i].max_level>=b)*((\
    ( n_rep-float(k) )/( n_rep-float(0) )\
    )**parsys[step][i].ind_sur) for i in range(n_rep)])
    # print('Q_iter',step)
    # return p_n * (1. - float(k)/n_rep)**step

    E = p_n * (1. - float(k)/n_rep)**step
    anc_list = [parsys[step][i].ancestor for i in range(n_rep)]
    anc_list = list(set(anc_list))
    # if step == 0:
    #   var_estim =\
    #       np.var([float(parsys[0][i].max_level>b) for i in range(n_rep)])
    if selection_method == 'multinomial':
        var_estim = E**2 -\
                (n_rep-float(k))**(2*step)/((n_rep*(n_rep-1.))**(step+1)) *\
                ((p_n *n_rep)**2 -
                        np.sum([np.sum([\
                                float(parsys[step][j].max_level>=b)
                            for j in range(n_rep) if
                            parsys[step][j].ancestor == i\
                                    ])**2\
                                    for i in anc_list]))
    elif selection_method == 'keep_survived':
        ####
        var_estim = E**2 -\
                (n_rep-float(k))**(2*step)/((n_rep*(n_rep-1.))**(step+1)) *\
                ((p_n_ind_sur)**2 -
                        np.sum([np.sum([\
                                float(parsys[step][j].max_level>=b)*((\
                                ( n_rep-float(k) )/( n_rep-float(0) )\
                                )**parsys[step][j].ind_sur)
                            for j in range(n_rep) if
                            parsys[step][j].ancestor == i\
                                    ])**2\
                                    for i in anc_list]))


    # delta = 2*1.96/np.sqrt(n_rep)*np.sqrt(var_estim)

    # print('E estim: ',E)
    # print('var_estim: ', var_estim)
    # print('delta:', delta)
    # return {'E':E, 'var_estim':var_estim,'delta':delta}
    return [E,var_estim]
    # print(list(set(E)))
    # return parsys

def NaiveMC(n_sim):
    """
    A naive Monte Carlo reference
    the reference value is around 0.00036
    same as in GAMS.pdf
    """
    t_0 = time()
    par = particle(1,x_0,1,1)
    I = 0.
    for i in tqdm(range(n_sim)):
        par.update_free()
        I += par.max_level >b
    estim = I/n_sim
    print("%.12f"%estim)
    print('time spent: ', time() - t_0, ' s')



if __name__ == '__main__':

    # # test particle class
    # par = particle(1,x_0,1,1)
    # par.update_cond(x_0)
    # print('traj:',par.traj)
    # print('max level: ',par.max_level)
    # par.calculate_trans_traj(Z_q = par.max_level-0.02)
    # print('transmissible traj: ', par.trans_traj)
    # par1 = particle(1,par.trans_traj,1,1)
    # par1.update_cond(Z_q = 1.15)
    # print('inh_traj: ',par1.inh_traj)
    # print('traj:',par1.traj)
    # print('max level: ',par1.max_level)

    # Calculation of reference probability
    # # for b = 1.3, the reference probability is 0.043
    # # for b = 1.2, the reference probability is 0.099
    # # for b = 1.1, the reference probability is 0.202
    cal_naive_ref = 0

    if cal_naive_ref:
        from joblib import Parallel,delayed
        import multiprocessing
        par = particle(1,x_0,1,1)
        num_cores = multiprocessing.cpu_count()
        print("number of CPUs: ", num_cores)
        def processInput():
            par.update_free()
            return float(par.max_level >b)
        t_0 = time()
        n_sim = 100000
        results =\
                Parallel(n_jobs=num_cores)(delayed(processInput)()\
                        for i in range(n_sim))
        print('reference value: ', np.sum(results)/float(n_sim))
        print('time spent(parallel): ', time() - t_0, ' s')
    ########################
    
    
    #sim_parsys(n_rep = n_rep, k = k_test, selection_method='keep_survived')
    test = 1
    if test:
        n_sim = 1000
        # method_test = 'keep_survived'
        method_test = 'multinomial'
    
        from joblib import Parallel,delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        t_0 = time()
        def processInput():
            return sim_parsys(n_rep = n_rep, k = k_test,
                    selection_method=method_test)
        results =\
                Parallel(n_jobs=num_cores)(delayed(processInput)()\
                        for i in tqdm(range(n_sim)))
        E_list = [results[i][0] for i in range(n_sim)]
        V_list = [results[i][1] for i in range(n_sim)]
        print('------------------------------------------------------------')
        print("number of CPUs: ", num_cores)
        print('a:',a,'\t','b:',b,'dt:',dt)
        print('n_rep:',n_rep,'\t','k:',k_test,'\t','n_sim:',n_sim)
        print('sampling method:', method_test)
        print('------------------------------------------------------------')
        print('mean:', np.mean(E_list))
        print('naive var:', np.var(E_list))
        print('mean var estim:', np.mean(V_list))
        print('mean var estim (with out 0):'\
            ,np.mean([V_list[i] for i in range(n_sim) if V_list[i]>0]))
        #print('delta:', 2*(1.96/np.sqrt(n_rep))*np.sqrt(np.var(E_list)))
        print('------------------------------------------------------------')
    
    # print('time spent(parallel): ', time() - t_0, ' s')
    # results = []
    # for i in tqdm(range(n_sim)):
    #     results += [sim_parsys(n_rep = n_rep, k = k_test,
    #         selection_method='keep_survived')]
    # 
    # #print(results) 
    # E_list = [results[i]['E'] for i in range(n_sim)]
    # V_list = [results[i]['var_estim'] for i in range(n_sim)]
    # print('mean: ', np.mean(E_list))
    # print('naive var: ', np.var(E_list))
    # print('mean var estim: ', np.mean(V_list))
    else:
        print(sim_parsys(n_rep = n_rep, k = k_test,\
                    selection_method='keep_survived'))
        


