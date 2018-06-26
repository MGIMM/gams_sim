import numpy as np
from time import time
from tqdm import tqdm
from joblib import Parallel,delayed
import multiprocessing
import json
#from math import sqrt, exp

from numba import jit, jitclass, float32, int32,float64

############################################################
# Two-dimensional bi-channel example

## dynamics: overdamped Langevin process
## d X_t =  -\Delta V (X_t)dt + \sqrt{2\beta^{-1}}d W_t 
## X_0 = x_0 = (-0.9,0); dt = 0.05

## V(x,y) = 0.2x^4+0.2(y-\frac{1}{3})^4 + 3 e^{-x^2
## -(y-\frac{1}{3})^2} -3 e^{-x^2-(y-\frac{5}{3})^2}
## -5 e^{-(x-1)^2-y^2} - 5e^{-(x+1)^2 - y^2}

## rare events:
## m_A = (x_A,y_A) = (-1,0) 
## m_B = (x_B,y_B) = (1,0) 
## A = B(m_A,\pho) 
## B = B(m_B,\pho) 
## with \rho = 0.05 \in (0,1)

## reaction coordinate:
## \xi^1 (x,y) = \sqrt{(x - m_A)^2 + (y-y^A)^2}
## \xi^2 (x,y) = \xi^1(x,y) - \sqrt{(x - m_B)^2 + (y-y^B)^2}
## \xi^3 (x,y) = x

## inverse temprature:
## \beta \in \{8.67,9.33,10\}

## reference probability value:
## (1\cdot 10^{-10},2\cdot 10^{-9})
############################################################

def reset_random_state():
    a = 0
    a = 1+2.
    f = open("/dev/random","rb")
    rnd_str = f.read(4)
    rnd_int = int.from_bytes(rnd_str, byteorder = 'big')
    np.random.seed(rnd_int)

# potential
@jit
def V(x,y):
    x2 = x**2
    y2 = y**2
    y13 = (y-1./3.)**2
    y35 = (y-5./3.)**2
    V = 0.2*x2**2 + 0.2*y13**2 +3*np.exp(-x2 -y13) -3*np.exp(-x2-y35)\
        -5*np.exp(-(x-1.)**2-y2) - 5*np.exp(-(x+1)**2-y2)
    return V

@jit
def gradV(x,y):
    dV_x = 0.8*x**3\
            -6*x*np.exp(-x**2 - (y-1./3.)**2)\
            +6.*x*np.exp(-x**2-(y-5./3.)**2)\
            +10.*(x-1.)*np.exp(-(x-1.)**2-y**2)\
            +10.*(x+1.)*np.exp(-(x+1.)**2-y**2)
    dV_y = 0.8*(y-1./3.)**3\
            -(6.*y-2.)*np.exp(-x**2-(y-1./3.)**2)\
            +(6.*y-10.)*np.exp(-x**2-(y-5./3.)**2)\
            +10.*y*(np.exp(-(x-1.)**2 - y**2)\
            + np.exp(-(x+1.)**2-y**2))
    return dV_x,dV_y

# reaction coordinates
@jit
def xi_1(x, y):
    return np.sqrt((x+1.)**2+y**2)
    
@jit
def xi_2(x, y):
    return 2. -np.sqrt((x-1.)**2 + y**2) 

@jit
def xi_3(x, y):
    return x 


############################################################
## global variables
beta=np.float64(4.67)
inv_beta = np.float64(2./beta)
dt=np.float64(.05)

## rare events settings
z_max=np.float64(1.9)
rho = np.float64(0.05)
X_0 = np.float64(-0.9)
#z_max=np.float64(0.7)
#rho = np.float64(0.1)
#X_0 = np.float64(-0.7)

Y_0 = np.float64(0.)
## reaction coordinate settings

xi = xi_1
n_rep_test = 100
k_test = 5 
method_test = 'keep_survived'
#method_test = 'multinomial'
n_sim = 10000
############################################################

@jit
def _update_state(x,y):
    g1,g2 = gradV(x,y)
    NUM1 = np.sqrt(inv_beta*dt)
    x_new = x - g1*dt+NUM1*np.random.normal(size=1)[0]
    y_new = y - g2*dt+NUM1*np.random.normal(size=1)[0]
    return x_new,y_new

@jit
def update_traj(x,y):
    x_t = x[-1]
    y_t = y[-1]
    while(xi_1(x_t,y_t) > rho and (x_t-1)**2 + y_t**2>rho**2):
        x_t,y_t = _update_state(x_t,y_t)
        x.append(x_t)
        y.append(y_t)
    return x,y

@jit
def find_max_level(x,y):
    levels = []
    for i in range(len(x)):
        levels.append(xi(x[i],y[i]))
    return np.max(levels)


# spec = [\
#         ('traj_x', float32[:]),\
#         ('traj_y', float32[:]),\
#         ('inh_traj_x', float32[:]),\
#         ('inh_traj_y', float32[:]),\
#         ('survive_history', int32[:]),\
#         ('parent', int32),\
#         ('ancestor', int32),\
#        ]

class particle:
    def __init__(self,\
            inherited_traj_x,\
            inherited_traj_y,\
            survive_history,\
            parent_index,\
            ancestor_index):
        """
        :parameter inherited_traj: trajectory of parent until the
                   first time beyond the current level.
                   Seperate x and y to compile with numba. 
        :parameter survive_history: 0-1 valued list, indicating
                   survive or not at the corresponding step.
        :parameter parent_index: int index of parent
        :parameter ancestor_index: int index of parent at step 0
        :method    update_traj: run Markov transition kernel to
                   until it reaches A.
        """
        self.traj_x = []
        self.traj_y = []
        self.inh_traj_x = inherited_traj_x
        self.inh_traj_y = inherited_traj_y
        self.survive_history = survive_history
        self.parent = parent_index
        self.ancestor = ancestor_index

    @property
    def max_level(self):
        return(find_max_level(self.traj_x,self.traj_y))

    def update(self):
        self.traj_x, self.traj_y =\
        update_traj(self.inh_traj_x,self.inh_traj_y)
def get_transmissible_traj(x,y,Z):
    """
    parameter: x: traj_x
    parameter: y: traj_y
    parameter: Z: current level

    return: trans_traj_x,trans_traj_y

    we remark that we stop at the first time when the level
    of the trajectory is above the current level.
    """
    trans_traj_x = []
    trans_traj_y = []
    x_t = X_0
    y_t = Y_0
    t=0
    while(xi(x_t,y_t)<=Z):
        trans_traj_x.append(x_t)
        trans_traj_y.append(y_t)
        t += 1
        x_t = x[t]
        y_t = y[t]
    trans_traj_x.append(x_t)
    trans_traj_y.append(y_t)
    return trans_traj_x,trans_traj_y

############################################################
## Visulization of V(x,y)  
# X = np.arange(-2.5,2.5,0.1)
# Y = np.arange(-3,3,0.1)
# x_grid,y_grid = np.meshgrid(X,Y)
# import matplotlib.pyplot as plt
# from matplotlib import cm
# plt.figure(figsize=(15,10))
# plt.contour(x_grid,y_grid,V(x_grid,y_grid),25,cmap=cm.gist_heat)
# par = particle([-0.5],[0.],[],2,3)
# par.update()
# 
# plt.quiver(np.array(par.traj_x[:-1]),np.array(par.traj_y[:-1]),\
#         np.array(par.traj_x[1:])-np.array(par.traj_x[:-1]),\
#         np.array(par.traj_y[1:])-np.array(par.traj_y[:-1]),\
#         scale_units='x', angles='xy', scale=15,\
#         color='darkred',alpha = 0.3)
# plt.plot(par.traj_x[:-1],par.traj_y[:-1],'-',color='darkred')
# plt.show()
# trans_x,trans_y =\
# get_transmissible_traj(par.traj_x,par.traj_y,par.max_level-0.3)
# plt.plot(xi(np.array(par.traj_x),np.array(par.traj_y)))
# plt.show()
############################################################


############################################################
## GAMS



@jit
def calculate_level(list_max_levels,k):
    """
    return the current level given a layer of particles
    and k the minimum number of particles to kill
    """
    return np.partition(list_max_levels,k)[k-1]
@jit
def varphi(x,y):
    return np.float64((x-1.)**2 +y**2 <= rho**2)

def GAMS(n_rep,k,selection_method):
    """
    Implementation of original GAMS algorithm (without
    resamplling to remove extinction). 
    """
    # prevent identical random seed for parallel computing
    reset_random_state()

    parsys = [[]]
    list_max_levels=[]
    for i in range(n_rep):
        par = particle([X_0],[Y_0],[],i,i)
        par.update()
        list_max_levels.append(par.max_level)
        parsys[0].append(par)

    step  = 0 
    current_level = calculate_level(list_max_levels,k)
    #print('level: ', current_level)
    # Initiation of K
    K=[]
    ## Evolution 
    __=0
    while(current_level<z_max):
        __+=1
        if __>50000:
            break
        list_max_levels = []
        I_on = []
        I_off = []
        for i in range(n_rep):
            if parsys[step][i].max_level <= current_level:
                I_off.append(i)
            else:
                I_on.append(i)
        if len(I_off) == n_rep:
            # stop when distinction happens
            break
        K.append(np.float64(len(I_off)))
        #print(K[step])
        parsys.append([]) # add an empty layer
        if selection_method == 'multinomial':
            for i in range(n_rep):
                parent_id = np.random.choice(I_on,size = 1)[0]
                parent = parsys[step][parent_id]
                tr_x,tr_y = get_transmissible_traj(parent.traj_x,\
                            parent.traj_y,\
                            current_level)
                par = particle(tr_x,tr_y,\
                        parent.survive_history,\
                        parent.parent,\
                        parent.ancestor)
                par.update()
                parsys[step+1].append(par)
                list_max_levels.append(par.max_level)

        elif selection_method == 'keep_survived':
            for i in range(n_rep):
                if i in I_off:
                    parent_id = np.random.choice(I_on,1)[0]
                    parent = parsys[step][parent_id]
                    tr_x,tr_y = get_transmissible_traj(parent.traj_x,\
                                parent.traj_y,\
                                current_level)
                    par = particle(tr_x,tr_y,\
                          parent.survive_history,\
                          parent.parent,\
                          parent.ancestor)
                    par.survive_history.append(np.float64(0))

                elif i in I_on:
                    parent_id = i
                    parent = parsys[step][parent_id]
                    tr_x,tr_y = get_transmissible_traj(parent.traj_x,\
                                parent.traj_y,\
                                current_level)
                    par = particle(tr_x,tr_y,\
                          parent.survive_history,\
                          parent.parent,\
                          parent.ancestor)
                    par.survive_history.append(np.float64(1))
                par.update()
                list_max_levels.append(par.max_level)
                parsys[step+1].append(par)

        # update step number and calculate next level
        step += 1
        #print('level: ', current_level)
        current_level = calculate_level(list_max_levels,k)

    ## Estimations
    list_Qiter = []
    for i in range(n_rep):
        list_Qiter.append(varphi(parsys[step][i].traj_x[-1],parsys[step][i].traj_y[-1]))
    E_SUM = np.sum(list_Qiter)
    n_rep_int = n_rep
    n_rep = np.float64(n_rep)
    E = E_SUM/n_rep
    gamma_1 = np.float64(1)
    #Q = np.float64(step)
    for i in range(step):
        gamma_1 *= (n_rep - K[i])/n_rep
    E *= gamma_1
    if step == 0:
        #selection_method ='multinomial'
        V = E**2 - np.float64(1)/(n_rep*(n_rep-np.float64(1)))*((E_SUM)**2-\
            np.sum([list_Qiter[i]**2 for i in range(n_rep_int)]))
        return E,V
    if selection_method =='multinomial':
        list_anc = [parsys[step][i].ancestor for i in\
                range(n_rep_int)]
        list_anc = list(set(list_anc))
        NUM3 =np.float64(0)
        if len(list_anc) >= 2:
            NUM3 = E_SUM**2 -\
                    np.sum([np.sum([varphi(parsys[step][j].traj_x[-1],parsys[step][j].traj_y[-1])\
                    for j in range(n_rep_int)\
                    if parsys[step][j].ancestor == i])**2\
                    for i in list_anc])
            V = E**2
            V -= gamma_1**2 *((n_rep/(n_rep-np.float64(1)))**(step+1)/n_rep**2)*NUM3
        else:
            V = E**2
    if selection_method == 'keep_survived':
        NUM3 = np.float64(0) 
        list_anc = [parsys[step][i].ancestor for i in\
                range(n_rep_int)]
        list_anc = list(set(list_anc))
        if len(list_anc) >= 2:
            layer = parsys[step]
            for i in range(n_rep_int):
                for j in range(n_rep_int):
                    if layer[i].ancestor != layer[j].ancestor:

                        NUM5=np.float64(1)
                        for m in range(step):
                            if np.float64(layer[i].survive_history[m] +\
                                layer[j].survive_history[m]) <=\
                                np.float64(1):  
                                NUM5 *= ((n_rep-K[m])/n_rep)**2
                            elif np.float64(layer[i].survive_history[m] +\
                                    layer[j].survive_history[m]) ==\
                                    np.float64(2):  
                                NUM5 *=\
                                ((n_rep-K[m])/n_rep)*((n_rep-K[m]-np.float64(1))/n_rep)

                        NUM3 +=\
                        varphi(parsys[step][i].traj_x[-1],parsys[step][i].traj_y[-1])*\
                        varphi(parsys[step][j].traj_x[-1],parsys[step][j].traj_y[-1])*NUM5

        else:
            V=E**2
        V =\
        E**2-NUM3*((n_rep/(n_rep-np.float64(1)))**(step+1))/n_rep/n_rep
                
        
    # print('E: {}'.format(E)) 
    #print('number of eves:',len(list_anc))
    
    return E,V,len(list_anc)
from time import time    
# t0=time()
# a = GAMS(n_rep_test,k_test,selection_method='keep_survived')
# print('time used: {} s'.format(time()-t0))
# 
############################################################
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, writers
# from matplotlib import cm
# def Viz_layer(layer):
#     X = np.arange(-2,2,0.01)
#     Y = np.arange(-2,2,0.01)
#     x_grid,y_grid = np.meshgrid(X,Y)
#     
#     fig = plt.figure(figsize = (15,9))
#     p1 = plt.axes(xlim=(-2, 2), ylim=(-2,2))
#     p1.contour(x_grid,y_grid,V(x_grid,y_grid),35,cmap=cm.gray)
#     
#     xdata = [[] for i in range(n_rep_test)]
#     ydata = [[] for i in range(n_rep_test)]
#     #xdata,ydata = [[],[] for i in range(10)]
#     lines = []
#     #plotcols = ['darkred','darkgreen']
#     for index in range(n_rep_test):
#         lobj = p1.plot([],[],'-o',markersize = 3, alpha = 0.1)[0]
#         lines.append(lobj)
#     
#     num_frames = np.max([len(layer[i].traj_x) for i in\
#         range(n_rep_test)]) - 1
#     def update(frame):
#         for i in range(n_rep_test):
#             if frame < len(layer[i].traj_x):
#                 xdata[i].append(layer[i].traj_x[frame])
#                 ydata[i].append(layer[i].traj_y[frame])
#             else:
#                 xdata[i].append(layer[i].traj_x[-1])
#                 ydata[i].append(layer[i].traj_y[-1])
#                 
#         for ind,line in enumerate(lines):
#             line.set_data(xdata[ind], ydata[ind])
#         return lines
#     
#     ani = FuncAnimation(fig, update, frames=num_frames,\
#                         blit = True, interval = 0.01)
#     Writer = writers['ffmpeg']
#     writer = Writer(fps=10, metadata=dict(artist='MG'), bitrate=180)    
#     #ani.save('im_little.mp4', writer=writer)
#     plt.show()
# 
# #Viz_layer(a)
# ############################################################
## parallelization

num_cores = multiprocessing.cpu_count()
if num_cores >300:
    num_cores -= 10
t_0 = time()
results =\
Parallel(n_jobs=num_cores)(delayed(GAMS)\
(n_rep = n_rep_test, k = k_test,\
selection_method=method_test)\
                for i in tqdm(range(n_sim)))



E_list = [results[i][0] for i in range(n_sim)]
V_list = [results[i][1] for i in range(n_sim)]
nb_eve_list = [results[i][2] for i in range(n_sim)]

@jit
def final_calculs():
    V_naive_list = []
    V_mean_list = []
    E_mean = []
    for i in range(n_sim):
        V_naive_list.append(np.var(E_list[0:i]))
        V_mean_list.append(np.mean(V_list[0:i]))
        E_mean.append(np.mean(E_list[0:i]))
    results_dict={\
        'E':E_mean,\
        'nb_eve':nb_eve_list,\
        'V_naive':V_naive_list,\
        'V_mean':V_mean_list
        }
    return results_dict
    
# V_naive_list = [np.var(E_list[0:i]) for i in range(n_sim)]
# V_mean_list = [np.mean(V_list[0:i]) for i in range(n_sim)]
# NUM_delta = 1.96/np.sqrt(n_sim)
# # delta_naive_list = [np.sqrt(V_naive_list[i])*NUM_delta for i in range(n_sim)]
# # delta_mean_list = [np.sqrt(V_mean_list[i])*NUM_delta for i in range(n_sim)]
#  
E_mean = np.mean(E_list)
V_naive = np.var(E_list)
V_mean = np.mean(V_list)
# results_dict={\
#     'E':E_list,\
#     'V':V_list,\
#     'nb_eve':nb_eve_list,\
#     'V_naive':V_naive_list,\
#     'V_mean':V_mean_list,\
#     'delta_naive':delta_naive_list,\
#     'delta_mean':delta_mean_list
#     }
results_dict = final_calculs()
print('------------------------------------------------------------')
print('GAMS: ')
print("number of CPUs: "+ str(num_cores))
print('beta: ', beta)
print('n_rep: '+str(n_rep_test)+'\t'+'k: '+str(k_test)+'\t'+'n_sim: '+str(n_sim))
print('sampling method: '+ str(method_test))
print('reaction coordinate: '+ xi.__name__)
print('------------------------------------------------------------')
print('mean: '+str(E_mean))
print('naive var estimator: '+str( V_naive))
print('mean of var estimator: '+str( V_mean))
print('var of variance estimator: '+str(np.var(V_list)))
print('nb of eve (mean): '+str(np.mean(nb_eve_list)))
print('time spent (parallel):  '+ str(time() - t_0)+' s')
print('------------------------------------------------------------\n')


info = method_test+'_n_rep_'+\
        str(n_rep_test)+'_k_'+str(k_test)+'_n_sim_'\
        +str(n_sim)+'_beta_'+str(beta)+'_dt_'+str(dt)+'_'+xi.__name__
json_file = 'json_low_tem/'+info+'.json'
log_file = 'log_low_tem/'+info+'.log'

with open(json_file, 'w') as f:
    json.dump(results_dict, f)

file = open(log_file,'w')
file.write('------------------------------------------------------------\n')
file.write('GAMS: \n')
file.write("number of CPUs: "+ str(num_cores)+'\n')
file.write('beta: '+ str(beta)+'\n')
file.write('n_rep: '+str(n_rep_test)+'   '+'k: '+str(k_test)+'   '+'n_sim: '+str(n_sim)+'\n')
file.write('sampling method: '+ str(method_test)+'\n')
file.write('reaction coordinate: '+ xi.__name__ +'\n')
file.write('------------------------------------------------------------'+'\n')
file.write('mean: '+str(E_mean)+'\n')
file.write('naive var estimator: '+str( V_naive)+'\n')
file.write('mean of var estimator: '+str( V_mean)+'\n')
file.write('var of variance estimator: '+str(np.var(V_list))+'\n')
file.write('nb of eve (mean): '+str(np.mean(nb_eve_list))+'\n')
file.write('time spent (parallel):  '+ str(time() - t_0)+' s'+'\n')
file.write('------------------------------------------------------------\n')
file.close()
