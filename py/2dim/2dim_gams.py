import numpy as np
from time import time
from tqdm import tqdm
from joblib import Parallel,delayed
import multiprocessing
import json

from numba import jit, jitclass

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
    for i in range(5):
        a += i**2
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
## Global variables
beta=2.67
inv_beta = 2./beta
dt=.05
z_max=1.9
rho = 0.05
xi = xi_1
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
    while(xi_1(x_t,y_t) > rho):
        x_t,y_t = _update_state(x_t,y_t)
        x.append(x_t)
        y.append(y_t)
    return x,y



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

    def update(self):
        self.traj_x, self.traj_y =\
        update_traj(self.inh_traj_x,self.inh_traj_y)



############################################################
## Visulization of V(x,y)  
# X = np.arange(-1.5,1.5,0.1)
# Y = np.arange(-2,2,0.1)
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
############################################################

