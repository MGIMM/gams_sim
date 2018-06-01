import numpy as np
from time import time
from tqdm import tqdm
from joblib import Parallel,delayed
import multiprocessing
import json

from numba import autojit

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
@autojit
def V(x,y):
    x2 = x**2
    y2 = y**2
    y13 = (y-1./3.)**2
    y35 = (y-5./3.)**2
    V = 0.2*x2**2 + 0.2*y13**2 +3*np.exp(-x2 -y13) -3*np.exp(-x2-y35)\
        -5*np.exp(-(x-1.)**2-y2) - 5*np.exp(-(x+1)**2-y2)
    return V

@autojit
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
@autojit
def xi_1(x, y):
    return np.sqrt((x+1.)**2+y**2)
    
@autojit
def xi_2(x, y):
    return 2. -np.sqrt((x-1.)**2 + y**2) 

@autojit
def xi_3(x, y):
    return x 

############################################################
## Visulization of V(x,y)  
# X = np.arange(-1.5,1.5,0.1)
# Y = np.arange(-2,2,0.1)
# x_grid,y_grid = np.meshgrid(X,Y)
# import matplotlib.pyplot as plt
# from matplotlib import cm
# plt.contour(x_grid,y_grid,V(x_grid,y_grid),25,cmap=cm.gist_heat)
# plt.savefig('test.pdf')
############################################################

@autojit
def update_state(x,y, beta=8.67, dt=.05):
    NUM1 = dt*gradV(x,y)
    NUM2 = np.sqrt(2./beta*dt)
    x_new = x - NUM1+NUM2*np.random.normal(size=1)[0]
    y_new = y - NUM1+NUM2*np.random.normal(size=1)[0]
    return(x_new,y_new)
   
class particle:
    def __init__(self,\
            update_dynamic,\
            reaction_coordinate,\
            index_survive,\
            inherited_traj,\
            parent_index,\
            ancestor_index):
        """
        :parameter update_dynamic: transition kernel of Markov Chain
        :parameter index_survive: 0-1 valued list, indicating
                   survive or not at each step.
        :parameter inherited_traj: trajectory of parent until the first time beyond 
                   the current level.
        :parameter parent_index: int index of parent
        :parameter ancestor_index: int index of parent at step 0
        """

        self.inh_traj = inherited_traj
        self.parent = parent_index
        self.ancestor = ancestor_index
        self.traj = self.inh_traj
        self.trans_traj = self.inh_traj
        self.max_level = 0.
        self.ind_sur = index_survive
        self.xi = reaction_coordinate

    def calculate_max_level(self):
        self.max_level = np.max(self.traj)

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

