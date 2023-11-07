# -*- coding: utf-8 -*-


import time
from pylab import *
import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import time
from pylab import *
import matplotlib.gridspec as gridspec
import numpy as np
#construct background states, observations with error

def x_to_y(X): # averaging in 2*2 windows (4 pixels)
    dim = X.shape[0]
    dim = 20
    Y = np.zeros((int(dim/2),int(dim/2)))
    for i in range(int(dim/2)):
        for j in range(int(dim/2)):
            Y[i,j] = X[2*i,2*j] + X[2*i+1,2*j] + X[2*i,2*j+1] + X[2*i+1,2*j+1]

            Y_noise = np.random.multivariate_normal(np.zeros(100),0.0000 * np.eye(100))
            Y_noise.shape = (10,10)
            Y = Y + Y_noise
    return Y


class shallow(object):

    # domain

    #N = 100
    #L = 1.
    #dx =  L / N
    #dt = dx / 100.

    # Initial Conditions

    #u = zeros((N,N)) # velocity in x direction
    #v = zeros((N,N)) # velocity in y direction

    #h_ini = 1.
    #h = h_ini * ones((N,N)) # pressure deviation (like height)
    #x,y = mgrid[:N,:N]

    time = 0

    plt = []
    fig = []


    def __init__(self, x=[],y=[],h_ini = 1.,u=[],v = [],dx=0.01,dt=0.0001, N=64,L=1., px=16, py=16, R=64, Hp=0.1, g=1., b=0.): # How define no default argument before?


        # add a perturbation in pressure surface


        self.px, self.py = px, py
        self.R = R
        self.Hp = Hp



        # Physical parameters

        self.g = g
        self.b = b
        self.L=L
        self.N=N

        # limits for h,u,v


        #self.dx =  self.L / self.N # a changer
        #self.dt = self.dx / 100.
        self.dx=dx
        self.dt=dt

        self.x,self.y = mgrid[:self.N,:self.N]

        self.u=zeros((self.N,self.N))
        self.v=zeros((self.N,self.N))

        self.h_ini=h_ini

        self.h=self.h_ini * ones((self.N,self.N))

        rr = (self.x-px)**2 + (self.y-py)**2
        self.h[rr<R] = self.h_ini + Hp #set initial conditions

        self.lims = [(self.h_ini-self.Hp,self.h_ini+self.Hp),(-0.02,0.02),(-0.02,0.02)]



    def dxy(self, A, axis=0):
        """
        Compute derivative of array A using balanced finite differences
        Axis specifies direction of spatial derivative (d/dx or d/dy)
        dA[i]/dx =  (A[i+1] - A[i-1] )  / 2dx
        """
        return (roll(A, -1, axis) - roll(A, 1, axis)) / (self.dx*2.) # roll: shift the array axis=0 shift the horizontal axis

    def d_dx(self, A):
        return self.dxy(A,1)

    def d_dy(self, A):
        return self.dxy(A,0)


    def d_dt(self, h, u, v):
        """
        http://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
        """
        for x in [h, u, v]: # type check
           assert isinstance(x, ndarray) and not isinstance(x, matrix)

        g,b,dx = self.g, self.b, self.dx

        du_dt = -g*self.d_dx(h) - b*u
        dv_dt = -g*self.d_dy(h) - b*v

        H = 0 #h.mean() - our definition of h includes this term
        dh_dt = -self.d_dx(u * (H+h)) - self.d_dy(v * (H+h))

        return dh_dt, du_dt, dv_dt


    def evolve(self):
        """
        Evolve state (h, u, v) forward in time using simple Euler method
        x_{N+1} = x_{N} +   dx/dt * d_t
        """

        dh_dt, du_dt, dv_dt = self.d_dt(self.h, self.u, self.v)
        dt = self.dt

        self.h += dh_dt * dt
        self.u += du_dt * dt
        self.v += dv_dt * dt
        self.time += dt

        return self.h, self.u, self.v




import random

iteration_times= 4100


frame_data = np.zeros((20000,64,64))
index = 0
parameter_data = []

#for simulation in range(50):
for simulation in range(200):
  print(simulation)
#  px=random.randint(20, 44)*1.
#  py=random.randint(20, 44)*1.
  px=random.randint(27, 37)*1.
  py=random.randint(27, 37)*1.
  R=random.randint(40, 80)*1.
  Hp=random.randint(5, 20)*0.01
  b=random.randint(1, 100)*0.1

  video_evolution = []

  # chose a point (x,y) to check the evolution

  #SW.plot()

  #for repition in range(20):
  for repition in range(5):
    SW = shallow(N=64,px=px,py=py,R=R,Hp=Hp,b=b)
    for i in range(iteration_times):
        SW.evolve()

        if i % 150 == 0 and i >= 1100:
            parameter_data.append([px,py,R,Hp,b,i])


            #frame_data.append(SW.u)
            frame_data[index,:,:] = SW.u
            index += 1
            #print('#########################################################################',b)
            #plt.imshow(SW.u)
            #plt.show()
            #plt.close()

"""# simulation all fields"""

import random

iteration_times= 4100


frame_data_u = np.zeros((20000,64,64))
frame_data_v = np.zeros((20000,64,64))
frame_data_h = np.zeros((20000,64,64))

index = 0
parameter_data = []

#for simulation in range(50):
for simulation in range(200):
  print(simulation)
#  px=random.randint(20, 44)*1.
#  py=random.randint(20, 44)*1.
  px=random.randint(27, 37)*1.
  py=random.randint(27, 37)*1.
  R=random.randint(40, 80)*1.
  Hp=random.randint(5, 20)*0.01
  b=random.randint(1, 100)*0.1

  video_evolution = []

  # chose a point (x,y) to check the evolution

  #SW.plot()

  #for repition in range(20):
  for repition in range(5):
    SW = shallow(N=64,px=px,py=py,R=R,Hp=Hp,b=b)
    for i in range(iteration_times):
        SW.evolve()

        if i % 150 == 0 and i >= 1100:
            parameter_data.append([px,py,R,Hp,b,i])


            #frame_data.append(SW.u)
            frame_data_u[index,:,:] = SW.u
            frame_data_v[index,:,:] = SW.v
            frame_data_h[index,:,:] = SW.h
            index += 1
            #print('#########################################################################',b)
            #plt.imshow(SW.u)
            #plt.show()
            #plt.close()

np.array(frame_data).shape

np.sum(np.isnan(frame_data))

np.array(parameter_data).shape

plt.imshow(frame_data[149,:,:])

from google.colab import drive
drive.mount('/content/drive')

#np.save('SW_contrastive/data/frame_data.npy',np.array(frame_data))

#np.save('SW_contrastive/data/parameter_data.npy',np.array(parameter_data))


np.save('SW_contrastive/data/frame_data_200.npy',np.array(frame_data))

np.save('SW_contrastive/data/parameter_data_200.npy',np.array(parameter_data))

np.array(frame_data).shape

# plot some examples
plt.imshow(frame_data[1710,:,:])

plt.imshow(frame_data[1,:,:])

plt.imshow(frame_data[1129,:,:])




