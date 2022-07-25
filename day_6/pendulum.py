# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:07:03 2022

@author: ljbak
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def pendulum(x, time):
    "pendulum ODE"
    x_dot = np.zeros(2)
    
    x_dot[0] = x[1]
    x_dot[1] = -g/l*np.sin(x[0])
    
    return x_dot

def RK4(func, x0, t):
    """Explicit integrator Runge-Kutta Order 4"""
    n = len(t)
    x = np.zeros((n, len(x0)))
    x[0] = x0
    x_current = x[0]
    t_current = t[0]
    for i in range(n):
        # update time and solution
        t_old = t_current 
 
        x_old = x_current
        x_current = pendulum(x_old,t[i])
        
        # RK4 constants
        k1 = pendulum(x_old, t_old)
        k2 = pendulum(x_old + k1*stepsize/2, t_old + stepsize/2)
        k3 = pendulum(x_old + k2*stepsize/2, t_old + stepsize/2)
        k4 = pendulum(x_old + k3*stepsize, t_old + stepsize)
        
        # calculate current time and solution 
        t_current = t[i+1]
        x_current = x_old + stepsize*(k1 + 2*k2 + 2*k3 + k4)/6
        
        # store solution 
        x[i,] = x_current
        return x


# pendulum properties
g = 9.81 # gravity
l = 3    # length

# initial conditions
x0 = np.zeros(2)
x0[0] = np.pi/3.0
x0[1] = 0
t0 = 0
tf = 10
stepsize = 0.1

"Evaluate the exact solution"
time = np.linspace(t0,tf,num=100)  # time spanned
x_true = odeint(pendulum, x0, time)  # solution

time_RK4 = np.arange(t0, tf, step=stepsize)
x_RK4 = RK4(pendulum, x0, time_RK4)  # RK4 solution

fig = plt.figure()
plt.plot(time, x_true)
plt.plot(time_RK4, x_RK4)
#subfigs = fig.subfigures(1, 2, wspace=0.07)