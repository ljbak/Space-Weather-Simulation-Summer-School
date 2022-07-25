# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:03:08 2022

@author: ljbak
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lorenz(x, t, sigma, rho, beta):
    """ODE right hand side"""
    x_dot = np.zeros(3)
    x_dot[0] = sigma*(x[1] - x[0])
    x_dot[1] = x[0]*(rho - x[2]) - x[1]
    x_dot[2] = x[0]*x[1] - beta*x[2]
    
    return x_dot

def RK4(func, x0, t0, tf, stepsize):
    "Fourth order Runge-Kutta"
    solution4 = np.array([])  # empty array to contain solution
    timeline4 = np.array([solution4])
    x_current = x0  # inital solution
    t_current = t0  # initial time
    timeline4 = np.append(timeline4, t_current)
    solution4 = np.append(solution4, x_current)
    while t_current < tf-stepsize: 
        # update time and solution
        t_old = t_current 
        x_old = x_current 
        
        # RK4 constants
        k1 = func(x_old, t_old, sigma, rho, beta)
        k2 = func(x_old + k1*stepsize/2, t_old + stepsize/2, sigma, rho, beta)
        k3 = func(x_old + k2*stepsize/2, t_old + stepsize/2, sigma, rho, beta)
        k4 = func(x_old + k3*stepsize, t_old + stepsize, sigma, rho, beta)
        
        # calculate current time and solution 
        t_current = t_old + stepsize
        x_current = x_old + stepsize*(k1 + 2*k2 + 2*k3 + k4)/6
        
        # store time and solution 
        timeline4 = np.append(timeline4, t_current)
        solution4 = np.append(solution4, x_current)
    
    solution4 = np.reshape(solution4,(np.size(timeline4),3))
    return solution4

"Set the problem"
x0 = np.transpose([5, 5, 5]) # initial condition
t0 = 0 # initial time
tf = 20 # final time

sigma = 10
rho = 28
beta = 8/3

"Evaluate the exact solution"
time = np.linspace(t0,tf,num=1000)  # time spanned

solution4 = odeint(lorenz, x0, time, args=(sigma, rho, beta))

"Numerical integration"
stepsize = .01 # step size

solution4 = RK4(lorenz, x0, t0, tf, stepsize)

"plot solution"
fig1 = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(solution4[:,0], solution4[:,1], solution4[:,2], 'gray')

"Random initial conditions"
number_of_ICs = 20

x0 = np.random.rand(number_of_ICs,3)
x0[:,0] = x0[:,0]*40-20
x0[:,1] = x0[:,1]*60-30
x0[:,2] = x0[:,2]*50

"plot new solutions"
for i in range(number_of_ICs):
    solution4 = RK4(lorenz, x0[i,:], t0, tf, stepsize)
    ax.plot3D(solution4[:,0], solution4[:,1], solution4[:,2])