# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:10:44 2022

@author: ljbak
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def RHS(f, t):
    """ODE right hand side"""
    return -2*f

"Set the problem"
f0 = 3 # initial condition
t0 = 0 # initial time
tf = 2 # final time

"Evaluate the exact solution"
time = np.linspace(t0,tf,num=100)  # time spanned
f_true = odeint(RHS, f0, time)  # solution


"Numerical integration"
stepsize = .2 # step size

"First order Runge-Kutta or Euler Method"
solution1 = np.array([])  # empty array to contain solution
timeline1 = np.array([solution1])
f_current = f0  # inital solution
t_current = t0  # initial time
timeline1 = np.append(timeline1, t_current)
solution1 = np.append(solution1, f_current)
while t_current < tf-stepsize: 
    # update time and solution
    t_old = t_current 
    f_old = f_current 
    
    # calculate current time and solution 
    t_current = t_old + stepsize
    f_current = f_old + stepsize*RHS(f_old, t_current)
    
    # store time and solution 
    timeline1 = np.append(timeline1, t_current)
    solution1 = np.append(solution1, f_current)
    
    
"Second order Runge-Kutta"
solution2 = np.array([])  # empty array to contain solution
timeline2 = np.array([solution2])
f_current = f0  # inital solution
t_current = t0  # initial time
timeline2 = np.append(timeline2, t_current)
solution2 = np.append(solution2, f_current)
while t_current < tf-stepsize: 
    # update time and solution
    t_old = t_current 
    f_old = f_current 
    
    # half-step time and solution values
    t_half = t_old + stepsize/2
    f_half = f_old + stepsize/2*RHS(f_old, t_old)
    
    # calculate current time and solution 
    t_current = t_old + stepsize
    f_current = f_old + stepsize*RHS(f_half, t_half)
    
    # store time and solution 
    timeline2 = np.append(timeline2, t_current)
    solution2 = np.append(solution2, f_current)
    
    
"Fourth order Runge-Kutta"
solution4 = np.array([])  # empty array to contain solution
timeline4 = np.array([solution4])
f_current = f0  # inital solution
t_current = t0  # initial time
timeline4 = np.append(timeline4, t_current)
solution4 = np.append(solution4, f_current)
while t_current < tf-stepsize: 
    # update time and solution
    t_old = t_current 
    f_old = f_current 
    
    # RK4 constants
    k1 = RHS(f_old, t_old)
    k2 = RHS(f_old + k1*stepsize/2, t_old + stepsize/2)
    k3 = RHS(f_old + k2*stepsize/2, t_old + stepsize/2)
    k4 = RHS(f_old + k3*stepsize, t_old + stepsize)
    
    # calculate current time and solution 
    t_current = t_old + stepsize
    f_current = f_old + stepsize*(k1 + 2*k2 + 2*k3 + k4)/6
    
    # store time and solution 
    timeline4 = np.append(timeline4, t_current)
    solution4 = np.append(solution4, f_current)


"plot solution"
fig1 = plt.figure()
plt.plot(time, f_true, 'k-', label='truth')
plt.grid()
plt.xlabel('time')
plt.ylabel('$f(t)$')
plt.plot(timeline1, solution1, 'b.-', label='RK1')
plt.plot(timeline2, solution2, 'r.-', label='RK2')
plt.plot(timeline4, solution4, 'g.-', label='RK4')
plt.legend()