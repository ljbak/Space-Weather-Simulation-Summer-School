# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:51:49 2022

@author: ljbak
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt


def function_cos(x):    
    """
    computes the function f(x)=cos(x) + x*sin(x) of an 
    input variable x

    Parameters
    ----------
    x : input variable to compute the function (vector)

    Returns
    -------
    
    f(x) : outputs function f(x)=cos(x) + x*sin(x)

    """
    return np.cos(x) + x*np.sin(x)


def derivative_of_function_cos(x):
    """
    
    f'(x) : outputs f'(x)=x*cos(x)

    -------
    Parameters
    ----------
    x : input variable to compute the function (vector)

    Returns
    -------
    
    df/fx : derivative of function f(x)=cos(x) + x*sin(x)
    
    """
    return x*np.cos(x)


def finite_difference_of_function_fwd(x, stepsize):
    """
    calculates forward finite difference of function f(x)=cos(x) + x*sin(x) 
    given an x at which to compute the finite difference and a step size

    Parameters
    ----------
    x : x values at which to compute finite difference
    stepsize : size of step over which to compute finite difference

    Returns
    -------
    dfdx_finite : finite forward difference of f(x) evaluated at x

    """
    # define empty array for derivative
    derivative = np.array([])

    # iterate over x
    counter = 0 # count the elements of x
    while counter < len(x)-1:  
        # forward finite difference
        dfdx_finite = (function_cos(x[counter+1]) - function_cos(x[counter]))/stepsize
        # append finite difference to the output array
        derivative = np.append(derivative, dfdx_finite)
        # increment counter
        counter = counter + 1
    
    return derivative


def finite_difference_of_function_bkwd(x, stepsize):
    """
    calculates backward finite difference of function f(x)=cos(x) + x*sin(x) 
    given an x at which to compute the finite difference and a step size

    Parameters
    ----------
    x : x values at which to compute finite difference
    stepsize : size of step over which to compute finite difference

    Returns
    -------
    dfdx_finite : finite backward difference of f(x) evaluated at x

    """
    # define empty array for derivative
    derivative = np.array([])

    # iterate over x
    counter = 1 # count the elements of x
    while counter < len(x):  
        # backward finite difference
        dfdx_finite = (function_cos(x[counter]) - function_cos(x[counter-1]))/stepsize
        # append finite difference to the output array
        derivative = np.append(derivative, dfdx_finite)
        # increment counter
        counter = counter + 1
    
    return derivative


def finite_difference_of_function_cen(x, stepsize):
    """
    calculates central finite difference of function f(x)=cos(x) + x*sin(x) 
    given x at which to compute the finite difference and a step size

    Parameters
    ----------
    x : x values at which to compute finite difference
    stepsize : size of step over which to compute finite difference

    Returns
    -------
    dfdx_finite : finite central difference of f(x) evaluated at x

    """
    # define empty array for derivative
    derivative = np.array([])

    # iterate over x
    counter = 1 # count the elements of x
    while counter < len(x)-1:  
        # central finite difference
        dfdx_finite = (function_cos(x[counter+1]) - function_cos(x[counter-1]))/(2*stepsize)
        # append finite difference to the output array
        derivative = np.append(derivative, dfdx_finite)
        # increment counter
        counter = counter + 1
    
    return derivative
    

# compute the function over these elements
x = np.linspace(-6,6,num=100)  

# calculate the function and its derivative
f = function_cos(x)
dfdx = derivative_of_function_cos(x)  

# step size for computing finite difference
stepsize = 0.25

# compute finite differences using step size stepsize
x_finite = np.arange(x[0],x[-1],step=stepsize)
dfdx_fwd = finite_difference_of_function_fwd(x_finite, stepsize)
dfdx_bkwd = finite_difference_of_function_bkwd(x_finite, stepsize)
dfdx_cen = finite_difference_of_function_cen(x_finite, stepsize)
    
# plot the function, its derivative, and finite differences
plt.plot(x, f, 'b-',label='f(x)')
plt.plot(x, dfdx, 'r-',label='f\'(x)')
plt.xlabel('x')
plt.ylabel('f(x), f''(x)')
plt.plot(x_finite[:-1],dfdx_fwd,'r>',label='fwd diff')
plt.plot(x_finite[1:],dfdx_bkwd,'r<',label='bkwd diff')
plt.plot(x_finite[1:-1],dfdx_cen,'r.',label='central diff')
plt.legend()