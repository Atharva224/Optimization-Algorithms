# Optimization for Engineers - Dr.Johannes Hild
# projected backtracking line search

# Purpose: Find t to satisfy f(x+t*d)< f(x) - sigma/t*norm(x-P(x - t*gradient))**2

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# P: box projection class with method .project()
# x: column vector in R**n (domain point)
# d: column vector in R**n (search direction)
# sigma: value in (0,1), marks quality of decrease. Default value: 1.0e-4.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, 1)
# should return t = 0.5

import numpy as np
from simpleValleyObjective import simpleValleyObjective
from projectionInBox import projectionInBox

def matrnr():
    # set your matriculation number here
    matrnr = 23353068
    return matrnr

def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, verbose=0):
    xp = P.project(x)
    gradx = f.gradient(xp)
    decrease = gradx.T @ d

    if decrease >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 1:
        raise TypeError('range of sigma is wrong!')

    if verbose:
        print('Start projectedBacktrackingSearch...')

    beta = 0.5
    t = 1
    # INCOMPLETE CODE STARTS# 
    
    def W1(t):
        new_x = x + t * d
        projected_new_x = P.project(new_x)
        lhs = f.objective(projected_new_x)
        rhs = f.objective(x) - sigma * t * np.linalg.norm(x - P.project(x - t * gradx))**2
        return lhs <= rhs

   
    while not W1(t):
        t *= beta


    return t

    # INCOMPLETE CODE ENDS
'''
# Test case
p = np.array([[0], [1]])
myObjective = simpleValleyObjective(p)
a = np.array([[-2], [1]])
b = np.array([[2], [2]])
eps = 1.0e-6
myBox = projectionInBox(a, b, eps)
x = np.array([[1], [1]])
d = np.array([[-1.99], [0]])
sigma = 0.5
t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, verbose=1)
print('Returned t:', t)  # Should return t = 0.5
'''