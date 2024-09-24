# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]


# projectedInexactNewtonCG.py

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA
from simpleValleyObjective import simpleValleyObjective
from projectionInBox import projectionInBox

def matrnr(): 
    # set your matriculation number here
    matrnr = 23353068
    return matrnr

def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xk = P.project(x0)

    # INCOMPLETE CODE STARTS
    gradx = f.gradient(xk)
    stationarity = np.linalg.norm(xk - P.project(xk - gradx))
    eta_k = min(0.5, np.sqrt(stationarity)) * stationarity

    while stationarity > eps:
        xj = xk.copy()
        rj = gradx
        dj = -rj

        while np.linalg.norm(rj) > eta_k:
            dA = PHA.projectedHessApprox(f, P, xk, dj)
            rhoj = dj.T @ dA

            if rhoj <= eps * np.linalg.norm(dj)**2:
                break

            tj = (np.linalg.norm(rj)**2) / rhoj
            xj = xj + tj * dj
            rold = rj
            rj = rold + tj * dA
            betaj = (np.linalg.norm(rj)**2) / (np.linalg.norm(rold)**2)
            dj = -rj + betaj * dj

        if np.linalg.norm(rj) <= eta_k:
            dk = xj - xk
        else:
            dk = -gradx

        tk = PB.projectedBacktrackingSearch(f, P, xk, dk, sigma=1.0e-4, verbose=verbose)
        xk = P.project(xk + tk * dk)
        gradx = f.gradient(xk)
        stationarity = np.linalg.norm(xk - P.project(xk - gradx))
        eta_k = min(0.5, np.sqrt(stationarity)) * stationarity

        countIter += 1

        # INCOMPLETE CODE ENDS

    if verbose:
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xk
'''
# Test case
p = np.array([[1], [1]])
myObjective = simpleValleyObjective(p)
a = np.array([[1], [1]])
b = np.array([[2], [2]])
myBox = projectionInBox(a, b)
x0 = np.array([[2], [2]], dtype=float)
eps = 1.0e-3
xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, verbose=1)
print("xmin =", xmin)

'''