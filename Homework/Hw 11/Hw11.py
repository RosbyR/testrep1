import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm

def driver():
    f = lambda x: 1/(1 + x**2)
    a = -5
    b = 5
    
    # Exact integral
    I_ex = 2 * np.arctan(5)
    
    # Number of intervals (n must be even for Simpson's)
    n = 510
    
    # Trapezoidal rule calculation
    I_trap = CompTrap(a, b, n, f)
    print('I_trap =', I_trap)
    err = abs(I_ex - I_trap)
    print('Trapezoidal absolute error =', err)    
    
    # Simpson's rule calculation
    I_simp = CompSimp(a, b, n, f)
    print('I_simp =', I_simp)
    err = abs(I_ex - I_simp)
    print('Simpson absolute error =', err)    

def CompTrap(a, b, n, f):
    h = (b - a) / n
    xnode = a + np.arange(0, n + 1) * h
    
    I_trap = h * (f(xnode[0]) / 2)
    for j in range(1, n):
        I_trap += h * f(xnode[j])
    I_trap += h * (f(xnode[n]) / 2)
    
    return I_trap     

def CompSimp(a, b, n, f):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")
    
    h = (b - a) / n
    xnode = a + np.arange(0, n + 1) * h
    
    I_simp = f(xnode[0]) + f(xnode[n])
    for j in range(1, n, 2):  # odd terms
        I_simp += 4 * f(xnode[j])
    for j in range(2, n - 1, 2):  # even terms
        I_simp += 2 * f(xnode[j])
    I_simp = (h / 3) * I_simp
    
    return I_simp 

driver()
