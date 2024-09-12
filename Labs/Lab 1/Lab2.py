import numpy as np

def compute_f(x):
    return (x**2)*(x-1)

def bisect(f, a, b, tol, nmax):
    n = 0
    while n < nmax and (b - a) > 2 * tol:
      
        xn = (a + b) / 2
        
        
        if f(xn) == 0 or (b - a) < tol:
            return xn
        
        
        if f(a) * f(xn) < 0:
            b = xn  
        else:
            a = xn
        
        n += 1
    return (a+b)/2


a = -1
b = 2
nmax = 100
tol = .01



root = bisect(compute_f, a, b, tol, nmax)
print(f"The root is approximately: {root}")
