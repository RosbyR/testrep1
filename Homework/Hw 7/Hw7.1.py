import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm

def driver(N):
    f = lambda x: 1/(1 + (10 * x) ** 2)  # The given function f(x)
    
    a = -1  # Left bound of the interval
    b = 1   # Right bound of the interval
    ''' Create interpolation nodes'''
    xint = np.linspace(a, b, N + 1)
    
    ''' Create interpolation data'''
    yint = f(xint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint, N)
    
    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
    
    ''' Apply inverse to create the coefficients'''
    coef = Vinv @ yint
    
    ''' Create evaluation grid '''
    Neval = 1000  # Finer grid for plotting
    xeval = np.linspace(a, b, Neval + 1)
    yeval = eval_monomial(xeval, coef, N, Neval)
    
    ''' Exact function for comparison'''
    yexact = f(xeval)
    
    ''' Compute error (for diagnostic purposes) '''
    err = norm(yexact - yeval)
    print(f'N = {N}, Error = {err:.6f}')
    
    ''' Plotting '''
    plt.figure(figsize=(10, 6))
    
    # Plot the original data points
    plt.plot(xint, yint, 'ro', label='Data points')
    
    # Plot the interpolating polynomial
    plt.plot(xeval, yeval, 'b-', label=f'Interpolating polynomial (N = {N})')
    
    # Plot the exact function
    plt.plot(xeval, yexact, 'g--', label='Exact function f(x)')
    
    # Add title and labels
    plt.title(f'Interpolation for N = {N}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    plt.show()

def eval_monomial(xeval, coef, N, Neval):
    yeval = coef[0] * np.ones(Neval + 1)
    
    for j in range(1, N + 1):
        yeval += coef[j] * xeval ** j

    return yeval

def Vandermonde(xint, N):
    V = np.zeros((N + 1, N + 1))
    
    ''' Fill the Vandermonde matrix '''
    for j in range(N + 1):
        V[j, 0] = 1.0
    
    for i in range(1, N + 1):
        V[:, i] = xint ** i

    return V     

# Run the driver function with increasing values of N to observe Runge's phenomenon
for N in range(4, 5):
    driver(N)
