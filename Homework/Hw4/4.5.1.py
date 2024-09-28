import numpy as np
import matplotlib.pyplot as plt
import scipy
def driver():
    # Test function 
    f = lambda x: x**6 - x - 1
    fp = lambda x: 6*x**5 - 1
    Nmax = 100
    tol = 1e-10
    
    # Test newton's method
    p0 = 2
    [p, pstar, info, it1] = newton(f, fp, p0, tol, Nmax)
    print('Newton method:')
    print('The approximate root is', '%16.16e' % pstar)
    print('The error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it1)
    error1 = abs(p[1:it1]-pstar)
    print(error1)
    error1_next = abs(p[1:it1+1] - pstar)
    # Test secant method

    print('\nSecant method:')
    x0 = 2
    x1 = 1
    [x, xstar, info, it2] = secant(f, x0, x1, tol, Nmax)
    print('The approximate root is', '%16.16e' % xstar)
    print('The error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it2)
    error2 = abs(x[1:it2]-xstar)
    error2_next = abs(x[1:it2+1] - xstar)
   
    print(error2)
    plt.loglog(error1, error1_next, label="Newton Method", marker='o')
    plt.loglog(error2, error2_next, label="Secant Method", marker='x')
    plt.xlabel('Iteration')
    plt.ylabel('log10(Error)')
    plt.title('xk - xk+1')
    plt.legend()
    plt.grid(True)
    plt.show()



def newton(f, fp, p0, tol, Nmax):
    """
    Newton iteration.
    """
    p = np.zeros(Nmax + 1)
    p[0] = p0
    
    for it1 in range(Nmax):
        p1 = p0 - f(p0) / fp(p0)
        p[it1 + 1] = p1
        if abs(p1 - p0) < tol:
            pstar = p1
            info = 0
            return [p, pstar, info, it1]
        p0 = p1
    
    pstar = p1
    info = 1
    return [p, pstar, info, it1]

def secant(f, x0, x1, tol, Nmax):
    """
    Secant iteration.
    """
    x = np.zeros(Nmax + 1)
    x[0] = x0
    x[1] = x1
    
    for it2 in range(1, Nmax):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x[it2 + 1] = x2
        if abs(x2 - x1) < tol:
            xstar = x2
            info = 0
            return [x, xstar, info, it2]
        x0, x1 = x1, x2
    
    xstar = x2
    info = 1
    return [x, xstar, info, it2]

driver()
