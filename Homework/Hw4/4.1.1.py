# import libraries
import numpy as np
from scipy import special
import matplotlib.pyplot as plt

def driver():
    ti = 20.0
    ts = -15.0
    a = 0.138 * 10**-6
    t = 5.184 * 10**6
    
    # Define the function f(x)
    f = lambda x: ts + (ti - ts) * special.erf(x / (2 * np.sqrt(a * t)))
    # Derivative for Newton's method
    fp = lambda x: (ti-ts)/(np.sqrt(np.pi*a*t)*np.exp(-x/(2*np.sqrt(a*t))))
    
    # Define the interval for bisection
    a_int = 0
    b_int = 2
    
    # Tolerance for root finding
    tol = 1e-13
    
    # Plot the function f(x) from x=0 to x=2
    plot_function(f, a_int, b_int)

    # Perform bisection method to find the root
    [astar, ier] = bisection(f, a_int, b_int, tol)
    print('Bisection: The approximate root is', astar)
    print('Bisection: The error message reads:', ier)
    print('Bisection: f(astar) =', f(astar))

    # Initial guess for Newton's method
    p0 = 1.2
    Nmax = 100
    tol_newton = 1.e-14

    # Perform Newton's method to find the root
    (p, pstar, info, it) = newton(f, fp, p0, tol_newton, Nmax)
    print('Newton: The approximate root is', '%16.16e' % pstar)
    print('Newton: The error message reads:', '%d' % info)
    print('Newton: Number of iterations:', '%d' % it)

# Function to plot f(x)
def plot_function(f, a, b):
    x_vals = np.linspace(a, b, 100)
    f_vals = [f(x) for x in x_vals]

    plt.plot(x_vals, f_vals, label='f(x)')
    plt.axhline(0, color='black',linewidth=0.5)  # Line at y=0
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Function Plot')
    plt.grid(True)
    plt.legend()
    plt.show()

# Bisection method definition
def bisection(f, a, b, tol):
    fa = f(a)
    fb = f(b)
    
    if fa * fb > 0:
        print(f"No root found in the interval [{a}, {b}]")
        ier = 1
        astar = a
        return [astar, ier]

    count = 0
    d = 0.5 * (a + b)
    
    while abs(b - a) > tol:
        d = 0.5 * (a + b)
        fd = f(d)
        
        if fd == 0:
            astar = d
            ier = 0
            return [astar, ier]
        
        if fa * fd < 0:
            b = d
        else:
            a = d
            fa = fd
            
        count += 1
  
