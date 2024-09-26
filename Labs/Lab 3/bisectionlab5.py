# import libraries
import numpy as np

def driver():

    # use routines    
    f = lambda x: np.exp(x**2+7*x-30)-1
    fp = lambda x: (2*x+7)*np.exp(x**2+7*x-30)
    fpp = lambda x: ((2*x+7)**2*np.exp(x**2+7*x-30))+(2*np.exp(x**2+7*x-30))
    g = lambda x: (f(x)*fpp(x))/(fp(x))**2  # fixed the power operator
    a = 2
    b = 4.5

    # tolerance
    tol = 1e-3

    # Call the bisection method
    [astar, ier] = bisection(f, a, b, tol, fp, fpp, g)
    print('The approximate root is', astar)
    print('The error message reads:', ier)
    print('f(astar) =', f(astar))

# define routines
def bisection(f, a, b, tol, fp, fpp, g,):
    
    # Inputs:
    # f, a, b - function and endpoints of the initial interval
    # tol - bisection stops when interval length < tol

    # Returns:
    # astar - approximation of root
    # ier - error message (0 for success, 1 for failure)

    # First, verify there is a root we can find in the interval 
    fa = f(a)
    fb = f(b)

    if (fa * fb > 0):
        ier = 1
        astar = a
        return [astar, ier]

    # Verify end points are not a root 
    if (fa == 0):
        astar = a
        ier = 0
        return [astar, ier]

    if (fb == 0):
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    while (g(0.5 * (a + b)) < 1):  # loop until interval size < tolerance
        d = 0.5 * (a + b)
        fd = f(d)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier]

        if (fa * fd < 0):
            b = d
        else: 
            a = d
            fa = fd
        count += 1

        # Print or store intermediate values if needed for debugging
        # print(f'Iteration {count}: Interval = [{a}, {b}], Midpoint = {d}')

    astar = 0.5 * (a + b)
    ier = 0
    return [astar, ier]

def newton(f, fp, p0, tol, Nmax,astar):
    """
    Newton iteration.
  
    Inputs:
        f, fp - function and derivative
        p0   - initial guess for root
        tol  - iteration stops when p_n, p_{n+1} are within tol
        Nmax - max number of iterations
    Returns:
        p     - an array of the iterates
        pstar - the last iterate
        info  - success message (0 for success, 1 if hit Nmax)
    """
    p = astar
    p[0] = p0
    for it in range(Nmax):
        p1 = p0 - f(p0) / fp(p0)
        p[it + 1] = p1
        if abs(p1 - p0) < tol:
            pstar = p1
            info = 0
            return [p, pstar, info, it]
        p0 = p1
    pstar = p1
    info = 1
    return [p, pstar, info, it]

# Run the driver function
driver()
