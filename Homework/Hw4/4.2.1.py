import numpy as np

def driver():
    # Test function 
    f = lambda x: np.exp(3*x) + 27*x**4*np.exp(x) - 9*x**2*np.exp(2*x)  # Fixed point is alpha1 = 1.4987....
    fp = lambda x: 3*np.exp(3*x) + 27 * 4 * x**3 * np.exp(x) - 9 * 2 * x * np.exp(2*x) - 18 * x**2 * np.exp(2*x)  # Corrected derivative
    Nmax = 100
    tol = 1e-10
    
    # Test f1
    p0 = 3.5
    (p, pstar, info, it) = newton(f, fp, p0, tol, Nmax)
    print('The approximate root is', '%16.16e' % pstar)
    print('The error message reads:', '%d' % info)
    print('Number of iterations:', '%d' % it)

    if info == 0:  # Only compute order if converged
        compute_order(p, pstar)

def newton(f, fp, p0, tol, Nmax):
    """
    Newton iteration.
    
    Inputs:
      f, fp - function and derivative
      p0    - initial guess for root
      tol   - iteration stops when p_n, p_{n+1} are within tol
      Nmax  - max number of iterations
    Returns:
      p      - an array of the iterates
      pstar   - the last iterate
      info    - success message
              - 0 if we met tol
              - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax + 1)
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

def compute_order(x, xstar):
    # Compute the order of convergence
    diff1 = np.abs(x[1:] - xstar)
    diff2 = np.abs(x[:-1] - xstar)

    # Ensure there are enough points to fit a line
    if len(diff1) < 2 or len(diff2) < 2:
        print("Not enough data to compute order.")
        return

    log_diff1 = np.log(diff1[1:])  # Use diff1 from index 1
    log_diff2 = np.log(diff2[:-1])  # Use diff2 up to the second last index

    fit = np.polyfit(log_diff2, log_diff1, 1)

    print('The order is:')
    print('lambda = ' + str(np.exp(fit[1])))
    print('alpha = ' + str(fit[0]))

    return [fit, diff1, diff2]

# Uncomment the aitken function if needed; otherwise, it can be removed.
# def aitken(x, xstar):
#     top = np.abs(x[1:] - x[:-1])
#     bottom = np.abs(x[2:] - 2 * x[1:] + x[:-1])
#     px = top / bottom
#     phat = x - px
#     print(phat)
#     return [top, bottom, px, phat]

driver()
