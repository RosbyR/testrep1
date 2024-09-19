import numpy as np

def driver():
    # Test function 
    f1 = lambda x: (10/(x+4))**(1/2) # Fixed point is alpha1 = 1.4987....
    
    Nmax = 100
    tol = 1e-10

    # Test f1
    x0 = 1.5
    print("Testing f1")
    [xstar, ier, x] = fixedpt(f1, x0, tol, Nmax)
    print('The approximate fixed point is:', xstar)
    print('f1(xstar):', f1(xstar))
    print('Error message reads:', ier)

    if ier == 0:  # Only compute order if converged
        compute_order(x, xstar)

# Define fixed point routine
def fixedpt(f, x0, tol, Nmax):
    x = np.zeros((Nmax, 1))  # Store the results of each iteration
    x[0] = x0

    count = 0
    while count < Nmax - 1:  # Nmax - 1 to prevent index overflow
        count += 1
        x1 = f(x[count-1])
        x[count] = x1
        if abs(x1 - x[count-1]) < tol:
            xstar = x1
            ier = 0
            break
    else:
        xstar = x1
        ier = 1

    # Print results in column format
    print("Iteration Results:")
    for i in range(count + 1):
        print(f"Iteration {i}: {x[i, 0]}")

    return [xstar, ier, x[:count + 1]]  # Return x values up to the count

def compute_order(x, xstar):
    # Avoid zero values in difference
    diff1 = np.abs(x[1:] - xstar)
    diff2 = np.abs(x[:-1] - xstar)

    # Filter out any zero differences
    mask = (diff1 > 0) & (diff2 > 0)
    log_diff1 = np.log(diff1[mask])
    log_diff2 = np.log(diff2[mask])

    # Ensure there are enough points to fit a line
    if len(log_diff1) < 2 or len(log_diff2) < 2:
        print("Not enough data to compute order.")
        return

    fit = np.polyfit(log_diff2, log_diff1, 1)

    print('The order is:')
    print('lambda = ' + str(np.exp(fit[1])))
    print('alpha = ' + str(fit[0]))

    return [fit, diff1, diff2]



def aitken(x, xstar):
    top = np.abs(x[1::]-x[0:-1])
    bottom = np.abs(x[2::]-2*x[1::]+x[0:-1])
    px = top/bottom
    phat = x-px
    print(phat)
    return [top, bottom, px, phat]
driver()
