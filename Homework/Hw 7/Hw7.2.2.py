import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def driver():
    f = lambda x: 1 / (1 + (10 * x) ** 2)  # Example function

    N = 10  # Number of interpolation nodes
    ''' interval '''
    a = -1
    b = 1

    ''' create Chebyshev interpolation nodes '''
    xint = np.array([np.cos((2 * j - 1) * np.pi / (2 * (N + 1))) for j in range(1, N + 2)])  # N+1 points

    ''' create interpolation data '''
    yint = f(xint)

    ''' create points for evaluating the Lagrange interpolating polynomial '''
    Neval = 1000
    xeval = np.linspace(a, b, Neval + 1)
    yeval_l = np.zeros(Neval + 1)  # Lagrange polynomial values (barycentric)

    ''' Compute barycentric weights '''
    w = barycentric_weights(xint, N + 1)  # Correct the size of weights to N+1

    ''' Evaluate Lagrange polynomial using barycentric formula '''
    for kk in range(Neval + 1):
        yeval_l[kk] = eval_lagrange_barycentric(xeval[kk], xint, yint, w, N + 1)  # Pass N+1

    ''' create vector with exact values '''
    fex = f(xeval)

    ''' Plotting '''
    plt.figure()
    plt.plot(xeval, fex, 'r-', label="Exact function f(x)")
    plt.plot(xeval, yeval_l, 'b--', label="Barycentric Lagrange interpolation")
    plt.plot(xint, yint, 'go', label="Interpolation points")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Lagrange Interpolation with Barycentric Formula')
    plt.grid(True)
    plt.show()


def barycentric_weights(xint, N):
    ''' Compute barycentric weights for the interpolation nodes '''
    w = np.ones(N)
    for j in range(N):
        for k in range(N):
            if j != k:
                w[j] /= (xint[j] - xint[k])
    return w


def eval_lagrange_barycentric(xeval, xint, yint, w, N):
    ''' Evaluate the Lagrange interpolating polynomial using barycentric formula '''
    numerator = 0.0
    denominator = 0.0
    for j in range(N):
        if xeval == xint[j]:  # Avoid division by zero if xeval equals any node
            return yint[j]
        term = w[j] / (xeval - xint[j])
        numerator += term * yint[j]
        denominator += term
    return numerator / denominator


driver()
