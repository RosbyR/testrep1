# import libraries
import numpy as np
import math

        
def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2

  f = lambda x: math.exp(3*x) - 27*x**6 + 27*x**4*math.exp(x) - 9*x**2*math.exp(2*x) # Fixed point is alpha1 = 1.4987....
  fp = lambda x: 3*(math.exp(x) - 6 * x)*(math.exp(x) - 3*x**2)**2
  p0 = 3.5
  m = 3

  h = lambda x: (math.exp(x) - 3*x**2)/(3*math.exp(x) - 3*x**2)**2
  hp = lambda x: (6*x**2 + math.exp(x)*(2 - 4*x + x**2))/(math.exp(x) - 6*x)**2
  Nmax = 100
  tol = 1.e-14
  print ('newton1')
  [p,pstar,info,it] = newton(f,fp,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  erorr = abs(p-pstar)
  sol = erorr[1:it]/erorr[0:it-1]
  print (sol)

  print('part b')
  [p,pstar,info,it] = newton(h, hp, p0, tol , Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  erorr = abs(p-pstar)
  sol = erorr[1:it]/erorr[0:it-1]
  print (sol)
  
  print('part b')
  [p,pstar,info,it] = newton1(m ,f, fp, p0, tol , Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  erorr = abs(p-pstar)
  sol = erorr[1:it]/erorr[0:it-1]
  print (sol)




def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]

def newton1(m , f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-m*(f(p0)/fp(p0))
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]    
        
driver()
