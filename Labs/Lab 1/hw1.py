import numpy as np
import matplotlib.pyplot as plt

X = np.arange(1.920, 2.080, .001)
print(X)
Y = (X-2)**9
Z = X**9 - 18*X**8 + 144*X**7 - 672*X**6 + 2016*X**5 - 4032*X**4 + 5376*X**3 - 4608*X**2 + 2304*X -512
plt.plot(X,Y, label='(X-2)^9')
plt.plot(X,Z, label='Coefficients')
plt.xlabel('X')
plt.ylabel('P(X)')
plt.title('Coefficients vs expression')
plt.legend()
plt.show()
print('hello')