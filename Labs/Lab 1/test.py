import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 5)
Y = np.arange(1,10,2)
print(X)
print('the first three entries are',Y[0:3])
print(Y)
w = 10**(-np.linspace(1,10,10))

print('W', w)
z = len(w)
x = np.linspace(0,10,z)
print('vector x',x)
plt.plot(x,w)
plt.xlabel('x')
plt.ylabel('w')
plt.show()

s = 3*w
ptint('s',s)

