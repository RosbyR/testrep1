import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0,np.pi, (np.pi)/30)
y = np.cos(t)



s = np.sum(t * y)
print(f"The sum is: {s}")
theta = np.linspace(0 , 2*np.pi, 1000)

R = 1.2
dr = 0.1
f = 15
p = 0

x = R*(1 + dr*np.sin(f*theta + p))*np.cos(theta)
y = R*(1 + dr*np.sin(f*theta + p))*np.sin(theta)
plt.plot(x,y,)
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal scaling
plt.grid(True)
plt.show()


