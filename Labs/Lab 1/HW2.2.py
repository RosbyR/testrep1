import numpy as np
import matplotlib.pyplot as plt
import random





theta = np.linspace(0 , 2*np.pi, 1000)
for i in range (1,10):

    R = i 
    dr = 0.05
    f = 2 + i
    p = random.uniform(0 , 2)

    x = R*(1 + dr*np.sin(f*theta + p))*np.cos(theta)
    y = R*(1 + dr*np.sin(f*theta + p))*np.sin(theta)
    plt.plot(x,y,)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal scaling
    plt.grid(True)
plt.show()