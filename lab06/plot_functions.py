import numpy as np
import matplotlib.pyplot as plt
from lab06 import rosenbrock, salomon, whitley, DOMAINS

fig = plt.figure(figsize=(15, 5))
functions = [rosenbrock, salomon, whitley]
titles = ['Generalized Rosenbrock', 'Salomon', 'Whitley']

for idx, (func, domain, title) in enumerate(zip(functions, DOMAINS, titles), 1):
    X, Y = np.meshgrid(
        np.linspace(domain[0], domain[1], 100),
        np.linspace(domain[0], domain[1], 100))
    Z = np.zeros(X.shape)
    
    for i in range(100):
        for j in range(100):
            Z[i, j], _ = func(np.array([X[i, j], Y[i, j]]))
    
    ax = fig.add_subplot(1, 3, idx, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()