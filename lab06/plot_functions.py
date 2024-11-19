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


fig = plt.figure(figsize=(15, 10)) 

# First row: 3D Surface Plots

for idx, (func, domain, title) in enumerate(zip(functions, DOMAINS, titles), 1):
    X, Y = np.meshgrid(
        np.linspace(domain[0], domain[1], 100),
        np.linspace(domain[0], domain[1], 100))
    Z = np.zeros(X.shape)
    
    for i in range(100):
        for j in range(100):
            Z[i, j], _ = func(np.array([X[i, j], Y[i, j]]))
    
    # Apply a logarithmic transformation to highlight small changes
    Z_log = np.log1p(np.abs(Z))
    
    # Plot the 3D log surface
    ax = fig.add_subplot(2, 3, idx, projection='3d')
    surface = ax.plot_surface(X, Y, Z_log, cmap='viridis', edgecolor='none')
    ax.set_title(f"{title} (Log Transformed)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Log Transformed Z")
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

# Second row: 2D Heatmaps for the same functions
for idx, (func, domain, title) in enumerate(zip(functions, DOMAINS, titles), 1):
    X, Y = np.meshgrid(
        np.linspace(domain[0], domain[1], 100),
        np.linspace(domain[0], domain[1], 100))
    Z = np.zeros(X.shape)
    
    for i in range(100):
        for j in range(100):
            Z[i, j], _ = func(np.array([X[i, j], Y[i, j]]))
    
    # Apply a logarithmic transformation to highlight small changes
    Z_log = np.log1p(np.abs(Z))
    
    # Plot the 2D heatmap
    ax = fig.add_subplot(2, 3, idx + 3)
    c = ax.pcolormesh(X, Y, Z_log, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_title(f"{title} (Heatmap)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.tight_layout()
plt.show()


# Reduced ranges for close optimum points analysis
REDUCED_DOMAINS = np.array([[0, 2], [-2, 2], [0, 2]])

# Plot the log surfaces for reduced ranges

fig = plt.figure(figsize=(15, 10)) 

# First row: 3D Surface Plots with reduced ranges
for idx, (func, domain, title) in enumerate(zip(functions, REDUCED_DOMAINS, titles), 1):
    X, Y = np.meshgrid(
        np.linspace(domain[0], domain[1], 100),
        np.linspace(domain[0], domain[1], 100))
    Z = np.zeros(X.shape)
    
    for i in range(100):
        for j in range(100):
            Z[i, j], _ = func(np.array([X[i, j], Y[i, j]]))
    
    Z_log = np.log1p(np.abs(Z))
    
    # Plot the 3D transformed surface
    ax = fig.add_subplot(2, 3, idx, projection='3d')
    surface = ax.plot_surface(X, Y, Z_log, cmap='viridis', edgecolor='none')
    ax.set_title(f"{title} (Log Transformed) - Reduced Range")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Log Transformed Z")
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

# Second row: 2D Heatmaps for the same functions with reduced ranges
for idx, (func, domain, title) in enumerate(zip(functions, REDUCED_DOMAINS, titles), 1):
    X, Y = np.meshgrid(
        np.linspace(domain[0], domain[1], 100),
        np.linspace(domain[0], domain[1], 100))
    Z = np.zeros(X.shape)
    
    # Evaluate the function
    for i in range(100):
        for j in range(100):
            Z[i, j], _ = func(np.array([X[i, j], Y[i, j]]))
    
    Z_log = np.log1p(np.abs(Z))
    
    # Plot the 2D heatmap
    ax = fig.add_subplot(2, 3, idx + 3)
    c = ax.pcolormesh(X, Y, Z_log, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax)
    ax.set_title(f"{title} (Heatmap) - Reduced Range")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.tight_layout()
plt.show()