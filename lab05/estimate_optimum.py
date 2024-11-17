import numpy as np
from scipy.optimize import differential_evolution

# Implementation of first objective function
def objective_function(x):
    sum_squares = np.sum(np.square(x))
    expr1 = 5 / (1 + sum_squares)
    cotangent = 1 / np.tan(np.exp(-expr1))
    return -expr1 + np.sin(cotangent)

n = 10
bounds = [(-3, 3) for _ in range(n)]

# Estimate optimum using differential evolution, uncomment to run

#result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=100000, tol=1e-8)
#print("Optimum parameters estimation x:", result.x)
#print("Optimum value estimation f(x):", result.fun)


# Check optimum
x = np.zeros(10)
print(f"value for x=(0,0,0...,0): {objective_function(x)}")

x = np.array([-0.02678404, -0.02164306, -0.01061552,  0.04325569, 
              0.03961267,  0.00105597, -0.00928009,  0.04510646, 
              0.02464627,  0.00933702])

print(f"value for our parameters: {objective_function(x)}")