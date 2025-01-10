# NSGA-II implementation

import numpy as np
import random
import copy
import matplotlib.pyplot as plt

DOMAIN = [0.0, 1.0]
M=2 # dimensionality of the problem
MAX_COST = 10000*M
N = 100 # seize of the population

def ZDT1(x):
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h = 1 - (f1 / g) ** 0.5
    f2 = g * h
    return [f1, f2]

def evaluation(P):

    for instance in P:
        x = instance[0] # take only coordinates
        instance[2] = ZDT1(x)

    evaluation.ncalls += len(P)
    return P, evaluation.ncalls

def initialize_population():
    population = []
    for _ in range(N):
        x1 = random.uniform(DOMAIN[0], DOMAIN[1])
        x2 = random.uniform(DOMAIN[0], DOMAIN[1])
        coord = [x1, x2]
        sigma = [1.0, 1.0]
        evaluation = [0.0, 0.0]
        fitness = 0 # is equal to the pareto set it belongs to. 0 is undefined, 1 is the best
        crowd_distance = 0.0
        population.append([coord, sigma, evaluation, fitness, crowd_distance])
    return population

evaluation.ncalls = 0

# f1-min, f2-min
def front_fitness_assignment(P):
    
    for point in P:
        point[3] = 0

    P = sorted(P, key=lambda x: (x[2][0], x[2][1])) # sort by minimizing f1. If f1 is the same, minimize f2
    P_new = []
    i = 1
    while len(P) != 0:
        pareto_front = front_kung(P)
        for point in pareto_front:
            point[3] = i
            P_new.append(point)

        P = [point for point in P if point[3] == 0]
        i += 1
    return P_new

def front_kung(P):
    if len(P) == 1:
        return P
    else:
        middle = len(P)//2
        T = front_kung(P[:middle])
        B = front_kung(P[middle:])
        P_new = T
        for point_B in B:
            is_dominated = False
            for point_T in T:
                if point_T[2][1] < point_B[2][1]:
                    is_dominated = True
                    break
            if is_dominated is False:
                P_new.append(point_B)
        return P_new

# Tournament Selection
def selection(P):
    P_size = len(P)
    P_selected = []
    for i in range(P_size):
        p1 = P[random.randint(0, P_size-1)]
        p2 = P[random.randint(0, P_size-1)]
        if p1[3] < p2[3] or (p1[3] == p2[3] and p1[4] > p2[4]):
            P_selected.append(p1)
        elif p1[3] > p2[3] or (p1[3] == p2[3] and p1[4] <= p2[4]):
            P_selected.append(p2)
    return P_selected

def crowding_distance_assignment(P):
    i = 0
    k = 1
    front = []
    while i != len(P):
        if P[i][3] == k:
            front.append(P[i])
            i += 1
        else:
            k += 1
            if len(front) != 0:
                crowding_distance(front)
                front = []
    return P

def crowding_distance(front):
    for point in front:
        point[4] = 0.0
    for m in range(2):
        front = sorted(front, key=lambda x: x[2][m])
        front[0][4] = np.inf
        front[-1][4] = np.inf
        for i in range(1, len(front)-1):
            front[i][4] += (front[i+1][2][m] - front[i-1][2][m]) / (front[-1][2][m] - front[0][2][m])
    return front

# Intermediate recombination
def recombination(P_selected):
    P_new = []
    P_selected_len = len(P_selected)
    while len(P_new) != P_selected_len:
        p1 = P_selected[random.randint(0, P_selected_len-1)]
        p2 = P_selected[random.randint(0, P_selected_len-1)]

        x1 = (p1[0][0] + p2[0][0]) / 2
        x2 = (p1[0][1] + p2[0][1]) / 2
        sigma1 = (p1[1][0] + p2[1][0]) / 2
        sigma2 = (p1[1][1] + p2[1][1]) / 2
        evaluation = [0.0, 0.0]
        fitness = 0 # is equal to the pareto set it belongs to. 0 is undefined, 1 is the best
        crowd_distance = 0.0
        P_new.append([[x1, x2], [sigma1, sigma2], evaluation, fitness, crowd_distance])
    return P_new

def reflective_clipping(value, lower, upper):
    """
    Simple reflective clipping.
    """
    while value < lower or value > upper:
        if value < lower:
            value = lower + (lower - value)
        elif value > upper:
            value = upper - (value - upper)
    return value


def mutation(P_recombined):

    for i in range(len(P_recombined)):
            n = len(P_recombined[i][0]) # number of dimensions
            random_for_all = np.random.normal(0, 1) # for every dimension of one individual
            for j in range(n):
                sigma = P_recombined[i][1][j] * np.exp(
                    random_for_all / np.sqrt(2*n)
                    + np.random.normal(0, 1) / np.sqrt(2*np.sqrt(n)))
                P_recombined[i][1][j] = reflective_clipping(sigma, 1e-6, 1)
                xi = P_recombined[i][0][j] + np.random.normal(0, P_recombined[i][1][j])
                P_recombined[i][0][j] = reflective_clipping(xi, DOMAIN[0], DOMAIN[1])
    return P_recombined

def replacement(P, offspring):
    R = P + offspring
    R = front_fitness_assignment(R)
    R = crowding_distance_assignment(R)
    P_new = []

    i = 0
    k = 1
    front = []
    while i != len(R):
        if R[i][3] == k:
            front.append(R[i])
            i += 1
        else:
            k += 1
            if len(front) != 0:
                if len(P_new) + len(front) <= N:
                    P_new += front
                else:
                    #crowding_distance(front)
                    front = sorted(front, key=lambda x: x[4], reverse=True)
                    P_new += front[:N-len(P_new)]
                    break
                front = []
    return P_new

P = initialize_population()
P, cost = evaluation(P)
P = front_fitness_assignment(P)
P = crowding_distance_assignment(P)

while True:
    P_selected = selection(P)
    offspring = recombination(P_selected)
    offspring = mutation(offspring)
    offspring, cost = evaluation(offspring)
    # if cost > MAX_COST:
    #     break
    P = replacement(P, offspring)
    if cost + N > MAX_COST:
        break

x_coords = [point[2][0] for point in P]
y_coords = [point[2][1] for point in P]

plt.figure(figsize=(8, 6))
plt.scatter(x_coords, y_coords, color='blue')
plt.title("Pareto optimal front of ZDT1")
plt.xlabel("f1")
plt.ylabel("f2")
plt.grid(True)
plt.legend()
plt.show()
print("Done")