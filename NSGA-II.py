# NSGA-II implementation

import numpy as np
import random
import matplotlib.pyplot as plt

DOMAIN = [0.0, 1.0]
M = 50 # dimensionality of the problem
MAX_ITERATIONS = 500
N = 150 # seize of the population

def ZDT1(x):
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h = 1 - (f1 / g) ** 0.5
    f2 = g * h
    return [f1, f2]

def ZDT2(x):
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h = 1 - (f1 / g) ** 2
    f2 = g * h
    return [f1, f2]

def ZDT3(x):
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    h = 1 - (f1 / g) ** 0.5 - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    return [f1, f2]

def ZDT4(x):
    f1 = x[0]
    g = 1 + 10 * (len(x) - 1) + sum([xi**2 - 10 * np.cos(4 * np.pi * xi) for xi in x[1:]])
    h = 1 - (f1 / g) ** 0.5
    f2 = g * h
    return [f1, f2]

def ZDT6(x):
    f1 = 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0])) ** 6
    g = 1 + 9 * (sum(x[1:]) / (len(x) - 1)) ** 0.25
    h = 1 - (f1 / g) ** 2
    f2 = g * h
    return [f1, f2]


def evaluation(P):

    for instance in P:
        x = instance[0] # take only coordinates
        instance[2] = ZDT2(x)

    return P


# Each instance of the population is a list of 5 elements: [coordinates, sigma, evaluation, fitness, crowding distance].
def initialize_population():
    population = []
    for _ in range(N):
        coord = []
        for _ in range(M):
            x = random.uniform(DOMAIN[0], DOMAIN[1])
            coord.append(x)
        sigma = [1.0 for _ in range(M)]
        evaluation = [0.0, 0.0]
        fitness = 0 # is equal to the pareto set it belongs to. 0 is undefined, 1 is the best
        crowd_distance = 0.0
        population.append([coord, sigma, evaluation, fitness, crowd_distance])
    return population

# Fitness assignment based on Pareto sets
# f1-min, f2-min
def front_fitness_assignment(P):

    for point in P:
        point[3] = 0

    P = sorted(P, key=lambda p: (p[2][0], p[2][1])) # sort by minimizing f1. If f1 is the same, minimize f2
    P_new = []
    i = 1
    while len(P) != 0:
        pareto_front = front_kung(P)
        for point in pareto_front:
            point[3] = i
            P_new.append(point)

        P = [point for point in P if point[3] == 0]
        P = sorted(P, key=lambda p: (p[2][0], p[2][1]))
        i += 1
    return P_new

def front_kung(P):
    if len(P) == 1:
        return P
    else:
        middle = len(P)//2
        T = front_kung(P[:middle])
        B = front_kung(P[middle:])
        P_new = T[:]
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
# point with better Pareto front is selected,
# if the fronts are the same, the one with the smaller crowding distance is selected
def selection(P):
    P_size = len(P)
    P_selected = []
    for i in range(P_size):
        p1 = P[random.randint(0, P_size-1)]
        p2 = P[random.randint(0, P_size-1)]
        if p1[3] < p2[3] or (p1[3] == p2[3] and p1[4] > p2[4]):
            P_selected.append(p1)
        else:
            P_selected.append(p2)
    return P_selected

# Calculate crowding distance in every Pareto front
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

            crowding_distance(front)
            front = []
    crowding_distance(front)
    return P

# Calculate crowding distance in specific front
def crowding_distance(front):
    if len(front) == 1:
        front[0][4] = np.inf
        return front
    
    for point in front:
        point[4] = 0.0
    for m in range(2):
        front = sorted(front, key=lambda x: x[2][m], reverse=True)
        front[0][4] = np.inf
        front[-1][4] = np.inf
        for i in range(1, len(front)-1):
            front[i][4] += (front[i+1][2][m] - front[i-1][2][m]) / (front[-1][2][m] - front[0][2][m])
    return front

#Intermediate recombination
# def recombination(P_selected):
#     P_new = []
#     P_selected_len = len(P_selected)
#     while len(P_new) != P_selected_len:
#         p1 = P_selected[random.randint(0, P_selected_len-1)]
#         p2 = P_selected[random.randint(0, P_selected_len-1)]
#         p3 = P_selected[random.randint(0, P_selected_len-1)]

#         x_mean = [(p1[0][i] + p2[0][i] + p3[0][i]) / 3 for i in range(M)]
#         sigma_mean = [(p1[1][i] + p2[1][i] + p3[1][i]) / 3 for i in range(M)]

#         evaluation = [0.0, 0.0]
#         fitness = 0
#         crowd_distance = 0.0
#         P_new.append([x_mean, sigma_mean, evaluation, fitness, crowd_distance])
#     return P_new

# SBX recombination
def recombination(P_selected, eta_max = 20, eta_min = 10):
    #random.shuffle(P_selected)
    P_new = []
    eta = eta_max - (eta_max - eta_min) * (cur_iteration / MAX_ITERATIONS)
    u = np.random.uniform(0, 1)
    if u <= 0.5:
        beta = (2*u)**(1/(1+eta))
    else:
        beta = (1/(2*(1-u)))**(1/(1+eta))
    for i in range(0, len(P_selected), 2):
        n = len(P_selected[i][0])
        offspring1_x = []
        offspring2_x = []
        offspring1_sigma = []
        offspring2_sigma = []
        for j in range(n):
            offspring1_x.append(reflective_clipping(0.5 * ((1 + beta) * P_selected[i][0][j] + (1 - beta) * P_selected[i + 1][0][j]), DOMAIN[0], DOMAIN[1]))
            offspring2_x.append(reflective_clipping(0.5 * ((1 - beta) * P_selected[i][0][j] + (1 + beta) * P_selected[i + 1][0][j]), DOMAIN[0], DOMAIN[1]))

            offspring1_sigma.append(reflective_clipping(0.5 * ((1 + beta) * P_selected[i][1][j] + (1 - beta) * P_selected[i + 1][1][j]), 1e-6, 1))
            offspring2_sigma.append(reflective_clipping(0.5 * ((1 - beta) * P_selected[i][1][j] + (1 + beta) * P_selected[i + 1][1][j]), 1e-6, 1)) 

        evaluation = [0.0, 0.0]
        fitness = 0
        crowd_distance = 0.0

        P_new.append([offspring1_x, offspring1_sigma, evaluation, fitness, crowd_distance])
        P_new.append([offspring2_x, offspring2_sigma, evaluation, fitness, crowd_distance])
    return P_new
        
# Simple reflective clipping
def reflective_clipping(value, lower, upper):

    while value < lower or value > upper:
        if value < lower:
            value = lower + (lower - value)
        elif value > upper:
            value = upper - (value - upper)
    return value

# Implementation of mutation from lecture where individual consists of two vectors: x and sigma.
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

# Next generation consists of the best pareto sets from sum of parents and offspring
# The last picked pareto set that fits into the next generetion that exceeds the population size is cut off by crowding distance
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

            if len(P_new) + len(front) <= N:
                P_new += front
            else:
                front = sorted(front, key=lambda x: x[4], reverse=True)
                P_new += front[:N-len(P_new)]
                break
            front = []
    return P_new

# Draw evolution of pareto fronts
# assumption of 4 checkpoints
def draw_pareto_fronts(pareto_fronts, objective_func_name, num_dimensions):
    iterations = [20, 50, 100 , 500]
    colors = ["red", "yellow", "green", "blue"]

    i = 0

    plt.figure(figsize=(10, 8))

    for pareto_front in pareto_fronts:
        x_coords = [point[2][0] for point in pareto_front]
        y_coords = [point[2][1] for point in pareto_front]
        
        plt.scatter(x_coords, y_coords, color=colors[i], label=f"Iteration: {iterations[i]}")

        not_dominated_points = [point for point in pareto_front if point[3] == 1]
        if len(not_dominated_points) > 1:
            not_dominated_points = sorted(not_dominated_points, key=lambda point: point[2][0])
            x_coords_not_dominated = [point[2][0] for point in not_dominated_points]
            y_coords_not_dominated = [point[2][1] for point in not_dominated_points]
        
            plt.plot(x_coords_not_dominated, y_coords_not_dominated, color=colors[i], linestyle='-')
        

        i += 1

    plt.suptitle(f"NSGA-II {objective_func_name}, {num_dimensions} dimensions")
    plt.title(f"Population size: {N}", fontsize=10)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()  
    plt.tight_layout()
    plt.savefig(f"pareto_fronts_{objective_func_name}_{num_dimensions}.png")
    plt.show()

P = initialize_population()
P = evaluation(P)
P = front_fitness_assignment(P)
P = crowding_distance_assignment(P)

cur_iteration = 1
checkpoints = [20, 50, 100, 500]
pareto_evolution = []
while True:
    if(cur_iteration in checkpoints):
        pareto_evolution.append([[point[0], point[1], point[2], point[3], point[4]] for point in P])
        if(cur_iteration == MAX_ITERATIONS):
            break
    P_selected = selection(P)
    offspring = recombination(P_selected)
    offspring = mutation(offspring)
    offspring = evaluation(offspring)
    P = replacement(P, offspring)
    cur_iteration += 1

draw_pareto_fronts(pareto_evolution, "ZDT", M)