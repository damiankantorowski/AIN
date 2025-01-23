# SPEA-II implementation

import numpy as np
import random
import matplotlib.pyplot as plt
import copy

DOMAIN = [0.0, 1.0]
M = 50 # dimensionality of the problem
MAX_ITERATIONS = 500
N = 150 # size of the population

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
        instance[2] = ZDT1(x)



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
        fitness = 0.0 # is equal to the pareto set it belongs to. 0 is undefined, 1 is the best
        raw_fitness = 0
        population.append([coord, sigma, evaluation, fitness, raw_fitness])
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
        if p1[3] < p2[3]:
            P_selected.append(copy.deepcopy(p1))
        else:
            P_selected.append(copy.deepcopy(p2))
    return P_selected

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

# ! CHECK IF CORRECT
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
        offspring1_x = []
        offspring2_x = []
        offspring1_sigma = []
        offspring2_sigma = []
        for j in range(M):
            offspring1_x.append(reflective_clipping(0.5 * ((1 + beta) * P_selected[i][0][j] + (1 - beta) * P_selected[i + 1][0][j]), DOMAIN[0], DOMAIN[1]))
            offspring2_x.append(reflective_clipping(0.5 * ((1 - beta) * P_selected[i][0][j] + (1 + beta) * P_selected[i + 1][0][j]), DOMAIN[0], DOMAIN[1]))

            offspring1_sigma.append(reflective_clipping(0.5 * ((1 + beta) * P_selected[i][1][j] + (1 - beta) * P_selected[i + 1][1][j]), 1e-6, 1))
            offspring2_sigma.append(reflective_clipping(0.5 * ((1 - beta) * P_selected[i][1][j] + (1 + beta) * P_selected[i + 1][1][j]), 1e-6, 1)) 

        evaluation = [0.0, 0.0]
        fitness = 0.0
        raw_fitness = 0
        P_new.append([offspring1_x, offspring1_sigma, evaluation, fitness, raw_fitness])
        P_new.append([offspring2_x, offspring2_sigma, evaluation, fitness, raw_fitness])
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
            
            random_for_all = np.random.normal(0, 1) # for every dimension of one individual
            for j in range(M):
                sigma = P_recombined[i][1][j] * np.exp(
                    random_for_all / np.sqrt(2*M)
                    + np.random.normal(0, 1) / np.sqrt(2*np.sqrt(M)))
                P_recombined[i][1][j] = reflective_clipping(sigma, 1e-6, 1)
                xi = P_recombined[i][0][j] + np.random.normal(0, P_recombined[i][1][j])
                P_recombined[i][0][j] = reflective_clipping(xi, DOMAIN[0], DOMAIN[1])
    return P_recombined

# Next generation consists of the best pareto sets from sum of parents and offspring
# The last picked pareto set that fits into the next generetion that exceeds the population size is cut off by crowding distance

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

        not_dominated_points = [point for point in pareto_front if point[4] == 0]
        if len(not_dominated_points) > 1:
            not_dominated_points = sorted(not_dominated_points, key=lambda point: point[2][0])
            x_coords_not_dominated = [point[2][0] for point in not_dominated_points]
            y_coords_not_dominated = [point[2][1] for point in not_dominated_points]
        
            plt.plot(x_coords_not_dominated, y_coords_not_dominated, color=colors[i], linestyle='-')
        

        i += 1

    plt.suptitle(f"SPEA-II {objective_func_name}, {num_dimensions} dimensions")
    plt.title(f"Population size: {N}", fontsize=10)
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.grid(True)
    plt.legend()  
    plt.tight_layout()
    plt.savefig(f"pareto_fronts_{objective_func_name}_{num_dimensions}.png")
    plt.show()

# domination check for 2 dimensional evaluation space
def dominates(p1, p2):
    if p1[2][0] < p2[2][0] and p1[2][1] <= p2[2][1] or p1[2][0] <= p2[2][0] and p1[2][1] < p2[2][1]:
        return True
    return False
    

def fitness_assignment(P):
    S = [0] * len(P)

    # calculate strength
    for i, p1 in enumerate(P):
        for p2 in P:
            if dominates(p1, p2):
                S[i] += 1

    # calculate raw fitness R            
    for p1 in P:
        p1[3] = 0
        for j, p2 in enumerate(P):
            if dominates(p2, p1):
                p1[4] += S[j] # add to raw fitness
                p1[3] += S[j] # add to final fitness
    
    # calculate density

    k = int(np.sqrt(len(P)))
    D = [0.0] * len(P) # distance list
    for i, p1 in enumerate(P):
        for j, p2 in enumerate(P):
            if i == j:
                D[j] = np.inf
            else:
                D[j] += np.sqrt((p1[2][0] - p2[2][0])**2 + (p1[2][1] - p2[2][1])**2) # euclidean distance for 2 dimensions
        D_indx = np.argsort(D)
        p1[3] += 1 / (D[D_indx[k]] + 2) # add distance component to final fitness

        D = [0.0] * len(P)

# TODO ADD SAME POINTS REMOVAL
def update(P):
    A = [p for p in P if p[4] == 0]
    
    if(len(A) > N):
        D = np.zeros((len(A), len(A)))
        # calculate distance matrix
        for i, p1 in enumerate(A):
            for j, p2 in enumerate(A):
                if i == j:
                    D[i][j] = np.inf
                else:
                    D[i][j] += np.sqrt((p1[2][0] - p2[2][0])**2 + (p1[2][1] - p2[2][1])**2) # euclidean distance for 2 dimensions
        # adding points with the smallest distance
        while len(A) != N:
            min_dist = np.min(D)
            min_indices = np.where(D == min_dist)
            min_index = min_indices[0][0]
            A.pop(min_index)
            D = np.delete(D, min_index, axis=0)
            D = np.delete(D, min_index, axis=1)
        # TODO ADD BREAKING TIES MECHANISM
        #min_dist = np.min(d[0] for d in D_list)
        #min_indices = [i for i, d in enumerate(D_list) if d[0] == min_dist]

        
    elif(len(A) < N):
        dominated_P = [p for p in P if p[4] != 0]
        dominated_P = sorted(dominated_P, key=lambda x: x[3]) # sort by final fitness
        A.extend(dominated_P[:N-len(A)])
    return A
    

A = initialize_population() # archive
evaluation(A)

cur_iteration = 1
checkpoints = [20, 50, 100, MAX_ITERATIONS]
pareto_evolution = []
while True:
    fitness_assignment(A)
    if(cur_iteration in checkpoints):
        pareto_evolution.append([[point[0], point[1], point[2], point[3], point[4]] for point in A])
        print("checkpoint: ", cur_iteration)
        if(cur_iteration == MAX_ITERATIONS):
            break
    P_selected = selection(A)
    offspring = recombination(P_selected)
    offspring = mutation(offspring)
    evaluation(offspring)
    fitness_assignment(offspring + A) # ! WORKS?
    A = update(offspring + A)
    cur_iteration += 1

draw_pareto_fronts(pareto_evolution, "ZDT", M)