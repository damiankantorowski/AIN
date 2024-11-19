import numpy as np
from numba import njit
from argparse import ArgumentParser

POP_SIZE = 20
ELITE_SIZE = 2
BITS = 16
GRAY = True
MUTATION_RATE = 1 / BITS
DOMAINS = np.array([[-30, 30], [-100, 100], [-10.24, 10.24]])

@njit
def gray_to_binary(gray):
    binary = gray.copy()
    for i in range(1, len(gray)):
        binary[i] = binary[i - 1] ^ gray[i]
    return binary

@njit
def bin2real(binary, num_range):
    if GRAY:
        binary = gray_to_binary(binary)
    integer = np.sum(2 ** np.arange(BITS) * binary)
    return 2 * num_range / 2 ** BITS * integer - num_range

@njit
def rosenbrock(x, cost=0):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2), cost + 1

@njit
def salomon(x, cost=0):
    root = np.sqrt(np.sum(x**2))
    return 1 - np.cos(2 * np.pi * root) + 0.1 * root, cost + 1

@njit
def whitley(x, cost=0):
    n = len(x)
    result = 0
    for i in range(n):
        for j in range(n):
            y = 100 * (x[i] - x[j]**2)**2 + (1 - x[j])**2
            result += (y ** 2)/4000 - np.cos(y) + 1
    return result, cost + 1

@njit
def select(population, scores, k=2):
    n_selected = POP_SIZE-ELITE_SIZE
    selected = np.zeros((n_selected,) + population[0].shape, dtype=population.dtype)
    selected_scores = np.zeros(n_selected, dtype=scores.dtype)
    for i in range(n_selected):
        indices = np.random.choice(len(population), k, replace=False)
        best = np.min(scores[indices])
        best_index = np.where(scores == best)[0][0]
        selected[i] = population[best_index]
        selected_scores[i] = best
    return selected, selected_scores

@njit
def crossover(parents):
    offspring = parents.copy()
    for i in range(0, len(offspring), 2):
        for j in range(len(offspring[i])):
            point1, point2 = sorted(np.random.choice(BITS, 2, replace=False))
            offspring[i][j][:point1] = parents[i][j][:point1]
            offspring[i][j][point1:point2] = parents[i + 1][j][point1:point2]
            offspring[i][j][point2:] = parents[i][j][point2:]

            offspring[i + 1][j][:point1] = parents[i + 1][j][:point1]
            offspring[i + 1][j][point1:point2] = parents[i][j][point1:point2]
            offspring[i + 1][j][point2:] = parents[i + 1][j][point2:]
    return offspring

@njit
def mutate(population):
    mutated = population.copy()
    for i in range(len(mutated)):
        for j in range(len(mutated[i])):
            mask = np.random.uniform(0.0, 1.0, size=mutated[i][j].shape) < MUTATION_RATE
            mutated[i][j] ^= mask
    return mutated

@njit
def add_elites(population, offspring, scores):
    elites = population[np.argsort(scores)[:ELITE_SIZE]]
    return np.concatenate((elites, offspring))

@njit
def evaluate(f, population, cost):
    scores = np.zeros(len(population))
    for i in range(len(population)):
        x_real = np.zeros(len(population[i]))
        for j in range(len(population[i])):
            x_real[j] = bin2real(population[i][j], DOMAINS[f-1][1])
        if f == 1:
            scores[i], cost = rosenbrock(x_real, cost)
        elif f == 2:
            scores[i], cost = salomon(x_real, cost)
        else:
            scores[i], cost = whitley(x_real, cost)
    return scores, cost

def get_result(x, num_range):
    return tuple(map(lambda xi: float(bin2real(xi, num_range)), x))

def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--f", type=int, choices=[1, 2, 3], default=1)
    args = parser.parse_args()
    
    max_cost = 10000 * args.n
    cost = 0
    population = np.random.randint(0, 2, (POP_SIZE, args.n, BITS))
    scores, cost = evaluate(args.f, population, cost)
    best_x, best_score = population[i := np.argmin(scores)], scores[i]
    while cost < max_cost:
        selected, selected_scores = select(population, scores)
        offspring = crossover(selected)
        offspring = mutate(offspring)
        population = add_elites(population, offspring, scores)
        scores, cost = evaluate(args.f, population, cost)
        if scores[new_i := np.argmin(scores)] < best_score:  
            best_x, best_score = population[new_i], scores[new_i]
    print(f"After {cost} evaluations:")
    print(f"Best parameters: {get_result(best_x, DOMAINS[args.f-1][1])}")
    print(f"Best score: {best_score}")

if __name__ == "__main__":
    main()
