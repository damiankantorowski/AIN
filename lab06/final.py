import optuna
import numpy as np
from numba import njit

DOMAINS = np.array([[-30, 30], [-100, 100], [-10.24, 10.24]])

@njit
def rosenbrock(x, cost):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2), cost + 1

@njit
def salomon(x, cost):
    root = np.sqrt(np.sum(x**2))
    return 1 - np.cos(2 * np.pi * root) + 0.1 * root, cost + 1

@njit
def whitley(x, cost):
    n = len(x)
    result = 0
    for i in range(n):
        for j in range(n):
            y = 100 * (x[i] - x[j]**2)**2 + (1 - x[j])**2
            result += (y ** 2)/4000 - np.cos(y) + 1
    return result, cost + 1

@njit
def select(population, scores, k, n_elites):
    pop_size = len(population)
    n_selected = pop_size-n_elites
    selected = np.zeros((n_selected,) + population[0].shape, dtype=population.dtype)
    selected_scores = np.zeros(n_selected, dtype=scores.dtype)
    
    for i in range(n_selected):
        indices = np.random.choice(pop_size, k, replace=False)
        best = np.min(scores[indices])
        best_index = np.where(scores == best)[0][0]
        selected[i] = population[best_index]
        selected_scores[i] = best

    return selected, selected_scores

@njit
def reflective_clipping(value, lower, upper):
    while value < lower or value > upper:
        if value < lower:
            value = lower + (lower - value)
        elif value > upper:
            value = upper - (value - upper)
    return value

# It only works when parents are even.
@njit
def crossover(parents, scores, f):

    parents_len = len(parents)
    offspring = parents.copy()  

    parents = parents[np.argsort(scores)[::-1]]
    weights = np.array([np.log(parents_len + 1) - np.log(i+1) for i in range(parents_len)])
    
    sigma_term = np.zeros(parents.shape[2])  # One entry per dimension (sigma)
    xi_term = np.zeros(parents.shape[2])     # One entry per dimension (xi)

    # Calculate weighted sum for sigma_term and xi_term across all parents for each dimension
    for j in range(parents.shape[2]):  # Loop over each dimension
        sigma_term[j] = np.sum(parents[:, 1, j] * weights) / np.sum(weights)  # Weighted sum for sigma
        xi_term[j] = np.sum(parents[:, 0, j] * weights) / np.sum(weights)  # Weighted sum for xi

    for i in range(parents_len):
        n = len(parents[i][0])
        random_for_all = np.random.normal(0, 1)
        for j in range(n):
            sigma = sigma_term[j] * np.exp(
            random_for_all / np.sqrt(2 * n) + np.random.normal(0, 1) / np.sqrt(2 * np.sqrt(n))
            )
            offspring[i][1][j] = reflective_clipping(sigma, 0, 1)
            xi = xi_term[j] + offspring[i][1][j] * np.random.normal(0, 1)
            offspring[i][0][j] = reflective_clipping(xi, DOMAINS[f][0], DOMAINS[f][1])
    return offspring

@njit
def mutate(population, f=1):
    mutated = population.copy()
    for i in range(len(mutated)):
            n = len(mutated[i][0])
            random_for_all = np.random.normal(0, 1) # for every dimension of one individual
            for j in range(n):
                sigma = mutated[i][1][j] * np.exp(
                    random_for_all / np.sqrt(2*n)
                    + np.random.normal(0, 1) / np.sqrt(2*np.sqrt(n)))
                mutated[i][1][j] = reflective_clipping(sigma, 1e-6, 1)
                xi = mutated[i][0][j] + np.random.normal(0, mutated[i][1][j])
                mutated[i][0][j] = reflective_clipping(xi, DOMAINS[f][0], DOMAINS[f][1])
    return mutated


@njit
def add_elites(prev_gen, offspring, prev_gen_scores, offspring_scores, n_elites):
    elite_indices = np.argsort(prev_gen_scores)[:n_elites]
    elites = prev_gen[elite_indices]
    elite_scores = prev_gen_scores[elite_indices]
    return np.concatenate((elites, offspring)), np.concatenate((elite_scores, offspring_scores))

@njit
def evaluate(f, population, cost):
    scores = np.zeros(len(population))
    for i in range(len(population)):
        xi = population[i][0]
        if f == 0:
            scores[i], cost = rosenbrock(xi, cost)
        elif f == 1:
            scores[i], cost = salomon(xi, cost)
        else:
            scores[i], cost = whitley(xi, cost)
    return scores, cost

def main():

    pop_size = 22
    n_elites = 10
    k = 6

    n=5 # Number of dimensions
    max_cost = 10000*n  # Max evaluations

    num_runs = 100  # Number of runs per parameter configuration per function

    results = {0: [], 1: [], 2: []}  # Store results for each function

    for f in [0, 1, 2]:
        for _ in range(num_runs):
            values = np.random.uniform(DOMAINS[f][0], DOMAINS[f][1], (pop_size, n))
            sigma = np.ones_like(values)
            population = np.stack((values, sigma), axis=1)

            cost = 0

            scores, cost = evaluate(f, population, cost)
            best_x, best_score = population[i := np.argmin(scores)], scores[i]
            while cost < max_cost:
                selected, selected_scores = select(population, scores, k, n_elites)
                offspring = crossover(selected, selected_scores, f)
                offspring = mutate(offspring, f)
                # only for clarity               
                prev_gen_scores = scores
                # Evaluate the offspring without including elites
                offspring_scores, cost = evaluate(f, offspring, cost)
                if cost > max_cost:
                    break
                # Concatenate the offspring with the elites from the previous generation 
                population, scores = add_elites(population, offspring, prev_gen_scores, offspring_scores, n_elites)

                if scores[new_i := np.argmin(scores)] < best_score:
                    best_x, best_score = population[new_i], scores[new_i]

            results[f].append(best_score)

    # Calculate the mean score for each function
    mean_scores = {f: np.mean(results[f]) for f in results}

    overall_mean_score = np.mean(list(mean_scores.values()))
    print(overall_mean_score)


if __name__ == "__main__":
    main()
