import numpy as np
from numba import njit # for faster execution

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
    """
    Basic tournament selection. Number of selected individuals is equal to the population size minus the number of elites.
    :param population: current population
    :param scores: scores of population
    :param k: number of individuals to compete in each tournament
    :param n_elites: number of elites to keep in the next generation
    :return: selected individuals and their scores
    """ 
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
    """
    Simple reflective clipping.
    """
    while value < lower or value > upper:
        if value < lower:
            value = lower + (lower - value)
        elif value > upper:
            value = upper - (value - upper)
    return value

@njit
def crossover(parents, scores, f):
    """
    Weighted recombination from Lecture #5, first and second slide in page 5th.
    :param parents: parents
    :param scores: scores of parents
    :param f: objective function index
    :return: offspring
    """

    parents_len = len(parents)
    offspring = parents.copy()  

    parents = parents[np.argsort(scores)[::-1]]
    weights = np.array([np.log(parents_len + 1) - np.log(i+1) for i in range(parents_len)])
    
    sigma_term = np.zeros(parents.shape[2])  # One entry per dimension (sigma)
    xi_term = np.zeros(parents.shape[2])     # One entry per dimension (xi)

    # Calculate weighted sum for sigma_term and xi_term across all parents for each dimension
    for j in range(parents.shape[2]):  # Loop over each dimension
        sigma_term[j] = np.sum(parents[:, 1, j] * weights) / np.sum(weights)
        xi_term[j] = np.sum(parents[:, 0, j] * weights) / np.sum(weights)
    # Loop across every parent
    for i in range(parents_len):
        n = len(parents[i][0])
        random_for_all = np.random.normal(0, 1)
        # Loop acroos every dimension
        for j in range(n):
            sigma = sigma_term[j] * np.exp(
            random_for_all / np.sqrt(2 * n) + np.random.normal(0, 1) / np.sqrt(2 * np.sqrt(n))
            )
            offspring[i][1][j] = reflective_clipping(sigma, 0, 1)
            xi = xi_term[j] + offspring[i][1][j] * np.random.normal(0, 1)
            offspring[i][0][j] = reflective_clipping(xi, DOMAINS[f][0], DOMAINS[f][1])
    return offspring

@njit
def mutate(offspring, f):
    """
    Implementation of mutation from lecture where individual consists of two vectors: x and sigma.
    :param offspring: 
    :param f: objective function index
    :return: mutated offspring
    """
    mutated = offspring.copy()
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
    """
    Add n_elites best individuals from the previous generation to the offspring.
    :param prev_gen: previous generation
    :param offspring: offspring
    :param prev_gen_scores: scores of previous generation
    :param offspring_scores: scores of offspring
    :param n_elites: number of elites to keep
    """
    elite_indices = np.argsort(prev_gen_scores)[:n_elites]
    elites = prev_gen[elite_indices]
    elite_scores = prev_gen_scores[elite_indices]

    # Combine elites with offspring and their scores
    return np.concatenate((elites, offspring)), np.concatenate((elite_scores, offspring_scores))

@njit
def evaluate(f, population, cost):
    """
    Redirection to the appropriate function for evaluation.
    :param f: objective function index
    :param population: population to evaluate
    :param cost: current cost
    """
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
    # Algorithm parameters
    pop_size = 22
    n_elites = 10
    k = 6

    n=15 # Number of dimensions
    max_cost = 10000*n  # Max evaluations

    f = 2 # Objective function index, 0 for Rosenbrock, 1 for Salomon, 2 for Whitley

    values = np.random.uniform(DOMAINS[f][0], DOMAINS[f][1], (pop_size, n))
    sigma = np.ones_like(values)
    population = np.stack((values, sigma), axis=1) # E.g. for shape (22, 2, 5) - 22 individuals, each with 2 vectors of 5 dimensions, one for x and one for sigma

    cost = 0
    scores, cost = evaluate(f, population, cost)
    best_x, best_score = population[i := np.argmin(scores)], scores[i]

    # Loop until max_cost is reached
    while True:
        selected, selected_scores = select(population, scores, k, n_elites)
        offspring = crossover(selected, selected_scores, f)
        offspring = mutate(offspring, f)

        # Evaluate the offspring without including elites
        offspring_scores, cost = evaluate(f, offspring, cost)
        # Check if the cost limit is reached
        if cost > max_cost:
            break
        # Concatenate the offspring with the elites from the previous generation 
        population, scores = add_elites(population, offspring, scores, offspring_scores, n_elites)

        if scores[new_i := np.argmin(scores)] < best_score:
            best_x, best_score = population[new_i], scores[new_i]
            print(f"Best score: {best_score}")
        


if __name__ == "__main__":
    main()
