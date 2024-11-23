import optuna
import numpy as np
from numba import njit

# TODO implement different clipping?
# TODO 

#POP_SIZE = 20 # Has to be even
#ELITE_SIZE = 2 # Has to be even
# TODO decide if these should be a parameters
#P_CROSSOVER_MAX = 0.9 
#P_CROSSOVER_MIN = 0.3 
#MAX_EVALUATIONS = 10000
#CUR_EVALUATIONS = 0

DOMAINS = np.array([[-30, 30], [-100, 100], [-10.24, 10.24]])



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

# TODO Does this work?
@njit
def weighted_random_choice(pop_size, selection_probabilities):
    cumulative_sum = np.cumsum(selection_probabilities)
    r = np.random.rand()
    for i in range(pop_size):
        if r <= cumulative_sum[i]:
            return i


# WE HAVE PERMISSION ONLY FROM: lectures.
# TODO rank selection
# @njit
# def select(population, scores, selection_type="tournament", k=2, n_elites=2):
#     if selection_type == "tournament":
#         n_selected = len(population)-n_elites
#         selected = np.zeros((n_selected,) + population[0].shape, dtype=population.dtype)
#         selected_scores = np.zeros(n_selected, dtype=scores.dtype)
#         for i in range(n_selected):
#             indices = np.random.choice(len(population), k, replace=False)
#             best = np.min(scores[indices])
#             best_index = np.where(scores == best)[0][0]
#             selected[i] = population[best_index]
#             selected_scores[i] = best
#         return selected, selected_scores
@njit
def select(population, scores, selection_type="tournament", k=2, a=0.0005, epsilon=1e-6, n_elites=2):
    pop_size = len(population)
    n_selected = pop_size-n_elites
    selected = np.zeros((n_selected,) + population[0].shape, dtype=population.dtype)
    selected_scores = np.zeros(n_selected, dtype=scores.dtype)
    
    if selection_type == "tournament":
        for i in range(n_selected):
            indices = np.random.choice(pop_size, k, replace=False)
            best = np.min(scores[indices])
            best_index = np.where(scores == best)[0][0]
            selected[i] = population[best_index]
            selected_scores[i] = best


            # DIFFERENT OPTION
            # indices = np.random.choice(pop_size, k, replace=False)
            # best_index = indices[np.argmin(scores[indices])]  # Use indices directly with np.argmin
            # selected[i] = population[best_index]
            # selected_scores[i] = scores[best_index]

    if selection_type == "rank":
        sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order
        #a = 1 / (pop_size*10)
        d = ((2 * 1) / pop_size - (2 * a)) / (pop_size - 1)
        selection_probabilities = np.zeros(pop_size)  # Initialize selection probabilities
        for i in range(pop_size):
            cur_index = sorted_indices[i]
            selection_probabilities[cur_index] = a + i*d
        if not np.isclose(np.sum(selection_probabilities), 1.0, atol=1e-9):
            print("ERROR: Selection probabilities do not sum to 1.")
        # Select individuals based on ranking probabilities
        for i in range(n_selected):
            #chosen_index = np.random.choice(pop_size, p=selection_probabilities)  # Rank-based selection
            chosen_index = weighted_random_choice(pop_size, selection_probabilities)
            selected[i] = population[int(chosen_index)]
            selected_scores[i] = scores[int(chosen_index)]
    if selection_type == "roulette":
        fitness = 1 / (epsilon + scores)
        selection_probabilities = fitness / np.sum(fitness)

        for i in range(n_selected):
            #chosen_index = np.random.choice(pop_size, p=selection_probabilities)
            chosen_index = weighted_random_choice(pop_size, selection_probabilities)
            selected[i] = population[int(chosen_index)]
            selected_scores[i] = scores[int(chosen_index)]

    return selected, selected_scores

# Function to decrease the crossover probability based on evaluations
#@njit
# def get_crossover_probability():
#     # Linearly decrease crossover probability based on the number of evaluations
#     return P_CROSSOVER_MAX - (P_CROSSOVER_MAX - P_CROSSOVER_MIN) * (CUR_EVALUATIONS / MAX_EVALUATIONS)

@njit
def reflective_clipping(value, lower, upper):
    while value < lower or value > upper:
        if value < lower:
            value = lower + (lower - value)
        elif value > upper:
            value = upper - (value - upper)
    return value

# TODO implement advanced intermediate recombination - BLX-alpha(?)
# It only works when parents are even.
# WE HAVE PERMISSION ONLY FROM: lectures + literature
@njit
def crossover(parents, scores, crossover_type, eta, f):

    #crossover_prob = get_crossover_probability()

    parents_len = len(parents)
    offspring = parents.copy()  

    if crossover_type == "weighted":
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

    elif crossover_type == "SBX":
        u = np.random.uniform(0, 1)
        if u <= 0.5:
            beta = (2*u)**(1/(1+eta))
        else:
            beta = (1/(2*(1-u)))**(1/(1+eta))
        for i in range(0, parents_len, 2):
            n = len(parents[i][0])
            for j in range(n):
                offspring[i][0][j] = reflective_clipping(0.5 * ((1 + beta) * parents[i][0][j] + (1 - beta) * parents[i + 1][0][j]), DOMAINS[f][0], DOMAINS[f][1]) 
                offspring[i + 1][0][j] = reflective_clipping(0.5 * ((1 - beta) * parents[i][0][j] + (1 + beta) * parents[i + 1][0][j]), DOMAINS[f][0], DOMAINS[f][1]) 
    

    return offspring

# WE HAVE PERMISSION ONLY FROM: lectures + literature
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
                mutated[i][1][j] = reflective_clipping(sigma, 0, 1) # TODO sigma should have minimal value that is not 0
                xi = mutated[i][0][j] + np.random.normal(0, mutated[i][1][j])
                mutated[i][0][j] = reflective_clipping(xi, DOMAINS[f][0], DOMAINS[f][1])
    return mutated

# @njit
# def add_elites(population, offspring, scores, n_elites):
#     elites = population[np.argsort(scores)[:n_elites]]
#     return np.concatenate((elites, offspring))

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

def get_result(x, num_range):
    return tuple(map(lambda xi: float(xi), x)) # TODO is it correct?

def objective(trial):
    # Define the search space for selection, crossover, mutation types and rates
    #global GRAY
    # global ELITE_SIZE
    #global POP_SIZE
    # global P_CROSSOVER_MAX
    # global P_CROSSOVER_MIN
    # global CUR_EVALUATIONS
    # global MAX_EVALUATIONS

    #P_CROSSOVER_MAX = trial.suggest_float("P_CROSSOVER_MAX", 0.7, 0.95)
    #P_CROSSOVER_MIN = trial.suggest_float("P_CROSSOVER_MIN", 0.1, 0.3)
    pop_size = trial.suggest_categorical("pop_size", [20, 22, 24, 26, 28, 30]) # Has to be even
    n_elites = trial.suggest_categorical("n_elites", [0, 2, 4, 6]) # Has to be even
    #GRAY = trial.suggest_categorical("GRAY", [True, False])
    selection_type = trial.suggest_categorical("selection_type", ["tournament", "rank", "roulette"])
    #selection_type = "rank"
    crossover_type = trial.suggest_categorical("crossover_type", ["weighted", "SBX"])
    #crossover_type = "SBX"
    if crossover_type == "SBX":
        eta = trial.suggest_int("eta", 10, 20)
    else:
        eta = 1

    if selection_type == "tournament":
        k = trial.suggest_int("k", 2, 6)
        a = 1
        epsilon = 1
    elif selection_type == "rank":
        k = 1
        a = trial.suggest_float("a", 0.00005, 0.005)
        epsilon = 1
    elif selection_type == "roulette":
        k = 1
        a = 1
        epsilon = trial.suggest_float("epsilon", 1e-9, 1)

    n=15 # Number of dimensions
    max_cost = 10000*n  # Max evaluations
    #MAX_EVALUATIONS = max_cost

    num_runs = 5  # Number of runs per parameter configuration per function

    results = {0: [], 1: [], 2: []}  # Store results for each function

    for f in [0, 1, 2]:
        for _ in range(num_runs):
            values = np.random.uniform(DOMAINS[f][0], DOMAINS[f][1], (pop_size, n))
            sigma = np.ones_like(values)
            population = np.stack((values, sigma), axis=1)

            cost = 0
            #CUR_EVALUATIONS = cost
            scores, cost = evaluate(f, population, cost)
            best_x, best_score = population[i := np.argmin(scores)], scores[i]
            while cost < max_cost: # TODO loop should end when max_cost is reached, so better to do it in a evaluation function
                selected, selected_scores = select(population, scores, selection_type, k, a, epsilon, n_elites)
                offspring = crossover(selected, selected_scores, crossover_type, eta, f)
                offspring = mutate(offspring, f)
                # only for clarity               
                prev_gen_scores = scores
                # Evaluate the offspring without including elites
                offspring_scores, cost = evaluate(f, offspring, cost) # TODO prevent exceeding max_cost. Right now it exceeds by a little.
                # Concatenate the offspring with the elites from the previous generation 
                population, scores = add_elites(population, offspring, prev_gen_scores, offspring_scores, n_elites) # old_scores - scores of previous generation.

                # population = add_elites(population, offspring, scores, n_elites)
                # scores, cost = evaluate(f=f, population=population, cost=cost)
                #CUR_EVALUATIONS = cost
                if scores[new_i := np.argmin(scores)] < best_score:
                    best_x, best_score = population[new_i], scores[new_i]

            results[f].append(best_score)

    # Calculate the mean score for each function
    mean_scores = {f: np.mean(results[f]) for f in results}

    # Return means as part of trial information
    trial.set_user_attr("rosenbrock_mean", mean_scores[0])
    trial.set_user_attr("salomon_mean", mean_scores[1])
    trial.set_user_attr("whitley_mean", mean_scores[2])

    # Return the overall average score across all functions
    overall_mean_score = np.mean(list(mean_scores.values()))
    return overall_mean_score


def main():
    print(objective.__code__)
    print(evaluate.__code__)

    database_url = "sqlite:///study.db?timeout=60"
    #study = optuna.create_study(direction="minimize", storage=database_url, study_name="real_test_1", load_if_exists=True)
    study = optuna.create_study(direction="minimize")  # Minimize the cost or maximize the fitness
    study.optimize(objective, n_trials=10, n_jobs=1) # 2 used cores

# Retrieve and display trial results
    print("Best trial:")
    best_trial = study.best_trial
    print(f"Value: {best_trial.value}")
    print(f"Params: {best_trial.params}")
    print(f"Rosenbrock mean: {best_trial.user_attrs['rosenbrock_mean']}")
    print(f"Salomon mean: {best_trial.user_attrs['salomon_mean']}")
    print(f"Whitley mean: {best_trial.user_attrs['whitley_mean']}")

    # Optionally, print a summary of all trials
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))


    # Rename parameters columns for clarity
    trials_df = trials_df.rename(
        columns={
            col: col.replace("user_attrs_", "")
            for col in trials_df.columns
            if col.startswith("user_attrs_")
        }
    )
    trials_df = trials_df.rename(
        columns={
            col: col.replace("params_", "")
            for col in trials_df.columns
            if col.startswith("params_")
        }
    )
    trials_df_sorted = trials_df.sort_values(by="value", ascending=True)
    
    # Save to CSV file
    trials_df_sorted.to_csv('optuna.csv', index=False)
    
    #print(trials_df_sorted)   

if __name__ == "__main__":
    main()
