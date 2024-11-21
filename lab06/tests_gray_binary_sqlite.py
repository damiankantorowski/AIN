import optuna
import numpy as np
from numba import njit

#POP_SIZE = 20 # Has to be even
#ELITE_SIZE = 2 # Has to be even
BITS = 16
GRAY = True
#MUTATION_RATE = 1 / BITS
#P_CROSSOVER_MAX = 0.9
#P_CROSSOVER_MIN = 0.3
#MAX_EVALUATIONS = 10000
#CUR_EVALUATIONS = 0
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

# WE HAVE PERMISSION ONLY FROM: lectures.
# TODO rank selection
@njit
def select(population, scores, selection_type="tournament", k=2, n_elites=2):
    if selection_type == "tournament":
        n_selected = len(population)-n_elites
        selected = np.zeros((n_selected,) + population[0].shape, dtype=population.dtype)
        selected_scores = np.zeros(n_selected, dtype=scores.dtype)
        for i in range(n_selected):
            indices = np.random.choice(len(population), k, replace=False)
            best = np.min(scores[indices])
            best_index = np.where(scores == best)[0][0]
            selected[i] = population[best_index]
            selected_scores[i] = best
        return selected, selected_scores


# Function to decrease the crossover probability based on evaluations
@njit
def get_crossover_probability(p_crossover_max, p_crossover_min, cur_evaluations, max_evaluations):
    # Linearly decrease crossover probability based on the number of evaluations
    return p_crossover_max - (p_crossover_max - p_crossover_min) * (cur_evaluations / max_evaluations)

# It only works when parents are even.
# WE HAVE PERMISSION ONLY FROM: lectures + literature
@njit
def crossover(parents, crossover_type, p_crossover_max, p_crossover_min, cur_evaluations, max_evaluations):

    crossover_prob = get_crossover_probability(p_crossover_max, p_crossover_min, cur_evaluations, max_evaluations)

    offspring = parents.copy()
    for i in range(0, len(offspring), 2):
        # Two parents crossover with probability crossover_prob
        if np.random.uniform(0.0, 1.0) < crossover_prob:
            for j in range(len(offspring[i])):

                
                if crossover_type == "one_point": 
                    point = np.random.randint(1, BITS)
                    offspring[i][j][point:] = parents[i + 1][j][point:]
                    offspring[i + 1][j][point:] = parents[i][j][point:]
                elif crossover_type == "two_point":
                    point1, point2 = sorted(np.random.choice(BITS, 2, replace=False))
                    offspring[i][j][:point1] = parents[i][j][:point1]
                    offspring[i][j][point1:point2] = parents[i + 1][j][point1:point2]
                    offspring[i][j][point2:] = parents[i][j][point2:]

                    offspring[i + 1][j][:point1] = parents[i + 1][j][:point1]
                    offspring[i + 1][j][point1:point2] = parents[i][j][point1:point2]
                    offspring[i + 1][j][point2:] = parents[i + 1][j][point2:]                    
                elif crossover_type == "uniform":
                    for k in range(BITS):
                        if np.random.uniform() < 0.5:
                            offspring[i][j][k], offspring[i + 1][j][k] = (
                                offspring[i + 1][j][k],
                                offspring[i][j][k],
                            )
    
    return offspring

# WE HAVE PERMISSION ONLY FROM: lectures + literature
@njit
def mutate(population, mutation_rate):
    mutated = population.copy()
    for i in range(len(mutated)):
        for j in range(len(mutated[i])):
            mask = np.random.uniform(0.0, 1.0, size=mutated[i][j].shape) < mutation_rate
            mutated[i][j] ^= mask
    return mutated

@njit
def add_elites(population, offspring, scores, n_elites=2):
    elites = population[np.argsort(scores)[:n_elites]]
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

def objective(trial):
    # Define the search space for selection, crossover, mutation types and rates
    #global GRAY
    #global ELITE_SIZE
    #global POP_SIZE
    #global P_CROSSOVER_MAX
    #global P_CROSSOVER_MIN
    #global CUR_EVALUATIONS
    #global MAX_EVALUATIONS

    p_crossover_max = trial.suggest_float("p_crossover_max", 0.8, 0.95)
    p_crossover_min = trial.suggest_float("p_crossover_min", 0.1, 0.3)
    pop_size = trial.suggest_categorical("pop_size", [20, 22, 24, 26, 28, 30]) # Has to be even
    n_elites = trial.suggest_categorical("n_elites", [0, 2, 4, 6]) # Has to be even
    #GRAY = trial.suggest_categorical("GRAY", [True, False])
    #selection_type = trial.suggest_categorical("selection_type", ["rank", "tournament"])
    selection_type = "tournament"
    crossover_type = trial.suggest_categorical("crossover_type", ["one_point", "two_point", "uniform"])
    #mutation_rate = trial.suggest_float("mutation_rate", 0.001, 0.1)
    mutation_rate = 0.05
    if selection_type == "tournament":
        k = trial.suggest_int("k", 2, 6)
    else:
        k = None


    n=15
    max_cost = 10000*n  # Max evaluations
    num_runs = 5  # Number of runs per parameter configuration per function
    results = {1: [], 2: [], 3: []}  # Store results for each function

    for f in [1, 2, 3]:  # Loop through all functions (Rosenbrock, Salomon, Whitley)
        for _ in range(num_runs):
            population = np.random.randint(0, 2, (pop_size, n, BITS))
            cost = 0
            scores, cost = evaluate(f=f, population=population, cost=cost)
            best_x, best_score = population[i := np.argmin(scores)], scores[i]

            while cost < max_cost:
                if selection_type == "tournament":
                    selected, selected_scores = select(population, scores, selection_type, k, n_elites)
                elif selection_type == "rank":
                    selected, selected_scores = select(population, scores, selection_type)
                offspring = crossover(selected, crossover_type, p_crossover_max, p_crossover_min, cost, max_cost)
                offspring = mutate(offspring, mutation_rate)
                population = add_elites(population, offspring, scores, n_elites)
                scores, cost = evaluate(f=f, population=population, cost=cost)
                if scores[new_i := np.argmin(scores)] < best_score:
                    best_x, best_score = population[new_i], scores[new_i]

            results[f].append(best_score)

    # Calculate the mean score for each function
    mean_scores = {f: np.mean(results[f]) for f in results}

    # Return means as part of trial information
    trial.set_user_attr("rosenbrock_mean", mean_scores[1])
    trial.set_user_attr("salomon_mean", mean_scores[2])
    trial.set_user_attr("whitley_mean", mean_scores[3])

    # Return the overall average score across all functions
    overall_mean_score = np.mean(list(mean_scores.values()))
    return overall_mean_score


def main():

    database_url = "sqlite:///study.db?timeout=60"
    study = optuna.create_study(direction="minimize", storage=database_url, study_name="gray_test_1", load_if_exists=True)
    #study = optuna.create_study(direction="minimize")  # Minimize the cost or maximize the fitness
    study.optimize(objective, n_trials=None, n_jobs=7) # 2 used cores

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
