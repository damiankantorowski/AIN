import numpy as np
from argparse import ArgumentParser

BITS = 16
POP_SIZE = 20 # Has to be odd
MAX_COST = 10000
ELITE_SIZE= 2
MUTATION_RATE = 1 / BITS
GRAY = True

def bin2real(x, num_range):

    def gray_to_binary(gray):
        binary = gray.copy()
        for i in range(1, len(gray)):
            binary[i] = binary[i - 1] ^ gray[i]
        return binary
    
    # Decode Gray code to binary
    if(GRAY == True):
        x = gray_to_binary(x)  

    return 2 * num_range / 2 ** BITS * int("".join(x.astype(str)), 2) - num_range

def evaluate1(x):
    evaluate1.cost += 1
    if isinstance(x[0][0], np.integer):
        x = np.array([bin2real(xi, 3) for xi in x])
    else:
        x = x[0, :]
    sum_squares = np.sum(np.square(x))
    expr1 = 5 / (1 + sum_squares)
    cotangent = 1 / np.tan(np.exp(-expr1))
    return -expr1 + np.sin(cotangent)
evaluate1.domain = (-3, 3)

def evaluate2(x, a=20, b=0.2, c=2*np.pi):
    evaluate2.cost += 1
    if isinstance(x[0][0], np.integer):
        x = np.array([bin2real(xi, 32.768) for xi in x])
    else:
        x = x[0, :]
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(x)) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum_sq_term + cos_term + a + np.exp(1)
evaluate2.domain = (-32.768, 32.768)

# ? implement simulated annealing? elite_size would increase with time
def select(population, scores, k=2, elite_size=ELITE_SIZE):
    
    elite_indices = np.argsort(scores)[:elite_size]
    elites = [population[i] for i in elite_indices]  # Store elite individuals
    elite_scores = [scores[i] for i in elite_indices]  # Store elite scores

    # ---------------------------- RANK SELECTION ----------------------------
    # !
    # selected, selected_scores = [], []
    # sorted_indices = np.argsort(scores)[::-1]  # Sort in descending order

    # a = 1 / (POP_SIZE*2)
    # d = ((2 * 1) / POP_SIZE - (2 * a)) / (POP_SIZE - 1)
    # selection_probabilities = np.zeros(POP_SIZE)  # Initialize selection probabilities
    # for i in range(POP_SIZE):
    #     cur_index = sorted_indices[i]
    #     selection_probabilities[cur_index] = a + i*d
    
    # # Select individuals based on ranking probabilities until we reach POP_SIZE
    # while len(selected) < (POP_SIZE - ELITE_SIZE):
    #     chosen_index = np.random.choice(POP_SIZE, p=selection_probabilities)  # Rank-based selection
    #     selected.append(population[chosen_index])
    #     selected_scores.append(scores[chosen_index])

    # ---------------------------- TOURNAMENT SELECTION ----------------------------

    selected, selected_scores = [], []

    while len(selected) < POP_SIZE - elite_size:
        indices = np.random.choice(POP_SIZE, k, replace=False)  # Select `k` random individuals
        best = np.min(scores[indices])  # Find the best score in this subset
        best_index = np.where(scores == best)[0][0]  # Get index of this best score
        selected.append(population[best_index])  # Add best individual to selected list
        selected_scores.append(best)  # Add best score to selected scores

    return (np.array(selected), np.array(selected_scores)), (np.array(elites), np.array(elite_scores))

def crossover(parents, scores, domain):
    offspring = parents.copy()
    if isinstance(parents[0][0][0], np.integer):

        # ---------------------------- UNIFORM RECOMBINATION FOR BINARY REPRESENTATION ----------------------------
        
        # for i in range(0, len(offspring), 2):
        #     for j in range(len(offspring[i])):
        #         for k in range(BITS):
        #             if np.random.uniform() < 1 / BITS:
        #                 offspring[i][j][k], offspring[i + 1][j][k] = (
        #                     offspring[i + 1][j][k],
        #                     offspring[i][j][k],
        #                 )

        # ---------------------------- ONE-POINT CROSSOVER FOR BINARY REPRESENTATION ----------------------------
         
        # for i in range(0, len(offspring), 2):
        #     for j in range(len(offspring[i])):
        #         # Choose a random crossover point in the middle of the bit sequence
        #         crossover_point = BITS // 2
                
        #         # Create children by swapping bits at the crossover point
        #         offspring[i][j][:crossover_point] = parents[i][j][:crossover_point]
        #         offspring[i][j][crossover_point:] = parents[i + 1][j][crossover_point:]
                
        #         offspring[i + 1][j][:crossover_point] = parents[i + 1][j][:crossover_point]
        #         offspring[i + 1][j][crossover_point:] = parents[i][j][crossover_point:]

        # ---------------------------- TWO-POINT CROSSOVER FOR BINARY REPRESENTATION ----------------------------

        for i in range(0, len(offspring), 2):
            for j in range(len(offspring[i])):
                # Choose two random crossover points
                point1, point2 = sorted(np.random.choice(BITS, 2, replace=False))
                
                # Create children by swapping bits between the two points
                offspring[i][j][:point1] = parents[i][j][:point1]
                offspring[i][j][point1:point2] = parents[i + 1][j][point1:point2]
                offspring[i][j][point2:] = parents[i][j][point2:]
                
                offspring[i + 1][j][:point1] = parents[i + 1][j][:point1]
                offspring[i + 1][j][point1:point2] = parents[i][j][point1:point2]
                offspring[i + 1][j][point2:] = parents[i + 1][j][point2:]

    else:
        #weighted recombination for real representation
        parents = parents[np.argsort(scores)[::-1]]
        weights = np.array([np.log(POP_SIZE - ELITE_SIZE + 1) - np.log(i+1) for i in range(POP_SIZE - ELITE_SIZE)])
        for i in range(POP_SIZE - ELITE_SIZE):
            n = len(parents[i][0])
            for j in range(n):
                sigma = (
                    np.sum(parents[:, 1, j] * weights)
                    / np.sum(weights)
                    * np.exp(
                        np.random.normal(0, 1) / np.sqrt(2 * n)
                        + np.random.normal(0, 1) / np.sqrt(2 * np.sqrt(n))
                    )
                )
                offspring[i][1][j] = np.clip(sigma, 0, 1)
                xi = np.sum(parents[:, 0, j] * weights) / np.sum(weights) 
                + offspring[i][1][j] * np.random.normal(0, 1)
                offspring[i][0][j] = np.clip(xi, domain[0], domain[1])
    return offspring

def mutate(population, domain):
    mutated = population.copy()
    if isinstance(mutated[0][0][0], np.integer):
        #bit flip mutation for binary representation
        for i in range(len(mutated)):
            for j in range(len(mutated[i])):
                mutated[i][j] ^= np.random.uniform(size=mutated[i][j].shape) < MUTATION_RATE

    else:
        #adaptive mutation for real representation
        for i in range(len(mutated)):
            n = len(mutated[i][0])
            for j in range(n):
                sigma = mutated[i][1][j] * np.exp(
                    np.random.normal(0, 1) / np.sqrt(2*n)
                    + np.random.normal(0, 1) / np.sqrt(2*np.sqrt(n)))
                mutated[i][1][j] = np.clip(sigma, 0, 1)
                xi = mutated[i][0][j] + np.random.normal(0, mutated[i][1][j])
                mutated[i][0][j] = np.clip(xi, domain[0], domain[1])
    return mutated

# coordinates change to printable real representation
def get_result(x, num_range):
    if isinstance(x[0][0], np.integer):
        return tuple(map(lambda xi: float(bin2real(xi, num_range)), x))
    return tuple(map(float, x[0, :]))

def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--function", type=int, choices=[1, 2], default=2)
    parser.add_argument("--repr", type=str, choices=["real", "binary"], default="real")
    args = parser.parse_args()

    evaluate = evaluate1 if args.function == 1 else evaluate2
    num_range = evaluate.domain[1]

    for i in range(1, 11):
        evaluate.cost = 0
        filename = f"results_{i}.txt"
        with open(filename, "w") as file:
            file.write("x, score\n") 
            if args.repr == "real":
                values = np.random.uniform(-num_range, num_range, (POP_SIZE, args.n))
                sigma = np.ones_like(values)
                population = np.stack((values, sigma), axis=1)
            else:
                population = np.random.randint(0, 2, (POP_SIZE, args.n, BITS))
            scores = np.array([evaluate(x) for x in population])
     
            best_parameters, best_score = population[i := np.argmin(scores)], scores[i]
            file.write(f"{get_result(best_parameters, num_range)}, {best_score}\n")
            while evaluate.cost < MAX_COST:

                (selected, scores), (elites, elite_scores) = select(population, scores)

                offspring = crossover(selected, scores, evaluate.domain)
                offspring = mutate(offspring, evaluate.domain)
                population = np.vstack((elites, offspring))

                scores = np.array([evaluate(x) for x in population])

                if scores[new_i := np.argmin(scores)] < best_score:  
                    best_parameters, best_score = population[new_i], scores[new_i]
                file.write(f"{get_result(best_parameters, num_range)}, {best_score}\n")
            print(f"{best_score}")            
if __name__ == "__main__":
    main()
