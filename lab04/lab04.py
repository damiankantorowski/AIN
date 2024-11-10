import numpy as np
from argparse import ArgumentParser

BITS = 16
POP_SIZE = 20
MAX_COST = 10000

def bin2real(x, num_range):
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

#tournament selection
def select(population, scores, k=2):
    selected = []
    selected_scores = []
    for _ in range(POP_SIZE):
        indices = np.random.choice(POP_SIZE, k, replace=False)
        best = np.min(scores[indices])
        best_index = np.where(scores == best)[0][0]
        selected.append(population[best_index])
        selected_scores.append(best)
    return (np.array(selected), np.array(selected_scores))

def crossover(parents, scores, domain):
    offspring = parents.copy()
    if isinstance(parents[0][0][0], np.integer):
        #uniform recombination for binary representation
        for i in range(0, len(offspring), 2):
            for j in range(len(offspring[i])):
                for k in range(BITS):
                    if np.random.uniform() < 1 / BITS:
                        offspring[i][j][k], offspring[i + 1][j][k] = (
                            offspring[i + 1][j][k],
                            offspring[i][j][k],
                        )
    else:
        #weighted recombination for real representation
        parents = parents[np.argsort(scores)[::-1]]
        weights = np.array([np.log(POP_SIZE+1) - np.log(i+1) for i in range(POP_SIZE)])
        for i in range(POP_SIZE):
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
                mutated[i][j] ^= np.random.uniform(size=mutated[i][j].shape) < 1 / BITS
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

def get_result(x, num_range):
    if isinstance(x[0][0], np.integer):
        return tuple(map(lambda xi: float(bin2real(xi, num_range)), x))
    return tuple(map(float, x[0, :]))

def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--function", type=int, choices=[1, 2], default=1)
    parser.add_argument("--repr", type=str, choices=["real", "binary"], default="binary")
    args = parser.parse_args()

    evaluate = evaluate1 if args.function == 1 else evaluate2
    num_range = evaluate.domain[1]
    evaluate.cost = 0

    with open("results.txt", "w") as file:
        file.write("x, score\n") 
        if args.repr == "real":
            values = np.random.uniform(-num_range, num_range, (POP_SIZE, args.n))
            sigma = np.ones_like(values)
            population = np.stack((values, sigma), axis=1)
        else:
            population = np.random.randint(0, 2, (POP_SIZE, args.n, BITS))
        scores = np.array([evaluate(x) for x in population])
        x, score = population[i := np.argmin(scores)], scores[i]
        file.write(f"{get_result(x, num_range)}, {score}\n")
        while evaluate.cost < MAX_COST:
            selected, scores = select(population, scores)
            ofspring = crossover(selected, scores, evaluate.domain)
            population = mutate(ofspring, evaluate.domain)
            scores = np.array([evaluate(x) for x in population])
            if scores[new_i := np.argmin(scores)] < score:
                x, score = population[new_i], scores[new_i]
            file.write(f"{get_result(x, num_range)}, {score}\n")

if __name__ == "__main__":
    main()
