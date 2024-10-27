import numpy as np
from argparse import ArgumentParser
import random

BITS = 16
MAX_COST = 10000
NUM_RANGE_t1 = 3
NUM_RANGE_t2 = 32.768

def bin2int(x):
    return sum(2 ** i * x[BITS - i - 1] for i in range(BITS))

def bin2real(x, num_range):
    return 2 * num_range / 2 ** BITS * bin2int(x) - num_range

def evaluate_t1(x):
    evaluate_t1.cost += 1

    sum_squares = np.sum(np.square(x))

    expr1 = 5 / (1 + sum_squares)
    #cotangent = 1 / np.tan(np.exp(-expr1) + 1e-10)  # epsilon to avoid division by zero in cotangent
    cotangent = 1 / np.tan(np.exp(-expr1))

    return -expr1 + np.sin(cotangent)

def evaluate_t2(x, a=20, b=0.2, c=2 * np.pi):
    evaluate_t2.cost += 1
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    return sum_sq_term + cos_term + a + np.exp(1)

def generate_neighbour_binary(x, m):
    x_new = x.copy()
    for i in range(len(x)):
        for j in range(BITS):
            if np.random.uniform() < m / BITS:
                x_new[i][j] = int(not x_new[i][j])
    return x_new

def generate_neighbour_real(x, num_range):
    x_new = x.copy()
    for i in range(len(x)):
        while True:
            x_candidate = x[i] + random.gauss(0, 1)
            if -num_range <= x_candidate <= num_range:
                    x_new[i] = x_candidate
                    break
    return x_new

def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=2) # number of coordinates
    parser.add_argument("--m", type=int, default=1) # mutation probability
    parser.add_argument("--test_function", type=int, choices=[1, 2], default=1) # 1 for evaluate_t1, 2 for evaluate_t2
    parser.add_argument("--number_representation", type=str, choices=["binary", "real"], default="real") # binary or real
    args = parser.parse_args()

    evaluate_t1.cost = 0
    evaluate_t2.cost = 0

    with open("results1.txt", "w") as file:
        file.write("Coordinates, evaluation\n")

        # Binary representation and evaluate_t1
        if args.number_representation == "binary" and args.test_function == 1:
            x = np.array([np.ones(BITS, dtype=int)] * args.n)

            cur_eval = evaluate_t1([bin2real(xi, NUM_RANGE_t1) for xi in x])
            real_values = tuple(map(lambda xi: float(bin2real(xi, NUM_RANGE_t1)), x))
            file.write(f"{real_values} {cur_eval}\n")

            while evaluate_t1.cost < MAX_COST:
                x_new = generate_neighbour_binary(x, args.m)
                new_eval = evaluate_t1([bin2real(xi, NUM_RANGE_t1) for xi in x_new])

                if new_eval < cur_eval:
                    x = x_new
                    cur_eval = new_eval

                real_values = tuple(map(lambda xi: float(bin2real(xi, NUM_RANGE_t1)), x))
                file.write(f"{real_values} {cur_eval}\n")
     # Binary representation and evaluate_t2
        elif args.number_representation == "binary" and args.test_function == 2:
            x = np.array([np.ones(BITS, dtype=int)] * args.n)

            cur_eval = evaluate_t2([bin2real(xi, NUM_RANGE_t2) for xi in x])
            real_values = tuple(map(lambda xi: bin2real(xi, NUM_RANGE_t2), x))
            file.write(f"{real_values} {cur_eval}\n")

            while evaluate_t2.cost < MAX_COST:
                x_new = generate_neighbour_binary(x, args.m)
                new_eval = evaluate_t2([bin2real(xi, NUM_RANGE_t2) for xi in x_new])

                if new_eval < cur_eval:
                    x = x_new
                    cur_eval = new_eval

                real_values = tuple(map(lambda xi: float(bin2real(xi, NUM_RANGE_t1)), x))
                file.write(f"{real_values} {cur_eval}\n")

        # Real representation and evaluate_t1
        elif args.number_representation == "real" and args.test_function == 1:
            x = np.ones(args.n) * NUM_RANGE_t1

            cur_eval = evaluate_t1(x)
            real_values = tuple(map(float, x))
            file.write(f"{real_values} {cur_eval}\n")

            while evaluate_t1.cost < MAX_COST:
                x_new = generate_neighbour_real(x, NUM_RANGE_t1)
                new_eval = evaluate_t1(x_new)

                if new_eval < cur_eval:
                    x = x_new
                    cur_eval = new_eval

                real_values = tuple(map(float, x))
                file.write(f"{real_values} {cur_eval}\n")

        # Real representation and evaluate_t2
        elif args.number_representation == "real" and args.test_function == 2:
            x = np.ones(args.n) * NUM_RANGE_t2

            cur_eval = evaluate_t2(x)
            real_values = tuple(map(float, x))
            file.write(f"{real_values} {cur_eval}\n")

            while evaluate_t2.cost < MAX_COST:
                x_new = generate_neighbour_real(x, NUM_RANGE_t2)
                new_eval = evaluate_t2(x_new)

                if new_eval < cur_eval:
                    x = x_new
                    cur_eval = new_eval

                real_values = tuple(map(float, x))
                file.write(f"{real_values} {cur_eval}\n")




if __name__ == "__main__":
    main()