import numpy as np
from argparse import ArgumentParser
import random

BITS = 16
MAX_COST = 10000
NUM_RANGE_T1 = 3
NUM_RANGE_T2 = 32.768

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
    x = np.array(x)
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

def T_schedule_t1(T0=1, Tn=0.22, n = 99, k=0):
    #return T0 - k * (T0 - Tn) / n
    A = ((T0 - Tn)*(n+1)) / n
    return A/(k+1) + T0 - A

def T_schedule_t2(T0=1, Tn=0.22, n = 99, k=0):
    return T0 * (Tn / T0) ** (k / n)

def get_result(x, repre, num_range):
    return tuple(map(float, x) if repre == "real" else map(lambda xi: float(bin2real(xi, num_range)), x))

def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=10) # number of coordinates
    parser.add_argument("--m", type=int, default=1) # mutation probability
    parser.add_argument("--function", type=int, choices=[1, 2], default=1) # 1 for evaluate_t1, 2 for evaluate_t2
    parser.add_argument("--repr", type=str, choices=["real", "binary"], default="real") # binary or real
    args = parser.parse_args()

    evaluate = evaluate_t1 if args.function == 1 else evaluate_t2
    num_range = NUM_RANGE_T1 if args.function == 1 else NUM_RANGE_T2
    T_schedule = T_schedule_t1 if args.function == 1 else T_schedule_t2
    Tn = 0.1 if args.repr == "real" and args.function == 1 else 0.22
    evaluate.cost = 0

    with open("results1.txt", "w") as file:
        file.write("Coordinates, evaluation\n")
        if args.repr == "real":
            x = np.ones(args.n) * num_range
            cur_eval = evaluate(x)
        else:
            x = np.array([np.ones(BITS, dtype=int)] * args.n)  
            cur_eval = evaluate([bin2real(xi, num_range) for xi in x])
        file.write(f"{get_result(x, args.repr, num_range)} {cur_eval}\n")
        k = 0
        T = 1
        while evaluate.cost < MAX_COST:
            for _ in range(100):
                if args.repr == "real":
                    x_new = generate_neighbour_real(x, num_range)
                    new_eval = evaluate(x)
                else:
                    x_new = generate_neighbour_binary(x, args.m)
                    new_eval = evaluate([bin2real(xi, num_range) for xi in x])
                if new_eval < cur_eval:
                    x = x_new
                    cur_eval = new_eval
                else:
                    if np.random.uniform(0, 1) < np.exp((cur_eval - new_eval) / T):
                        x = x_new
                        cur_eval = new_eval
                file.write(f"{get_result(x, args.repr, num_range)} {cur_eval}\n")
            k += 1
            T = T_schedule(1, Tn, 99, k)

if __name__ == "__main__":
    main()