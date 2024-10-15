import numpy as np
from argparse import ArgumentParser

BITS = 16
MAX_COST = 10000
NUM_RANGE = 10

def bin2int(x):
    return sum(2 ** i * x[BITS - i - 1] for i in range(BITS))

def bin2real(x):
    return 2 * NUM_RANGE / 2 ** BITS * bin2int(x) - NUM_RANGE

def evaluate(x):
    evaluate.cost += 1
    return sum([x ** 2 for x in map(bin2real, x)])
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, choices=[1, 4], default=1)
    args = parser.parse_args()
    x = np.array([np.ones(BITS, dtype=int)] * args.n) #x0 = [10, ..., 10]
    evaluate.cost = 0
    while evaluate.cost < MAX_COST:
        x_new = x.copy()
        for i in range(args.n):
            for j in range(BITS):
                if np.random.uniform() < args.m / BITS:
                    x_new[i][j] = int(not x_new[i][j])
        if evaluate(x_new) < evaluate(x):
            x = x_new
            print(tuple(map(float, map(bin2real, x))))

if __name__ == "__main__":
    main()

