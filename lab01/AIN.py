import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--low", type=float, default=0)
    parser.add_argument("--high", type=float, default=1)
    parser.add_argument("--mean", type=float, default=0)
    parser.add_argument("--std", type=float, default=1)
    args = parser.parse_args()
    uniform = np.random.uniform(args.low, args.high, args.n)
    normal = np.random.normal(args.mean, args.std, args.n)
    np.savetxt("uniform.txt", uniform, fmt='%.6f')
    np.savetxt("normal.txt", normal, fmt='%.6f')

if __name__ == "__main__":
    main()
