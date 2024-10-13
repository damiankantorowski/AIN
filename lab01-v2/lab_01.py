import numpy as np
from argparse import ArgumentParser
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--low", type=float, default=0)
    parser.add_argument("--high", type=float, default=1)
    parser.add_argument("--mean", type=float, default=0)
    parser.add_argument("--std", type=float, default=1)
    parser.add_argument("--square_side_length", type=int, default=1)        # Kwadrat zdefiniowany przez wierzchołki (0,0) i (side_lenght, side_lenght)
    parser.add_argument("--radius", type=float, default=0.5)
    args = parser.parse_args()

    uniform = np.random.uniform(args.low, args.high, args.n)
    normal = np.random.normal(args.mean, args.std, args.n)
    np.savetxt("uniform.txt", uniform, fmt='%.6f')
    np.savetxt("normal.txt", normal, fmt='%.6f')

    i = 0
    j = 0
    points = []
    while i < args.n - 1:
        point = (uniform[i], uniform[i+1]) 
        points.append(point)
        i += 2
        j += 1

    center = (args.square_side_length / 2, args.square_side_length / 2)
    hits = 0

    for x,y in points:
        if args.radius > (pow(center[0] - x, 2) + pow(center[1] - y, 2))**0.5:
            hits += 1

    ratio = hits / len(points)
    square_area = args.square_side_length*args.square_side_length
    estimated_area = ratio*square_area
    real_area = np.pi*pow(args.radius, 2)
    metrics = ["Radius", "Side Length of Square", "Number of Points", "Hits Inside Circle", "Ratio", "Estimated Area", "Real Area"]
    values = [args.radius, args.square_side_length, len(points), hits, ratio, estimated_area, real_area]
    monte_carlo_data = np.array([metrics, values], dtype=object)
    np.savetxt("monte_carlo.txt", monte_carlo_data, fmt='%s', delimiter=', ')


if __name__ == "__main__":
    main()