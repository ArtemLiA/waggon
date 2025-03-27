import argparse
import waggon.functions as f
from waggon.optim import PyXABOptimizer


FUNCS = {
    'ackley':     f.ackley,
    'himmelblau': f.himmelblau,
    'holder':     f.holder,
    'levi':       f.levi,
    'rosenbrock': f.rosenbrock,
    'tang':       f.tang,
    'thc':        f.three_hump_camel
}


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--function', 
        help='Function to optimize', 
        default='thc',
        choices=[
            'thc', 
            'ackley', 
            'levi', 
            'himmelblau', 
            'rosenbrock', 
            'tang', 
            'holder'
        ]
    )
    parser.add_argument(
        '-a', '--algorithm',
        help='Multi-armed (X-armed) bandit optimization algorithm',
        default='T-HOO',
        choices=[
            "Zooming",
            "T-HOO",
            "DOO",
            "SOO",
            "StoSOO",
            "HCT",
            "POO",
            "GPO",
            "PCT",
            "SequOOL",
            "StroquOOL", 
            "VROOM",
            "VHCT",
            "VPCT",
        ],
        type=str
    )
    parser.add_argument(
        '-p', '--partition',
        help='Partition method',
        default='binary',
        choices=[
            'binary',
            'rand-binary',
            'dim-binary',
            'kary',
            'rand-kary',
        ],
        type=str
    )
    parser.add_argument(
        '-d', '--dimensions',
        help='Dimensionality of the experiment',
        type=int,
        default=None
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Increase output verbose',
        choices=[0, 1, 2],
        default=1,
        type=int
    )
    parser.add_argument(
        '-e', '--experiments', 
        help="Number of experiments", 
        default=5, 
        type=int
    )
    parser.add_argument(
        '-eps', '--epsilon', 
        help="Required error rate", 
        default=1e-1, 
        type=float
    )
    parser.add_argument(
        '--max_iter',
        help="Maximum number of iterations",
        default=1000,
        type=int
    )

    args = parser.parse_args()

    for i in range(args.experiments):
        print(f"Experiment #{i}")

        func = FUNCS[args.function]
        dim = args.dimensions

        if dim is not None:
            func = func(dim)
        else:
            func = func()
        
        algo = args.algorithm
        partition = args.partition
        eps = args.epsilon

        bandit = PyXABOptimizer(
            func=func, algo=algo, partition=partition, eps=eps, max_iter=args.max_iter
        )
        bandit.optimise()

        print(f"Iterations: {bandit.res.size}")
        print(f"Minimum: {bandit.params[-1]}")
        print(f"Func min: {func.glob_min}")
        print(f"Error: {bandit.errors[-1]}")
        print()

if __name__ == "__main__":
    test()

