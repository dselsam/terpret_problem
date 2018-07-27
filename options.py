import argparse

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', action='store', dest='k', type=int, default=32, help="Length of parity chain")
    parser.add_argument('--v', action='store', dest='v', type=int, default=32, help="Number of marginals to print")
    parser.add_argument('--alpha', action='store', dest='alpha', type=float, default=1.0, help="Dirichlet hyperparameter")
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--n_epochs', action='store', dest='n_epochs', type=int, default=100000, help="Number of epochs")
    parser.add_argument('--max_ent', action='store', dest='max_ent', type=int, default=0, help="Maximum entropy initialization")
    parser.add_argument('--seed', action='store', dest='seed', type=int, default=0, help="Random seed")
    opts = parser.parse_args()
    return opts
