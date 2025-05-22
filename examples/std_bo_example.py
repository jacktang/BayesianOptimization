import os, sys
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()
basedir = os.path.realpath(f"{cwd}/../")
sys.path.append(basedir)

from bayes_opt import BayesianOptimization

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

# pbounds = {'x': (2, 4), 'y': (-3, 3)}

def test_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    """
    Test function for SAAS optimization.
    
    Only x1 and x2 are relevant, the rest are noise dimensions.
    This is a 10D version of the 2D function:
    f(x1, x2) = sin(x1) * cos(x2) - 0.5 * x1^2 - 0.5 * x2^2
    """
    # Only dimensions 1 and 2 are relevant
    return np.sin(x1) * np.cos(x2) - 0.5 * x1**2 - 0.5 * x2**2

pbounds = {'x1': (0, 1), 'x2': (0, 1), 'x3': (0, 1), 'x4': (0, 1), 'x5': (0, 1), 'x6': (0, 1), 'x7': (0, 1), 'x8': (0, 1), 'x9': (0, 1), 'x10': (0, 1)}

optimizer = BayesianOptimization(
    # f=black_box_function,
    f=test_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

# optimizer.maximize(
#     init_points=2,
#     n_iter=3,
# )

optimizer.maximize(
    init_points=5,
    n_iter=15,
)

print(optimizer.res)