import nevergrad as ng
import numpy as np


print("Optimization of continuous hyperparameters =========")


def train_and_return_test_error(x):
    return np.linalg.norm([int(50. * abs(x_ - 0.2)) for x_ in x])

parametrization = ng.p.Array(shape=(300,))  # optimize on R^300

budget = 1200  # How many trainings we will do before concluding.

names = ["RandomSearch", "TwoPointsDE", "CMA", "PSO", "ScrHammersleySearch"]

for name in names:
    optim = ng.optimizers.registry[name](parametrization=parametrization, budget=budget)
    for u in range(budget // 3):
        x1 = optim.ask()
        # Ask and tell can be asynchronous.
        # Just be careful that you "tell" something that was asked.
        # Here we ask 3 times and tell 3 times in order to fake asynchronicity
        x2 = optim.ask()
        x3 = optim.ask()
        # The three folowing lines could be parallelized.
        # We could also do things asynchronously, i.e. do one more ask
        # as soon as a training is over.
        y1 = train_and_return_test_error(*x1.args, **x1.kwargs)  # here we only defined an arg, so we could omit kwargs
        y2 = train_and_return_test_error(*x2.args, **x2.kwargs)  # (keeping it here for the sake of consistency)
        y3 = train_and_return_test_error(*x3.args, **x3.kwargs)
        optim.tell(x1, y1)
        optim.tell(x2, y2)
        optim.tell(x3, y3)
    recommendation = optim.recommend()
    print("* ", name, " provides a vector of parameters with test error ",
          train_and_return_test_error(*recommendation.args, **recommendation.kwargs))