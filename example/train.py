import sys
import os

from hpman.m import _

o_path = os.getcwd()
sys.path.append(o_path)
from hpnevergrad import hpng
import nevergrad as ng


def train(learning_rate: float, batch_size: int, architecture: str) -> float:
    lr = _('learning_rate', 1e-3, range=[1e-3, 1.0], scale='log')
    bs = _('batch_size', 1, range=[1, 12])
    architecture = _('architecture', 'conv', choices=['conv', 'fc'])

    accuracy = (learning_rate - 0.2)**2 + (batch_size - 4)**2 + (
        0 if architecture == "conv" else 10)
    return accuracy


if __name__ == "__main__":
    _.parse_file(__file__)
    # define hyperparameters in nevergrad parametrization type
    parametrization = hpng.get_parametrization(_)
    # load hyperparameter in hpman to objective_function: a warpper
    objective_function = hpng.get_objective_function(train, _)
    optimizer = ng.optimizers.NGO(parametrization=parametrization, budget=100)
    recommendation = optimizer.minimize(objective_function)

    print(recommendation.kwargs)
