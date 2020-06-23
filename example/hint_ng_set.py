import sys
import os

from hpman.hpm_db import L
from hpman.m import _

import argparse
o_path = os.getcwd()
sys.path.append(o_path)
import hpnevergrad
import nevergrad as ng


def fake_training(learning_rate: float, batch_size: int,
                  architecture: str) -> float:
    # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
    return (learning_rate - 0.2)**2 + (batch_size - 4)**2 + (
        0 if architecture == "conv" else 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # `lr.set_mutation(custom='gaussian')` can finished simply in hpman hyperparameter declaration.
    lr = _('learning_rate',
           1e-3,
           range=[1e-3, 1.0],
           scale='log',
           custom='gaussian')
    # `bs.set_bounds(method='clipping')` can finished simply in hpman hyperparameter declaration.
    bs = _('batch_size', 1, range=[1, 12], method='clipping')

    arch = _('architecture', 'conv', choices=['conv', 'fc'])
    _.parse_file(__file__)

    # define hyperparameters in nevergrad parametrization type
    parametrization = hpnevergrad.hpng(_)
    optim = ng.optimizers.OnePlusOne(parametrization=parametrization,
                                     budget=100)

    recommendation = optim.minimize(fake_training)
    print(recommendation.kwargs)
