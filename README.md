# hpnevergrad

A [nevergrad](https://github.com/facebookresearch/nevergrad/) extension for [hpman](https://github.com/megvii-research/hpman)

# Introduction
After using hpman to define the hyperparameters, call hpnevergrad through a single line of code to define hyperparameters in nevergrad parametrization type, to specify what are the parameters that the optimization should be performed upon.

# Example
```python
from hpman.m import _
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

>>>{'architecture': 'conv', 'batch_size': 1.4889356356452565, 'learning_rate': 0.0016799143450021905}
```


