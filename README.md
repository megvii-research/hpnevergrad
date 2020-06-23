# hpnevergrad

A [nevergrad](https://github.com/facebookresearch/nevergrad/) extension for [hpman](https://github.com/megvii-research/hpman)

# Introduction
After using hpman to define the hyperparameters, call hpnevergrad through a single line of code to define hyperparameters in nevergrad parametrization type, to specify what are the parameters that the optimization should be performed upon.

# Example
the example is modified based on [nevergrad's example](https://github.com/facebookresearch/nevergrad/blob/master/README.md).
```python
from hpman.m import _
import argparse
import hpnevergrad
import nevergrad as ng

def fake_training(learning_rate: float, batch_size: int,
                  architecture: str) -> float:
    # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
    return (learning_rate - 0.2)**2 + (batch_size - 4)**2 + (
        0 if architecture == "conv" else 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    lr = _('learning_rate', 1e-3, range=[1e-3, 1.0], scale='log')
    bs = _('batch_size', 1, range=[1, 12])  
    arch = _('architecture', 'conv', choices=['conv', 'fc'])
    _.parse_file(__file__)
    # define hyperparameters in nevergrad parametrization type
    parametrization = hpnevergrad.hpng(_)
    optim = ng.optimizers.OnePlusOne(parametrization=parametrization,
                                        budget=100)
                                        
    recommendation = optim.minimize(fake_training)
    print(recommendation.kwargs)
>>>{'architecture': 'conv', 'batch_size': 4.000222753115852, 'learning_rate': 0.21394340980606086}
```


# Hint the behaviors of the nevergrad
Hint the [behaviors of the nevergrad](https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.Array), including set_bounds, set_mutation, set_integer_casting. These behaviors can be finished simply in hpman hyperparameter declaration.
```python
# `lr.set_mutation(custom='gaussian')` can finished simply in hpman hyperparameter declaration.
lr = _('learning_rate', 1e-3, range=[1e-3, 1.0], scale='log',custom='gaussian')
# `bs.set_bounds(method='clipping')` can finished simply in hpman hyperparameter declaration.
bs = _('batch_size', 1, range=[1, 12],method='clipping')  
# define hyperparameters in nevergrad parametrization type
parametrization = hpnevergrad.hpng(_)
```
is equivalent to
```python
lr=ng.p.Log(lower=0.001, upper=1.0)
bs=ng.p.Scalar(init=1,lower=1, upper=12)

lr.set_mutation(custom='gaussian')
bs.set_bounds(method='clipping')

parametrization = ng.p.Instrumentation(
    learning_rate=lr,
    batch_size=bs
)
```