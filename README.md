# hpnevergrad

A [nevergrad](https://github.com/facebookresearch/nevergrad/) extension for [hpman](https://github.com/megvii-research/hpman)

# Introduction
After using hpman to define the hyperparameters, call hpargparse through a single line of code to start new experiments, analyze the results and adjust the parameters, and find the best combination of hyperparameters.

# Example
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

    parametrization = hpnevergrad.hpng(_)
    optim = ng.optimizers.OnePlusOne(parametrization=parametrization,
                                        budget=100)
                                        
    recommendation = optim.minimize(fake_training)
    print(recommendation.kwargs)
>>>{'architecture': 'conv', 'batch_size': 4.000222753115852, 'learning_rate': 0.21394340980606086}
```


# TODO
User can assign an optimization type in hpman. eg:
```python
lr = _('learning_rate', 1e-3, range=[1e-3, 1.0], scale='log',optimizer='RandomSearch')
parametrization = hpnevergrad.hpng(_)
recommendation = optim.minimize(fake_training)
```
is equivalent to

```python
lr = _('learning_rate', 1e-3, range=[1e-3, 1.0], scale='log',optimizer='RandomSearch')
parametrization = hpnevergrad.hpng(_)
recommendation = optim.minimize(fake_training)
optim = ng.optimizers.registry['RandomSearch'](parametrization=parametrization, budget=budget)
```