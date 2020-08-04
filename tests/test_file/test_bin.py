import nevergrad as ng
from hpman.m import _

from hpnevergrad import hpng


def train() -> float:
    lr = _('lr', 1e-3, range=[1e-3, 1.0], scale='log')
    bs = _('bs', 1, range=[1, 12])
    architecture = _('architecture', 'conv', choices=['conv', 'fc'])
    accuracy = (lr - 0.2)**2 + (bs -
                                4)**2 + (0 if architecture == "conv" else 10)
    return accuracy
