import nevergrad as ng
import argparse
import hpman
from hpman import (HyperParameterManager, HyperParameterOccurrence, L)


class NgMethod(object):
    """
    Get hyperparameter nevergrad parameter type.
    """
    def __init__(self, value, hint):
        """
        :param value: float. Initial value of the variable. 
        :param hint: Dict. Hints provided by user of this occurrence of the hyperparameter.
        """
        self.value = value
        self.hint = hint

    def log_ng(self):
        """
        :return: nevergrad.p.Log. Parameter representing a positive variable, 
        mutated by Gaussian mutation in log-scale.
        """
        self.hint.pop('scale')
        if 'range' in self.hint.keys():
            self.hint['lower'], self.hint['upper'] = self.hint.pop('range')
        self.hint['init'] = self.value

        return ng.p.Log(**self.hint)

    def scalar_ng(self):
        """
        :return: nevergrad.p.Scalar. Parameter representing a scalar. 
        """
        if 'range' in self.hint.keys():
            self.hint['lower'], self.hint['upper'] = self.hint.pop('range')
        self.hint['init'] = self.value
        return ng.p.Scalar(**self.hint)

    def choice_ng(self):
        """
        :return: nevergrad.p.Choice. Unordered categorical parameter, 
            randomly choosing one of the provided choice options as a value. 
        """
        return ng.p.Choice(**self.hint)

    def transition_choice_ng(self):
        """
        :return: nevergrad.p.TransitionChoice. Ordered categorical parameter, 
            choosing one of the provided choice options as a value, with continuous transitions.
        """
        return ng.p.TransitionChoice(**self.hint)

    def array_ng(self):
        """
        :return: nevergrad.p.Array.  
        """
        self.hint['init'] = self.value
        return ng.p.Array(**self.hint)


def get_method(value, hint):
    if 'choices' in hint.keys():
        if 'transitions' in hint.keys():
            method_type = 'transition_choice_ng'
        else:
            method_type = 'choice_ng'
    elif isinstance(value, (float, int)):
        if 'scale' in hint.keys() and hint['scale'] == 'log':
            method_type = 'log_ng'
        else:
            method_type = 'scalar_ng'
    else:
        method_type = 'array_ng'
    return method_type


def hpng(hp_mgr: hpman.HyperParameterManager):
    """Bridging the gap between nevergrad and hpman.

    :param hp_mgr: The hyperparameter manager from `hpman`. It is
        usually an 'underscore' variable obtained by `from hpman.m import _`
    :return: ng.p.Instrumentation. Container of parameters available.
    """
    kw = {}
    for k, d in sorted(hp_mgr.db.group_by("name").items()):
        for i, oc in enumerate(
                d.select(L.exist_attr("filename")).sorted(
                    L.order_by("filename"))):
            if oc["hints"] is not None:
                hint = oc["hints"]
                value = oc["value"]
                name = oc["name"]
                method_type = get_method(value, hint)
                method = getattr(NgMethod(value, hint), method_type)
                kw[name] = method()
    return ng.p.Instrumentation(**kw)
