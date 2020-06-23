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
        self.bounds_kwargs = {}
        self.mutation_kwargs = {}
        self.casting_kwargs = {}
        self.method = ng.p.Parameter()

    def get_sets(self):
        """
        Save specific behaviors kwargs from hpman's hint to feed nevergrad methods, including set_bounds, set_mutation, set_integer_casting.
        """
        # Save set_bounds kwargs from hpman's hint.
        if 'range' in self.hint.keys():
            self.bounds_kwargs['lower'], self.bounds_kwargs[
                'upper'] = self.hint.pop('range')
        if 'method' in self.hint.keys():
            self.bounds_kwargs['method'] = self.hint.pop('method')
        if 'full_range_sampling' in self.hint.keys():
            self.bounds_kwargs['full_range_sampling'] = self.hint.pop(
                'full_range_sampling')

        # Save set_mutation kwargs from hpman's hint.
        if 'sigma' in self.hint.keys():
            self.mutation_kwargs['sigma'] = self.hint.pop('sigma')
        if 'exponent' in self.hint.keys():
            self.mutation_kwargs['exponent'] = self.hint.pop('exponent')
        if 'custom' in self.hint.keys():
            self.mutation_kwargs['custom'] = self.hint.pop('custom')

        # Save set_integer_casting kwargs from hpman's hint.
        if 'set_integer_casting' in self.hint.keys(
        ) and self.hint['set_integer_casting'] == True:
            self.casting_kwargs['set_integer_casting'] = self.hint.pop(
                'set_integer_casting')

    def add_sets(self):
        """
        Feed specific behaviors kwargs to nevergrad methods.
        """
        # hint nevergrad method set_bounds
        if len(self.bounds_kwargs) > 0:
            self.method.set_bounds(**self.bounds_kwargs)
        # hint nevergrad method set_mutation
        if len(self.mutation_kwargs) > 0:
            self.method.set_mutation(**self.mutation_kwargs)
        # hint nevergrad method set_integer_casting
        if len(self.casting_kwargs) > 0:
            self.method.set_integer_casting()

    def log_ng(self):
        """
        :return: nevergrad.p.Log. Parameter representing a positive variable, 
        mutated by Gaussian mutation in log-scale.
        """
        self.hint.pop('scale')
        self.hint['init'] = self.value
        self.get_sets()
        self.method = ng.p.Log(**self.hint)
        self.add_sets()
        return self.method

    def scalar_ng(self):
        """
        :return: nevergrad.p.Scalar. Parameter representing a scalar. 
        """
        self.hint['init'] = self.value
        self.get_sets()
        self.method = ng.p.Scalar(**self.hint)
        self.add_sets()
        return self.method

    def choice_ng(self):
        """
        :return: nevergrad.p.Choice. Unordered categorical parameter, 
            randomly choosing one of the provided choice options as a value. 
        """
        self.method = ng.p.Choice(**self.hint)
        return self.method

    def transition_choice_ng(self):
        """
        :return: nevergrad.p.TransitionChoice. Ordered categorical parameter, 
            choosing one of the provided choice options as a value, with continuous transitions.
        """
        self.hint.pop('transitions')
        self.method = ng.p.TransitionChoice(**self.hint)
        return self.method

    def array_ng(self):
        """
        :return: nevergrad.p.Array.  
        """
        self.hint['init'] = self.value
        self.get_sets()
        self.method = ng.p.Array(**self.hint)
        self.add_sets()
        return self.method


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
