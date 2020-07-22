import nevergrad as ng
import argparse
import hpman
from hpman import L
import numpy as np
import typing as tp
from typing import Callable
import importlib


class NgMethod(object):
    """
    Get nevergrad parameter type from hpman.
    """

    value = None  # type: float
    """Elaborate value"""

    hint = None  # type: Dict[TODO]
    """Hints of the hyperparameter."""

    # XXX: (https://en.wikipedia.org/wiki/Passive_data_structure)
    bounds_kwargs = None  # type: Dict[TODO]
    """Save `set_bounds()` kwargs from hpman's hint."""
    mutation_kwargs = None
    """Save `set_mutation()` kwargs from hpman's hint."""
    casting_kwargs = None
    """Save `set_integer_casting()` kwargs from hpman's hint."""

    method = ng.p.Parameter()

    def __init__(self, value, hint):
        """
        :param value: float. Initial value of the hyparameter. 
        :param hint: Dict. Hints provided by user of this occurrence 
            of the hyperparameter.
        :bounds_kwargs: Dict. Save `set_bounds()` kwargs from hpman's hint.
        :mutation_kwargs: Dict. Save `set_mutation()` kwargs from hpman's hint.
        :casting_kwargs: Dict. Save `set_integer_casting()` kwargs from hpman's hint.
        :method: type of parameter in nevergrad.
        """
        self.value = value
        self.hint = hint
        self.bounds_kwargs = {}
        self.mutation_kwargs = {}
        self.casting_kwargs = {}
        self.method = ng.p.Parameter()

    def get_sets(self):
        """
        Save specific behaviors kwargs from hpman's hint to feed nevergrad 
            methods, including set_bounds, set_mutation, set_integer_casting.
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
        # feed nevergrad method set_bounds
        if len(self.bounds_kwargs) > 0:
            self.method.set_bounds(**self.bounds_kwargs)
        # feed nevergrad method set_mutation
        if len(self.mutation_kwargs) > 0:
            self.method.set_mutation(**self.mutation_kwargs)
        # feed nevergrad method set_integer_casting
        if len(self.casting_kwargs) > 0:
            self.method.set_integer_casting()

    def log_ng(self):
        """
        :return: nevergrad.p.Log. 
        """
        self.hint.pop('scale')
        self.hint['init'] = self.value
        self.get_sets()
        self.method = ng.p.Log(**self.hint)
        self.add_sets()
        return self.method

    def scalar_ng(self):
        """
        :return: nevergrad.p.Scalar.
        """
        self.hint['init'] = self.value
        self.get_sets()
        self.method = ng.p.Scalar(**self.hint)
        self.add_sets()
        return self.method

    def choice_ng(self):
        """
        :return: nevergrad.p.Choice. 
        """
        self.method = ng.p.Choice(**self.hint)
        return self.method

    def transition_choice_ng(self):
        """
        :return: nevergrad.p.TransitionChoice. 
        """
        self.hint.pop('transitions')
        self.method = ng.p.TransitionChoice(**self.hint)
        return self.method

    def array_ng(self):
        """
        :return: nevergrad.p.Array.  
        """
        self.hint['init'] = np.array(self.value)
        self.get_sets()
        self.method = ng.p.Array(**self.hint)
        self.add_sets()
        return self.method


def get_method_type(value, hint):
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
    # Hpman can not support ndarray type, we transfer list to ndarray.
    elif isinstance(value, list):
        method_type = 'array_ng'
    else:
        raise TypeError("type error")
    return method_type


def get_parametrization(hp_mgr: hpman.HyperParameterManager):
    """Define hyperparameters in nevergrad parametrization type.

    :param hp_mgr: The hyperparameter manager from `hpman`. It is
        usually an 'underscore' variable obtained by `from hpman.m import _`
    :return: ng.p.Instrumentation. Container of parameters available.
    """
    kw = {}
    for k, d in sorted(hp_mgr.db.group_by("name").items()):
        for i, oc in enumerate(
                d.select(L.exist_attr("filename")).sorted(
                    L.order_by("filename"))):
            if len(oc["hints"]) > 0:
                hint = oc["hints"]
                value = oc["value"]
                name = oc["name"]
                method_type = get_method_type(value, hint)
                method = getattr(NgMethod(value, hint), method_type)
                kw[name] = method()
    return ng.p.Instrumentation(**kw)





def get_objective_function(train: Callable[[], float],
                           hpm: hpman.HyperParameterManager):
    def objective_function(**kwargs):
        hpm.set_values(kwargs)
        return train()

    return objective_function


#hpng command line tool


def optimizer_warpper(optim_type: str, budget: int, param: ng.p.Instrumentation):
    optim = ng.optimizers.registry[optim_type](parametrization=param,
                                               budget=budget)
    return optim


def split_module(module):
    """
    split module string to file name and function name.

    :param module: A string of the command line `module.py:obj`.
    """
    parts = module.split(":", 1)
    if len(parts) != 2:
        raise ImportError("Failed to find attribute")
    f, obj = parts[0], parts[1]
    return f, obj


def import_func(module: str, obj: str):
    """
    parse the command line `module.py:obj` into the objective function

    :param module: A string of file name to parse.
    :param obj: A string of the function in module.
    """

    # make the module readable for importlib
    module = module.replace('/', '.')
    module = module.rsplit('.', 1)[0]

    mod = importlib.import_module(module)

    try:
        func = getattr(mod, obj)
    except:
        raise ImportError("Failed to find attribute %r in %r." % (obj, module))

    if func is None:
        raise ImportError("Failed to find application object: %r" % obj)

    if not callable(func):
        raise ImportError("Application object must be callable.")
    return func