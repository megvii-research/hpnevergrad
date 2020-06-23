import pytest
import hpman

import argparse

import nevergrad as ng

import os
import sys

o_path = os.getcwd()
sys.path.append(o_path)
import hpnevergrad


class Test(object):
    def _make_basic(self):
        #fpath = str(fpath)
        _ = hpman.HyperParameterManager("_")
        parser = argparse.ArgumentParser()
        _.parse_file(__file__)
        return parser, _

    def test_array_basic(self) -> None:
        parser, _ = self._make_basic()
        arr1 = _('arr1', ((1, 2), (3, 4)), mutable_sigma=True)
        parametrization = hpnevergrad.hpng(_)
        assert 'arr1' in parametrization.kwargs.keys()
        arr2 = _('arr2', (300, ),
                 method='constraint',
                 set_integer_casting=True,
                 exponent=3,
                 custom='cauchy')
        parametrization = hpnevergrad.hpng(_)
        assert 'arr2' in parametrization.kwargs.keys()

    def test_log(self) -> None:
        parser, _ = self._make_basic()
        log1 = _('log1', 0.02, range=[1e-3, 1.0], scale='log', exponent=3.0)
        parametrization = hpnevergrad.hpng(_)
        assert 'log1' in parametrization.kwargs.keys()
        log2 = _('log2',
                 0.02,
                 range=[1e-3, 1.0],
                 scale='log',
                 exponent=3.0,
                 sigma=3,
                 custom='gaussian')
        parametrization = hpnevergrad.hpng(_)
        assert 'log2' in parametrization.kwargs.keys()

    def test_choice_repetitions(self) -> None:
        parser, _ = self._make_basic()
        choice = _('choice', 0, choices=[0, 1, 2, 3], repetitions=2)
        parametrization = hpnevergrad.hpng(_)
        assert 'choice' in parametrization.kwargs.keys()

    def test_ordered_choice(self) -> None:
        parser, _ = self._make_basic()
        transition_choice = _('transition_choice',
                              2,
                              choices=[0, 1, 2, 3],
                              transitions=[-1000000, 10])
        parametrization = hpnevergrad.hpng(_)
        assert 'transition_choice' in parametrization.kwargs.keys()
