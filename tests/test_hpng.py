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
        arr = _('arr',((1,2),(3,4)),mutable_sigma=True)
        parametrization = hpnevergrad.hpng(_)
        assert 'arr' in parametrization.kwargs.keys()

    def test_log(self) -> None:
        parser, _ = self._make_basic()
        log = _('log',0.02, range=[1e-3, 1.0], scale='log',exponent=3.0,optimizer='balbala')
        parametrization = hpnevergrad.hpng(_)
        assert 'log' in parametrization.kwargs.keys()


    def test_choice_repetitions(self) -> None:
        parser, _ = self._make_basic()
        choice = _('choice',0, choices=[0, 1, 2, 3],repetitions=2)
        parametrization = hpnevergrad.hpng(_)
        assert 'choice' in parametrization.kwargs.keys()


    def test_ordered_choice(self) -> None:
        parser, _ = self._make_basic()
        transition_choice = _('transition_choice',2, choices=[0, 1, 2, 3],transitions=[-1000000, 10])
        parametrization = hpnevergrad.hpng(_)
        assert 'transition_choice' in parametrization.kwargs.keys()
        
