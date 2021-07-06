import hpman
import nevergrad as ng
import pytest

from hpnevergrad import hpng


class Test(object):
    def _make_basic(self):
        _ = hpman.HyperParameterManager("_")
        _.parse_file(__file__)
        return _

    def test_array_basic(self) -> None:
        arr1_hpm = hpman.HyperParameterManager("arr1_hpm")
        arr1_hpm.parse_file(__file__)
        arr1 = arr1_hpm("arr1", [[1, 2], [3, 4]], mutable_sigma=True)
        parametrization = hpng.get_parametrization(arr1_hpm)
        assert "arr1" in parametrization.kwargs.keys()

        arr2_hpm = hpman.HyperParameterManager("arr2_hpm")
        arr2_hpm.parse_file(__file__)
        arr2 = arr2_hpm(
            "arr2",
            [
                300,
            ],
            method="constraint",
            set_integer_casting=True,
            exponent=3,
            custom="cauchy",
        )
        parametrization = hpng.get_parametrization(arr2_hpm)
        assert "arr2" in parametrization.kwargs.keys()

    def test_log(self) -> None:
        log1_hpm = hpman.HyperParameterManager("log1_hpm")
        log1_hpm.parse_file(__file__)
        log1 = log1_hpm("log1", 0.02, range=[1e-3, 1.0], scale="log", exponent=3.0)
        parametrization = hpng.get_parametrization(log1_hpm)
        assert "log1" in parametrization.kwargs.keys()

        log2_hpm = hpman.HyperParameterManager("log2_hpm")
        log2_hpm.parse_file(__file__)
        log2 = log2_hpm(
            "log2",
            0.02,
            range=[1e-3, 1.0],
            scale="log",
            exponent=3.0,
            sigma=3,
            custom="gaussian",
        )
        parametrization = hpng.get_parametrization(log2_hpm)
        assert "log2" in parametrization.kwargs.keys()

    def test_choice_repetitions(self) -> None:
        choice_hpm = hpman.HyperParameterManager("choice_hpm")
        choice_hpm.parse_file(__file__)
        choice = choice_hpm("choice", 0, choices=[0, 1, 2, 3], repetitions=2)
        parametrization = hpng.get_parametrization(choice_hpm)
        assert "choice" in parametrization.kwargs.keys()

    def test_ordered_choice(self) -> None:
        transition_choice_hpm = hpman.HyperParameterManager("transition_choice_hpm")
        transition_choice_hpm.parse_file(__file__)
        transition_choice = transition_choice_hpm(
            "transition_choice", 2, choices=[0, 1, 2, 3], transitions=[-1000000, 10]
        )
        parametrization = hpng.get_parametrization(transition_choice_hpm)
        assert "transition_choice" in parametrization.kwargs.keys()

    def test_get_objective_function(self) -> None:
        def func(*args, **kwargs):
            return (x - 0.01) ** 2

        objective_function_hpm = hpman.HyperParameterManager("objective_function_hpm")
        objective_function_hpm.parse_file(__file__)
        x = objective_function_hpm(
            "x", 0.02, range=[1e-3, 1.0], scale="log", exponent=3.0
        )
        objective_function = hpng.get_objective_function(func, objective_function_hpm)
        assert callable(objective_function)
        ans = objective_function(x=x)
        assert ans == 0.0001

    def test_optimizer_warpper(self):
        optim_type = "RandomSearch"
        budget = 10
        parametrization = ng.p.Instrumentation(ng.p.Array(shape=(2,)), y=ng.p.Scalar())
        optimizer = hpng.optimizer_warpper(optim_type, budget, parametrization)
        assert isinstance(
            optimizer.provide_recommendation(), ng.p.Parameter
        ), "Recommendation should be available from start"

    def test_split_module(self):
        s = "a.py:run"
        f, obj = hpng.split_module(s)
        assert f == "a.py"
        assert obj == "run"

    def test_import_func(self):
        f = "test_file/test_bin.py"
        obj = "train"
        f = hpng.import_func(f, obj)
        assert callable(f)
