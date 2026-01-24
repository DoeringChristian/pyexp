"""Tests for the Experiment class and experiment decorator."""

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pyexp
from pyexp import experiment, Config, Tensor, sweep


class TestExperimentDecorator:
    """Tests for the @experiment decorator."""

    def test_decorator_returns_experiment(self):
        @experiment
        def my_exp(config):
            return config["x"] * 2

        assert isinstance(my_exp, pyexp.Experiment)

    def test_experiment_callable(self):
        @experiment
        def my_exp(config):
            return config["x"] * 2

        result = my_exp({"x": 5})
        assert result == 10

    def test_configs_decorator(self):
        @experiment
        def my_exp(config):
            return config["x"]

        @my_exp.configs
        def configs():
            return [{"x": 1}, {"x": 2}]

        assert my_exp._configs_fn is not None
        assert my_exp._configs_fn() == [{"x": 1}, {"x": 2}]

    def test_report_decorator(self):
        @experiment
        def my_exp(config):
            return config["x"]

        @my_exp.report
        def report(configs, results):
            return sum(results)

        assert my_exp._report_fn is not None


class TestExperimentRun:
    """Tests for Experiment.run() execution."""

    def test_run_executes_pipeline(self, tmp_path):
        @experiment
        def my_exp(config):
            return config["x"] * 2

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(configs, results):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == 10

    def test_run_with_passed_functions(self, tmp_path):
        @experiment
        def my_exp(config):
            return config["x"] + 1

        def my_configs():
            return [{"name": "t", "x": 10}]

        def my_report(configs, results):
            return results.tolist()

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(configs=my_configs, report=my_report, output_dir=tmp_path)

        assert result == [11]

    def test_run_caches_results(self, tmp_path):
        call_count = 0

        @experiment
        def my_exp(config):
            nonlocal call_count
            call_count += 1
            return config["x"]

        @my_exp.configs
        def configs():
            return [{"name": "cached", "x": 42}]

        @my_exp.report
        def report(configs, results):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result1 = my_exp.run(output_dir=tmp_path)
            result2 = my_exp.run(output_dir=tmp_path)

        assert result1 == 42
        assert result2 == 42
        assert call_count == 1  # Only called once due to caching

    def test_run_rerun_flag(self, tmp_path):
        call_count = 0

        @experiment
        def my_exp(config):
            nonlocal call_count
            call_count += 1
            return call_count

        @my_exp.configs
        def configs():
            return [{"name": "rerun", "x": 1}]

        @my_exp.report
        def report(configs, results):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        with patch.object(sys, "argv", ["test", "--rerun"]):
            result = my_exp.run(output_dir=tmp_path)

        assert call_count == 2
        assert result == 2

    def test_run_report_flag(self, tmp_path):
        @experiment
        def my_exp(config):
            return config["x"]

        @my_exp.configs
        def configs():
            return [{"name": "rep", "x": 99}]

        @my_exp.report
        def report(configs, results):
            return results[0]

        # First run to create cache
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        # Report-only run
        with patch.object(sys, "argv", ["test", "--report"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == 99

    def test_run_report_flag_no_cache_raises(self, tmp_path):
        @experiment
        def my_exp(config):
            return config["x"]

        @my_exp.configs
        def configs():
            return [{"name": "nocache", "x": 1}]

        @my_exp.report
        def report(configs, results):
            return results

        with patch.object(sys, "argv", ["test", "--report"]):
            with pytest.raises(RuntimeError, match="No cached result"):
                my_exp.run(output_dir=tmp_path)

    def test_run_creates_output_dir(self, tmp_path):
        out_dir = tmp_path / "nested" / "output"

        @experiment
        def my_exp(config):
            return 1

        @my_exp.configs
        def configs():
            return [{"name": "dir", "x": 1}]

        @my_exp.report
        def report(configs, results):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=out_dir)

        assert out_dir.exists()

    def test_config_receives_out(self, tmp_path):
        received_out = None

        @experiment
        def my_exp(config):
            nonlocal received_out
            received_out = config.out
            return 1

        @my_exp.configs
        def configs():
            return [{"name": "outdir", "x": 1}]

        @my_exp.report
        def report(configs, results):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert received_out is not None
        assert isinstance(received_out, Path)
        assert received_out.exists()

    def test_config_is_config_type(self, tmp_path):
        received_config = None

        @experiment
        def my_exp(config):
            nonlocal received_config
            received_config = config
            return 1

        @my_exp.configs
        def configs():
            return [{"name": "type", "nested": {"a": 1}}]

        @my_exp.report
        def report(configs, results):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert isinstance(received_config, Config)
        assert received_config.nested.a == 1

    def test_run_no_configs_raises(self, tmp_path):
        @experiment
        def my_exp(config):
            return 1

        @my_exp.report
        def report(configs, results):
            return results

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(RuntimeError, match="No configs function"):
                my_exp.run(output_dir=tmp_path)

    def test_run_no_report_raises(self, tmp_path):
        @experiment
        def my_exp(config):
            return 1

        @my_exp.configs
        def configs():
            return [{"name": "x", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(RuntimeError, match="No report function"):
                my_exp.run(output_dir=tmp_path)

    def test_out_in_config_raises(self, tmp_path):
        @experiment
        def my_exp(config):
            return 1

        @my_exp.configs
        def configs():
            return [{"name": "bad", "out": "/some/path"}]

        @my_exp.report
        def report(configs, results):
            return results

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(AssertionError, match="out"):
                my_exp.run(output_dir=tmp_path)

    def test_multiple_configs(self, tmp_path):
        @experiment
        def my_exp(config):
            return config["x"] ** 2

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(configs, results):
            return results.tolist()

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == [4, 9, 16]

    def test_report_receives_tensors(self, tmp_path):
        """Report function should receive configs and results as Tensors."""
        received_configs = None
        received_results = None

        @experiment
        def my_exp(config):
            return config["x"] * 2

        @my_exp.configs
        def configs():
            return [{"name": "a", "x": 1}, {"name": "b", "x": 2}]

        @my_exp.report
        def report(configs, results):
            nonlocal received_configs, received_results
            received_configs = configs
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert isinstance(received_configs, Tensor)
        assert isinstance(received_results, Tensor)
        assert received_configs.shape == (2,)
        assert received_results.shape == (2,)

    def test_report_tensors_preserve_sweep_shape(self, tmp_path):
        """Tensors should preserve shape from sweep operations."""
        received_configs = None
        received_results = None

        @experiment
        def my_exp(config):
            return {"val": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"x": 1}, {"x": 2}])
            cfgs = sweep(cfgs, [{"y": 10}, {"y": 20}])
            return cfgs

        @my_exp.report
        def report(configs, results):
            nonlocal received_configs, received_results
            received_configs = configs
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert received_configs.shape == (1, 2, 2)
        assert received_results.shape == (1, 2, 2)
        # Check indexing works the same way
        assert received_configs[0, 0, 0]["x"] == 1
        assert received_results[0, 0, 0]["val"] == 11
