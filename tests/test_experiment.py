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
        def report(results):
            return len(results)

        assert my_exp._report_fn is not None


class TestExperimentRun:
    """Tests for Experiment.run() execution."""

    def test_run_executes_pipeline(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"result": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == 10

    def test_run_with_passed_functions(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"result": config["x"] + 1}

        def my_configs():
            return [{"name": "t", "x": 10}]

        def my_report(results):
            return [r["result"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(configs=my_configs, report=my_report, output_dir=tmp_path)

        assert result == [11]

    def test_run_caches_results(self, tmp_path):
        call_count = 0

        @experiment
        def my_exp(config):
            nonlocal call_count
            call_count += 1
            return {"result": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cached", "x": 42}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            # Use executor="inline" to track call_count in same process
            result1 = my_exp.run(output_dir=tmp_path, executor="inline")
            result2 = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result1 == 42
        assert result2 == 42
        assert call_count == 1  # Only called once due to caching

    def test_run_rerun_flag(self, tmp_path):
        call_count = 0

        @experiment
        def my_exp(config):
            nonlocal call_count
            call_count += 1
            return {"result": call_count}

        @my_exp.configs
        def configs():
            return [{"name": "rerun", "x": 1}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            # Use executor="inline" to track call_count in same process
            my_exp.run(output_dir=tmp_path, executor="inline")

        with patch.object(sys, "argv", ["test", "--rerun"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert call_count == 2
        assert result == 2

    def test_run_report_flag(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"result": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "rep", "x": 99}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

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
        def report(results):
            return results

        with patch.object(sys, "argv", ["test", "--report"]):
            with pytest.raises(RuntimeError, match="No cached result"):
                my_exp.run(output_dir=tmp_path)

    def test_run_creates_output_dir(self, tmp_path):
        out_dir = tmp_path / "nested" / "output"

        @experiment
        def my_exp(config):
            return {"x": 1}

        @my_exp.configs
        def configs():
            return [{"name": "dir", "x": 1}]

        @my_exp.report
        def report(results):
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
            return {"x": 1}

        @my_exp.configs
        def configs():
            return [{"name": "outdir", "x": 1}]

        @my_exp.report
        def report(results):
            return results

        with patch.object(sys, "argv", ["test"]):
            # Use executor="inline" to capture received_out in same process
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert received_out is not None
        assert isinstance(received_out, Path)
        assert received_out.exists()

    def test_config_is_config_type(self, tmp_path):
        received_config = None

        @experiment
        def my_exp(config):
            nonlocal received_config
            received_config = config
            return {"x": 1}

        @my_exp.configs
        def configs():
            return [{"name": "type", "nested": {"a": 1}}]

        @my_exp.report
        def report(results):
            return results

        with patch.object(sys, "argv", ["test"]):
            # Use executor="inline" to capture received_config in same process
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert isinstance(received_config, Config)
        assert received_config.nested.a == 1

    def test_run_no_configs_raises(self, tmp_path):
        @experiment
        def my_exp(config):
            return 1

        @my_exp.report
        def report(results):
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
        def report(results):
            return results

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(AssertionError, match="out"):
                my_exp.run(output_dir=tmp_path)

    def test_multiple_configs(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"result": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(results):
            return [r["result"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == [4, 9, 16]

    def test_report_receives_tensor(self, tmp_path):
        """Report function should receive results as Tensor."""
        received_results = None

        @experiment
        def my_exp(config):
            return {"result": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "a", "x": 1}, {"name": "b", "x": 2}]

        @my_exp.report
        def report(results):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert isinstance(received_results, Tensor)
        assert received_results.shape == (2,)

    def test_results_contain_config_and_name(self, tmp_path):
        """Each result should contain the config and name."""
        received_results = None

        @experiment
        def my_exp(config):
            return {"accuracy": 0.95}

        @my_exp.configs
        def configs():
            return [{"name": "test", "lr": 0.01, "epochs": 10}]

        @my_exp.report
        def report(results):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        result = received_results[0]
        assert result["name"] == "test"
        assert result["config"]["lr"] == 0.01
        assert result["config"]["epochs"] == 10
        assert result["accuracy"] == 0.95
        # out should not be in config
        assert "out" not in result["config"]

    def test_results_filterable_by_config(self, tmp_path):
        """Results should be filterable by config values."""
        received_results = None

        @experiment
        def my_exp(config):
            return {"result": config["x"] * config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "x1", "x": 1}, {"name": "x2", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "y10", "y": 10}, {"name": "y20", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        # Filter by config.x
        x1_results = received_results[{"config.x": 1}]
        assert x1_results.shape == (1, 1, 2)
        assert all(r["config"]["x"] == 1 for r in x1_results)

        # Filter by config.y
        y10_results = received_results[{"config.y": 10}]
        assert y10_results.shape == (1, 2, 1)
        assert all(r["config"]["y"] == 10 for r in y10_results)

    def test_report_tensors_preserve_sweep_shape(self, tmp_path):
        """Tensors should preserve shape from sweep operations."""
        received_results = None

        @experiment
        def my_exp(config):
            return {"val": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "c", "y": 10}, {"name": "d", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert received_results.shape == (1, 2, 2)
        assert received_results[0, 0, 0]["config"]["x"] == 1
        assert received_results[0, 0, 0]["val"] == 11

    def test_non_dict_result_wrapped_in_value(self, tmp_path):
        """Non-dict results should be wrapped with 'value' key."""
        received_results = None

        @experiment
        def my_exp(config):
            return config["x"] * 2  # Returns int, not dict

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        result = received_results[0]
        assert result["name"] == "test"
        assert result["config"]["x"] == 5
        assert result["value"] == 10


class TestExecutorSystem:
    """Tests for the modular executor system."""

    def test_decorator_default_executor_subprocess(self):
        """Default executor is 'subprocess' when not specified."""
        @experiment
        def my_exp(config):
            return config["x"]

        assert my_exp._executor_default == "subprocess"

    def test_decorator_executor_string(self):
        """Can set executor as string in decorator."""
        @experiment(executor="inline")
        def my_exp(config):
            return config["x"]

        assert my_exp._executor_default == "inline"

    def test_run_uses_decorator_default(self, tmp_path):
        """run() uses the decorator's executor default."""
        call_count = 0

        @experiment(executor="inline")
        def my_exp(config):
            nonlocal call_count
            call_count += 1
            return {"result": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        # If executor="inline" was used, call_count would be updated
        assert call_count == 1
        assert result == 5

    def test_run_can_override_decorator_default(self, tmp_path):
        """run(executor=...) can override the decorator default."""
        @experiment(executor="inline")
        def my_exp(config):
            return {"result": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            # Override executor="inline" with executor="subprocess"
            result = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert result == 10

class TestSubprocessExecution:
    """Tests for subprocess-based experiment execution."""

    def test_subprocess_runs_experiment(self, tmp_path):
        """Experiments run in subprocess with executor='subprocess'."""
        @experiment
        def my_exp(config):
            return {"result": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert result == 10

    def test_subprocess_handles_exception(self, tmp_path):
        """Exceptions in subprocess are captured and returned as error results."""
        @experiment
        def my_exp(config):
            raise ValueError("Test error")

        @my_exp.configs
        def configs():
            return [{"name": "failing", "x": 1}]

        @my_exp.report
        def report(results):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert result["__error__"] is True
        assert result["type"] == "ValueError"
        assert "Test error" in result["message"]
        assert result["name"] == "failing"

    def test_subprocess_continues_after_failure(self, tmp_path):
        """Other experiments continue even if one fails."""
        @experiment
        def my_exp(config):
            if config["x"] == 2:
                raise ValueError("Fail on x=2")
            return {"result": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 1},
                {"name": "b", "x": 2},  # This will fail
                {"name": "c", "x": 3},
            ]

        @my_exp.report
        def report(results):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="subprocess")

        # First and third succeed
        assert results[0]["result"] == 1
        assert results[2]["result"] == 3

        # Second failed
        assert results[1]["__error__"] is True
        assert results[1]["type"] == "ValueError"

    def test_subprocess_multiple_configs(self, tmp_path):
        """Multiple configs all run in separate subprocesses."""
        @experiment
        def my_exp(config):
            return {"result": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(results):
            return [r["result"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert result == [4, 9, 16]

    def test_subprocess_with_sweep(self, tmp_path):
        """Subprocess execution works with sweep configurations."""
        @experiment
        def my_exp(config):
            return {"result": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "c", "y": 10}, {"name": "d", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert results.shape == (1, 2, 2)
        assert results[0, 0, 0]["result"] == 11  # x=1, y=10
        assert results[0, 1, 1]["result"] == 22  # x=2, y=20


import os

@pytest.mark.skipif(not hasattr(os, "fork"), reason="Fork not available on this platform")
class TestForkExecution:
    """Tests for fork-based experiment execution (Unix only)."""

    def test_fork_runs_experiment(self, tmp_path):
        """Experiments run correctly with fork executor."""
        @experiment
        def my_exp(config):
            return {"result": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="fork")

        assert result == 10

    def test_fork_handles_exception(self, tmp_path):
        """Exceptions in forked process are captured and returned as error results."""
        @experiment
        def my_exp(config):
            raise ValueError("Test error in fork")

        @my_exp.configs
        def configs():
            return [{"name": "failing", "x": 1}]

        @my_exp.report
        def report(results):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="fork")

        assert result["__error__"] is True
        assert result["type"] == "ValueError"
        assert "Test error in fork" in result["message"]
        assert result["name"] == "failing"

    def test_fork_continues_after_failure(self, tmp_path):
        """Other experiments continue even if one fails in fork."""
        @experiment
        def my_exp(config):
            if config["x"] == 2:
                raise ValueError("Fail on x=2")
            return {"result": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 1},
                {"name": "b", "x": 2},  # This will fail
                {"name": "c", "x": 3},
            ]

        @my_exp.report
        def report(results):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="fork")

        # First and third succeed
        assert results[0]["result"] == 1
        assert results[2]["result"] == 3

        # Second failed
        assert results[1]["__error__"] is True
        assert results[1]["type"] == "ValueError"

    def test_fork_multiple_configs(self, tmp_path):
        """Multiple configs all run in separate forked processes."""
        @experiment
        def my_exp(config):
            return {"result": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(results):
            return [r["result"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="fork")

        assert result == [4, 9, 16]

    def test_fork_decorator_default(self, tmp_path):
        """Can set fork as default executor in decorator."""
        @experiment(executor="fork")
        def my_exp(config):
            return {"result": config["x"] * 3}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 7}]

        @my_exp.report
        def report(results):
            return results[0]["result"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == 21

    def test_fork_with_sweep(self, tmp_path):
        """Fork execution works with sweep configurations."""
        @experiment
        def my_exp(config):
            return {"result": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "c", "y": 10}, {"name": "d", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="fork")

        assert results.shape == (1, 2, 2)
        assert results[0, 0, 0]["result"] == 11  # x=1, y=10
        assert results[0, 1, 1]["result"] == 22  # x=2, y=20


class TestCustomExecutor:
    """Tests for custom executor support."""

    def test_custom_executor_instance(self, tmp_path):
        """Can pass a custom Executor instance."""
        from pyexp import Executor

        class CountingExecutor(Executor):
            def __init__(self):
                self.call_count = 0

            def run(self, fn, config, result_path):
                self.call_count += 1
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result = fn(config)
                with open(result_path, "wb") as f:
                    pickle.dump(result, f)
                return result

        custom_executor = CountingExecutor()

        @experiment
        def my_exp(config):
            return {"result": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "a", "x": 1}, {"name": "b", "x": 2}]

        @my_exp.report
        def report(results):
            return [r["result"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor=custom_executor)

        assert result == [2, 4]
        assert custom_executor.call_count == 2

    def test_get_executor_unknown_raises(self):
        """get_executor raises for unknown executor name."""
        from pyexp import get_executor

        with pytest.raises(ValueError, match="Unknown executor"):
            get_executor("nonexistent")
