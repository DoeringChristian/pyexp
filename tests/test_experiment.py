"""Tests for the Experiment class and experiment decorator."""

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import pyexp
from pyexp import experiment, Config, Tensor, sweep, Executor


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
        def report(results, out):
            return len(results)

        assert my_exp._report_fn is not None


class TestExperimentRun:
    """Tests for Experiment.run() execution."""

    def test_run_executes_pipeline(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == 10

    def test_run_with_passed_functions(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"value": config["x"] + 1}

        def my_configs():
            return [{"name": "t", "x": 10}]

        def my_report(results, out):
            return [r.result["value"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(configs=my_configs, report=my_report, output_dir=tmp_path)

        assert result == [11]

    def test_run_caches_results(self, tmp_path):
        call_count = 0

        @experiment
        def my_exp(config):
            nonlocal call_count
            call_count += 1
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cached", "x": 42}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            # Use executor="inline" to track call_count in same process
            result1 = my_exp.run(output_dir=tmp_path, executor="inline")
            result2 = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result1 == 42
        assert result2 == 42
        assert call_count == 1  # Only called once due to caching

    def test_run_report_flag(self, tmp_path):
        @experiment(timestamp=False)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "rep", "x": 99}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

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
        def report(results, out):
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
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=out_dir)

        assert out_dir.exists()

    def test_log_saved_to_file(self, tmp_path):
        """Log output should be saved to log.out in experiment folder."""
        @experiment(timestamp=False)
        def my_exp(config):
            print("Hello from experiment")
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "logged", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="subprocess")

        # Check log.out exists and contains output
        exp_dir = tmp_path / "my_exp"
        log_files = list(exp_dir.rglob("log.out"))
        assert len(log_files) == 1
        assert "Hello from experiment" in log_files[0].read_text()

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
        def report(results, out):
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
        def report(results, out):
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
        def report(results, out):
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
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(AssertionError, match="out"):
                my_exp.run(output_dir=tmp_path)

    def test_multiple_configs(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"value": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(results, out):
            return [r.result["value"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == [4, 9, 16]

    def test_report_receives_tensor(self, tmp_path):
        """Report function should receive results as Tensor."""
        received_results = None

        @experiment
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "a", "x": 1}, {"name": "b", "x": 2}]

        @my_exp.report
        def report(results, out):
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
        def report(results, out):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        result = received_results[0]
        assert result.config["name"] == "test"
        assert result.config["lr"] == 0.01
        assert result.config["epochs"] == 10
        assert result.result["accuracy"] == 0.95
        # out should not be in config
        assert "out" not in result.config

    def test_results_filterable_by_config(self, tmp_path):
        """Results should be filterable by config values."""
        received_results = None

        @experiment
        def my_exp(config):
            return {"value": config["x"] * config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "x1", "x": 1}, {"name": "x2", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "y10", "y": 10}, {"name": "y20", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results, out):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        # Filter by config.x
        x1_results = received_results[{"config.x": 1}]
        assert x1_results.shape == (1, 1, 2)
        assert all(r.config["x"] == 1 for r in x1_results)

        # Filter by config.y
        y10_results = received_results[{"config.y": 10}]
        assert y10_results.shape == (1, 2, 1)
        assert all(r.config["y"] == 10 for r in y10_results)

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
        def report(results, out):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert received_results.shape == (1, 2, 2)
        assert received_results[0, 0, 0].config["x"] == 1
        assert received_results[0, 0, 0].result["val"] == 11

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
        def report(results, out):
            nonlocal received_results
            received_results = results
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        result = received_results[0]
        assert result.config["name"] == "test"
        assert result.config["x"] == 5
        assert result.result == 10


class TestOutputFolderStructure:
    """Tests for output folder naming and timestamp features."""

    def test_default_name_is_function_name(self):
        """Experiment name defaults to function name."""
        @experiment
        def my_custom_experiment(config):
            return config["x"]

        assert my_custom_experiment._name == "my_custom_experiment"

    def test_custom_name_in_decorator(self):
        """Can set custom name in decorator."""
        @experiment(name="mnist_classifier")
        def my_exp(config):
            return config["x"]

        assert my_exp._name == "mnist_classifier"

    def test_timestamp_default_true(self):
        """Timestamp defaults to True."""
        @experiment
        def my_exp(config):
            return config["x"]

        assert my_exp._timestamp_default is True

    def test_timestamp_false_in_decorator(self):
        """Can disable timestamp in decorator."""
        @experiment(timestamp=False)
        def my_exp(config):
            return config["x"]

        assert my_exp._timestamp_default is False

    def test_output_structure_with_timestamp(self, tmp_path):
        """Output folder includes timestamp when timestamp=True."""
        @experiment(name="test_exp", timestamp=True)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Check structure: tmp_path/test_exp/<timestamp>/cfg-<hash>/
        exp_dir = tmp_path / "test_exp"
        assert exp_dir.exists()

        # Should have one timestamp folder
        timestamp_dirs = list(exp_dir.iterdir())
        assert len(timestamp_dirs) == 1
        assert timestamp_dirs[0].is_dir()

        # Timestamp folder should contain config folder and report folder
        contents = list(timestamp_dirs[0].iterdir())
        config_dirs = [d for d in contents if d.name.startswith("cfg-")]
        assert len(config_dirs) == 1
        assert (timestamp_dirs[0] / "report").exists()

    def test_output_structure_without_timestamp(self, tmp_path):
        """Output folder has no timestamp when timestamp=False."""
        @experiment(name="test_exp", timestamp=False)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Check structure: tmp_path/test_exp/cfg-<hash>/
        exp_dir = tmp_path / "test_exp"
        assert exp_dir.exists()

        # Should directly contain config folder and report folder (no timestamp)
        contents = list(exp_dir.iterdir())
        config_dirs = [d for d in contents if d.name.startswith("cfg-")]
        assert len(config_dirs) == 1
        assert (exp_dir / "report").exists()

    def test_cli_timestamp_continues_run(self, tmp_path):
        """--timestamp CLI arg continues a specific run."""
        @experiment(name="test_exp", timestamp=True)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        # First run with specific timestamp
        with patch.object(sys, "argv", ["test", "--timestamp", "2024-01-01_12-00-00"]):
            result1 = my_exp.run(output_dir=tmp_path, executor="inline")

        # Second run with same timestamp should use cache
        with patch.object(sys, "argv", ["test", "--timestamp", "2024-01-01_12-00-00"]):
            result2 = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result1 == 1
        assert result2 == 1

        # Check folder exists
        timestamp_dir = tmp_path / "test_exp" / "2024-01-01_12-00-00"
        assert timestamp_dir.exists()

    def test_continue_uses_latest_timestamp(self, tmp_path):
        """--continue uses the most recent timestamp folder."""
        @experiment(name="test_exp", timestamp=True)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        # Create two runs with specific timestamps
        with patch.object(sys, "argv", ["test", "--timestamp", "2024-01-01_10-00-00"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        with patch.object(sys, "argv", ["test", "--timestamp", "2024-01-02_10-00-00"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # --continue should use the latest (2024-01-02)
        with patch.object(sys, "argv", ["test", "--continue"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result == 1
        # Verify it used the latest timestamp folder
        assert (tmp_path / "test_exp" / "2024-01-02_10-00-00").exists()

    def test_continue_no_previous_runs_raises(self, tmp_path):
        """--continue raises error when no previous runs exist."""
        @experiment(name="new_exp", timestamp=True)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test", "--continue"]):
            with pytest.raises(RuntimeError, match="No previous runs found"):
                my_exp.run(output_dir=tmp_path, executor="inline")

    def test_name_override_in_run(self, tmp_path):
        """Can override name in run()."""
        @experiment(name="default_name", timestamp=False)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, name="override_name", executor="inline")

        # Should use overridden name
        assert (tmp_path / "override_name").exists()
        assert not (tmp_path / "default_name").exists()

    def test_timestamp_override_in_run(self, tmp_path):
        """Can override timestamp in run()."""
        @experiment(name="test_exp", timestamp=True)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, timestamp=False, executor="inline")

        # Should have no timestamp folder
        exp_dir = tmp_path / "test_exp"
        contents = list(exp_dir.iterdir())
        config_dirs = [d for d in contents if d.name.startswith("cfg-")]
        assert len(config_dirs) == 1
        assert (exp_dir / "report").exists()


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
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        # If executor="inline" was used, call_count would be updated
        assert call_count == 1
        assert result == 5

    def test_run_can_override_decorator_default(self, tmp_path):
        """run(executor=...) can override the decorator default."""
        @experiment(executor="inline")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

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
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

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
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert result.error is not None
        assert "ValueError" in result.error
        assert "Test error" in result.error
        assert result.config["name"] == "failing"

    def test_subprocess_continues_after_failure(self, tmp_path):
        """Other experiments continue even if one fails."""
        @experiment
        def my_exp(config):
            if config["x"] == 2:
                raise ValueError("Fail on x=2")
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 1},
                {"name": "b", "x": 2},  # This will fail
                {"name": "c", "x": 3},
            ]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="subprocess")

        # First and third succeed
        assert results[0].result["value"] == 1
        assert results[2].result["value"] == 3

        # Second failed
        assert results[1].error is not None
        assert "ValueError" in results[1].error

    def test_subprocess_multiple_configs(self, tmp_path):
        """Multiple configs all run in separate subprocesses."""
        @experiment
        def my_exp(config):
            return {"value": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(results, out):
            return [r.result["value"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert result == [4, 9, 16]

    def test_subprocess_with_sweep(self, tmp_path):
        """Subprocess execution works with sweep configurations."""
        @experiment
        def my_exp(config):
            return {"value": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "c", "y": 10}, {"name": "d", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="subprocess")

        assert results.shape == (1, 2, 2)
        assert results[0, 0, 0].result["value"] == 11  # x=1, y=10
        assert results[0, 1, 1].result["value"] == 22  # x=2, y=20


import os

@pytest.mark.skipif(not hasattr(os, "fork"), reason="Fork not available on this platform")
class TestForkExecution:
    """Tests for fork-based experiment execution (Unix only)."""

    def test_fork_runs_experiment(self, tmp_path):
        """Experiments run correctly with fork executor."""
        @experiment
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

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
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="fork")

        assert result.error is not None
        assert "ValueError" in result.error
        assert "Test error in fork" in result.error
        assert result.config["name"] == "failing"

    def test_fork_continues_after_failure(self, tmp_path):
        """Other experiments continue even if one fails in fork."""
        @experiment
        def my_exp(config):
            if config["x"] == 2:
                raise ValueError("Fail on x=2")
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 1},
                {"name": "b", "x": 2},  # This will fail
                {"name": "c", "x": 3},
            ]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="fork")

        # First and third succeed
        assert results[0].result["value"] == 1
        assert results[2].result["value"] == 3

        # Second failed
        assert results[1].error is not None
        assert "ValueError" in results[1].error

    def test_fork_multiple_configs(self, tmp_path):
        """Multiple configs all run in separate forked processes."""
        @experiment
        def my_exp(config):
            return {"value": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(results, out):
            return [r.result["value"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="fork")

        assert result == [4, 9, 16]

    def test_fork_decorator_default(self, tmp_path):
        """Can set fork as default executor in decorator."""
        @experiment(executor="fork")
        def my_exp(config):
            return {"value": config["x"] * 3}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 7}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == 21

    def test_fork_with_sweep(self, tmp_path):
        """Fork execution works with sweep configurations."""
        @experiment
        def my_exp(config):
            return {"value": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "c", "y": 10}, {"name": "d", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            results = my_exp.run(output_dir=tmp_path, executor="fork")

        assert results.shape == (1, 2, 2)
        assert results[0, 0, 0].result["value"] == 11  # x=1, y=10
        assert results[0, 1, 1].result["value"] == 22  # x=2, y=20


def ray_available():
    """Check if Ray is installed."""
    try:
        import ray
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not ray_available(), reason="Ray not installed")
class TestRayExecution:
    """Tests for Ray-based experiment execution."""

    def test_ray_address_in_decorator(self):
        """Can set remote cluster address via executor='ray://...' in decorator."""
        @experiment(executor="ray://cluster:10001")
        def my_exp(config):
            return config["x"]

        assert my_exp._executor_default == "ray://cluster:10001"

    def test_ray_address_in_run(self, tmp_path):
        """Can set remote cluster address via executor='ray://...' or 'ray:<address>' in run()."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        # This will use local Ray since we don't have a cluster
        # Just testing that the parameter is accepted
        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="ray")

        assert result == 5

    def test_ray_executor_with_options(self):
        """RayExecutor accepts configuration options."""
        from pyexp import RayExecutor

        # Should not raise - just testing initialization
        executor = RayExecutor(num_cpus=2)
        assert executor._ray.is_initialized()

    def test_ray_executor_with_runtime_env(self, tmp_path):
        """RayExecutor works with runtime_env configuration."""
        from pyexp import RayExecutor

        executor = RayExecutor(
            runtime_env={"working_dir": str(tmp_path)}
        )

        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 42}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor=executor)

        assert result == 42

    def test_ray_runs_experiment(self, tmp_path):
        """Experiments run correctly with ray executor."""
        @experiment
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="ray")

        assert result == 10

    def test_ray_handles_exception(self, tmp_path):
        """Exceptions in Ray task are captured and returned as error results."""
        @experiment
        def my_exp(config):
            raise ValueError("Test error in ray")

        @my_exp.configs
        def configs():
            return [{"name": "failing", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="ray")

        assert result.error is not None
        assert "ValueError" in result.error
        assert "Test error in ray" in result.error
        assert result.config["name"] == "failing"

    def test_ray_multiple_configs(self, tmp_path):
        """Multiple configs run with Ray executor."""
        @experiment
        def my_exp(config):
            return {"value": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        @my_exp.report
        def report(results, out):
            return [r.result["value"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="ray")

        assert result == [4, 9, 16]

    def test_ray_decorator_default(self, tmp_path):
        """Can set ray as default executor in decorator."""
        @experiment(executor="ray")
        def my_exp(config):
            return {"value": config["x"] * 3}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 7}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path)

        assert result == 21


class TestCustomExecutor:
    """Tests for custom executor support."""

    def test_custom_executor_instance(self, tmp_path):
        """Can pass a custom Executor instance."""
        from pyexp import Executor

        class CountingExecutor(Executor):
            def __init__(self):
                self.call_count = 0

            def run(self, fn, config, result_path, capture=True):
                self.call_count += 1
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result = fn(config)
                structured = {"result": result, "error": None, "log": ""}
                with open(result_path, "wb") as f:
                    pickle.dump(structured, f)
                return structured

        custom_executor = CountingExecutor()

        @experiment
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "a", "x": 1}, {"name": "b", "x": 2}]

        @my_exp.report
        def report(results, out):
            return [r.result["value"] for r in results]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor=custom_executor)

        assert result == [2, 4]
        assert custom_executor.call_count == 2

    def test_get_executor_unknown_raises(self):
        """get_executor raises for unknown executor name."""
        from pyexp import get_executor

        with pytest.raises(ValueError, match="Unknown executor"):
            get_executor("nonexistent")


class TestRetry:
    """Tests for the retry functionality."""

    def test_retry_default_is_4(self, tmp_path):
        """Default retry count should be 4."""
        @experiment
        def my_exp(config):
            return {"value": 1}

        assert my_exp._retry_default == 4

    def test_retry_in_decorator(self, tmp_path):
        """Retry can be set in decorator."""
        @experiment(retry=2)
        def my_exp(config):
            return {"value": 1}

        assert my_exp._retry_default == 2

    def test_retry_on_failure(self, tmp_path):
        """Failed experiments should be retried."""
        attempt_count = 0

        class RetryTestExecutor(Executor):
            def run(self, fn, config, result_path, capture=True):
                nonlocal attempt_count
                attempt_count += 1
                result_path.parent.mkdir(parents=True, exist_ok=True)
                # Fail on first attempt, succeed on retry
                if attempt_count == 1:
                    structured = {"result": None, "error": "First attempt failed", "log": ""}
                else:
                    structured = {"result": {"value": 42}, "error": None, "log": ""}
                with open(result_path, "wb") as f:
                    pickle.dump(structured, f)
                return structured

        @experiment(retry=4)
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor=RetryTestExecutor())

        assert attempt_count == 2  # First attempt failed, second succeeded
        assert result.result["value"] == 42

    def test_retry_exhausted(self, tmp_path):
        """When retries exhausted, error should be captured."""
        attempt_count = 0

        class AlwaysFailExecutor(Executor):
            def run(self, fn, config, result_path, capture=True):
                nonlocal attempt_count
                attempt_count += 1
                result_path.parent.mkdir(parents=True, exist_ok=True)
                structured = {"result": None, "error": "Always fails", "log": f"Attempt {attempt_count}"}
                with open(result_path, "wb") as f:
                    pickle.dump(structured, f)
                return structured

        @experiment(retry=2)
        def my_exp(config):
            raise RuntimeError("Always fails")

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor=AlwaysFailExecutor())

        # 1 initial attempt + 2 retries = 3 total
        assert attempt_count == 3
        assert result.error == "Always fails"

    def test_retry_zero_no_retries(self, tmp_path):
        """With retry=0, no retries should happen."""
        attempt_count = 0

        class FailOnceExecutor(Executor):
            def run(self, fn, config, result_path, capture=True):
                nonlocal attempt_count
                attempt_count += 1
                result_path.parent.mkdir(parents=True, exist_ok=True)
                structured = {"result": None, "error": "Failed", "log": ""}
                with open(result_path, "wb") as f:
                    pickle.dump(structured, f)
                return structured

        @experiment(retry=0)
        def my_exp(config):
            raise RuntimeError("Fails")

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor=FailOnceExecutor())

        assert attempt_count == 1  # Only 1 attempt, no retries
        assert result.error == "Failed"

    def test_retry_cli_override(self, tmp_path):
        """CLI --retry should override decorator setting."""
        attempt_count = 0

        class AlwaysFailExecutor(Executor):
            def run(self, fn, config, result_path, capture=True):
                nonlocal attempt_count
                attempt_count += 1
                result_path.parent.mkdir(parents=True, exist_ok=True)
                structured = {"result": None, "error": "Always fails", "log": ""}
                with open(result_path, "wb") as f:
                    pickle.dump(structured, f)
                return structured

        @experiment(retry=10)  # Decorator says 10 retries
        def my_exp(config):
            raise RuntimeError("Always fails")

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        # CLI overrides to 1 retry
        with patch.object(sys, "argv", ["test", "--retry", "1"]):
            result = my_exp.run(output_dir=tmp_path, executor=AlwaysFailExecutor())

        # 1 initial attempt + 1 retry = 2 total (not 11)
        assert attempt_count == 2

    def test_retry_run_override(self, tmp_path):
        """run() retry should override decorator setting."""
        attempt_count = 0

        class AlwaysFailExecutor(Executor):
            def run(self, fn, config, result_path, capture=True):
                nonlocal attempt_count
                attempt_count += 1
                result_path.parent.mkdir(parents=True, exist_ok=True)
                structured = {"result": None, "error": "Always fails", "log": ""}
                with open(result_path, "wb") as f:
                    pickle.dump(structured, f)
                return structured

        @experiment(retry=10)  # Decorator says 10 retries
        def my_exp(config):
            raise RuntimeError("Always fails")

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        # run() overrides to 1 retry
        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor=AlwaysFailExecutor(), retry=1)

        # 1 initial attempt + 1 retry = 2 total
        assert attempt_count == 2
