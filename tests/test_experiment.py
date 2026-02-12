"""Tests for experiment framework."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pyexp import Config, Tensor, experiment, sweep, Experiment, ExperimentRunner


class TestExperimentDecorator:
    """Tests for the @experiment decorator syntax."""

    def test_decorator_returns_runner(self):
        @experiment
        def my_exp(config):
            return config["x"]

        assert isinstance(my_exp, ExperimentRunner)

    def test_experiment_class_accessible(self):
        @experiment
        def my_exp(config):
            return config["x"] * 2

        # The experiment class should be accessible
        assert my_exp._experiment_class is not None

    def test_configs_decorator(self):
        @experiment
        def my_exp(config):
            return config["x"]

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        assert my_exp._configs_fn is not None

    def test_report_decorator(self):
        @experiment
        def my_exp(config):
            return config["x"]

        @my_exp.report
        def report(results, out):
            return results

        assert my_exp._report_fn is not None


class TestExperimentRun:
    """Tests for ExperimentRunner.run() execution."""

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
            result = my_exp.run(output_dir=tmp_path, executor="inline")

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
            result = my_exp.run(
                configs=my_configs, report=my_report, output_dir=tmp_path, executor="inline"
            )

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
        @experiment
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
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Report-only run (--report implies --continue to latest)
        with patch.object(sys, "argv", ["test", "--report"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result == 99

    def test_report_uses_saved_configs(self, tmp_path):
        """--report should use saved configs, not recompute from configs function."""
        config_value = [42]  # Mutable to allow modification

        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": config_value[0]}]

        @my_exp.report
        def report(results, out):
            return results[0].cfg["x"]

        # First run with x=42
        with patch.object(sys, "argv", ["test"]):
            result1 = my_exp.run(output_dir=tmp_path, executor="inline")
        assert result1 == 42

        # Change configs function to return different value
        config_value[0] = 999

        # --report should still use saved config with x=42
        with patch.object(sys, "argv", ["test", "--report"]):
            result2 = my_exp.run(output_dir=tmp_path, executor="inline")
        assert result2 == 42  # Should be original value, not 999

    def test_continue_uses_saved_configs(self, tmp_path):
        """--continue should use saved configs, not recompute from configs function."""
        config_value = [42]  # Mutable to allow modification

        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": config_value[0]}]

        @my_exp.report
        def report(results, out):
            return results[0].cfg["x"]

        # First run with x=42
        with patch.object(sys, "argv", ["test"]):
            result1 = my_exp.run(output_dir=tmp_path, executor="inline")
        assert result1 == 42

        # Change configs function to return different value
        config_value[0] = 999

        # --continue should still use saved config with x=42
        with patch.object(sys, "argv", ["test", "--continue"]):
            result2 = my_exp.run(output_dir=tmp_path, executor="inline")
        assert result2 == 42  # Should be original value, not 999

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

        # --report with no previous runs should raise "No previous runs found"
        with patch.object(sys, "argv", ["test", "--report"]):
            with pytest.raises(RuntimeError, match="No previous runs found"):
                my_exp.run(output_dir=tmp_path, executor="inline")

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
            my_exp.run(output_dir=out_dir, executor="inline")

        assert out_dir.exists()

    def test_log_saved_to_file(self, tmp_path):
        """Log output should be saved to log.out in experiment folder."""

        @experiment
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
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        # Check log contains output
        assert "Hello from experiment" in result.log

    def test_config_receives_out(self, tmp_path):
        received_out = None

        @experiment
        def my_exp(config):
            nonlocal received_out
            # In new API, 'out' is not in config, it's on the instance
            # But for decorator API, config still has what was passed
            # Actually, let's check what's available
            return {"x": 1}

        @my_exp.configs
        def configs():
            return [{"name": "outdir", "x": 1}]

        @my_exp.report
        def report(results, out):
            nonlocal received_out
            # In report, we can access exp.out
            received_out = results[0].out
            return results

        with patch.object(sys, "argv", ["test"]):
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

    def test_run_no_report_succeeds(self, tmp_path):
        """Running without a report function should succeed (report is optional)."""

        @experiment
        def my_exp(config):
            return 1

        @my_exp.configs
        def configs():
            return [{"name": "x", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
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
            result = my_exp.run(output_dir=tmp_path, executor="inline")

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
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert isinstance(received_results, Tensor)
        assert received_results.shape == (2,)

    def test_results_contain_config_and_name(self, tmp_path):
        """Each result should contain the cfg and name."""
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
            my_exp.run(output_dir=tmp_path, executor="inline")

        result = received_results[0]
        assert result.cfg["name"] == "test"
        assert result.cfg["lr"] == 0.01
        assert result.cfg["epochs"] == 10
        assert result.result["accuracy"] == 0.95
        # out is now a property on the experiment, not in cfg
        assert result.out.exists()
        # name is a shorthand for cfg.get("name", "")
        assert result.name == "test"

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
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Filter by cfg.x (note: changed from config.x to cfg.x)
        x1_results = received_results[{"cfg.x": 1}]
        assert x1_results.shape == (1, 1, 2)
        assert all(r.cfg["x"] == 1 for r in x1_results)

        # Filter by cfg.y
        y10_results = received_results[{"cfg.y": 10}]
        assert y10_results.shape == (1, 2, 1)
        assert all(r.cfg["y"] == 10 for r in y10_results)

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
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert received_results.shape == (1, 2, 2)
        assert received_results[0, 0, 0].cfg["x"] == 1
        assert received_results[0, 0, 0].result["val"] == 11

    def test_non_dict_result_wrapped_in_result(self, tmp_path):
        """Non-dict results are stored in .result attribute."""
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
            my_exp.run(output_dir=tmp_path, executor="inline")

        result = received_results[0]
        assert result.cfg["name"] == "test"
        assert result.cfg["x"] == 5
        assert result.result == 10

    def test_config_json_saved(self, tmp_path):
        """Each experiment run should save a config.json."""
        import json

        @experiment
        def my_exp(config):
            return {"value": 1}

        @my_exp.configs
        def configs():
            return [{"name": "test", "lr": 0.01, "nested": {"a": 1}}]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        config_files = list(tmp_path.rglob("config.json"))
        assert len(config_files) == 1
        data = json.loads(config_files[0].read_text())
        assert data["name"] == "test"
        assert data["lr"] == 0.01
        assert data["nested"]["a"] == 1

    def test_invalid_config_type_raises(self, tmp_path):
        """Configs with non-base types should be rejected."""

        @experiment
        def my_exp(config):
            return {"value": 1}

        @my_exp.configs
        def configs():
            return [{"name": "bad", "fn": lambda x: x}]

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(TypeError, match="Invalid config value"):
                my_exp.run(output_dir=tmp_path, executor="inline")

    def test_filter_runs_subset(self, tmp_path):
        """--filter should only run configs matching the regex."""
        runs = []

        @experiment
        def my_exp(config):
            runs.append(config["name"])
            return {"name": config["name"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "lr_0.01_batch_32"},
                {"name": "lr_0.01_batch_64"},
                {"name": "lr_0.001_batch_32"},
                {"name": "lr_0.001_batch_64"},
            ]

        @my_exp.report
        def report(results, out):
            return [r.result["name"] for r in results]

        # First run all configs
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Now filter to only lr_0.01 configs using --continue
        runs.clear()
        with patch.object(sys, "argv", ["test", "--continue", "--filter", "lr_0\\.01"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        # Since all are cached, runs list stays empty
        assert len(runs) == 0

    def test_filter_no_match_returns_none(self, tmp_path):
        """--filter with no matches should return None."""

        @experiment
        def my_exp(config):
            return {"name": config["name"]}

        @my_exp.configs
        def configs():
            return [{"name": "alpha"}, {"name": "beta"}]

        @my_exp.report
        def report(results, out):
            return results

        # First run to create configs
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Filter with no matches
        with patch.object(sys, "argv", ["test", "--continue", "--filter", "gamma"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result is None

    def test_filter_runs_only_matching(self, tmp_path):
        """--filter should only execute matching configs on fresh run."""

        @experiment
        def my_exp(config):
            return {"name": config["name"], "ran": True}

        @my_exp.configs
        def configs():
            return [
                {"name": "exp_a"},
                {"name": "exp_b"},
                {"name": "other_c"},
            ]

        @my_exp.report
        def report(results, out):
            return [r.result["name"] for r in results]

        # Run with filter - only exp_* should run
        with patch.object(sys, "argv", ["test", "--filter", "^exp_"]):
            # This will fail at report because not all results exist
            try:
                my_exp.run(output_dir=tmp_path, executor="inline")
            except FileNotFoundError:
                pass  # Expected - report tries to load all configs

        # Check which experiments actually ran by looking for experiment.pkl files
        run_dir = tmp_path / "my_exp"
        timestamp_dir = list(run_dir.iterdir())[0]
        ran_names = []
        for exp_dir in timestamp_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != "report":
                exp_pkl = exp_dir / "experiment.pkl"
                if exp_pkl.exists():
                    import pickle
                    with open(exp_pkl, "rb") as f:
                        exp = pickle.load(f)
                    if exp.result and exp.result.get("ran"):
                        ran_names.append(exp.result["name"])

        assert sorted(ran_names) == ["exp_a", "exp_b"]


class TestResultsMethod:
    """Tests for the experiment.results() method."""

    def test_results_loads_latest_run(self, tmp_path):
        """results() loads the latest run by default."""

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
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Load results
        results = my_exp.results(output_dir=tmp_path)

        assert results.shape == (1,)
        assert results[0].cfg["name"] == "test"
        assert results[0].cfg["x"] == 5
        assert results[0].result["value"] == 10
        # out is a property on the experiment
        assert results[0].out is not None
        assert results[0].out.exists()

    def test_results_loads_specific_timestamp(self, tmp_path):
        """results() can load a specific timestamp."""
        import time

        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        # Run first experiment
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Get the first timestamp
        base_dir = tmp_path / "my_exp"
        first_timestamp = sorted(base_dir.iterdir())[0].name

        time.sleep(1.1)  # Ensure different timestamp

        # Modify configs for second run
        @my_exp.configs
        def configs2():
            return [{"name": "test", "x": 99}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Load first run by timestamp
        results = my_exp.results(timestamp=first_timestamp, output_dir=tmp_path)
        assert results[0].result["value"] == 1

        # Load latest (second run)
        results_latest = my_exp.results(output_dir=tmp_path)
        assert results_latest[0].result["value"] == 99

    def test_results_preserves_tensor_shape(self, tmp_path):
        """results() preserves the original tensor shape from sweep."""

        @experiment
        def my_exp(config):
            return {"value": config["x"] * config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "base"}]
            cfgs = sweep(cfgs, [{"name": "x1", "x": 1}, {"name": "x2", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "y10", "y": 10}, {"name": "y20", "y": 20}])
            return cfgs

        @my_exp.report
        def report(results, out):
            return results

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)

        # Should preserve 3D shape: (1 base) x (2 x values) x (2 y values)
        assert results.shape == (1, 2, 2)

    def test_results_no_runs_raises(self, tmp_path):
        """results() raises FileNotFoundError when no runs exist."""

        @experiment
        def my_exp(config):
            return {"value": 1}

        with pytest.raises(FileNotFoundError, match="No runs found"):
            my_exp.results(output_dir=tmp_path)

    def test_results_invalid_timestamp_raises(self, tmp_path):
        """results() raises FileNotFoundError for invalid timestamp."""

        @experiment
        def my_exp(config):
            return {"value": 1}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        with pytest.raises(FileNotFoundError, match="Run not found"):
            my_exp.results(timestamp="1999-01-01_00-00-00", output_dir=tmp_path)

    def test_configs_json_saved(self, tmp_path):
        """run() saves runs.json with run folder references and shape."""
        import json

        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 1},
                {"name": "b", "x": 2},
            ]

        @my_exp.report
        def report(results, out):
            return None

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Find the run directory
        base_dir = tmp_path / "my_exp"
        run_dir = sorted(base_dir.iterdir())[0]

        # Check runs.json exists and has correct content
        configs_path = run_dir / "runs.json"
        assert configs_path.exists()

        data = json.loads(configs_path.read_text())
        assert data["shape"] == [2]
        assert len(data["runs"]) == 2
        # runs should be folder names like "a-<hash>" and "b-<hash>"
        assert data["runs"][0].startswith("a-")
        assert data["runs"][1].startswith("b-")

        # Individual config.json files should contain full configs
        for run_folder in data["runs"]:
            config_json = run_dir / run_folder / "config.json"
            assert config_json.exists()
            config_data = json.loads(config_json.read_text())
            assert "x" in config_data


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

    def test_always_creates_timestamp_folder(self, tmp_path):
        """Output folder always includes a timestamp directory."""

        @experiment(name="test_exp")
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

    def test_continue_specific_timestamp(self, tmp_path):
        """--continue=TIMESTAMP continues a specific run."""

        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        # First run
        with patch.object(sys, "argv", ["test"]):
            result1 = my_exp.run(output_dir=tmp_path, executor="inline")

        # Get the timestamp that was created
        exp_dir = tmp_path / "test_exp"
        timestamp = list(exp_dir.iterdir())[0].name

        # Second run with --continue should use cache
        with patch.object(sys, "argv", ["test", f"--continue={timestamp}"]):
            result2 = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result1 == 1
        assert result2 == 1

    def test_continue_uses_latest_timestamp(self, tmp_path):
        """--continue uses the most recent timestamp folder."""

        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        # Create two runs (sleep to ensure different timestamps)
        import time

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")
        time.sleep(1.1)
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        exp_dir = tmp_path / "test_exp"
        timestamps = sorted(d.name for d in exp_dir.iterdir())
        assert len(timestamps) == 2

        # --continue should use the latest
        with patch.object(sys, "argv", ["test", "--continue"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result == 1

    def test_continue_no_previous_runs_raises(self, tmp_path):
        """--continue raises error when no previous runs exist."""

        @experiment(name="new_exp")
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

        @experiment(name="default_name")
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

    def test_list_shows_runs(self, tmp_path, capsys):
        """--list should show all runs."""

        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results

        # Create a run
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # List runs
        with patch.object(sys, "argv", ["test", "--list"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result is None
        captured = capsys.readouterr()
        assert "Runs for test_exp" in captured.out
        assert "1 passed" in captured.out

    def test_default_output_dir_relative_to_file(self):
        """Default output_dir is relative to experiment file's directory."""
        # Create an experiment - __file__ should be this test file
        @experiment
        def my_exp(config):
            return config["x"]

        # The default output dir should be relative to the experiment file
        # Since my_exp is defined here, its __file__ is tests/test_experiment.py
        expected_dir = Path(__file__).parent / "out"
        assert my_exp._output_dir_default == expected_dir

    def test_output_dir_override_in_decorator(self, tmp_path):
        """Can override output_dir in decorator."""
        custom_dir = tmp_path / "custom_out"

        @experiment(output_dir=custom_dir)
        def my_exp(config):
            return config["x"]

        assert my_exp._output_dir_default == custom_dir


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

        @experiment(executor="subprocess")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            # Override subprocess with inline
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result == 10


def _can_subprocess_import_pyexp():
    """Check if subprocess can import pyexp (with inherited sys.path)."""
    import subprocess

    # Match what SubprocessExecutor does: pass sys.path via PYTHONPATH
    env = os.environ.copy()
    pythonpath = os.pathsep.join(sys.path)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    result = subprocess.run(
        [sys.executable, "-c", "import pyexp"],
        capture_output=True,
        env=env,
    )
    return result.returncode == 0


@pytest.mark.skipif(
    not _can_subprocess_import_pyexp(),
    reason="pyexp not importable in subprocess (not installed)",
)
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
        assert result.cfg["name"] == "failing"

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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Fork not available on this platform")
class TestForkExecution:
    """Tests for fork-based experiment execution (Unix only)."""

    def test_fork_runs_experiment(self, tmp_path):
        """Experiments run correctly with fork executor."""

        @experiment
        def my_exp(config):
            return {"value": config["x"] * 3}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="fork")

        assert result == 15

    def test_fork_handles_exception(self, tmp_path):
        """Exceptions in fork are captured."""

        @experiment
        def my_exp(config):
            raise RuntimeError("Fork error")

        @my_exp.configs
        def configs():
            return [{"name": "fail", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="fork")

        assert result.error is not None
        assert "RuntimeError" in result.error

    def test_fork_multiple_configs(self, tmp_path):
        """Multiple configs run correctly with fork executor."""

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


class TestDecoratorWithOut:
    """Tests for the decorator API with out parameter."""

    def test_decorator_with_out_parameter(self, tmp_path):
        """Decorator passes out when function has 2 parameters."""
        received_out = None

        @experiment
        def my_exp(cfg, out):
            nonlocal received_out
            received_out = out
            return {"value": cfg["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert received_out is not None
        assert isinstance(received_out, Path)
        assert received_out.exists()

    def test_decorator_without_out_parameter(self, tmp_path):
        """Decorator works with just cfg parameter."""

        @experiment
        def my_exp(cfg):
            return {"value": cfg["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline")

        assert result == 10


class TestClassBasedExperiment:
    """Tests for the class-based Experiment API."""

    def test_class_based_experiment_runs(self, tmp_path):
        """Class-based experiments can be run with ExperimentRunner."""

        class MyExperiment(Experiment):
            accuracy: float

            def experiment(self):
                self.accuracy = self.cfg["x"] * 2

            @staticmethod
            def configs():
                return [{"name": "test", "x": 5}]

            @staticmethod
            def report(results, out):
                return results[0].accuracy

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert result == 10

    def test_class_based_experiment_attributes(self, tmp_path):
        """Class-based experiments can set custom attributes."""

        class MyExperiment(Experiment):
            model_name: str
            accuracy: float
            epochs_run: int

            def experiment(self):
                self.model_name = "test_model"
                self.accuracy = 0.95
                self.epochs_run = self.cfg["epochs"]

            @staticmethod
            def configs():
                return [{"name": "test", "epochs": 10}]

            @staticmethod
            def report(results, out):
                exp = results[0]
                return {
                    "model": exp.model_name,
                    "accuracy": exp.accuracy,
                    "epochs": exp.epochs_run,
                }

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert result["model"] == "test_model"
        assert result["accuracy"] == 0.95
        assert result["epochs"] == 10

    def test_class_experiment_cfg_property(self, tmp_path):
        """Class-based experiments have access to cfg property."""

        class MyExperiment(Experiment):
            received_lr: float

            def experiment(self):
                self.received_lr = self.cfg.lr

            @staticmethod
            def configs():
                return [{"name": "test", "lr": 0.01}]

            @staticmethod
            def report(results, out):
                return results[0].received_lr

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert result == 0.01

    def test_class_experiment_out_property(self, tmp_path):
        """Class-based experiments have access to out property."""

        class MyExperiment(Experiment):
            out_path: Path

            def experiment(self):
                self.out_path = self.out

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return results[0].out_path

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert isinstance(result, Path)
        assert result.exists()


class TestCheckpointDecorator:
    """Tests for the @chkpt checkpoint decorator."""

    def test_chkpt_saves_checkpoint(self, tmp_path):
        """@chkpt saves a checkpoint file after method completes."""
        from pyexp import chkpt

        class MyExperiment(Experiment):
            model: dict

            @chkpt
            def train(self):
                self.model = {"weights": [1, 2, 3]}

            def experiment(self):
                self.train()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return results[0].model

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert result == {"weights": [1, 2, 3]}

        # Check checkpoint file was created
        exp_dir = tmp_path / "MyExperiment"
        checkpoint_dirs = list(exp_dir.rglob(".checkpoints"))
        assert len(checkpoint_dirs) == 1
        assert (checkpoint_dirs[0] / "train.pkl").exists()

    def test_chkpt_restores_from_checkpoint(self, tmp_path):
        """@chkpt restores state from checkpoint on subsequent calls."""
        from pyexp import chkpt
        import json

        # Use a file to track calls across pickle boundaries
        call_log = tmp_path / "call_log.json"
        call_log.write_text(json.dumps({"train": 0, "evaluate": 0}))

        class MyExperiment(Experiment):
            model: dict
            accuracy: float

            @chkpt
            def train(self):
                log = json.loads(call_log.read_text())
                log["train"] += 1
                call_log.write_text(json.dumps(log))
                self.model = {"weights": [1, 2, 3]}

            @chkpt
            def evaluate(self):
                log = json.loads(call_log.read_text())
                log["evaluate"] += 1
                call_log.write_text(json.dumps(log))
                self.accuracy = 0.95

            def experiment(self):
                self.train()
                self.evaluate()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return {"model": results[0].model, "accuracy": results[0].accuracy}

        # First run
        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result1 = runner.run(executor="inline")

        log = json.loads(call_log.read_text())
        assert log["train"] == 1
        assert log["evaluate"] == 1
        assert result1["accuracy"] == 0.95

        # Get the timestamp for continue
        exp_dir = tmp_path / "MyExperiment"
        timestamps = [d for d in exp_dir.iterdir() if d.is_dir()]
        timestamp = timestamps[0].name

        # Delete experiment.pkl to force re-run but keep checkpoints
        for exp_pkl in exp_dir.rglob("experiment.pkl"):
            exp_pkl.unlink()

        # Second run with --continue should use checkpoints
        with patch.object(sys, "argv", ["test", f"--continue={timestamp}"]):
            result2 = runner.run(executor="inline")

        log = json.loads(call_log.read_text())
        # train and evaluate should NOT be called again (restored from checkpoint)
        assert log["train"] == 1
        assert log["evaluate"] == 1
        assert result2["accuracy"] == 0.95

    def test_chkpt_with_retry(self, tmp_path):
        """@chkpt(retry=N) retries on failure."""
        from pyexp import chkpt
        import json

        # Use a file to track attempts across pickle boundaries
        attempt_log = tmp_path / "attempt_log.json"
        attempt_log.write_text(json.dumps({"train": 0}))

        class MyExperiment(Experiment):
            model: dict

            @chkpt(retry=3)
            def train(self):
                log = json.loads(attempt_log.read_text())
                log["train"] += 1
                attempt_log.write_text(json.dumps(log))
                if log["train"] < 3:
                    raise ValueError("Training failed")
                self.model = {"weights": [1, 2, 3]}

            def experiment(self):
                self.train()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return results[0].model

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        log = json.loads(attempt_log.read_text())
        # Should succeed on 3rd attempt
        assert log["train"] == 3
        assert result == {"weights": [1, 2, 3]}

    def test_chkpt_retry_exhausted_raises(self, tmp_path):
        """@chkpt raises after retry count exhausted."""
        from pyexp import chkpt

        class MyExperiment(Experiment):
            @chkpt(retry=2)
            def train(self):
                raise ValueError("Always fails")

            def experiment(self):
                self.train()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return results[0]

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        # Experiment should have error set
        assert result.error is not None
        assert "ValueError" in result.error

    def test_chkpt_preserves_return_value(self, tmp_path):
        """@chkpt caches and restores return values."""
        from pyexp import chkpt

        class MyExperiment(Experiment):
            result_value: int

            @chkpt
            def compute(self):
                return 42

            def experiment(self):
                self.result_value = self.compute()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return results[0].result_value

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert result == 42

    def test_chkpt_multiple_sequential(self, tmp_path):
        """Multiple @chkpt methods work correctly in sequence."""
        from pyexp import chkpt

        call_order = []

        class MyExperiment(Experiment):
            step1_done: bool
            step2_done: bool
            step3_done: bool

            @chkpt
            def step1(self):
                call_order.append("step1")
                self.step1_done = True

            @chkpt
            def step2(self):
                call_order.append("step2")
                self.step2_done = True

            @chkpt
            def step3(self):
                call_order.append("step3")
                self.step3_done = True

            def experiment(self):
                self.step1()
                self.step2()
                self.step3()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                exp = results[0]
                return (exp.step1_done, exp.step2_done, exp.step3_done)

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert call_order == ["step1", "step2", "step3"]
        assert result == (True, True, True)

        # Check all checkpoint files exist
        exp_dir = tmp_path / "MyExperiment"
        checkpoint_dirs = list(exp_dir.rglob(".checkpoints"))
        assert len(checkpoint_dirs) == 1
        checkpoint_files = list(checkpoint_dirs[0].iterdir())
        assert len(checkpoint_files) == 3
        assert {f.name for f in checkpoint_files} == {"step1.pkl", "step2.pkl", "step3.pkl"}

    def test_chkpt_state_isolation(self, tmp_path):
        """@chkpt doesn't affect runner-managed attributes."""
        from pyexp import chkpt

        class MyExperiment(Experiment):
            user_attr: str

            @chkpt
            def setup(self):
                self.user_attr = "modified"

            def experiment(self):
                # Store original cfg reference
                original_cfg = self.cfg
                self.setup()
                # cfg should still be the same object
                assert self.cfg is original_cfg

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return results[0].user_attr

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert result == "modified"

    def test_chkpt_partial_failure_resume(self, tmp_path):
        """Resume after partial failure skips completed checkpoints."""
        from pyexp import chkpt
        import json

        # Use a file to track calls across pickle boundaries
        call_log = tmp_path / "call_log.json"
        call_log.write_text(json.dumps({"step1": 0, "step2": 0, "should_fail": True}))

        class MyExperiment(Experiment):
            step1_result: int
            step2_result: int

            @chkpt
            def step1(self):
                log = json.loads(call_log.read_text())
                log["step1"] += 1
                call_log.write_text(json.dumps(log))
                self.step1_result = 100

            @chkpt
            def step2(self):
                log = json.loads(call_log.read_text())
                log["step2"] += 1
                call_log.write_text(json.dumps(log))
                if log.get("should_fail", False):
                    raise ValueError("Step 2 failed")
                self.step2_result = 200

            def experiment(self):
                self.step1()
                self.step2()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                return results[0]

        # First run - step1 succeeds, step2 fails (disable retries to simplify)
        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path, retry=0)
            result1 = runner.run(executor="inline")

        log = json.loads(call_log.read_text())
        assert log["step1"] == 1
        assert log["step2"] == 1
        assert result1.error is not None

        # Get timestamp
        exp_dir = tmp_path / "MyExperiment"
        timestamps = [d for d in exp_dir.iterdir() if d.is_dir()]
        timestamp = timestamps[0].name

        # Delete experiment.pkl but keep checkpoints
        for exp_pkl in exp_dir.rglob("experiment.pkl"):
            exp_pkl.unlink()

        # Verify step1 checkpoint exists, step2 does not
        checkpoint_dir = list(exp_dir.rglob(".checkpoints"))[0]
        assert (checkpoint_dir / "step1.pkl").exists()
        assert not (checkpoint_dir / "step2.pkl").exists()

        # Fix step2 and resume
        log["should_fail"] = False
        call_log.write_text(json.dumps(log))

        with patch.object(sys, "argv", ["test", f"--continue={timestamp}"]):
            result2 = runner.run(executor="inline")

        log = json.loads(call_log.read_text())
        # step1 should NOT be called again (checkpoint hit)
        assert log["step1"] == 1  # Still 1, not re-run
        # step2 should be called again (no checkpoint due to previous failure)
        assert log["step2"] == 2  # Called again
        assert result2.step1_result == 100
        assert result2.step2_result == 200

    def test_chkpt_with_complex_state(self, tmp_path):
        """@chkpt handles complex state (nested dicts, lists)."""
        from pyexp import chkpt

        class MyExperiment(Experiment):
            model: dict
            history: list
            metadata: dict

            @chkpt
            def train(self):
                self.model = {
                    "layers": [{"weights": [1, 2, 3]}, {"weights": [4, 5, 6]}],
                    "config": {"lr": 0.01, "epochs": 100},
                }
                self.history = [0.5, 0.4, 0.3, 0.2]
                self.metadata = {"version": "1.0", "tags": ["test", "experiment"]}

            def experiment(self):
                self.train()

            @staticmethod
            def configs():
                return [{"name": "test", "x": 1}]

            @staticmethod
            def report(results, out):
                exp = results[0]
                return {
                    "model": exp.model,
                    "history": exp.history,
                    "metadata": exp.metadata,
                }

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        assert result["model"]["layers"][0]["weights"] == [1, 2, 3]
        assert result["history"] == [0.5, 0.4, 0.3, 0.2]
        assert result["metadata"]["tags"] == ["test", "experiment"]

    def test_chkpt_multiple_configs(self, tmp_path):
        """@chkpt works correctly with multiple configs."""
        from pyexp import chkpt

        class MyExperiment(Experiment):
            lr_used: float

            @chkpt
            def train(self):
                self.lr_used = self.cfg.lr

            def experiment(self):
                self.train()

            @staticmethod
            def configs():
                return [
                    {"name": "fast", "lr": 0.1},
                    {"name": "slow", "lr": 0.01},
                ]

            @staticmethod
            def report(results, out):
                return [exp.lr_used for exp in results]

        with patch.object(sys, "argv", ["test"]):
            runner = ExperimentRunner(MyExperiment, output_dir=tmp_path)
            result = runner.run(executor="inline")

        # Verify results are correct
        assert result == [0.1, 0.01]

        # Check each config has its own checkpoint directory
        exp_dir = tmp_path / "MyExperiment"
        checkpoint_dirs = list(exp_dir.rglob(".checkpoints"))
        assert len(checkpoint_dirs) == 2

        # Verify each checkpoint directory has train.pkl
        for checkpoint_dir in checkpoint_dirs:
            assert (checkpoint_dir / "train.pkl").exists()


def _init_git_repo(path: Path) -> None:
    """Initialize a git repo with an initial commit at the given path."""
    subprocess.check_call(["git", "init"], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@test.com"], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.name", "Test"], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Create a file and initial commit
    (path / "README.md").write_text("init")
    subprocess.check_call(["git", "add", "."], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "commit", "-m", "init"], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _init_git_repo_no_commits(path: Path) -> None:
    """Initialize a git repo with NO commits at the given path."""
    subprocess.check_call(["git", "init"], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.email", "test@test.com"], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call(["git", "config", "user.name", "Test"], cwd=path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Add a file but don't commit
    (path / "README.md").write_text("init")


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo and chdir into it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    old_cwd = os.getcwd()
    os.chdir(repo)
    yield repo
    os.chdir(old_cwd)


@pytest.fixture
def git_repo_no_commits(tmp_path):
    """Create a temporary git repo with no commits and chdir into it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo_no_commits(repo)
    old_cwd = os.getcwd()
    os.chdir(repo)
    yield repo
    os.chdir(old_cwd)


class TestSourceSnapshotting:
    """Tests for git-based source code snapshotting."""

    def test_snapshot_created_on_fresh_run(self, git_repo, tmp_path):
        """.src snapshot dir should be created on a fresh run with stash enabled."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        # Find run dir and check .src exists
        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        src_dir = timestamp_dir / ".src"
        assert src_dir.exists(), ".src snapshot directory should be created"
        assert src_dir.is_dir()

    def test_no_stash_prevents_snapshot(self, git_repo, tmp_path):
        """--no-stash should prevent snapshot creation."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test", "--no-stash"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Find run dir and check .src does NOT exist
        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        src_dir = timestamp_dir / ".src"
        assert not src_dir.exists(), ".src should not be created with --no-stash"

    def test_commit_hash_in_runs_json(self, git_repo, tmp_path):
        """Commit hash should be stored in runs.json."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        # Find run dir and check runs.json
        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        runs_json = json.loads((timestamp_dir / "runs.json").read_text())
        assert "commit" in runs_json, "runs.json should contain commit hash"
        assert len(runs_json["commit"]) == 40, "commit hash should be 40 chars"

    def test_no_commit_hash_without_stash(self, git_repo, tmp_path):
        """runs.json should not contain commit hash when stash is disabled."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test", "--no-stash"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        runs_json = json.loads((timestamp_dir / "runs.json").read_text())
        assert "commit" not in runs_json

    def test_snapshot_persists_after_run(self, git_repo, tmp_path):
        """Snapshot should persist after run completes (kept as artifact)."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        with patch.object(sys, "argv", ["test"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        assert result == 1

        # Snapshot should still exist after run
        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        src_dir = timestamp_dir / ".src"
        assert src_dir.exists(), ".src should persist after run completes"

    def test_snapshot_contains_repo_files(self, git_repo, tmp_path):
        """Snapshot should contain copies of repository files."""
        # Add a source file to the repo
        (git_repo / "source.py").write_text("x = 42")

        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        # Check snapshot has the source file
        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        src_dir = timestamp_dir / ".src"
        assert (src_dir / "source.py").exists()
        assert (src_dir / "source.py").read_text() == "x = 42"

    def test_continue_reuses_existing_snapshot(self, git_repo, tmp_path):
        """--continue should detect and reuse existing .src snapshot."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results[0].result["value"]

        # First run creates snapshot
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        timestamp = timestamp_dir.name
        src_dir = timestamp_dir / ".src"
        assert src_dir.exists()

        # Continue run should reuse snapshot (not fail)
        with patch.object(sys, "argv", ["test", f"--continue={timestamp}"]):
            result = my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        assert result == 1
        assert src_dir.exists()

    def test_list_runs_excludes_src(self, git_repo, tmp_path, capsys):
        """--list should not count .src as a config directory."""
        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        @my_exp.report
        def report(results, out):
            return results

        # Create a run (with snapshot)
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        # List runs
        with patch.object(sys, "argv", ["test", "--list"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        captured = capsys.readouterr()
        assert "1 passed" in captured.out
        assert "(1 configs)" in captured.out

    def test_snapshot_works_in_repo_with_no_commits(self, git_repo_no_commits, tmp_path):
        """Stash and snapshot should work even in a repo with no prior commits."""
        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        # Find run dir and check .src exists
        base_dir = tmp_path / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        src_dir = timestamp_dir / ".src"
        assert src_dir.exists(), ".src snapshot should be created even without prior commits"

        # Check commit hash is in runs.json
        runs_json = json.loads((timestamp_dir / "runs.json").read_text())
        assert "commit" in runs_json
        assert len(runs_json["commit"]) == 40

    def test_snapshot_symlinks_submodules(self, git_repo, tmp_path):
        """Submodule directories in snapshot should be symlinked to originals."""
        # Create a separate repo to use as a submodule
        sub_repo = tmp_path / "sub_repo"
        sub_repo.mkdir()
        _init_git_repo(sub_repo)
        (sub_repo / "lib.py").write_text("value = 99")
        subprocess.check_call(["git", "add", "."], cwd=sub_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "commit", "-m", "add lib"], cwd=sub_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Add it as a submodule in the main repo
        subprocess.check_call(
            ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub_repo), "mylib"],
            cwd=git_repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(["git", "commit", "-m", "add submodule"], cwd=git_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        @experiment
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path / "out", executor="inline", stash=True)

        # Find the .src dir
        base_dir = tmp_path / "out" / "my_exp"
        timestamp_dir = sorted(base_dir.iterdir())[0]
        src_dir = timestamp_dir / ".src"

        # Submodule should be a symlink pointing to the real submodule dir
        mylib_in_snapshot = src_dir / "mylib"
        assert mylib_in_snapshot.is_symlink(), "submodule should be symlinked"
        assert mylib_in_snapshot.resolve() == (git_repo / "mylib").resolve()
        assert (mylib_in_snapshot / "lib.py").read_text() == "value = 99"

