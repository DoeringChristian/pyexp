"""Tests for experiment framework."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pyexp import Config, Runs, experiment, sweep, Result, ExperimentRunner, Experiment


class TestExperimentDecorator:
    """Tests for the @experiment decorator syntax."""

    def test_decorator_returns_experiment(self):
        @experiment
        def my_exp(config):
            return config["x"]

        assert isinstance(my_exp, Experiment)

    def test_experiment_fn_accessible(self):
        @experiment
        def my_exp(config):
            return config["x"] * 2

        assert my_exp._fn is not None

    def test_configs_decorator(self):
        @experiment
        def my_exp(config):
            return config["x"]

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        assert my_exp._configs_fn is not None

    def test_isinstance_experiment(self):
        @experiment
        def my_exp(config):
            return config["x"]

        assert isinstance(my_exp, Experiment)
        assert not isinstance(my_exp, ExperimentRunner)


class TestExperimentRun:
    """Tests for Experiment.run() execution."""

    def test_run_executes_pipeline(self, tmp_path):
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 10

    def test_run_with_passed_configs(self, tmp_path):
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] + 1}

        def my_configs():
            return [{"name": "t", "x": 10}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(
                configs=my_configs, output_dir=tmp_path, executor="inline"
            )

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 11

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

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # --continue should use cached results
        with patch.object(sys, "argv", ["test", "--continue"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 42
        assert call_count == 1  # Only called once due to caching

    def test_continue_uses_saved_configs(self, tmp_path):
        """--continue should use saved configs, not recompute from configs function."""
        config_value = [42]  # Mutable to allow modification

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": config_value[0]}]

        # First run with x=42
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results1 = my_exp.results(output_dir=tmp_path)
        assert results1[0].cfg["x"] == 42

        # Change configs function to return different value
        config_value[0] = 999

        # --continue should still use saved config with x=42
        with patch.object(sys, "argv", ["test", "--continue"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results2 = my_exp.results(output_dir=tmp_path)
        assert results2[0].cfg["x"] == 42  # Should be original value, not 999

    def test_run_creates_output_dir(self, tmp_path):
        out_dir = tmp_path / "nested" / "output"

        @experiment
        def my_exp(config):
            return {"x": 1}

        @my_exp.configs
        def configs():
            return [{"name": "dir", "x": 1}]

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

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert "Hello from experiment" in results[0].log

    def test_config_receives_out(self, tmp_path):
        @experiment
        def my_exp(config):
            return {"x": 1}

        @my_exp.configs
        def configs():
            return [{"name": "outdir", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].out is not None
        assert isinstance(results[0].out, Path)
        assert results[0].out.exists()

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

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert isinstance(received_config, Config)
        assert received_config.nested.a == 1

    def test_run_no_configs_raises(self, tmp_path):
        @experiment
        def my_exp(config):
            return 1

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(RuntimeError, match="No configs function"):
                my_exp.run(output_dir=tmp_path)

    def test_run_succeeds_without_report(self, tmp_path):
        """Running without a report function should succeed."""

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

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(AssertionError, match="out"):
                my_exp.run(output_dir=tmp_path)

    def test_multiple_configs(self, tmp_path):
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        values = [r.result["value"] for r in results]
        assert values == [4, 9, 16]

    def test_results_are_runs(self, tmp_path):
        """Results should be Runs instances."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "a", "x": 1}, {"name": "b", "x": 2}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert isinstance(results, Runs)
        assert len(results) == 2

    def test_results_contain_config_and_name(self, tmp_path):
        """Each result should contain the cfg and name."""

        @experiment
        def my_exp(config):
            return {"accuracy": 0.95}

        @my_exp.configs
        def configs():
            return [{"name": "test", "lr": 0.01, "epochs": 10}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        result = results[0]
        assert result.cfg["name"] == "test"
        assert result.cfg["lr"] == 0.01
        assert result.cfg["epochs"] == 10
        assert result.result["accuracy"] == 0.95
        assert result.out.exists()
        assert result.name == "test"

    def test_results_filterable_by_config(self, tmp_path):
        """Results should be filterable by config values."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "x1", "x": 1}, {"name": "x2", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "y10", "y": 10}, {"name": "y20", "y": 20}])
            return cfgs

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert len(results) == 4

        x1_results = results[{"cfg.x": 1}]
        assert all(r.cfg["x"] == 1 for r in x1_results)

        y10_results = results[{"cfg.y": 10}]
        assert all(r.cfg["y"] == 10 for r in y10_results)

    def test_results_are_1d(self, tmp_path):
        """Result Runs are always 1D regardless of sweep shape."""

        @experiment
        def my_exp(config):
            return {"val": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "c", "y": 10}, {"name": "d", "y": 20}])
            return cfgs

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert len(results) == 4
        exp_a_c = results["exp_a_c"]
        assert exp_a_c.cfg["x"] == 1
        assert exp_a_c.result["val"] == 11

    def test_non_dict_result_wrapped_in_result(self, tmp_path):
        """Non-dict results are stored in .result attribute."""

        @experiment
        def my_exp(config):
            return config["x"] * 2  # Returns int, not dict

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        result = results[0]
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
                {"name": "lr_0.01_batch_32", "lr": 0.01, "batch": 32},
                {"name": "lr_0.01_batch_64", "lr": 0.01, "batch": 64},
                {"name": "lr_0.001_batch_32", "lr": 0.001, "batch": 32},
                {"name": "lr_0.001_batch_64", "lr": 0.001, "batch": 64},
            ]

        # First run all configs
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Now filter to only lr_0.01 configs using --continue
        runs.clear()
        with patch.object(sys, "argv", ["test", "--continue", "--filter", "lr_0\\.01"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Since all are cached, runs list stays empty
        assert len(runs) == 0

    def test_filter_runs_only_matching(self, tmp_path):
        """--filter should only execute matching configs on fresh run."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"name": config["name"], "ran": True}

        @my_exp.configs
        def configs():
            return [
                {"name": "exp_a", "x": 1},
                {"name": "exp_b", "x": 2},
                {"name": "other_c", "x": 3},
            ]

        # Run with filter - only exp_* should run
        with patch.object(sys, "argv", ["test", "--filter", "^exp_"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Check which experiments actually ran by looking for experiment.pkl files
        base_dir = tmp_path / "my_exp"
        ran_names = []
        for exp_pkl in base_dir.rglob("experiment.pkl"):
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

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)

        assert len(results) == 1
        assert results[0].cfg["name"] == "test"
        assert results[0].cfg["x"] == 5
        assert results[0].result["value"] == 10
        assert results[0].out is not None
        assert results[0].out.exists()

    def test_results_loads_specific_timestamp(self, tmp_path):
        """results() can load a specific timestamp."""
        import time

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        # Run first experiment
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Get the first timestamp from batch manifests
        base_dir = tmp_path / "my_exp"
        batches_dir = base_dir / ".batches"
        first_timestamp = sorted(batches_dir.glob("*.json"))[0].stem

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

    def test_results_always_1d(self, tmp_path):
        """results() always returns a 1D Runs regardless of sweep shape."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "base"}]
            cfgs = sweep(cfgs, [{"name": "x1", "x": 1}, {"name": "x2", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "y10", "y": 10}, {"name": "y20", "y": 20}])
            return cfgs

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert len(results) == 4

    def test_results_no_runs_returns_empty(self, tmp_path):
        """results() returns empty Runs when no runs exist."""

        @experiment
        def my_exp(config):
            return {"value": 1}

        results = my_exp.results(output_dir=tmp_path)
        assert isinstance(results, Runs)
        assert len(results) == 0

    def test_results_invalid_timestamp_raises(self, tmp_path):
        """results() raises FileNotFoundError for invalid timestamp."""

        @experiment
        def my_exp(config):
            return {"value": 1}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        with pytest.raises(FileNotFoundError, match="Run not found"):
            my_exp.results(timestamp="1999-01-01_00-00-00", output_dir=tmp_path)

    def test_batch_manifest_saved(self, tmp_path):
        """run() saves .batches/<timestamp>.json with run names."""
        import json

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 1},
                {"name": "b", "x": 2},
            ]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        batches_dir = base_dir / ".batches"
        assert batches_dir.exists()

        manifests = list(batches_dir.glob("*.json"))
        assert len(manifests) == 1

        data = json.loads(manifests[0].read_text())
        assert len(data["runs"]) == 2
        assert data["runs"] == ["a", "b"]
        assert "timestamp" in data

        timestamp = data["timestamp"]
        for run_name in data["runs"]:
            config_json = base_dir / run_name / timestamp / "config.json"
            assert config_json.exists()
            config_data = json.loads(config_json.read_text())
            assert "x" in config_data


class TestGetitem:
    """Tests for Experiment.__getitem__."""

    def test_getitem_loads_result(self, tmp_path):
        """my_exp['test'] loads a result by name."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        # Override output_dir for results lookup
        my_exp._output_dir = tmp_path
        result = my_exp["test"]
        assert result.result["value"] == 10
        assert result.name == "test"

    def test_getitem_glob_pattern(self, tmp_path):
        """my_exp['pretrain.*'] returns matching results."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "pretrain_a", "x": 1},
                {"name": "pretrain_b", "x": 2},
                {"name": "finetune", "x": 3},
            ]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        my_exp._output_dir = tmp_path
        pretrain_results = my_exp["pretrain.*"]
        assert isinstance(pretrain_results, Runs)
        assert len(pretrain_results) == 2


class TestFinishedFlag:
    """Tests for the Result.finished property."""

    def test_finished_false_by_default(self):
        """New Result instances have finished=False."""
        from pyexp import Config
        exp = Result(cfg=Config({"name": "test"}), name="test", out=Path("/tmp/test"))
        assert exp.finished is False

    def test_finished_true_after_successful_run(self, tmp_path):
        """finished=True after a successful experiment run."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert len(results) == 1
        assert results[0].finished is True

    def test_finished_true_after_failed_run(self, tmp_path):
        """finished=True after a failed experiment run."""

        @experiment(name="my_exp")
        def my_exp(config):
            raise ValueError("intentional failure")

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", retry=0)

        results = my_exp.results(output_dir=tmp_path)
        assert len(results) == 1
        assert results[0].finished is True
        assert results[0].error is not None

    def test_initial_pkl_saved_before_execution(self, tmp_path):
        """experiment.pkl should exist on disk even before execution completes."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": 1}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].finished is True

    def test_continue_skips_finished_experiments(self, tmp_path):
        """--continue should skip experiments that are already finished."""
        call_count = 0

        @experiment(name="my_exp")
        def my_exp(config):
            nonlocal call_count
            call_count += 1
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        # First run
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert call_count == 1

        # Continue run should skip (already finished)
        with patch.object(sys, "argv", ["test", "--continue"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        assert call_count == 1  # Not called again

    def test_results_finished_only(self, tmp_path):
        """results(finished=True) with explicit timestamp only returns finished experiments."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "a", "x": 1}, {"name": "b", "x": 2}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        timestamp = sorted((base_dir / ".batches").glob("*.json"))[0].stem

        # All finished - should return both
        results = my_exp.results(timestamp=timestamp, output_dir=tmp_path, finished=True)
        assert len(results) == 2

        # Remove .finished from one experiment to simulate unfinished
        for d in base_dir.rglob(".finished"):
            d.unlink()
            break

        # With finished=True, only the finished one should be returned
        results = my_exp.results(timestamp=timestamp, output_dir=tmp_path, finished=True)
        assert len(results) == 1

        # Without finished flag, both should still be returned
        results = my_exp.results(timestamp=timestamp, output_dir=tmp_path)
        assert len(results) == 2

    def test_results_finished_searches_back_in_time(self, tmp_path):
        """results(finished=True) searches back to find a fully finished batch."""
        import time

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        # First run - will be finished
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        time.sleep(1.1)

        # Second run - will also be finished
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        timestamps = sorted((base_dir / ".batches").glob("*.json"))
        latest_ts = timestamps[-1].stem

        # Remove .finished from latest run to simulate unfinished
        for d in base_dir.rglob(".finished"):
            if latest_ts in str(d):
                d.unlink()

        # results(finished=True) should skip the latest and find the first
        results = my_exp.results(output_dir=tmp_path, finished=True)
        assert len(results) == 1

        first_ts = timestamps[0].stem
        assert first_ts in str(results[0].out)

    def test_results_finished_returns_empty_when_none_finished(self, tmp_path):
        """results(finished=True) returns empty Runs when no batch is fully finished."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        for d in base_dir.rglob(".finished"):
            d.unlink()

        results = my_exp.results(output_dir=tmp_path, finished=True)
        assert isinstance(results, Runs)
        assert len(results) == 0


class TestOutputFolderStructure:
    """Tests for output folder naming and timestamp features."""

    def test_default_name_is_filename_functionname(self):
        """Experiment name defaults to <filename>.<function_name>."""

        @experiment
        def my_custom_experiment(config):
            return config["x"]

        assert my_custom_experiment._name == "test_experiment.my_custom_experiment"

    def test_custom_name_in_decorator(self):
        """Can set custom name in decorator."""

        @experiment(name="mnist_classifier")
        def my_exp(config):
            return config["x"]

        assert my_exp._name == "mnist_classifier"

    def test_new_directory_layout(self, tmp_path):
        """Output uses <experiment>/<run_name>/<timestamp>/ layout."""

        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        exp_dir = tmp_path / "test_exp"
        assert exp_dir.exists()

        cfg_dir = exp_dir / "cfg"
        assert cfg_dir.exists()

        timestamp_dirs = [d for d in cfg_dir.iterdir() if d.is_dir()]
        assert len(timestamp_dirs) == 1

        assert (timestamp_dirs[0] / "config.json").exists()
        assert (timestamp_dirs[0] / "experiment.pkl").exists()

        batches_dir = exp_dir / ".batches"
        assert batches_dir.exists()

    def test_continue_specific_timestamp(self, tmp_path):
        """--continue=TIMESTAMP continues a specific run."""

        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        # First run
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        exp_dir = tmp_path / "test_exp"
        timestamp = sorted((exp_dir / ".batches").glob("*.json"))[0].stem

        # Second run with --continue should use cache
        with patch.object(sys, "argv", ["test", f"--continue={timestamp}"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 1

    def test_continue_uses_latest_timestamp(self, tmp_path):
        """--continue uses the most recent batch timestamp."""

        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

        import time

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")
        time.sleep(1.1)
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        exp_dir = tmp_path / "test_exp"
        batches = sorted((exp_dir / ".batches").glob("*.json"))
        assert len(batches) == 2

        # --continue should use the latest
        with patch.object(sys, "argv", ["test", "--continue"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

    def test_continue_no_previous_runs_raises(self, tmp_path):
        """--continue raises error when no previous runs exist."""

        @experiment(name="new_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

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

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, name="override_name", executor="inline")

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
        @experiment
        def my_exp(config):
            return config["x"]

        expected_dir = Path(__file__).parent / "out"
        assert my_exp._output_dir == expected_dir

    def test_output_dir_override_in_decorator(self, tmp_path):
        """Can override output_dir in decorator."""
        custom_dir = tmp_path / "custom_out"

        @experiment(output_dir=custom_dir)
        def my_exp(config):
            return config["x"]

        assert my_exp._output_dir == custom_dir


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

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path)

        assert call_count == 1
        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 5

    def test_run_can_override_decorator_default(self, tmp_path):
        """run(executor=...) can override the decorator default."""

        @experiment(executor="subprocess")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 10


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

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * 2}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="subprocess")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 10

    def test_subprocess_handles_exception(self, tmp_path):
        """Exceptions in subprocess are captured and returned as error results."""

        @experiment
        def my_exp(config):
            raise ValueError("Test error")

        @my_exp.configs
        def configs():
            return [{"name": "failing", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="subprocess")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].error is not None
        assert "ValueError" in results[0].error
        assert "Test error" in results[0].error
        assert results[0].cfg["name"] == "failing"

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

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="subprocess")

        results = my_exp.results(output_dir=tmp_path)

        assert results[0].result["value"] == 1
        assert results[2].result["value"] == 3
        assert results[1].error is not None
        assert "ValueError" in results[1].error

    def test_subprocess_multiple_configs(self, tmp_path):
        """Multiple configs all run in separate subprocesses."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="subprocess")

        results = my_exp.results(output_dir=tmp_path)
        values = [r.result["value"] for r in results]
        assert values == [4, 9, 16]

    def test_subprocess_with_sweep(self, tmp_path):
        """Subprocess execution works with sweep configurations."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] + config["y"]}

        @my_exp.configs
        def configs():
            cfgs = [{"name": "exp"}]
            cfgs = sweep(cfgs, [{"name": "a", "x": 1}, {"name": "b", "x": 2}])
            cfgs = sweep(cfgs, [{"name": "c", "y": 10}, {"name": "d", "y": 20}])
            return cfgs

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="subprocess")

        results = my_exp.results(output_dir=tmp_path)
        assert len(results) == 4
        assert results["exp_a_c"].result["value"] == 11  # x=1, y=10
        assert results["exp_b_d"].result["value"] == 22  # x=2, y=20


@pytest.mark.skipif(not hasattr(os, "fork"), reason="Fork not available on this platform")
class TestForkExecution:
    """Tests for fork-based experiment execution (Unix only)."""

    def test_fork_runs_experiment(self, tmp_path):
        """Experiments run correctly with fork executor."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] * 3}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 5}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="fork")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 15

    def test_fork_handles_exception(self, tmp_path):
        """Exceptions in fork are captured."""

        @experiment
        def my_exp(config):
            raise RuntimeError("Fork error")

        @my_exp.configs
        def configs():
            return [{"name": "fail", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="fork")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].error is not None
        assert "RuntimeError" in results[0].error

    def test_fork_multiple_configs(self, tmp_path):
        """Multiple configs run correctly with fork executor."""

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"] ** 2}

        @my_exp.configs
        def configs():
            return [
                {"name": "a", "x": 2},
                {"name": "b", "x": 3},
                {"name": "c", "x": 4},
            ]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="fork")

        results = my_exp.results(output_dir=tmp_path)
        values = [r.result["value"] for r in results]
        assert values == [4, 9, 16]


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

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 10


class TestSubmitAPI:
    """Tests for the ExperimentRunner.submit() API."""

    def test_submit_runs_experiments(self, tmp_path):
        """submit() API runs experiments with per-config functions."""

        def train_fn(cfg):
            return {"value": cfg["x"] * 2}

        runner = ExperimentRunner(name="my_exp", output_dir=tmp_path)
        runner.submit(train_fn, {"name": "a", "x": 5})
        runner.submit(train_fn, {"name": "b", "x": 10})

        with patch.object(sys, "argv", ["test"]):
            runner.run(executor="inline")

        from pyexp.runner import _get_latest_timestamp, _load_experiments
        base_dir = tmp_path / "my_exp"
        ts = _get_latest_timestamp(base_dir)
        results = _load_experiments(base_dir, ts)
        values = [r.result["value"] for r in results]
        assert values == [10, 20]

    def test_result_dataclass_fields(self):
        """Result dataclass has expected fields."""
        from pyexp import Config

        exp = Result(
            cfg=Config({"name": "test", "lr": 0.01}),
            name="test",
            out=Path("/tmp/test"),
        )
        assert exp.cfg["lr"] == 0.01
        assert exp.name == "test"
        assert exp.out == Path("/tmp/test")
        assert exp.result is None
        assert exp.error is None
        assert exp.log == ""
        assert exp.finished is False
        assert exp.skipped is False

    def test_submit_no_submissions_raises(self, tmp_path):
        """Runner with no submissions raises RuntimeError."""
        runner = ExperimentRunner(name="my_exp", output_dir=tmp_path)

        with patch.object(sys, "argv", ["test"]):
            with pytest.raises(RuntimeError, match="No experiments submitted"):
                runner.run(executor="inline")


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

    def _get_snapshot_dir(self, base_dir):
        """Helper to find the snapshot directory under .snapshots/."""
        snapshots_dir = base_dir / ".snapshots"
        if not snapshots_dir.exists():
            return None
        commit_dirs = list(snapshots_dir.iterdir())
        if not commit_dirs:
            return None
        return commit_dirs[0]

    def test_snapshot_created_on_fresh_run(self, git_repo, tmp_path):
        """.snapshots/<commit>/ should be created on a fresh run with stash enabled."""
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        snapshot_dir = self._get_snapshot_dir(base_dir)
        assert snapshot_dir is not None, ".snapshots/<commit>/ should be created"
        assert snapshot_dir.is_dir()

    def test_no_stash_prevents_snapshot(self, git_repo, tmp_path):
        """--no-stash should prevent snapshot creation."""
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test", "--no-stash"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        snapshots_dir = base_dir / ".snapshots"
        assert not snapshots_dir.exists(), ".snapshots should not be created with --no-stash"

    def test_commit_hash_in_batch_manifest(self, git_repo, tmp_path):
        """Commit hash should be stored in batch manifest."""
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        manifest = list((base_dir / ".batches").glob("*.json"))[0]
        data = json.loads(manifest.read_text())
        assert "commit" in data, "batch manifest should contain commit hash"
        assert len(data["commit"]) == 40, "commit hash should be 40 chars"

    def test_commit_hash_in_each_run_dir(self, git_repo, tmp_path):
        """Each run directory should have a .commit file with the commit hash."""
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "fast", "x": 1}, {"name": "slow", "x": 2}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        manifest = list((base_dir / ".batches").glob("*.json"))[0]
        data = json.loads(manifest.read_text())
        commit = data["commit"]

        for run_name in data["runs"]:
            commit_file = base_dir / run_name / data["timestamp"] / ".commit"
            assert commit_file.exists(), f".commit should exist in {run_name} run dir"
            assert commit_file.read_text() == commit

    def test_no_commit_hash_without_stash(self, git_repo, tmp_path):
        """Batch manifest should not contain commit hash when stash is disabled."""
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test", "--no-stash"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        manifest = list((base_dir / ".batches").glob("*.json"))[0]
        data = json.loads(manifest.read_text())
        assert "commit" not in data

    def test_snapshot_persists_after_run(self, git_repo, tmp_path):
        """Snapshot should persist after run completes (kept as artifact)."""
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        snapshot_dir = self._get_snapshot_dir(base_dir)
        assert snapshot_dir is not None, ".snapshots/<commit>/ should persist after run"

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 1

    def test_snapshot_contains_repo_files(self, git_repo, tmp_path):
        """Snapshot should contain copies of repository files."""
        (git_repo / "source.py").write_text("x = 42")

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        snapshot_dir = self._get_snapshot_dir(base_dir)
        assert (snapshot_dir / "source.py").exists()
        assert (snapshot_dir / "source.py").read_text() == "x = 42"

    def test_continue_reuses_existing_snapshot(self, git_repo, tmp_path):
        """--continue should detect and reuse existing snapshot."""
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        # First run creates snapshot
        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        timestamp = sorted((base_dir / ".batches").glob("*.json"))[0].stem
        snapshot_dir = self._get_snapshot_dir(base_dir)
        assert snapshot_dir is not None

        # Continue run should reuse snapshot (not fail)
        with patch.object(sys, "argv", ["test", f"--continue={timestamp}"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        results = my_exp.results(output_dir=tmp_path)
        assert results[0].result["value"] == 1
        assert snapshot_dir.exists()

    def test_list_runs_correct_count(self, git_repo, tmp_path, capsys):
        """--list should show correct config count."""
        @experiment(name="test_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "cfg", "x": 1}]

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
        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="inline", stash=True)

        base_dir = tmp_path / "my_exp"
        snapshot_dir = self._get_snapshot_dir(base_dir)
        assert snapshot_dir is not None, ".snapshots/<commit>/ should be created even without prior commits"

        manifest = list((base_dir / ".batches").glob("*.json"))[0]
        data = json.loads(manifest.read_text())
        assert "commit" in data
        assert len(data["commit"]) == 40

    def test_snapshot_symlinks_submodules(self, git_repo, tmp_path):
        """Submodule directories in snapshot should be symlinked to originals."""
        sub_repo = tmp_path / "sub_repo"
        sub_repo.mkdir()
        _init_git_repo(sub_repo)
        (sub_repo / "lib.py").write_text("value = 99")
        subprocess.check_call(["git", "add", "."], cwd=sub_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.check_call(["git", "commit", "-m", "add lib"], cwd=sub_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.check_call(
            ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub_repo), "mylib"],
            cwd=git_repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(["git", "commit", "-m", "add submodule"], cwd=git_repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path / "out", executor="inline", stash=True)

        base_dir = tmp_path / "out" / "my_exp"
        snapshot_dir = self._get_snapshot_dir(base_dir)

        mylib_in_snapshot = snapshot_dir / "mylib"
        assert mylib_in_snapshot.is_symlink(), "submodule should be symlinked"
        assert mylib_in_snapshot.resolve() == (git_repo / "mylib").resolve()
        assert (mylib_in_snapshot / "lib.py").read_text() == "value = 99"

    @pytest.mark.skipif(
        not _can_subprocess_import_pyexp(),
        reason="pyexp not importable in subprocess (not installed)",
    )
    def test_subprocess_executes_from_snapshot(self, git_repo, tmp_path):
        """Subprocess executor should keep cwd at the original repo, not the snapshot."""
        (git_repo / "marker.txt").write_text("repo_content")

        @experiment(name="my_exp")
        def my_exp(config):
            import os
            from pathlib import Path

            cwd = os.getcwd()
            marker = Path("marker.txt")
            content = marker.read_text() if marker.exists() else "NOT_FOUND"
            return {"cwd": cwd, "marker": content}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="subprocess", stash=True)

        results = my_exp.results(output_dir=tmp_path)
        result = results[0]

        base_dir = tmp_path / "my_exp"
        snapshot_dir = self._get_snapshot_dir(base_dir)
        assert snapshot_dir is not None
        assert not result.result["cwd"].startswith(str(snapshot_dir)), (
            f"Subprocess should NOT change cwd to snapshot dir {snapshot_dir}, "
            f"but cwd was {result.result['cwd']}"
        )
        assert result.result["marker"] == "repo_content"
        assert (snapshot_dir / "marker.txt").read_text() == "repo_content"

    @pytest.mark.skipif(
        not hasattr(os, "fork"), reason="Fork not available on this platform"
    )
    def test_fork_executes_from_snapshot(self, git_repo, tmp_path):
        """Fork executor should keep cwd at the original repo, not the snapshot."""
        (git_repo / "marker.txt").write_text("repo_content")

        @experiment(name="my_exp")
        def my_exp(config):
            import os
            from pathlib import Path

            cwd = os.getcwd()
            marker = Path("marker.txt")
            content = marker.read_text() if marker.exists() else "NOT_FOUND"
            return {"cwd": cwd, "marker": content}

        @my_exp.configs
        def configs():
            return [{"name": "test", "x": 1}]

        with patch.object(sys, "argv", ["test"]):
            my_exp.run(output_dir=tmp_path, executor="fork", stash=True)

        results = my_exp.results(output_dir=tmp_path)
        result = results[0]

        base_dir = tmp_path / "my_exp"
        snapshot_dir = self._get_snapshot_dir(base_dir)
        assert snapshot_dir is not None
        assert not result.result["cwd"].startswith(str(snapshot_dir)), (
            f"Fork should NOT change cwd to snapshot dir {snapshot_dir}, "
            f"but cwd was {result.result['cwd']}"
        )
        assert result.result["marker"] == "repo_content"


class TestFilterWithDependencies:
    """Tests for --filter interacting with depends_on dependency graphs."""

    def test_filter_loads_finished_dependency(self, tmp_path):
        """--filter + --continue should load finished deps from disk and not crash."""
        import pickle

        @experiment(name="pipeline")
        def pipeline(cfg, out, deps):
            if cfg["name"] == "finetune":
                return {"ft_result": True, "pretrain_val": deps[0].result["pt_val"]}
            return {"pt_val": 42}

        @pipeline.configs
        def configs():
            return [
                {"name": "pretrain", "lr": 0.01},
                {"name": "finetune", "lr": 0.001, "depends_on": "pretrain"},
            ]

        # Phase 1: run full pipeline (no filter)
        with patch.object(sys, "argv", ["test", "--no-stash"]):
            pipeline.run(output_dir=tmp_path, executor="inline")

        # Verify both ran successfully
        base_dir = tmp_path / "pipeline"
        from pyexp.runner import _get_latest_timestamp, _discover_experiment_dirs

        ts = _get_latest_timestamp(base_dir)
        exp_dirs = _discover_experiment_dirs(base_dir, ts)
        assert len(exp_dirs) == 2
        for d in exp_dirs:
            assert (d / ".finished").exists()

        # Phase 2: re-run with --continue --filter "finetune"
        with patch.object(
            sys,
            "argv",
            ["test", "--continue", "--no-stash", "--filter", "finetune"],
        ):
            pipeline.run(output_dir=tmp_path, executor="inline")

        # finetune should still be cached and valid
        for d in exp_dirs:
            cfg = json.loads((d / "config.json").read_text())
            if cfg["name"] == "finetune":
                with open(d / "experiment.pkl", "rb") as f:
                    ft_exp = pickle.load(f)
                assert ft_exp.result["pretrain_val"] == 42
                assert ft_exp.finished
                break

    def test_filter_rerun_with_dependency_access(self, tmp_path):
        """--filter on a fresh downstream run should inject finished upstream deps."""
        import pickle

        @experiment(name="pipeline2")
        def pipeline2(cfg, out, deps):
            if cfg["name"] == "finetune":
                return {"ft_result": True, "pretrain_val": deps[0].result["pt_val"]}
            return {"pt_val": 99}

        @pipeline2.configs
        def configs():
            return [
                {"name": "pretrain", "lr": 0.01},
                {"name": "finetune", "lr": 0.001, "depends_on": "pretrain"},
            ]

        # Phase 1: run full pipeline
        with patch.object(sys, "argv", ["test", "--no-stash"]):
            pipeline2.run(output_dir=tmp_path, executor="inline")

        # Phase 2: delete finetune's .finished marker so it re-runs
        base_dir = tmp_path / "pipeline2"
        from pyexp.runner import _get_latest_timestamp, _discover_experiment_dirs

        ts = _get_latest_timestamp(base_dir)
        exp_dirs = _discover_experiment_dirs(base_dir, ts)
        for d in exp_dirs:
            cfg = json.loads((d / "config.json").read_text())
            if cfg["name"] == "finetune":
                (d / ".finished").unlink()
                (d / "experiment.pkl").unlink()
                break

        # Phase 3: re-run with --continue --filter "finetune"
        with patch.object(
            sys,
            "argv",
            ["test", "--continue", "--no-stash", "--filter", "finetune"],
        ):
            pipeline2.run(output_dir=tmp_path, executor="inline")

        # Verify finetune re-ran and got the pretrain dependency
        for d in exp_dirs:
            cfg = json.loads((d / "config.json").read_text())
            if cfg["name"] == "finetune":
                with open(d / "experiment.pkl", "rb") as f:
                    ft_exp = pickle.load(f)
                assert ft_exp.result["pretrain_val"] == 99
                assert ft_exp.finished
                assert not ft_exp.skipped
                break

    def test_filter_skips_when_dependency_not_finished(self, tmp_path):
        """--filter should skip downstream if upstream dep has no finished result."""
        import pickle

        @experiment(name="pipeline3")
        def pipeline3(cfg, out, deps):
            return {"val": 1}

        @pipeline3.configs
        def configs():
            return [
                {"name": "pretrain", "lr": 0.01},
                {"name": "finetune", "lr": 0.001, "depends_on": "pretrain"},
            ]

        # Run with filter "finetune" only -- pretrain never ran, no finished result
        with patch.object(sys, "argv", ["test", "--no-stash", "--filter", "finetune"]):
            pipeline3.run(output_dir=tmp_path, executor="inline")

        # Verify finetune was marked as skipped
        base_dir = tmp_path / "pipeline3"
        from pyexp.runner import _get_latest_timestamp, _discover_experiment_dirs

        ts = _get_latest_timestamp(base_dir)
        exp_dirs = _discover_experiment_dirs(base_dir, ts)
        for d in exp_dirs:
            cfg = json.loads((d / "config.json").read_text())
            if cfg["name"] == "finetune":
                with open(d / "experiment.pkl", "rb") as f:
                    ft_exp = pickle.load(f)
                assert ft_exp.skipped
                break


class TestCrossTimestampResults:
    """Tests for results() returning experiments across multiple batch timestamps.

    When a user runs all experiments, then runs only a subset (e.g. with --filter),
    results() should still return all experiments — each from its own latest run.
    """

    def _setup_two_batches(self, tmp_path):
        """Helper: run batch 1 with a,b,c then batch 2 with a only.

        Returns (exp instance, base_dir, ts1, ts2).
        """
        import time

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "run_a", "x": 1},
                {"name": "run_b", "x": 2},
                {"name": "run_c", "x": 3},
            ]

        # Batch 1: all three
        with patch.object(sys, "argv", ["test", "--no-stash"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        ts1 = sorted((base_dir / ".batches").glob("*.json"))[0].stem

        time.sleep(1.1)

        # Batch 2: only run_a (via filter)
        with patch.object(sys, "argv", ["test", "--no-stash", "--filter", "run_a"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        ts2 = sorted((base_dir / ".batches").glob("*.json"))[-1].stem
        assert ts1 != ts2

        return my_exp, base_dir, ts1, ts2

    def test_results_returns_all_experiments_across_batches(self, tmp_path):
        """results() with no timestamp should return all experiments, each from its latest run."""
        my_exp, base_dir, ts1, ts2 = self._setup_two_batches(tmp_path)

        results = my_exp.results(output_dir=tmp_path)
        names = sorted(r.name for r in results)
        assert names == ["run_a", "run_b", "run_c"]

        # run_a should come from the newer batch (ts2)
        run_a = results["run_a"]
        assert ts2 in str(run_a.out)

        # run_b and run_c should come from the older batch (ts1)
        run_b = results["run_b"]
        run_c = results["run_c"]
        assert ts1 in str(run_b.out)
        assert ts1 in str(run_c.out)

    def test_explicit_timestamp_returns_only_that_batch(self, tmp_path):
        """results(timestamp=ts1) should return only experiments from that batch."""
        my_exp, base_dir, ts1, ts2 = self._setup_two_batches(tmp_path)

        # ts1 had all three
        results_ts1 = my_exp.results(timestamp=ts1, output_dir=tmp_path)
        assert len(results_ts1) == 3

        # ts2 had only run_a
        results_ts2 = my_exp.results(timestamp=ts2, output_dir=tmp_path)
        assert len(results_ts2) == 1
        assert results_ts2[0].name == "run_a"

    def test_finished_only_works_per_experiment(self, tmp_path):
        """results(finished=True) should find each experiment's latest finished run."""
        import time

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "run_a", "x": 1},
                {"name": "run_b", "x": 2},
            ]

        # Batch 1: both finished
        with patch.object(sys, "argv", ["test", "--no-stash"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"
        ts1 = sorted((base_dir / ".batches").glob("*.json"))[0].stem

        time.sleep(1.1)

        # Batch 2: only run_a
        with patch.object(sys, "argv", ["test", "--no-stash", "--filter", "run_a"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        ts2 = sorted((base_dir / ".batches").glob("*.json"))[-1].stem

        # Remove .finished from run_a in ts2 to simulate unfinished
        run_a_ts2_dir = base_dir / "run_a" / ts2
        (run_a_ts2_dir / ".finished").unlink()

        # finished=True: run_a should fall back to ts1, run_b from ts1
        results = my_exp.results(output_dir=tmp_path, finished=True)
        names = sorted(r.name for r in results)
        assert names == ["run_a", "run_b"]

        run_a = results["run_a"]
        assert ts1 in str(run_a.out), "run_a should fall back to the older finished run"

    def test_experiment_only_in_old_batch_still_found(self, tmp_path):
        """An experiment that only exists in an older batch should still appear in results()."""
        import time
        import pickle

        @experiment(name="my_exp")
        def my_exp(config):
            return {"value": config["x"]}

        @my_exp.configs
        def configs():
            return [
                {"name": "run_a", "x": 1},
                {"name": "run_b", "x": 2},
            ]

        # Batch 1: both experiments
        with patch.object(sys, "argv", ["test", "--no-stash"]):
            my_exp.run(output_dir=tmp_path, executor="inline")

        base_dir = tmp_path / "my_exp"

        time.sleep(1.1)

        # Batch 2: completely new experiment (run_c) — simulate by creating manually
        ts2 = "2099-01-01_00-00-00"
        run_c_dir = base_dir / "run_c" / ts2
        run_c_dir.mkdir(parents=True)
        (run_c_dir / "config.json").write_text('{"name": "run_c", "x": 99}')
        from pyexp import Config
        result_obj = Result(
            cfg=Config({"name": "run_c", "x": 99}),
            name="run_c",
            result={"value": 99},
            finished=True,
        )
        with open(run_c_dir / "experiment.pkl", "wb") as f:
            pickle.dump(result_obj, f)
        (run_c_dir / ".finished").touch()

        results = my_exp.results(output_dir=tmp_path)
        names = sorted(r.name for r in results)
        assert names == ["run_a", "run_b", "run_c"]

    def test_getitem_finds_across_timestamps(self, tmp_path):
        """__getitem__ should find experiments across timestamps."""
        my_exp, base_dir, ts1, ts2 = self._setup_two_batches(tmp_path)

        my_exp._output_dir = tmp_path

        # run_b only exists in ts1
        run_b = my_exp["run_b"]
        assert run_b.name == "run_b"
        assert run_b.result["value"] == 2

        # run_a exists in both, should get the latest (ts2)
        run_a = my_exp["run_a"]
        assert run_a.name == "run_a"
        assert ts2 in str(run_a.out)
