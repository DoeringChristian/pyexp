"""Subprocess worker for running isolated experiments.

This module is invoked as a subprocess to run a single experiment in isolation.
It receives a serialized payload (fn + experiment dataclass) via a temp file,
executes the experiment function, and writes the pickled experiment to the
designated output path.

Usage:
    python -m pyexp.worker <payload_path>

The payload file contains cloudpickle-serialized data:
    {
        "fn": <experiment function>,
        "experiment": <Result dataclass with cfg/name/out set>,
        "result_path": <path to write pickled experiment>,
        "stash": <bool>,
        "wants_out": <bool>,
        "wants_deps": <bool>,
    }

Dependencies are resolved from disk (config.json + experiment.pkl files)
rather than being serialized in the payload.
"""

import json
import pickle
import sys
import traceback
from pathlib import Path

import cloudpickle


def _resolve_deps_from_disk(result_path: Path):
    """Resolve and load dependency experiments from disk.

    Reads depends_on from config.json, discovers the latest finished experiments
    in the base directory, resolves dependency names, and loads their pickled results.

    Returns:
        A Runs instance containing the resolved dependency Results, or None if
        no dependencies are declared.
    """
    from pyexp.config import Runs
    from pyexp.runner import (
        _normalize_depends_on,
        _resolve_depends_on,
        _discover_all_experiments_latest,
    )

    experiment_dir = result_path.parent
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return None

    config_data = json.loads(config_path.read_text())
    depends_on = config_data.get("depends_on")
    if not depends_on:
        return None

    # base_dir is 3 levels up: experiment.pkl -> <timestamp> -> <name> -> base_dir
    base_dir = experiment_dir.parent.parent

    # Discover all latest finished experiments to build a Runs for resolution
    latest_dirs = _discover_all_experiments_latest(base_dir, finished_only=True)
    all_configs = []
    dir_by_name = {}
    for d in latest_dirs:
        cfg = json.loads((d / "config.json").read_text())
        all_configs.append(cfg)
        dir_by_name[cfg.get("name", "")] = d

    configs_runs = Runs(all_configs)
    dep_keys = _normalize_depends_on(depends_on)
    config_name = config_data.get("name", "")
    resolved_names = _resolve_depends_on(dep_keys, configs_runs, config_name)

    dep_results = []
    for dep_name in resolved_names:
        dep_dir = dir_by_name.get(dep_name)
        if dep_dir is None:
            continue
        dep_pkl = dep_dir / "experiment.pkl"
        if dep_pkl.exists():
            with open(dep_pkl, "rb") as f:
                dep_result = pickle.load(f)
            dep_result.out = dep_dir
            dep_results.append(dep_result)

    return Runs(dep_results)


def run_worker(payload_path: str) -> int:
    """Run a single experiment from a serialized payload.

    Args:
        payload_path: Path to the cloudpickle-serialized payload file.

    Returns:
        Exit code: 0 for success, 1 for failure.

    The worker saves the pickled experiment (with results).
    """
    experiment = None
    result_path = None
    try:
        # Load the payload
        with open(payload_path, "rb") as f:
            payload = cloudpickle.load(f)

        fn = payload["fn"]
        experiment = payload["experiment"]
        result_path = Path(payload["result_path"])
        experiment.out = result_path.parent
        wants_out = payload.get("wants_out", False)
        wants_deps = payload.get("wants_deps", False)

        # Resolve deps from disk if needed
        deps = None
        if wants_deps:
            deps = _resolve_deps_from_disk(result_path)

        # Call the experiment function
        if wants_deps:
            experiment.result = fn(experiment.cfg, experiment.out, deps)
        elif wants_out:
            experiment.result = fn(experiment.cfg, experiment.out)
        else:
            experiment.result = fn(experiment.cfg)

        # Write pickled experiment
        experiment.finished = True
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "wb") as f:
            pickle.dump(experiment, f)
        (result_path.parent / ".finished").touch()

        return 0

    except Exception as e:
        # Write error information to experiment and save
        try:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            if experiment is not None:
                experiment.error = error_msg
                experiment.finished = True
                if result_path is not None:
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(result_path, "wb") as f:
                        pickle.dump(experiment, f)
                    (result_path.parent / ".finished").touch()
        except Exception:
            # If we can't even write the error, just print it
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m pyexp.worker <payload_path>", file=sys.stderr)
        sys.exit(2)

    sys.exit(run_worker(sys.argv[1]))
