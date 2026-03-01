"""Subprocess worker for running isolated experiments.

This module is invoked as a subprocess to run a single experiment in isolation.
It receives a serialized payload (fn + experiment dataclass) via a temp file,
executes the experiment function, and writes the result to the designated output path.

Usage:
    python -m pyexp.worker <payload_path>

The payload file contains cloudpickle-serialized data:
    {
        "fn": <experiment function>,
        "experiment": <Result dataclass with cfg/name/out set>,
        "result_path": <path to write pickled result>,
        "stash": <bool>,
        "wants_out": <bool>,
        "wants_deps": <bool>,
    }

Dependencies are resolved from disk (config.json + result.pkl files)
rather than being serialized in the payload.

Output files written:
    - result.pkl: The pickled return value of the experiment function
    - error.txt: Error message if the experiment failed
    - log.out: Captured stdout/stderr (written by the executor after worker completes)
    - .finished: Marker file indicating the experiment completed
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
    in the base directory, resolves dependency names, and loads their results.

    Returns:
        A Runs instance containing the resolved dependency Results, or None if
        no dependencies are declared.
    """
    from pyexp.config import Config, Runs
    from pyexp.runner import (
        Result,
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

    # base_dir is 3 levels up: result.pkl -> <timestamp> -> <name> -> base_dir
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

        # Load dependency result from new format
        dep_config_path = dep_dir / "config.json"
        dep_result_path = dep_dir / "result.pkl"
        dep_error_path = dep_dir / "error.txt"
        dep_log_path = dep_dir / "log.out"

        if dep_config_path.exists():
            dep_cfg_data = json.loads(dep_config_path.read_text())
            dep_cfg_clean = {k: v for k, v in dep_cfg_data.items() if k != "depends_on"}
            dep_cfg = Config(dep_cfg_clean)

            result_value = None
            if dep_result_path.exists():
                with open(dep_result_path, "rb") as f:
                    result_value = pickle.load(f)

            error = None
            if dep_error_path.exists():
                error = dep_error_path.read_text()

            log = ""
            if dep_log_path.exists():
                log = dep_log_path.read_text()

            dep_result = Result(
                cfg=dep_cfg,
                name=dep_cfg_data.get("name", ""),
                out=dep_dir,
                result=result_value,
                error=error,
                log=log,
                finished=True,
                skipped=(dep_dir / ".skipped").exists(),
            )
            dep_results.append(dep_result)

    return Runs(dep_results)


def run_worker(payload_path: str) -> int:
    """Run a single experiment from a serialized payload.

    Args:
        payload_path: Path to the cloudpickle-serialized payload file.

    Returns:
        Exit code: 0 for success, 1 for failure.

    The worker saves only the result value to result.pkl and errors to error.txt.
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
        experiment_dir = result_path.parent
        experiment.out = experiment_dir
        wants_out = payload.get("wants_out", False)
        wants_deps = payload.get("wants_deps", False)

        # Resolve deps from disk if needed
        deps = None
        if wants_deps:
            deps = _resolve_deps_from_disk(result_path)

        # Call the experiment function
        if wants_deps:
            result_value = fn(experiment.cfg, experiment.out, deps)
        elif wants_out:
            result_value = fn(experiment.cfg, experiment.out)
        else:
            result_value = fn(experiment.cfg)

        # Write only the result value to result.pkl
        experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(result_path, "wb") as f:
            pickle.dump(result_value, f)
        (experiment_dir / ".finished").touch()

        return 0

    except Exception as e:
        # Write error information to error.txt
        try:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            if result_path is not None:
                experiment_dir = result_path.parent
                experiment_dir.mkdir(parents=True, exist_ok=True)
                (experiment_dir / "error.txt").write_text(error_msg)
                (experiment_dir / ".finished").touch()
            else:
                # Can't write to disk — print so the executor can capture it
                print(error_msg, file=sys.stderr)
        except Exception:
            # If we can't even write the error, just print it
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python -m pyexp.worker <payload_path>", file=sys.stderr)
        sys.exit(2)

    sys.exit(run_worker(sys.argv[1]))
