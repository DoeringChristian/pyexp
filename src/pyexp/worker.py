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
        "deps": <Runs of dependency experiments or None>,
        "result_path": <path to write pickled experiment>,
        "stash": <bool>,
        "wants_out": <bool>,
        "wants_deps": <bool>,
    }
"""

import pickle
import sys
import traceback
from pathlib import Path

import cloudpickle


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
        deps = payload.get("deps")
        wants_out = payload.get("wants_out", False)
        wants_deps = payload.get("wants_deps", False)

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
