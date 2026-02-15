"""Subprocess worker for running isolated experiments.

This module is invoked as a subprocess to run a single experiment in isolation.
It receives a serialized payload (experiment instance) via a temp file, executes
the experiment, and writes the pickled instance to the designated output path.

Usage:
    python -m pyexp.worker <payload_path>

The payload file contains cloudpickle-serialized data:
    {
        "instance": <experiment instance with cfg/out set>,
        "result_path": <path to write pickled instance>,
        "stash": <bool>,
    }
"""

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

    The worker saves the pickled experiment instance (with results).
    """
    instance = None
    result_path = None
    try:
        # Load the payload
        with open(payload_path, "rb") as f:
            payload = cloudpickle.load(f)

        instance = payload["instance"]
        result_path = Path(payload["result_path"])

        # Restore transient deps if provided
        deps = payload.get("deps")
        if deps is not None:
            instance._Experiment__deps = deps

        # Run the experiment
        instance.experiment()

        # Write pickled instance using cloudpickle for dynamic classes
        instance._Experiment__finished = True
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "wb") as f:
            cloudpickle.dump(instance, f)
        (result_path.parent / ".finished").touch()

        return 0

    except Exception as e:
        # Write error information to instance and save
        try:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            if instance is not None:
                instance._Experiment__error = error_msg
                instance._Experiment__finished = True
                if result_path is not None:
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(result_path, "wb") as f:
                        cloudpickle.dump(instance, f)
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
