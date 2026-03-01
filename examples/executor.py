from os import abort
from signal import SIGSEGV
import signal
import pyexp


def fn():
    print("test")
    signal.signal(SIGSEGV)


if __name__ == "__main__":
    ex = pyexp.SubprocessExecutor()
    res = ex.run(fn, capture=False)
    print(f"{res.log=}")
    print(f"{res.ok=}")
