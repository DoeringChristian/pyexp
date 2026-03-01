from os import abort
from signal import SIGSEGV
import signal
import pyexp


def fn():
    print("test")
    signal.signal(SIGSEGV)


if __name__ == "__main__":
    ex = pyexp.SubprocessExecutor(snapshot=True, capture=False)
    future = ex.submit(fn)
    print(f"{future.log=}")
    print(f"{future.exception()=}")
