from os import abort
from signal import SIGSEGV
import signal
import pyexp


def fn():
    print("test")

    return "test"


if __name__ == "__main__":
    ex = pyexp.SshExecutor(snapshot=True, capture=False, hosts=["doeringc@rgllab"])
    future = ex.submit(fn)
    res = future.result()
    print(f"{res=}")
    print(f"{future.log=}")
    print(f"{future.exception()=}")
