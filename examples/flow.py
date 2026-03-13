import pyexp


def test(): ...


@pyexp.task
def task1():
    return "test"


@pyexp.task
def task2(t1):
    test("test")
    print(f"{t1=}")
    return f"task2({t1})"


@pyexp.flow
def flow(test: bool = False):
    t1 = task1()

    # Tasks don't have to be returned.
    t2 = task2(t1)


if __name__ == "__main__":

    # test is also exposed as a cli argument
    result = flow(test=True)
