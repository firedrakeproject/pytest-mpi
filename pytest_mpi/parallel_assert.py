from collections.abc import Callable

from mpi4py import MPI


def parallel_assert(assertion: Callable, participating: bool = True, msg: str = "") -> None:
    """Make an assertion across ``COMM_WORLD``.

    Parameters
    ----------
    assertion :
        Callable that will be tested for truthyness (usually evaluates some assertion).
        This should be the same across all ranks.
    participating :
        Whether the given rank should evaluate the assertion.
    msg :
        Optional error message to print out on failure.

    Notes
    -----
    It is very important that ``parallel_assert`` is called collectively on all
    ranks simulataneously.

    Example
    -------
    Where in serial code one would have previously written:
    ```python
    x = f()
    assert x < 5
    ```

    Now write:
    ```python
    x = f()
    parallel_assert(lambda: x < 5)
    ```

    """
    result = assertion() if participating else True
    all_results = MPI.COMM_WORLD.allgather(result)
    if not min(all_results):
        raise AssertionError(
            "Parallel assertion failed on ranks: "
            f"{[rank for rank, result in enumerate(all_results) if not result]}\n"
            + msg
        )
