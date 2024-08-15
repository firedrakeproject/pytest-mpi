from mpi4py import MPI

def parallel_assert(assertion, subset=None, msg=""):
    """Make an assertion across MPI.COMM_WORLD

    Parameters:
    -----------
    assertion:
        Callable that will be tested for truthyness (usually evaluates some assertion).
        This should be the same across all ranks.
    subset:


    Example:
    --------
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
    if subset:
        if MPI.COMM_WORLD.rank in subset:
            evaluation = assertion()
        else:
            evaluation = True
    else:
        evaluation = assertion()
    all_results = MPI.COMM_WORLD.allgather(evaluation)
    if not min(all_results):
        raise AssertionError(
            "Parallel assertion failed on ranks: "
            f"{[ii for ii, b in enumerate(all_results) if not b]}\n" + msg
        )
