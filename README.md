# pytest-mpi

Pytest plugin that lets you run tests in parallel with MPI.

## Usage

Writing a parallel test simply requires marking the test with the `parallel` marker:

```py
@pytest.mark.parallel(nprocs=5)  # run in parallel with 5 processes
def test_my_code_on_5_procs():
    ...

@pytest.mark.parallel  # run in parallel with the default number of processes (3)
def test_my_code_on_3_procs():
    ...
```

## How it works

1. The user calls `pytest` (not `mpiexec -n <# proc> pytest`!). This launches the "parent" `pytest` process.
2. This parent `pytest` process collects all the tests and then begins to run them.
3. When a test is found with the `parallel` marker, rather than executing the function as before, a subprocess is forked calling `mpiexec -np <# proc> pytest this_specific_test_file.py::this_specific_test`. This produces `<# proc>` "child" `pytest` processes that execute the test together.
4. If this terminates successfully then the test is assumed to have passed.

## FAQs

### Why call `mpiexec` internally instead of wrapping `pytest` in it?

It is not a good idea to wrap `pytest` in an `mpiexec` because it would not compose well with various `pytest` plugins. In particular it would likely cause deadlocks when used with `pytest-xdist`.

## Caveats

Unfortunately `pytest-mpi` will only work for MPI distributions that support nested calls to `MPI_Init` (e.g. [MPICH](https://www.mpich.org/)). This is because of step 3 above. The parent `pytest` needs to import all of the test scripts before tests can be run and, unless you have been extremely careful with your imports, it is very likely that a line like `from mpi4py import MPI` or `from petsc4py import PETSc` will be encountered and these will go away and call `MPI_Init`. Then, launching the child `pytest` processes with `mpiexec` will trigger another call to `MPI_Init`, breaking things.
