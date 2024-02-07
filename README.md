# pytest-mpi

Pytest plugin that lets you run tests in parallel with MPI.

## Installation

To install `pytest-mpi` simply run:

```
$ python -m pip install /path/to/pytest-mpi-repo
```

## Usage

Writing a parallel test simply requires marking the test with the `parallel` marker:

```py
@pytest.mark.parallel(nprocs=5)  # run in parallel with 5 processes
def test_my_code_on_5_procs():
    ...

@pytest.mark.parallel(5)  # the "nprocs" kwarg is optional
def test_my_code_on_5_procs_again():
    ...

@pytest.mark.parallel  # run in parallel with the default number of processes (3)
def test_my_code_on_3_procs():
    ...

@pytest.mark.parallel()  # the brackets are optional
def test_my_code_on_3_procs_again():
    ...
```

One can also mark a test with a sequence of values for `nprocs`:

```py
@pytest.mark.parallel(nprocs=[1, 2, 3])  # run in parallel on 1, 2 and 3 processes
def test_my_code_on_variable_nprocs():
    ...

@pytest.mark.parallel([1, 2, 3])  # again the "nprocs" kwarg is optional
def test_my_code_on_variable_nprocs_again():
    ...
```

If multiple numbers of processes are requested then the tests are parametrised
and renamed to, in this case, `test_my_code_on_variable_nprocs[nprocs=1]`,
`test_my_code_on_variable_nprocs[nprocs=2]` and
`test_my_code_on_variable_nprocs[nprocs=3]`.

### Extra markers

When running the code with these `parallel` markers, `pytest-mpi` adds extra markers
to each test to allow one to select all tests with a particular number of processors.
For example, to select all parallel tests on 3 processors, one should run

```bash
$ pytest -m "parallel[3]"
```

For serial tests - those either unmarked or marked `@pytest.mark.parallel(1)` - one
can select these by running

```bash
$ pytest -m "not parallel or parallel[1]"
```

### Forking mode

`pytest-mpi` can be run in one of two modes: forking or non-forking. The former
works as follows:

1. The user calls `pytest` (not `mpiexec -n <# proc> pytest`!). This launches
   the "parent" `pytest` process.
2. This parent `pytest` process collects all the tests and begins to run them.
3. When a test is found with the `parallel` marker, rather than executing the
   function as before, a subprocess is forked calling
   `mpiexec -np <# proc> pytest this_specific_test_file.py::this_specific_test`.
   This produces `<# proc>` "child" `pytest` processes that execute the
   test together.
4. If this terminates successfully then the test is considered to have passed.

This is convenient for development for a number of reasons:

* The plugin composes better with other pytest plugins like `pytest-xdist`.
* It is not necessary to wrap `pytest` invocations with `mpiexec` calls, and
  all parallel and serial tests can be run at once.

However, the forking mode of `pytest-mpi` is restricted in that only one mainstream
MPI distribution ([MPICH](https://www.mpich.org/)) supports nested calls to
`MPI_Init`. If your "parent" `pytest` process initialises MPI (for instance by
executing `from petsc4py import PETSc`) then this will cause non-MPICH MPI
distributions to crash. Further, forking a subprocess can be expensive since a
completely fresh Python interpreter is launched each time.

### Non-forking mode

With these significant limitations in mind, `pytest-mpi` therefore also supports
a non-forking mode. To use it, one simply needs to wrap the `pytest` invocation
with `mpiexec`, no additional configuration is necessary. For example, to run
all of the parallel tests on 2 ranks one needs to execute:

```bash
$ mpiexec -n 2 pytest -m "parallel[2]"
```

This approach is agnostic to MPI distribution, and free from the forking startup
overhead, but has a number of disadvantages:

* `pytest-xdist` is strictly disallowed as threading would lead to deadlocks. It
  is therefore impossible to take full advantage of machines with many cores.
* A different `mpiexec` instruction is required for each level of parallelism.
  Attempting to run with `mpiexec` with a mismatching number of processes to the
  parallel marker will result in an error.

## Configuration

`pytest-mpi` respects the environment variable `PYTEST_MPI_MAX_NPROCS`, which defines
the maximum number of processes that can be requested by a parallel marker. If this
value is exceeded an error will be raised.
