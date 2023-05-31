import os
import subprocess

import mpi4py
import pytest


mpi4py.rc.initialize = False
from mpi4py import MPI


CHILD_PROCESS_FLAG = "_PYTEST_MPI_CHILD_PROCESS"
"""Environment variable set for the processes spawned by the mpiexec call."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors (default: 3)"
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("parallel") and not _is_parallel_child_process():
        _set_parallel_callback(item)


def _is_parallel_child_process():
    return CHILD_PROCESS_FLAG in os.environ


def _set_parallel_callback(item):
    """Replace the callback for a test item with one that calls ``mpiexec``.

    Parameters
    ----------
    item : _pytest.nodes.Item
        The test item to run.

    """
    if MPI.Is_initialized():
        raise pytest.UsageError(
            "MPI should not be initialised before the inner pytest invocation"
        )

    nprocs = item.get_closest_marker("parallel").kwargs.get("nprocs", 3)
    if nprocs < 2:
        raise pytest.UsageError("Need to specify at least two processes for a parallel test")

    # Run xfailing tests to ensure that errors are reported to calling process
    pytest_args = ["--runxfail", "-s", "-q", f"{item.fspath}::{item.name}"]
    # Try to generate less output on other ranks so stdout is easier to read
    quieter_pytest_args = (pytest_args
                           + ["--tb=no", "--no-summary", "--no-header",
                              "--disable-warnings", "--show-capture=no"])

    cmd = (["mpiexec", "-n", "1", "-genv", CHILD_PROCESS_FLAG, "1", "python", "-m", "pytest"]
            + pytest_args
            + [":", "-n", f"{nprocs-1}", "python", "-m", "pytest"]
            + quieter_pytest_args)

    def parallel_callback(*args, **kwargs):
         subprocess.run(cmd, check=True)

    item.obj = parallel_callback
