import collections
import numbers
import os
import subprocess

import pytest
from mpi4py import MPI


CHILD_PROCESS_FLAG = "_PYTEST_MPI_CHILD_PROCESS"
"""Environment variable set for the processes spawned by the mpiexec call."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors (default: 3)"
    )


def pytest_generate_tests(metafunc):
    """Identify tests with parallel markers and break them apart if necessary.

    This hook turns tests with marks like ``@pytest.mark.parallel([2, 3, 4])``
    into multiple tests, one for each requested size. The tests are then
    distinguished by ID. For example ``test_abc[nprocs=2]``, ``test_abc[nprocs=3]``
    and ``test_abc[nprocs=4]``. If only one parallel size is requested then this
    is skipped.

    """
    markers = tuple(m for m in metafunc.function.pytestmark if m.name == "parallel")

    if not markers:
        return

    marker, = markers
    nprocs = _parse_marker_nprocs(marker)
    # Only label tests if more than one parallel size is requested
    if len(nprocs) > 1:
        # Trick the function into thinking that it needs an extra fixture argument
        metafunc.fixturenames.append("_nprocs")
        metafunc.parametrize("_nprocs", nprocs, ids=lambda n: f"nprocs={n}")


def pytest_collection_modifyitems(config, items):
    # Only modify items if parallel markers are being used
    if not any(item.get_closest_marker("parallel") for item in items):
        return

    if MPI.COMM_WORLD.size > 1 and not _is_parallel_child_process():
        raise pytest.UsageError(
            "pytest should not be called from within a parallel context "
            "(e.g. mpiexec -n 3 pytest ...)"
        )

    # Add extra markers to each test to allow for querying specific levels of
    # parallelism (e.g. "-m parallel[3]")
    for item in items:
        if item.get_closest_marker("parallel"):
            nprocs = _extract_nprocs_for_single_test(item)
            new_marker = f"parallel[{nprocs}]"
            if new_marker not in pytest.mark._markers:
                config.addinivalue_line(
                    "markers",
                    f"{new_marker}: internal marker"
                )
            item.add_marker(getattr(pytest.mark, new_marker))


def pytest_runtest_setup(item):
    if item.get_closest_marker("parallel") and not _is_parallel_child_process():
        _set_parallel_callback(item)


def _is_parallel_child_process():
    return CHILD_PROCESS_FLAG in os.environ


def _set_parallel_callback(item):
    """Replace the callback for a test item with one that calls ``mpiexec``.

    If the number of processes requested is 1 then this function does nothing.

    Parameters
    ----------
    item : _pytest.nodes.Item
        The test item to run.

    """
    nprocs = _extract_nprocs_for_single_test(item)
    assert isinstance(nprocs, numbers.Integral)

    if nprocs == 1:
        return

    # Run xfailing tests to ensure that errors are reported to calling process
    pytest_args = ["--runxfail", "-s", "-q", f"{item.fspath}::{item.name}"]
    # Try to generate less output on other ranks so stdout is easier to read
    quieter_pytest_args = pytest_args + [
        "--tb=no", "--no-summary", "--no-header",
        "--disable-warnings", "--show-capture=no"
    ]

    cmd = [
        "mpiexec", "-n", "1", "-genv", CHILD_PROCESS_FLAG, "1", "python", "-m", "pytest"
    ] + pytest_args + [
        ":", "-n", f"{nprocs-1}", "python", "-m", "pytest"
    ] + quieter_pytest_args

    def parallel_callback(*args, **kwargs):
        subprocess.run(cmd, check=True)

    item.obj = parallel_callback


def _extract_nprocs_for_single_test(item):
    """Extract the number of processes that a test is supposed to be run with.

    Unlike `_parse_marker_nprocs`, this function applies to tests that have already
    been set to require a fixed level of parallelism. In other words, if the
    parallel marker requested, say, ``[2, 3]`` processes, the tests input to
    this function have already been split into ``[nprocs=2]`` and ``[nprocs=3]``
    versions. Therefore, this function returns an integer, rather than a tuple.

    """
    # First check to see if we have parametrised nprocs (if multiple were requested)
    if hasattr(item, "callspec") and "_nprocs" in item.callspec.params:
        nprocs = item.callspec.params["_nprocs"]
    else:
        # The parallel marker must just want one value of nprocs
        marker = item.get_closest_marker("parallel")
        nprocs, = _parse_marker_nprocs(marker)
    return nprocs


def _parse_marker_nprocs(marker):
    """Return the number of processes requested from a parallel marker.

    This function enables one to use the parallel marker with or without
    using the ``nprocs`` keyword argument.

    The returned process counts are provided as a tuple, even if only a
    single value is requested.

    """
    assert marker.name == "parallel"

    if len(marker.args) == 1 and not marker.kwargs:
        return _as_tuple(marker.args[0])
    elif len(marker.kwargs) == 1 and not marker.args:
        return _as_tuple(marker.kwargs["nprocs"])
    elif not marker.args and not marker.kwargs:
        return (3,)
    else:
        raise pytest.UsageError("Bad arguments given to parallel marker")


def _as_tuple(arg):
    return tuple(arg) if isinstance(arg, collections.abc.Iterable) else (arg,)
