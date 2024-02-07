import collections
import numbers
import os
import subprocess

import pytest
from mpi4py import MPI


MAX_NPROCS_FLAG = "PYTEST_MPI_MAX_NPROCS"
"""Environment variable that can be set to limit the maximum number of processes.

If set then requesting a parallel test with more processes than it will raise an
error. If unset then any value is accepted.
"""


CHILD_PROCESS_FLAG = "_PYTEST_MPI_CHILD_PROCESS"
"""Environment variable set for the processes spawned by the mpiexec call."""


_plugin_in_use = False
"""Global variable set internally to indicate that parallel markers are used."""


@pytest.hookimpl()
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parallel(nprocs): mark test to run in parallel on nprocs processors (default: 3)"
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    if MPI.COMM_WORLD.size > 1 and not _is_parallel_child_process() and _xdist_active(session):
        raise pytest.UsageError(
            "Wrapping pytest calls in mpiexec is only supported if pytest-xdist "
            "is not in use"
        )


@pytest.hookimpl()
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
    nprocss = _parse_marker_nprocs(marker)

    if MAX_NPROCS_FLAG in os.environ:
        max_nprocs = int(os.environ[MAX_NPROCS_FLAG])
        for nprocs in nprocss:
            if nprocs > max_nprocs:
                raise pytest.UsageError(
                    "Requested a parallel test with too many ranks "
                    f"({nprocs} > {MAX_NPROCS_FLAG}={max_nprocs})"
                )

    # Only label tests if more than one parallel size is requested
    if len(nprocss) > 1:
        # Trick the function into thinking that it needs an extra fixture argument
        metafunc.fixturenames.append("_nprocs")
        metafunc.parametrize("_nprocs", nprocss, ids=lambda n: f"nprocs={n}")


@pytest.hookimpl()
def pytest_collection_modifyitems(config, items):
    global _plugin_in_use

    _plugin_in_use = any(item.get_closest_marker("parallel") for item in items)

    if not _plugin_in_use:
        return

    for item in items:
        if item.get_closest_marker("parallel"):
            # Add extra markers to each test to allow for querying specific levels of
            # parallelism (e.g. "-m parallel[3]")
            nprocs = _extract_nprocs_for_single_test(item)
            new_marker = f"parallel[{nprocs}]"
            if new_marker not in pytest.mark._markers:
                config.addinivalue_line(
                    "markers",
                    f"{new_marker}: internal marker"
                )
            item.add_marker(getattr(pytest.mark, new_marker))


@pytest.hookimpl()
def pytest_runtest_setup(item):
    if not _plugin_in_use:
        return

    if item.get_closest_marker("parallel"):
        if MPI.COMM_WORLD.size == 1:
            # If using pytest-mpi in "forking" mode, add a callback to item
            # that calls mpiexec
            assert not _is_parallel_child_process()
            _set_parallel_callback(item)
        elif _is_parallel_child_process():
            # Already a forked subprocess, run the test unmodified
            pass
        else:
            # Outer mpiexec used, do not fork a subprocess but fail if the
            # requested parallelism does not match the provided amount
            nprocs = _extract_nprocs_for_single_test(item)
            if nprocs != MPI.COMM_WORLD.size:
                raise pytest.UsageError(
                    "Attempting to run parallel tests inside an mpiexec call "
                    "where the requested and provided process counts do not match"
                )
    else:
        # serial test
        if MPI.COMM_WORLD.size != 1:
            raise pytest.UsageError(
                "Serial tests should not be run by multiple processes, consider "
                "adding a parallel marker to the test"
            )


@pytest.fixture(scope="function", autouse=True)
def barrier_finalize(request):
    """Call an MPI barrier at the end of each test.

    This should help localise tests that are not fully collective.

    """
    if _plugin_in_use:
        request.addfinalizer(lambda: MPI.COMM_WORLD.barrier())


def _is_parallel_child_process():
    return CHILD_PROCESS_FLAG in os.environ


def _xdist_active(session):
    try:
        import xdist
        return xdist.is_xdist_controller(session) or xdist.is_xdist_worker(session)
    except ImportError:
        return False


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
