"""Test the SweepExp class."""
from __future__ import annotations

import pytest

from sweepexp import SweepExp, SweepExpParallel, sweepexp

# ================================================================
#  Helpers
# ================================================================
# Try to import mpi4py, if not available, skip the tests
modes = ["parallel", "sequential"]
try:
    from sweepexp import SweepExpMPI
    mpi_available = True
    modes.append("mpi")
except ImportError:
    SweepExpMPI = None
    modes.append(pytest.param("mpi", marks=pytest.mark.skip(
        reason="mpi4py not available")))
    mpi_available = False

# ================================================================
#  Fixtures
# ================================================================

# ================================================================
#  Tests
# ================================================================

def test_init_default():
    """Test the initialization of the SweepExp class without a file."""
    def my_experiment(x: int) -> dict:
        return {"result": x * 2}
    parameters = {"x": [1, 2, 3]}
    timeit = True
    # Create the experiment
    exp = sweepexp(func=my_experiment, parameters=parameters, timeit=timeit)
    assert isinstance(exp, SweepExp)
    assert exp.func == my_experiment
    assert exp.parameters == parameters
    assert exp.timeit == timeit

@pytest.mark.parametrize("mode", modes)
def test_init_with_given_mode(mode):
    mode_type = {"parallel": SweepExpParallel,
                 "mpi": SweepExpMPI,
                 "sequential": SweepExp}[mode]
    def my_experiment(x: int) -> dict:
        return {"result": x * 2}
    parameters = {"x": [1, 2, 3]}
    timeit = True
    exp = sweepexp(
        func=my_experiment,
        parameters=parameters,
        mode=mode,
        timeit=timeit,
    )
    assert isinstance(exp, mode_type)
    assert exp.func == my_experiment
    assert exp.parameters == parameters
    assert exp.timeit == timeit

def test_mpi_fallback(caplog):
    with caplog.at_level("WARNING"):
        sweep = sweepexp(func=None, parameters={}, mode="mpi")
    if mpi_available:
        assert isinstance(sweep, SweepExpMPI)
    else:
        assert isinstance(sweep, SweepExpParallel)
        assert "Fallback to 'parallel' mode." in caplog.text