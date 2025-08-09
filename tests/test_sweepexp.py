"""Test the SweepExp class."""
from __future__ import annotations

import pytest

from sweepexp import SweepExp, SweepExpMPI, SweepExpParallel, sweepexp

# ================================================================
#  Helpers
# ================================================================

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

@pytest.mark.parametrize("mode", ["parallel", "mpi", "sequential"])
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
