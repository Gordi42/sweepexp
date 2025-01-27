"""
Test the SweepExpParallel class.

Description
-----------
This test should be run with mpi and 4 CPUs available. For example with srun:

.. code-block:: bash

    srun -n 4 pytest tests/test_sweepexp_mpi.py

Or with mpirun:

.. code-block:: bash

    mpirun -n 4 pytest tests/test_sweepexp_mpi.py

"""

from __future__ import annotations

import pytest
from mpi4py import MPI

# Try to import mpi4py and SweetExpMPI
try:
    from mpi4py import MPI

    from sweepexp import SweepExpMPI
except ImportError:
    MPI = None
    SweepExpMPI = None

# ================================================================
#  Helpers
# ================================================================
class MyObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: MyObject) -> bool:
        if not isinstance(other, MyObject):
            return False
        return self.value == other.value

# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture(params=[".pkl", ".zarr", ".nc"])
def save_path(temp_dir, request):
    return temp_dir / f"test{request.param}"

# ================================================================
#  Tests
# ================================================================

@pytest.mark.mpi
def test_mpi_world_size():
    """Test the number of ranks."""
    size = MPI.COMM_WORLD.Get_size()
    expected_size = 4
    assert size == expected_size
