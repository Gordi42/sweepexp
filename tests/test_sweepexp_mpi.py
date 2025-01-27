"""
Test the SweepExpParallel class.

Description
-----------
This test should be run with mpi. For example with srun:

.. code-block:: bash

    srun -n 2 pytest tests/test_sweepexp_mpi.py

Or with mpirun:

.. code-block:: bash

    mpirun -n 2 pytest tests/test_sweepexp_mpi.py

"""
from __future__ import annotations

import pytest

# Try to import mpi4py and SweetExpMPI
try:
    from mpi4py import MPI
except ImportError:
    # Skip the test if mpi4py is not available
    pytest.skip("mpi4py is not available", allow_module_level=True)


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

@pytest.mark.mpi(min_size=2)
def test_mpi_world_size():
    """Test the number of ranks."""
    size = MPI.COMM_WORLD.Get_size()
    min_size = 2
    assert size >= min_size
