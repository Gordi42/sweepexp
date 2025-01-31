"""
A python package for running parallel experiments across parameter grids with MPI.

For more information, visit the project's GitHub repository:
https://github.com/Gordi42/sweepexp
"""
__author__ = """Silvano Gordian Rosenau"""
__email__ = "silvano.rosenau@uni-hamburg.de"
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .logger import log
    from .sweepexp import SweepExp
    from .sweepexp_mpi import SweepExpMPI
    from .sweepexp_parallel import SweepExpParallel

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = { }

all_imports_by_origin = {
    "sweepexp.logger": ["log"],
    "sweepexp.sweepexp": ["SweepExp"],
    "sweepexp.sweepexp_mpi": ["SweepExpMPI"],
    "sweepexp.sweepexp_parallel": ["SweepExpParallel"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
