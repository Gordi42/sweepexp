"""
SweepExp
========

Description
-----------
A python package for running parallel experiments across parameter grids with MPI.

For more information, visit the project's GitHub repository:
https://github.com/Gordi42/sweepexp
"""
__author__ = """Silvano Gordian Rosenau"""
__email__ = 'silvano.rosenau@uni-hamburg.de'
__version__ = '0.1.0'

from lazypimp import setup
from typing import TYPE_CHECKING

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    ...
    # import modules
    # from . import my_module

    # other imports
    # from .my_module import my_function

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    # "sweepexp": ["my_module"],
}

all_imports_by_origin = {
    # "sweepexp.my_module": ["my_function"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)

