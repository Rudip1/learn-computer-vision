"""Multi-view geometry: C++ core with a thin Python layer.

Point sets are passed as ``N x 2`` (image) or ``N x 3`` (scene) NumPy arrays, one point per row. Every algorithm
lives in the compiled module :mod:`mvg._core`; the Python files only plot, generate synthetic data and glue
notebooks together.
"""

from . import _core
from ._core import *  # noqa: F401,F403

__version__ = "0.1.0"
