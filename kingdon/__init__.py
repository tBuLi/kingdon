"""Top-level package for Kingdon."""

__author__ = """Martin Roelfs"""
__email__ = 'martinroelfs@yahoo.com'
__version__ = '2.1.1'

import sys

from sympy import symbols

from kingdon.algebra import Algebra
from kingdon.multivector import (
    MultiVector, # Generic MultiVector type.
    Scalar, Vector, Bivector, Trivector, Quadvector, Pentavector, Hexavector, Heptavector, Octovector, # k-vectors
    Bireflection, # compositions
    Direction, EVector, UPoint, Point, Translation,  # PGA types
    stack
)
from kingdon.matrixreps import expr_as_matrix


def load_ipython_extension(ipython):
    """ Register the :code:`%%graph` cell magic; called by :code:`%load_ext kingdon`. """
    from kingdon.graph_magic import register_magics
    register_magics(ipython)


# Register the magic when kingdon is imported into a running IPython, without importing IPython
# ourselves: if it is not already imported, there is no shell for us to register with anyway.
if (_ipython := sys.modules.get('IPython')) and (_shell := _ipython.get_ipython()):
    load_ipython_extension(_shell)
