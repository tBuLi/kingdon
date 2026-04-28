"""Top-level package for Kingdon."""

__author__ = """Martin Roelfs"""
__email__ = 'martinroelfs@yahoo.com'
__version__ = '2.1.1'

from sympy import symbols

from kingdon.algebra import Algebra
from kingdon.multivector import (
    MultiVector, Scalar, Vector, Bivector, PseudoBivector, PseudoVector, PseudoScalar, 
    Blade2, Reflection2,
    Direction, EVector, UPoint, Point, Translation, Line,  # PGA types
    stack
)
from kingdon.matrixreps import expr_as_matrix
