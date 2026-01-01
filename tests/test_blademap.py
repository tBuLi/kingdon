#!/usr/bin/env python

"""Tests for `kingdon.blademap` module."""

import pytest
import numpy as np

from kingdon import Algebra, MultiVector
from kingdon.blademap import BladeMap


@pytest.fixture
def stap():
    return Algebra(3, 1, 1, start_index=0)


@pytest.fixture
def pga3d():
    return Algebra(3, 0, 1, start_index=0)


@pytest.fixture
def pga2d():
    return Algebra(2, 0, 1, start_index=0)


def test_automatic_blade_alignment(stap, pga3d):
    """Test automatic blade alignment between two algebras."""
    auto_blade_map = BladeMap(alg1=stap, alg2=pga3d)
    assert auto_blade_map.alg1 == stap
    assert auto_blade_map.alg2 == pga3d
    assert len(auto_blade_map.blades_map) > 0


def test_manual_blade_mapping(stap, pga3d):
    """Test manual blade mapping."""
    manual_blade_map = BladeMap([
        (MultiVector(stap, e0=1), MultiVector(pga3d, e0=1)),
        (MultiVector(stap, e1=1), MultiVector(pga3d, e1=1)),
    ])
    assert len(manual_blade_map.blades_map) > 0


def test_mapping_multivector(stap, pga3d):
    """Test mapping a multivector in both directions."""
    auto_blade_map = BladeMap(alg1=stap, alg2=pga3d)
    test_mv = MultiVector(stap, e=2, e0=1, e1=3)
    mapped = auto_blade_map(test_mv)
    assert mapped.algebra == pga3d

    mapped_back = auto_blade_map(mapped)
    assert mapped_back.algebra == stap


def test_numpy_array_coefficients(stap, pga3d):
    """Test mapping multivectors with numpy array coefficients."""
    auto_blade_map = BladeMap(alg1=stap, alg2=pga3d)
    array_mv = MultiVector(stap, e=np.array([1, 2, 3]), e0=np.array([2, 1, 0]))
    mapped = auto_blade_map(array_mv)
    assert mapped.algebra == pga3d


def test_invalid_algebra_raises_error(stap, pga3d):
    """Test that mapping a multivector from unknown algebra raises error."""
    auto_blade_map = BladeMap(alg1=stap, alg2=pga3d)
    wrong_algebra = Algebra(4, 0, 0)
    test_mv = MultiVector(wrong_algebra, e=1)
    with pytest.raises(ValueError):
        auto_blade_map(test_mv)


def test_blademap_properties(stap, pga3d):
    """Test BladeMap properties."""
    auto_blade_map = BladeMap(alg1=stap, alg2=pga3d)
    assert auto_blade_map.b1[0].algebra == stap
    assert auto_blade_map.b2[0].algebra == pga3d
