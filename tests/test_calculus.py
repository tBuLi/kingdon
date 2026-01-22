#!/usr/bin/env python

"""Tests for `kingdon.calculus` package."""
import pytest
import numpy as np

from kingdon import Algebra
from kingdon.calculus import d, da, chain, product


@pytest.fixture
def sta():
    """Spacetime algebra (STA) equivalent - signature (1,3,0)"""
    return Algebra(1, 3, 0)


def is_close(a, b, tol=1e-6):
    """Check if two multivectors are close within tolerance."""
    try:
        diff = a - b
        # Check if all coefficients are small
        if hasattr(diff, '_values') and diff._values:
            return all(abs(val) < tol for val in diff._values)
        return True  # Zero multivector case
    except:
        return abs(a - b) < tol


def is_closish(a, b, tol=1e-3):
    """Check if two multivectors are close within a looser tolerance."""
    return is_close(a, b, tol)


def test_eq_1p6(sta):
    """Definition a-derivative in terms of d of equation 1.6"""
    f = lambda x: x * x
    a = sta.random_vector()
    af = lambda x: a * f(x)

    x = sta.random_vector()

    lhs = da(f=f, a=a)(x)
    rhs = (1/2 * (a * d(sta, f)(x) + d(sta, af)(x)))

    assert is_closish(lhs, rhs)


def test_d1(sta):
    """Test derivative of inner product with bivector"""
    x = sta.random_vector()
    B = sta.random_bivector()

    f = lambda x: x | B  # Inner product with bivector
    df = d(sta, f, tau=1e-5)(x)

    expected = 2 * B

    assert is_close(df, expected)


def test_d2(sta):
    """Test derivative of x^2 (simple scalar field)"""
    x = sta.random_vector()
    f = lambda x: x.normsq()  # Use normsq() instead of x**2
    df = d(sta, f, tau=1e-8)(x)

    expected = 2 * x

    assert is_close(df, expected)


def test_d3(sta):
    """Test derivative with trivector"""
    x = sta.random_vector()
    T = sta.random_trivector()

    f = lambda x: x | T  # Inner product with trivector
    df = d(sta, f, tau=1e-5)(x)

    expected = 3 * T

    assert is_close(df, expected)


def test_chain_rule(sta):
    """Test chain rule"""
    x = sta.random_vector()

    # Simple functions for testing
    f = lambda x: x
    g = lambda x: x.normsq()

    fog = lambda x: f(g(x))  # Composition f(g(x))

    # Test chain rule
    chain_result = chain(sta, f, g)(x)
    d_fog_result = d(sta, fog)(x)

    assert is_closish(chain_result, d_fog_result)


def test_product_rule(sta):
    """Test product rule"""
    x = sta.random_vector()

    # Simple functions for testing
    f = lambda x: x
    g = lambda x: x.normsq()

    fg = lambda x: f(x) * g(x)  # Product f(x)*g(x)

    # Test product rule
    product_result = product(sta, f, g)(x)
    d_fg_result = d(sta, fg)(x)

    assert is_closish(product_result, d_fg_result)