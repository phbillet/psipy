"""
test_factorization.py

Tests for:
    - factorize_symbolic
    - evaluate_decomposition_quality

These tests check:
    - exact separable reconstructions,
    - low-rank polynomial reconstructions,
    - zero/constant expressions,
    - 1D and 2D symbols,
    - complex-valued symbols,
    - non-polynomial smooth symbols,
    - Monte Carlo quality evaluation,
    - reproducibility with fixed seeds,
    - basic input validation.

The tolerances are deliberately somewhat conservative because:
    - factorize_symbolic converts numerical coefficients to SymPy floats,
    - evaluate_decomposition_quality uses random Monte Carlo samples.
"""

import numpy as np
import pytest
import sympy as sp
from sympy import symbols, cos, exp, I
import numpy.testing as npt

from psiop import factorize_symbolic, evaluate_decomposition_quality


# -----------------------------------------------------------------------------
# Shared keys and helpers
# -----------------------------------------------------------------------------

QUALITY_KEYS = ("rel_l2_error", "max_abs_error", "mean_abs_error")
FACTOR_KEYS = QUALITY_KEYS + ("svd_energy_retained_pct", "singular_values")


def _assert_quality_metrics(metrics, keys=QUALITY_KEYS):
    """Check that the quality metric dictionary is well-formed."""
    assert isinstance(metrics, dict)

    for key in keys:
        assert key in metrics

    assert np.isfinite(metrics["rel_l2_error"])
    assert np.isfinite(metrics["max_abs_error"])
    assert np.isfinite(metrics["mean_abs_error"])

    assert metrics["rel_l2_error"] >= 0.0
    assert metrics["max_abs_error"] >= 0.0
    assert metrics["mean_abs_error"] >= 0.0


def _assert_factorization_metrics(metrics):
    """Check metrics returned specifically by factorize_symbolic."""
    _assert_quality_metrics(metrics, keys=FACTOR_KEYS)

    assert 0.0 <= metrics["svd_energy_retained_pct"] <= 100.0 + 1e-12
    assert isinstance(metrics["singular_values"], np.ndarray)
    assert metrics["singular_values"].ndim == 1
    assert np.all(metrics["singular_values"] >= 0.0)


def _assert_symbolic_pairs(pairs, x_syms=None, xi_syms=None):
    """
    Check that the returned separable pairs are symbolic expressions.

    If x_syms / xi_syms are provided, also check variable separation:
        a_k should only depend on spatial variables,
        q_k should only depend on frequency variables.
    """
    assert isinstance(pairs, list)

    x_set = set(x_syms) if x_syms is not None else None
    xi_set = set(xi_syms) if xi_syms is not None else None

    for item in pairs:
        assert isinstance(item, tuple)
        assert len(item) == 2

        a_raw, q_raw = item
        a = sp.sympify(a_raw)
        q = sp.sympify(q_raw)

        assert isinstance(a, sp.Expr)
        assert isinstance(q, sp.Expr)

        if x_set is not None:
            assert a.free_symbols <= x_set

        if xi_set is not None:
            assert q.free_symbols <= xi_set


# -----------------------------------------------------------------------------
# factorize_symbolic: 1D polynomial / separable tests
# -----------------------------------------------------------------------------

def test_factorize_exact_separable_1d_polynomial():
    """Exact separable 1D polynomial should be recovered with one pair."""
    x, xi = symbols("x xi", real=True)
    expr = (1 + x**2) * (2 - 3*xi + xi**2)

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=3,
        tol=1e-8,
        num_samples=2000,
        seed=123,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    # Ideally exactly one pair. The error check is the main correctness check.
    assert len(pairs) == 1
    assert metrics["rel_l2_error"] < 1e-4
    assert metrics["max_abs_error"] < 1e-3
    assert metrics["svd_energy_retained_pct"] > 99.9


def test_factorize_rank_two_1d_polynomial():
    """Sum of two separable polynomial terms should be low-rank."""
    x, xi = symbols("x xi", real=True)
    expr = x*xi + x**2 * xi**2

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=2,
        tol=1e-8,
        num_samples=2000,
        seed=123,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    assert len(pairs) <= 2
    assert metrics["rel_l2_error"] < 1e-4
    assert metrics["max_abs_error"] < 1e-3
    assert metrics["svd_energy_retained_pct"] > 99.9


def test_factorize_frequency_only_expression():
    """A frequency-only symbol should be represented as constant(x) * q(xi)."""
    x, xi = symbols("x xi", real=True)
    expr = 3*xi**2 - xi

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=2,
        tol=1e-8,
        num_samples=2000,
        seed=77,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    assert len(pairs) <= 2
    assert metrics["rel_l2_error"] < 1e-4
    assert metrics["svd_energy_retained_pct"] > 99.9


def test_factorize_constant_expression():
    """A nonzero constant should be factorized without error."""
    x, xi = symbols("x xi", real=True)
    expr = sp.Integer(7)

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=1,
        tol=1e-8,
        num_samples=500,
        seed=11,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    assert len(pairs) >= 1
    assert metrics["rel_l2_error"] < 1e-4
    assert metrics["svd_energy_retained_pct"] > 99.9


def test_factorize_zero_expression():
    """Zero symbol should return no pairs and zero errors."""
    x, xi = symbols("x xi", real=True)
    expr = sp.Integer(0)

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=2,
        tol=1e-8,
        num_samples=200,
        seed=0,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    assert pairs == []
    assert metrics["rel_l2_error"] == 0.0
    assert metrics["max_abs_error"] == 0.0
    assert metrics["mean_abs_error"] == 0.0
    assert metrics["svd_energy_retained_pct"] == 100.0
    assert metrics["singular_values"].size == 0


def test_factorize_invalid_degree():
    """degree < 1 should raise ValueError."""
    x, xi = symbols("x xi", real=True)
    expr = x*xi

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    with pytest.raises(ValueError):
        factorize_symbolic(
            expr,
            [x],
            [xi],
            bounds,
            degree=0,
        )


# -----------------------------------------------------------------------------
# factorize_symbolic: 2D tests
# -----------------------------------------------------------------------------

def test_factorize_exact_separable_2d_polynomial():
    """Exact separable 2D polynomial should be accurately recovered."""
    x, y, xi, eta = symbols("x y xi eta", real=True)
    expr = (1 + x + y**2) * (2 - xi + eta**2)

    bounds = {
        x: (-1.0, 1.0),
        y: (-1.0, 1.0),
        xi: (-1.0, 1.0),
        eta: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x, y],
        [xi, eta],
        bounds,
        degree=2,
        tol=1e-8,
        num_samples=2000,
        seed=3,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x, y], xi_syms=[xi, eta])
    _assert_factorization_metrics(metrics)

    # Ideally one pair; allow a small numerical margin.
    assert len(pairs) <= 2
    assert metrics["rel_l2_error"] < 1e-4
    assert metrics["max_abs_error"] < 1e-3
    assert metrics["svd_energy_retained_pct"] > 99.9


def test_factorize_rank_two_2d_polynomial():
    """2D sum x*xi + y*eta should be low-rank and accurately reconstructed."""
    x, y, xi, eta = symbols("x y xi eta", real=True)
    expr = x*xi + y*eta

    bounds = {
        x: (-1.0, 1.0),
        y: (-1.0, 1.0),
        xi: (-1.0, 1.0),
        eta: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x, y],
        [xi, eta],
        bounds,
        degree=2,
        tol=1e-8,
        num_samples=2000,
        seed=5,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x, y], xi_syms=[xi, eta])
    _assert_factorization_metrics(metrics)

    assert len(pairs) <= 2
    assert metrics["rel_l2_error"] < 1e-4
    assert metrics["svd_energy_retained_pct"] > 99.9


# -----------------------------------------------------------------------------
# factorize_symbolic: complex and non-polynomial tests
# -----------------------------------------------------------------------------

def test_factorize_complex_rank_two_polynomial():
    """Complex-valued polynomial symbols should be handled correctly."""
    x, xi = symbols("x xi", real=True)
    expr = (1 + 2*I) * x*xi + (3 - 4*I) * x**2 * xi**2

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=2,
        tol=1e-8,
        num_samples=2000,
        seed=13,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    assert len(pairs) <= 2
    assert metrics["rel_l2_error"] < 1e-4
    assert metrics["svd_energy_retained_pct"] > 99.9


def test_factorize_nonpolynomial_separable_approximation():
    """
    Separable non-polynomial symbol exp(x)*cos(xi).

    This is not represented exactly by a finite Chebyshev expansion, but
    a moderate degree should give a good approximation on [-1, 1]^2.
    """
    x, xi = symbols("x xi", real=True)
    expr = exp(x) * cos(xi)

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=8,
        tol=1e-8,
        num_samples=2000,
        seed=17,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    assert len(pairs) >= 1
    assert metrics["rel_l2_error"] < 1e-2
    assert metrics["svd_energy_retained_pct"] > 99.0


def test_factorize_nonseparable_smooth_function():
    """
    Non-separable smooth symbol exp(x*xi).

    This checks that the Chebyshev/SVD pipeline produces a finite,
    reasonably accurate low-rank approximation.
    """
    x, xi = symbols("x xi", real=True)
    expr = exp(x*xi)

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, metrics = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=6,
        tol=1e-8,
        num_samples=3000,
        seed=19,
        digits=10,
    )

    _assert_symbolic_pairs(pairs, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics)

    assert len(pairs) >= 1
    assert metrics["rel_l2_error"] < 0.2
    assert metrics["svd_energy_retained_pct"] > 90.0


# -----------------------------------------------------------------------------
# evaluate_decomposition_quality tests
# -----------------------------------------------------------------------------

def test_evaluate_decomposition_quality_exact_separable():
    """Exact separable pair should give zero Monte Carlo error."""
    x, xi = symbols("x xi", real=True)
    expr = (1 + x) * cos(xi)

    pairs = [
        (1 + x, cos(xi)),
    ]

    bounds = {
        x: (-2.0, 2.0),
        xi: (-1.0, 1.0),
    }

    metrics = evaluate_decomposition_quality(
        expr,
        pairs,
        [x],
        [xi],
        bounds,
        num_samples=2000,
        seed=99,
    )

    _assert_quality_metrics(metrics)
    assert metrics["rel_l2_error"] < 1e-10
    assert metrics["max_abs_error"] < 1e-10
    assert metrics["mean_abs_error"] < 1e-10


def test_evaluate_decomposition_quality_exact_sum():
    """Exact sum of separable pairs should give zero Monte Carlo error."""
    x, xi = symbols("x xi", real=True)
    expr = x*xi + (1 - x**2) * cos(xi)

    pairs = [
        (x, xi),
        (1 - x**2, cos(xi)),
    ]

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    metrics = evaluate_decomposition_quality(
        expr,
        pairs,
        [x],
        [xi],
        bounds,
        num_samples=2000,
        seed=101,
    )

    _assert_quality_metrics(metrics)
    assert metrics["rel_l2_error"] < 1e-10
    assert metrics["max_abs_error"] < 1e-10


def test_evaluate_decomposition_quality_zero_original_empty_pairs():
    """Zero original expression with empty approximation should be exact."""
    x, xi = symbols("x xi", real=True)
    expr = sp.Integer(0)

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    metrics = evaluate_decomposition_quality(
        expr,
        [],
        [x],
        [xi],
        bounds,
        num_samples=100,
        seed=0,
    )

    _assert_quality_metrics(metrics)
    assert metrics["rel_l2_error"] == 0.0
    assert metrics["max_abs_error"] == 0.0
    assert metrics["mean_abs_error"] == 0.0


def test_evaluate_decomposition_quality_nonzero_original_empty_pairs():
    """Empty approximation of a nonzero expression should have relative error 1."""
    x, xi = symbols("x xi", real=True)
    expr = 1 + x*xi

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    metrics = evaluate_decomposition_quality(
        expr,
        [],
        [x],
        [xi],
        bounds,
        num_samples=1000,
        seed=2,
    )

    _assert_quality_metrics(metrics)

    # If approximation is identically zero:
    #   ||orig - 0|| / ||orig|| = 1
    npt.assert_allclose(metrics["rel_l2_error"], 1.0, rtol=1e-12, atol=1e-12)


def test_evaluate_decomposition_quality_wrong_approximation():
    """A deliberately wrong approximation should produce a large error."""
    x, xi = symbols("x xi", real=True)
    expr = x*xi

    # Approximate x*xi by x only.
    pairs = [
        (x, sp.Integer(1)),
    ]

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    metrics = evaluate_decomposition_quality(
        expr,
        pairs,
        [x],
        [xi],
        bounds,
        num_samples=2000,
        seed=7,
    )

    _assert_quality_metrics(metrics)
    assert metrics["rel_l2_error"] > 0.1
    assert metrics["max_abs_error"] > 0.0


def test_evaluate_decomposition_quality_complex_exact():
    """Complex-valued exact separable expression should give zero error."""
    x, xi = symbols("x xi", real=True)
    expr = (1 + 2*I) * x * xi

    pairs = [
        ((1 + 2*I) * x, xi),
    ]

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    metrics = evaluate_decomposition_quality(
        expr,
        pairs,
        [x],
        [xi],
        bounds,
        num_samples=2000,
        seed=23,
    )

    _assert_quality_metrics(metrics)
    assert metrics["rel_l2_error"] < 1e-10
    assert metrics["max_abs_error"] < 1e-10


def test_evaluate_decomposition_quality_constant_expression():
    """Constant expression with constant separable pair should be exact."""
    x, xi = symbols("x xi", real=True)
    expr = sp.Integer(5)

    pairs = [
        (sp.Integer(5), sp.Integer(1)),
    ]

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    metrics = evaluate_decomposition_quality(
        expr,
        pairs,
        [x],
        [xi],
        bounds,
        num_samples=500,
        seed=29,
    )

    _assert_quality_metrics(metrics)
    assert metrics["rel_l2_error"] < 1e-12
    assert metrics["max_abs_error"] < 1e-12


# -----------------------------------------------------------------------------
# Combined factorize + evaluate tests
# -----------------------------------------------------------------------------

def test_factorize_then_evaluate_on_fresh_samples():
    """Pairs from factorize_symbolic should evaluate well on fresh samples."""
    x, xi = symbols("x xi", real=True)
    expr = x*xi + x**2 * xi**2

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    pairs, _ = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        degree=2,
        tol=1e-8,
        num_samples=500,
        seed=23,
        digits=10,
    )

    metrics = evaluate_decomposition_quality(
        expr,
        pairs,
        [x],
        [xi],
        bounds,
        num_samples=1500,
        seed=24,
    )

    _assert_quality_metrics(metrics)
    assert metrics["rel_l2_error"] < 1e-4


def test_factorize_reproducible_with_same_seed():
    """Same seed should give the same Monte Carlo quality metric."""
    x, xi = symbols("x xi", real=True)
    expr = exp(x*xi)

    bounds = {
        x: (-1.0, 1.0),
        xi: (-1.0, 1.0),
    }

    kwargs = dict(
        degree=4,
        tol=1e-6,
        num_samples=700,
        digits=10,
    )

    pairs_1, metrics_1 = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        seed=123,
        **kwargs,
    )

    pairs_2, metrics_2 = factorize_symbolic(
        expr,
        [x],
        [xi],
        bounds,
        seed=123,
        **kwargs,
    )

    _assert_symbolic_pairs(pairs_1, x_syms=[x], xi_syms=[xi])
    _assert_symbolic_pairs(pairs_2, x_syms=[x], xi_syms=[xi])
    _assert_factorization_metrics(metrics_1)
    _assert_factorization_metrics(metrics_2)

    assert len(pairs_1) == len(pairs_2)
    assert metrics_1["rel_l2_error"] == pytest.approx(
        metrics_2["rel_l2_error"],
        rel=1e-12,
        abs=1e-12,
    )


# -----------------------------------------------------------------------------
# Input validation tests
# -----------------------------------------------------------------------------

def test_factorize_missing_bound_raises():
    """factorize_symbolic should fail if a frequency bound is missing."""
    x, xi = symbols("x xi", real=True)
    expr = x*xi

    bounds = {
        x: (-1.0, 1.0),
        # xi missing
    }

    with pytest.raises(KeyError):
        factorize_symbolic(
            expr,
            [x],
            [xi],
            bounds,
            degree=1,
        )


def test_evaluate_missing_bound_raises():
    """evaluate_decomposition_quality should fail if a bound is missing."""
    x, xi = symbols("x xi", real=True)
    expr = x*xi

    pairs = [
        (x, xi),
    ]

    bounds = {
        x: (-1.0, 1.0),
        # xi missing
    }

    with pytest.raises(KeyError):
        evaluate_decomposition_quality(
            expr,
            pairs,
            [x],
            [xi],
            bounds,
            num_samples=10,
            seed=0,
        )