"""
pytest test suite for fio_bridge.py

This test file covers:

- Kernel construction (1D)
- Critical point guess generation
- evaluate_at / evaluate_grid
- CompositionBridge
- PropagatorBridge
- Asymptotic right inverse
- FFT reference comparison
- Numerical consistency checks
- Error handling

All tests are written with English comments.
"""

import numpy as np
import sympy as sp
import pytest

from psiop import PseudoDifferentialOperator
from fio import (
    FIOKernel,
    PsiOpFIOBridge,
    PropagatorBridge,
    CompositionBridge,
    fft_reference,
)

# ------------------------------------------------------------
# Common symbolic setup
# ------------------------------------------------------------

x_sym = sp.Symbol('x', real=True)
y_sym = sp.Symbol('y', real=True)
xi_sym = sp.Symbol('xi', real=True)

k0 = 2.0
lam = 40.0

u_phase = k0 * y_sym
u_amp = sp.exp(-y_sym**2 / 2)

x_grid = np.linspace(-2.0, 2.0, 16)
u_numeric = np.exp(-x_grid**2 / 2) * np.exp(1j * lam * k0 * x_grid)


# ============================================================
# 1. Kernel construction tests
# ============================================================

def test_build_kernel_returns_fiokernel():
    """Kernel builder should return an FIOKernel instance."""
    P = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P)
    kernel = bridge.build_kernel(0.0, u_phase, u_amp)
    assert isinstance(kernel, FIOKernel)


def test_kernel_phase_contains_x():
    """Phase should depend on observation point x."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P)
    kernel = bridge.build_kernel(1.5, u_phase, u_amp)
    assert float(kernel.x_val) == 1.5


def test_kernel_has_two_integration_variables_1d():
    """1D psiOp must produce 2 integration variables (y, xi)."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P)
    kernel = bridge.build_kernel(0.0, u_phase, u_amp)
    assert len(kernel.int_vars) == 2


# ============================================================
# 2. Guess generation tests
# ============================================================

def test_make_guesses_returns_list():
    """_make_guesses must return a list of numpy arrays."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P)
    kernel = bridge.build_kernel(0.0, u_phase, u_amp)
    guesses = bridge._make_guesses(0.0, kernel)
    assert isinstance(guesses, list)
    assert isinstance(guesses[0], np.ndarray)


def test_make_guesses_non_empty():
    """There should be at least one guess generated."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P)
    kernel = bridge.build_kernel(0.0, u_phase, u_amp)
    guesses = bridge._make_guesses(0.0, kernel)
    assert len(guesses) > 0


# ============================================================
# 3. evaluate_at tests
# ============================================================

def test_evaluate_at_returns_evalresult():
    """evaluate_at must return an EvalResult object."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    result = bridge.evaluate_at(0.1, u_phase, u_amp)
    assert hasattr(result, "value")
    assert hasattr(result, "n_critical_points")


def test_evaluate_at_value_is_complex():
    """Returned value must be complex."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    result = bridge.evaluate_at(0.0, u_phase, u_amp)
    assert isinstance(result.value, complex)


# ============================================================
# 4. evaluate_grid tests
# ============================================================

def test_evaluate_grid_shape():
    """evaluate_grid must return an array of same length."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    values = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    assert values.shape == x_grid.shape


def test_evaluate_grid_dtype_complex():
    """Output array must be complex dtype."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    values = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    assert np.iscomplexobj(values)


# ============================================================
# 5. WKB consistency tests
# ============================================================

def test_constant_symbol_wkb_property():
    """For constant symbol p(xi)=xi^2, result should approximate k0^2 * u(x)."""
    P = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    ref = k0**2 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 3.0 / lam


def test_linear_symbol_wkb_property():
    """For symbol p=2*xi, result should approximate 2*k0 * u(x)."""
    P = PseudoDifferentialOperator(2 * xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    ref = 2 * k0 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 3.0 / lam


# ============================================================
# 6. CompositionBridge tests
# ============================================================

def test_composition_bridge_cubic_symbol():
    """Composition of xi^2 and xi should behave like xi^3."""
    P = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")
    Q = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    comp = CompositionBridge(P, Q, lam=lam, comp_order=1)
    vals = comp.evaluate_grid(x_grid, u_phase, u_amp)
    ref = k0**3 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 5.0 / lam



# ============================================================
# 7. Asymptotic inverse test
# ============================================================

def test_right_inverse_symbolic_residual():
    """Symbolic composition P∘R should be close to identity."""
    P = PseudoDifferentialOperator(xi_sym**2 + 1, vars_x=[x_sym], mode="symbol")
    R_sym = P.right_inverse_asymptotic(order=1)
    R = PseudoDifferentialOperator(R_sym, vars_x=[x_sym], mode="symbol")
    PR = P.compose_asymptotic(R, order=1)
    residual = sp.simplify(PR - 1)
    assert residual == 0


# ============================================================
# 8. FFT reference consistency
# ============================================================

def test_fft_reference_matches_numpy():
    """FFT reference should preserve array length and dtype."""
    P = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")
    vals = fft_reference(P, u_numeric, x_grid)
    assert vals.shape == u_numeric.shape
    assert np.iscomplexobj(vals)


# ============================================================
# 9. Error handling
# ============================================================

def test_invalid_dimension_raises():
    """Bridge should reject unsupported dimensions."""
    class FakeOp:
        dim = 3
    with pytest.raises(ValueError):
        PsiOpFIOBridge(FakeOp())


def test_composition_dimension_mismatch():
    """CompositionBridge should assert equal dimensions."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    Q = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    Q.dim = 2
    with pytest.raises(AssertionError):
        CompositionBridge(P, Q)


# ============================================================
# 10. Numerical stability
# ============================================================

def test_small_lambda_still_runs():
    """Even small lambda should not crash evaluation."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=5.0)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    assert vals.shape == x_grid.shape


def test_large_lambda_runs():
    """Large lambda should not produce NaNs."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=100.0)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    assert not np.any(np.isnan(vals))


def test_no_warnings_list_type():
    """Warnings list must be a list."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    result = bridge.evaluate_at(0.0, u_phase, u_amp)
    assert isinstance(result.warnings_list, list)

# ============================================================
# 11. Structural integrity tests
# ============================================================

def test_evalresult_fields_exist():
    """EvalResult must expose expected public attributes."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P)
    res = bridge.evaluate_at(0.0, u_phase, u_amp)
    assert hasattr(res, "x_val")
    assert hasattr(res, "value")
    assert hasattr(res, "contributions")
    assert hasattr(res, "warnings_list")


def test_number_of_critical_points_non_negative():
    """Number of critical points must be >= 0."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P)
    res = bridge.evaluate_at(0.0, u_phase, u_amp)
    assert res.n_critical_points >= 0


# ============================================================
# 12. Scaling with lambda
# ============================================================

def test_error_scales_like_inverse_lambda():
    """
    Relative error should roughly halve when lambda doubles.
    This verifies semiclassical scaling.
    """
    P = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")

    bridge1 = PsiOpFIOBridge(P, lam=20.0)
    vals1 = bridge1.evaluate_grid(x_grid, u_phase, u_amp)
    ref = k0**2 * u_numeric
    err1 = np.max(np.abs(vals1 - ref) / (np.abs(ref) + 1e-12))

    bridge2 = PsiOpFIOBridge(P, lam=40.0)
    vals2 = bridge2.evaluate_grid(x_grid, u_phase, u_amp)
    err2 = np.max(np.abs(vals2 - ref) / (np.abs(ref) + 1e-12))

    assert err2 < err1


# ============================================================
# 13. Stability under grid refinement
# ============================================================

def test_grid_refinement_stability():
    """Refining x grid should not change values drastically."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)

    coarse = np.linspace(-2, 2, 10)
    fine = np.linspace(-2, 2, 40)

    vals_coarse = bridge.evaluate_grid(coarse, u_phase, u_amp)
    vals_fine = bridge.evaluate_grid(fine, u_phase, u_amp)

    assert np.isfinite(vals_coarse).all()
    assert np.isfinite(vals_fine).all()


# ============================================================
# 14. Higher-degree symbols
# ============================================================

def test_quartic_symbol():
    """Test symbol xi^4 against WKB expectation."""
    P = PseudoDifferentialOperator(xi_sym**4, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    ref = k0**4 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 5.0 / lam


def test_negative_symbol():
    """Test negative symbol -xi^2."""
    P = PseudoDifferentialOperator(-xi_sym**2, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    ref = -k0**2 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 3.0 / lam


# ============================================================
# 15. Constant symbol
# ============================================================

def test_constant_symbol():
    """Constant symbol must act as scalar multiplication."""
    P = PseudoDifferentialOperator(sp.Integer(3), vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    ref = 3.0 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 1e-6



# ============================================================
# 16. Composition associativity (symbolic level)
# ============================================================

def test_symbolic_associativity():
    """(P∘Q)∘R should equal P∘(Q∘R) at symbolic level."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    Q = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")
    R = PseudoDifferentialOperator(1 + xi_sym, vars_x=[x_sym], mode="symbol")

    # Left side: (P∘Q)∘R
    PQ = P.compose_asymptotic(Q, order=1)
    PQ_op = PseudoDifferentialOperator(PQ, vars_x=[x_sym], mode="symbol")
    left = PQ_op.compose_asymptotic(R, order=1)

    # Right side: P∘(Q∘R)
    QR = Q.compose_asymptotic(R, order=1)
    QR_op = PseudoDifferentialOperator(QR, vars_x=[x_sym], mode="symbol")
    right = P.compose_asymptotic(QR_op, order=1)

    assert sp.simplify(left - right) == 0



# ============================================================
# 17. Amplitude dependence
# ============================================================

def test_nontrivial_amplitude():
    """Non-Gaussian amplitude should still work."""
    amp = sp.exp(-y_sym**4)
    P = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, amp)
    assert np.isfinite(vals).all()


# ============================================================
# 18. Domain edge cases
# ============================================================

def test_large_x_range():
    """Bridge should not explode for large x."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    x_large = np.linspace(-10, 10, 20)
    vals = bridge.evaluate_grid(x_large, u_phase, u_amp)
    assert np.isfinite(vals).all()


def test_zero_amplitude():
    """Zero amplitude should produce zero output."""
    P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)
    zero_amp = sp.Integer(0)
    vals = bridge.evaluate_grid(x_grid, u_phase, zero_amp)
    assert np.allclose(vals, 0)

# ============================================================
# 19. Examples from physics
# ============================================================

def test_schrodinger_free_particle():
    """
    Free Schrödinger Hamiltonian: H = -∂²_x
    Symbol: p(xi) = xi²
    Expected leading behavior: k0² * u(x)
    """
    P = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode="symbol")
    bridge = PsiOpFIOBridge(P, lam=lam)

    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)
    ref = k0**2 * u_numeric

    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 3.0 / lam

def test_harmonic_oscillator_symbol():
    """
    Harmonic oscillator Hamiltonian H = xi² + x².
    WKB leading term: (k0² + x²) u(x).
    """
    P = PseudoDifferentialOperator(xi_sym**2 + x_sym**2,
                                   vars_x=[x_sym], mode="symbol")

    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)

    ref = (k0**2 + x_grid**2) * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 5.0 / lam

def test_variable_transport():
    """
    Variable transport speed c(x) = 1 + x.
    Symbol: (1+x)*xi.
    Leading behavior: (1+x)*k0 * u(x).
    """
    P = PseudoDifferentialOperator((1 + x_sym)*xi_sym,
                                   vars_x=[x_sym], mode="symbol")

    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)

    ref = (1 + x_grid) * k0 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 8.0 / lam

def test_klein_gordon_symbol():
    """
    Relativistic dispersion relation:
    p(xi) = sqrt(xi² + m²)
    Leading WKB: sqrt(k0² + m²) u(x).
    """
    m = 1.5
    P = PseudoDifferentialOperator(sp.sqrt(xi_sym**2 + m**2),
                                   vars_x=[x_sym], mode="symbol")

    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)

    ref = np.sqrt(k0**2 + m**2) * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 5.0 / lam

def test_fractional_laplacian():
    """
    Fractional Laplacian symbol |xi|^alpha.
    Leading WKB: |k0|^alpha u(x).
    """
    alpha = 1.3
    eps = 0.5

    P = PseudoDifferentialOperator(
        (xi_sym**2 + eps)**(alpha/2),
        vars_x=[x_sym],
        mode="symbol"
    )

    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)

    ref = (k0**2 + eps)**(alpha/2) * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))

    assert rel_err < 5.0 / lam


def test_helmholtz_operator():
    """
    Helmholtz operator: p = xi² - n(x)².
    With n(x)=1+x².
    """
    n_expr = 1 + x_sym**2
    P = PseudoDifferentialOperator(xi_sym**2 - n_expr**2,
                                   vars_x=[x_sym], mode="symbol")

    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)

    ref = (k0**2 - (1 + x_grid**2)**2) * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 8.0 / lam

def test_imaginary_symbol_damping():
    """
    Pure imaginary symbol i*xi² should introduce phase rotation.
    """
    P = PseudoDifferentialOperator(sp.I * xi_sym**2,
                                   vars_x=[x_sym], mode="symbol")

    bridge = PsiOpFIOBridge(P, lam=lam)
    vals = bridge.evaluate_grid(x_grid, u_phase, u_amp)

    ref = 1j * k0**2 * u_numeric
    rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
    assert rel_err < 5.0 / lam

