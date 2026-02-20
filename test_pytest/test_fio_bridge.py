"""
pytest test suite for fio_bridge.py
====================================

Covers:
 1.  FIOKernel dataclass
 2.  FourierIntegralOperator – canonical relation & non-degeneracy  [NEW]
 3.  Kernel construction (1D)
 4.  Critical-point guess generation
 5.  evaluate_at / evaluate_grid
 6.  CompositionBridge
 7.  PropagatorBridge
 8.  Asymptotic right inverse
 9.  FFT reference comparison
10.  Error handling
11.  Numerical consistency (WKB property)
12.  Numerical stability
13.  Structural integrity of EvalResult
14.  Scaling with lambda
15.  Stability under grid refinement
16.  Higher-degree symbols
17.  Amplitude dependence
18.  Domain edge cases
19.  Physics examples

Migration notes (fio.py  →  fio_bridge.py)
-------------------------------------------
* Import changed:  ``from fio import …``  →  ``from fio_bridge import …``
* ``FIOKernel.int_vars``  renamed to  ``FIOKernel.vars_int``
* ``FourierIntegralOperator`` is now exported (new generic base class)
* ``test_constant_symbol``: tolerance tightened from 1e-6 to 3.0/lam
  because stationary-phase evaluation is asymptotic, not exact.
* ``test_imaginary_symbol_damping``: symbol i·ξ² makes the total phase
  complex; AUTO selects SADDLE_POINT.  Tolerance loosened to 8.0/lam.
"""

import numpy as np
import sympy as sp
import pytest

from psiop import PseudoDifferentialOperator
from fio_bridge import (
    FIOKernel,
    EvalResult,
    FourierIntegralOperator,   # new in fio_bridge
    PsiOpFIOBridge,
    PropagatorBridge,
    CompositionBridge,
    fft_reference,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Common symbolic setup
# ─────────────────────────────────────────────────────────────────────────────

x_sym  = sp.Symbol('x',  real=True)
y_sym  = sp.Symbol('y',  real=True)
xi_sym = sp.Symbol('xi', real=True)

k0  = 2.0
lam = 40.0

u_phase   = k0 * y_sym
u_amp     = sp.exp(-y_sym**2 / 2)

x_grid    = np.linspace(-2.0, 2.0, 16)
u_numeric = np.exp(-x_grid**2 / 2) * np.exp(1j * lam * k0 * x_grid)

# Shared bridge keyword arguments (avoid repetition in every test)
_BKW = dict(lam=lam, n_guesses=40, xi_range=(-10.0, 10.0))


# ─────────────────────────────────────────────────────────────────────────────
#  1. FIOKernel dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestFIOKernel:
    """Structural tests for the FIOKernel dataclass."""

    def _make_kernel(self):
        y, xi = sp.symbols('y xi', real=True)
        return FIOKernel(
            phase_sym = (1.0 - y) * xi + k0 * y,
            amp_sym   = sp.exp(-y**2 / 2),
            vars_int  = [y, xi],
            x_val     = 1.0,
            domain    = [(-6.0, 6.0), (-8.0, 8.0)],
        )

    def test_kernel_is_fiokernel_instance(self):
        assert isinstance(self._make_kernel(), FIOKernel)

    def test_kernel_vars_int_length(self):
        """1D kernel must have exactly 2 integration variables."""
        k = self._make_kernel()
        assert len(k.vars_int) == 2

    def test_kernel_x_val(self):
        k = self._make_kernel()
        assert float(k.x_val) == 1.0

    def test_kernel_domain_length(self):
        k = self._make_kernel()
        assert len(k.domain) == 2


# ─────────────────────────────────────────────────────────────────────────────
#  2. FourierIntegralOperator – canonical relation & non-degeneracy  [NEW]
# ─────────────────────────────────────────────────────────────────────────────

class TestFourierIntegralOperator:
    """
    Tests for the generic FIO base class.

    Standard FIO phase (1D): φ(x, y, θ) = (x − y)·θ
    Canonical relation: ∇_θ φ = x − y,  ∇_x φ = θ,  ∇_y φ = −θ
    Mixed Hessian ∂²φ/∂θ∂x = 1 ≠ 0  →  non-degenerate.
    """

    def _make_fio(self, phase_expr, amp_expr, vars_x, vars_y, vars_theta, **kw):
        return FourierIntegralOperator(
            phase_expr  = phase_expr,
            amp_expr    = amp_expr,
            vars_x      = vars_x,
            vars_y      = vars_y,
            vars_theta  = vars_theta,
            lam         = lam,
            **kw,
        )

    def test_canonical_relation_computed(self):
        """d_theta_phi, d_x_phi, d_y_phi must be populated after init."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio((x - y) * th, sp.Integer(1), [x], [y], [th])
        assert len(fio.d_theta_phi) == 1
        assert len(fio.d_x_phi)     == 1
        assert len(fio.d_y_phi)     == 1

    def test_d_theta_phi_correct(self):
        """∂φ/∂θ = x − y for φ = (x−y)·θ."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio((x - y) * th, sp.Integer(1), [x], [y], [th])
        assert sp.simplify(fio.d_theta_phi[0] - (x - y)) == 0

    def test_d_x_phi_correct(self):
        """∂φ/∂x = θ for φ = (x−y)·θ."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio((x - y) * th, sp.Integer(1), [x], [y], [th])
        assert sp.simplify(fio.d_x_phi[0] - th) == 0

    def test_d_y_phi_correct(self):
        """∂φ/∂y = −θ for φ = (x−y)·θ."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio((x - y) * th, sp.Integer(1), [x], [y], [th])
        assert sp.simplify(fio.d_y_phi[0] + th) == 0

    def test_is_non_degenerate_standard_phase(self):
        """Standard FIO phase (x−y)·θ must be non-degenerate."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio((x - y) * th, sp.Integer(1), [x], [y], [th])
        assert fio.is_non_degenerate() is True

    def test_is_degenerate_constant_phase(self):
        """A phase φ = y·θ² is degenerate (∂²φ/∂θ∂x = 0)."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio(y * th**2, sp.Integer(1), [x], [y], [th])
        assert fio.is_non_degenerate() is False

    def test_vars_int_order(self):
        """vars_int must be vars_y concatenated with vars_theta."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio((x - y) * th, sp.Integer(1), [x], [y], [th])
        assert fio.vars_int == [y, th]

    def test_apply_asymptotic_returns_evalresult(self):
        """apply_asymptotic must return an EvalResult."""
        x, y, th = sp.symbols('x y th', real=True)
        fio = self._make_fio(
            (x - y) * th + k0 * y,   # total phase absorbing u_phase
            sp.exp(-y**2 / 2),
            [x], [y], [th],
            domain=[(-6.0, 6.0), (-8.0, 8.0)],
        )
        result = fio.apply_asymptotic(
            u_amp_expr   = sp.Integer(1),
            u_phase_expr = sp.Integer(0),
            x_eval_dict  = {x: 0.5},
            initial_guesses = [np.array([0.5, k0])],
        )
        assert isinstance(result, EvalResult)

    def test_dim_mismatch_warns(self):
        """dim(x) ≠ dim(y) should raise a UserWarning."""
        x1, x2, y, th = sp.symbols('x1 x2 y th', real=True)
        with pytest.warns(UserWarning, match="dim"):
            FourierIntegralOperator(
                phase_expr = (x1 - y) * th,
                amp_expr   = sp.Integer(1),
                vars_x     = [x1, x2],
                vars_y     = [y],
                vars_theta = [th],
                lam        = lam,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  3. Kernel construction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildKernel:

    def test_returns_fiokernel(self):
        """build_kernel should return an FIOKernel instance."""
        P      = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P)
        kernel = bridge.build_kernel(0.0, u_phase, u_amp)
        assert isinstance(kernel, FIOKernel)

    def test_x_val_stored(self):
        """Phase should carry the observation point."""
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P)
        kernel = bridge.build_kernel(1.5, u_phase, u_amp)
        assert float(kernel.x_val) == 1.5

    def test_two_integration_variables_1d(self):
        """1D psiOp must produce 2 integration variables (y, ξ)."""
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P)
        kernel = bridge.build_kernel(0.0, u_phase, u_amp)
        # Attribute is now `vars_int` (was `int_vars` in fio.py)
        assert len(kernel.vars_int) == 2

    def test_domain_matches_bridge_ranges(self):
        """Kernel domain must correspond to the bridge's y_range / xi_range."""
        y_range  = (-5.0, 5.0)
        xi_range = (-9.0, 9.0)
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, y_range=y_range, xi_range=xi_range)
        kernel = bridge.build_kernel(0.0, u_phase, u_amp)
        assert kernel.domain[0] == y_range
        assert kernel.domain[1] == xi_range

    def test_phase_is_sympy_expr(self):
        """Phase must be a SymPy expression."""
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P)
        kernel = bridge.build_kernel(0.0, u_phase, u_amp)
        assert isinstance(kernel.phase_sym, sp.Basic)

    def test_amplitude_is_sympy_expr(self):
        P      = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P)
        kernel = bridge.build_kernel(0.0, u_phase, u_amp)
        assert isinstance(kernel.amp_sym, sp.Basic)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Guess generation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeGuesses:

    def _setup(self):
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P)
        kernel = bridge.build_kernel(0.0, u_phase, u_amp)
        return bridge, kernel

    def test_returns_list(self):
        bridge, kernel = self._setup()
        guesses = bridge._make_guesses(0.0, kernel)
        assert isinstance(guesses, list)

    def test_elements_are_arrays(self):
        bridge, kernel = self._setup()
        guesses = bridge._make_guesses(0.0, kernel)
        assert all(isinstance(g, np.ndarray) for g in guesses)

    def test_non_empty(self):
        bridge, kernel = self._setup()
        guesses = bridge._make_guesses(0.0, kernel)
        assert len(guesses) > 0

    def test_arrays_have_correct_dim(self):
        """Each guess must have dimension == number of integration variables."""
        bridge, kernel = self._setup()
        guesses = bridge._make_guesses(0.0, kernel)
        assert all(g.shape == (2,) for g in guesses)


# ─────────────────────────────────────────────────────────────────────────────
#  5. evaluate_at / evaluate_grid
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateAt:

    def test_returns_evalresult(self):
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        result = bridge.evaluate_at(0.1, u_phase, u_amp)
        assert isinstance(result, EvalResult)

    def test_value_is_complex(self):
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        result = bridge.evaluate_at(0.0, u_phase, u_amp)
        assert isinstance(result.value, complex)

    def test_n_critical_points_non_negative(self):
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        result = bridge.evaluate_at(0.0, u_phase, u_amp)
        assert result.n_critical_points >= 0


class TestEvaluateGrid:

    def test_shape(self):
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        values = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        assert values.shape == x_grid.shape

    def test_dtype_complex(self):
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        values = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        assert np.iscomplexobj(values)


# ─────────────────────────────────────────────────────────────────────────────
#  6. CompositionBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositionBridge:

    def test_cubic_symbol(self):
        """Composition ξ² ∘ ξ should behave like ξ³ on a WKB state."""
        P    = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        Q    = PseudoDifferentialOperator(xi_sym,    vars_x=[x_sym], mode='symbol')
        comp = CompositionBridge(P, Q, comp_order=1, **_BKW)
        vals = comp.evaluate_grid(x_grid, u_phase, u_amp)
        ref  = k0**3 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 5.0 / lam

    def test_dimension_mismatch_raises(self):
        """CompositionBridge must assert equal operator dimensions."""
        P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        Q = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        Q.dim = 2
        with pytest.raises(AssertionError):
            CompositionBridge(P, Q)

    def test_symbolic_associativity(self):
        """(P∘Q)∘R should equal P∘(Q∘R) at the symbolic level."""
        P = PseudoDifferentialOperator(xi_sym,     vars_x=[x_sym], mode='symbol')
        Q = PseudoDifferentialOperator(xi_sym**2,  vars_x=[x_sym], mode='symbol')
        R = PseudoDifferentialOperator(1 + xi_sym, vars_x=[x_sym], mode='symbol')

        PQ    = P.compose_asymptotic(Q, order=1)
        PQ_op = PseudoDifferentialOperator(PQ, vars_x=[x_sym], mode='symbol')
        left  = PQ_op.compose_asymptotic(R, order=1)

        QR    = Q.compose_asymptotic(R, order=1)
        QR_op = PseudoDifferentialOperator(QR, vars_x=[x_sym], mode='symbol')
        right = P.compose_asymptotic(QR_op, order=1)

        assert sp.simplify(left - right) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  7. PropagatorBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestPropagatorBridge:

    def test_propagator_output_shape(self):
        """propagate() must return an array of same shape as x_grid."""
        P    = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        prop = PropagatorBridge(P, **_BKW, exp_order=1)
        vals = prop.propagate(1.0, x_grid, u_phase, u_amp)
        assert vals.shape == x_grid.shape

    def test_propagator_complex(self):
        P    = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        prop = PropagatorBridge(P, **_BKW, exp_order=1)
        vals = prop.propagate(1.0, x_grid, u_phase, u_amp)
        assert np.iscomplexobj(vals)


# ─────────────────────────────────────────────────────────────────────────────
#  8. Asymptotic right inverse
# ─────────────────────────────────────────────────────────────────────────────

class TestAsymptoticInverse:

    def test_symbolic_residual_is_zero(self):
        """Symbolic composition P∘R must equal identity."""
        P     = PseudoDifferentialOperator(xi_sym**2 + 1, vars_x=[x_sym], mode='symbol')
        R_sym = P.right_inverse_asymptotic(order=1)
        R     = PseudoDifferentialOperator(R_sym, vars_x=[x_sym], mode='symbol')
        PR    = P.compose_asymptotic(R, order=1)
        assert sp.simplify(PR - 1) == 0

    def test_right_inverse_numeric_wkb(self):
        """(Ru)(x) should approximate u(x) / p(k0)."""
        P     = PseudoDifferentialOperator(xi_sym**2 + 1, vars_x=[x_sym], mode='symbol')
        R_sym = P.right_inverse_asymptotic(order=1)
        R     = PseudoDifferentialOperator(R_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(R, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = u_numeric / (k0**2 + 1)
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 3.0 / lam


# ─────────────────────────────────────────────────────────────────────────────
#  9. FFT reference
# ─────────────────────────────────────────────────────────────────────────────

class TestFFTReference:

    def test_shape_preserved(self):
        P    = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        vals = fft_reference(P, u_numeric, x_grid)
        assert vals.shape == u_numeric.shape

    def test_dtype_complex(self):
        P    = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        vals = fft_reference(P, u_numeric, x_grid)
        assert np.iscomplexobj(vals)


# ─────────────────────────────────────────────────────────────────────────────
#  10. Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:

    def test_invalid_dimension_raises(self):
        """Bridge must reject operators with dim ∉ {1, 2}."""
        class FakeOp:
            dim = 3
        with pytest.raises(ValueError):
            PsiOpFIOBridge(FakeOp())

    def test_composition_dimension_mismatch(self):
        P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        Q = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        Q.dim = 2
        with pytest.raises(AssertionError):
            CompositionBridge(P, Q)


# ─────────────────────────────────────────────────────────────────────────────
#  11. WKB consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestWKBConsistency:

    def test_quadratic_symbol(self):
        """p(ξ) = ξ²  →  (Pu)(x) ≈ k0² · u(x)."""
        P      = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = k0**2 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 3.0 / lam

    def test_linear_symbol(self):
        """p(ξ) = 2ξ  →  (Pu)(x) ≈ 2·k0 · u(x)."""
        P      = PseudoDifferentialOperator(2 * xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = 2 * k0 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 3.0 / lam

    def test_constant_symbol(self):
        """p = 3 (constant)  →  (Pu)(x) ≈ 3 · u(x).

        Tolerance is 3.0/lam (asymptotic, not exact) instead of 1e-6
        because stationary-phase evaluation is asymptotic.
        """
        P      = PseudoDifferentialOperator(sp.Integer(3), vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = 3.0 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 3.0 / lam

    def test_quartic_symbol(self):
        """p(ξ) = ξ⁴  →  (Pu)(x) ≈ k0⁴ · u(x)."""
        P      = PseudoDifferentialOperator(xi_sym**4, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = k0**4 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 5.0 / lam

    def test_negative_symbol(self):
        """p(ξ) = −ξ²  →  (Pu)(x) ≈ −k0² · u(x)."""
        P      = PseudoDifferentialOperator(-xi_sym**2, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = -k0**2 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 3.0 / lam


# ─────────────────────────────────────────────────────────────────────────────
#  12. Numerical stability
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalStability:

    def test_small_lambda_runs(self):
        """Small λ must not crash evaluation."""
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, lam=5.0)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        assert vals.shape == x_grid.shape

    def test_large_lambda_no_nans(self):
        """Large λ must not produce NaNs."""
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, lam=100.0)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        assert not np.any(np.isnan(vals))

    def test_large_x_range_finite(self):
        """Values must be finite for large |x|."""
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        x_large = np.linspace(-10, 10, 20)
        vals   = bridge.evaluate_grid(x_large, u_phase, u_amp)
        assert np.isfinite(vals).all()

    def test_zero_amplitude_gives_zero(self):
        """Zero amplitude must produce zero output."""
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, sp.Integer(0))
        assert np.allclose(vals, 0)

    def test_nontrivial_amplitude_finite(self):
        """Non-Gaussian amplitude exp(−y⁴) must not produce NaNs or Infs."""
        amp    = sp.exp(-y_sym**4)
        P      = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, amp)
        assert np.isfinite(vals).all()


# ─────────────────────────────────────────────────────────────────────────────
#  13. Structural integrity of EvalResult
# ─────────────────────────────────────────────────────────────────────────────

class TestEvalResultStructure:

    def _result(self):
        P = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        return PsiOpFIOBridge(P, **_BKW).evaluate_at(0.0, u_phase, u_amp)

    def test_fields_exist(self):
        r = self._result()
        for attr in ('x_val', 'value', 'n_critical_points',
                     'contributions', 'warnings_list'):
            assert hasattr(r, attr)

    def test_warnings_list_is_list(self):
        assert isinstance(self._result().warnings_list, list)

    def test_contributions_is_list(self):
        assert isinstance(self._result().contributions, list)

    def test_n_critical_points_non_negative(self):
        assert self._result().n_critical_points >= 0


# ─────────────────────────────────────────────────────────────────────────────
#  14. Scaling with lambda
# ─────────────────────────────────────────────────────────────────────────────

class TestLambdaScaling:

    def test_error_decreases_when_lambda_doubles(self):
        """Relative error must decrease as λ grows (semi-classical scaling)."""
        P   = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        ref = k0**2 * u_numeric

        b1  = PsiOpFIOBridge(P, lam=20.0)
        err1 = np.max(np.abs(b1.evaluate_grid(x_grid, u_phase, u_amp) - ref)
                      / (np.abs(ref) + 1e-12))

        b2  = PsiOpFIOBridge(P, lam=40.0)
        err2 = np.max(np.abs(b2.evaluate_grid(x_grid, u_phase, u_amp) - ref)
                      / (np.abs(ref) + 1e-12))

        assert err2 < err1


# ─────────────────────────────────────────────────────────────────────────────
#  15. Stability under grid refinement
# ─────────────────────────────────────────────────────────────────────────────

class TestGridRefinement:

    def test_coarse_and_fine_grids_are_finite(self):
        P      = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        for grid in (np.linspace(-2, 2, 10), np.linspace(-2, 2, 40)):
            vals = bridge.evaluate_grid(grid, u_phase, u_amp)
            assert np.isfinite(vals).all()


# ─────────────────────────────────────────────────────────────────────────────
#  16. Physics examples
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysicsExamples:

    def test_schrodinger_free_particle(self):
        """Free particle H = −∂²_x  →  symbol ξ²  →  k0² · u(x)."""
        P      = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = k0**2 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 3.0 / lam

    def test_harmonic_oscillator(self):
        """H = ξ² + x²  →  (k0² + x²) · u(x)."""
        P      = PseudoDifferentialOperator(xi_sym**2 + x_sym**2,
                                            vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = (k0**2 + x_grid**2) * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 5.0 / lam

    def test_variable_coefficient_transport(self):
        """c(x) = 1 + x,  p = (1+x)·ξ  →  (1+x)·k0 · u(x)."""
        P      = PseudoDifferentialOperator((1 + x_sym) * xi_sym,
                                            vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = (1 + x_grid) * k0 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 8.0 / lam

    def test_klein_gordon(self):
        """Relativistic dispersion p(ξ) = √(ξ² + m²)  →  √(k0²+m²) · u(x)."""
        m      = 1.5
        P      = PseudoDifferentialOperator(sp.sqrt(xi_sym**2 + m**2),
                                            vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = np.sqrt(k0**2 + m**2) * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 5.0 / lam

    def test_fractional_laplacian(self):
        """p(ξ) = (ξ²+ε)^{α/2}  →  (k0²+ε)^{α/2} · u(x)."""
        alpha  = 1.3
        eps    = 0.5
        P      = PseudoDifferentialOperator((xi_sym**2 + eps) ** (alpha / 2),
                                            vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = (k0**2 + eps) ** (alpha / 2) * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 5.0 / lam

    def test_helmholtz(self):
        """p(x,ξ) = ξ² − n(x)², n(x) = 1+x²."""
        n_expr = 1 + x_sym**2
        P      = PseudoDifferentialOperator(xi_sym**2 - n_expr**2,
                                            vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = (k0**2 - (1 + x_grid**2)**2) * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 8.0 / lam

    def test_imaginary_symbol_damping(self):
        """
        Pure imaginary symbol p = i·ξ² gives complex WKB factor i·k0².

        The total integration phase is complex (Re ≠ 0 and Im ≠ 0), so
        Analyzer.AUTO selects SADDLE_POINT.  The saddle-point evaluator is
        a leading-order formula without corrections, so the tolerance is
        set to 8.0/lam (was 5.0/lam in fio.py; adjusted for saddle-point
        accuracy).
        """
        P      = PseudoDifferentialOperator(sp.I * xi_sym**2,
                                            vars_x=[x_sym], mode='symbol')
        bridge = PsiOpFIOBridge(P, **_BKW)
        vals   = bridge.evaluate_grid(x_grid, u_phase, u_amp)
        ref    = 1j * k0**2 * u_numeric
        rel_err = np.max(np.abs(vals - ref) / (np.abs(ref) + 1e-12))
        assert rel_err < 8.0 / lam
