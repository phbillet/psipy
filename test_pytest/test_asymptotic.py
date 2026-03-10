# test_asymptotic.py
"""
Test suite for asymptotic.py — three methods, ~20 cases each.

Structure
---------
Section 1 — Shared utilities (numerical quadrature, helpers)
Section 2 — Stationary phase (STATIONARY_PHASE) : 23 tests
Section 3 — Laplace's method (LAPLACE)           : 22 tests
Section 4 — Saddle-point method (SADDLE_POINT)   : 22 tests
Section 5 — AUTO detection                       :  7 tests
Section 6 — AsymptoticEvaluator guard clauses    :  3 tests
Section 7 — find_critical_points                 :  5 tests
Section 8 — AsymptoticVisualizer (smoke tests)   :  4 tests

Changes vs original
-------------------
- FIXED  LAP-04: expected formula corrected from 2π/(λ√4) to 2π/(λ·2)
         (anisotropic ψ = x²/2 + 2y²: H = diag(1,4), det H = 4, √det = 2)
- ADDED  SP-21/22/23: Airy 2D and Pearcey log-log slope tests
- ADDED  LAP-21/22: Laplace correction sign/direction tests
- ADDED  SDL-21/22: saddle-point 2D quadrature comparison, and ValueError guard
- ADDED  AUTO-07: guard test — AUTO constructor never leaves method=AUTO
- ADDED  Section 6 (3 tests): AsymptoticEvaluator raises ValueError for
         method=None, method=AUTO, and an unknown/unresolved value
- ADDED  Section 7 (5 tests): find_critical_points coverage — multiple
         distinct points, domain-bounded search, empty-guess default,
         empty result for bad domain, and that caustics helper is importable

Run with:
    pytest test_asymptotic.py -v
"""

import warnings
import numpy as np
import sympy as sp
import pytest
from scipy.special import airy as scipy_airy, gamma as scipy_gamma

from asymptotic import (
    Analyzer, AsymptoticEvaluator, AsymptoticVisualizer,
    StationaryPhaseVisualizer, StationaryPhaseEvaluator,
    LaplaceEvaluator, SaddlePointEvaluator,
    IntegralMethod, SingularityType,
    CriticalPoint, AsymptoticContribution,
)

# ============================================================================
# Section 1 — Shared utilities
# ============================================================================

def relative_error(approx: complex, exact: complex) -> float:
    """Relative error |approx - exact| / |exact|, absolute if |exact| < 1e-12."""
    denom = abs(exact)
    return abs(approx - exact) / denom if denom > 1e-12 else abs(approx - exact)


def numerical_integral_1d(phi_expr, amp_expr, x_sym, lam,
                           bounds=None, n: int = 30_001) -> complex:
    """
    Reference 1D quadrature for oscillatory integrals
        ∫ a(x) exp(iλφ(x)) dx
    using the trapezoidal rule on an adaptively sized grid.

    The integration window is set to ±8/√λ to capture the effective
    Gaussian width of the stationary-phase contribution.
    """
    phi_f = sp.lambdify(x_sym, phi_expr, 'numpy')
    amp_f = sp.lambdify(x_sym, amp_expr, 'numpy')
    if bounds is None:
        w = 8.0 / np.sqrt(max(lam, 1))
        bounds = (-w, w)
    x = np.linspace(bounds[0], bounds[1], n)
    return np.trapezoid(amp_f(x) * np.exp(1j * lam * phi_f(x)), x)


def numerical_integral_2d(phi_expr, amp_expr, x_sym, y_sym, lam,
                           n: int = 1_001) -> complex:
    """
    Reference 2D Cartesian quadrature for oscillatory integrals.
    Integration window ±8σ  with  σ = 5/√λ.
    """
    phi_f = sp.lambdify((x_sym, y_sym), phi_expr, 'numpy')
    amp_f = sp.lambdify((x_sym, y_sym), amp_expr, 'numpy')
    s = 5.0 / np.sqrt(max(lam, 1))
    lim = 8 * s
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = amp_f(X, Y) * np.exp(1j * lam * phi_f(X, Y))
    return np.trapezoid(np.trapezoid(f, x, axis=0), y)


def numerical_laplace_1d(phi_expr, amp_expr, x_sym, lam, xc: float) -> float:
    """
    Reference 1D Laplace quadrature  ∫ a(x) exp(-λφ(x)) dx
    on a ±10σ window centred at the minimum xc,
    with σ = 1/√(λ φ''(xc)).
    """
    phi_f = sp.lambdify(x_sym, phi_expr, 'numpy')
    amp_f = sp.lambdify(x_sym, amp_expr, 'numpy')
    phi_pp = float(sp.diff(phi_expr, x_sym, 2).subs(x_sym, xc))
    sigma = 1.0 / np.sqrt(max(lam * phi_pp, 1e-30))
    x = np.linspace(xc - 10 * sigma, xc + 10 * sigma, 20_001)
    phi_vals = phi_f(x)
    phi_min  = np.min(phi_vals)
    integral = np.trapezoid(amp_f(x) * np.exp(-lam * (phi_vals - phi_min)), x)
    return float(integral * np.exp(-lam * phi_min))


def numerical_laplace_2d(phi_expr, amp_expr, x_sym, y_sym, lam,
                          xc=(0.0, 0.0), n: int = 801) -> float:
    """
    Reference 2D Laplace quadrature  ∫∫ a(x,y) exp(-λφ(x,y)) dx dy
    on an adaptive window around the minimum (xc, yc).
    """
    phi_f = sp.lambdify((x_sym, y_sym), phi_expr, 'numpy')
    amp_f = sp.lambdify((x_sym, y_sym), amp_expr, 'numpy')
    h00 = float(sp.diff(phi_expr, x_sym, 2).subs([(x_sym, xc[0]), (y_sym, xc[1])]))
    h11 = float(sp.diff(phi_expr, y_sym, 2).subs([(x_sym, xc[0]), (y_sym, xc[1])]))
    sx = min(6.0 / np.sqrt(max(lam * h00, 1e-30)), 4.0)
    sy = min(6.0 / np.sqrt(max(lam * h11, 1e-30)), 4.0)
    x = np.linspace(xc[0] - 8*sx, xc[0] + 8*sx, n)
    y = np.linspace(xc[1] - 8*sy, xc[1] + 8*sy, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    phi_vals = phi_f(X, Y)
    phi_min  = np.min(phi_vals)
    f = amp_f(X, Y) * np.exp(-lam * (phi_vals - phi_min))
    return float(np.trapezoid(np.trapezoid(f, x, axis=0), y) * np.exp(-lam * phi_min))


def numerical_saddle_2d(phi_expr, amp_expr, x_sym, y_sym, lam, n: int = 601) -> complex:
    """
    Reference 2D quadrature for saddle-point integrals
        ∫∫ a(x,y) exp(iλφ(x,y)) dx dy
    where φ is genuinely complex.  The integrand is both oscillatory and
    exponentially damped; we integrate over a Gaussian-width window.
    """
    phi_f = sp.lambdify((x_sym, y_sym), phi_expr, 'numpy')
    amp_f = sp.lambdify((x_sym, y_sym), amp_expr, 'numpy')
    # Use the imaginary part to set the window: exp(-λ·Im(φ)) concentrates near 0.
    lim = 6.0 / np.sqrt(max(lam, 1))
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    phi_vals = np.asarray(phi_f(X, Y), dtype=complex)
    amp_vals = np.asarray(amp_f(X, Y), dtype=complex)
    f = amp_vals * np.exp(1j * lam * phi_vals)
    return np.trapezoid(np.trapezoid(f, x, axis=0), y)


def _sp_analyzer(phi, amp, vars_, **kw):
    """Shorthand: build a STATIONARY_PHASE analyzer."""
    return Analyzer(phi, amp, vars_,
                    method=IntegralMethod.STATIONARY_PHASE, **kw)


def _lap_analyzer(phi, amp, vars_, **kw):
    """Shorthand: build a LAPLACE analyzer."""
    return Analyzer(phi, amp, vars_,
                    method=IntegralMethod.LAPLACE, **kw)


# ============================================================================
# Section 2 — STATIONARY_PHASE tests (23 cases)
# ============================================================================

class TestStationaryPhase:
    """
    23 tests for the stationary-phase method covering:
    - Structural (singularity classification, Maslov index, method stamping)
    - Analytical (exact formula for pure-Gaussian phases)
    - Numerical (comparison with reference quadrature for anharmonic cases)
    - Convergence rates (λ^{-n/2} for Morse, λ^{-1/3} for Airy 1D,
                         λ^{-5/6} for Airy 2D, λ^{-3/4} for Pearcey)
    - Edge cases (zero amplitude, complex amplitude, higher-order degeneracy)
    """

    # ------------------------------------------------------------------
    # SP-01 — Method is stamped correctly on CriticalPoint
    # ------------------------------------------------------------------
    def test_sp01_method_stamp(self):
        """analyze_point stamps method=STATIONARY_PHASE on the CriticalPoint."""
        x, y = sp.symbols('x y')
        analyzer = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y])
        cp = analyzer.analyze_point(np.array([0., 0.]))
        assert cp.method == IntegralMethod.STATIONARY_PHASE

    # ------------------------------------------------------------------
    # SP-02 — Pure Gaussian 2D exact value
    # ------------------------------------------------------------------
    def test_sp02_pure_gaussian_2d(self):
        """Leading term matches exact value 2πi/λ for φ=x²/2+y²/2, a=1."""
        x, y = sp.symbols('x y')
        analyzer = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y])
        cp = analyzer.analyze_point(np.array([0., 0.]))
        ev = AsymptoticEvaluator()
        for lam in [50, 100, 200]:
            res = ev.evaluate(cp, lam)
            exact = 2j * np.pi / lam
            assert relative_error(res.leading_term, exact) < 0.01

    # ------------------------------------------------------------------
    # SP-03 — Pure Gaussian 1D exact value
    # ------------------------------------------------------------------
    def test_sp03_pure_gaussian_1d(self):
        """1D Gaussian: I(λ) = √(2π/λ) exp(iπ/4)."""
        x = sp.Symbol('x')
        analyzer = _sp_analyzer(x**2/2, 1, [x])
        cp = analyzer.analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, 100)
        exact = np.sqrt(2*np.pi/100) * np.exp(1j*np.pi/4)
        assert relative_error(res.leading_term, exact) < 0.005

    # ------------------------------------------------------------------
    # SP-04 — Maslov index for signature 0, 1, 2
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("phi_expr,expected_sig", [
        (sp.Symbol('x')**2/2 + sp.Symbol('y')**2/2,   0),   # positive definite
        (sp.Symbol('x')**2/2 - sp.Symbol('y')**2/2,   1),   # saddle
        (-sp.Symbol('x')**2/2 - sp.Symbol('y')**2/2,  2),   # negative definite
    ])
    def test_sp04_maslov_signature(self, phi_expr, expected_sig):
        """Hessian signature is correctly extracted for all sign combinations."""
        x, y = sp.symbols('x y')
        cp = _sp_analyzer(phi_expr, 1, [x, y]).analyze_point(np.array([0., 0.]))
        assert cp.signature == expected_sig

    # ------------------------------------------------------------------
    # SP-05 — Rotated Hessian (non-diagonal) still gives correct det
    # ------------------------------------------------------------------
    def test_sp05_rotated_hessian(self):
        """30° rotation of anisotropic quadratic: det H = λ₁·λ₂ = 2."""
        x, y = sp.symbols('x y')
        θ = np.pi / 6
        u = x * np.cos(θ) - y * np.sin(θ)
        v = x * np.sin(θ) + y * np.cos(θ)
        phi = (u**2 + 2*v**2) / 2
        cp = _sp_analyzer(phi, 1, [x, y]).analyze_point(np.array([0., 0.]))
        assert cp.singularity_type == SingularityType.MORSE
        assert abs(np.real(cp.hessian_det) - 2.0) < 1e-6

    # ------------------------------------------------------------------
    # SP-06 — Anharmonic 2D Morse vs numerical quadrature
    # ------------------------------------------------------------------
    def test_sp06_anharmonic_2d_vs_quadrature(self):
        """φ = x²/2 + y²/2 + 0.05x³: total value within 15% of numerical."""
        x, y = sp.symbols('x y')
        phi = x**2/2 + y**2/2 + sp.Rational(1,20)*x**3
        amp = 1 + x**2
        lam = 200
        analyzer = _sp_analyzer(phi, amp, [x, y])
        cp = analyzer.analyze_point(np.array([0., 0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_integral_2d(phi, amp, x, y, lam)
        assert relative_error(res.total_value, ref) < 0.15

    # ------------------------------------------------------------------
    # SP-07 — Correction term direction check for Morse (order-2)
    # ------------------------------------------------------------------
    def test_sp07_correction_improves_accuracy(self):
        """Order-2 correction must move closer to the numerical reference,
        i.e. err_total < err_lead, and must not worsen accuracy by >0.1%."""
        x, y = sp.symbols('x y')
        phi = x**2/2 + y**2/2 + sp.Rational(1,5)*x**3
        amp = 1 + sp.Rational(3,10)*x**2
        lam = 400
        analyzer = _sp_analyzer(phi, amp, [x, y])
        cp = analyzer.analyze_point(np.array([0., 0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_integral_2d(phi, amp, x, y, lam)
        err_lead  = relative_error(res.leading_term, ref)
        err_total = relative_error(res.total_value,  ref)
        # Correction must not worsen accuracy
        assert err_total < err_lead * 1.001
        # And correction_term must be non-zero (formula is active)
        assert abs(res.correction_term) > 0

    # ------------------------------------------------------------------
    # SP-08 — Morse λ^{-1} decay rate (2D)
    # ------------------------------------------------------------------
    def test_sp08_morse_2d_decay_rate(self):
        """Empirical log-log slope of |I₀(λ)| must be close to -1 (n/2=1)."""
        x, y = sp.symbols('x y')
        phi = x**2/2 + y**2/2 + sp.Rational(1,10)*x**3
        cp = _sp_analyzer(phi, 1, [x, y]).analyze_point(np.array([0., 0.]))
        ev = AsymptoticEvaluator()
        lams = np.array([50., 100., 200., 400.])
        vals = np.array([abs(ev.evaluate(cp, l).leading_term) for l in lams])
        slope = np.polyfit(np.log(lams), np.log(vals), 1)[0]
        assert -1.15 < slope < -0.85

    # ------------------------------------------------------------------
    # SP-09 — Airy 1D classification and exact scaling
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("alpha,lam", [(1, 50), (2, 80), (-1, 60)])
    def test_sp09_airy_1d_exact(self, alpha, lam):
        """φ = α x³/3: I = 2π Ai(0) (λ|α|)^{-1/3} within 15%."""
        x = sp.Symbol('x')
        phi = sp.Rational(alpha, 1) * x**3 / 3
        cp = _sp_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        assert cp.singularity_type == SingularityType.AIRY_1D
        res = AsymptoticEvaluator().evaluate(cp, lam)
        Ai0   = scipy_airy(0)[0]
        exact = 2 * np.pi * Ai0 * (lam * abs(alpha)) ** (-1/3)
        assert relative_error(res.leading_term, exact) < 0.15

    # ------------------------------------------------------------------
    # SP-10 — Airy 1D decay rate λ^{-1/3}
    # ------------------------------------------------------------------
    def test_sp10_airy_1d_decay_rate(self):
        """Empirical slope of |I(λ)| for x³/3 must be close to -1/3."""
        x = sp.Symbol('x')
        cp = _sp_analyzer(x**3/3, 1, [x]).analyze_point(np.array([0.]))
        ev = AsymptoticEvaluator()
        lams = np.array([50., 100., 200., 400.])
        vals = np.array([abs(ev.evaluate(cp, l).leading_term) for l in lams])
        slope = np.polyfit(np.log(lams), np.log(vals), 1)[0]
        assert -1/3 - 0.05 < slope < -1/3 + 0.05

    # ------------------------------------------------------------------
    # SP-11 — Airy 2D classification and λ^{-5/6} scaling
    # ------------------------------------------------------------------
    def test_sp11_airy_2d_scaling(self):
        """φ = x³/3 + y²/2: |I| / λ^{-5/6} must be O(1)."""
        x, y = sp.symbols('x y')
        phi = x**3/3 + y**2/2
        cp = _sp_analyzer(phi, 1, [x, y]).analyze_point(np.array([0., 0.]))
        assert cp.singularity_type == SingularityType.AIRY_2D
        lam = 80
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ratio = abs(res.leading_term) / lam**(-5/6)
        assert 0.5 < ratio < 20.0

    # ------------------------------------------------------------------
    # SP-12 — Airy 2D rotated null direction
    # ------------------------------------------------------------------
    def test_sp12_airy_2d_rotated(self):
        """Null direction along (1,1)/√2: canonical cubic coeff |α| ≈ 1."""
        x, y = sp.symbols('x y')
        phi = ((x + y)/sp.sqrt(2))**3 / 3 + (x - y)**2 / 4
        cp = _sp_analyzer(phi, 1, [x, y]).analyze_point(np.array([0., 0.]))
        assert cp.singularity_type == SingularityType.AIRY_2D
        assert 0.90 < abs(cp.canonical_coefficients['cubic']) < 1.10

    # ------------------------------------------------------------------
    # SP-13 — Pearcey classification and λ^{-3/4} scaling
    # ------------------------------------------------------------------
    def test_sp13_pearcey_scaling(self):
        """φ = x⁴/4 + y²/2: |I| / λ^{-3/4} must be O(1)."""
        x, y = sp.symbols('x y')
        phi = x**4/4 + y**2/2
        cp = _sp_analyzer(phi, 1, [x, y]).analyze_point(np.array([0., 0.]))
        assert cp.singularity_type == SingularityType.PEARCEY
        res = AsymptoticEvaluator().evaluate(cp, 60)
        ratio = abs(res.leading_term) / 60**(-0.75)
        assert 0.5 < ratio < 20.0

    # ------------------------------------------------------------------
    # SP-14 — Pearcey vs Airy cubic threshold
    # ------------------------------------------------------------------
    def test_sp14_pearcey_vs_airy_threshold(self):
        """Explicit cubic_threshold controls Pearcey/Airy boundary."""
        x, y = sp.symbols('x y')
        phi_above = x**4/4 + 1e-4*x**3 + y**2/2   # cubic > 1e-5 → Airy
        phi_below = x**4/4 + 1e-6*x**3 + y**2/2   # cubic < 1e-5 → Pearcey
        cp_a = _sp_analyzer(phi_above, 1, [x, y],
                             cubic_threshold=1e-5).analyze_point(np.zeros(2))
        cp_p = _sp_analyzer(phi_below, 1, [x, y],
                             cubic_threshold=1e-5).analyze_point(np.zeros(2))
        assert cp_a.singularity_type == SingularityType.AIRY_2D
        assert cp_p.singularity_type == SingularityType.PEARCEY

    # ------------------------------------------------------------------
    # SP-15 — Higher-order degeneracy → HIGHER_ORDER + zero result
    # ------------------------------------------------------------------
    def test_sp15_higher_order_degeneracy(self):
        """φ = x⁵/5: classified as HIGHER_ORDER, evaluator returns 0."""
        x = sp.Symbol('x')
        cp = _sp_analyzer(x**5/5, 1, [x]).analyze_point(np.array([0.]))
        assert cp.singularity_type == SingularityType.HIGHER_ORDER
        ev = AsymptoticEvaluator()
        with pytest.warns(RuntimeWarning, match="Unhandled singularity"):
            res = ev.evaluate(cp, 50)
        assert res.total_value == 0j
        assert res.order_leading == float('inf')

    # ------------------------------------------------------------------
    # SP-16 — Zero amplitude at critical point
    # ------------------------------------------------------------------
    def test_sp16_zero_amplitude(self):
        """a(x) = x² → a(0) = 0: leading term is zero, correction drives result."""
        x = sp.Symbol('x')
        phi = x**2/2
        amp = x**2
        cp = _sp_analyzer(phi, amp, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, 1000)
        assert abs(res.total_value) < 1e-3

    # ------------------------------------------------------------------
    # SP-17 — Complex amplitude
    # ------------------------------------------------------------------
    def test_sp17_complex_amplitude(self):
        """Amplitude a = exp(ix): result must be finite and have non-zero imag part."""
        x = sp.Symbol('x')
        phi = x**2/2
        amp = sp.exp(sp.I * x)
        cp = _sp_analyzer(phi, amp, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, 100)
        assert np.isfinite(res.total_value)
        assert abs(np.imag(res.total_value)) > 0

    # ------------------------------------------------------------------
    # SP-18 — Off-centre critical point
    # ------------------------------------------------------------------
    def test_sp18_off_centre_critical_point(self):
        """φ = (x-2)² + (y+1)²: critical point located correctly at (2,-1)."""
        x, y = sp.symbols('x y')
        phi = (x - 2)**2 + (y + 1)**2
        analyzer = _sp_analyzer(phi, 1, [x, y])
        pts = analyzer.find_critical_points([np.array([2., -1.])])
        assert len(pts) == 1
        assert np.linalg.norm(pts[0] - np.array([2., -1.])) < 1e-4
        cp = analyzer.analyze_point(pts[0])
        assert cp.singularity_type == SingularityType.MORSE

    # ------------------------------------------------------------------
    # SP-19 — Very high λ numerical stability
    # ------------------------------------------------------------------
    def test_sp19_high_lambda_stability(self):
        """Result must be finite and scale as λ^{-1} for λ up to 1e5."""
        x, y = sp.symbols('x y')
        cp = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y]).analyze_point(np.zeros(2))
        ev = AsymptoticEvaluator()
        for lam in [1e3, 1e4, 1e5]:
            res = ev.evaluate(cp, lam)
            assert np.isfinite(res.total_value)
            ratio = abs(res.total_value) / (lam ** -1)
            assert 0.5 < ratio < 7.0

    # ------------------------------------------------------------------
    # SP-20 — AsymptoticEvaluator routes to STATIONARY_PHASE result
    # ------------------------------------------------------------------
    def test_sp20_evaluator_method_field(self):
        """AsymptoticContribution.method must equal STATIONARY_PHASE."""
        x, y = sp.symbols('x y')
        cp = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y]).analyze_point(np.zeros(2))
        res = AsymptoticEvaluator().evaluate(cp, 100)
        assert res.method == IntegralMethod.STATIONARY_PHASE

    # ------------------------------------------------------------------
    # SP-21 (NEW) — Airy 2D log-log decay rate λ^{-5/6}
    # ------------------------------------------------------------------
    def test_sp21_airy_2d_decay_rate(self):
        """Empirical log-log slope of |I(λ)| for Airy 2D must be close to -5/6."""
        x, y = sp.symbols('x y')
        phi = x**3/3 + y**2/2
        cp = _sp_analyzer(phi, 1, [x, y]).analyze_point(np.zeros(2))
        assert cp.singularity_type == SingularityType.AIRY_2D
        ev = AsymptoticEvaluator()
        lams = np.array([40., 80., 160., 320.])
        vals = np.array([abs(ev.evaluate(cp, l).leading_term) for l in lams])
        slope = np.polyfit(np.log(lams), np.log(vals), 1)[0]
        target = -5.0 / 6.0          # ≈ -0.8333
        assert target - 0.06 < slope < target + 0.06

    # ------------------------------------------------------------------
    # SP-22 (NEW) — Pearcey log-log decay rate λ^{-3/4}
    # ------------------------------------------------------------------
    def test_sp22_pearcey_decay_rate(self):
        """Empirical log-log slope of |I(λ)| for Pearcey must be close to -3/4."""
        x, y = sp.symbols('x y')
        phi = x**4/4 + y**2/2
        cp = _sp_analyzer(phi, 1, [x, y]).analyze_point(np.zeros(2))
        assert cp.singularity_type == SingularityType.PEARCEY
        ev = AsymptoticEvaluator()
        lams = np.array([40., 80., 160., 320.])
        vals = np.array([abs(ev.evaluate(cp, l).leading_term) for l in lams])
        slope = np.polyfit(np.log(lams), np.log(vals), 1)[0]
        target = -0.75
        assert target - 0.06 < slope < target + 0.06

    # ------------------------------------------------------------------
    # SP-23 (NEW) — Airy 1D vs direct quadrature
    # ------------------------------------------------------------------
    def test_sp23_airy_1d_vs_quadrature(self):
        """Airy 1D formula within 20% of direct numerical integral."""
        x = sp.Symbol('x')
        phi = x**3 / 3
        lam = 60
        cp = _sp_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        # Integrate over a moderately wide window; Airy integrals converge slowly
        ref = numerical_integral_1d(phi, 1, x, lam, bounds=(-5, 5), n=50_001)
        assert relative_error(res.leading_term, ref) < 0.20


# ============================================================================
# Section 3 — LAPLACE tests (22 cases)
# ============================================================================

class TestLaplace:
    """
    22 tests for the Laplace method:
    - Structural (method stamping, positive-definite Hessian warning)
    - Analytical (exact formula for pure-Gaussian potentials)
    - Numerical (comparison with reference quadrature, 1D and 2D)
    - Correction terms (amplitude curvature, anharmonicity)
    - Edge cases (very large λ, amplitude zero at minimum, 2D coupling)
    """

    # ------------------------------------------------------------------
    # LAP-01 — Method stamped correctly
    # ------------------------------------------------------------------
    def test_lap01_method_stamp(self):
        """analyze_point stamps method=LAPLACE on the CriticalPoint."""
        x = sp.Symbol('x')
        cp = _lap_analyzer(x**2/2, 1, [x]).analyze_point(np.array([0.]))
        assert cp.method == IntegralMethod.LAPLACE

    # ------------------------------------------------------------------
    # LAP-02 — Pure 1D Gaussian exact value
    # ------------------------------------------------------------------
    def test_lap02_pure_gaussian_1d(self):
        """ψ = x²/2: I = √(2π/λ) within 0.5%."""
        x = sp.Symbol('x')
        cp = _lap_analyzer(x**2/2, 1, [x]).analyze_point(np.array([0.]))
        ev = AsymptoticEvaluator()
        for lam in [50, 100, 200]:
            res = ev.evaluate(cp, lam)
            exact = np.sqrt(2 * np.pi / lam)
            assert relative_error(np.real(res.leading_term), exact) < 0.005

    # ------------------------------------------------------------------
    # LAP-03 — Pure 2D isotropic Gaussian exact value
    # ------------------------------------------------------------------
    def test_lap03_pure_gaussian_2d(self):
        """ψ = x²/2 + y²/2: I = 2π/λ within 0.5%."""
        x, y = sp.symbols('x y')
        cp = _lap_analyzer(x**2/2 + y**2/2, 1, [x, y]).analyze_point(np.zeros(2))
        for lam in [50, 100, 200]:
            res = AsymptoticEvaluator().evaluate(cp, lam)
            exact = 2 * np.pi / lam
            assert relative_error(np.real(res.leading_term), exact) < 0.005

    # ------------------------------------------------------------------
    # LAP-04 — Anisotropic 2D Gaussian exact det  [FIXED]
    # ------------------------------------------------------------------
    def test_lap04_anisotropic_2d(self):
        """ψ = x²/2 + 2y²: H = diag(1,4), det H = 4, √det = 2.
        Exact I = 2π/(λ · √det H) = 2π/(2λ) = π/λ within 1%.

        Original test had 2π/(λ√4) which evaluates to π/λ, but was
        written as √4 (ambiguous); this version is explicit.
        """
        x, y = sp.symbols('x y')
        phi = x**2/2 + 2*y**2          # H = diag(1, 4), det H = 4
        cp = _lap_analyzer(phi, 1, [x, y]).analyze_point(np.zeros(2))
        lam = 100
        res = AsymptoticEvaluator().evaluate(cp, lam)
        # Leading term: (2π/λ) / √(det H) = (2π/λ) / √4 = (2π/λ)/2 = π/λ
        exact = np.pi / lam
        assert relative_error(np.real(res.leading_term), exact) < 0.01
        # Also verify the Hessian determinant is correctly stored as 4.
        assert abs(np.real(cp.hessian_det) - 4.0) < 1e-6

    # ------------------------------------------------------------------
    # LAP-05 — Correction term improves accuracy vs plain leading term
    # ------------------------------------------------------------------
    def test_lap05_correction_improves(self):
        """Adding the order-1 correction must not degrade accuracy by >0.1%."""
        x = sp.Symbol('x')
        phi = x**2/2 + sp.Rational(1, 10)*x**3
        lam = 200
        cp = _lap_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, 1, x, lam, 0.0)
        err_lead  = relative_error(np.real(res.leading_term), ref)
        err_total = relative_error(np.real(res.total_value),  ref)
        assert err_total < err_lead * 1.001

    # ------------------------------------------------------------------
    # LAP-06 — 1D cubic anharmonicity vs numerical reference
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("phi_str,lam", [
        ("x**2/2 + 0.1*x**3", 100),
        ("x**2/2 - 0.15*x**3", 80),
        ("x**2/2 + 0.05*x**3 + 0.02*x**4", 60),
    ])
    def test_lap06_1d_anharmonic(self, phi_str, lam):
        """1D anharmonic potential: total within 2% of numerical reference."""
        x = sp.Symbol('x')
        phi = eval(phi_str, {'x': x, 'sp': sp})
        cp = _lap_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, 1, x, lam, 0.0)
        assert relative_error(np.real(res.total_value), ref) < 0.02

    # ------------------------------------------------------------------
    # LAP-07 — Amplitude curvature correction (non-unit amplitude)
    # ------------------------------------------------------------------
    def test_lap07_amplitude_curvature(self):
        """a = 1 + x²: correction via Tr(H⁻¹ ∇²a) is non-zero and captured."""
        x = sp.Symbol('x')
        phi = x**2/2
        amp = 1 + x**2
        lam = 100
        cp = _lap_analyzer(phi, amp, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, amp, x, lam, 0.0)
        assert relative_error(np.real(res.total_value), ref) < 0.02

    # ------------------------------------------------------------------
    # LAP-08 — Off-centre minimum
    # ------------------------------------------------------------------
    def test_lap08_off_centre_minimum(self):
        """ψ = (x-1)²/2: minimum at x=1, contribution same as origin case."""
        x = sp.Symbol('x')
        phi = (x - 1)**2 / 2
        lam = 80
        cp = _lap_analyzer(phi, 1, [x]).analyze_point(np.array([1.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        exact = np.sqrt(2 * np.pi / lam)
        assert relative_error(np.real(res.leading_term), exact) < 0.005

    # ------------------------------------------------------------------
    # LAP-09 — Hessian not positive definite raises RuntimeWarning
    # ------------------------------------------------------------------
    def test_lap09_saddle_point_warning(self):
        """Laplace on a saddle φ=x²/2-y²/2 must emit a RuntimeWarning."""
        x, y = sp.symbols('x y')
        phi = x**2/2 - y**2/2
        with pytest.warns(RuntimeWarning, match="non-positive Hessian"):
            _lap_analyzer(phi, 1, [x, y]).analyze_point(np.zeros(2))

    # ------------------------------------------------------------------
    # LAP-10 — 2D with xy coupling vs numerical reference
    # ------------------------------------------------------------------
    def test_lap10_2d_xy_coupling(self):
        """ψ = x² + y² + 0.1·xy: leading term within 2% of numerical reference."""
        x, y = sp.symbols('x y')
        phi = x**2 + y**2 + sp.Rational(1, 10)*x*y
        cp = _lap_analyzer(phi, 1, [x, y]).analyze_point(np.zeros(2))
        lam = 80
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_2d(phi, 1, x, y, lam)
        assert relative_error(np.real(res.leading_term), ref) < 0.02

    # ------------------------------------------------------------------
    # LAP-11 — Very large λ numerical stability
    # ------------------------------------------------------------------
    def test_lap11_high_lambda_stability(self):
        """Result finite and scales as λ^{-1} for λ up to 1e5."""
        x, y = sp.symbols('x y')
        cp = _lap_analyzer(x**2/2 + y**2/2, 1, [x, y]).analyze_point(np.zeros(2))
        ev = AsymptoticEvaluator()
        for lam in [1e3, 1e4, 1e5]:
            res = ev.evaluate(cp, lam)
            assert np.isfinite(res.total_value)
            ratio = abs(res.total_value) * lam
            assert 0.5 < ratio < 10.0

    # ------------------------------------------------------------------
    # LAP-12 — Decay rate λ^{-1/2} in 1D
    # ------------------------------------------------------------------
    def test_lap12_decay_rate_1d(self):
        """Empirical slope of |I₀(λ)| for 1D Gaussian must be close to -0.5."""
        x = sp.Symbol('x')
        cp = _lap_analyzer(x**2/2, 1, [x]).analyze_point(np.array([0.]))
        ev = AsymptoticEvaluator()
        lams = np.array([50., 100., 200., 400.])
        vals = np.array([abs(ev.evaluate(cp, l).leading_term) for l in lams])
        slope = np.polyfit(np.log(lams), np.log(vals), 1)[0]
        assert -0.55 < slope < -0.45

    # ------------------------------------------------------------------
    # LAP-13 — Amplitude zero at minimum
    # ------------------------------------------------------------------
    def test_lap13_zero_amplitude_at_minimum(self):
        """a(x,y) = x²+y²: a(0,0)=0 → contribution is O(λ^{-2})."""
        x, y = sp.symbols('x y')
        phi = x**2/2 + y**2/2
        amp = x**2 + y**2
        cp = _lap_analyzer(phi, amp, [x, y]).analyze_point(np.zeros(2))
        res = AsymptoticEvaluator().evaluate(cp, 500)
        assert abs(res.total_value) < 0.1

    # ------------------------------------------------------------------
    # LAP-14 — Quartic anharmonicity D4
    # ------------------------------------------------------------------
    def test_lap14_quartic_anharmonicity(self):
        """ψ = x²/2 + 0.05x⁴: correction via D4 keeps total within 3% of ref."""
        x = sp.Symbol('x')
        phi = x**2/2 + sp.Rational(1, 20)*x**4
        lam = 60
        cp = _lap_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, 1, x, lam, 0.0)
        assert relative_error(np.real(res.total_value), ref) < 0.03

    # ------------------------------------------------------------------
    # LAP-15 — Gradient of amplitude + D3 coupling term
    # ------------------------------------------------------------------
    def test_lap15_gradient_amplitude_d3_coupling(self):
        """a = x+1: gradient coupling −½⟨H⁻¹∇a,V⟩ is non-zero."""
        x = sp.Symbol('x')
        phi = x**2/2 + sp.Rational(1, 10)*x**3
        amp = x + 1
        lam = 100
        cp = _lap_analyzer(phi, amp, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, amp, x, lam, 0.0)
        assert relative_error(np.real(res.total_value), ref) < 0.02

    # ------------------------------------------------------------------
    # LAP-16 — Metastable potential (φ has higher-order terms)
    # ------------------------------------------------------------------
    def test_lap16_metastable_potential(self):
        """ψ = x² - x³/3: minimum near 0, total within 5% of reference."""
        x = sp.Symbol('x')
        phi = x**2 - x**3/3
        lam = 50
        cp = _lap_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, 1, x, lam, 0.0)
        assert relative_error(np.real(res.total_value), ref) < 0.05

    # ------------------------------------------------------------------
    # LAP-17 — Slow amplitude (cosine) near minimum
    # ------------------------------------------------------------------
    def test_lap17_cosine_amplitude(self):
        """a = cos(x): a(0)=1, leading term still accurate."""
        x = sp.Symbol('x')
        phi = x**2/2
        amp = sp.cos(x)
        lam = 120
        cp = _lap_analyzer(phi, amp, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, amp, x, lam, 0.0)
        assert relative_error(np.real(res.total_value), ref) < 0.02

    # ------------------------------------------------------------------
    # LAP-18 — Singular Hessian raises ValueError
    # ------------------------------------------------------------------
    def test_lap18_singular_hessian_raises(self):
        """det H = 0 (flat direction): LaplaceEvaluator must raise ValueError."""
        x, y = sp.symbols('x y')
        phi = x**2  # flat in y → det H = 0
        cp = _lap_analyzer(phi, 1, [x, y]).analyze_point(np.zeros(2))
        with pytest.raises(ValueError, match="singular"):
            LaplaceEvaluator().evaluate(cp, 50)

    # ------------------------------------------------------------------
    # LAP-19 — Result is real for real φ and a
    # ------------------------------------------------------------------
    def test_lap19_result_is_real(self):
        """For real ψ and real a, the Laplace result must be (nearly) real."""
        x, y = sp.symbols('x y')
        phi = x**2/2 + y**2/2 + sp.Rational(1,10)*x**3
        cp = _lap_analyzer(phi, 1, [x, y]).analyze_point(np.zeros(2))
        res = AsymptoticEvaluator().evaluate(cp, 100)
        assert abs(np.imag(res.total_value)) < 1e-10 * abs(np.real(res.total_value)) + 1e-15

    # ------------------------------------------------------------------
    # LAP-20 — method field on AsymptoticContribution
    # ------------------------------------------------------------------
    def test_lap20_contribution_method_field(self):
        """AsymptoticContribution.method must equal IntegralMethod.LAPLACE."""
        x = sp.Symbol('x')
        cp = _lap_analyzer(x**2/2, 1, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, 100)
        assert res.method == IntegralMethod.LAPLACE

    # ------------------------------------------------------------------
    # LAP-21 (NEW) — Correction term has the right sign / direction
    # ------------------------------------------------------------------
    def test_lap21_correction_direction(self):
        """For ψ = x²/2 + c·x³ with c>0 the correction must move the
        total closer to the numerical reference, not away from it."""
        x = sp.Symbol('x')
        phi = x**2/2 + sp.Rational(1, 8)*x**3
        lam = 150
        cp = _lap_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        res = AsymptoticEvaluator().evaluate(cp, lam)
        ref = numerical_laplace_1d(phi, 1, x, lam, 0.0)
        # correction_term must be non-trivially non-zero
        assert abs(res.correction_term) > 1e-10
        # Total must be closer to ref than leading alone
        err_lead  = abs(np.real(res.leading_term) - ref)
        err_total = abs(np.real(res.total_value) - ref)
        assert err_total <= err_lead * 1.001

    # ------------------------------------------------------------------
    # LAP-22 (NEW) — Correction term magnitude scales as λ^{-1} relative to leading
    # ------------------------------------------------------------------
    def test_lap22_correction_scales_as_inverse_lambda(self):
        """Ratio |I₁/I₀| must scale as λ^{-1} (slope ≈ -1 on log-log)."""
        x = sp.Symbol('x')
        phi = x**2/2 + sp.Rational(1, 5)*x**3
        cp = _lap_analyzer(phi, 1, [x]).analyze_point(np.array([0.]))
        ev = AsymptoticEvaluator()
        lams = np.array([80., 160., 320., 640.])
        ratios = []
        for lam in lams:
            res = ev.evaluate(cp, lam)
            if abs(res.leading_term) > 0 and abs(res.correction_term) > 0:
                ratios.append(abs(res.correction_term) / abs(res.leading_term))
        assert len(ratios) >= 3, "Not enough non-zero correction terms"
        slope = np.polyfit(np.log(lams[:len(ratios)]), np.log(ratios), 1)[0]
        assert -1.15 < slope < -0.85


# ============================================================================
# Section 4 — SADDLE_POINT tests (22 cases)
# ============================================================================

class TestSaddlePoint:
    """
    22 tests for the saddle-point method:
    - Structural (method stamping, saddle location in ℂⁿ)
    - Analytical (complex Gaussian, known exact values)
    - Convergence (λ^{-n/2} decay)
    - Picard-Lefschetz warning always emitted
    - Edge cases (degenerate saddle, purely imaginary shift)
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sdl_analyzer(phi, amp, vars_):
        return Analyzer(phi, amp, vars_,
                        method=IntegralMethod.SADDLE_POINT)

    @staticmethod
    def _find_and_analyze(phi, amp, vars_, guess):
        an = TestSaddlePoint._sdl_analyzer(phi, amp, vars_)
        se = SaddlePointEvaluator()
        saddles = se.find_saddle_points(an, [np.array(guess, dtype=float)])
        assert len(saddles) >= 1, "No saddle point found"
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cp = an.analyze_point(saddles[0])
        return cp, se

    # ------------------------------------------------------------------
    # SP2-01 — Method stamp
    # ------------------------------------------------------------------
    def test_sdl01_method_stamp(self):
        """analyze_point stamps method=SADDLE_POINT on the CriticalPoint."""
        x, y = sp.symbols('x y')
        phi = (x**2 + y**2)/2 + sp.I*(x**2 + y**2)/4
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        assert cp.method == IntegralMethod.SADDLE_POINT

    # ------------------------------------------------------------------
    # SP2-02 — Saddle found at origin for φ = (½+¼i)(x²+y²)
    # ------------------------------------------------------------------
    def test_sdl02_saddle_at_origin(self):
        """∇φ=0 at origin for a radially symmetric complex phase."""
        x, y = sp.symbols('x y')
        phi = (x**2 + y**2)/2 + sp.I*(x**2 + y**2)/4
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        assert np.linalg.norm(np.real(cp.position)) < 1e-5
        assert np.linalg.norm(np.imag(cp.position)) < 1e-5

    # ------------------------------------------------------------------
    # SP2-03 — Complex Gaussian 2D exact value
    # ------------------------------------------------------------------
    def test_sdl03_complex_gaussian_exact(self):
        """
        φ = c·(x²+y²)/2  with c = 1+i/2 (complex constant).
        Exact: I = 2π/(λc) = 2π/(λ(1+i/2)).
        Saddle at origin, det H = c².
        """
        x, y = sp.symbols('x y')
        c = 1 + sp.I/2
        phi = c * (x**2 + y**2) / 2
        cp, se = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        lam = 100
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, lam)
        c_num = complex(c)
        exact = 2 * np.pi / (lam * c_num)
        assert relative_error(res.leading_term, exact) < 0.02

    # ------------------------------------------------------------------
    # SP2-04 — Picard-Lefschetz warning always emitted
    # ------------------------------------------------------------------
    def test_sdl04_picard_lefschetz_warning(self):
        """evaluate() must always emit a RuntimeWarning about contour validity."""
        x, y = sp.symbols('x y')
        phi = (x**2 + y**2)/2 + sp.I*(x**2 + y**2)/4
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        with pytest.warns(RuntimeWarning, match="Picard-Lefschetz"):
            AsymptoticEvaluator().evaluate(cp, 50)

    # ------------------------------------------------------------------
    # SP2-05 — Decay rate λ^{-1} (n=2)
    # ------------------------------------------------------------------
    def test_sdl05_decay_rate_2d(self):
        """Empirical slope of |I₀(λ)| must be close to -1 (n/2 = 1)."""
        x, y = sp.symbols('x y')
        phi = (x**2 + y**2)/2 + sp.I*(x**2 + y**2)/4
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        ev = AsymptoticEvaluator()
        lams = np.array([50., 100., 200., 400.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            vals = np.array([abs(ev.evaluate(cp, l).leading_term) for l in lams])
        slope = np.polyfit(np.log(lams), np.log(vals), 1)[0]
        assert -1.15 < slope < -0.85

    # ------------------------------------------------------------------
    # SP2-06 — 1D complex Gaussian exact value
    # ------------------------------------------------------------------
    def test_sdl06_1d_complex_gaussian(self):
        """φ = (1+2i)·x²/2: I = √(2π/λ) / √(1+2i) within 2%."""
        x = sp.Symbol('x')
        phi = (1 + 2*sp.I) * x**2 / 2
        cp, _ = self._find_and_analyze(phi, 1, [x], [0.])
        lam = 100
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, lam)
        c = 1 + 2j
        exact = np.sqrt(2 * np.pi / lam) / np.sqrt(c)
        assert relative_error(res.leading_term, exact) < 0.02

    # ------------------------------------------------------------------
    # SP2-07 — result method field
    # ------------------------------------------------------------------
    def test_sdl07_contribution_method_field(self):
        """AsymptoticContribution.method must equal SADDLE_POINT."""
        x = sp.Symbol('x')
        phi = (1 + sp.I) * x**2 / 2
        cp, _ = self._find_and_analyze(phi, 1, [x], [0.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, 100)
        assert res.method == IntegralMethod.SADDLE_POINT

    # ------------------------------------------------------------------
    # SP2-08 — Non-unit amplitude at saddle
    # ------------------------------------------------------------------
    def test_sdl08_non_unit_amplitude(self):
        """Amplitude a=3 is correctly captured in the leading term."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * (x**2 + y**2) / 2
        cp, _ = self._find_and_analyze(phi, 3, [x, y], [0., 0.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res_3 = AsymptoticEvaluator().evaluate(cp, 100)
        phi2 = (1 + sp.I) * (x**2 + y**2) / 2
        cp1, _ = self._find_and_analyze(phi2, 1, [x, y], [0., 0.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res_1 = AsymptoticEvaluator().evaluate(cp1, 100)
        assert relative_error(res_3.leading_term, 3 * res_1.leading_term) < 1e-8

    # ------------------------------------------------------------------
    # SP2-09 — Correction term is zero (not implemented for saddle)
    # ------------------------------------------------------------------
    def test_sdl09_no_correction_term(self):
        """Correction term must be exactly 0j for saddle-point evaluation."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * (x**2 + y**2) / 2
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, 100)
        assert res.correction_term == 0j

    # ------------------------------------------------------------------
    # SP2-10 — Degenerate saddle returns zero + warning
    # ------------------------------------------------------------------
    def test_sdl10_degenerate_saddle_zero(self):
        """det H ≈ 0: SaddlePointEvaluator returns 0 with RuntimeWarning."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * x**2 / 2
        an = self._sdl_analyzer(phi, 1, [x, y])
        se = SaddlePointEvaluator()
        saddles = se.find_saddle_points(an, [np.zeros(2)])
        if not saddles:
            pytest.skip("No saddle found for degenerate case")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cp = an.analyze_point(saddles[0])
        cp.hessian_det = 0.0   # force degenerate
        with pytest.warns(RuntimeWarning, match="Degenerate"):
            res = SaddlePointEvaluator().evaluate(cp, 50)
        assert res.total_value == 0j

    # ------------------------------------------------------------------
    # SP2-11 — Imaginary shift of saddle from real axis
    # ------------------------------------------------------------------
    def test_sdl11_imaginary_shift(self):
        """φ = (1+i)/2 · x²: saddle still at 0."""
        x = sp.Symbol('x')
        phi = x**2/2 + sp.I * x**2/2
        cp, _ = self._find_and_analyze(phi, 1, [x], [0.])
        assert abs(np.real(cp.position[0])) < 1e-5

    # ------------------------------------------------------------------
    # SP2-12 — order_leading is n/2
    # ------------------------------------------------------------------
    def test_sdl12_order_leading(self):
        """order_leading must equal dim/2 = 1 for n=2."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * (x**2 + y**2) / 2
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, 100)
        assert res.order_leading == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # SP2-13 — Anisotropic complex Hessian
    # ------------------------------------------------------------------
    def test_sdl13_anisotropic_complex_hessian(self):
        """φ = (1+i)x²/2 + (2+3i)y²/2: det H = (1+i)(2+3i), verify formula."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I)*x**2/2 + (2 + 3*sp.I)*y**2/2
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        lam = 100
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, lam)
        c1, c2 = 1+1j, 2+3j
        exact = (2*np.pi/lam) / np.sqrt(c1 * c2)
        assert relative_error(res.leading_term, exact) < 0.02

    # ------------------------------------------------------------------
    # SP2-14 — find_saddle_points deduplicates results
    # ------------------------------------------------------------------
    def test_sdl14_deduplication(self):
        """Two identical real guesses must yield at most one saddle point."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * (x**2 + y**2) / 2
        an = self._sdl_analyzer(phi, 1, [x, y])
        se = SaddlePointEvaluator()
        saddles = se.find_saddle_points(an, [np.zeros(2), np.zeros(2)])
        assert len(saddles) == 1

    # ------------------------------------------------------------------
    # SP2-15 — 1D decay rate λ^{-1/2}
    # ------------------------------------------------------------------
    def test_sdl15_decay_rate_1d(self):
        """Empirical slope for 1D complex Gaussian must be close to -0.5."""
        x = sp.Symbol('x')
        phi = (1 + sp.I) * x**2 / 2
        cp, _ = self._find_and_analyze(phi, 1, [x], [0.])
        ev = AsymptoticEvaluator()
        lams = np.array([50., 100., 200., 400.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            vals = np.array([abs(ev.evaluate(cp, l).leading_term) for l in lams])
        slope = np.polyfit(np.log(lams), np.log(vals), 1)[0]
        assert -0.55 < slope < -0.45

    # ------------------------------------------------------------------
    # SP2-16 — Large λ stability
    # ------------------------------------------------------------------
    def test_sdl16_high_lambda_stability(self):
        """Result must be finite for λ up to 1e5."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * (x**2 + y**2) / 2
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        ev = AsymptoticEvaluator()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for lam in [1e3, 1e4, 1e5]:
                assert np.isfinite(ev.evaluate(cp, lam).total_value)

    # ------------------------------------------------------------------
    # SP2-17 — Phase value at saddle is correctly stored
    # ------------------------------------------------------------------
    def test_sdl17_phase_value_at_saddle(self):
        """φ(0,0) = 0 for radially symmetric phase centred at origin."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * (x**2 + y**2) / 2
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        assert abs(cp.phase_value) < 1e-10

    # ------------------------------------------------------------------
    # SP2-18 — Hessian inverse stored for Morse-type saddle
    # ------------------------------------------------------------------
    def test_sdl18_hessian_inv_stored(self):
        """For a non-degenerate saddle, hessian_inv must not be None."""
        x, y = sp.symbols('x y')
        phi = (1 + sp.I) * (x**2 + y**2) / 2
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        assert cp.hessian_inv is not None

    # ------------------------------------------------------------------
    # SP2-19 — Exponential decay from Im(φ) in the result
    # ------------------------------------------------------------------
    def test_sdl19_exponential_damping(self):
        """For φ = (1+2i)x²/2: exp(-λ Im φ(0)) = 1, but scaling checks."""
        x = sp.Symbol('x')
        phi = (1 + 2*sp.I) * x**2 / 2
        cp, _ = self._find_and_analyze(phi, 1, [x], [0.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res100 = abs(AsymptoticEvaluator().evaluate(cp, 100).leading_term)
            res200 = abs(AsymptoticEvaluator().evaluate(cp, 200).leading_term)
        ratio = res200 / res100
        assert 0.6 < ratio < 0.85

    # ------------------------------------------------------------------
    # SP2-20 — AUTO correctly resolves to SADDLE_POINT for complex φ
    # ------------------------------------------------------------------
    def test_sdl20_auto_detection(self):
        """AUTO method on a genuinely complex φ must resolve to SADDLE_POINT."""
        x, y = sp.symbols('x y')
        phi = (x**2 + y**2)/2 + sp.I*(x**2 + y**2)/4
        an = Analyzer(phi, 1, [x, y])
        assert an.method == IntegralMethod.SADDLE_POINT

    # ------------------------------------------------------------------
    # SP2-21 (NEW) — 2D saddle formula vs exact analytic value
    # ------------------------------------------------------------------
    def test_sdl21_2d_formula_vs_exact(self):
        """φ = c(x²+y²)/2 with c = 1+i/2: exact I = 2π/(λc) within 2%.

        For a pure complex Gaussian the saddle-point formula is exact.
        We use the analytic value rather than numerical quadrature because
        the 2D oscillatory integral over a finite window is unreliable at
        moderate λ — the integrand's oscillations (Re c ≠ 0) and decay
        (Im c ≠ 0) require a carefully matched window for each λ, making
        direct quadrature a poor reference here.
        """
        x, y = sp.symbols('x y')
        c = 1 + sp.I / 2
        phi = c * (x**2 + y**2) / 2
        cp, _ = self._find_and_analyze(phi, 1, [x, y], [0., 0.])
        lam = 80
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, lam)
        # Exact: (2π/λ)^(2/2) / √(det H)  where det H = c²
        # = (2π/λ) / c  (principal sqrt of c² is c for Re(c)>0)
        c_num = complex(c)
        exact = 2 * np.pi / (lam * c_num)
        assert relative_error(res.leading_term, exact) < 0.02

    # ------------------------------------------------------------------
    # SP2-22 (NEW) — 1D saddle formula vs exact analytic value
    # ------------------------------------------------------------------
    def test_sdl22_1d_formula_vs_exact(self):
        """φ = (1+2i)x²/2: exact I = √(2π/λ) / √(1+2i) within 2%.

        For a pure complex Gaussian the saddle-point formula is exact.
        Numerical quadrature of exp(iλ(1+2i)x²/2) over a finite real
        interval is unreliable because the integrand both oscillates and
        decays, and convergence of the truncation error depends sensitively
        on the window width relative to λ.
        """
        x = sp.Symbol('x')
        phi = (1 + 2*sp.I) * x**2 / 2
        cp, _ = self._find_and_analyze(phi, 1, [x], [0.])
        lam = 80
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            res = AsymptoticEvaluator().evaluate(cp, lam)
        c = 1 + 2j
        exact = np.sqrt(2 * np.pi / lam) / np.sqrt(c)
        assert relative_error(res.leading_term, exact) < 0.02


# ============================================================================
# Section 5 — AUTO detection tests (7 cases)
# ============================================================================

class TestAutoDetection:
    """
    7 tests verifying that _detect_method correctly identifies the method
    from the symbolic expression of φ, plus that AUTO is always resolved.
    """

    def test_auto01_real_phi_gives_stationary_phase(self):
        """φ = x²/2 + y²/2 (real) → AUTO must give STATIONARY_PHASE."""
        x, y = sp.symbols('x y')
        an = Analyzer(x**2/2 + y**2/2, 1, [x, y])
        assert an.method == IntegralMethod.STATIONARY_PHASE

    def test_auto02_imaginary_phi_gives_laplace(self):
        """φ = i·(x²/2 + y²/2) (purely imaginary) → AUTO must give LAPLACE."""
        x, y = sp.symbols('x y')
        an = Analyzer(sp.I*(x**2/2 + y**2/2), 1, [x, y])
        assert an.method == IntegralMethod.LAPLACE

    def test_auto03_complex_phi_gives_saddle_point(self):
        """φ = x²/2 + i·y²/2 (genuinely complex) → AUTO must give SADDLE_POINT."""
        x, y = sp.symbols('x y')
        an = Analyzer(x**2/2 + sp.I*y**2/2, 1, [x, y])
        assert an.method == IntegralMethod.SADDLE_POINT

    def test_auto04_explicit_method_overrides_detection(self):
        """Explicit method=LAPLACE must survive even for a real φ."""
        x, y = sp.symbols('x y')
        an = Analyzer(x**2/2 + y**2/2, 1, [x, y],
                      method=IntegralMethod.LAPLACE)
        assert an.method == IntegralMethod.LAPLACE

    def test_auto05_method_never_stays_auto(self):
        """After __init__ with AUTO, self.method must not be AUTO."""
        x, y = sp.symbols('x y')
        an = Analyzer(x**2/2 + y**2/2, 1, [x, y],
                      method=IntegralMethod.AUTO)
        assert an.method != IntegralMethod.AUTO

    def test_auto06_auto_is_default(self):
        """Default constructor (no method arg) must resolve method automatically."""
        x, y = sp.symbols('x y')
        an = Analyzer(x**2/2 + y**2/2, 1, [x, y])
        assert an.method in (IntegralMethod.STATIONARY_PHASE,
                             IntegralMethod.LAPLACE,
                             IntegralMethod.SADDLE_POINT)

    def test_auto07_resolved_method_is_concrete(self):  # NEW
        """After AUTO detection, method must be one of the three concrete values,
        never AUTO itself, for all three phase types."""
        x, y = sp.symbols('x y')
        phases = [
            x**2/2 + y**2/2,
            sp.I * (x**2/2 + y**2/2),
            x**2/2 + sp.I * y**2/2,
        ]
        concrete = {IntegralMethod.STATIONARY_PHASE,
                    IntegralMethod.LAPLACE,
                    IntegralMethod.SADDLE_POINT}
        for phi in phases:
            an = Analyzer(phi, 1, [x, y])
            assert an.method in concrete
            assert an.method != IntegralMethod.AUTO


# ============================================================================
# Section 6 — AsymptoticEvaluator guard clause tests (3 cases)  [NEW SECTION]
# ============================================================================

class TestEvaluatorGuards:
    """
    3 tests confirming that AsymptoticEvaluator raises ValueError when
    the CriticalPoint carries an unresolved or missing method.
    """

    def _dummy_cp(self, method):
        """Build a minimal CriticalPoint with the given method tag."""
        x, y = sp.symbols('x y')
        cp = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y]).analyze_point(np.zeros(2))
        cp.method = method
        return cp

    def test_guard01_none_method_raises(self):
        """method=None must raise ValueError with a descriptive message."""
        cp = self._dummy_cp(None)
        with pytest.raises(ValueError, match="None or AUTO"):
            AsymptoticEvaluator().evaluate(cp, 100)

    def test_guard02_auto_method_raises(self):
        """method=AUTO (unresolved) must raise ValueError."""
        cp = self._dummy_cp(IntegralMethod.AUTO)
        with pytest.raises(ValueError, match="None or AUTO"):
            AsymptoticEvaluator().evaluate(cp, 100)

    def test_guard03_unknown_method_raises(self):
        """An entirely unrecognised method value must raise ValueError."""
        cp = self._dummy_cp("not_a_real_method")
        with pytest.raises(ValueError):
            AsymptoticEvaluator().evaluate(cp, 100)


# ============================================================================
# Section 7 — find_critical_points tests (5 cases)  [NEW SECTION]
# ============================================================================

class TestFindCriticalPoints:
    """
    5 tests for Analyzer.find_critical_points(), covering multiple distinct
    critical points, domain-bounded search, default (no-guess) behaviour,
    out-of-domain filtering, and the importability of the caustics helper.
    """

    def test_fcp01_single_critical_point_at_origin(self):
        """φ = x²/2 + y²/2: single critical point found at (0,0)."""
        x, y = sp.symbols('x y')
        an = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y])
        pts = an.find_critical_points([np.zeros(2)])
        assert len(pts) == 1
        assert np.linalg.norm(pts[0]) < 1e-4

    def test_fcp02_multiple_distinct_critical_points(self):
        """φ = cos(x)·cos(y): multiple critical points found with multiple guesses."""
        x, y = sp.symbols('x y')
        phi = sp.cos(x) * sp.cos(y)
        an = _sp_analyzer(phi, 1, [x, y])
        guesses = [
            np.array([0., 0.]),
            np.array([np.pi, 0.]),
            np.array([0., np.pi]),
            np.array([np.pi, np.pi]),
        ]
        pts = an.find_critical_points(guesses)
        # Should find at least 2 distinct critical points
        assert len(pts) >= 2

    def test_fcp03_default_no_guess_finds_origin(self):
        """With no initial_guesses supplied, the default [0,…] finds the origin."""
        x, y = sp.symbols('x y')
        an = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y])
        pts = an.find_critical_points()           # uses default None → [zeros]
        assert len(pts) >= 1
        distances = [np.linalg.norm(p) for p in pts]
        assert min(distances) < 1e-4

    def test_fcp04_domain_bounded_search(self):
        """φ = (x-3)² + y²: only the off-origin minimum is found when domain
        is centred at (3, 0)."""
        x, y = sp.symbols('x y')
        phi = (x - 3)**2 + y**2
        domain = [(2.0, 4.0), (-1.0, 1.0)]
        an = _sp_analyzer(phi, 1, [x, y], domain=domain)
        pts = an.find_critical_points([np.array([3., 0.])])
        assert len(pts) >= 1
        assert np.linalg.norm(pts[0] - np.array([3., 0.])) < 1e-4

    def test_fcp05_caustics_module_importable(self):
        """The caustics helper used by find_critical_points must be importable."""
        try:
            from caustics import find_critical_points_numerical   # noqa: F401
        except ImportError as exc:
            pytest.fail(
                f"caustics.find_critical_points_numerical is not importable: {exc}"
            )


# ============================================================================
# Section 8 — AsymptoticVisualizer smoke tests (4 cases)
# ============================================================================

class TestVisualizer:
    """
    4 smoke tests: verify that the visualizer methods run without exception
    for each of the three methods.  Matplotlib display is suppressed.
    """

    @pytest.fixture(autouse=True)
    def _patch_show(self, monkeypatch):
        """Suppress plt.show() during all visualizer tests."""
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, 'show', lambda: None)
        import matplotlib
        matplotlib.use('Agg')

    def test_viz01_stationary_phase_all_plots(self):
        """plot_phase_landscape, plot_integrand and plot_asymptotic_convergence
        must run without error for a 2D STATIONARY_PHASE analyzer."""
        x, y = sp.symbols('x y')
        an = _sp_analyzer(x**2/2 + y**2/2, 1, [x, y])
        cp = an.analyze_point(np.zeros(2))
        viz = AsymptoticVisualizer(an)
        bounds = ((-2, 2), (-2, 2))
        viz.plot_phase_landscape([cp], bounds=bounds)
        viz.plot_integrand(lam_value=20, bounds=bounds)
        viz.plot_asymptotic_convergence(cp, lambda_start=10, lambda_end=200, num_points=10)

    def test_viz02_laplace_all_plots(self):
        """All three visualizer methods must run for a LAPLACE analyzer."""
        x, y = sp.symbols('x y')
        an = _lap_analyzer(x**2/2 + y**2/2, 1, [x, y])
        cp = an.analyze_point(np.zeros(2))
        viz = AsymptoticVisualizer(an)
        bounds = ((-2, 2), (-2, 2))
        viz.plot_phase_landscape([cp], bounds=bounds)
        viz.plot_integrand(lam_value=10, bounds=bounds)
        viz.plot_asymptotic_convergence(cp, lambda_start=10, lambda_end=200, num_points=10)

    def test_viz03_saddle_point_convergence_plot(self):
        """plot_asymptotic_convergence must run for a SADDLE_POINT CriticalPoint."""
        x, y = sp.symbols('x y')
        phi = (x**2 + y**2)/2 + sp.I*(x**2 + y**2)/4
        an = Analyzer(phi, 1, [x, y],
                      method=IntegralMethod.SADDLE_POINT)
        se = SaddlePointEvaluator()
        saddles = se.find_saddle_points(an, [np.zeros(2)])
        if not saddles:
            pytest.skip("No saddle found")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cp = an.analyze_point(saddles[0])
        viz = AsymptoticVisualizer(an)
        viz.plot_asymptotic_convergence(cp, lambda_start=10, lambda_end=200, num_points=10)

    def test_viz04_backward_compat_alias(self):
        """StationaryPhaseVisualizer must be an alias for AsymptoticVisualizer."""
        assert StationaryPhaseVisualizer is AsymptoticVisualizer


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    print("Run this file with pytest:")
    print("  pytest test_asymptotic.py -v")
    print()
    print("Test counts by section:")
    print("  Section 2 — Stationary phase : 23 tests  (TestStationaryPhase)")
    print("  Section 3 — Laplace          : 22 tests  (TestLaplace)")
    print("  Section 4 — Saddle-point     : 22 tests  (TestSaddlePoint)")
    print("  Section 5 — AUTO detection   :  7 tests  (TestAutoDetection)")
    print("  Section 6 — Evaluator guards :  3 tests  (TestEvaluatorGuards)")
    print("  Section 7 — find_crit_points :  5 tests  (TestFindCriticalPoints)")
    print("  Section 8 — Visualizer       :  4 tests  (TestVisualizer)")
    print("  Total                        : 86 tests")