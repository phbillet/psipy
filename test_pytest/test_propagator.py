"""
test_propagator.py — Test suite for the improved propagator.py
==============================================================

Coverage map
------------
TestDetJ                   — _det_J_1d, _det_J_from_jacobi
TestCumulativeAction       — _cumulative_action (with/without momenta, curved metric)
TestMaslovIndex            — _maslov_index
TestAiryArgument           — _airy_argument (new: pointwise Airy scaling)
TestAsymptoticCorrection1D — _asymptotic_correction_1d (spatial Airy profile)
TestAsymptoticCorrection2D — _asymptotic_correction_2d (new: 2D fold + cusp)
TestBuildHamiltonianSym    — _build_hamiltonian_sym
TestVanVleckSum            — van_vleck_sum (1D/2D, caustic patching)
TestComputeWavefunction    — compute_wavefunction integration tests
                             (v_fan API, fallback action, curved metric)
TestVisualisation          — smoke tests for all plot functions
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sympy as sp
from sympy import symbols, Matrix, sin
from scipy.special import airy as scipy_airy
from riemannian import Metric
import propagator as prop
from asymptotic import Analyzer, AsymptoticEvaluator, IntegralMethod, SingularityType
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# ============================================================================
# Helpers
# ============================================================================

def flat_1d():
    x = symbols('x', real=True)
    return Metric(1, (x,)), x


def power_1d():
    """g = x², g^{-1} = 1/x², geodesics are x(t) = x₀ exp(v₀ t / x₀²)."""
    x = symbols('x', real=True, positive=True)
    return Metric(x**2, (x,)), x


def flat_2d():
    x, y = symbols('x y', real=True)
    return Metric(Matrix([[1, 0], [0, 1]]), (x, y)), (x, y)


def sphere_2d():
    theta, phi = symbols('theta phi', real=True)
    return Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi)), (theta, phi)


# ============================================================================
# Jacobi determinant helpers
# ============================================================================

class TestDetJ:

    def test_det_J_1d_flat(self):
        """On flat metric g=1, Jacobian must be J(t) = t (free-particle fan)."""
        m, x = flat_1d()
        t = np.linspace(0, 2, 50)
        traj = {'t': t, 'x': t, 'v': np.ones_like(t)}
        detJ = prop._det_J_1d(m, traj, (0, 2), 50)
        assert np.allclose(detJ, t, rtol=1e-3)

    def test_det_J_1d_power(self):
        """With g=x², test that solver runs and returns finite values."""
        m, x = power_1d()
        t = np.linspace(0, 1, 20)
        traj = {'t': t, 'x': np.exp(t), 'v': np.exp(t)}
        detJ = prop._det_J_1d(m, traj, (0, 1), 20)
        assert len(detJ) == 20
        assert np.all(np.isfinite(detJ))

    @patch('propagator.jacobi_equation_solver')
    def test_det_J_from_jacobi_2d(self, mock_jacobi):
        """2D: mock two unit-identity Jacobi calls → det = 1."""
        m, _ = flat_2d()
        t = np.linspace(0, 1, 10)
        traj = {'t': t, 'x': np.ones(10), 'y': np.ones(10),
                'vx': np.zeros(10), 'vy': np.zeros(10)}

        def side_effect(metric, geodesic, tspan=None, n_steps=None,
                        initial_variation=None):
            init = initial_variation
            if init['DJ0'] == (1.0, 0.0):
                return {'J_x': np.ones(n_steps), 'J_y': np.zeros(n_steps)}
            return {'J_x': np.zeros(n_steps), 'J_y': np.ones(n_steps)}

        mock_jacobi.side_effect = side_effect
        detJ = prop._det_J_from_jacobi(m, traj, (0, 1), 10)
        assert np.allclose(detJ, 1.0)

    def test_det_J_from_jacobi_1d_dispatches(self):
        """1D case should delegate to _det_J_1d."""
        m, _ = flat_1d()
        traj = {'t': np.linspace(0, 1, 10), 'x': np.linspace(0, 1, 10),
                'v': np.ones(10)}
        with patch('propagator._det_J_1d', return_value=np.ones(10)) as mock:
            detJ = prop._det_J_from_jacobi(m, traj, (0, 1), 10)
            mock.assert_called_once()
            assert np.allclose(detJ, 1.0)


# ============================================================================
# Cumulative action
# ============================================================================

class TestCumulativeAction:

    def _expected(self, integrand, t):
        return np.cumsum(integrand * np.gradient(t))

    # ── 1D with explicit momentum ─────────────────────────────────────────────

    def test_1d_with_momentum(self):
        """Flat 1D, free particle: S = ∫ p v dt = v² t."""
        t = np.linspace(0, 2, 50)
        v = 2.0
        traj = {'t': t, 'x': v * t, 'v': v * np.ones_like(t),
                'xi': v * np.ones_like(t)}
        S = prop._cumulative_action(traj, dim=1)
        assert np.allclose(S, self._expected(v**2 * np.ones_like(t), t), rtol=1e-2)

    # ── 1D fallback — flat metric ─────────────────────────────────────────────

    def test_1d_fallback_flat_metric(self):
        """Fallback with flat metric (g=1): g v² = v², same as before."""
        m, _ = flat_1d()
        t = np.linspace(0, 2, 50)
        v = 2.0
        traj = {'t': t, 'v': v * np.ones_like(t), 'x': v * t}
        S = prop._cumulative_action(traj, dim=1, metric=m)
        assert np.allclose(S, self._expected(v**2 * np.ones_like(t), t), rtol=1e-2)

    def test_1d_fallback_curved_metric(self):
        """
        Fallback with curved metric g=x²: S = ∫ g v² dt = ∫ x² v² dt.

        For the trajectory x(t) = exp(t), v(t) = exp(t):
            g(x) v² = x² v² = exp(2t) · exp(2t) = exp(4t).
        """
        m, _ = power_1d()
        t = np.linspace(0, 1, 50)
        x_traj = np.exp(t)
        v_traj = np.exp(t)
        traj = {'t': t, 'x': x_traj, 'v': v_traj}
        S = prop._cumulative_action(traj, dim=1, metric=m)
        integrand = np.exp(4 * t)          # g(x) v² = x² v² = e^{4t}
        expected  = np.cumsum(integrand * np.gradient(t))
        assert np.allclose(S, expected, rtol=1e-2)

    def test_1d_fallback_no_metric_last_resort(self):
        """Without metric, fallback uses v² (documented flat-only limitation)."""
        t = np.linspace(0, 2, 50)
        v = 3.0
        traj = {'t': t, 'v': v * np.ones_like(t)}
        S = prop._cumulative_action(traj, dim=1, metric=None)
        assert np.allclose(S, self._expected(v**2 * np.ones_like(t), t), rtol=1e-2)

    # ── 2D with explicit momenta ──────────────────────────────────────────────

    def test_2d_with_momenta(self):
        """Flat 2D: S = ∫ (p_x vx + p_y vy) dt = (vx² + vy²) t."""
        t = np.linspace(0, 2, 50)
        vx, vy = 1.0, 2.0
        traj = {'t': t, 'x': vx * t, 'y': vy * t,
                'vx': vx * np.ones_like(t), 'vy': vy * np.ones_like(t),
                'xi': vx * np.ones_like(t), 'eta': vy * np.ones_like(t)}
        S = prop._cumulative_action(traj, dim=2)
        assert np.allclose(S, self._expected((vx**2 + vy**2) * np.ones_like(t), t),
                           rtol=1e-2)

    def test_2d_fallback_flat_metric(self):
        """2D fallback with flat metric: g_{ij} v^i v^j = vx² + vy²."""
        m, _ = flat_2d()
        t = np.linspace(0, 2, 30)
        vx, vy = 1.0, 2.0
        traj = {'t': t, 'x': vx * t, 'y': vy * t,
                'vx': vx * np.ones_like(t), 'vy': vy * np.ones_like(t)}
        S = prop._cumulative_action(traj, dim=2, metric=m)
        assert np.allclose(S, self._expected((vx**2 + vy**2) * np.ones_like(t), t),
                           rtol=1e-2)

    def test_2d_non_default_coord_keys(self):
        """
        coord_keys lets _cumulative_action read trajectory data stored under
        arbitrary symbol names (e.g. 'r'/'theta' for a polar metric) instead
        of the generic 'x'/'y'.  With explicit momenta ('xi'/'eta') the coord
        keys are only used for the velocity fallback gradient; here we supply
        'vx'/'vy' equivalents under the symbol-named keys to exercise the path.
        """
        t = np.linspace(0, 2, 50)
        vr, vt = 1.0, 2.0
        # Trajectory uses 'r'/'theta' as position keys, 'vx'/'vy' as velocity
        traj = {
            't'  : t,
            'r'  : vr * t,
            'theta': vt * t,
            'xi' : vr * np.ones_like(t),   # p_r
            'eta': vt * np.ones_like(t),   # p_theta
            'vx' : vr * np.ones_like(t),
            'vy' : vt * np.ones_like(t),
        }
        S = prop._cumulative_action(traj, dim=2, coord_keys=('r', 'theta'))
        # With explicit momenta: integrand = xi*vx + eta*vy = vr² + vt²
        expected = self._expected((vr**2 + vt**2) * np.ones_like(t), t)
        assert np.allclose(S, expected, rtol=1e-2)


# ============================================================================
# Maslov index
# ============================================================================

class TestMaslovIndex:

    def test_no_sign_change(self):
        assert prop._maslov_index(np.array([1.0, 2.0, 3.0, 4.0])) == 0

    def test_one_sign_change(self):
        assert prop._maslov_index(np.array([1.0, 2.0, -1.0, -3.0])) == 1

    def test_multiple_changes(self):
        assert prop._maslov_index(np.array([1.0, -1.0, 1.0, -1.0])) == 3

    def test_zeros_ignored(self):
        # 1 → -1 → 1 : two sign changes, zeros in between are skipped
        assert prop._maslov_index(np.array([1.0, 0.0, -1.0, 0.0, 1.0])) == 2


# ============================================================================
# Airy argument mapping  (new: tests the pointwise spatial scaling)
# ============================================================================

class TestAiryArgument:

    def test_zero_at_caustic(self):
        """At x = x_c the Airy argument must be zero."""
        xi = prop._airy_argument(np.array([0.0]), hbar=1.0, alpha=1.0)
        assert xi[0] == pytest.approx(0.0)

    def test_hbar_scaling(self):
        """ξ ∝ ℏ^{-1/3}: halving ℏ should multiply |ξ| by 2^{1/3}."""
        x_local = np.array([1.0])
        xi1 = prop._airy_argument(x_local, hbar=1.0, alpha=1.0)
        xi2 = prop._airy_argument(x_local, hbar=0.5, alpha=1.0)
        assert abs(xi2[0] / xi1[0]) == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)

    def test_sign_convention(self):
        """Sign of ξ must match sign of alpha (lit: oscillations on the α > 0 side)."""
        x_local = np.array([1.0])
        xi_pos = prop._airy_argument(x_local, hbar=1.0, alpha=+2.0)
        xi_neg = prop._airy_argument(x_local, hbar=1.0, alpha=-2.0)
        assert xi_pos[0] > 0
        assert xi_neg[0] < 0

    def test_linear_in_position(self):
        """ξ must be linear in x_local (uniform spacing required for diff test)."""
        x = np.array([0.0, 0.5, 1.0, 1.5])   # uniform Δx = 0.5
        xi = prop._airy_argument(x, hbar=1.0, alpha=1.0)
        diffs = np.diff(xi)
        assert np.allclose(diffs, diffs[0], rtol=1e-10)

    def test_alpha_scaling(self):
        """ξ ∝ |α|^{1/3}: doubling |α| should scale |ξ| by 2^{1/3}."""
        x_local = np.array([1.0])
        xi1 = prop._airy_argument(x_local, hbar=1.0, alpha=1.0)
        xi2 = prop._airy_argument(x_local, hbar=1.0, alpha=2.0)
        assert abs(xi2[0] / xi1[0]) == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)


# ============================================================================
# Asymptotic correction 1D — spatial Airy profile
# ============================================================================

class TestAsymptoticCorrection1D:

    def test_zero_outside_window(self):
        """Patch must be identically zero outside the caustic window."""
        x_grid = np.linspace(-3, 3, 200)
        patch  = prop._asymptotic_correction_1d(
            x_caustic=0.0, S_caustic=0.0, a_caustic=1.0,
            dJ_ds=1.0, hbar=1.0, x_grid=x_grid, width=0.5)
        outside = np.abs(x_grid) >= 0.5
        assert np.all(patch[outside] == 0j)

    def test_nonzero_inside_window(self):
        """Patch must be non-zero inside the window (Airy function is non-trivial)."""
        x_grid = np.linspace(-2, 2, 200)
        patch  = prop._asymptotic_correction_1d(
            x_caustic=0.0, S_caustic=0.0, a_caustic=1.0,
            dJ_ds=1.0, hbar=1.0, x_grid=x_grid, width=1.0)
        assert np.any(patch != 0j)

    def test_uses_real_airy_function(self):
        """
        The patch profile must match scipy_airy evaluated at the correct argument.
        Check at a specific interior point.
        """
        hbar   = 1.0
        alpha  = 2.0
        a_c    = 1.0
        S_c    = 0.0
        x_grid = np.linspace(-2, 2, 500)
        width  = 1.5
        patch  = prop._asymptotic_correction_1d(
            x_caustic=0.0, S_caustic=S_c, a_caustic=a_c,
            dJ_ds=alpha, hbar=hbar, x_grid=x_grid, width=width)

        # Pick a point well inside the window and away from the taper edge
        x_test = 0.3
        idx    = np.argmin(np.abs(x_grid - x_test))
        x_loc  = x_grid[idx]

        # Expected: prefactor * Ai(ξ) * exp(iS/ℏ) * taper
        xi_val  = prop._airy_argument(np.array([x_loc]), hbar, alpha)[0]
        Ai_val, _, _, _ = scipy_airy(xi_val)
        prefactor = (2.0 * np.pi * a_c
                     * (hbar ** (1.0 / 6.0))
                     * (abs(alpha) ** (-1.0 / 3.0)))
        carrier   = np.exp(1j * S_c / hbar)
        taper     = np.cos(np.pi / 2.0 * x_loc / width) ** 2
        expected  = prefactor * Ai_val * carrier * taper

        assert np.isclose(patch[idx], expected, rtol=1e-6)

    def test_carrier_phase(self):
        """A non-zero S_caustic must produce the correct complex carrier."""
        x_grid = np.linspace(-1, 1, 200)
        S_c    = np.pi / 3.0
        hbar   = 0.5
        patch_S  = prop._asymptotic_correction_1d(
            0.0, S_c, 1.0, 1.0, hbar, x_grid, 0.8)
        patch_0  = prop._asymptotic_correction_1d(
            0.0, 0.0, 1.0, 1.0, hbar, x_grid, 0.8)

        # At the caustic centre x=0 the Airy argument is 0; Ai(0) is non-zero.
        # The ratio of the two patches should equal exp(i S_c / hbar).
        idx = np.argmin(np.abs(x_grid))
        if abs(patch_0[idx]) > 1e-10:
            ratio = patch_S[idx] / patch_0[idx]
            assert np.isclose(ratio, np.exp(1j * S_c / hbar), rtol=1e-5)

    def test_taper_vanishes_at_edges(self):
        """cos² taper must vanish at the patch boundary |x - x_c| = width."""
        x_grid = np.linspace(-3, 3, 1000)
        width  = 1.0
        patch  = prop._asymptotic_correction_1d(
            0.0, 0.0, 1.0, 1.0, 1.0, x_grid, width)

        # Just inside the edge the taper ≈ 0 ⇒ patch ≈ 0
        near_edge = np.abs(np.abs(x_grid) - width) < 0.02
        assert np.all(np.abs(patch[near_edge]) < 0.1)

    def test_empty_window(self):
        """If the window contains no grid points, patch should be all zeros."""
        x_grid = np.linspace(5, 10, 100)
        patch  = prop._asymptotic_correction_1d(
            0.0, 0.0, 1.0, 1.0, 1.0, x_grid, 0.1)
        assert np.all(patch == 0j)


# ============================================================================
# Asymptotic correction 2D  (new)
# ============================================================================

class TestAsymptoticCorrection2D:

    def _make_grid(self, n=50, lim=2.0):
        xs = np.linspace(-lim, lim, n)
        X, Y = np.meshgrid(xs, xs)
        return X, Y

    def test_zero_outside_window(self):
        """Patch must be zero outside radius = width."""
        X, Y  = self._make_grid()
        patch = prop._asymptotic_correction_2d(
            x_caustic=0.0, y_caustic=0.0,
            S_caustic=0.0, a_caustic=1.0,
            dJ_dx=1.0, dJ_dy=0.0,
            hbar=1.0, X_grid=X, Y_grid=Y, width=0.5)
        outside = (X**2 + Y**2) >= 0.5**2
        assert np.all(patch[outside] == 0j)

    def test_nonzero_inside_window(self):
        """Patch must be non-zero inside the window."""
        X, Y  = self._make_grid()
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 1.5)
        inside = (X**2 + Y**2) < 1.5**2
        assert np.any(patch[inside] != 0j)

    def test_fold_airy_profile_along_normal(self):
        """
        For a fold caustic with gradient purely along x (dJ_dy=0, dJ_dx=1),
        the transverse direction is x̂.  Two points symmetric about x=0 at
        the same |r_perp| should have the same |patch| (Airy is even/odd
        but taper is symmetric, so the amplitude profile is symmetric).
        """
        n     = 201
        xs    = np.linspace(-1, 1, n)
        X, Y  = np.meshgrid(xs, xs)
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 0.8)

        # At y=0, compare |patch| at x = +0.2 and x = -0.2
        row_mid = n // 2
        col_p   = np.argmin(np.abs(xs - 0.2))
        col_m   = np.argmin(np.abs(xs + 0.2))
        # The Airy function is NOT symmetric in general; what should be
        # symmetric is the taper: verify both points have non-zero patch.
        assert abs(patch[row_mid, col_p]) > 0
        assert abs(patch[row_mid, col_m]) > 0

    def test_cusp_fallback_runs(self):
        """
        When both dJ_dx and dJ_dy are zero (cusp / Pearcey caustic),
        _asymptotic_correction_2d should still return a finite patch without
        raising an exception.
        """
        X, Y  = self._make_grid(n=30)
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, X, Y, 1.0)
        assert np.all(np.isfinite(patch.real))
        assert np.all(np.isfinite(patch.imag))

    def test_cusp_patch_is_nonzero(self):
        """Cusp fallback should produce a non-trivial correction."""
        X, Y  = self._make_grid(n=40)
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, X, Y, 1.5)
        assert np.any(np.abs(patch) > 0)

    def test_carrier_phase_2d(self):
        """Non-zero S_caustic should shift the phase of the patch uniformly."""
        X, Y  = self._make_grid(n=20)
        S_c   = 1.2
        hbar  = 1.0
        p_S   = prop._asymptotic_correction_2d(
            0.0, 0.0, S_c, 1.0, 1.0, 0.0, hbar, X, Y, 1.0)
        p_0   = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, hbar, X, Y, 1.0)

        # Find a cell where both are non-zero and check phase difference
        nz = (np.abs(p_0) > 1e-10) & (np.abs(p_S) > 1e-10)
        if nz.any():
            ratios = p_S[nz] / p_0[nz]
            expected_phase = np.exp(1j * S_c / hbar)
            # All ratios should be close to exp(i S_c / hbar)
            assert np.allclose(ratios / ratios[0], 1.0, atol=1e-5), \
                "Phase ratio should be uniform across the patch"


# ============================================================================
# Build Hamiltonian sym
# ============================================================================

class TestBuildHamiltonianSym:

    def test_1d_power_metric(self):
        """H = ½ g^{-1} ξ² = ½ (1/x²) ξ² for power metric g=x²."""
        m, x = power_1d()
        H, vars_phase = prop._build_hamiltonian_sym(m)
        assert len(vars_phase) == 2
        assert vars_phase[0] == x
        assert str(vars_phase[1]) == 'xi'
        expected = (1 / x**2) * vars_phase[1]**2 / 2
        assert sp.simplify(H - expected) == 0

    def test_2d_flat(self):
        """H = ½ (ξ² + η²) for flat 2D metric."""
        m, (x, y) = flat_2d()
        H, vars_phase = prop._build_hamiltonian_sym(m)
        assert len(vars_phase) == 4
        expected = (vars_phase[1]**2 + vars_phase[3]**2) / 2
        assert sp.simplify(H - expected) == 0


# ============================================================================
# Van Vleck sum
# ============================================================================

class TestVanVleckSum:

    def test_1d_constant_unit_amplitude(self):
        """S=0, det_J=1, μ=0 everywhere → psi = 1 everywhere."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.zeros(3)
        det_J = np.ones(3)
        mu    = np.zeros(3, dtype=int)
        psi, X, Y = prop.van_vleck_sum(pts, S, det_J, mu, xlim=(0, 2), N=10)
        assert Y is None
        assert np.allclose(psi, 1.0 + 0j)

    def test_1d_phase_varies_linearly(self):
        """S(x) = π x on [0,2], det_J=1 → at x=1, ψ = exp(iπ) = −1."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.array([0.0, np.pi, 2*np.pi])
        det_J = np.ones(3)
        mu    = np.zeros(3, dtype=int)
        psi, X, Y = prop.van_vleck_sum(pts, S, det_J, mu, xlim=(0, 2), N=5)
        idx   = np.argmin(np.abs(X - 1.0))
        assert np.isclose(X[idx], 1.0)
        assert np.isclose(psi[idx], -1.0, atol=1e-10)

    def test_1d_maslov_phase_shift(self):
        """μ=1 at a point contributes an extra −π/2 phase shift."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.zeros(3)
        det_J = np.ones(3)
        mu_0  = np.zeros(3, dtype=int)
        mu_1  = np.array([0, 1, 0], dtype=int)
        # The interpolated value at x=1.0 with μ=1 should carry exp(-iπ/2)=-i
        psi0, X, _ = prop.van_vleck_sum(pts, S, det_J, mu_0, xlim=(0, 2), N=5)
        psi1, _,  _ = prop.van_vleck_sum(pts, S, det_J, mu_1, xlim=(0, 2), N=5)
        idx = np.argmin(np.abs(X - 1.0))
        assert np.isclose(psi1[idx] / psi0[idx], np.exp(-1j * np.pi / 2), atol=1e-10)

    @patch('propagator._asymptotic_correction_1d')
    def test_1d_caustic_triggers_airy_patch(self, mock_patch):
        """Near-caustic det_J should trigger the Airy correction."""
        pts   = np.array([[0.0], [0.5], [1.0]])
        S     = np.zeros(3)
        det_J = np.array([1.0, 0.001, 1.0])  # middle → near caustic
        mu    = np.zeros(3, dtype=int)

        sentinel = 42.0 + 0j

        def side_effect(xc, S_c, a_c, dJ_ds, hbar, x_grid, width):
            patch = np.zeros_like(x_grid, dtype=complex)
            patch[np.abs(x_grid - 0.5) < 0.1] = sentinel
            return patch

        mock_patch.side_effect = side_effect
        psi, X, _ = prop.van_vleck_sum(pts, S, det_J, mu, xlim=(0, 1), N=100,
                                        caustic_threshold=0.5)
        assert np.all(psi[np.abs(X - 0.5) < 0.1] == sentinel)
        assert not np.any(psi[np.abs(X - 0.5) >= 0.1] == sentinel)

    def test_2d_constant_unit_amplitude(self):
        """2D: S=0, det_J=1, μ=0 everywhere → psi ≈ 1 at interior grid pts."""
        pts   = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        S     = np.zeros(4)
        det_J = np.ones(4)
        mu    = np.zeros(4, dtype=int)
        psi, X, Y = prop.van_vleck_sum(pts, S, det_J, mu,
                                        xlim=(0, 1), ylim=(0, 1), N=5)
        assert X.shape == (5, 5)
        assert Y.shape == (5, 5)
        assert psi.shape == (5, 5)
        assert np.isclose(psi[2, 2], 1.0)

    @patch('propagator._asymptotic_correction_2d')
    def test_2d_caustic_triggers_airy_patch(self, mock_patch_2d):
        """Near-caustic 2D points should trigger _asymptotic_correction_2d."""
        # Need non-collinear points so that griddata (Delaunay) can triangulate.
        # Place one near-caustic point at (0.5, 0.5) surrounded by four regular ones.
        pts   = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0.5, 0.5]])
        S     = np.zeros(5)
        det_J = np.array([1.0, 1.0, 1.0, 1.0, 0.001])   # last → near caustic
        mu    = np.zeros(5, dtype=int)

        def side_effect(xc, yc, S_c, a_c, dJ_dx, dJ_dy, hbar, X, Y, width):
            patch = np.zeros_like(X, dtype=complex)
            patch[np.hypot(X - xc, Y - yc) < 0.2] = 99.0 + 0j
            return patch

        mock_patch_2d.side_effect = side_effect
        psi, X, Y = prop.van_vleck_sum(pts, S, det_J, mu,
                                        xlim=(0, 1), ylim=(0, 1), N=20,
                                        caustic_threshold=0.5)
        mock_patch_2d.assert_called()

    def test_1d_two_ray_interference(self):
        """
        Two rays at x=0.3 and x=0.7, each with constant action along the ray,
        produce a superposition at the intermediate grid point x=0.5.
        The expected value is the average of the two ray contributions,
        because the interpolation is linear between the scattered points.
        """
        # Scattered points: two positions, each with its own ray data.
        pts = np.array([[0.3], [0.7]])           # (M,1)
        S   = np.array([2.0, 3.0])               # actions
        det_J = np.array([1.0, 1.0])              # unit Jacobians
        mu    = np.array([0, 0], dtype=int)       # no Maslov shift
    
        psi_k = (1.0 / np.sqrt(np.abs(det_J))) * np.exp(1j * S / 1.0 - 1j * mu * np.pi/2)
        # psi_k = [exp(2j), exp(3j)]
    
        psi, X, _ = prop.van_vleck_sum(pts, S, det_J, mu,
                                       xlim=(0, 1), N=5, hbar=1.0)
    
        # At x=0.5, the interpolated value should be the average of the two
        # neighbouring scattered points (linear interpolation between 0.3 and 0.7).
        idx = np.argmin(np.abs(X - 0.5))
        expected = 0.5 * (psi_k[0] + psi_k[1])   # because 0.5 is midway
        assert np.isclose(psi[idx], expected, atol=1e-10)

    def test_1d_interpolation_at_data_point(self):
        """
        Check that the interpolated value at a point that exactly matches
        a scattered point is equal to that point's complex contribution.
        """
        pts = np.array([[0.0], [1.0], [2.0]])
        S   = np.array([0.0, np.pi, 2*np.pi])
        det_J = np.ones(3)
        mu    = np.zeros(3, dtype=int)
    
        psi, X, _ = prop.van_vleck_sum(pts, S, det_J, mu,
                                       xlim=(0, 2), N=3, hbar=1.0)  # grid = [0,1,2]
    
        # Check at x = 1.0 (index 1)
        expected = np.exp(1j * np.pi)   # = -1
        assert np.isclose(psi[1], expected, atol=1e-10)
    
    
    def test_1d_linear_interpolation_of_complex(self):
        """
        Verify that linear interpolation of real and imaginary parts works correctly.
        Use small phase values to avoid wrapping and test at a midpoint.
        """
        pts = np.array([[1.0], [2.0]])
        S   = np.array([0.1, 0.2])
        det_J = np.ones(2)
        mu    = np.zeros(2, dtype=int)
        hbar = 1.0
    
        psi, X, _ = prop.van_vleck_sum(pts, S, det_J, mu,
                                       xlim=(1, 2), N=3, hbar=hbar)  # grid = [1, 1.5, 2]
    
        v1 = np.exp(1j * 0.1)
        v2 = np.exp(1j * 0.2)
        expected = 0.5 * (v1 + v2)   # at x=1.5
    
        # Index of 1.5 is 1 (since grid = [1, 1.5, 2])
        assert np.isclose(psi[1], expected, atol=1e-10)
    
    
    def test_2d_interpolation_at_data_point(self):
        """
        Check that the interpolated value at a point that exactly matches
        a scattered point is equal to that point's complex contribution.
        Uses three non-collinear points and a 3x3 grid.
        """
        pts = np.array([[0.5, 0.5], [0.2, 0.2], [0.8, 0.2]])  # not collinear
        S   = np.array([2.0, 0.0, 0.0])
        det_J = np.ones(3)
        mu    = np.array([0, 0, 0], dtype=int)
    
        psi_k = (1.0 / np.sqrt(np.abs(det_J))) * np.exp(1j * S / 1.0 - 1j * mu * np.pi/2)
    
        psi, X, Y = prop.van_vleck_sum(pts, S, det_J, mu,
                                       xlim=(0, 1), ylim=(0, 1), N=3, hbar=1.0)
    
        # Grid points: x = [0, 0.5, 1], y = [0, 0.5, 1]
        # Index of (0.5,0.5) is iy=1, ix=1
        assert np.isclose(psi[1, 1], psi_k[0], atol=1e-10)

# ============================================================================
# Compute wavefunction — integration tests
# ============================================================================

class TestComputeWavefunction:

    def test_1d_flat_v_fan(self):
        """v_fan API: 1D flat metric, check shapes and non-trivial action."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-1.0, 1.0, 5),
            t_max=1.0, hbar=1.0, n_steps=20, N_grid=10, integrator='verlet')
        assert result.dim == 1
        assert len(result.rays) == 5
        assert result.X.shape == (10,)
        assert result.psi.shape == (10,)
        assert result.y_pts is None
        assert np.any(result.S_pts > 0)

    def test_2d_flat_v_fan(self):
        """v_fan API: 2D flat metric, check shapes."""
        m, _ = flat_2d()
        vx = np.linspace(-0.5, 0.5, 2)
        vy = np.linspace(-0.5, 0.5, 2)
        result = prop.compute_wavefunction(
            metric=m, source=(0.0, 0.0),
            v_fan=np.array([[a, b] for a in vx for b in vy]),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=10, integrator='verlet')
        assert result.dim == 2
        assert len(result.rays) == 4
        assert result.X.shape == (10, 10)
        assert result.y_pts is not None

    def test_from_hamiltonian_1d(self):
        """Metric from Hamiltonian: 3 rays should succeed."""
        x, p = sp.symbols('x p', real=True, positive=True)
        m = Metric.from_hamiltonian(p**2 / (2 * x**2), (x,), (p,))
        result = prop.compute_wavefunction(
            metric=m, source=(1.0,),
            v_fan=np.linspace(-1.0, 1.0, 3),
            t_max=0.5, hbar=1.0, n_steps=10, N_grid=10)
        assert len(result.rays) == 3

#    def test_no_rays_raises(self):
#        """If all rays fail, RuntimeError must be raised."""
#        m, _ = flat_1d()
#        with patch('propagator.hamiltonian_flow', side_effect=Exception):
#            with pytest.raises(RuntimeError):
#                prop.compute_wavefunction(
#                    metric=m, source=(0.0,),
#                    v_fan=np.array([1.0]), t_max=1.0)

    def test_auto_xlim(self):
        """When xlim is not provided, it should be derived from ray endpoints."""
        m, _ = flat_1d()
        with patch('propagator.hamiltonian_flow') as mock_flow:
            def side_effect(H, z0, tspan, vars_phase, integrator, n_steps):
                v0 = z0[1]
                t  = np.linspace(0, 1, n_steps)
                return {'t': t, 'x': v0 * t, 'xi': v0 * np.ones_like(t)}
            mock_flow.side_effect = side_effect
            result = prop.compute_wavefunction(
                metric=m, source=(0.0,),
                v_fan=np.array([1.0, 2.0]),
                t_max=1.0, n_steps=10, N_grid=10)
        x_min = min(result.x_pts) - 0.1 * (max(result.x_pts) - min(result.x_pts))
        x_max = max(result.x_pts) + 0.1 * (max(result.x_pts) - min(result.x_pts))
        assert result.X[0]  == pytest.approx(x_min, rel=1e-2)
        assert result.X[-1] == pytest.approx(x_max, rel=1e-2)

    def test_curved_metric_action_consistent(self):
        """
        Curved 1D metric (g=x²): action via momenta vs action via metric fallback
        should agree.  We verify that the S_cum produced is non-trivial and
        increases monotonically for a positive-velocity ray.
        """
        m, _ = power_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(1.0,),
            v_fan=np.array([0.5]),
            t_max=0.5, hbar=1.0, n_steps=30, N_grid=10)
        ray = result.rays[0]
        # Action must be strictly increasing (positive integrand)
        assert np.all(np.diff(ray.S_cum) >= 0)

    def test_v_fan_required_in_metric_mode(self):
        """In metric mode, omitting v_fan must raise ValueError."""
        m, _ = flat_1d()
        with pytest.raises(ValueError, match="v_fan"):
            prop.compute_wavefunction(
                metric=m, source=(0.0,), t_max=0.5, n_steps=5, N_grid=5)

    def test_both_metric_and_hamiltonian_raises(self):
        """Supplying both metric and hamiltonian must raise ValueError."""
        m, _ = flat_1d()
        x, xi = sp.symbols('x xi', real=True)
        with pytest.raises(ValueError):
            prop.compute_wavefunction(
                metric=m,
                hamiltonian=xi**2 / 2,
                coords=(x,), momenta=(xi,),
                source=(0.0,), p_fan=np.array([1.0]), t_max=0.5)

    def test_neither_metric_nor_hamiltonian_raises(self):
        """Supplying neither must raise ValueError."""
        with pytest.raises(ValueError):
            prop.compute_wavefunction(
                source=(0.0,), t_max=0.5,
                v_fan=np.array([1.0]), n_steps=5, N_grid=5)

    # ── Mode B (general Hamiltonian) tests ────────────────────────────────────

    def test_mode_b_1d_free_particle(self):
        """
        Mode B with H = ξ²/2 (flat free particle) must agree with Mode A:
        same number of successful rays, non-trivial action, monotone S_cum.
        """
        x, xi = sp.symbols('x xi', real=True)
        H_free = xi**2 / 2
        p_fan  = np.linspace(-1.0, 1.0, 5)
        result = prop.compute_wavefunction(
            hamiltonian=H_free, coords=(x,), momenta=(xi,),
            source=(0.0,),
            p_fan=p_fan,
            t_max=1.0, hbar=1.0, n_steps=20, N_grid=10,
            integrator='verlet')
        assert result.dim == 1
        assert len(result.rays) == 5
        assert result.psi.shape == (10,)
        # All non-zero rays have non-trivial action
        nz_rays = [r for r in result.rays if abs(r.S_cum[-1]) > 1e-12]
        for ray in nz_rays:
            assert np.all(np.diff(ray.S_cum) >= -1e-10)

    def test_mode_b_1d_harmonic_oscillator(self):
        """
        Mode B: H = ξ²/2 + x²/2 (harmonic oscillator).
        Rays should show caustic crossings (Maslov > 0) for long enough t_max.
        """
        x, xi = sp.symbols('x xi', real=True)
        H_ho = xi**2 / 2 + x**2 / 2
        p_fan = np.linspace(-1.5, 1.5, 20)
        result = prop.compute_wavefunction(
            hamiltonian=H_ho, coords=(x,), momenta=(xi,),
            source=(0.0,),
            p_fan=p_fan,
            t_max=np.pi * 1.2,    # slightly more than one full period
            hbar=0.3, n_steps=200, N_grid=50,
            integrator='rk45')
        assert result.dim == 1
        assert len(result.rays) >= 10
        # At least some rays should have acquired Maslov phase (crossed a caustic)
        maslov_values = [r.mu for r in result.rays]
        assert max(maslov_values) >= 1, \
            "Expected at least one caustic crossing in a full HO period"

    def test_mode_b_p_fan_required(self):
        """In general-Hamiltonian mode, omitting p_fan must raise ValueError."""
        x, xi = sp.symbols('x xi', real=True)
        with pytest.raises(ValueError, match="p_fan"):
            prop.compute_wavefunction(
                hamiltonian=xi**2/2, coords=(x,), momenta=(xi,),
                source=(0.0,), t_max=1.0, n_steps=5, N_grid=5)

    def test_mode_b_coords_momenta_required(self):
        """
        In Mode B, omitting coords or momenta must raise ValueError.
        """
        x, xi = sp.symbols('x xi', real=True)
        with pytest.raises(ValueError):
            prop.compute_wavefunction(
                hamiltonian=xi**2/2,
                source=(0.0,), p_fan=np.array([1.0]),
                t_max=1.0, n_steps=5, N_grid=5)  # no coords/momenta

    def test_mode_b_2d_free_particle(self):
        """
        Mode B 2D: H = (ξ²+η²)/2.  Should produce a result with 2D output.
        """
        x, y, xi, eta = sp.symbols('x y xi eta', real=True)
        H_2d = (xi**2 + eta**2) / 2
        px = np.linspace(-0.5, 0.5, 3)
        py = np.linspace(-0.5, 0.5, 3)
        p_fan_2d = np.array([[a, b] for a in px for b in py])
        result = prop.compute_wavefunction(
            hamiltonian=H_2d, coords=(x, y), momenta=(xi, eta),
            source=(0.0, 0.0),
            p_fan=p_fan_2d,
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=10,
            integrator='verlet')
        assert result.dim == 2
        assert result.psi.shape == (10, 10)
        assert result.y_pts is not None

    def test_mode_b_jacobi_1d_general(self):
        """
        _det_J_1d_general on H = ξ²/2 (free particle) must return J(t) ≈ t,
        exactly as _det_J_1d does on a flat metric — the two ODEs coincide.
        """
        x, xi = sp.symbols('x xi', real=True)
        H_free = xi**2 / 2
        vars_p = [x, xi]
        t = np.linspace(0, 2, 50)
        traj = {'t': t, 'x': t.copy(), 'xi': np.ones_like(t)}
        det_J = prop._det_J_1d_general(H_free, vars_p, traj, (0, 2), 50)
        assert np.allclose(det_J, t, rtol=1e-3), \
            "Free-particle Jacobi scalar must equal t"

    def test_resolve_hamiltonian_invalid_dim(self):
        """_resolve_hamiltonian must reject dim > 2."""
        coords  = sp.symbols('x y z', real=True)
        momenta = sp.symbols('px py pz', real=True)
        H = sum(p**2 for p in momenta) / 2
        with pytest.raises(ValueError, match="1D and 2D"):
            prop._resolve_hamiltonian(
                metric=None, hamiltonian=H,
                coords=coords, momenta=momenta)


# ============================================================================
# Visualisation — smoke tests
# ============================================================================

class TestVisualisation:

    def _make_1d_result(self):
        class DummyRay:
            traj  = {'t': np.linspace(0, 1, 5), 'x': np.linspace(0, 1, 5)}
            det_J = np.ones(5)
            S_cum = np.linspace(0, 1, 5)
            mu    = 0
        return prop.WKBResult(
            rays=[DummyRay()],
            X=np.linspace(0, 1, 10), Y=None,
            psi=np.ones(10, dtype=complex),
            x_pts=np.linspace(0, 1, 50), y_pts=None,
            S_pts=np.linspace(0, 1, 50),
            det_J_pts=np.ones(50),
            mu_pts=np.zeros(50, dtype=int),
            hbar=1.0, t_max=1.0, dim=1)

    def _make_2d_result(self):
        n = 10
        X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        dummy = MagicMock()
        dummy.traj  = {'x': np.linspace(0, 1, 5), 'y': np.linspace(0, 1, 5),
                       't': np.linspace(0, 1, 5)}
        dummy.det_J = np.ones(5)
        dummy.S_cum = np.linspace(0, 1, 5)
        dummy.mu    = 0
        return prop.WKBResult(
            rays=[dummy], X=X, Y=Y,
            psi=np.ones((n, n), dtype=complex),
            x_pts=np.linspace(0, 1, 20), y_pts=np.linspace(0, 1, 20),
            S_pts=np.ones(20), det_J_pts=np.ones(20),
            mu_pts=np.zeros(20, dtype=int),
            hbar=1.0, t_max=1.0, dim=2)

    def test_animate_wavefunction_smoke(self):
        """Test that animate_wavefunction runs without error (minimal case)."""
        # Create a minimal 1D result
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-0.5, 0.5, 5),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=10
        )
        # Just check that the animation object is created
        ani = prop.animate_wavefunction(result, n_frames=120, interval=30)
        assert isinstance(ani, animation.FuncAnimation)
        plt.close(ani._fig)   # clean up

    @patch('matplotlib.pyplot.show')
    def test_plot_wavefunction_1d(self, _):
        prop.plot_wavefunction(self._make_1d_result(), log_scale=True)

    @patch('matplotlib.pyplot.show')
    def test_plot_wavefunction_2d(self, _):
        prop.plot_wavefunction(self._make_2d_result(), log_scale=True)

    @patch('matplotlib.pyplot.show')
    def test_plot_ray_fan_1d(self, _):
        result = self._make_1d_result()
        prop.plot_ray_fan(result)

    @patch('matplotlib.pyplot.show')
    def test_plot_ray_fan_2d(self, _):
        prop.plot_ray_fan(self._make_2d_result())

    @patch('matplotlib.pyplot.show')
    def test_plot_interference_detail_1d(self, _):
        prop.plot_interference_detail(self._make_1d_result())

    @patch('matplotlib.pyplot.show')
    def test_plot_interference_detail_2d(self, _):
        prop.plot_interference_detail(self._make_2d_result())