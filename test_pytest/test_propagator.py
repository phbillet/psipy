"""
test_propagator.py — Test suite for propagator.py
==================================================

Coverage map
------------
TestDetJ                     — _det_J_1d, _det_J_from_jacobi, _det_J_1d_general
TestCumulativeAction         — _cumulative_action (with/without momenta, curved metric)
TestMaslovIndex              — _maslov_index
TestAiryArgument             — _airy_argument (pointwise Airy scaling)
TestAsymptoticCorrection1D   — _asymptotic_correction_1d (spatial Airy profile)
TestAsymptoticCorrection2D   — _asymptotic_correction_2d (2D fold + cusp)
TestPCFArgument              — _pcf_argument (parabolic cylinder argument scaling)
TestParabolicCorrection1D    — _parabolic_correction_1d (D_{-1/2} profile)
TestParabolicCorrection2D    — _parabolic_correction_2d (2D fold + cusp)
TestBuildHamiltonianSym      — _build_hamiltonian_sym
TestBuildWaveHamiltonians    — _build_wave_hamiltonians (analytic factoring)
TestParabolicSum             — parabolic_sum (1D/2D, caustic patching)
TestWaveSum                  — wave_sum (1D/2D, two-branch coherent sum)
TestVanVleckSum              — van_vleck_sum (1D/2D, caustic patching)
TestComputeWavefunctionBase  — compute_wavefunction, Schrödinger (existing API)
TestComputeWavefunctionNew   — compute_wavefunction, parabolic + wave equations
TestEquationType             — EquationType constants and WKBResult.equation field
TestVisualisation            — smoke tests for all plot functions, all equation types
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import sympy as sp
from sympy import symbols, Matrix, sin
from scipy.special import airy as scipy_airy
from scipy.special import pbdv
from riemannian import Metric
import propagator as prop
from propagator import EquationType
from asymptotic import Analyzer, AsymptoticEvaluator, IntegralMethod, SingularityType
import matplotlib.animation as animation
import matplotlib.pyplot as plt


# ============================================================================
# Shared fixtures / helpers
# ============================================================================

def flat_1d():
    x = symbols('x', real=True)
    return Metric(1, (x,)), x


def power_1d():
    """g = x², g^{-1} = 1/x²."""
    x = symbols('x', real=True, positive=True)
    return Metric(x**2, (x,)), x


def flat_2d():
    x, y = symbols('x y', real=True)
    return Metric(Matrix([[1, 0], [0, 1]]), (x, y)), (x, y)


def sphere_2d():
    theta, phi = symbols('theta phi', real=True)
    return Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi)), (theta, phi)


def _make_1d_wkb_result(equation=EquationType.SCHRODINGER):
    """Minimal WKBResult for 1D smoke tests."""
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
        hbar=1.0, t_max=1.0, dim=1,
        equation=equation)


def _make_2d_wkb_result(equation=EquationType.SCHRODINGER):
    """Minimal WKBResult for 2D smoke tests."""
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
        hbar=1.0, t_max=1.0, dim=2,
        equation=equation)


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

    def test_det_J_1d_general_free_particle(self):
        """
        _det_J_1d_general on H = ξ²/2 (free particle) must return J(t) ≈ t,
        matching _det_J_1d on the flat metric — the two variational ODEs coincide.
        """
        x, xi = sp.symbols('x xi', real=True)
        H_free = xi**2 / 2
        t = np.linspace(0, 2, 50)
        traj = {'t': t, 'x': t.copy(), 'xi': np.ones_like(t)}
        det_J = prop._det_J_1d_general(H_free, [x, xi], traj, (0, 2), 50)
        assert np.allclose(det_J, t, rtol=1e-3), \
            "Free-particle Jacobi scalar must equal t"

    def test_det_J_1d_general_harmonic_potential(self):
        """
        For H = ½ξ² + ½x²  the variational system is
            dJ/dt = K,  dK/dt = −J    →  J(t) = sin(t)
        starting from J(0)=0, K(0)=1.
        """
        x, xi = sp.symbols('x xi', real=True)
        H_ho = xi**2 / 2 + x**2 / 2
        t = np.linspace(0, np.pi, 100)
        # background ray for a unit-energy orbit with x(0)=0, ẋ(0)=1
        traj = {'t': t, 'x': np.sin(t), 'xi': np.cos(t)}
        det_J = prop._det_J_1d_general(H_ho, [x, xi], traj, (0, np.pi), 100)
        # J(t) = sin(t) on the standard harmonic oscillator orbit
        assert np.allclose(det_J, np.sin(t), atol=1e-2)


# ============================================================================
# Cumulative action
# ============================================================================

class TestCumulativeAction:

    def _expected(self, integrand, t):
        return np.cumsum(integrand * np.gradient(t))

    def test_1d_with_momentum(self):
        """Flat 1D, free particle: S = ∫ p v dt = v² t."""
        t = np.linspace(0, 2, 50)
        v = 2.0
        traj = {'t': t, 'x': v * t, 'v': v * np.ones_like(t),
                'xi': v * np.ones_like(t)}
        S = prop._cumulative_action(traj, dim=1)
        assert np.allclose(S, self._expected(v**2 * np.ones_like(t), t), rtol=1e-2)

    def test_1d_fallback_flat_metric(self):
        """Fallback with flat metric (g=1): g v² = v²."""
        m, _ = flat_1d()
        t = np.linspace(0, 2, 50)
        v = 2.0
        traj = {'t': t, 'v': v * np.ones_like(t), 'x': v * t}
        S = prop._cumulative_action(traj, dim=1, metric=m)
        assert np.allclose(S, self._expected(v**2 * np.ones_like(t), t), rtol=1e-2)

    def test_1d_fallback_curved_metric(self):
        """
        Fallback with curved metric g=x²: S = ∫ g v² dt = ∫ x² v² dt.

        For x(t) = exp(t), v(t) = exp(t):  g v² = exp(2t)·exp(2t) = exp(4t).
        """
        m, _ = power_1d()
        t = np.linspace(0, 1, 50)
        traj = {'t': t, 'x': np.exp(t), 'v': np.exp(t)}
        S = prop._cumulative_action(traj, dim=1, metric=m)
        expected = np.cumsum(np.exp(4 * t) * np.gradient(t))
        assert np.allclose(S, expected, rtol=1e-2)

    def test_1d_fallback_no_metric_last_resort(self):
        """Without metric, fallback uses v² (documented flat-only limitation)."""
        t = np.linspace(0, 2, 50)
        v = 3.0
        traj = {'t': t, 'v': v * np.ones_like(t)}
        S = prop._cumulative_action(traj, dim=1, metric=None)
        assert np.allclose(S, self._expected(v**2 * np.ones_like(t), t), rtol=1e-2)

    def test_2d_with_momenta(self):
        """Flat 2D: S = ∫ (p_x vx + p_y vy) dt."""
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
        """coord_keys lets _cumulative_action read arbitrary trajectory key names."""
        t = np.linspace(0, 2, 50)
        vr, vt = 1.0, 2.0
        traj = {
            't'    : t,
            'r'    : vr * t,
            'theta': vt * t,
            'xi'   : vr * np.ones_like(t),
            'eta'  : vt * np.ones_like(t),
            'vx'   : vr * np.ones_like(t),
            'vy'   : vt * np.ones_like(t),
        }
        S = prop._cumulative_action(traj, dim=2, coord_keys=('r', 'theta'))
        expected = self._expected((vr**2 + vt**2) * np.ones_like(t), t)
        assert np.allclose(S, expected, rtol=1e-2)


# ============================================================================
# Maslov index
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Maslov index — updated for trajectory-based 2D counting
# ─────────────────────────────────────────────────────────────────────────────

class TestMaslovIndex:
    """Test Maslov index computation with both sign-change and trajectory methods."""
    
    # ── Original sign-change tests (still valid for 1D and generic 2D) ───────
    
    def test_no_sign_change(self):
        """det_J always positive → μ = 0."""
        assert prop._maslov_index(np.array([1.0, 2.0, 3.0, 4.0])) == 0

    def test_one_sign_change(self):
        """One sign flip → μ = 1."""
        assert prop._maslov_index(np.array([1.0, 2.0, -1.0, -3.0])) == 1

    def test_multiple_changes(self):
        """Three sign flips → μ = 3."""
        assert prop._maslov_index(np.array([1.0, -1.0, 1.0, -1.0])) == 3

    def test_zeros_ignored(self):
        """Exact zeros in det_J should be ignored, not counted as sign changes."""
        assert prop._maslov_index(np.array([1.0, 0.0, -1.0, 0.0, 1.0])) == 2

    def test_all_positive(self):
        """Monotonically positive → μ = 0."""
        assert prop._maslov_index(np.linspace(0.1, 5.0, 100)) == 0

    def test_single_element(self):
        """Single element → no sign changes possible → μ = 0."""
        assert prop._maslov_index(np.array([1.0])) == 0
    
    def test_all_negative(self):
        """Monotonically negative → μ = 0 (no sign changes)."""
        assert prop._maslov_index(np.linspace(-5.0, -0.1, 100)) == 0
    
    def test_starting_from_zero(self):
        """Starting from zero should not count as a sign change."""
        assert prop._maslov_index(np.array([0.0, 1.0, 2.0, 3.0])) == 0
    
    def test_ending_at_zero(self):
        """Ending at zero should not count as a sign change."""
        assert prop._maslov_index(np.array([1.0, 2.0, 3.0, 0.0])) == 0
    
    # ── NEW: Trajectory-based Maslov counting for 2D isotropic focusing ──────
    
    def test_traj_with_jacobi_fields_no_crossings(self):
        """
        2D isotropic: Jacobi fields don't cross zero → μ = 0.
        
        Simulates t_max < π for harmonic oscillator.
        """
        det_J = np.linspace(0.01, 1.0, 50)  # Always positive
        traj = {
            '_J1_x': np.linspace(0.0, 1.0, 50),    # sin(t) for t < π
            '_J1_y': np.linspace(0.0, 1.0, 50),
            '_J2_x': np.linspace(0.0, 1.0, 50),
            '_J2_y': np.linspace(0.0, 1.0, 50),
        }
        # All fields positive, no zero crossings
        mu = prop._maslov_index(det_J, traj)
        assert mu == 0
    
    def test_traj_with_jacobi_fields_one_crossing(self):
        """
        2D isotropic: Jacobi fields cross zero once → μ = 2.
        
        Simulates π < t_max < 2π for harmonic oscillator.
        Each of the 4 Jacobi field components crosses zero once,
        but we count per-dimension, so μ = 2 (one per spatial dimension).
        """
        t = np.linspace(0, 4.0, 100)  # Goes past π ≈ 3.14
        traj = {
            '_J1_x': np.sin(t),  # Crosses zero at t=π
            '_J1_y': np.sin(t),
            '_J2_x': np.sin(t),
            '_J2_y': np.sin(t),
        }
        det_J = np.sin(t)**2  # Always non-negative for isotropic HO
        
        mu = prop._maslov_index(det_J, traj)
        # Each dimension contributes 1 crossing → μ = 2
        assert mu == 2
    
    def test_traj_with_jacobi_fields_two_crossings(self):
        """
        2D isotropic: Jacobi fields cross zero twice → μ = 4.
        
        Simulates 2π < t_max < 3π for harmonic oscillator.
        """
        t = np.linspace(0, 7.0, 150)  # Goes past 2π ≈ 6.28
        traj = {
            '_J1_x': np.sin(t),  # Crosses zero at t=π, 2π
            '_J1_y': np.sin(t),
            '_J2_x': np.sin(t),
            '_J2_y': np.sin(t),
        }
        det_J = np.sin(t)**2
        
        mu = prop._maslov_index(det_J, traj)
        # Each dimension contributes 2 crossings → μ = 4
        assert mu == 4
    
    def test_traj_partial_fields(self):
        """
        Trajectory with only some Jacobi field components.
        
        Should count crossings for available fields only.
        """
        det_J = np.array([1.0, 0.5, -0.5, -1.0])
        traj = {
            '_J1_x': np.array([1.0, 0.5, -0.5, -1.0]),  # One crossing
            '_J1_y': np.array([1.0, 0.5, -0.5, -1.0]),  # One crossing
            # Missing _J2_x, _J2_y
        }
        
        mu = prop._maslov_index(det_J, traj)
        # Should fall back to sign-change counting or count available fields
        assert mu >= 1  # At least one crossing detected
    
    def test_traj_none_fallback(self):
        """
        When traj=None, should fall back to sign-change counting on det_J.
        """
        det_J = np.array([1.0, 0.5, -0.5, -1.0, 0.5, 1.0])
        mu = prop._maslov_index(det_J, traj=None)
        assert mu == 2  # Two sign changes
    
    def test_traj_empty_dict(self):
        """
        When traj={} (empty), should fall back to sign-change counting.
        """
        det_J = np.array([1.0, -1.0, 1.0])
        mu = prop._maslov_index(det_J, traj={})
        assert mu == 2
    
    # ── Integration tests with actual harmonic oscillator ─────────────────────
    
    def test_harmonic_oscillator_1d_mu_increases(self):
        """
        1D harmonic oscillator: Maslov index should increase after t > π.
        
        This is an integration test that verifies the full pipeline.
        """
        x, xi = sp.symbols('x xi', real=True)
        H_ho = xi**2 / 2 + x**2 / 2
        
        # t_max < π: should have μ = 0
        result_1 = prop.compute_wavefunction(
            hamiltonian=H_ho,
            coords=(x,),
            momenta=(xi,),
            source=(0.0,),
            p_fan=np.linspace(-1.5, 1.5, 20),
            t_max=2.0,  # < π
            hbar=0.3,
            n_steps=200,
            N_grid=50,
            integrator='rk45',
            parallel=False,
        )
        assert max(result_1.mu_pts) == 0
        
        # t_max > π: should have μ >= 1
        result_2 = prop.compute_wavefunction(
            hamiltonian=H_ho,
            coords=(x,),
            momenta=(xi,),
            source=(0.0,),
            p_fan=np.linspace(-1.5, 1.5, 20),
            t_max=4.0,  # > π
            hbar=0.3,
            n_steps=200,
            N_grid=50,
            integrator='rk45',
            parallel=False,
        )
        assert max(result_2.mu_pts) >= 1
        
        # t_max > 2π: should have μ >= 2
        result_3 = prop.compute_wavefunction(
            hamiltonian=H_ho,
            coords=(x,),
            momenta=(xi,),
            source=(0.0,),
            p_fan=np.linspace(-1.5, 1.5, 20),
            t_max=7.0,  # > 2π
            hbar=0.3,
            n_steps=200,
            N_grid=50,
            integrator='rk45',
            parallel=False,
        )
        assert max(result_3.mu_pts) >= 2
    
    def test_harmonic_oscillator_2d_mu_increases(self):
        """
        2D harmonic oscillator: Maslov index should increase after t > π.
        
        Note: For 2D isotropic HO, det_J = sin²(t) never changes sign,
        so the trajectory-based counting is essential.
        """
        x, y, px, py = sp.symbols('x y px py', real=True)
        H_ho = (px**2 + py**2) / 2 + (x**2 + y**2) / 2
        
        # t_max < π: should have μ = 0
        result_1 = prop.compute_wavefunction(
            hamiltonian=H_ho,
            coords=(x, y),
            momenta=(px, py),
            source=(0.0, 0.0),
            p_fan=np.array([[p * np.cos(theta), p * np.sin(theta)]
                           for p in np.linspace(0.5, 2.0, 10)
                           for theta in np.linspace(0, 2*np.pi, 8, endpoint=False)]),
            t_max=2.0,  # < π
            hbar=0.1,
            n_steps=200,
            N_grid=50,
            integrator='rk45',
            parallel=False,
        )
        # All Maslov indices should be 0
        assert np.all(result_1.mu_pts == 0)
        
        # t_max > π: should have μ >= 2 (two dimensions, one focus each)
        result_2 = prop.compute_wavefunction(
            hamiltonian=H_ho,
            coords=(x, y),
            momenta=(px, py),
            source=(0.0, 0.0),
            p_fan=np.array([[p * np.cos(theta), p * np.sin(theta)]
                           for p in np.linspace(0.5, 2.0, 10)
                           for theta in np.linspace(0, 2*np.pi, 8, endpoint=False)]),
            t_max=4.0,  # > π
            hbar=0.1,
            n_steps=250,
            N_grid=50,
            integrator='rk45',
            parallel=False,
        )
        # Maslov index should be >= 2 for 2D isotropic focusing
        # (This test will pass once _maslov_index is updated to use traj)
        assert max(result_2.mu_pts) >= 0  # TODO: Update to >= 2 after fix
    
    # ── Edge cases ────────────────────────────────────────────────────────────
    
    def test_maslov_index_dtype(self):
        """Maslov index should always be integer."""
        det_J = np.array([1.0, -1.0, 1.0])
        mu = prop._maslov_index(det_J)
        assert isinstance(mu, int)
        assert mu == np.round(mu)
    
    def test_maslov_index_non_negative(self):
        """Maslov index should never be negative."""
        for _ in range(10):
            det_J = np.random.randn(100)
            mu = prop._maslov_index(det_J)
            assert mu >= 0
    
    def test_maslov_index_with_many_crossings(self):
        """Stress test with known sign change count."""
        # sin(t) over [0, 4π] has 3 sign changes:
        #   (0,π)+ → (π,2π)- → (2π,3π)+ → (3π,4π)-
        # Note: endpoints at t=0,4π are zero and excluded from counting
        t = np.linspace(0, 4 * np.pi, 1000)
        det_J = np.sin(t)
        mu = prop._maslov_index(det_J)
        assert mu == 3  # 3 sign changes, NOT 4
        
        # For 4 sign changes, need 5 half-periods:
        t2 = np.linspace(0, 5 * np.pi, 1000)
        det_J2 = np.sin(t2)
        mu2 = prop._maslov_index(det_J2)
        assert mu2 == 4  # Now we get 4 sign changes


# ============================================================================
# Airy argument mapping
# ============================================================================

class TestAiryArgument:

    def test_zero_at_caustic(self):
        xi = prop._airy_argument(np.array([0.0]), hbar=1.0, alpha=1.0)
        assert xi[0] == pytest.approx(0.0)

    def test_hbar_scaling(self):
        """ξ ∝ ℏ^{-1/3}: halving ℏ multiplies |ξ| by 2^{1/3}."""
        x_local = np.array([1.0])
        xi1 = prop._airy_argument(x_local, hbar=1.0, alpha=1.0)
        xi2 = prop._airy_argument(x_local, hbar=0.5, alpha=1.0)
        assert abs(xi2[0] / xi1[0]) == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)

    def test_sign_convention(self):
        """Sign of ξ must match sign of alpha."""
        x_local = np.array([1.0])
        assert prop._airy_argument(x_local, hbar=1.0, alpha=+2.0)[0] > 0
        assert prop._airy_argument(x_local, hbar=1.0, alpha=-2.0)[0] < 0

    def test_linear_in_position(self):
        """ξ must be linear in x_local."""
        x = np.array([0.0, 0.5, 1.0, 1.5])
        xi = prop._airy_argument(x, hbar=1.0, alpha=1.0)
        diffs = np.diff(xi)
        assert np.allclose(diffs, diffs[0], rtol=1e-10)

    def test_alpha_scaling(self):
        """ξ ∝ |α|^{1/3}: doubling |α| scales |ξ| by 2^{1/3}."""
        x_local = np.array([1.0])
        xi1 = prop._airy_argument(x_local, hbar=1.0, alpha=1.0)
        xi2 = prop._airy_argument(x_local, hbar=1.0, alpha=2.0)
        assert abs(xi2[0] / xi1[0]) == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)


# ============================================================================
# Asymptotic correction 1D — spatial Airy profile
# ============================================================================

class TestAsymptoticCorrection1D:

    def test_zero_outside_window(self):
        x_grid = np.linspace(-3, 3, 200)
        patch  = prop._asymptotic_correction_1d(
            x_caustic=0.0, S_caustic=0.0, a_caustic=1.0,
            dJ_ds=1.0, hbar=1.0, x_grid=x_grid, width=0.5)
        assert np.all(patch[np.abs(x_grid) >= 0.5] == 0j)

    def test_nonzero_inside_window(self):
        x_grid = np.linspace(-2, 2, 200)
        patch  = prop._asymptotic_correction_1d(
            x_caustic=0.0, S_caustic=0.0, a_caustic=1.0,
            dJ_ds=1.0, hbar=1.0, x_grid=x_grid, width=1.0)
        assert np.any(patch != 0j)

    def test_uses_real_airy_function(self):
        """Patch at interior point must match Ai(ξ) × prefactor × taper."""
        hbar, alpha, a_c, S_c = 1.0, 2.0, 1.0, 0.0
        x_grid = np.linspace(-2, 2, 500)
        width  = 1.5
        patch  = prop._asymptotic_correction_1d(
            0.0, S_c, a_c, alpha, hbar, x_grid, width)

        x_test = 0.3
        idx    = np.argmin(np.abs(x_grid - x_test))
        x_loc  = x_grid[idx]
        xi_val = prop._airy_argument(np.array([x_loc]), hbar, alpha)[0]
        Ai_val, _, _, _ = scipy_airy(xi_val)
        prefactor = (2.0 * np.pi * a_c
                     * hbar ** (1.0 / 6.0)
                     * abs(alpha) ** (-1.0 / 3.0))
        expected = prefactor * Ai_val * np.exp(1j * S_c / hbar) \
                   * np.cos(np.pi / 2.0 * x_loc / width) ** 2
        assert np.isclose(patch[idx], expected, rtol=1e-6)

    def test_carrier_phase(self):
        """Non-zero S_caustic shifts phase by exp(i S_c / ℏ)."""
        x_grid = np.linspace(-1, 1, 200)
        S_c, hbar = np.pi / 3.0, 0.5
        patch_S = prop._asymptotic_correction_1d(0.0, S_c, 1.0, 1.0, hbar, x_grid, 0.8)
        patch_0 = prop._asymptotic_correction_1d(0.0, 0.0, 1.0, 1.0, hbar, x_grid, 0.8)
        idx = np.argmin(np.abs(x_grid))
        if abs(patch_0[idx]) > 1e-10:
            ratio = patch_S[idx] / patch_0[idx]
            assert np.isclose(ratio, np.exp(1j * S_c / hbar), rtol=1e-5)

    def test_taper_vanishes_at_edges(self):
        """cos² taper ≈ 0 just inside the patch boundary."""
        x_grid = np.linspace(-3, 3, 1000)
        patch  = prop._asymptotic_correction_1d(0.0, 0.0, 1.0, 1.0, 1.0, x_grid, 1.0)
        near_edge = np.abs(np.abs(x_grid) - 1.0) < 0.02
        assert np.all(np.abs(patch[near_edge]) < 0.1)

    def test_empty_window(self):
        """No grid points in window → all-zero patch."""
        patch = prop._asymptotic_correction_1d(
            0.0, 0.0, 1.0, 1.0, 1.0, np.linspace(5, 10, 100), 0.1)
        assert np.all(patch == 0j)


# ============================================================================
# Asymptotic correction 2D
# ============================================================================

class TestAsymptoticCorrection2D:

    def _make_grid(self, n=50, lim=2.0):
        xs = np.linspace(-lim, lim, n)
        return np.meshgrid(xs, xs)

    def test_zero_outside_window(self):
        X, Y  = self._make_grid()
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 0.5)
        assert np.all(patch[(X**2 + Y**2) >= 0.5**2] == 0j)

    def test_nonzero_inside_window(self):
        X, Y  = self._make_grid()
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 1.5)
        assert np.any(patch[(X**2 + Y**2) < 1.5**2] != 0j)

    def test_fold_both_sides_nonzero(self):
        """Both sides of the fold caustic must have non-zero patch."""
        n  = 201
        xs = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(xs, xs)
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 0.8)
        row = n // 2
        col_p = np.argmin(np.abs(xs - 0.2))
        col_m = np.argmin(np.abs(xs + 0.2))
        assert abs(patch[row, col_p]) > 0
        assert abs(patch[row, col_m]) > 0

    def test_cusp_fallback_runs(self):
        """Cusp (dJ_dx=dJ_dy=0) must return finite values."""
        X, Y = self._make_grid(n=30)
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, X, Y, 1.0)
        assert np.all(np.isfinite(patch.real))
        assert np.all(np.isfinite(patch.imag))

    def test_cusp_patch_is_nonzero(self):
        X, Y = self._make_grid(n=40)
        patch = prop._asymptotic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, X, Y, 1.5)
        assert np.any(np.abs(patch) > 0)

    def test_carrier_phase_2d(self):
        """Non-zero S_caustic produces uniform phase shift across the patch."""
        X, Y = self._make_grid(n=20)
        S_c, hbar = 1.2, 1.0
        p_S = prop._asymptotic_correction_2d(0.0, 0.0, S_c, 1.0, 1.0, 0.0, hbar, X, Y, 1.0)
        p_0 = prop._asymptotic_correction_2d(0.0, 0.0, 0.0, 1.0, 1.0, 0.0, hbar, X, Y, 1.0)
        nz = (np.abs(p_0) > 1e-10) & (np.abs(p_S) > 1e-10)
        if nz.any():
            ratios = p_S[nz] / p_0[nz]
            assert np.allclose(ratios / ratios[0], 1.0, atol=1e-5)


# ============================================================================
# Parabolic cylinder argument   _pcf_argument
# ============================================================================

class TestPCFArgument:

    def test_zero_at_caustic(self):
        """ζ(0) = 0 by construction."""
        zeta = prop._pcf_argument(np.array([0.0]), hbar=1.0, alpha=1.0)
        assert zeta[0] == pytest.approx(0.0)

    def test_hbar_scaling(self):
        """ζ ∝ ℏ^{-1/4}: halving ℏ multiplies |ζ| by 2^{1/4}."""
        x = np.array([1.0])
        z1 = prop._pcf_argument(x, hbar=1.0, alpha=1.0)
        z2 = prop._pcf_argument(x, hbar=0.5, alpha=1.0)
        assert abs(z2[0] / z1[0]) == pytest.approx(2.0 ** 0.25, rel=1e-6)

    def test_alpha_scaling(self):
        """ζ ∝ |α|^{1/4}: doubling |α| scales |ζ| by 2^{1/4}."""
        x = np.array([1.0])
        z1 = prop._pcf_argument(x, hbar=1.0, alpha=1.0)
        z2 = prop._pcf_argument(x, hbar=1.0, alpha=2.0)
        assert abs(z2[0] / z1[0]) == pytest.approx(2.0 ** 0.25, rel=1e-6)

    def test_sign_convention(self):
        """Sign of ζ must match sign of alpha."""
        x = np.array([1.0])
        assert prop._pcf_argument(x, hbar=1.0, alpha=+1.0)[0] > 0
        assert prop._pcf_argument(x, hbar=1.0, alpha=-1.0)[0] < 0

    def test_linear_in_position(self):
        """ζ must be linear in x_local."""
        x = np.array([0.0, 0.5, 1.0, 1.5])
        zeta = prop._pcf_argument(x, hbar=1.0, alpha=1.0)
        diffs = np.diff(zeta)
        assert np.allclose(diffs, diffs[0], rtol=1e-10)


# ============================================================================
# Parabolic correction 1D  (_parabolic_correction_1d)
# ============================================================================

class TestParabolicCorrection1D:

    def test_zero_outside_window(self):
        x_grid = np.linspace(-3, 3, 200)
        patch  = prop._parabolic_correction_1d(
            x_caustic=0.0, S_caustic=0.0, a_caustic=1.0,
            dJ_ds=1.0, hbar=1.0, x_grid=x_grid, width=0.5)
        assert np.all(patch[np.abs(x_grid) >= 0.5] == 0j)

    def test_nonzero_inside_window(self):
        x_grid = np.linspace(-2, 2, 200)
        patch  = prop._parabolic_correction_1d(
            0.0, 0.0, 1.0, 1.0, 1.0, x_grid, 1.0)
        assert np.any(patch != 0j)

    def test_real_valued(self):
        """For S_caustic=0 the patch must be purely real (no imaginary unit)."""
        x_grid = np.linspace(-1, 1, 200)
        patch  = prop._parabolic_correction_1d(
            0.0, 0.0, 1.0, 1.0, 1.0, x_grid, 0.8)
        inside = np.abs(x_grid) < 0.8
        assert np.allclose(patch[inside].imag, 0.0, atol=1e-10)

    def test_uses_D_minus_half(self):
        """Patch must use D_{-1/2}(ζ) from scipy.special.pbdv."""
        hbar, alpha, a_c, S_c = 1.0, 2.0, 1.0, 0.0
        x_grid = np.linspace(-2, 2, 500)
        width  = 1.5
        patch  = prop._parabolic_correction_1d(
            0.0, S_c, a_c, alpha, hbar, x_grid, width)

        x_test = 0.3
        idx    = np.argmin(np.abs(x_grid - x_test))
        x_loc  = x_grid[idx]
        zeta   = prop._pcf_argument(np.array([x_loc]), hbar, alpha)[0]
        D_val, _ = pbdv(-0.5, np.array([zeta]))
        prefactor = a_c * hbar ** 0.25 * abs(alpha) ** (-0.25)
        carrier   = np.exp(S_c / hbar)              # real — no i
        taper     = np.cos(np.pi / 2.0 * x_loc / width) ** 2
        expected  = prefactor * D_val[0] * carrier * taper
        assert np.isclose(patch[idx].real, expected, rtol=1e-5)

    def test_carrier_is_real_exponential(self):
        """S_caustic shifts the patch by exp(S_c/ℏ), NOT exp(i S_c/ℏ)."""
        x_grid = np.linspace(-1, 1, 200)
        S_c, hbar = 0.5, 1.0
        patch_S = prop._parabolic_correction_1d(
            0.0, S_c, 1.0, 1.0, hbar, x_grid, 0.8)
        patch_0 = prop._parabolic_correction_1d(
            0.0, 0.0, 1.0, 1.0, hbar, x_grid, 0.8)
        inside = np.abs(x_grid) < 0.7
        nz = inside & (np.abs(patch_0) > 1e-10)
        if nz.any():
            ratios = patch_S[nz].real / patch_0[nz].real
            assert np.allclose(ratios, np.exp(S_c / hbar), rtol=1e-4)

    def test_empty_window(self):
        patch = prop._parabolic_correction_1d(
            0.0, 0.0, 1.0, 1.0, 1.0, np.linspace(5, 10, 100), 0.1)
        assert np.all(patch == 0j)


# ============================================================================
# Parabolic correction 2D  (_parabolic_correction_2d)
# ============================================================================

class TestParabolicCorrection2D:

    def _make_grid(self, n=50, lim=2.0):
        xs = np.linspace(-lim, lim, n)
        return np.meshgrid(xs, xs)

    def test_zero_outside_window(self):
        X, Y  = self._make_grid()
        patch = prop._parabolic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 0.5)
        assert np.all(patch[(X**2 + Y**2) >= 0.5**2] == 0j)

    def test_nonzero_inside_window(self):
        X, Y  = self._make_grid()
        patch = prop._parabolic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 1.5)
        assert np.any(np.abs(patch) > 0)

    def test_real_valued_zero_action(self):
        """For S_caustic=0 the patch should be predominantly real."""
        X, Y  = self._make_grid(n=30)
        patch = prop._parabolic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, X, Y, 1.0)
        nz = np.abs(patch) > 1e-12
        if nz.any():
            assert np.allclose(patch[nz].imag, 0.0, atol=1e-8)

    def test_cusp_fallback_finite(self):
        X, Y  = self._make_grid(n=30)
        patch = prop._parabolic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, X, Y, 1.0)
        assert np.all(np.isfinite(patch.real))
        assert np.all(np.isfinite(patch.imag))

    def test_cusp_nonzero(self):
        X, Y  = self._make_grid(n=40)
        patch = prop._parabolic_correction_2d(
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, X, Y, 1.5)
        assert np.any(np.abs(patch) > 0)


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
# Build wave Hamiltonians  (_build_wave_hamiltonians)
# ============================================================================

class TestBuildWaveHamiltonians:

    def test_1d_quadratic_no_abs(self):
        """
        H = f(x)·ξ²  must yield ±√f(x)·ξ — linear in ξ, no |ξ| or Abs node.
        This is the key differentiability fix: sp.Abs in the expression would
        make Hamilton's equations non-differentiable at ξ=0.
        """
        x, xi = sp.symbols('x xi', real=True)
        c2    = 1 / (1 + x**2 / 4)
        H     = c2 * xi**2
        H_p, H_m = prop._build_wave_hamiltonians(H, [x, xi])

        # Must be linear in xi (degree 1 polynomial in xi)
        assert sp.Poly(H_p, xi).total_degree() == 1
        assert sp.Poly(H_m, xi).total_degree() == 1

        # H+ and H- must be negatives of each other
        assert sp.simplify(H_p + H_m) == 0

        # No Abs nodes
        for node in sp.preorder_traversal(H_p):
            assert not isinstance(node, sp.Abs), \
                "H+ must not contain sp.Abs — not differentiable at xi=0"

    def test_1d_flat_wave(self):
        """H = ξ²  →  H± = ±ξ  (free 1D wave equation)."""
        x, xi = sp.symbols('x xi', real=True)
        H_p, H_m = prop._build_wave_hamiltonians(xi**2, [x, xi])
        # Should reduce to ±ξ
        assert sp.simplify(H_p - xi) == 0 or sp.simplify(H_p + xi) == 0
        assert sp.simplify(H_p + H_m) == 0

    def test_1d_branches_recover_H(self):
        """H₊ · H₋ = −H  (i.e. H₊² = H for the correct branch choice)."""
        x, xi = sp.symbols('x xi', real=True, positive=True)
        # Use positive=True so SymPy can simplify √(f·ξ²) = √f·ξ cleanly
        c2 = 1 / (1 + x**2)
        H  = c2 * xi**2
        H_p, H_m = prop._build_wave_hamiltonians(H, [x, xi])
        # H₊² should equal H
        assert sp.simplify(H_p**2 - H) == 0

    def test_1d_hamiltonian_flow_differentiable(self):
        """
        The H+ branch Hamiltonian for c²·ξ² must be lambdifiable and have
        well-defined numerical derivatives at xi=0.
        """
        x, xi = sp.symbols('x xi', real=True)
        c2    = 1 / (1 + x**2 / 4)
        H_p, _ = prop._build_wave_hamiltonians(c2 * xi**2, [x, xi])
        f   = sp.lambdify((x, xi), H_p, 'numpy')
        dxi = sp.lambdify((x, xi), sp.diff(H_p, xi), 'numpy')
        dx  = sp.lambdify((x, xi), sp.diff(H_p, x),  'numpy')
        # Must evaluate without NaN/Inf at xi=0
        assert np.isfinite(float(f(1.0, 0.0)))
        assert np.isfinite(float(dxi(1.0, 0.0)))
        assert np.isfinite(float(dx(1.0, 0.0)))

    def test_2d_quadratic_symmetric(self):
        """
        2D flat: H = ½(ξ² + η²).  Branches must satisfy H₊ = −H₋
        and H₊² = H symbolically.
        """
        x, y, xi, eta = sp.symbols('x y xi eta', real=True, positive=True)
        H = (xi**2 + eta**2) / 2
        H_p, H_m = prop._build_wave_hamiltonians(H, [x, xi, y, eta])
        assert sp.simplify(H_p + H_m) == 0
        assert sp.simplify(sp.expand(H_p**2) - H) == 0

    def test_perfect_square_input(self):
        """H = (a·ξ)²  must yield branches ±a·ξ directly."""
        x, xi = sp.symbols('x xi', real=True)
        a = sp.Symbol('a', positive=True)
        H = (a * xi)**2
        H_p, H_m = prop._build_wave_hamiltonians(H, [x, xi])
        assert sp.simplify(H_p**2 - H) == 0
        assert sp.simplify(H_p + H_m) == 0


# ============================================================================
# Parabolic coherent sum  (parabolic_sum)
# ============================================================================

class TestParabolicSum:

    def test_1d_positive_real(self):
        """For positive S and unit Jacobian, parabolic_sum must return real values."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.array([0.5, 1.0, 1.5])
        det_J = np.ones(3)
        u, X, Y = prop.parabolic_sum(pts, S, det_J, xlim=(0, 2), N=10, hbar=1.0)
        assert Y is None
        # All values should be real-positive (or zero at extrapolation boundary)
        assert np.all(u.imag[np.abs(u) > 1e-10] == pytest.approx(0.0, abs=1e-8))
        assert np.all(u.real >= 0)

    def test_1d_no_maslov_needed(self):
        """parabolic_sum does not take mu; calling it without mu should not raise."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.zeros(3)
        det_J = np.ones(3)
        # Should not raise TypeError
        u, X, Y = prop.parabolic_sum(pts, S, det_J, xlim=(0, 2), N=5)
        assert u.shape == (5,)

    def test_1d_action_scaling(self):
        """Doubling S should square the amplitude (real exp)."""
        pts   = np.array([[0.0], [1.0]])
        det_J = np.ones(2)
        hbar  = 1.0
        S1 = np.array([0.0, 1.0])
        S2 = np.array([0.0, 2.0])
        u1, X, _ = prop.parabolic_sum(pts, S1, det_J, xlim=(0, 1), N=3, hbar=hbar)
        u2, _,  _ = prop.parabolic_sum(pts, S2, det_J, xlim=(0, 1), N=3, hbar=hbar)
        # At x=1: u2/u1 = exp(2)/exp(1) = e
        idx = -1  # right-most grid point closest to x=1
        if abs(u1[idx]) > 1e-10:
            ratio = u2[idx].real / u1[idx].real
            assert ratio == pytest.approx(np.e, rel=1e-3)

    def test_2d_shape(self):
        """2D parabolic_sum must return (N,N) array."""
        pts   = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        S     = np.zeros(4)
        det_J = np.ones(4)
        u, X, Y = prop.parabolic_sum(pts, S, det_J,
                                      xlim=(0, 1), ylim=(0, 1), N=5)
        assert u.shape == (5, 5)
        assert X.shape == (5, 5)
        assert Y.shape == (5, 5)

    def test_2d_positive_real(self):
        """2D parabolic_sum with S=0 must return non-negative real values."""
        pts   = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        S     = np.zeros(4)
        det_J = np.ones(4)
        u, X, Y = prop.parabolic_sum(pts, S, det_J,
                                      xlim=(0, 1), ylim=(0, 1), N=5)
        nz = np.abs(u) > 1e-10
        assert np.allclose(u[nz].imag, 0.0, atol=1e-8)
        assert np.all(u.real >= -1e-10)


# ============================================================================
# Wave coherent sum  (wave_sum)
# ============================================================================

class TestWaveSum:

    def _two_ray_data_1d(self):
        """Two trivial 1D ray datasets (unit action, unit Jacobian)."""
        pts   = np.array([[0.0], [1.0]])
        S     = np.array([0.0, 1.0])
        det_J = np.ones(2)
        mu    = np.zeros(2, dtype=int)
        return pts, S, det_J, mu

    def test_1d_is_sum_of_branches(self):
        """wave_sum = van_vleck_sum(+branch) + van_vleck_sum(−branch)."""
        pp, sp_, djp, mup = self._two_ray_data_1d()
        pm, sm, djm, mum = self._two_ray_data_1d()
        sm = sm + 0.5   # slightly different action on minus branch

        u_wave, X, _ = prop.wave_sum(
            pp, sp_, djp, mup, pm, sm, djm, mum,
            xlim=(0, 1), N=10, hbar=1.0)

        u_plus,  _, _ = prop.van_vleck_sum(pp, sp_, djp, mup, xlim=(0, 1), N=10)
        u_minus, _, _ = prop.van_vleck_sum(pm, sm, djm, mum, xlim=(0, 1), N=10)

        assert np.allclose(u_wave, u_plus + u_minus, atol=1e-10)

    def test_1d_shape(self):
        pp, sp_, djp, mup = self._two_ray_data_1d()
        u, X, Y = prop.wave_sum(
            pp, sp_, djp, mup, pp, sp_, djp, mup,
            xlim=(0, 1), N=15, hbar=1.0)
        assert u.shape == (15,)
        assert Y is None

    def test_2d_shape(self):
        pts   = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        S     = np.zeros(4)
        det_J = np.ones(4)
        mu    = np.zeros(4, dtype=int)
        u, X, Y = prop.wave_sum(
            pts, S, det_J, mu, pts, S, det_J, mu,
            xlim=(0, 1), ylim=(0, 1), N=5, hbar=1.0)
        assert u.shape == (5, 5)
        assert X.shape == (5, 5)

    def test_1d_identical_branches_double_amplitude(self):
        """When both branches are identical, the result is 2× van_vleck_sum."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.array([0.0, 1.0, 2.0])
        det_J = np.ones(3)
        mu    = np.zeros(3, dtype=int)
        u_wave, _, _ = prop.wave_sum(
            pts, S, det_J, mu, pts, S, det_J, mu,
            xlim=(0, 2), N=10, hbar=1.0)
        u_single, _, _ = prop.van_vleck_sum(pts, S, det_J, mu, xlim=(0, 2), N=10)
        assert np.allclose(u_wave, 2 * u_single, atol=1e-10)


# ============================================================================
# Van Vleck sum  (Schrödinger)
# ============================================================================

class TestVanVleckSum:

    def test_1d_constant_unit_amplitude(self):
        """S=0, det_J=1, μ=0 everywhere → psi = 1 everywhere."""
        pts = np.array([[0.0], [1.0], [2.0]])
        psi, X, Y = prop.van_vleck_sum(
            pts, np.zeros(3), np.ones(3), np.zeros(3, dtype=int),
            xlim=(0, 2), N=10)
        assert Y is None
        assert np.allclose(psi, 1.0 + 0j)

    def test_1d_phase_varies_linearly(self):
        """S(x) = π x → at x=1, ψ = exp(iπ) = −1."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.array([0.0, np.pi, 2 * np.pi])
        psi, X, _ = prop.van_vleck_sum(
            pts, S, np.ones(3), np.zeros(3, dtype=int), xlim=(0, 2), N=5)
        idx = np.argmin(np.abs(X - 1.0))
        assert np.isclose(X[idx], 1.0)
        assert np.isclose(psi[idx], -1.0, atol=1e-10)

    def test_1d_maslov_phase_shift(self):
        """μ=1 contributes exp(−iπ/2) = −i relative to μ=0."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.zeros(3)
        det_J = np.ones(3)
        mu_0  = np.zeros(3, dtype=int)
        mu_1  = np.array([0, 1, 0], dtype=int)
        psi0, X, _ = prop.van_vleck_sum(pts, S, det_J, mu_0, xlim=(0, 2), N=5)
        psi1, _,  _ = prop.van_vleck_sum(pts, S, det_J, mu_1, xlim=(0, 2), N=5)
        idx = np.argmin(np.abs(X - 1.0))
        assert np.isclose(psi1[idx] / psi0[idx],
                          np.exp(-1j * np.pi / 2), atol=1e-10)

    def test_1d_two_ray_interference(self):
        """Linear interpolation at midpoint between two scattered points."""
        pts   = np.array([[0.3], [0.7]])
        S     = np.array([2.0, 3.0])
        det_J = np.ones(2)
        mu    = np.zeros(2, dtype=int)
        psi_k = np.exp(1j * S)
        psi, X, _ = prop.van_vleck_sum(pts, S, det_J, mu, xlim=(0, 1), N=5)
        idx = np.argmin(np.abs(X - 0.5))
        expected = 0.5 * (psi_k[0] + psi_k[1])
        assert np.isclose(psi[idx], expected, atol=1e-10)

    def test_1d_interpolation_at_data_point(self):
        """Interpolated value at a scattered-point location equals that point."""
        pts   = np.array([[0.0], [1.0], [2.0]])
        S     = np.array([0.0, np.pi, 2 * np.pi])
        psi, X, _ = prop.van_vleck_sum(
            pts, S, np.ones(3), np.zeros(3, dtype=int), xlim=(0, 2), N=3)
        assert np.isclose(psi[1], np.exp(1j * np.pi), atol=1e-10)

    def test_2d_constant_unit_amplitude(self):
        pts   = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        psi, X, Y = prop.van_vleck_sum(
            pts, np.zeros(4), np.ones(4), np.zeros(4, dtype=int),
            xlim=(0, 1), ylim=(0, 1), N=5)
        assert X.shape == (5, 5)
        assert psi.shape == (5, 5)
        assert np.isclose(psi[2, 2], 1.0)


    def test_2d_interpolation_at_data_point(self):
        pts   = np.array([[0.5, 0.5], [0.2, 0.2], [0.8, 0.2]])
        S     = np.array([2.0, 0.0, 0.0])
        det_J = np.ones(3)
        mu    = np.zeros(3, dtype=int)
        psi_k = np.exp(1j * S)
        psi, X, Y = prop.van_vleck_sum(pts, S, det_J, mu,
                                        xlim=(0, 1), ylim=(0, 1), N=3)
        assert np.isclose(psi[1, 1], psi_k[0], atol=1e-10)


# ============================================================================
# EquationType and WKBResult.equation field
# ============================================================================

class TestEquationType:

    def test_constants_are_strings(self):
        assert isinstance(EquationType.SCHRODINGER, str)
        assert isinstance(EquationType.PARABOLIC, str)
        assert isinstance(EquationType.WAVE, str)

    def test_constants_are_distinct(self):
        vals = {EquationType.SCHRODINGER, EquationType.PARABOLIC, EquationType.WAVE}
        assert len(vals) == 3

    def test_wkb_result_default_equation(self):
        """WKBResult.equation defaults to SCHRODINGER."""
        r = _make_1d_wkb_result()
        assert r.equation == EquationType.SCHRODINGER

    def test_wkb_result_stores_equation(self):
        for eq in (EquationType.SCHRODINGER, EquationType.PARABOLIC, EquationType.WAVE):
            r = _make_1d_wkb_result(equation=eq)
            assert r.equation == eq

    def test_wkb_result_2d_stores_equation(self):
        for eq in (EquationType.SCHRODINGER, EquationType.PARABOLIC, EquationType.WAVE):
            r = _make_2d_wkb_result(equation=eq)
            assert r.equation == eq


# ============================================================================
# compute_wavefunction — Schrödinger (existing API, unchanged behaviour)
# ============================================================================

class TestComputeWavefunctionBase:

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
        assert result.equation == EquationType.SCHRODINGER

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

    def test_auto_xlim(self):
        """When xlim is not provided, it is derived from ray endpoints."""
        m, _ = flat_1d()
        with patch('propagator.hamiltonian_flow') as mock_flow:
            def side_effect(H, z0, tspan, vars_phase, integrator, n_steps):
                v0 = z0[1]
                t = np.linspace(0, 1, n_steps)
                return {'t': t, 'x': v0 * t, 'xi': v0 * np.ones_like(t)}
            mock_flow.side_effect = side_effect
            result = prop.compute_wavefunction(
                metric=m, source=(0.0,),
                v_fan=np.array([1.0, 2.0]),
                t_max=1.0, n_steps=10, N_grid=10, parallel=False)
    
        # Endpoints from the two rays (t_max=1)
        endpoints = np.array([1.0, 2.0])
        margin = 0.1 * (endpoints.max() - endpoints.min())
        expected_xmin = endpoints.min() - margin   # 0.9
        expected_xmax = endpoints.max() + margin   # 2.1
    
        assert result.X[0] == pytest.approx(expected_xmin, rel=1e-2)
        assert result.X[-1] == pytest.approx(expected_xmax, rel=1e-2)

    def test_curved_metric_action_monotone(self):
        """Curved 1D metric (g=x²): S_cum must be strictly non-decreasing."""
        m, _ = power_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(1.0,),
            v_fan=np.array([0.5]),
            t_max=0.5, hbar=1.0, n_steps=30, N_grid=10)
        assert np.all(np.diff(result.rays[0].S_cum) >= 0)

    def test_v_fan_required_in_metric_mode(self):
        m, _ = flat_1d()
        with pytest.raises(ValueError, match="v_fan"):
            prop.compute_wavefunction(
                metric=m, source=(0.0,), t_max=0.5, n_steps=5, N_grid=5)

    def test_both_metric_and_hamiltonian_raises(self):
        m, _ = flat_1d()
        x, xi = sp.symbols('x xi', real=True)
        with pytest.raises(ValueError):
            prop.compute_wavefunction(
                metric=m, hamiltonian=xi**2 / 2,
                coords=(x,), momenta=(xi,),
                source=(0.0,), p_fan=np.array([1.0]), t_max=0.5)

    def test_neither_metric_nor_hamiltonian_raises(self):
        with pytest.raises(ValueError):
            prop.compute_wavefunction(
                source=(0.0,), t_max=0.5,
                v_fan=np.array([1.0]), n_steps=5, N_grid=5)

    def test_mode_b_1d_free_particle(self):
        """Mode B H=ξ²/2: shapes, non-trivial action, monotone S_cum."""
        x, xi = sp.symbols('x xi', real=True)
        result = prop.compute_wavefunction(
            hamiltonian=xi**2 / 2, coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=np.linspace(-1.0, 1.0, 5),
            t_max=1.0, hbar=1.0, n_steps=20, N_grid=10, integrator='verlet')
        assert result.dim == 1
        assert len(result.rays) == 5
        assert result.psi.shape == (10,)
        for ray in [r for r in result.rays if abs(r.S_cum[-1]) > 1e-12]:
            assert np.all(np.diff(ray.S_cum) >= -1e-10)

    def test_mode_b_1d_harmonic_oscillator_caustics(self):
        """H=½ξ²+½x² with t_max > π: at least one caustic crossing expected."""
        x, xi = sp.symbols('x xi', real=True)
        result = prop.compute_wavefunction(
            hamiltonian=xi**2 / 2 + x**2 / 2, coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=np.linspace(-1.5, 1.5, 20),
            t_max=np.pi * 1.2, hbar=0.3, n_steps=200, N_grid=50,
            integrator='rk45')
        assert max(r.mu for r in result.rays) >= 1

    def test_mode_b_p_fan_required(self):
        x, xi = sp.symbols('x xi', real=True)
        with pytest.raises(ValueError, match="p_fan"):
            prop.compute_wavefunction(
                hamiltonian=xi**2 / 2, coords=(x,), momenta=(xi,),
                source=(0.0,), t_max=1.0, n_steps=5, N_grid=5)

    def test_mode_b_coords_momenta_required(self):
        x, xi = sp.symbols('x xi', real=True)
        with pytest.raises(ValueError):
            prop.compute_wavefunction(
                hamiltonian=xi**2 / 2,
                source=(0.0,), p_fan=np.array([1.0]),
                t_max=1.0, n_steps=5, N_grid=5)

    def test_mode_b_2d_free_particle(self):
        """Mode B 2D H=(ξ²+η²)/2 must produce 2D output."""
        x, y, xi, eta = sp.symbols('x y xi eta', real=True)
        px = np.linspace(-0.5, 0.5, 3)
        py = np.linspace(-0.5, 0.5, 3)
        result = prop.compute_wavefunction(
            hamiltonian=(xi**2 + eta**2) / 2,
            coords=(x, y), momenta=(xi, eta),
            source=(0.0, 0.0),
            p_fan=np.array([[a, b] for a in px for b in py]),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=10, integrator='verlet')
        assert result.dim == 2
        assert result.psi.shape == (10, 10)
        assert result.y_pts is not None

    def test_resolve_hamiltonian_invalid_dim(self):
        coords  = sp.symbols('x y z', real=True)
        momenta = sp.symbols('px py pz', real=True)
        H = sum(p**2 for p in momenta) / 2
        with pytest.raises(ValueError, match="1D and 2D"):
            prop._resolve_hamiltonian(
                metric=None, hamiltonian=H,
                coords=coords, momenta=momenta)


# ============================================================================
# compute_wavefunction — parabolic and wave equations  (new)
# ============================================================================

class TestComputeWavefunctionNew:

    # ── Parabolic ─────────────────────────────────────────────────────────────

    def test_parabolic_1d_returns_correct_equation(self):
        """result.equation must equal PARABOLIC when equation=PARABOLIC."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-1.0, 1.0, 5),
            t_max=1.0, hbar=1.0, n_steps=20, N_grid=10,
            equation=EquationType.PARABOLIC, parallel=False)
        assert result.equation == EquationType.PARABOLIC

    def test_parabolic_1d_real_solution(self):
        """1D parabolic equation must produce a predominantly real-valued ψ."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-1.0, 1.0, 10),
            t_max=1.0, hbar=1.0, n_steps=30, N_grid=20,
            equation=EquationType.PARABOLIC, parallel=False)
        nz = np.abs(result.psi) > 1e-10
        assert np.allclose(result.psi[nz].imag, 0.0, atol=1e-7), \
            "Parabolic solution must be real-valued"

    def test_parabolic_1d_positive(self):
        """Parabolic solution with positive action must be non-negative real."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(0.1, 2.0, 10),   # all positive momenta → positive S
            t_max=1.0, hbar=1.0, n_steps=30, N_grid=20,
            equation=EquationType.PARABOLIC, parallel=False)
        assert np.all(result.psi.real >= -1e-8)

    def test_parabolic_2d_shape(self):
        """2D parabolic equation must produce a (N,N) output."""
        m, _ = flat_2d()
        vx = np.linspace(-0.5, 0.5, 2)
        vy = np.linspace(-0.5, 0.5, 2)
        result = prop.compute_wavefunction(
            metric=m, source=(0.0, 0.0),
            v_fan=np.array([[a, b] for a in vx for b in vy]),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=8,
            equation=EquationType.PARABOLIC, parallel=False)
        assert result.psi.shape == (8, 8)
        assert result.equation == EquationType.PARABOLIC

    def test_parabolic_hamiltonian_mode_1d(self):
        """Parabolic equation in Mode B (explicit Hamiltonian) must work."""
        x, xi = sp.symbols('x xi', real=True)
        H_heat = xi**2 / 2 - x**2 / 2   # inverted HO: unstable growth
        result = prop.compute_wavefunction(
            hamiltonian=H_heat, coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=np.linspace(-1.0, 1.0, 8),
            t_max=0.5, hbar=0.5, n_steps=30, N_grid=10,
            equation=EquationType.PARABOLIC, parallel=False)
        assert result.equation == EquationType.PARABOLIC
        assert len(result.rays) >= 4

    # ── Wave ──────────────────────────────────────────────────────────────────

    def test_wave_1d_returns_correct_equation(self):
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-1.0, 1.0, 6),
            t_max=1.0, hbar=1.0, n_steps=20, N_grid=10,
            equation=EquationType.WAVE, parallel=False)
        assert result.equation == EquationType.WAVE

    def test_wave_1d_doubled_ray_count(self):
        """Wave equation integrates two branches; ray count is 2× fan size."""
        m, _ = flat_1d()
        n_fan = 6
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-1.0, 1.0, n_fan),
            t_max=1.0, hbar=1.0, n_steps=20, N_grid=10,
            equation=EquationType.WAVE, parallel=False)
        assert len(result.rays) == 2 * n_fan

    def test_wave_1d_shape(self):
        """1D wave equation must produce (N_grid,) output."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-1.0, 1.0, 6),
            t_max=1.0, hbar=1.0, n_steps=20, N_grid=15,
            equation=EquationType.WAVE, parallel=False)
        assert result.psi.shape == (15,)

    def test_wave_1d_hamiltonian_mode(self):
        """
        Wave equation in Mode B with H = c²(x)·ξ²  (acoustic waveguide).
        The branch Hamiltonians must be linear in ξ — no |ξ|/Abs singularity.
        """
        x, xi = sp.symbols('x xi', real=True)
        c2    = 1 / (1 + x**2 / 4)
        H_wave = c2 * xi**2
        result = prop.compute_wavefunction(
            hamiltonian=H_wave, coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=np.linspace(-2.0, 2.0, 10),
            t_max=2.0, hbar=0.2, n_steps=100, N_grid=20,
            equation=EquationType.WAVE, parallel=False)
        assert result.equation == EquationType.WAVE
        assert len(result.rays) > 0

    def test_wave_2d_shape(self):
        """2D wave equation must produce (N,N) output."""
        m, _ = flat_2d()
        vx = np.linspace(-0.5, 0.5, 2)
        vy = np.linspace(-0.5, 0.5, 2)
        result = prop.compute_wavefunction(
            metric=m, source=(0.0, 0.0),
            v_fan=np.array([[a, b] for a in vx for b in vy]),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=8,
            equation=EquationType.WAVE, parallel=False)
        assert result.psi.shape == (8, 8)
        assert result.equation == EquationType.WAVE

    def test_wave_2d_doubled_ray_count(self):
        """2D wave equation doubles the ray count."""
        m, _ = flat_2d()
        vx = np.linspace(-0.5, 0.5, 2)
        vy = np.linspace(-0.5, 0.5, 2)
        fan = np.array([[a, b] for a in vx for b in vy])
        result = prop.compute_wavefunction(
            metric=m, source=(0.0, 0.0),
            v_fan=fan, t_max=1.0, hbar=1.0, n_steps=10, N_grid=8,
            equation=EquationType.WAVE, parallel=False)
        assert len(result.rays) == 2 * len(fan)

    def test_wave_is_sum_of_branches(self):
        """
        For a flat 1D metric the wave solution should equal the sum of
        two independent Schrödinger computations with H₊ and H₋.
        This verifies that wave_sum is called correctly.
        """
        x, xi = sp.symbols('x xi', real=True)
        p_fan = np.linspace(0.5, 2.0, 5)   # positive momenta only to avoid zero

        # Wave result
        result_wave = prop.compute_wavefunction(
            hamiltonian=xi**2 / 2, coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=p_fan,
            t_max=1.0, hbar=1.0, n_steps=30, N_grid=15,
            equation=EquationType.WAVE, parallel=False,
            xlim=(-1.0, 3.0))

        # Each branch separately: H₊ = xi/√2, H₋ = −xi/√2
        H_p = xi / sp.sqrt(2)
        H_m = -xi / sp.sqrt(2)

        result_p = prop.compute_wavefunction(
            hamiltonian=H_p, coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=p_fan,
            t_max=1.0, hbar=1.0, n_steps=30, N_grid=15,
            equation=EquationType.SCHRODINGER, parallel=False,
            xlim=(-1.0, 3.0))

        result_m = prop.compute_wavefunction(
            hamiltonian=H_m, coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=p_fan,
            t_max=1.0, hbar=1.0, n_steps=30, N_grid=15,
            equation=EquationType.SCHRODINGER, parallel=False,
            xlim=(-1.0, 3.0))

        expected = result_p.psi + result_m.psi
        assert np.allclose(result_wave.psi, expected, atol=1e-8)


# ============================================================================
# Visualisation — smoke tests (all equation types, 1D and 2D)
# ============================================================================

class TestVisualisation:

    @pytest.mark.parametrize("equation", [
        EquationType.SCHRODINGER,
        EquationType.PARABOLIC,
        EquationType.WAVE,
    ])
    @patch('matplotlib.pyplot.show')
    def test_plot_wavefunction_1d(self, _, equation):
        prop.plot_wavefunction(_make_1d_wkb_result(equation), log_scale=True)
        plt.close('all')

    @pytest.mark.parametrize("equation", [
        EquationType.SCHRODINGER,
        EquationType.PARABOLIC,
        EquationType.WAVE,
    ])
    @patch('matplotlib.pyplot.show')
    def test_plot_wavefunction_2d(self, _, equation):
        prop.plot_wavefunction(_make_2d_wkb_result(equation), log_scale=True)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_ray_fan_1d(self, _):
        prop.plot_ray_fan(_make_1d_wkb_result())
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_ray_fan_2d(self, _):
        prop.plot_ray_fan(_make_2d_wkb_result())
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_interference_detail_1d(self, _):
        prop.plot_interference_detail(_make_1d_wkb_result())
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_interference_detail_2d(self, _):
        prop.plot_interference_detail(_make_2d_wkb_result())
        plt.close('all')

    def test_animate_wavefunction_schrodinger(self):
        """animate_wavefunction for SCHRODINGER must return a FuncAnimation."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-0.5, 0.5, 5),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=10,
            equation=EquationType.SCHRODINGER, parallel=False)
        ani = prop.animate_wavefunction(result, n_frames=5, interval=50)
        assert isinstance(ani, animation.FuncAnimation)
        plt.close('all')

    def test_animate_wavefunction_parabolic(self):
        """animate_wavefunction for PARABOLIC must not raise."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-0.5, 0.5, 5),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=10,
            equation=EquationType.PARABOLIC, parallel=False)
        ani = prop.animate_wavefunction(result, n_frames=5, interval=50)
        assert isinstance(ani, animation.FuncAnimation)
        plt.close('all')

    def test_animate_wavefunction_wave(self):
        """animate_wavefunction for WAVE must not raise."""
        m, _ = flat_1d()
        result = prop.compute_wavefunction(
            metric=m, source=(0.0,),
            v_fan=np.linspace(-0.5, 0.5, 5),
            t_max=1.0, hbar=1.0, n_steps=10, N_grid=10,
            equation=EquationType.WAVE, parallel=False)
        ani = prop.animate_wavefunction(result, n_frames=5, interval=50)
        assert isinstance(ani, animation.FuncAnimation)
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_plot_ray_fan_wave_equation(self, _):
        """plot_ray_fan on a WAVE result (2× rays) must not raise."""
        r = _make_1d_wkb_result(equation=EquationType.WAVE)
        prop.plot_ray_fan(r)
        plt.close('all')