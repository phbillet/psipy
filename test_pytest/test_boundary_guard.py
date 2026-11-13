"""
test_boundary_guard.py — Tests for the boundary-condition handling of PDESolver
==============================================================================

Covers the refactored `_apply_boundary` and the `boundary_tol` runtime guard:

1. `_apply_boundary` semantics (periodic no-op, Dirichlet, Neumann, invalid BC)
2. Guard behaviour (no warning for localized data, warning once otherwise,
   `boundary_tol` configurable, `_bc_max_correction` tracking)
3. Large-box equivalence: Dirichlet/Neumann clipping == periodic solution for
   compactly supported data (the regime where the heuristic is valid)
4. Regression: periodic order-1 evolution must not be corrupted by clipping
"""

import warnings

import pytest
import numpy as np
from sympy import symbols, Function, Eq, diff, sin

from solver import PDESolver, psiOp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diffusion_1d(bc, ic, Lx=2 * np.pi, Nx=64, Lt=0.5, Nt=100, **setup_kw):
    """u_t = -Op(xi^2 + 1) u in 1D, set up (not solved)."""
    t, x, xi = symbols('t x xi', real=True)
    u = Function('u')
    eq = Eq(diff(u(t, x), t), -psiOp(xi**2 + 1, u(t, x)))
    solver = PDESolver(eq)
    solver.setup(Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
                 boundary_condition=bc, initial_condition=ic,
                 n_frames=5, plot=False, **setup_kw)
    return solver


def _stationary_solver(dim, bc):
    """Cheap solver used only to exercise `_apply_boundary` directly."""
    if dim == 1:
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=16, boundary_condition=bc,
                     initial_condition=None, plot=False)
    else:
        x, y, xi, eta = symbols('x y xi eta', real=True)
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -x * y)
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16, Ny=16,
                     boundary_condition=bc, initial_condition=None,
                     plot=False)
    # Start every test with a clean guard state.
    solver._bc_max_correction = 0.0
    solver._bc_warned = False
    return solver


def _gaussian(x, x0=0.0, w=1.0):
    return np.exp(-((x - x0) / w) ** 2)


def _no_runtime_warning():
    """Context manager turning RuntimeWarning into an error."""
    cm = warnings.catch_warnings()
    cm.__enter__()
    warnings.simplefilter("error", RuntimeWarning)
    return cm


# ===========================================================================
# 1.  _apply_boundary semantics
# ===========================================================================

class TestApplyBoundary:

    @pytest.mark.parametrize("dim", [1, 2])
    def test_periodic_is_noop(self, dim):
        solver = _stationary_solver(dim, 'periodic')
        rng = np.random.default_rng(0)
        shape = (16,) if dim == 1 else (16, 16)
        u = rng.standard_normal(shape)
        u0 = u.copy()
        solver._apply_boundary(u)
        np.testing.assert_array_equal(u, u0)

    def test_periodic_never_touches_guard_state(self):
        solver = _stationary_solver(1, 'periodic')
        solver._apply_boundary(np.arange(16, dtype=float))
        assert solver._bc_max_correction == 0.0
        assert solver._bc_warned is False

    def test_dirichlet_1d_zeros_node0_only(self):
        solver = _stationary_solver(1, 'nonperiodic')
        u = np.arange(1.0, 17.0)
        ref = u.copy()
        with pytest.warns(RuntimeWarning):
            solver._apply_boundary(u)
        # endpoint=False grid: node 0 is both walls; the last node is free.
        assert u[0] == 0
        np.testing.assert_array_equal(u[1:], ref[1:])

    def test_dirichlet_2d_zeros_row0_and_col0_only(self):
        solver = _stationary_solver(2, 'nonperiodic')
        u = np.ones((16, 16))
        with pytest.warns(RuntimeWarning):
            solver._apply_boundary(u)
        assert np.all(u[0, :] == 0) and np.all(u[:, 0] == 0)
        assert np.all(u[1:, 1:] == 1)

    def test_neumann_1d_copies_inner_neighbour(self):
        solver = _stationary_solver(1, 'neumann')
        u = np.arange(1.0, 17.0)
        with pytest.warns(RuntimeWarning):
            solver._apply_boundary(u)
        assert u[0] == u[1]
        assert u[-1] == u[-2]

    def test_neumann_2d_copies_inner_neighbour(self):
        solver = _stationary_solver(2, 'neumann')
        rng = np.random.default_rng(1)
        u = rng.standard_normal((16, 16))
        with pytest.warns(RuntimeWarning):
            solver._apply_boundary(u)
        np.testing.assert_array_equal(u[0, :], u[1, :])
        np.testing.assert_array_equal(u[-1, :], u[-2, :])
        np.testing.assert_array_equal(u[:, 0], u[:, 1])
        np.testing.assert_array_equal(u[:, -1], u[:, -2])

    def test_neumann_is_not_dirichlet(self):
        """Regression: Neumann used to be documented as 'sets to zero'."""
        solver = _stationary_solver(1, 'neumann')
        u = np.full(16, 3.0)
        solver._apply_boundary(u)
        assert u[0] == 3.0 and u[-1] == 3.0

    def test_invalid_bc_raises(self):
        solver = _stationary_solver(1, 'nonperiodic')
        solver.boundary_condition = 'robin'
        with pytest.raises(ValueError, match="robin"):
            solver._apply_boundary(np.zeros(16))

    def test_complex_field_supported(self):
        solver = _stationary_solver(1, 'nonperiodic')
        u = (1.0 + 1.0j) * np.ones(16)
        with pytest.warns(RuntimeWarning):
            solver._apply_boundary(u)
        assert u[0] == 0 and u[-1] == 1.0 + 1.0j


# ===========================================================================
# 2.  Guard behaviour
# ===========================================================================

class TestBoundaryGuard:

    def test_default_tolerance(self):
        solver = _diffusion_1d('nonperiodic', lambda x: _gaussian(x),
                               Lx=40.0, Nx=128)
        assert solver.boundary_tol == pytest.approx(1e-8)

    def test_localized_data_no_warning_dirichlet(self):
        cm = _no_runtime_warning()
        try:
            solver = _diffusion_1d('nonperiodic', lambda x: _gaussian(x),
                                   Lx=40.0, Nx=256)
            solver.solve()
        finally:
            cm.__exit__(None, None, None)
        assert solver._bc_max_correction < solver.boundary_tol
        assert solver._bc_warned is False

    def test_localized_data_no_warning_neumann(self):
        cm = _no_runtime_warning()
        try:
            solver = _diffusion_1d('neumann', lambda x: _gaussian(x),
                                   Lx=40.0, Nx=256)
            solver.solve()
        finally:
            cm.__exit__(None, None, None)
        assert solver._bc_max_correction < solver.boundary_tol

    def test_nonzero_boundary_warns_at_setup(self):
        """sin(x+0.3) is non-zero at the left edge: clipping must be flagged."""
        with pytest.warns(RuntimeWarning, match="boundary"):
            solver = _diffusion_1d('nonperiodic', lambda x: np.sin(x + 0.3))
        assert solver._bc_warned is True
        assert solver._bc_max_correction > solver.boundary_tol

    def test_warning_emitted_only_once(self):
        solver = _stationary_solver(1, 'nonperiodic')
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in range(5):
                solver._apply_boundary(np.ones(16))
        n = sum(issubclass(w.category, RuntimeWarning) for w in rec)
        assert n == 1

    def test_max_correction_is_monotone(self):
        solver = _stationary_solver(1, 'nonperiodic')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            big = np.zeros(16)
            big[0] = 1.0
            big[8] = 1.0          # relative correction = 1
            solver._apply_boundary(big)
            c1 = solver._bc_max_correction
            small = np.zeros(16)
            small[0] = 1e-3
            small[8] = 1.0        # relative correction = 1e-3
            solver._apply_boundary(small)
        assert c1 == pytest.approx(1.0)
        assert solver._bc_max_correction == pytest.approx(c1)

    def test_relative_scale(self):
        """Correction is relative to max|u|, so an amplitude rescaling of the
        whole field must not change the reported value."""
        results = []
        for amp in (1.0, 1e6):
            solver = _stationary_solver(1, 'nonperiodic')
            u = np.zeros(16)
            u[0] = 1e-4 * amp
            u[8] = amp
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                solver._apply_boundary(u)
            results.append(solver._bc_max_correction)
        assert results[0] == pytest.approx(results[1])
        assert results[0] == pytest.approx(1e-4)

    def test_large_tol_suppresses_warning(self):
        cm = _no_runtime_warning()
        try:
            solver = _diffusion_1d('nonperiodic', lambda x: np.sin(x + 0.3),
                                   boundary_tol=10.0)
        finally:
            cm.__exit__(None, None, None)
        assert solver.boundary_tol == 10.0
        assert solver._bc_warned is False

    def test_gaussian_tail_above_tol_warns(self):
        """A Gaussian whose value at the edge is ~1e-6 (> default 1e-8)
        must be flagged: the box is too small for the large-box regime."""
        ic = lambda x: _gaussian(x, w=1.0)
        L = 2 * np.sqrt(-np.log(1e-6))         # edge at |x| = L/2 -> u ~ 1e-6
        with pytest.warns(RuntimeWarning):
            _diffusion_1d('nonperiodic', ic, Lx=L, Nx=128)

    def test_periodic_never_warns(self):
        cm = _no_runtime_warning()
        try:
            solver = _diffusion_1d('periodic', lambda x: np.sin(x + 0.3))
            solver.solve()
        finally:
            cm.__exit__(None, None, None)
        assert solver._bc_max_correction == 0.0


# ===========================================================================
# 3.  Large-box equivalence (regime where the heuristic is valid)
# ===========================================================================

class TestLargeBoxEquivalence:

    @pytest.mark.parametrize("bc", ["periodic", "nonperiodic"])
    def test_clipping_equals_periodic_for_gaussian(self, bc):
        ic = lambda x: _gaussian(x, w=1.0)
        kw = dict(Lx=40.0, Nx=256, Lt=1.0, Nt=200)

        ref = _diffusion_1d('periodic', ic, **kw)
        ref.solve()

        cm = _no_runtime_warning()
        try:
            sol = _diffusion_1d(bc, ic, **kw)
            sol.solve()
        finally:
            cm.__exit__(None, None, None)

        np.testing.assert_allclose(np.real(sol.u_prev), np.real(ref.u_prev),
                                   atol=1e-10)

    def test_dirichlet_gaussian_matches_exact_heat_kernel(self):
        """u_t = -(xi^2+1)u  =>  u(x,t) = e^{-t} * heat-kernel(Gaussian)."""
        w, T = 1.0, 0.5
        kw = dict(Lx=40.0, Nx=512, Lt=T, Nt=200)
        sol = _diffusion_1d('nonperiodic', lambda x: _gaussian(x, w=w), **kw)
        sol.solve()
        # exp(-(x/w)^2) evolved by u_t = u_xx : width^2 -> w^2 + 4T
        w2 = w**2 + 4 * T
        exact = np.exp(-T) * (w / np.sqrt(w2)) * np.exp(-sol.X**2 / w2)
        np.testing.assert_allclose(np.real(sol.u_prev), exact, atol=1e-6)


# ===========================================================================
# 4.  Regression: periodic evolution must not be clipped
# ===========================================================================

class TestPeriodicRegression:

    def test_periodic_decay_with_nonzero_edge_value(self):
        """sin(x+0.3) has u[0] != u[-2]; the old periodic 'copy' clipping
        (u[0]=u[-2], u[-1]=u[1]) would corrupt it at O(h)."""
        phase, T = 0.3, 1.0
        solver = _diffusion_1d('periodic', lambda x: np.sin(x + phase),
                               Nx=64, Lt=T, Nt=200)
        solver.solve()
        exact = np.sin(solver.X + phase) * np.exp(-2.0 * T)   # k0=1 -> k0^2+1
        np.testing.assert_allclose(np.real(solver.u_prev), exact, atol=1e-6)

    def test_periodic_initial_data_untouched(self):
        ic = lambda x: np.sin(x + 0.3)
        solver = _diffusion_1d('periodic', ic, Nx=64)
        np.testing.assert_allclose(np.real(solver.u_prev), ic(solver.X),
                                   atol=0, rtol=0)


# ===========================================================================
# 5.  'nonperiodic' mode and deprecated aliases
# ===========================================================================

class TestNonperiodicMode:

    def test_nonperiodic_accepted_and_clips_to_zero(self):
        solver = _diffusion_1d('nonperiodic', lambda x: _gaussian(x), Lx=40.0, Nx=128)
        assert solver.boundary_condition == 'nonperiodic'
        u = np.ones(128)
        with pytest.warns(RuntimeWarning):
            solver._apply_boundary(u)
        # node 0 carries both walls (endpoint=False grid); the last node is free
        assert u[0] == 0 and u[-1] == 1

    def test_unknown_bc_rejected_at_setup(self):
        with pytest.raises(ValueError, match="robin"):
            _diffusion_1d('robin', lambda x: _gaussian(x))

    def test_nonperiodic_requires_psiOp(self):
        t, x = symbols('t x', real=True)
        u = Function('u')
        solver = PDESolver(Eq(diff(u(t, x), t), diff(u(t, x), x, 2)))
        with pytest.raises(ValueError, match="psiOp"):
            solver.setup(Lx=2 * np.pi, Nx=32, Lt=0.1, Nt=10,
                         boundary_condition='nonperiodic',
                         initial_condition=lambda x: np.sin(x), plot=False)