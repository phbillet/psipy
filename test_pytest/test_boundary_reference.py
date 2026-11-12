"""
test_boundary_reference.py — Reference tests for wall boundary treatments
=========================================================================

Equation:  u_t = -Op(xi^2 + 1) u   (i.e. u_t = u_xx - u)

Exact Dirichlet solution on [-L/2, L/2] by the method of images, for an
initial Gaussian exp(-(x-x0)^2 / A0):

    u(x,t) = sum_n [ G(x; x0 + 2Ln, t) - G(x; -L - x0 + 2Ln, t) ]
    G(x; c, t) = e^{-t} sqrt(A0/a) exp(-(x-c)^2 / a),   a = A0 + 4t

Tests
-----
1. Odd extension on a doubled periodic domain reproduces the exact Dirichlet
   solution (validates the "method of images" recommendation).
2. 'nonperiodic' (boundary clipping) is exact in the large-box regime, and
   clearly worse than the images when the pulse hits the wall (and the guard
   fires).
3. Optional periodic edge guard (`periodic_edge_tol`).

Assumptions to check on first run: the final field is read from
`solver.u_prev`, and `solver.X` is the spatial grid.
"""

import warnings

import numpy as np
import pytest
from sympy import symbols, Function, Eq, diff

from solver import PDESolver, psiOp

A0 = 0.5          # Gaussian parameter: exp(-(x-c)^2 / A0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _G(x, c, t):
    a = A0 + 4.0 * t
    return np.exp(-t) * np.sqrt(A0 / a) * np.exp(-((x - c) ** 2) / a)


def _dirichlet_exact(x, x0, t, L, nmax=3):
    """Exact solution with homogeneous Dirichlet walls at x = -L/2 and +L/2."""
    s = np.zeros_like(x, dtype=float)
    for n in range(-nmax, nmax + 1):
        s += _G(x, x0 + 2 * L * n, t) - _G(x, -L - x0 + 2 * L * n, t)
    return s


def _diffusion_1d(bc, ic, Lx, Nx, Lt, Nt=100, **setup_kw):
    t, x, xi = symbols('t x xi', real=True)
    u = Function('u')
    eq = Eq(diff(u(t, x), t), -psiOp(xi**2 + 1, u(t, x)))
    solver = PDESolver(eq)
    solver.setup(Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
                 boundary_condition=bc, initial_condition=ic,
                 n_frames=5, plot=False, **setup_kw)
    return solver


def _final(solver):
    return np.real(solver.u_prev)


def _run_recording(bc, ic, Lx, Nx, Lt, Nt=100, **kw):
    """Setup + solve, returning (solver, list of RuntimeWarnings)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        solver = _diffusion_1d(bc, ic, Lx, Nx, Lt, Nt, **kw)
        solver.solve()
    rt = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    return solver, rt


# ===========================================================================
# 1.  Method of images with the periodic solver
# ===========================================================================

class TestMethodOfImages:

    L, X0, T = 10.0, 3.5, 0.5

    def test_odd_extension_reproduces_exact_dirichlet(self):
        L, x0, T = self.L, self.X0, self.T
        # Doubled periodic domain [-L, L): odd about both walls (-L/2, +L/2).
        ic_ext = lambda x: _dirichlet_exact(x, x0, 0.0, L)
        sol = _diffusion_1d('periodic', ic_ext, Lx=2 * L, Nx=256, Lt=T, Nt=100)
        sol.solve()

        exact = _dirichlet_exact(sol.X, x0, T, L)
        np.testing.assert_allclose(_final(sol), exact, atol=1e-8)

        # The walls are zeros of the solution (odd symmetry), to spectral accuracy.
        i_left = np.argmin(np.abs(sol.X + L / 2))
        assert abs(_final(sol)[i_left]) < 1e-8

    def test_images_beat_clipping_when_wall_is_hit(self):
        L, x0, T = self.L, self.X0, self.T
        ic = lambda x: _G(x, x0, 0.0)

        # (a) images on the doubled periodic domain
        ic_ext = lambda x: _dirichlet_exact(x, x0, 0.0, L)
        img = _diffusion_1d('periodic', ic_ext, Lx=2 * L, Nx=256, Lt=T, Nt=100)
        img.solve()
        err_img = np.max(np.abs(_final(img) - _dirichlet_exact(img.X, x0, T, L)))

        # (b) heuristic clipping on the physical box
        clip, rt = _run_recording('nonperiodic', ic, Lx=L, Nx=128, Lt=T, Nt=100)
        err_clip = np.max(np.abs(_final(clip) - _dirichlet_exact(clip.X, x0, T, L)))

        assert err_img < 1e-8
        assert err_clip > 100 * err_img
        # ... and the guard must have flagged the situation.
        assert len(rt) >= 1
        assert clip._bc_max_correction > clip.boundary_tol

    def test_clipping_exact_when_pulse_far_from_wall(self):
        L, x0, T = 40.0, 0.0, 0.5
        ic = lambda x: _G(x, x0, 0.0)
        clip, rt = _run_recording('nonperiodic', ic, Lx=L, Nx=512, Lt=T, Nt=100)

        err = np.max(np.abs(_final(clip) - _dirichlet_exact(clip.X, x0, T, L)))
        assert err < 1e-8
        assert rt == []
        assert clip._bc_max_correction < clip.boundary_tol


# ===========================================================================
# 2b. Wall symmetry of the clipping (endpoint=False grid)
# ===========================================================================

def _mirror(u):
    """x -> -x on the grid x_j = -L/2 + j h  (node j <-> node N-j, 0 fixed)."""
    return np.roll(u[::-1], 1)


class TestWallSymmetry:

    def test_clipping_is_mirror_symmetric(self):
        """A pulse at +x0 and its mirror image at -x0 must evolve into mirror
        images of each other. Clipping both node 0 and node N-1 breaks this
        (the right wall would sit at L/2 - h); clipping node 0 only keeps it."""
        L, x0, T = 10.0, 3.5, 0.5
        ic_right = lambda x: _G(x, +x0, 0.0)
        ic_left = lambda x: _G(x, -x0, 0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            right = _diffusion_1d('nonperiodic', ic_right, Lx=L, Nx=128, Lt=T)
            right.solve()
            left = _diffusion_1d('nonperiodic', ic_left, Lx=L, Nx=128, Lt=T)
            left.solve()

        np.testing.assert_allclose(_final(left), _mirror(_final(right)), atol=1e-10)

    def test_mirror_helper(self):
        u = np.arange(8.0)
        np.testing.assert_array_equal(_mirror(u), [0, 7, 6, 5, 4, 3, 2, 1])


# ===========================================================================
# 3.  Periodic edge guard
# ===========================================================================

class TestPeriodicEdgeGuard:

    def test_disabled_by_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            sol = _diffusion_1d('periodic', lambda x: np.sin(x + 0.3),
                                Lx=2 * np.pi, Nx=64, Lt=0.5)
            sol.solve()
        assert sol.periodic_edge_tol is None
        assert sol._edge_max == 0.0

    def test_warns_when_pulse_sits_on_the_edge(self):
        ic = lambda x: _G(x, 4.5, 0.0)          # edge of [-5, 5)
        with pytest.warns(RuntimeWarning, match="wraps around"):
            sol = _diffusion_1d('periodic', ic, Lx=10.0, Nx=128, Lt=0.2,
                                periodic_edge_tol=1e-8)
        assert sol._edge_warned is True
        assert sol._edge_max > 1e-8

    def test_silent_for_well_centered_pulse(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            sol = _diffusion_1d('periodic', lambda x: _G(x, 0.0, 0.0),
                                Lx=40.0, Nx=512, Lt=0.5,
                                periodic_edge_tol=1e-8)
            sol.solve()
        assert sol._edge_warned is False
        assert sol._edge_max < 1e-8

    def test_detects_wraparound_during_evolution(self):
        """Edge amplitude is ~1e-22 initially; diffusion brings it to ~1e-2."""
        ic = lambda x: _G(x, 0.0, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)       # setup is clean
            sol = _diffusion_1d('periodic', ic, Lx=10.0, Nx=128, Lt=1.0,
                                Nt=100, periodic_edge_tol=1e-8)
        with pytest.warns(RuntimeWarning, match="wraps around"):
            sol.solve()

    def test_guard_does_not_modify_solution(self):
        ic = lambda x: _G(x, 4.5, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = _diffusion_1d('periodic', ic, Lx=10.0, Nx=128, Lt=0.2,
                              periodic_edge_tol=1e-8)
            a.solve()
        b = _diffusion_1d('periodic', ic, Lx=10.0, Nx=128, Lt=0.2)
        b.solve()
        np.testing.assert_array_equal(_final(a), _final(b))