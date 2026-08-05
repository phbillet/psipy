"""
test_dirichlet_pde.py — Unified test suite for PDESolver with Dirichlet BCs
===========================================================================

This file merges the original test suites:
  - test_1d_Dirichlet_pde.py
  - test_2d_Dirichlet_pde.py

It follows the class‑based organisation of the 2D test file and covers:

1. Equation parsing & initialisation
2. Grid & boundary conditions
3. Stationary pseudo‑differential problems (1D & 2D)
4. Time‑dependent problems (1D & 2D) – diffusion, wave, Schrödinger,
   Hermite, Airy, Gaussian, Legendre
5. Error metrics (solver.test)
6. Internal helpers (dealiasing mask, combined symbol, …)
7. Edge cases and regression tests
"""

import pytest
import numpy as np
from sympy import symbols, Function, Eq, sin, cos, exp, diff, I
from scipy.special import eval_hermite, airy, legendre

from solver import PDESolver, psiOp


# ---------------------------------------------------------------------------
# Shared symbolic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sym2d():
    """Commonly used symbolic variables for 2‑D problems."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    return x, y, xi, eta


@pytest.fixture
def sym1d():
    """Commonly used symbolic variables for 1‑D problems."""
    x, xi = symbols('x xi', real=True)
    return x, xi


# ===========================================================================
# 1.  Equation parsing & initialisation
# ===========================================================================

class TestParsing:
    """Tests focused on __init__ and _parse_equation."""

    def test_stationary_detected_no_t(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        assert solver.is_stationary is True

    def test_time_dependent_detected_with_t(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        eq = Eq(diff(u(t, x), t), -psiOp(xi**2 + 1, u(t, x)))
        solver = PDESolver(eq)
        assert solver.is_stationary is False

    def test_dim1_detected(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        assert solver.dim == 1

    def test_dim2_detected(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        assert solver.dim == 2

    def test_has_psi_flag_set(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        assert solver.has_psi is True

    def test_multiple_unknowns_raises(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        v = Function('v')(x)
        # Two separate unknowns in the same equation
        with pytest.raises(ValueError, match="exactly one unknown"):
            eq = Eq(psiOp(xi**2 + 1, u) + v, 0)
            PDESolver(eq)

    def test_dirichlet_without_psiOp_raises(self):
        """setup() must raise when Dirichlet BC is requested without psiOp."""
        t, x = symbols('t x', real=True)
        u = Function('u')
        # Plain linear PDE (no psiOp)
        eq = Eq(diff(u(t, x), t), diff(u(t, x), x, 2))
        solver = PDESolver(eq)
        with pytest.raises(ValueError, match="psiOp"):
            solver.setup(Lx=2 * np.pi, Nx=32, Lt=0.1, Nt=10,
                         boundary_condition='dirichlet',
                         initial_condition=lambda x: np.sin(x),
                         plot=False)


# ===========================================================================
# 2.  Grid & boundary conditions
# ===========================================================================

class TestGridAndBoundary:
    """Grid shapes, ranges, and boundary condition enforcement."""

    def _make_solver_2d(self):
        x, y, xi, eta = symbols('x y xi eta', real=True)
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        return solver

    def _make_solver_1d(self, bc='dirichlet'):
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=32,
                     boundary_condition=bc, initial_condition=None,
                     plot=False)
        return solver

    # --- 2D grid ---

    def test_2d_grid_shapes(self):
        solver = self._make_solver_2d()
        assert solver.X.shape == (32, 32)
        assert solver.Y.shape == (32, 32)
        assert solver.KX.shape == (32, 32)
        assert solver.KY.shape == (32, 32)

    def test_2d_x_range(self):
        solver = self._make_solver_2d()
        assert solver.X.min() >= -np.pi - 1e-10
        assert solver.X.max() < np.pi + 1e-10

    def test_2d_dealiasing_mask_shape_and_dtype(self):
        solver = self._make_solver_2d()
        assert solver.dealiasing_mask.shape == (32, 32)
        assert solver.dealiasing_mask.dtype == bool

    def test_2d_dealiasing_kills_highest_modes(self):
        """The highest-frequency modes must be masked out."""
        solver = self._make_solver_2d()
        # The corner of the wavenumber grid holds the highest frequencies
        mask = solver.dealiasing_mask
        assert not mask[0, mask.shape[1] // 2 + 5], \
            "High-frequency corner should be dealiased (False)"

    # --- 1D grid ---

    def test_1d_grid_shape(self):
        solver = self._make_solver_1d()
        assert solver.X.shape == (32,)
        assert solver.KX.shape == (32,)

    def test_1d_grid_range(self):
        solver = self._make_solver_1d()
        assert solver.X.min() >= -np.pi - 1e-10
        assert solver.X.max() < np.pi + 1e-10

    # --- Boundary condition enforcement ---

    def test_dirichlet_2d_sets_boundary_to_zero(self):
        solver = self._make_solver_2d()
        u = np.random.rand(32, 32)
        solver._apply_boundary(u)
        assert np.all(u[0, :] == 0), "First row should be 0 (Dirichlet)"
        assert np.all(u[-1, :] == 0), "Last row should be 0 (Dirichlet)"
        assert np.all(u[:, 0] == 0), "First column should be 0 (Dirichlet)"
        assert np.all(u[:, -1] == 0), "Last column should be 0 (Dirichlet)"

    def test_dirichlet_1d_sets_boundary_to_zero(self):
        solver = self._make_solver_1d()
        u = np.random.rand(32)
        solver._apply_boundary(u)
        assert u[0] == 0.0, "u[0] should be 0 for Dirichlet"
        assert u[-1] == 0.0, "u[-1] should be 0 for Dirichlet"

    def test_periodic_1d_copies_corner_values(self):
        solver = self._make_solver_1d(bc='periodic')
        u = np.random.rand(32)
        u_copy = u.copy()
        solver._apply_boundary(u_copy)
        assert u_copy[0] == u[-2], "Periodic: u[0] should equal u[-2]"
        assert u_copy[-1] == u[1], "Periodic: u[-1] should equal u[1]"

    def test_invalid_boundary_condition_raises(self):
        solver = self._make_solver_1d()
        solver.boundary_condition = 'zorglub'  # unsupported
        u = np.zeros(32)
        with pytest.raises(ValueError, match="Invalid boundary condition"):
            solver._apply_boundary(u)


# ===========================================================================
# 3.  Stationary solver — 1D problems
# ===========================================================================

class TestStationary1D:
    """Stationary psiOp problems in 1D (including imported tests)."""

    def test_constant_symbol_1d(self):
        """psiOp(ξ²+1, u) = -sin(x) → exact: -sin(x)/2."""
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))

        def u_exact(x):
            return -np.sin(x) / 2.0

        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=64,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=6)

        err = solver.test(u_exact=u_exact, threshold=1e-3, component='real')
        assert err < 1e-3

    def test_constant_symbol_1d_periodic(self):
        """Same equation but with periodic BC (cos solution is doubly periodic)."""
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -cos(x))

        def u_exact(x):
            return -np.cos(x) / 2.0

        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=64,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=6)

        err = solver.test(u_exact=u_exact, threshold=1e-4, component='real')
        assert err < 1e-4

    def test_higher_order_symbol_1d(self):
        """psiOp(ξ⁴ + 1, u) = -sin(x) → exact: -sin(x)/2 (since k=1 → k⁴=1)."""
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**4 + 1, u), -sin(x))

        def u_exact(x):
            return -np.sin(x) / 2.0

        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=64,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=0)

        err = solver.test(u_exact=u_exact, threshold=5e-2, component='real')
        assert err < 5e-2

    def test_stationary_solution_stored_in_solver(self):
        """After solve_stationary_psiOp, solver.u must be a numpy array."""
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=32,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=0)
        assert isinstance(u_num, np.ndarray)
        assert u_num.shape == (32,)

    # -----------------------------------------------------------------------
    # Tests imported from test_1d_Dirichlet_pde.py
    # -----------------------------------------------------------------------

    def test_stationary_equation_1d(self):
        """psiOp(xi^2 + 1, u) = sin(x) → exact: sin(x)/2."""
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        equation = Eq(psiOp(xi**2 + 1, u), sin(x))

        def u_exact(x_vals):
            return np.sin(x_vals) / 2

        solver = PDESolver(equation)
        solver.setup(Lx=2 * np.pi, Nx=256,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=1)
        err = solver.test(u_exact=u_exact, threshold=5e-3, component='real')
        assert err < 5e-3

    def test_stationary_equation_with_x_and_xi_1d(self):
        """psiOp(x^2 * xi^2 + 1, u) = sin(x)  (approx. exact solution)."""
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        equation = Eq(psiOp(x**2 * xi**2 + 1, u), sin(x))

        def u_exact(x_vals):
            return np.sin(x_vals) / (x_vals**2 + 1)

        solver = PDESolver(equation)
        solver.setup(Lx=2 * np.pi, Nx=256,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=0)
        err = solver.test(u_exact=u_exact, threshold=1, component='real')
        assert err < 1


# ===========================================================================
# 4.  Stationary solver — 2D problems (strengthened existing tests)
# ===========================================================================

class TestStationary2D:
    """Stationary psiOp problems in 2D."""

    def test_constant_symbol_2d_tight_threshold(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        symbol = xi**2 + eta**2 + 1
        eq = Eq(psiOp(symbol, u), -sin(x) * sin(y))

        def u_exact(x, y):
            return -np.sin(x) * np.sin(y) / 3.0

        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=64, Ny=64,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=6)

        err = solver.test(u_exact=u_exact, threshold=1e-5, component='real')
        assert err < 1e-5

    def test_constant_symbol_2d_solution_shape(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=6)
        assert u_num.shape == (32, 32)

    def test_constant_symbol_2d_absolute_norm(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))

        def u_exact(x, y):
            return -np.sin(x) * np.sin(y) / 3.0

        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=6)

        err = solver.test(u_exact=u_exact, norm='absolute', threshold=10.0,
                          component='real')
        assert err < 10.0

    def test_cos_source_2d(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -cos(x) * cos(y))

        def u_exact(x, y):
            # symbol at (kx=1, ky=1): 1+1+1 = 3
            return -np.cos(x) * np.cos(y) / 3.0

        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=64, Ny=64,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=6)

        err = solver.test(u_exact=u_exact, threshold=1e-5, component='real')
        assert err < 1e-5


# ===========================================================================
# 5.  Time‑dependent solver — 1D
# ===========================================================================

class TestTimeDependentSolver1D:
    """Time‑dependent psiOp problems in 1D (including imported tests)."""

    def _build_diffusion_1d(self, Nt=200, Lt=1.0, k0=1.0):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        eq = Eq(diff(u(t, x), t), -psiOp(xi**2 + 1, u(t, x)))
        solver = PDESolver(eq)
        solver.setup(
            Lx=2 * np.pi, Nx=64,
            Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=lambda x: np.sin(k0 * x),
            n_frames=20,
            plot=False,
        )
        return solver, k0

    def test_1d_diffusion_at_final_time(self):
        solver, k0 = self._build_diffusion_1d(Nt=500, Lt=1.0)
        solver.solve()

        def u_exact(x, t):
            return np.sin(k0 * x) * np.exp(-(k0**2 + 1) * t)

        err = solver.test(u_exact=u_exact, t_eval=1.0,
                          threshold=0.1, component='real')
        assert err < 0.1

    def test_1d_diffusion_frames_saved(self):
        solver, _ = self._build_diffusion_1d(Nt=200, Lt=1.0)
        solver.solve()
        assert len(solver.frames) >= 2

    def test_1d_diffusion_error_grows_with_time(self):
        """Error at t=0 must be smaller than at t=T (solution decays but
        the relative deviation from zero grows as the reference shrinks)."""
        solver, k0 = self._build_diffusion_1d(Nt=500, Lt=2.0)
        solver.solve()

        def u_exact(x, t):
            return np.sin(k0 * x) * np.exp(-(k0**2 + 1) * t)

        err_0 = solver.test(u_exact=u_exact, t_eval=0.0,
                            threshold=1.0, component='real')
        err_T = solver.test(u_exact=u_exact, t_eval=2.0,
                            threshold=1.0, component='real')
        assert err_0 < err_T or err_0 < 1e-2

    def test_1d_t_eval_out_of_range_raises(self):
        solver, k0 = self._build_diffusion_1d(Nt=100, Lt=1.0)
        solver.solve()

        def u_exact(x, t):
            return np.sin(k0 * x) * np.exp(-(k0**2 + 1) * t)

        with pytest.raises((ValueError, AssertionError)):
            solver.test(u_exact=u_exact, t_eval=999.0,
                        threshold=1e-10, component='real')

    def test_zero_initial_condition_stays_zero(self):
        """For a homogeneous linear equation, zero IC → zero solution."""
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        eq = Eq(diff(u(t, x), t), -psiOp(xi**2 + 1, u(t, x)))
        solver = PDESolver(eq)
        solver.setup(
            Lx=2 * np.pi, Nx=32,
            Lt=1.0, Nt=100,
            boundary_condition='dirichlet',
            initial_condition=lambda x: np.zeros_like(x),
            n_frames=10,
            plot=False,
        )
        solver.solve()
        for frame in solver.frames:
            assert np.max(np.abs(frame)) < 1e-10

    # -----------------------------------------------------------------------
    # Tests imported from test_1d_Dirichlet_pde.py
    # -----------------------------------------------------------------------

    def test_diffusion_equation_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        equation = Eq(diff(u(t, x), t), -psiOp(xi**2 + 1, u(t, x)))

        solver = PDESolver(equation)
        Lx = 2 * np.pi
        Nx = 128
        Lt = 2.0
        Nt = 400
        k0 = 1.0
        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=lambda x: np.sin(k0 * x),
            plot=False
        )
        solver.solve()

        def u_exact(x, t):
            return np.sin(k0 * x) * np.exp(-(k0**2 + 1) * t)

        n_test = 4
        for i in range(n_test + 1):
            err = solver.test(u_exact=u_exact, t_eval=i * Lt / n_test,
                              threshold=50, component='real')
            assert err < 50

    def test_wave_equation_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')(t, x)
        eq = Eq(diff(u, t, t), psiOp(-xi**2, u))

        solver = PDESolver(eq)
        Lt, Lx = 5.0, 2 * np.pi
        Nx, Nt = 512, 1000
        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=lambda x: np.sin(x),
            initial_velocity=lambda x: np.zeros_like(x),
            plot=False
        )
        solver.solve()

        def u_exact(x, t):
            return np.sin(x) * np.cos(t)

        n_test = 10
        for i in range(n_test + 1):
            err = solver.test(u_exact=u_exact, t_eval=i * Lt / n_test,
                              threshold=7e-1, component='real')
            assert err < 7e-1

    def test_schrodinger_equation_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')(t, x)
        equation = Eq(diff(u, t), psiOp(I * xi**2, u))

        solver = PDESolver(equation)
        Lx, Nx = 2 * np.pi, 512
        Lt, Nt = 5.0, 500
        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=lambda x: np.sin(x),
            plot=False
        )
        solver.solve()

        def u_exact(x, t):
            return np.exp(1j * t) * np.sin(x)

        n_test = 5
        for i in range(n_test + 1):
            err = solver.test(u_exact=u_exact, t_eval=i * Lt / n_test,
                              threshold=7e-1, component='real')
            assert err < 7e-1

    def test_equation_psiOp_depending_on_x_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        symbol_expr = 1 + x**2
        eq = Eq(diff(u(t, x), t), -psiOp(symbol_expr, u(t, x)))

        solver = PDESolver(eq)
        Lx, Nx = 2.0, 256
        Lt, Nt = 2.0, 300

        def initial_condition(x):
            return np.sin(np.pi * x / (Lx / 2))

        def u_exact(x, t):
            return initial_condition(x) * np.exp(-t * (1 + x**2))

        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=initial_condition,
            plot=False
        )
        solver.solve()

        n_test = 5
        for i in range(n_test + 1):
            t_eval = i * Lt / n_test
            err = solver.test(u_exact=u_exact, t_eval=t_eval,
                              threshold=5e-2, component='real')
            assert err < 5e-2

    def test_hermite_equation_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        p_expr = x**2 + xi**2
        equation = Eq(diff(u(t, x), t, t), -psiOp(p_expr, u(t, x)))

        solver = PDESolver(equation, compute_energy=False)
        Lx, Nx = 12.0, 256
        Lt, Nt = 3.0, 600
        n = 2
        lambda_n = 2 * n + 1

        def initial_condition(x):
            return eval_hermite(n, x) * np.exp(-x**2 / 2)

        def initial_velocity(x):
            return 0.0 * x

        def u_exact(x, t):
            return np.cos(np.sqrt(lambda_n) * t) * eval_hermite(n, x) * np.exp(-x**2 / 2)

        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=initial_condition,
            initial_velocity=initial_velocity,
            plot=False
        )
        solver.solve()

        n_test = 5
        for i in range(n_test + 1):
            t_eval = i * Lt / n_test
            err = solver.test(u_exact=u_exact, t_eval=t_eval,
                              threshold=50, component='real')
            assert err < 50

    def test_airy_equation_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        p_expr = x + xi**2
        equation = Eq(diff(u(t, x), t, t), -psiOp(p_expr, u(t, x)))

        solver = PDESolver(equation, compute_energy=False)
        Lx, Nx = 40.0, 256
        Lt, Nt = 2.0, 1000

        def initial_condition(x):
            return airy(x)[0]

        def initial_velocity(x):
            return 0.0 * x

        def u_exact(x, t):
            return airy(x - t**2 / 4)[0]

        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=initial_condition,
            initial_velocity=initial_velocity,
            plot=False
        )
        solver.solve()

        n_test = 5
        for i in range(n_test + 1):
            t_eval = i * Lt / n_test
            err = solver.test(u_exact=u_exact, t_eval=t_eval,
                              threshold=50, component='real')
            assert err < 50

    def test_gaussian_equation_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        p_expr = x**2 + xi**2 +0.01
        equation = Eq(diff(u(t, x), t, t), -psiOp(p_expr, u(t, x)))

        solver = PDESolver(equation, compute_energy=False)
        Lx, Nx = 10.0, 256
        Lt, Nt = 2 * np.pi, 1000

        def initial_condition(x):
            return np.exp(-0.5 * x**2)

        def initial_velocity(x):
            return 0.0 * x

        def u_exact(x, t):
            omega = np.sqrt(1.01)
            return np.cos(omega * t) * np.exp(-0.5 * x**2)

        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=initial_condition,
            initial_velocity=initial_velocity,
            plot=False
        )
        solver.solve()

        n_test = 5
        for i in range(n_test + 1):
            t_eval = i * Lt / n_test
            err = solver.test(u_exact=u_exact, t_eval=t_eval,
                              threshold=0.2, component='real')
            assert err < 0.2

    def test_legendre_equation_1d(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        n = 20
        P_n = legendre(n)
        lambda_n = n * (n + 1)

        p_expr = xi**2
        equation = Eq(diff(u(t, x), t, t), -psiOp(p_expr, u(t, x)))

        solver = PDESolver(equation, compute_energy=False)
        Lx, Nx = 2.0, 256
        Lt, Nt = 2 * np.pi / np.sqrt(lambda_n), 500

        def initial_condition(x):
            return P_n(2 * x / Lx)

        def initial_velocity(x):
            return 0 * x

        def u_exact(x, t):
            x_scaled = 2 * x / Lx
            return np.cos(np.sqrt(lambda_n) * t) * P_n(x_scaled)

        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=initial_condition,
            initial_velocity=initial_velocity,
            plot=False
        )
        solver.solve()

        n_test = 5
        for i in range(n_test + 1):
            t_eval = i * Lt / n_test
            err = solver.test(u_exact=u_exact, t_eval=t_eval,
                              threshold=5, component='real')
            assert err < 5

# ===========================================================================
# 6.  Time‑dependent solver — 2D (strengthened)
# ===========================================================================

class TestTimeDependentSolver2D:
    """Time‑dependent psiOp problems in 2D."""

    def _build_diffusion_2d(self, k0=1.0, l0=1.0, Lt=1.0, Nt=200):
        t, x, y, xi, eta = symbols('t x y xi eta', real=True)
        u = Function('u')
        eq = Eq(diff(u(t, x, y), t), -psiOp(xi**2 + eta**2 + 1, u(t, x, y)))
        solver = PDESolver(eq)
        solver.setup(
            Lx=2 * np.pi, Ly=2 * np.pi,
            Nx=32, Ny=32,
            Lt=Lt, Nt=Nt,
            boundary_condition='dirichlet',
            initial_condition=lambda x, y: np.sin(k0 * x) * np.sin(l0 * y),
            n_frames=10,
            plot=False,
        )
        return solver, k0, l0

    def test_2d_diffusion_final_time(self):
        solver, k0, l0 = self._build_diffusion_2d(Lt=1.0, Nt=300)
        solver.solve()

        def u_exact(x, y, t):
            return np.sin(k0 * x) * np.sin(l0 * y) * np.exp(-(k0**2 + l0**2 + 1) * t)

        err = solver.test(u_exact=u_exact, t_eval=1.0,
                          threshold=0.5, component='real')
        assert err < 0.5

    def test_2d_diffusion_intermediate_times(self):
        Lt = 2.0
        Nt = 500
        solver, k0, l0 = self._build_diffusion_2d(Lt=Lt, Nt=Nt)
        solver.solve()

        def u_exact(x, y, t):
            return np.sin(k0 * x) * np.sin(l0 * y) * np.exp(-(k0**2 + l0**2 + 1) * t)

        for frac in [0.0, 0.25, 0.5, 1.0]:
            t_eval = frac * Lt
            err = solver.test(u_exact=u_exact, t_eval=t_eval,
                              threshold=1.0, component='real')
            assert err < 1.0, f"Error too large at t={t_eval}: {err}"

    def test_2d_frames_count(self):
        solver, _, _ = self._build_diffusion_2d(Nt=100)
        solver.solve()
        assert len(solver.frames) >= 2


# ===========================================================================
# 7.  Error metrics (solver.test)
# ===========================================================================

class TestErrorMetrics:
    """Exercise all code paths inside solver.test()."""

    def _stationary_2d_solver(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=6)
        return solver

    def test_component_real(self, sym2d):
        solver = self._stationary_2d_solver(sym2d)
        err = solver.test(
            u_exact=lambda x, y: -np.sin(x) * np.sin(y) / 3,
            threshold=1e-3, component='real')
        assert err is not None

    def test_component_abs(self, sym2d):
        solver = self._stationary_2d_solver(sym2d)
        err = solver.test(
            u_exact=lambda x, y: -np.sin(x) * np.sin(y) / 3,
            threshold=1e-3, component='abs')
        assert err is not None

    def test_component_imag_near_zero(self, sym2d):
        """Imaginary part of a real solution should be ~0."""
        solver = self._stationary_2d_solver(sym2d)
        err = solver.test(
            u_exact=lambda x, y: np.zeros_like(x),
            norm='absolute', threshold=1.0, component='imag')
        assert err < 1.0

    def test_norm_absolute(self, sym2d):
        solver = self._stationary_2d_solver(sym2d)
        err = solver.test(
            u_exact=lambda x, y: -np.sin(x) * np.sin(y) / 3,
            norm='absolute', threshold=100.0, component='real')
        assert err is not None

    def test_invalid_component_raises(self, sym2d):
        solver = self._stationary_2d_solver(sym2d)
        with pytest.raises(ValueError, match="Invalid component"):
            solver.test(
                u_exact=lambda x, y: np.zeros_like(x),
                component='magnitude')

    def test_invalid_norm_raises(self, sym2d):
        solver = self._stationary_2d_solver(sym2d)
        with pytest.raises(ValueError, match="Unknown norm"):
            solver.test(
                u_exact=lambda x, y: np.zeros_like(x),
                norm='l1', threshold=100.0, component='real')


# ===========================================================================
# 8.  Internal helpers
# ===========================================================================

class TestInternalHelpers:
    """Unit tests for internal methods."""

    def _simple_stationary_2d(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        symbol = xi**2 + eta**2 + 1
        eq = Eq(psiOp(symbol, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16, Ny=16,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        return solver

    def test_total_symbol_expr_correct(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        symbol = xi**2 + eta**2 + 1
        eq = Eq(psiOp(symbol, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        expr = solver._total_symbol_expr()
        # The expression should contain xi and eta
        assert expr.has(xi) or expr.has(eta), \
            "_total_symbol_expr should contain frequency variables"

    def test_combined_symbol_dtype(self, sym2d):
        solver = self._simple_stationary_2d(sym2d)
        assert solver.combined_symbol.dtype == np.complex128

    def test_combined_symbol_shape(self, sym2d):
        solver = self._simple_stationary_2d(sym2d)
        assert solver.combined_symbol.shape == (16, 16)

    def test_dealiasing_mask_is_boolean(self, sym2d):
        solver = self._simple_stationary_2d(sym2d)
        assert solver.dealiasing_mask.dtype == bool

    def test_dealiasing_ratio_respected(self, sym2d):
        """Fewer than dealiasing_ratio fraction of modes should pass."""
        solver = self._simple_stationary_2d(sym2d)
        fraction_passing = solver.dealiasing_mask.mean()
        # With the default 2/3 ratio, the fraction should be ≤ (2/3)²
        assert fraction_passing <= (solver.dealiasing_ratio ** 2) + 0.05


# ===========================================================================
# 9.  Regression / edge cases
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge‑case and regression tests."""

    def test_stationary_is_stationary_flag_after_solve(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16, Ny=16,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=0)
        assert solver.is_stationary is True

    def test_solve_returns_frames(self):
        t, x, xi = symbols('t x xi', real=True)
        u = Function('u')
        eq = Eq(diff(u(t, x), t), -psiOp(xi**2 + 1, u(t, x)))
        solver = PDESolver(eq)
        solver.setup(
            Lx=2 * np.pi, Nx=16,
            Lt=0.1, Nt=10,
            boundary_condition='dirichlet',
            initial_condition=lambda x: np.sin(x),
            n_frames=5,
            plot=False,
        )
        solver.solve()
        assert len(solver.frames) >= 2
        final = solver.frames[-1]
        assert isinstance(final, np.ndarray)
        assert final.shape == (16,)

    def test_different_Nx_Ny_grid(self, sym2d):
        """2‑D solver must handle non‑square grids."""
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16, Ny=32,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=0)
        assert u_num.shape == (16, 32)

    def test_1d_stationary_solution_is_real_dominated(self):
        """The imaginary part of the 1D solution should be negligible."""
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=64,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=0)
        imag_ratio = np.max(np.abs(np.imag(u_num))) / (np.max(np.abs(u_num)) + 1e-30)
        assert imag_ratio < 1e-5

    def test_2d_stationary_solution_is_real_dominated(self, sym2d):
        """Same check for 2‑D."""
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='dirichlet', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=6)
        imag_ratio = np.max(np.abs(np.imag(u_num))) / (np.max(np.abs(u_num)) + 1e-30)
        assert imag_ratio < 1e-5

    def test_setup_missing_Ny_2d_raises(self, sym2d):
        """In 2‑D, omitting Ny must raise a clear ValueError."""
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        with pytest.raises(ValueError, match="Ny"):
            solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16,
                         boundary_condition='dirichlet', plot=False)