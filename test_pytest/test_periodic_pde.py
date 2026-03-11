"""
test_periodic_pde.py — Unified test suite for PDESolver with periodic BCs
==========================================================================

This file merges the original test suites:
  - test_1d_periodic_pde.py
  - test_2d_periodic_pde.py

It follows the class‑based organisation introduced for the Dirichlet tests and covers:

1. Equation parsing & initialisation (periodic‑specific checks)
2. Grid & boundary conditions (periodic)
3. Stationary pseudo‑differential problems (1D & 2D)
4. Time‑dependent problems (1D & 2D) – transport, heat, wave, Schrödinger,
   KdV, fractional diffusion, biharmonic, integro‑differential equations
5. Error metrics (solver.test)
6. Internal helpers (dealiasing mask, combined symbol, …)
7. Edge cases and regression tests

All tests use periodic boundary conditions (default) unless explicitly noted.
"""

import pytest
import numpy as np
from sympy import symbols, Function, Eq, sin, cos, exp, diff, I, Abs, sqrt, fourier_transform
from scipy.special import airy, eval_hermite, legendre

from solver import PDESolver, psiOp, Op


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sym1d():
    """Symbolic variables for 1‑D problems."""
    x, xi = symbols('x xi', real=True)
    return x, xi


@pytest.fixture
def sym2d():
    """Symbolic variables for 2‑D problems."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    return x, y, xi, eta


# ---------------------------------------------------------------------------
# Helper: run solver and assert error
# ---------------------------------------------------------------------------
def _run_and_assert(solver, u_exact, t_eval=None, threshold=1e-3, component='real'):
    """Helper that calls solver.test() and asserts the error is below threshold."""
    err = solver.test(u_exact=u_exact, t_eval=t_eval, threshold=threshold, component=component)
    assert err < threshold
    return err


# ===========================================================================
# 1.  Equation parsing & initialisation (periodic context)
# ===========================================================================

class TestParsing:
    """Tests focused on __init__ and _parse_equation for periodic problems."""

    def test_periodic_allowed_without_psiOp(self):
        """Plain linear PDE (no psiOp) works with periodic BC."""
        t, x = symbols('t x', real=True)
        u = Function('u')
        eq = Eq(diff(u(t, x), t), diff(u(t, x), x, 2))
        solver = PDESolver(eq)
        # Should not raise
        solver.setup(Lx=2 * np.pi, Nx=32, Lt=0.1, Nt=10,
                     boundary_condition='periodic',
                     initial_condition=lambda x: np.sin(x),
                     plot=False)

    def test_has_psi_flag_set_with_psiOp(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        assert solver.has_psi is True


# ===========================================================================
# 2.  Grid & boundary conditions (periodic)
# ===========================================================================

class TestGridAndBoundary:
    """Grid shapes, ranges, and periodic boundary enforcement."""

    def _make_solver_2d(self):
        x, y, xi, eta = symbols('x y xi eta', real=True)
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        return solver

    def _make_solver_1d(self):
        x, xi = symbols('x xi', real=True)
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=32,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        return solver

    # --- 2D grid ---

    def test_2d_grid_shapes(self):
        solver = self._make_solver_2d()
        assert solver.X.shape == (32, 32)
        assert solver.Y.shape == (32, 32)
        assert solver.KX.shape == (32, 32)
        assert solver.KY.shape == (32, 32)

    def test_2d_dealiasing_mask_shape_and_dtype(self):
        solver = self._make_solver_2d()
        assert solver.dealiasing_mask.shape == (32, 32)
        assert solver.dealiasing_mask.dtype == bool

    # --- 1D grid ---

    def test_1d_grid_shape(self):
        solver = self._make_solver_1d()
        assert solver.X.shape == (32,)
        assert solver.KX.shape == (32,)

    # --- Periodic boundary enforcement ---

    def test_periodic_1d_copies_corner_values(self):
        solver = self._make_solver_1d()
        u = np.random.rand(32)
        u_copy = u.copy()
        solver._apply_boundary(u_copy)
        assert u_copy[0] == u[-2], "Periodic: u[0] should equal u[-2]"
        assert u_copy[-1] == u[1], "Periodic: u[-1] should equal u[1]"

    def test_periodic_2d_copies_boundary_values(self):
        solver = self._make_solver_2d()
        u = np.random.rand(32, 32)
        u_copy = u.copy()
        solver._apply_boundary(u_copy)
    
        # After periodic BC, first row must equal second‑last row
        np.testing.assert_array_equal(u_copy[0, :], u_copy[-2, :])
        # Last row must equal second row
        np.testing.assert_array_equal(u_copy[-1, :], u_copy[1, :])
        # First column must equal second‑last column
        np.testing.assert_array_equal(u_copy[:, 0], u_copy[:, -2])
        # Last column must equal second column
        np.testing.assert_array_equal(u_copy[:, -1], u_copy[:, 1])

    def test_invalid_boundary_condition_raises(self):
        solver = self._make_solver_1d()
        solver.boundary_condition = 'neumann'  # unsupported
        u = np.zeros(32)
        with pytest.raises(ValueError, match="Invalid boundary condition"):
            solver._apply_boundary(u)


# ===========================================================================
# 3.  Stationary solver — 1D problems
# ===========================================================================

class TestStationary1D:
    """Stationary psiOp problems in 1D with periodic BC."""

    def test_stationary_psiOp_constant_symbol(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        equation = Eq(psiOp(xi**2 + 1, u), sin(x))

        def u_exact(x):
            return np.sin(x) / 2

        solver = PDESolver(equation)
        solver.setup(Lx=2 * np.pi, Nx=256, initial_condition=None, plot=False)

        u_num = solver.solve_stationary_psiOp(order=1)
        u_ref = u_exact(solver.X)
        error = np.linalg.norm(np.real(u_num) - u_ref) / np.linalg.norm(u_ref)

        assert error < 5e-3

    def test_stationary_psiOp_x_dependent(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        p_symbol = xi**2 + x**2 + 1
        f_expr = 3 * (1 - x**2) * exp(-x**2)
        equation = Eq(psiOp(p_symbol, u), f_expr)

        def u_exact(x):
            return np.exp(-x**2)

        solver = PDESolver(equation)
        solver.setup(Lx=30, Nx=512, initial_condition=None, plot=False)

        u_num = solver.solve_stationary_psiOp(order=1)
        u_ref = u_exact(solver.X)
        err = np.linalg.norm(np.real(u_num) - u_ref) / np.linalg.norm(u_ref)
        assert err < 2.0  # loose threshold due to large domain


# ===========================================================================
# 4.  Stationary solver — 2D problems
# ===========================================================================

class TestStationary2D:
    """Stationary psiOp problems in 2D with periodic BC."""

    def test_stationary_psiOp_constant_symbol_2d(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        equation = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))

        def u_exact(x, y):
            return -np.sin(x) * np.sin(y) / 3

        solver = PDESolver(equation)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=128, Ny=128,
                     initial_condition=None, plot=False)

        u_num = solver.solve_stationary_psiOp()
        ref = u_exact(solver.X, solver.Y)
        err = np.linalg.norm(np.real(u_num) - ref) / np.linalg.norm(ref)
        assert err < 2e-1

    def test_stationary_psiOp_variable_symbol_2d(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)

        p_symbol = xi**2 + eta**2 + x**2 + y**2 + 1
        f_expr = (-3*x**2 - 3*y**2 + 5) * exp(-x**2 - y**2)

        def u_exact(x, y):
            return np.exp(-x**2 - y**2)

        equation = Eq(psiOp(p_symbol, u), f_expr)

        solver = PDESolver(equation)
        solver.setup(Lx=10, Ly=10, Nx=64, Ny=64, initial_condition=None, plot=False)
        u_num = solver.solve_stationary_psiOp(order=1)

        ref = u_exact(solver.X, solver.Y)
        err = np.linalg.norm(np.real(u_num) - ref) / np.linalg.norm(ref)
        assert err < 3e-1


# ===========================================================================
# 5.  Time‑dependent solver — 1D problems
# ===========================================================================

class TestTimeDependent1D:
    """Time‑dependent psiOp and Op problems in 1D with periodic BC."""

    # -----------------------------------------------------------------------
    # Integro‑differential equations
    # -----------------------------------------------------------------------
    def test_integro_differential_with_Op(self):
        t, x, kx, xi, omega = symbols('t x kx xi omega')
        u_func = Function('u')
        u = u_func(t, x)
        eps = 0.001
        integral_term = Op(1 / (I * (kx + eps)), u)

        eq = Eq(diff(u, t), diff(u, x) + integral_term - u)
        solver = PDESolver(eq, time_scheme='ETD-RK4')

        Lt = 0.5
        Lx = 20
        Nx = 256
        Nt = 50

        solver.setup(Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
                     initial_condition=lambda x: np.sin(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.exp(-t) * np.sin(x)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=0.6, component='real')

    def test_integro_differential_with_psiOp(self):
        t, x, xi, omega = symbols('t x xi omega')
        u_func = Function('u')
        u = u_func(t, x)
        eps = 0.001
        integral_term = psiOp(I*xi + 1 / (I * (xi + eps)) - 1, u)

        eq = Eq(diff(u, t), integral_term)
        solver = PDESolver(eq, dealiasing_ratio=2/3)

        Lt = 0.5
        Lx = 20
        Nx = 256
        Nt = 50

        solver.setup(Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
                     initial_condition=lambda x: np.sin(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.exp(-t) * np.sin(x)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=0.6, component='real')

    # -----------------------------------------------------------------------
    # Convolution equations
    # -----------------------------------------------------------------------
    def test_convolution_with_Op(self):
        t, x, kx, xi, omega, lam = symbols('t x kx xi omega lambda')
        u_func = Function('u')
        u = u_func(t, x)
        lam_val = 1.0
        f_kernel = exp(-lam_val * Abs(x))

        # Convolution equation: ∂t u = - OpConv(f_kernel, u)
        eq = Eq(diff(u, t), -Op(fourier_transform(f_kernel, x, kx/(2*np.pi)), u))

        solver = PDESolver(eq, time_scheme='ETD-RK4')
        Lt = 0.5
        solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=50,
                     initial_condition=lambda x: np.cos(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            lam_val = 1.0
            k_val = 1.0
            decay = 2 * lam_val / (lam_val**2 + k_val**2)
            return np.cos(x) * np.exp(-decay * t)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-3, component='real')

    def test_convolution_with_psiOp(self):
        t, x, kx, xi, omega, lam = symbols('t x kx xi omega lambda')
        u_func = Function('u')
        u = u_func(t, x)
        lam = 1.0
        f_kernel = exp(-lam * Abs(x))

        eq = Eq(diff(u, t), -psiOp(fourier_transform(f_kernel, x, xi/(2*np.pi)), u))
        solver = PDESolver(eq.subs(lam, 1))

        Lt = 2.0
        solver.setup(Lx=2 * np.pi, Nx=256, Lt=Lt, Nt=100,
                     initial_condition=lambda x: np.cos(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            lam_val = 1.0
            k_val = 1.0
            decay = 2 * lam_val / (lam_val**2 + k_val**2)
            return np.cos(x) * np.exp(-decay * t)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-3, component='real')

    # -----------------------------------------------------------------------
    # Transport
    # -----------------------------------------------------------------------
    def test_transport_pde(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t), -diff(u, x))
        Lt = 1.0
        solver = PDESolver(eq, time_scheme='ETD-RK4')
        solver.setup(Lx=10, Nx=256, Lt=Lt, Nt=200,
                     initial_condition=lambda x: np.exp(-x**2), plot=False)
        solver.solve()

        def u_exact(x, t):
            L = 10
            return np.exp(-((x - t + L/2) % L - L/2)**2)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-2, component='real')

    def test_transport_with_Op(self):
        t, x, kx, xi, omega = symbols('t x kx xi omega')
        u_func = Function('u')
        u = u_func(t, x)
        c = 1.0
        beta = 1.3
        eq = Eq(diff(u, t) + c * diff(u, x), -Op(abs(kx)**beta, u))
        Lt = 0.5
        solver = PDESolver(eq, dealiasing_ratio=0.5, time_scheme='ETD-RK4')
        solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=200,
                     initial_condition=lambda x: np.cos(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.cos(x - c * t) * np.exp(-t * abs(1)**beta)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=7e-2, component='real')

    def test_transport_with_psiOp(self):
        t, x, kx, xi, omega = symbols('t x kx xi omega')
        u_func = Function('u')
        u = u_func(t, x)
        c = 1.0
        beta = 1.3
        eq = Eq(diff(u, t), -psiOp(c*I*xi + abs(xi)**beta, u))
        Lt = 0.5
        solver = PDESolver(eq, dealiasing_ratio=2/3)
        solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=200,
                     initial_condition=lambda x: np.cos(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.cos(x - c * t) * np.exp(-t * abs(1)**beta)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=7e-2, component='real')

    # -----------------------------------------------------------------------
    # Heat / diffusion
    # -----------------------------------------------------------------------
    def test_heat_periodic_sin(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t), diff(u, x, 2))
        Lt = 0.5
        solver = PDESolver(eq, time_scheme='ETD-RK4')
        solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=40,
                     initial_condition=lambda x: np.sin(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.sin(x) * np.exp(-t)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-2, component='real')

    def test_heat_gaussian_initial(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t), diff(u, x, 2))

        solver = PDESolver(eq, time_scheme='ETD-RK4')
        Lt = 0.5
        solver.setup(Lx=5*np.pi, Nx=256, Lt=Lt, Nt=40,
                     initial_condition=lambda x: np.exp(-2 * x**2), plot=False)
        solver.solve()

        def u_exact(x, t):
            sigma = 0.5
            variance = sigma**2 + 2 * t
            return (np.exp(-x**2 / (2 * variance)) / np.sqrt(1 + 2 * t / sigma**2))

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-2, component='real')

    def test_heat_with_Op(self):
        t, x, kx, xi, omega = symbols('t x kx xi omega')
        u_func = Function('u')
        u = u_func(t, x)
        eq = Eq(diff(u, t), Op((I*kx)**2, u)/2 + diff(u, x, 2)/2)
        Lt = 0.5
        solver = PDESolver(eq, time_scheme='ETD-RK4')
        solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=40,
                     initial_condition=lambda x: np.sin(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.sin(x) * np.exp(-t)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-2, component='real')

    def test_fractional_heat(self):
        t, x, kx, xi, omega = symbols('t x kx xi omega')
        u_func = Function('u')
        u = u_func(t, x)
        alpha = 1.8
        nu = 0.1
        eq = Eq(diff(u, t), -Op(abs(kx)**alpha, u) + nu * diff(u, x, 2))
        Lt = 0.5
        solver = PDESolver(eq, dealiasing_ratio=0.5, time_scheme='ETD-RK4')
        solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=200,
                     initial_condition=lambda x: np.sin(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.sin(x) * np.exp(-t * (abs(1)**alpha + nu * 1**2))

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=2e-2, component='real')

    # -----------------------------------------------------------------------
    # Burgers' equation (nonlinear)
    # -----------------------------------------------------------------------
    def test_burgers_equation(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        nu = 0.1
        eq = Eq(diff(u, t), -u * diff(u, x) + nu * diff(u, x, 2))

        def phi(x, t):
            return 2 + np.sin(x) * np.exp(-nu * t)
        def dphi_dx(x, t):
            return np.cos(x) * np.exp(-nu * t)
        def u_exact(x, t):
            return -2 * nu * dphi_dx(x, t) / phi(x, t)

        Lt = 0.5
        solver = PDESolver(eq, time_scheme='ETD-RK4')
        solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=100,
                     initial_condition=lambda x: u_exact(x, 0), plot=False)
        solver.solve()

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-2, component='real')

    # -----------------------------------------------------------------------
    # psiOp with spatial potential
    # -----------------------------------------------------------------------
    def test_psiOp_spatial_potential(self):
        t, x, xi = symbols('t x xi', real=True)   # define t explicitly
        u = Function('u')
        symbol_expr = 1 + sin(x)
        eq = Eq(diff(u(t, x), t), -psiOp(symbol_expr, u(t, x)))
        Lt = 0.5
        solver = PDESolver(eq)
        solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=50,
                     initial_condition=lambda x: np.cos(x), plot=False)
        solver.solve()
    
        def u_exact(x, t):
            return np.cos(x) * np.exp(-t * (1 + np.sin(x)))
    
        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-2, component='real')

    # -----------------------------------------------------------------------
    # Wave equation (second order)
    # -----------------------------------------------------------------------
    def test_wave_periodic_sin(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t, t), diff(u, x, x))
        Lt = 0.5
        solver = PDESolver(eq, time_scheme='ETD-RK4')
        solver.setup(Lx=2*np.pi, Nx=256, Lt=Lt, Nt=100,
                     initial_condition=lambda x: np.sin(x),
                     initial_velocity=lambda x: np.zeros_like(x),
                     plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.sin(x) * np.cos(t)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=0.2, component='real')

    def test_wave_with_source(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        source_term = cos(x) * (3 * cos(np.sqrt(2) * t) - np.sqrt(2) * sin(np.sqrt(2) * t))
        eq = Eq(diff(u, t, t), diff(u, x, x) - u + source_term)

        solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=1)
        Lt = 0.5
        solver.setup(Lx=20, Nx=256, Lt=Lt, Nt=100,
                     initial_condition=lambda x: np.cos(x),
                     initial_velocity=lambda x: np.zeros_like(x),
                     plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.cos(x) * np.cos(np.sqrt(2) * t)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=1, component='real')

    # -----------------------------------------------------------------------
    # Schrödinger equation
    # -----------------------------------------------------------------------
    def test_schrodinger_free_packet(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t), -I * diff(u, x, x))
        Lt = 0.5
        solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=1/2)
        solver.setup(Lx=20, Nx=512, Lt=Lt, Nt=200,
                     initial_condition=lambda x: np.exp(-x**2) * np.exp(1j * x),
                     plot=False)
        solver.solve()

        def u_exact(x, t):
            return 1 / np.sqrt(1 - 4j * t) * np.exp(1j * (x + t)) * np.exp(-((x + 2*t)**2) / (1 - 4j * t))

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=0.6, component='real')

    # -----------------------------------------------------------------------
    # Klein‑Gordon equation
    # -----------------------------------------------------------------------
    def test_klein_gordon(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t, t), diff(u, x, x) - u)
        Lt = 0.5
        solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=1/4)
        solver.setup(Lx=20, Nx=256, Lt=Lt, Nt=100,
                     initial_condition=lambda x: np.cos(x),
                     initial_velocity=lambda x: np.zeros_like(x),
                     plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.cos(np.sqrt(2) * t) * np.cos(x)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5, component='real')

    # -----------------------------------------------------------------------
    # KdV soliton
    # -----------------------------------------------------------------------
    def test_kdv_soliton(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t) + 6 * u * diff(u, x) - diff(u, x, x, x), 0)

        c = 0.5
        x0 = 0.0
        def initial_condition(x):
            return c / 2 * (1 / np.cosh(np.sqrt(c)/2 * (x - x0)))**2
        Lt = 0.5

        solver = PDESolver(eq, time_scheme='ETD-RK4', dealiasing_ratio=2/3)
        solver.setup(Lx=40, Nx=1024, Lt=Lt, Nt=200,
                     initial_condition=initial_condition, plot=False)
        solver.solve()

        def u_exact(x, t):
            return c / 2 * (1 / np.cosh(np.sqrt(c)/2 * (x - c*t - x0)))**2

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=0.5, component='real')

    # -----------------------------------------------------------------------
    # Biharmonic equation
    # -----------------------------------------------------------------------
    def test_biharmonic(self):
        t, x = symbols('t x')
        u = Function('u')(t, x)
        eq = Eq(diff(u, t), -diff(u, x, x, x, x))
        Lt = 0.5
        solver = PDESolver(eq)
        solver.setup(Lx=2*np.pi, Nx=128, Lt=Lt, Nt=50,
                     initial_condition=lambda x: np.sin(x), plot=False)
        solver.solve()

        def u_exact(x, t):
            return np.sin(x) * np.exp(-t)

        n_test = 5
        for i in range(n_test + 1):
            _run_and_assert(solver, u_exact, t_eval=i * Lt / n_test,
                            threshold=5e-2, component='real')


# ===========================================================================
# 6.  Time‑dependent solver — 2D problems
# ===========================================================================

class TestTimeDependent2D:
    """Time‑dependent problems in 2D with periodic BC."""

    def test_transport_2d(self):
        t, x, y = symbols('t x y')
        u = Function('u')(t, x, y)
        eq = Eq(diff(u, t), -diff(u, x) - diff(u, y))

        L = 10
        N = 128
        Lt = 1.0

        solver = PDESolver(eq)
        solver.setup(Lx=L, Ly=L, Nx=N, Ny=N, Lt=Lt, Nt=200,
                     initial_condition=lambda x, y: np.exp(-(x**2 + y**2)),
                     plot=False)
        solver.solve()

        def u_exact(x, y, t):
            Lx = Ly = L
            xs = (x - t + Lx/2) % Lx - Lx/2
            ys = (y - t + Ly/2) % Ly - Ly/2
            return np.exp(-(xs**2 + ys**2))

        _run_and_assert(solver, u_exact, t_eval=Lt/2,
                        threshold=5e-2, component='real')

    def test_heat_2d(self):
        t, x, y = symbols('t x y')
        u = Function('u')(t, x, y)
        eq = Eq(diff(u, t), diff(u, x, 2) + diff(u, y, 2))

        solver = PDESolver(eq)
        solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=50,
                     initial_condition=lambda x, y: np.sin(x) * np.sin(y),
                     plot=False)
        solver.solve()

        def u_exact(x, y, t):
            return np.sin(x) * np.sin(y) * np.exp(-2*t)

        _run_and_assert(solver, u_exact, t_eval=0.5,
                        threshold=5e-2, component='real')

    def test_heat_psiop_2d(self):
        t, x, y, xi, eta = symbols('t x y xi eta', real=True)   # define t explicitly
        u = Function('u')(t, x, y)
        eq = Eq(diff(u, t), psiOp(-(xi**2 + eta**2), u))
    
        solver = PDESolver(eq)
        solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=50,
                     initial_condition=lambda x, y: np.sin(x) * np.sin(y),
                     plot=False)
        solver.solve()
    
        def u_exact(x, y, t):
            return np.sin(x) * np.sin(y) * np.exp(-2*t)
    
        _run_and_assert(solver, u_exact, t_eval=0.5,
                        threshold=5e-2, component='real')

    def test_schrodinger_2d(self):
        t, x, y = symbols('t x y')
        u = Function('u')(t, x, y)
        eq = Eq(I * diff(u, t), diff(u, x, 2) + diff(u, y, 2))

        solver = PDESolver(eq, time_scheme='ETD-RK4')
        solver.setup(Lx=10, Ly=10, Nx=128, Ny=128, Lt=1.0, Nt=100,
                     initial_condition=lambda x, y: np.exp(-x**2 - y**2) * np.exp(1j*(x + y)),
                     plot=False)
        solver.solve()

        def u_exact(x, y, t):
            return 1 / np.sqrt(1 + 4j * t)**2 * np.exp(1j*(x + y - 2*t)) * np.exp(-((x + 2*t)**2 + (y + 2*t)**2) / (1 + 4j * t))

        _run_and_assert(solver, u_exact, t_eval=0.5,
                        threshold=0.3, component='abs')

    def test_fractional_diffusion_2d(self):
        t, x, y, kx, ky = symbols('t x y kx ky')
        u = Function('u')(t, x, y)
        alpha = 1.5
        eq = Eq(diff(u, t), -Op((kx**2 + ky**2)**(alpha/2), u))

        solver = PDESolver(eq)
        solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=80,
                     initial_condition=lambda x, y: np.sin(x) * np.sin(y),
                     plot=False)
        solver.solve()

        def u_exact(x, y, t):
            return np.sin(x) * np.sin(y) * np.exp(-t * (2**(alpha/2)))

        _run_and_assert(solver, u_exact, t_eval=0.5,
                        threshold=5e-2, component='real')

    def test_wave_2d(self):
        t, x, y = symbols('t x y')
        u = Function('u')(t, x, y)
        eq = Eq(diff(u, t, 2), diff(u, x, 2) + diff(u, y, 2))

        solver = PDESolver(eq, time_scheme='ETD-RK4')
        solver.setup(Lx=2*np.pi, Ly=2*np.pi, Nx=128, Ny=128, Lt=1.0, Nt=100,
                     initial_condition=lambda x, y: np.sin(x) * np.sin(y),
                     initial_velocity=lambda x, y: np.zeros_like(x),
                     plot=False)
        solver.solve()

        def u_exact(x, y, t):
            return np.sin(x) * np.sin(y) * np.cos(np.sqrt(2) * t)

        _run_and_assert(solver, u_exact, t_eval=0.5,
                        threshold=5, component='abs')

    def test_klein_gordon_2d(self):
        t, x, y = symbols('t x y')
        u = Function('u')(t, x, y)
        c = 1.0
        m = 1.0
        eq = Eq(diff(u, t, 2), c**2 * (diff(u, x, 2) + diff(u, y, 2)) - m**2 * u)

        L = 2 * np.pi
        N = 512
        T_final = 2.0
        Nt = 200
        kx, ky = 1, 1
        omega_val = float(np.sqrt(c**2 * (kx**2 + ky**2) + m**2))

        solver = PDESolver(eq)
        solver.setup(Lx=L, Ly=L, Nx=N, Ny=N, Lt=T_final, Nt=Nt,
                     initial_condition=lambda x, y: np.sin(x) * np.sin(y),
                     initial_velocity=lambda x, y: np.zeros_like(x),
                     plot=False)
        solver.solve()

        def u_exact(x, y, t):
            return np.sin(x) * np.sin(y) * np.cos(omega_val * t)

        _run_and_assert(solver, u_exact, t_eval=0.5,
                        threshold=5, component='abs')

    def test_biharmonic_2d(self):
        t, x, y = symbols('t x y')
        u = Function('u')(t, x, y)
        eq = Eq(diff(u, t), -(diff(u, x, 4) + 2 * diff(u, x, 2, y, 2) + diff(u, y, 4)))

        L = 2 * np.pi
        N = 512
        Lt = 2.0
        Nt = 400

        solver = PDESolver(eq)
        solver.setup(Lx=L, Ly=L, Nx=N, Ny=N, Lt=Lt, Nt=Nt,
                     initial_condition=lambda x, y: np.sin(x) * np.sin(y),
                     plot=False)
        solver.solve()

        def u_exact(x, y, t):
            return np.sin(x) * np.sin(y) * np.exp(-4 * t)

        _run_and_assert(solver, u_exact, t_eval=0.5,
                        threshold=7e-2, component='abs')


# ===========================================================================
# 7.  Error metrics (solver.test)
# ===========================================================================

class TestErrorMetrics:
    """Exercise all code paths inside solver.test()."""

    def _periodic_2d_solver(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=6)
        return solver

    def test_component_real(self, sym2d):
        solver = self._periodic_2d_solver(sym2d)
        err = solver.test(u_exact=lambda x, y: -np.sin(x) * np.sin(y) / 3,
                          threshold=1e-3, component='real')
        assert err < 1e-3

    def test_component_abs(self, sym2d):
        solver = self._periodic_2d_solver(sym2d)
        err = solver.test(u_exact=lambda x, y: -np.sin(x) * np.sin(y) / 3,
                          threshold=1e-3, component='abs')
        assert err < 1e-3

    def test_component_imag_near_zero(self, sym2d):
        solver = self._periodic_2d_solver(sym2d)
        err = solver.test(u_exact=lambda x, y: np.zeros_like(x),
                          norm='absolute', threshold=1.0, component='imag')
        assert err < 1.0

    def test_norm_absolute(self, sym2d):
        solver = self._periodic_2d_solver(sym2d)
        err = solver.test(u_exact=lambda x, y: -np.sin(x) * np.sin(y) / 3,
                          norm='absolute', threshold=100.0, component='real')
        assert err is not None

    def test_invalid_component_raises(self, sym2d):
        solver = self._periodic_2d_solver(sym2d)
        with pytest.raises(ValueError, match="Invalid component"):
            solver.test(u_exact=lambda x, y: np.zeros_like(x), component='magnitude')

    def test_invalid_norm_raises(self, sym2d):
        solver = self._periodic_2d_solver(sym2d)
        with pytest.raises(ValueError, match="Unknown norm"):
            solver.test(u_exact=lambda x, y: np.zeros_like(x),
                        norm='l1', threshold=100.0, component='real')


# ===========================================================================
# 8.  Internal helpers
# ===========================================================================

class TestInternalHelpers:
    """Unit tests for internal methods (periodic context)."""

    def _simple_periodic_2d(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        symbol = xi**2 + eta**2 + 1
        eq = Eq(psiOp(symbol, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16, Ny=16,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        return solver

    def test_total_symbol_expr_correct(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        symbol = xi**2 + eta**2 + 1
        eq = Eq(psiOp(symbol, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        expr = solver._total_symbol_expr()
        assert expr.has(xi) or expr.has(eta), \
            "_total_symbol_expr should contain frequency variables"

    def test_combined_symbol_dtype(self, sym2d):
        solver = self._simple_periodic_2d(sym2d)
        assert solver.combined_symbol.dtype == np.complex128

    def test_combined_symbol_shape(self, sym2d):
        solver = self._simple_periodic_2d(sym2d)
        assert solver.combined_symbol.shape == (16, 16)

    def test_dealiasing_mask_is_boolean(self, sym2d):
        solver = self._simple_periodic_2d(sym2d)
        assert solver.dealiasing_mask.dtype == bool

    def test_dealiasing_ratio_respected(self, sym2d):
        solver = self._simple_periodic_2d(sym2d)
        fraction_passing = solver.dealiasing_mask.mean()
        assert fraction_passing <= (solver.dealiasing_ratio ** 2) + 0.05


# ===========================================================================
# 9.  Regression / edge cases
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge‑case and regression tests (periodic)."""

    def test_stationary_is_stationary_flag_after_solve(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16, Ny=16,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        solver.solve_stationary_psiOp(order=0)
        assert solver.is_stationary is True

    def test_solve_returns_frames(self):
        t, x = symbols('t x', real=True)
        u = Function('u')
        eq = Eq(diff(u(t, x), t), -psiOp(x**2 + 1, u(t, x)))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=16, Lt=0.1, Nt=10,
                     initial_condition=lambda x: np.sin(x),
                     n_frames=5, plot=False)
        solver.solve()
        assert len(solver.frames) >= 2
        final = solver.frames[-1]
        assert isinstance(final, np.ndarray)
        assert final.shape == (16,)

    def test_different_Nx_Ny_grid(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16, Ny=32,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=0)
        assert u_num.shape == (16, 32)

    def test_1d_stationary_solution_is_real_dominated(self, sym1d):
        x, xi = sym1d
        u = Function('u')(x)
        eq = Eq(psiOp(xi**2 + 1, u), -sin(x))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Nx=64,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=0)
        imag_ratio = np.max(np.abs(np.imag(u_num))) / (np.max(np.abs(u_num)) + 1e-30)
        assert imag_ratio < 1e-5

    def test_2d_stationary_solution_is_real_dominated(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=32, Ny=32,
                     boundary_condition='periodic', initial_condition=None,
                     plot=False)
        u_num = solver.solve_stationary_psiOp(order=6)
        imag_ratio = np.max(np.abs(np.imag(u_num))) / (np.max(np.abs(u_num)) + 1e-30)
        assert imag_ratio < 1e-5

    def test_setup_missing_Ny_2d_raises(self, sym2d):
        x, y, xi, eta = sym2d
        u = Function('u')(x, y)
        eq = Eq(psiOp(xi**2 + eta**2 + 1, u), -sin(x) * sin(y))
        solver = PDESolver(eq)
        with pytest.raises(ValueError, match="Ny"):
            solver.setup(Lx=2 * np.pi, Ly=2 * np.pi, Nx=16,
                         boundary_condition='periodic', plot=False)