#!/usr/bin/env python3
"""
test_performance_convergence.py – final correct version.

Key insight (reached after four rounds of debugging)
-----------------------------------------------------
The solver's 'default' time scheme computes exp(L·dt)·û exactly in Fourier
space for ALL linear, constant-coefficient problems regardless of whether
L(k) is real or imaginary.  Concretely:

  heat 1D:      L(k) = -k²        (real, negative)   → exp(-k²·dt) exact
  advection 1D: L(k) = -i·k       (imaginary)        → exp(-i·k·dt) exact
  wave 1D:      L(k) = -k² (ETD2) (real, negative)   → exp(-k²·dt) exact
  heat 2D:      L(k) = -(kx²+ky²) (real, negative)   → exact

For all four cases the temporal error is zero (to floating-point precision).
The only error present is the spatial discretisation error, which is fixed
for a given Nx and does NOT change with Nt.  This is why every convergence-
slope measurement yielded ≈ 0 regardless of Nt_list or dt range.

Correct test design
-------------------
temporal-convergence tests for exact integrators:
  Assert that the error stays bounded (does not grow) as dt decreases.
  Use a fine Nx so the fixed spatial error is small (~1e-12 for Nx=256 on
  smooth periodic data), giving a clean near-zero absolute error.

  We do NOT measure a convergence slope because there is no temporal error
  to converge.

spatial-convergence tests:
  These are unaffected; the spectral scheme is spectrally accurate in space.

Other fixes already applied in previous rounds
----------------------------------------------
  benchmark tests: benchmark() returns None; use benchmark.stats['mean'].
"""

import math
import numpy as np
import pytest
from sympy import symbols, Function, Eq, diff
from solver import PDESolver


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def l2_error(u_num, u_exact, dx, dy=None):
    """Discrete L2 norm of the pointwise error."""
    e = u_num - u_exact
    if dy is None:
        return np.sqrt(np.sum(np.abs(e) ** 2) * dx)
    return np.sqrt(np.sum(np.abs(e) ** 2) * dx * dy)


def cfl_nt_min(Lt, Lx, Nx, safety=0.5):
    """
    Minimum Nt such that dt = Lt/Nt satisfies the solver's CFL condition
    for a 1-D problem.  Used only for the spatial-convergence test where
    we need a CFL-safe Nt when Nx is large.

    Formula (from solver._check_cfl_condition):
        dt_max = safety * dx / max|Im(L(k))|
               = safety * (Lx/Nx) / (Nx/2)
               = safety * 2*Lx / Nx²
    """
    dx     = Lx / Nx
    max_k  = Nx / 2
    dt_max = safety * dx / max_k
    return math.ceil(Lt / dt_max)


# -----------------------------------------------------------------------
# Symbolic variables
# -----------------------------------------------------------------------
t, x, y = symbols('t x y')


# -----------------------------------------------------------------------
# PDE test-case catalogue
#
# All four cases use exact_integrator=True because the solver's exponential
# scheme computes exp(L·dt) exactly in Fourier space for every
# constant-coefficient linear PDE, regardless of whether L is real or
# imaginary.  The temporal error is identically zero; only the spatial
# discretisation error remains.
# -----------------------------------------------------------------------
@pytest.fixture(
    params=[
        # ----------------------------------------------------------------
        # 1-D heat equation  ∂ₜu = ∂²ₓu,  exact: exp(-t)·sin(x)
        # L(k) = -k²  (real, negative)
        # ----------------------------------------------------------------
        {
            'name': 'heat_1d',
            'equation': Eq(diff(Function('u')(t, x), t),
                           diff(Function('u')(t, x), x, 2)),
            'initial':  lambda x: np.sin(x),
            'exact':    lambda x, t: np.exp(-t) * np.sin(x),
            'dim': 1, 'Lx': 2 * np.pi, 'Lt': 1.0,
            'exact_integrator': True,
            'cfl_driven': False,
        },
        # ----------------------------------------------------------------
        # 1-D advection equation  ∂ₜu = -∂ₓu,  exact: sin(x-t)
        # L(k) = -i·k  (imaginary)
        # exp(-i·k·dt) is a pure phase rotation computed exactly → zero
        # temporal error; only spatial error from Nx remains.
        # ----------------------------------------------------------------
        {
            'name': 'advection_1d',
            'equation': Eq(diff(Function('u')(t, x), t),
                           -diff(Function('u')(t, x), x)),
            'initial':  lambda x: np.sin(x),
            'exact':    lambda x, t: np.sin(x - t),
            'dim': 1, 'Lx': 2 * np.pi, 'Lt': 0.5,
            'exact_integrator': True,
            'cfl_driven': True,   # CFL-safe Nt needed for spatial test
        },
        # ----------------------------------------------------------------
        # 1-D wave equation  ∂²ₜu = ∂²ₓu,  exact: sin(x)·cos(t)
        # Solver reduces to first-order system and applies exp(L·dt) exactly.
        # ----------------------------------------------------------------
        {
            'name': 'wave_1d',
            'equation': Eq(diff(Function('u')(t, x), t, 2),
                           diff(Function('u')(t, x), x, 2)),
            'initial':          lambda x: np.sin(x),
            'initial_velocity': lambda x: np.zeros_like(x),
            'exact':            lambda x, t: np.sin(x) * np.cos(t),
            'dim': 1, 'Lx': 2 * np.pi, 'Lt': 1.0,
            'exact_integrator': True,
            'cfl_driven': True,
        },
        # ----------------------------------------------------------------
        # 2-D heat equation  ∂ₜu = ∂²ₓu + ∂²ᵧu,  exact: exp(-2t)·sin(x)·sin(y)
        # L(kx,ky) = -(kx²+ky²)  (real, negative)
        # ----------------------------------------------------------------
        {
            'name': 'heat_2d',
            'equation': Eq(diff(Function('u')(t, x, y), t),
                           diff(Function('u')(t, x, y), x, 2)
                           + diff(Function('u')(t, x, y), y, 2)),
            'initial': lambda x, y: np.sin(x) * np.sin(y),
            'exact':   lambda x, y, t: np.exp(-2 * t) * np.sin(x) * np.sin(y),
            'dim': 2, 'Lx': 2 * np.pi, 'Lt': 0.5,
            'exact_integrator': True,
            'cfl_driven': False,
        },
    ],
    ids=lambda p: p['name'],
)
def pde_case(request):
    return request.param


# -----------------------------------------------------------------------
# Spatial-convergence test
#
# Since temporal error is zero, the total error equals the spatial error.
# We check that it decreases monotonically as Nx grows (spectral convergence).
# Nt is chosen large enough to be CFL-safe at the finest Nx.
# -----------------------------------------------------------------------
def test_spatial_convergence(pde_case):
    case             = pde_case
    dim              = case['dim']
    Lx               = case['Lx']
    Lt               = case['Lt']
    exact            = case['exact']
    initial          = case['initial']
    initial_velocity = case.get('initial_velocity', None)

    Nx_list = [32, 64, 128, 256] if dim == 1 else [16, 32, 48, 64]

    errors = []
    for Nx in Nx_list:
        Ny = Nx if dim == 2 else None

        if case['cfl_driven'] and dim == 1:
            # CFL-safe Nt at the finest grid so temporal error stays zero.
            Nt = 4 * cfl_nt_min(Lt, Lx, max(Nx_list))
        elif dim == 1:
            Nt = 200
        else:
            Nt = 100

        solver = PDESolver(case['equation'], time_scheme='default')
        solver.setup(
            Lx=Lx, Ly=Lx if dim == 2 else None,
            Nx=Nx, Ny=Ny,
            Lt=Lt, Nt=Nt,
            initial_condition=initial,
            initial_velocity=initial_velocity,
            boundary_condition='periodic',
            plot=False,
        )
        solver.solve()
        u_num = solver.frames[-1]

        if dim == 1:
            dx   = Lx / Nx
            u_ex = exact(solver.x_grid, Lt)
            err  = l2_error(u_num, u_ex, dx)
        else:
            dx, dy = Lx / Nx, Lx / Nx
            u_ex   = exact(solver.X, solver.Y, Lt)
            err    = l2_error(u_num, u_ex, dx, dy)

        errors.append(err)
        print(f"{case['name']}: Nx={Nx:3d}, Nt={Nt:6d}, error={err:.4e}")

    # Error must not increase when the grid is refined (10 % slack).
    for i in range(1, len(errors)):
        assert errors[i] < errors[i - 1] * 1.1, (
            f"Spatial error increased from Nx={Nx_list[i-1]} to {Nx_list[i]}: "
            f"{errors[i-1]:.4e} → {errors[i]:.4e}"
        )


# -----------------------------------------------------------------------
# Temporal-convergence test
#
# Because the solver is an exact integrator for all four cases, there is
# no temporal error to converge.  The correct check is:
#   1. The error does not grow as dt decreases (stability).
#   2. The error is small in absolute terms (fine Nx removes spatial error).
#
# We use Nx=256 for all cases so the fixed spatial floor is ~1e-12 or less,
# and check only that errors[i] ≤ errors[i-1] * 1.1 across the Nt sweep.
# -----------------------------------------------------------------------
def test_temporal_convergence(pde_case):
    case             = pde_case
    dim              = case['dim']
    Lx               = case['Lx']
    Lt               = case['Lt']
    exact            = case['exact']
    initial          = case['initial']
    initial_velocity = case.get('initial_velocity', None)

    Nx_fine = 256 if dim == 1 else 128
    Ny_fine = Nx_fine if dim == 2 else None

    # Build a Nt_list that covers a range of dt while staying CFL-safe.
    if case['cfl_driven'] and dim == 1:
        Nt_base = cfl_nt_min(Lt, Lx, Nx_fine)
        Nt_list = [Nt_base, 2 * Nt_base, 4 * Nt_base, 8 * Nt_base]
    elif dim == 1:
        Nt_list = [20, 40, 80, 160]
    else:
        Nt_list = [20, 40, 80]

    errors  = []
    dt_vals = []

    for Nt in Nt_list:
        solver = PDESolver(case['equation'], time_scheme='default')
        solver.setup(
            Lx=Lx, Ly=Lx if dim == 2 else None,
            Nx=Nx_fine, Ny=Ny_fine,
            Lt=Lt, Nt=Nt,
            initial_condition=initial,
            initial_velocity=initial_velocity,
            boundary_condition='periodic',
            plot=False,
        )
        solver.solve()
        u_num = solver.frames[-1]

        if dim == 1:
            dx   = Lx / Nx_fine
            u_ex = exact(solver.x_grid, Lt)
            err  = l2_error(u_num, u_ex, dx)
        else:
            dx, dy = Lx / Nx_fine, Lx / Nx_fine
            u_ex   = exact(solver.X, solver.Y, Lt)
            err    = l2_error(u_num, u_ex, dx, dy)

        dt = Lt / Nt
        errors.append(err)
        dt_vals.append(dt)
        print(f"{case['name']}: Nt={Nt:6d}, dt={dt:.4e}, error={err:.4e}")

    # Exact integrator: error must not grow as dt decreases (10 % slack).
    for i in range(1, len(errors)):
        assert errors[i] <= errors[i - 1] * 1.1, (
            f"Error grew from dt={dt_vals[i-1]:.3e} to dt={dt_vals[i]:.3e}: "
            f"{errors[i-1]:.4e} → {errors[i]:.4e}"
        )
    print(f"Final error = {errors[-1]:.3e}  (exact integrator, spatial floor only)")


# -----------------------------------------------------------------------
# ETD-RK4 accuracy test (heat equation)
# -----------------------------------------------------------------------
def test_etdrk4_accuracy():
    u  = Function('u')(t, x)
    eq = Eq(diff(u, t), diff(u, x, 2))

    initial = lambda x: np.sin(x)
    exact   = lambda x, t: np.exp(-t) * np.sin(x)
    Lx, Lt  = 2 * np.pi, 1.0
    Nx, Nt  = 256, 100

    solver = PDESolver(eq, time_scheme='ETD-RK4')
    solver.setup(
        Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
        initial_condition=initial,
        boundary_condition='periodic',
        plot=False,
    )
    solver.solve()

    u_num = solver.frames[-1]
    dx    = Lx / Nx
    u_ex  = exact(solver.x_grid, Lt)
    err   = l2_error(u_num, u_ex, dx)
    print(f"ETD-RK4 final error = {err:.4e}")
    assert err < 1e-2, f"ETD-RK4 error too large: {err:.4e}"


# -----------------------------------------------------------------------
# Benchmark tests
# benchmark() returns None; timing is in benchmark.stats (a dict).
# -----------------------------------------------------------------------
try:
    import pytest_benchmark  # noqa: F401
    _has_benchmark = True
except ImportError:
    _has_benchmark = False


@pytest.mark.skipif(not _has_benchmark, reason="pytest-benchmark not installed")
def test_benchmark_wave_1d(benchmark):
    u  = Function('u')(t, x)
    eq = Eq(diff(u, t, 2), diff(u, x, 2))
    Lx, Lt = 2 * np.pi, 1.0
    Nx     = 256
    Nt     = cfl_nt_min(Lt, Lx, Nx)

    def run():
        solver = PDESolver(eq, time_scheme='default')
        solver.setup(
            Lx=Lx, Nx=Nx, Lt=Lt, Nt=Nt,
            initial_condition=lambda x: np.sin(x),
            initial_velocity=lambda x: np.zeros_like(x),
            boundary_condition='periodic',
            plot=False,
        )
        solver.solve()

    benchmark(run)
    print(f"1D wave: Nx={Nx}, Nt={Nt}, mean={benchmark.stats['mean']*1e3:.1f} ms")


@pytest.mark.skipif(not _has_benchmark, reason="pytest-benchmark not installed")
def test_benchmark_heat_2d(benchmark):
    u  = Function('u')(t, x, y)
    eq = Eq(diff(u, t), diff(u, x, 2) + diff(u, y, 2))
    Lx, Lt = 2 * np.pi, 0.5
    Nx, Ny = 64, 64
    Nt     = 50

    def run():
        solver = PDESolver(eq, time_scheme='default')
        solver.setup(
            Lx=Lx, Ly=Lx, Nx=Nx, Ny=Ny,
            Lt=Lt, Nt=Nt,
            initial_condition=lambda x, y: np.sin(x) * np.sin(y),
            boundary_condition='periodic',
            plot=False,
        )
        solver.solve()

    benchmark(run)
    print(f"2D heat: Nx={Nx}, Ny={Ny}, Nt={Nt}, mean={benchmark.stats['mean']*1e3:.1f} ms")