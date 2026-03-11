# Copyright 2026 Philippe Billet assisted by LLMs.
# Licensed under the Apache License, Version 2.0.
"""
propagator.py — Semiclassical (Van Vleck–Pauli–Morette) wavefunction
=====================================================================

Rebuilt on riemannian.py + symplectic.py exclusively.  geometry.py is
not used.

Physics
-------
    ψ(x, t) = Σ_k  exp(i S_k/ℏ − i μ_k π/2) / √|det J_k|

Package roles
-------------
riemannian.py
    Metric           — encodes the geometry (= kinetic part of H)
    geodesic_solver  — integrates rays (position + velocity)
    jacobi_equation_solver — integrates Jacobi fields → det J → caustics

symplectic.py
    hamiltonian_flow — high-accuracy ray integration with symplectic
                       integrators; also used for action ∫ p dq via
                       the momentum arrays it returns
    action_integral  — closed-orbit action (used for periodic orbits)

New unique contribution of this module
---------------------------------------
    van_vleck_sum    — assembles ψ from (pts, S, det_J, μ) scattered data
                       onto a regular grid via scipy.interpolate.griddata
"""

from __future__ import annotations

import sys
import types
import builtins

# ── psipy shim: riemannian.py / symplectic.py both do `from imports import *`
# ── but there is no imports.py on PYTHONPATH.  We create a minimal shim that
# ── re-exports everything they actually need.
def _make_imports_shim():
    import numpy as np_
    import sympy as sp_
    import scipy.integrate, scipy.optimize, scipy.interpolate, scipy.stats
    m = types.ModuleType('imports')
    # scipy first, then sympy overwrites conflicting names (log, exp, …)
    for mod in (scipy.integrate, scipy.optimize, scipy.interpolate, scipy.stats):
        m.__dict__.update({k: getattr(mod, k) for k in dir(mod)
                           if not k.startswith('_')})
    m.__dict__.update(sp_.__dict__)
    m.np = np_
    m.numpy = np_
    sys.modules.setdefault('imports', m)
    builtins.np = np_     # symplectic uses bare 'np' inside functions

_make_imports_shim()

import numpy as np
import sympy as sp
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ── psipy imports ─────────────────────────────────────────────────────────────
from riemannian import Metric, geodesic_solver, jacobi_equation_solver
from symplectic import hamiltonian_flow
from asymptotic import (
    Analyzer, AsymptoticEvaluator,
    IntegralMethod, SingularityType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RayData:
    """Enriched ray: trajectory + Jacobi field det + cumulative action."""
    traj    : dict          # output of geodesic_solver / hamiltonian_flow
    det_J   : np.ndarray   # shape (n_steps,)  — det of Jacobi 2×2 matrix
    S_cum   : np.ndarray   # shape (n_steps,)  — cumulative action ∫ p dx
    mu      : int           # Maslov index (number of det J sign changes)


@dataclass
class WKBResult:
    """Output of compute_wavefunction."""
    rays      : List[RayData]
    X         : np.ndarray          # (N,) 1D or (N,N) 2D
    Y         : Optional[np.ndarray]# None in 1D
    psi       : np.ndarray          # complex
    # scattered raw data (all rays concatenated)
    x_pts     : np.ndarray
    y_pts     : Optional[np.ndarray]
    S_pts     : np.ndarray
    det_J_pts : np.ndarray
    mu_pts    : np.ndarray
    hbar      : float
    t_max     : float
    dim       : int                  # 1 or 2


# ─────────────────────────────────────────────────────────────────────────────
# 1 — Jacobi matrix determinant  (uses riemannian.jacobi_equation_solver)
# ─────────────────────────────────────────────────────────────────────────────

def _det_J_1d(metric: Metric, traj: dict,
              tspan: tuple, n_steps: int) -> np.ndarray:
    """
    1D Jacobi scalar J = ∂x/∂p₀  via the variational ODE
        dJ/dt = g_inv(x) * K
        dK/dt = -½ (d g_inv / dx) * p * J    (linearised geodesic)
    with J(0)=0, K(0)=1.

    Uses scipy.integrate.solve_ivp directly (riemannian has no 1D Jacobi).
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    x_sym = metric.coords[0]
    g_inv_prime = sp.lambdify(x_sym,
                              sp.diff(metric.g_inv_expr, x_sym), 'numpy')

    x_interp = interp1d(traj['t'], traj['x'], kind='linear')
    v_interp = interp1d(traj['t'], traj['v'], kind='linear')

    def jac_ode(t, state):
        J, K = state
        xv = float(x_interp(t))
        vv = float(v_interp(t))
        g_i = float(metric.g_inv_func(xv))
        gp  = float(g_inv_prime(xv))
        dJ  = g_i * K
        dK  = -0.5 * gp * vv * J
        return [dJ, dK]

    sol = solve_ivp(jac_ode, tspan, [0.0, 1.0],
                    t_eval=np.linspace(tspan[0], tspan[1], n_steps),
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y[0]


def _det_J_from_jacobi(metric: Metric, traj: dict,
                        tspan: tuple, n_steps: int) -> np.ndarray:
    """
    Compute det(J) along a ray.

    1D: variational ODE solved locally (riemannian.jacobi_equation_solver
        is 2D-only).
    2D: two Jacobi fields via riemannian.jacobi_equation_solver.
        ICs J(0)=0, J'(0)=eᵢ correspond to a point-source fan.
    """
    if metric.dim == 1:
        return _det_J_1d(metric, traj, tspan, n_steps)

    # 2D: two independent Jacobi fields → 2×2 matrix → det
    jac1 = jacobi_equation_solver(
        metric, traj,
        initial_variation={'J0': (0.0, 0.0), 'DJ0': (1.0, 0.0)},
        tspan=tspan, n_steps=n_steps,
    )
    jac2 = jacobi_equation_solver(
        metric, traj,
        initial_variation={'J0': (0.0, 0.0), 'DJ0': (0.0, 1.0)},
        tspan=tspan, n_steps=n_steps,
    )
    return jac1['J_x'] * jac2['J_y'] - jac1['J_y'] * jac2['J_x']


# ─────────────────────────────────────────────────────────────────────────────
# 2 — Cumulative action  (uses symplectic.hamiltonian_flow momentum arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _cumulative_action(traj: dict, dim: int) -> np.ndarray:
    """
    S(t) = ∫₀ᵗ p · (dx/dt') dt'   using the momentum arrays from
    symplectic.hamiltonian_flow (keys 'p' in 1D, 'px'/'py' in 2D).

    Falls back to velocity proxy when no momentum key is present
    (e.g. result from geodesic_solver without Hamiltonian formulation).
    """
    t = traj['t']
    dt = np.gradient(t)
    if dim == 1:
        if 'p' in traj:
            p  = traj['p']
            vx = traj.get('v', np.gradient(traj['x'], t))
            integrand = p * vx
        else:
            # fallback: kinetic approximation
            integrand = traj['v'] ** 2
        return np.cumsum(integrand * dt)
    else:
        if 'px' in traj and 'py' in traj:
            px, py = traj['px'], traj['py']
            vx = traj.get('vx', np.gradient(traj['x'], t))
            vy = traj.get('vy', np.gradient(traj['y'], t))
            integrand = px * vx + py * vy
        else:
            integrand = traj['vx'] ** 2 + traj['vy'] ** 2
        return np.cumsum(integrand * dt)


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Maslov index  (sign-change count on det J)
# ─────────────────────────────────────────────────────────────────────────────

def _maslov_index(det_J: np.ndarray) -> int:
    """Count sign changes of det J (each = one caustic crossing, μ += 1)."""
    signs = np.sign(det_J)
    signs = signs[signs != 0]              # ignore exact zeros
    return int(np.sum(np.abs(np.diff(signs)) > 0))


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Van Vleck coherent sum  (the unique new contribution)
# ─────────────────────────────────────────────────────────────────────────────

def _asymptotic_correction_1d(
    x_caustic : float,
    S_caustic : float,
    a_caustic : float,
    dJ_ds     : float,
    hbar      : float,
    x_grid    : np.ndarray,
    width     : float,
) -> np.ndarray:
    """
    Replace the WKB amplitude near a 1D fold caustic with the Airy uniform
    approximation from asymptotic.Analyzer + AsymptoticEvaluator.

    The local phase near the fold is  φ(p) = S(p)/ℏ  with  S ~ S_c + α(p-p_c)³/3.
    asymptotic.Analyzer is initialised with this cubic phase and unit amplitude;
    the Airy evaluator returns the correct O(λ^{-1/3}) scaling automatically.

    Returns  patch  : complex array, same shape as x_grid, zero outside [x_c ± width].
    """
    patch = np.zeros_like(x_grid, dtype=complex)
    mask  = np.abs(x_grid - x_caustic) < width
    if not np.any(mask):
        return patch

    x_local = x_grid[mask] - x_caustic
    lam     = 1.0 / hbar                    # large parameter λ = 1/ℏ

    # Build the cubic phase  φ(t) = α t³/3  with α chosen so that the
    # Hessian vanishes at t=0 and ∂³φ/∂t³ = α = dJ_ds (slope of det J).
    alpha = float(dJ_ds) if abs(dJ_ds) > 1e-12 else 1.0
    t_sym = sp.Symbol('t', real=True)
    phase_sym = sp.Rational(1, 3) * alpha * t_sym**3  # cubic normal form

    analyzer  = Analyzer(
        phase_expr     = phase_sym,
        amplitude_expr = sp.Integer(1),
        variables      = [t_sym],
        method         = IntegralMethod.STATIONARY_PHASE,
    )
    evaluator = AsymptoticEvaluator()

    # The single critical point of the cubic is at t=0
    xc = np.array([0.0])
    cp = analyzer.analyze_point(xc)     # → SingularityType.AIRY_1D
    contrib = evaluator.evaluate(cp, lam)

    # Uniform Airy value (scalar) scaled by amplitude at caustic and carrier phase
    airy_val = contrib.total_value * a_caustic * np.exp(1j * S_caustic / hbar)

    # Taper smoothly to zero at the patch edges
    taper = np.cos(np.pi / 2 * x_local / width) ** 2
    patch[mask] = airy_val * taper

    return patch


def van_vleck_sum(
    pts    : np.ndarray,             # (M, 1) or (M, 2)
    S      : np.ndarray,             # (M,)
    det_J  : np.ndarray,             # (M,)
    mu     : np.ndarray,             # (M,) integer
    xlim   : Tuple[float, float],
    ylim   : Optional[Tuple[float, float]] = None,
    N      : int   = 300,
    hbar   : float = 1.0,
    reg    : float = 1e-4,
    method : str   = "linear",
    caustic_threshold : float = 0.05,  # |det J| / max|det J| below this → Airy patch
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Hybrid Van Vleck assembler: griddata everywhere, asymptotic.py at caustics.

    Strategy
    --------
    1. **Regular points** (|det J| large):
       Standard WKB amplitude  1/√|det J|  interpolated onto the grid via
       scipy.interpolate.griddata — fast, O(M log M).

    2. **Caustic points** (|det J| / max|det J| < caustic_threshold):
       The 1/√|det J| amplitude diverges.  Instead, asymptotic.Analyzer is
       initialised with the local cubic normal-form phase (∝ t³) and
       AsymptoticEvaluator dispatches to StationaryPhaseEvaluator._eval_airy_1d,
       which returns the correct O(λ^{-1/3}) Airy scaling.  The Airy patch is
       blended into the grid over a smoothly tapered window.

    Why this split?
    ---------------
    asymptotic.py evaluates  I(λ) = ∫ a(t) e^{iλφ(t)} dt  at a *single point*
    given a symbolic phase.  Repeating this for every grid cell (N² times) would
    be prohibitively expensive.  The hybrid uses it only where it matters: inside
    the caustic zone, which is a small fraction of the grid.

    Parameters
    ----------
    pts               : scattered ray positions, shape (M, 1) or (M, 2)
    S                 : accumulated action at each scattered point
    det_J             : Jacobi determinant at each scattered point
    mu                : Maslov index at each scattered point
    xlim, ylim        : grid bounds (ylim=None → 1D mode)
    N                 : grid resolution
    hbar              : reduced Planck constant
    reg               : amplitude regularisation floor (avoids 1/0 at exact zeros)
    method            : griddata interpolation method ('linear', 'cubic', 'nearest')
    caustic_threshold : relative |det J| threshold for Airy patching

    Returns
    -------
    psi : complex ndarray  (N,) in 1D or (N, N) in 2D
    X   : x-grid coordinates
    Y   : y-grid coordinates (None in 1D)
    """
    # ── standard WKB amplitude everywhere (regularised) ──────────────────────
    abs_det   = np.abs(det_J)
    amp       = 1.0 / np.sqrt(np.maximum(abs_det, reg))
    psi_k     = amp * np.exp(1j * S / hbar - 1j * mu * np.pi / 2)

    # ── identify caustic scattered points ────────────────────────────────────
    det_max   = abs_det.max() if abs_det.max() > 0 else 1.0
    near_caus = abs_det < caustic_threshold * det_max

    if ylim is None:
        # ════════════════════ 1D ════════════════════════════════════════════
        x_grid = np.linspace(*xlim, N)
        order  = np.argsort(pts[:, 0])
        xs     = pts[order, 0]
        pk_ord = psi_k[order]

        psi = (np.interp(x_grid, xs, pk_ord.real, left=0, right=0)
             + 1j * np.interp(x_grid, xs, pk_ord.imag, left=0, right=0))

        # ── Airy patches at each detected caustic cluster ─────────────────
        if np.any(near_caus):
            caus_xs = pts[near_caus, 0]
            # cluster caustic positions that are within 2% of grid span
            span    = xlim[1] - xlim[0]
            for xc in caus_xs[np.argsort(caus_xs)]:
                # representative S and amplitude at this caustic
                idx_c    = np.argmin(np.abs(pts[:, 0] - xc))
                S_c      = float(S[idx_c])
                a_c      = float(amp[idx_c]) * float(det_max) ** 0.5  # undo 1/√det
                # dJ/ds ≈ slope of det_J near the caustic
                nearby   = np.abs(pts[:, 0] - xc) < 0.05 * span
                if nearby.sum() >= 2:
                    dJ_ds = float(np.gradient(det_J[nearby],
                                              pts[nearby, 0]).mean())
                else:
                    dJ_ds = 1.0
                width = max(0.04 * span, 3 * (x_grid[1] - x_grid[0]))
                patch = _asymptotic_correction_1d(
                    xc, S_c, a_c, dJ_ds, hbar, x_grid, width)
                # blend: replace WKB inside the Airy window
                blend   = np.abs(patch) > 0
                psi[blend] = patch[blend]

        return psi, x_grid, None

    else:
        # ════════════════════ 2D ════════════════════════════════════════════
        xs, ys = np.linspace(*xlim, N), np.linspace(*ylim, N)
        X, Y   = np.meshgrid(xs, ys)
        grid   = np.c_[X.ravel(), Y.ravel()]
        kw     = dict(points=pts, xi=grid, method=method, fill_value=0.0)
        psi    = (griddata(values=psi_k.real, **kw)
                + 1j * griddata(values=psi_k.imag, **kw)).reshape(N, N)
        # 2D caustic patching: asymptotic.py Airy / Pearcey corrections would
        # require a 2D Analyzer per caustic cell — expensive.  The divergence
        # is already suppressed by `reg`; a future extension can add
        # _eval_airy_2d / _eval_pearcey patches following the same pattern.
        return psi, X, Y


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _build_hamiltonian_sym(metric: Metric) -> Tuple[sp.Expr, list]:
    """
    Build H = ½ gⁱʲ pᵢ pⱼ from the Metric object.
    Returns (H_expr, vars_phase) in the form [x, p] or [x, px, y, py].
    """
    if metric.dim == 1:
        x    = metric.coords[0]
        p    = sp.Symbol('p', real=True)
        H    = metric.g_inv_expr * p**2 / 2
        return H, [x, p]
    else:
        x, y   = metric.coords
        px, py = sp.symbols('px py', real=True)
        g_inv  = metric.g_inv_matrix
        H      = (g_inv[0, 0] * px**2
                + 2 * g_inv[0, 1] * px * py
                + g_inv[1, 1] * py**2) / 2
        return H, [x, px, y, py]


def compute_wavefunction(
    metric       : Metric,
    source       : Tuple,                # (x0,) or (x0, y0)
    p_fan        : np.ndarray,           # (n_rays,) 1D or (n_rays, 2) 2D
    t_max        : float,
    hbar         : float = 1.0,
    n_steps      : int   = 400,
    N_grid       : int   = 300,
    xlim         : Optional[Tuple] = None,
    ylim         : Optional[Tuple] = None,
    integrator   : str   = 'verlet',     # passed to symplectic.hamiltonian_flow
) -> WKBResult:
    """
    Full semiclassical wavefunction pipeline using riemannian + symplectic.

    Parameters
    ----------
    metric     : riemannian.Metric  (encodes the geometry)
    source     : initial position (x0,) or (x0, y0)
    p_fan      : initial momenta — shape (n_rays,) for 1D,
                                         (n_rays, 2) for 2D
    t_max      : integration time
    hbar       : reduced Planck constant
    n_steps    : time steps per ray
    N_grid     : output grid resolution
    xlim, ylim : grid bounds (auto-detected from ray endpoints if None)
    integrator : 'verlet' (default, symplectic), 'rk45' (higher accuracy)

    Returns
    -------
    WKBResult
    """
    dim    = metric.dim
    tspan  = (0.0, t_max)
    H_sym, vars_phase = _build_hamiltonian_sym(metric)

    rays = []
    for p0 in p_fan:
        try:
            # ── initial state for symplectic.hamiltonian_flow ──────────────
            if dim == 1:
                # convert velocity to momentum: p = g(x0) * v0
                g0  = float(metric.g_func(source[0]))
                mom = float(g0 * p0)
                z0  = [source[0], mom]
            else:
                # p = g · v
                g0  = metric.eval(source[0], source[1])['g']
                mom = g0 @ np.array(p0, dtype=float)
                z0  = [source[0], float(mom[0]), source[1], float(mom[1])]

            # ── ray integration (symplectic) ────────────────────────────────
            traj = hamiltonian_flow(
                H_sym, z0, tspan,
                vars_phase=vars_phase,
                integrator=integrator,
                n_steps=n_steps,
            )

            # ── Jacobi determinant (riemannian) ────────────────────────────
            # geodesic_solver gives the traj dict format jacobi_equation_solver
            # expects; we rebuild it from hamiltonian_flow output.
            if dim == 1:
                x_sym = vars_phase[0]
                p_sym = vars_phase[1]
                geo_traj = {
                    't'  : traj['t'],
                    'x'  : traj[str(x_sym)],
                    'v'  : metric.g_inv_func(traj[str(x_sym)])
                           * traj[str(p_sym)],
                }
            else:
                x_sym, px_sym, y_sym, py_sym = vars_phase
                x_arr = traj[str(x_sym)];  y_arr = traj[str(y_sym)]
                px_arr = traj[str(px_sym)]; py_arr = traj[str(py_sym)]
                g00 = metric.g_inv_func[(0, 0)](x_arr, y_arr)
                g01 = metric.g_inv_func[(0, 1)](x_arr, y_arr)
                g10 = metric.g_inv_func[(1, 0)](x_arr, y_arr)
                g11 = metric.g_inv_func[(1, 1)](x_arr, y_arr)
                geo_traj = {
                    't'  : traj['t'],
                    'x'  : x_arr, 'y' : y_arr,
                    'vx' : g00 * px_arr + g01 * py_arr,
                    'vy' : g10 * px_arr + g11 * py_arr,
                }

            det_J = _det_J_from_jacobi(metric, geo_traj, tspan, n_steps)
            S_cum = _cumulative_action(traj, dim)
            mu    = _maslov_index(det_J)

            rays.append(RayData(traj=traj, det_J=det_J, S_cum=S_cum, mu=mu))

        except Exception as e:
            continue   # skip failed rays silently

    if not rays:
        raise RuntimeError("All rays failed to integrate.")

    # ── collect scattered data ────────────────────────────────────────────────
    if dim == 1:
        x_sym_str = str(vars_phase[0])
        x_all  = np.concatenate([r.traj[x_sym_str] for r in rays])
        S_all  = np.concatenate([r.S_cum            for r in rays])
        dJ_all = np.concatenate([r.det_J            for r in rays])
        mu_all = np.concatenate([np.full(len(r.det_J), r.mu) for r in rays])
        pts    = x_all[:, None]
        if xlim is None:
            m = 0.1 * (x_all.max() - x_all.min())
            xlim = (x_all.min() - m, x_all.max() + m)
        psi, X, Y = van_vleck_sum(pts, S_all, dJ_all, mu_all,
                                   xlim=xlim, N=N_grid, hbar=hbar)
        return WKBResult(rays=rays, X=X, Y=Y, psi=psi,
                         x_pts=x_all, y_pts=None,
                         S_pts=S_all, det_J_pts=dJ_all, mu_pts=mu_all,
                         hbar=hbar, t_max=t_max, dim=1)

    else:
        x_sym_str = str(vars_phase[0])
        y_sym_str = str(vars_phase[2])
        x_all  = np.concatenate([r.traj[x_sym_str] for r in rays])
        y_all  = np.concatenate([r.traj[y_sym_str] for r in rays])
        S_all  = np.concatenate([r.S_cum            for r in rays])
        dJ_all = np.concatenate([r.det_J            for r in rays])
        mu_all = np.concatenate([np.full(len(r.det_J), r.mu) for r in rays])
        pts    = np.c_[x_all, y_all]
        if xlim is None:
            mx = 0.1 * (x_all.max() - x_all.min())
            xlim = (x_all.min() - mx, x_all.max() + mx)
        if ylim is None:
            my = 0.1 * (y_all.max() - y_all.min())
            ylim = (y_all.min() - my, y_all.max() + my)
        psi, X, Y = van_vleck_sum(pts, S_all, dJ_all, mu_all,
                                   xlim=xlim, ylim=ylim, N=N_grid, hbar=hbar)
        return WKBResult(rays=rays, X=X, Y=Y, psi=psi,
                         x_pts=x_all, y_pts=y_all,
                         S_pts=S_all, det_J_pts=dJ_all, mu_pts=mu_all,
                         hbar=hbar, t_max=t_max, dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_BG   = "#0e0e1a"
_DARK = "#444"

def _style(fig, axes):
    """Apply dark theme to all axes."""
    fig.patch.set_facecolor(_BG)
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(_BG)
        ax.tick_params(colors="white", labelsize=7)
        for lbl in (ax.xaxis.label, ax.yaxis.label, ax.title):
            lbl.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor(_DARK)


def plot_wavefunction(result: WKBResult, log_scale=True,
                      save_path=None) -> plt.Figure:
    """
    Master figure:
    1D → 4 panels: density · phase · Re/Im · ray fan
    2D → 5 panels: density · phase · ray fan+caustics · det J · Maslov
    """
    if result.dim == 1:
        return _plot_1d(result, log_scale, save_path)
    return _plot_2d(result, log_scale, save_path)


def _plot_1d(result: WKBResult, log_scale: bool, save_path) -> plt.Figure:
    x, psi = result.X, result.psi
    den    = np.log1p(np.abs(psi)**2) if log_scale else np.abs(psi)**2
    dlabel = r"$\log(1+|\psi|^2)$" if log_scale else r"$|\psi|^2$"

    fig = plt.figure(figsize=(16, 8))
    gs  = GridSpec(2, 4, fig, hspace=0.45, wspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.fill_between(x, den, alpha=0.8, color=plt.cm.inferno(0.65))
    ax0.plot(x, den, lw=0.9, color="white", alpha=0.55)
    ax0.set(title=dlabel, xlabel="$x$", ylabel=dlabel)

    ax1 = fig.add_subplot(gs[0, 2:4])
    ax1.plot(x, np.angle(psi), color=plt.cm.hsv(0.28), lw=1.1)
    ax1.axhline(0, color="white", lw=0.5, ls="--")
    ax1.set(title=r"Phase $\arg(\psi)$", xlabel="$x$", ylabel="rad",
            ylim=(-np.pi - 0.3, np.pi + 0.3))

    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.plot(x, psi.real, lw=1.0, color="#4fc3f7", label=r"Re $\psi$")
    ax2.plot(x, psi.imag, lw=1.0, color="#ef9a9a", label=r"Im $\psi$", alpha=0.8)
    ax2.axhline(0, color="white", lw=0.4, ls="--")
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.set(title=r"Re / Im $\psi$", xlabel="$x$")

    ax3 = fig.add_subplot(gs[1, 2:4])
    vars_str = list(result.rays[0].traj.keys())
    x_key = [k for k in vars_str if k not in ('t', 'energy') and 'p' not in k][0]
    for ray in result.rays:
        c = plt.cm.plasma(0.3 + 0.5 * float(np.mean(np.abs(ray.det_J)))
                          / (float(np.mean(np.abs(ray.det_J))) + 1.0))
        ax3.plot(ray.traj['t'], ray.traj[x_key], lw=0.5, alpha=0.3, color=c)
    ax3.set(title="Ray fan  $x(t)$", xlabel="$t$", ylabel="$x$")

    fig.suptitle(
        rf"Van Vleck wavefunction  ($\hbar={result.hbar}$, "
        rf"$t_{{max}}={result.t_max}$, {len(result.rays)} rays)",
        color="white", fontsize=11, fontweight="bold", y=1.01)
    _style(fig, fig.axes)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def _plot_2d(result: WKBResult, log_scale: bool, save_path) -> plt.Figure:
    X, Y, psi = result.X, result.Y, result.psi
    den    = np.log1p(np.abs(psi)**2) if log_scale else np.abs(psi)**2
    dlabel = r"$\log(1+|\psi|^2)$" if log_scale else r"$|\psi|^2$"

    fig = plt.figure(figsize=(20, 8))
    gs  = GridSpec(2, 3, fig, hspace=0.42, wspace=0.32)

    # density
    ax0 = fig.add_subplot(gs[0, 0:2])
    im0 = ax0.pcolormesh(X, Y, den, cmap="inferno", shading="auto")
    fig.colorbar(im0, ax=ax0, label=dlabel, pad=0.02)
    ax0.set_aspect("equal")
    ax0.set(title=dlabel, xlabel="$x$", ylabel="$y$")

    # phase
    ax1 = fig.add_subplot(gs[0, 2])
    im1 = ax1.pcolormesh(X, Y, np.angle(psi), cmap="hsv",
                          shading="auto", vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im1, ax=ax1, label=r"$\arg(\psi)$", pad=0.02)
    ax1.set_aspect("equal")
    ax1.set(title=r"Phase  $\arg(\psi)$", xlabel="$x$", ylabel="$y$")

    # ray fan + caustics
    ax2 = fig.add_subplot(gs[1, 0])
    vars_str = list(result.rays[0].traj.keys())
    x_key = [k for k in vars_str if k not in ('t','energy') and 'p' not in k][0]
    y_key = [k for k in vars_str if k not in ('t','energy') and 'p' not in k
             and k != x_key][0]
    cmap_r = plt.cm.cool
    n_r = max(len(result.rays) - 1, 1)
    for i, ray in enumerate(result.rays):
        ax2.plot(ray.traj[x_key], ray.traj[y_key],
                 lw=0.5, alpha=0.3, color=cmap_r(i / n_r))
        # caustic positions (sign changes of det J)
        signs = np.sign(ray.det_J)
        cidx  = np.where(np.diff(signs) != 0)[0]
        if len(cidx):
            ax2.scatter(ray.traj[x_key][cidx], ray.traj[y_key][cidx],
                        s=10, color="yellow", zorder=5, alpha=0.7)
    ax2.set_aspect("equal")
    ax2.set(title="Ray fan  +  caustics (yellow)", xlabel="$x$", ylabel="$y$")

    # det J scatter
    ax3 = fig.add_subplot(gs[1, 1])
    sc3 = ax3.scatter(result.x_pts, result.y_pts,
                      c=np.log1p(np.abs(result.det_J_pts)),
                      cmap="plasma", s=0.8, alpha=0.45, rasterized=True)
    fig.colorbar(sc3, ax=ax3, label=r"$\log(1+|\det J|)$", pad=0.02)
    ax3.set_aspect("equal")
    ax3.set(title=r"Jacobian $|\det J|$", xlabel="$x$", ylabel="$y$")

    # Maslov
    ax4 = fig.add_subplot(gs[1, 2])
    mu  = result.mu_pts.astype(float)
    sc4 = ax4.scatter(result.x_pts, result.y_pts,
                      c=mu, cmap="RdBu_r", s=0.8, alpha=0.45,
                      vmin=mu.min(), vmax=mu.max(), rasterized=True)
    fig.colorbar(sc4, ax=ax4, label=r"Maslov $\mu$", pad=0.02)
    ax4.set_aspect("equal")
    ax4.set(title=r"Maslov index $\mu$", xlabel="$x$", ylabel="$y$")

    fig.suptitle(
        rf"Van Vleck wavefunction 2D  ($\hbar={result.hbar}$, "
        rf"$t_{{max}}={result.t_max}$, {len(result.rays)} rays)",
        color="white", fontsize=11, fontweight="bold")
    _style(fig, fig.axes)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_ray_fan(result: WKBResult, save_path=None) -> plt.Figure:
    """Ray fan coloured by total accumulated action."""
    is2d = (result.dim == 2)
    fig, ax = plt.subplots(figsize=(10, 6))

    S_finals = np.array([r.S_cum[-1] for r in result.rays])
    S_norm   = (S_finals - S_finals.min()) / (np.ptp(S_finals) + 1e-30)

    vars_str = list(result.rays[0].traj.keys())
    x_key = [k for k in vars_str if k not in ('t','energy') and 'p' not in k][0]
    if is2d:
        y_key = [k for k in vars_str if k not in ('t','energy') and 'p' not in k
                 and k != x_key][0]

    for i, ray in enumerate(result.rays):
        c = plt.cm.viridis(S_norm[i])
        if is2d:
            ax.plot(ray.traj[x_key], ray.traj[y_key],
                    lw=0.7, alpha=0.4, color=c)
            signs = np.sign(ray.det_J)
            cidx  = np.where(np.diff(signs) != 0)[0]
            if len(cidx):
                ax.scatter(ray.traj[x_key][cidx], ray.traj[y_key][cidx],
                           s=14, color="yellow", zorder=5, alpha=0.8)
        else:
            ax.plot(ray.traj['t'], ray.traj[x_key], lw=0.7, alpha=0.4, color=c)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                                norm=mcolors.Normalize(S_finals.min(),
                                                       S_finals.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Action $S$")
    ax.set_aspect("equal" if is2d else "auto")
    ax.set(title="Ray fan coloured by action  (yellow = caustic)",
           xlabel="$x$" if is2d else "$t$",
           ylabel="$y$" if is2d else "$x$")
    _style(fig, ax)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_interference_detail(result: WKBResult,
                              save_path=None) -> plt.Figure:
    """Three-panel: Re(ψ) fringes · |ψ|² · scattered action vs position."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if result.dim == 1:
        x, psi = result.X, result.psi
        axes[0].plot(x, psi.real, lw=1.0, color="#80cbc4")
        axes[0].fill_between(x, psi.real, alpha=0.3, color="#80cbc4")
        axes[0].set(title=r"Re $\psi$  (interference fringes)", xlabel="$x$")

        den = np.abs(psi)**2
        axes[1].fill_between(x, den, alpha=0.85, color=plt.cm.inferno(0.6))
        axes[1].plot(x, den, lw=0.8, color="white", alpha=0.5)
        axes[1].set(title=r"$|\psi|^2$", xlabel="$x$")

        sc = axes[2].scatter(result.x_pts, result.S_pts,
                             c=result.mu_pts, cmap="RdYlGn", s=0.5,
                             alpha=0.4, rasterized=True)
        fig.colorbar(sc, ax=axes[2], label=r"Maslov $\mu$")
        axes[2].set(title=r"Action $S(x)$  (colour = $\mu$)",
                    xlabel="$x$", ylabel="$S$")
    else:
        X, Y, psi = result.X, result.Y, result.psi
        im0 = axes[0].pcolormesh(X, Y, psi.real, cmap="RdBu_r", shading="auto")
        fig.colorbar(im0, ax=axes[0], label=r"Re $\psi$")
        axes[0].set_aspect("equal")
        axes[0].set(title=r"Re $\psi$  (interference fringes)",
                    xlabel="$x$", ylabel="$y$")

        im1 = axes[1].pcolormesh(X, Y, np.abs(psi)**2,
                                  cmap="inferno", shading="auto")
        fig.colorbar(im1, ax=axes[1], label=r"$|\psi|^2$")
        axes[1].set_aspect("equal")
        axes[1].set(title=r"$|\psi|^2$", xlabel="$x$", ylabel="$y$")

        sc = axes[2].scatter(result.x_pts, result.S_pts,
                             c=result.mu_pts, cmap="RdYlGn", s=0.5,
                             alpha=0.4, rasterized=True)
        fig.colorbar(sc, ax=axes[2], label=r"Maslov $\mu$")
        axes[2].set(title=r"Action $S(x)$  (colour = $\mu$)",
                    xlabel="$x$", ylabel="$S$")

    fig.tight_layout()
    _style(fig, axes)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig