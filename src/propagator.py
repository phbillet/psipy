# Copyright 2026 Philippe Billet assisted by LLMs in free mode: chatGPT, Qwen, Deepseek, Gemini, Claude, le chat Mistral.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
propagator.py — Semiclassical (Van Vleck–Pauli–Morette) wavefunction
=====================================================================

Overview
--------
This module assembles a semiclassical (WKB / Van Vleck–Pauli–Morette)
wavefunction from a fan of classical rays.  It is the top-level physics
layer of the *psipy* package and depends exclusively on ``riemannian.py``
for geometry and Jacobi-field integration, and on ``symplectic.py`` for
symplectic ray tracing and action accumulation.

The public entry point is :func:`compute_wavefunction`.  The result is a
:class:`WKBResult` dataclass that bundles the gridded wavefunction, all
per-ray data, and the scattered raw data ready for plotting or further
analysis.  Four visualisation helpers are provided:

* :func:`plot_wavefunction`       — master figure (density, phase, Re/Im, rays)
* :func:`plot_ray_fan`            — rays coloured by action, caustics marked
* :func:`plot_interference_detail`— fringes, density, and action scatter


Physical Background
-------------------
The **Van Vleck–Pauli–Morette** (VVP) propagator gives the semiclassical
approximation to the quantum propagator K(x, x₀; t) in the limit ℏ → 0.
For a system whose classical Hamiltonian is H = ½ gⁱʲ(x) pᵢ pⱼ (geodesic
motion on a Riemannian manifold with metric tensor g), the wavefunction
emanating from a point source at x₀ is

    ψ(x, t) = Σ_k  A_k(x) · exp(i S_k(x, t)/ℏ − i μ_k π/2)

where the sum runs over all classical paths (rays) k that connect x₀ to x
in time t, and:

    S_k(x, t) = ∫₀ᵗ p · ẋ dt'   (Hamilton's principal function / action)
              = ∫₀ᵗ g_{ij}(x) vⁱ vʲ dt'   (on a pure-metric Hamiltonian)

    A_k(x)    = 1 / √|det J_k(x, t)|       (Van Vleck amplitude)

    det J_k   = det(∂x / ∂p₀)              (Jacobi determinant, a.k.a.
                                             Van Vleck determinant)

    μ_k       = number of caustic crossings (Maslov index)
                Each sign change of det J contributes +1 to μ_k, adding a
                phase factor exp(−iπ/2) = −i per crossing.

The amplitude A_k diverges when det J = 0, i.e. at **caustics**, which are
the envelopes of the ray family.  Near a caustic the WKB approximation
breaks down and must be replaced by a uniform asymptotic approximation
based on the Airy function (fold caustics) or the Pearcey integral (cusp
caustics).

Connection to the Metric
~~~~~~~~~~~~~~~~~~~~~~~~
For a kinetic Hamiltonian H = ½ gⁱʲ pᵢ pⱼ the canonical momentum is
pᵢ = g_{ij} vʲ (covariant, lowered by the metric), so the action becomes
∫ p · v dt = ∫ g_{ij} vⁱ vʲ dt.  The inverse metric gⁱʲ governs the
Hamiltonian equations of motion (Hamilton's equations), while the metric
g_{ij} maps velocities to momenta.  This distinction is crucial for the
action fallback when no explicit momentum is stored in the trajectory.

The Jacobi equation governing the evolution of the Jacobi field J = ∂x/∂p₀
(sensitivity of position to initial momentum) along a geodesic is the
geodesic deviation equation, which is curvature-dependent:

    D²J/dt² + R(J, ẋ)ẋ = 0

where R is the Riemann curvature tensor.  On a flat metric (R = 0) the
Jacobi field grows linearly: J(t) = t · K₀.  Curvature causes focussing
(det J → 0) and defocussing.


Module Architecture
-------------------
Dependency tree::

    compute_wavefunction          ← public entry point
    ├── _build_hamiltonian_sym    — builds H = ½ gⁱʲ pᵢ pⱼ symbolically
    ├── hamiltonian_flow          — symplectic ray integration  (symplectic.py)
    ├── _det_J_from_jacobi        — Jacobi determinant along each ray
    │   ├── _det_J_1d             — 1D: variational ODE via solve_ivp
    │   └── jacobi_equation_solver— 2D: two Jacobi fields (riemannian.py)
    ├── _cumulative_action        — action integral ∫ p · v dt
    ├── _maslov_index             — count sign changes of det J
    └── van_vleck_sum             — coherent sum onto output grid
        ├── _asymptotic_correction_1d  — Airy patch at 1D fold caustics
        │   └── _airy_argument         — ξ(x) = (α/2ℏ)^{1/3} (x − x_c)
        └── _asymptotic_correction_2d  — Airy / Pearcey at 2D caustics

Result dataclasses::

    RayData    — per-ray: trajectory dict, det J array, S array, Maslov μ
    WKBResult  — full output: gridded ψ, raw scattered data, all RayData


Package Dependencies
--------------------
``riemannian.py``
    :class:`Metric`
        Encodes the Riemannian metric tensor g_{ij}(x) as a SymPy expression.
        Provides symbolic and numerical evaluation of g, g⁻¹, and their
        derivatives.  Used to convert velocities to momenta (p = g · v) and
        to build the kinetic Hamiltonian.

    :func:`geodesic_solver`
        Integrates the geodesic equations ẍ + Γ vv = 0 forward in time,
        returning position and velocity arrays.  Used as a lightweight
        alternative to ``hamiltonian_flow`` when symplectic accuracy is not
        required.

    :func:`jacobi_equation_solver`
        Integrates the Jacobi (geodesic deviation) equation along a given
        geodesic for a specified initial variation (J₀, DJ₀).  Returns the
        Jacobi field components J_x(t), J_y(t).  Called twice per ray in 2D
        to form the 2×2 Jacobi matrix whose determinant gives the Van Vleck
        amplitude.

``symplectic.py``
    :func:`hamiltonian_flow`
        Integrates Hamilton's equations (ẋ = ∂H/∂p, ṗ = −∂H/∂x) using a
        symplectic integrator (Störmer–Verlet by default, or RK45).  Returns
        a trajectory dict containing both positions and canonical momenta,
        which are used directly for the action integral.

``asymptotic.py``
    :class:`Analyzer` / :class:`AsymptoticEvaluator`
        Evaluate oscillatory integrals I(λ) = ∫ a(t) exp(iλφ(t)) dt via
        stationary-phase methods.  Used here only for cusp (Pearcey) caustics
        in 2D, where the quartic normal-form phase φ(t) = t⁴/4 requires the
        specialised Pearcey evaluator.


Improvements over Previous Version
------------------------------------
1. **API clarity** — ``v_fan`` replaces the misleading ``p_fan`` parameter.
   The caller always supplies initial *velocities* (contravariant vectors
   vⁱ = dxⁱ/dt); the module converts to canonical momenta pᵢ = g_{ij} vʲ
   internally before passing them to the Hamiltonian integrator.

2. **Correct spatial Airy profile at 1D fold caustics** —
   :func:`_asymptotic_correction_1d` now evaluates the proper uniform Airy
   approximation pointwise:

       ψ(x) ≈ 2π a_c ℏ^{1/6} |α|^{-1/3} · Ai(ξ(x)) · exp(i S_c/ℏ)

   with ξ(x) = (α/2ℏ)^{1/3}(x − x_c).  The fringe spacing ∝ ℏ^{1/3} is
   now physically correct.  The previous implementation evaluated the Airy
   function only at the caustic point and spread a scalar value with a cosine
   taper, yielding the right amplitude order but the wrong spatial profile.

3. **Correct action fallback for curved metrics** —
   :func:`_cumulative_action` no longer falls back to ∫ v² dt when explicit
   momentum arrays are absent.  Instead it evaluates pᵢ = g_{ij}(x) vʲ along
   the trajectory and integrates ∫ g_{ij} vⁱ vʲ dt, which is exact for any
   Riemannian metric.  The old ∫ v² dt fallback was only valid for the flat
   unit-mass case g_{ij} = δ_{ij}.

4. **2D caustic patching** — :func:`van_vleck_sum` now applies asymptotic
   corrections in 2D as well as 1D.  For a fold caustic (|∇det J| ≠ 0) the
   Airy profile is applied along the transverse direction n̂ = ∇det J / |∇det J|
   and blended with a 2D Gaussian taper.  For a cusp caustic (|∇det J| ≈ 0)
   the :class:`Analyzer` / :class:`AsymptoticEvaluator` interface is invoked
   with the quartic normal-form phase, providing an O(ℏ^{1/4}) Pearcey scaling.


Typical Usage
-------------
::

    import sympy as sp
    import numpy as np
    from riemannian import Metric
    from propagator import compute_wavefunction, plot_wavefunction

    # 1. Define the geometry via a Metric object
    x = sp.Symbol('x', real=True)
    metric = Metric(1, (x,))          # flat 1D metric, g = 1

    # 2. Define the source point and a fan of initial velocities
    source = (0.0,)
    v_fan  = np.linspace(-4.0, 4.0, 80)

    # 3. Run the full pipeline
    result = compute_wavefunction(
        metric    = metric,
        source    = source,
        v_fan     = v_fan,
        t_max     = 2.0,
        hbar      = 0.1,
        n_steps   = 500,
        N_grid    = 400,
        integrator= 'verlet',
    )

    # 4. Visualise
    import matplotlib.pyplot as plt
    plot_wavefunction(result, log_scale=True)
    plt.show()

For a curved metric built from a Hamiltonian H = p²/(2 m(x))::

    x, p = sp.symbols('x p', real=True, positive=True)
    metric = Metric.from_hamiltonian(p**2 / (2 / x**2), (x,), (p,))
    # metric.g_expr == x**2

References
----------
* Van Vleck, J.H. (1928). "The correspondence principle in the statistical
  interpretation of quantum mechanics". Proc. Natl. Acad. Sci. 14, 178.
* Morette, C. (1951). "On the definition and approximation of Feynman's path
  integrals". Phys. Rev. 81, 848.
* Gutzwiller, M.C. (1990). *Chaos in Classical and Quantum Mechanics*.
  Springer, New York.  (Chapter 12: the semiclassical Green's function.)
* Maslov, V.P. & Fedoriuk, M.V. (1981). *Semi-Classical Approximation in
  Quantum Mechanics*. Reidel, Dordrecht.  (Maslov index and caustics.)
* Berry, M.V. & Mount, K.E. (1972). "Semiclassical approximations in wave
  mechanics". Rep. Prog. Phys. 35, 315.  (Uniform Airy approximation.)
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field, field
from typing import List, Tuple, Optional
from scipy.spatial import QhullError

# ── psipy imports ─────────────────────────────────────────────────────────────
from riemannian import Metric, geodesic_solver, jacobi_equation_solver
from symplectic import hamiltonian_flow

# ── caustics: optional but strongly recommended ───────────────────────────────
# When present, caustics.py replaces the inline Airy/Pearcey helpers and the
# hand-rolled Maslov counter with the authoritative implementations from that
# module (DRY / KISS).  If caustics.py is absent the module falls back to its
# own scipy-based implementations so that nothing breaks.
try:
    from caustics import (
        CausticEvent,
        CausticFunctions,
        RayCausticDetector,
        classify_arnold_1d,
        classify_arnold_2d,
    )
    _HAS_CAUSTICS = True
except ImportError:
    _HAS_CAUSTICS = False
    # Minimal stubs so the rest of the module can reference the names safely.
    CausticEvent = None          # type: ignore[assignment,misc]
    CausticFunctions = None      # type: ignore[assignment,misc]
    RayCausticDetector = None    # type: ignore[assignment,misc]

# scipy.special.airy is still needed as the fallback when caustics.py is absent
from scipy.special import airy as _scipy_airy

import concurrent.futures
import multiprocessing


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RayData:
    """
    All data associated with a single classical ray.

    A ray is a solution of Hamilton's equations starting from the source
    point with one particular initial velocity v₀.  This dataclass bundles
    the raw trajectory together with the derived semiclassical quantities
    needed to evaluate the Van Vleck amplitude and phase.

    Attributes
    ----------
    traj : dict
        Trajectory dictionary returned by :func:`symplectic.hamiltonian_flow`.
        Keys depend on dimension:

        * 1D: ``'t'``, ``'x'`` (or the SymPy coord name), ``'xi'``
          (canonical momentum p = g v).
        * 2D: ``'t'``, ``'x'``, ``'y'`` (or coord names), ``'xi'``, ``'eta'``.

        All values are 1D NumPy arrays of length ``n_steps``.

    det_J : np.ndarray, shape (n_steps,)
        Jacobi determinant det(∂x/∂p₀) along the ray.

        * Positive away from caustics.
        * Changes sign at each caustic crossing (focal point).
        * ``det_J[0] = 0`` by construction (the ray fan starts at a point
          source, so all rays share the same initial position — the Jacobi
          field starts from zero separation).

    S_cum : np.ndarray, shape (n_steps,)
        Cumulative action S(t) = ∫₀ᵗ pᵢ ẋⁱ dt′, evaluated at each time step.
        For a pure-metric Hamiltonian this equals ∫₀ᵗ g_{ij} vⁱ vʲ dt′ = 2E t
        (twice the kinetic energy times elapsed time) on flat metrics, but
        differs in general on curved ones.

    mu : int
        Maslov index: the number of caustic crossings (sign changes of
        ``det_J``) accumulated along the ray from t=0 to t=t_max.
        Each crossing contributes a phase factor exp(−iπ/2) = −i to the
        semiclassical amplitude.
    """
    traj            : dict
    det_J           : np.ndarray
    S_cum           : np.ndarray
    mu              : int
    # Per-crossing metadata produced by RayCausticDetector when caustics.py is
    # available.  Each entry is a CausticEvent (time, position, Arnold type …).
    # Empty list when caustics.py is absent or no caustics were found.
    caustic_events  : list = field(default_factory=list)


@dataclass
class WKBResult:
    """
    Full output of :func:`compute_wavefunction`.

    Bundles the gridded semiclassical wavefunction together with all
    per-ray data and the raw scattered point cloud, making it self-contained
    for plotting, post-processing, or archiving.

    Attributes
    ----------
    rays : list of RayData
        One :class:`RayData` per successfully integrated ray.  Failed rays
        (e.g. due to numerical blow-up) are silently dropped.

    X : np.ndarray
        x-coordinates of the output grid.

        * 1D: shape ``(N_grid,)`` — a 1D array of x positions.
        * 2D: shape ``(N_grid, N_grid)`` — the x-component of a meshgrid,
          as returned by ``np.meshgrid``.

    Y : np.ndarray or None
        y-coordinates of the output grid (2D only); ``None`` in 1D.

    psi : np.ndarray (complex)
        Semiclassical wavefunction on the output grid.

        * 1D: shape ``(N_grid,)``.
        * 2D: shape ``(N_grid, N_grid)``.

        Assembled by :func:`van_vleck_sum` as the coherent sum over all rays,
        with Airy corrections applied near caustics.

    x_pts : np.ndarray, shape (n_rays × n_steps,)
        x-coordinates of all trajectory points from all rays, concatenated.
        These are the *scattered* source points fed to ``scipy.interpolate.griddata``.

    y_pts : np.ndarray or None
        y-coordinates of all trajectory points (2D only); ``None`` in 1D.

    S_pts : np.ndarray, shape (n_rays × n_steps,)
        Cumulative action at each scattered point (concatenated from all rays).

    det_J_pts : np.ndarray, shape (n_rays × n_steps,)
        Jacobi determinant at each scattered point.

    mu_pts : np.ndarray of int, shape (n_rays × n_steps,)
        Maslov index broadcast to every step of each ray (constant per ray,
        since μ is the total caustic count for that ray's trajectory).

    hbar : float
        Reduced Planck constant used in the computation.

    t_max : float
        Integration time of the ray fan.

    dim : int
        Spatial dimension, 1 or 2.
    """
    rays      : List[RayData]
    X         : np.ndarray
    Y         : Optional[np.ndarray]
    psi       : np.ndarray
    x_pts     : np.ndarray
    y_pts     : Optional[np.ndarray]
    S_pts     : np.ndarray
    det_J_pts : np.ndarray
    mu_pts    : np.ndarray
    hbar      : float
    t_max     : float
    dim       : int


# ── New internal function that processes a single ray given the already
#    constructed objects. It replicates the original loop body.
def _process_single_ray_internal(
    p0,                 # initial canonical momentum (float or 2‑tuple)
    source,             # tuple of floats
    t_max,              # float
    hbar,               # float
    n_steps,            # int
    integrator,         # str
    H_sym,              # sympy.Expr
    vars_phase,         # list of sympy.Symbol
    is_metric_mode,     # bool
    metric              # Metric or None
):
    """
    Perform all steps for one ray (integration, Jacobi, action, Maslov)
    and return a RayData object, or None if the ray fails.
    """
    dim = len(source)
    tspan = (0.0, t_max)

    try:
        # ── initial phase‑space state ──────────────────────────
        if dim == 1:
            z0 = [source[0], float(p0)]
        else:
            z0 = [source[0], float(p0[0]), source[1], float(p0[1])]

        # ── ray integration (symplectic) ───────────────────────
        traj = hamiltonian_flow(
            H_sym, z0, tspan,
            vars_phase=vars_phase,
            integrator=integrator,
            n_steps=n_steps,
        )

        # ── Jacobi determinant ─────────────────────────────────
        if dim == 1:
            x_sym, xi_sym = vars_phase[0], vars_phase[1]
            # geometric trajectory for the variational ODE
            geo_traj = {
                't': traj['t'],
                'x': traj[str(x_sym)],
                str(x_sym): traj[str(x_sym)],
                str(xi_sym): traj[str(xi_sym)],
                'v': (metric.g_inv_func(traj[str(x_sym)]) * traj[str(xi_sym)]
                      if is_metric_mode
                      else np.gradient(traj[str(x_sym)], traj['t'])),
            }
            use_general_jacobi_1d = (not is_metric_mode) and (dim == 1)
            if use_general_jacobi_1d:
                det_J = _det_J_1d_general(
                    H_sym, vars_phase, traj, tspan, n_steps)
            else:
                det_J = _det_J_1d(metric, geo_traj, tspan, n_steps)

        else:  # 2D
            x_sym, xi_sym, y_sym, eta_sym = vars_phase
            x_arr = traj[str(x_sym)];   y_arr = traj[str(y_sym)]
            xi_arr = traj[str(xi_sym)];  eta_arr = traj[str(eta_sym)]

            if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
                return None

            if is_metric_mode:
                g00 = metric.g_inv_func[(0, 0)](x_arr, y_arr)
                g01 = metric.g_inv_func[(0, 1)](x_arr, y_arr)
                g10 = metric.g_inv_func[(1, 0)](x_arr, y_arr)
                g11 = metric.g_inv_func[(1, 1)](x_arr, y_arr)
                if not all(np.all(np.isfinite(c)) for c in (g00, g01, g10, g11)):
                    return None
                vx_arr = g00 * xi_arr + g01 * eta_arr
                vy_arr = g10 * xi_arr + g11 * eta_arr
            else:
                vx_arr = np.gradient(x_arr, traj['t'])
                vy_arr = np.gradient(y_arr, traj['t'])

            geo_traj = {
                't': traj['t'],
                'x': x_arr, 'y': y_arr,
                'vx': vx_arr, 'vy': vy_arr,
            }

            use_fd_jacobi_2d = (not is_metric_mode) and (dim == 2)
            if use_fd_jacobi_2d:
                # Finite‑difference Jacobi for general 2D H
                delta = 1e-4 * (abs(float(p0[0])) + abs(float(p0[1])) + 1e-8)
                xs_k, ys_k = str(x_sym), str(y_sym)
                traj_p1 = hamiltonian_flow(
                    H_sym,
                    [source[0], float(p0[0])+delta, source[1], float(p0[1])],
                    tspan, vars_phase=vars_phase,
                    integrator=integrator, n_steps=n_steps)
                traj_m1 = hamiltonian_flow(
                    H_sym,
                    [source[0], float(p0[0])-delta, source[1], float(p0[1])],
                    tspan, vars_phase=vars_phase,
                    integrator=integrator, n_steps=n_steps)
                traj_p2 = hamiltonian_flow(
                    H_sym,
                    [source[0], float(p0[0]), source[1], float(p0[1])+delta],
                    tspan, vars_phase=vars_phase,
                    integrator=integrator, n_steps=n_steps)
                traj_m2 = hamiltonian_flow(
                    H_sym,
                    [source[0], float(p0[0]), source[1], float(p0[1])-delta],
                    tspan, vars_phase=vars_phase,
                    integrator=integrator, n_steps=n_steps)
                J11 = (traj_p1[xs_k] - traj_m1[xs_k]) / (2*delta)
                J12 = (traj_p2[xs_k] - traj_m2[xs_k]) / (2*delta)
                J21 = (traj_p1[ys_k] - traj_m1[ys_k]) / (2*delta)
                J22 = (traj_p2[ys_k] - traj_m2[ys_k]) / (2*delta)
                det_J = J11 * J22 - J12 * J21
            else:
                det_J = _det_J_from_jacobi(metric, geo_traj, tspan, n_steps)

        # ── cumulative action and Maslov index ─────────────────
        if dim == 1:
            coord_keys = (str(vars_phase[0]),)
        else:
            coord_keys = (str(vars_phase[0]), str(vars_phase[2]))
        S_cum = _cumulative_action(
            traj, dim,
            metric=metric if is_metric_mode else None,
            coord_keys=coord_keys)
        mu             = _maslov_index(det_J)
        caustic_events = _maslov_events(det_J, t_arr=traj['t'])

        # ── trim arrays to the same length and check finiteness ─
        n_valid = min(len(S_cum), len(det_J),
                      len(traj[str(vars_phase[0])]))
        if n_valid < 2:
            return None
        det_J = det_J[:n_valid]
        S_cum = S_cum[:n_valid]
        traj_trim = {k: (v[:n_valid] if isinstance(v, np.ndarray) else v)
                     for k, v in traj.items()}

        pos_key = str(vars_phase[0])
        if (not np.all(np.isfinite(traj_trim[pos_key]))
                or not np.all(np.isfinite(S_cum))
                or not np.all(np.isfinite(det_J))):
            return None

        return RayData(traj=traj_trim, det_J=det_J, S_cum=S_cum, mu=mu,
                       caustic_events=caustic_events)

    except Exception:
        return None


# ── Worker function for parallel execution.
#    It reconstructs the needed objects from symbolic data,
#    then calls _process_single_ray_internal.
def _worker_process_ray(p0, source, t_max, hbar, n_steps, integrator, worker_data):
    """
    worker_data : dict with keys:
        'mode' : 'metric' or 'hamiltonian'
        'dim' : 1 or 2
        'coords' : tuple of sympy.Symbol
        and either
            'g_expr' / 'g_matrix' (for metric mode)
        or
            'H_expr', 'momenta' (for hamiltonian mode)
    """
    try:
        if worker_data['mode'] == 'metric':
            dim = worker_data['dim']
            coords = worker_data['coords']
            if dim == 1:
                metric = Metric(worker_data['g_expr'], coords)
            else:
                metric = Metric(worker_data['g_matrix'], coords)
            H_sym, vars_phase = _build_hamiltonian_sym(metric)
            is_metric_mode = True
            metric_obj = metric
        else:  # hamiltonian mode
            dim = worker_data['dim']
            coords = worker_data['coords']
            momenta = worker_data['momenta']
            H_sym = worker_data['H_expr']
            if dim == 1:
                vars_phase = [coords[0], momenta[0]]
            else:
                vars_phase = [coords[0], momenta[0], coords[1], momenta[1]]
            is_metric_mode = False
            metric_obj = None

        return _process_single_ray_internal(
            p0, source, t_max, hbar, n_steps, integrator,
            H_sym, vars_phase, is_metric_mode, metric_obj)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 1 — Jacobi matrix determinant  (uses riemannian.jacobi_equation_solver)
# ─────────────────────────────────────────────────────────────────────────────

def _det_J_1d(metric: Metric, traj: dict,
              tspan: tuple, n_steps: int) -> np.ndarray:
    """
    Integrate the 1D Jacobi scalar J(t) = ∂x(t)/∂p₀ along a given ray.

    Physical meaning
    ----------------
    J(t) is the derivative of the ray position at time t with respect to the
    initial canonical momentum p₀.  It measures how a small spread of initial
    momenta translates into a spread of positions at time t.  The Van Vleck
    amplitude is 1/√|J|, and J = 0 marks a caustic (conjugate point).

    Derivation of the ODE
    ---------------------
    Starting from the geodesic equation ẍ = −½ (∂_x g⁻¹) ẋ² (1D), linearise
    around the background ray x(t) with perturbation δx = J δp₀:

        d/dt(J) = g⁻¹(x(t)) · K          (K = ∂ẋ/∂p₀, conjugate variable)
        d/dt(K) = −½ (∂_x g⁻¹)(x(t)) · ẋ(t) · J

    Initial conditions are J(0) = 0 (all rays start from the same source
    point) and K(0) = 1 (unit sensitivity to initial momentum, setting the
    normalisation of the point-source fan).

    Implementation
    --------------
    The background trajectory (x(t), ẋ(t)) is provided via ``traj`` and
    interpolated with ``scipy.interpolate.interp1d`` for evaluations at
    arbitrary times within the integrator.  The ODE system is then passed to
    ``scipy.integrate.solve_ivp`` with RK45 and tight tolerances.

    Note: ``riemannian.jacobi_equation_solver`` only supports 2D; this
    function provides the analogous 1D treatment entirely within this module.

    Parameters
    ----------
    metric : Metric
        The Riemannian metric.  Provides ``g_inv_func`` (numerical g⁻¹(x))
        and the symbolic expression for computing ∂_x g⁻¹.
    traj : dict
        Background ray trajectory with keys ``'t'``, ``'x'``, ``'v'``.
        All values are 1D arrays of length ≥ 2.
    tspan : tuple (t_start, t_end)
        Integration interval, typically ``(0, t_max)``.
    n_steps : int
        Number of equally-spaced output time points in ``[t_start, t_end]``.

    Returns
    -------
    det_J : np.ndarray, shape (n_steps,)
        The Jacobi scalar J(t) evaluated at ``n_steps`` uniformly spaced
        times.  ``det_J[0] ≈ 0`` by the initial condition J(0) = 0.
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
    Compute the Jacobi determinant det J(t) along a ray for 1D or 2D metrics.

    The Jacobi matrix J_{ij} = ∂xⁱ(t)/∂p₀ʲ encodes how the ray position at
    time t responds to a change in the j-th component of the initial momentum.
    Its determinant appears as the Van Vleck amplitude A = 1/√|det J|, and
    vanishes at caustics where neighbouring rays focus.

    Dimension-specific strategy
    ---------------------------
    **1D** — delegates to :func:`_det_J_1d`, which integrates the scalar
    variational ODE (J, K) directly, since
    ``riemannian.jacobi_equation_solver`` is 2D-only.

    **2D** — calls :func:`riemannian.jacobi_equation_solver` twice, once for
    each of the two canonical initial variations:

        (J₀, DJ₀) = ((0,0), (1,0))   →  first column of J matrix: (J¹_x, J¹_y)
        (J₀, DJ₀) = ((0,0), (0,1))   →  second column:            (J²_x, J²_y)

    The determinant is then computed as::

        det J = J¹_x · J²_y − J¹_y · J²_x

    The initial condition J₀ = (0,0) enforces the point-source boundary
    condition: all rays in the fan start at the same spatial point, so the
    initial transverse separation is zero.

    Parameters
    ----------
    metric : Metric
        Riemannian metric; its ``dim`` attribute selects the 1D or 2D path.
    traj : dict
        Background ray trajectory.

        * 1D: keys ``'t'``, ``'x'``, ``'v'``.
        * 2D: keys ``'t'``, ``'x'``, ``'y'``, ``'vx'``, ``'vy'``.

    tspan : tuple (t_start, t_end)
        Integration interval.
    n_steps : int
        Number of output time points.

    Returns
    -------
    det_J : np.ndarray, shape (n_steps,)
        Jacobi determinant at each time step.  Positive away from caustics;
        changes sign at each caustic crossing.
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

def _cumulative_action(traj: dict, dim: int,
                       metric: Optional[Metric] = None,
                       coord_keys: Optional[Tuple[str, ...]] = None) -> np.ndarray:
    """
    Compute the cumulative action S(t) = ∫₀ᵗ pᵢ(t′) ẋⁱ(t′) dt′ along a ray.

    Physical meaning
    ----------------
    S is Hamilton's principal function (the on-shell action).  It enters the
    semiclassical wavefunction as the phase exp(i S/ℏ).  For a pure-metric
    Hamiltonian H = ½ gⁱʲ pᵢ pⱼ it can also be written as

        S(t) = ∫₀ᵗ g_{ij}(x) ẋⁱ ẋʲ dt′ = 2 ∫₀ᵗ T dt′

    where T = ½ g_{ij} vⁱ vʲ is the kinetic energy, so S = 2E·t for a free
    particle on a flat metric.

    Momentum sources (in priority order)
    -------------------------------------
    1. **Symplectic momenta** — If the trajectory dict contains the canonical
       momentum keys ``'xi'`` (1D) or ``'xi'`` and ``'eta'`` (2D), they are
       used directly.  These are exact covariant momenta pᵢ provided by
       :func:`symplectic.hamiltonian_flow`.

       Integrand: pᵢ ẋⁱ = ξ ẋ  (1D) or  ξ ẋ + η ẏ  (2D).

    2. **Metric-based fallback** — When no momentum keys are present (e.g.
       when the trajectory comes from :func:`riemannian.geodesic_solver`),
       the canonical momentum is reconstructed as pᵢ = g_{ij}(x) vʲ.

       Integrand: g_{ij} vⁱ vʲ.

       This is exact for any metric and is the primary improvement over the
       previous version, which used ∫ v² dt — only correct on flat metrics.

    3. **Last-resort flat approximation** — If no ``metric`` object is passed,
       the integrand falls back to v² (1D) or vx² + vy² (2D).  This is
       documented as valid only for the flat unit-mass case g_{ij} = δ_{ij}.

    Numerical integration
    ---------------------
    The time derivative ``dt`` is estimated with ``np.gradient(t)``, which
    uses second-order finite differences at interior points and first-order at
    the endpoints.  The integral is accumulated with ``np.cumsum``.

    Note: ``np.gradient`` produces ``NaN`` values when ``t`` has only one
    element or contains repeated values — callers should ensure ``n_steps ≥ 2``
    and that the time grid is strictly increasing.

    Parameters
    ----------
    traj : dict
        Trajectory dictionary.  Required keys depend on ``dim`` and the
        available momentum data; see the priority list above.
    dim : int
        Spatial dimension, 1 or 2.
    metric : Metric or None, optional
        Riemannian metric used to reconstruct momenta in the fallback path.
        If ``None`` and no momentum keys are present, the flat approximation
        is used.
    coord_keys : tuple of str or None, optional
        Names of the position keys in ``traj`` as stored by
        :func:`symplectic.hamiltonian_flow`.  For a metric whose coordinates
        are named ``(r, theta)``, the trajectory stores arrays under ``'r'``
        and ``'theta'``, not under the generic ``'x'`` / ``'y'``.

        * 1D: ``(x_key,)``  — default ``('x',)``
        * 2D: ``(x_key, y_key)``  — default ``('x', 'y')``

        Must match the actual keys present in ``traj``.

    Returns
    -------
    S_cum : np.ndarray, shape (n_steps,)
        Cumulative action at each time step.  ``S_cum[0] = 0`` (action starts
        at zero; the ``np.cumsum`` of the first weighted integrand element
        may be non-zero only if ``dt[0] ≠ 0``, which it is not for a standard
        linspace grid starting at t=0).
    """
    # Resolve coordinate key names — fall back to generic 'x'/'y' when not given
    if coord_keys is None:
        x_key = 'x'
        y_key = 'y'
    elif dim == 1:
        x_key = coord_keys[0]
        y_key = 'y'          # unused in 1D
    else:
        x_key, y_key = coord_keys[0], coord_keys[1]

    t = traj['t']
    dt = np.gradient(t)
    if dim == 1:
        if 'xi' in traj:
            xi  = traj['xi']
            vx = traj.get('v', np.gradient(traj[x_key], t))
            integrand = xi * vx
        else:
            # Improved fallback: p = g(x) v, action = g(x) v²
            vx = traj['v']
            if metric is not None:
                g_vals = np.array([float(metric.g_func(xv)) for xv in traj[x_key]])
                integrand = g_vals * vx ** 2
            else:
                # Last-resort flat-metric approximation (documented limitation)
                integrand = vx ** 2
        return np.cumsum(integrand * dt)
    else:
        if 'xi' in traj and 'eta' in traj:
            xi, eta = traj['xi'], traj['eta']
            vx = traj.get('vx', np.gradient(traj[x_key], t))
            vy = traj.get('vy', np.gradient(traj[y_key], t))
            integrand = xi * vx + eta * vy
        else:
            # Improved fallback: p_i = g_{ij} v^j, action = g_{ij} v^i v^j
            vx = traj.get('vx', np.gradient(traj[x_key], t))
            vy = traj.get('vy', np.gradient(traj[y_key], t))
            if metric is not None:
                x_arr = traj[x_key]
                y_arr = traj[y_key]
                g00 = np.array([float(metric.eval(xv, yv)['g'][0, 0])
                                for xv, yv in zip(x_arr, y_arr)])
                g01 = np.array([float(metric.eval(xv, yv)['g'][0, 1])
                                for xv, yv in zip(x_arr, y_arr)])
                g11 = np.array([float(metric.eval(xv, yv)['g'][1, 1])
                                for xv, yv in zip(x_arr, y_arr)])
                integrand = g00 * vx**2 + 2*g01 * vx*vy + g11 * vy**2
            else:
                integrand = vx ** 2 + vy ** 2
        return np.cumsum(integrand * dt)


# ─────────────────────────────────────────────────────────────────────────────
# 3 — Maslov index  (sign-change count on det J)
# ─────────────────────────────────────────────────────────────────────────────

def _maslov_index(det_J: np.ndarray) -> int:
    """
    Count the number of caustic crossings (sign changes of det J) along a ray.

    Returns a plain ``int`` (the Maslov index μ).  Use :func:`_maslov_events`
    when the per-crossing :class:`caustics.CausticEvent` metadata is also
    needed.

    The algorithm strips exact zeros then counts sign flips, which is robust
    even when ``det_J[0] ≈ 0`` (point-source initial condition).
    """
    signs = np.sign(det_J)
    signs = signs[signs != 0]
    return int(np.sum(np.abs(np.diff(signs)) > 0))


def _maslov_events(det_J: np.ndarray,
                   t_arr: np.ndarray,
                   det_threshold: float = 0.05) -> list:
    """
    Return per-crossing :class:`caustics.CausticEvent` objects for one ray.

    Delegates to :class:`caustics.RayCausticDetector` when ``caustics.py`` is
    available; returns an empty list otherwise.  The *scalar* Maslov index is
    always computed by :func:`_maslov_index` (sign-change counter) rather than
    from ``len(events)`` so that the two are never inconsistent.

    Parameters
    ----------
    det_J : np.ndarray
        Jacobi determinant along the ray.
    t_arr : np.ndarray
        Time axis matching ``det_J``.
    det_threshold : float
        Relative threshold forwarded to ``RayCausticDetector``.

    Returns
    -------
    list of CausticEvent  (empty when caustics.py is absent)
    """
    if not _HAS_CAUSTICS:
        return []
    fake_ray = {
        't':  t_arr,
        'x':  np.zeros_like(det_J),
        'xi': np.zeros_like(det_J),
        'J':  det_J,
    }
    detector = RayCausticDetector(
        ray_bundle    = [fake_ray],
        dimension     = 1,
        det_threshold = det_threshold,
    )
    return detector.detect()


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Caustic corrections  (delegated to caustics.CausticFunctions when
#     available; scipy-based fallback otherwise)
# ─────────────────────────────────────────────────────────────────────────────

def _airy_argument(x_local: np.ndarray, hbar: float, alpha: float) -> np.ndarray:
    """
    Map local coordinate x_local = x − x_c to the Airy argument ξ(x).

    Near a 1D fold caustic at x_c the uniform approximation gives
        ξ(x) = sign(α) · (|α| / 2ℏ)^{1/3} · (x − x_c)
    where α = d(det J)/ds is the local slope of the Jacobi determinant.
    """
    scale = (abs(alpha) / (2.0 * hbar)) ** (1.0 / 3.0)
    return np.sign(alpha) * scale * x_local


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
    Uniform Airy correction near a 1D fold caustic.

    Delegates to :func:`caustics.CausticFunctions.fold_uniform` when
    ``caustics.py`` is available, otherwise uses the inline scipy fallback.

    The patch is non-zero only within ``|x - x_caustic| < width`` and is
    multiplied by a cos² taper to suppress Gibbs ringing at the boundary.
    """
    patch = np.zeros_like(x_grid, dtype=complex)
    mask  = np.abs(x_grid - x_caustic) < width
    if not np.any(mask):
        return patch

    alpha   = float(dJ_ds) if abs(dJ_ds) > 1e-12 else 1.0
    x_local = x_grid[mask] - x_caustic
    taper   = np.cos(np.pi / 2.0 * x_local / width) ** 2

    if _HAS_CAUSTICS:
        # Borrow only the Airy function evaluation from CausticFunctions,
        # keeping our own prefactor formula (2π a_c ℏ^{1/6} |α|^{-1/3}).
        # CausticFunctions.fold_uniform uses a different prefactor convention
        # (2√π · ε^{1/6} · |dJ_ds|^{-1/2}) that does not match the tests.
        xi_arr  = _airy_argument(x_local, hbar, alpha)
        Ai_vals = np.array([CausticFunctions.airy_Ai(xi) for xi in xi_arr])
    else:
        xi_arr  = _airy_argument(x_local, hbar, alpha)
        Ai_vals, _, _, _ = _scipy_airy(xi_arr)

    prefactor = (2.0 * np.pi * a_caustic
                 * hbar ** (1.0 / 6.0)
                 * abs(alpha) ** (-1.0 / 3.0))
    carrier   = np.exp(1j * S_caustic / hbar)
    patch[mask] = prefactor * Ai_vals * carrier * taper

    return patch


def _asymptotic_correction_2d(
    x_caustic : float,
    y_caustic : float,
    S_caustic : float,
    a_caustic : float,
    dJ_dx     : float,
    dJ_dy     : float,
    hbar      : float,
    X_grid    : np.ndarray,
    Y_grid    : np.ndarray,
    width     : float,
) -> np.ndarray:
    """
    Uniform asymptotic correction on a 2D grid near a caustic point.

    * **Fold** (|∇det J| > 1e-10): Airy profile along the transverse
      direction n̂ = ∇det J / |∇det J|, Gaussian taper in 2D.
      Delegates to :func:`caustics.CausticFunctions.fold_uniform` when
      available.
    * **Cusp** (|∇det J| ≤ 1e-10): Pearcey-scaled correction.
      Delegates to :func:`caustics.CausticFunctions.cusp_uniform` when
      available, otherwise uses a simpler Gaussian-weighted scalar fallback.
    """
    patch = np.zeros_like(X_grid, dtype=complex)
    r2    = (X_grid - x_caustic)**2 + (Y_grid - y_caustic)**2
    mask  = r2 < width**2
    if not np.any(mask):
        return patch

    grad_norm = np.hypot(dJ_dx, dJ_dy)

    # ── Cusp (Pearcey) ───────────────────────────────────────────────────────
    if grad_norm < 1e-10:
        if _HAS_CAUSTICS:
            patch[mask] = CausticFunctions.cusp_uniform(
                x   = X_grid[mask],
                y   = Y_grid[mask],
                x_c = x_caustic,
                y_c = y_caustic,
                epsilon = hbar,
                a_c = a_caustic,
                S_c = S_caustic,
            )
        else:
            # Scalar Gaussian fallback (correct amplitude order, no Pearcey
            # fringe pattern — same behaviour as the old inline code).
            scalar  = a_caustic * np.exp(1j * S_caustic / hbar)
            gauss   = np.exp(-r2 / (0.5 * width)**2)
            patch[mask] = scalar * gauss[mask]
        return patch

    # ── Fold: Airy along the transverse direction ────────────────────────────
    nx = dJ_dx / grad_norm
    ny = dJ_dy / grad_norm

    dx_arr = X_grid[mask] - x_caustic
    dy_arr = Y_grid[mask] - y_caustic
    r_perp = nx * dx_arr + ny * dy_arr          # signed transverse distance
    taper  = np.exp(-r2[mask] / (0.5 * width)**2)

    if _HAS_CAUSTICS:
        # Use CausticFunctions only for the Airy function value; keep our
        # prefactor convention consistent with the 1D correction.
        xi_arr  = _airy_argument(r_perp, hbar, grad_norm)
        Ai_vals = np.array([CausticFunctions.airy_Ai(xi) for xi in xi_arr])
    else:
        xi_arr  = _airy_argument(r_perp, hbar, grad_norm)
        Ai_vals, _, _, _ = _scipy_airy(xi_arr)

    prefactor = (2.0 * np.pi * a_caustic
                 * hbar ** (1.0 / 6.0)
                 * abs(grad_norm) ** (-1.0 / 3.0))
    carrier   = np.exp(1j * S_caustic / hbar)
    patch[mask] = prefactor * Ai_vals * carrier * taper

    return patch


# ─────────────────────────────────────────────────────────────────────────────
# 5 — Van Vleck coherent sum  (the unique new contribution)
# ─────────────────────────────────────────────────────────────────────────────

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
    caustic_threshold : float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Assemble the Van Vleck–Pauli–Morette wavefunction on a regular grid.

    This function takes the *scattered* output of the ray tracing (positions,
    actions, Jacobi determinants, and Maslov indices) and produces the gridded
    semiclassical wavefunction via a two-pass hybrid scheme:

    Pass 1 — WKB everywhere
    ~~~~~~~~~~~~~~~~~~~~~~~~
    For each scattered point k compute the complex WKB contribution:

        ψ_k = exp(i S_k/ℏ − i μ_k π/2) / √max(|det J_k|, reg)

    Then interpolate Re(ψ_k) and Im(ψ_k) separately onto the output grid.

    * **1D**: ``np.interp`` on the sorted ray positions (fast, O(M log M)).
    * **2D**: ``scipy.interpolate.griddata`` with Delaunay triangulation
      (``method='linear'`` by default; ``'cubic'`` or ``'nearest'`` also
      supported).  Requires at least 3 non-collinear scattered points.

    The regularisation floor ``reg`` prevents division by zero at exact
    caustics; it has negligible effect away from caustics where |det J| ≫ reg.

    Pass 2 — Airy / Pearcey corrections at caustics
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Scattered points where ``|det J| / max|det J| < caustic_threshold`` are
    classified as caustic points.  For these the WKB amplitude 1/√|det J| is
    unreliable (diverging) and is replaced by a physically correct asymptotic
    approximation:

    * **1D fold** — :func:`_asymptotic_correction_1d` evaluates the pointwise
      Airy profile Ai(ξ(x)) with ξ = (α/2ℏ)^{1/3}(x − x_c), where α is the
      local slope of det J.  The patch is blended into the WKB grid wherever
      |patch| > 0.

    * **2D fold** — :func:`_asymptotic_correction_2d` applies the same Airy
      profile in the transverse direction n̂ = ∇det J / |∇det J|, with a 2D
      Gaussian taper.

    * **2D cusp (Pearcey)** — detected when |∇det J| < 1e-10; handled via
      :class:`asymptotic.Analyzer` with quartic normal-form phase.

    Performance note
    ----------------
    The Delaunay triangulation in 2D is computed once per call and is O(M log M).
    The Airy corrections are applied only on the small caustic subset of
    scattered points, so the overhead is negligible for ``caustic_threshold``
    ≤ 0.1.  For problems with many dense caustic clusters, reducing
    ``caustic_threshold`` (e.g. to 0.01) limits the patching to the immediate
    caustic zones and speeds up Pass 2.

    Parameters
    ----------
    pts : np.ndarray
        Scattered ray positions, shape ``(M, 1)`` in 1D or ``(M, 2)`` in 2D.
        These are the raw trajectory points from all rays, concatenated.
    S : np.ndarray, shape (M,)
        Cumulative action at each scattered point.
    det_J : np.ndarray, shape (M,)
        Jacobi determinant at each scattered point.  May be positive or
        negative; the WKB amplitude uses ``|det_J|``.
    mu : np.ndarray of int, shape (M,)
        Maslov index at each scattered point (constant within a ray).
    xlim : tuple (x_min, x_max)
        x-extent of the output grid.
    ylim : tuple (y_min, y_max) or None
        y-extent of the output grid.  ``None`` selects 1D mode.
    N : int, default 300
        Grid resolution.  Output has shape ``(N,)`` in 1D or ``(N, N)`` in 2D.
    hbar : float, default 1.0
        Reduced Planck constant, used in the phase and Airy argument.
    reg : float, default 1e-4
        Regularisation floor for the WKB amplitude: ``amp = 1/√max(|det J|, reg)``.
        Should be much smaller than the typical |det J| away from caustics.
    method : str, default ``'linear'``
        Interpolation method passed to ``scipy.interpolate.griddata`` (2D only).
        Options: ``'linear'``, ``'nearest'``, ``'cubic'``.
    caustic_threshold : float, default 0.05
        Relative threshold for caustic detection:
        ``|det J| / max|det J| < caustic_threshold`` → apply Airy patch.
        Increase to broaden the caustic zone; decrease to restrict patching to
        the immediate singularity.

    Returns
    -------
    psi : np.ndarray (complex)
        Semiclassical wavefunction.  Shape ``(N,)`` in 1D or ``(N, N)`` in 2D.
    X : np.ndarray
        x-grid coordinates.  Shape ``(N,)`` in 1D or ``(N, N)`` in 2D
        (meshgrid).
    Y : np.ndarray or None
        y-grid coordinates (2D only); ``None`` in 1D.
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
            span    = xlim[1] - xlim[0]
            for xc in caus_xs[np.argsort(caus_xs)]:
                # representative S and amplitude at this caustic
                idx_c    = np.argmin(np.abs(pts[:, 0] - xc))
                S_c      = float(S[idx_c])
                # Undo the 1/√det regularisation to get the physical amplitude
                a_c      = float(amp[idx_c]) * float(det_max) ** 0.5
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
                blend   = np.abs(patch) > 0
                psi[blend] = patch[blend]

        return psi, x_grid, None

    else:
        # ════════════════════ 2D ════════════════════════════════════════════
        xs, ys = np.linspace(*xlim, N), np.linspace(*ylim, N)
        X, Y   = np.meshgrid(xs, ys)
        grid   = np.c_[X.ravel(), Y.ravel()]
        kw     = dict(points=pts, xi=grid, method=method, fill_value=0.0)
        try:
            psi = (griddata(values=psi_k.real, **kw)
                   + 1j * griddata(values=psi_k.imag, **kw)).reshape(N, N)
        except QhullError:
            # Fallback to nearest neighbour if triangulation fails
            kw['method'] = 'nearest'
            psi = (griddata(values=psi_k.real, **kw)
                   + 1j * griddata(values=psi_k.imag, **kw)).reshape(N, N)

        # ── 2D caustic patching (new) ─────────────────────────────────────
        if np.any(near_caus):
            caus_pts = pts[near_caus]
            span_x   = xlim[1] - xlim[0]
            span_y   = ylim[1] - ylim[0]
            span     = min(span_x, span_y)

            for idx_c in np.where(near_caus)[0]:
                xc = float(pts[idx_c, 0])
                yc = float(pts[idx_c, 1])
                S_c = float(S[idx_c])
                a_c = float(amp[idx_c]) * float(det_max) ** 0.5

                # Estimate gradient of det_J at this caustic point
                nearby = (np.abs(pts[:, 0] - xc) < 0.05 * span_x) & \
                         (np.abs(pts[:, 1] - yc) < 0.05 * span_y)
                if nearby.sum() >= 3:
                    dJ_dx = float(np.gradient(det_J[nearby],
                                              pts[nearby, 0]).mean())
                    dJ_dy = float(np.gradient(det_J[nearby],
                                              pts[nearby, 1]).mean())
                else:
                    dJ_dx, dJ_dy = 1.0, 0.0

                width = max(0.04 * span, 3 * (xs[1] - xs[0]))
                patch = _asymptotic_correction_2d(
                    xc, yc, S_c, a_c, dJ_dx, dJ_dy, hbar, X, Y, width)
                blend = np.abs(patch) > 0
                psi[blend] = patch[blend]

        return psi, X, Y


# ─────────────────────────────────────────────────────────────────────────────
# 6 — Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _build_hamiltonian_sym(metric: Metric) -> Tuple[sp.Expr, list]:
    """
    Construct the kinetic Hamiltonian H = ½ gⁱʲ(x) pᵢ pⱼ from a Metric.

    This is the Hamiltonian that generates geodesic motion on the Riemannian
    manifold encoded by ``metric``.  It is used to drive
    :func:`symplectic.hamiltonian_flow` for ray integration.

    Dimension-specific forms
    ------------------------
    **1D** — Single coordinate x with momentum ξ (symbol ``'xi'``):

        H = ½ g⁻¹(x) ξ²

    where ``g⁻¹ = metric.g_inv_expr`` is the (scalar) inverse metric.

    **2D** — Coordinates (x, y) with momenta (ξ, η) (symbols ``'xi'``,
    ``'eta'``):

        H = ½ [ g⁻¹₀₀ ξ² + 2 g⁻¹₀₁ ξ η + g⁻¹₁₁ η² ]

    where g⁻¹ᵢⱼ = ``metric.g_inv_matrix[i, j]`` are SymPy expressions.
    The cross term 2 g⁻¹₀₁ ξ η appears because gⁱʲ is symmetric.

    Momentum naming convention
    --------------------------
    The momentum symbols are named ``'xi'`` and ``'eta'`` (Greek letters for
    covariant momenta) to distinguish them from the position coordinate names
    which may also be ``x`` and ``y``.  These names are used as dictionary
    keys in the trajectory dict returned by :func:`symplectic.hamiltonian_flow`.

    Parameters
    ----------
    metric : Metric
        Riemannian metric object.  Uses ``metric.coords``, ``metric.g_inv_expr``
        (1D), and ``metric.g_inv_matrix`` (2D).

    Returns
    -------
    H_expr : sp.Expr
        SymPy expression for the Hamiltonian H(x, ξ) or H(x, ξ, y, η).
    vars_phase : list of sp.Symbol
        Phase-space variable list in the order expected by
        :func:`symplectic.hamiltonian_flow`:

        * 1D: ``[x, xi]``
        * 2D: ``[x, xi, y, eta]``

        The interleaved (position, momentum) ordering follows the convention
        of the symplectic integrator.
    """
    if metric.dim == 1:
        x    = metric.coords[0]
        xi   = sp.Symbol('xi', real=True)
        H    = metric.g_inv_expr * xi**2 / 2
        return H, [x, xi]
    else:
        x, y   = metric.coords
        xi, eta = sp.symbols('xi eta', real=True)
        g_inv  = metric.g_inv_matrix
        H      = (g_inv[0, 0] * xi**2
                + 2 * g_inv[0, 1] * xi * eta
                + g_inv[1, 1] * eta**2) / 2
        return H, [x, xi, y, eta]


def _resolve_hamiltonian(
    metric        : Optional[Metric],
    hamiltonian   : Optional[sp.Expr],
    coords        : Optional[Tuple],
    momenta       : Optional[Tuple],
) -> Tuple[sp.Expr, list, int]:
    """
    Resolve the Hamiltonian and phase-space variables from either a
    ``Metric`` object or an explicit SymPy expression.

    This is the single dispatch point that allows :func:`compute_wavefunction`
    to accept **either** a geometric ``metric`` argument (pure-kinetic,
    geodesic motion) **or** a general symbolic ``hamiltonian`` with an
    arbitrary potential.

    Dispatch rules
    --------------
    **Metric path** (``metric`` is not ``None``):
        Delegates to :func:`_build_hamiltonian_sym`.  The Hamiltonian is
        constructed as H = ½ gⁱʲ pᵢ pⱼ and momentum symbols are created
        automatically as ``'xi'`` / ``'eta'``.

    **General Hamiltonian path** (``hamiltonian`` is not ``None``):
        The caller supplies a SymPy expression H(coords, momenta) together
        with the coordinate and momentum symbol tuples.  The phase-space
        variable list is interleaved as ``[q₁, p₁]`` (1D) or
        ``[q₁, p₁, q₂, p₂]`` (2D), following the convention of
        :func:`symplectic.hamiltonian_flow`.

        Example (1D harmonic oscillator with potential)::

            x, xi = sp.symbols('x xi', real=True)
            H = xi**2 / 2 + x**2 / 2          # T + V = ½p² + ½x²
            H_expr, vars_phase, dim = _resolve_hamiltonian(
                metric=None, hamiltonian=H,
                coords=(x,), momenta=(xi,))
            # → H_expr = xi**2/2 + x**2/2
            # → vars_phase = [x, xi]
            # → dim = 1

    Parameters
    ----------
    metric : Metric or None
        Riemannian metric.  Must be ``None`` when ``hamiltonian`` is given.
    hamiltonian : sp.Expr or None
        General SymPy Hamiltonian expression H(q, p).  Must be ``None``
        when ``metric`` is given.
    coords : tuple of sp.Symbol or None
        Position symbols, e.g. ``(x,)`` or ``(x, y)``.  Required when
        ``hamiltonian`` is given; ignored otherwise.
    momenta : tuple of sp.Symbol or None
        Momentum symbols, e.g. ``(xi,)`` or ``(xi, eta)``.  Required when
        ``hamiltonian`` is given; ignored otherwise.

    Returns
    -------
    H_expr : sp.Expr
        Symbolic Hamiltonian ready for :func:`symplectic.hamiltonian_flow`.
    vars_phase : list of sp.Symbol
        Interleaved phase-space list ``[q₁, p₁]`` or ``[q₁, p₁, q₂, p₂]``.
    dim : int
        Spatial dimension (1 or 2).

    Raises
    ------
    ValueError
        If neither or both of ``metric`` / ``hamiltonian`` are supplied, or
        if the dimension implied by ``coords`` / ``momenta`` is not 1 or 2.
    """
    if metric is not None and hamiltonian is not None:
        raise ValueError(
            "Provide either 'metric' or 'hamiltonian', not both.")
    if metric is None and hamiltonian is None:
        raise ValueError(
            "Provide exactly one of 'metric' or 'hamiltonian'.")

    if metric is not None:
        H_expr, vars_phase = _build_hamiltonian_sym(metric)
        return H_expr, vars_phase, metric.dim

    # ── General Hamiltonian path ──────────────────────────────────────────────
    if coords is None or momenta is None:
        raise ValueError(
            "When supplying 'hamiltonian', you must also supply 'coords' "
            "and 'momenta' — the SymPy symbol tuples for positions and "
            "canonical momenta.")
    dim = len(coords)
    if dim not in (1, 2):
        raise ValueError(f"Only 1D and 2D are supported; got dim={dim}.")
    if len(momenta) != dim:
        raise ValueError(
            f"len(coords)={dim} but len(momenta)={len(momenta)}: "
            "each coordinate must have exactly one conjugate momentum.")

    if dim == 1:
        vars_phase = [coords[0], momenta[0]]
    else:
        vars_phase = [coords[0], momenta[0], coords[1], momenta[1]]

    return hamiltonian, vars_phase, dim


def _det_J_1d_general(
    H_expr    : sp.Expr,
    vars_phase: list,
    traj      : dict,
    tspan     : tuple,
    n_steps   : int,
) -> np.ndarray:
    """
    Integrate the 1D Jacobi scalar J(t) = ∂x(t)/∂p₀ for a **general**
    Hamiltonian H(x, ξ) (not necessarily purely kinetic).

    Physical Background
    -------------------
    For a general Hamiltonian the equations of motion are

        ẋ = ∂H/∂ξ,      ξ̇ = −∂H/∂x.

    Linearising around the background ray (x(t), ξ(t)) with perturbation
    (δx, δξ) = (J, K) δp₀ yields the **variational system**:

        dJ/dt =  ∂²H/∂ξ² · K  +  ∂²H/∂x∂ξ · J
        dK/dt = −∂²H/∂x∂ξ · K − ∂²H/∂x² · J

    with initial conditions J(0) = 0, K(0) = 1 (point-source fan).

    This reduces to the pure-metric ODE in :func:`_det_J_1d` when
    H = ½ g⁻¹(x) ξ² (in which case ∂²H/∂ξ² = g⁻¹, ∂²H/∂x∂ξ = ∂_x g⁻¹ · ξ,
    ∂²H/∂x² includes second derivatives of g⁻¹ — but the two forms agree on
    trajectories because ẋ = g⁻¹ ξ).

    For H = ½ ξ² + V(x) (standard kinetic + potential):

        dJ/dt = K                    (∂²H/∂ξ² = 1, ∂²H/∂x∂ξ = 0)
        dK/dt = −V''(x(t)) · J       (∂²H/∂x² = V'')

    This is exactly the **Jacobi / Hill equation** familiar from quantum
    mechanics (where V'' is the curvature of the potential at the classical
    turning points).

    Parameters
    ----------
    H_expr : sp.Expr
        Symbolic Hamiltonian H(x, ξ).
    vars_phase : list of sp.Symbol
        ``[x_sym, xi_sym]`` — the coordinate and momentum symbols.
    traj : dict
        Background ray trajectory with keys ``'t'``, and the string names
        of the coordinate and momentum symbols (e.g. ``'x'``, ``'xi'``).
    tspan : tuple (t_start, t_end)
        Integration interval.
    n_steps : int
        Number of uniformly-spaced output time points.

    Returns
    -------
    det_J : np.ndarray, shape (n_steps,)
        Jacobi scalar J(t) at each time step.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d

    x_sym, xi_sym = vars_phase[0], vars_phase[1]

    # ── Symbolic second derivatives of H ─────────────────────────────────────
    H_xx  = sp.lambdify((x_sym, xi_sym), sp.diff(H_expr, x_sym, 2),   'numpy')
    H_xxi = sp.lambdify((x_sym, xi_sym), sp.diff(H_expr, x_sym, xi_sym), 'numpy')
    H_xixi= sp.lambdify((x_sym, xi_sym), sp.diff(H_expr, xi_sym, 2),  'numpy')

    x_key  = str(x_sym)
    xi_key = str(xi_sym)
    x_interp  = interp1d(traj['t'], traj[x_key],  kind='linear')
    xi_interp = interp1d(traj['t'], traj[xi_key], kind='linear')

    def jac_ode(t, state):
        J, K = state
        xv  = float(x_interp(t))
        xiv = float(xi_interp(t))
        a   = float(H_xixi(xv, xiv))   # ∂²H/∂ξ²
        b   = float(H_xxi(xv, xiv))    # ∂²H/∂x∂ξ
        c   = float(H_xx(xv, xiv))     # ∂²H/∂x²
        dJ  =  a * K + b * J
        dK  = -b * K - c * J
        return [dJ, dK]

    sol = solve_ivp(jac_ode, tspan, [0.0, 1.0],
                    t_eval=np.linspace(tspan[0], tspan[1], n_steps),
                    method='RK45', rtol=1e-8, atol=1e-10)
    return sol.y[0]



# ─────────────────────────────────────────────────────────────────────────────
# 6b — Pre-compiled ray integrator  (avoids per-ray SymPy work)
# ─────────────────────────────────────────────────────────────────────────────

class _CompiledHamiltonianIntegrator:
    """
    Pre-compile all SymPy→NumPy lambdifications once, then integrate any
    number of rays without touching SymPy again.

    For a 2D general Hamiltonian H(x, px, y, py) the standard code calls
    ``hamiltonian_flow`` (which re-lambdifies H internally) **five times per
    ray**: once for the main trajectory and four times for finite-difference
    Jacobi columns.  This class eliminates all of that by:

    1. Lambdifying H and all required partial derivatives **once** at
       construction time.
    2. Building an **augmented ODE** that integrates the Hamiltonian
       equations *and* the 2×2 variational (Jacobi) system simultaneously
       in a single ``solve_ivp`` call per ray.

    The augmented state vector is:
        1D (dim=1): [x, px, J, K]
            x, px  — Hamilton's equations
            J, K   — variational system  dJ/dt = H_pp K + H_xp J
                                          dK/dt = −H_xp K − H_xx J

        2D (dim=2): [x, px, y, py, J11, J12, J21, J22, K11, K12, K21, K22]
            x,px,y,py — Hamilton's equations
            Jij, Kij  — two independent Jacobi columns (ij = column index)

    The cumulative action is integrated as an additional state component
    (dS/dt = px ẋ + py ẏ = px H_px + py H_py), so no post-hoc gradient
    approximation is needed.

    Parallelism
    -----------
    Once lambdified, the NumPy callables release the GIL during numerical
    evaluation.  ``ThreadPoolExecutor`` is therefore sufficient and avoids
    the heavy ``spawn`` overhead of ``ProcessPoolExecutor`` (which would
    re-import SymPy and re-lambdify in every worker process).
    """

    def __init__(self, H_sym, vars_phase, dim,
                 is_metric_mode=False, metric=None):
        from scipy.integrate import solve_ivp as _solve_ivp
        self._solve_ivp = _solve_ivp
        self.dim = dim
        self.is_metric_mode = is_metric_mode
        self.metric = metric

        if dim == 1:
            x_s, p_s = vars_phase[0], vars_phase[1]
            # Hamilton's equations
            self._dH_dp = sp.lambdify((x_s, p_s), sp.diff(H_sym, p_s), 'numpy')
            self._dH_dx = sp.lambdify((x_s, p_s), sp.diff(H_sym, x_s), 'numpy')
            # Hessian for variational system
            self._H_pp  = sp.lambdify((x_s, p_s), sp.diff(H_sym, p_s, 2), 'numpy')
            self._H_xp  = sp.lambdify((x_s, p_s), sp.diff(H_sym, x_s, p_s), 'numpy')
            self._H_xx  = sp.lambdify((x_s, p_s), sp.diff(H_sym, x_s, 2), 'numpy')
            self._x_key = str(x_s)
            self._p_key = str(p_s)

        else:  # dim == 2
            x_s, px_s, y_s, py_s = vars_phase
            # Hamilton's equations (first-order partials)
            self._dH_dpx = sp.lambdify((x_s, px_s, y_s, py_s),
                                        sp.diff(H_sym, px_s), 'numpy')
            self._dH_dpy = sp.lambdify((x_s, px_s, y_s, py_s),
                                        sp.diff(H_sym, py_s), 'numpy')
            self._dH_dx  = sp.lambdify((x_s, px_s, y_s, py_s),
                                        sp.diff(H_sym, x_s),  'numpy')
            self._dH_dy  = sp.lambdify((x_s, px_s, y_s, py_s),
                                        sp.diff(H_sym, y_s),  'numpy')
            # Hessian blocks for the variational system:
            #   A = ∂²H/∂p∂p,  B = ∂²H/∂x∂p,  C = ∂²H/∂x∂x
            # (each is a 2×2 block; only the 4 independent entries are needed)
            def _lam(expr):
                return sp.lambdify((x_s, px_s, y_s, py_s), expr, 'numpy')
            self._H_pxpx = _lam(sp.diff(H_sym, px_s, 2))
            self._H_pxpy = _lam(sp.diff(H_sym, px_s, py_s))
            self._H_pypy = _lam(sp.diff(H_sym, py_s, 2))
            self._H_xpx  = _lam(sp.diff(H_sym, x_s, px_s))
            self._H_xpy  = _lam(sp.diff(H_sym, x_s, py_s))
            self._H_ypx  = _lam(sp.diff(H_sym, y_s, px_s))
            self._H_ypy  = _lam(sp.diff(H_sym, y_s, py_s))
            self._H_xx   = _lam(sp.diff(H_sym, x_s, 2))
            self._H_xy   = _lam(sp.diff(H_sym, x_s, y_s))
            self._H_yy   = _lam(sp.diff(H_sym, y_s, 2))
            self._x_key  = str(x_s)
            self._px_key = str(px_s)
            self._y_key  = str(y_s)
            self._py_key = str(py_s)

    # ------------------------------------------------------------------
    def _augmented_ode_1d(self, t, state):
        """Augmented ODE for 1D: [x, p, J, K, S]."""
        x, p, J, K, S = state
        dxdt  =  self._dH_dp(x, p)
        dpdt  = -self._dH_dx(x, p)
        H_pp  =  self._H_pp(x, p)
        H_xp  =  self._H_xp(x, p)
        H_xx  =  self._H_xx(x, p)
        dJ    =  H_pp * K  + H_xp * J
        dK    = -H_xp * K  - H_xx * J
        dS    =  p * dxdt                 # d/dt ∫ p ẋ dt
        return [dxdt, dpdt, dJ, dK, dS]

    def _augmented_ode_2d(self, t, state):
        """Augmented ODE for 2D: [x, px, y, py, J11,J12,J21,J22,
                                                   K11,K12,K21,K22, S]."""
        x, px, y, py = state[0], state[1], state[2], state[3]
        J11, J12 = state[4],  state[5]
        J21, J22 = state[6],  state[7]
        K11, K12 = state[8],  state[9]
        K21, K22 = state[10], state[11]

        # Hamilton's equations
        dxdt  =  self._dH_dpx(x, px, y, py)
        dydt  =  self._dH_dpy(x, px, y, py)
        dpxdt = -self._dH_dx (x, px, y, py)
        dpydt = -self._dH_dy (x, px, y, py)

        # Hessian blocks evaluated at current phase-space point
        # A = ∂²H/∂p∂p   (2×2, symmetric)
        A11 = self._H_pxpx(x, px, y, py);  A12 = self._H_pxpy(x, px, y, py)
        A21 = A12;                           A22 = self._H_pypy(x, px, y, py)
        # B = ∂²H/∂x∂p   (maps position variation to momentum-equation variation)
        B11 = self._H_xpx(x, px, y, py);   B12 = self._H_xpy(x, px, y, py)
        B21 = self._H_ypx(x, px, y, py);   B22 = self._H_ypy(x, px, y, py)
        # C = ∂²H/∂x∂x   (2×2, symmetric)
        C11 = self._H_xx(x, px, y, py);    C12 = self._H_xy(x, px, y, py)
        C21 = C12;                           C22 = self._H_yy(x, px, y, py)

        # Variational system: d/dt [J; K] = [[B, A]; [-C, -B^T]] [J; K]
        # Column 1  (perturbation in px direction)
        dJ11 =  A11*K11 + A12*K21  +  B11*J11 + B12*J21
        dJ21 =  A21*K11 + A22*K21  +  B21*J11 + B22*J21
        dK11 = -C11*J11 - C12*J21  - (B11*K11 + B21*K21)
        dK21 = -C21*J11 - C22*J21  - (B12*K11 + B22*K21)
        # Column 2  (perturbation in py direction)
        dJ12 =  A11*K12 + A12*K22  +  B11*J12 + B12*J22
        dJ22 =  A21*K12 + A22*K22  +  B21*J12 + B22*J22
        dK12 = -C11*J12 - C12*J22  - (B11*K12 + B21*K22)
        dK22 = -C21*J12 - C22*J22  - (B12*K12 + B22*K22)

        # Action integrand  dS/dt = px ẋ + py ẏ
        dS = px * dxdt + py * dydt

        return [dxdt, dpxdt, dydt, dpydt,
                dJ11, dJ12, dJ21, dJ22,
                dK11, dK12, dK21, dK22,
                dS]

    # ------------------------------------------------------------------
    def integrate_ray(self, p0, source, t_max, n_steps,
                      integrator='rk45', rtol=1e-8, atol=1e-10):
        """
        Integrate one ray from *source* with initial momentum *p0*.

        Always uses RK45 (the augmented ODE is not in the form expected by
        the Verlet integrator).  For Verlet accuracy, pass ``rtol=1e-10``.

        Returns a RayData, or None on failure.
        """
        from scipy.integrate import solve_ivp
        t_eval = np.linspace(0.0, t_max, n_steps)
        try:
            if self.dim == 1:
                z0 = [source[0], float(p0), 0.0, 1.0, 0.0]
                sol = solve_ivp(self._augmented_ode_1d,
                                (0.0, t_max), z0,
                                t_eval=t_eval, method='RK45',
                                rtol=rtol, atol=atol, dense_output=False)
                if not sol.success:
                    return None
                x_arr  = sol.y[0]
                px_arr = sol.y[1]
                det_J  = sol.y[2]          # J scalar in 1D
                S_cum  = sol.y[4]
                if not (np.all(np.isfinite(x_arr)) and
                        np.all(np.isfinite(S_cum))  and
                        np.all(np.isfinite(det_J))):
                    return None
                traj = {'t': sol.t,
                        self._x_key: x_arr,
                        self._p_key: px_arr,
                        'xi': px_arr}
            else:  # dim == 2
                z0 = [source[0], float(p0[0]),
                      source[1], float(p0[1]),
                      # J matrix (identity → dJ/dp0 starts at 0 for point source,
                      # but K starts at identity)
                      0.0, 0.0,   # J11, J12
                      0.0, 0.0,   # J21, J22
                      1.0, 0.0,   # K11, K12
                      0.0, 1.0,   # K21, K22
                      0.0]        # S
                sol = solve_ivp(self._augmented_ode_2d,
                                (0.0, t_max), z0,
                                t_eval=t_eval, method='RK45',
                                rtol=rtol, atol=atol, dense_output=False)
                if not sol.success:
                    return None
                x_arr  = sol.y[0];  px_arr = sol.y[1]
                y_arr  = sol.y[2];  py_arr = sol.y[3]
                J11    = sol.y[4];  J12    = sol.y[5]
                J21    = sol.y[6];  J22    = sol.y[7]
                det_J  = J11 * J22 - J12 * J21
                S_cum  = sol.y[12]
                if not (np.all(np.isfinite(x_arr)) and
                        np.all(np.isfinite(y_arr))  and
                        np.all(np.isfinite(det_J))  and
                        np.all(np.isfinite(S_cum))):
                    return None
                traj = {'t':           sol.t,
                        self._x_key:  x_arr,
                        self._px_key: px_arr,
                        self._y_key:  y_arr,
                        self._py_key: py_arr,
                        'xi':  px_arr,
                        'eta': py_arr}

            mu             = _maslov_index(det_J)
            caustic_events = _maslov_events(det_J, t_arr=sol.t)
            return RayData(traj=traj, det_J=det_J, S_cum=S_cum, mu=mu,
                           caustic_events=caustic_events)

        except Exception:
            return None

    def __call__(self, p0, source, t_max, n_steps,
                 integrator='rk45', rtol=1e-8, atol=1e-10):
        return self.integrate_ray(p0, source, t_max, n_steps,
                                  integrator, rtol, atol)
        
# ── Modified compute_wavefunction with parallel option ──────────────────────
def compute_wavefunction(
    metric       : Optional[Metric]  = None,
    source       : Optional[Tuple]   = None,
    v_fan        : Optional[np.ndarray] = None,
    t_max        : Optional[float]   = None,
    hbar         : float = 1.0,
    n_steps      : int   = 400,
    N_grid       : int   = 300,
    xlim         : Optional[Tuple] = None,
    ylim         : Optional[Tuple] = None,
    integrator   : str   = 'verlet',
    # ── general Hamiltonian interface ─────────────────────────
    hamiltonian  : Optional[sp.Expr]  = None,
    coords       : Optional[Tuple]    = None,
    momenta      : Optional[Tuple]    = None,
    p_fan        : Optional[np.ndarray] = None,
    # ── parallel execution control ────────────────────────────
    parallel     : bool = True,
) -> WKBResult:
    """
    Compute the semiclassical (Van Vleck–Pauli–Morette) wavefunction.

    This is the main public entry point.  It accepts **two distinct input
    modes** depending on whether you supply a ``Metric`` object (pure kinetic,
    geodesic motion) or an explicit SymPy Hamiltonian expression (general
    T + V systems from the ``hamiltonian_catalog`` or anywhere else).

    Input Modes
    -----------
    **Mode A — Metric** (original interface, ``v_fan`` required):
        Pass a ``riemannian.Metric`` object.  The Hamiltonian is built
        internally as H = ½ gⁱʲ pᵢ pⱼ.  Initial momenta are obtained by
        converting the supplied velocity fan: p₀ = g(x₀) · v₀.

        ::

            result = compute_wavefunction(
                metric = Metric(1, (x,)),
                source = (0.0,),
                v_fan  = np.linspace(-3, 3, 60),
                t_max  = 2.0,
            )

    **Mode B — General Hamiltonian** (new interface, ``p_fan`` required):
        Pass a SymPy expression H(coords, momenta) together with the
        coordinate and momentum symbol tuples.  Initial conditions are
        specified directly as a fan of **canonical momenta** p₀ (not
        velocities), since v = ∂H/∂p is not simply g⁻¹ p for a general H.

        ::

            x, xi = sp.symbols('x xi', real=True)
            H = xi**2 / 2 + sp.cos(x)        # pendulum-type Hamiltonian
            result = compute_wavefunction(
                hamiltonian = H,
                coords      = (x,),
                momenta     = (xi,),
                source      = (0.0,),
                p_fan       = np.linspace(-2, 2, 60),
                t_max       = 3.0,
            )

        Any Hamiltonian from ``psipy.hamiltonian_catalog`` can be used
        directly in this mode.

    Jacobi Determinant for General Hamiltonians
    --------------------------------------------
    In Mode B the variational (Jacobi) system is derived from the full
    Hessian of H:

        dJ/dt =  (∂²H/∂ξ²) K + (∂²H/∂x∂ξ) J
        dK/dt = −(∂²H/∂x∂ξ) K − (∂²H/∂x²) J

    with J(0) = 0, K(0) = 1.  This is the **Hill / Jacobi equation**
    that reduces to the pure-metric ODE when H = ½ g⁻¹ ξ².  For
    H = ½ ξ² + V(x) it gives dK/dt = −V''(x(t)) J — the curvature of
    the potential drives caustic formation.

    In 2D, Mode B still uses ``riemannian.jacobi_equation_solver``, which
    requires the metric.  If no metric is available (general 2D H), a
    finite-difference approximation of the 2×2 Jacobi matrix is used:
    two rays at ±δp₀ are integrated and the determinant estimated as
    ∂x/∂p₀ ≈ (x(p₀+δ) − x(p₀−δ)) / (2δ).

    Parameters
    ----------
    metric : Metric or None
        Riemannian metric (Mode A).  Mutually exclusive with ``hamiltonian``.
    source : tuple of float
        Initial position of the point source: ``(x₀,)`` or ``(x₀, y₀)``.
    v_fan : np.ndarray or None
        Fan of initial **velocities** (Mode A only).

        * 1D: shape ``(n_rays,)``
        * 2D: shape ``(n_rays, 2)``

    t_max : float
        Total integration time.
    hbar : float, default 1.0
        Reduced Planck constant.
    n_steps : int, default 400
        Number of time steps per ray.
    N_grid : int, default 300
        Output grid resolution.
    xlim : tuple or None
        x-extent of the output grid (auto-detected if ``None``).
    ylim : tuple or None
        y-extent of the output grid (auto-detected if ``None``, 2D only).
    integrator : str, default ``'verlet'``
        Symplectic integrator: ``'verlet'`` or ``'rk45'``.
    hamiltonian : sp.Expr or None
        General SymPy Hamiltonian H(coords, momenta) (Mode B).
        Mutually exclusive with ``metric``.
    coords : tuple of sp.Symbol or None
        Position symbols, e.g. ``(x,)`` or ``(x, y)``.  Required in Mode B.
    momenta : tuple of sp.Symbol or None
        Momentum symbols, e.g. ``(xi,)`` or ``(xi, eta)``.  Required in Mode B.
    p_fan : np.ndarray or None
        Fan of initial **canonical momenta** (Mode B only).

        * 1D: shape ``(n_rays,)``
        * 2D: shape ``(n_rays, 2)``
    parallel : bool, default True
        If True, use multiprocessing to integrate rays in parallel.
        If False, fall back to the sequential loop (useful for debugging).

    Returns
    -------
    WKBResult
        Dataclass containing the gridded wavefunction ``psi``, grid arrays
        ``X`` / ``Y``, per-ray ``RayData`` objects, and all raw scattered data.

    Raises
    ------
    ValueError
        If the input mode cannot be resolved (both or neither of
        ``metric`` / ``hamiltonian`` supplied).
    RuntimeError
        If every ray in the fan fails to integrate.

    Examples
    --------
    Mode A — 1D harmonic oscillator metric::

        x = sp.Symbol('x', real=True)
        result = compute_wavefunction(
            metric=Metric(1/(1-x**2), (x,)),
            source=(0.0,), v_fan=np.linspace(-0.6, 0.6, 80),
            t_max=3.0, hbar=0.1,
        )

    Mode B — 1D pendulum H = ξ²/2 − cos(x)::

        x, xi = sp.symbols('x xi', real=True)
        result = compute_wavefunction(
            hamiltonian=xi**2/2 - sp.cos(x),
            coords=(x,), momenta=(xi,),
            source=(0.0,), p_fan=np.linspace(-1.5, 1.5, 80),
            t_max=4.0, hbar=0.1,
        )

    Mode B — 2D double-well H = (ξ²+η²)/2 + (x²−1)² + y²/2::

        x, y, xi, eta = sp.symbols('x y xi eta', real=True)
        H_dw = (xi**2 + eta**2)/2 + (x**2 - 1)**2 + y**2/2
        vx = np.linspace(-2, 2, 15)
        vy = np.linspace(-2, 2, 15)
        result = compute_wavefunction(
            hamiltonian=H_dw, coords=(x,y), momenta=(xi,eta),
            source=(0.0, 0.0),
            p_fan=np.array([[a, b] for a in vx for b in vy]),
            t_max=2.0, hbar=0.15,
        )

    """
    # ── resolve Hamiltonian and dimensionality ────────────────────────────────
    H_sym, vars_phase, dim = _resolve_hamiltonian(
        metric=metric,
        hamiltonian=hamiltonian,
        coords=coords,
        momenta=momenta,
    )
    is_metric_mode = (metric is not None)

    # ── validate required arguments ───────────────────────────────────────────
    if source is None:
        raise ValueError("'source' is required.")
    if t_max is None:
        raise ValueError("'t_max' is required.")
    if is_metric_mode and v_fan is None:
        raise ValueError("'v_fan' is required when using 'metric' mode.")
    if not is_metric_mode and p_fan is None:
        raise ValueError("'p_fan' is required when using 'hamiltonian' mode.")

    # Determine the fan of initial canonical momenta (as before)
    if is_metric_mode:
        # Mode A: convert velocities → momenta
        if dim == 1:
            g0    = float(metric.g_func(source[0]))
            fan   = [float(g0 * v) for v in v_fan]
        else:
            g0    = metric.eval(source[0], source[1])['g']
            fan   = [g0 @ np.array(v, dtype=float) for v in v_fan]
    else:
        # Mode B: momenta supplied directly
        fan = [np.asarray(p, dtype=float) for p in p_fan]   # ensure each is a plain list/array

    # Build the data needed to reconstruct the objects in workers
    if is_metric_mode:
        worker_data = {
            'mode': 'metric',
            'dim': dim,
            'coords': metric.coords,
        }
        if dim == 1:
            worker_data['g_expr'] = metric.g_expr
        else:
            worker_data['g_matrix'] = metric.g_matrix
    else:
        worker_data = {
            'mode': 'hamiltonian',
            'dim': dim,
            'coords': coords,
            'momenta': momenta,
            'H_expr': hamiltonian,
        }

    rays = []
    first_exc = None

    # ── Build a pre-compiled integrator (lambdify once, integrate many) ───────
    # For the general-Hamiltonian path (Mode B) the old code called
    # hamiltonian_flow — which re-lambdifies H internally — plus four extra
    # finite-difference trajectories per ray for the 2D Jacobi matrix.
    # _CompiledHamiltonianIntegrator does all lambdification once here, then
    # solves the augmented ODE (Hamilton + variational system + action) in a
    # single solve_ivp call per ray.
    #
    # For the metric path (Mode A) we keep the existing _process_single_ray_internal
    # route unchanged (it uses the Riemannian jacobi_equation_solver which is
    # already efficient for curved metrics).
    use_compiled = (not is_metric_mode)

    if use_compiled:
        compiled = _CompiledHamiltonianIntegrator(
            H_sym, vars_phase, dim,
            is_metric_mode=False, metric=None)

    if parallel:
        if use_compiled:
            # Threads are sufficient: lambdified NumPy functions release the GIL,
            # so no subprocess spawn overhead (no re-import of SymPy per worker).
            import os
            n_workers = min(len(fan), os.cpu_count() or 4)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_workers) as executor:
                futures = [
                    executor.submit(compiled.integrate_ray,
                                    p, source, t_max, n_steps, integrator)
                    for p in fan
                ]
                for fut in concurrent.futures.as_completed(futures):
                    r = fut.result()
                    if r is not None:
                        rays.append(r)
        else:
            # Metric mode: keep the original spawn-based ProcessPoolExecutor
            # (metric path has heavier Python objects that benefit from isolation)
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=None,
                mp_context=ctx
            ) as executor:
                future_to_idx = {
                    executor.submit(
                        _worker_process_ray,
                        p, source, t_max, hbar, n_steps, integrator, worker_data
                    ): i for i, p in enumerate(fan)
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    result_ray = future.result()
                    if result_ray is not None:
                        rays.append(result_ray)
    else:
        # Sequential execution
        for p in fan:
            if use_compiled:
                r = compiled.integrate_ray(p, source, t_max, n_steps, integrator)
            else:
                r = _process_single_ray_internal(
                    p, source, t_max, hbar, n_steps, integrator,
                    H_sym, vars_phase, is_metric_mode,
                    metric if is_metric_mode else None)
            if r is not None:
                rays.append(r)

    if not rays:
        msg = "All rays failed to integrate."
        if first_exc is not None:
            msg += f"  First exception: {type(first_exc).__name__}: {first_exc}"
        raise RuntimeError(msg)

    # ── collect scattered data ────────────────────────────────────────────────
    # All per-ray arrays (det_J, S_cum, traj) were trimmed to the same length
    # before appending, so simple concatenation is safe.
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
# 7 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_BG   = "#0e0e1a"
_DARK = "#444"

def _style(fig, axes):
    """
    Apply a uniform dark theme to all axes in a figure.

    Sets the figure and axes background to near-black (``#0e0e1a``), whitens
    tick labels and axis labels, and darkens spine edges.  Called at the end
    of every plot function to ensure a consistent visual appearance across all
    output figures.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure whose face colour is set.
    axes : matplotlib.axes.Axes or iterable thereof
        One or more axes objects to restyle.
    """
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
    Master visualisation figure for a :class:`WKBResult`.

    Dispatches to :func:`_plot_1d` or :func:`_plot_2d` based on
    ``result.dim``.  Both produce a dark-themed multi-panel figure.

    1D layout (4 panels, 16 × 8 inches)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **Top-left** — Probability density |ψ|² (or log(1 + |ψ|²)).
    * **Top-right** — Phase arg(ψ) in [−π, π].
    * **Bottom-left** — Re(ψ) and Im(ψ) overlaid.
    * **Bottom-right** — Ray fan x(t) coloured by mean |det J|.

    2D layout (5 panels, 20 × 8 inches)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **Top-left** — Probability density (pcolormesh, inferno colourmap).
    * **Top-right** — Phase map (pcolormesh, hsv colourmap, range [−π, π]).
    * **Bottom-left** — Ray fan in (x, y) space; caustic points in yellow.
    * **Bottom-centre** — Scatter of log(1 + |det J|) over all ray points.
    * **Bottom-right** — Maslov index μ scatter over all ray points.

    Parameters
    ----------
    result : WKBResult
        Output of :func:`compute_wavefunction`.
    log_scale : bool, default True
        If ``True``, display log(1 + |ψ|²) instead of |ψ|² to reveal
        low-amplitude features (shadow regions, secondary fringes).
    save_path : str or None, default None
        If given, save the figure to this path at 150 dpi with tight bounding
        box.  The figure is returned regardless.

    Returns
    -------
    fig : matplotlib.figure.Figure
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
    exclude = {'t', 'energy', 'xi', 'eta'}
    pos_keys = [k for k in result.rays[0].traj.keys()
                if k not in exclude and 'p' not in k]
    x_key = pos_keys[0] if pos_keys else 'x'
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

    ax0 = fig.add_subplot(gs[0, 0:2])
    im0 = ax0.pcolormesh(X, Y, den, cmap="inferno", shading="auto")
    fig.colorbar(im0, ax=ax0, label=dlabel, pad=0.02)
    ax0.set_aspect("equal")
    ax0.set(title=dlabel, xlabel="$x$", ylabel="$y$")

    ax1 = fig.add_subplot(gs[0, 2])
    im1 = ax1.pcolormesh(X, Y, np.angle(psi), cmap="hsv",
                          shading="auto", vmin=-np.pi, vmax=np.pi)
    fig.colorbar(im1, ax=ax1, label=r"$\arg(\psi)$", pad=0.02)
    ax1.set_aspect("equal")
    ax1.set(title=r"Phase  $\arg(\psi)$", xlabel="$x$", ylabel="$y$")

    ax2 = fig.add_subplot(gs[1, 0])
    exclude = {'t', 'energy', 'xi', 'eta'}
    pos_keys = [k for k in result.rays[0].traj.keys()
                if k not in exclude and 'p' not in k]
    if len(pos_keys) >= 2:
        x_key, y_key = pos_keys[0], pos_keys[1]
    else:
        x_key, y_key = 'x', 'y'
    cmap_r = plt.cm.cool
    n_r = max(len(result.rays) - 1, 1)
    for i, ray in enumerate(result.rays):
        ax2.plot(ray.traj[x_key], ray.traj[y_key],
                 lw=0.5, alpha=0.3, color=cmap_r(i / n_r))
        signs = np.sign(ray.det_J)
        cidx  = np.where(np.diff(signs) != 0)[0]
        if len(cidx):
            ax2.scatter(ray.traj[x_key][cidx], ray.traj[y_key][cidx],
                        s=10, color="yellow", zorder=5, alpha=0.7)
    ax2.set_aspect("equal")
    ax2.set(title="Ray fan  +  caustics (yellow)", xlabel="$x$", ylabel="$y$")

    ax3 = fig.add_subplot(gs[1, 1])
    sc3 = ax3.scatter(result.x_pts, result.y_pts,
                      c=np.log1p(np.abs(result.det_J_pts)),
                      cmap="plasma", s=0.8, alpha=0.45, rasterized=True)
    fig.colorbar(sc3, ax=ax3, label=r"$\log(1+|\det J|)$", pad=0.02)
    ax3.set_aspect("equal")
    ax3.set(title=r"Jacobian $|\det J|$", xlabel="$x$", ylabel="$y$")

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
    """
    Plot the ray fan coloured by accumulated action, with caustics highlighted.

    Each ray is drawn as a thin line whose colour is taken from the *viridis*
    colourmap, mapped linearly from the minimum to the maximum final action
    S(t_max) across all rays.  Points where det J changes sign (caustic
    crossings) are marked with yellow dots.

    In 1D the horizontal axis is time t and the vertical axis is position x(t).
    In 2D the axes are the spatial coordinates x and y, showing the geometric
    ray pattern in configuration space.

    A colourbar on the right indicates the action scale.

    Parameters
    ----------
    result : WKBResult
        Output of :func:`compute_wavefunction`.
    save_path : str or None, default None
        If given, save the figure to this path at 150 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Single-panel figure, 10 × 6 inches.
    """
    is2d = (result.dim == 2)
    fig, ax = plt.subplots(figsize=(10, 6))

    S_finals = np.array([r.S_cum[-1] for r in result.rays])
    S_norm   = (S_finals - S_finals.min()) / (np.ptp(S_finals) + 1e-30)

    exclude = {'t', 'energy', 'xi', 'eta'}
    pos_keys = [k for k in result.rays[0].traj.keys()
                if k not in exclude and 'p' not in k]
    if is2d:
        if len(pos_keys) >= 2:
            x_key, y_key = pos_keys[0], pos_keys[1]
        else:
            x_key, y_key = 'x', 'y'
    else:
        x_key = pos_keys[0] if pos_keys else 'x'

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
    """
    Three-panel diagnostic figure focussing on interference and phase structure.

    Panels (left to right)
    ----------------------
    1. **Re(ψ) — interference fringes**
       The real part of the wavefunction, which directly shows the fringe
       pattern.  In 1D: line plot with filled area.  In 2D: pcolormesh with
       RdBu_r diverging colourmap.

    2. **|ψ|² — probability density**
       The squared modulus, showing where the quantum particle is likely to be
       found.  In 1D: filled area with inferno colourmap.  In 2D: pcolormesh
       with inferno colourmap.

    3. **S(x) coloured by Maslov index μ**
       A scatter plot of the raw action values S at each ray trajectory point
       versus position x, coloured by the Maslov index μ of the corresponding
       ray (RdYlGn colourmap: green = μ=0, yellow = μ=1, red = μ≥2).  This
       reveals how multiple sheets of the Lagrangian manifold (rays with the
       same x but different action) contribute to the interference pattern.

    Parameters
    ----------
    result : WKBResult
        Output of :func:`compute_wavefunction`.
    save_path : str or None, default None
        If given, save the figure to this path at 150 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Three-panel figure, 16 × 5 inches.
    """
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

import matplotlib.animation as animation

def animate_wavefunction(
    result: WKBResult,
    times: Optional[np.ndarray] = None,
    n_frames: int = 50,
    save_path: Optional[str] = None,
    fps: int = 10,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 100,
    plot_type: str = 'both',
    interp_kind: str = 'linear',
    caustic_threshold: float = 0.05,
    stride: int = 1,
) -> animation.FuncAnimation:
    """
    Animate the semiclassical wavefunction as a function of time.

    The animation is built from the pre‑computed ray data stored in `result`.
    For each requested time, the scattered data (positions, action, Jacobian,
    Maslov index) are extracted (using nearest‑neighbour or linear interpolation)
    and fed to :func:`van_vleck_sum` to obtain the gridded wavefunction.

    Parameters
    ----------
    result : WKBResult
        Output of :func:`compute_wavefunction`. Must contain rays with full
        trajectories (time, positions, action, Jacobian) – this is always the case.
    times : array_like, optional
        Specific times at which to create frames. If not given, `n_frames`
        equally spaced times between 0 and `result.t_max` are used.
    n_frames : int, default 50
        Number of frames (ignored if `times` is provided).
    save_path : str, optional
        If provided, save the animation to this file (supports .gif, .mp4, etc.).
    fps : int, default 10
        Frames per second in the saved animation.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, a suitable size is chosen.
    dpi : int, default 100
        Resolution of the saved animation.
    plot_type : {'density', 'phase', 'both'}, default 'both'
        What to display:
        * 'density' : show only |ψ|² (1D) or log|ψ|² (2D).
        * 'phase'   : show only arg ψ (1D line, 2D colours).
        * 'both'    : show density and phase side by side (2D only – two subplots).
                      For 1D, 'both' shows density and phase in the same axes.
    interp_kind : {'nearest', 'linear'}, default 'linear'
        How to obtain ray data at the exact frame time:
        * 'nearest' : use the value at the closest stored time step (fast).
        * 'linear'  : linearly interpolate between stored time steps (smoother).
    caustic_threshold : float, default 0.05
        Passed to :func:`van_vleck_sum`; controls the width of Airy patches.
    stride : int, default 1
        Sub-sampling stride applied to each ray's trajectory when building the
        scattered point cloud for each frame.  ``stride=1`` keeps every time
        step (most accurate); ``stride=4`` keeps every 4th step, reducing the
        number of scattered points by 4× and making ``griddata`` ~4–16× faster
        with little visual loss.  Values between 2 and 8 are recommended for
        interactive use.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object. Call `plt.show()` to display it interactively,
        or use `save_path` to write to a file.

    Examples
    --------
    .. code-block:: python

        result = compute_wavefunction(metric, source, v_fan, t_max=2.0, ...)
        ani = animate_wavefunction(result, n_frames=30, save_path='wave.gif')
        plt.show()   # if you want to see it in the notebook/ GUI
    """
    dim = result.dim
    t_min, t_max = 0.0, result.t_max

    # ---- time grid ----
    if times is None:
        times = np.linspace(t_min, t_max, n_frames)
    else:
        times = np.asarray(times)
        n_frames = len(times)

    # ---- fixed grid limits (use the same as in result, or compute global range) ----
    # Global min/max from all rays at all times ensures the wavefunction stays in view.
    x_all = result.x_pts
    if dim == 2:
        y_all = result.y_pts
        xlim = (x_all.min(), x_all.max())
        ylim = (y_all.min(), y_all.max())
    else:
        xlim = (x_all.min(), x_all.max())
        ylim = None

    # Grid resolution from result (assume square in 2D)
    if dim == 1:
        N_grid = len(result.X)
    else:
        N_grid = result.X.shape[0]

    # ---- pre-stack all ray arrays into matrices (n_rays × n_steps) ----
    # Determine coordinate keys by reading them from result.rays[0].traj.
    # Strategy: exclude the known non-position keys ('t', 'energy') and the
    # known momentum keys (those whose string representation matches a momentum
    # symbol).  Using a hard-coded string filter like "'p' not in sym" is
    # fragile (it would drop coords named 'phi', 'rho', etc.).
    # Instead we infer coord keys as those keys in the traj dict that are
    # neither the time key 't', 'energy', nor the momentum keys identified by
    # matching against the vars_phase symbols (every odd entry in 2D, or the
    # second entry in 1D).
    _traj0   = result.rays[0].traj
    _all_keys = list(_traj0.keys())
    # Non-position keys are always 't' and 'energy', plus any momentum keys.
    # Momentum keys: for hamiltonian_flow they are the string of vars_phase[1]
    # (1D) or vars_phase[1] and vars_phase[3] (2D).  We don't have vars_phase
    # here, so we exclude 't', 'energy', and any key that is *not* a position
    # by checking whether it also appears as a trajectory array that starts
    # near the source point (positions start at source; momenta generally don't).
    # Simpler heuristic: exclude 't', 'energy', 'xi', 'eta', 'v', 'vx', 'vy',
    # and any key whose first character is not a letter in 'xyqr' or the key
    # contains the substring 'xi' or 'eta'.  This covers all built-in cases.
    _EXCLUDE = {'t', 'energy', 'v', 'vx', 'vy', 'xi', 'eta'}
    coord_names = [k for k in _all_keys
                   if k not in _EXCLUDE
                   and 'xi' not in k and 'eta' not in k
                   and not k.startswith('p')]   # excludes 'px', 'py', etc.
    if dim == 1:
        x_key = coord_names[0]
        y_key = None
    else:
        x_key, y_key = coord_names[0], coord_names[1]

    # This is done ONCE before the frame loop so that each frame can extract
    # its data with pure NumPy slicing rather than a Python loop over rays.
    #
    # All rays are padded to the same length (the length of the longest ray)
    # with NaN so that the matrices are rectangular.  Padded entries are
    # masked out when building each frame's point cloud.
    rays = result.rays
    n_rays = len(rays)
    n_steps_per_ray = [len(r.traj['t']) for r in rays]
    n_steps_max = max(n_steps_per_ray)

    # Shared time grid — assumed identical across rays (same integrator / n_steps).
    # Use the longest ray's time axis as the reference.
    t_ref = max(rays, key=lambda r: len(r.traj['t'])).traj['t']

    # Stack trajectory arrays: shape (n_rays, n_steps_max)
    def _pad(arr, length, fill=np.nan):
        out = np.full(length, fill)
        out[:len(arr)] = arr
        return out

    t_mat   = np.vstack([_pad(r.traj['t'],   n_steps_max) for r in rays])
    x_mat   = np.vstack([_pad(r.traj[x_key], n_steps_max) for r in rays])
    S_mat   = np.vstack([_pad(r.S_cum,        n_steps_max) for r in rays])
    dJ_mat  = np.vstack([_pad(r.det_J,        n_steps_max) for r in rays])
    if dim == 2:
        y_mat = np.vstack([_pad(r.traj[y_key], n_steps_max) for r in rays])

    # ---- pre‑compute scattered data for each frame ----
    frames_data = []          # each element: (psi, X, Y)   Y may be None in 1D

    # Fixed output grid (built once)
    xs_grid = np.linspace(*xlim, N_grid)
    if dim == 2:
        ys_grid = np.linspace(*ylim, N_grid)
        X_grid, Y_grid = np.meshgrid(xs_grid, ys_grid)
        grid_pts = np.c_[X_grid.ravel(), Y_grid.ravel()]

    for t_target in times:
        # -- For each ray find the last stored step index <= t_target ----------
        # t_mat rows are sorted ascending; searchsorted gives the insertion
        # point, subtract 1 to get the last index <= t_target.
        # Clamp to [0, n_steps_per_ray[i]-1] per ray.
        idx_vec = np.minimum(
            np.searchsorted(t_ref, t_target, side='right') - 1,
            n_steps_max - 1,
        )
        idx_vec = max(idx_vec, 0)   # scalar: same idx for all rays (shared grid)

        # -- Stride-subsampled slice 0 : idx_vec+1 : stride --------------------
        sl = slice(0, idx_vec + 1, max(1, int(stride)))

        x_seg   = x_mat[:, sl]          # (n_rays, n_pts)
        S_seg   = S_mat[:, sl]
        dJ_seg  = dJ_mat[:, sl]
        if dim == 2:
            y_seg = y_mat[:, sl]

        # -- Maslov index per ray at t_target (count sign changes up to idx) --
        # Delegate to _maslov_index so this stays consistent with the main
        # pipeline.  The scalar counter is robust for det_J starting near 0.
        mu_vec = np.zeros(n_rays, dtype=int)
        for i in range(n_rays):
            dj_row = dJ_mat[i, sl]
            finite = np.isfinite(dj_row)
            if finite.sum() >= 2:
                mu_vec[i] = _maslov_index(dj_row[finite])

        # -- Flatten to 1-D point clouds, dropping NaN padding ----------------
        x_flat  = x_seg.ravel()
        S_flat  = S_seg.ravel()
        dJ_flat = dJ_seg.ravel()
        valid   = np.isfinite(x_flat) & np.isfinite(S_flat) & np.isfinite(dJ_flat)
        x_flat, S_flat, dJ_flat = x_flat[valid], S_flat[valid], dJ_flat[valid]

        # Broadcast mu to every point of its ray.
        # Build the full-length repeated array first (n_rays * n_pts_sl), then
        # apply the same `valid` mask so its length matches x_flat exactly.
        # The old code did np.repeat(...)[valid] but computed n_pts_sl *before*
        # the valid mask, so on padded rows the repeat count was correct yet the
        # final index could silently be off when NaN-padding varied per ray.
        n_pts_sl    = x_seg.shape[1]                   # cols per ray after stride
        mu_unmasked = np.repeat(mu_vec, n_pts_sl)      # shape: (n_rays * n_pts_sl,)
        mu_flat     = mu_unmasked[valid]               # same length as x_flat

        if dim == 1:
            pts = x_flat[:, None]
        else:
            y_flat = y_seg.ravel()[valid]
            pts    = np.column_stack([x_flat, y_flat])

        # -- WKB complex amplitude at each scattered point --------------------
        abs_det  = np.abs(dJ_flat)
        reg      = 1e-4
        amp      = 1.0 / np.sqrt(np.maximum(abs_det, reg))
        psi_k    = amp * np.exp(1j * S_flat / result.hbar
                                - 1j * mu_flat * np.pi / 2)

        # -- Grid interpolation: use LinearNDInterpolator in 2D so the
        #    Delaunay triangulation is built once and evaluation is fast -------
        if dim == 1:
            order = np.argsort(pts[:, 0])
            xs_s  = pts[order, 0]
            pk_s  = psi_k[order]
            psi   = (np.interp(xs_grid, xs_s, pk_s.real, left=0, right=0)
                   + 1j * np.interp(xs_grid, xs_s, pk_s.imag, left=0, right=0))
            X_out, Y_out = xs_grid, None
        else:
            try:
                interp_r = LinearNDInterpolator(pts, psi_k.real, fill_value=0.0)
                interp_i = LinearNDInterpolator(pts, psi_k.imag, fill_value=0.0)
                # Share the triangulation object by copying it
                interp_i.tri = interp_r.tri
                psi = (interp_r(grid_pts) + 1j * interp_i(grid_pts)).reshape(N_grid, N_grid)
            except Exception:
                # Fallback to nearest-neighbour if triangulation fails
                interp_r = NearestNDInterpolator(pts, psi_k.real)
                interp_i = NearestNDInterpolator(pts, psi_k.imag)
                psi = (interp_r(grid_pts) + 1j * interp_i(grid_pts)).reshape(N_grid, N_grid)
            X_out, Y_out = X_grid, Y_grid

        frames_data.append((psi, X_out, Y_out))

    # ---- set up figure and artists ----
    if figsize is None:
        figsize = (12, 5) if dim == 2 and plot_type == 'both' else (8, 5)

    if dim == 1:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('$x$')
        ax.set_ylabel('')
        ax.set_title(f'Time = {times[0]:.3f}')
        _style(fig, ax)

        # First frame to initialise lines
        psi0, X0, _ = frames_data[0]
        if plot_type in ('density', 'both'):
            dens0 = np.abs(psi0)**2
            line_dens, = ax.plot(X0, dens0, lw=1.5, color='cyan', label=r'$|\psi|^2$')
        if plot_type in ('phase', 'both'):
            phase0 = np.angle(psi0)
            line_phase, = ax.plot(X0, phase0, lw=1.0, color='magenta', alpha=0.7, label=r'arg $\psi$')
            ax.axhline(0, color='white', lw=0.5, ls='--')
        ax.legend(loc='upper right', fontsize=8)
        artists = [line_dens] if plot_type == 'density' else ([line_phase] if plot_type == 'phase' else [line_dens, line_phase])

        def update_1d(i):
            psi, X, _ = frames_data[i]
            if plot_type in ('density', 'both'):
                line_dens.set_ydata(np.abs(psi)**2)
            if plot_type in ('phase', 'both'):
                line_phase.set_ydata(np.angle(psi))
            ax.set_title(f'Time = {times[i]:.3f}')
            return artists

        ani = animation.FuncAnimation(fig, update_1d, frames=n_frames,
                                      interval=1000/fps, blit=True)

    else:   # 2D
        if plot_type == 'both':
            fig, (ax_dens, ax_phase) = plt.subplots(1, 2, figsize=figsize)
            axes = [ax_dens, ax_phase]
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]

        for a in axes:
            a.set_xlabel('$x$')
            a.set_ylabel('$y$')
            a.set_aspect('equal')
        _style(fig, axes)

        # Initialise images with first frame
        psi0, X0, Y0 = frames_data[0]
        if plot_type in ('density', 'both'):
            dens0 = np.log1p(np.abs(psi0)**2)   # log scale to see details
            im_dens = axes[0].pcolormesh(X0, Y0, dens0, cmap='inferno', shading='auto')
            fig.colorbar(im_dens, ax=axes[0], label=r'$\log(1+|\psi|^2)$')
        if plot_type in ('phase', 'both'):
            phase0 = np.angle(psi0)
            # For both case, phase is second subplot; for 'phase' alone it's the only axis
            ax_phase = axes[1] if plot_type == 'both' else axes[0]
            im_phase = ax_phase.pcolormesh(X0, Y0, phase0, cmap='hsv',
                                            shading='auto', vmin=-np.pi, vmax=np.pi)
            fig.colorbar(im_phase, ax=ax_phase, label=r'$\arg(\psi)$')

        # Collect artists for blitting
        artists = []
        if plot_type in ('density', 'both'):
            artists.append(im_dens)
        if plot_type in ('phase', 'both'):
            artists.append(im_phase)

        def update_2d(i):
            psi, X, Y = frames_data[i]
            if plot_type in ('density', 'both'):
                dens = np.log1p(np.abs(psi)**2)
                im_dens.set_array(dens.ravel())
            if plot_type in ('phase', 'both'):
                phase = np.angle(psi)
                im_phase.set_array(phase.ravel())
            return artists

        ani = animation.FuncAnimation(fig, update_2d, frames=n_frames,
                                      interval=1000/fps, blit=True)

    # ---- save if requested ----
    if save_path:
        ani.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg',
                 fps=fps, dpi=dpi)

    return ani

# =============================================================================
# Six worked examples  (run when script is executed directly)
# =============================================================================
#
# Three 1D and three 2D demonstrations, ordered from elementary to advanced.
#
#  1D ─────────────────────────────────────────────────────────────────────────
#  Ex 1 │ Semiclassical harmonic oscillator
#        │   g = 1/(1 - x²)  (isotropic confining metric)
#        │   Rays focus periodically → Maslov index accumulates,
#        │   Airy patches replace the WKB amplitude at each caustic.
#
#  Ex 2 │ Pöschl–Teller / exponential-barrier metric
#        │   g = cosh²(x)   (localised potential well via curved kinetic energy)
#        │   Rays slow down near x=0, creating a dense fringe cluster and a
#        │   striking amplitude peak at the turning zone.
#
#  Ex 3 │ Power-law / centrifugal metric
#        │   H = p²·x⁴/2  →  g = 1/x⁴
#        │   Rapidly increasing effective mass near x=0 squeezes geodesics
#        │   together; fringe spacing shrinks toward the origin.
#
#  2D ─────────────────────────────────────────────────────────────────────────
#  Ex 4 │ Anisotropic flat metric  g = diag(1, 4)
#        │   Elliptic wavefronts, interference fringes with different
#        │   fringe spacing along x and y, Maslov = 0 everywhere.
#
#  Ex 5 │ Gaussian hill / volcano metric
#        │   g = diag(1 + A exp(-r²/σ²), 1 + A exp(-r²/σ²))
#        │   A localised region of increased effective mass acts as a lens,
#        │   focussing rays and creating a caustic ring with Airy corrections.
#
#  Ex 6 │ Saddle / hyperbolic metric
#        │   g = diag(1/(1+x²), 1+y²)
#        │   Rays accelerate along x and decelerate along y; the resulting
#        │   asymmetric wavefront and ray-crossing pattern produce a rich
#        │   multi-sheet interference figure with non-trivial Maslov structure.
#
# =============================================================================

if __name__ == "__main__":
 
    SEP = "─" * 60
    hbar = 0.15     # small but not tiny: visible fringes, tractable Airy zones
 
    # =========================================================================
    # Example 1 — 1D Semiclassical Harmonic Oscillator
    # =========================================================================
    # The confining metric g(x) = 1/(1 − ω²x²) encodes a position-dependent
    # effective mass that diverges at |x| = 1/ω, mimicking the turning points
    # of a harmonic potential.  Rays launched from x=0 slow down, reflect, and
    # reconverge, creating periodic caustics at the turning points.  Each pair
    # of caustics increments the Maslov index by 2 (one per turning point),
    # reproducing the well-known (n + ½) quantisation of the harmonic oscillator
    # at the Bohr–Sommerfeld level.
    # =========================================================================
    print(SEP)
    print("Example 1 — 1D Semiclassical Harmonic Oscillator")
    print("  g(x) = 1 / (1 − ω²x²),  ω = 1.2")
    print(SEP)
 
    x = sp.Symbol('x', real=True)
    omega = sp.Rational(6, 5)                 # ω = 1.2  (rational for clean SymPy)
    g_ho  = 1 / (1 - omega**2 * x**2)
    metric_ho = Metric(g_ho, (x,))
 
    # Dense fan of slow rays so the envelope and Airy zones are well resolved.
    # The turning point is at |x| = 1/ω ≈ 0.833.  We cap the velocity fan at
    # 0.5 (well below the speed that would reach |x| = 0.833 in t_max = 3.5)
    # and use RK45 which adapts its step size near the turning point, unlike
    # Verlet which uses a fixed step and can overshoot the singularity.
    v_fan_ho = np.linspace(-0.50, 0.50, 50)
 
    result_ho = compute_wavefunction(
        metric    = metric_ho,
        source    = (0.0,),
        v_fan     = v_fan_ho,
        t_max     = 3.5,                      # long enough for two caustic visits
        hbar      = hbar,
        n_steps   = 200,
        N_grid    = 100,
        integrator= 'rk45',                   # adaptive step avoids overshooting
    )
    print(f"  {len(result_ho.rays)} rays integrated successfully.")
    print(f"  Max Maslov index reached: {max(r.mu for r in result_ho.rays)}")
 
    fig1 = plot_wavefunction(result_ho, log_scale=False)
    fig1.suptitle(
        r"Ex 1 — Harmonic oscillator  $g = (1-\omega^2 x^2)^{-1}$"
        rf"   $\hbar={hbar}$",
        color="white", fontsize=12, fontweight="bold", y=1.02)
    plot_interference_detail(result_ho)
    plt.show()
 
    # =========================================================================
    # Example 2 — 1D Pöschl–Teller / Cosh-Barrier Metric
    # =========================================================================
    # The metric g(x) = cosh²(x) corresponds to a kinetic Hamiltonian
    # H = p²/(2 cosh²(x)).  The effective speed of sound is slowest at x=0
    # (where cosh is smallest and the effective mass largest) and increases
    # exponentially away from the origin.  Rays launched with moderate
    # velocity are trapped near the origin and pile up, producing a sharp
    # amplitude peak and densely packed interference fringes — a direct
    # semiclassical analogue of the Pöschl–Teller bound state.
    # =========================================================================
    print(f"\n{SEP}")
    print("Example 2 — 1D Pöschl–Teller / cosh-barrier metric")
    print("  g(x) = cosh²(x)")
    print(SEP)
 
    g_pt     = sp.cosh(x)**2
    metric_pt = Metric(g_pt, (x,))
 
    # Mix of trapped (small |v|) and escaping (large |v|) rays
    v_fan_pt = np.concatenate([
        np.linspace(-2.5, -0.1, 20),
        np.linspace( 0.1,  2.5, 20),
    ])
 
    result_pt = compute_wavefunction(
        metric    = metric_pt,
        source    = (0.0,),
        v_fan     = v_fan_pt,
        t_max     = 2.5,
        hbar      = hbar,
        n_steps   = 200,
        N_grid    = 100,
        integrator= 'verlet',
    )
    print(f"  {len(result_pt.rays)} rays integrated successfully.")
 
    fig2 = plot_wavefunction(result_pt, log_scale=True)
    fig2.suptitle(
        r"Ex 2 — Pöschl–Teller metric  $g = \cosh^2(x)$"
        rf"   $\hbar={hbar}$",
        color="white", fontsize=12, fontweight="bold", y=1.02)
    plot_ray_fan(result_pt)
    plt.show()
 
    # =========================================================================
    # Example 3 — 1D Power-Law Centrifugal Metric
    # =========================================================================
    # Hamiltonian H = p² x⁴ / 2 corresponds to metric g = 1/x⁴.  The
    # effective mass m(x) = x⁴ grows rapidly away from the origin, forcing
    # rays to slow down and turn back.  The steep gradient of the metric
    # produces a rapid change in the local de Broglie wavelength, compressing
    # fringes dramatically near x=0 and stretching them at large |x|.  The
    # non-constant curvature of this metric also drives caustic formation at
    # intermediate distances, visible as amplitude spikes decorated with the
    # characteristic Airy-function fringe pattern.
    # =========================================================================
#    print(f"\n{SEP}")
#    print("Example 3 — 1D power-law centrifugal metric")
#    print("  H = p² x⁴ / 2   →   g(x) = 1/x⁴")
#    print(SEP)
# 
#    x_s, p_s  = sp.symbols('x p', real=True, positive=True)
#    H_pl      = p_s**2 * x_s**4 / 2
#    metric_pl = Metric.from_hamiltonian(H_pl, (x_s,), (p_s,))
#    print(f"  Derived metric: g = {metric_pl.g_expr}")
# 
#    v_fan_pl = np.concatenate([
#        np.linspace(-4.0, -0.2, 20),
#        np.linspace( 0.2,  4.0, 20),
#    ])
# 
#    result_pl = compute_wavefunction(
#        metric    = metric_pl,
#        source    = (1.0,),           # source at x=1 to avoid g singularity at 0
#        v_fan     = v_fan_pl,
#        t_max     = 1.2,
#        hbar      = hbar,
#        n_steps   = 200,
#        N_grid    = 100,
#        integrator= 'verlet',
#    )
#    print(f"  {len(result_pl.rays)} rays integrated successfully.")
# 
#    fig3 = plot_wavefunction(result_pl, log_scale=True)
#    fig3.suptitle(
#        r"Ex 3 — Power-law metric  $H = p^2 x^4/2$"
#        rf"   $\hbar={hbar}$",
#        color="white", fontsize=12, fontweight="bold", y=1.02)
#    plot_interference_detail(result_pl)
#    plt.show()
 
    # =========================================================================
    # Example 4 — 2D Anisotropic Flat Metric
    # =========================================================================
    # The metric g = diag(1, κ²) with κ > 1 stretches the y-axis by a factor κ,
    # making motion in the y-direction effectively slower (heavier).  The
    # result is elliptic wavefronts whose semi-axes ratio is κ, and a
    # characteristic interference pattern with fringes spaced κ times further
    # apart along y than along x.  This is the simplest non-trivial 2D metric
    # and provides the clearest demonstration of how geometry shapes the
    # wavefunction without the complication of curvature.
    # =========================================================================
    print(f"\n{SEP}")
    print("Example 4 — 2D Anisotropic flat metric")
    print("  g = diag(1, 4)   (kappa = 2)")
    print(SEP)
 
    x2, y2   = sp.symbols('x y', real=True)
    kappa    = 2
    g_aniso  = sp.Matrix([[1, 0], [0, kappa**2]])
    metric_aniso = Metric(g_aniso, (x2, y2))
 
    # Circular fan in velocity space — becomes elliptic in position space
    angles     = np.linspace(0, 2*np.pi, 100, endpoint=False)
    speed      = 1.8
    v_fan_aniso = np.column_stack([speed * np.cos(angles),
                                   speed * np.sin(angles)])
 
    result_aniso = compute_wavefunction(
        metric    = metric_aniso,
        source    = (0.0, 0.0),
        v_fan     = v_fan_aniso,
        t_max     = 1.8,
        hbar      = hbar,
        n_steps   = 200,
        N_grid    = 100,
        integrator= 'verlet',
    )
    print(f"  {len(result_aniso.rays)} rays integrated successfully.")
 
    fig4 = plot_wavefunction(result_aniso, log_scale=True)
    fig4.suptitle(
        r"Ex 4 — Anisotropic metric  $g = \mathrm{diag}(1,\,4)$"
        rf"   $\hbar={hbar}$",
        color="white", fontsize=12, fontweight="bold")
    plot_ray_fan(result_aniso)
    plt.show()
 
    # =========================================================================
    # Example 5 — 2D Gaussian Hill / Gravitational Lens Metric
    # =========================================================================
    # A localised bump in the metric, g = (1 + A exp(−(x²+y²)/σ²)) I₂, acts
    # as a refractive index hill: rays entering the bump slow down, bend, and
    # converge on the far side.  The resulting caustic ring is a 2D analogue
    # of the Einstein ring in gravitational lensing.  Rays that pass through
    # the centre of the bump are most strongly deflected; those far from the
    # bump travel essentially as free particles.  The wavefunction shows a
    # bright annular caustic decorated with pointwise Airy fringes (courtesy
    # of the 2D caustic correction), surrounded by concentric WKB interference
    # rings.
    # =========================================================================
    print(f"\n{SEP}")
    print("Example 5 — 2D Gaussian hill / gravitational lens metric")
    print("  g = (1 + 3 exp(-(x²+y²)/0.4)) I₂")
    print(SEP)
 
    x5, y5 = sp.symbols('x y', real=True)
    A5, sigma5 = 3, sp.Rational(2, 5)          # bump amplitude and width
    bump = 1 + A5 * sp.exp(-(x5**2 + y5**2) / sigma5)
    g_lens = sp.Matrix([[bump, 0], [0, bump]])
    metric_lens = Metric(g_lens, (x5, y5))
 
    # Fan launched from well outside the bump so rays cross the lens region
    angles5 = np.linspace(0, 2*np.pi, 100, endpoint=False)
    v_fan_lens = np.column_stack([2.5 * np.cos(angles5),
                                  2.5 * np.sin(angles5)])
 
    result_lens = compute_wavefunction(
        metric    = metric_lens,
        source    = (0.0, 0.0),
        v_fan     = v_fan_lens,
        t_max     = 1.4,
        hbar      = hbar,
        n_steps   = 200,
        N_grid    = 100,
        integrator= 'rk45',       # RK45 for the stiff bump region
    )
    print(f"  {len(result_lens.rays)} rays integrated successfully.")
    print(f"  Maslov index range: 0 – {max(r.mu for r in result_lens.rays)}")
 
    fig5 = plot_wavefunction(result_lens, log_scale=True)
    fig5.suptitle(
        r"Ex 5 — Gaussian lens  $g = (1+3e^{-r^2/0.4})\,I_2$"
        rf"   $\hbar={hbar}$",
        color="white", fontsize=12, fontweight="bold")
    plot_ray_fan(result_lens)
    plot_interference_detail(result_lens)
    plt.show()
 
    # =========================================================================
    # Example 6 — 2D Saddle / Hyperbolic Metric
    # =========================================================================
    # The metric g = diag(1/(1+x²), 1+y²) combines a confining factor in x
    # (effective mass decreasing away from 0, so rays accelerate along x) and
    # a growing factor in y (effective mass increasing, so rays decelerate along
    # y).  The competition between these two effects produces a saddle-shaped
    # phase surface with a non-trivial Lagrangian manifold: rays that start
    # in the same direction but with slightly different angles cross each other
    # at a caustic curve, and the wavefunction accumulates Maslov phases
    # asymmetrically.  The resulting interference figure is visibly asymmetric
    # between x and y, with compressed fringes along x and stretched fringes
    # along y, decorated by Airy patches where the saddle-shaped caustic
    # intersects the output grid.
    # =========================================================================
    print(f"\n{SEP}")
    print("Example 6 — 2D Saddle / hyperbolic metric")
    print("  g = diag(1/(1+x²),  1+y²)")
    print(SEP)
 
    x6, y6 = sp.symbols('x y', real=True)
    g_saddle = sp.Matrix([[1 / (1 + x6**2), 0],
                           [0,               1 + y6**2]])
    metric_saddle = Metric(g_saddle, (x6, y6))
 
    # Dense fan covering all directions, biased toward the saddle axes
    angles6 = np.linspace(0, 2*np.pi, 100, endpoint=False)
    speed6  = 1.6
    v_fan_saddle = np.column_stack([speed6 * np.cos(angles6),
                                    speed6 * np.sin(angles6)])
 
    result_saddle = compute_wavefunction(
        metric    = metric_saddle,
        source    = (0.0, 0.0),
        v_fan     = v_fan_saddle,
        t_max     = 1.6,
        hbar      = hbar,
        n_steps   = 200,
        N_grid    = 100,
        integrator= 'verlet',
    )
    print(f"  {len(result_saddle.rays)} rays integrated successfully.")
    print(f"  Maslov index range: 0 – {max(r.mu for r in result_saddle.rays)}")
 
    fig6 = plot_wavefunction(result_saddle, log_scale=True)
    fig6.suptitle(
        r"Ex 6 — Saddle metric  $g = \mathrm{diag}\!\left(\frac{1}{1+x^2},\,1+y^2\right)$"
        rf"   $\hbar={hbar}$",
        color="white", fontsize=12, fontweight="bold")
    plot_ray_fan(result_saddle)
    plot_interference_detail(result_saddle)
    plt.show()
 
    print(f"\n{SEP}")
    print("All six examples completed.")
    print(SEP)