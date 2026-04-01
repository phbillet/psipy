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

The module also supports three types of first‑order‑in‑time (or second‑order)
partial differential equations, selected by the ``equation`` argument:

* **Schrödinger** (default) — −i ∂ψ/∂t = ψOp(p, ψ)   (Van Vleck propagator)
* **Parabolic** (heat)      — ∂u/∂t = ψOp(p, u)      (real exponent)
* **Wave** (hyperbolic)     — ∂²u/∂t² = ψOp(p, u)    (two branches, ±√H)

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

    * S_k(x, t) = ∫₀ᵗ p · ẋ dt'   (Hamilton's principal function / action) = ∫₀ᵗ g_{ij}(x) vⁱ vʲ dt' (on a pure-metric Hamiltonian)
    * A_k(x)    = 1 / √|det J_k(x, t)|  (Van Vleck amplitude)
    * det J_k   = det(∂x / ∂p₀)         (Jacobi determinant, a.k.a. Van Vleck determinant)
    * μ_k       = number of caustic crossings (Maslov index). Each sign change of det J contributes +1 to μ_k, adding a phase factor exp(−iπ/2) = −i per crossing.

The amplitude A_k diverges when det J = 0, i.e. at **caustics**, which are
the envelopes of the ray family.  Near a caustic the WKB approximation
breaks down and is replaced by a uniform asymptotic approximation:

* In 1D, a **fold caustic** is corrected with the Airy function Ai(ξ) where
  ξ(x) = (α/2ℏ)^{1/3} (x − x_c).  The fringe spacing ∝ ℏ^{1/3} is
  physically correct, and the uniform amplitude is
  ψ(x) ≈ 2π a_c ℏ^{1/6} |α|^{-1/3} · Ai(ξ(x)) · exp(i S_c/ℏ).
* In 2D, **fold caustics** are patched with Airy functions in the transverse
  direction n̂ = ∇det J / |∇det J|, blended with a 2D Gaussian taper.
  **Cusp caustics** (where ∇det J ≈ 0) are treated with the Pearcey integral
  via the `asymptotic` module, giving an O(ℏ^{1/4}) scaling.

For the parabolic (heat) equation ∂u/∂t = ψOp(p, u), the WKB ansatz has a
real exponent, and fold caustics are corrected with the parabolic cylinder
function D_{-1/2}(ζ), ζ = (α/ℏ)^{1/4} (x − x_c).  For the wave equation
∂²u/∂t² = ψOp(p, u), the dispersion relation splits into two branches,
and the wavefunction is the coherent sum of both families.

Connection to the Metric
~~~~~~~~~~~~~~~~~~~~~~~~
For a kinetic Hamiltonian H = ½ gⁱʲ pᵢ pⱼ the canonical momentum is
pᵢ = g_{ij} vʲ (covariant, lowered by the metric), so the action becomes
∫ p · v dt = ∫ g_{ij} vⁱ vʲ dt.  The inverse metric gⁱʲ governs the
Hamiltonian equations of motion (Hamilton's equations), while the metric
g_{ij} maps velocities to momenta.  This distinction is crucial for the
action fallback: when no explicit momentum is stored in the trajectory,
the code reconstructs pᵢ = g_{ij}(x) vʲ and integrates ∫ g_{ij} vⁱ vʲ dt,
which is exact for any Riemannian metric.  The old ∫ v² dt fallback (valid
only for the flat unit‑mass case) is no longer used.

The Jacobi equation governing the evolution of the Jacobi field J = ∂x/∂p₀
along a geodesic is the geodesic deviation equation, which is curvature‑
dependent:

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
    ├── _cumulative_action        — action integral ∫ p · v dt (exact for metric)
    ├── _maslov_index             — count sign changes of det J
    └── van_vleck_sum             — coherent sum onto output grid
        ├── _asymptotic_correction_1d  — Airy patch at 1D fold caustics
        │   └── _airy_argument         — ξ(x) = (α/2ℏ)^{1/3} (x − x_c)
        └── _asymptotic_correction_2d  — Airy / Pearcey at 2D caustics

The entry point accepts either a `Metric` object (geodesic motion) or a
general SymPy Hamiltonian.  For metric mode, the fan of initial conditions
is specified as **velocities** vⁱ (contravariant) via the ``v_fan`` argument;
the module automatically converts them to canonical momenta pᵢ = g_{ij} vʲ
before integrating.  For a general Hamiltonian the fan is given directly as
canonical momenta via ``p_fan``.

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
        Evaluate oscillatory integrals I(λ) = ∫ a(t) exp(iλ φ(t)) dt via
        stationary-phase methods.  Used here only for cusp (Pearcey) caustics
        in 2D, where the quartic normal-form phase φ(t) = t⁴/4 requires the
        specialised Pearcey evaluator.


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

For a general Hamiltonian (e.g. harmonic oscillator)::

    x, xi = sp.symbols('x xi', real=True)
    H = xi**2/2 + x**2/2
    result = compute_wavefunction(
        hamiltonian = H,
        coords      = (x,),
        momenta     = (xi,),
        source      = (0.0,),
        p_fan       = np.linspace(-2, 2, 80),
        t_max       = 5.0,
        hbar        = 0.2,
    )

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
from scipy.interpolate import griddata
from scipy.special import airy as scipy_airy
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

import concurrent.futures
import multiprocessing
from scipy.special import pbdv   # parabolic cylinder functions for heat-type caustics


# ─────────────────────────────────────────────────────────────────────────────
# Equation-type selector
# ─────────────────────────────────────────────────────────────────────────────

class EquationType:
    """
    Selector for the type of PDE solved by :func:`compute_wavefunction`.

    Three first-order-in-time (or second-order) equations are supported.
    Pass one of these three string constants as the ``equation`` argument:

    ``EquationType.SCHRODINGER``  (default, original behaviour)
        −i ∂u/∂t = ψOp(p, u)

        WKB ansatz:  u = A exp( i S/ℏ )
        Phase factor at caustics: exp(−i μ π/2)  (Maslov).
        Caustic correction: Airy function  Ai(ξ).

    ``EquationType.PARABOLIC``
        ∂u/∂t = ψOp(p, u)

        WKB ansatz:  u = A exp( S/ℏ )   (real exponent — no i)
        The action S is accumulated with the *same* classical rays but enters
        the exponent without the imaginary unit.  Consequently:

        * The solution is real-valued (when the initial data are real).
        * There are no oscillatory fringes; instead, the solution concentrates
          exponentially around rays with the largest action.
        * Caustic corrections use the **parabolic cylinder function** D_{-1/2}
          (the real-axis analogue of the Airy function for fold caustics).
        * No Maslov phase — sign changes of det J contribute a real factor
          |det J|^{-1/2} that diverges, patched by D_{-1/2}.

    ``EquationType.WAVE``
        ∂²u/∂t² = ψOp(p, u)

        The dispersion relation is  ω² = H(x, p), giving **two branches**:
        ω₊ = +√H  and  ω₋ = −√H.  For each initial ray direction the code
        integrates two Hamiltonians, H₊ = +√H and H₋ = −√H, effectively
        doubling the ray fan.  The wavefunction is the coherent sum:

            u = Σ_{k,±}  A_k exp( i S_k^±/ℏ − i μ_k^± π/2 )

        where S^± = ∫ p · ẋ dt along rays driven by H₊ / H₋.

        This covers:
        * The scalar wave equation  □u = 0  (H = −|p|²)
        * Acoustics in an inhomogeneous medium  (H = −c²(x)|p|²)
        * Any second-order hyperbolic operator.

    Usage::

        result = compute_wavefunction(
            ...,
            equation = EquationType.WAVE,
        )
    """
    SCHRODINGER = 'schrodinger'
    PARABOLIC   = 'parabolic'
    WAVE        = 'wave'


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
    traj    : dict
    det_J   : np.ndarray
    S_cum   : np.ndarray
    mu      : int


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
    equation  : str = EquationType.SCHRODINGER   # which PDE was solved


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
    metric,             # Metric or None
    equation=EquationType.SCHRODINGER,   # NEW
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
        mu = _maslov_index(det_J, traj)

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

        return RayData(traj=traj_trim, det_J=det_J, S_cum=S_cum, mu=mu)

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
        'equation' : str  (EquationType constant)
        and either
            'g_expr' / 'g_matrix' (for metric mode)
        or
            'H_expr', 'momenta' (for hamiltonian mode)
    """
    try:
        equation = worker_data.get('equation', EquationType.SCHRODINGER)
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
            H_sym, vars_phase, is_metric_mode, metric_obj,
            equation=equation)
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
    
    det_J = jac1['J_x'] * jac2['J_y'] - jac1['J_y'] * jac2['J_x']
    
    # NEW: Store individual Jacobi field components for Maslov counting
    # This allows proper counting even when det_J doesn't change sign
    traj['_J1_x'] = jac1['J_x']
    traj['_J1_y'] = jac1['J_y']
    traj['_J2_x'] = jac2['J_x']
    traj['_J2_y'] = jac2['J_y']
    
    return det_J

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

def _maslov_index(det_J: np.ndarray, traj: Optional[dict] = None) -> int:
    """
    Count the number of caustic crossings (sign changes of det J) along a ray.

    Physical meaning
    ----------------
    The Maslov index μ counts how many times the ray has passed through a
    caustic (a point where det J = 0, i.e. where neighbouring rays focus).
    At each crossing, det J changes sign and the semiclassical wavefunction
    accumulates an extra phase factor exp(−iπ/2) = −i, corresponding to a
    phase advance of −π/2.  The total Maslov correction to the phase is
    −μ π/2.

    For a 1D free particle starting from a point source, det J = t > 0 always
    (no caustics), giving μ = 0.  For a harmonic oscillator the ray fan
    focuses at t = π/ω, T = 2π/ω, ..., incrementing μ by 1 at each focus.

    Algorithm
    ---------
    1. Compute ``signs = np.sign(det_J)``.
    2. Remove exact zeros (the ray is at a caustic; the sign is ill-defined).
    3. Count the number of sign flips in the reduced array.

    A sign flip (+1 → −1 or −1 → +1) corresponds to a single caustic
    crossing.  Multiple consecutive zeros between two non-zero values of the
    same sign are treated as a single pass through the caustic locus, not
    multiple crossings.

    Parameters
    ----------
    det_J : np.ndarray
        Jacobi determinant values along the ray, shape (n_steps,).

    Returns
    -------
    mu : int
        Non-negative integer Maslov index.  Equal to the number of sign
        changes of the non-zero elements of ``det_J``.
    """
    # If trajectory has individual Jacobi fields, count their zeros
    if traj is not None and '_J1_x' in traj:
        # 2D case: count zeros of each Jacobi field component
        mu = 0
        for key in ['_J1_x', '_J1_y', '_J2_x', '_J2_y']:
            if key in traj:
                field = traj[key]
                signs = np.sign(field)
                signs = signs[signs != 0]
                mu += int(np.sum(np.abs(np.diff(signs)) > 0))
        # Each focus in 2D contributes 2 (one per dimension)
        # But we counted each field separately, so divide by 2
        return mu // 2
    
    # Fallback: original sign-change counting for det_J
    signs = np.sign(det_J)
    signs = signs[signs != 0]
    return int(np.sum(np.abs(np.diff(signs)) > 0))


# ─────────────────────────────────────────────────────────────────────────────
# 4 — Caustic corrections using proper Airy / Pearcey profiles
# ─────────────────────────────────────────────────────────────────────────────

def _airy_argument(x_local: np.ndarray, hbar: float, alpha: float) -> np.ndarray:
    """
    Map the local coordinate x_local = x − x_c to the Airy argument ξ(x).

    Derivation
    ----------
    Near a 1D fold caustic at x = x_c, the classical phase is stationary at
    momentum p = p_c.  Expanding the phase to cubic order in the momentum
    deviation δp = p − p_c gives the **cubic normal form**:

        φ(p) = S_c/ℏ + α (p − p_c)³ / 3

    where α = dJ/ds is the slope of the Jacobi determinant with respect to
    the ray-parameter (or equivalently, the coefficient of the cubic term in
    the Legendre transform).

    Performing the stationary-phase integral of exp(i λ φ(p)) over p with
    λ = 1/ℏ, and mapping the result to a function of position x (via the
    inverse Legendre transform), yields the uniform Airy approximation

        ψ(x) ∝ Ai(ξ(x))  with  ξ(x) = (α / (2ℏ))^{1/3} · (x − x_c)

    The factor (α / 2ℏ)^{1/3} sets the correct fringe scale: fringes on the
    illuminated side of the caustic have spacing ∝ ℏ^{1/3}, which is
    parametrically larger than the WKB wavelength ∝ ℏ.

    Sign convention
    ---------------
    Following Berry & Mount (1972), the sign of ξ is chosen so that:

    * ξ > 0  on the shadow side (det J < 0 after the caustic) — Ai(ξ) decays
      exponentially, representing the evanescent tail.
    * ξ < 0  on the illuminated side (det J > 0) — Ai(ξ) oscillates with
      increasing frequency, reproducing the classical interference fringes.

    The factor ``np.sign(alpha)`` implements this convention.

    Parameters
    ----------
    x_local : np.ndarray
        Coordinate relative to the caustic: x_local = x − x_c.  May be
        positive (shadow side) or negative (illuminated side), depending on
        the sign of α.
    hbar : float
        Reduced Planck constant.  The fringe scale ∝ ℏ^{1/3}.
    alpha : float
        Cubic coefficient α = dJ/ds.  Controls the direction of oscillations
        (via its sign) and the fringe frequency (via its magnitude).

    Returns
    -------
    xi : np.ndarray, same shape as x_local
        Airy argument ξ(x), dimensionless.  Zero at the caustic (x_local = 0)
        by construction.
    """
    scale = (abs(alpha) / (2.0 * hbar)) ** (1.0 / 3.0)
    # sign of α controls which side of the caustic has oscillations
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
    Replace the WKB amplitude near a 1D fold caustic with the pointwise
    uniform Airy approximation.

    Background
    ----------
    The WKB amplitude A(x) = 1/√|det J| diverges as det J → 0 at a caustic.
    Near a fold caustic the stationary-phase integral can be evaluated
    uniformly in terms of the Airy function (Berry & Mount 1972):

        ψ(x) ≈ P(ℏ, α) · Ai(ξ(x)) · exp(i S_c/ℏ)

    where:

    * ``P(ℏ, α) = 2π a_c ℏ^{1/6} |α|^{-1/3}`` is the uniform prefactor,
      derived by matching the WKB and Airy asymptotics away from the caustic.
    * ``Ai(ξ)`` is the real Airy function evaluated at ``ξ(x)`` given by
      :func:`_airy_argument`.
    * ``exp(i S_c/ℏ)`` is the carrier phase at the caustic position.

    The patch is multiplied by a cosine² taper to avoid Gibbs-like
    discontinuities at the patch boundary, while preserving the Airy
    zero-crossings and fringe structure deep inside the window.

    This replaces the previous implementation, which evaluated the Airy
    function only once at ξ = 0 and spread the resulting scalar value with
    a cosine taper.  That approach gave the correct O(ℏ^{1/6}) amplitude
    order but the wrong spatial fringe pattern.

    Parameters
    ----------
    x_caustic : float
        Position x_c of the caustic (centre of the correction window).
    S_caustic : float
        Accumulated action S(x_c) at the caustic; determines the carrier
        phase exp(i S_c/ℏ).
    a_caustic : float
        Physical (unregularised) WKB amplitude at the caustic, i.e. the
        value of 1/√|det J| · √|det J_max| recovered by undoing the
        ``reg``-floor applied in :func:`van_vleck_sum`.
    dJ_ds : float
        Local slope of the Jacobi determinant with respect to position at
        the caustic, α = d(det J)/dx.  Used to compute the Airy argument
        and the prefactor amplitude.  Guarded against zero: if ``|dJ_ds|
        < 1e-12`` a fallback value of 1.0 is used.
    hbar : float
        Reduced Planck constant.
    x_grid : np.ndarray, shape (N,)
        Output grid on which to evaluate the Airy correction.
    width : float
        Half-width of the correction window in physical units.  Points
        outside ``|x - x_caustic| >= width`` receive a zero patch.

    Returns
    -------
    patch : np.ndarray (complex), shape (N,)
        Airy-corrected wavefunction contribution near the caustic.
        Zero outside the window ``|x - x_caustic| < width``.
        The caller in :func:`van_vleck_sum` replaces the WKB value wherever
        ``|patch| > 0``.
    """
    patch = np.zeros_like(x_grid, dtype=complex)
    mask  = np.abs(x_grid - x_caustic) < width
    if not np.any(mask):
        return patch

    x_local = x_grid[mask] - x_caustic
    alpha   = float(dJ_ds) if abs(dJ_ds) > 1e-12 else 1.0

    # Airy argument — physically correct pointwise mapping
    xi_arr  = _airy_argument(x_local, hbar, alpha)
    Ai_vals, _, _, _ = scipy_airy(xi_arr)   # real Airy function

    # Uniform amplitude prefactor  2π a_c ℏ^{1/6} |α|^{-1/3}
    prefactor = 2.0 * np.pi * a_caustic * (hbar ** (1.0 / 6.0)) * (abs(alpha) ** (-1.0 / 3.0))

    # Carrier phase from accumulated action at the caustic
    carrier = np.exp(1j * S_caustic / hbar)

    # Smooth edge taper to avoid Gibbs ringing at the patch boundary
    taper = np.cos(np.pi / 2.0 * x_local / width) ** 2

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
    Apply an asymptotic caustic correction on a 2D grid.

    Handles two topologically distinct caustic types based on the gradient
    of the Jacobi determinant at the caustic point:

    Fold caustic (|∇det J| > threshold)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A generic fold caustic in 2D is locally a cylindrical surface: the
    wavefield factorises as

        ψ(x,y) ≈ ψ_Airy(r_⊥) · ψ_WKB(r_∥)

    where r_⊥ is the coordinate transverse to the caustic surface (in the
    direction of ∇det J) and r_∥ is the coordinate along the caustic.

    Algorithm:

    1. Compute the unit normal n̂ = ∇det J / |∇det J|.
    2. For each masked grid point (x, y), compute the signed transverse
       distance r_⊥ = n̂ · (x − x_c, y − y_c).
    3. Evaluate the Airy argument ξ = (|∇det J| / (2ℏ))^{1/3} · r_⊥  via
       :func:`_airy_argument` with α = |∇det J|.
    4. Apply the uniform Airy formula:
       patch = 2π a_c ℏ^{1/6} |α|^{-1/3} · Ai(ξ) · exp(i S_c/ℏ) · taper(r²).
    5. Blend with a radial Gaussian taper exp(−r²/(0.5 width)²) to smoothly
       join the WKB background outside the correction zone.

    Cusp caustic (|∇det J| ≈ 0, ``grad_norm < 1e-10``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When both partial derivatives ∂_x det J and ∂_y det J vanish
    simultaneously, the caustic is a cusp (Pearcey-type singularity).  The
    fold approximation breaks down and a higher-order treatment is required.

    The phase near a cusp has the **quartic normal form** φ(t) = t⁴/4.
    :class:`asymptotic.Analyzer` is initialised with this phase and
    :class:`asymptotic.AsymptoticEvaluator` returns an O(ℏ^{1/4}) Pearcey
    scaling.  Because the full 2D Pearcey integral is expensive to evaluate
    on a grid, the scalar result is spread with a 2D Gaussian taper, which
    gives the correct amplitude order near the cusp but not the exact Pearcey
    fringe pattern.  A full pointwise Pearcey correction is noted as a future
    extension.

    Parameters
    ----------
    x_caustic, y_caustic : float
        Position (x_c, y_c) of the caustic point.
    S_caustic : float
        Accumulated action S(x_c, y_c) at the caustic; sets carrier phase.
    a_caustic : float
        Physical WKB amplitude at the caustic (after undoing regularisation).
    dJ_dx, dJ_dy : float
        Components of the gradient of det J at the caustic point, estimated
        from nearby scattered ray data in :func:`van_vleck_sum`.
    hbar : float
        Reduced Planck constant.
    X_grid, Y_grid : np.ndarray, shape (N, N)
        Meshgrid arrays (output of ``np.meshgrid``) on which to evaluate the
        correction.
    width : float
        Radius of the correction disk in physical units.  Points outside
        ``r = sqrt((x-x_c)² + (y-y_c)²) >= width`` receive a zero patch.

    Returns
    -------
    patch : np.ndarray (complex), shape (N, N)
        Caustic correction on the 2D grid.  Zero outside the disk of radius
        ``width`` centred on ``(x_caustic, y_caustic)``.
    """
    patch = np.zeros_like(X_grid, dtype=complex)
    r2    = (X_grid - x_caustic)**2 + (Y_grid - y_caustic)**2
    mask  = r2 < width**2

    if not np.any(mask):
        return patch

    grad_norm = np.hypot(dJ_dx, dJ_dy)

    # ── Cusp (Pearcey) caustic: both partial derivatives vanish ──────────────
    if grad_norm < 1e-10:
        # Use the asymptotic.Analyzer scalar approach (as documented)
        t_sym  = sp.Symbol('t', real=True)
        phase_sym = sp.Rational(1, 4) * t_sym**4   # quartic normal form

        try:
            analyzer  = Analyzer(
                phase_expr     = phase_sym,
                amplitude_expr = sp.Integer(1),
                variables      = [t_sym],
                method         = IntegralMethod.STATIONARY_PHASE,
            )
            evaluator = AsymptoticEvaluator()
            xc_pt     = np.array([0.0])
            cp        = analyzer.analyze_point(xc_pt)
            contrib   = evaluator.evaluate(cp, 1.0 / hbar)
            scalar    = contrib.total_value * a_caustic * np.exp(1j * S_caustic / hbar)
        except Exception:
            scalar = a_caustic * np.exp(1j * S_caustic / hbar)

        gauss = np.exp(-r2 / (0.5 * width)**2)
        patch[mask] = scalar * gauss[mask]
        return patch

    # ── Fold caustic: Airy along transverse direction ─────────────────────────
    # Unit normal to the caustic (direction of det-J gradient)
    nx = dJ_dx / grad_norm
    ny = dJ_dy / grad_norm

    # Transverse coordinate of each masked grid point
    dx_arr = X_grid[mask] - x_caustic
    dy_arr = Y_grid[mask] - y_caustic
    r_perp  = nx * dx_arr + ny * dy_arr    # signed transverse distance

    # Airy argument along the transverse direction
    alpha   = grad_norm                    # |∇det J| acts as the cubic coefficient
    xi_arr  = _airy_argument(r_perp, hbar, alpha)
    Ai_vals, _, _, _ = scipy_airy(xi_arr)

    prefactor = (2.0 * np.pi * a_caustic
                 * (hbar ** (1.0 / 6.0))
                 * (abs(alpha) ** (-1.0 / 3.0)))
    carrier   = np.exp(1j * S_caustic / hbar)

    # Gaussian taper in 2D (radial)
    taper = np.exp(-r2[mask] / (0.5 * width)**2)

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
        psi    = (griddata(values=psi_k.real, **kw)
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
# 5b — Parabolic (heat-type) coherent sum   ∂u/∂t = ψOp u
# ─────────────────────────────────────────────────────────────────────────────

def _pcf_argument(x_local: np.ndarray, hbar: float, alpha: float) -> np.ndarray:
    """
    Map the local coordinate x_local = x − x_c to the argument of the
    parabolic cylinder function D_{-1/2}(ζ) used near a fold caustic of the
    **parabolic (heat-type)** equation.

    Background
    ----------
    For the heat-type PDE  ∂u/∂t = ψOp u  the WKB amplitude is
    A = exp(S/ℏ) / √|det J|, real-valued.  Near a fold caustic the uniform
    approximation replaces the singular √|det J| by the parabolic cylinder
    function  D_{-1/2}(ζ)  (real axis), with argument

        ζ(x) = (α / ℏ)^{1/4} · (x − x_c)

    where α = d(det J)/dx is the local slope of the Jacobi determinant.
    The ℏ^{1/4} fringe scale is coarser than the WKB scale ℏ, consistent
    with the real (diffusive) nature of the equation.

    Parameters
    ----------
    x_local : np.ndarray   (x − x_c)
    hbar    : float
    alpha   : float        (d det_J / dx at caustic)

    Returns
    -------
    zeta : np.ndarray
    """
    scale = (abs(alpha) / hbar) ** 0.25
    return np.sign(alpha) * scale * x_local


def _parabolic_correction_1d(
    x_caustic : float,
    S_caustic : float,
    a_caustic : float,
    dJ_ds     : float,
    hbar      : float,
    x_grid    : np.ndarray,
    width     : float,
) -> np.ndarray:
    """
    Replace the real WKB amplitude near a 1D fold caustic with the uniform
    **parabolic cylinder** approximation for the heat-type equation.

    For ∂u/∂t = ψOp u the leading-order WKB approximation is

        u(x) ≈ A_c · ℏ^{1/4} |α|^{-1/4} · D_{-1/2}(ζ(x)) · exp(S_c/ℏ)

    where D_{-1/2} is the parabolic cylinder function of order −½ and
    ζ(x) = (α/ℏ)^{1/4}(x − x_c).  There is no imaginary unit; the solution
    remains real.

    The scipy implementation ``pbdv(ν, ζ)`` returns (D_ν(ζ), D_ν'(ζ)); we
    use ν = −½.

    Parameters
    ----------
    (identical layout to :func:`_asymptotic_correction_1d`)

    Returns
    -------
    patch : np.ndarray (real, cast to complex for uniform API)
    """
    patch = np.zeros_like(x_grid, dtype=complex)
    mask  = np.abs(x_grid - x_caustic) < width
    if not np.any(mask):
        return patch

    x_local = x_grid[mask] - x_caustic
    alpha   = float(dJ_ds) if abs(dJ_ds) > 1e-12 else 1.0

    zeta     = _pcf_argument(x_local, hbar, alpha)
    D_vals, _ = pbdv(-0.5, zeta)          # D_{-1/2}(ζ)

    # Uniform prefactor  A_c · ℏ^{1/4} |α|^{-1/4}
    prefactor = a_caustic * (hbar ** 0.25) * (abs(alpha) ** (-0.25))
    carrier   = np.exp(S_caustic / hbar)   # real exponential (no i)
    taper     = np.cos(np.pi / 2.0 * x_local / width) ** 2

    patch[mask] = (prefactor * D_vals * carrier * taper).astype(complex)
    return patch


def _parabolic_correction_2d(
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
    Apply the parabolic cylinder correction at a 2D fold caustic for the
    heat-type equation.

    Fold caustic: the correction is applied along the transverse direction
    n̂ = ∇det J / |∇det J|, exactly as in the Schrödinger case, but using
    D_{-1/2}(ζ) instead of Ai(ξ) and a real exponential carrier.

    Cusp caustic (|∇det J| ≈ 0): a scalar D_{-1/2}(0) value spread with a
    Gaussian taper (same strategy as the Pearcey fallback in Schrödinger).
    """
    patch = np.zeros_like(X_grid, dtype=complex)
    r2    = (X_grid - x_caustic)**2 + (Y_grid - y_caustic)**2
    mask  = r2 < width**2
    if not np.any(mask):
        return patch

    grad_norm = np.hypot(dJ_dx, dJ_dy)
    carrier   = float(np.exp(S_caustic / hbar))

    if grad_norm < 1e-10:
        # Cusp: scalar PCF value spread with Gaussian
        D_val, _ = pbdv(-0.5, np.array([0.0]))
        scalar   = float(a_caustic * (hbar ** 0.25) * D_val[0] * carrier)
        gauss    = np.exp(-r2 / (0.5 * width)**2)
        patch[mask] = (scalar * gauss[mask]).astype(complex)
        return patch

    # Fold: transverse direction
    nx = dJ_dx / grad_norm
    ny = dJ_dy / grad_norm
    dx_arr = X_grid[mask] - x_caustic
    dy_arr = Y_grid[mask] - y_caustic
    r_perp  = nx * dx_arr + ny * dy_arr

    alpha    = grad_norm
    zeta     = _pcf_argument(r_perp, hbar, alpha)
    D_vals, _ = pbdv(-0.5, zeta)

    prefactor = a_caustic * (hbar ** 0.25) * (abs(alpha) ** (-0.25))
    taper     = np.exp(-r2[mask] / (0.5 * width)**2)
    patch[mask] = (prefactor * D_vals * carrier * taper).astype(complex)
    return patch


def parabolic_sum(
    pts    : np.ndarray,
    S      : np.ndarray,
    det_J  : np.ndarray,
    xlim   : Tuple[float, float],
    ylim   : Optional[Tuple[float, float]] = None,
    N      : int   = 300,
    hbar   : float = 1.0,
    reg    : float = 1e-4,
    method : str   = 'linear',
    caustic_threshold : float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Assemble the semiclassical solution for the **parabolic (heat-type)**
    equation  ∂u/∂t = ψOp u  on a regular grid.

    WKB formula
    -----------
    Unlike the Schrödinger case there is no imaginary unit in the exponent:

        u_k(x) = exp( S_k(x)/ℏ ) / √max(|det J_k|, reg)

    The real exponential means:

    * Rays with larger action dominate exponentially.
    * There are no oscillatory fringes between ray contributions.
    * At caustics the amplitude diverges as in the Schrödinger case, but is
      patched with the parabolic cylinder function D_{-1/2} instead of Ai.
    * The Maslov index is irrelevant (no phase); sign changes of det J are
      handled by the absolute value and the caustic patch.
    * ``mu`` is not required for the parabolic equation.

    The interpolation strategy is identical to :func:`van_vleck_sum`.

    Parameters
    ----------
    pts, S, det_J, xlim, ylim, N, hbar, reg, method, caustic_threshold :
        Same meaning as in :func:`van_vleck_sum`.

    Returns
    -------
    u, X, Y : same layout as :func:`van_vleck_sum`.

    """
    abs_det = np.abs(det_J)
    amp     = 1.0 / np.sqrt(np.maximum(abs_det, reg))
    u_k     = amp * np.exp(S / hbar)          # real — no i, no Maslov

    det_max   = abs_det.max() if abs_det.max() > 0 else 1.0
    near_caus = abs_det < caustic_threshold * det_max

    if ylim is None:
        # ── 1D ───────────────────────────────────────────────────────────────
        x_grid = np.linspace(*xlim, N)
        order  = np.argsort(pts[:, 0])
        xs, uk_ord = pts[order, 0], u_k[order]
        u = (np.interp(x_grid, xs, uk_ord.real, left=0, right=0)
           + 1j * np.interp(x_grid, xs, uk_ord.imag, left=0, right=0))

        if np.any(near_caus):
            caus_xs = pts[near_caus, 0]
            span    = xlim[1] - xlim[0]
            for xc in caus_xs[np.argsort(caus_xs)]:
                idx_c = np.argmin(np.abs(pts[:, 0] - xc))
                S_c   = float(S[idx_c])
                a_c   = float(amp[idx_c]) * float(det_max) ** 0.5
                nearby = np.abs(pts[:, 0] - xc) < 0.05 * span
                dJ_ds = (float(np.gradient(det_J[nearby],
                                           pts[nearby, 0]).mean())
                         if nearby.sum() >= 2 else 1.0)
                width = max(0.04 * span, 3 * (x_grid[1] - x_grid[0]))
                patch = _parabolic_correction_1d(
                    xc, S_c, a_c, dJ_ds, hbar, x_grid, width)
                blend = np.abs(patch) > 0
                u[blend] = patch[blend]
        return u, x_grid, None

    else:
        # ── 2D ───────────────────────────────────────────────────────────────
        xs, ys = np.linspace(*xlim, N), np.linspace(*ylim, N)
        X, Y   = np.meshgrid(xs, ys)
        grid   = np.c_[X.ravel(), Y.ravel()]
        kw     = dict(points=pts, xi=grid, method=method, fill_value=0.0)
        u      = (griddata(values=u_k.real, **kw)
                + 1j * griddata(values=u_k.imag, **kw)).reshape(N, N)

        if np.any(near_caus):
            span_x = xlim[1] - xlim[0]
            span_y = ylim[1] - ylim[0]
            span   = min(span_x, span_y)
            for idx_c in np.where(near_caus)[0]:
                xc  = float(pts[idx_c, 0])
                yc  = float(pts[idx_c, 1])
                S_c = float(S[idx_c])
                a_c = float(amp[idx_c]) * float(det_max) ** 0.5
                nearby = ((np.abs(pts[:, 0] - xc) < 0.05 * span_x) &
                          (np.abs(pts[:, 1] - yc) < 0.05 * span_y))
                if nearby.sum() >= 3:
                    dJ_dx = float(np.gradient(det_J[nearby],
                                              pts[nearby, 0]).mean())
                    dJ_dy = float(np.gradient(det_J[nearby],
                                              pts[nearby, 1]).mean())
                else:
                    dJ_dx, dJ_dy = 1.0, 0.0
                width = max(0.04 * span, 3 * (xs[1] - xs[0]))
                patch = _parabolic_correction_2d(
                    xc, yc, S_c, a_c, dJ_dx, dJ_dy, hbar, X, Y, width)
                blend = np.abs(patch) > 0
                u[blend] = patch[blend]
        return u, X, Y


# ─────────────────────────────────────────────────────────────────────────────
# 5c — Wave (hyperbolic) coherent sum   ∂²u/∂t² = ψOp u
# ─────────────────────────────────────────────────────────────────────────────

def wave_sum(
    pts_plus  : np.ndarray,
    S_plus    : np.ndarray,
    det_J_plus: np.ndarray,
    mu_plus   : np.ndarray,
    pts_minus : np.ndarray,
    S_minus   : np.ndarray,
    det_J_minus: np.ndarray,
    mu_minus  : np.ndarray,
    xlim      : Tuple[float, float],
    ylim      : Optional[Tuple[float, float]] = None,
    N         : int   = 300,
    hbar      : float = 1.0,
    reg       : float = 1e-4,
    method    : str   = 'linear',
    caustic_threshold : float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Assemble the semiclassical wavefunction for the **wave (hyperbolic)**
    equation  ∂²u/∂t² = ψOp u  on a regular grid.

    Two-branch structure
    --------------------
    The dispersion relation ω² = H factors into two branches H₊ = +√H and
    H₋ = −√H, each generating its own family of classical rays.  The
    wavefunction is the coherent superposition:

        u(x) = Σ_{k∈H₊}  A_k exp(i S_k⁺/ℏ − i μ_k⁺ π/2) + Σ_{k∈H₋}  A_k exp(i S_k⁻/ℏ − i μ_k⁻ π/2)

    where S_k^± = ∫₀ᵗ p · ẋ dt′ along the respective branch rays.  Each
    branch is summed via :func:`van_vleck_sum`, then the two grids are added.

    Caustic corrections (Airy / Pearcey) are applied independently on each
    branch before superposition.

    Parameters
    ----------
    pts_plus, S_plus, det_J_plus, mu_plus :
        Scattered ray data for the H₊ = +√H branch.
    pts_minus, S_minus, det_J_minus, mu_minus :
        Scattered ray data for the H₋ = −√H branch.
    xlim, ylim, N, hbar, reg, method, caustic_threshold :
        Grid and numerical parameters (same as :func:`van_vleck_sum`).

    Returns
    -------
    u, X, Y : same layout as :func:`van_vleck_sum`.
    """
    u_plus,  X, Y = van_vleck_sum(
        pts_plus,  S_plus,  det_J_plus,  mu_plus,
        xlim=xlim, ylim=ylim, N=N, hbar=hbar,
        reg=reg, method=method,
        caustic_threshold=caustic_threshold,
    )
    u_minus, _, _ = van_vleck_sum(
        pts_minus, S_minus, det_J_minus, mu_minus,
        xlim=xlim, ylim=ylim, N=N, hbar=hbar,
        reg=reg, method=method,
        caustic_threshold=caustic_threshold,
    )
    return u_plus + u_minus, X, Y


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


def _build_wave_hamiltonians(
    H_sym      : sp.Expr,
    vars_phase : list,
) -> Tuple[sp.Expr, sp.Expr]:
    """
    Construct the two smooth dispersion branches H₊ and H₋ for the wave
    equation  ∂²u/∂t² = ψOp(p, u).

    Physical background
    -------------------
    The WKB ansatz u = A exp(i S/ℏ) reduces the wave PDE to the eikonal
    equation  (∂S/∂t)² = H(x, ∇S), which factors into two branches:

        H₊ =  +√H    (forward-propagating)
        H₋ =  −√H    (backward-propagating)

    The wavefunction is the coherent sum over both ray families.

    The differentiability problem
    -----------------------------
    The naïve ``sp.sqrt(H)`` fails for the most common wave Hamiltonians.
    For  H = f(x)·ξ²  (1D acoustic):  √H = √f·|ξ|, which is non-differentiable
    at ξ = 0 — the symplectic integrator needs ∂H/∂ξ = √f·sign(ξ), which is
    discontinuous.  This makes every ray fail with a numerical error near p = 0.

    Solution — analytic factoring by momentum degree
    -------------------------------------------------
    1. **1D quadratic**  H = a(x)·ξ²:
       Return  H± = ±√a(x)·ξ  (linear, smooth everywhere).
       The signed form is correct because the ray fan covers both ξ > 0 and
       ξ < 0, so both propagation directions are represented without |ξ|.

    2. **2D quadratic form**  H = a·ξ² + 2b·ξη + c·η²:
       Use sp.sqrt directly — SymPy often simplifies positive-definite forms.

    3. **Perfect square**  H = expr²:
       Return ±expr immediately.

    4. **Fallback**: assume momenta positive, take sqrt, restore unsigned symbols.

    Parameters
    ----------
    H_sym : sp.Expr
        Spatial Hamiltonian H(x, p) ≥ 0 on the real phase space.
    vars_phase : list of sp.Symbol
        Phase-space variables [x, ξ] (1D) or [x, ξ, y, η] (2D).

    Returns
    -------
    H_plus, H_minus : sp.Expr, sp.Expr
        Smooth branch Hamiltonians for ``hamiltonian_flow``.
    """
    # Extract momentum symbols (every second entry of vars_phase)
    mom_syms = vars_phase[1::2]   # [ξ] or [ξ, η]

    # ── Step 1: simplify H ────────────────────────────────────────────────────
    H_simplified = sp.powsimp(sp.expand(H_sym), force=True)

    # ── Step 2: 1D purely-quadratic case  H = a(x)·ξ²  →  H± = ±√a(x)·ξ ────
    # This is the most common wave Hamiltonian (acoustic, Schrödinger symbol).
    # √(a·ξ²) = √a·|ξ| is NOT smooth; use ±√a·ξ (signed, linear in ξ).
    # The fan covers both ξ>0 and ξ<0, so both propagation directions are
    # represented without needing |ξ|.
    if len(mom_syms) == 1:
        xi = mom_syms[0]
        try:
            poly_H = sp.Poly(H_simplified, xi)
            if poly_H.total_degree() == 2:
                a = poly_H.nth(2)   # coeff of ξ²
                b = poly_H.nth(1)   # coeff of ξ  (should be 0 for even H)
                c = poly_H.nth(0)   # constant in ξ (should be 0)
                if b == 0 and c == 0 and a != 0:
                    sqrt_a  = sp.sqrt(sp.simplify(a))
                    return sqrt_a * xi, -sqrt_a * xi
        except Exception:
            pass

    # ── Step 3: 2D  H = a·ξ² + 2b·ξ·η + c·η²  →  use sp.sqrt(H) directly ──
    # For the diagonal case H = a·ξ² + c·η² the branches are not linear in
    # momenta, but sp.sqrt of a positive-definite quadratic form is smooth
    # everywhere except the origin (which is never reached by non-trivial rays).
    # SymPy can handle this symbolically when a, c are positive.
    if len(mom_syms) == 2:
        xi, eta = mom_syms
        try:
            poly_H = sp.Poly(H_simplified, xi, eta)
            if poly_H.total_degree() == 2:
                # Attempt direct sqrt — SymPy may simplify to a clean expression
                sqrt_H = sp.sqrt(H_simplified)
                sqrt_H_s = sp.simplify(sqrt_H)
                return sqrt_H_s, -sqrt_H_s
        except Exception:
            pass

    # ── Step 4: check if H is already a perfect square  (H = expr²) ─────────
    H_factored = sp.factor(H_simplified)
    if H_factored.is_Pow and H_factored.exp == 2:
        base = H_factored.base
        return base, -base

    # ── Step 5: assume momenta positive, take sqrt, restore signs ────────────
    # Replaces |p| by p in the final expression (valid since fan covers ±p).
    pos_subs = {p: sp.Symbol(str(p), positive=True) for p in mom_syms}
    H_pos    = H_simplified.subs(pos_subs)
    sqrt_pos = sp.sqrt(H_pos)
    inv_subs = {v: k for k, v in pos_subs.items()}
    H_plus   = sqrt_pos.subs(inv_subs)
    return H_plus, -H_plus


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
    # ── equation type ─────────────────────────────────────────
    equation     : str  = EquationType.SCHRODINGER,
) -> WKBResult:
    """
    Compute the semiclassical (Van Vleck–Pauli–Morette) wavefunction.

    This is the main public entry point.  It accepts **two distinct input
    modes** depending on whether you supply a ``Metric`` object (pure kinetic,
    geodesic motion) or an explicit SymPy Hamiltonian expression (general
    T + V systems).

    The returned wavefunction `psi` is evaluated **only at the final time**
    `t = t_max` using the **endpoint** of each classical ray.  (Full ray
    histories are stored in `result.rays` and can be used for animation.)

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
        specified directly as a fan of **canonical momenta** p₀.

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
    equation : str, default ``EquationType.SCHRODINGER``
        Type of PDE: ``'schrodinger'``, ``'parabolic'``, or ``'wave'``.

    Returns
    -------
    WKBResult
        Dataclass containing:
        - ``psi`` : wavefunction on the grid at `t = t_max`.
        - ``X``, ``Y`` : grid coordinates.
        - ``rays`` : list of :class:`RayData` with full time histories.
        - ``x_pts``, ``y_pts`` : scattered positions of **all** ray points
          (useful for debugging, but note that the static ``psi`` uses
          only the final points).
        - ``S_pts``, ``det_J_pts``, ``mu_pts`` : corresponding action,
          Jacobi determinant and Maslov index.
        - ``hbar``, ``t_max``, ``dim``, ``equation``.

    Raises
    ------
    ValueError
        If the input mode cannot be resolved.
    RuntimeError
        If every ray in the fan fails to integrate.
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
        fan = [np.asarray(p, dtype=float) for p in p_fan]

    # Build the data needed to reconstruct the objects in workers
    if is_metric_mode:
        worker_data = {
            'mode': 'metric',
            'dim': dim,
            'coords': metric.coords,
            'equation': equation,
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
            'equation': equation,
        }

    # ── For the wave equation: build the two branch Hamiltonians ─────────────
    if equation == EquationType.WAVE:
        H_plus_sym, H_minus_sym = _build_wave_hamiltonians(H_sym, vars_phase)
        worker_data_plus  = dict(worker_data)
        worker_data_minus = dict(worker_data)
        if is_metric_mode:
            # Replace with branch Hamiltonians in hamiltonian-mode worker format
            worker_data_plus = {
                'mode': 'hamiltonian', 'dim': dim,
                'coords': tuple(vars_phase[::2]),
                'momenta': tuple(vars_phase[1::2]),
                'H_expr': H_plus_sym,
                'equation': equation,
            }
            worker_data_minus = dict(worker_data_plus)
            worker_data_minus['H_expr'] = H_minus_sym
        else:
            worker_data_plus['H_expr']  = H_plus_sym
            worker_data_minus['H_expr'] = H_minus_sym

    def _run_fan(wdata):
        """Integrate a full ray fan with the given worker_data."""
        result_rays = []
        if parallel:
            ctx = multiprocessing.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=None, mp_context=ctx
            ) as executor:
                future_to_idx = {
                    executor.submit(
                        _worker_process_ray,
                        p, source, t_max, hbar, n_steps, integrator, wdata
                    ): i for i, p in enumerate(fan)
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    r = future.result()
                    if r is not None:
                        result_rays.append(r)
        else:
            # resolve H_sym for sequential run
            if wdata['mode'] == 'metric':
                if wdata['dim'] == 1:
                    _m = Metric(wdata['g_expr'], wdata['coords'])
                else:
                    _m = Metric(wdata['g_matrix'], wdata['coords'])
                _H, _vp = _build_hamiltonian_sym(_m)
                _is_m, _m_obj = True, _m
            else:
                _dim = wdata['dim']
                _c, _p = wdata['coords'], wdata['momenta']
                _H = wdata['H_expr']
                _vp = ([_c[0], _p[0]] if _dim == 1
                       else [_c[0], _p[0], _c[1], _p[1]])
                _is_m, _m_obj = False, None
            for p in fan:
                r = _process_single_ray_internal(
                    p, source, t_max, hbar, n_steps, integrator,
                    _H, _vp, _is_m, _m_obj,
                    equation=wdata.get('equation', EquationType.SCHRODINGER))
                if r is not None:
                    result_rays.append(r)
        return result_rays

    rays = []
    first_exc = None

    if equation == EquationType.WAVE:
        # Integrate both branches
        rays_plus  = _run_fan(worker_data_plus)
        rays_minus = _run_fan(worker_data_minus)
        if not rays_plus and not rays_minus:
            raise RuntimeError("All rays failed to integrate on both wave branches.")
        rays = rays_plus + rays_minus
    else:
        rays = _run_fan(worker_data)

    if not rays:
        raise RuntimeError("All rays failed to integrate.")

    # ── collect scattered data (only the final time step) ─────────────────────
    # For animation we keep the full ray data in result.rays, but the static
    # wavefunction psi is built exclusively from the endpoints at t = t_max.
    # This makes the static result identical to the last frame of the animation.

    if dim == 1:
        x_key = str(vars_phase[0])

        if equation == EquationType.WAVE:
            # Plus branch
            x_p_last = np.array([r.traj[x_key][-1] for r in rays_plus])
            S_p_last = np.array([r.S_cum[-1] for r in rays_plus])
            dJ_p_last = np.array([r.det_J[-1] for r in rays_plus])
            mu_p_last = np.array([r.mu for r in rays_plus])

            # Minus branch
            x_m_last = np.array([r.traj[x_key][-1] for r in rays_minus])
            S_m_last = np.array([r.S_cum[-1] for r in rays_minus])
            dJ_m_last = np.array([r.det_J[-1] for r in rays_minus])
            mu_m_last = np.array([r.mu for r in rays_minus])

            pts_p = x_p_last[:, None]
            pts_m = x_m_last[:, None]

            if xlim is None:
                all_x = np.concatenate([x_p_last, x_m_last])
                m = 0.1 * (all_x.max() - all_x.min())
                xlim = (all_x.min() - m, all_x.max() + m)

            psi, X, Y = wave_sum(
                pts_p, S_p_last, dJ_p_last, mu_p_last,
                pts_m, S_m_last, dJ_m_last, mu_m_last,
                xlim=xlim, N=N_grid, hbar=hbar,
            )
        else:
            # Non‑wave: Schrödinger or parabolic
            x_last = np.array([r.traj[x_key][-1] for r in rays])
            S_last = np.array([r.S_cum[-1] for r in rays])
            dJ_last = np.array([r.det_J[-1] for r in rays])
            mu_last = np.array([r.mu for r in rays])

            pts = x_last[:, None]   # shape (n_rays, 1)

            if xlim is None:
                m = 0.1 * (x_last.max() - x_last.min())
                xlim = (x_last.min() - m, x_last.max() + m)

            if equation == EquationType.PARABOLIC:
                psi, X, Y = parabolic_sum(
                    pts, S_last, dJ_last,
                    xlim=xlim, N=N_grid, hbar=hbar,
                )
            else:
                psi, X, Y = van_vleck_sum(
                    pts, S_last, dJ_last, mu_last,
                    xlim=xlim, N=N_grid, hbar=hbar,
                )

        # For completeness, also collect all points (used for scatter data)
        x_all = np.concatenate([r.traj[x_key] for r in rays])
        S_all = np.concatenate([r.S_cum for r in rays])
        dJ_all = np.concatenate([r.det_J for r in rays])
        mu_all = np.concatenate([np.full(len(r.det_J), r.mu) for r in rays])

        return WKBResult(
            rays=rays, X=X, Y=None, psi=psi,
            x_pts=x_all, y_pts=None,
            S_pts=S_all, det_J_pts=dJ_all, mu_pts=mu_all,
            hbar=hbar, t_max=t_max, dim=1, equation=equation,
        )

    else:  # dim == 2
        x_key, y_key = str(vars_phase[0]), str(vars_phase[2])

        if equation == EquationType.WAVE:
            # Plus branch
            x_p_last = np.array([r.traj[x_key][-1] for r in rays_plus])
            y_p_last = np.array([r.traj[y_key][-1] for r in rays_plus])
            S_p_last = np.array([r.S_cum[-1] for r in rays_plus])
            dJ_p_last = np.array([r.det_J[-1] for r in rays_plus])
            mu_p_last = np.array([r.mu for r in rays_plus])

            # Minus branch
            x_m_last = np.array([r.traj[x_key][-1] for r in rays_minus])
            y_m_last = np.array([r.traj[y_key][-1] for r in rays_minus])
            S_m_last = np.array([r.S_cum[-1] for r in rays_minus])
            dJ_m_last = np.array([r.det_J[-1] for r in rays_minus])
            mu_m_last = np.array([r.mu for r in rays_minus])

            pts_p = np.c_[x_p_last, y_p_last]
            pts_m = np.c_[x_m_last, y_m_last]

            if xlim is None:
                all_x = np.concatenate([x_p_last, x_m_last])
                mx = 0.1 * (all_x.max() - all_x.min())
                xlim = (all_x.min() - mx, all_x.max() + mx)
            if ylim is None:
                all_y = np.concatenate([y_p_last, y_m_last])
                my = 0.1 * (all_y.max() - all_y.min())
                ylim = (all_y.min() - my, all_y.max() + my)

            psi, X, Y = wave_sum(
                pts_p, S_p_last, dJ_p_last, mu_p_last,
                pts_m, S_m_last, dJ_m_last, mu_m_last,
                xlim=xlim, ylim=ylim, N=N_grid, hbar=hbar,
            )
        else:
            # Non‑wave
            x_last = np.array([r.traj[x_key][-1] for r in rays])
            y_last = np.array([r.traj[y_key][-1] for r in rays])
            S_last = np.array([r.S_cum[-1] for r in rays])
            dJ_last = np.array([r.det_J[-1] for r in rays])
            mu_last = np.array([r.mu for r in rays])

            pts = np.c_[x_last, y_last]

            if xlim is None:
                mx = 0.1 * (x_last.max() - x_last.min())
                xlim = (x_last.min() - mx, x_last.max() + mx)
            if ylim is None:
                my = 0.1 * (y_last.max() - y_last.min())
                ylim = (y_last.min() - my, y_last.max() + my)

            if equation == EquationType.PARABOLIC:
                psi, X, Y = parabolic_sum(
                    pts, S_last, dJ_last,
                    xlim=xlim, ylim=ylim, N=N_grid, hbar=hbar,
                )
            else:
                psi, X, Y = van_vleck_sum(
                    pts, S_last, dJ_last, mu_last,
                    xlim=xlim, ylim=ylim, N=N_grid, hbar=hbar,
                )

        # Collect all points for scatter data
        x_all = np.concatenate([r.traj[x_key] for r in rays])
        y_all = np.concatenate([r.traj[y_key] for r in rays])
        S_all = np.concatenate([r.S_cum for r in rays])
        dJ_all = np.concatenate([r.det_J for r in rays])
        mu_all = np.concatenate([np.full(len(r.det_J), r.mu) for r in rays])

        return WKBResult(
            rays=rays, X=X, Y=Y, psi=psi,
            x_pts=x_all, y_pts=y_all,
            S_pts=S_all, det_J_pts=dJ_all, mu_pts=mu_all,
            hbar=hbar, t_max=t_max, dim=2, equation=equation,
        )

# ─────────────────────────────────────────────────────────────────────────────
# 7 — Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_BG   = "#0e0e1a"
_DARK = "#444"

# Known momentum / non-position trajectory keys that must never be mistaken
# for coordinate keys.  This explicit set is used by every plotting and
# animation function to detect the actual position keys stored in ray.traj,
# regardless of what coordinate names the user chose (x, y, theta, phi, r, …).
_KNOWN_NON_POS = frozenset({
    't', 'energy',
    'xi', 'eta',               # standard momentum symbols used internally
    'px', 'py', 'pz',          # alternative momentum names
    'p', 'p1', 'p2', 'p3',
    'vx', 'vy', 'vz',          # velocity components stored by some integrators
    'v',
})


def _pos_keys(traj: dict, dim: int):
    """
    Return the position key(s) from a trajectory dictionary.

    Filters out all known non-position keys (momenta, velocities, time,
    energy) by exact match against ``_KNOWN_NON_POS``.  The remaining keys
    are returned in insertion order (Python 3.7+ dict ordering), which
    matches the order in which ``hamiltonian_flow`` stores coordinates.

    Unlike the previous ``'p' not in k`` substring test, this approach is
    safe for coordinate names that contain the letter 'p' — most importantly
    ``'phi'`` for spherical / polar coordinates.

    Parameters
    ----------
    traj : dict
        Trajectory dictionary from ``RayData.traj``.
    dim : int
        Expected spatial dimension (1 or 2).

    Returns
    -------
    x_key : str          (always)
    y_key : str or None  (None when dim == 1)
    """
    pos = [k for k in traj if k not in _KNOWN_NON_POS]
    if dim == 1:
        return (pos[0] if pos else 'x'), None
    if len(pos) >= 2:
        return pos[0], pos[1]
    return 'x', 'y'   # last-resort fallback (should never be reached)

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
    eq = result.equation

    # ── choose display quantity based on equation type ────────────────────────
    if eq == EquationType.PARABOLIC:
        # Real-valued solution: display the real part as primary
        primary     = psi.real
        pri_label   = r"$u(x)$  [Re]"
        density     = np.log1p(primary**2) if log_scale else primary**2
        den_label   = (r"$\log(1+u^2)$" if log_scale else r"$u^2$")
        phase_label = r"$\log|u|$"
        phase_data  = np.log1p(np.abs(psi))
    else:
        primary     = psi
        pri_label   = (r"$\log(1+|\psi|^2)$" if log_scale else r"$|\psi|^2$")
        density     = np.log1p(np.abs(psi)**2) if log_scale else np.abs(psi)**2
        den_label   = pri_label
        phase_label = r"Phase $\arg(\psi)$"
        phase_data  = np.angle(psi)

    _EQ_TITLES = {
        EquationType.SCHRODINGER: r"Van Vleck wavefunction  $-i\,\partial_t u = \psi\mathrm{Op}\,u$",
        EquationType.PARABOLIC:   r"Semiclassical heat kernel  $\partial_t u = \psi\mathrm{Op}\,u$",
        EquationType.WAVE:        r"Semiclassical wave solution  $\partial_{tt} u = \psi\mathrm{Op}\,u$",
    }

    fig = plt.figure(figsize=(16, 8))
    gs  = GridSpec(2, 4, fig, hspace=0.45, wspace=0.38)

    ax0 = fig.add_subplot(gs[0, 0:2])
    ax0.fill_between(x, density, alpha=0.8, color=plt.cm.inferno(0.65))
    ax0.plot(x, density, lw=0.9, color="white", alpha=0.55)
    ax0.set(title=den_label, xlabel="$x$", ylabel=den_label)

    ax1 = fig.add_subplot(gs[0, 2:4])
    if eq == EquationType.PARABOLIC:
        ax1.fill_between(x, phase_data, alpha=0.7, color=plt.cm.plasma(0.5))
        ax1.plot(x, phase_data, lw=1.1, color=plt.cm.plasma(0.8))
    else:
        ax1.plot(x, phase_data, color=plt.cm.hsv(0.28), lw=1.1)
        ax1.axhline(0, color="white", lw=0.5, ls="--")
        ax1.set_ylim(-np.pi - 0.3, np.pi + 0.3)
    ax1.set(title=phase_label, xlabel="$x$", ylabel=phase_label)

    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.plot(x, psi.real, lw=1.0, color="#4fc3f7", label=r"Re $\psi$")
    if eq != EquationType.PARABOLIC:
        ax2.plot(x, psi.imag, lw=1.0, color="#ef9a9a",
                 label=r"Im $\psi$", alpha=0.8)
    ax2.axhline(0, color="white", lw=0.4, ls="--")
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.set(title=r"Re / Im", xlabel="$x$")

    ax3 = fig.add_subplot(gs[1, 2:4])
    x_key, _ = _pos_keys(result.rays[0].traj, dim=1)
    for ray in result.rays:
        c = plt.cm.plasma(0.3 + 0.5 * float(np.mean(np.abs(ray.det_J)))
                          / (float(np.mean(np.abs(ray.det_J))) + 1.0))
        ax3.plot(ray.traj['t'], ray.traj[x_key], lw=0.5, alpha=0.3, color=c)
    branch_note = "  (+/− branches)" if eq == EquationType.WAVE else ""
    ax3.set(title=f"Ray fan  $x(t)${branch_note}", xlabel="$t$", ylabel="$x$")

    fig.suptitle(
        rf"{_EQ_TITLES[eq]}  ($\hbar={result.hbar}$, "
        rf"$t_{{max}}={result.t_max}$, {len(result.rays)} rays)",
        color="white", fontsize=10, fontweight="bold", y=1.01)
    _style(fig, fig.axes)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def _plot_2d(result: WKBResult, log_scale: bool, save_path) -> plt.Figure:
    X, Y, psi = result.X, result.Y, result.psi
    eq = result.equation
    den    = np.log1p(np.abs(psi)**2) if log_scale else np.abs(psi)**2
    dlabel = r"$\log(1+|\psi|^2)$" if log_scale else r"$|\psi|^2$"

    _EQ_TITLES_2D = {
        EquationType.SCHRODINGER: r"Van Vleck 2D  $-i\,\partial_t u = \psi\mathrm{Op}\,u$",
        EquationType.PARABOLIC:   r"Heat kernel 2D  $\partial_t u = \psi\mathrm{Op}\,u$",
        EquationType.WAVE:        r"Wave 2D  $\partial_{tt} u = \psi\mathrm{Op}\,u$",
    }

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
    x_key, y_key = _pos_keys(result.rays[0].traj, dim=2)
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
    mu = result.mu_pts
    
    # Check if mu is integer-valued (it should be!)
    if np.all(mu == np.round(mu)):
        # Use discrete coloring for integer Maslov index
        unique_mu = np.unique(mu)
        sc4 = ax4.scatter(result.x_pts, result.y_pts,
                          c=mu, cmap= "RdYlGn", s=0.8, alpha=0.45,
                          vmin=unique_mu.min(), vmax=unique_mu.max(),
                          rasterized=True)
        
        # Force integer ticks on colorbar
        cbar = fig.colorbar(sc4, ax=ax4, label=r"Maslov $\mu$", pad=0.02)
        cbar.set_ticks(unique_mu.astype(int))
        cbar.set_ticklabels([str(int(m)) for m in unique_mu])
    else:
        # Fallback for non-integer (shouldn't happen)
        sc4 = ax4.scatter(result.x_pts, result.y_pts,
                          c=mu.astype(float), cmap= "RdBu_r", s=0.8, alpha=0.45,
                          rasterized=True)
        fig.colorbar(sc4, ax=ax4, label=r"Maslov $\mu$", pad=0.02)

    branch_note = "  (+/− branches)" if eq == EquationType.WAVE else ""
    fig.suptitle(
        rf"{_EQ_TITLES_2D[eq]}  ($\hbar={result.hbar}$, "
        rf"$t_{{max}}={result.t_max}$, {len(result.rays)} rays{branch_note})",
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
    x_key, y_key = _pos_keys(result.rays[0].traj, dim=result.dim)

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


def animate_wavefunction(
    result: WKBResult,
    frame_times=None,
    n_frames: int = 100,
    interval: int = 50,
    save_path: str | None = None,
    log_scale: bool = True,
) -> "matplotlib.animation.FuncAnimation":
    """
    Animate the time evolution of the semiclassical wavefunction.

    Overview
    --------
    This function creates an animation of ψ(x, t) (or u(x, t) for parabolic
    equation) by re‑assembling the Van Vleck–Pauli–Morette sum at each frame
    from the already integrated ray data.  Unlike a static snapshot that is
    merely rotated in phase, this recomputes the full coherent superposition
    at each time slice, correctly capturing the spreading, focusing, caustic
    births, and true |ψ|² dynamics.

    The animation calls the appropriate summation routine
    (:func:`van_vleck_sum`, :func:`parabolic_sum`, or :func:`wave_sum`) for
    each frame.  No ray integration is repeated – only the gridding step is
    performed, so the overhead is modest.

    Physical Background
    -------------------
    For the Schrödinger equation (``equation = EquationType.SCHRODINGER``) the
    wavefunction is
        ψ(x,t) = Σ_k A_k(t) · exp(i S_k(t)/ℏ − i μ_k(t) π/2)
    with A_k = 1/√|det J_k|.  For the heat‑type equation
    (``EquationType.PARABOLIC``) the sum is real‑exponential
        u(x,t) = Σ_k A_k(t) · exp(S_k(t)/ℏ)
    and for the wave equation (``EquationType.WAVE``) it is the coherent sum
    over two branches H₊ = +√H and H₋ = −√H.

    Parameters
    ----------
    result : WKBResult
        Output of :func:`compute_wavefunction`.  Must contain the full per‑ray
        time series (``traj``, ``det_J``, ``S_cum``).  The animation uses the
        raw data stored in ``result.rays`` to evaluate the wavefunction at
        arbitrary times within [0, result.t_max].
    frame_times : array‑like or None
        Physical times at which to render frames.  Each value is matched to
        the nearest stored time step.  If ``None``, ``n_frames`` equally
        spaced times between 0 and ``result.t_max`` are used.
    n_frames : int, default 100
        Number of frames (ignored when ``frame_times`` is supplied).
    interval : int, default 50
        Delay between frames in milliseconds.
    save_path : str or None, default None
        If given, save the animation to this path.  Format is inferred from
        the extension: ``.gif`` uses Pillow, ``.mp4`` uses ffmpeg.
    log_scale : bool, default True
        If ``True``, display the density as ``log(1 + |ψ|²)`` (or
        ``log(1 + u²)`` for parabolic) to reveal low‑amplitude features.
        Otherwise show the linear density.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object.  Call ``plt.show()`` to display it, or use
        ``anim.save(...)`` directly (though the function already supports
        saving via the ``save_path`` argument).

    Notes
    -----
    * Axis limits for Re(ψ), Im(ψ), and the density are pre‑scanned over
      **all** frames to ensure a stable scale throughout the animation.
    * For 1D the figure layout mirrors :func:`plot_wavefunction`: density,
      phase, real part, imaginary part (the latter hidden for parabolic).
    * For 2D the layout consists of real part, imaginary part, density, and
      phase (or log|u| for parabolic).
    * For the wave equation the ray fan is assumed to be split into two halves
      representing the H₊ and H₋ branches (the order used by
      :func:`compute_wavefunction`).  Both are combined in the sum at each
      frame.
    * The function uses the appropriate summation method based on the
      ``result.equation`` attribute.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import sympy as sp
        from riemannian import Metric
        from propagator import compute_wavefunction, animate_wavefunction

        # 1D free particle on a flat metric
        x = sp.Symbol('x', real=True)
        metric = Metric(1, (x,))
        source = (0.0,)
        v_fan  = np.linspace(-4.0, 4.0, 100)

        result = compute_wavefunction(
            metric=metric, source=source, v_fan=v_fan,
            t_max=2.0, hbar=0.05, n_steps=500, N_grid=400,
        )

        # Create animation and display
        anim = animate_wavefunction(result, n_frames=80, interval=40)
        plt.show()

        # Save as GIF
        animate_wavefunction(result, save_path="wavepacket.gif")
    """
    from matplotlib.animation import FuncAnimation

    # ── time index selection ──────────────────────────────────────────────────
    t_arr   = result.rays[0].traj['t']
    n_steps = len(t_arr)

    if frame_times is None:
        indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)
    else:
        indices = np.array([np.argmin(np.abs(t_arr - t)) for t in frame_times],
                           dtype=int)

    # ── position key detection ────────────────────────────────────────────────
    x_key, y_key = _pos_keys(result.rays[0].traj, dim=result.dim)

    # ── grid parameters ───────────────────────────────────────────────────────
    if result.dim == 1:
        x      = result.X
        xlim   = (float(x[0]), float(x[-1]))
        N_grid = len(x)
        ylim_grid = None
    else:
        X, Y   = result.X, result.Y
        xlim   = (float(X[0, 0]),  float(X[0, -1]))
        ylim_grid = (float(Y[0, 0]),  float(Y[-1, 0]))
        N_grid = X.shape[0]

    hbar = result.hbar
    eq   = result.equation

    _ANIM_TITLES = {
        EquationType.SCHRODINGER: "Van Vleck wavefunction",
        EquationType.PARABOLIC:   "Semiclassical heat kernel",
        EquationType.WAVE:        "Semiclassical wave",
    }
    anim_title = _ANIM_TITLES.get(eq, "Semiclassical wavefunction")

    # ── pre-scan all frames to get global amplitude limits ────────────────────
    def _psi_at(t_idx: int) -> np.ndarray:
        """Assemble u on the grid at a single time index, equation-aware."""
        if result.dim == 1:
            pts = np.array([[ray.traj[x_key][t_idx]] for ray in result.rays])
        else:
            pts = np.array([[ray.traj[x_key][t_idx],
                             ray.traj[y_key][t_idx]] for ray in result.rays])
            pts += 1e-10 * np.random.default_rng(0).standard_normal(pts.shape)

        S_vals    = np.array([ray.S_cum[t_idx]  for ray in result.rays])
        detJ_vals = np.array([ray.det_J[t_idx]  for ray in result.rays])
        mu_vals   = np.array([ray.mu             for ray in result.rays])

        if eq == EquationType.PARABOLIC:
            psi, _, _ = parabolic_sum(
                pts, S_vals, detJ_vals,
                xlim=xlim, ylim=ylim_grid, N=N_grid, hbar=hbar,
            )
        elif eq == EquationType.WAVE:
            # For animation the rays are already interleaved (+ and − branches);
            # split them by the sign of their final action derivative as a proxy
            # — or simply use the full pool for both branches (conservative).
            half = len(result.rays) // 2
            rays_p = result.rays[:half]
            rays_m = result.rays[half:]
            def _pts_S_dJ_mu(rlist):
                if result.dim == 1:
                    _pts = np.array([[r.traj[x_key][t_idx]] for r in rlist])
                else:
                    _pts = np.array([[r.traj[x_key][t_idx],
                                      r.traj[y_key][t_idx]] for r in rlist])
                    _pts += 1e-10 * np.random.default_rng(1).standard_normal(_pts.shape)
                return (_pts,
                        np.array([r.S_cum[t_idx]  for r in rlist]),
                        np.array([r.det_J[t_idx]  for r in rlist]),
                        np.array([r.mu             for r in rlist]))
            pp, sp, djp, mup = _pts_S_dJ_mu(rays_p)
            pm, sm, djm, mum = _pts_S_dJ_mu(rays_m)
            psi, _, _ = wave_sum(
                pp, sp, djp, mup,
                pm, sm, djm, mum,
                xlim=xlim, ylim=ylim_grid, N=N_grid, hbar=hbar,
            )
        else:
            psi, _, _ = van_vleck_sum(
                pts, S_vals, detJ_vals, mu_vals,
                xlim=xlim, ylim=ylim_grid, N=N_grid, hbar=hbar,
            )
        return psi

    # Sample a subset of frames (every 5th) to estimate limits without full scan
    sample_idx = indices[::max(1, len(indices) // 20)]
    psi_samples = [_psi_at(int(i)) for i in sample_idx]

    # ── pre-scan limits ───────────────────────────────────────────────────────
    # psi_max = max of |Re| and |Im| separately (not |ψ|, which over-inflates
    # the Re/Im axes when the wavefunction is nearly real or nearly imaginary).
    re_max_global = max(float(np.max(np.abs(p.real))) for p in psi_samples) or 1.0
    im_max_global = max(float(np.max(np.abs(p.imag))) for p in psi_samples) or 1.0
    psi_max  = max(re_max_global, im_max_global)     # kept for 2D colourbar init
    den_vals = [(np.log1p(np.abs(p)**2) if log_scale else np.abs(p)**2)
                for p in psi_samples]
    den_max  = max(float(np.max(d)) for d in den_vals) or 1.0

    # Equation-aware labels (mirrors _plot_1d / _plot_2d)
    if eq == EquationType.PARABOLIC:
        dlabel      = r"$\log(1+u^2)$"   if log_scale else r"$u^2$"
        phase_label = r"$\log|u|$"
    else:
        dlabel      = r"$\log(1+|\psi|^2)$" if log_scale else r"$|\psi|^2$"
        phase_label = r"Phase $\arg\psi(x,t)$"

    # ── build figure ──────────────────────────────────────────────────────────
    if result.dim == 1:
        fig, axes_2d = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes_2d.flatten()
        _style(fig, axes)
        fig.subplots_adjust(hspace=0.38, wspace=0.35)

        psi0 = _psi_at(int(indices[0]))
        den0 = np.log1p(np.abs(psi0)**2) if log_scale else np.abs(psi0)**2
        if eq == EquationType.PARABOLIC:
            phase0 = np.log1p(np.abs(psi0))
        else:
            phase0 = np.angle(psi0)

        # Panel layout — identical to _plot_1d:
        #   axes[0] = density  |ψ|²  (or u²)
        #   axes[1] = phase    arg ψ  (or log|u|)
        #   axes[2] = Re ψ
        #   axes[3] = Im ψ  (hidden for PARABOLIC)
        (line_den,)   = axes[0].plot(x, den0,        lw=0.9, color="white",  alpha=0.55)
        (line_phase,) = axes[1].plot(x, phase0,       lw=1.1, color=plt.cm.hsv(0.28))
        (line_re,)    = axes[2].plot(x, psi0.real,    lw=1.1, color="#4fc3f7")
        (line_im,)    = axes[3].plot(x, psi0.imag,    lw=1.1, color="#ef9a9a")

        axes[0].fill_between(x, den0,     alpha=0.75, color=plt.cm.inferno(0.65))
        axes[2].fill_between(x, psi0.real, alpha=0.25, color="#4fc3f7")
        axes[3].fill_between(x, psi0.imag, alpha=0.25, color="#ef9a9a")

        axes[0].set_xlim(xlim);  axes[0].set_ylim(0, 1.25 * den_max)
        axes[1].set_xlim(xlim)
        if eq != EquationType.PARABOLIC:
            axes[1].set_ylim(-np.pi - 0.3, np.pi + 0.3)
            axes[1].axhline(0, color="white", lw=0.4, ls="--", alpha=0.4)
            axes[1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            axes[1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "$0$",
                                      r"$\pi/2$", r"$\pi$"])
        for ax in axes[2:]:
            ax.set_xlim(xlim)
            ax.set_ylim(-1.35 * psi_max, 1.35 * psi_max)
            ax.axhline(0, color="white", lw=0.4, ls="--", alpha=0.4)

        axes[0].set(title=dlabel,      xlabel="$x$", ylabel=dlabel)
        axes[1].set(title=phase_label, xlabel="$x$", ylabel=phase_label)
        axes[2].set(title=r"Re $\psi(x,t)$", xlabel="$x$", ylabel=r"Re $\psi$")
        axes[3].set(title=r"Im $\psi(x,t)$", xlabel="$x$", ylabel=r"Im $\psi$")

        if eq == EquationType.PARABOLIC:
            axes[3].set_visible(False)   # Im ≡ 0 for parabolic

        time_text = fig.suptitle(
            rf"{anim_title}  ($\hbar={hbar}$,  $t={t_arr[indices[0]]:.3f}$)",
            color="white", fontsize=10, fontweight="bold")
        _style(fig, axes)

        def _update_1d(frame):
            t_idx = int(indices[frame])
            psi   = _psi_at(t_idx)
            den   = np.log1p(np.abs(psi)**2) if log_scale else np.abs(psi)**2
            if eq == EquationType.PARABOLIC:
                phase = np.log1p(np.abs(psi))
            else:
                phase = np.angle(psi)

            # ── axes[0]: density ─────────────────────────────────────────────
            line_den.set_ydata(den)
            for coll in axes[0].collections[:]: coll.remove()       # ← Bug 2 fix
            axes[0].fill_between(x, den, alpha=0.75, color=plt.cm.inferno(0.65))
            _den_max = float(np.max(den)) or 1.0
            axes[0].set_ylim(0, 1.25 * _den_max)                    # ← Bug 1 fix

            # ── axes[1]: phase / log|u| ──────────────────────────────────────
            line_phase.set_ydata(phase)
            if eq != EquationType.PARABOLIC:
                axes[1].set_ylim(-np.pi - 0.3, np.pi + 0.3)         # ← Bug 3 fix

            # ── axes[2]: Re ψ ─────────────────────────────────────────────────
            line_re.set_ydata(psi.real)
            for coll in axes[2].collections[:]: coll.remove()
            axes[2].fill_between(x, psi.real, alpha=0.25, color="#4fc3f7")
            _re_max = float(np.max(np.abs(psi.real))) or 1.0
            axes[2].set_ylim(-1.35 * _re_max, 1.35 * _re_max)

            # ── axes[3]: Im ψ  (skipped for PARABOLIC) ───────────────────────
            if eq != EquationType.PARABOLIC:
                line_im.set_ydata(psi.imag)
                for coll in axes[3].collections[:]: coll.remove()
                axes[3].fill_between(x, psi.imag, alpha=0.25, color="#ef9a9a")
                _im_max = float(np.max(np.abs(psi.imag))) or 1.0
                axes[3].set_ylim(-1.35 * _im_max, 1.35 * _im_max)

            time_text.set_text(
                rf"{anim_title}  ($\hbar={hbar}$,  $t={t_arr[t_idx]:.3f}$)")
            return line_den, line_phase, line_re, line_im

        anim = FuncAnimation(fig, _update_1d, frames=len(indices),
                             interval=interval, blit=False)

    else:  # 2D
        fig, axes_2d = plt.subplots(2, 2, figsize=(13, 11))
        axes = axes_2d.flatten()
        _style(fig, axes)
        fig.subplots_adjust(hspace=0.32, wspace=0.32)

        psi0 = _psi_at(int(indices[0]))
        den0 = np.log1p(np.abs(psi0)**2) if log_scale else np.abs(psi0)**2
        psi_max_2d = max(float(np.max(np.abs(psi0.real))),
                         float(np.max(np.abs(psi0.imag)))) or 1.0

        im_re = axes[0].pcolormesh(X, Y, psi0.real, cmap="RdBu_r",
                                    shading="auto", vmin=-psi_max_2d, vmax=psi_max_2d)
        fig.colorbar(im_re, ax=axes[0], label=r"Re $\psi$", pad=0.02)
        axes[0].set_aspect("equal")
        axes[0].set(title=r"Re $\psi(x,y,t)$", xlabel="$x$", ylabel="$y$")

        im_im = axes[1].pcolormesh(X, Y, psi0.imag, cmap="RdBu_r",
                                    shading="auto", vmin=-psi_max_2d, vmax=psi_max_2d)
        fig.colorbar(im_im, ax=axes[1], label=r"Im $\psi$", pad=0.02)
        axes[1].set_aspect("equal")
        axes[1].set(title=r"Im $\psi(x,y,t)$", xlabel="$x$", ylabel="$y$")

        im_den = axes[2].pcolormesh(X, Y, den0, cmap="inferno",
                                     shading="auto", vmin=0, vmax=den_max)
        fig.colorbar(im_den, ax=axes[2], label=dlabel, pad=0.02)
        axes[2].set_aspect("equal")
        axes[2].set(title=dlabel, xlabel="$x$", ylabel="$y$")

        if eq == EquationType.PARABOLIC:
            phase0_2d = np.log1p(np.abs(psi0))
            im_phase = axes[3].pcolormesh(X, Y, phase0_2d, cmap="plasma",
                                           shading="auto")
            fig.colorbar(im_phase, ax=axes[3], label=r"$\log|u|$", pad=0.02)
            axes[3].set(title=r"$\log|u(x,y,t)|$", xlabel="$x$", ylabel="$y$")
        else:
            im_phase = axes[3].pcolormesh(X, Y, np.angle(psi0), cmap="hsv",
                                           shading="auto", vmin=-np.pi, vmax=np.pi)
            cbar_phase = fig.colorbar(im_phase, ax=axes[3],
                                       label=r"$\arg\psi$  [rad]", pad=0.02)
            cbar_phase.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cbar_phase.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "$0$",
                                        r"$\pi/2$", r"$\pi$"])
            axes[3].set(title=r"Phase $\arg\psi(x,y,t)$", xlabel="$x$", ylabel="$y$")
        axes[3].set_aspect("equal")

        if eq == EquationType.PARABOLIC:
            axes[1].set_visible(False)   # Im ≡ 0

        time_text = fig.suptitle(
            rf"{anim_title} 2D  ($\hbar={hbar}$,  $t={t_arr[indices[0]]:.3f}$)",
            color="white", fontsize=10, fontweight="bold")
        _style(fig, axes)

        def _update_2d(frame):
            t_idx = int(indices[frame])
            psi   = _psi_at(t_idx)
            den   = np.log1p(np.abs(psi)**2) if log_scale else np.abs(psi)**2

            _re_max = float(np.max(np.abs(psi.real))) or 1.0
            im_re.set_array(psi.real.ravel())
            im_re.set_clim(-_re_max, _re_max)

            if eq != EquationType.PARABOLIC:
                _im_max = float(np.max(np.abs(psi.imag))) or 1.0
                im_im.set_array(psi.imag.ravel())
                im_im.set_clim(-_im_max, _im_max)

            _den_max = float(np.max(den)) or 1.0          # ← Bug 6 fix
            im_den.set_array(den.ravel())
            im_den.set_clim(0, _den_max)

            if eq == EquationType.PARABOLIC:
                im_phase.set_array(np.log1p(np.abs(psi)).ravel())
            else:
                im_phase.set_array(np.angle(psi).ravel())

            time_text.set_text(
                rf"{anim_title} 2D  ($\hbar={hbar}$,  $t={t_arr[t_idx]:.3f}$)")
            return im_re, im_im, im_den, im_phase

        anim = FuncAnimation(fig, _update_2d, frames=len(indices),
                             interval=interval, blit=False)

    # ── optional save ─────────────────────────────────────────────────────────
    if save_path:
        ext = save_path.rsplit('.', 1)[-1].lower()
        writer = 'pillow' if ext == 'gif' else 'ffmpeg'
        anim.save(save_path, writer=writer, dpi=120,
                  savefig_kwargs={'facecolor': fig.get_facecolor()})

    return anim