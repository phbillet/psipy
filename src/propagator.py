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

# (at the top of the file, add the import)
import concurrent.futures

# ── Global variables for parallel worker processes ───────────────────────────
_worker_metric = None          # will hold a Metric instance per process
_worker_metric_data = None     # symbolic data passed to initializer

def _worker_init(metric_data):
    """
    Initializer for each worker process.
    Reconstructs a Metric object from the symbolic data and stores it in
    the global _worker_metric.
    """
    global _worker_metric, _worker_metric_data
    _worker_metric_data = metric_data
    dim, coords, expr_or_matrix = metric_data
    if dim == 1:
        _worker_metric = Metric(expr_or_matrix, coords)
    else:
        _worker_metric = Metric(expr_or_matrix, coords)


def _process_single_ray(v0, source, t_max, hbar, n_steps, integrator):
    """
    Worker function that processes a single initial velocity v0.
    Uses the global _worker_metric (set by _worker_init).
    Returns a RayData object on success, None on failure.
    """
    global _worker_metric
    metric = _worker_metric
    dim = metric.dim
    tspan = (0.0, t_max)
    H_sym, vars_phase = _build_hamiltonian_sym(metric)

    try:
        # Convert velocity to momentum
        if dim == 1:
            g0 = float(metric.g_func(source[0]))
            mom = float(g0 * v0)
            z0 = [source[0], mom]
        else:
            g0 = metric.eval(source[0], source[1])['g']
            mom = g0 @ np.array(v0, dtype=float)
            z0 = [source[0], float(mom[0]), source[1], float(mom[1])]

        # Integrate ray
        traj = hamiltonian_flow(
            H_sym, z0, tspan,
            vars_phase=vars_phase,
            integrator=integrator,
            n_steps=n_steps,
        )

        # Reconstruct geometric trajectory for Jacobi solver
        if dim == 1:
            x_sym = vars_phase[0]
            xi_sym = vars_phase[1]
            geo_traj = {
                't': traj['t'],
                'x': traj[str(x_sym)],
                'v': metric.g_inv_func(traj[str(x_sym)]) * traj[str(xi_sym)],
            }
        else:
            x_sym, xi_sym, y_sym, eta_sym = vars_phase
            x_arr = traj[str(x_sym)]; y_arr = traj[str(y_sym)]
            xi_arr = traj[str(xi_sym)]; eta_arr = traj[str(eta_sym)]

            if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
                return None

            g00 = metric.g_inv_func[(0, 0)](x_arr, y_arr)
            g01 = metric.g_inv_func[(0, 1)](x_arr, y_arr)
            g10 = metric.g_inv_func[(1, 0)](x_arr, y_arr)
            g11 = metric.g_inv_func[(1, 1)](x_arr, y_arr)

            if not all(np.all(np.isfinite(c)) for c in (g00, g01, g10, g11)):
                return None

            geo_traj = {
                't': traj['t'],
                'x': x_arr, 'y': y_arr,
                'vx': g00 * xi_arr + g01 * eta_arr,
                'vy': g10 * xi_arr + g11 * eta_arr,
            }

        # Compute Jacobi determinant
        det_J = _det_J_from_jacobi(metric, geo_traj, tspan, n_steps)

        # Compute cumulative action
        if dim == 1:
            ck = (str(vars_phase[0]),)
        else:
            ck = (str(vars_phase[0]), str(vars_phase[2]))
        S_cum = _cumulative_action(traj, dim, metric=metric, coord_keys=ck)

        # Maslov index
        mu = _maslov_index(det_J)

        return RayData(traj=traj, det_J=det_J, S_cum=S_cum, mu=mu)

    except Exception:
        # Silently skip failed rays
        return None


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
    signs = np.sign(det_J)
    signs = signs[signs != 0]              # ignore exact zeros
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

def compute_wavefunction(
    metric       : Metric,
    source       : Tuple,
    v_fan        : np.ndarray,
    t_max        : float,
    hbar         : float = 1.0,
    n_steps      : int   = 400,
    N_grid       : int   = 300,
    xlim         : Optional[Tuple] = None,
    ylim         : Optional[Tuple] = None,
    integrator   : str   = 'verlet',
) -> WKBResult:
    """
    Compute the semiclassical (Van Vleck–Pauli–Morette) wavefunction.

    This is the main public entry point.  It orchestrates the full pipeline:

    1. Build the kinetic Hamiltonian H = ½ gⁱʲ pᵢ pⱼ from ``metric``.
    2. For each initial velocity ``v0`` in ``v_fan``:

       a. Convert velocity to canonical momentum: p₀ = g(x₀) · v₀.
       b. Integrate Hamilton's equations with :func:`symplectic.hamiltonian_flow`
          to obtain the ray trajectory (positions + momenta).
       c. Reconstruct the velocity trajectory from the momenta (v = g⁻¹ p)
          and pass it to :func:`_det_J_from_jacobi` to compute the Jacobi
          determinant along the ray.
       d. Compute the cumulative action with :func:`_cumulative_action`.
       e. Compute the Maslov index with :func:`_maslov_index`.
       f. Store the result as a :class:`RayData` object.

    3. Concatenate all scattered ray data (positions, S, det J, μ).
    4. Call :func:`van_vleck_sum` to interpolate the VVP amplitude onto a
       regular output grid and apply Airy corrections near caustics.
    5. Return a :class:`WKBResult` containing the gridded ψ and all raw data.

    Velocity vs momentum convention
    --------------------------------
    The parameter ``v_fan`` specifies the fan in terms of *initial velocities*
    (contravariant vectors vⁱ = dxⁱ/dt), not canonical momenta.  Internally,
    the conversion p₀ = g(x₀) · v₀ is applied before the Hamiltonian
    integrator.  On a flat metric g = I this distinction is irrelevant, but on
    a curved metric (e.g. g = x² for a position-dependent mass) it matters:
    a uniform velocity fan produces uniformly-spaced geodesics, while a
    momentum fan would oversample fast regions.

    Error handling
    --------------
    Individual rays that raise any exception during integration (e.g. due to
    the ray escaping the domain, a singularity in the metric, or a numerical
    blow-up) are silently skipped.  If *all* rays fail, a ``RuntimeError`` is
    raised.  Callers may inspect ``len(result.rays)`` to check how many rays
    succeeded.

    Grid bounds
    -----------
    If ``xlim`` (and ``ylim`` for 2D) are not provided, they are set to the
    range of the ray endpoints with a 10% margin added on each side.  This
    auto-detection works well for most problems but may produce a grid that
    crops the wavefunction if rays are highly inhomogeneous; in that case
    supply explicit limits.

    Parameters
    ----------
    metric : Metric
        Riemannian metric encoding the geometry.  Must have ``dim`` attribute
        equal to 1 or 2.  Can be constructed directly (``Metric(g_expr, coords)``)
        or from a Hamiltonian (``Metric.from_hamiltonian(H, coords, momenta)``).
    source : tuple of float
        Initial position of the point source.

        * 1D: ``(x₀,)``
        * 2D: ``(x₀, y₀)``

    v_fan : np.ndarray
        Fan of initial velocities.

        * 1D: shape ``(n_rays,)`` — one velocity per ray.
        * 2D: shape ``(n_rays, 2)`` — one velocity vector ``[vx, vy]`` per ray.

        All velocities are contravariant (ẋ, ẏ), not covariant momenta.
        Conversion p = g · v is done internally.
    t_max : float
        Total integration time.  Rays are propagated from t = 0 to t = t_max.
    hbar : float, default 1.0
        Reduced Planck constant.  Appears in the phase exp(i S/ℏ) and in the
        fringe scale of the Airy correction ∝ ℏ^{1/3}.
    n_steps : int, default 400
        Number of time steps per ray.  Higher values improve accuracy of the
        action integral and Jacobi determinant.  The time grid is
        ``np.linspace(0, t_max, n_steps)``.
    N_grid : int, default 300
        Output grid resolution.  The wavefunction is evaluated on an N_grid ×
        N_grid mesh (2D) or N_grid points (1D).
    xlim : tuple (x_min, x_max) or None
        Explicit x-range of the output grid.  If ``None``, auto-detected from
        ray data with a 10% margin.
    ylim : tuple (y_min, y_max) or None
        Explicit y-range (2D only).  If ``None``, auto-detected.
    integrator : str, default ``'verlet'``
        Symplectic integrator passed to :func:`symplectic.hamiltonian_flow`.

        * ``'verlet'`` — Störmer–Verlet (leapfrog), 2nd-order symplectic.
          Fast; recommended for most problems.
        * ``'rk45'``   — Runge–Kutta 4/5, non-symplectic but higher formal
          accuracy.  Suitable when ℏ is very small and phase accuracy matters
          more than energy conservation.

    Returns
    -------
    WKBResult
        Dataclass containing:

        * ``psi`` — semiclassical wavefunction on the output grid (complex).
        * ``X``, ``Y`` — grid coordinate arrays.
        * ``rays`` — list of :class:`RayData` for each successful ray.
        * ``x_pts``, ``y_pts``, ``S_pts``, ``det_J_pts``, ``mu_pts`` — raw
          scattered data from all rays concatenated.
        * ``hbar``, ``t_max``, ``dim`` — input metadata.

    Raises
    ------
    RuntimeError
        If every ray in ``v_fan`` fails to integrate.

    Examples
    --------
    1D flat metric::

        import sympy as sp
        import numpy as np
        from riemannian import Metric
        from propagator import compute_wavefunction

        x = sp.Symbol('x', real=True)
        result = compute_wavefunction(
            metric=Metric(1, (x,)),
            source=(0.0,),
            v_fan=np.linspace(-3, 3, 60),
            t_max=2.0,
            hbar=0.5,
        )

    2D curved metric from Hamiltonian::

        r, theta = sp.symbols('r theta', real=True, positive=True)
        pr, pth  = sp.symbols('p_r p_theta', real=True)
        H = (pr**2 + pth**2 / r**2) / 2   # polar-coordinate kinetic energy
        metric = Metric.from_hamiltonian(H, (r, theta), (pr, pth))

        vr  = np.linspace(-0.5, 0.5, 10)
        vth = np.linspace(-0.5, 0.5, 10)
        v_fan = np.array([[a, b] for a in vr for b in vth])

        result = compute_wavefunction(
            metric=metric, source=(1.0, 0.0),
            v_fan=v_fan, t_max=1.0, hbar=0.2,
        )
    """
    dim = metric.dim
    tspan = (0.0, t_max)
    H_sym, vars_phase = _build_hamiltonian_sym(metric)   # still needed for coordinate names, etc.

    # Prepare symbolic data for worker processes
    if dim == 1:
        metric_data = (dim, metric.coords, metric.g_expr)
    else:
        metric_data = (dim, metric.coords, metric.g_matrix)

    rays = []
    # Use ProcessPoolExecutor to parallelize rays
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=None,
        initializer=_worker_init,
        initargs=(metric_data,)
    ) as executor:
        # Submit all rays
        future_to_v0 = {
            executor.submit(
                _process_single_ray,
                v0, source, t_max, hbar, n_steps, integrator
            ): v0 for v0 in v_fan
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_v0):
            result = future.result()
            if result is not None:
                rays.append(result)

    if not rays:
        raise RuntimeError("All rays failed to integrate.")
        if first_exc is not None:
            msg += f"  First exception: {type(first_exc).__name__}: {first_exc}"
        raise RuntimeError(msg)

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


# =============================================================================
# Example usage (run when script is executed directly)
# =============================================================================
if __name__ == "__main__":
    print("Running Van Vleck wavefunction examples...\n")

    # ----------------------------------------------------------------------
    # Example 1: 1D free particle on flat metric  g = 1
    # ----------------------------------------------------------------------
    print("1D free particle (flat metric) – computing wavefunction...")
    x = sp.Symbol('x', real=True)
    metric_1d = Metric(1, (x,))                     # g = 1
    source_1d = (0.0,)
    v_fan_1d = np.linspace(-5.0, 5.0, 50)           # initial velocities
    t_max_1d = 2.0
    hbar = 1.0

    result_1d = compute_wavefunction(
        metric=metric_1d,
        source=source_1d,
        v_fan=v_fan_1d,
        t_max=t_max_1d,
        hbar=hbar,
        n_steps=500,
        N_grid=300,
        integrator='verlet'
    )
    plot_wavefunction(result_1d, log_scale=True)
    plt.show()

    # ----------------------------------------------------------------------
    # Example 2: 2D free particle on flat metric  g = [[1,0],[0,1]]
    # ----------------------------------------------------------------------
    print("\n2D free particle (flat metric) – computing wavefunction...")
    x, y = sp.symbols('x y', real=True)
    metric_2d = Metric([[1, 0], [0, 1]], (x, y))    # Euclidean metric
    source_2d = (0.0, 0.0)

    v1_vals = np.linspace(-3.0, 3.0, 20)
    v2_vals = np.linspace(-3.0, 3.0, 20)
    v_fan_2d = np.array([[a, b] for a in v1_vals for b in v2_vals])

    t_max_2d = 1.5
    result_2d = compute_wavefunction(
        metric=metric_2d,
        source=source_2d,
        v_fan=v_fan_2d,
        t_max=t_max_2d,
        hbar=hbar,
        n_steps=300,
        N_grid=150,
        integrator='verlet'
    )
    plot_wavefunction(result_2d, log_scale=True)
    plt.show()

    # ----------------------------------------------------------------------
    # Example 3: 1D with Hamiltonian  H = p^2 / (2 m(x)),  m(x) = 1/x^2
    # This yields metric g = m(x) = x^2.
    # ----------------------------------------------------------------------
    print("\n1D position-dependent mass (Hamiltonian input) – computing wavefunction...")
    x_sym, p_sym = sp.symbols('x p', real=True)
    m_expr = 1 / x_sym**2
    H_1d = p_sym**2 / (2 * m_expr)
    metric_from_H_1d = Metric.from_hamiltonian(H_1d, (x_sym,), (p_sym,))
    print(f"  Metric from Hamiltonian: g = {metric_from_H_1d.g_expr}")

    source_1d_h = (1.0,)
    v_fan_1d_h = np.linspace(-3.0, 3.0, 50)
    t_max_1d_h = 1.5
    result_1d_h = compute_wavefunction(
        metric=metric_from_H_1d,
        source=source_1d_h,
        v_fan=v_fan_1d_h,
        t_max=t_max_1d_h,
        hbar=hbar,
        n_steps=500,
        N_grid=300,
        integrator='verlet'
    )
    plot_wavefunction(result_1d_h, log_scale=True)
    plt.show()

    # ----------------------------------------------------------------------
    # Example 4: 2D polar coordinates H = (p_r^2 + p_theta^2 / r^2) / 2
    #
    # Three cautions specific to polar coordinates:
    #
    # (a) Coordinate singularity at r = 0: the metric component g^{θθ} = 1/r²
    #     diverges.  Keep the source away from r=0 (r₀=1 is fine) and use
    #     velocities small enough that no ray reaches r=0 during integration.
    #
    # (b) Coordinate singularity at θ = 0 / 2π for SymPy with positive=True:
    #     evaluating g at θ=0 can trigger domain errors in lambdified
    #     expressions.  A small offset θ₀ = 0.1 rad avoids this.
    #
    # (c) Zero-velocity rays: a fan that contains v=(0,0) produces a
    #     degenerate constant trajectory; the Jacobi solver then receives a
    #     trivially zero det J and the Maslov index computation is unreliable.
    #     Build the fan from two 1D grids that exclude zero.
    # ----------------------------------------------------------------------
    print("\n2D polar coordinates (Hamiltonian input) – computing wavefunction...")
    r, theta = sp.symbols('r theta', real=True, positive=True)
    pr, ptheta = sp.symbols('p_r p_theta', real=True)
    H_2d = (pr**2 + ptheta**2 / r**2) / 2
    metric_from_H_2d = Metric.from_hamiltonian(H_2d, (r, theta), (pr, ptheta))

    # (b) small θ offset so SymPy lambdify never evaluates at θ = 0
    source_2d_h = (1.0, 0.1)

    # (c) exclude zero: use linspace on strictly positive/negative halves
    vr_pos  = np.linspace(0.05, 0.4, 5)
    vr_vals = np.concatenate([-vr_pos[::-1], vr_pos])   # [-0.4,…,-0.05, 0.05,…,0.4]
    vt_pos  = np.linspace(0.05, 0.4, 5)
    vt_vals = np.concatenate([-vt_pos[::-1], vt_pos])
    v_fan_2d_h = np.array([[a, b] for a in vr_vals for b in vt_vals])

    t_max_2d_h = 0.8          # shorter: keeps rays well away from r = 0
    result_2d_h = compute_wavefunction(
        metric=metric_from_H_2d,
        source=source_2d_h,
        v_fan=v_fan_2d_h,
        t_max=t_max_2d_h,
        hbar=hbar,
        n_steps=600,          # more steps for accuracy on curved metric
        N_grid=150,
        integrator='rk45'     # RK45 handles the r² denominator more robustly
    )
    plot_wavefunction(result_2d_h, log_scale=True)
    plt.show()

    print("\nPlotting additional diagnostics for the 1D variable-mass example...")
    plot_ray_fan(result_1d_h)
    plt.show()
    plot_interference_detail(result_1d_h)
    plt.show()

    print("\nExamples finished.")