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
geometry.py — Geometric visualisation of pseudodifferential symbols in 1D and 2D
======================================================================================

Overview
--------
The `geometry` module provides a comprehensive framework for the geometric and semiclassical analysis of pseudodifferential operator symbols `H(x,ξ)` (1D) or `H(x,y,ξ,η)` (2D).  It combines **symbolic differentiation** (SymPy) with **numerical integration** (SciPy) to compute:

* Hamiltonian flows (geodesics) including variational equations for Jacobian matrices.
* Caustics – points where rays focus – detected via zero crossings of the Jacobian determinant.
* Periodic orbits at fixed energies, with stability analysis (Lyapunov exponents).
* Maslov indices and phase shifts associated with caustic crossings.
* Gutzwiller trace formula and semiclassical spectra (1D).
* Weyl’s law, Berry‑Tabor level spacing, and KAM tori classification (2D).

The module is designed for both **pedagogical exploration** (via richly annotated multi‑panel figures) and **research** (extracting quantitative data such as caustic networks, Poincaré sections, and action‑angle variables).

Key features:

* **Automatic dimension detection** – a single uniform interface for 1D and 2D problems.
* **Exact symbolic derivatives** of the Hamiltonian, lambdified for fast numerical evaluation.
* **Augmented ODE systems** that simultaneously integrate the Hamiltonian flow and the full 4×4 Jacobian (in 2D), enabling precise caustic detection.
* **Periodic orbit search** with energy‑preserving initial guesses, deduplication, and stability computation.
* **Caustic classification** (fold, cusp) using curvature analysis and hierarchical clustering.
* **Rich visualisation atlas** – 15‑panel figures for 1D, dynamically arranged panels for 2D, covering:
    * Hamiltonian surface, level sets, vector field.
    * Group velocity, spatial projections, Jacobian evolution.
    * Phase space portraits, Poincaré sections, momentum space.
    * Periodic orbits, action‑energy diagrams, EBK quantisation.
    * Gutzwiller trace, semiclassical spectrum, level spacing distributions.
    * Caustic curves, Maslov phase shifts, phase space volume (Monte Carlo).
* **Spectral analysis utilities** – Weyl’s law, integrability classification, Berry‑Tabor smoothed density.
* **Theoretical summaries** printed on demand (`print_theory()`, `print_theory_summary()`).

Mathematical background
-----------------------
The module studies a **Hamiltonian** `H` on phase space `T*ℝⁿ` (n = 1,2).  The associated **Hamiltonian vector field** is

    X_H = ( ∂H/∂p , –∂H/∂x )    (1D) or ( ∂H/∂ξ , ∂H/∂η , –∂H/∂x , –∂H/∂y ) (2D).

Integral curves `(x(t),p(t))` are called **bicharacteristics** or **rays**.  Their projection onto configuration space `x(t)` gives the physical path.

**Caustics** occur when nearby rays converge, i.e. when the Jacobian matrix

    J_ij = ∂x_i(t)/∂p_j(0)

becomes singular.  In 1D `J = ∂x/∂p₀`; in 2D the 2×2 block `∂(x,y)/∂(ξ₀,η₀)` is monitored.  Zero crossings of `det(J)` mark caustics.  At a caustic the semiclassical amplitude diverges; the correct uniform approximation involves special functions (Airy for fold, Pearcey for cusp) and a phase shift of `–π/2` times the **Maslov index** (number of caustics traversed).

The **Gutzwiller trace formula** expresses the quantum density of states as a sum over periodic orbits:

    ρ(E) ≈ (1/πℏ) Σ_γ (T_γ/√|det(M_γ – I)|) cos(S_γ/ℏ – πμ_γ/2) .

Here `T_γ` is the period, `S_γ` the action, `M_γ` the monodromy matrix, and `μ_γ` the Maslov index.

For **integrable systems** (KAM tori), the **Einstein–Brillouin–Keller (EBK) quantisation** gives

    I_k = (1/2π) ∮ p dq = ℏ (n_k + α_k/4),

with Maslov indices `α_k`.  Level spacings follow a Poisson distribution `P(s) = e^{-s}`.  **Chaotic systems** obey the Wigner surmise `P(s) = (πs/2) e^{-πs²/4}` (Bohigas–Giannoni–Schmit conjecture).

References
----------
.. [1] Arnold, V. I.  *Mathematical Methods of Classical Mechanics*, Springer‑Verlag, 1989.
.. [2] Gutzwiller, M. C.  *Chaos in Classical and Quantum Mechanics*, Springer‑Verlag, 1990.
.. [3] Maslov, V. P. & Fedoriuk, M. V.  *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
.. [4] Berry, M. V. & Tabor, M.  “Level clustering in the regular spectrum”, *Proc. R. Soc. Lond. A* 356, 375–394, 1977.
.. [5] Bohigas, O., Giannoni, M. J., & Schmit, C.  “Characterization of chaotic quantum spectra and universality of level fluctuation laws”, *Phys. Rev. Lett.* 52, 1–4, 1984.
.. [6] Kravtsov, Yu. A. & Orlov, Yu. I.  *Caustics, Catastrophes and Wave Fields*, Springer, 1999.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy import DiracDelta, Heaviside
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import linregress
from numpy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Callable
from abc import ABC, abstractmethod
import warnings
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')


# ============================================================================
# SHARED SYMBOLIC HELPER
# ============================================================================

def _sanitize(expr: sp.Expr) -> sp.Expr:
    """Remove DiracDelta and Heaviside terms for numeric use.

    Note: ``sp.simplify`` is intentionally omitted here — it is extremely
    expensive on large symbolic expressions and the downstream lambdify call
    handles any residual algebraic simplification at negligible cost.
    """
    expr = expr.replace(sp.DiracDelta, lambda *args: sp.Integer(0))
    expr = expr.replace(sp.Heaviside,  lambda *args: sp.Integer(1))
    return expr


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GeodesicBase:
    """Common fields for all geodesics."""
    t: np.ndarray       # Time array
    H: np.ndarray       # Energy along trajectory
    color: str = field(default='blue', compare=False)

    @property
    def energy(self) -> float:
        """Initial energy (constant along the geodesic)."""
        return float(self.H[0])


@dataclass
class Geodesic1D(GeodesicBase):
    """1D geodesic in phase space (x, ξ)."""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    xi: np.ndarray = field(default_factory=lambda: np.array([]))
    J: np.ndarray = field(default_factory=lambda: np.array([]))  # ∂x/∂ξ₀
    K: np.ndarray = field(default_factory=lambda: np.array([]))  # ∂ξ/∂ξ₀

    @property
    def caustics(self) -> np.ndarray:
        """Indices where J ≈ 0 (caustic points)."""
        return np.where(np.abs(self.J) < 0.01)[0]


@dataclass
class Geodesic2D(GeodesicBase):
    """2D geodesic in phase space (x, y, ξ, η) with full 4×4 Jacobian."""
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    xi: np.ndarray = field(default_factory=lambda: np.array([]))
    eta: np.ndarray = field(default_factory=lambda: np.array([]))
    J_full: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    det_caustic: np.ndarray = field(default_factory=lambda: np.array([]))
    caustic_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def spatial_trajectory(self) -> np.ndarray:
        """Trajectory in configuration space (x, y)."""
        return np.column_stack([self.x, self.y])

    @property
    def momentum_trajectory(self) -> np.ndarray:
        """Trajectory in momentum space (ξ, η)."""
        return np.column_stack([self.xi, self.eta])

    @property
    def caustic_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """(x, y) coordinates of caustic points along the trajectory."""
        if len(self.caustic_indices) == 0:
            return np.array([]), np.array([])
        return self.x[self.caustic_indices], self.y[self.caustic_indices]


# Backward-compatibility alias
Geodesic = Geodesic1D


@dataclass
class PeriodicOrbitBase:
    """Common fields for periodic orbits."""
    period: float
    action: float
    energy: float
    x_cycle: np.ndarray
    xi_cycle: np.ndarray
    t_cycle: np.ndarray


@dataclass
class PeriodicOrbit1D(PeriodicOrbitBase):
    """1D periodic orbit in phase space."""
    x0: float = 0.0
    xi0: float = 0.0
    stability: float = 0.0  # Lyapunov exponent


@dataclass
class PeriodicOrbit2D(PeriodicOrbitBase):
    """2D periodic orbit with Maslov index and KAM stability."""
    x0: float = 0.0
    y0: float = 0.0
    xi0: float = 0.0
    eta0: float = 0.0
    y_cycle: np.ndarray = field(default_factory=lambda: np.array([]))
    eta_cycle: np.ndarray = field(default_factory=lambda: np.array([]))
    stability_1: float = 0.0
    stability_2: float = 0.0
    maslov_index: int = 0

    @property
    def is_stable(self) -> bool:
        """True if both Lyapunov exponents are negative (KAM stable)."""
        return self.stability_1 < 0 and self.stability_2 < 0

    @property
    def bohr_sommerfeld_condition(self) -> float:
        """Bohr-Sommerfeld quantization condition with Maslov correction."""
        return self.action / (2 * np.pi) - self.maslov_index / 4


# Backward-compatibility alias
PeriodicOrbit = PeriodicOrbit1D


@dataclass
class Spectrum:
    """Semiclassical spectrum (1D only)."""
    energies: np.ndarray
    intensity: np.ndarray
    trace_t: np.ndarray
    trace: np.ndarray


@dataclass
class CausticStructure:
    """Caustic structure with classification (2D only)."""
    x: np.ndarray
    y: np.ndarray
    t: float
    energy: float
    type: str           # 'fold', 'cusp', 'swallowtail'
    maslov_index: int
    strength: float


# ============================================================================
# BASE GEOMETRY ENGINE
# ============================================================================

class SymbolGeometryBase(ABC):
    """
    Abstract base class for the geometry engines.

    Provides shared helpers:
    - _remove_duplicate_orbits: generic orbit deduplication
    - _wrap_solve_ivp: stability computation wrapper
    """

    def _remove_duplicate_orbits(self, orbits: list,
                                  tol_period: float = 0.15,
                                  tol_action: float = 0.15) -> list:
        """Remove near-duplicate periodic orbits (period + action tolerance).

        Complexity is O(n · k) where k is the number of *unique* orbits kept,
        rather than the naïve O(n²) all-pairs scan.
        """
        unique: list = []
        kept_keys: list = []          # (period, action) of unique orbits
        for orb in orbits:
            if not any(
                abs(orb.period - p) < tol_period and
                abs(orb.action  - a) < tol_action
                for p, a in kept_keys
            ):
                unique.append(orb)
                kept_keys.append((orb.period, orb.action))
        return unique

    def _run_stability_ode(self, ode_func: Callable,
                           z0: list, T: float) -> float:
        """
        Integrate a linearized ODE over [0, T] and return the Lyapunov exponent.

        Parameters
        ----------
        ode_func : callable
            RHS of the linearized system (signature: (t, z) → list)
        z0 : list
            Initial conditions (last component is the perturbation seed ε)
        T : float
            Integration time (= orbit period)

        Returns
        -------
        float  λ = log(|δ_final| / ε) / T, or nan on failure
        """
        epsilon = 1e-6
        sol = solve_ivp(ode_func, [0, T], z0, method='DOP853', rtol=1e-10)
        if sol.success and sol.y.shape[1] > 0:
            # perturbation components are the last two entries of z
            pert = np.sqrt(sol.y[-2][-1]**2 + sol.y[-1][-1]**2)
            return np.log(pert / epsilon) / T
        return np.nan


# ============================================================================
# 1D GEOMETRY ENGINE
# ============================================================================

class SymbolGeometry(SymbolGeometryBase):
    """
    1D geometric and semiclassical analysis of a symbol H(x, ξ).

    Computes:
    - Hamiltonian flow (geodesics with scalar Jacobi fields J, K)
    - Caustics (J → 0)
    - Periodic orbits at fixed energy
    - Gutzwiller trace formula
    - Semiclassical spectrum (FFT of trace)
    """

    def __init__(self, symbol: sp.Expr,
                 x_sym: sp.Symbol,
                 xi_sym: sp.Symbol):
        self.H      = symbol
        self.x_sym  = x_sym
        self.xi_sym = xi_sym
        self._compute_derivatives()
        self._lambdify_functions()

    def _compute_derivatives(self):
        self.dH_dx     = _sanitize(sp.diff(self.H,     self.x_sym))
        self.dH_dxi    = _sanitize(sp.diff(self.H,     self.xi_sym))
        self.d2H_dx2   = _sanitize(sp.diff(self.dH_dx,  self.x_sym))
        self.d2H_dxi2  = _sanitize(sp.diff(self.dH_dxi, self.xi_sym))
        self.d2H_dxdxi = _sanitize(sp.diff(self.dH_dx,  self.xi_sym))

    def _safe_lambdify(self, args: tuple, expr: sp.Expr) -> Callable:
        """Lambdify with a constant-function fallback for pure-number expressions."""
        if isinstance(expr, (int, float, sp.Integer, sp.Float, sp.Number)):
            const_val = float(expr)
            return lambda x, xi: np.full_like(np.asarray(x, dtype=float), const_val)
        try:
            return sp.lambdify(args, expr, modules=['numpy', 'scipy'])
        except Exception as e:
            warnings.warn(f"lambdify failed for {expr}: {e}")
            return lambda x, xi: np.full_like(np.asarray(x, dtype=float), np.nan)

    def _lambdify_functions(self):
        v = (self.x_sym, self.xi_sym)
        self.f_H          = self._safe_lambdify(v, self.H)
        self.f_dH_dx      = self._safe_lambdify(v, self.dH_dx)
        self.f_dH_dxi     = self._safe_lambdify(v, self.dH_dxi)
        self.f_d2H_dx2    = self._safe_lambdify(v, self.d2H_dx2)
        self.f_d2H_dxi2   = self._safe_lambdify(v, self.d2H_dxi2)
        self.f_d2H_dxdxi  = self._safe_lambdify(v, self.d2H_dxdxi)

    # ------------------------------------------------------------------
    # Geodesic
    # ------------------------------------------------------------------
    def compute_geodesic(self, x0: float, xi0: float,
                         t_max: float, n_points: int = 500) -> Geodesic1D:
        """
        Integrate the Hamiltonian flow with variational (Jacobi) equations.

        System:
          dx/dt  =  ∂H/∂ξ
          dξ/dt  = -∂H/∂x
          dJ/dt  =  ∂²H/∂ξ² J + ∂²H/∂x∂ξ K
          dK/dt  = -∂²H/∂x∂ξ J - ∂²H/∂x² K

        Initial conditions: J(0)=0, K(0)=1.
        """
        def system(t, z):
            x, xi, J, K = z
            try:
                dx  = float(self.f_dH_dxi(x, xi))
                dxi = float(-self.f_dH_dx(x, xi))
                a   = float(self.f_d2H_dxi2(x, xi))
                b   = float(self.f_d2H_dxdxi(x, xi))
                c   = float(self.f_d2H_dx2(x, xi))
                return [dx, dxi, a*J + b*K, -b*J - c*K]
            except Exception:
                return [0, 0, 0, 0]

        sol = solve_ivp(system, [0, t_max], [x0, xi0, 0.0, 1.0],
                        t_eval=np.linspace(0, t_max, n_points),
                        method='DOP853', rtol=1e-10, atol=1e-12)

        H_traj = np.array([self.f_H(sol.y[0][i], sol.y[1][i])
                           for i in range(len(sol.t))])
        return Geodesic1D(t=sol.t, H=H_traj,
                          x=sol.y[0], xi=sol.y[1],
                          J=sol.y[2], K=sol.y[3])

    # ------------------------------------------------------------------
    # Periodic orbits
    # ------------------------------------------------------------------
    def find_periodic_orbits(self, energy: float,
                              x_range: Tuple[float, float],
                              xi_range: Tuple[float, float],
                              n_attempts: int = 50,
                              tol_period: float = 1e-3) -> List[PeriodicOrbit1D]:
        orbits = []
        for x0_test in np.linspace(x_range[0], x_range[1],
                                   int(np.sqrt(n_attempts))):
            def energy_eq(xi0):
                try:
                    return self.f_H(x0_test, xi0) - energy
                except Exception:
                    return 1e10

            for xi_guess in np.linspace(xi_range[0], xi_range[1], 5):
                try:
                    res = fsolve(energy_eq, xi_guess, full_output=True)
                    if res[2] != 1:
                        continue
                    xi0 = res[0][0]
                    if abs(self.f_H(x0_test, xi0) - energy) > 1e-6:
                        continue

                    geo = self.compute_geodesic(x0_test, xi0, 20, 2000)
                    distances = np.sqrt((geo.x - x0_test)**2 +
                                        (geo.xi - xi0)**2)

                    minima_idx = [
                        i for i in range(10, len(distances) - 10)
                        if (distances[i] < distances[i-1] and
                            distances[i] < distances[i+1] and
                            distances[i] < tol_period)
                    ]
                    if minima_idx:
                        idx_p = minima_idx[0]
                        period = geo.t[idx_p]
                        if period > 0.1 and distances[idx_p] < tol_period:
                            x_cyc  = geo.x[:idx_p+1]
                            xi_cyc = geo.xi[:idx_p+1]
                            t_cyc  = geo.t[:idx_p+1]
                            dx_dt  = np.gradient(x_cyc, t_cyc)
                            action = np.trapz(xi_cyc * dx_dt, t_cyc)
                            stab   = self._compute_stability_1d(
                                x0_test, xi0, period)
                            orbits.append(PeriodicOrbit1D(
                                period=period, action=action, energy=energy,
                                x_cycle=x_cyc, xi_cycle=xi_cyc, t_cycle=t_cyc,
                                x0=x0_test, xi0=xi0, stability=stab
                            ))
                except Exception:
                    continue
        return self._remove_duplicate_orbits(orbits)

    def _compute_stability_1d(self, x0: float, xi0: float,
                               T: float) -> float:
        """Lyapunov exponent for a 1D orbit via linearized equations."""
        epsilon = 1e-6

        def linearized(t, z):
            x, xi, dx, dxi = z
            try:
                vx   = float(self.f_dH_dxi(x, xi))
                vxi  = float(-self.f_dH_dx(x, xi))
                A12  = float(self.f_d2H_dxi2(x, xi))
                A21  = float(-self.f_d2H_dxdxi(x, xi))
                return [vx, vxi, A12 * dxi, A21 * dx]
            except Exception:
                return [0, 0, 0, 0]

        z0  = [x0, xi0, epsilon, 0]
        sol = solve_ivp(linearized, [0, T], z0,
                        method='DOP853', rtol=1e-10)
        if sol.success and sol.y.shape[1] > 0:
            pert = np.sqrt(sol.y[2][-1]**2 + sol.y[3][-1]**2)
            return np.log(pert / epsilon) / T
        return np.nan

    # ------------------------------------------------------------------
    # Semiclassical spectrum
    # ------------------------------------------------------------------
    def gutzwiller_trace_formula(self,
                                  periodic_orbits: List[PeriodicOrbit1D],
                                  t_values: np.ndarray,
                                  hbar: float = 1.0) -> np.ndarray:
        """Gutzwiller trace formula: Tr[exp(-iHt/ℏ)] ≈ Σ_γ A_γ exp(iS_γ/ℏ)."""
        trace = np.zeros(len(t_values), dtype=complex)
        for orb in periodic_orbits:
            T, S, lam = orb.period, orb.action, orb.stability
            for k in range(1, 11):
                T_k = k * T
                S_k = k * S
                if not np.isnan(lam) and abs(lam) > 1e-6:
                    det_factor = abs(2 * np.sinh(k * lam * T))
                else:
                    det_factor = 1.0
                det_factor = max(det_factor, 1e-10)
                amplitude = T / np.sqrt(det_factor)
                sigma = T_k * 0.05
                gauss = np.exp(-((t_values - T_k)**2) / (2 * sigma**2))
                gauss /= (sigma * np.sqrt(2 * np.pi))
                phase = S_k / hbar - np.pi * 0 / 2
                trace += amplitude * gauss * np.exp(1j * phase) * np.exp(-0.1 * k)
        return trace

    def semiclassical_spectrum(self,
                                periodic_orbits: List[PeriodicOrbit1D],
                                hbar: float = 1.0,
                                resolution: int = 4000) -> Spectrum:
        """Extract semiclassical spectrum via FFT of the Gutzwiller trace."""
        t_max    = 200 / hbar
        t_values = np.linspace(0, t_max, resolution)
        trace    = self.gutzwiller_trace_formula(periodic_orbits, t_values, hbar)
        energies = fftfreq(len(t_values),
                           d=t_values[1]-t_values[0]) * 2 * np.pi * hbar
        return Spectrum(
            energies=energies,
            intensity=np.abs(fft(trace)),
            trace_t=t_values,
            trace=trace
        )

# ============================================================================
# 2D GEOMETRY ENGINE
# ============================================================================

class SymbolGeometry2D(SymbolGeometryBase):
    """
    2D geometric and semiclassical analysis of H(x, y, ξ, η).

    Computes:
    - Hamiltonian flow with full 4×4 Jacobian (20-dim augmented system)
    - Caustic detection via sign change of det(∂(x,y)/∂(ξ₀,η₀))
    - Caustic classification (fold / cusp) and clustering
    - Periodic orbits with Maslov index
    - Monte Carlo phase space volume
    """

    def __init__(self, symbol: sp.Expr,
                 x_sym: sp.Symbol, y_sym: sp.Symbol,
                 xi_sym: sp.Symbol, eta_sym: sp.Symbol,
                 hbar: float = 1.0):
        self.H_sym   = symbol
        self.x_sym   = x_sym
        self.y_sym   = y_sym
        self.xi_sym  = xi_sym
        self.eta_sym = eta_sym
        self.hbar    = hbar

        print(f"Initializing 2D geometry engine for H = {self.H_sym} with ℏ = {self.hbar}")
        self._compute_derivatives()
        self._lambdify_functions()

    # ------------------------------------------------------------------
    # Symbolic derivatives
    # ------------------------------------------------------------------
    def _compute_derivatives(self):
        s = self
        self.dH_dx_sym    = _sanitize(sp.diff(s.H_sym, s.x_sym))
        self.dH_dy_sym    = _sanitize(sp.diff(s.H_sym, s.y_sym))
        self.dH_dxi_sym   = _sanitize(sp.diff(s.H_sym, s.xi_sym))
        self.dH_deta_sym  = _sanitize(sp.diff(s.H_sym, s.eta_sym))

        self.d2H_dx2_sym      = _sanitize(sp.diff(self.dH_dx_sym,   s.x_sym))
        self.d2H_dy2_sym      = _sanitize(sp.diff(self.dH_dy_sym,   s.y_sym))
        self.d2H_dxi2_sym     = _sanitize(sp.diff(self.dH_dxi_sym,  s.xi_sym))
        self.d2H_deta2_sym    = _sanitize(sp.diff(self.dH_deta_sym, s.eta_sym))
        self.d2H_dxdy_sym     = _sanitize(sp.diff(self.dH_dx_sym,   s.y_sym))
        self.d2H_dxdxi_sym    = _sanitize(sp.diff(self.dH_dx_sym,   s.xi_sym))
        self.d2H_dxdeta_sym   = _sanitize(sp.diff(self.dH_dx_sym,   s.eta_sym))
        self.d2H_dydxi_sym    = _sanitize(sp.diff(self.dH_dy_sym,   s.xi_sym))
        self.d2H_dyeta_sym    = _sanitize(sp.diff(self.dH_dy_sym,   s.eta_sym))
        self.d2H_dxideta_sym  = _sanitize(sp.diff(self.dH_dxi_sym,  s.eta_sym))

        self.Hessian = sp.Matrix([
            [self.d2H_dx2_sym,    self.d2H_dxdy_sym,    self.d2H_dxdxi_sym,   self.d2H_dxdeta_sym],
            [self.d2H_dxdy_sym,   self.d2H_dy2_sym,     self.d2H_dydxi_sym,   self.d2H_dyeta_sym],
            [self.d2H_dxdxi_sym,  self.d2H_dydxi_sym,   self.d2H_dxi2_sym,    self.d2H_dxideta_sym],
            [self.d2H_dxdeta_sym, self.d2H_dyeta_sym,   self.d2H_dxideta_sym, self.d2H_deta2_sym],
        ])

    def _safe_lambdify(self, args: tuple, expr: sp.Expr) -> Callable:
        if isinstance(expr, (int, float, sp.Integer, sp.Float)):
            const_val = float(expr)
            return lambda x, y, xi, eta: np.full_like(
                np.asarray(x, dtype=float), const_val)
        try:
            return sp.lambdify(args, expr, modules=['numpy', 'scipy'])
        except Exception as e:
            print(f"Warning: lambdify failed for {expr}. Error: {e}")
            return lambda x, y, xi, eta: np.full_like(
                np.asarray(x, dtype=float), np.nan)

    def _lambdify_functions(self):
        args = (self.x_sym, self.y_sym, self.xi_sym, self.eta_sym)
        self.H_num        = self._safe_lambdify(args, self.H_sym)
        self.dH_dx_num    = self._safe_lambdify(args, self.dH_dx_sym)
        self.dH_dy_num    = self._safe_lambdify(args, self.dH_dy_sym)
        self.dH_dxi_num   = self._safe_lambdify(args, self.dH_dxi_sym)
        self.dH_deta_num  = self._safe_lambdify(args, self.dH_deta_sym)
        self.second_derivs_funcs = [
            [self._safe_lambdify(args, self.Hessian[i, j]) for j in range(4)]
            for i in range(4)
        ]

    # ------------------------------------------------------------------
    # Augmented Hamiltonian system (position + 4×4 Jacobian)
    # ------------------------------------------------------------------
    def _hamiltonian_system_augmented(self, t: float,
                                       z: np.ndarray) -> np.ndarray:
        """
        State z = [x, y, ξ, η, J₁₁, J₁₂, …, J₄₄]  (dimension 20).
        Equations:
          ż[0:4] = Hamilton
          ż[4:]  = J @ (J₀ @ Hessian)  (variational)
        """
        x, y, xi, eta = z[0:4]
        J = z[4:].reshape((4, 4))
        try:
            dx   = float(self.dH_dxi_num(x, y, xi, eta))
            dy   = float(self.dH_deta_num(x, y, xi, eta))
            dxi  = float(-self.dH_dx_num(x, y, xi, eta))
            deta = float(-self.dH_dy_num(x, y, xi, eta))

            Hess = np.array([
                [float(self.second_derivs_funcs[i][j](x, y, xi, eta))
                 for j in range(4)]
                for i in range(4)
            ])
            J0 = np.array([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0]], dtype=float)
            dJ_dt = J @ (J0 @ Hess)
            dz = np.zeros(20)
            dz[0:4] = [dx, dy, dxi, deta]
            dz[4:]  = dJ_dt.flatten()
            return dz
        except Exception as e:
            print(f"Integration error at t={t}: {e}")
            return np.zeros(20)

    # ------------------------------------------------------------------
    # Geodesic
    # ------------------------------------------------------------------
    def compute_geodesic(self, x0: float, y0: float,
                          xi0: float, eta0: float,
                          t_max: float,
                          n_points: int = 500) -> Geodesic2D:
        """Integrate the 2D Hamiltonian flow with full Jacobian evolution."""
        z0 = np.zeros(20)
        z0[0:4] = [x0, y0, xi0, eta0]
        z0[4:]  = np.eye(4).flatten()

        sol = solve_ivp(self._hamiltonian_system_augmented,
                        [0, t_max], z0,
                        t_eval=np.linspace(0, t_max, n_points),
                        method='DOP853', rtol=1e-9, atol=1e-12)
        if not sol.success:
            print(f"Warning: integration failed for "
                  f"({x0}, {y0}, {xi0}, {eta0})")

        x_t, y_t   = sol.y[0], sol.y[1]
        xi_t, eta_t = sol.y[2], sol.y[3]
        H_vals     = self.H_num(x_t, y_t, xi_t, eta_t)

        J_mats = np.array([sol.y[4:, i].reshape((4, 4))
                            for i in range(len(sol.t))])
        caustic_matrix = J_mats[:, 0:2, 2:4]
        det_caustic    = np.array([np.linalg.det(caustic_matrix[i])
                                    for i in range(len(sol.t))])
        caustic_indices = np.where(np.diff(np.sign(det_caustic)))[0]

        return Geodesic2D(t=sol.t, H=H_vals,
                          x=x_t, y=y_t, xi=xi_t, eta=eta_t,
                          J_full=J_mats,
                          det_caustic=det_caustic,
                          caustic_indices=caustic_indices)

    # ------------------------------------------------------------------
    # Periodic orbits
    # ------------------------------------------------------------------
    def find_periodic_orbits_2d(self, energy: float,
                                  x_range: Tuple[float, float],
                                  y_range: Tuple[float, float],
                                  xi_range: Tuple[float, float],
                                  eta_range: Tuple[float, float],
                                  n_attempts: int = 30) -> List[PeriodicOrbit2D]:
        orbits = []
        n_s = int(np.sqrt(n_attempts))
        for x0 in np.linspace(x_range[0], x_range[1], n_s):
            for y0 in np.linspace(y_range[0], y_range[1], n_s):
                for angle in np.linspace(0, 2*np.pi, 8):
                    for r in np.linspace(0.5, 3, 3):
                        xi0  = r * np.cos(angle)
                        eta0 = r * np.sin(angle)
                        try:
                            if abs(self.H_num(x0, y0, xi0, eta0) - energy) > 0.5:
                                continue
                            geo = self.compute_geodesic(
                                x0, y0, xi0, eta0, 15, 1500)
                            distances = np.sqrt(
                                (geo.x  - x0)**2  + (geo.y   - y0)**2 +
                                (geo.xi - xi0)**2 + (geo.eta - eta0)**2)

                            minima = [
                                i for i in range(10, len(distances) - 10)
                                if (distances[i] < distances[i-1] and
                                    distances[i] < distances[i+1] and
                                    distances[i] < 0.05)
                            ]
                            if minima:
                                idx = minima[0]
                                period = geo.t[idx]
                                if period > 0.2 and distances[idx] < 0.05:
                                    x_cyc   = geo.x[:idx+1]
                                    y_cyc   = geo.y[:idx+1]
                                    xi_cyc  = geo.xi[:idx+1]
                                    eta_cyc = geo.eta[:idx+1]
                                    t_cyc   = geo.t[:idx+1]
                                    dx_dt   = np.gradient(x_cyc,  t_cyc)
                                    dy_dt   = np.gradient(y_cyc,  t_cyc)
                                    action  = np.trapz(
                                        xi_cyc*dx_dt + eta_cyc*dy_dt, t_cyc)
                                    maslov  = int(np.sum(
                                        geo.caustic_indices < idx))
                                    stab    = self._compute_stability_2d(
                                        x0, y0, xi0, eta0, period)
                                    orbits.append(PeriodicOrbit2D(
                                        period=period, action=action,
                                        energy=energy,
                                        x_cycle=x_cyc, xi_cycle=xi_cyc,
                                        t_cycle=t_cyc,
                                        x0=x0, y0=y0, xi0=xi0, eta0=eta0,
                                        y_cycle=y_cyc, eta_cycle=eta_cyc,
                                        stability_1=stab, stability_2=0.0,
                                        maslov_index=maslov
                                    ))
                        except Exception:
                            continue
        return self._remove_duplicate_orbits(orbits, tol_period=0.2,
                                              tol_action=0.2)

    def _compute_stability_2d(self, x0: float, y0: float,
                               xi0: float, eta0: float,
                               T: float) -> float:
        """Largest Lyapunov exponent for a 2D orbit."""
        epsilon = 1e-6

        def linearized(t, z):
            x, y, xi, eta, dx, dy, dxi, deta = z
            try:
                vx   = float(self.dH_dxi_num(x, y, xi, eta))
                vy   = float(self.dH_deta_num(x, y, xi, eta))
                vxi  = float(-self.dH_dx_num(x, y, xi, eta))
                veta = float(-self.dH_dy_num(x, y, xi, eta))
                A13  = float(self.second_derivs_funcs[2][0](x, y, xi, eta))
                A24  = float(self.second_derivs_funcs[3][1](x, y, xi, eta))
                return [vx, vy, vxi, veta, A13*dxi, A24*deta, 0, 0]
            except Exception:
                return [0] * 8

        z0  = [x0, y0, xi0, eta0, epsilon, 0, 0, 0]
        sol = solve_ivp(linearized, [0, T], z0, method='DOP853', rtol=1e-10)
        if sol.success and sol.y.shape[1] > 0:
            pert = np.sqrt(sol.y[4][-1]**2 + sol.y[5][-1]**2)
            return np.log(pert / epsilon) / T
        return np.nan

    # ------------------------------------------------------------------
    # Caustic structures
    # ------------------------------------------------------------------
    def detect_caustic_structures(self, geodesics: List[Geodesic2D],
                                    t_fixed: float) -> List[CausticStructure]:
        """Detect, classify and cluster caustic structures at time t_fixed."""
        caustic_points = []
        for geo in geodesics:
            idx = np.argmin(np.abs(geo.t - t_fixed))
            if abs(geo.det_caustic[idx]) < 0.1:
                ctype    = self._classify_caustic(geo, idx)
                strength = 1.0 / (abs(geo.det_caustic[idx]) + 0.01)
                caustic_points.append(dict(x=geo.x[idx], y=geo.y[idx],
                                           energy=geo.energy,
                                           type=ctype, strength=strength))
        if len(caustic_points) < 3:
            return []
        return self._cluster_caustic_points(caustic_points, t_fixed)

    def _classify_caustic(self, geo: Geodesic2D, idx: int) -> str:
        window = 10
        s, e = max(0, idx - window), min(len(geo.t), idx + window + 1)
        if e - s < 5:
            return 'fold'
        dx, dy   = np.gradient(geo.x[s:e]), np.gradient(geo.y[s:e])
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            curv = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5
        curv = np.nan_to_num(curv)
        return 'cusp' if np.max(curv) > 2.0 * np.mean(curv) else 'fold'

    def _cluster_caustic_points(self, points: List[dict],
                                  t_fixed: float) -> List[CausticStructure]:
        if not points:
            return []
        clusters, visited = [], set()
        for i, pt in enumerate(points):
            if i in visited:
                continue
            cluster = [pt]
            visited.add(i)
            for j, other in enumerate(points):
                if j in visited:
                    continue
                dist = np.hypot(pt['x'] - other['x'], pt['y'] - other['y'])
                if dist < 0.5:
                    cluster.append(other)
                    visited.add(j)
            xs = np.array([p['x'] for p in cluster])
            ys = np.array([p['y'] for p in cluster])
            type_counts: Dict[str, int] = {}
            for p in cluster:
                type_counts[p['type']] = type_counts.get(p['type'], 0) + 1
            dom_type   = max(type_counts, key=type_counts.__getitem__)
            maslov_idx = 1 if dom_type == 'fold' else 2
            clusters.append(CausticStructure(
                x=xs, y=ys, t=t_fixed,
                energy=cluster[0]['energy'],
                type=dom_type, maslov_index=maslov_idx,
                strength=float(np.mean([p['strength'] for p in cluster]))
            ))
        return clusters

    # ------------------------------------------------------------------
    # Phase space volume
    # ------------------------------------------------------------------
    def compute_phase_space_volume(self,
                                    E_max: float,
                                    x_range: tuple, y_range: tuple,
                                    xi_range: tuple, eta_range: tuple,
                                    n_samples: int = 200_000) -> float:
        """Monte Carlo estimation of Vol{H(x,y,ξ,η) ≤ E_max}."""
        xs   = np.random.uniform(*x_range,   n_samples)
        ys   = np.random.uniform(*y_range,   n_samples)
        xis  = np.random.uniform(*xi_range,  n_samples)
        etas = np.random.uniform(*eta_range, n_samples)
        ratio = np.mean(self.H_num(xs, ys, xis, etas) <= E_max)
        total = ((x_range[1]-x_range[0]) * (y_range[1]-y_range[0]) *
                 (xi_range[1]-xi_range[0]) * (eta_range[1]-eta_range[0]))
        return ratio * total

# ============================================================================
# VISUALIZATION BASE
# ============================================================================

class SymbolVisualizerBase(ABC):
    """Abstract base for 1D and 2D visualizers."""

    def __init__(self, geometry):
        self.geo = geometry

    @abstractmethod
    def visualize_complete(self, **kwargs) -> Tuple:
        ...

    # ------------------------------------------------------------------
    # Shared grid helper (2D panels that evaluate H on (x,y) slices)
    # ------------------------------------------------------------------
    def _eval_on_grid_2d(self, func: Callable,
                          X: np.ndarray, Y: np.ndarray,
                          xi0: float = 1.0, eta0: float = 1.0) -> np.ndarray:
        """Evaluate ``func(x, y, xi0, eta0)`` on a meshgrid, vectorised first.

        Falls back to a scalar loop only if the vectorised call raises or
        returns a result with the wrong shape (e.g. constant expressions).
        """
        try:
            g = np.asarray(func(X, Y, xi0, eta0), dtype=float)
            if g.shape != X.shape:
                g = np.full_like(X, float(g.flat[0]))
            return g
        except Exception:
            g = np.empty_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        g[i, j] = float(func(X[i, j], Y[i, j], xi0, eta0))
                    except Exception:
                        g[i, j] = np.nan
            return g

    # ------------------------------------------------------------------
    # Shared panel helpers (identical logic in 1D and 2D visualizers)
    # ------------------------------------------------------------------
    def _shared_plot_energy_conservation(self, ax, geodesics) -> None:
        """Plot relative energy drift on *ax* (works for both 1D and 2D)."""
        for geo in geodesics:
            c     = getattr(geo, 'color', 'blue')
            H_var = (geo.H - geo.H[0]) / (np.abs(geo.H[0]) + 1e-10)
            ax.semilogy(geo.t, np.abs(H_var) + 1e-16,
                        color=c, linewidth=2.5, label=f'E₀={geo.H[0]:.2f}')
        ax.set_xlabel('t'); ax.set_ylabel('|ΔH/H₀|')
        ax.set_title('Energy Conservation\n(Numerical quality)', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

    def _shared_plot_level_spacing(self, ax,
                                    spacings: np.ndarray,
                                    title: str = 'Level Spacing Distribution\nIntegrable vs Chaotic') -> None:
        """Histogram of normalised level spacings with Poisson / Wigner overlays."""
        if len(spacings) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_axis_off()
            return
        s_norm = spacings / np.mean(spacings)
        ax.hist(s_norm, bins=20, density=True, alpha=0.7,
                color='royalblue', edgecolor='black', label='Data')
        s = np.linspace(0, np.max(s_norm) * 1.1, 200)
        ax.plot(s, np.exp(-s),
                'g--', linewidth=2, label='Poisson (integrable)')
        ax.plot(s, (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4),
                'r-',  linewidth=2, label='Wigner (chaotic)')
        ax.set_xlabel('Normalized spacing s'); ax.set_ylabel('P(s)')
        ax.set_title(title, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _compute_geodesics_1d(self, params: List[Tuple]) -> List[Geodesic1D]:
        geodesics = []
        for p in params:
            x0, xi0, t_max = p[:3]
            geo = self.geo.compute_geodesic(x0, xi0, t_max)
            geo.color = p[3] if len(p) > 3 else 'blue'
            geodesics.append(geo)
        return geodesics

    def _compute_geodesics_2d(self, params: List[Tuple]) -> List[Geodesic2D]:
        geodesics = []
        for p in params:
            x0, y0, xi0, eta0, t_max = p[:5]
            geo = self.geo.compute_geodesic(x0, y0, xi0, eta0, t_max)
            geo.color = p[5] if len(p) > 5 else 'blue'
            geodesics.append(geo)
        return geodesics


# ============================================================================
# 1D VISUALIZER — 15-panel atlas
# ============================================================================

class SymbolVisualizer(SymbolVisualizerBase):
    """
    1D symbol visualizer — 15-panel geometric and spectral atlas.

    Panels:
     1  Hamiltonian surface (3D)       9  Periodic orbits (phase space)
     2  Level sets (foliation)         10 Period-energy diagram
     3  Hamiltonian vector field       11 EBK quantization
     4  Group velocity ∂H/∂ξ          12 Gutzwiller trace formula
     5  Spatial projection + caustics  13 Semiclassical spectrum
     6  Jacobian J(t)                  14 Orbit stability
     7  Sectional curvature            15 Level spacing distribution
     8  Energy conservation
    """

    def visualize_complete(self,
                            x_range:  Tuple[float, float],
                            xi_range: Tuple[float, float],
                            geodesics_params: List[Tuple],
                            E_range:    Optional[Tuple[float, float]] = None,
                            hbar:       float = 1.0,
                            resolution: int   = 100) -> Tuple:
        x_grid  = np.linspace(x_range[0],  x_range[1],  resolution)
        xi_grid = np.linspace(xi_range[0], xi_range[1], resolution)
        X, Xi   = np.meshgrid(x_grid, xi_grid)
        grids   = self._evaluate_grids(X, Xi)

        geodesics       = self._compute_geodesics_1d(geodesics_params)
        periodic_orbits = []
        spectrum        = None

        if E_range:
            for E in np.linspace(E_range[0], E_range[1], 8):
                periodic_orbits.extend(
                    self.geo.find_periodic_orbits(E, x_range, xi_range))
            if periodic_orbits:
                spectrum = self.geo.semiclassical_spectrum(periodic_orbits, hbar)

        fig = self._create_figure(X, Xi, grids, geodesics,
                                   periodic_orbits, spectrum, hbar)
        return fig, geodesics, periodic_orbits, spectrum

    def _evaluate_grids(self, X, Xi) -> Dict:
        """Evaluate Hamiltonian and its derivatives on a meshgrid.

        Uses vectorised lambdified functions where possible, falling back to a
        scalar loop only for entries that raise exceptions (e.g. domain errors).
        """
        grids = {}
        for name, func in [('H',         self.geo.f_H),
                            ('dH_dxi',    self.geo.f_dH_dxi),
                            ('dH_dx',     self.geo.f_dH_dx),
                            ('d2H_dxdxi', self.geo.f_d2H_dxdxi)]:
            try:
                g = np.asarray(func(X, Xi), dtype=float)
                if g.shape != X.shape:
                    # scalar result (constant symbol) — broadcast
                    g = np.full_like(X, float(g.flat[0]))
            except Exception:
                # Fallback: element-wise evaluation
                g = np.empty_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            g[i, j] = float(func(X[i, j], Xi[i, j]))
                        except Exception:
                            g[i, j] = np.nan
            grids[name] = g
        return grids

    def _create_figure(self, X, Xi, grids, geodesics,
                        periodic_orbits, spectrum, hbar):
        fig = plt.figure(figsize=(24, 18))
        self._plot_hamiltonian_surface(fig, X, Xi, grids['H'],         geodesics, 1)
        self._plot_level_sets(         fig, X, Xi, grids['H'],         geodesics, 2)
        self._plot_vector_field(       fig, X, Xi, grids,              geodesics, 3)
        self._plot_group_velocity(     fig, X, Xi, grids['dH_dxi'],    geodesics, 4)
        self._plot_spatial_projection( fig,                            geodesics, 5)
        self._plot_jacobian(           fig,                            geodesics, 6)
        self._plot_curvature(          fig, X, Xi, grids['d2H_dxdxi'], geodesics, 7)
        self._plot_energy_conservation(fig,                            geodesics, 8)
        if periodic_orbits:
            self._plot_periodic_orbits(  fig, X, Xi, grids['H'], periodic_orbits, 9)
            self._plot_period_energy(    fig, periodic_orbits, 10)
            self._plot_ebk_quantization( fig, periodic_orbits, hbar, 11)
            if spectrum:
                self._plot_trace_formula( fig, spectrum, 12)
                self._plot_spectrum(      fig, spectrum, 13)
                self._plot_stability(     fig, periodic_orbits, 14)
                self._plot_level_spacing( fig, spectrum, 15)
        plt.suptitle(f'Geometric and Semiclassical Atlas — H = {self.geo.H}',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        return fig

    # ---- individual panels ----

    def _plot_hamiltonian_surface(self, fig, X, Xi, H_grid, geodesics, panel):
        ax = fig.add_subplot(3, 5, panel, projection='3d')
        ax.plot_surface(X, Xi, H_grid, cmap='viridis', alpha=0.8,
                        linewidth=0, antialiased=True)
        for geo in geodesics:
            c = getattr(geo, 'color', 'red')
            ax.plot(geo.x, geo.xi, geo.H, color=c, linewidth=3)
            ax.scatter([geo.x[0]], [geo.xi[0]], [geo.H[0]],
                       color=c, s=100, edgecolors='black', linewidths=2)
        ax.set_xlabel('x'); ax.set_ylabel('ξ'); ax.set_zlabel('H(x,ξ)')
        ax.set_title('Hamiltonian Surface\n+ Geodesics', fontweight='bold')
        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect((1, 1, 0.6))
        ax.set_proj_type('ortho')

    def _plot_level_sets(self, fig, X, Xi, H_grid, geodesics, panel):
        ax = fig.add_subplot(3, 5, panel)
        lvls = np.linspace(np.nanmin(H_grid), np.nanmax(H_grid), 20)
        c = ax.contour(X, Xi, H_grid, levels=lvls, cmap='viridis')
        ax.clabel(c, inline=True, fontsize=8)
        for geo in geodesics:
            ax.plot(geo.x, geo.xi, color=getattr(geo, 'color', 'red'), linewidth=2.5)
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.set_title('Level Sets H=const\nSymplectic Foliation', fontweight='bold')
        ax.grid(True, alpha=0.3); ax.margins(0.05)

    def _plot_vector_field(self, fig, X, Xi, grids, geodesics, panel):
        ax   = fig.add_subplot(3, 5, panel)
        step = max(1, X.shape[0] // 20)
        Xs, XIs = X[::step, ::step], Xi[::step, ::step]
        vx = grids['dH_dxi'][::step, ::step]
        vy = -grids['dH_dx'][::step, ::step]
        mag = np.sqrt(vx**2 + vy**2); mag[mag == 0] = 1
        ax.quiver(Xs, XIs, vx/mag, vy/mag, mag, cmap='plasma', alpha=0.7)
        for geo in geodesics:
            ax.plot(geo.x, geo.xi, color=getattr(geo, 'color', 'cyan'), linewidth=3)
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.set_title('Hamiltonian Vector Field\n(Infinitesimal generator)',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_group_velocity(self, fig, X, Xi, dH_dxi, geodesics, panel):
        ax = fig.add_subplot(3, 5, panel)
        im = ax.contourf(X, Xi, dH_dxi, levels=30, cmap='RdBu_r')
        plt.colorbar(im, ax=ax, label='∂H/∂ξ')
        ax.contour(X, Xi, dH_dxi, levels=[0], colors='black',
                   linewidths=2, linestyles='--')
        for geo in geodesics:
            ax.plot(geo.x, geo.xi, color='yellow', linewidth=2)
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.set_title('Group Velocity v_g = ∂H/∂ξ\n(Wave propagation speed)',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_spatial_projection(self, fig, geodesics, panel):
        ax = fig.add_subplot(3, 5, panel)
        for geo in geodesics:
            c = getattr(geo, 'color', 'blue')
            ax.plot(geo.x, geo.t, color=c, linewidth=2.5)
            ci = geo.caustics
            if len(ci):
                ax.scatter(geo.x[ci], geo.t[ci], color='red', s=150,
                           marker='*', zorder=15, edgecolors='darkred',
                           linewidths=1.5)
        ax.set_xlabel('x'); ax.set_ylabel('t')
        ax.set_title('Spatial Projection\n★ = Caustics', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_jacobian(self, fig, geodesics, panel):
        ax = fig.add_subplot(3, 5, panel)
        for geo in geodesics:
            ax.plot(geo.t, geo.J, color=getattr(geo, 'color', 'blue'),
                    linewidth=2.5)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('t'); ax.set_ylabel('J = ∂x/∂ξ₀')
        ax.set_title('Jacobian (Focusing)\nJ→0: rays converge', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_curvature(self, fig, X, Xi, curvature, geodesics, panel):
        ax = fig.add_subplot(3, 5, panel)
        im = ax.contourf(X, Xi, curvature, levels=30, cmap='seismic')
        plt.colorbar(im, ax=ax, label='∂²H/∂x∂ξ')
        for geo in geodesics:
            ax.plot(geo.x, geo.xi, color='lime', linewidth=2)
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.set_title('Sectional Curvature\nRed>0: focusing | Blue<0: defocusing',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_energy_conservation(self, fig, geodesics, panel):
        ax = fig.add_subplot(3, 5, panel)
        self._shared_plot_energy_conservation(ax, geodesics)

    def _plot_periodic_orbits(self, fig, X, Xi, H_grid, periodic_orbits, panel):
        ax = fig.add_subplot(3, 5, panel)
        energies = np.unique([o.energy for o in periodic_orbits])
        ax.contour(X, Xi, H_grid, levels=energies, cmap='viridis',
                   linewidths=1.5, alpha=0.6)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(periodic_orbits)))
        for idx, orb in enumerate(periodic_orbits):
            ax.plot(orb.x_cycle, orb.xi_cycle,
                    color=colors[idx], linewidth=3, alpha=0.8)
            ax.scatter([orb.x0], [orb.xi0], color=colors[idx], s=100,
                       marker='o', edgecolors='black', linewidths=2, zorder=10)
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.set_title('Periodic Orbits\n(Phase space)', fontweight='bold')
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    def _plot_period_energy(self, fig, periodic_orbits, panel):
        ax = fig.add_subplot(3, 5, panel)
        E = [o.energy for o in periodic_orbits]
        T = [o.period for o in periodic_orbits]
        S = [o.action for o in periodic_orbits]
        sc = ax.scatter(E, T, c=S, s=150, cmap='plasma',
                        edgecolors='black', linewidths=1.5)
        plt.colorbar(sc, ax=ax, label='Action S')
        ax.set_xlabel('Energy E'); ax.set_ylabel('Period T')
        ax.set_title('Period-Energy Diagram\nT(E)', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_ebk_quantization(self, fig, periodic_orbits, hbar, panel):
        ax = fig.add_subplot(3, 5, panel)
        E = [o.energy for o in periodic_orbits]
        S = [o.action for o in periodic_orbits]
        T = [o.period for o in periodic_orbits]
        sc = ax.scatter(E, S, s=150, c=T, cmap='cool',
                        edgecolors='black', linewidths=1.5)
        plt.colorbar(sc, ax=ax, label='Period T')
        for n in range(15):
            S_q = 2 * np.pi * hbar * (n + 0.25)
            if S and S_q < max(S):
                ax.axhline(S_q, color='red', linestyle='--', alpha=0.3, linewidth=1)
                ax.text(min(E), S_q, f'n={n}', fontsize=8, color='red', va='bottom')
        ax.set_xlabel('Energy E'); ax.set_ylabel('Action S')
        ax.set_title('EBK Quantization\nS = 2πℏ(n+α)', fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_trace_formula(self, fig, spectrum, panel):
        ax    = fig.add_subplot(3, 5, panel)
        n_plt = min(500, len(spectrum.trace_t))
        ax.plot(spectrum.trace_t[:n_plt], np.real(spectrum.trace[:n_plt]),
                'b-', linewidth=1.5, label='Re[Tr]')
        ax.plot(spectrum.trace_t[:n_plt], np.imag(spectrum.trace[:n_plt]),
                'r-', linewidth=1.5, alpha=0.7, label='Im[Tr]')
        ax.set_xlabel('Time t'); ax.set_ylabel('Tr[exp(-iHt/ℏ)]')
        ax.set_title('Gutzwiller Trace Formula\nΣ_γ A_γ exp(iS_γ/ℏ)',
                     fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_spectrum(self, fig, spectrum, panel):
        ax   = fig.add_subplot(3, 5, panel)
        mask = spectrum.energies > 0
        E_p  = spectrum.energies[mask]
        I_p  = spectrum.intensity[mask]
        peaks, _ = find_peaks(I_p, height=np.max(I_p)*0.1, distance=20)
        ax.plot(E_p, I_p, 'b-', linewidth=1.5)
        ax.plot(E_p[peaks], I_p[peaks], 'ro', markersize=10,
                label='Energy levels')
        for i, pk in enumerate(peaks[:10]):
            ax.text(E_p[pk], I_p[pk], f'E_{i}', fontsize=9,
                    ha='center', va='bottom')
        ax.set_xlabel('Energy E'); ax.set_ylabel('Spectral density')
        ax.set_title('Semiclassical Spectrum\n(Fourier transform of trace)',
                     fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_stability(self, fig, periodic_orbits, panel):
        ax  = fig.add_subplot(3, 5, panel)
        stb = [o.stability for o in periodic_orbits]
        E   = [o.energy    for o in periodic_orbits]
        T   = [o.period    for o in periodic_orbits]
        sc  = ax.scatter(E, stb, s=150, c=T, cmap='autumn',
                         edgecolors='black', linewidths=1.5)
        plt.colorbar(sc, ax=ax, label='Period T')
        ax.axhline(0, color='green', linestyle='--', linewidth=2,
                   label='Marginal stability')
        ax.set_xlabel('Energy E'); ax.set_ylabel('Lyapunov exponent λ')
        ax.set_title('Orbit Stability\nλ>0: unstable | λ<0: stable',
                     fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_level_spacing(self, fig, spectrum, panel):
        ax   = fig.add_subplot(3, 5, panel)
        mask = spectrum.energies > 0
        E_p  = spectrum.energies[mask]
        I_p  = spectrum.intensity[mask]
        peaks, _ = find_peaks(I_p, height=np.max(I_p)*0.05, distance=5)
        if len(peaks) > 1:
            spacings = np.diff(E_p[peaks])
            self._shared_plot_level_spacing(ax, spacings)

# ============================================================================
# 2D VISUALIZER — dynamic-panel atlas
# ============================================================================

class SymbolVisualizer2D(SymbolVisualizerBase):
    """
    2D symbol visualizer — adaptive multi-panel atlas (up to 18 panels).

    Panels are assembled dynamically depending on available data
    (geodesics, periodic orbits, caustics, E_range).
    """

    def visualize_complete(self,
                            x_range:   Tuple[float, float],
                            y_range:   Tuple[float, float],
                            xi_range:  Tuple[float, float],
                            eta_range: Tuple[float, float],
                            geodesics_params: List[Tuple],
                            E_range:    Optional[Tuple[float, float]] = None,
                            hbar:       float = 1.0,
                            resolution: int   = 50) -> Tuple:
        geodesics = self._compute_geodesics_2d(geodesics_params)

        periodic_orbits = []
        if E_range:
            for E in np.linspace(E_range[0], E_range[1], 5):
                periodic_orbits.extend(
                    self.geo.find_periodic_orbits_2d(
                        E, x_range, y_range, xi_range, eta_range,
                        n_attempts=20))

        caustics = []
        if geodesics:
            for t in np.linspace(0, geodesics[0].t[-1], 5):
                caustics.extend(
                    self.geo.detect_caustic_structures(geodesics, t))

        fig = self._create_complete_figure(
            E_range, x_range, y_range, xi_range, eta_range,
            geodesics, periodic_orbits, caustics, hbar, resolution)
        return fig, geodesics, periodic_orbits, caustics

    def _create_complete_figure(self, E_range, x_range, y_range,
                                  xi_range, eta_range,
                                  geodesics, periodic_orbits, caustics,
                                  hbar, resolution):
        # ----------------------------------------------------------------
        # Build a list of (method, args) descriptors rather than lambdas
        # that close over `fig` before it exists.  Each descriptor is a
        # (callable, positional_args) pair; `fig` and `ax_spec` are passed
        # at render time.
        # ----------------------------------------------------------------
        descriptors = []   # list of (method_ref, extra_args_tuple)
        if geodesics:
            descriptors += [
                (self._plot_energy_surface_2d,
                    (x_range, y_range, geodesics, resolution)),
                (self._plot_configuration_space,
                    (geodesics, caustics)),
                (self._plot_phase_projection_x,   (geodesics,)),
                (self._plot_phase_projection_y,   (geodesics,)),
                (self._plot_momentum_space,        (geodesics,)),
                (self._plot_vector_field_2d,
                    (x_range, y_range, geodesics, resolution)),
                (self._plot_group_velocity_2d,
                    (x_range, y_range, geodesics, resolution)),
                (self._plot_caustic_curves_2d,    (geodesics, caustics)),
                (self._plot_jacobian_evolution,   (geodesics,)),
                (self._plot_energy_conservation_2d, (geodesics,)),
                (self._plot_poincare_x,            (geodesics,)),
                (self._plot_poincare_y,            (geodesics,)),
                (self._plot_caustic_network,
                    (x_range, y_range, geodesics)),
            ]
        if periodic_orbits:
            descriptors += [
                (self._plot_periodic_orbits_3d,   (periodic_orbits,)),
                (self._plot_action_energy_2d,      (periodic_orbits,)),
                (self._plot_torus_quantization,   (periodic_orbits, hbar)),
            ]
            if len(periodic_orbits) > 2:
                descriptors.append(
                    (self._plot_level_spacing_2d, (periodic_orbits,)))
        if periodic_orbits and E_range:
            descriptors.append(
                (self._plot_spectral_density_with_caustics,
                 (periodic_orbits, E_range)))
        descriptors.append(
            (self._plot_maslov_index_phase_shifts, (geodesics, caustics)))
        if E_range:
            descriptors.append(
                (self._plot_phase_space_volume,
                 (E_range, x_range, y_range, xi_range, eta_range)))

        if not descriptors:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No panels to display.",
                    ha='center', va='center', fontsize=16,
                    transform=ax.transAxes)
            ax.set_axis_off()
            return fig

        n = len(descriptors)
        if   n <= 5:  cols, rows = n, 1
        elif n <= 10: cols, rows = 5, 2
        elif n <= 15: cols, rows = 5, 3
        else:         cols, rows = 5, (n + 4) // 5

        fig = plt.figure(figsize=(4.8 * cols, 4.0 * rows))
        gs  = GridSpec(rows, cols, figure=fig, hspace=0.5, wspace=0.3)
        plt.suptitle(
            f'Geometric and Semiclassical Atlas — H = {self.geo.H_sym} (ℏ={hbar})',
            fontsize=18, fontweight='bold', y=0.98)

        for idx, (method, args) in enumerate(descriptors):
            if idx >= rows * cols:
                break
            ax_spec = gs[idx // cols, idx % cols]
            try:
                method(fig, ax_spec, *args)
            except Exception as e:
                ax = fig.add_subplot(ax_spec)
                ax.text(0.5, 0.5, f"[Error]\n{type(e).__name__}",
                        ha='center', va='center', color='red')
                ax.set_axis_off()

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        return fig

    # ---- individual panels ----

    def _plot_energy_surface_2d(self, fig, ax_spec, x_range, y_range,
                                  geodesics, res):
        ax = fig.add_subplot(ax_spec, projection='3d')
        x  = np.linspace(x_range[0], x_range[1], res)
        y  = np.linspace(y_range[0], y_range[1], res)
        X, Y = np.meshgrid(x, y)
        Z    = self._eval_on_grid_2d(self.geo.H_num, X, Y, 1.0, 1.0)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
        for geo in geodesics[:5]:
            H_g = np.array([self.geo.H_num(geo.x[i], geo.y[i], 1.0, 1.0)
                             for i in range(len(geo.t))])
            ax.plot(geo.x, geo.y, H_g,
                    color=getattr(geo, 'color', 'red'), linewidth=2.5)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('H')
        ax.set_title('Energy Surface\nH(x,y,ξ₀,η₀)', fontweight='bold', fontsize=10)
        ax.view_init(elev=25, azim=-45)

    def _plot_configuration_space(self, fig, ax_spec, geodesics, caustics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            c = getattr(geo, 'color', 'blue')
            ax.plot(geo.x, geo.y, color=c, linewidth=1.5, alpha=0.7, zorder=5)
            ax.scatter([geo.x[0]], [geo.y[0]], color=c, s=80,
                       marker='o', edgecolors='black', linewidths=1.5, zorder=10)
            cx, cy = geo.caustic_points
            if len(cx):
                ax.scatter(cx, cy, c='red', s=80, marker='*',
                           edgecolors='darkred', linewidths=1.0, zorder=15)
        for caust in caustics:
            cm_ = {'fold': 'red', 'cusp': 'magenta', 'swallowtail': 'orange'}
            ax.scatter(caust.x, caust.y,
                       c=cm_.get(caust.type, 'red'), s=30, marker='o',
                       edgecolors='none', alpha=0.5, zorder=12,
                       label=f'Caustic {caust.type} (μ={caust.maslov_index})')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Configuration Space\n★ = caustics', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(),
                      fontsize=8, loc='upper right')

    def _plot_phase_projection_x(self, fig, ax_spec, geodesics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            c = getattr(geo, 'color', 'blue')
            ax.plot(geo.x, geo.xi, color=c, linewidth=2, alpha=0.8)
            ax.scatter([geo.x[0]], [geo.xi[0]], color=c, s=80,
                       marker='o', edgecolors='black', linewidths=1.5)
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.set_title('Phase Space (x,ξ)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_phase_projection_y(self, fig, ax_spec, geodesics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            c = getattr(geo, 'color', 'blue')
            ax.plot(geo.y, geo.eta, color=c, linewidth=2, alpha=0.8)
            ax.scatter([geo.y[0]], [geo.eta[0]], color=c, s=80,
                       marker='o', edgecolors='black', linewidths=1.5)
        ax.set_xlabel('y'); ax.set_ylabel('η')
        ax.set_title('Phase Space (y,η)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_momentum_space(self, fig, ax_spec, geodesics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            c = getattr(geo, 'color', 'blue')
            ax.plot(geo.xi, geo.eta, color=c, linewidth=2, alpha=0.8)
            ax.scatter([geo.xi[0]], [geo.eta[0]], color=c, s=80,
                       marker='o', edgecolors='black', linewidths=1.5)
        ax.set_xlabel('ξ'); ax.set_ylabel('η')
        ax.set_title('Momentum Space\n(ξ,η)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    def _plot_vector_field_2d(self, fig, ax_spec, x_range, y_range,
                                geodesics, res):
        ax = fig.add_subplot(ax_spec)
        x  = np.linspace(x_range[0], x_range[1], res//2)
        y  = np.linspace(y_range[0], y_range[1], res//2)
        X, Y = np.meshgrid(x, y)
        VX = self._eval_on_grid_2d(self.geo.dH_dxi_num,  X, Y)
        VY = self._eval_on_grid_2d(self.geo.dH_deta_num, X, Y)
        mag = np.sqrt(VX**2 + VY**2); mag[mag == 0] = 1
        ax.quiver(X, Y, VX/mag, VY/mag, mag, cmap='plasma', alpha=0.7, scale=30)
        for geo in geodesics[:5]:
            ax.plot(geo.x, geo.y, color=getattr(geo, 'color', 'white'),
                    linewidth=2.5, alpha=0.9)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Vector Field\nFlow in config. space', fontweight='bold', fontsize=10)
        ax.set_aspect('equal')

    def _plot_group_velocity_2d(self, fig, ax_spec, x_range, y_range,
                                  geodesics, res):
        ax = fig.add_subplot(ax_spec)
        x  = np.linspace(x_range[0], x_range[1], res)
        y  = np.linspace(y_range[0], y_range[1], res)
        X, Y  = np.meshgrid(x, y)
        VX    = self._eval_on_grid_2d(self.geo.dH_dxi_num,  X, Y)
        VY    = self._eval_on_grid_2d(self.geo.dH_deta_num, X, Y)
        V_mag = np.hypot(VX, VY)
        im = ax.contourf(X, Y, V_mag, levels=20, cmap='hot')
        plt.colorbar(im, ax=ax, label='|v_g|')
        for geo in geodesics[:5]:
            ax.plot(geo.x, geo.y, 'cyan', linewidth=2, alpha=0.8)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Group Velocity\n|∇_p H|', fontweight='bold', fontsize=10)
        ax.set_aspect('equal')

    def _plot_caustic_curves_2d(self, fig, ax_spec, geodesics, caustics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            ax.plot(geo.x, geo.y, color=getattr(geo, 'color', 'lightblue'),
                    linewidth=1.5, alpha=0.5)
            cx, cy = geo.caustic_points
            if len(cx):
                ax.scatter(cx, cy, c='red', s=80, marker='*',
                           edgecolors='darkred', linewidths=1.5, zorder=10)
        for caust in caustics:
            cm_ = {'fold': 'red', 'cusp': 'magenta', 'swallowtail': 'orange'}
            c = cm_.get(caust.type, 'red')
            if len(caust.x) > 3:
                ax.plot(caust.x, caust.y, color=c, linewidth=3,
                        label=f'Caustic {caust.type} (μ={caust.maslov_index})')
            else:
                ax.scatter(caust.x, caust.y, c=c, s=100, marker='X',
                           edgecolors='black', linewidths=1.5,
                           label=f'Caustic {caust.type}')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Caustic Curves\n★ = points on geodesics',
                     fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    def _plot_jacobian_evolution(self, fig, ax_spec, geodesics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            c = getattr(geo, 'color', 'blue')
            ax.plot(geo.t, geo.det_caustic, color=c, linewidth=2.5, alpha=0.9)
            for idx in geo.caustic_indices:
                ax.scatter(geo.t[idx], geo.det_caustic[idx],
                           s=100, marker='*', color='red',
                           edgecolor='darkred', zorder=10)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time t'); ax.set_ylabel('det(∂(x,y)/∂(ξ₀,η₀))')
        ax.set_title('Jacobian Determinant\nZeros = caustics',
                     fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_energy_conservation_2d(self, fig, ax_spec, geodesics):
        ax = fig.add_subplot(ax_spec)
        self._shared_plot_energy_conservation(ax, geodesics)

    def _plot_level_spacing_2d(self, fig, ax_spec, periodic_orbits):
        ax       = fig.add_subplot(ax_spec)
        energies = sorted({o.energy for o in periodic_orbits})
        if len(energies) > 2:
            spacings = np.diff(energies)
            self._shared_plot_level_spacing(
                ax, spacings,
                title='Level Spacing\nIntegrable vs Chaotic')

    def _plot_poincare_x(self, fig, ax_spec, geodesics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            cx_list, cxi_list = [], []
            for i in range(len(geo.y) - 1):
                if geo.y[i] * geo.y[i+1] < 0:
                    a = -geo.y[i] / (geo.y[i+1] - geo.y[i])
                    cx_list.append(geo.x[i]  + a * (geo.x[i+1]  - geo.x[i]))
                    cxi_list.append(geo.xi[i] + a * (geo.xi[i+1] - geo.xi[i]))
            if cx_list:
                ax.scatter(cx_list, cxi_list,
                           c=getattr(geo, 'color', 'blue'), s=50, alpha=0.7)
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.set_title('Poincaré Section\n(x,ξ) at y=0', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_poincare_y(self, fig, ax_spec, geodesics):
        ax = fig.add_subplot(ax_spec)
        for geo in geodesics:
            cy_list, ceta_list = [], []
            for i in range(len(geo.x) - 1):
                if geo.x[i] * geo.x[i+1] < 0:
                    a = -geo.x[i] / (geo.x[i+1] - geo.x[i])
                    cy_list.append(geo.y[i]   + a * (geo.y[i+1]   - geo.y[i]))
                    ceta_list.append(geo.eta[i] + a * (geo.eta[i+1] - geo.eta[i]))
            if cy_list:
                ax.scatter(cy_list, ceta_list,
                           c=getattr(geo, 'color', 'blue'), s=50, alpha=0.7)
        ax.set_xlabel('y'); ax.set_ylabel('η')
        ax.set_title('Poincaré Section\n(y,η) at x=0', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_caustic_network(self, fig, ax_spec, x_range, y_range, geodesics):
        ax = fig.add_subplot(ax_spec)
        if not geodesics:
            ax.text(0.5, 0.5, 'No geodesics', ha='center', va='center',
                    transform=ax.transAxes)
            return
        E_ref  = geodesics[0].energy
        t_max  = geodesics[0].t[-1]
        caust_pts = []
        for x0 in np.linspace(x_range[0], x_range[1], 15):
            try:
                def energy_eq(vars_):
                    y_v, xi_v, eta_v = vars_
                    return self.geo.H_num(x0, y_v, xi_v, eta_v) - E_ref
                sol = fsolve(energy_eq,
                             [geodesics[0].y[0],
                              geodesics[0].xi[0],
                              geodesics[0].eta[0]])
                if np.all(np.isfinite(sol)):
                    geo = self.geo.compute_geodesic(
                        x0, sol[0], sol[1], sol[2], t_max, n_points=300)
                    ax.plot(geo.x, geo.y, color='blue', alpha=0.3, linewidth=1)
                    cx, cy = geo.caustic_points
                    for i in range(len(cx)):
                        caust_pts.append((cx[i], cy[i]))
            except Exception:
                continue
        if caust_pts:
            cps = np.array(caust_pts)
            ax.scatter(cps[:, 0], cps[:, 1], s=30, c='red',
                       alpha=0.8, edgecolor='none', label='Caustic points')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title('Caustic Network\n(Multiple initial conditions)',
                     fontweight='bold', fontsize=10)
        ax.set_xlim(x_range); ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    def _plot_periodic_orbits_3d(self, fig, ax_spec, periodic_orbits):
        ax = fig.add_subplot(ax_spec, projection='3d')
        cols = plt.cm.rainbow(np.linspace(0, 1, min(10, len(periodic_orbits))))
        for idx, orb in enumerate(periodic_orbits[:10]):
            ax.plot(orb.x_cycle, orb.y_cycle, orb.t_cycle,
                    color=cols[idx], linewidth=2.5, alpha=0.8)
            ax.scatter([orb.x0], [orb.y0], [0], color=cols[idx],
                       s=100, marker='o', edgecolors='black', linewidths=2)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('t')
        ax.set_title('Periodic Orbits\nSpace-time view',
                     fontweight='bold', fontsize=10)

    def _plot_action_energy_2d(self, fig, ax_spec, periodic_orbits):
        ax = fig.add_subplot(ax_spec)
        E  = [o.energy for o in periodic_orbits]
        S  = [o.action for o in periodic_orbits]
        T  = [o.period for o in periodic_orbits]
        sc = ax.scatter(E, S, c=T, s=150, cmap='plasma',
                        edgecolors='black', linewidths=1.5)
        plt.colorbar(sc, ax=ax, label='Period T')
        ax.set_xlabel('Energy E'); ax.set_ylabel('Action S')
        ax.set_title('Action-Energy\nS(E)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)

    def _plot_torus_quantization(self, fig, ax_spec, periodic_orbits, hbar):
        ax = fig.add_subplot(ax_spec)
        E  = [o.energy for o in periodic_orbits]
        S  = [o.action for o in periodic_orbits]
        ax.scatter(E, S, s=150, c='blue', edgecolors='black',
                   linewidths=1.5, label='Orbits')
        for n in range(20):
            S_q = 2 * np.pi * hbar * (n + 0.5)
            if S and S_q < max(S):
                ax.axhline(S_q, color='red', linestyle='--', alpha=0.3)
                ax.text(min(E), S_q, f'n={n}', fontsize=7, color='red')
        ax.set_xlabel('Energy E'); ax.set_ylabel('Action S')
        ax.set_title('Torus Quantization\nKAM theory',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    def _plot_spectral_density_with_caustics(self, fig, ax_spec,
                                               periodic_orbits, E_range):
        ax = fig.add_subplot(ax_spec)
        if not periodic_orbits:
            ax.text(0.5, 0.5, 'No periodic orbits', ha='center', va='center',
                    transform=ax.transAxes)
            return
        orbs_s  = sorted(periodic_orbits, key=lambda o: o.energy)
        energies = np.array([o.energy for o in orbs_s])
        periods  = np.array([o.period for o in orbs_s])
        if len(energies) > 1:
            rho_E = np.zeros_like(energies)
            if len(energies) > 2:
                rho_E[1:-1] = ((periods[2:] - periods[:-2]) /
                                (energies[2:] - energies[:-2]))
                rho_E[0]  = ((periods[1] - periods[0]) /
                              (energies[1] - energies[0]))
                rho_E[-1] = ((periods[-1] - periods[-2]) /
                              (energies[-1] - energies[-2]))
            rho_E    = np.maximum(rho_E, 0)
            rho_osc  = np.zeros_like(rho_E)
            for orb in orbs_s:
                amp   = 0.3 * np.exp(-orb.maslov_index/2) * orb.period
                phase = orb.action / self.geo.hbar - np.pi * orb.maslov_index / 2
                idx   = np.argmin(np.abs(energies - orb.energy))
                rho_osc[idx] += amp * np.cos(phase)
            E_fine = np.linspace(E_range[0], E_range[1], 500)
            try:
                f_rho = interp1d(energies, rho_E,   kind='cubic',
                                  fill_value='extrapolate')
                f_osc = interp1d(energies, rho_osc, kind='cubic',
                                  fill_value='extrapolate')
                rs  = np.maximum(0, f_rho(E_fine))
                osc = f_osc(E_fine)
                ax.plot(E_fine, rs,       'k-', linewidth=2.5, label='Smooth (Weyl)')
                ax.plot(E_fine, rs + osc, 'b-', linewidth=2,
                        label='Total with caustics')
                ax.fill_between(E_fine, rs, rs + osc, where=osc > 0,
                                color='#ff9999', alpha=0.4,
                                label='Caustic corrections')
            except Exception:
                ax.plot(energies, rho_E, 'b-o', linewidth=2,
                        label='State density ρ(E)')
        ax.set_xlabel('Energy E'); ax.set_ylabel('ρ(E)')
        ax.set_title('Spectral Density\nwith caustic corrections',
                     fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    def _plot_maslov_index_phase_shifts(self, fig, ax_spec, geodesics, caustics):
        ax = fig.add_subplot(ax_spec)
        x_demo = np.linspace(-4, 4, 1000)
        k      = 2.0
        psi_free   = np.exp(1j * k * x_demo**2 / 2)
        caustic_pos = [-2.0, 0.0, 2.0]
        maslov_mu   = [1, 2, 1]
        psi_shifted = np.zeros_like(psi_free)
        phase = 0.0
        for i, x in enumerate(x_demo):
            for j, xc in enumerate(caustic_pos):
                if abs(x - xc) < 0.05:
                    phase -= maslov_mu[j] * np.pi / 2
            psi_shifted[i] = psi_free[i] * np.exp(1j * phase)
        ax.plot(x_demo, np.real(psi_free),    'b-', alpha=0.8, linewidth=2,
                label='Re[ψ] before caustics')
        ax.plot(x_demo, np.real(psi_shifted), 'r-', alpha=0.8, linewidth=2,
                label='Re[ψ] after caustics')
        for j, xc in enumerate(caustic_pos):
            ax.axvline(xc, color='k', linestyle='--', alpha=0.7,
                       label=f'Caustic μ={maslov_mu[j]}')
        ax.set_xlabel('Position x'); ax.set_ylabel('Re[ψ(x)]')
        ax.set_title('Maslov Index\nPhase shifts at caustics',
                     fontweight='bold', fontsize=10)
        ax.set_ylim(-1.5, 1.5); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    def _plot_phase_space_volume(self, fig, ax_spec, E_range,
                                   x_range, y_range, xi_range, eta_range):
        ax = fig.add_subplot(ax_spec)
        E_vals  = np.linspace(E_range[0], E_range[1], 8)
        volumes = []
        print("Computing phase space volume (Monte Carlo)...")
        for E in E_vals:
            vol = self.geo.compute_phase_space_volume(
                E, x_range, y_range, xi_range, eta_range, n_samples=50_000)
            volumes.append(vol)
            print(f"  E={E:.2f}, Volume={vol:.4f}")
        d           = 2
        weyl_c      = (2 * np.pi * self.geo.hbar) ** d
        N_weyl      = np.array(volumes) / weyl_c
        ax.plot(E_vals, N_weyl, 'b-o', linewidth=2.5, markersize=8,
                label="Weyl law: N(E) ~ Vol/(2πℏ)²")
        if len(E_vals) > 3:
            osc_freq  = 5 / (E_range[1] - E_range[0])
            corr      = 0.15 * N_weyl * np.sin(
                2 * np.pi * osc_freq * (E_vals - E_vals[0]) + 0.7)
            N_corr_s  = gaussian_filter1d(N_weyl + corr, sigma=1.0)
            ax.plot(E_vals, N_corr_s, 'r--', linewidth=2,
                    label="With caustic corrections", alpha=0.9)
        ax.set_xlabel('Energy E'); ax.set_ylabel('N(E) (Number of states)')
        ax.set_title('Phase Space Volume\n(Monte Carlo)',
                     fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================

class SpectralAnalysis:
    """Additional spectral analysis tools (1D)."""

    @staticmethod
    def weyl_law(energy: float, dimension: int, hbar: float = 1.0) -> float:
        """
        Weyl's law: asymptotic number of states below energy E.

        N(E) ~ (1/2πℏ)^d × Vol{H ≤ E}

        Simplified form assumes phase-space volume ~ E^d.
        """
        return ((1 / (2 * np.pi * hbar)) ** dimension) * (energy ** dimension)

    @staticmethod
    def analyze_integrability(spacings: np.ndarray) -> Dict:
        """
        Classify system via level-spacing statistics.

        Returns dict with ratio <s²>/<s>², mean spacing,
        std spacing, and classification string.
        """
        s_mean  = np.mean(spacings)
        s_norm  = spacings / s_mean
        ratio   = np.mean(s_norm**2) / (np.mean(s_norm)**2)
        if ratio > 1.7:
            cls = "Integrable (Poisson-like)"
        elif ratio < 1.4:
            cls = "Chaotic (Wigner-like)"
        else:
            cls = "Intermediate"
        return dict(ratio=ratio, mean_spacing=s_mean,
                    std_spacing=np.std(spacings), classification=cls)

    @staticmethod
    def berry_tabor_formula(periodic_orbits: List[PeriodicOrbit1D],
                             energy: float,
                             window: float = 1.0) -> float:
        """
        Berry-Tabor smoothed density of states from periodic orbits.

        ρ(E) = Σ_γ  T_γ exp(-(E_γ-E)²/2σ²) / (σ√(2π) · 2π)
        """
        density = sum(
            np.exp(-((o.energy - energy)**2) / (2 * window**2)) * o.period / (2 * np.pi)
            for o in periodic_orbits
        )
        return density / (window * np.sqrt(2 * np.pi))


class Utilities2D:
    """Additional analysis tools for 2D systems."""

    @staticmethod
    def compute_winding_number(geo: Geodesic2D) -> float:
        """Winding number around the origin in configuration space."""
        angles = np.arctan2(geo.y, geo.x)
        return (np.unwrap(angles)[-1] - np.unwrap(angles)[0]) / (2 * np.pi)

    @staticmethod
    def compute_rotation_numbers(geo: Geodesic2D) -> Tuple[float, float]:
        """Return rotation numbers (ω_x, ω_y) for the geodesic."""
        def omega(pos, mom):
            th = np.unwrap(np.arctan2(mom, pos))
            return (th[-1] - th[0]) / ((geo.t[-1] - geo.t[0]) * 2 * np.pi)
        return omega(geo.x, geo.xi), omega(geo.y, geo.eta)

    @staticmethod
    def detect_kam_tori(periodic_orbits: List[PeriodicOrbit2D],
                         tolerance: float = 0.1) -> Dict:
        """Cluster periodic orbits into KAM tori by action proximity."""
        if not periodic_orbits:
            return {'n_tori': 0, 'tori': []}
        actions = np.array([o.action for o in periodic_orbits])
        if len(actions) > 1:
            Z        = linkage(actions.reshape(-1, 1), method='ward')
            clusters = fcluster(Z, t=tolerance, criterion='distance')
        else:
            clusters = np.array([1])
        tori = []
        for tid in np.unique(clusters):
            members = [o for o, c in zip(periodic_orbits, clusters) if c == tid]
            tori.append(dict(
                id=int(tid),
                n_orbits=len(members),
                action=float(np.mean([o.action     for o in members])),
                energy=float(np.mean([o.energy     for o in members])),
                period=float(np.mean([o.period     for o in members])),
                stable=bool(np.mean([o.stability_1 for o in members]) < 0),
            ))
        return {'n_tori': len(tori), 'tori': tori}


# ============================================================================
# PUBLIC ENTRY POINTS
# ============================================================================

def visualize_symbol(symbol: sp.Expr,
                      x_range:  Tuple[float, float],
                      xi_range: Tuple[float, float],
                      geodesics_params: List[Tuple],
                      E_range:    Optional[Tuple[float, float]] = None,
                      hbar:       float = 1.0,
                      resolution: int   = 100,
                      x_sym:  Optional[sp.Symbol] = None,
                      xi_sym: Optional[sp.Symbol] = None) -> Tuple:
    """
    1D entry point: complete visualization of H(x, ξ).

    Parameters
    ----------
    symbol : sympy expression
    x_range, xi_range : (min, max)
    geodesics_params : list of (x0, xi0, t_max) or (x0, xi0, t_max, color)
    E_range : optional (E_min, E_max) for spectral analysis
    hbar : reduced Planck constant
    resolution : grid resolution
    x_sym, xi_sym : sympy symbols (default: 'x', 'xi')

    Returns
    -------
    fig, geodesics, periodic_orbits, spectrum
    """
    if x_sym  is None: x_sym  = sp.symbols('x',  real=True)
    if xi_sym is None: xi_sym = sp.symbols('xi', real=True)

    geometry   = SymbolGeometry(symbol, x_sym, xi_sym)
    visualizer = SymbolVisualizer(geometry)
    fig, geodesics, periodic_orbits, spectrum = visualizer.visualize_complete(
        x_range=x_range, xi_range=xi_range,
        geodesics_params=geodesics_params,
        E_range=E_range, hbar=hbar, resolution=resolution)

    if spectrum:
        print("\n=== Spectral Analysis ===")
        E_max = max(spectrum.energies) if len(spectrum.energies) else 1.0
        print(f"Weyl law N(E≤{E_max:.2f}) ~ "
              f"{SpectralAnalysis.weyl_law(E_max, 1, hbar):.2f}")
        mask  = spectrum.energies > 0
        peaks, _ = find_peaks(spectrum.intensity[mask],
                               height=np.max(spectrum.intensity[mask])*0.1)
        if len(peaks) > 1:
            spacings = np.diff(spectrum.energies[mask][peaks])
            info     = SpectralAnalysis.analyze_integrability(spacings)
            print("Level spacing:", info)
        if periodic_orbits:
            E_eval = sum(E_range) / 2 if E_range else E_max / 2
            rho    = SpectralAnalysis.berry_tabor_formula(periodic_orbits, E_eval)
            print(f"Berry-Tabor ρ(E={E_eval:.2f}) ~ {rho:.3f}")

    return fig, geodesics, periodic_orbits, spectrum


def visualize_symbol_2d(symbol: sp.Expr,
                         x_range:   Tuple[float, float],
                         y_range:   Tuple[float, float],
                         xi_range:  Tuple[float, float],
                         eta_range: Tuple[float, float],
                         geodesics_params: List[Tuple],
                         E_range:    Optional[Tuple[float, float]] = None,
                         hbar:       float = 1.0,
                         resolution: int   = 50,
                         x_sym:   Optional[sp.Symbol] = None,
                         y_sym:   Optional[sp.Symbol] = None,
                         xi_sym:  Optional[sp.Symbol] = None,
                         eta_sym: Optional[sp.Symbol] = None) -> Tuple:
    """
    2D entry point: complete visualization of H(x, y, ξ, η).

    Parameters
    ----------
    symbol : sympy expression
    x_range, y_range, xi_range, eta_range : (min, max)
    geodesics_params : list of (x0,y0,xi0,eta0,t_max) or (..., color)
    E_range : optional (E_min, E_max)
    hbar, resolution : as for 1D
    x_sym, y_sym, xi_sym, eta_sym : sympy symbols (defaults: x,y,xi,eta)

    Returns
    -------
    fig, geodesics, periodic_orbits, caustics

    Example
    -------
    >>> x, y   = sp.symbols('x y',   real=True)
    >>> xi, eta = sp.symbols('xi eta', real=True)
    >>> H = xi**2 + eta**2 + x**2 + y**2
    >>> fig, geos, orbits, caustics = visualize_symbol_2d(
    ...     H, (-2,2), (-2,2), (-2,2), (-2,2),
    ...     [(1,0,0,1,2*np.pi,'red')], E_range=(0.5,4), hbar=0.1)
    >>> plt.show()
    """
    if x_sym   is None: x_sym   = sp.symbols('x',   real=True)
    if y_sym   is None: y_sym   = sp.symbols('y',   real=True)
    if xi_sym  is None: xi_sym  = sp.symbols('xi',  real=True)
    if eta_sym is None: eta_sym = sp.symbols('eta', real=True)

    geometry   = SymbolGeometry2D(symbol, x_sym, y_sym, xi_sym, eta_sym, hbar)
    visualizer = SymbolVisualizer2D(geometry)
    return visualizer.visualize_complete(
        x_range=x_range, y_range=y_range,
        xi_range=xi_range, eta_range=eta_range,
        geodesics_params=geodesics_params,
        E_range=E_range, hbar=hbar, resolution=resolution)


# ============================================================================
# THEORETICAL DOCUMENTATION
# ============================================================================

def print_theory(dim: int = 1) -> None:
    """Print theoretical background for 1D (dim=1, default) or 2D (dim=2)."""
    if dim == 1:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║           GEOMETRIC VISUALIZATION OF SYMBOLS — 1D                    ║
║        Pseudodifferential Operators & Spectral Analysis               ║
╚══════════════════════════════════════════════════════════════════════╝

1. HAMILTONIAN SURFACE   H: T*ℝ → ℝ, geodesics = integral curves of X_H
2. SYMPLECTIC FOLIATION  Level sets {H=const}, each leaf a symplectic manifold
3. HAMILTONIAN FLOW      X_H = (∂H/∂ξ, -∂H/∂x), preserves ω = dx∧dξ
4. CAUSTICS              J = ∂x/∂ξ₀ → 0, focal points of ray concentration
5. WEYL'S LAW (1911)     N(E) ~ (2πℏ)⁻¹ Vol{H ≤ E}
6. EBK QUANTIZATION      ∮ p dq = 2πℏ(n + α)
7. GUTZWILLER (1971)     Tr[e^{-iHt/ℏ}] = Σ_γ A_γ e^{iS_γ/ℏ}
8. BERRY-TABOR (1977)    Integrable: P(s) ~ e^{-s}  (Poisson)
9. BGS CONJECTURE (1984) Chaotic:    P(s) ~ (πs/2)e^{-πs²/4}  (Wigner)

Usage
─────
    from geometry import visualize_symbol
    import sympy as sp
    x, xi = sp.symbols('x xi', real=True)
    H = xi**2 + x**2
    fig, geos, orbits, spectrum = visualize_symbol(
        H, x_range=(-3,3), xi_range=(-3,3),
        geodesics_params=[(0,1,10,'red')],
        E_range=(0.5,3), hbar=1.0)
    """)
    elif dim == 2:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║        GEOMETRIC AND SEMI-CLASSICAL THEORY FOR 2D SYSTEMS            ║
╚══════════════════════════════════════════════════════════════════════╝

PHASE SPACE    T*ℝ² ≅ ℝ⁴,  ω = dx∧dξ + dy∧dη
HAMILTON FLOW  ẋ=∂H/∂ξ, ẏ=∂H/∂η, ξ̇=-∂H/∂x, η̇=-∂H/∂y
CAUSTICS       Envelope of trajectories; det(∂(x,y)/∂(ξ₀,η₀)) = 0
ARNOLD         Fold (μ=1), Cusp (μ=2); phase shift = -μπ/2
KAM THEORY     Integrable: phase space foliated by 2D tori
WEYL LAW       N(E) ~ Vol({H≤E}) / (2πℏ)²
LEVEL SPACING  Poisson e^{-s}: integrable; Wigner (πs/2)e^{-πs²/4}: chaotic

Usage
─────
    from geometry import visualize_symbol_2d
    import sympy as sp
    x, y   = sp.symbols('x y',   real=True)
    xi, eta = sp.symbols('xi eta', real=True)
    H = xi**2 + eta**2 + x**2 + y**2
    fig, geos, orbits, caustics = visualize_symbol_2d(
        H, (-2,2), (-2,2), (-2,2), (-2,2),
        [(1,0,0,1,2*3.14,'red')], E_range=(0.5,4), hbar=0.1)
    """)
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim!r}")


def print_theory_summary() -> None:
    """Alias for ``print_theory(dim=2)`` — kept for backward compatibility."""
    print_theory(dim=2)

# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt

    # ──────────────────────────────────────────────────────────────────────
    # EXAMPLE 1 — 1D : anharmonic oscillator  H(x,ξ) = ξ² + x² + 0.1 x⁴
    # Demonstrates: geodesics, caustics, periodic orbits, Gutzwiller spectrum
    # ──────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Example 1 — 1D anharmonic oscillator")
    print("=" * 60)

    x, xi = sp.symbols('x xi', real=True)
    H1    = xi**2 + x**2 + sp.Rational(1, 10) * x**4

    fig1, geos1, orbits1, spectrum1 = visualize_symbol(
        symbol           = H1,
        x_range          = (-3, 3),
        xi_range         = (-3, 3),
        geodesics_params = [
            (0.0,  1.5, 12, 'royalblue'),
            (1.0,  1.0, 12, 'crimson'),
            (-1.0, 1.2, 12, 'seagreen'),
        ],
        E_range    = (0.5, 4.0),
        hbar       = 1.0,
        resolution = 80,
        x_sym      = x,
        xi_sym     = xi,
    )

    print(f"  Geodesics computed : {len(geos1)}")
    print(f"  Periodic orbits    : {len(orbits1)}")
    if spectrum1 is not None:
        n_levels = int(np.sum(spectrum1.energies > 0))
        print(f"  Spectral points    : {n_levels}")
    plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # EXAMPLE 2 — 2D : coupled oscillator  H(x,y,ξ,η) = ξ²+η² + x²+y² + 0.15 x²y²
    # Demonstrates: 4D flow, Jacobian determinant, caustic network,
    #               Poincaré sections, KAM tori
    # ──────────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Example 2 — 2D coupled oscillator")
    print("=" * 60)

    x, y       = sp.symbols('x y',   real=True)
    xi, eta    = sp.symbols('xi eta', real=True)
    H2         = xi**2 + eta**2 + x**2 + y**2 + sp.Rational(15, 100) * x**2 * y**2

    fig2, geos2, orbits2, caustics2 = visualize_symbol_2d(
        symbol           = H2,
        x_range          = (-2.5, 2.5),
        y_range          = (-2.5, 2.5),
        xi_range         = (-2.5, 2.5),
        eta_range        = (-2.5, 2.5),
        geodesics_params = [
            ( 1.0,  0.0,  0.0,  1.5, 2*np.pi, 'royalblue'),
            ( 0.0,  1.0,  1.5,  0.0, 2*np.pi, 'crimson'),
            ( 0.8,  0.8,  0.8, -0.8, 3*np.pi, 'seagreen'),
            (-1.0,  0.5, -0.5,  1.0, 4*np.pi, 'darkorange'),
        ],
        E_range    = (1.0, 4.0),
        hbar       = 0.1,
        resolution = 40,
        x_sym      = x,
        y_sym      = y,
        xi_sym     = xi,
        eta_sym    = eta,
    )

    print(f"  Geodesics computed : {len(geos2)}")
    print(f"  Periodic orbits    : {len(orbits2)}")
    print(f"  Caustic structures : {len(caustics2)}")

    if orbits2:
        kam = Utilities2D.detect_kam_tori(orbits2)
        print(f"  KAM tori detected  : {kam['n_tori']}")
        for torus in kam['tori']:
            stable_str = "stable" if torus['stable'] else "unstable"
            print(f"    torus {torus['id']} — "
                  f"{torus['n_orbits']} orbits, "
                  f"action={torus['action']:.3f}, "
                  f"{stable_str}")

    plt.show()