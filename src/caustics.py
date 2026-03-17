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
caustics.py — Catastrophe classification, ray caustic detection, and uniform asymptotic corrections
===========================================================================================================

Overview
--------
The `caustics` module provides a comprehensive toolkit for working with caustics in semiclassical analysis and catastrophe theory.  It consolidates previously scattered functionality into a single, well‑tested module with a clean API.  The main components are:

1. **Arnold classification** (algebraic) – Symbolic classification of critical points of a function `H(ξ)` (1D) or `H(ξ,η)` (2D) into elementary catastrophes: Morse, fold (A2), cusp (A3), swallowtail (A4), butterfly (A5), and umbilics (D4±).  Uses successive derivatives and Hessian analysis.

2. **Catastrophe detection** (adaptive numerical solver) – Robustly finds all critical points of `H` by combining symbolic solving, coarse grid scanning, Newton refinement, and clustering.  Then classifies them using the algebraic routines.

3. **Ray‑based caustic detection** – Operates on a bundle of rays (output of `wkb.py`) that contain the integrated **stability matrix** `J = ∂x(t)/∂q`.  Caustics are detected when `det(J)` crosses zero.  This corrects the common mistake of using velocity zeros or momentum minima as caustic indicators.  Maslov indices and phase shifts are computed automatically.

4. **Uniform special functions** – Implements the Airy function (fold), the Pearcey integral (cusp), and a swallowtail integral (A4), together with the correct prefactors for uniform WKB approximations near caustics.

5. **Visualisation helpers** – Functions to plot catastrophe points on 1D curves or 2D surfaces, and to overlay caustic events on ray bundles.

The module is designed to be **independent** of the other packages (it does not import `wkb.py`, `psiop.py`, etc.), but it is used by `wkb.py` for caustic detection and correction.

Mathematical background
-----------------------
**Caustics** are envelopes of families of rays (bicharacteristics) where the ray density becomes infinite.  In the semiclassical approximation, the amplitude diverges at a caustic, and the standard WKB ansatz breaks down.  Uniform approximations replace the oscillatory exponential by special functions.

For a Hamiltonian system with `n` degrees of freedom, a family of rays is parameterised by an initial‑condition parameter `q ∈ ℝⁿ`.  The **stability matrix**

    J(t) = ∂x(t)/∂q

satisfies the variational equation `dJ/dt = H_{px}·J` along each ray.  A caustic occurs when `det(J(t)) = 0`.  The **Maslov index** `μ` is the signed number of such zero crossings; each crossing contributes a phase shift of `π/2` to the wave function.

**Arnold’s classification of catastrophes** (singularities of gradient maps) provides a taxonomy of caustics:

* **A₂ (fold)**: normal form `ξ³` – the simplest caustic, described by the Airy function.
* **A₃ (cusp)**: normal form `ξ⁴` – described by the Pearcey integral.
* **A₄ (swallowtail)**: normal form `ξ⁵` – a three‑parameter integral.
* **D₄± (umbilics)**: normal forms `ξ³ + η³` (hyperbolic) and `ξ³ – 3ξη²` (elliptic) – degenerate cases with Hessian rank 0.

The module classifies a critical point (where `∇H = 0`) by examining its Hessian rank and higher derivatives, using algebraic invariants.

References
----------
.. [1] Arnold, V. I.  *Catastrophe Theory*, Springer‑Verlag, 1986.
.. [2] Duistermaat, J. J.  “Oscillatory integrals, Lagrange immersions and unfolding of singularities”, *Comm. Pure Appl. Math.* **27**, 207–281, 1974.
.. [3] Maslov, V. P. & Fedoriuk, M. V.  *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
.. [4] Kravtsov, Yu. A. & Orlov, Yu. I.  *Caustics, Catastrophes and Wave Fields*, Springer, 1999.
.. [5] Berry, M. V. & Howls, C. J.  “High orders of the Weyl expansion for quantum billiards”, *Phys. Rev. E* **50**(5), 3577–3595, 1994.
.. [6] Connor, J. N. L.  “Practical methods for the uniform asymptotic evaluation of oscillatory integrals”, *Mol. Phys.* **31**(1), 33–55, 1976.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy as sp

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import fsolve
    from scipy.special import airy
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not found — numeric fallbacks disabled.", ImportWarning)

try:
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Arnold classification (algebraic)
# ══════════════════════════════════════════════════════════════════════════════

def classify_arnold_1d(f: sp.Expr,
                       xi: sp.Symbol,
                       point: float,
                       max_order: int = 8,
                       tol: float = 1e-8) -> dict:
    """
    Classify a 1D singularity of f(xi) at xi = point.

    Algorithm: evaluate successive derivatives f^(k)(point) until the first
    non-vanishing one determines the A_k type.

    Parameters
    ----------
    f : sympy.Expr
        Function of xi.
    xi : sympy.Symbol
    point : float
        Critical point (should satisfy f'(point) ≈ 0).
    max_order : int
        Maximum derivative order to check (default 8 covers up to A7).
    tol : float
        Threshold for "non-zero" derivative.

    Returns
    -------
    dict with keys:
        'type'        : str   e.g. "A2 (Fold)", "A3 (Cusp)", ...
        'order'       : int   k such that f^(k)(point) ≠ 0
        'derivatives' : dict  {k: value} for k = 1..max_order
        'point'       : float
    """
    subs = {xi: point}
    derivs = {}
    for k in range(1, max_order + 1):
        val = float(sp.N(sp.diff(f, (xi, k)).subs(subs)))
        derivs[k] = val

    # k=1 non-zero → regular point, not a caustic
    if abs(derivs.get(1, 0.0)) > tol:
        return {"type": "regular (not critical)", "order": 1,
                "derivatives": derivs, "point": point}

    _ARNOLD_1D = {
        2: "A1 (Morse/non-degenerate)",
        3: "A2 (Fold)",
        4: "A3 (Cusp)",
        5: "A4 (Swallowtail)",
        6: "A5 (Butterfly)",
        7: "A6",
        8: "A7",
    }
    for k in range(2, max_order + 1):
        if abs(derivs.get(k, 0.0)) > tol:
            label = _ARNOLD_1D.get(k, f"A{k-1} (higher)")
            return {"type": label, "order": k,
                    "derivatives": derivs, "point": point}

    return {"type": f"flat (all derivatives vanish up to order {max_order})",
            "order": None, "derivatives": derivs, "point": point}


def classify_arnold_2d(H: sp.Expr,
                       xi: sp.Symbol,
                       eta: sp.Symbol,
                       point: dict,
                       tol: float = 1e-8) -> dict:
    """
    Classify a 2D catastrophe at a critical point of H(xi, eta).

    Follows Arnold's classification:
        Morse   rank 2  → non-degenerate
        A_k     rank 1  → fold (A2), cusp (A3), swallowtail (A4),
                          butterfly (A5), A6+
        D4±     rank 0  → hyperbolic umbilic (I>0) / elliptic umbilic (I<0)
        higher  rank 0, I=0 → E6 or beyond

    Parameters
    ----------
    H : sympy.Expr
        Function of (xi, eta).
    xi, eta : sympy.Symbol
    point : dict
        Numeric values {'xi': float, 'eta': float}.
        Extra keys (space coordinates) are ignored.
    tol : float

    Returns
    -------
    dict with keys:
        'type'                   : str
        'hessian'                : 2×2 list
        'third_order_tensor'     : dict of H_xxx, H_xxy, H_xyy, H_yyy
        'directional_derivatives': dict of D3…D6  (rank-1 case)
        'cubic_invariant_I'      : float           (rank-0 case)
    """

    # --- Determine xi and eta values (accept both symbol and string keys) ---
    if xi in point:
        xi_val = float(point[xi])
        eta_val = float(point[eta])
    elif "xi" in point and "eta" in point:
        xi_val = float(point["xi"])
        eta_val = float(point["eta"])
    else:
        raise KeyError(f"Point dict must contain keys {xi} or 'xi'/'eta'")

    subs = {xi: xi_val, eta: eta_val}
#    subs = {xi: float(point["xi"]), eta: float(point["eta"])}

    # ── Hessian ──────────────────────────────────────────────────
    H_xx = float(sp.N(sp.diff(H, (xi, 2)).subs(subs)))
    H_yy = float(sp.N(sp.diff(H, (eta, 2)).subs(subs)))
    H_xy = float(sp.N(sp.diff(H, xi, eta).subs(subs)))
    Hess = np.array([[H_xx, H_xy], [H_xy, H_yy]], dtype=float)
    rank = int(np.linalg.matrix_rank(Hess, tol=tol))
    eigvals, eigvecs = np.linalg.eigh(Hess)   # eigh: real symmetric

    # ── Third-order tensor ────────────────────────────────────────
    H_xxx = float(sp.N(sp.diff(H, (xi, 3)).subs(subs)))
    H_xxy = float(sp.N(sp.diff(H, xi, xi, eta).subs(subs)))
    H_xyy = float(sp.N(sp.diff(H, xi, eta, eta).subs(subs)))
    H_yyy = float(sp.N(sp.diff(H, (eta, 3)).subs(subs)))
    third = {"H_xxx": H_xxx, "H_xxy": H_xxy, "H_xyy": H_xyy, "H_yyy": H_yyy}

    base = {"hessian": Hess.tolist(), "third_order_tensor": third}

    # ── Case 1 : Morse (rank 2) ───────────────────────────────────
    if rank == 2:
        return {**base, "type": "Morse (non-degenerate)"}

    # ── Case 2 : rank 1 → A_k series ─────────────────────────────
    if rank == 1:
        # Null direction = eigenvector of the zero eigenvalue
        null_idx = int(np.argmin(np.abs(eigvals)))
        null_dir = eigvecs[:, null_idx]
        vx, vy = float(null_dir[0]), float(null_dir[1])

        # Directional derivative operator D
        def _D(expr):
            return vx * sp.diff(expr, xi) + vy * sp.diff(expr, eta)

        # Build D^k H symbolically up to order 6
        DH  = _D(H)
        D2H = _D(DH)
        D3H = _D(D2H)
        D4H = _D(D3H)
        D5H = _D(D4H)
        D6H = _D(D5H)

        D3 = float(sp.N(D3H.subs(subs)))
        D4 = float(sp.N(D4H.subs(subs)))
        D5 = float(sp.N(D5H.subs(subs)))
        D6 = float(sp.N(D6H.subs(subs)))

        directional = {"D3": D3, "D4": D4, "D5": D5, "D6": D6,
                       "null_direction": null_dir.tolist()}

        _AK = {3: "A2 (Fold)", 4: "A3 (Cusp)",
               5: "A4 (Swallowtail)", 6: "A5 (Butterfly)"}

        for order, (val, lbl) in zip(
                [3, 4, 5, 6], [(D3, _AK[3]), (D4, _AK[4]),
                                (D5, _AK[5]), (D6, _AK[6])]):
            if abs(val) > tol:
                return {**base, "type": lbl,
                        "directional_derivatives": directional}

        return {**base, "type": "A6+ or higher degeneracy",
                "directional_derivatives": directional}

    # ── Case 3 : rank 0 → D4 umbilics ────────────────────────────
    # Discriminant of the cubic form C(v,w) = a v³ + 3b v²w + 3c vw² + d w³
    # with  a=H_xxx, b=H_xxy, c=H_xyy, d=H_yyy
    # Δ > 0  → three distinct real roots → D4- (elliptic umbilic)
    # Δ < 0  → one real root             → D4+ (hyperbolic umbilic)
    # Δ = 0  → repeated root             → higher degeneracy (E6…)
    a, b, c, d = H_xxx, H_xxy, H_xyy, H_yyy
    I = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
    if abs(I) < tol:
        return {**base, "type": "D4 degenerate (I≈0) — possible E6 or higher",
                "cubic_invariant_I": float(I)}
    elif I < 0:
        # Δ < 0 → one real root of cubic → D4+ (hyperbolic umbilic)
        # Normal form: x³ + y³  (three-cusped shape)
        return {**base, "type": "D4+ (Hyperbolic umbilic)",
                "cubic_invariant_I": float(I)}
    else:
        # Δ > 0 → three distinct real roots → D4- (elliptic umbilic)
        # Normal form: x³ - 3xy²  (three-petalled shape)
        return {**base, "type": "D4- (Elliptic umbilic)",
                "cubic_invariant_I": float(I)}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Catastrophe detection with adaptive solver
# ══════════════════════════════════════════════════════════════════════════════

def find_critical_points_numerical(
    grad_func,
    initial_guesses: List[np.ndarray],
    tolerance: float = 1e-6,
    domain: Optional[List[Tuple[float, float]]] = None,
    cluster_tol: float = 1e-4,
) -> List[np.ndarray]:
    """
    Shared numerical kernel for locating critical points where ∇φ = 0.

    Minimises |∇φ(x)|² from each initial guess using L-BFGS-B, then
    deduplicates nearby solutions with a greedy DBSCAN-style clustering.

    This function is the single source of truth used by:
      - ``asymptotic.Analyzer.find_critical_points``
      - ``fio_bridge._BoundAnalyzer.find_critical_points``
    It can also be called directly for lightweight use cases.

    Parameters
    ----------
    grad_func : callable
        Function ``grad_func(*x) -> array-like`` returning the gradient ∇φ
        evaluated at the point ``x`` (unpacked coordinates).
    initial_guesses : list of np.ndarray
        Starting points for the optimisation.
    tolerance : float
        Convergence threshold: a point is accepted when |∇φ|² < tolerance.
    domain : list of (lo, hi) tuples, optional
        If provided, only points satisfying ``lo <= x_i <= hi`` are kept.
    cluster_tol : float
        Distance threshold for deduplication; solutions closer than this
        are merged into a single representative point (cluster mean).

    Returns
    -------
    list of np.ndarray
        Unique critical points found within the given tolerance and domain.
    """
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for find_critical_points_numerical.")

    from scipy.optimize import minimize as _minimize

    def objective(x):
        g = np.asarray(grad_func(*x), dtype=float)
        return float(np.dot(g, g))

    raw = []
    for guess in initial_guesses:
        try:
            res = _minimize(objective, guess, tol=tolerance, method='L-BFGS-B')
            if res.success and res.fun < tolerance:
                xc = res.x
                if domain is not None:
                    if not all(lo <= xi <= hi for xi, (lo, hi) in zip(xc, domain)):
                        continue
                raw.append(xc)
        except Exception:
            pass

    # Greedy DBSCAN-style deduplication
    clusters: List[List[np.ndarray]] = []
    for pt in raw:
        for cluster in clusters:
            if np.linalg.norm(pt - cluster[0]) < cluster_tol:
                cluster.append(pt)
                break
        else:
            clusters.append([pt])

    return [np.mean(c, axis=0) for c in clusters]


class AdaptiveCriticalPointSolver:
    """
    Find critical points of H(xi_vars) robustly.

    Strategy
    --------
    1. Symbolic solve (sp.solve) — exact when polynomial.
    2. Coarse grid scan — evaluate |∇H| on a grid, keep near-zero cells
       as seeds for Newton refinement.
    3. Newton / fsolve refinement — high-accuracy convergence from seeds.
    4. DBSCAN-style deduplication — merge solutions closer than `cluster_tol`.

    This replaces the fragile uniform-grid + sp.nsolve approach that missed
    solutions and was slow.

    Parameters
    ----------
    H_expr : sp.Expr
    xi_vars : sequence of sp.Symbol
        Frequency variables (the unknowns for ∇H = 0).
    coords : sequence of sp.Symbol, optional
        Additional parameter symbols (treated as unknowns too).
    bounds : dict {symbol: (lo, hi)}, optional
        Search bounds per symbol.  Defaults to (-5, 5) for all.
    coarse_n : int
        Points per axis for the coarse scan (default 30).
    cluster_tol : float
        Distance threshold for deduplication (default 1e-4).
    grad_tol : float
        |∇H| threshold to accept a point as critical (default 1e-8).
    """

    def __init__(self,
                 H_expr: sp.Expr,
                 xi_vars: Sequence[sp.Symbol],
                 coords: Optional[Sequence[sp.Symbol]] = None,
                 bounds: Optional[Dict] = None,
                 coarse_n: int = 30,
                 cluster_tol: float = 1e-4,
                 grad_tol: float = 1e-8):

        self.H_expr     = H_expr
        self.xi_vars    = list(xi_vars)
        self.coords     = list(coords) if coords else []
        self.unknowns   = self.xi_vars + self.coords
        self.dim        = len(self.unknowns)
        self.coarse_n   = coarse_n
        self.cluster_tol = cluster_tol
        self.grad_tol   = grad_tol

        # Default bounds
        _default = (-5.0, 5.0)
        self.bounds = {v: _default for v in self.unknowns}
        if bounds:
            self.bounds.update(bounds)

        # Pre-compile gradient as numpy function
        self._grad_exprs = [sp.diff(H_expr, v) for v in self.unknowns]
        self._grad_func  = sp.lambdify(self.unknowns, self._grad_exprs, "numpy")
        self._H_func     = sp.lambdify(self.unknowns, H_expr, "numpy")

    # ── Private helpers ──────────────────────────────────────────

    def _eval_grad(self, pt: np.ndarray) -> np.ndarray:
        """Evaluate ∇H at a point as a numpy array."""
        try:
            g = self._grad_func(*pt)
            return np.array(g, dtype=float).ravel()
        except Exception:
            return np.full(self.dim, np.inf)

    def _coarse_seeds(self) -> List[np.ndarray]:
        """Coarse grid scan: return points where |∇H| is locally small."""
        axes = [np.linspace(self.bounds[v][0], self.bounds[v][1], self.coarse_n)
                for v in self.unknowns]
        grid_pts = list(itertools.product(*axes))

        # Evaluate |∇H|² on all grid points
        grad_norms = []
        for pt in grid_pts:
            g = self._eval_grad(np.array(pt))
            grad_norms.append(np.dot(g, g))

        grad_norms = np.array(grad_norms)

        # Keep seeds where |∇H|² is in the bottom 10% OR below an absolute threshold
        finite_norms = grad_norms[np.isfinite(grad_norms)]
        if len(finite_norms) == 0:
            return []
        thresh = min(np.percentile(finite_norms, 10),
                     0.1 * float(np.median(finite_norms)) + 1e-6)
        seeds = [np.array(grid_pts[i])
                 for i, v in enumerate(grad_norms)
                 if v <= thresh and np.isfinite(v)]
        return seeds

    def _newton_refine(self, seed: np.ndarray) -> Optional[np.ndarray]:
        """Newton refinement from a seed point.
        Uses scipy.fsolve when available, else a simple numpy Newton loop."""
        if _HAS_SCIPY:
            try:
                from scipy.optimize import fsolve
                sol, info, ier, _ = fsolve(
                    self._eval_grad, seed,
                    full_output=True, xtol=1e-12, ftol=1e-12
                )
                if ier == 1:
                    residual = np.max(np.abs(self._eval_grad(sol)))
                    if residual < self.grad_tol:
                        return sol
            except Exception:
                pass
            return None

        # Pure-numpy Newton fallback (finite-difference Jacobian)
        pt = seed.copy().astype(float)
        eps_fd = 1e-6
        for _ in range(50):
            g = self._eval_grad(pt)
            if np.max(np.abs(g)) < self.grad_tol:
                return pt
            # Build finite-difference Jacobian of ∇H
            n = len(pt)
            J = np.zeros((n, n))
            for j in range(n):
                dv = np.zeros(n); dv[j] = eps_fd
                J[:, j] = (self._eval_grad(pt + dv) - g) / eps_fd
            try:
                delta = np.linalg.solve(J, -g)
                pt = pt + delta
            except np.linalg.LinAlgError:
                break
        g_final = self._eval_grad(pt)
        if np.max(np.abs(g_final)) < self.grad_tol:
            return pt
        return None

    def _deduplicate(self, solutions: List[np.ndarray]) -> List[np.ndarray]:
        """
        Merge solutions closer than cluster_tol (greedy DBSCAN-style).
        Returns a list of representative points (cluster means).
        """
        if not solutions:
            return []
        clusters: List[List[np.ndarray]] = []
        for sol in solutions:
            placed = False
            for cluster in clusters:
                if np.linalg.norm(sol - cluster[0]) < self.cluster_tol:
                    cluster.append(sol)
                    placed = True
                    break
            if not placed:
                clusters.append([sol])
        return [np.mean(c, axis=0) for c in clusters]

    def _symbolic_solve(self) -> Optional[List[np.ndarray]]:
        """Attempt symbolic solution with sp.solve."""
        eqs = [sp.Eq(g, 0) for g in self._grad_exprs]
        try:
            sol_list = sp.solve(eqs, self.unknowns, dict=True)
            if not sol_list:
                return None
            result = []
            for sol in sol_list:
                try:
                    pt = np.array([float(sp.N(sol.get(v, v)))
                                   for v in self.unknowns], dtype=float)
                    if np.all(np.isfinite(pt)):
                        result.append(pt)
                except Exception:
                    continue
            return result if result else None
        except Exception:
            return None

    # ── Public API ───────────────────────────────────────────────

    def solve(self, method: str = "auto") -> List[np.ndarray]:
        """
        Find all critical points.

        Parameters
        ----------
        method : "symbolic" | "adaptive" | "auto"
            - "symbolic" : sp.solve only (fast for polynomials)
            - "adaptive" : coarse grid + Newton (always numerical)
            - "auto"     : try symbolic first, fall back to adaptive

        Returns
        -------
        list of np.ndarray, each of shape (dim,)
            Each array gives the values of (xi_vars + coords) at the
            critical point.
        """
        solutions = []

        if method in ("symbolic", "auto"):
            sym_sols = self._symbolic_solve()
            if sym_sols:
                solutions.extend(sym_sols)
                if method == "symbolic":
                    return self._deduplicate(solutions)

        if method in ("adaptive", "auto") and not solutions:
            seeds = self._coarse_seeds()
            refined = []
            for seed in seeds:
                pt = self._newton_refine(seed)
                if pt is not None:
                    refined.append(pt)
            solutions.extend(refined)

        return self._deduplicate(solutions)

    def to_dict_list(self, solutions: Optional[List[np.ndarray]] = None
                     ) -> List[Dict]:
        """
        Convert solutions to list of dicts {symbol: value}.
        If solutions is None, calls self.solve() first.
        """
        if solutions is None:
            solutions = self.solve()
        return [{v: float(pt[i]) for i, v in enumerate(self.unknowns)}
                for pt in solutions]


def detect_catastrophes(H_expr: sp.Expr,
                        xi_vars: Sequence[sp.Symbol],
                        coords: Optional[Sequence[sp.Symbol]] = None,
                        method: str = "auto",
                        bounds: Optional[Dict] = None,
                        coarse_n: int = 30,
                        max_order: int = 6,
                        tol: float = 1e-8) -> List[Dict]:
    """
    Detect and classify catastrophes of H_expr(xi_vars).

    Combines AdaptiveCriticalPointSolver (robust search) with
    classify_arnold_1d / classify_arnold_2d (typing).

    Parameters
    ----------
    H_expr : sp.Expr
    xi_vars : tuple of sp.Symbol   — (xi,) in 1D, (xi, eta) in 2D
    coords : tuple of sp.Symbol, optional
        Additional parameter symbols (e.g. spatial coordinates x, y).
        When provided, catastrophes are tracked as families.
    method : "symbolic" | "adaptive" | "auto"
    bounds : dict {symbol: (lo, hi)}, optional
    coarse_n : int    — coarse grid resolution per axis
    max_order : int   — max derivative order for 1D classification
    tol : float

    Returns
    -------
    list of dict, each with:
        'point'      : dict {symbol: value}
        'type'       : str   Arnold type
        'details'    : dict  hessian / derivatives / invariants
    """
    dim = len(xi_vars)
    if dim not in (1, 2):
        raise NotImplementedError("detect_catastrophes supports dim 1 or 2.")

    solver = AdaptiveCriticalPointSolver(
        H_expr, xi_vars, coords=coords,
        bounds=bounds, coarse_n=coarse_n, grad_tol=tol
    )
    raw_points = solver.to_dict_list()

    results = []
    for pt_dict in raw_points:
        if dim == 1:
            xi = xi_vars[0]
            xi_val = pt_dict.get(xi, None)
            if xi_val is None:
                continue
            classification = classify_arnold_1d(
                H_expr, xi, xi_val, max_order=max_order, tol=tol
            )
            results.append({
                "point"  : pt_dict,
                "type"   : classification["type"],
                "details": classification,
            })
        else:
            xi, eta = xi_vars[0], xi_vars[1]
            if xi not in pt_dict or eta not in pt_dict:
                continue
            classification = classify_arnold_2d(
                H_expr, xi, eta, pt_dict, tol=tol
            )
            results.append({
                "point"  : pt_dict,
                "type"   : classification["type"],
                "details": classification,
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Ray-based caustic detection (geometric, corrected)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CausticEvent:
    """
    A caustic crossing event along a ray.

    Attributes
    ----------
    ray_idx : int       Index of the ray in the bundle.
    time    : float     Integration time t* where det(J) = 0.
    time_idx: int       Index in the time array closest to t*.
    position: np.ndarray  Spatial position x(t*) (shape (dim,)).
    momentum: np.ndarray  Momentum ξ(t*)          (shape (dim,)).
    det_J   : float     Value of det(J) at t* (close to 0).
    sign_change: int    +1 or -1 (sign of d(det J)/dt at crossing).
    maslov_contribution: int   +1 (= π/2 phase shift).
    arnold_type: str    Classification from classify_arnold_* (if available).
    """
    ray_idx   : int
    time      : float
    time_idx  : int
    position  : np.ndarray
    momentum  : np.ndarray
    det_J     : float
    sign_change: int          = 0
    maslov_contribution: int  = 1
    arnold_type: str          = "unknown"


class RayCausticDetector:
    """
    Detect caustics in a ray bundle by tracking the stability matrix J.

    Mathematical background
    -----------------------
    Given a Hamiltonian H(x, ξ) and a family of rays parameterised by
    initial condition s ∈ ℝ^d, the stability matrix is

        J(t) = ∂x(t, s) / ∂s₀

    and satisfies the variational equation along the ray:

        dJ/dt = H_{px}(x(t), ξ(t)) · J

    where H_{px} = ∂²H/∂ξ∂x is the mixed Hessian (a d×d matrix in d-D).

    A caustic occurs at t* when det(J(t*)) = 0.
    The Maslov index μ counts the number of such crossings (with sign).

    Correct approach vs old CausticDetector
    ----------------------------------------
    OLD (wrong):
        1D: caustic ↔ dx/dt = 0   (velocity turning point, NOT a caustic)
        2D: caustic ↔ |ξ| ≈ 0     (small momentum, NOT a caustic)

    NEW (correct):
        1D: caustic ↔ J(t) = ∂x(t)/∂x₀ = 0  (requires integrating J)
        2D: caustic ↔ det([[∂x/∂x₀, ∂x/∂y₀],[∂y/∂x₀, ∂y/∂y₀]]) = 0

    Parameters
    ----------
    ray_bundle : list of dict
        Each dict is an integrated ray as returned by wkb.py:
        1D keys: 't', 'x', 'xi', 'S', 'a0', ...
        2D keys: 't', 'x', 'y', 'xi', 'eta', 'S', 'a0', ...
        Each ray must additionally contain the stability matrix entries:
        1D: 'J'          (array, shape (n_steps,))
        2D: 'J11','J12','J21','J22'  (arrays, shape (n_steps,))
        If J is absent, RayCausticDetector will attempt to compute it
        numerically from neighbouring rays if a full bundle is provided.
    dimension : int
        1 or 2.
    det_threshold : float
        |det(J)| < det_threshold to flag a caustic (default 0.05).
        Relative to the initial value |det(J(0))|.
    H_expr : sp.Expr, optional
        Symbolic Hamiltonian.  If provided, classify_arnold_* is called
        at each caustic to give the Arnold type.
    xi_syms : tuple of sp.Symbol, optional
        Momentum symbols matching H_expr.
    x_syms  : tuple of sp.Symbol, optional
        Spatial symbols matching H_expr.
    """

    def __init__(self,
                 ray_bundle: List[Dict],
                 dimension: int,
                 det_threshold: float = 0.05,
                 H_expr: Optional[sp.Expr] = None,
                 xi_syms: Optional[Tuple] = None,
                 x_syms : Optional[Tuple] = None):

        self.rays          = ray_bundle
        self.dimension     = dimension
        self.det_threshold = det_threshold
        self.H_expr        = H_expr
        self.xi_syms       = xi_syms
        self.x_syms        = x_syms
        self._events: List[CausticEvent] = []

    # ── Stability matrix ─────────────────────────────────────────

    def _det_J_1d(self, ray: Dict) -> np.ndarray:
        """
        Extract or estimate det(J) = J for a 1D ray.

        If 'J' key is present: use directly.
        Else: estimate by finite difference with a neighbouring ray
              if the bundle contains it, otherwise raise.
        """
        if 'J' in ray:
            return np.asarray(ray['J'], dtype=float)
        raise KeyError(
            "Ray dict is missing key 'J' (stability matrix). "
            "Integrate J alongside the ray ODE: dJ/dt = (∂²H/∂ξ∂x)·J, J(0)=1."
        )

    def _det_J_2d(self, ray: Dict) -> np.ndarray:
        """
        Extract or compute det(J) for a 2D ray.

        Expects keys 'J11','J12','J21','J22' in the ray dict.
        """
        if all(k in ray for k in ('J11', 'J12', 'J21', 'J22')):
            J11 = np.asarray(ray['J11'], dtype=float)
            J12 = np.asarray(ray['J12'], dtype=float)
            J21 = np.asarray(ray['J21'], dtype=float)
            J22 = np.asarray(ray['J22'], dtype=float)
            return J11 * J22 - J12 * J21
        raise KeyError(
            "Ray dict is missing stability matrix keys 'J11'..'J22'. "
            "Integrate the 2×2 matrix ODE dJ/dt = H_px · J alongside the ray."
        )

    @staticmethod
    def stability_matrix_ode_1d(H_px_func):
        """
        Return the RHS for the 1D stability ODE to be integrated with the ray.

        Usage in your ray integrator
        ----------------------------
        Add a state component J (scalar, initially 1) to the ray ODE.
        The extra equation is:

            dJ/dt = H_px(x(t), ξ(t)) · J

        where H_px = ∂²H/∂ξ∂x (scalar in 1D).

        Parameters
        ----------
        H_px_func : callable (x, xi) → float
            Lambdified ∂²H/∂ξ∂x.

        Returns
        -------
        callable (t, J_val, x_val, xi_val) → dJ/dt
        """
        def rhs(t, J_val, x_val, xi_val):
            return H_px_func(x_val, xi_val) * J_val
        return rhs

    @staticmethod
    def stability_matrix_ode_2d(H_px_func):
        """
        Return the RHS for the 2D stability ODE.

        The 2×2 matrix J satisfies  dJ/dt = H_{px} · J
        where H_{px} = [[∂²H/∂ξ∂x, ∂²H/∂ξ∂y],[∂²H/∂η∂x, ∂²H/∂η∂y]].

        Pack J as a flat vector [J11, J12, J21, J22] in the state.

        Parameters
        ----------
        H_px_func : callable (x, y, xi, eta) → 2×2 array
            Lambdified H_{px} matrix.

        Returns
        -------
        callable (t, J_flat, x, y, xi, eta) → dJ_flat/dt  (4-vector)
        """
        def rhs(t, J_flat, x, y, xi, eta):
            J = J_flat.reshape(2, 2)
            Hpx = np.asarray(H_px_func(x, y, xi, eta), dtype=float).reshape(2, 2)
            dJ = Hpx @ J
            return dJ.ravel()
        return rhs

    # ── Detection ────────────────────────────────────────────────

    def _find_sign_changes(self, arr: np.ndarray,
                           t_arr: np.ndarray,
                           threshold: float) -> List[Tuple[int, float]]:
        """
        Find indices where |arr| crosses below threshold
        and changes sign, indicating det(J) = 0.

        Returns list of (index, interpolated_time).
        """
        events = []
        norm0 = abs(arr[0]) if abs(arr[0]) > 1e-12 else 1.0
        scaled = arr / norm0

        for i in range(len(scaled) - 1):
            a, b = scaled[i], scaled[i + 1]
            if abs(a) < threshold or abs(b) < threshold:
                continue
            if a * b < 0:  # sign change
                # Linear interpolation for zero crossing time
                t_cross = t_arr[i] + (t_arr[i+1] - t_arr[i]) * abs(a) / (abs(a) + abs(b))
                events.append((i, t_cross))
        return events

    def detect(self) -> List[CausticEvent]:
        """
        Detect all caustic events in the ray bundle.

        Returns
        -------
        list of CausticEvent
        """
        self._events = []

        for ray_idx, ray in enumerate(self.rays):
            t = np.asarray(ray['t'], dtype=float)

            try:
                if self.dimension == 1:
                    det_J = self._det_J_1d(ray)
                    x_arr = np.asarray(ray['x'], dtype=float)
                    xi_arr = np.asarray(ray['xi'], dtype=float)

                    crossings = self._find_sign_changes(det_J, t, self.det_threshold)
                    for idx, t_cross in crossings:
                        sign = int(np.sign(det_J[idx + 1] - det_J[idx]))
                        ev = CausticEvent(
                            ray_idx   = ray_idx,
                            time      = t_cross,
                            time_idx  = idx,
                            position  = np.array([x_arr[idx]]),
                            momentum  = np.array([xi_arr[idx]]),
                            det_J     = float(det_J[idx]),
                            sign_change = sign,
                            maslov_contribution = 1,
                        )
                        if self.H_expr is not None and self.xi_syms:
                            ev.arnold_type = self._classify_at_event_1d(ev)
                        self._events.append(ev)

                else:  # dim == 2
                    det_J = self._det_J_2d(ray)
                    x_arr  = np.asarray(ray['x'],   dtype=float)
                    y_arr  = np.asarray(ray['y'],   dtype=float)
                    xi_arr = np.asarray(ray['xi'],  dtype=float)
                    eta_arr= np.asarray(ray['eta'], dtype=float)

                    crossings = self._find_sign_changes(det_J, t, self.det_threshold)
                    for idx, t_cross in crossings:
                        sign = int(np.sign(det_J[idx + 1] - det_J[idx]))
                        ev = CausticEvent(
                            ray_idx   = ray_idx,
                            time      = t_cross,
                            time_idx  = idx,
                            position  = np.array([x_arr[idx], y_arr[idx]]),
                            momentum  = np.array([xi_arr[idx], eta_arr[idx]]),
                            det_J     = float(det_J[idx]),
                            sign_change = sign,
                            maslov_contribution = 1,
                        )
                        if self.H_expr is not None and self.xi_syms:
                            ev.arnold_type = self._classify_at_event_2d(ev)
                        self._events.append(ev)

            except KeyError as e:
                warnings.warn(f"Ray {ray_idx}: {e}", UserWarning)
                continue

        return self._events

    def _classify_at_event_1d(self, ev: CausticEvent) -> str:
        """Call classify_arnold_1d at the caustic position."""
        if self.H_expr is None or not self.xi_syms:
            return "unknown"
        xi_sym = self.xi_syms[0]
        xi_val = float(ev.momentum[0])
        try:
            res = classify_arnold_1d(self.H_expr, xi_sym, xi_val)
            return res["type"]
        except Exception:
            return "classification failed"

    def _classify_at_event_2d(self, ev: CausticEvent) -> str:
        """Call classify_arnold_2d at the caustic position."""
        if self.H_expr is None or not self.xi_syms or len(self.xi_syms) < 2:
            return "unknown"
        xi_sym, eta_sym = self.xi_syms[0], self.xi_syms[1]
        pt = {"xi": float(ev.momentum[0]), "eta": float(ev.momentum[1])}
        try:
            res = classify_arnold_2d(self.H_expr, xi_sym, eta_sym, pt)
            return res["type"]
        except Exception:
            return "classification failed"

    # ── Maslov index ─────────────────────────────────────────────

    def maslov_index(self, ray_idx: int) -> int:
        """
        Return the Maslov index for ray ray_idx.

        The Maslov index is the signed count of caustic crossings:
            μ = Σ_k  sign(d det(J)/dt at t_k*)

        Each crossing contributes a phase shift of π/2 to the WKB amplitude.

        Parameters
        ----------
        ray_idx : int

        Returns
        -------
        int   (total signed count; use abs() for the number of crossings)
        """
        if not self._events:
            self.detect()
        return sum(ev.maslov_contribution * ev.sign_change
                   for ev in self._events if ev.ray_idx == ray_idx)

    def maslov_phase(self, ray_idx: int) -> float:
        """
        Total Maslov phase = μ · π/2 for ray ray_idx.
        """
        return self.maslov_index(ray_idx) * np.pi / 2

    def summary(self) -> Dict:
        """Return a summary dict of detected caustic events."""
        if not self._events:
            self.detect()
        by_type = {}
        for ev in self._events:
            by_type.setdefault(ev.arnold_type, []).append(ev)
        return {
            "n_events"   : len(self._events),
            "by_ray"     : {i: [e for e in self._events if e.ray_idx == i]
                            for i in set(e.ray_idx for e in self._events)},
            "by_type"    : by_type,
            "maslov_indices": {
                i: self.maslov_index(i)
                for i in set(e.ray_idx for e in self._events)
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Special functions for uniform corrections
# ══════════════════════════════════════════════════════════════════════════════

class CausticFunctions:
    """
    Special functions for uniform asymptotic corrections near caustics.

    All functions include the correct normalisation prefactors for the
    uniform WKB approximation (Maslov-Fedoriuk / Duistermaat).

    Fold (A2)   : Airy function Ai
    Cusp (A3)   : Pearcey integral P(x,y)
    Swallowtail (A4) : SW integral (3-parameter oscillatory)
    """

    # ── Fold : Airy ───────────────────────────────────────────────

    @staticmethod
    def airy_Ai(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Airy function Ai(z).

        Fold caustic uniform approximation:
            u(x) ≈ 2√π · ε^{1/6} · |∂_s J|^{-1/2} · Ai(-ε^{-2/3} ζ(x)) · e^{iS_c/ε}
        where ζ(x) is the local coordinate measuring distance to the caustic.
        """
        if not _HAS_SCIPY:
            raise RuntimeError("scipy required for Airy function.")
        ai, _, _, _ = airy(z)
        return ai

    @staticmethod
    def airy_Ai_prime(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Derivative Ai'(z) of the Airy function."""
        if not _HAS_SCIPY:
            raise RuntimeError("scipy required.")
        _, aip, _, _ = airy(z)
        return aip

    @staticmethod
    def airy_Bi(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Second Airy function Bi(z) (exponentially growing branch)."""
        if not _HAS_SCIPY:
            raise RuntimeError("scipy required.")
        _, _, bi, _ = airy(z)
        return bi

    @staticmethod
    def fold_uniform(x: np.ndarray,
                     x_c: float,
                     epsilon: float,
                     a_c: float,
                     S_c: float,
                     dJ_ds: float) -> np.ndarray:
        """
        Uniform approximation near a fold caustic (A2).

        Formula (Duistermaat 1974, eq. 2.4):
            u(x) = 2√π · ε^{1/6} · a_c · |dJ/ds|^{-1/2} · Ai(-ε^{-2/3} (x - x_c)) · exp(i S_c / ε)

        Parameters
        ----------
        x       : evaluation grid
        x_c     : caustic position
        epsilon : small parameter
        a_c     : WKB amplitude at caustic
        S_c     : phase at caustic
        dJ_ds   : ∂(det J)/∂s at the caustic (slope of det J vs parameter s)

        Returns
        -------
        np.ndarray of complex
        """
        zeta = -(x - x_c) / epsilon**(2/3)
        prefactor = (2 * np.sqrt(np.pi)
                     * epsilon**(1/6)
                     * a_c
                     / np.sqrt(abs(dJ_ds) + 1e-30))
        return prefactor * CausticFunctions.airy_Ai(zeta) * np.exp(1j * S_c / epsilon)

    # ── Cusp : Pearcey ───────────────────────────────────────────

    @staticmethod
    def pearcey(x: float, y: float,
                t_range: float = 6.0,
                n_pts: int = 500) -> complex:
        """
        Pearcey integral for cusp caustic (A3):

            P(x, y) = ∫_{-∞}^{∞} exp(i(t⁴ + x t² + y t)) dt

        Evaluated by adaptive quadrature on [-t_range, t_range].

        Parameters
        ----------
        x, y    : control parameters
        t_range : integration half-width (default 6.0; increase for large |x|,|y|)
        n_pts   : number of quadrature points

        Returns
        -------
        complex
        """
        t  = np.linspace(-t_range, t_range, n_pts)
        dt = t[1] - t[0]
        phase = t**4 + x * t**2 + y * t
        return complex(np.trapezoid(np.exp(1j * phase), dx=dt))

    @staticmethod
    def pearcey_grid(X: np.ndarray, Y: np.ndarray,
                     t_range: float = 6.0,
                     n_pts: int = 500) -> np.ndarray:
        """
        Vectorised Pearcey integral over 2D grids X, Y.
        Returns complex array of same shape as X.
        """
        shape = X.shape
        result = np.empty(shape, dtype=complex)
        for idx in np.ndindex(shape):
            result[idx] = CausticFunctions.pearcey(
                float(X[idx]), float(Y[idx]), t_range=t_range, n_pts=n_pts
            )
        return result

    @staticmethod
    def cusp_uniform(x: np.ndarray,
                     y: np.ndarray,
                     x_c: float,
                     y_c: float,
                     epsilon: float,
                     a_c: float,
                     S_c: float) -> np.ndarray:
        """
        Uniform approximation near a cusp caustic (A3).

        Scaled Pearcey:
            u(x,y) ≈ ε^{1/4} · a_c · P(ε^{-1/2}(x-x_c), ε^{-3/4}(y-y_c)) · exp(i S_c / ε)

        Returns
        -------
        np.ndarray of complex, same shape as x and y.
        """
        X_sc = (x - x_c) / epsilon**(1/2)
        Y_sc = (y - y_c) / epsilon**(3/4)
        P = CausticFunctions.pearcey_grid(X_sc, Y_sc)
        return epsilon**(1/4) * a_c * P * np.exp(1j * S_c / epsilon)

    # ── Swallowtail (A4) ─────────────────────────────────────────

    @staticmethod
    def swallowtail(x: float, y: float, z: float,
                    t_range: float = 5.0,
                    n_pts: int = 400) -> complex:
        """
        Swallowtail integral for A4 caustic:

            SW(x,y,z) = ∫_{-∞}^{∞} exp(i(t⁵ + x t³ + y t² + z t)) dt

        Parameters
        ----------
        x, y, z : control parameters
        t_range, n_pts : quadrature parameters

        Returns
        -------
        complex
        """
        t = np.linspace(-t_range, t_range, n_pts)
        dt = t[1] - t[0]
        phase = t**5 + x * t**3 + y * t**2 + z * t
        return complex(np.trapezoid(np.exp(1j * phase), dx=dt))

    # ── Phase shift from Maslov index ─────────────────────────────

    @staticmethod
    def maslov_phase_shift(mu: int) -> complex:
        """
        WKB phase correction factor from Maslov index μ.

        Each caustic crossing (each zero of det J) contributes a phase
        shift of π/2.  The total correction factor is exp(i μ π/2).

        Parameters
        ----------
        mu : int   Maslov index (number of signed caustic crossings).

        Returns
        -------
        complex   exp(i μ π/2)
        """
        return np.exp(1j * mu * np.pi / 2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_catastrophe(H: sp.Expr,
                     xi_vars: Sequence[sp.Symbol],
                     points: List[Dict],
                     xi_bounds: Tuple[float, float] = (-3.0, 3.0),
                     eta_bounds: Tuple[float, float] = (-3.0, 3.0),
                     n: int = 300,
                     title: Optional[str] = None) -> None:
    """
    Plot H and mark classified catastrophe points.

    Parameters
    ----------
    H        : sympy.Expr
    xi_vars  : (xi,) or (xi, eta)
    points   : list of dicts as returned by detect_catastrophes()
               each must have keys 'point' and 'type'
    xi_bounds, eta_bounds : axis ranges
    n        : grid resolution
    title    : optional plot title
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available.")

    dim = len(xi_vars)
    _TYPE_COLORS = {
        "A2 (Fold)"           : "red",
        "A3 (Cusp)"           : "orange",
        "A4 (Swallowtail)"    : "purple",
        "A5 (Butterfly)"      : "blue",
        "D4+ (Hyperbolic umbilic)": "green",
        "D4- (Elliptic umbilic)"  : "cyan",
    }

    if dim == 1:
        xi = xi_vars[0]
        X  = np.linspace(xi_bounds[0], xi_bounds[1], n)
        Hf = sp.lambdify(xi, H, "numpy")
        Y  = np.asarray(Hf(X), dtype=float)

        fig, ax = _plt.subplots(figsize=(9, 5))
        ax.plot(X, Y, "k-", lw=1.5, label="H(ξ)")

        for p in points:
            pt  = p["point"]
            typ = p["type"]
            xv  = float(pt.get(xi, 0.0))
            yv  = float(Hf(xv))
            col = _TYPE_COLORS.get(typ, "gray")
            ax.scatter([xv], [yv], color=col, s=80, zorder=5)
            ax.annotate(typ, (xv, yv), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=col)

        ax.set_xlabel("ξ")
        ax.set_ylabel("H(ξ)")
        ax.set_title(title or "Catastrophe plot (1D)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _plt.tight_layout()
        _plt.show()

    elif dim == 2:
        xi, eta = xi_vars
        Xv = np.linspace(xi_bounds[0],  xi_bounds[1],  n // 2)
        Yv = np.linspace(eta_bounds[0], eta_bounds[1], n // 2)
        XX, YY = np.meshgrid(Xv, Yv)
        Hf = sp.lambdify((xi, eta), H, "numpy")
        ZZ = np.asarray(Hf(XX, YY), dtype=float)

        fig = _plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot_surface(XX, YY, ZZ, alpha=0.55, rstride=3, cstride=3,
                        cmap="viridis")

        for p in points:
            pt  = p["point"]
            typ = p["type"]
            xv  = float(pt.get(xi,  0.0))
            yv  = float(pt.get(eta, 0.0))
            zv  = float(Hf(xv, yv))
            col = _TYPE_COLORS.get(typ, "gray")
            ax.scatter([xv], [yv], [zv], color=col, s=80, zorder=5)
            ax.text(xv, yv, zv + 0.1, typ, fontsize=7, color=col)

        ax.set_xlabel("ξ")
        ax.set_ylabel("η")
        ax.set_zlabel("H")
        ax.set_title(title or "Catastrophe surface (2D)")
        _plt.tight_layout()
        _plt.show()

    else:
        raise NotImplementedError("plot_catastrophe supports only dim 1 or 2.")


def plot_caustic_events(ray_bundle: List[Dict],
                        events: List[CausticEvent],
                        dimension: int,
                        n_rays_plot: int = 20,
                        title: Optional[str] = None) -> None:
    """
    Overlay caustic events on the ray bundle plot.

    Parameters
    ----------
    ray_bundle : list of ray dicts (with 'x', 't' etc.)
    events     : list of CausticEvent from RayCausticDetector.detect()
    dimension  : 1 or 2
    n_rays_plot: max number of rays to draw
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available.")

    fig, ax = _plt.subplots(figsize=(10, 6))

    n_total = len(ray_bundle)
    step    = max(1, n_total // n_rays_plot)
    indices = range(0, n_total, step)

    _TYPE_COLORS = {
        "A2 (Fold)"        : "red",
        "A3 (Cusp)"        : "orange",
        "A4 (Swallowtail)" : "purple",
        "A5 (Butterfly)"   : "blue",
    }

    if dimension == 1:
        for i in indices:
            ray = ray_bundle[i]
            ax.plot(ray['t'], ray['x'], 'b-', alpha=0.3, lw=0.8)

        for ev in events:
            col = _TYPE_COLORS.get(ev.arnold_type, "red")
            ax.scatter([ev.time], [ev.position[0]],
                       color=col, s=60, zorder=5,
                       label=ev.arnold_type)

        ax.set_xlabel("t")
        ax.set_ylabel("x(t)")

    else:  # 2D
        for i in indices:
            ray = ray_bundle[i]
            ax.plot(ray['x'], ray['y'], 'b-', alpha=0.3, lw=0.8)

        for ev in events:
            col = _TYPE_COLORS.get(ev.arnold_type, "red")
            ax.scatter([ev.position[0]], [ev.position[1]],
                       color=col, s=60, zorder=5,
                       label=ev.arnold_type)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=8)

    ax.set_title(title or f"Ray bundle with caustic events (dim={dimension})")
    ax.grid(True, alpha=0.3)
    _plt.tight_layout()
    _plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Self-test
# ══════════════════════════════════════════════════════════════════════════════

def _run_tests(verbose: bool = True) -> Dict[str, bool]:
    """
    Quick self-test covering classification and adaptive solver.

    Tests
    -----
    T1  1D fold        f = ξ³             → A2 (Fold)
    T2  1D cusp        f = ξ⁴             → A3 (Cusp)
    T3  1D swallowtail f = ξ⁵             → A4 (Swallowtail)
    T4  2D Morse       H = ξ² + η²        → Morse
    T5  2D fold        H = ξ³ + η²        → A2
    T6  2D cusp        H = ξ⁴ + η²        → A3
    T7  2D hyp.umbilic H = ξ³ + η³        → D4+
    T8  2D ell.umbilic H = ξ³ - 3ξη²     → D4-
    T9  Adaptive solver on f = ξ⁴ - ξ²   → two critical points
    """
    xi_s  = sp.Symbol('xi',  real=True)
    eta_s = sp.Symbol('eta', real=True)

    results = {}

    def _check(label, got, expected_substr):
        ok = expected_substr.lower() in got.lower()
        results[label] = ok
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}]  {label:35s}  got: {got}")
        return ok

    if verbose:
        print("\n" + "="*60)
        print("  caustics.py — self-test")
        print("="*60)

    # T1–T3 : 1D Arnold
    for label, f_expr, expected in [
        ("T1 1D A2 (Fold)",        xi_s**3,      "A2"),
        ("T2 1D A3 (Cusp)",        xi_s**4,      "A3"),
        ("T3 1D A4 (Swallowtail)", xi_s**5,      "A4"),
    ]:
        res = classify_arnold_1d(f_expr, xi_s, 0.0)
        _check(label, res["type"], expected)

    # T4–T8 : 2D Arnold
    for label, H_expr, expected in [
        ("T4 2D Morse",            xi_s**2 + eta_s**2,           "Morse"),
        ("T5 2D A2 (Fold)",        xi_s**3 + eta_s**2,           "A2"),
        ("T6 2D A3 (Cusp)",        xi_s**4 + eta_s**2,           "A3"),
        ("T7 2D D4+ (Hyp umbilic)",xi_s**3 + eta_s**3,           "D4+"),
        ("T8 2D D4- (Ell umbilic)",xi_s**3 - 3*xi_s*eta_s**2,   "D4-"),
    ]:
        pt = {"xi": 0.0, "eta": 0.0}
        res = classify_arnold_2d(H_expr, xi_s, eta_s, pt)
        _check(label, res["type"], expected)

    # T9 : Adaptive solver — use 'auto' to try symbolic first
    f9 = xi_s**4 - xi_s**2   # critical points at ξ=0, ±1/√2
    solver9 = AdaptiveCriticalPointSolver(f9, [xi_s],
                                          bounds={xi_s: (-2.0, 2.0)},
                                          coarse_n=40)
    sols9 = solver9.solve(method="auto")
    ok9 = len(sols9) >= 2
    results["T9 Adaptive solver (2+ points)"] = ok9
    if verbose:
        status = "PASS" if ok9 else "FAIL"
        pts_str = [f"{s[0]:.3f}" for s in sols9]
        print(f"  [{status}]  {'T9 Adaptive solver (2+ points)':35s}"
              f"  found {len(sols9)} pts: {pts_str}")

    # Summary
    n_pass = sum(results.values())
    n_total = len(results)
    if verbose:
        print(f"\n  {n_pass}/{n_total} tests passed\n")

    return results


if __name__ == "__main__":
    _run_tests(verbose=True)