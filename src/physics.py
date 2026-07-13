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
physics.py — Lagrangian and Hamiltonian toolkit: Legendre transforms & symbolic PDEs
============================================================================================

Overview
--------
The `physics` module provides a unified interface for transforming between Lagrangian and Hamiltonian descriptions of physical systems, both symbolically and numerically.  It implements the classical Legendre transform as well as the more general **Legendre–Fenchel transform** (convex conjugate) for non‑quadratic or even non‑smooth Lagrangians.  The module also bridges to pseudo‑differential operator formalisms by decomposing a Hamiltonian into its local (polynomial) and non‑local parts and generating formal PDEs of Schrödinger, wave, or stationary type.

Key features include:

* Conversion from Lagrangian `L(x, u, p)` to Hamiltonian `H(x, u, ξ)` (and vice‑versa) using:
    * Symbolic Legendre transform (fast path for quadratic Lagrangians, general `solve`‑based inversion).
    * Symbolic Legendre–Fenchel transform (for convex, possibly non‑invertible relations).
    * Numeric Legendre–Fenchel transform (grid‑based or SciPy optimisation) for 1D and 2D problems.
* Decomposition of a Hamiltonian into polynomial and non‑polynomial (non‑local) parts – a heuristic useful for identifying the principal symbol and lower‑order terms.
* Generation of formal symbolic PDEs using a placeholder `ψOp` to represent the action of a pseudo‑differential operator; supports stationary (eigenvalue), Schrödinger, and wave equations.

Mathematical background
-----------------------
In classical mechanics, the **Lagrangian** `L(x, u, p)` depends on position `x`, the field `u` (optional), and the generalised velocities `p` (often denoted `ẋ`).  The conjugate momenta are defined as

    ξ = ∂L/∂p.

When this relation can be inverted (i.e. the Hessian `∂²L/∂p²` is non‑singular), the **Hamiltonian** is obtained by the **Legendre transform**

    H(x, u, ξ) = ξ·p − L(x, u, p), with p expressed in terms of ξ.

If the Hessian is singular, or if `L` is not strictly convex, the transform must be replaced by the **Legendre–Fenchel conjugate**

    H(x, u, ξ) = sup_p ( ξ·p − L(x, u, p) ),

which always yields a convex (in ξ) function.  This module provides both symbolic and numerical evaluations of this supremum.

In the context of pseudo‑differential operators, a Hamiltonian `H(x, ξ)` can be seen as the symbol of an operator.  The module splits `H` into a **polynomial part** (local operator) and a **non‑polynomial part** (non‑local, e.g. containing `√(1+ξ²)`).  This decomposition guides the construction of the corresponding PDE:

    ψOp(H, u) = E u                     (stationary),
    
    i ∂_t u = ψOp(H, u)                  (Schrödinger),
    
    ∂_tt u + ψOp(H, u) = 0                (wave),

where `ψOp` is a placeholder for the pseudo‑differential operator with symbol `H`.


References
----------
.. [1] Arnold, V. I.  *Mathematical Methods of Classical Mechanics*, Springer‑Verlag, 1989 (2nd ed.).  §14: Legendre Transform.
.. [2] Rockafellar, R. T.  *Convex Analysis*, Princeton University Press, 1970.  Chapter 12: Conjugate Functions.
.. [3] Evans, L. C.  *Partial Differential Equations*, American Mathematical Society, 2010 (2nd ed.).  §4.3: Hamilton–Jacobi Equations.
.. [4] Folland, G. B.  *Quantum Field Theory: A Tourist Guide for Mathematicians*, American Mathematical Society, 2008.  §1: Legendre Transform and Quantisation.
"""

import math as _math
import sympy as sp
from sympy import Matrix
import numpy as _np

# Optional SciPy for numeric optimization
try:
    from scipy import optimize as _optimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Optional plotting
try:
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# --------------------------
# Lagrangian <-> Hamiltonian
# --------------------------
class LagrangianHamiltonianConverter:
    """
    Bidirectional converter between Lagrangian and Hamiltonian (Legendre transform),
    with optional Legendre–Fenchel (convex conjugate) support and robust numeric fallback.

    Main API:
      L_to_H(L_expr, coords, u, p_vars, return_symbol_only=False, force=False,
             method="legendre", fenchel_opts=None)

        - method: "legendre" (default), "fenchel_symbolic", "fenchel_numeric"
        - If method == "fenchel_numeric" returns (H_repr, xi_vars, numeric_callable)
          otherwise returns (H_expr, xi_vars)
    """

    _numeric_cache = {}

    # --------------------
    # Utilities
    # --------------------
    @staticmethod
    def _is_quadratic_in_p(L_expr, p_vars):
        """
        Robust test: returns True only if L_expr is polynomial of degree ≤ 2 in each p_var.
        Falls back to False for non-polynomial expressions (Abs, sqrt, etc.).
        """
        for p in p_vars:
            # Quick test: is L polynomial in p?
            if not L_expr.is_polynomial(p):
                return False
            try:
                deg = sp.degree(L_expr, p)
            except Exception:
                return False
            if deg is None or deg > 2:
                return False
        return True

    @staticmethod
    def _quadratic_legendre(L_expr, p_vars, xi_vars):
        """
        Analytic Legendre transform for quadratic L: L = 1/2 p^T A p + b^T p + c
        Returns (H_expr, sol_map) and raises ValueError if Hessian singular.
        """
        A = Matrix([[sp.diff(sp.diff(L_expr, p_i), p_j) for p_j in p_vars] for p_i in p_vars])
        grad = Matrix([sp.diff(L_expr, p) for p in p_vars])
        try:
            A_inv = A.inv()
        except Exception:
            raise ValueError("Quadratic analytic path: Hessian A is singular (non-invertible).")
        subs_zero = {p: 0 for p in p_vars}
        b_vec = grad.subs(subs_zero)
        xi_vec = Matrix(xi_vars)
        p_solution_vec = A_inv * (xi_vec - b_vec)
        sol = {p_vars[i]: sp.simplify(p_solution_vec[i]) for i in range(len(p_vars))}
        H_expr = sum(xi_vars[i] * sol[p_vars[i]] for i in range(len(p_vars))) - sp.simplify(L_expr.subs(sol))
        return sp.simplify(H_expr), sol

    # ----------------------------
    # Numeric Legendre-Fenchel helpers
    # ----------------------------
    @staticmethod
    def _legendre_fenchel_1d_numeric_callable(L_func, p_bounds=(-10.0, 10.0), n_grid=2001, mode="auto",
                                             scipy_multistart=5):
        """
        Return a callable H_numeric(xi) = sup_p (xi*p - L(p)) for 1D L_func(p).
        - L_func: callable p -> L(p)
        - mode: "auto" | "scipy" | "grid"
        """
        pmin, pmax = float(p_bounds[0]), float(p_bounds[1])

        def _compute_by_grid(xi):
            grid = _np.linspace(pmin, pmax, int(n_grid))
            Lvals = _np.array([float(L_func(p)) for p in grid], dtype=float)
            S = xi * grid - Lvals
            idx = int(_np.argmax(S))
            return float(S[idx]), float(grid[idx])

        def _compute_by_scipy(xi):
            if not _HAS_SCIPY:
                return _compute_by_grid(xi)

            def negS(p):
                p0 = float(p[0])
                return -(xi * p0 - float(L_func(p0)))

            best_val = -_math.inf
            best_p = None
            inits = _np.linspace(pmin, pmax, max(3, int(scipy_multistart)))
            for x0 in inits:
                try:
                    res = _optimize.minimize(negS, x0=[float(x0)], bounds=[(pmin, pmax)], method="L-BFGS-B")
                    if res.success:
                        pstar = float(res.x[0])
                        sval = float(xi * pstar - float(L_func(pstar)))
                        if sval > best_val:
                            best_val = sval
                            best_p = pstar
                except Exception:
                    continue
            if best_p is None:
                return _compute_by_grid(xi)
            return best_val, best_p

        compute = _compute_by_scipy if (_HAS_SCIPY and mode != "grid") else _compute_by_grid

        def H_numeric(xi_in):
            xi_arr = _np.atleast_1d(xi_in).astype(float)
            out = _np.empty_like(xi_arr, dtype=float)
            for i, xi in enumerate(xi_arr):
                val, _ = compute(float(xi))
                out[i] = val
            if _np.isscalar(xi_in):
                return float(out[0])
            return out

        return H_numeric

    @staticmethod
    def _legendre_fenchel_nd_numeric_callable(L_func, dim, p_bounds, n_grid_per_dim=41, mode="auto",
                                              scipy_multistart=10, multistart_restarts=8):
        """
        Return callable H_numeric(xi_vector) approximating sup_p (xi·p - L(p)) for dim>=2.
        - L_func: callable p_vector -> L(p)
        - p_bounds: tuple/list of per-dimension bounds
        """
        pmin_list, pmax_list = p_bounds
        pmin = [float(v) for v in pmin_list]
        pmax = [float(v) for v in pmax_list]

        def compute_by_grid(xi_vec):
            import itertools
            grids = [_np.linspace(pmin[d], pmax[d], int(n_grid_per_dim)) for d in range(dim)]
            best = -_math.inf
            best_p = None
            for pt in itertools.product(*grids):
                pt_arr = _np.array(pt, dtype=float)
                sval = float(_np.dot(xi_vec, pt_arr) - L_func(pt_arr))
                if sval > best:
                    best = sval
                    best_p = pt_arr
            return best, best_p

        def compute_by_scipy(xi_vec):
            if not _HAS_SCIPY:
                return compute_by_grid(xi_vec)

            def negS(p):
                p = _np.asarray(p, dtype=float)
                return - (float(_np.dot(xi_vec, p)) - float(L_func(p)))

            best_val = -_math.inf
            best_p = None
            center = _np.array([(pmin[d] + pmax[d]) / 2.0 for d in range(dim)], dtype=float)
            rng = _np.random.default_rng(123456)
            inits = [center]
            for k in range(multistart_restarts):
                r = rng.random(dim)
                start = _np.array([pmin[d] + r[d] * (pmax[d] - pmin[d]) for d in range(dim)], dtype=float)
                inits.append(start)
            for x0 in inits:
                try:
                    res = _optimize.minimize(negS, x0=x0, bounds=tuple((pmin[d], pmax[d]) for d in range(dim)),
                                             method="L-BFGS-B")
                    if res.success:
                        pstar = _np.asarray(res.x, dtype=float)
                        sval = float(_np.dot(xi_vec, pstar) - L_func(pstar))
                        if sval > best_val:
                            best_val = sval
                            best_p = pstar
                except Exception:
                    continue
            if best_p is None:
                return compute_by_grid(xi_vec)
            return best_val, best_p

        compute = compute_by_scipy if (_HAS_SCIPY and mode != "grid") else compute_by_grid

        def H_numeric(xi_in):
            xi_arr = _np.atleast_2d(xi_in).astype(float)
            if xi_arr.shape[-1] != dim:
                xi_arr = xi_arr.reshape(-1, dim)
            out = _np.empty((xi_arr.shape[0],), dtype=float)
            for i, xivec in enumerate(xi_arr):
                val, _ = compute(xivec)
                out[i] = val
            if out.shape[0] == 1:
                return float(out[0])
            return out

        return H_numeric

    # ----------------------------
    # Main methods
    # ----------------------------
    @staticmethod
    def L_to_H(L_expr, coords, u, p_vars, return_symbol_only=False, force=False,
               method="legendre", fenchel_opts=None):
        """
        Convert L(x,u,p) -> H(x,u,xi) with options for generalized Legendre (Fenchel).

        Parameters:
          - method: "legendre" (default), "fenchel_symbolic", "fenchel_numeric"
          - fenchel_opts: dict with options for numeric fenchel
        """
        dim = len(coords)
        if dim == 1:
            xi_vars = (sp.Symbol('xi', real=True),)
        elif dim == 2:
            xi_vars = (sp.Symbol('xi', real=True), sp.Symbol('eta', real=True))
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")

        # Quadratic fast-path (symbolic)
        if method in ("legendre", "fenchel_symbolic") and LagrangianHamiltonianConverter._is_quadratic_in_p(L_expr, p_vars):
            try:
                H_expr, sol = LagrangianHamiltonianConverter._quadratic_legendre(L_expr, p_vars, xi_vars)
                if return_symbol_only:
                    H_expr = H_expr.subs(u, 0)
                return H_expr, xi_vars
            except Exception:
                if not force and method == "legendre":
                    raise

        # CLASSICAL LEGENDRE
        if method == "legendre":
            H_p = None
            try:
                H_p = sp.hessian(L_expr, p_vars)
                det_H = sp.simplify(H_p.det())
            except Exception:
                det_H = None

            if det_H is not None and det_H == 0 and not force:
                raise ValueError("Legendre transform not invertible: Hessian singular. Use force=True or Fenchel method.")
            if det_H is None and not force:
                raise ValueError("Unable to verify Hessian determinant symbolically. Use force=True to attempt solve().")

            eqs = [sp.Eq(sp.diff(L_expr, p_vars[i]), xi_vars[i]) for i in range(dim)]
            sol_list = sp.solve(eqs, p_vars, dict=True)
            if not sol_list:
                if not force:
                    raise ValueError("Unable to solve symbolic Legendre relations. Use force=True or Fenchel fallback.")
            if sol_list:
                sol = sol_list[0]
                if isinstance(sol, tuple) and len(sol) == len(p_vars):
                    sol = {p_vars[i]: sol[i] for i in range(len(p_vars))}
                H_expr = sum(xi_vars[i]*sol[p_vars[i]] for i in range(dim)) - L_expr.subs(sol)
                H_expr = sp.simplify(H_expr)
                if return_symbol_only:
                    H_expr = H_expr.subs(u, 0)
                return H_expr, xi_vars
            raise ValueError("Legendre inversion failed even with solve().")

        # FENCHEL: symbolic attempt
        # -----------------------------------------------------
        #  Prevent symbolic Fenchel when L is non-differentiable
        # -----------------------------------------------------
        if method == "fenchel_symbolic":
            if L_expr.has(sp.Abs) or L_expr.has(sp.sign) or any(
                sp.diff(L_expr, p).has(sp.sign, sp.Abs) for p in p_vars
            ):
                raise ValueError(
                    "Symbolic Fenchel not possible for nonsmooth L (Abs, sign). "
                    "Use method='fenchel_numeric' instead."
                )

        if method == "fenchel_symbolic":
            eqs = [sp.Eq(sp.diff(L_expr, p_vars[i]), xi_vars[i]) for i in range(dim)]
            sol_list = sp.solve(eqs, p_vars, dict=True)
            if sol_list:
                candidates = []
                for sol in sol_list:
                    if isinstance(sol, tuple) and len(sol) == len(p_vars):
                        sol = {p_vars[i]: sol[i] for i in range(len(p_vars))}
                    S_expr = sum(xi_vars[i] * sol[p_vars[i]] for i in range(dim)) - L_expr.subs(sol)
                    candidates.append(sp.simplify(S_expr))
                H_candidates = sp.simplify(sp.Max(*candidates)) if len(candidates) > 1 else candidates[0]
                if return_symbol_only:
                    H_candidates = H_candidates.subs(u, 0)
                return H_candidates, xi_vars
            raise ValueError("Symbolic Fenchel conjugate not found; use method='fenchel_numeric' for numeric computation.")

        # FENCHEL: numeric path
        if method == "fenchel_numeric":
            if fenchel_opts is None:
                fenchel_opts = {}
            if dim == 1:
                p_bounds = fenchel_opts.get("p_bounds", (-10.0, 10.0))
                n_grid = int(fenchel_opts.get("n_grid", 2001))
                mode = fenchel_opts.get("mode", "auto")
                scipy_multistart = int(fenchel_opts.get("scipy_multistart", 8))

                # Build numeric L_func (try lambdify)
                try:
                    f_lamb = sp.lambdify((p_vars[0],), L_expr, "numpy")
                    def L_func_scalar(p):
                        return float(f_lamb(p))
                except Exception:
                    try:
                        f_lamb = sp.lambdify(p_vars[0], L_expr, "numpy")
                        def L_func_scalar(p):
                            return float(f_lamb(p))
                    except Exception:
                        def L_func_scalar(p):
                            return float(sp.N(L_expr.subs({p_vars[0]: p})))

                H_numeric = LagrangianHamiltonianConverter._legendre_fenchel_1d_numeric_callable(
                    L_func_scalar, p_bounds=p_bounds, n_grid=n_grid, mode=mode,
                    scipy_multistart=scipy_multistart
                )
                H_func = sp.Function("H_numeric")
                H_repr = H_func(xi_vars[0])
                LagrangianHamiltonianConverter._numeric_cache[id(H_repr)] = H_numeric
                return H_repr, xi_vars, H_numeric

            else:
                # dim == 2
                p_bounds = fenchel_opts.get("p_bounds", [(-10.0, 10.0), (-10.0, 10.0)])
                n_grid_per_dim = int(fenchel_opts.get("n_grid_per_dim", 41))
                mode = fenchel_opts.get("mode", "auto")
                scipy_multistart = int(fenchel_opts.get("scipy_multistart", 20))
                multistart_restarts = int(fenchel_opts.get("multistart_restarts", 8))

                f_lamb = None
                try:
                    f_lamb = sp.lambdify((p_vars[0], p_vars[1]), L_expr, "numpy")
                    def L_func_nd(p):
                        return float(f_lamb(float(p[0]), float(p[1])))
                except Exception:
                    try:
                        f_lamb = sp.lambdify((p_vars,), L_expr, "numpy")
                        def L_func_nd(p):
                            return float(f_lamb(tuple(float(v) for v in p)))
                    except Exception:
                        def L_func_nd(p):
                            subs_map = {p_vars[i]: float(p[i]) for i in range(2)}
                            return float(sp.N(L_expr.subs(subs_map)))

                H_numeric = LagrangianHamiltonianConverter._legendre_fenchel_nd_numeric_callable(
                    L_func_nd, dim=2, p_bounds=(p_bounds[0], p_bounds[1]),
                    n_grid_per_dim=n_grid_per_dim, mode=mode,
                    scipy_multistart=scipy_multistart, multistart_restarts=multistart_restarts
                )
                H_func = sp.Function("H_numeric")
                H_repr = H_func(*xi_vars)
                LagrangianHamiltonianConverter._numeric_cache[id(H_repr)] = H_numeric
                return H_repr, xi_vars, H_numeric

        raise ValueError("Unknown method '{}'. Choose 'legendre', 'fenchel_symbolic' or 'fenchel_numeric'.".format(method))

    @staticmethod
    def H_to_L(H_expr, coords, u, xi_vars, force=False):
        """
        Inverse Legendre (classical). Does not attempt Fenchel inverse.
        """
        dim = len(coords)
        if dim == 1:
            p_vars = (sp.Symbol('p', real=True),)
        elif dim == 2:
            p_vars = (sp.Symbol('p_x', real=True), sp.Symbol('p_y', real=True))
        else:
            raise ValueError("Only 1D and 2D are supported.")

        eqs = [sp.Eq(sp.diff(H_expr, xi_vars[i]), p_vars[i]) for i in range(dim)]
        sol = sp.solve(eqs, xi_vars, dict=True)
        if not sol:
            if not force:
                raise ValueError("Unable to symbolically solve p = ∂H/∂ξ for ξ. Use force=True.")
            sol = sp.solve(eqs, xi_vars)
        if not sol:
            raise ValueError("Inverse Legendre transform failed; cannot find ξ(p).")
        sol = sol[0] if isinstance(sol, list) else sol
        if isinstance(sol, tuple) and len(sol) == len(xi_vars):
            sol = {xi_vars[i]: sol[i] for i in range(len(xi_vars))}
        if not isinstance(sol, dict):
            if isinstance(sol, list) and sol and isinstance(sol[0], dict):
                sol = sol[0]
            else:
                raise ValueError("Unexpected output from solve(); cannot construct ξ(p).")
        L_expr = sum(sol[xi_vars[i]] * p_vars[i] for i in range(dim)) - H_expr.subs(sol)
        return sp.simplify(L_expr), p_vars


# ---------------------------------------------------------------------------
# HamiltonianSymbolicConverter
# ---------------------------------------------------------------------------
class HamiltonianSymbolicConverter:
    """
    Symbolic converter between Hamiltonians and formal PDEs (psiOp).
    """

    @staticmethod
    def decompose_hamiltonian(H_expr, xi_vars):
        """
        Decomposes the Hamiltonian into polynomial (local) and non-polynomial (nonlocal) parts.
        The heuristic treats terms containing sqrt, Abs, or sign as nonlocal.
        """
        xi = xi_vars if isinstance(xi_vars, (tuple, list)) else (xi_vars,)
        poly_terms, nonlocal_terms = 0, 0
        H_expand = sp.expand(H_expr)
        for term in H_expand.as_ordered_terms():
            # Heuristic: treat terms containing sqrt/Abs/sign as nonlocal explicitly
            # Check if the *current* 'term' (from the outer loop) has these functions.
            # The original code had a scoping bug in the 'any' statement.
            if any(func in term.free_symbols for func in [sp.sqrt, sp.Abs, sp.sign]) or \
               term.has(sp.sqrt) or term.has(sp.Abs) or term.has(sp.sign):
                # Alternative and more robust check:
                # This checks if the specific 'term' object contains the specified functions.
                nonlocal_terms += term
            elif all(term.is_polynomial(xi_i) for xi_i in xi):
                poly_terms += term
            else:
                nonlocal_terms += term
        return sp.simplify(poly_terms), sp.simplify(nonlocal_terms)

    @classmethod
    def hamiltonian_to_symbolic_pde(cls, H_expr, coords, t, u, mode="schrodinger"):
        dim = len(coords)
        if dim == 1:
            xi_vars = (sp.Symbol("xi", real=True),)
        elif dim == 2:
            xi_vars = (sp.Symbol("xi", real=True), sp.Symbol("eta", real=True))
        else:
            raise ValueError("Only 1D and 2D Hamiltonians are supported.")

        H_poly, H_nonlocal = cls.decompose_hamiltonian(H_expr, xi_vars)
        H_total = H_poly + H_nonlocal
        psiOp_H_u = sp.Function("psiOp")(H_total, u)

        if mode == "stationary":
            E = sp.Symbol("E", real=True)
            pde = sp.Eq(psiOp_H_u, E * u)
            formal = "ψOp(H, u) = E u"
        elif mode == "heat":
            pde = sp.Eq(sp.Derivative(u, t), -psiOp_H_u)
            formal = "∂_t u = -ψOp(H, u)"
        elif mode == "schrodinger":
            pde = sp.Eq(sp.I * sp.Derivative(u, t), psiOp_H_u)
            formal = "i ∂_t u = ψOp(H, u)"
        elif mode == "wave":
            pde = sp.Eq(sp.Derivative(u, (t, 2)), -psiOp_H_u)
            formal = "∂_{tt} u + ψOp(H, u) = 0"
        else:
            raise ValueError("mode must be one of: 'stationary', 'heat', 'schrodinger' or 'wave'.")

        coord_str = ", ".join(str(c) for c in coords)
        xi_str = ", ".join(str(x) for x in xi_vars)
        formal += f"   (H = H({coord_str}; {xi_str}))"

        return {
            "pde": sp.simplify(pde),
            "H_poly": H_poly,
            "H_nonlocal": H_nonlocal,
            "formal_string": formal,
            "mode": mode
        }
