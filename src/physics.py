"""
Toolkit for Lagrangian and Hamiltonian manipulation and Catastrophe analysis

Physics utilities:
 - Lagrangian <-> Hamiltonian converter with Legendre & Legendre-Fenchel (numeric + symbolic)
 - HamiltonianSymbolicConverter (formal psiOp PDE generation)
 - Catastrophe detection (1D & 2D) and Arnold classification for 2D
 - Plotting helper for catastrophes

Notes: Use L_to_H(..., method="fenchel_numeric", fenchel_opts=...) for numeric Fenchel.
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
        elif mode == "schrodinger":
            pde = sp.Eq(sp.I * sp.Derivative(u, t), psiOp_H_u)
            formal = "i ∂_t u = ψOp(H, u)"
        elif mode == "wave":
            pde = sp.Eq(sp.Derivative(u, (t, 2)), -psiOp_H_u)
            formal = "∂_{tt} u + ψOp(H, u) = 0"
        else:
            raise ValueError("mode must be one of: 'stationary', 'schrodinger', 'wave'.")

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


# --------------------------
# Catastrophe detector (1D & 2D) + numeric fallback
# --------------------------
def _try_symbolic_solve(eqs, unknowns):
    try:
        sol = sp.solve(eqs, unknowns, dict=True)
        if sol:
            return sol
        return None
    except Exception:
        return None


def _numeric_solve_system(eqs, unknowns, guesses):
    """
    Try nsolve with several initial guesses.
    Returns list of numeric solution dicts.
    """
    sol_set = []
    for g in guesses:
        try:
            solv = sp.nsolve(eqs, unknowns, g, tol=1e-12, maxsteps=50)
            vals = [float(v) for v in sp.Matrix(solv)]
            mapping = {unknowns[i]: vals[i] for i in range(len(unknowns))}
            # deduplicate approx
            already = False
            for s in sol_set:
                if all(abs(float(s[k]) - mapping[k]) < 1e-6 for k in unknowns):
                    already = True
                    break
            if not already:
                sol_set.append(mapping)
        except Exception:
            continue
    return sol_set


def _classify_1d_at_point(H, xi_sym, pt, max_order=5):
    """
    Classify 1D catastrophe type at xi=pt by checking derivatives.
    """
    derivs = {}
    for k in range(1, max_order + 1):
        derivs[k] = float(sp.N(sp.diff(H, (xi_sym, k)).subs({xi_sym: pt})))
    for k in range(1, max_order + 1):
        if abs(derivs[k]) > 1e-8:
            if k == 1:
                return "regular (not critical)", derivs
            elif k == 2:
                return "fold (A2)", derivs
            elif k == 3:
                return "cusp (A3)", derivs
            elif k == 4:
                return "swallowtail (A4)", derivs
            else:
                return f"A{k} (higher)", derivs
    return "flat (all derivatives vanish up to order {})".format(max_order), derivs


def detect_catastrophes(H_expr, xi_vars, coords=None, method="auto", numeric_grid=None, max_order=5):
    """
    Detect catastrophes for a Hamiltonian symbol H_expr.

    Parameters
    ----------
    H_expr : sympy.Expr
    xi_vars : tuple of sympy symbols (xi,) or (xi,eta)
    coords : tuple of sympy symbols (parameters) to solve for as well (optional)
    method : "symbolic" | "numeric" | "auto"
    numeric_grid : dict for numeric search options
    max_order : int (for 1D derivative checks)
    """
    results = []
    dim = len(xi_vars)
    coords = tuple(coords) if coords is not None else tuple()
    unknowns = tuple(list(xi_vars) + list(coords))

    if dim == 1:
        xi = xi_vars[0]
        eq = sp.diff(H_expr, xi)
        eqs = [sp.Eq(eq, 0)]
        sol_list = None
        if method in ("symbolic", "auto"):
            sol_list = _try_symbolic_solve(eqs, unknowns)
        if not sol_list and method in ("numeric", "auto"):
            grid = numeric_grid or {}
            xi_b = grid.get("xi_bounds", (-5.0, 5.0))
            nxi = int(grid.get("n_xi", 41))
            xi_inits = list(_np.linspace(xi_b[0], xi_b[1], nxi))
            param_inits = [()]
            if coords:
                p_bounds = grid.get("param_bounds", [(-1, 1) for _ in coords])
                pn = int(grid.get("n_param", 7))
                arrays = [ _np.linspace(b[0], b[1], pn) for b in p_bounds ]
                import itertools
                param_inits = list(itertools.product(*arrays))
            guesses = []
            for xi0 in xi_inits:
                for p in param_inits:
                    guesses.append(tuple([xi0] + list(p)))
            sol_num = _numeric_solve_system([eq], unknowns, guesses)
            sol_list = sol_num if sol_num else None

        if not sol_list:
            return results

        if isinstance(sol_list, dict):
            sol_list = [sol_list]
        for sol in sol_list:
            if isinstance(sol, dict):
                mapping = {k: float(sp.N(sol[k])) for k in sol}
            else:
                continue
            xi_val = mapping.get(xi, None)
            if xi_val is None:
                continue
            typ, derivs = _classify_1d_at_point(H_expr, xi, xi_val, max_order=max_order)
            res = {"point": mapping, "type": typ, "details": {"derivatives": derivs}}
            results.append(res)
        return results

    if dim == 2:
        xi, eta = xi_vars
        dxi = sp.diff(H_expr, xi)
        deta = sp.diff(H_expr, eta)
        eqs = [sp.Eq(dxi, 0), sp.Eq(deta, 0)]
        sol_list = None
        if method in ("symbolic", "auto"):
            sol_list = _try_symbolic_solve(eqs, unknowns)
        if not sol_list and method in ("numeric", "auto"):
            grid = numeric_grid or {}
            xi_b = grid.get("xi_bounds", (-5.0, 5.0))
            eta_b = grid.get("eta_bounds", (-5.0, 5.0))
            nxi = int(grid.get("n_xi", 21))
            neta = int(grid.get("n_eta", 21))
            xi_inits = list(_np.linspace(xi_b[0], xi_b[1], nxi))
            eta_inits = list(_np.linspace(eta_b[0], eta_b[1], neta))
            param_inits = [()]
            if coords:
                p_bounds = grid.get("param_bounds", [(-1,1) for _ in coords])
                pn = int(grid.get("n_param", 5))
                arrays = [ _np.linspace(b[0], b[1], pn) for b in p_bounds ]
                import itertools
                param_inits = list(itertools.product(*arrays))
            guesses = []
            for xi0 in xi_inits:
                for eta0 in eta_inits:
                    for p in param_inits:
                        guesses.append(tuple([xi0, eta0] + list(p)))
            sol_num = _numeric_solve_system(eqs, unknowns, guesses)
            sol_list = sol_num if sol_num else None

        if not sol_list:
            return results

        for sol in sol_list:
            if isinstance(sol, dict):
                mapping = {k: float(sp.N(sol[k])) for k in sol}
            else:
                continue
            subs_map = {k: mapping[k] for k in mapping}
            H_xixi = float(sp.N(sp.diff(H_expr, (xi, 2)).subs(subs_map)))
            H_etaeta = float(sp.N(sp.diff(H_expr, (eta, 2)).subs(subs_map)))
            H_xieta = float(sp.N(sp.diff(H_expr, xi, eta).subs(subs_map)))
            Hess = _np.array([[H_xixi, H_xieta], [H_xieta, H_etaeta]], dtype=float)
            w, v = _np.linalg.eig(Hess)
            rank = _np.linalg.matrix_rank(Hess, tol=1e-8)
            info = {"hessian": Hess.tolist(), "eigenvals": w.tolist(), "eigenvecs": v.tolist(), "rank": int(rank)}
            if rank == 2:
                typ = "non-degenerate critical point (no catastrophe)"
            elif rank == 1:
                null_idx = int(_np.argmin(_np.abs(w)))
                null_vec = v[:, null_idx]
                vx, vy = float(null_vec[0]), float(null_vec[1])
                D = vx*sp.diff(H_expr, xi) + vy*sp.diff(H_expr, eta)
                D2 = vx*sp.diff(D, xi) + vy*sp.diff(D, eta)
                D3 = vx*sp.diff(D2, xi) + vy*sp.diff(D2, eta)
                D3_val = float(sp.N(D3.subs(subs_map)))
                if abs(D3_val) > 1e-8:
                    typ = "fold/cusp family (one zero eigenvalue, D3 != 0)"
                else:
                    typ = "higher degeneracy (one zero eigenvalue and directional third = 0)"
                info["directional_third"] = D3_val
            else:
                # rank == 0
                typ = "high degeneracy (possible umbilic or higher catastrophe)"
            results.append({"point": mapping, "type": typ, "details": info})
        return results

    raise NotImplementedError("detect_catastrophes only supports dim 1 or 2.")


# --------------------------
# Arnold 2D classifier (detailed)
# --------------------------
def classify_arnold_2d(H, xi, eta, point, tol=1e-8):
    """
    Classify a 2D catastrophe at a critical point of H(xi, eta),
    following Arnold invariants: Morse, A2, A3, A4, A5, D4+, D4-.
    point is a dict mapping sympy symbols to numeric values.
    Returns a dict with classification and derivative tensors.

    Arnold A_k Classification:
    - A_k: k-th catastrophe (k ≥ 2)  
    - Characterized by D_{k+1} = first non-zero directional derivative
    - Examples:
      * A2 (Fold): H ~ ξ³
      * A3 (Cusp): H ~ ξ⁴  
      * A4 (Swallowtail): H ~ ξ⁵
      * A5 (Butterfly): H ~ ξ⁶
    """
    subs = {xi: point["xi"], eta: point["eta"]}

    # --- 1. Hessian (2nd order) ---
    H_xx = float(sp.N(sp.diff(H, (xi, 2)).subs(subs)))
    H_yy = float(sp.N(sp.diff(H, (eta, 2)).subs(subs)))
    H_xy = float(sp.N(sp.diff(H, xi, eta).subs(subs)))
    Hess = _np.array([[H_xx, H_xy], [H_xy, H_yy]], dtype=float)
    rank = _np.linalg.matrix_rank(Hess, tol=tol)
    eigvals, eigvecs = _np.linalg.eig(Hess)

    # --- 2. Third-order tensor ---
    H_xxx = float(sp.N(sp.diff(H, (xi, 3)).subs(subs)))
    H_xxy = float(sp.N(sp.diff(sp.diff(H, (xi, 2)), eta).subs(subs)))
    H_xyy = float(sp.N(sp.diff(sp.diff(H, xi, (eta, 2))).subs(subs)))
    H_yyy = float(sp.N(sp.diff(H, (eta, 3)).subs(subs)))
    third = {"H_xxx": H_xxx, "H_xxy": H_xxy, "H_xyy": H_xyy, "H_yyy": H_yyy}

    # --- Case 1: Morse point (non-degenerate) ---
    if rank == 2:
        return {"type": "Morse (non-degenerate)", "hessian": Hess.tolist(), "third_order_tensor": third}

    # --- Case 2: Rank 1 (A_k family) ---
    if rank == 1:
        # Find the null direction (eigenvector associated with the smallest eigenvalue)
        null_idx = int(_np.argmin(_np.abs(eigvals)))
        null_dir = eigvecs[:, null_idx]
        vx, vy = float(null_dir[0]), float(null_dir[1])

        # Build the directional operator D = vx * d/dxi + vy * d/deta
        # Compute successive directional derivatives
        D1 = vx*sp.diff(H, xi) + vy*sp.diff(H, eta)  # = 0 at critical point
        D2 = vx*sp.diff(D1, xi) + vy*sp.diff(D1, eta) # = 0 because rank 1
        D3 = vx*sp.diff(D2, xi) + vy*sp.diff(D2, eta)
        D4 = vx*sp.diff(D3, xi) + vy*sp.diff(D3, eta)
        D5 = vx*sp.diff(D4, xi) + vy*sp.diff(D4, eta)
        # --- FIX: Add D6 for A5 (Butterfly) test case ---
        D6 = vx*sp.diff(D5, xi) + vy*sp.diff(D5, eta)

        D3_val = float(sp.N(D3.subs(subs)))
        D4_val = float(sp.N(D4.subs(subs)))
        D5_val = float(sp.N(D5.subs(subs)))
        # --- FIX: Evaluate D6 ---
        D6_val = float(sp.N(D6.subs(subs)))

        # --- FIX: Store D6 in results ---
        directional = {"D3": D3_val, "D4": D4_val, "D5": D5_val, "D6": D6_val}

        # --- FIX: Corrected A_k classification (A_k <-> D_{k+1}) ---
        # The first non-vanishing derivative D_{k+1} determines the A_k catastrophe.
        if abs(D3_val) > tol:
            # A2 (Fold): D3 is the first non-zero
            return {"type": "A2 (Fold)", "hessian": Hess.tolist(), "third_order_tensor": third, "directional_derivatives": directional}
        elif abs(D4_val) > tol:
            # A3 (Cusp): D4 is the first non-zero
            return {"type": "A3 (Cusp)", "hessian": Hess.tolist(), "third_order_tensor": third, "directional_derivatives": directional}
        elif abs(D5_val) > tol:
            # A4 (Swallowtail): D5 is the first non-zero
            return {"type": "A4 (Swallowtail)", "hessian": Hess.tolist(), "third_order_tensor": third, "directional_derivatives": directional}
        elif abs(D6_val) > tol:
            # A5 (Butterfly): D6 is the first non-zero
            return {"type": "A5 (Butterfly)", "hessian": Hess.tolist(), "third_order_tensor": third, "directional_derivatives": directional}
        else:
            # A6 or higher
            return {"type": "A6+ or higher degeneracy", "hessian": Hess.tolist(), "third_order_tensor": third, "directional_derivatives": directional}


    # --- Case 3: Rank 0 (D4 umbilics and higher) ---
    # The Hessian is zero. We examine the cubic tensor.
    # (Note: This logic is correct for D4 (Tests 5, 6).
    # It does not handle "corang 2" cases where H_3 = 0 but H_4 != 0 (like Test 7),
    # so it correctly classifies them as a high degeneracy.)
    
    I = H_xxx * H_yyy - H_xxy * H_xyy
    if abs(I) < tol:
        # This includes D4+/- where I=0, or higher degeneracies (E6, etc.)
        return {"type": "D4 degenerate (I=0) or higher (E6?)", "hessian": Hess.tolist(), "third_order_tensor": third, "cubic_invariant_I": I}
    elif I > 0:
        # This could be a valid case for D4+ if I != 0 (depends on specific H)
        return {"type": "D4+ (Hyperbolic umbilic - I>0)", "hessian": Hess.tolist(), "third_order_tensor": third, "cubic_invariant_I": I}
    else: # I < 0
        # This could be a valid case for D4- if I != 0 (depends on specific H)
        return {"type": "D4- (Elliptic umbilic - I<0)", "hessian": Hess.tolist(), "third_order_tensor": third, "cubic_invariant_I": I}
        
# --------------------------
# Plotting helper for catastrophes
# --------------------------
def plot_catastrophe(H, xi_vars, points, xi_bounds=(-3, 3), eta_bounds=(-3, 3), n=300):
    """
    Plot H and mark catastrophe points. Works for 1D and 2D.
    Requires matplotlib.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib not available; install matplotlib to use plot_catastrophe().")
    dim = len(xi_vars)
    if dim == 1:
        xi = xi_vars[0]
        X = _np.linspace(xi_bounds[0], xi_bounds[1], n)
        Hf = sp.lambdify(xi, H, "numpy")
        Y = Hf(X)
        _plt.figure(figsize=(8, 5))
        _plt.plot(X, Y, label="H(xi)")
        for p in points:
            xv = p["point"][xi]
            yv = float(Hf(xv))
            _plt.scatter([xv], [yv], color='red')
            _plt.text(xv, yv, p["type"])
        _plt.xlabel("xi")
        _plt.ylabel("H")
        _plt.title("Catastrophe plot (1D)")
        _plt.grid()
        _plt.show()
    elif dim == 2:
        xi, eta = xi_vars
        X = _np.linspace(xi_bounds[0], xi_bounds[1], int(n/2))
        Y = _np.linspace(eta_bounds[0], eta_bounds[1], int(n/2))
        XX, YY = _np.meshgrid(X, Y)
        Hf = sp.lambdify((xi, eta), H, "numpy")
        ZZ = Hf(XX, YY)
        fig = _plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(XX, YY, ZZ, alpha=0.6, rstride=3, cstride=3)
        for p in points:
            xv = p["point"][xi]
            yv = p["point"][eta]
            zv = float(Hf(xv, yv))
            ax.scatter([xv], [yv], [zv], color='red', s=80)
            ax.text(xv, yv, zv, p["type"])
        ax.set_xlabel("xi")
        ax.set_ylabel("eta")
        ax.set_zlabel("H")
        _plt.title("Catastrophe surface (2D)")
        _plt.show()
    else:
        raise NotImplementedError("plot_catastrophe supports only dim 1 or 2.")
