import sympy as sp

import sympy as sp
from sympy import Matrix

class LagrangianHamiltonianConverter:
    """
    Bidirectional converter between Lagrangian and Hamiltonian
    (Legendre transform), compatible with 1D and 2D.

    Improvements over the simple version:
      - checks convexity / Hessian invertibility before inversion,
      - analytic fast path for quadratic-in-p Lagrangians,
      - 'force' flag to bypass checks when user deliberately wants to proceed,
      - clearer error messages and diagnostic information.

    Public API:
      L_to_H(L_expr, coords, u, p_vars, return_symbol_only=False, force=False)
      H_to_L(H_expr, coords, u, xi_vars, force=False)
    """

    @staticmethod
    def _is_quadratic_in_p(L_expr, p_vars):
        # Return True if L_expr is (at most) quadratic in all p_vars
        for p in p_vars:
            if sp.degree(L_expr, p) is None:
                return False
            if sp.degree(L_expr, p) > 2:
                return False
        return True

    @staticmethod
    def _quadratic_legendre(L_expr, p_vars, xi_vars):
        """
        Analytic Legendre transform for quadratic L of the form:
          L = 1/2 * p^T A(x,u) p + b(x,u)^T p + c(x,u)
        Returns tuple (H_expr, sol) where sol is mapping p_vars -> p(xi,...)
        """
        # Build Hessian (A) and linear term (b)
        A = sp.Matrix([[sp.diff(sp.diff(L_expr, p_i), p_j) for p_j in p_vars] for p_i in p_vars])
        # Linear term: b_i = ∂L/∂p_i - sum_j A_ij p_j  (but simpler: evaluate gradient and remove A p)
        grad = Matrix([sp.diff(L_expr, p) for p in p_vars])
        # For quadratic L, grad = A * p + b  => b = grad - A*p (so b depends on p symbolically)
        # We can deduce b as gradient with p set to 0 if there are no constant-in-p contributions in A
        # Simpler approach: write ansatz p = A^{-1} (xi - b). We'll solve for p symbolically by linear solve.
        # Build symbolic linear system: A * p - (xi - b) = 0 ; but b may depend on p in degenerate cases.
        try:
            A_inv = A.inv()
        except Exception:
            raise ValueError("Quadratic path: Hessian matrix A is singular (non-invertible).")

        # Solve linear system ∂L/∂p = xi  => A p + b = xi  => p = A^{-1} (xi - b)
        # To obtain b(x,u) we compute grad and substitute p_vars -> 0 to get pure linear terms in p (works for standard quadratic L).
        b_vec = grad.subs({p: 0 for p in p_vars})
        # Construct p solution
        xi_vec = Matrix(xi_vars)
        p_solution_vec = A_inv * (xi_vec - b_vec)
        sol = {p_vars[i]: sp.simplify(p_solution_vec[i]) for i in range(len(p_vars))}
        # H = xi·p - L(p->sol)
        H_expr = sum(xi_vars[i]*sol[p_vars[i]] for i in range(len(p_vars))) - sp.simplify(L_expr.subs(sol))
        return sp.simplify(H_expr), sol

    @staticmethod
    def L_to_H(L_expr, coords, u, p_vars, return_symbol_only=False, force=False):
        """
        Transforms a Lagrangian L(x,u,p) into a Hamiltonian H(x,u,ξ)
        via the Legendre transform.

        Parameters
        ----------
        L_expr : sympy.Expr
            Lagrangian expression L(x, u, p).
        coords : tuple
            Spatial coordinates (x,) or (x, y).
        u : sympy.Function or sympy.Symbol
            Dependent variable placeholder in L (kept symbolic).
        p_vars : tuple
            Momentum variables (p,) or (p_x, p_y).
        return_symbol_only : bool
            If True, remove explicit dependence on u (useful for pseudo-differential symbols).
        force : bool
            If True, attempt inversion even if Hessian is singular or convexity cannot be proven.

        Returns
        -------
        H_expr, xi_vars
        """
        dim = len(coords)
        if dim == 1:
            xi_vars = (sp.Symbol('xi', real=True),)
        elif dim == 2:
            xi_vars = (sp.Symbol('xi', real=True), sp.Symbol('eta', real=True))
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")

        # 1) Compute Hessian w.r.t. momentum variables
        try:
            H_p = sp.hessian(L_expr, p_vars)
        except Exception:
            # If Hessian fails, fall back to a best-effort behaviour below
            H_p = None

        if H_p is not None:
            det_H = sp.simplify(H_p.det())
        else:
            det_H = None

        # 2) If quadratic in p, use analytic path (fast and robust when A invertible)
        if LagrangianHamiltonianConverter._is_quadratic_in_p(L_expr, p_vars):
            if H_p is None:
                # unlikely, but be defensive
                if not force:
                    raise ValueError("Cannot compute Hessian for quadratic L; aborting. Use force=True to bypass.")
            else:
                # try the analytic quadratic inversion
                try:
                    H_expr, sol = LagrangianHamiltonianConverter._quadratic_legendre(L_expr, p_vars, xi_vars)
                    if return_symbol_only:
                        H_expr = H_expr.subs(u, 0)
                    return H_expr, xi_vars
                except ValueError as e:
                    if not force:
                        raise
                    # else fall through to solve() attempt

        # 3) If Hessian determinant is symbolically zero -> not invertible (unless force)
        if det_H is not None:
            # If det_H simplifies to 0, then Hessian singular -> Legendre not (globally) invertible
            if det_H == 0:
                if not force:
                    raise ValueError(
                        "Legendre transform is not invertible: Hessian w.r.t. p is singular (determinant == 0). "
                        "This indicates L is not strictly convex in p. If you are sure you want to proceed, call with force=True."
                    )
                # else: proceed with solve() (best-effort)
            else:
                # det_H nonzero symbolically: OK to attempt inversion / solve
                pass
        else:
            # Couldn't compute determinant symbolically: warn user unless forced
            if not force:
                raise ValueError(
                    "Unable to symbolically verify Hessian determinant. "
                    "Call with force=True to attempt a best-effort inversion using solve()."
                )

        # 4) Default fallback: set up Legendre relations and try solve()
        eqs = [sp.Eq(sp.diff(L_expr, p_vars[i]), xi_vars[i]) for i in range(dim)]
        sol_list = sp.solve(eqs, p_vars, dict=True)

        if not sol_list:
            if not force:
                raise ValueError(
                    "Unable to solve the Legendre relations ξ = ∂L/∂p. "
                    "This may be because the system is not algebraically solvable or the mapping is not invertible. "
                    "Try using a quadratic L or set force=True to attempt a numeric/local inversion."
                )
            else:
                # If forced, try a less structured solve (non-dict) or manual substitution
                sol_list = sp.solve(eqs, p_vars)  # fall back; may be empty or complicated

        if not sol_list:
            raise ValueError("Legendre inversion failed even with force=True.")

        sol = sol_list[0] if isinstance(sol_list, list) else sol_list
        # If solve returned a list of expressions, convert to mapping
        if isinstance(sol, tuple) and len(sol) == len(p_vars):
            sol = {p_vars[i]: sol[i] for i in range(len(p_vars))}
        if not isinstance(sol, dict):
            # if solve produced a dict-like mapping in a list, try first entry
            if isinstance(sol_list, list) and sol_list and isinstance(sol_list[0], dict):
                sol = sol_list[0]
            else:
                # try to build mapping when solve returns expressions in order
                try:
                    sol = {p_vars[i]: sol[i] for i in range(len(p_vars))}
                except Exception:
                    if not force:
                        raise ValueError("Unexpected solve() output; cannot construct p(ξ).")
                    # else continue and try best-effort below

        # 5) Hamiltonian: H = Σ ξ_i * p_i - L(p->sol)
        H_expr = sum(xi_vars[i]*sol[p_vars[i]] for i in range(dim)) - L_expr.subs(sol)
        H_expr = sp.simplify(H_expr)

        if return_symbol_only:
            H_expr = H_expr.subs(u, 0)

        return H_expr, xi_vars


    @staticmethod
    def H_to_L(H_expr, coords, u, xi_vars, force=False):
        """
        Transforms a Hamiltonian H(x,u,ξ) into a Lagrangian L(x,u,p)
        via the inverse Legendre transform.

        Parameters
        ----------
        H_expr : sympy.Expr
        coords : tuple
        u : sympy.Function or sympy.Symbol
        xi_vars : tuple of dual variables (xi,) or (xi,eta)
        force : bool
            Bypass some symbolic checks and attempt solve() forcibly.

        Returns
        -------
        L_expr, p_vars
        """
        dim = len(coords)
        if dim == 1:
            p_vars = (sp.Symbol('p', real=True),)
        elif dim == 2:
            p_vars = (sp.Symbol('p_x', real=True), sp.Symbol('p_y', real=True))
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")

        # Build inverse relations: p_i = ∂H/∂ξ_i  → solve for ξ_i in terms of p_i
        eqs = [sp.Eq(sp.diff(H_expr, xi_vars[i]), p_vars[i]) for i in range(dim)]
        sol = sp.solve(eqs, xi_vars, dict=True)

        if not sol:
            if not force:
                raise ValueError(
                    "Unable to symbolically solve p = ∂H/∂ξ for ξ. "
                    "Either H is not convex/invertible in ξ or the system is not algebraically solvable. "
                    "Use force=True for a best-effort attempt."
                )
            # try fallback
            sol = sp.solve(eqs, xi_vars)

        if not sol:
            raise ValueError("Inverse Legendre transform failed; cannot find ξ(p).")

        sol = sol[0] if isinstance(sol, list) else sol
        # Ensure mapping form
        if isinstance(sol, tuple) and len(sol) == len(xi_vars):
            sol = {xi_vars[i]: sol[i] for i in range(len(xi_vars))}
        if not isinstance(sol, dict):
            if isinstance(sol, list) and sol and isinstance(sol[0], dict):
                sol = sol[0]
            else:
                raise ValueError("Unexpected output from solve(); cannot construct ξ(p).")

        L_expr = sum(sol[xi_vars[i]] * p_vars[i] for i in range(dim)) - H_expr.subs(sol)
        return sp.simplify(L_expr), p_vars
        
class HamiltonianSymbolicConverter:
    """
    Symbolic converter between Hamiltonians and formal PDEs.

    This class provides tools to construct symbolic partial differential equations
    (stationary, Schrödinger, or wave-type) from a given **Hamiltonian symbol**
    H(x, ξ[, η]), using the pseudo-differential notation ψOp(H, u).

    It does **not** apply any operator to `u(x,t)` directly — the purpose is
    to preserve the symbolic dependence in phase space (x, ξ, [η]).

    ---
    Example (1D Schrödinger-like):

        >>> x, t, xi = sp.symbols("x t xi", real=True)
        >>> u = sp.Function("u")(x, t)
        >>> V = sp.Function("V")(x)
        >>> H = 0.5*xi**2 + V
        >>> res = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
        ...     H, (x,), t, u, mode="schrodinger")
        >>> print(res["pde"])
        Eq(I*Derivative(u(x, t), t), psiOp(0.5*xi**2 + V(x), u(x, t)))
    """

    # ------------------------------------------------------------
    # 1. Hamiltonian decomposition
    # ------------------------------------------------------------
    @staticmethod
    def decompose_hamiltonian(H_expr, xi_vars):
        """
        Decompose a Hamiltonian into its **polynomial** and **non-local** parts.

        Args:
            H_expr : sympy.Expr
                The Hamiltonian symbol H(x, ξ[, η]).
            xi_vars : tuple
                The dual (frequency) variables, typically (xi,) or (xi, eta).

        Returns:
            tuple (H_poly, H_nonlocal)
                - H_poly : polynomial in ξ (and η)
                - H_nonlocal : non-polynomial part (e.g. |ξ|, sqrt(ξ² + m²), …)
        """
        xi = xi_vars if isinstance(xi_vars, (tuple, list)) else (xi_vars,)
        poly_terms, nonlocal_terms = 0, 0
        H_expand = sp.expand(H_expr)

        for term in H_expand.as_ordered_terms():
            if all(term.is_polynomial(xi_i) for xi_i in xi):
                poly_terms += term
            else:
                nonlocal_terms += term

        return sp.simplify(poly_terms), sp.simplify(nonlocal_terms)

    # ------------------------------------------------------------
    # 2. Hamiltonian → Symbolic PDE (ψOp-form)
    # ------------------------------------------------------------
    @classmethod
    def hamiltonian_to_symbolic_pde(cls, H_expr, coords, t, u, mode="schrodinger"):
        """
        Convert a Hamiltonian symbol H(x, ξ[, η]) into a **formal PDE** expressed
        with pseudo-differential operators ψOp(H, u).

        Supported PDE forms:
          • Stationary   :  ψOp(H, u) = E u
          • Schrödinger  :  i ∂ₜu = ψOp(H, u)
          • Wave         :  ∂ₜₜu + ψOp(H, u) = 0

        Args:
            H_expr : sympy.Expr
                Hamiltonian H(x, ξ[, η]).
            coords : tuple
                Spatial coordinates (x,) or (x, y).
            t : sympy.Symbol
                Time variable.
            u : sympy.Function
                Dependent variable u(x[, y], t).
            mode : str
                PDE type, one of {"stationary", "schrodinger", "wave"}.

        Returns:
            dict :
                {
                    "pde"          : sympy.Eq — the symbolic PDE,
                    "H_poly"       : polynomial part of H,
                    "H_nonlocal"   : nonlocal part of H,
                    "formal_string": textual representation,
                    "mode"         : PDE mode.
                }
        """
        dim = len(coords)
        if dim == 1:
            xi_vars = (sp.Symbol("xi", real=True),)
        elif dim == 2:
            xi_vars = (sp.Symbol("xi", real=True), sp.Symbol("eta", real=True))
        else:
            raise ValueError("Only 1D and 2D Hamiltonians are supported.")

        # --- Decompose the Hamiltonian into polynomial / nonlocal parts
        H_poly, H_nonlocal = cls.decompose_hamiltonian(H_expr, xi_vars)
        H_total = H_poly + H_nonlocal

        # --- Formal pseudo-differential operator
        psiOp_H_u = sp.Function("psiOp")(H_total, u)

        # --- Build the PDE depending on the mode
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

