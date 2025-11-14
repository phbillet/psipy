import sympy as sp

class LagrangianHamiltonianConverter:
    """
    Bidirectional converter between Lagrangian and Hamiltonian
    (Legendre transform), compatible with 1D and 2D.

    Can be used:
    - for classical analytical formulations (mechanics, scalar field),
    - or to generate a pseudo-differential symbol H(x,ξ)
      from a Lagrangian quadratic in p.
    """

    @staticmethod
    def L_to_H(L_expr, coords, u, p_vars, return_symbol_only=False):
        """
        Transforms a Lagrangian L(x,u,p) into a Hamiltonian H(x,u,ξ)
        via the Legendre transform.
        Arguments:
            L_expr: sympy expression of the Lagrangian L(x,u,p)
            coords: tuple of coordinates (x,) or (x,y)
            u: sympy dependent variable
            p_vars: tuple of momentum variables (p,) or (p_x,p_y)
            return_symbol_only: if True, removes any dependence on u
                                (useful for pseudo-differential symbols)

        Returns:
            H_expr: sympy expression of the Hamiltonian
            xi_vars: tuple of dual variables (ξ,) or (ξ,η)
        """
        dim = len(coords)
        if dim == 1:
            xi_vars = (sp.Symbol('xi', real=True),)
        elif dim == 2:
            xi_vars = (sp.Symbol('xi', real=True), sp.Symbol('eta', real=True))
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")
        # Legendre relations: ξ_i = ∂L/∂p_i
        eqs = [sp.Eq(sp.diff(L_expr, p_vars[i]), xi_vars[i]) for i in range(dim)]
        # Solve for p_i in terms of ξ_i
        sol = sp.solve(eqs, p_vars, dict=True)
        if not sol:
            raise ValueError("Unable to solve the relation ξ = ∂L/∂p.")
        sol = sol[0]
        # Hamiltonian: H = Σ ξ_i * p_i - L
        H_expr = sum(xi_vars[i]*sol[p_vars[i]] for i in range(dim)) - L_expr.subs(sol)
        H_expr = sp.simplify(H_expr)
        if return_symbol_only:
            # Remove any dependence on u
            H_expr = H_expr.subs(u, 0)
        return H_expr, xi_vars

    @staticmethod
    def H_to_L(H_expr, coords, u, xi_vars):
        """
        Transforms a Hamiltonian H(x,u,ξ) into a Lagrangian L(x,u,p)
        via the inverse Legendre transform.
        Arguments:
            H_expr: sympy expression of the Hamiltonian H(x,u,ξ)
            coords: tuple of coordinates (x,) or (x,y)
            u: sympy dependent variable
            xi_vars: tuple of dual variables (ξ,) or (ξ,η)

        Returns:
            L_expr: sympy expression of the Lagrangian
            p_vars: tuple of momentum variables (p,) or (p_x,p_y)
        """
        dim = len(coords)
        if dim == 1:
            p_vars = (sp.Symbol('p', real=True),)
        elif dim == 2:
            p_vars = (sp.Symbol('p_x', real=True), sp.Symbol('p_y', real=True))
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")
        # Inverse relations: p_i = ∂H/∂ξ_i
        eqs = [sp.Eq(sp.diff(H_expr, xi_vars[i]), p_vars[i]) for i in range(dim)]
        # Solve for ξ_i in terms of p_i
        sol = sp.solve(eqs, xi_vars, dict=True)
        if not sol:
            raise ValueError("Unable to solve the relation p = ∂H/∂ξ.")
        sol = sol[0]
        # Lagrangian: L = Σ ξ_i * p_i - H
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

