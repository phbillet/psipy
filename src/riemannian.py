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
riemannian.py — Unified 1D/2D Riemannian geometry toolkit
================================================================

Overview
--------
The `riemannian` module provides a unified framework for working with Riemannian manifolds in one and two dimensions.  A single class `Metric` dispatches all computations automatically based on the dimension, making it easy to switch between 1D curves and 2D surfaces without changing the calling interface.

Key features include:

* Symbolic construction of the metric tensor from an explicit expression (1D scalar, 2D matrix) or by extraction from a Hamiltonian kinetic energy.
* Automatic computation of Christoffel symbols (1D and 2D).
* Geodesic integration with multiple numerical schemes (RK4, adaptive, symplectic/Verlet, Hamiltonian flow via the companion `symplectic` module).
* Curvature: Riemann tensor, Ricci tensor, Gaussian curvature, scalar curvature.
* Laplace–Beltrami operator: full symbol (principal + subprincipal parts) ready for microlocal analysis.
* Riemannian volume (arc length in 1D) via symbolic or numerical integration.
* **2D only**: Exponential map, geodesic distance (shooting or optimisation), Jacobi equation solver (geodesic deviation), Hodge star operator, de Rham Laplacian on 0‑ and 1‑forms, numerical verification of the Gauss–Bonnet theorem.
* Rich visualisation suite: geodesic trajectories, curvature maps (Gaussian/Ricci), metric components.

Mathematical background
-----------------------
A **Riemannian metric** `g` on an `n`-dimensional manifold assigns an inner product to each tangent space.  In local coordinates `(x¹,…,xⁿ)` the metric is written as

    ds² = gᵢⱼ(x) dxⁱ dxʲ

and its inverse is denoted `gⁱʲ`.  The **Christoffel symbols** are derived from the metric:

    Γⁱⱼₖ = ½ gⁱˡ (∂ⱼ gₖₗ + ∂ₖ gⱼₗ − ∂ₗ gⱼₖ)

and determine the **geodesic equation**

    ẍⁱ + Γⁱⱼₖ ẋʲ ẋᵏ = 0.

For a 1D metric `g₁₁(x)` the geodesic equation simplifies to
    ẍ + Γ¹₁₁ ẋ² = 0, Γ¹₁₁ = ½ (log g₁₁)′.

**Curvature** is encoded in the Riemann tensor `Rⁱⱼₖₗ`, the Ricci tensor `Rᵢⱼ = Rᵏᵢₖⱼ`, and the scalar curvature `R = gⁱʲ Rᵢⱼ`.  For a 2D surface the Gaussian curvature `K` satisfies `R₁₂₁₂ = K |g|` and is the only independent component.

The **Laplace–Beltrami operator** acting on functions is

    Δ = |g|^{-½} ∂ᵢ ( |g|^{½} gⁱʲ ∂ⱼ ),

and its principal symbol is `gⁱʲ ξᵢ ξⱼ`.  The subprincipal symbol encodes the lower‑order terms.

The module also implements the **Hodge star** on differential forms and the **de Rham Laplacian** `Δ = dδ + δd` for 0‑ and 1‑forms in 2D.


References
----------
.. [1] do Carmo, M. P.  *Riemannian Geometry*, Birkhäuser, 1992.
.. [2] Jost, J.  *Riemannian Geometry and Geometric Analysis*, Springer, 2011 (6th ed.).
.. [3] Lee, J. M.  *Riemannian Manifolds: An Introduction to Curvature*, Springer, 1997.
.. [4] Petersen, P.  *Riemannian Geometry*, Springer, 2016 (3rd ed.).
.. [5] Frankel, T.  *The Geometry of Physics*, Cambridge University Press, 2011 (3rd ed.).

"""

from imports import *
from symplectic import hamiltonian_flow as symp_hamiltonian_flow

# Consolidate all scipy imports here so they are not re-imported inside
# every function call (negligible overhead, but noisy and hard to audit).
from scipy.integrate import (
    quad, dblquad, solve_ivp, cumulative_trapezoid,
)
from scipy.interpolate import interp1d
from scipy.optimize import minimize


# ============================================================================
# Unified Metric class
# ============================================================================

class Metric:
    """
    Riemannian metric on a 1D or 2D manifold.
    
    The dimension is inferred automatically from the supplied inputs:
    
    - **1D**: ``g_input`` is a scalar SymPy expression in one coordinate;
      ``coords`` is a 1-tuple ``(x,)`` or a bare symbol.
    - **2D**: ``g_input`` is a 2×2 SymPy ``Matrix`` (or a nested list that
      will be promoted to one); ``coords`` is a 2-tuple ``(x, y)``.
    
    On construction the metric is simplified once, and all derived symbolic
    quantities — inverse metric, determinant, square-root of the determinant,
    and Christoffel symbols — are computed and stored. Numerical callables
    (``lambdify``-produced functions) are also built and cached so that
    integration and visualization routines have zero symbolic overhead at
    run-time.
    
    Parameters
    ----------
    g_input : sympy.Expr or sympy.Matrix or list
        Metric tensor. A scalar SymPy expression for 1D, or a 2×2 SymPy
        Matrix (or equivalent nested list of expressions) for 2D.
    coords : tuple of sympy.Symbol or sympy.Symbol
        Coordinate symbols in order. The length determines the manifold
        dimension (1 or 2). A bare symbol is accepted for the 1D case.
    
    Attributes
    ----------
    dim : int
        Manifold dimension, either 1 or 2.
    coords : tuple of sympy.Symbol
        Coordinate symbols, always stored as a tuple.
    g_expr : sympy.Expr
        (1D only) Simplified metric component g₁₁(x).
    g_inv_expr : sympy.Expr
        (1D only) Inverse metric component g¹¹(x) = 1/g₁₁(x).
    sqrt_det_expr : sympy.Expr
        (1D only) Square root of the metric determinant, √|g₁₁(x)|.
    christoffel_sym : sympy.Expr or dict
        Symbolic Christoffel symbols. 
        - 1D: Expression for Γ¹₁₁ = ½ (log g₁₁)′.
        - 2D: Nested dict ``christoffel_sym[i][j][k]`` → SymPy expression.
    g_func : callable or dict
        Numerical metric function.
        - 1D: ``g₁₁(x_val)`` → float or ndarray.
        - 2D: Dict ``{(i, j): callable}`` of component functions.
    g_inv_func : callable or dict
        Numerical inverse-metric function.
        - 1D: ``g¹¹(x_val)`` → float or ndarray.
        - 2D: Dict ``{(i, j): callable}`` of component functions.
    sqrt_det_func : callable
        (1D only) Numerical function √|g₁₁|(x_val) → float or ndarray.
    sqrt_det_g_func : callable
        (2D only) Numerical function √|det(g)|(x_val, y_val).
    christoffel_func : callable or dict
        Numerical Christoffel callables.
        - 1D: ``Γ¹₁₁(x_val)`` → float or ndarray.
        - 2D: Nested dict of callables ``Γⁱⱼₖ(x_val, y_val)``.
    g_matrix : sympy.Matrix
        (2D only) Simplified 2×2 metric tensor matrix.
    det_g : sympy.Expr
        (2D only) Determinant of the metric, det(g).
    sqrt_det_g : sympy.Expr
        (2D only) Square root of the absolute determinant, √|det(g)|.
    g_inv_matrix : sympy.Matrix
        (2D only) Symbolic inverse metric g⁻¹.
    det_g_func : callable
        (2D only) Numerical function det(g)(x_val, y_val).
    
    Raises
    ------
    ValueError
        If the number of coordinate symbols is neither 1 nor 2.
    ValueError
        If ``g_input`` is not a 2×2 matrix when ``len(coords) == 2``.
    
    Examples
    --------
    **1D — cone-like metric** g = x²:
    
    >>> from sympy import symbols, Matrix, sin, simplify
    >>> x = symbols('x', real=True, positive=True)
    >>> m = Metric(x**2, (x,))
    >>> m.dim
    1
    >>> m.christoffel_sym          # Γ¹₁₁ = 1/x
    1/x
    >>> m.gauss_curvature()        # intrinsic curvature of a curve is 0
    0
    
    **2D — unit sphere** ds² = dθ² + sin²θ dφ²:
    
    >>> theta, phi = symbols('theta phi', real=True)
    >>> g = Matrix([[1, 0], [0, sin(theta)**2]])
    >>> m = Metric(g, (theta, phi))
    >>> m.dim
    2
    >>> simplify(m.gauss_curvature())   # K = 1 everywhere on the unit sphere
    1
    
    **2D — Poincaré half-plane** g = diag(1/y², 1/y²):
    
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1/y**2, 0], [0, 1/y**2]]), (x, y))
    >>> simplify(m.gauss_curvature())   # constant negative curvature
    -1
    """
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, g_input, coords):
        # Normalise coords to a tuple
        if isinstance(coords, (list, tuple)):
            self.coords = tuple(coords)
        else:
            self.coords = (coords,)

        self.dim = len(self.coords)

        if self.dim == 1:
            self._init_1d(g_input)
        elif self.dim == 2:
            self._init_2d(g_input)
        else:
            raise ValueError("Only 1D and 2D manifolds are supported.")

    # ---- 1D initialisation -------------------------------------------

    def _init_1d(self, g_expr):
        """
        Initialise all symbolic and numerical attributes for a 1D metric.

        Called internally by ``__init__`` when ``len(coords) == 1``.  Performs
        a single ``simplify`` pass on the user-supplied expression, then derives
        the inverse metric, the square-root determinant, and the lone Christoffel
        symbol Γ¹₁₁ = ½ (d/dx) log|g₁₁|.  Finally creates ``lambdify``-based
        numerical callables for every symbolic quantity.

        Parameters
        ----------
        g_expr : sympy.Expr
            Raw (un-simplified) metric component g₁₁(x).
        """
        x = self.coords[0]
        # One simplify pass on the input is sufficient; derived expressions
        # (g_inv, sqrt_det, Christoffel) are kept in raw symbolic form and
        # simplified only once at the end by lambdify's own canonicalisation.
        self.g_expr        = simplify(g_expr)
        self.g_inv_expr    = 1 / self.g_expr
        self.sqrt_det_expr = sqrt(abs(self.g_expr))

        # Christoffel: Γ¹₁₁ = ½ (log g₁₁)'
        self.christoffel_sym = diff(log(abs(self.g_expr)), x) / 2

        # Numerical lambdas
        self.g_func           = lambdify(x, self.g_expr,           'numpy')
        self.g_inv_func       = lambdify(x, self.g_inv_expr,       'numpy')
        self.sqrt_det_func    = lambdify(x, self.sqrt_det_expr,    'numpy')
        self.christoffel_func = lambdify(x, self.christoffel_sym,  'numpy')

        # Aliases used by dimension-agnostic helpers
        self._g_func_dict     = {(0, 0): self.g_func}
        self._g_inv_func_dict = {(0, 0): self.g_inv_func}

    # ---- 2D initialisation -------------------------------------------

    def _init_2d(self, g_matrix):
        """
        Initialise all symbolic and numerical attributes for a 2D metric.

        Called internally by ``__init__`` when ``len(coords) == 2``.  Accepts
        either a SymPy ``Matrix`` or a nested list/tuple (automatically promoted
        to a Matrix).  Performs a single ``simplify`` pass, computes the exact
        symbolic inverse and determinant, and builds all 8 Christoffel symbols
        Γⁱⱼₖ via ``_compute_christoffel_2d``.  Numerical callables are stored
        in dict form indexed by ``(i, j)`` or nested ``[i][j][k]``.

        Parameters
        ----------
        g_matrix : sympy.Matrix or list
            The 2×2 metric tensor.  A nested list is promoted to a Matrix.

        Raises
        ------
        ValueError
            If the provided matrix is not 2×2.
        """
        if not isinstance(g_matrix, Matrix):
            g_matrix = Matrix(g_matrix)
        if g_matrix.shape != (2, 2):
            raise ValueError("Metric requires a 2×2 matrix for dim=2.")

        x, y = self.coords
        # Simplify the user-supplied matrix once; derived quantities inherit
        # the simplified form without needing a second simplify pass.
        self.g_matrix    = simplify(g_matrix)
        self.det_g       = self.g_matrix.det()           # exact, no extra simplify
        self.sqrt_det_g  = sqrt(abs(self.det_g))
        self.g_inv_matrix = self.g_matrix.inv()          # exact inverse

        # Christoffel symbols Γⁱⱼₖ
        self.christoffel_sym = self._compute_christoffel_2d()

        # Numerical lambdas
        self.g_func = {
            (i, j): lambdify((x, y), self.g_matrix[i, j], 'numpy')
            for i in range(2) for j in range(2)
        }
        self.g_inv_func = {
            (i, j): lambdify((x, y), self.g_inv_matrix[i, j], 'numpy')
            for i in range(2) for j in range(2)
        }
        self.det_g_func      = lambdify((x, y), self.det_g,      'numpy')
        self.sqrt_det_g_func = lambdify((x, y), self.sqrt_det_g, 'numpy')

        # Christoffel funcs: dict[i][j][k]
        self.christoffel_func = {
            i: {
                j: {
                    k: lambdify((x, y), self.christoffel_sym[i][j][k], 'numpy')
                    for k in range(2)
                }
                for j in range(2)
            }
            for i in range(2)
        }

        # Aliases for uniform access
        self._g_func_dict     = self.g_func
        self._g_inv_func_dict = self.g_inv_func

    def _compute_christoffel_2d(self):
        """
        Compute all 2D Christoffel symbols of the second kind symbolically.

        Uses the standard formula

            Γⁱⱼₖ = ½ gⁱˡ (∂ⱼ gₖₗ + ∂ₖ gⱼₗ − ∂ₗ gⱼₖ)

        summing over the repeated index ℓ.  Each of the 8 independent
        components is simplified before storage.

        Returns
        -------
        dict
            Nested dict ``Gamma[i][j][k]`` of SymPy expressions, with
            indices i, j, k ∈ {0, 1} corresponding to (x, y) coordinates.
        """
        x, y = self.coords
        g = self.g_matrix
        g_inv = self.g_inv_matrix
        Gamma = {}
        for i in range(2):
            Gamma[i] = {}
            for j in range(2):
                Gamma[i][j] = {}
                for k in range(2):
                    expr = 0
                    for ell in range(2):
                        term1 = diff(g[k, ell], [x, y][j])
                        term2 = diff(g[j, ell], [x, y][k])
                        term3 = diff(g[j, k], [x, y][ell])
                        expr += g_inv[i, ell] * (term1 + term2 - term3) / 2
                    Gamma[i][j][k] = simplify(expr)
        return Gamma

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_hamiltonian(cls, H_expr, coords, momenta):
        """
        Construct a ``Metric`` by extracting the kinetic term from a Hamiltonian.

        For a free-particle (purely kinetic) Hamiltonian of the form

            H = ½ gⁱʲ(q) pᵢ pⱼ + V(q),

        the contravariant metric tensor gⁱʲ equals the Hessian of H with
        respect to the momenta.  This method computes that Hessian and inverts
        it to recover the covariant metric gᵢⱼ.  Any potential term V(q) that
        is independent of the momenta is automatically discarded.

        Parameters
        ----------
        H_expr : sympy.Expr
            Full Hamiltonian expressed in terms of both ``coords`` and
            ``momenta`` symbols.
        coords : tuple of sympy.Symbol
            Generalised position variables, length 1 (1D) or 2 (2D).
        momenta : tuple of sympy.Symbol
            Conjugate momentum variables, same length as ``coords``.

        Returns
        -------
        Metric
            A ``Metric`` instance whose dimension matches ``len(coords)``.

        Raises
        ------
        ValueError
            If ``len(coords)`` is neither 1 nor 2.

        Notes
        -----
        The extraction relies on ``∂²H/∂pᵢ∂pⱼ = gⁱʲ``, which is exact for
        quadratic kinetic terms.  Non-quadratic kinetic energy (e.g. relativistic
        Hamiltonians) will produce incorrect results.

        Examples
        --------
        **1D polar-like kinetic energy** H = p² / (2x²):

        >>> from sympy import symbols, simplify
        >>> x, p = symbols('x p', real=True)
        >>> H = p**2 / (2*x**2)
        >>> m = Metric.from_hamiltonian(H, (x,), (p,))
        >>> m.dim
        1
        >>> simplify(m.g_expr - x**2)    # recovers g₁₁ = x²
        0

        **2D polar coordinates** H = (p_r² + p_θ²/r²) / 2:

        >>> r, theta = symbols('r theta', real=True, positive=True)
        >>> pr, pt = symbols('p_r p_theta', real=True)
        >>> H = (pr**2 + pt**2/r**2) / 2
        >>> m = Metric.from_hamiltonian(H, (r, theta), (pr, pt))
        >>> m.dim
        2
        """
        coords = tuple(coords)
        momenta = tuple(momenta)
        n = len(coords)

        if n == 1:
            p = momenta[0]
            g_inv = diff(H_expr, p, 2)
            return cls(simplify(1 / g_inv), coords)
        elif n == 2:
            px, py = momenta
            g_inv_11 = diff(H_expr, px, 2)
            g_inv_12 = diff(diff(H_expr, px), py)
            g_inv_22 = diff(H_expr, py, 2)
            g_inv = Matrix([[g_inv_11, g_inv_12], [g_inv_12, g_inv_22]])
            return cls(simplify(g_inv.inv()), coords)
        else:
            raise ValueError("Only 1D and 2D Hamiltonians are supported.")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, *point):
        """
        Evaluate metric quantities numerically at a given coordinate point.

        All cached numerical callables are evaluated at the supplied point and
        collected into a single dict for convenient downstream use (e.g. inside
        ODE right-hand sides or distance computations).

        Parameters
        ----------
        *point : float or numpy.ndarray
            Coordinate values at which to evaluate.  Supply one argument for
            a 1D metric and two arguments for a 2D metric.  NumPy arrays are
            accepted, in which case all returned values are arrays of the same
            shape.

        Returns
        -------
        dict
            **1D** — keys and values:

            * ``'g'`` : float or ndarray — metric component g₁₁(x).
            * ``'g_inv'`` : float or ndarray — inverse metric g¹¹(x).
            * ``'sqrt_det'`` : float or ndarray — √|g₁₁(x)|.
            * ``'christoffel'`` : float or ndarray — Γ¹₁₁(x).

            **2D** — keys and values:

            * ``'g'`` : ndarray of shape (2, 2, ...) — full metric matrix.
            * ``'g_inv'`` : ndarray of shape (2, 2, ...) — inverse metric.
            * ``'det_g'`` : float or ndarray — det(g).
            * ``'sqrt_det'`` : float or ndarray — √|det(g)|.
            * ``'christoffel'`` : nested dict ``[i][j][k]`` — Γⁱⱼₖ values.

        Examples
        --------
        >>> x = symbols('x', real=True, positive=True)
        >>> m = Metric(x**2, (x,))
        >>> ev = m.eval(2.0)
        >>> ev['g']             # g₁₁(2) = 4
        4.0
        >>> ev['christoffel']   # Γ¹₁₁(2) = 1/2
        0.5
        """
        if self.dim == 1:
            x_val = point[0]
            return {
                'g': self.g_func(x_val),
                'g_inv': self.g_inv_func(x_val),
                'sqrt_det': self.sqrt_det_func(x_val),
                'christoffel': self.christoffel_func(x_val),
            }
        else:
            x_val, y_val = point
            g_arr = np.zeros((2, 2, *np.shape(x_val)))
            g_inv_arr = np.zeros_like(g_arr)
            for i in range(2):
                for j in range(2):
                    g_arr[i, j] = self.g_func[(i, j)](x_val, y_val)
                    g_inv_arr[i, j] = self.g_inv_func[(i, j)](x_val, y_val)
            christoffel_vals = {}
            for i in range(2):
                christoffel_vals[i] = {}
                for j in range(2):
                    christoffel_vals[i][j] = {}
                    for k in range(2):
                        christoffel_vals[i][j][k] = self.christoffel_func[i][j][k](x_val, y_val)
            return {
                'g': g_arr,
                'g_inv': g_inv_arr,
                'det_g': self.det_g_func(x_val, y_val),
                'sqrt_det': self.sqrt_det_g_func(x_val, y_val),
                'christoffel': christoffel_vals,
            }

    # ------------------------------------------------------------------
    # Curvature
    # ------------------------------------------------------------------

    def gauss_curvature(self):
        """
        Compute the Gaussian curvature K of the manifold.

        For a **1D** manifold (a curve), the intrinsic Gaussian curvature is
        identically zero and the method returns the SymPy integer ``0``
        immediately without any computation.

        For a **2D** surface the curvature is extracted from the Riemann tensor
        via

            K = R₁₂₁₂ / det(g),

        where R₁₂₁₂ = gₐ₁ Rᵃ₂₁₂ is the single independent component of the
        fully covariant Riemann tensor in 2D.

        Returns
        -------
        sympy.Expr
            Simplified symbolic expression for K as a function of the
            coordinates.  For flat metrics this simplifies to ``0``; for the
            unit sphere it simplifies to ``1``.

        Notes
        -----
        The Riemann tensor is recomputed each call (no caching).  For
        repeated curvature queries on the same metric consider caching the
        result yourself, e.g. ``K = simplify(m.gauss_curvature())``.

        Examples
        --------
        >>> from sympy import symbols, Matrix, sin, simplify
        >>> x, y = symbols('x y', real=True)
        >>> m_flat = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
        >>> m_flat.gauss_curvature()
        0

        >>> theta, phi = symbols('theta phi', real=True)
        >>> m_sphere = Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi))
        >>> simplify(m_sphere.gauss_curvature())
        1
        """
        if self.dim == 1:
            return sympify(0)
        R = self.riemann_tensor()
        g = self.g_matrix
        R_xyxy = g[0, 0] * R[0][1][0][1] + g[0, 1] * R[1][1][0][1]
        return simplify(R_xyxy / self.det_g)

    def riemann_tensor(self):
        """
        Compute the Riemann curvature tensor Rⁱⱼₖₗ (2D manifolds only).

        The tensor is computed from the Christoffel symbols via the standard
        formula

            Rⁱⱼₖₗ = ∂ₖΓⁱⱼₗ − ∂ₗΓⁱⱼₖ + ΓⁱₘₖΓᵐⱼₗ − ΓⁱₘₗΓᵐⱼₖ,

        with all 16 components evaluated and simplified.  In 2D the tensor has
        at most one independent component (R⁰₁₀₁ = −R⁰₁₁₀ = R¹₀₁₀ = …), and
        the Gaussian curvature is proportional to it.

        Returns
        -------
        dict
            Nested dict ``R[i][j][k][l]`` → SymPy expression for Rⁱⱼₖₗ, with
            all indices in {0, 1}.

        Raises
        ------
        NotImplementedError
            If called on a 1D metric (the Riemann tensor of a 1D manifold is
            identically zero and carries no information).

        Notes
        -----
        All 16 components are computed and returned; the user is responsible for
        exploiting symmetries (antisymmetry in k, l and in i, j when lowered)
        to reduce redundant evaluations.

        Examples
        --------
        >>> from sympy import symbols, Matrix, sin, simplify
        >>> theta, phi = symbols('theta phi', real=True)
        >>> m = Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi))
        >>> R = m.riemann_tensor()
        >>> simplify(R[0][1][0][1])   # non-zero component on unit sphere
        -sin(theta)**2
        """
        if self.dim == 1:
            raise NotImplementedError("Riemann tensor is zero for 1D manifolds.")
        x, y = self.coords
        Gamma = self.christoffel_sym
        R = {}
        for i in range(2):
            R[i] = {}
            for j in range(2):
                R[i][j] = {}
                for k in range(2):
                    R[i][j][k] = {}
                    for ell in range(2):
                        expr = diff(Gamma[i][j][ell], [x, y][k])
                        expr -= diff(Gamma[i][j][k], [x, y][ell])
                        for m in range(2):
                            expr += Gamma[i][m][k] * Gamma[m][j][ell]
                            expr -= Gamma[i][m][ell] * Gamma[m][j][k]
                        R[i][j][k][ell] = simplify(expr)
        return R

    def ricci_tensor(self):
        """
        Compute the Ricci tensor Rᵢⱼ (2D manifolds only).

        The Ricci tensor is the contraction

            Rᵢⱼ = Rᵏᵢₖⱼ

        of the Riemann tensor over the first and third indices.  In 2D the
        result is a symmetric 2×2 SymPy matrix.

        Returns
        -------
        sympy.Matrix
            Simplified 2×2 symmetric matrix of Ricci tensor components.

        Raises
        ------
        NotImplementedError
            If called on a 1D metric.

        Examples
        --------
        >>> from sympy import symbols, Matrix, sin, simplify
        >>> theta, phi = symbols('theta phi', real=True)
        >>> m = Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi))
        >>> Ric = m.ricci_tensor()
        >>> simplify(Ric[0, 0])   # R_θθ = 1 on the unit sphere
        1
        """
        if self.dim == 1:
            raise NotImplementedError("Ricci tensor is zero for 1D manifolds.")
        R_full = self.riemann_tensor()
        Ric = zeros(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    Ric[i, j] += R_full[k][i][k][j]
        return simplify(Ric)

    def ricci_scalar(self):
        """
        Compute the scalar (Ricci) curvature R.

        The scalar curvature is the full trace of the Ricci tensor with respect
        to the metric:

            R = gⁱʲ Rᵢⱼ.

        For **1D** manifolds the scalar curvature is identically zero and the
        method returns the SymPy integer ``0`` immediately.  For **2D** surfaces
        R = 2K, where K is the Gaussian curvature.

        Returns
        -------
        sympy.Expr
            Simplified symbolic expression for the scalar curvature.

        Notes
        -----
        This method internally calls :meth:`ricci_tensor`, which in turn calls
        :meth:`riemann_tensor`.  Both are recomputed on each invocation.

        Examples
        --------
        >>> from sympy import symbols, Matrix, sin, simplify
        >>> theta, phi = symbols('theta phi', real=True)
        >>> m = Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi))
        >>> simplify(m.ricci_scalar())   # R = 2 for the unit sphere
        2
        """
        if self.dim == 1:
            return sympify(0)
        Ric = self.ricci_tensor()
        g_inv = self.g_inv_matrix
        R = sum(g_inv[i, j] * Ric[i, j] for i in range(2) for j in range(2))
        return simplify(R)

    # ------------------------------------------------------------------
    # Laplace-Beltrami
    # ------------------------------------------------------------------

    def laplace_beltrami_symbol(self):
        """
        Compute the full symbol of the Laplace–Beltrami operator.

        The Laplace–Beltrami operator acting on smooth functions is

            Δ_g f = |g|^{-½} ∂ᵢ (|g|^{½} gⁱʲ ∂ⱼ f).

        Its **principal symbol** (the highest-order part) evaluated at
        cotangent vector ξ is the positive-definite quadratic form

            σ₂(Δ_g)(x, ξ) = gⁱʲ(x) ξᵢ ξⱼ.

        The **subprincipal symbol** captures the first-order (transport) part
        arising from the non-constancy of the metric.  The two are combined
        into a **full symbol** via

            σ_full = σ₂ + i σ₁     (microlocal convention),

        making the output ready for use in microlocal analysis or WKB
        approximations.

        Works for both 1D and 2D metrics.

        Returns
        -------
        dict with keys:

        * ``'principal'`` : sympy.Expr — σ₂(Δ_g)(x, ξ), a polynomial of
          degree 2 in the cotangent variables ξ (and η for 2D).
        * ``'subprincipal'`` : sympy.Expr — σ₁(Δ_g)(x, ξ), degree 1 in ξ.
        * ``'full'`` : sympy.Expr — σ₂ + i σ₁ (complex-valued symbol).

        Notes
        -----
        The cotangent variables are introduced as fresh SymPy symbols named
        ``xi`` (1D) or ``xi, eta`` (2D).  They do not conflict with the
        coordinate symbols.

        Examples
        --------
        **1D**, g = x²:

        >>> from sympy import symbols, simplify
        >>> x, xi = symbols('x xi', real=True, positive=True)
        >>> m = Metric(x**2, (x,))
        >>> lb = m.laplace_beltrami_symbol()
        >>> simplify(lb['principal'] - xi**2/x**2)
        0
        >>> simplify(lb['subprincipal'] - xi/x**3)
        0

        **2D**, polar coordinates g = diag(1, r²):

        >>> r, theta = symbols('r theta', real=True, positive=True)
        >>> xi_r, xi_t = symbols('xi eta', real=True)
        >>> m = Metric(Matrix([[1, 0], [0, r**2]]), (r, theta))
        >>> lb = m.laplace_beltrami_symbol()
        >>> simplify(lb['principal'] - (xi_r**2 + xi_t**2/r**2))
        0
        """
        if self.dim == 1:
            x = self.coords[0]
            xi = symbols('xi', real=True)
            principal    = self.g_inv_expr * xi**2
            log_sqrt_g   = log(self.sqrt_det_expr)
            # subprincipal = (d/dx log √g) * g^{-1} * ξ
            transport    = diff(log_sqrt_g, x) * self.g_inv_expr
            subprincipal = transport * xi
            # One simplify at the very end on the terms that matter
            p = simplify(principal)
            s = simplify(subprincipal)
            return {
                'principal':    p,
                'subprincipal': s,
                'full':         p + 1j * s,
            }
        else:
            x, y = self.coords
            xi, eta = symbols('xi eta', real=True)
            g_inv = self.g_inv_matrix
            principal = (g_inv[0, 0] * xi**2 +
                         2 * g_inv[0, 1] * xi * eta +
                         g_inv[1, 1] * eta**2)
            sqrt_g   = self.sqrt_det_g
            coeff_x  = diff(sqrt_g * g_inv[0, 0], x) + diff(sqrt_g * g_inv[0, 1], y)
            coeff_y  = diff(sqrt_g * g_inv[1, 0], x) + diff(sqrt_g * g_inv[1, 1], y)
            subprincipal = simplify((coeff_x * xi + coeff_y * eta) / sqrt_g)
            return {
                'principal':    simplify(principal),
                'subprincipal': subprincipal,
                'full':         simplify(principal + 1j * subprincipal),
            }

    # ------------------------------------------------------------------
    # Volume / arc length
    # ------------------------------------------------------------------

    def riemannian_volume(self, domain, method='numerical'):
        """
        Compute the Riemannian volume (area / arc length) of a rectangular domain.

        The Riemannian volume element is dV = √|det(g)| dx¹ ∧ … ∧ dxⁿ.  For a
        1D metric this equals the arc length ∫ √g₁₁ dx; for a 2D metric it
        equals the surface area ∫∫ √|det(g)| dx dy.

        Parameters
        ----------
        domain : tuple
            * **1D**: ``(x_min, x_max)`` — a pair of floats or SymPy numbers
              defining the integration interval.
            * **2D**: ``((x_min, x_max), (y_min, y_max))`` — a pair of pairs
              defining the rectangular integration region.
        method : {'symbolic', 'numerical'}, default 'numerical'
            * ``'symbolic'``: uses SymPy's ``integrate`` for an exact result.
              May be slow or fail to close for complicated metrics.
            * ``'numerical'``: uses ``scipy.integrate.quad`` (1D) or
              ``scipy.integrate.dblquad`` (2D) for a fast floating-point
              approximation.

        Returns
        -------
        sympy.Expr or float
            * Symbolic result (SymPy expression or number) when
              ``method='symbolic'``.
            * Float when ``method='numerical'``.

        Raises
        ------
        ValueError
            If ``method`` is not ``'symbolic'`` or ``'numerical'``.

        Examples
        --------
        **1D** — arc length of g = 1/x² on [1, e]:

        >>> import numpy as np
        >>> x = symbols('x', real=True, positive=True)
        >>> m = Metric(1/x**2, (x,))
        >>> float(m.riemannian_volume((1, np.e), method='symbolic'))  # = 1
        1.0
        >>> m.riemannian_volume((1, np.e), method='numerical')         # ≈ 1
        0.9999...

        **2D** — area of g = diag(4, 9) on [0,1]²:

        >>> x, y = symbols('x y', real=True)
        >>> m = Metric(Matrix([[4, 0], [0, 9]]), (x, y))
        >>> m.riemannian_volume(((0, 1), (0, 1)), method='symbolic')   # = 6
        6
        """
        if self.dim == 1:
            x_min, x_max = domain
            x = self.coords[0]
            if method == 'symbolic':
                return integrate(self.sqrt_det_expr, (x, x_min, x_max))
            elif method == 'numerical':
                result, _ = quad(self.sqrt_det_func, x_min, x_max)
                return result
            else:
                raise ValueError("method must be 'symbolic' or 'numerical'")
        else:
            (x_min, x_max), (y_min, y_max) = domain
            x, y = self.coords
            sqrt_g = self.sqrt_det_g
            if method == 'symbolic':
                return integrate(sqrt_g, (x, x_min, x_max), (y, y_min, y_max))
            elif method == 'numerical':
                integrand = lambda yv, xv: self.sqrt_det_g_func(xv, yv)
                result, _ = dblquad(integrand, x_min, x_max, y_min, y_max)
                return result
            else:
                raise ValueError("method must be 'symbolic' or 'numerical'")

    def arc_length(self, x_min, x_max, method='numerical'):
        """
        Compute the arc length of a 1D metric between two coordinate values.

        Convenience wrapper around :meth:`riemannian_volume` for the 1D case.
        The arc length is

            L = ∫_{x_min}^{x_max} √g₁₁(x) dx.

        Parameters
        ----------
        x_min, x_max : float or sympy.Expr
            Integration bounds along the coordinate x.
        method : {'symbolic', 'numerical'}, default 'numerical'
            Passed directly to :meth:`riemannian_volume`.

        Returns
        -------
        sympy.Expr or float
            Arc length, symbolic or numerical depending on ``method``.

        Raises
        ------
        NotImplementedError
            If called on a 2D metric.  Use :meth:`riemannian_volume` with a
            2D domain for surface area computations.
        """
        if self.dim != 1:
            raise NotImplementedError("arc_length is defined for 1D metrics only.")
        return self.riemannian_volume((x_min, x_max), method=method)

    def covariant_derivative_vector(self, V_components, do_simplify=True):
        """
        Compute the covariant derivative ∇V of a vector field (2D only).

        The covariant derivative of a contravariant vector field V = Vʲ ∂ⱼ in
        the direction of the coordinate basis vector ∂ᵢ is

            (∇V)ⁱⱼ = ∂ᵢ Vʲ + Γʲᵢₖ Vᵏ,

        where Γʲᵢₖ are the Christoffel symbols of the second kind.

        Parameters
        ----------
        V_components : list or tuple of two sympy.Expr
            Upper-index components (V¹, V²) of the vector field, expressed as
            functions of the coordinate symbols.
        do_simplify : bool, default True
            If ``True``, each entry of the resulting matrix is passed through
            ``sympy.simplify`` before being returned.  Set to ``False`` for
            faster (but potentially unsimplified) output.

        Returns
        -------
        sympy.Matrix
            A 2×2 matrix whose (i, j)-entry is ∇ᵢ Vʲ, i.e. the j-th
            component of the covariant derivative in the i-th coordinate
            direction.  Row index i = 0, 1 is the differentiation direction
            (lower); column index j = 0, 1 is the vector component (upper).

        Raises
        ------
        NotImplementedError
            If called on a 1D metric.

        Examples
        --------
        >>> from sympy import symbols, Matrix
        >>> x, y = symbols('x y', real=True)
        >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))   # flat metric
        >>> nabla_V = m.covariant_derivative_vector([x, y])
        >>> nabla_V   # equals the ordinary Jacobian for flat space
        Matrix([[1, 0], [0, 1]])
        """
        if self.dim != 2:
            raise NotImplementedError("Only 2D metrics are supported.")
        x, y = self.coords
        V1, V2 = V_components
        # partial derivatives
        dV = [[diff(V1, x), diff(V2, x)],   # ∂_x V^1, ∂_x V^2
              [diff(V1, y), diff(V2, y)]]   # ∂_y V^1, ∂_y V^2
    
        Gamma = self.christoffel_sym        # Gamma[i][j][k] = Γ^i_{jk}
        nabla = zeros(2, 2)
        for i in (0, 1):         # direction (lower index)
            for j in (0, 1):     # component (upper index)
                term = dV[i][j]   # ∂_i V^j
                # Γ^j_{i,k} * V^k
                for k in (0, 1):
                    term += Gamma[j][i][k] * (V1 if k == 0 else V2)
                nabla[i, j] = term
        return simplify(nabla) if do_simplify else nabla
    
    def covariant_derivative_covector(self, omega_components, do_simplify=True):
        """
        Compute the covariant derivative ∇ω of a covector field (2D only).

        The covariant derivative of a covariant 1-form ω = ωⱼ dxʲ in the
        direction of the coordinate basis vector ∂ᵢ is

            (∇ω)ᵢⱼ = ∂ᵢ ωⱼ − Γᵏᵢⱼ ωₖ,

        where the Christoffel term accounts for the change of basis under
        parallel transport.

        Parameters
        ----------
        omega_components : list or tuple of two sympy.Expr
            Lower-index components (ω₁, ω₂) of the 1-form, expressed as
            functions of the coordinate symbols.
        do_simplify : bool, default True
            If ``True``, each entry of the resulting matrix is passed through
            ``sympy.simplify`` before being returned.  Set to ``False`` for
            faster (but potentially unsimplified) output.

        Returns
        -------
        sympy.Matrix
            A 2×2 matrix whose (i, j)-entry is ∇ᵢ ωⱼ, i.e. the covariant
            derivative of the j-th component of ω in the i-th coordinate
            direction.  Both row index i and column index j are lower (covariant)
            indices taking values in {0, 1}.

        Raises
        ------
        NotImplementedError
            If called on a 1D metric.

        Examples
        --------
        >>> from sympy import symbols, Matrix
        >>> x, y = symbols('x y', real=True)
        >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))   # flat metric
        >>> nabla_omega = m.covariant_derivative_covector([x**2, y**2])
        >>> nabla_omega   # Christoffel terms vanish on flat space
        Matrix([[2*x, 0], [0, 2*y]])
        """
        if self.dim != 2:
            raise NotImplementedError("Only 2D metrics are supported.")
        x, y = self.coords
        w1, w2 = omega_components
        # partial derivatives
        dw = [[diff(w1, x), diff(w2, x)],   # ∂_x ω₁, ∂_x ω₂
              [diff(w1, y), diff(w2, y)]]   # ∂_y ω₁, ∂_y ω₂
    
        Gamma = self.christoffel_sym        # Gamma[i][j][k] = Γ^i_{jk}
        nabla = zeros(2, 2)
        for i in (0, 1):         # direction (lower index)
            for j in (0, 1):     # component (lower index)
                term = dw[i][j]   # ∂_i ω_j
                # - Γ^k_{i,j} * ω_k
                for k in (0, 1):
                    term -= Gamma[k][i][j] * (w1 if k == 0 else w2)
                nabla[i, j] = term
        return simplify(nabla) if do_simplify else nabla


    def riemannian_gradient(self, f_expr, do_simplify=True):
        """
        Compute the gradient of a scalar function f as a contravariant vector field.

        In local coordinates: ∇f = gⁱʲ ∂ⱼ f ∂ᵢ.

        Parameters
        ----------
        f_expr : sympy.Expr
            The scalar function expressed in the metric's coordinates.
        do_simplify : bool, default True
            Whether to simplify the resulting expressions.

        Returns
        -------
        tuple or dict
            - 1D: a single SymPy expression for the gradient component (∇f)¹.
            - 2D: a tuple (∇f¹, ∇f²) of SymPy expressions.
        """
        if self.dim == 1:
            x = self.coords[0]
            grad_expr = self.g_inv_expr * diff(f_expr, x)
            return simplify(grad_expr) if do_simplify else grad_expr
        else:
            x, y = self.coords
            grad1 = self.g_inv_matrix[0,0]*diff(f_expr, x) + self.g_inv_matrix[0,1]*diff(f_expr, y)
            grad2 = self.g_inv_matrix[1,0]*diff(f_expr, x) + self.g_inv_matrix[1,1]*diff(f_expr, y)
            if do_simplify:
                grad1 = simplify(grad1)
                grad2 = simplify(grad2)
            return (grad1, grad2)

    def riemannian_hessian(self, f_expr, do_simplify=True):
        """
        Compute the Hessian of a scalar function f as a (0,2)-tensor (covariant).

        In local coordinates: Hess(f)ᵢⱼ = ∂ᵢ∂ⱼ f − Γᵏᵢⱼ ∂ₖ f.

        Parameters
        ----------
        f_expr : sympy.Expr
            The scalar function expressed in the metric's coordinates.
        do_simplify : bool, default True
            Whether to simplify the resulting expressions.

        Returns
        -------
        sympy.Matrix (2D) or sympy.Expr (1D)
            The Hessian matrix (2×2) for 2D, or the single component H₁₁ for 1D.
        """
        if self.dim == 1:
            x = self.coords[0]
            f1 = diff(f_expr, x)
            f2 = diff(f_expr, x, x)
            Gamma = self.christoffel_sym   # Γ¹₁₁
            H = f2 - Gamma * f1
            return simplify(H) if do_simplify else H
        else:
            x, y = self.coords
            f1 = diff(f_expr, x)
            f2 = diff(f_expr, y)
            f11 = diff(f_expr, x, x)
            f12 = diff(f_expr, x, y)
            f22 = diff(f_expr, y, y)
            Gamma = self.christoffel_sym  # dict Gamma[i][j][k]

            H = zeros(2, 2)
            for i in range(2):
                for j in range(2):
                    term = diff(f_expr, [x, y][i], [x, y][j])
                    # subtract Γ^k_{ij} ∂_k f
                    for k in range(2):
                        term -= Gamma[k][i][j] * (f1 if k==0 else f2)
                    H[i,j] = term
            return simplify(H) if do_simplify else H

# ============================================================================
# Stand-alone helper functions (dimension-dispatching)
# ============================================================================

def christoffel(metric):
    """
    Return the pre-computed numerical Christoffel symbol callable(s) of a metric.

    This is a thin convenience accessor that exposes ``metric.christoffel_func``
    without needing to remember the attribute name.

    Parameters
    ----------
    metric : Metric
        A ``Metric`` instance of any dimension.

    Returns
    -------
    callable or dict
        * **1D**: a single callable ``Gamma(x_val)`` returning the scalar
          Γ¹₁₁(x) as a float or ndarray.
        * **2D**: a nested dict ``Gamma[i][j][k]`` of callables, each mapping
          ``(x_val, y_val)`` to Γⁱⱼₖ as a float or ndarray.

    Examples
    --------
    >>> x = symbols('x', real=True, positive=True)
    >>> m = Metric(x**2, (x,))
    >>> G = christoffel(m)
    >>> G(2.0)    # Γ¹₁₁(2) = 1/2
    0.5
    """
    return metric.christoffel_func


def geodesic_solver(metric, p0, v0, tspan, method='rk4', n_steps=1000,
                    reparametrize=False):
    """
    Integrate the geodesic equation on a 1D or 2D Riemannian manifold.

    The geodesic equation in local coordinates is

        ẍⁱ + Γⁱⱼₖ ẋʲ ẋᵏ = 0,

    supplemented by the initial conditions x(0) = p0, ẋ(0) = v0.  This
    function dispatches automatically to ``_geodesic_1d`` or ``_geodesic_2d``
    based on ``metric.dim``.

    Parameters
    ----------
    metric : Metric
        The Riemannian metric defining the manifold.
    p0 : float (1D) or tuple of two floats (2D)
        Initial position.  For 2D pass ``(x₀, y₀)``.
    v0 : float (1D) or tuple of two floats (2D)
        Initial velocity / tangent vector.  For 2D pass ``(vₓ₀, vᵧ₀)``.
    tspan : tuple of two floats
        Integration interval ``(t_start, t_end)``.
    method : str, default ``'rk4'``
        Numerical integration scheme.  Available methods by dimension:

        * **1D**: ``'rk4'`` (RK23 via SciPy), ``'adaptive'`` (RK45),
          ``'symplectic'`` (leapfrog/Störmer–Verlet in phase space).
        * **2D**: ``'rk45'`` (RK45), ``'rk4'`` (RK23), ``'symplectic'``
          or ``'verlet'`` (Störmer–Verlet via Hamiltonian flow).
    n_steps : int, default 1000
        Number of time steps (evaluation points for fixed-step methods;
        used as ``t_eval`` grid for adaptive methods).
    reparametrize : bool, default False
        **2D only.** If ``True``, also compute the accumulated arc-length
        parameter s(t) = ∫₀ᵗ ‖ẋ(τ)‖_g dτ and include it in the output dict
        under the key ``'arc_length'``.

    Returns
    -------
    dict
        Trajectory arrays.  Keys depend on dimension and method:

        * **1D**: ``'t'``, ``'x'``, ``'v'``.  Symplectic method additionally
          includes ``'p'`` (canonical momentum).
        * **2D**: ``'t'``, ``'x'``, ``'y'``, ``'vx'``, ``'vy'``.  Hamiltonian
          methods additionally include ``'px'``, ``'py'``, ``'energy'``.
          If ``reparametrize=True``, also includes ``'arc_length'``.

    Raises
    ------
    ValueError
        If ``method`` is not recognised for the given dimension.

    Examples
    --------
    **1D flat metric** — straight-line geodesic:

    >>> import numpy as np
    >>> x = symbols('x', real=True)
    >>> m = Metric(1, (x,))
    >>> traj = geodesic_solver(m, 0.0, 2.0, (0, 3.0))
    >>> np.isclose(traj['x'][-1], 6.0, rtol=1e-4)
    True

    **2D Euclidean metric** with arc-length reparametrisation:

    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> traj = geodesic_solver(m, (0, 0), (1, 1), (0, 5), reparametrize=True)
    >>> 'arc_length' in traj
    True
    """
    if metric.dim == 1:
        return _geodesic_1d(metric, p0, v0, tspan, method, n_steps)
    else:
        return _geodesic_2d(metric, p0, v0, tspan, method, n_steps, reparametrize)


def _geodesic_1d(metric, x0, v0, tspan, method, n_steps):
    """
    Internal geodesic integrator for 1D metrics.

    Solves the scalar ODE system

        ẋ = v,   v̇ = −Γ¹₁₁(x) v²,

    using the scheme selected by ``method``:

    * ``'rk4'``: fixed-step RK23 (SciPy RK23 with uniform ``t_eval``).
    * ``'adaptive'``: adaptive RK45 (SciPy RK45).
    * ``'symplectic'``: first-order leapfrog in the (x, p) phase space,
      with momentum p = g₁₁(x) v.  Exactly preserves the symplectic 2-form
      and provides better long-time energy conservation than the Runge–Kutta
      methods at the same step size.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 1``.
    x0, v0 : float
        Initial position and velocity.
    tspan : tuple of two floats
        Integration interval.
    method : {'rk4', 'adaptive', 'symplectic'}
    n_steps : int
        Number of time steps (or evaluation points for adaptive methods).

    Returns
    -------
    dict
        Keys ``'t'``, ``'x'``, ``'v'``, and ``'p'`` (symplectic only).
    """
    Gamma_func = metric.christoffel_func

    def ode(t, y):
        x, v = y
        return [v, -Gamma_func(x) * v**2]

    if method in ('rk4', 'adaptive'):
        sol = solve_ivp(
            ode, tspan, [x0, v0],
            method='RK23' if method == 'rk4' else 'RK45',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps)
        )
        return {'t': sol.t, 'x': sol.y[0], 'v': sol.y[1]}

    elif method == 'symplectic':
        dt = (tspan[1] - tspan[0]) / n_steps
        t_vals = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        p_vals = np.zeros(n_steps)

        x_vals[0] = x0
        # p = v / g⁻¹  →  p = v * g
        p_vals[0] = v0 / metric.g_inv_func(x0)

        g_inv_prime = lambdify(
            metric.coords[0],
            diff(metric.g_inv_expr, metric.coords[0]),
            'numpy'
        )

        for i in range(n_steps - 1):
            x = x_vals[i]
            p = p_vals[i]
            p_new = p - dt * 0.5 * g_inv_prime(x) * p**2
            x_new = x + dt * metric.g_inv_func(x) * p_new
            x_vals[i + 1] = x_new
            p_vals[i + 1] = p_new

        v_vals = np.array([metric.g_inv_func(xi) * pi
                           for xi, pi in zip(x_vals, p_vals)])
        return {'t': t_vals, 'x': x_vals, 'v': v_vals, 'p': p_vals}

    else:
        raise ValueError("1D method must be 'rk4', 'adaptive', or 'symplectic'.")


def _geodesic_2d(metric, p0, v0, tspan, method, n_steps, reparametrize):
    """
    Internal geodesic integrator for 2D metrics.

    Solves the system of four first-order ODEs

        ẋ = vₓ,   ẏ = vᵧ,
        v̇ₓ = −(Γ⁰₀₀ vₓ² + 2Γ⁰₀₁ vₓvᵧ + Γ⁰₁₁ vᵧ²),
        v̇ᵧ = −(Γ¹₀₀ vₓ² + 2Γ¹₀₁ vₓvᵧ + Γ¹₁₁ vᵧ²),

    using the scheme selected by ``method``:

    * ``'rk45'``: adaptive RK45 via ``scipy.integrate.solve_ivp``.
    * ``'rk4'``: fixed-step RK23 via ``scipy.integrate.solve_ivp``.
    * ``'symplectic'`` / ``'verlet'``: Störmer–Verlet in phase space via
      :func:`geodesic_hamiltonian_flow`.

    If ``reparametrize=True``, the arc-length parameter

        s(t) = ∫₀ᵗ √(gᵢⱼ ẋⁱ ẋʲ) dτ

    is appended to the result dict as ``'arc_length'`` using cumulative
    trapezoidal integration.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    p0, v0 : tuple of two floats
        Initial position ``(x₀, y₀)`` and velocity ``(vₓ₀, vᵧ₀)``.
    tspan : tuple of two floats
    method : {'rk45', 'rk4', 'symplectic', 'verlet'}
    n_steps : int
    reparametrize : bool

    Returns
    -------
    dict
        Keys ``'t'``, ``'x'``, ``'y'``, ``'vx'``, ``'vy'``, and optionally
        ``'arc_length'``, ``'px'``, ``'py'``, ``'energy'``.
    """
    Gamma = metric.christoffel_func

    def ode(t, state):
        x, y, vx, vy = state
        ax = -(Gamma[0][0][0](x, y) * vx**2 +
               2 * Gamma[0][0][1](x, y) * vx * vy +
               Gamma[0][1][1](x, y) * vy**2)
        ay = -(Gamma[1][0][0](x, y) * vx**2 +
               2 * Gamma[1][0][1](x, y) * vx * vy +
               Gamma[1][1][1](x, y) * vy**2)
        return [vx, vy, ax, ay]

    if method in ('rk45', 'rk4'):
        sol = solve_ivp(
            ode, tspan, [p0[0], p0[1], v0[0], v0[1]],
            method='RK45' if method == 'rk45' else 'RK23',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps)
        )
        result = {
            't': sol.t,
            'x': sol.y[0], 'y': sol.y[1],
            'vx': sol.y[2], 'vy': sol.y[3]
        }

    elif method in ('symplectic', 'verlet'):
        result = geodesic_hamiltonian_flow(
            metric, p0, v0, tspan, method='verlet', n_steps=n_steps
        )

    else:
        raise ValueError("2D method must be 'rk45', 'rk4', 'symplectic', or 'verlet'.")

    if reparametrize:
        ds = np.sqrt(
            metric.g_func[(0, 0)](result['x'], result['y']) * result['vx']**2 +
            2 * metric.g_func[(0, 1)](result['x'], result['y']) * result['vx'] * result['vy'] +
            metric.g_func[(1, 1)](result['x'], result['y']) * result['vy']**2
        )
        result['arc_length'] = cumulative_trapezoid(ds, result['t'], initial=0)

    return result

def riemannian_gradient_func(self, f_expr):
    grad_expr = self.riemannian_gradient(f_expr)
    if self.dim == 1:
        return lambdify(self.coords, grad_expr, 'numpy')
    else:
        return [lambdify(self.coords, grad_expr[i], 'numpy') for i in range(2)]

def riemannian_hessian_func(self, f_expr):
    H_expr = self.riemannian_hessian(f_expr)
    if self.dim == 1:
        return lambdify(self.coords, H_expr, 'numpy')
    else:
        return [[lambdify(self.coords, H_expr[i,j], 'numpy') for j in range(2)] for i in range(2)]

def geodesic_hamiltonian_flow(metric, p0, v0, tspan, method='verlet', n_steps=1000):
    """
    Integrate geodesic flow using the Hamiltonian formulation with symplectic schemes.

    Reformulates the geodesic equation as a Hamiltonian system with the
    kinetic-energy Hamiltonian

        H = ½ gⁱʲ(q) pᵢ pⱼ,

    where pᵢ = gᵢⱼ ẋʲ are the canonical momenta conjugate to the coordinates
    qⁱ.  Hamilton's equations

        q̇ⁱ = ∂H/∂pᵢ = gⁱʲ pⱼ,   ṗᵢ = −∂H/∂qⁱ

    are then integrated by the companion ``symplectic.hamiltonian_flow`` module.
    Initial velocities are converted to momenta automatically.

    Parameters
    ----------
    metric : Metric
        A 1D or 2D Riemannian metric.
    p0 : float (1D) or tuple of two floats (2D)
        Initial position in configuration space.
    v0 : float (1D) or tuple of two floats (2D)
        Initial velocity, converted internally to canonical momentum via
        pᵢ = gᵢⱼ(p0) v0ʲ.
    tspan : tuple of two floats
        Integration interval ``(t_start, t_end)``.
    method : str, default ``'verlet'``
        Symplectic integration scheme forwarded to ``symplectic.hamiltonian_flow``.
        The mapping from user-facing names to internal names is:

        * ``'verlet'`` → Störmer–Verlet (2nd order, time-reversible).
        * ``'stormer'`` → same as ``'verlet'`` (alias).
        * ``'symplectic'`` → same as ``'verlet'`` (backward-compatible alias).
        * ``'symplectic_euler'`` → symplectic Euler (1st order).
        * ``'rk45'`` → Dormand–Prince (not symplectic, but high accuracy).
    n_steps : int, default 1000
        Number of integration steps.

    Returns
    -------
    dict
        Phase-space trajectory.  Keys by dimension:

        * **1D**: ``'t'``, ``'x'``, ``'v'``, ``'p'``, ``'energy'``.
        * **2D**: ``'t'``, ``'x'``, ``'y'``, ``'vx'``, ``'vy'``,
          ``'px'``, ``'py'``, ``'energy'``.

        The ``'energy'`` array contains H evaluated at each time step;
        for a perfect symplectic integrator it is exactly conserved.

    Raises
    ------
    ValueError
        If ``method`` is not in the allowed list.
    NotImplementedError
        If ``metric.dim`` is neither 1 nor 2.

    Notes
    -----
    Symplectic integrators conserve a modified (shadow) Hamiltonian exactly,
    so ``energy`` will exhibit bounded oscillations rather than secular drift.
    The Störmer–Verlet scheme is second-order accurate and time-reversible,
    making it the default choice for long-time integrations.

    Examples
    --------
    **1D energy conservation** over 10 time units:

    >>> x = symbols('x', real=True, positive=True)
    >>> m = Metric(x**2, (x,))
    >>> res = geodesic_hamiltonian_flow(m, 2.0, 2.5, (0, 10), method='verlet', n_steps=2000)
    >>> import numpy as np
    >>> np.std(res['energy']) / res['energy'][0] < 0.01   # < 1% drift
    True
    """
    # Map method names to those understood by symplectic.hamiltonian_flow
    method_map = {
        'verlet': 'verlet',
        'stormer': 'verlet',          # identical
        'symplectic': 'verlet',        # treat 'symplectic' as second-order Verlet (original behaviour)
        'symplectic_euler': 'symplectic',  # first-order symplectic Euler
    }
    if method not in method_map and method not in ('rk45',):
        raise ValueError(f"Unknown method '{method}'. Allowed: verlet, stormer, symplectic, symplectic_euler, rk45")
    integrator = method_map.get(method, method)   # pass through if not in map (e.g. 'rk45')

    # Build Hamiltonian expression and initial state
    if metric.dim == 1:
        x = metric.coords[0]
        p_sym = symbols('p', real=True)
        vars_phase = [x, p_sym]
        H_expr = (metric.g_inv_expr * p_sym**2) / 2

        # Convert velocity to momentum: p = g * v
        g0 = metric.g_func(p0)          # p0 is the initial position
        p0_mom = float(g0 * v0)          # ensure scalar
        z0 = [p0, p0_mom]

        # Call unified Hamiltonian flow
        traj = symp_hamiltonian_flow(H_expr, z0, tspan,
                                      vars_phase=vars_phase,
                                      integrator=integrator,
                                      n_steps=n_steps)

        # Post‑process: extract arrays and compute velocities
        x_vals = traj[str(x)]
        p_vals = traj[str(p_sym)]
        # velocity = g_inv * p
        v_vals = metric.g_inv_func(x_vals) * p_vals
        energy = traj['energy']

        return {
            't': traj['t'],
            'x': x_vals,
            'v': v_vals,
            'p': p_vals,
            'energy': energy
        }

    elif metric.dim == 2:
        x, y = metric.coords
        px_sym, py_sym = symbols('px py', real=True)
        vars_phase = [x, y, px_sym, py_sym]

        g_inv = metric.g_inv_matrix
        H_expr = 0.5 * (g_inv[0,0] * px_sym**2 +
                        2 * g_inv[0,1] * px_sym * py_sym +
                        g_inv[1,1] * py_sym**2)

        # Convert velocity to momentum: p = g · v
        g_eval = metric.eval(p0[0], p0[1])
        g_mat = g_eval['g']
        p_mom = g_mat @ v0                     # shape (2,)
        z0 = [p0[0], p0[1], p_mom[0], p_mom[1]]

        traj = symp_hamiltonian_flow(H_expr, z0, tspan,
                                      vars_phase=vars_phase,
                                      integrator=integrator,
                                      n_steps=n_steps)

        x_vals  = traj[str(x)]
        y_vals  = traj[str(y)]
        px_vals = traj[str(px_sym)]
        py_vals = traj[str(py_sym)]

        # Vectorised velocity recovery: v = g_inv · p at each point.
        # Evaluate the four g_inv components as arrays, then apply the
        # 2×2 linear map without a Python for-loop.
        g00 = metric.g_inv_func[(0, 0)](x_vals, y_vals)
        g01 = metric.g_inv_func[(0, 1)](x_vals, y_vals)
        g10 = metric.g_inv_func[(1, 0)](x_vals, y_vals)
        g11 = metric.g_inv_func[(1, 1)](x_vals, y_vals)
        vx_vals = g00 * px_vals + g01 * py_vals
        vy_vals = g10 * px_vals + g11 * py_vals

        energy = traj['energy']

        return {
            't': traj['t'],
            'x': x_vals,
            'y': y_vals,
            'vx': vx_vals,
            'vy': vy_vals,
            'px': px_vals,
            'py': py_vals,
            'energy': energy
        }

    else:
        raise NotImplementedError("geodesic_hamiltonian_flow only supports 1D and 2D metrics.")

def laplace_beltrami(metric):
    """
    Return the symbol dict of the Laplace–Beltrami operator for a metric.

    Convenience wrapper around :meth:`Metric.laplace_beltrami_symbol`.  The
    returned dict contains the principal symbol (the inverse-metric quadratic
    form in the cotangent variables), the subprincipal symbol (the transport
    term), and their complex combination as the full microlocal symbol.

    Parameters
    ----------
    metric : Metric
        A 1D or 2D Riemannian metric.

    Returns
    -------
    dict
        With keys ``'principal'``, ``'subprincipal'``, and ``'full'``.
        See :meth:`Metric.laplace_beltrami_symbol` for full documentation.

    Examples
    --------
    >>> x = symbols('x', real=True, positive=True)
    >>> m = Metric(x**2, (x,))
    >>> lb = laplace_beltrami(m)
    >>> lb['principal']   # g¹¹ ξ² = ξ²/x²
    xi**2/x**2
    """
    return metric.laplace_beltrami_symbol()


# ============================================================================
# 1D-only helpers
# ============================================================================

def sturm_liouville_reduce(metric, potential_expr=None):
    """
    Reduce the Laplace–Beltrami eigenvalue problem to Sturm–Liouville form (1D only).

    The Laplace–Beltrami eigenvalue problem on a 1D Riemannian manifold,

        −Δ_g u + V u = λ u,

    is equivalent to the classical Sturm–Liouville problem

        −(p u′)′ + q u = λ w u,

    with weight function w = √g, coefficient p = √g · g¹¹, and potential
    coefficient q = V √g.  This standard form enables use of classical
    spectral theory, finite-element methods, and Sturm–Liouville solvers.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 1``.
    potential_expr : sympy.Expr or None, default None
        Optional potential V(x).  Pass ``None`` (default) for the pure
        Laplace–Beltrami operator (V = 0).

    Returns
    -------
    dict with keys:

    * ``'p'`` : sympy.Expr — coefficient p(x) = √g · g¹¹.
    * ``'q'`` : sympy.Expr — coefficient q(x) = V(x) √g (zero if no potential).
    * ``'w'`` : sympy.Expr — weight function w(x) = √g.
    * ``'p_func'`` : callable — numerical p(x_val).
    * ``'q_func'`` : callable — numerical q(x_val).
    * ``'w_func'`` : callable — numerical w(x_val).

    Raises
    ------
    NotImplementedError
        If called on a 2D metric.

    Examples
    --------
    **Cone metric** g = x², no potential:

    >>> x = symbols('x', real=True, positive=True)
    >>> m = Metric(x**2, (x,))
    >>> sl = sturm_liouville_reduce(m)
    >>> simplify(sl['p'])   # p = √(x²) · (1/x²) = 1/x
    1/x
    >>> simplify(sl['w'])   # w = √(x²) = x
    x
    >>> sl['q']             # q = 0 (no potential)
    0
    """
    if metric.dim != 1:
        raise NotImplementedError("sturm_liouville_reduce is for 1D metrics only.")
    x = metric.coords[0]
    sqrt_g = metric.sqrt_det_expr
    g_inv = metric.g_inv_expr

    p_expr = simplify(sqrt_g * g_inv)
    w_expr = sqrt_g
    q_expr = sympify(0) if potential_expr is None else simplify(potential_expr * sqrt_g)

    return {
        'p': p_expr, 'q': q_expr, 'w': w_expr,
        'p_func': lambdify(x, p_expr, 'numpy'),
        'q_func': lambdify(x, q_expr, 'numpy'),
        'w_func': lambdify(x, w_expr, 'numpy'),
    }


# ============================================================================
# 2D-only helpers
# ============================================================================

def exponential_map(metric, p, v, t=1.0, method='rk45'):
    """
    Evaluate the Riemannian exponential map exp_p(t·v) on a 2D manifold.

    The exponential map sends the tangent vector v at base point p along the
    unique unit-speed geodesic γ with γ(0) = p and γ̇(0) = v, returning the
    point γ(t).  Geodesic completeness is assumed; the result may be
    inaccurate near conjugate loci.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    p : tuple of two floats
        Base point (x₀, y₀) in the manifold.
    v : tuple of two floats
        Initial tangent vector (vₓ, vᵧ) at p.  The geodesic is parametrised
        so that ‖γ̇(0)‖_g = ‖v‖_g, i.e. the speed equals the norm of v.
    t : float, default 1.0
        Parameter value at which to evaluate the geodesic.
    method : str, default ``'rk45'``
        Geodesic integration method passed to :func:`geodesic_solver`.

    Returns
    -------
    tuple of two floats
        End point (x(t), y(t)) of the geodesic.

    Raises
    ------
    NotImplementedError
        If called on a 1D metric.

    Examples
    --------
    **Flat metric** — exp_p(tv) = p + tv:

    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> import numpy as np
    >>> end = exponential_map(m, (0, 0), (3, 4), t=1.0)
    >>> np.allclose(end, (3, 4), atol=1e-4)
    True
    """
    if metric.dim != 2:
        raise NotImplementedError("exponential_map is for 2D metrics only.")
    traj = geodesic_solver(metric, p, v, (0, t), method=method, n_steps=100)
    return (traj['x'][-1], traj['y'][-1])


def distance(metric, p, q, method='shooting', max_iter=50, tol=1e-6):
    """
    Compute the geodesic distance between two points on a 2D Riemannian manifold.

    Two complementary numerical methods are provided:

    * **Shooting** (default): iteratively refines an initial tangent vector at
      p by shooting a geodesic and comparing its endpoint to q.  Converges
      quadratically near the solution and is accurate for well-separated points
      without conjugate loci between them.
    * **Optimisation**: minimises the energy functional
      E(v) = ½ ‖v‖_g² + penalty · ‖exp_p(v) − q‖²
      over the initial tangent vector v using BFGS (``scipy.optimize.minimize``).
      More robust when the shooting iteration diverges, but generally less precise.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    p, q : tuple of two floats
        Start and end points ``(x, y)`` in the manifold.
    method : {'shooting', 'optimize'}, default ``'shooting'``
        Numerical method to use.
    max_iter : int, default 50
        Maximum number of shooting iterations (ignored for ``'optimize'``).
    tol : float, default 1e-6
        Convergence tolerance on the endpoint error ‖exp_p(v) − q‖ for the
        shooting method.

    Returns
    -------
    float
        Approximate geodesic distance d(p, q).

    Raises
    ------
    NotImplementedError
        If called on a 1D metric.
    ValueError
        If ``method`` is not ``'shooting'`` or ``'optimize'``.

    Notes
    -----
    Both methods rely on :func:`exponential_map` internally, so the accuracy
    of the geodesic integrator (controlled by ``n_steps`` inside
    ``exponential_map``) directly affects the result.  For very curved spaces
    or points near conjugate loci, neither method is guaranteed to converge.

    Examples
    --------
    **Flat metric** — Euclidean distance:

    >>> import numpy as np
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> np.isclose(distance(m, (0, 0), (3, 4), method='shooting'), 5.0, rtol=1e-3)
    True
    >>> np.isclose(distance(m, (0, 0), (3, 4), method='optimize'), 5.0, rtol=5e-2)
    True
    """
    if metric.dim != 2:
        raise NotImplementedError("distance is for 2D metrics only.")

    if method == 'shooting':
        v_guess = np.array([q[0] - p[0], q[1] - p[1]], dtype=float)
        for _ in range(max_iter):
            q_reached = exponential_map(metric, p, tuple(v_guess), t=1.0)
            error = np.array([q_reached[0] - q[0], q_reached[1] - q[1]])
            if np.linalg.norm(error) < tol:
                break
            v_guess -= 0.5 * error
        g_eval = metric.eval(p[0], p[1])
        dist_sq = (g_eval['g'][0, 0] * v_guess[0]**2 +
                   2 * g_eval['g'][0, 1] * v_guess[0] * v_guess[1] +
                   g_eval['g'][1, 1] * v_guess[1]**2)
        return float(np.sqrt(dist_sq))

    elif method == 'optimize':

        def energy_functional(v):
            q_reached = exponential_map(metric, p, tuple(v), t=1.0)
            err = (q_reached[0] - q[0])**2 + (q_reached[1] - q[1])**2
            g_eval = metric.eval(p[0], p[1])
            E = 0.5 * (g_eval['g'][0, 0] * v[0]**2 +
                       2 * g_eval['g'][0, 1] * v[0] * v[1] +
                       g_eval['g'][1, 1] * v[1]**2)
            return E + 1e6 * err

        v_init = np.array([q[0] - p[0], q[1] - p[1]], dtype=float)
        result = minimize(energy_functional, v_init, method='BFGS')
        return float(np.sqrt(2 * result.fun))

    else:
        raise ValueError("method must be 'shooting' or 'optimize'.")


def jacobi_equation_solver(metric, geodesic, initial_variation, tspan, n_steps=1000):
    """
    Solve the Jacobi equation (geodesic deviation equation) along a geodesic (2D only).

    The Jacobi equation describes how a one-parameter family of geodesics
    spreads apart.  Along a reference geodesic γ with tangent vector γ̇, the
    Jacobi field J satisfies

        D²J/dt² + R(J, γ̇)γ̇ = 0,

    where R is the Riemann curvature tensor and D/dt denotes the covariant
    derivative along γ.  In components this becomes

        D²Jⁱ/dt² = −Rⁱⱼₖₗ Jʲ γ̇ᵏ γ̇ˡ − Γⁱⱼₖ (DJʲ/dt) γ̇ᵏ − Γⁱⱼₖ Jʲ D(γ̇ᵏ)/dt.

    The geodesic is interpolated cubically to supply the (x(t), y(t), vx(t),
    vy(t)) values needed inside the ODE, and the result is integrated with
    ``scipy.integrate.solve_ivp``.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    geodesic : dict
        Output dict from :func:`geodesic_solver` with keys ``'t'``, ``'x'``,
        ``'y'``, ``'vx'``, ``'vy'``.  The time array must span at least
        ``tspan``.
    initial_variation : dict
        Initial conditions for the Jacobi field:

        * ``'J0'`` : tuple of two floats — initial value (Jₓ(0), Jᵧ(0)).
        * ``'DJ0'`` : tuple of two floats — initial covariant derivative
          (DJₓ/dt(0), DJᵧ/dt(0)).
    tspan : tuple of two floats
        Integration interval for the Jacobi ODE.  Must be a sub-interval of
        the time range covered by ``geodesic['t']``.
    n_steps : int, default 1000
        Number of output time steps.

    Returns
    -------
    dict with keys:

    * ``'t'`` : ndarray — time values.
    * ``'J_x'``, ``'J_y'`` : ndarray — Jacobi field components.
    * ``'DJ_x'``, ``'DJ_y'`` : ndarray — covariant derivative of the Jacobi field.

    Raises
    ------
    NotImplementedError
        If called on a 1D metric.

    Notes
    -----
    Conjugate points along the geodesic occur where the Jacobi field vanishes.
    Large Jacobi-field growth indicates negative sectional curvature in the
    geodesic direction, while oscillating fields indicate positive curvature.

    Examples
    --------
    >>> import numpy as np
    >>> from sympy import symbols, Matrix, sin
    >>> theta, phi = symbols('theta phi', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi))
    >>> geod = geodesic_solver(m, (np.pi/2, 0), (0, 1), (0, 2), n_steps=200)
    >>> jac = jacobi_equation_solver(m, geod, {'J0': (0, 0), 'DJ0': (0.1, 0)}, (0, 2))
    >>> 'J_x' in jac and 'J_y' in jac
    True
    """
    if metric.dim != 2:
        raise NotImplementedError("jacobi_equation_solver is for 2D metrics only.")

    x_sym, y_sym = metric.coords
    R     = metric.riemann_tensor()
    R_func = {
        i: {
            j: {
                k: {
                    ell: lambdify((x_sym, y_sym), R[i][j][k][ell], 'numpy')
                    for ell in range(2)
                }
                for k in range(2)
            }
            for j in range(2)
        }
        for i in range(2)
    }

    t_geod = geodesic['t']
    x_interp = interp1d(t_geod, geodesic['x'], kind='cubic')
    y_interp = interp1d(t_geod, geodesic['y'], kind='cubic')
    vx_interp = interp1d(t_geod, geodesic['vx'], kind='cubic')
    vy_interp = interp1d(t_geod, geodesic['vy'], kind='cubic')
    Gamma = metric.christoffel_func

    def jacobi_ode(t, state):
        J_x, J_y, DJ_x, DJ_y = state
        x = x_interp(t)
        y = y_interp(t)
        vx = vx_interp(t)
        vy = vy_interp(t)
        J = [J_x, J_y]
        v = [vx, vy]
        curv_x = sum(R_func[0][j][k][ell](x, y) * J[j] * v[k] * v[ell]
                     for j in range(2) for k in range(2) for ell in range(2))
        curv_y = sum(R_func[1][j][k][ell](x, y) * J[j] * v[k] * v[ell]
                     for j in range(2) for k in range(2) for ell in range(2))
        DDJ_x = -(Gamma[0][0][0](x, y) * DJ_x * vx +
                  Gamma[0][0][1](x, y) * (DJ_x * vy + DJ_y * vx) +
                  Gamma[0][1][1](x, y) * DJ_y * vy + curv_x)
        DDJ_y = -(Gamma[1][0][0](x, y) * DJ_x * vx +
                  Gamma[1][0][1](x, y) * (DJ_x * vy + DJ_y * vx) +
                  Gamma[1][1][1](x, y) * DJ_y * vy + curv_y)
        return [DJ_x, DJ_y, DDJ_x, DDJ_y]

    J0 = initial_variation['J0']
    DJ0 = initial_variation['DJ0']
    sol = solve_ivp(
        jacobi_ode, tspan, [J0[0], J0[1], DJ0[0], DJ0[1]],
        t_eval=np.linspace(tspan[0], tspan[1], n_steps)
    )
    return {'t': sol.t, 'J_x': sol.y[0], 'J_y': sol.y[1],
            'DJ_x': sol.y[2], 'DJ_y': sol.y[3]}


def hodge_star(metric, form_degree):
    """
    Return the Hodge star operator ⋆ on differential forms (2D only).

    The Hodge star is the metric-induced isomorphism between k-forms and
    (n−k)-forms.  In an oriented 2D Riemannian manifold with metric g and
    volume form dV = √|det g| dx∧dy, it acts as:

    * **0-forms** (functions): ⋆f = f √|det g|.
    * **1-forms**: if α = αₓ dx + αᵧ dy then
      ⋆α = (g⁻¹ antisymmetric contraction) · √|det g|,
      concretely

          ⋆α = (g⁰⁰ αᵧ − g⁰¹ αₓ) √|g| dx + (−g⁰¹ αᵧ + g¹¹ αₓ) √|g| dy.

    * **2-forms**: ⋆(f dx∧dy) = f / √|det g|.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    form_degree : {0, 1, 2}
        Degree k of the input differential form.

    Returns
    -------
    callable
        * For degrees 0 and 2: a function of a single SymPy expression
          (the coefficient of the form) returning the coefficient of the
          Hodge dual.
        * For degree 1: a function ``(alpha_x, alpha_y) → (beta_x, beta_y)``
          returning the two components of the Hodge dual 1-form.

    Raises
    ------
    NotImplementedError
        If called on a 1D metric.
    ValueError
        If ``form_degree`` is not 0, 1, or 2.

    Examples
    --------
    **Scaled Euclidean metric** g = diag(4, 9):

    >>> from sympy import symbols, Matrix, simplify
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[4, 0], [0, 9]]), (x, y))
    >>> star0 = hodge_star(m, 0)
    >>> simplify(star0(1))   # ⋆1 = √(4·9) = 6
    6
    >>> star2 = hodge_star(m, 2)
    >>> simplify(star2(12))  # ⋆(12 dx∧dy) = 12/6 = 2
    2
    """
    if metric.dim != 2:
        raise NotImplementedError("hodge_star is for 2D metrics only.")
    sqrt_g = metric.sqrt_det_g
    g_inv = metric.g_inv_matrix

    if form_degree == 0:
        return lambda f: f * sqrt_g
    elif form_degree == 1:
        def star_1form(alpha_x, alpha_y):
            beta_x = (g_inv[0, 0] * alpha_y - g_inv[0, 1] * alpha_x) * sqrt_g
            beta_y = (-g_inv[0, 1] * alpha_y + g_inv[1, 1] * alpha_x) * sqrt_g
            return (beta_x, beta_y)
        return star_1form
    elif form_degree == 2:
        return lambda f: f / sqrt_g
    else:
        raise ValueError("form_degree must be 0, 1, or 2.")

def hodge_decomposition(metric, omega_components, domain, resolution=50):
    """
    Numerically decompose a 1-form into exact, co-exact, and harmonic parts
    on a 2D rectangle.

    The decomposition follows the Hodge theorem:
        α = dφ  +  ⋆dψ  +  h

    It solves two Poisson problems with Dirichlet BC:
        Δ φ = δα           → α_exact   = dφ
        Δ ψ = δ(⋆α)        → α_coexact = ⋆dψ
        h   = α − dφ − ⋆dψ

    Parameters
    ----------
    metric : Metric
        Must be 2D.
    omega_components : tuple
        The components (α_x, α_y) as callables or SymPy expressions.
    domain : tuple
        ((x_min, x_max), (y_min, y_max))
    resolution : int
        Grid points per axis.
    """
    if metric.dim != 2:
        raise NotImplementedError("Hodge decomposition is only implemented for 2D.")

    from scipy.sparse import diags, lil_matrix
    from scipy.sparse.linalg import spsolve

    # 1. Grid setup
    x_vals = np.linspace(domain[0][0], domain[0][1], resolution)
    y_vals = np.linspace(domain[1][0], domain[1][1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]
    N = resolution
    N2 = N * N

    # 2. Metric evaluation (Broadcasted to ensure arrays even for flat metrics)
    sqrt_det = np.broadcast_to(metric.sqrt_det_g_func(X, Y), X.shape)
    g_inv00  = np.broadcast_to(metric.g_inv_func[(0, 0)](X, Y), X.shape)
    g_inv11  = np.broadcast_to(metric.g_inv_func[(1, 1)](X, Y), X.shape)
    g_inv01  = np.broadcast_to(metric.g_inv_func[(0, 1)](X, Y), X.shape)

    # 3. Form components evaluation
    def eval_comp(c):
        if callable(c): return np.broadcast_to(c(X, Y), X.shape)
        f = lambdify(metric.coords, c, 'numpy')
        return np.broadcast_to(f(X, Y), X.shape)

    alpha_x_vals = eval_comp(omega_components[0])
    alpha_y_vals = eval_comp(omega_components[1])

    # 4. Helpers
    def face_avg(arr, axis, direction):
        """Average array with neighbor; ensures indices stay within bounds."""
        s_src = [slice(None), slice(None)]
        s_dst = [slice(None), slice(None)]
        if direction == 1: # Forward
            s_src[axis] = slice(1, None)
            s_dst[axis] = slice(0, -1)
        else: # Backward
            s_src[axis] = slice(0, -1)
            s_dst[axis] = slice(1, None)
        
        out = np.zeros_like(arr)
        out[tuple(s_dst)] = 0.5 * (arr[tuple(s_dst)] + arr[tuple(s_src)])
        return out

    def gradient(arr):
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[1:-1, :] = (arr[2:, :] - arr[:-2, :]) / (2 * dx)
        gx[0, :] = (arr[1, :] - arr[0, :]) / dx
        gx[-1, :] = (arr[-1, :] - arr[-2, :]) / dx
        gy[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) / (2 * dy)
        gy[:, 0] = (arr[:, 1] - arr[:, 0]) / dy
        gy[:, -1] = (arr[:, -1] - arr[:, -2]) / dy
        return gx, gy

    def codifferential(f_x, f_y):
        flux_x = sqrt_det * (g_inv00 * f_x + g_inv01 * f_y)
        flux_y = sqrt_det * (g_inv01 * f_x + g_inv11 * f_y)
        div = np.zeros_like(f_x)
        div[1:-1, :] += (flux_x[2:, :] - flux_x[:-2, :]) / (2 * dx)
        div[:, 1:-1] += (flux_y[:, 2:] - flux_y[:, :-2]) / (2 * dy)
        div[0, :] = (flux_x[1, :] - flux_x[0, :] / dx)
        div[-1, :] = (flux_x[-1, :] - flux_x[-2, :] / dx)
        return div / (sqrt_det + 1e-14)

    def hodge_star_1form(f_x, f_y):
        return (
            (g_inv00 * f_y - g_inv01 * f_x) * sqrt_det,
            (g_inv11 * f_x - g_inv01 * f_y) * sqrt_det,
        )

    # 5. Sparse Laplacian Matrix Assembly
    # East/West faces (x-axis, axis 0)
    aE = face_avg(sqrt_det * g_inv00, 0,  1)
    aW = face_avg(sqrt_det * g_inv00, 0, -1)
    # North/South faces (y-axis, axis 1)
    bN = face_avg(sqrt_det * g_inv11, 1,  1)
    bS = face_avg(sqrt_det * g_inv11, 1, -1)

    c_center = -(aE + aW) / dx**2 - (bN + bS) / dy**2
    c_east   = aE / dx**2
    c_west   = aW / dx**2
    c_north  = bN / dy**2
    c_south  = bS / dy**2

    # Standard 5-point stencil
    diag_data = [
        (c_center.ravel(), 0),
        (c_east.ravel()[:-N], N),   # i+1 -> offset N
        (c_west.ravel()[N:], -N),   # i-1 -> offset -N
        (c_north.ravel()[:-1], 1),  # j+1 -> offset 1
        (c_south.ravel()[1:], -1),  # j-1 -> offset -1
    ]
    
    A = lil_matrix((N2, N2))
    for data, k in diag_data:
        A.setdiag(data, k)

    # Cross-derivative g01 terms using a central stencil
    src = np.arange(N2).reshape(N, N)
    cross_coeff = (sqrt_det * g_inv01) / (4 * dx * dy)
    
    # Offsets for (di, dj): (1,1), (1,-1), (-1,1), (-1,-1)
    for di, dj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        offset = di * N + dj
        ri = slice(0, N-1) if di > 0 else slice(1, N)
        rj = slice(0, N-1) if dj > 0 else slice(1, N)
        
        mask = np.zeros((N, N), dtype=bool)
        mask[ri, rj] = True
        idx_v = src[mask].flatten()
        # The cross term contribution (simplified central diff)
        A[idx_v, idx_v + offset] += (di * dj) * cross_coeff[mask].flatten()

    # 6. Solve Poisson Problems
    # RHS for exact and co-exact potentials
    rhs_phi = codifferential(alpha_x_vals, alpha_y_vals)
    star_ax, star_ay = hodge_star_1form(alpha_x_vals, alpha_y_vals)
    rhs_psi = codifferential(star_ax, star_ay)

    def solve_poisson(rhs):
        b = rhs.ravel()
        # Enforce Dirichlet BC: potential = 0 on all four boundaries
        boundary_mask = np.zeros((N, N), dtype=bool)
        boundary_mask[0, :] = boundary_mask[-1, :] = True
        boundary_mask[:, 0] = boundary_mask[:, -1] = True
        idx_bound = src[boundary_mask].flatten()
        
        A_bc = A.tocsr()
        # For simplicity in assembly, we zero the rows of the boundary points
        for idx in idx_bound:
            A_bc.data[A_bc.indptr[idx]:A_bc.indptr[idx+1]] = 0
            A_bc[idx, idx] = 1.0
            b[idx] = 0.0
        
        return spsolve(A_bc, b).reshape(N, N)

    phi = solve_poisson(rhs_phi)
    psi = solve_poisson(rhs_psi)

    # 7. Reconstruction
    ex_x, ex_y = gradient(phi)
    
    dpsi_x, dpsi_y = gradient(psi)
    co_x, co_y = hodge_star_1form(dpsi_x, dpsi_y)
    
    ha_x = alpha_x_vals - ex_x - co_x
    ha_y = alpha_y_vals - ex_y - co_y

    return {
        'potential_phi': phi,
        'potential_psi': psi,
        'alpha_exact': (ex_x, ex_y),
        'alpha_coexact': (co_x, co_y),
        'alpha_harmonic': (ha_x, ha_y)
    }

def parallel_transport(metric, curve, initial_vector, tspan=None, method='RK45'):
    """
    Transport a vector along a curve using parallel transport.

    The parallel transport equation Dv/dt = 0 is solved as a linear ODE:
        dv^i/dt = - Γ^i_{jk} v^j ẋ^k,
    where ẋ^k are the components of the curve's velocity.

    Parameters
    ----------
    metric : Metric
        The Riemannian metric.
    curve : dict
        A trajectory dict from `geodesic_solver` containing at least:
        - 't' : array of time values
        - for 1D: 'x' array
        - for 2D: 'x', 'y' arrays
    initial_vector : float (1D) or tuple of two floats (2D)
        Components of the vector to transport at t = curve['t'][0].
    tspan : tuple (t0, t1) or None
        Time interval over which to transport. If None, the whole curve is used.
    method : str, default 'RK45'
        Integration method passed to `solve_ivp`.

    Returns
    -------
    dict
        - 't' : array of times (same as curve['t'] within tspan)
        - For 1D: 'v' : transported vector components
        - For 2D: 'vx', 'vy' : transported vector components
    """
    if metric.dim == 1:
        x = curve['x']
        t_vals = curve['t']
        v0 = initial_vector
        Gamma = metric.christoffel_func
        # approximate velocity from curve (once)
        dxdt = np.gradient(x, t_vals)

        def ode_1d(t, v):
            # interpolate position and velocity at time t
            x_t = np.interp(t, t_vals, x)
            dxdt_t = np.interp(t, t_vals, dxdt)
            return -Gamma(x_t) * v * dxdt_t

        if tspan is None:
            tspan = (t_vals[0], t_vals[-1])
        # Use the same time grid as the curve (or a subset)
        t_eval = np.linspace(tspan[0], tspan[1], len(t_vals))
        sol = solve_ivp(ode_1d, tspan, [v0], t_eval=t_eval, method=method)
        return {'t': sol.t, 'v': sol.y[0]}

    else:  # 2D
        x = curve['x']; y = curve['y']; t_vals = curve['t']
        vx0, vy0 = initial_vector
        Gamma = metric.christoffel_func
        # approximate velocities
        dxdt = np.gradient(x, t_vals)
        dydt = np.gradient(y, t_vals)

        def ode_2d(t, state):
            vx, vy = state
            # interpolate
            x_t = np.interp(t, t_vals, x)
            y_t = np.interp(t, t_vals, y)
            dxdt_t = np.interp(t, t_vals, dxdt)
            dydt_t = np.interp(t, t_vals, dydt)
            # compute Christoffel terms
            G000 = Gamma[0][0][0](x_t, y_t)
            G001 = Gamma[0][0][1](x_t, y_t)
            G011 = Gamma[0][1][1](x_t, y_t)
            G100 = Gamma[1][0][0](x_t, y_t)
            G101 = Gamma[1][0][1](x_t, y_t)
            G111 = Gamma[1][1][1](x_t, y_t)

            dvx = -(G000 * vx * dxdt_t +
                    G001 * (vx * dydt_t + vy * dxdt_t) +
                    G011 * vy * dydt_t)
            dvy = -(G100 * vx * dxdt_t +
                    G101 * (vx * dydt_t + vy * dxdt_t) +
                    G111 * vy * dydt_t)
            return [dvx, dvy]

        if tspan is None:
            tspan = (t_vals[0], t_vals[-1])
        t_eval = np.linspace(tspan[0], tspan[1], len(t_vals))
        sol = solve_ivp(ode_2d, tspan, [vx0, vy0], t_eval=t_eval, method=method)
        return {'t': sol.t, 'vx': sol.y[0], 'vy': sol.y[1]}


def de_rham_laplacian(metric, form_degree):
    """
    Compute the principal symbol of the de Rham (Hodge) Laplacian on k-forms (2D only).

    The Hodge–de Rham Laplacian is defined as

        Δ = dδ + δd,

    where d is the exterior derivative and δ = ⋆d⋆ is its formal adjoint.
    For **0-forms** (smooth functions) this coincides with the Laplace–Beltrami
    operator, so the result is identical to :meth:`Metric.laplace_beltrami_symbol`.
    For **1-forms** the principal symbol is the same positive-definite quadratic
    form gⁱʲ ξᵢ ξⱼ (the de Rham Laplacian is a Laplace-type operator on all
    form degrees in 2D), and the subprincipal symbol vanishes at this level of
    approximation.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    form_degree : {0, 1}
        Degree k of the differential form.  Degree 2 is not implemented
        (it equals the degree-0 case by Hodge duality).

    Returns
    -------
    dict with keys:

    * ``'principal'`` : sympy.Expr — leading symbol gⁱʲ ξᵢ ξⱼ.
    * ``'subprincipal'`` : sympy.Expr or int — lower-order symbol
      (non-zero for k=0; zero for k=1 at this approximation level).
    * ``'full'`` : sympy.Expr — full microlocal symbol.

    Raises
    ------
    NotImplementedError
        If called on a 1D metric, or if ``form_degree`` is 2 or higher.

    Examples
    --------
    >>> from sympy import symbols, Matrix
    >>> r, theta = symbols('r theta', real=True, positive=True)
    >>> m = Metric(Matrix([[1, 0], [0, r**2]]), (r, theta))
    >>> lb = de_rham_laplacian(m, form_degree=0)
    >>> '\\nfor k=0 and k=1 the principal symbol is the same:'
    >>> lb_1 = de_rham_laplacian(m, form_degree=1)
    >>> from sympy import simplify
    >>> simplify(lb['principal'] - lb_1['principal'])
    0
    """
    if metric.dim != 2:
        raise NotImplementedError("de_rham_laplacian is for 2D metrics only.")
    if form_degree == 0:
        return metric.laplace_beltrami_symbol()
    elif form_degree == 1:
        xi, eta = symbols('xi eta', real=True)
        g_inv = metric.g_inv_matrix
        principal = (g_inv[0, 0] * xi**2 +
                     2 * g_inv[0, 1] * xi * eta +
                     g_inv[1, 1] * eta**2)
        return {'principal': principal, 'subprincipal': 0, 'full': principal}
    else:
        raise NotImplementedError("Only degrees 0 and 1 are implemented.")


def verify_gauss_bonnet(metric, domain, resolution=100):
    """
    Numerically verify the Gauss–Bonnet theorem over a rectangular domain (2D only).

    The Gauss–Bonnet theorem states that for a compact oriented surface M
    without boundary,

        ∫∫_M K dA = 2π χ(M),

    where K is the Gaussian curvature, dA = √|det g| dx dy is the Riemannian
    area element, and χ(M) is the Euler characteristic.  For the unit sphere
    χ = 2, giving ∫K dA = 4π; for a torus χ = 0, giving ∫K dA = 0.

    This function numerically evaluates the left-hand side over a rectangular
    domain using ``scipy.integrate.dblquad`` and compares it to the expected
    value 2π (assuming χ = 1 for the supplied region, e.g. a topological disk).

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    domain : tuple
        ``((x_min, x_max), (y_min, y_max))`` — rectangular integration region.
    resolution : int, default 100
        Unused by the current implementation (kept for API consistency; the
        adaptive ``dblquad`` routine controls its own resolution internally).

    Returns
    -------
    dict with keys:

    * ``'integral'`` : float — numerically computed ∫∫_M K dA.
    * ``'expected'`` : float — reference value 2π.
    * ``'integration_error'`` : float — absolute error estimate from ``dblquad``.
    * ``'relative_error'`` : float — |integral − 2π| / 2π.

    Raises
    ------
    NotImplementedError
        If called on a 1D metric.

    Notes
    -----
    Accuracy degrades near coordinate singularities (e.g. θ = 0 or π on the
    sphere) where the curvature expression or the metric determinant may become
    ill-conditioned numerically.  Restrict the domain away from such points.

    Examples
    --------
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))    # flat: K = 0
    >>> result = verify_gauss_bonnet(m, ((0, 1), (0, 1)))
    >>> abs(result['integral']) < 1e-10   # ∫K dA = 0 for flat metric
    True
    """
    if metric.dim != 2:
        raise NotImplementedError("verify_gauss_bonnet is for 2D metrics only.")

    K_expr = metric.gauss_curvature()
    sqrt_g = metric.sqrt_det_g
    x_sym, y_sym = metric.coords
    integrand_func = lambdify((x_sym, y_sym), K_expr * sqrt_g, 'numpy')

    (x_min, x_max), (y_min, y_max) = domain
    integral, error = dblquad(
        lambda y, x: integrand_func(x, y),
        x_min, x_max, y_min, y_max
    )
    expected = 2 * np.pi
    return {
        'integral': integral,
        'integration_error': error,
        'expected': expected,
        'relative_error': abs(integral - expected) / abs(expected)
    }


# ============================================================================
# Visualisation (unified)
# ============================================================================

def visualize_geodesics(metric, initial_conditions, tspan,
                        x_range=None, y_range=None,
                        colorby='speed', plot_curvature=True,
                        n_steps=500):
    """
    Visualise geodesic trajectories on a 1D or 2D Riemannian manifold.

    Dispatches to ``_visualize_geodesics_1d`` or ``_visualize_geodesics_2d``
    based on ``metric.dim``.  Each geodesic is integrated from the given
    initial conditions and plotted using Matplotlib.

    Parameters
    ----------
    metric : Metric
        The Riemannian metric defining the manifold.
    initial_conditions : list
        * **1D**: list of ``(x₀, v₀)`` pairs (floats).
        * **2D**: list of ``((x₀, y₀), (vₓ₀, vᵧ₀))`` pairs of tuples.
    tspan : tuple of two floats
        Integration interval ``(t_start, t_end)`` for all geodesics.
    x_range : tuple of two floats, optional
        Horizontal plotting range.  Inferred from trajectory extents if
        ``None``.
    y_range : tuple of two floats, optional
        Vertical plotting range (2D only).  Inferred from trajectories if
        ``None``.
    colorby : {'speed', 'time', 'curvature'}, default ``'speed'``
        **1D only.** Quantity used to colour the scatter plot of each
        geodesic point.  ``'speed'`` uses |ẋ|, ``'time'`` uses t, and
        ``'curvature'`` uses |Γ¹₁₁(x(t))|.
    plot_curvature : bool, default True
        **2D only.** If ``True``, render the Gaussian curvature as a
        colour-mapped background heatmap behind the geodesic curves.
    n_steps : int, default 500
        Number of integration time steps per geodesic.

    Returns
    -------
    None
        Displays a Matplotlib figure via ``plt.show()``.

    Examples
    --------
    **1D** — visualise two geodesics on the cone metric g = x²:

    >>> x = symbols('x', real=True, positive=True)
    >>> m = Metric(x**2, (x,))
    >>> visualize_geodesics(m, [(1.0, 0.5), (2.0, -1.0)], (0, 5))

    **2D** — geodesics on the Poincaré half-plane:

    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1/y**2, 0], [0, 1/y**2]]), (x, y))
    >>> visualize_geodesics(m, [((0, 1), (0, 1)), ((0, 1), (1, 0))], (0, 2))
    """
    if metric.dim == 1:
        _visualize_geodesics_1d(metric, initial_conditions, tspan,
                                x_range, colorby, n_steps)
    else:
        _visualize_geodesics_2d(metric, initial_conditions, tspan,
                                x_range, y_range, plot_curvature, n_steps)


def _plot_geodesic_1d_colored(ax, metric, traj, colorby, label):
    """
    Plot a single 1D geodesic trajectory on *ax*, coloured by a scalar field.

    The trajectory is rendered either as a scatter plot (if a valid ``colorby``
    quantity is found) or as a plain line.  The colour mapping uses the
    ``'viridis'`` colormap throughout.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw on.
    metric : Metric
        The 1D metric (used to compute Christoffel values for
        ``colorby='curvature'``).
    traj : dict
        Trajectory dict returned by :func:`geodesic_solver` (keys ``'t'``,
        ``'x'``, ``'v'``).
    colorby : {'speed', 'time', 'curvature', None}
        Quantity to map to colour:

        * ``'speed'``: absolute velocity |ẋ(t)|.
        * ``'time'``: time parameter t.
        * ``'curvature'``: absolute Christoffel value |Γ¹₁₁(x(t))|.
        * Any other value: falls back to a plain un-coloured line plot.
    label : str
        Legend label for the trajectory (used only in the un-coloured fallback).

    Returns
    -------
    matplotlib.collections.PathCollection or None
        The scatter collection (suitable for ``plt.colorbar``) if a colour
        quantity was used, or ``None`` for plain line plots.
    """
    if colorby == 'speed':
        colors = np.abs(traj['v'])
    elif colorby == 'time':
        colors = traj['t']
    elif colorby == 'curvature':
        colors = np.abs(metric.christoffel_func(traj['x']))
    else:
        colors = None

    if colors is not None:
        sc = ax.scatter(traj['t'], traj['x'], c=colors,
                        s=10, cmap='viridis', alpha=0.6)
        return sc
    else:
        ax.plot(traj['t'], traj['x'], alpha=0.7, label=label)
        return None


def _visualize_geodesics_1d(metric, initial_conditions, tspan,
                             x_range, colorby, n_steps):
    """
    Internal renderer for 1D geodesic visualisation.

    Produces a two-panel Matplotlib figure:

    * **Top panel**: the metric component g₁₁(x) plotted over ``x_range``.
    * **Bottom panel**: all geodesic trajectories x(t) coloured by ``colorby``
      (see :func:`_plot_geodesic_1d_colored`).

    All geodesics are integrated once and cached so that the trajectory data
    can be reused for both auto-detecting ``x_range`` and rendering.

    Parameters
    ----------
    metric : Metric (dim=1)
    initial_conditions : list of (x₀, v₀) pairs
    tspan : tuple of two floats
    x_range : tuple of two floats or None
        If ``None``, inferred from the union of all trajectory extents ± 0.5.
    colorby : {'speed', 'time', 'curvature'}
    n_steps : int
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Compute all trajectories once; reuse for both range detection and plotting.
    trajs = [
        (x0, v0, geodesic_solver(metric, x0, v0, tspan, n_steps=n_steps))
        for x0, v0 in initial_conditions
    ]

    if x_range is None:
        all_x = np.concatenate([traj['x'] for _, _, traj in trajs])
        x_range = (all_x.min() - 0.5, all_x.max() + 0.5)

    x_plot = np.linspace(x_range[0], x_range[1], 200)
    ax1.plot(x_plot, metric.g_func(x_plot), 'k-', linewidth=2, label='g₁₁(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('g₁₁(x)')
    ax1.set_title('Metric Component')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    scatter = None
    for x0, v0, traj in trajs:
        label = f'IC: x₀={x0:.2f}, v₀={v0:.2f}'
        sc    = _plot_geodesic_1d_colored(ax2, metric, traj, colorby, label)
        if sc is not None:
            scatter = sc

    ax2.set_xlabel('t')
    ax2.set_ylabel('x(t)')
    ax2.set_title('Geodesic Trajectories')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    if scatter is not None:
        plt.colorbar(scatter, ax=ax2).set_label(colorby.capitalize())
    plt.tight_layout()
    plt.show()


def _visualize_geodesics_2d(metric, initial_conditions, tspan,
                              x_range, y_range, plot_curvature, n_steps):
    """
    Internal renderer for 2D geodesic visualisation.

    Produces a single Matplotlib axes on which:

    * If ``plot_curvature`` is True, the Gaussian curvature K(x, y) is
      rendered as a translucent colour mesh (``pcolormesh``) in the background.
      Failures in curvature evaluation (e.g. near coordinate singularities)
      are silently ignored with a printed warning.
    * Each geodesic is drawn as a sequence of short coloured line segments
      whose colour progresses from green (start) to red (end) along the
      trajectory.

    Parameters
    ----------
    metric : Metric (dim=2)
    initial_conditions : list of ((x₀, y₀), (vₓ₀, vᵧ₀)) pairs
    tspan : tuple of two floats
    x_range, y_range : tuple of two floats or None
        Plotting ranges.  If ``None``, inferred from trajectory extents
        with a 10 % margin.
    plot_curvature : bool
    n_steps : int
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    trajectories = [geodesic_solver(metric, p0, v0, tspan, n_steps=n_steps)
                    for p0, v0 in initial_conditions]

    if x_range is None:
        all_x = np.concatenate([t['x'] for t in trajectories])
        m = 0.1 * (all_x.max() - all_x.min())
        x_range = (all_x.min() - m, all_x.max() + m)
    if y_range is None:
        all_y = np.concatenate([t['y'] for t in trajectories])
        m = 0.1 * (all_y.max() - all_y.min())
        y_range = (all_y.min() - m, all_y.max() + m)

    if plot_curvature:
        try:
            x_bg = np.linspace(x_range[0], x_range[1], 100)
            y_bg = np.linspace(y_range[0], y_range[1], 100)
            X_bg, Y_bg = np.meshgrid(x_bg, y_bg, indexing='ij')
            K_expr = metric.gauss_curvature()
            K_func = lambdify(metric.coords, K_expr, 'numpy')
            K_vals = K_func(X_bg, Y_bg)
            im = ax.pcolormesh(X_bg, Y_bg, K_vals, shading='auto',
                               cmap='RdBu_r', alpha=0.3, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label='Gaussian Curvature')
        except Exception:
            print("Warning: Could not compute curvature background.")

    for idx, traj in enumerate(trajectories):
        p0, v0 = initial_conditions[idx]
        cvals = plt.cm.viridis(np.linspace(0, 1, len(traj['x'])))
        for i in range(len(traj['x']) - 1):
            ax.plot(traj['x'][i:i + 2], traj['y'][i:i + 2],
                    color=cvals[i], alpha=0.8, linewidth=2)
        ax.plot(traj['x'][0], traj['y'][0], 'go', markersize=10,
                label=f'Start {idx + 1}')
        ax.plot(traj['x'][-1], traj['y'][-1], 'ro', markersize=10,
                label=f'End {idx + 1}')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Geodesics on Riemannian Manifold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_curvature(metric, x_range=None, y_range=None,
                        resolution=100, quantity='gauss', cmap='RdBu_r',
                        **kwargs):
    """
    Visualise a curvature-related quantity of a 1D or 2D Riemannian manifold.

    Dispatches to ``_visualize_curvature_1d`` or ``_visualize_curvature_2d``
    based on ``metric.dim`` and renders a Matplotlib figure.

    Parameters
    ----------
    metric : Metric
        The Riemannian metric to visualise.
    x_range : tuple of two floats, optional
        Coordinate range along the first axis for the plot.  For 1D, inferred
        from geodesic extents if ``initial_conditions`` is also passed;
        defaults to (-5, 5) if neither is provided.  For 2D, required.
    y_range : tuple of two floats, optional
        Coordinate range along the second axis.  **Required for 2D metrics.**
        Unused for 1D.
    resolution : int, default 100
        Number of sample points along each axis when building the background
        curve (1D) or colour mesh (2D).
    quantity : str, default ``'gauss'``
        Which scalar field to display:

        * **2D**: ``'gauss'`` — Gaussian curvature K(x, y);
          ``'ricci_scalar'`` — Ricci scalar R(x, y).
        * **1D**: ``'metric'`` or ``'gauss'`` — metric component g₁₁(x);
          ``'christoffel'`` — Christoffel symbol Γ¹₁₁(x).
    cmap : str, default ``'RdBu_r'``
        Matplotlib colormap name for the 2D colour mesh.  Ignored for 1D.
    **kwargs
        Additional keyword arguments forwarded to the **1D** renderer only:

        * ``initial_conditions`` : list of (x₀, v₀) — if provided, geodesics
          are overlaid on the lower subplot.
        * ``tspan`` : tuple — integration interval for geodesics (default (0, 10)).
        * ``colorby`` : str — colouring scheme for geodesics (default ``'speed'``).
        * ``n_steps`` : int — number of integration steps (default 500).

    Returns
    -------
    None
        Displays a Matplotlib figure via ``plt.show()``.

    Raises
    ------
    ValueError
        If ``x_range`` or ``y_range`` are missing for a 2D metric, or if
        ``quantity`` is not recognised for the given dimension.

    Examples
    --------
    **2D** — Gaussian curvature of the Poincaré half-plane:

    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1/y**2, 0], [0, 1/y**2]]), (x, y))
    >>> visualize_curvature(m, x_range=(-2, 2), y_range=(0.5, 3), quantity='gauss')

    **1D** — metric and geodesics for g = x²:

    >>> x = symbols('x', real=True, positive=True)
    >>> m = Metric(x**2, (x,))
    >>> visualize_curvature(m, x_range=(0.5, 5), quantity='metric',
    ...                     initial_conditions=[(1.0, 0.5)], tspan=(0, 4))
    """
    if metric.dim == 1:
        _visualize_curvature_1d(metric, x_range, resolution, quantity, **kwargs)
    elif metric.dim == 2:
        if x_range is None or y_range is None:
            raise ValueError("x_range and y_range are required for 2D visualization.")
        _visualize_curvature_2d(metric, x_range, y_range, resolution, quantity, cmap)
    else:
        raise ValueError("Only 1D and 2D manifolds are supported.")


def _visualize_curvature_1d(metric, x_range, resolution, quantity, **kwargs):
    """
    Internal renderer for 1D curvature / metric visualisation.

    Produces a two-panel Matplotlib figure:

    * **Top panel**: the selected scalar quantity (metric component g₁₁ or
      Christoffel symbol Γ¹₁₁) plotted as a line over ``x_range``.
    * **Bottom panel**: geodesic trajectories x(t) if ``initial_conditions``
      is supplied via ``**kwargs``; otherwise a placeholder message.

    Parameters
    ----------
    metric : Metric (dim=1)
    x_range : tuple of two floats or None
        Plotting range.  If ``None`` and trajectories are available, inferred
        from their extents ± 0.5; falls back to (−5, 5) if no trajectories.
    resolution : int
        Number of points used to sample the scalar quantity along x.
    quantity : {'metric', 'gauss', 'christoffel'}
        Which scalar to plot.  ``'metric'`` and ``'gauss'`` both render g₁₁.
    **kwargs
        Optional keys forwarded from :func:`visualize_curvature`:

        * ``initial_conditions`` : list of (x₀, v₀) — geodesics to overlay.
        * ``tspan`` : tuple — integration interval (default ``(0, 10)``).
        * ``colorby`` : str — colouring scheme (default ``'speed'``).
        * ``n_steps`` : int — integration steps (default 500).
    """
    tspan   = kwargs.get('tspan',   (0, 10))
    colorby = kwargs.get('colorby', 'speed')
    n_steps = kwargs.get('n_steps', 500)
    initial_conditions = kwargs.get('initial_conditions')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax_metric, ax_geo = axes

    # Pre-compute trajectories once (avoids double integration when x_range
    # must be inferred from the trajectory extents).
    trajs = []
    if initial_conditions:
        trajs = [
            (x0, v0, geodesic_solver(metric, x0, v0, tspan, n_steps=n_steps))
            for x0, v0 in initial_conditions
        ]

    if x_range is None:
        if trajs:
            all_x  = np.concatenate([traj['x'] for _, _, traj in trajs])
            x_range = (all_x.min() - 0.5, all_x.max() + 0.5)
        else:
            x_range = (-5, 5)

    x_plot = np.linspace(x_range[0], x_range[1], resolution)

    if quantity in ('metric', 'gauss'):
        y_plot = metric.g_func(x_plot)
        ylabel, title = 'g₁₁(x)', 'Metric Component'
    elif quantity == 'christoffel':
        y_plot = metric.christoffel_func(x_plot)
        ylabel, title = 'Γ¹₁₁(x)', 'Christoffel Symbol'
    else:
        raise ValueError("1D quantity must be 'metric' or 'christoffel'.")

    ax_metric.plot(x_plot, y_plot, 'k-', linewidth=2, label=ylabel)
    ax_metric.set_xlabel('x')
    ax_metric.set_ylabel(ylabel)
    ax_metric.set_title(title)
    ax_metric.grid(True, alpha=0.3)
    ax_metric.legend()

    scatter = None
    if trajs:
        for x0, v0, traj in trajs:
            label = f'IC: x₀={x0:.2f}, v₀={v0:.2f}'
            sc    = _plot_geodesic_1d_colored(ax_geo, metric, traj, colorby, label)
            if sc is not None:
                scatter = sc

        ax_geo.set_xlabel('t')
        ax_geo.set_ylabel('x(t)')
        ax_geo.set_title('Geodesic Trajectories')
        ax_geo.grid(True, alpha=0.3)
        ax_geo.legend()
        if scatter is not None:
            plt.colorbar(scatter, ax=ax_geo).set_label(colorby.capitalize())
    else:
        ax_geo.text(0.5, 0.5, 'No initial conditions provided.',
                    ha='center', va='center', transform=ax_geo.transAxes)

    plt.tight_layout()
    plt.show()


def _visualize_curvature_2d(metric, x_range, y_range, resolution, quantity, cmap):
    """
    Internal renderer for 2D curvature visualisation.

    Evaluates the selected scalar curvature quantity on a regular meshgrid and
    renders it as a ``pcolormesh`` colour map using Matplotlib.  Constant
    curvature expressions (which ``lambdify`` returns as scalars) are broadcast
    to the full meshgrid shape before plotting.

    Parameters
    ----------
    metric : Metric (dim=2)
    x_range, y_range : tuple of two floats
        Coordinate ranges along each axis.
    resolution : int
        Number of sample points along each axis (meshgrid is resolution × resolution).
    quantity : {'gauss', 'ricci_scalar'}
        Curvature quantity to display:

        * ``'gauss'``: Gaussian curvature K(x, y).
        * ``'ricci_scalar'``: Ricci scalar R(x, y).
    cmap : str
        Matplotlib colormap name.

    Raises
    ------
    ValueError
        If ``quantity`` is not ``'gauss'`` or ``'ricci_scalar'``.
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    if quantity == 'gauss':
        K_expr = metric.gauss_curvature()
        Z = lambdify(metric.coords, K_expr, 'numpy')(X, Y)
        title = 'Gaussian Curvature K(x, y)'
    elif quantity == 'ricci_scalar':
        R_expr = metric.ricci_scalar()
        Z = lambdify(metric.coords, R_expr, 'numpy')(X, Y)
        title = 'Ricci Scalar R(x, y)'
    else:
        raise ValueError("2D quantity must be 'gauss' or 'ricci_scalar'.")

    # When the curvature expression simplifies to a constant, lambdify returns
    # a 0-d scalar rather than a 2-D array.  Broadcast it to the meshgrid shape
    # so that pcolormesh always receives a properly shaped array.
    Z = np.broadcast_to(np.asarray(Z, dtype=float), X.shape).copy()

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
    plt.colorbar(label=title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
