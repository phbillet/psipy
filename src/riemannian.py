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
The ``riemannian`` module provides a unified symbolic and numerical framework
for working with Riemannian manifolds in one and two dimensions.  A single
class ``Metric`` dispatches all computations automatically based on the
dimension of its input, making it straightforward to switch between 1D curves
and 2D surfaces without changing the calling interface.

The module is organised in four broad layers:

1. **Symbolic core** — ``Metric`` and its methods build and cache all derived
   geometric quantities (inverse metric, Christoffel symbols, curvature
   tensors, Laplace–Beltrami symbol) as SymPy expressions.
2. **Geodesic layer** — standalone functions integrate geodesic and
   Hamiltonian flow, compute geodesic distance, solve the Jacobi deviation
   equation, and perform parallel transport along curves.
3. **Differential-form layer** (2D only) — the Hodge star, the de Rham
   Laplacian with Weitzenböck correction, and a full numerical Hodge
   decomposition backed by a sparse FEM solver on a ``RiemannianGrid``.
4. **Visualisation layer** — geodesic trajectory plots and curvature
   colour maps for both dimensions.


Public API summary
------------------

``Metric`` class
    Central object; accepts a 1D scalar expression or a 2×2 SymPy ``Matrix``
    together with coordinate symbols.  Can also be constructed from a
    free-particle Hamiltonian via ``Metric.from_hamiltonian``.

    Methods available on every ``Metric``:

    * ``eval(*point)`` — evaluate all metric quantities at a numerical point.
    * ``gauss_curvature()`` — Gaussian curvature K (2D) or zero (1D).
    * ``riemann_tensor()`` — full Riemann curvature tensor R^i_jkl (2D).
    * ``ricci_tensor()`` — Ricci tensor R_ij (2D).
    * ``ricci_scalar()`` — scalar curvature R (2D).
    * ``laplace_beltrami_symbol()`` — principal and subprincipal symbols of
      the Laplace-Beltrami operator, ready for microlocal analysis.
    * ``riemannian_volume(domain)`` — Riemannian volume (area in 2D) by
      symbolic or numerical integration.
    * ``arc_length(x_min, x_max)`` — arc length along a 1D metric.
    * ``covariant_derivative_vector(V)`` — covariant derivative of a
      contravariant vector field (2D).
    * ``covariant_derivative_covector(omega)`` — covariant derivative of a
      1-form (2D).
    * ``riemannian_gradient(f)`` — gradient of a scalar as a contravariant
      vector field, grad f = g^ij d_j f d_i (2D).
    * ``riemannian_hessian(f)`` — covariant Hessian
      Hess(f)_ij = d_i d_j f - Gamma^k_ij d_k f (2D).

Geodesic and flow functions
    * ``geodesic_solver(metric, p0, v0, tspan)`` — integrate the geodesic ODE
      with selectable schemes: ``'rk4'``, ``'adaptive'``, ``'symplectic'``.
    * ``geodesic_hamiltonian_flow(metric, p0, v0, tspan)`` — geodesic flow in
      Hamiltonian form via the companion ``symplectic`` module (Verlet and
      other symplectic integrators).
    * ``distance(metric, p, q)`` — geodesic distance by shooting or
      optimisation (2D).
    * ``jacobi_equation_solver(metric, geodesic, initial_variation, tspan)``
      — solve the Jacobi (geodesic deviation) equation along a reference
      geodesic (2D).
    * ``parallel_transport(metric, curve, initial_vector, tspan)`` — parallel
      transport of a vector along a parameterised curve (2D).
    * ``christoffel(metric)`` — convenience wrapper returning the Christoffel
      symbol dict.
    * ``exponential_map(metric, p, v, t)`` — exponential map exp_p(tv) at a
      base point (2D).

Spectral and operator functions
    * ``laplace_beltrami(metric)`` — return the Laplace-Beltrami symbol dict
      (wrapper around ``Metric.laplace_beltrami_symbol``).
    * ``sturm_liouville_reduce(metric, potential_expr)`` — reduce the 1D
      Laplace-Beltrami eigenvalue problem -Delta_g u + Vu = lambda u to
      canonical Sturm-Liouville form (1D only).

Differential-form layer (2D only)
    * ``hodge_star(metric, form_degree)`` — symbolic Hodge star on 0-, 1-,
      and 2-forms.
    * ``de_rham_laplacian(metric, form_degree)`` — symbolic de Rham-Hodge
      Laplacian Delta = d delta + delta d on 0- and 1-forms, including the
      Weitzenbock curvature correction K*id for 1-forms; returns a dict with
      the principal symbol, subprincipal symbol, Weitzenbock term, and a
      callable ``'action'`` that applies the operator symbolically.
    * ``hodge_decomposition(metric, omega, domain, form_degree)`` — numerical
      Hodge-Helmholtz decomposition of a 1-form or 2-form into exact,
      co-exact, and harmonic parts on a rectangular domain.  The exact
      potential phi is solved with homogeneous Dirichlet BC (phi = 0 on dOmega)
      and the co-exact potential psi with Neumann-pinned BC (d_n psi = 0 on
      dOmega, gauge fixed at one interior node).  The harmonic residual
      captures the de Rham cohomology class of the input form.
    * ``visualize_hodge_decomposition(decomp)`` — vector-field or scalar-field
      plot of the three Hodge components with a metric-weighted energy bar
      chart.

``RiemannianGrid`` class
    Helper assembled once from a ``Metric`` and a rectangular domain.  Caches
    the N^2 x N^2 sparse FEM matrix for Delta_0 (scalar Laplace-Beltrami) and
    the 2N^2 x 2N^2 block matrix for Delta_1 (de Rham Laplacian on 1-forms
    with Weitzenbock correction K*id).  Exposes two public solvers:

    * ``solve_poisson_dirichlet(rhs)`` — Delta_0 u = f with u = 0 on dOmega.
    * ``solve_poisson_neumann(rhs)`` — Delta_0 u = f with d_n u = 0 on dOmega;
      the RHS is automatically projected onto the compatible subspace
      (mean-zero with respect to sqrt(g)), and the gauge is fixed by pinning
      one interior node to zero.  Both solvers accept a single (N, N) array
      or a batch (K, N, N) array of right-hand sides.

Gauss-Bonnet verification
    * ``verify_gauss_bonnet(metric, domain)`` — numerically integrate K dA
      over a rectangular domain and compare with the topological prediction
      2 pi chi (2D only).

Visualisation
    * ``visualize_geodesics(metric, initial_conditions, tspan)`` — overlay
      geodesic trajectories on a curvature background (1D or 2D).
    * ``visualize_curvature(metric, x_range, y_range, quantity)`` — colour map
      of Gaussian curvature or Ricci scalar (2D), or line plot of the metric
      component / Christoffel symbol with optional geodesic overlay (1D).


Mathematical background
-----------------------
A **Riemannian metric** g on an n-dimensional manifold assigns an inner
product to each tangent space.  In local coordinates (x^1,...,x^n) it is
written

    ds^2 = g_ij(x) dx^i dx^j,

with inverse g^ij.  The **Christoffel symbols** are

    Gamma^i_jk = 1/2 * g^il (d_j g_kl + d_k g_jl - d_l g_jk)

and govern the **geodesic equation**  x^i_tt + Gamma^i_jk x^j_t x^k_t = 0.
In 1D this simplifies to  x_tt + Gamma^1_11 x_t^2 = 0  with
Gamma^1_11 = 1/2 (log g_11)'.

**Curvature** is encoded in the Riemann tensor R^i_jkl, the Ricci tensor
R_ij = R^k_ikj, and the scalar curvature R = g^ij R_ij.  In 2D the Gaussian
curvature K is the only independent component: R_1212 = K |g|.

The **Laplace-Beltrami operator** on scalar functions is

    Delta = |g|^{-1/2} d_i ( |g|^{1/2} g^ij d_j ),

with principal symbol g^ij xi_i xi_j.  The sparse FEM discretisation uses a
standard 5-point finite-volume stencil with cross-derivative terms for
off-diagonal metric components, divided by sqrt(g) to recover the true
operator.

The **de Rham Laplacian** Delta = d delta + delta d on k-forms coincides with
the Laplace-Beltrami operator on 0-forms and satisfies the **Weitzenbock
identity** on 1-forms in 2D:

    Delta alpha = nabla* nabla alpha + K * alpha,

where nabla* nabla is the rough (connection) Laplacian acting component-wise
and K is the Gaussian curvature.

The **Hodge decomposition** of a differential form on a compact domain Omega
in 2D is as follows:

* **0‑form** (scalar function) f:
    f = Δφ + h₀,
    where Δφ = δ(⋆dφ) is the co‑exact part (the Laplacian of a scalar potential
    φ) and h₀ is the constant harmonic part (the mean of f).  The exact part
    is zero because no (−1)-form exists.  The harmonic part corresponds to the
    de Rham cohomology class [f] ∈ H⁰_dR(Ω) ≅ ℝ.

* **1‑form** α = αₓ dx + αᵧ dy:
    α = dφ + ⋆dψ + h,
    where dφ is exact, ⋆dψ is co‑exact, and h is harmonic (Δh = 0).  φ and ψ
    are scalar potentials satisfying Poisson equations; h represents the de Rham
    cohomology class [α] ∈ H¹_dR(Ω).

* **2‑form** ω = f dx∧dy:
    ω = d(⋆dφ) + h₂,
    with no co‑exact component (no 3‑forms exist in 2D).  φ is a scalar
    potential solving Δφ = -⋆ω, and h₂ is the harmonic 2‑form, representing
    the cohomology class [ω] ∈ H²_dR(Ω).


References
----------
.. [1] do Carmo, M. P.  *Riemannian Geometry*, Birkhauser, 1992.
.. [2] Jost, J.  *Riemannian Geometry and Geometric Analysis*, Springer,
       2011 (6th ed.).
.. [3] Lee, J. M.  *Riemannian Manifolds: An Introduction to Curvature*,
       Springer, 1997.
.. [4] Petersen, P.  *Riemannian Geometry*, Springer, 2016 (3rd ed.).
.. [5] Frankel, T.  *The Geometry of Physics*, Cambridge University Press,
       2011 (3rd ed.).
.. [6] Warner, F. W.  *Foundations of Differentiable Manifolds and Lie
       Groups*, Springer, 1983.
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

from sympy import symbols, simplify, lambdify, diff, sqrt, log, zeros, Matrix
import numpy as np
from scipy.sparse import lil_matrix, diags, coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags


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

    The Jacobi equation describes how a one-parameter family of geodesics spreads apart.
    For a reference geodesic γ(t) with tangent vector v = γ̇, the Jacobi field J satisfies

        D²J/dt² + R(J, v)v = 0,

    where R is the Riemann curvature tensor and D/dt denotes the covariant derivative along γ.
    In local coordinates (x⁰, x¹) this system is equivalent to the first‑order ODEs:

        dJⁱ/dt = DJⁱ - Γⁱⱼₖ Jʲ vᵏ
        d(DJⁱ)/dt = -Rⁱⱼₖₗ Jʲ vᵏ vˡ - Γⁱⱼₖ DJʲ vᵏ

    The geodesic trajectory (x(t), y(t), vx(t), vy(t)) is interpolated cubically from the
    input dict. The ODE is integrated with high‑accuracy BDF (backward differentiation formula)
    from `scipy.integrate.solve_ivp`.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    geodesic : dict
        Output dict from :func:`geodesic_solver` with keys ``'t'``, ``'x'``, ``'y'``,
        ``'vx'``, ``'vy'``. The time array must span at least ``tspan``.
    initial_variation : dict
        Initial conditions for the Jacobi field:

        * ``'J0'`` : tuple of two floats — initial value (Jₓ(0), Jᵧ(0)).
        * ``'DJ0'`` : tuple of two floats — initial covariant derivative
          (DJₓ/dt(0), DJᵧ/dt(0)).
    tspan : tuple of two floats
        Integration interval for the Jacobi ODE. Must be a sub‑interval of the time range
        covered by ``geodesic['t']``.
    n_steps : int, default 1000
        Number of output time steps (equally spaced in the interval ``tspan``).

    Returns
    -------
    dict
        Keys:

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
    Large growth indicates negative sectional curvature in the geodesic direction,
    while oscillatory behaviour indicates positive curvature.

    The ODE is solved with high precision (`rtol=1e-9`, `atol=1e-12`) and a stiff
    solver (BDF) to handle possible rapid changes in curvature.

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
    R = metric.riemann_tensor()
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
        DJ = [DJ_x, DJ_y]

        # Γ(J, v)
        gamma_J_x = sum(Gamma[0][j][k](x, y) * J[j] * v[k] for j in range(2) for k in range(2))
        gamma_J_y = sum(Gamma[1][j][k](x, y) * J[j] * v[k] for j in range(2) for k in range(2))

        # Γ(DJ, v)
        gamma_DJ_x = sum(Gamma[0][j][k](x, y) * DJ[j] * v[k] for j in range(2) for k in range(2))
        gamma_DJ_y = sum(Gamma[1][j][k](x, y) * DJ[j] * v[k] for j in range(2) for k in range(2))

        # R(J, v)v  (R^i_{jkl} v^j J^k v^l)
        curv_x = sum(R_func[0][j][k][ell](x, y) * v[j] * J[k] * v[ell]
                     for j in range(2) for k in range(2) for ell in range(2))
        curv_y = sum(R_func[1][j][k][ell](x, y) * v[j] * J[k] * v[ell]
                     for j in range(2) for k in range(2) for ell in range(2))

        # dJ/dt = DJ - Γ(J, v)
        dJ_x = DJ_x - gamma_J_x
        dJ_y = DJ_y - gamma_J_y

        # d(DJ)/dt = -R(J, v)v - Γ(DJ, v)
        dDJ_x = -curv_x - gamma_DJ_x
        dDJ_y = -curv_y - gamma_DJ_y

        return [dJ_x, dJ_y, dDJ_x, dDJ_y]

    J0 = initial_variation['J0']
    DJ0 = initial_variation['DJ0']
    sol = solve_ivp(
        jacobi_ode, tspan, [J0[0], J0[1], DJ0[0], DJ0[1]],
        t_eval=np.linspace(tspan[0], tspan[1], n_steps),
        method='BDF',
        rtol=1e-9, atol=1e-12
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
    g = metric.g_matrix               # covariant metric components
    g11, g12, g22 = g[0,0], g[0,1], g[1,1]

    if form_degree == 0:
        return lambda f: f * sqrt_g
    elif form_degree == 1:
        def star_1form(alpha_x, alpha_y):
            # (⋆α)_x = √|g| ( -g12 α_x - g22 α_y )
            # (⋆α)_y = √|g| (  g11 α_x + g12 α_y )
            beta_x = sqrt_g * (-g12 * alpha_x - g22 * alpha_y)
            beta_y = sqrt_g * ( g11 * alpha_x + g12 * alpha_y)
            return (beta_x, beta_y)
        return star_1form
    elif form_degree == 2:
        return lambda f: f / sqrt_g
    else:
        raise ValueError("form_degree must be 0, 1, or 2.")

# =============================================================================
# Option A — de_rham_laplacian extended to form_degree=1 with the full
#            Weitzenböck correction, and a new symbolic action method.
# Option B — RiemannianGrid helper that assembles the sparse FEM matrix from
#            the operator symbol, and hodge_decomposition rewritten to use it.
# =============================================================================

# =============================================================================
# Option A — full symbolic de Rham Laplacian
# =============================================================================

def de_rham_laplacian(metric, form_degree):
    """
    Compute the symbol and symbolic action of the de Rham–Hodge Laplacian
    Δ = dδ + δd on k-forms (2D only).

    For ``form_degree=0`` the result is identical to
    :meth:`Metric.laplace_beltrami_symbol`.

    For ``form_degree=1`` the **Weitzenböck identity**

        Δα = ∇*∇α + Ric(α♯)♭

    is used.  In 2D the Ricci endomorphism on 1-forms is simply K·id
    (multiplication by the Gaussian curvature), so

        (Δα)ᵢ = (∇*∇α)ᵢ + K αᵢ ,

    where (∇*∇α)ᵢ is the component-wise rough Laplacian acting on each
    1-form component as a scalar.  This is strictly stronger than the
    previous implementation, which returned zero for the subprincipal
    symbol and omitted the curvature term entirely.

    Parameters
    ----------
    metric : Metric
        Must have ``metric.dim == 2``.
    form_degree : {0, 1}
        Degree of the input differential form.

    Returns
    -------
    dict with keys:

    * ``'principal'``    : sympy.Expr — leading symbol gⁱʲ ξᵢ ξⱼ.
    * ``'subprincipal'`` : sympy.Expr — first-order transport term
                          (non-zero for k=0; zero for k=1 at symbol level).
    * ``'full'``         : sympy.Expr — σ₂ + i σ₁ (microlocal convention).
    * ``'weitzenbock'``  : sympy.Expr or None — the Weitzenböck curvature
                          correction K (only present for ``form_degree=1``).
    * ``'action'``       : callable — ``action(alpha)`` applies the operator
                          symbolically to a form and returns the result.
                          - k=0: ``alpha`` is a scalar SymPy expression;
                            returns a scalar expression Δf.
                          - k=1: ``alpha`` is a 2-tuple of SymPy expressions
                            (α₀, α₁); returns a 2-tuple ((Δα)₀, (Δα)₁).

    Raises
    ------
    NotImplementedError
        If called on a 1D metric, or if ``form_degree`` ≥ 2.

    Examples
    --------
    **Flat metric** — Laplacian of a 0-form:

    >>> from sympy import symbols, Matrix, sin, cos, simplify
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> op = de_rham_laplacian(m, form_degree=0)
    >>> simplify(op['action'](sin(x) * cos(y)))   # −sin(x)cos(y) − sin(x)cos(y)
    -2*sin(x)*cos(y)

    **Unit sphere** — Weitzenböck correction K=1 on a 1-form:

    >>> theta, phi = symbols('theta phi', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi))
    >>> op = de_rham_laplacian(m, form_degree=1)
    >>> simplify(op['weitzenbock'])
    1
    """
    if metric.dim != 2:
        raise NotImplementedError("de_rham_laplacian is for 2D metrics only.")

    xi, eta = symbols('xi eta', real=True)
    g_inv   = metric.g_inv_matrix
    principal = (g_inv[0, 0] * xi**2
                 + 2 * g_inv[0, 1] * xi * eta
                 + g_inv[1, 1] * eta**2)

    if form_degree == 0:
        # Identical to laplace_beltrami_symbol — delegate to avoid duplication.
        sym = metric.laplace_beltrami_symbol()

        def _action_0(f_expr):
            """Apply Δ to a scalar function f (0-form)."""
            x, y = metric.coords
            sqrt_g = metric.sqrt_det_g
            g_inv_m = metric.g_inv_matrix
            # Δf = |g|^{-½} ∂ᵢ(|g|^{½} gⁱʲ ∂ⱼ f)
            div = sum(
                diff(sqrt_g * g_inv_m[i, j] * diff(f_expr, [x, y][j]), [x, y][i])
                for i in range(2) for j in range(2)
            )
            return simplify(div / sqrt_g)

        return {**sym, 'weitzenbock': None, 'action': _action_0}

    elif form_degree == 1:
        # ------------------------------------------------------------------
        # Symbol level: principal symbol same as k=0; subprincipal = 0.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Weitzenböck term: in 2D, Ric = K·g so the endomorphism on 1-forms
        # is K·id.  We extract K from the Riemann tensor.
        # ------------------------------------------------------------------
        K = simplify(metric.gauss_curvature())

        def _rough_laplacian_scalar(f_expr):
            """
            Rough (connection) Laplacian ∇*∇f = Laplace–Beltrami on a scalar.
            Re-uses the 0-form action so the formula is not repeated.
            """
            x, y    = metric.coords
            sqrt_g  = metric.sqrt_det_g
            g_inv_m = metric.g_inv_matrix
            div = sum(
                diff(sqrt_g * g_inv_m[i, j] * diff(f_expr, [x, y][j]), [x, y][i])
                for i in range(2) for j in range(2)
            )
            return simplify(div / sqrt_g)

        def _action_1(alpha):
            """
            Apply Δ = ∇*∇ + K to a 1-form α = (α₀, α₁).

            Each component is treated as a scalar by the rough Laplacian,
            then the Weitzenböck term K·αᵢ is added.

            Parameters
            ----------
            alpha : tuple of two sympy.Expr
                (α₀, α₁) — the two covariant components of the 1-form.

            Returns
            -------
            tuple of two sympy.Expr
                ((Δα)₀, (Δα)₁)
            """
            return tuple(
                simplify(_rough_laplacian_scalar(a) + K * a)
                for a in alpha
            )

        return {
            'principal':    simplify(principal),
            'subprincipal': 0,
            'full':         simplify(principal),
            'weitzenbock':  K,
            'action':       _action_1,
        }

    else:
        raise NotImplementedError("Only form_degree 0 and 1 are implemented.")


# =============================================================================
# Option B — RiemannianGrid: assembles the sparse FEM Laplacian matrix from
#            the operator symbol, shared by hodge_decomposition.
# =============================================================================

class RiemannianGrid:
    """
    A regular rectangular grid carrying numerical metric data and the
    assembled sparse FEM Laplace–Beltrami matrix.

    This is the bridge between the symbolic ``de_rham_laplacian`` operator
    and the numerical solvers in ``hodge_decomposition``.  Instead of each
    function building its own stencil, both consume a ``RiemannianGrid``.

    Parameters
    ----------
    metric : Metric
        A 2D Riemannian metric.
    domain : tuple
        ``((x_min, x_max), (y_min, y_max))``
    resolution : int
        Number of grid points per axis.

    Attributes
    ----------
    X, Y : ndarray (N, N)
        Meshgrid coordinate arrays.
    dx, dy : float
        Grid spacings.
    N, N2 : int
        Points per axis and total points.
    sqrt_det, g_inv00, g_inv11, g_inv01 : ndarray (N, N)
        Evaluated metric coefficients.
    A_scalar : scipy.sparse.csr_matrix
        Assembled FEM matrix for the scalar Laplace–Beltrami operator Δ₀.
        Suitable for solving Δ₀ u = f after applying Dirichlet BC.
    A_1form : scipy.sparse.csr_matrix
        Block-2×2 matrix for the de Rham Laplacian on 1-forms,
        incorporating the Weitzenböck curvature term.  Shape (2N², 2N²).
    idx_bound : ndarray of int
        Flat indices of all boundary nodes (for Dirichlet BC enforcement).

    Examples
    --------
    >>> from sympy import symbols, Matrix
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> grid = RiemannianGrid(m, ((0, 1), (0, 1)), resolution=30)
    >>> grid.A_scalar.shape
    (900, 900)
    >>> grid.A_1form.shape
    (1800, 1800)
    """

    def __init__(self, metric, domain, resolution):
        if metric.dim != 2:
            raise NotImplementedError("RiemannianGrid requires a 2D metric.")

        x_vals = np.linspace(domain[0][0], domain[0][1], resolution)
        y_vals = np.linspace(domain[1][0], domain[1][1], resolution)
        self.X, self.Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        self.dx = x_vals[1] - x_vals[0]
        self.dy = y_vals[1] - y_vals[0]
        self.N  = resolution
        self.N2 = resolution * resolution
        self._metric = metric

        def _to_float_array(val):
            return np.broadcast_to(val, self.X.shape).astype(float)
        
        self.sqrt_det = _to_float_array(metric.sqrt_det_g_func(self.X, self.Y))
        self.g_inv00  = _to_float_array(metric.g_inv_func[(0, 0)](self.X, self.Y))
        self.g_inv11  = _to_float_array(metric.g_inv_func[(1, 1)](self.X, self.Y))
        self.g_inv01  = _to_float_array(metric.g_inv_func[(0, 1)](self.X, self.Y))
        # Covariant metric components (needed for the Hodge star)
        self.g00 = _to_float_array(metric.g_func[(0, 0)](self.X, self.Y))
        self.g01 = _to_float_array(metric.g_func[(0, 1)](self.X, self.Y))
        self.g11 = _to_float_array(metric.g_func[(1, 1)](self.X, self.Y))

        # Boundary node indices (shared by both Poisson solvers)
        src = np.arange(self.N2).reshape(self.N, self.N)
        bm  = np.zeros((self.N, self.N), dtype=bool)
        bm[0, :] = bm[-1, :] = bm[:, 0] = bm[:, -1] = True
        self.idx_bound = src[bm].flatten()
        self._src      = src

        # Assemble both matrices once at construction time
        self.A_scalar = self._assemble_scalar_laplacian()
        self.A_1form  = self._assemble_1form_laplacian()
        self._pin = int(self._src[self.N // 2, self.N // 2])
        self._A_neumann = self.A_scalar.tolil()
        self._A_neumann[self._pin, :] = 0
        self._A_neumann[self._pin, self._pin] = 1.0
        self._A_neumann = self._A_neumann.tocsr()
        # Build Dirichlet version: zero out rows of boundary nodes, set diagonal to 1
        self._A_dirichlet = self.A_scalar.tolil()
        bnd = self.idx_bound
        self._A_dirichlet[bnd, :] = 0
        self._A_dirichlet[bnd, bnd] = 1.0
        self._A_dirichlet = self._A_dirichlet.tocsr()

    # ------------------------------------------------------------------
    # Private assembly helpers
    # ------------------------------------------------------------------

    def _face_avg(self, arr, axis, forward):
        out = np.zeros_like(arr)
        s_dst = [slice(None), slice(None)]
        s_src = [slice(None), slice(None)]
        if forward:
            s_dst[axis], s_src[axis] = slice(0, -1), slice(1, None)
        else:
            s_dst[axis], s_src[axis] = slice(1, None), slice(0, -1)
        out[tuple(s_dst)] = 0.5 * (arr[tuple(s_dst)] + arr[tuple(s_src)])
        return out

    def _assemble_scalar_laplacian(self):
        """
        Build the N²×N² sparse FEM matrix for the scalar Laplace–Beltrami
        operator Δ₀ = |g|^{-½} ∂ᵢ(|g|^{½} gⁱʲ ∂ⱼ).
 
        The stencil is a standard 5-point finite-volume scheme with cross
        terms for the off-diagonal g_inv01 component.
        """
        N, N2, dx, dy = self.N, self.N2, self.dx, self.dy
    
        face_specs = [
            (self.sqrt_det * self.g_inv00, dx, 0),
            (self.sqrt_det * self.g_inv11, dy, 1),
        ]
        c_center    = np.zeros_like(self.sqrt_det)
        stencil_diags = []
    
        for coeff, h, axis in face_specs:
            cf = self._face_avg(coeff, axis, forward=True)
            cb = self._face_avg(coeff, axis, forward=False)
            c_center -= (cf + cb) / h**2
            offset = N if axis == 0 else 1
            stencil_diags.append(((cf / h**2).ravel()[:-offset],  offset))
            stencil_diags.append(((cb / h**2).ravel()[offset:],  -offset))
    
        A = lil_matrix((N2, N2))
        A.setdiag(c_center.ravel(), 0)
        for data, k in stencil_diags:
            A.setdiag(data, k)
    
        cross = (self.sqrt_det * self.g_inv01) / (4 * dx * dy)
        for di, dj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            ri   = slice(0, N - 1) if di > 0 else slice(1, N)
            rj   = slice(0, N - 1) if dj > 0 else slice(1, N)
            mask = np.zeros((N, N), dtype=bool)
            mask[ri, rj] = True
            idx  = self._src[mask].flatten()
            A[idx, idx + di * N + dj] += (di * dj) * cross[mask].flatten()
    
        # ✅ FIX 1: divide every row by √g to get the true (1/√g)∂(√g g^{ij}∂) operator
        inv_sqrt_det = diags(1.0 / (self.sqrt_det.ravel() + 1e-14), format='csr')
        return inv_sqrt_det.dot(A.tocsr())
    

    def _assemble_1form_laplacian(self):
        """
        Build the 2N²×2N² sparse matrix for the de Rham Laplacian on 1-forms,

            Δ₁ = ∇*∇ + K·id  (Weitzenböck),

        where the rough Laplacian ∇*∇ acts component-wise (same stencil as
        Δ₀ for each scalar component), and K is the Gaussian curvature
        evaluated on the grid, adding a diagonal block K·I to each component.

        The matrix has the 2×2 block structure:
            [ Δ₀ + K·I      0      ]
            [    0       Δ₀ + K·I  ]
        """
        from scipy.sparse import eye as sp_eye, bmat as sp_bmat
    
        K_expr  = self._metric.gauss_curvature()
        K_func  = lambdify(self._metric.coords, K_expr, 'numpy')
        K_grid  = np.broadcast_to(K_func(self.X, self.Y), self.X.shape).copy()
    
        # ✅ FIX 2: Weitzenböck with geometer's sign is Δ₁ = Δ₀ + K·id,
        #    where Δ₀ is already negative semi-definite.
        #    K_diag must therefore be added (not subtracted), which is what
        #    the code does — but only works correctly once A_scalar is the
        #    true Δ₀ (Fix 1 above).  No change needed here beyond Fix 1.
        K_diag = sp_eye(self.N2, format='csr').multiply(K_grid.ravel())
        block  = self.A_scalar + K_diag
        return sp_bmat([[block, None], [None, block]], format='csr')

    # ------------------------------------------------------------------
    # Public solver
    # ------------------------------------------------------------------
    def solve_poisson_dirichlet(self, rhs):
        """
        Solve the scalar Laplace–Beltrami equation Δ₀ u = rhs with homogeneous Dirichlet boundary conditions.
    
        The system is assembled as a sparse FEM matrix `self._A_dirichlet`, which encodes
        the operator Δ₀ on a regular rectangular grid.  Homogeneous Dirichlet conditions
        (u = 0 on the boundary) are enforced by zeroing the rows corresponding to boundary
        nodes and setting the diagonal entry to 1; the right‑hand side is set to zero on
        those rows.
    
        The solver supports multiple right‑hand sides simultaneously, which is more
        efficient than solving each individually.
    
        Parameters
        ----------
        rhs : numpy.ndarray
            Right‑hand side field(s).  Allowed shapes:
    
            - 2D (N, N) : a single scalar field on the grid.
            - 3D (K, N, N) : K independent scalar fields, solved in one linear system.
    
        Returns
        -------
        numpy.ndarray
            Solution u of the Poisson equation, with the same shape as `rhs`:
    
            - (N, N) for a single RHS.
            - (K, N, N) for multiple RHS (first dimension = number of fields).
    
        Raises
        ------
        ValueError
            If the shape of `rhs` does not match the grid dimensions.
    
        Notes
        -----
        The linear system is solved using `scipy.sparse.linalg.spsolve`, which
        returns the solution with the same number of columns as the number of RHS
        vectors.  The result is reshaped back to the original spatial grid.
        """
        if rhs.ndim == 2:
            if rhs.shape != (self.N, self.N):
                raise ValueError("rhs must have shape (N, N)")
            rhs_flat = rhs.ravel().reshape(-1, 1)          # (N², 1)
            out_shape = (self.N, self.N)
        elif rhs.ndim == 3:
            K, N, M = rhs.shape
            if N != self.N or M != self.N:
                raise ValueError("rhs must have shape (K, N, N)")
            rhs_flat = rhs.reshape(K, -1).T                # (N², K)
            out_shape = (K, self.N, self.N)
        else:
            raise ValueError("rhs must be 2D or 3D array")

        b = rhs_flat.copy()
        # Enforce Dirichlet BC: set RHS to 0 on boundary nodes
        b[self.idx_bound, :] = 0.0

        sol = spsolve(self._A_dirichlet, b)
        if sol.ndim == 1:
            sol = sol[:, None]

        if rhs.ndim == 2:
            return sol.reshape(self.N, self.N)
        else:
            return sol.T.reshape(out_shape)
            

    def solve_poisson_neumann(self, rhs):
        """
        Solve the scalar Laplace–Beltrami equation Δ₀ u = rhs with homogeneous Neumann boundary conditions.
    
        The Neumann problem requires a compatibility condition: the right‑hand side must
        be orthogonal to the constant nullspace, i.e. ∫_M rhs · √g dV = 0.  This function
        automatically enforces that condition by subtracting the weighted mean from `rhs`
        before solving.  The gauge freedom (addition of an arbitrary constant) is fixed
        by pinning a single interior node (the grid point nearest to the centre) to zero
        in the assembled matrix `self._A_neumann`.
    
        The solver supports multiple right‑hand sides simultaneously, which is more
        efficient than solving each individually.
    
        Parameters
        ----------
        rhs : numpy.ndarray
            Right‑hand side field(s).  Allowed shapes:
    
            - 2D (N, N) : a single scalar field on the grid.
            - 3D (K, N, N) : K independent scalar fields, solved in one linear system.
    
        Returns
        -------
        numpy.ndarray
            Solution u of the Poisson equation, with the same shape as `rhs`:
    
            - (N, N) for a single RHS.
            - (K, N, N) for multiple RHS (first dimension = number of fields).
    
        Raises
        ------
        ValueError
            If the shape of `rhs` does not match the grid dimensions.
    
        Notes
        -----
        The weighted mean is computed as (∫ rhs · √g dV) / (∫ √g dV) and subtracted
        from each RHS vector.  The solution is unique because the gauge is fixed by
        the interior node condition.
        """
        # Determine input shape and flatten appropriately
        if rhs.ndim == 2:
            if rhs.shape != (self.N, self.N):
                raise ValueError("rhs must have shape (N, N)")
            rhs_flat = rhs.ravel().reshape(-1, 1)          # (N², 1)
            out_shape = (self.N, self.N)
        elif rhs.ndim == 3:
            K, N, M = rhs.shape
            if N != self.N or M != self.N:
                raise ValueError("rhs must have shape (K, N, N)")
            rhs_flat = rhs.reshape(K, -1).T                # (N², K)
            out_shape = (K, self.N, self.N)
        else:
            raise ValueError("rhs must be 2D or 3D array")

        b = rhs_flat.copy()
        weights = self.sqrt_det.ravel()

        # Subtract weighted mean to make RHS orthogonal to nullspace
        mean = (b.T.dot(weights) / weights.sum()).reshape(1, -1)
        b = b - weights[:, None] * mean
        # Zero the pin row so the gauge node solves to exactly 0
        b[self._pin, :] = 0.0

        # Solve all columns at once
        sol = spsolve(self._A_neumann, b)

        if sol.ndim == 1:
            sol = sol[:, None]                            # make it (N², 1)

        # Reshape back to original form
        if rhs.ndim == 2:
            return sol.reshape(self.N, self.N)
        else:
            return sol.T.reshape(out_shape)

# =============================================================================
# Numerical operators that re-use the grid (no stencil duplication)
# =============================================================================

def _fd_deriv(arr, axis, h):
    """
    Second-order finite-difference derivative along a specified axis.

    For interior points (index 1..-2) a centered difference of order O(h²) is used.
    At boundaries, a second-order one-sided scheme (three-point forward/backward)
    provides the same formal accuracy.

    Parameters
    ----------
    arr : ndarray
        2D input array.
    axis : {0, 1}
        Axis along which to differentiate (0 = x, 1 = y).
    h : float
        Grid spacing along the chosen axis.

    Returns
    -------
    ndarray
        Derivative array of the same shape as `arr`.
    """
    out  = np.zeros_like(arr)
    s_p  = [slice(None), slice(None)]; s_p[axis]  = slice(2, None)
    s_m  = [slice(None), slice(None)]; s_m[axis]  = slice(None, -2)
    s_c  = [slice(None), slice(None)]; s_c[axis]  = slice(1, -1)
    s_0  = [slice(None), slice(None)]; s_0[axis]  = 0
    s_1  = [slice(None), slice(None)]; s_1[axis]  = 1
    s_2  = [slice(None), slice(None)]; s_2[axis]  = 2       # ← new
    s_n1 = [slice(None), slice(None)]; s_n1[axis] = -1
    s_n2 = [slice(None), slice(None)]; s_n2[axis] = -2
    s_n3 = [slice(None), slice(None)]; s_n3[axis] = -3      # ← new
    # Interior: centred 2nd-order
    out[tuple(s_c)]  = (arr[tuple(s_p)] - arr[tuple(s_m)]) / (2 * h)
    # Boundary: one-sided 2nd-order
    out[tuple(s_0)]  = (-3*arr[tuple(s_0)]  + 4*arr[tuple(s_1)]  - arr[tuple(s_2)])  / (2 * h)
    out[tuple(s_n1)] = ( 3*arr[tuple(s_n1)] - 4*arr[tuple(s_n2)] + arr[tuple(s_n3)]) / (2 * h)
    return out


def _gradient(arr, dx, dy):
    """
    Compute the gradient of a scalar field using second-order finite differences.

    The gradient is returned as a tuple (∂ₓf, ∂ᵧf), where each component is
    computed with :func:`_fd_deriv`.

    Parameters
    ----------
    arr : ndarray
        2D scalar field.
    dx : float
        Grid spacing in the x-direction.
    dy : float
        Grid spacing in the y-direction.

    Returns
    -------
    tuple of ndarray
        (∂f/∂x, ∂f/∂y), each of the same shape as `arr`.
    """
    return _fd_deriv(arr, 0, dx), _fd_deriv(arr, 1, dy)


def _codifferential(f_x, f_y, g_inv00, g_inv01, g_inv11, sqrt_det, dx, dy):
    """
    Compute the codifferential δ of a 1‑form α = f_x dx + f_y dy on a Riemannian 2‑manifold.

    In coordinates the codifferential is given by

        δα = - (1/√g) ∂_i ( √g g^{ij} α_j ),

    where g^{ij} are the contravariant metric components and √g = √|det g|.
    The divergence term is evaluated with second‑order finite differences
    (see :func:`_fd_deriv`).  A tiny regularisation (1e-14) is added to √g
    to avoid division by zero in degenerate cases.

    Parameters
    ----------
    f_x, f_y : ndarray
        Components of the 1‑form (αₓ, αᵧ), each of shape (N, N).
    g_inv00, g_inv01, g_inv11 : ndarray
        Contravariant metric components g¹¹, g¹², g²² evaluated on the grid.
    sqrt_det : ndarray
        Square root of the determinant √|det g|.
    dx, dy : float
        Grid spacings in the x and y directions.

    Returns
    -------
    ndarray
        Scalar array representing δα, same shape as the input arrays.
    """
    flux_x = sqrt_det * (g_inv00 * f_x + g_inv01 * f_y)
    flux_y = sqrt_det * (g_inv01 * f_x + g_inv11 * f_y)
    div    = _fd_deriv(flux_x, 0, dx) + _fd_deriv(flux_y, 1, dy)
    return -div / (sqrt_det + 1e-14)


def hodge_decomposition(metric, omega_components, domain, resolution=50,
                        form_degree=1):
    """
    Numerically decompose a differential k‑form into its Hodge components
    on a 2D rectangular domain.

    Supported form degrees
    ----------------------
    ``form_degree=0``
        Decomposes a scalar function f (0‑form) as

            f = Δu + h₀,

        where Δu is the co‑exact part (the Laplacian of a scalar potential u)
        and h₀ is the constant harmonic part (the weighted mean of f).
        The exact part is zero (no (−1)-form). The potential u solves the
        Neumann problem Δu = f − h₀, with the gauge fixed by pinning one
        interior grid point to zero. The decomposition is then

            coexact   = Δu
            harmonic  = h₀

        The harmonic part corresponds to the de Rham cohomology class
        [f] ∈ H⁰_dR(M), which is simply the constant component.

    ``form_degree=1`` (default)
        Decomposes a 1‑form α = αₓ dx + αᵧ dy as

            α = dφ + ⋆dψ + h,

        where dφ is exact, ⋆dψ is co‑exact, and h is harmonic (Δh = 0).

        The scalar potentials φ and ψ satisfy the Poisson equations

            Δ₀ φ = δα            (Dirichlet boundary condition)
            Δ₀ ψ = -δ(⋆α)         (Neumann boundary condition)

        with the gauge fixed by pinning an interior grid point to zero
        for the Neumann problem.  The decomposition is then

            α_exact   = ∇φ = dφ
            α_coexact = ⋆dψ
            α_harmonic = α − α_exact − α_coexact

        The harmonic part corresponds to the de Rham cohomology class
        [α] ∈ H¹_dR(M).

    ``form_degree=2``
        Decomposes a 2‑form ω = f dx∧dy into exact and harmonic parts:

            ω = d(⋆dφ) + h,

        (there is no co‑exact component because no 3‑forms exist in 2D).
        The algorithm solves a single Dirichlet problem for the scalar
        potential φ defined by Δ₀ φ = -⋆ω, where ⋆ω = f / √|g| is the
        Hodge dual (a 0‑form).  The exact part is then

            ω_exact = d(⋆dφ)

        and the harmonic part is ω_harmonic = ω − ω_exact.

        The resulting harmonic 2‑form belongs to the second de Rham
        cohomology class [ω] ∈ H²_dR(M).

    Parameters
    ----------
    metric : Metric
        Must be 2D.
    omega_components : tuple of two (callable or sympy.Expr), or scalar
        - ``form_degree=0``: a scalar f (0‑form).
        - ``form_degree=1``: a 2‑tuple (αₓ, αᵧ).
        - ``form_degree=2``: a scalar f representing ω = f dx∧dy.
    domain : tuple
        ((x_min, x_max), (y_min, y_max))
    resolution : int, default 50
        Number of grid points per axis.
    form_degree : {0, 1, 2}, default 1
        Degree of the input differential form.

    Returns
    -------
    dict
        All cases contain a key ``'grid'`` holding the
        :class:`RiemannianGrid` instance used for the computation.

        **form_degree=0**:
            * ``'potential_u'`` – array : potential u (Neumann, pinned).
            * ``'coexact'``     – array : Δu = f − h₀.
            * ``'harmonic'``    – array : constant harmonic part h₀.

        **form_degree=1**:
            * ``'potential_phi'``  – array : φ potential (Dirichlet).
            * ``'potential_psi'``  – array : ψ potential (Neumann).
            * ``'alpha_exact'``    – tuple of two arrays : (dφ)ₓ, (dφ)ᵧ.
            * ``'alpha_coexact'``  – tuple of two arrays : (⋆dψ)ₓ, (⋆dψ)ᵧ.
            * ``'alpha_harmonic'`` – tuple of two arrays : hₓ, hᵧ.

        **form_degree=2**:
            * ``'potential_phi'``  – array : φ such that Δ₀ φ = ⋆ω.
            * ``'omega_exact'``    – array : coefficient of the exact part.
            * ``'omega_harmonic'`` – array : coefficient of the harmonic part.

    Raises
    ------
    NotImplementedError
        If ``metric.dim != 2`` or ``form_degree`` is not 0, 1, or 2.

    Examples
    --------
    **0‑form** — function with zero mean on the unit square:

    >>> from sympy import symbols, Matrix
    >>> import numpy as np
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> f = x**2 - y**2
    >>> dec0 = hodge_decomposition(m, f, ((0, 1), (0, 1)), resolution=40,
    ...                            form_degree=0)
    >>> # harmonic part ≈ 0, coexact part ≈ f

    **1‑form** — rotation form on the flat torus (purely harmonic):

    >>> dec1 = hodge_decomposition(m, (-y, x), ((0, 1), (0, 1)), resolution=40)
    >>> # harmonic part carries ≈ 100% of L² energy

    **2‑form** — constant vorticity ω = dx∧dy on the flat plane:

    >>> dec2 = hodge_decomposition(m, 1, ((0, 1), (0, 1)),
    ...                            resolution=40, form_degree=2)
    >>> # omega_exact ≈ 0, omega_harmonic ≈ 1 (contractible domain: b₂ = 0)
    """
    if metric.dim != 2:
        raise NotImplementedError("Hodge decomposition is only implemented for 2D.")
    if form_degree not in (0, 1, 2):
        raise NotImplementedError(
            "form_degree must be 1 or 2.  "
            "0-form decomposition is not supported (see docstring)."
        )

    # ------------------------------------------------------------------
    # Grid — assembled once, shared across all solves
    # ------------------------------------------------------------------
    grid = RiemannianGrid(metric, domain, resolution)
    X, Y, dx, dy = grid.X, grid.Y, grid.dx, grid.dy
    sd, gi00, gi11, gi01 = grid.sqrt_det, grid.g_inv00, grid.g_inv11, grid.g_inv01

    def _eval(c):
        fn = c if callable(c) else lambdify(metric.coords, c, 'numpy')
        return np.broadcast_to(fn(X, Y), X.shape).copy()

    # Partially-applied metric operators (avoid repeating 5 arguments everywhere)
    def _codf(fx, fy):
        return _codifferential(fx, fy, gi00, gi01, gi11, sd, dx, dy)

    def _star1(fx, fy):
        # Numerical counterpart of hodge_star(metric, 1) — must stay in sync.
        # In 0-indexed components: g00=g₁₁, g01=g₁₂, g11=g₂₂.
        # (⋆α)_x = √|g| ( -g₁₂ αₓ - g₂₂ αᵧ )
        # (⋆α)_y = √|g| (  g₁₁ αₓ + g₁₂ αᵧ )
        return ( sd * (-grid.g01 * fx - grid.g11 * fy),
                 sd * ( grid.g00 * fx + grid.g01 * fy) )

    def _star2(f):
        """Numerical ⋆ on a 2-form coefficient: ⋆(f dx∧dy) = f / √|g|."""
        return f / (sd + 1e-14)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    if form_degree == 0:
        # Scalar function decomposition
        f = _eval(omega_components) if not isinstance(omega_components, np.ndarray) else omega_components
        f = np.broadcast_to(f, (resolution, resolution)).copy()

        # Compute weighted mean
        mean = np.sum(f * grid.sqrt_det) / np.sum(grid.sqrt_det)

        # Solve Δu = f - mean with Neumann BC
        rhs = f - mean
        u = grid.solve_poisson_neumann(rhs)

        return {
            'potential_u': u,
            'coexact':     rhs,
            'harmonic':    np.full_like(f, mean),
            'grid':        grid
        }
    elif form_degree == 1:
        return _hodge_decomposition_1form(
            grid, omega_components, _eval, _codf, _star1, dx, dy
        )
    else:   # form_degree == 2
        return _hodge_decomposition_2form(
            grid, omega_components, _eval, _codf, _star1, _star2, dx, dy
        )


# -----------------------------------------------------------------------
# Private implementations — one per form degree
# -----------------------------------------------------------------------
def _hodge_decomposition_1form(grid, omega_components, _eval, _codf, _star1,
                                dx, dy):
    """
    Core 1-form Hodge decomposition.  Solves:
        Δ₀ φ = δα          (exact potential)
        Δ₀ ψ = -δ(⋆α)       (co-exact potential)
    with homogeneous Neumann BC (natural BC for Hodge potentials).
    The gauge is fixed by pinning one interior node to zero.
    """
    alpha_x = _eval(omega_components[0])
    alpha_y = _eval(omega_components[1])

    phi = grid.solve_poisson_dirichlet(-_codf(alpha_x, alpha_y))
    psi = grid.solve_poisson_neumann(_codf(*_star1(alpha_x, alpha_y)))

    ex_x, ex_y = _gradient(phi, dx, dy)
    co_x, co_y = _star1(*_gradient(psi, dx, dy))

    return {
        'potential_phi':  phi,
        'potential_psi':  psi,
        'alpha_exact':    np.stack([ex_x, ex_y]),
        'alpha_coexact':  np.stack([co_x, co_y]),
        'alpha_harmonic': np.stack([alpha_x - ex_x - co_x,
                                    alpha_y - ex_y - co_y]),
        'grid':           grid,
    }


def _hodge_decomposition_2form(grid, omega_components, _eval, _codf, _star1,
                                _star2, dx, dy):
    """
    Decompose a 2-form ω = f dx∧dy.  Uses homogeneous Dirichlet BC for the
    scalar potential φ (u = 0 on ∂M).
    """
    f       = _eval(omega_components)
    f_tilde = _star2(f)

    phi = -grid.solve_poisson_dirichlet(f_tilde)

    dphi_x, dphi_y       = _gradient(phi, dx, dy)
    star_dphi_x, star_dphi_y = _star1(dphi_x, dphi_y)

    omega_exact_coeff = (_fd_deriv(star_dphi_y, axis=0, h=grid.dx) -
                         _fd_deriv(star_dphi_x, axis=1, h=grid.dy))

    return {
        'potential_phi':   phi,
        'omega_exact':     omega_exact_coeff,
        'omega_harmonic':  f - omega_exact_coeff,
        'grid':            grid,
    }
    

def analyze_hodge_decomposition(decomp, original=None, print_report=True, show_plot=True):
    """
    Analyse the output of hodge_decomposition and print key metrics.

    Parameters
    ----------
    decomp : dict
        Dictionary returned by hodge_decomposition (must contain a 'grid' key).
    original : optional
        The original differential form used as input to hodge_decomposition.
        - For 1‑form: tuple of two arrays, callables, or SymPy expressions.
        - For 2‑form: a single array, callable, or SymPy expression.
        If not provided, reconstruction errors are not computed.
    print_report : bool, default True
        If True, print the analysis to the console.
    show_plot : bool, default True
        If True, call visualize_hodge_decomposition on decomp.

    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    # ------------------------------------------------------------------
    # Determine form degree
    # ------------------------------------------------------------------
    if 'alpha_exact' in decomp:
        form_degree = 1
    elif 'omega_exact' in decomp:
        form_degree = 2
    elif 'potential_u' in decomp:
        form_degree = 0
    else:
        raise ValueError("decomp does not contain expected keys for 0‑, 1‑ or 2‑form.")

    # ------------------------------------------------------------------
    # Grid data
    # ------------------------------------------------------------------
    grid = decomp['grid']
    X, Y = grid.X, grid.Y
    dx, dy = grid.dx, grid.dy
    sd = grid.sqrt_det                      # √|g|
    # Metric components (contravariant for 1‑form, covariant for 2‑form are also stored)
    gi00 = grid.g_inv00
    gi01 = grid.g_inv01
    gi11 = grid.g_inv11
    # Covariant components (for 2‑form's gradient)
    g00 = grid.g00
    g01 = grid.g01
    g11 = grid.g11

    # ------------------------------------------------------------------
    # Helper functions for metric‑weighted L² on 1‑forms
    # ------------------------------------------------------------------
    def weighted_dot_1form(fx, fy, gx, gy):
        return gi00 * fx * gx + gi01 * (fx * gy + fy * gx) + gi11 * fy * gy

    def weighted_norm_1form(fx, fy):
        return np.sqrt(np.sum(weighted_dot_1form(fx, fy, fx, fy) * sd * dx * dy))

    def weighted_inner_1form(fx, fy, gx, gy):
        return np.sum(weighted_dot_1form(fx, fy, gx, gy) * sd * dx * dy)

    def curl_1form(fx, fy):
        # dα = (∂θ αφ - ∂φ αθ) dθ∧dφ   (θ = x, φ = y)
        dtheta = np.gradient(fy, axis=0) / dx   # ∂/∂x of α_y
        dphi   = np.gradient(fx, axis=1) / dy   # ∂/∂y of α_x
        return dtheta - dphi

    def codifferential_1form(fx, fy):
        # δα = - (1/√g) ∂_i( √g g^{ij} α_j )
        flux_x = sd * (gi00 * fx + gi01 * fy)
        flux_y = sd * (gi01 * fx + gi11 * fy)
        div = np.gradient(flux_x, axis=0) / dx + np.gradient(flux_y, axis=1) / dy
        return -div / sd

    # ------------------------------------------------------------------
    # Helper functions for metric‑weighted L² on 2‑forms and 0‑forms
    # ------------------------------------------------------------------
    def weighted_norm_2form(f):
        return np.sqrt(np.sum(f**2 * sd * dx * dy))

    def weighted_inner_2form(f, g):
        return np.sum(f * g * sd * dx * dy)

    def grad_2form(f):
        # returns (∂f/∂x, ∂f/∂y) as 2D arrays
        return (np.gradient(f, axis=0) / dx, np.gradient(f, axis=1) / dy)

    # ------------------------------------------------------------------
    # Helper to convert original input to numerical arrays
    # ------------------------------------------------------------------
    def _eval_original(original):
        if original is None:
            return None
        if form_degree == 1:
            a, b = original
            # If both are numpy arrays, return them directly
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return a, b
            # If both are callables, call them
            if callable(a) and callable(b):
                return a(X, Y), b(X, Y)
            # Otherwise, treat as symbolic and use lambdify
            from sympy import lambdify, sympify
            a_sym = sympify(a)
            b_sym = sympify(b)
            f_a = lambdify(grid._metric.coords, a_sym, 'numpy')
            f_b = lambdify(grid._metric.coords, b_sym, 'numpy')
            return f_a(X, Y), f_b(X, Y)
        else:  # 2‑form or 0‑form
            # If it's a numpy array, return it directly
            if isinstance(original, np.ndarray):
                return original
            # If callable, call it
            if callable(original):
                return original(X, Y)
            # Otherwise, treat as symbolic
            from sympy import lambdify, sympify
            expr = sympify(original)
            f = lambdify(grid._metric.coords, expr, 'numpy')
            return f(X, Y)

    # ------------------------------------------------------------------
    # 0‑form analysis
    # ------------------------------------------------------------------
    if form_degree == 0:
        u = decomp['potential_u']
        coexact = decomp['coexact']
        harmonic = decomp['harmonic']
        f = coexact + harmonic   # original reconstructed

        # Compute energies (weighted)
        total_energy = weighted_norm_2form(f)**2
        energy_co = weighted_norm_2form(coexact)**2
        energy_ha = weighted_norm_2form(harmonic)**2

        # Reconstruction error if original provided
        if original is not None:
            true = _eval_original(original)
            if true is not None:
                diff = f - true
                max_err = np.max(np.abs(diff))
                l2_err = weighted_norm_2form(diff)
            else:
                max_err = l2_err = np.nan
        else:
            max_err = l2_err = np.nan

        # Orthogonality (exact part is zero)
        inner_co_ha = weighted_inner_2form(coexact, harmonic)

        # Norms
        norm_co = weighted_norm_2form(coexact)
        norm_ha = weighted_norm_2form(harmonic)
        norm_total = weighted_norm_2form(f)

        # Energy fractions
        total_energy = norm_total**2
        frac_co = (norm_co**2 / total_energy) * 100 if total_energy > 0 else 0
        frac_ha = (norm_ha**2 / total_energy) * 100 if total_energy > 0 else 0

        # Harmonic part properties: it should be constant
        harmonic_vals = harmonic.ravel()
        harmonic_mean = np.mean(harmonic_vals)
        harmonic_std = np.std(harmonic_vals)

        if print_report:
            print("\n=== Hodge Decomposition Analysis (0‑form) ===\n")
            print("Reconstruction error (max norm) : {:.2e}".format(max_err))
            print("Reconstruction error (L² norm)  : {:.2e}".format(l2_err))
            print("Orthogonality :")
            print(f"   ⟨coexact, harmonic⟩: {inner_co_ha:.2e}")
            print(f"Norm of coexact part : {norm_co:.3f}")
            print(f"Norm of harmonic part: {norm_ha:.3f}  (true: {norm_total:.3f})")
            print(f"Energy fractions: coexact {frac_co:.1f}%, harmonic {frac_ha:.1f}%")
            print(f"Harmonic part: mean = {harmonic_mean:.6f}, std = {harmonic_std:.2e}")

        result = {
            'form_degree': 0,
            'reconstruction_max_error': max_err,
            'reconstruction_l2_error': l2_err,
            'inner_coexact_harmonic': inner_co_ha,
            'norm_coexact': norm_co,
            'norm_harmonic': norm_ha,
            'norm_total': norm_total,
            'energy_fraction_coexact': frac_co,
            'energy_fraction_harmonic': frac_ha,
            'harmonic_mean': harmonic_mean,
            'harmonic_std': harmonic_std,
        }
        if show_plot:
            from riemannian import visualize_hodge_decomposition
            visualize_hodge_decomposition(decomp)
        return result

    # ------------------------------------------------------------------
    # 1‑form analysis
    # ------------------------------------------------------------------

    elif form_degree == 1:
        # Extract components
        ex_x, ex_y = decomp['alpha_exact']
        co_x, co_y = decomp['alpha_coexact']
        ha_x, ha_y = decomp['alpha_harmonic']
        orig_x = ex_x + co_x + ha_x
        orig_y = ex_y + co_y + ha_y

        # Evaluate original if provided
        true_xy = _eval_original(original)
        if true_xy is not None:
            true_x, true_y = true_xy
        else:
            true_x = true_y = None

        # Reconstruction errors
        if true_x is not None:
            diff_x = orig_x - true_x
            diff_y = orig_y - true_y
            max_err = np.max(np.sqrt(diff_x**2 + diff_y**2))
            l2_err = weighted_norm_1form(diff_x, diff_y)
        else:
            max_err = l2_err = np.nan

        # Orthogonality
        inner_ex_co = weighted_inner_1form(ex_x, ex_y, co_x, co_y)
        inner_ex_ha = weighted_inner_1form(ex_x, ex_y, ha_x, ha_y)
        inner_co_ha = weighted_inner_1form(co_x, co_y, ha_x, ha_y)

        # Norms
        norm_ex = weighted_norm_1form(ex_x, ex_y)
        norm_co = weighted_norm_1form(co_x, co_y)
        norm_ha = weighted_norm_1form(ha_x, ha_y)
        norm_total = weighted_norm_1form(orig_x, orig_y)

        # Energy fractions
        total_energy = norm_total**2
        frac_ex = (norm_ex**2 / total_energy) * 100 if total_energy > 0 else 0
        frac_co = (norm_co**2 / total_energy) * 100 if total_energy > 0 else 0
        frac_ha = (norm_ha**2 / total_energy) * 100 if total_energy > 0 else 0

        # Harmonic part properties
        curl_ha = curl_1form(ha_x, ha_y)
        codiff_ha = codifferential_1form(ha_x, ha_y)
        max_curl_ha = np.max(np.abs(curl_ha))
        max_codiff_ha = np.max(np.abs(codiff_ha))

        # Assemble result dict
        result = {
            'form_degree': 1,
            'reconstruction_max_error': max_err,
            'reconstruction_l2_error': l2_err,
            'inner_exact_coexact': inner_ex_co,
            'inner_exact_harmonic': inner_ex_ha,
            'inner_coexact_harmonic': inner_co_ha,
            'norm_exact': norm_ex,
            'norm_coexact': norm_co,
            'norm_harmonic': norm_ha,
            'norm_total': norm_total,
            'energy_fraction_exact': frac_ex,
            'energy_fraction_coexact': frac_co,
            'energy_fraction_harmonic': frac_ha,
            'curl_harmonic_max': max_curl_ha,
            'codiff_harmonic_max': max_codiff_ha,
        }

        # Print report
        if print_report:
            print("\n=== Hodge Decomposition Analysis (1‑form) ===\n")
            print("Reconstruction error (max norm) : {:.2e}".format(max_err))
            print("Reconstruction error (L² norm)  : {:.2e}".format(l2_err))
            print("Orthogonality :")
            print(f"   ⟨exact, coexact⟩   : {inner_ex_co:.2e}")
            print(f"   ⟨exact, harmonic⟩  : {inner_ex_ha:.2e}")
            print(f"   ⟨coexact, harmonic⟩: {inner_co_ha:.2e}")
            print(f"Norm of exact part   : {norm_ex:.3f}")
            print(f"Norm of coexact part : {norm_co:.3f}")
            print(f"Norm of harmonic part: {norm_ha:.3f}  (true: {norm_total:.3f})")
            print(f"Energy fractions: exact {frac_ex:.1f}%, coexact {frac_co:.1f}%, harmonic {frac_ha:.1f}%")
            print(f"Curl of harmonic part (max): {max_curl_ha:.2e}")
            print(f"Codiff of harmonic part (max): {max_codiff_ha:.2e}")

    # ------------------------------------------------------------------
    # 2‑form analysis
    # ------------------------------------------------------------------
    else:  # form_degree == 2
        exact = decomp['omega_exact']
        harmonic = decomp['omega_harmonic']
        orig = exact + harmonic

        # Evaluate original if provided
        true = _eval_original(original)

        # Reconstruction errors
        if true is not None:
            diff = orig - true
            max_err = np.max(np.abs(diff))
            l2_err = weighted_norm_2form(diff)
        else:
            max_err = l2_err = np.nan

        # Orthogonality
        inner_ex_ha = weighted_inner_2form(exact, harmonic)

        # Norms
        norm_ex = weighted_norm_2form(exact)
        norm_ha = weighted_norm_2form(harmonic)
        norm_total = weighted_norm_2form(orig)

        # Energy fractions
        total_energy = norm_total**2
        frac_ex = (norm_ex**2 / total_energy) * 100 if total_energy > 0 else 0
        frac_ha = (norm_ha**2 / total_energy) * 100 if total_energy > 0 else 0

        # Harmonic part properties: gradient of coefficient and gradient of coefficient/√g
        grad_harmonic = grad_2form(harmonic)
        grad_norm_harmonic = np.sqrt(grad_harmonic[0]**2 + grad_harmonic[1]**2)
        max_grad_harmonic = np.max(grad_norm_harmonic)

        # For co‑closedness: δω = 0 ⇔ ∇(harmonic / √g) = 0
        harmonic_over_sqrt = harmonic / (sd + 1e-14)
        grad_over_sqrt = grad_2form(harmonic_over_sqrt)
        grad_norm_over_sqrt = np.sqrt(grad_over_sqrt[0]**2 + grad_over_sqrt[1]**2)
        max_grad_over_sqrt = np.max(grad_norm_over_sqrt)

        # Assemble result dict
        result = {
            'form_degree': 2,
            'reconstruction_max_error': max_err,
            'reconstruction_l2_error': l2_err,
            'inner_exact_harmonic': inner_ex_ha,
            'norm_exact': norm_ex,
            'norm_harmonic': norm_ha,
            'norm_total': norm_total,
            'energy_fraction_exact': frac_ex,
            'energy_fraction_harmonic': frac_ha,
            'max_gradient_harmonic': max_grad_harmonic,
            'max_gradient_harmonic_over_sqrt': max_grad_over_sqrt,
        }

        if print_report:
            print("\n=== Hodge Decomposition Analysis (2‑form) ===\n")
            print("Reconstruction error (max norm) : {:.2e}".format(max_err))
            print("Reconstruction error (L² norm)  : {:.2e}".format(l2_err))
            print("Orthogonality :")
            print(f"   ⟨exact, harmonic⟩  : {inner_ex_ha:.2e}")
            print(f"Norm of exact part   : {norm_ex:.3f}")
            print(f"Norm of harmonic part: {norm_ha:.3f}  (true: {norm_total:.3f})")
            print(f"Energy fractions: exact {frac_ex:.1f}%, harmonic {frac_ha:.1f}%")
            print(f"Max |∇(harmonic coefficient)|         : {max_grad_harmonic:.2e}")
            print(f"Max |∇(harmonic / √g)| (co‑closedness): {max_grad_over_sqrt:.2e}")

    # ------------------------------------------------------------------
    # Call visualisation if requested
    # ------------------------------------------------------------------
    if show_plot:
        from riemannian import visualize_hodge_decomposition
        visualize_hodge_decomposition(decomp)

    return result
    
def visualize_hodge_decomposition(decomp, domain=None, resolution=50,
                                   cmap='RdBu_r', quiver_stride=3,
                                   form_degree=None):
    """
    Visualise the Hodge decomposition of a differential form on a 2D manifold.

    This function works for decompositions of both 1‑forms and 2‑forms as
    returned by :func:`hodge_decomposition`.  It automatically detects the
    form degree from the keys present in `decomp`, or you can specify it
    explicitly with the `form_degree` parameter.

    For a 1‑form α = dφ + ⋆dψ + h, the visualisation shows:
        - Top row: original vector field and its three components
          (exact, co‑exact, harmonic) as quiver plots overlaid on a
          heatmap of the magnitude.
        - Bottom row: scalar potentials φ and ψ, residual error, and an
          energy bar chart.

    For a 2‑form ω = f dx∧dy, the decomposition is ω = d(⋆dφ) + h,
    where φ is the scalar potential obtained from solving Δφ = -⋆ω with
    Dirichlet boundary conditions.  The co‑exact part is identically zero
    (no 3‑forms exist in 2D) and is therefore omitted from the visualisation.
    The display consists of:
        - Top row: original scalar field, exact part, harmonic part.
        - Bottom row: scalar potential φ, residual error, and an energy
          bar chart comparing the exact and harmonic contributions.

    The energy distribution uses the metric‑weighted L² norm:
        - For 1‑forms: ∫ ‖α‖_g² dV
        - For 2‑forms: ∫ ω² dV
    This requires that the decomposition dictionary contains a `'grid'` key
    with a :class:`RiemannianGrid` instance that provides the metric weights.
    If the grid is missing, the unweighted sum of squared values is used.

    Parameters
    ----------
    decomp : dict
        Output of :func:`hodge_decomposition`. Must contain keys appropriate
        for the form degree:
          - 1‑form: `'alpha_exact'`, `'alpha_coexact'`, `'alpha_harmonic'`,
            `'potential_phi'`, `'potential_psi'`.
          - 2‑form: `'omega_exact'`, `'omega_harmonic'`,
            `'potential_phi'`.
        For 2‑forms the original form is reconstructed from the three parts
        if not already present.
    domain : tuple, optional
        ((x_min, x_max), (y_min, y_max)) – coordinate bounds.  If the
        decomposition does not contain a `'grid'` key, this parameter is
        required to build the coordinate mesh.  If a grid is present,
        `domain` is ignored.
    resolution : int, default 50
        Number of grid points per axis.  Used only when a grid is not
        available in `decomp`.
    cmap : str, default 'RdBu_r'
        Matplotlib colormap for scalar fields (and magnitude for 1‑forms).
    quiver_stride : int, default 3
        Sub‑sampling stride for quiver arrows (only used for 1‑forms).
    form_degree : {1, 2}, optional
        Explicitly specify the form degree.  If `None`, the degree is
        inferred from the dictionary keys.

    Returns
    -------
    None
        Displays a Matplotlib figure.

    Examples
    --------
    **1‑form** on the flat torus:

    >>> from sympy import symbols, Matrix
    >>> x, y = symbols('x y', real=True)
    >>> m = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> dec = hodge_decomposition(m, (-y, x), ((0, 2*np.pi), (0, 2*np.pi)))
    >>> visualize_hodge_decomposition(dec)

    **2‑form** on the unit square (constant vorticity):

    >>> dec2 = hodge_decomposition(m, 1, ((0, 1), (0, 1)), form_degree=2)
    >>> visualize_hodge_decomposition(dec2)
    """
    # ------------------------------------------------------------------
    # Determine form degree
    # ------------------------------------------------------------------
    if form_degree is None:
        if 'alpha_exact' in decomp:
            form_degree = 1
        elif 'omega_exact' in decomp:
            form_degree = 2
        elif 'potential_u' in decomp:
            form_degree = 0
        else:
            raise ValueError(
                "Cannot infer form degree from dictionary keys. "
                "Please specify `form_degree=1` or `form_degree=2`."
            )

    # ------------------------------------------------------------------
    # Obtain grid (for coordinates and metric weights)
    # ------------------------------------------------------------------
    grid = decomp.get('grid', None)
    if grid is not None:
        # Use the grid stored in the decomposition
        X = grid.X
        Y = grid.Y
        sqrt_det = grid.sqrt_det
        resolution = X.shape[0]   # override
    else:
        # Build a simple mesh without metric weights
        if domain is None:
            raise ValueError(
                "No grid found in decomposition and `domain` not provided."
            )
        x_vals = np.linspace(domain[0][0], domain[0][1], resolution)
        y_vals = np.linspace(domain[1][0], domain[1][1], resolution)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        sqrt_det = None   # indicates no metric weights

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _magnitude(fx, fy):
        return np.sqrt(fx**2 + fy**2)

    def _weighted_energy_1form(fx, fy, w):
        """Weighted L² energy of a 1‑form."""
        mag2 = fx**2 + fy**2
        if w is not None:
            return np.sum(mag2 * w)
        else:
            return np.sum(mag2)

    def _weighted_energy_2form(f, w):
        """Weighted L² energy of a 2‑form coefficient."""
        if w is not None:
            return np.sum(f**2 * w)
        else:
            return np.sum(f**2)

    # ------------------------------------------------------------------
    # 1‑form case
    # ------------------------------------------------------------------
    if form_degree == 1:
        # Extract components
        ex_x, ex_y = decomp['alpha_exact']
        co_x, co_y = decomp['alpha_coexact']
        ha_x, ha_y = decomp['alpha_harmonic']
        alpha_x = ex_x + co_x + ha_x
        alpha_y = ex_y + co_y + ha_y
        phi = decomp['potential_phi']
        psi = decomp['potential_psi']

        # Compute energies (weighted if possible)
        total_energy = _weighted_energy_1form(alpha_x, alpha_y, sqrt_det)
        energies = [
            _weighted_energy_1form(ex_x, ex_y, sqrt_det),
            _weighted_energy_1form(co_x, co_y, sqrt_det),
            _weighted_energy_1form(ha_x, ha_y, sqrt_det)
        ]

        # Figure layout
        fig = plt.figure(figsize=(18, 9))
        gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)

        def _panel_field(ax, fx, fy, title):
            mag = _magnitude(fx, fy)
            pcm = ax.pcolormesh(X, Y, mag, shading='auto', cmap=cmap)
            plt.colorbar(pcm, ax=ax, label='‖·‖')
            sl = slice(None, None, quiver_stride)
            ax.quiver(X[sl, sl], Y[sl, sl], fx[sl, sl], fy[sl, sl],
                      color='k', alpha=0.6, scale=None, scale_units='xy')
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')

        def _panel_scalar(ax, Z, title):
            pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
            plt.colorbar(pcm, ax=ax)
            ax.contour(X, Y, Z, levels=12, colors='k', linewidths=0.5, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')

        # Top row
        _panel_field(fig.add_subplot(gs[0, 0]), alpha_x, alpha_y, 'Original  α')
        _panel_field(fig.add_subplot(gs[0, 1]), ex_x, ex_y, 'Exact part  dφ')
        _panel_field(fig.add_subplot(gs[0, 2]), co_x, co_y, 'Co‑exact part  ⋆dψ')
        _panel_field(fig.add_subplot(gs[0, 3]), ha_x, ha_y, 'Harmonic part  h')

        # Bottom row
        _panel_scalar(fig.add_subplot(gs[1, 0]), phi, 'Scalar potential  φ')
        _panel_scalar(fig.add_subplot(gs[1, 1]), psi, 'Co‑scalar potential  ψ')
        res_x = alpha_x - ex_x - co_x - ha_x
        res_y = alpha_y - ex_y - co_y - ha_y
        _panel_field(fig.add_subplot(gs[1, 2]), res_x, res_y,
                     f'Residual  ‖α - dφ - ⋆dψ - h‖\n'
                     f'(max = {_magnitude(res_x, res_y).max():.2e})')

        ax_energy = fig.add_subplot(gs[1, 3])
        labels = ['dφ', '⋆dψ', 'h']
        fracs = np.array(energies) / (total_energy + 1e-30) * 100
        bars = ax_energy.bar(labels, fracs, color=['steelblue', 'tomato', 'seagreen'])
        ax_energy.bar_label(bars, fmt='%.1f%%', padding=3)
        ax_energy.set_ylabel('% of total L² energy')
        ax_energy.set_title('Energy distribution')
        ax_energy.set_ylim(0, max(fracs) * 1.2 + 5)

        fig.suptitle('Hodge Decomposition of 1-form  α = dφ + ⋆dψ + h', fontsize=14, y=1.01)

    # ------------------------------------------------------------------
    # 2‑form case (no coexact component)
    # ------------------------------------------------------------------
    elif form_degree == 2:   # form_degree == 2
        # Retrieve components; reconstruct original if not present
        exact = decomp['omega_exact']
        harmonic = decomp['omega_harmonic']
        phi = decomp['potential_phi']

        # Original 2‑form coefficient (reconstructed)
        omega = exact + harmonic   # coexact is zero

        # Compute energies (weighted if possible)
        total_energy = _weighted_energy_2form(omega, sqrt_det)
        energies = [
            _weighted_energy_2form(exact, sqrt_det),
            _weighted_energy_2form(harmonic, sqrt_det)
        ]

        # Figure layout: 2 rows, 3 columns
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        def _panel_scalar(ax, Z, title):
            pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
            plt.colorbar(pcm, ax=ax)
            ax.contour(X, Y, Z, levels=12, colors='k', linewidths=0.5, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')

        # Top row: original, exact, harmonic
        _panel_scalar(fig.add_subplot(gs[0, 0]), omega, 'Original  ω')
        _panel_scalar(fig.add_subplot(gs[0, 1]), exact, 'Exact part  d(⋆dφ)')
        _panel_scalar(fig.add_subplot(gs[0, 2]), harmonic, 'Harmonic part  h')

        # Bottom row: potential φ, residual, energy bar chart
        _panel_scalar(fig.add_subplot(gs[1, 0]), phi, 'Potential  φ')
        residual = omega - exact - harmonic
        _panel_scalar(fig.add_subplot(gs[1, 1]), residual,
                     f'Residual ‖ω - d(⋆dφ) - h‖\n'
                     f'(max = {np.abs(residual).max():.2e})')

        ax_energy = fig.add_subplot(gs[1, 2])
        labels = ['exact', 'harmonic']
        fracs = np.array(energies) / (total_energy + 1e-30) * 100
        bars = ax_energy.bar(labels, fracs, color=['steelblue', 'seagreen'])
        ax_energy.bar_label(bars, fmt='%.1f%%', padding=3)
        ax_energy.set_ylabel('% of total weighted L² energy')
        ax_energy.set_title('Energy distribution')
        ax_energy.set_ylim(0, max(fracs) * 1.2 + 5)

        fig.suptitle('Hodge Decomposition of a 2‑Form   ω = d(⋆dφ) + h', fontsize=14, y=1.01)
    else:
        u = decomp['potential_u']
        coexact = decomp['coexact']
        harmonic = decomp['harmonic']
        f = coexact + harmonic

        total_energy = _weighted_energy_2form(f, sqrt_det)
        energies = [
            _weighted_energy_2form(coexact, sqrt_det),
            _weighted_energy_2form(harmonic, sqrt_det)
        ]

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        def _panel_scalar(ax, Z, title):
            pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
            plt.colorbar(pcm, ax=ax)
            ax.contour(X, Y, Z, levels=12, colors='k', linewidths=0.5, alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_aspect('equal')

        # Top row: original, coexact, harmonic
        _panel_scalar(fig.add_subplot(gs[0, 0]), f, 'Original  f')
        _panel_scalar(fig.add_subplot(gs[0, 1]), coexact, 'Co‑exact part  Δu')
        _panel_scalar(fig.add_subplot(gs[0, 2]), harmonic, 'Harmonic part  h₀')

        # Bottom row: potential u, residual, energy bar chart
        _panel_scalar(fig.add_subplot(gs[1, 0]), u, 'Potential  u')
        residual = f - coexact - harmonic
        _panel_scalar(fig.add_subplot(gs[1, 1]), residual,
                     f'Residual ‖f - Δu - h₀‖\n'
                     f'(max = {np.abs(residual).max():.2e})')

        ax_energy = fig.add_subplot(gs[1, 2])
        labels = ['coexact', 'harmonic']
        fracs = np.array(energies) / (total_energy + 1e-30) * 100
        bars = ax_energy.bar(labels, fracs, color=['tomato', 'seagreen'])
        ax_energy.bar_label(bars, fmt='%.1f%%', padding=3)
        ax_energy.set_ylabel('% of total weighted L² energy')
        ax_energy.set_title('Energy distribution')
        ax_energy.set_ylim(0, max(fracs) * 1.2 + 5)

        fig.suptitle('Hodge Decomposition of a 0‑Form   f = Δu + h₀', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.show()
    


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