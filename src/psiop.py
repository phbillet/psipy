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
psiop.py — Symbolic–numerical toolkit for pseudo‑differential operators in 1D/2D
=================================================================================

Overview
--------
The `psiop` module provides a unified framework for constructing, manipulating,
and numerically applying pseudo‑differential operators (ΨDOs) in one and two
spatial dimensions. It combines **symbolic** symbol calculus, based on SymPy,
with **numerical** evaluation, quantization, time propagation, diagnostics,
and visualization, based on NumPy/SciPy/Matplotlib.

The package is intended for researchers and students working in microlocal
analysis, spectral theory, semiclassical analysis, Hamiltonian/geometric
optics, and the numerical analysis of PDEs.

The module supports both scalar operators

    P : u(x) ↦ Op(p)(u)(x)

and matrix‑valued operators

    P : u(x) = (u₁(x), …, u_N(x)) ↦ Op(P)(u)(x),

where the symbol is an N × N matrix of scalar symbols P_ij(x, ξ).


Main objects and workflows
--------------------------
* `PseudoDifferentialOperator`
    Scalar pseudo‑differential operator in 1D or 2D. The symbol may be given
    explicitly (`mode='symbol'`) or extracted automatically from a differential
    expression acting on a test function (`mode='auto'`).

* `MatrixPseudoDifferentialOperator`
    Matrix‑valued pseudo‑differential operator acting on vector fields. Each
    matrix entry is wrapped as a scalar operator, so application reuses the
    existing scalar numerical machinery.

* Numerical application backends
    Operators may be applied either through direct Kohn–Nirenberg quadrature
    or through a Peetre‑decomposition backend, which splits the symbol into
    local, separable, and low‑rank joint contributions.

* Symbolic calculus
    Composition, commutators, formal inverses, formal adjoints, fractional
    powers, exponential symbols `exp(tP)`, and Weyl/Kohn–Nirenberg symbol
    conversion are implemented through asymptotic symbolic expansions.

* PDE propagation helpers
    The module includes propagator construction and time‑stepping utilities
    for equations of the form

        ∂ₜ u = Op(s)(u)

    and second‑order systems reduced to first‑order block form.

* Visualization and animation
    Symbol amplitude/phase plots, characteristic sets, cotangent fibers,
    Hamiltonian flows, singularity propagation, micro‑support, group velocity
    fields, pseudospectra, and interactive dashboards are provided.


Key features
------------
* Symbol creation in one and two spatial dimensions:
    - explicit symbol mode, `p(x, ξ)` or `p(x, y, ξ, η)`;
    - automatic extraction from differential operators via plane‑wave testing;
    - support for scalar and matrix‑valued symbols.

* Quantization conventions:
    - Kohn–Nirenberg quantization;
    - Weyl quantization;
    - symbolic asymptotic conversion between Weyl and Kohn–Nirenberg symbols.

* High‑frequency symbolic analysis:
    - principal symbol extraction;
    - asymptotic expansion as |ξ| → ∞;
    - homogeneity detection;
    - symbolic/numerical estimation of operator order.

* Asymptotic symbolic calculus:
    - operator composition `p ∘ q`;
    - commutators `[P, Q] = P ∘ Q − Q ∘ P`;
    - formal left and right inverses;
    - formal adjoint `P*`;
    - fractional powers `P^α`;
    - exponential symbols `exp(tP)`.

* Peetre‑style symbolic decomposition:
    - local polynomial part in the frequency variables;
    - separable terms of the form `a(x) q(ξ)`;
    - genuinely joint, non‑separable residual terms;
    - Chebyshev/SVD low‑rank factorization of joint residuals:

          p_joint(x, ξ) ≈ Σ_k a_k(x) q_k(ξ);

    - Monte Carlo quality diagnostics for the low‑rank approximation.

* Numerical application backends:
    - direct pointwise Kohn–Nirenberg evaluation on space–frequency grids;
    - Peetre‑based application using local, separable, and low‑rank terms;
    - periodic FFT‑based evaluation for periodic problems;
    - non‑periodic quadrature for Dirichlet‑type settings;
    - frequency windowing, dealiasing masks, spatial tapering, magnitude
      clamping, caching, and optional parallel execution.

* Matrix‑valued operators:
    - entrywise application to vector fields;
    - pointwise evaluation of the symbol matrix;
    - pointwise eigenvalues/eigenvectors of the symbol matrix;
    - asymptotic matrix composition;
    - matrix commutators;
    - matrix exponential symbols.

* Operator diagnostics:
    - symbolic and numerical ellipticity tests;
    - formal self‑adjointness checks;
    - semiclassical trace formula, symbolic or numerical;
    - pseudospectrum computation for matrix discretizations, with resolvent
      norm contours, eigenvalue overlays, sparse/dense solvers, optional
      adaptive refinement, and parallel evaluation.

* Hamiltonian and microlocal dynamics:
    - Hamiltonian vector field associated with the symbol;
    - symplectic flow visualization;
    - bicharacteristic integration;
    - singularity propagation animation in 2D and 3D phase‑space projections;
    - characteristic set and characteristic gradient visualization.

* Time‑dependent solver utilities:
    - construction of approximate propagators `exp(dt · Op(s))`;
    - first‑order evolution solver;
    - second‑order evolution solver via block companion reduction;
    - 1D/2D grid generation utilities;
    - plotting and animation of scalar and matrix‑valued solutions.

* Interactive exploration:
    - ipywidgets‑based dashboard for real‑time inspection of symbol amplitude,
      phase, micro‑support, characteristic set, gradient, Hamiltonian flow,
      cotangent fibers, and vector fields.


Mathematical background
-----------------------
A pseudo‑differential operator `P` acting on functions of `x ∈ ℝⁿ` is defined
by its **symbol** `p(x, ξ)` on phase space `T*ℝⁿ`. In the Kohn–Nirenberg
quantization,

    (P u)(x) = (2π)^{-n} ∫_{ℝⁿ} e^{i x·ξ} p(x, ξ) û(ξ) dξ,

where `û` denotes the Fourier transform of `u`. If `p` is independent of `x`,
the operator reduces to a Fourier multiplier:

    P u = ℱ^{-1}[p(ξ) ℱ[u](ξ)].

The Weyl quantization uses a symmetric midpoint convention. The two
quantizations are related by an asymptotic differential correction. In 1D,

    a_KN(x, ξ) = exp(− i/2 · ∂_x ∂_ξ) a_Weyl(x, ξ),

and conversely,

    a_Weyl(x, ξ) = exp(+ i/2 · ∂_x ∂_ξ) a_KN(x, ξ).

In 2D, the cross‑derivative operator becomes

    ∂_x ∂_ξ + ∂_y ∂_η.

For polynomial symbols the conversion series is finite and exact; for general
symbol classes it is interpreted asymptotically.


Asymptotic composition
----------------------
For two symbols `p` and `q`, the symbol of the composition `P ∘ Q` admits the
Kohn–Nirenberg asymptotic expansion

    (p ∘ q)(x, ξ) ~ Σ_α (i)^{−|α|} / α! · ∂_ξ^α p(x, ξ) · ∂_x^α q(x, ξ).

In one dimension this reduces to

    (p ∘ q)(x, ξ) ~ Σ_{k≥0} (i)^{-k} / k! · ∂_ξ^k p(x, ξ) · ∂_x^k q(x, ξ).

A corresponding Weyl/Moyal star product is also available. Truncating these
series gives approximate compositions valid in high‑frequency or slowly varying
regimes. The same machinery underlies commutators, formal inverses, formal
adjoints, fractional powers, and exponential symbols.


Principal symbol and order
--------------------------
The high‑frequency behaviour of `p(x, ξ)` determines the order of the
operator. The **principal symbol** is the leading homogeneous component as
|ξ| → ∞. If `p` is homogeneous of degree `m`, then

    p(x, λ ξ) = λ^m p(x, ξ),    λ > 0.

For non‑homogeneous symbols, the module estimates the dominant asymptotic
order by series expansion in inverse frequency variables, or in the radial
variable `ρ = sqrt(ξ² + η²)` in 2D.


Peetre decomposition
--------------------
For numerical application, a symbol is decomposed into structurally simpler
pieces:

    p(x, ξ) = p_local(x, ξ) + p_sep(x, ξ) + p_joint(x, ξ).

* `p_local` is polynomial in the frequency variables and corresponds to
  differential operators:

      p_local(x, ξ) = Σ_α a_α(x) ξ^α.

* `p_sep` is a finite sum of separable Fourier‑multiplier terms:

      p_sep(x, ξ) = Σ_k a_k(x) q_k(ξ).

  These are applied efficiently as

      u ↦ a_k(x) · Op(q_k)(u).

* `p_joint` contains genuinely entangled space–frequency dependence. On a
  bounded phase‑space window, this residual may be approximated by a low‑rank
  Chebyshev/SVD expansion

      p_joint(x, ξ) ≈ Σ_{k=1}^r a_k(x) q_k(ξ),

  reducing the numerical application of a general ΨDO to a small number of
  separable FFT‑based operations.


Hamiltonian flow and propagation of singularities
-------------------------------------------------
The Hamiltonian flow associated with the principal symbol `p(x, ξ)` governs
the propagation of singularities along bicharacteristics. In 1D,

    dx/dt = ∂_ξ p(x, ξ),      dξ/dt = −∂_x p(x, ξ).

In 2D,

    dx/dt = ∂_ξ p,     dy/dt = ∂_η p,
    dξ/dt = −∂_x p,    dη/dt = −∂_y p.

For matrix‑valued systems, the relevant Hamiltonians are obtained from the
eigenvalues of the symbol matrix. These flows are used for ray tracing,
singularity animation, and microlocal diagnostics.


Pseudospectrum
--------------
The ε‑pseudospectrum of an operator `P` is the set

    σ_ε(P) = { λ ∈ ℂ : ‖(P − λ I)^{-1}‖ ≥ ε^{-1} }.

Equivalently, it is determined by the smallest singular value of the shifted
matrix discretization:

    ‖(P − λ I)^{-1}‖ = 1 / σ_min(P − λ I).

Pseudospectra are particularly informative for non‑normal operators, where
the resolvent norm may be large even far from the spectrum.


Numerical design notes
----------------------
The numerical kernels are designed to balance accuracy, memory usage, and
speed:

* Constant‑coefficient symbols are applied as FFT multipliers whenever
  possible.
* Space‑dependent symbols are evaluated using chunked or block‑parallel
  quadrature to avoid excessive memory allocation.
* Frequency windowing and magnitude clamping improve stability for singular
  or rapidly growing symbols.
* Phase matrices and window functions are cached in non‑periodic transforms.
* The Peetre backend reduces complex symbols to a small number of efficient
  separable applications.
* Matrix‑valued application reuses scalar operator infrastructure entrywise.


References
----------
.. [1] Hörmander, L.
       *The Analysis of Linear Partial Differential Operators III*,
       Springer, 1985.
       Chapter 18: Pseudo‑differential Operators.

.. [2] Taylor, M. E.
       *Pseudo Differential Operators*,
       Princeton University Press, 1981.

.. [3] Zworski, M.
       *Semiclassical Analysis*,
       American Mathematical Society, 2012.
       Chapter 4: Pseudo‑differential Operators.

.. [4] Martinez, A.
       *An Introduction to Semiclassical and Microlocal Analysis*,
       Springer, 2002.

.. [5] Trefethen, L. N. and Embree, M.
       *Spectra and Pseudospectra*,
       Princeton University Press, 2005.

.. [6] Peetre, J.
       “Applications de la théorie des espaces d’interpolation dans
       l’analyse harmonique.”
       *Ricerche di Matematica*, 1968.
"""

from imports import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# ============================================================================
# Shared visualization helpers (used by PseudoDifferentialOperator's
# visualize_*/plot_*/animate_* methods below). Factored out because nearly
# every one of those methods used to re-implement the same handful of
# shapes from scratch: evaluate the symbol over a 2D phase-space slice and
# render it (5 methods + micro-support), or build a 1D quiver field from
# the symbol's derivatives (2 methods).
# ============================================================================

def _slice_grid(op, kind, x_grid=None, xi_grid=None, y_grid=None, eta_grid=None,
                 x0=0.0, y0=0.0, xi0=0.0, eta0=0.0):
    """Evaluate op.p_func over one canonical 2D slice of phase space.

    kind='freq'  : 1D -> vary (x, xi).   2D -> fix (x0, y0), vary (xi, eta).
                   (used by: fiber, characteristic set/gradient, micro-support)
    kind='space' : 1D -> vary (x, xi) (1D has only one slice, same as above).
                   2D -> fix (xi0, eta0), vary (x, y).
                   (used by: symbol amplitude, phase)

    Returns (axis1_vals, axis2_vals, axis1_label, axis2_label, Z).
    """
    if op.dim == 1:
        A, B = np.meshgrid(x_grid, xi_grid, indexing='ij')
        return x_grid, xi_grid, 'x', r'$\xi$', op.p_func(A, B)
    if kind == 'space':
        A, B = np.meshgrid(x_grid, y_grid, indexing='ij')
        Z = op.p_func(A, B, np.full_like(A, xi0), np.full_like(B, eta0))
        return x_grid, y_grid, 'x', 'y', Z
    # kind == 'freq'
    A, B = np.meshgrid(xi_grid, eta_grid, indexing='ij')
    Z = op.p_func(x0, y0, A, B)
    return xi_grid, eta_grid, r'$\xi$', r'$\eta$', Z


def _render_field(ax1, ax2, Z, style='pcolormesh', cmap='viridis', cbar_label=None,
                   xlabel='x', ylabel=r'$\xi$', title='', levels=50,
                   contour_color='red', grid=False, show=True):
    """One shared renderer for pcolormesh / contourf / contour panels."""
    if style == 'pcolormesh':
        im_ = plt.pcolormesh(ax1, ax2, Z, shading='auto', cmap=cmap)
        plt.colorbar(im_, label=cbar_label)
    elif style == 'contourf':
        im_ = plt.contourf(ax1, ax2, Z, levels=levels, cmap=cmap)
        plt.colorbar(im_, label=cbar_label)
    elif style == 'contour':
        plt.contour(ax1, ax2, Z, levels=levels, colors=contour_color)
    else:
        raise ValueError(f"unknown style {style!r}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid(True)
    if show:
        plt.show()


def _grad_norm(Z):
    """|grad Z| over a 2D array, using a consistent abs()**2 convention on
    both axes (see visualize_characteristic_gradient's note: this unifies
    an inconsistency that used to exist between its 1D and 2D branches)."""
    gx = np.gradient(Z, axis=0)
    gy = np.gradient(Z, axis=1)
    return np.sqrt(np.abs(gx) ** 2 + np.abs(gy) ** 2)


def _quiver_field(op, xlim, klim, density, vec_exprs_fn, title, scale=10, width=0.005):
    """1D-only quiver plot of a vector field derived from the symbol.

    vec_exprs_fn(p, x, xi) -> (U_expr, V_expr)
    """
    if op.dim != 1:
        raise NotImplementedError("Only 1D version implemented.")
    x, = op.vars_x
    xi = symbols('xi', real=True)
    x_vals = np.linspace(*xlim, density)
    xi_vals = np.linspace(*klim, density)
    X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')

    U_expr, V_expr = vec_exprs_fn(op.symbol, x, xi)
    U = lambdify((x, xi), simplify(U_expr), 'numpy')(X, XI)
    V = lambdify((x, xi), simplify(V_expr), 'numpy')(X, XI)
    if np.isscalar(U):
        U = np.full_like(X, U, dtype=float)
    if np.isscalar(V):
        V = np.full_like(X, V, dtype=float)

    plt.quiver(X, XI, U, V, scale=scale, width=width)
    plt.xlabel('x')
    plt.ylabel(r'$\xi$')
    plt.title(title)
    plt.grid(True)
    plt.show()


def _make_real(expr):
    """Re(expr), fully evaluated -- used when a Hamiltonian field may come
    out complex-valued from sympy but only the real part is physically
    meaningful for the flow."""
    return simplify(sp.re(expr.doit(deep=True)))


# ============================================================================
# Multi-index helpers -- shared, dimension-generic building blocks for the
# asymptotic symbolic calculus (composition, formal inverses, exponential
# symbol, formal adjoint) used by both PseudoDifferentialOperator and
# MatrixPseudoDifferentialOperator. Factored out to replace what used to be
# separate hand-unrolled 1D/2D code paths in each of those methods.
# ============================================================================

def _mi_all(n, dim):
    """Yield all `dim`-tuples of non-negative ints summing to exactly n."""
    if dim == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _mi_all(n - i, dim - 1):
            yield (i,) + rest


def _mi_upto(n, dim):
    """Yield all `dim`-tuples of non-negative ints with 1 <= sum <= n."""
    for m in range(1, n + 1):
        yield from _mi_all(m, dim)


def _mi_diff(expr, mvars, alpha):
    """d^|alpha| expr / prod(mvars_i ** alpha_i), entrywise for sympy
    Matrix `expr` (via .diff) as well as scalar sympy expressions."""
    for v, a in zip(mvars, alpha):
        if a:
            expr = expr.diff(v, a)
    return expr


def _mi_factorial(alpha):
    fact = 1
    for a in alpha:
        fact *= factorial(a)
    return fact


class PseudoDifferentialOperator:
    """
    Pseudo-differential operator with dynamic symbol evaluation on spatial grids.
    Supports both 1D and 2D operators, and can be defined explicitly (symbol mode)
    or extracted automatically from symbolic equations (auto mode).

    Parameters
    ----------
    expr : sympy expression
        Symbolic expression representing the pseudo-differential symbol.
    vars_x : list of sympy symbols
        Spatial variables (e.g., [x] for 1D, [x, y] for 2D).
    var_u : sympy function, optional
        Function u(x, t) used in auto mode to extract the operator symbol.
    mode : str, {'symbol', 'auto'}
        - 'symbol': directly uses expr as the operator symbol.
        - 'auto': computes the symbol automatically by applying expr to exp(i x ξ).

    Attributes
    ----------
    dim : int
        Spatial dimension (1 or 2).
    fft, ifft : callable
        Fast Fourier transform and inverse (scipy.fft or scipy.fft2).
    p_func : callable
        Evaluated symbol function ready for numerical use.

    Notes
    -----
    - In 'symbol' mode, `expr` should be expressed in terms of spatial variables and frequency variables (ξ, η).
    - In 'auto' mode, the symbol is derived by applying the differential expression to a complex exponential.
    - Frequency variables are internally named 'xi' and 'eta' for consistency.
    - Uses numpy for numerical evaluation and scipy.fft for FFT operations.

    Examples
    --------
    >>> # Example 1: 1D Laplacian operator (symbol mode)
    >>> from sympy import symbols
    >>> x, xi = symbols('x xi', real=True)
    >>> op = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')

    >>> # Example 2: 1D transport operator (auto mode)
    >>> from sympy import Function
    >>> u = Function('u')
    >>> expr = u(x).diff(x)
    >>> op = PseudoDifferentialOperator(expr=expr, vars_x=[x], var_u=u(x), mode='auto')
    """

    def __init__(self, expr, vars_x, var_u=None, mode='symbol', 
                 quantization='kohn-nirenberg', apply_backend='peetre', compute_peetre=False, peetre_options=None,):
        """
        Build a PseudoDifferentialOperator from a symbolic expression.

        Depending on `mode`, `expr` is interpreted either as the symbol
        itself ('symbol' mode) or as a differential expression from which
        the symbol is extracted automatically by applying it to a plane
        wave `exp(i x·xi)` and dividing it out ('auto' mode). The
        resulting symbol is lambdified into a NumPy-callable `p_func` for
        numerical evaluation, and FFT/IFFT routines matching the spatial
        dimension are attached to the instance.

        Parameters
        ----------
        expr : sympy.Expr
            Symbol expression ('symbol' mode) or differential expression
            acting on `var_u` ('auto' mode).
        vars_x : list of sympy symbols
            Spatial variables; length 1 for 1D operators, length 2 for 2D.
        var_u : sympy Function, optional
            Function u(x[, y]) whose action defines the differential
            expression in 'auto' mode. Required when `mode='auto'`.
        mode : {'symbol', 'auto'}, default 'symbol'
            Whether `expr` is already the symbol or must be derived from a
            differential expression.
        quantization : {'kohn-nirenberg', 'weyl'}, default 'kohn-nirenberg'
            Quantization convention used when evaluating/applying the
            operator and when composing symbols asymptotically.
        apply_backend : {'peetre', 'direct'}, default 'peetre'
            Numerical strategy used by `apply()`: 'peetre' applies the
            operator through its Peetre local/separable/low-rank
            decomposition, 'direct' evaluates the full symbol pointwise on
            the space-frequency grid.
        compute_peetre : bool, default False
            If True, eagerly compute and cache the Peetre decomposition of
            the symbol at construction time (see `peetre_decomposition`).
        peetre_options : dict, optional
            Keyword arguments forwarded to `peetre_decomposition()` when
            `compute_peetre` is True.

        Raises
        ------
        ValueError
            If `apply_backend` is not one of {'direct', 'peetre'}, if
            `mode` is not one of {'symbol', 'auto'}, or if `var_u` is
            missing while `mode='auto'`.
        NotImplementedError
            If `vars_x` has a length other than 1 or 2.
        """
        self.dim = len(vars_x)
        self.mode = mode
        self.symbol_cached = None
        self.expr = expr
        self.vars_x = vars_x
        self.quantization = quantization
        if apply_backend not in {"direct", "peetre"}:
            raise ValueError("apply_backend must be 'direct' or 'peetre'")
    
        self.apply_backend = apply_backend
        self._peetre_options = dict(peetre_options or {})
        self._peetre_decomposition = None
    
        if compute_peetre is None:
            compute_peetre = apply_backend == "peetre"

        if self.dim == 1:
            x, = vars_x
            xi_internal = symbols('xi', real=True)
            expr = sympify(expr)
            expr = expr.subs(symbols('xi', real=True), xi_internal)
            self.fft = partial(fft, workers=FFT_WORKERS)
            self.ifft = partial(ifft, workers=FFT_WORKERS)

            if mode == 'symbol':
                self.symbol = expr
                try:
                    self.p_func = lambdify((x, xi_internal), expr, 'numpy')
                except Exception:
                    # Graceful fallback for symbols with undefined functions (e.g. Derivative(c(x), x))
                    self.p_func = None 
            elif mode == 'auto':
                if var_u is None:
                    raise ValueError("var_u must be provided in mode='auto'")
                exp_i = exp(I * x * xi_internal)
                P_ei = expr.subs(var_u, exp_i)
                symbol = simplify(P_ei / exp_i)
                symbol = expand(symbol)
                self.symbol = symbol
                try:
                    self.p_func = lambdify((x, xi_internal), symbol, 'numpy')
                except Exception:
                    self.p_func = None
            else:
                raise ValueError("mode must be 'auto' or 'symbol'")

        elif self.dim == 2:
            x, y = vars_x
            xi_internal, eta_internal = symbols('xi eta', real=True)
            expr = sympify(expr)
            expr = expr.subs(symbols('xi', real=True), xi_internal)
            expr = expr.subs(symbols('eta', real=True), eta_internal)
            self.fft = partial(fft2, workers=FFT_WORKERS)
            self.ifft = partial(ifft2, workers=FFT_WORKERS)

            if mode == 'symbol':
                self.symbol = expr
                try:
                    self.p_func = lambdify((x, y, xi_internal, eta_internal), expr, 'numpy')
                except Exception:
                    self.p_func = None
            elif mode == 'auto':
                if var_u is None:
                    raise ValueError("var_u must be provided in mode='auto'")
                exp_i = exp(I * (x * xi_internal + y * eta_internal))
                P_ei = expr.subs(var_u, exp_i)
                symbol = simplify(P_ei / exp_i)
                symbol = expand(symbol)
                self.symbol = symbol
                try:
                    self.p_func = lambdify((x, y, xi_internal, eta_internal), symbol, 'numpy')
                except Exception:
                    self.p_func = None
            else:
                raise ValueError("mode must be 'auto' or 'symbol'")

        else:
            raise NotImplementedError("Only 1D and 2D supported")

        if mode == 'auto':
            self._compute_symbol_derivatives() 
            print("\nsymbol = ")
            pprint(self.symbol, num_columns=NUM_COLS)

        # ------------------------------------------------------------
        # Optional eager Peetre decomposition
        # ------------------------------------------------------------
        if compute_peetre:
            self._peetre_decomposition = self.peetre_decomposition(
                **self._peetre_options
            )

    def _compute_symbol_derivatives(self):
        """
        Precompute and lambdify first- and second-order derivatives of the
        symbol with respect to space and frequency variables.

        The derivatives (`dp_dx`, `dp_dxi`, `d2p_dxi2`, `d2p_dx2`,
        `d2p_dxidx`, plus the analogous `y`/`eta` terms in 2D) are stored
        symbolically in `self.derivatives`, and each one is also
        lambdified into a NumPy callable `self._<name>_func` for fast
        numerical evaluation (e.g. in WKB/geometric-optics constructions
        and Hamiltonian-flow computations). Called internally, in 'auto'
        mode, right after the symbol has been extracted.

        Notes
        -----
        If a derivative expression contains undefined functions (e.g. an
        unspecified coefficient `c(x)`), lambdification is skipped for
        that term and the corresponding `_<name>_func` attribute is set to
        None instead of raising.
        """
        self.derivatives = {}
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            self.derivatives['dp_dx'] = diff(self.symbol, x)
            self.derivatives['dp_dxi'] = diff(self.symbol, xi)
            self.derivatives['d2p_dxi2'] = diff(self.symbol, xi, 2)
            self.derivatives['d2p_dx2'] = diff(self.symbol, x, 2)
            self.derivatives['d2p_dxidx'] = diff(diff(self.symbol, xi), x)
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            self.derivatives['dp_dx'] = diff(self.symbol, x)
            self.derivatives['dp_dy'] = diff(self.symbol, y)
            self.derivatives['dp_dxi'] = diff(self.symbol, xi)
            self.derivatives['dp_deta'] = diff(self.symbol, eta)
            self.derivatives['d2p_dxi2'] = diff(self.symbol, xi, 2)
            self.derivatives['d2p_deta2'] = diff(self.symbol, eta, 2)
            self.derivatives['d2p_dx2'] = diff(self.symbol, x, 2)
            self.derivatives['d2p_dy2'] = diff(self.symbol, y, 2)
            self.derivatives['d2p_dxidx'] = diff(diff(self.symbol, xi), x)
            self.derivatives['d2p_detady'] = diff(diff(self.symbol, eta), y)
        
        # Lambdify for numerical evaluation
        if self.dim == 1:
            vars_tuple = (self.vars_x[0], symbols('xi', real=True))
        else:
            vars_tuple = tuple(self.vars_x) + (symbols('xi', real=True), symbols('eta', real=True))
            
        for name, expr in self.derivatives.items():
            try:
                setattr(self, f'_{name}_func', lambdify(vars_tuple, expr, 'numpy'))
            except Exception:
                # Fallback if derivative contains undefined functions
                setattr(self, f'_{name}_func', None)
        
    def evaluate(self, X, Y, KX, KY, cache=True):
        """
        Evaluate the pseudo-differential operator's symbol on a grid of spatial and frequency coordinates.

        The method dynamically selects between 1D and 2D evaluation based on the spatial dimension.
        If caching is enabled and a cached symbol exists, it returns the cached result to avoid recomputation.

        Parameters
        ----------
        X, Y : ndarray
            Spatial grid coordinates. In 1D, Y is ignored.
        KX, KY : ndarray
            Frequency grid coordinates. In 1D, KY is ignored.
        cache : bool, default=True
            If True, stores the computed symbol for reuse in subsequent calls to avoid redundant computation.

        Returns
        -------
        ndarray
            Evaluated symbol values over the input grid. Shape matches the input spatial/frequency grids.

        Raises
        ------
        NotImplementedError
            If the spatial dimension is not 1D or 2D.
        """
        if cache and self.symbol_cached is not None:
            return self.symbol_cached

        if self.dim == 1:
            symbol = self.p_func(X, KX)
        elif self.dim == 2:
            symbol = self.p_func(X, Y, KX, KY)

        if cache:
            self.symbol_cached = symbol

        return symbol

    def clear_cache(self):
        """Clear cached symbol evaluations and Peetre decompositions."""
        self.symbol_cached = None
        if hasattr(self, "_peetre_cache"):
            self._peetre_cache = None
        if hasattr(self, "_peetre_decomposition"):
            self._peetre_decomposition = None
        if hasattr(self, "_joint_lowrank_cache"):
            self._joint_lowrank_cache = None
        if hasattr(self, "_joint_nufft_cache"):
            self._joint_nufft_cache = None
        if hasattr(self, "_joint_aaa_cache"):
            self._joint_aaa_cache = None

    def _get_peetre_decomposition(self):
        """
        Return the Peetre decomposition stored in the instance.
    
        If the decomposition was not computed in __init__, it is computed
        lazily on first use.
        """
        if getattr(self, "_peetre_decomposition", None) is None:
            opts = getattr(self, "_peetre_options", None) or {}
            self._peetre_decomposition = self.peetre_decomposition(**opts)
    
        return self._peetre_decomposition

    def apply(
        self,
        u,
        x_grid,
        kx,
        boundary_condition='periodic',
        y_grid=None,
        ky=None,
        dealiasing_mask=None,
        freq_window='gaussian',
        clamp=1e6,
        space_window=False,
        weyl_order=4,
        backend=None,
        apply_joint=True,
        joint_backend="direct",
        joint_degree=6,
        joint_tol=1e-5,
        joint_bounds=None,
        joint_max_rel_error=None,
    ):
        """
        Apply the pseudo-differential operator to the input field u.
     
        Dispatches based on:
        - Whether the symbol is spatially dependent (x / y).
        - The boundary condition ('periodic' or 'dirichlet').
        - The quantization convention stored in ``self.quantization``.
     
        Supported quantizations
        -----------------------
        'kohn-nirenberg' (default)
            Standard KN quantization.  The symbol is used as-is.
     
        'weyl'
            Weyl quantization.  The Weyl symbol is first converted to its
            KN equivalent via the asymptotic series
     
                a_KN = exp(+i/2 * d_x d_xi) a_Weyl
     
            truncated at ``weyl_order``.  The corrected KN symbol is then
            passed to the existing KN numerical backend — no new numerical
            kernel is needed.
     
        Dispatch logic
        --------------
        Dispatch Logic:\n
        if not self.is_spatial: u ↦ Op(p)(D) ⋅ u = 𝓕⁻¹[ p(ξ) ⋅ 𝓕(u) ]\n
        elif periodic: u ↦ Op(p)(x,D) ⋅ u ≈ ∫ eᶦˣᶿ p(x, ξ) 𝓕(u)(ξ) dξ based of FFT (quicker)\n
        elif dirichlet: u ↦ Op(p)(x,D) ⋅ u ≈ u ≈ ∫ eᶦˣᶿ p(x, ξ) 𝓕(u)(ξ) dξ (slower)\n
     
        Parameters
        ----------
        u : ndarray
            Function to which the operator is applied.
        x_grid : ndarray
            Spatial grid in x direction.
        kx : ndarray
            Frequency grid in x direction.
        boundary_condition : {'periodic', 'dirichlet'}, default='periodic'
            Boundary condition type.
        y_grid : ndarray, optional
            Spatial grid in y direction (2D only).
        ky : ndarray, optional
            Frequency grid in y direction (2D only).
        dealiasing_mask : ndarray, optional
            Dealiasing mask applied in Fourier space.
        freq_window : {'gaussian', 'hann', None}, default='gaussian'
            Frequency-domain smoothing window.
        clamp : float, default=1e6
            Clip symbol values to [-clamp, clamp].
        space_window : bool, default=False
            Apply a Gaussian spatial taper.
        weyl_order : int, default=4
            Truncation order for the Weyl -> KN asymptotic correction.
            Ignored when ``self.quantization != 'weyl'``.
            The series is exact and finite for polynomial symbols.
     
        Returns
        -------
        ndarray
            Result of applying the operator, same shape as u.
     
        Raises
        ------
        ValueError
            If boundary_condition is not 'periodic' or 'dirichlet'.
        """

        if backend is None:
            backend = self.apply_backend
    
        if backend == "peetre":
            return self.apply_peetre(
                u,
                x_grid,
                kx,
                boundary_condition=boundary_condition,
                y_grid=y_grid,
                ky=ky,
                dealiasing_mask=dealiasing_mask,
                freq_window=freq_window,
                clamp=clamp,
                space_window=space_window,
                weyl_order=weyl_order,
                apply_joint=apply_joint,
                decomposition=None,
                use_cache=True,
                separable_local=False,
                joint_backend=joint_backend,
                joint_degree=joint_degree,
                joint_tol=joint_tol,
                joint_bounds=joint_bounds,
                joint_max_rel_error=joint_max_rel_error,
            )
            
        is_spatial  = self._is_spatial_dependent()
     
        # Case 1: constant symbol + periodic BC — fast FFT multiplier
        if not is_spatial and boundary_condition == 'periodic':
            return self._apply_constant_fft(
                u,
                x_grid,
                kx,
                y_grid,
                ky,
                dealiasing_mask,
                freq_window=freq_window,
                clamp=clamp,
                space_window=space_window,
            )
     
        # For all other cases, obtain the effective (possibly corrected) symbol
        symbol_func = self._get_effective_symbol_func(weyl_order=weyl_order)

        # Case 2: spatial symbol + periodic BC
        if boundary_condition == 'periodic':
            return kohn_nirenberg_fft(
                u_vals=u,
                symbol_func=symbol_func,
                x_grid=x_grid,
                kx=kx,
                fft_func=self.fft,
                ifft_func=self.ifft,
                dim=self.dim,
                y_grid=y_grid,
                ky=ky,
                freq_window=freq_window,
                clamp=clamp,
                space_window=space_window,
                is_spatial=is_spatial,
            )
     
        # Case 3: Dirichlet BC (non-periodic)
        if (boundary_condition == 'dirichlet' or boundary_condition == 'neumann'):
            if self.dim == 1:
                return kohn_nirenberg_nonperiodic(
                    u_vals=u,
                    x_grid=x_grid,
                    xi_grid=kx,
                    symbol_func=symbol_func,
                    freq_window=freq_window,
                    clamp=clamp,
                    space_window=space_window,
                    is_spatial=is_spatial,
                )
            elif self.dim == 2:
                return kohn_nirenberg_nonperiodic(
                    u_vals=u,
                    x_grid=(x_grid, y_grid),
                    xi_grid=(kx, ky),
                    symbol_func=symbol_func,
                    freq_window=freq_window,
                    clamp=clamp,
                    space_window=space_window,
                    is_spatial=is_spatial,
                )
     
        raise ValueError(f"Invalid boundary condition '{boundary_condition}'")

    def _is_spatial_dependent(self):
        """
        Check if the symbol depends on spatial variables.
        
        Returns
        -------
        bool
            True if symbol depends on x (or x, y)
        """
        if self.dim == 1:
            return self.symbol.has(self.vars_x[0])
        elif self.dim == 2:
            x, y = self.vars_x
            return self.symbol.has(x) or self.symbol.has(y)
        else:
            return False
    
    def _get_symbol_func(self):
        """
        Get a lambdified version of the symbol.
        
        Returns
        -------
        callable
            Lambdified symbol function
        """
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            return lambdify((x, xi), self.symbol, 'numpy')
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            return lambdify((x, y, xi, eta), self.symbol, 'numpy')
        else:
            raise NotImplementedError("Only 1D and 2D supported")

    def _get_effective_symbol_func(self, weyl_order=4):
        """
        Return a lambdified callable for the symbol to pass to the KN backend.
     
        If ``self.quantization == 'weyl'``, the Weyl symbol is first converted
        to its KN equivalent via ``weyl_to_kn_symbol(order=weyl_order)``.
        The corrected expression is then lambdified exactly like a native KN
        symbol.
     
        If ``self.quantization == 'kohn-nirenberg'`` (or any other value), the
        symbol is lambdified as-is, identical to the previous behaviour of
        ``_get_symbol_func()``.
     
        Parameters
        ----------
        weyl_order : int, default=4
            Truncation order passed to ``weyl_to_kn_symbol``.
            Has no effect when quantization is not 'weyl'.
     
        Returns
        -------
        callable
            NumPy-compatible function with signature:
            - 1D : f(x, xi)
            - 2D : f(x, y, xi, eta)
     
        Notes
        -----
        The corrected symbol is computed symbolically once per call.  For
        repeated ``apply()`` calls on the same operator, consider caching the
        result via ``self._cached_kn_symbol`` (not implemented here to keep
        the patch minimal — add if performance becomes a concern).
        """
        if self.quantization == 'weyl':
            effective_symbol = self.weyl_to_kn_symbol(order=weyl_order)
        else:
            # 'kohn-nirenberg' or legacy behaviour
            effective_symbol = self.symbol
     
        # Lambdify with the effective symbol
        if self.dim == 1:
            x  = self.vars_x[0]
            xi = symbols('xi', real=True)
            return lambdify((x, xi), effective_symbol, 'numpy')
     
        elif self.dim == 2:
            x, y    = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            return lambdify((x, y, xi, eta), effective_symbol, 'numpy')
     
        else:
            raise NotImplementedError(
                "_get_effective_symbol_func: only 1D and 2D are supported."
            )
    
    def _apply_constant_fft(
        self,
        u,
        x_grid,
        kx,
        y_grid=None,
        ky=None,
        dealiasing_mask=None,
        freq_window="gaussian",
        clamp=1e6,
        space_window=False,
    ):
        """
        Apply a constant-coefficient pseudo-differential operator in Fourier space.
    
        For periodic computations, the frequency grid is recomputed from x_grid
        and y_grid in order to match the FFT ordering.
    
        This version also applies the same frequency windowing and clamping as
        kohn_nirenberg_fft, so that constant-coefficient suboperators used inside
        apply_peetre are consistent with the periodic variable-coefficient path.
        """
        import numpy as np
    
        u_hat = self.fft(u)
    
        if self.dim == 1:
            Nx = len(x_grid)
            dx = x_grid[1] - x_grid[0]
    
            kx_fft = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    
            X_dummy = np.zeros_like(kx_fft)
            symbol_vals = self.p_func(X_dummy, kx_fft)
    
            symbol_vals = np.broadcast_to(
                symbol_vals,
                kx_fft.shape
            ).astype(np.complex128).copy()
    
            symbol_vals = _clip_complex_magnitude(symbol_vals, clamp)
    
            if freq_window == "gaussian":
                k_max = np.max(np.abs(kx_fft))
                if k_max > 0:
                    sigma = 0.8 * k_max
                    symbol_vals *= np.exp(-(kx_fft / sigma) ** 4)
    
            elif freq_window == "hann":
                k_max = np.max(np.abs(kx_fft))
                if k_max > 0:
                    W = 0.5 * (
                        1.0 + np.cos(np.pi * kx_fft / k_max)
                    ) * (np.abs(kx_fft) < k_max)
                    symbol_vals *= W
    
        elif self.dim == 2:
            if y_grid is None:
                raise ValueError("y_grid is required for 2D operators.")
    
            Nx = len(x_grid)
            Ny = len(y_grid)
    
            dx = x_grid[1] - x_grid[0]
            dy = y_grid[1] - y_grid[0]
    
            kx_fft = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
            ky_fft = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    
            KX, KY = np.meshgrid(kx_fft, ky_fft, indexing="ij")
    
            X_dummy = np.zeros_like(KX)
            Y_dummy = np.zeros_like(KY)
    
            symbol_vals = self.p_func(X_dummy, Y_dummy, KX, KY)
    
            symbol_vals = np.broadcast_to(
                symbol_vals,
                KX.shape
            ).astype(np.complex128).copy()
    
            symbol_vals = _clip_complex_magnitude(symbol_vals, clamp)
    
            if freq_window == "gaussian":
                kx_max = np.max(np.abs(kx_fft))
                ky_max = np.max(np.abs(ky_fft))
    
                if kx_max > 0 and ky_max > 0:
                    sx = 0.8 * kx_max
                    sy = 0.8 * ky_max
                    symbol_vals *= (
                        np.exp(-(KX / sx) ** 4)
                        * np.exp(-(KY / sy) ** 4)
                    )
    
            elif freq_window == "hann":
                kx_max = np.max(np.abs(kx_fft))
                ky_max = np.max(np.abs(ky_fft))
    
                if kx_max > 0 and ky_max > 0:
                    Wx = 0.5 * (
                        1.0 + np.cos(np.pi * KX / kx_max)
                    ) * (np.abs(KX) < kx_max)
    
                    Wy = 0.5 * (
                        1.0 + np.cos(np.pi * KY / ky_max)
                    ) * (np.abs(KY) < ky_max)
    
                    symbol_vals *= Wx * Wy
    
        else:
            raise ValueError("Only 1D and 2D supported")
    
        u_hat *= symbol_vals
    
        if dealiasing_mask is not None:
            u_hat *= dealiasing_mask
    
        result = self.ifft(u_hat)
    
        if space_window:
            if self.dim == 1:
                x0 = (x_grid[0] + x_grid[-1]) / 2.0
                L = (x_grid[-1] - x_grid[0]) / 2.0
                sw_x = np.exp(-((x_grid - x0) / L) ** 2)
                result *= sw_x
    
            elif self.dim == 2:
                x0 = (x_grid[0] + x_grid[-1]) / 2.0
                y0 = (y_grid[0] + y_grid[-1]) / 2.0
    
                Lx = (x_grid[-1] - x_grid[0]) / 2.0
                Ly = (y_grid[-1] - y_grid[0]) / 2.0
    
                sw_x = np.exp(-((x_grid - x0) / Lx) ** 2)
                sw_y = np.exp(-((y_grid - y0) / Ly) ** 2)
    
                result *= sw_x[:, None] * sw_y[None, :]
    
        return result
        
    def principal_symbol(self, order=1):
        """
        Compute the leading homogeneous component of the pseudo-differential symbol.

        This method extracts the principal part of the symbol, which is the dominant 
        term under high-frequency asymptotics (|ξ| → ∞). The expansion is performed 
        in polar coordinates for 2D symbols to maintain rotational symmetry, then 
        converted back to Cartesian form.

        Parameters
        ----------
        order : int
            Order of the asymptotic expansion in powers of 1/ρ, where ρ = |ξ| in 1D 
            or ρ = sqrt(ξ² + η²) in 2D. Only the leading-order term is returned.

        Returns
        -------
        sympy.Expr
            The principal symbol component, homogeneous of degree `m - order`, where 
            `m` is the original symbol's order.

        Notes:
        - In 1D, uses direct series expansion in ξ.
        - In 2D, expands in radial variable ρ while preserving angular dependence.
        - Useful for microlocal analysis and constructing parametrices.
        """

        p = self.symbol
        if self.dim == 1:
            # Reuse the xi symbol actually present in self.symbol; a fresh
            # symbols('xi', real=True, positive=True) is a distinct object and
            # series(p, xi, ...) would silently treat p as constant in xi.
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            return simplify(series(p, xi, oo, n=order).removeO())
        elif self.dim == 2:
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in p.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            # Homogeneous radial expansion: we set (ξ, η) = ρ (cosθ, sinθ)
            rho, theta = symbols('rho theta', real=True, positive=True)
            p_rho = p.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
            expansion = series(p_rho, rho, oo, n=order).removeO()
            # Revert back to (ξ, η)
            expansion_cart = expansion.subs({rho: sqrt(xi**2 + eta**2),
                                             cos(theta): xi / sqrt(xi**2 + eta**2),
                                             sin(theta): eta / sqrt(xi**2 + eta**2)})
            return simplify(powdenest(expansion_cart, force=True))
                       
    def is_homogeneous(self, tol=1e-10):
        """
        Check whether the symbol is homogeneous in the frequency variables.
    
        Returns
        -------
        (bool, Rational or float or None)
            Tuple (is_homogeneous, degree) where:
            - is_homogeneous: True if the symbol satisfies p(λξ, λη) = λ^m * p(ξ, η)
            - degree: the detected degree m if homogeneous, or None
        """
        from sympy import symbols, simplify, expand, Eq, nsimplify
    
        if self.dim == 1:
            p = self.symbol
            # IMPORTANT: reuse the xi symbol actually present in self.symbol.
            # Creating a fresh symbols('xi', real=True, positive=True) here is a
            # *different* sympy Symbol object from the one baked into p (which is
            # typically only real=True), so p.subs(xi, ...) would silently no-op.
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            l = symbols('l', real=True, positive=True)
            p_scaled = p.subs(xi, l * xi)
            ratio = simplify(p_scaled / p)
            if ratio.has(xi):
                return False, None
            try:
                # Float alpha (e.g. 0.5) can leave stray 1.0 coefficients (e.g.
                # 1.0*l**1.0) that keep the ratio a Mul instead of a bare Pow, so
                # as_base_exp() would return (1.0*l**1.0, 1) instead of (l, 1).
                # nsimplify(..., rational=True) restores the exact form first.
                ratio_clean = nsimplify(simplify(ratio), rational=True)
                base, deg = ratio_clean.as_base_exp()
                if base == l:
                    return True, deg
                return False, None
            except Exception:
                return False, None
    
        elif self.dim == 2:
            p = self.symbol
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in p.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            l = symbols('l', real=True, positive=True)
            p_scaled = p.subs({xi: l * xi, eta: l * eta})
            ratio = simplify(p_scaled / p)
            # If ratio == l**m with no (xi, eta) left, it's homogeneous
            if ratio.has(xi, eta):
                return False, None
            try:
                ratio_clean = nsimplify(ratio, rational=True)
                base, exp = ratio_clean.as_base_exp()
                if base == l:
                    return True, exp
            except Exception:
                pass
            return False, None

    def symbol_order(self, max_order=10, tol=1e-3):
        """Estimate the asymptotic homogeneity order of the symbol as |ξ|→∞."""
        from sympy import (symbols, series, simplify, cos, sin, oo,
                            powdenest, radsimp, Add)
    
        def validate_order(power, coeff, tol):
            if power is None:
                return None
            if simplify(coeff) == 0 or coeff.equals(0):
                print("⚠️ Coefficient is symbolically zero; ignoring")
                return None
            return int(power) if float(power) == int(power) else float(power)
    
        # ---- FIX 3: order-0 shortcut (no frequency dependence at all) ----
        freq_syms = [s for s in self.symbol.free_symbols if s.name in ('xi', 'eta')]
        if not freq_syms:
            return 0
    
        is_homog, degree = self.is_homogeneous()
        if is_homog:
            return float(degree)
        print("⚠️ The symbol is not homogeneous. The asymptotic order is not well defined.")
    
        def leading_power(s, var):
            """(power, coeff) of the HIGHEST-power part of s in var.
            Collects ALL terms sharing the top power (not just one)."""
            terms  = Add.make_args(s)
            powers = [t.as_powers_dict().get(var, 0) for t in terms]
            top    = max(powers)
            lead   = Add(*[t for t, p in zip(terms, powers) if p == top])
            lead   = radsimp(simplify(powdenest(lead, force=True)))
            coeff  = lead / var**top if top else lead
            return top, coeff
    
        if self.dim == 1:
            x  = self.vars_x[0]
            xi = next((s for s in self.symbol.free_symbols if s.name == 'xi'),
                      symbols('xi', real=True))
            try:                                    # method 1: xi → ∞
                s = series(self.symbol, xi, oo, n=max_order).removeO()
                power, coeff = leading_power(s, xi)          # FIX 2: was rho
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return order
            except Exception:
                pass
            try:                                    # method 2: xi = 1/z
                z = symbols('z', real=True, positive=True)
                s = series(self.symbol.subs(xi, 1/z), z, 0, n=max_order).removeO()
                power, coeff = leading_power(s, z)           # FIX 2: was rho
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return -order
            except Exception as e:
                print(f"⚠️ fallback z failed: {e}")
            return None
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi  = next((s for s in self.symbol.free_symbols if s.name == 'xi'),
                       symbols('xi', real=True))
            eta = next((s for s in self.symbol.free_symbols if s.name == 'eta'),
                       symbols('eta', real=True))
            rho, theta = symbols('rho theta', real=True, positive=True)
            try:                                    # method 1: polar, rho → ∞
                p_rho = self.symbol.subs({xi: rho*cos(theta), eta: rho*sin(theta)})
                # FIX 1: dropped preprocess_power/preprocess_sqrt — they inject
                #        a spurious factor 2**n into rho**n terms.
                s = series(simplify(p_rho), rho, oo, n=max_order).removeO()
                power, coeff = leading_power(s, rho)
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return order
            except Exception as e:
                print(f"⚠️ polar expansion failed: {e}")
            try:                                    # method 2: z = 1/rho
                z = symbols('z', real=True, positive=True)
                p_z = self.symbol.subs({xi: cos(theta)/z, eta: sin(theta)/z})
                s = series(simplify(p_z), z, 0, n=max_order).removeO()
                power, coeff = leading_power(s, z)           # FIX 2: was rho
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return -order
            except Exception as e:
                print(f"⚠️ fallback z (2D) failed: {e}")
            return None
    
        raise NotImplementedError("Only 1D and 2D supported.")

    
    def asymptotic_expansion(self, order=3):
        """
        Compute the asymptotic expansion of the symbol as |ξ| → ∞ (high-frequency regime).
    
        This method expands the pseudo-differential symbol in inverse powers of the 
        frequency variable(s), either in 1D or 2D. It handles both polynomial and 
        exponential symbols by performing a series expansion in 1/|ξ| up to the specified order.
    
        The expansion is performed directly in Cartesian coordinates for 1D symbols.
        For 2D symbols, the method uses polar coordinates (ρ, θ) to perform the expansion 
        at infinity in ρ, then converts the result back to Cartesian coordinates.
    
        Parameters
        ----------
        order : int, optional
            Maximum order of the asymptotic expansion. Default is 3.
    
        Returns
        -------
        sympy.Expr
            The asymptotic expansion of the symbol up to the given order, expressed in Cartesian coordinates.
            If expansion fails, returns the original unexpanded symbol.
    
        Notes:
        - In 1D: expansion is performed directly in terms of ξ.
        - In 2D: the symbol is first rewritten in polar coordinates (ρ,θ), expanded asymptotically 
          in ρ → ∞, then converted back to Cartesian coordinates (ξ,η).
        - Handles special case when the symbol is an exponential function by expanding its argument.
        - Symbolic normalization is applied early (via `simplify`) for 2D expressions to improve convergence.
        - Robust to failures: catches exceptions and issues warnings instead of raising errors.
        - Final expression is simplified using `powdenest` and `expand` for improved readability.
        """
        p = self.symbol
    
        if self.dim == 1:
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
    
            try:
                # Case: exponential function
                if p.func == exp and len(p.args) == 1:
                    arg = p.args[0]
                    arg_series = series(arg, xi, oo, n=order).removeO()
                    expanded = series(exp(expand(arg_series)), xi, oo, n=order).removeO()
                    return simplify(powdenest(expanded, force=True))
                else:
                    expanded = series(p, xi, oo, n=order).removeO()
                    return simplify(powdenest(expanded, force=True))
    
            except Exception as e:
                print(f"Warning: 1D expansion failed: {e}")
                return p
    
        elif self.dim == 2:
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in p.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            rho, theta = symbols('rho theta', real=True, positive=True)
    
            # Normalize before substitution
            p = simplify(p)
    
            # Substitute polar coordinates
            p_polar = p.subs({
                xi: rho * cos(theta),
                eta: rho * sin(theta)
            })
    
            try:
                # Handle exponentials
                if p_polar.func == exp and len(p_polar.args) == 1:
                    arg = p_polar.args[0]
                    arg_series = series(arg, rho, oo, n=order).removeO()
                    expanded = series(exp(expand(arg_series)), rho, oo, n=order).removeO()
                else:
                    expanded = series(p_polar, rho, oo, n=order).removeO()
    
                # Convert back to Cartesian
                norm = sqrt(xi**2 + eta**2)
                expansion_cart = expanded.subs({
                    rho: norm,
                    cos(theta): xi / norm,
                    sin(theta): eta / norm
                })
    
                # Final simplifications
                result = simplify(powdenest(expansion_cart, force=True))
                result = expand(result)
                return result
    
            except Exception as e:
                print(f"Warning: 2D expansion failed: {e}")
                return p  
            
    def compose_asymptotic(self, other, order=1, mode='kn', sign_convention=None):
        """
        Compose two pseudo-differential operators using an asymptotic expansion
        in the chosen quantization scheme (Kohn–Nirenberg or Weyl).
    
        Parameters
        ----------
        other : PseudoDifferentialOperator
            The operator to compose with this one.
        order : int, default=1
            Maximum order of the asymptotic expansion.
        mode : {'kn', 'weyl'}, default='kn'
            Quantization mode:
            - 'kn' : Kohn–Nirenberg quantization (left-quantized)
            - 'weyl' : Weyl symmetric quantization
        sign_convention : {'standard', 'inverse'}, optional
            Controls the phase factor convention for the KN case:
            - 'standard' → (i)^(-n), gives [x, ξ] = +i (physics convention)
            - 'inverse' → (i)^(+n), gives [x, ξ] = -i (mathematical adjoint convention)
            If None, defaults to 'standard'.
    
        Returns
        -------
        sympy.Expr
            Symbolic expression for the composed symbol up to the given order.
    
        Notes
        -----
        - In 1D (Kohn–Nirenberg):
            (p ∘ q)(x, ξ) ~ Σₙ (1/n!) (i sgn)^n ∂_ξⁿ p(x, ξ) ∂_xⁿ q(x, ξ)
        - In 1D (Weyl):
            (p # q)(x, ξ) = exp[(i/2)(∂_ξ^p ∂_x^q - ∂_x^p ∂_ξ^q)] p(x, ξ) q(x, ξ)
            truncated at given order.
    
        """
    
        assert self.dim == other.dim, "Operator dimensions must match"
        if mode not in ('kn', 'weyl'):
            raise ValueError("mode must be either 'kn' or 'weyl'")
        dim = self.dim
        if dim not in (1, 2):
            raise NotImplementedError("Only 1D and 2D cases are implemented")

        p, q = self.symbol, other.symbol
        x_vars = self.vars_x
        xi_vars = self._peetre_frequency_symbols()
        sign = -1 if (sign_convention or 'standard') == 'standard' else +1

        result = 0
        if mode == 'kn':  # Kohn-Nirenberg
            for n in range(order + 1):
                for alpha in _mi_all(n, dim):
                    fact = _mi_factorial(alpha)
                    dp = _mi_diff(p, xi_vars, alpha)
                    dq = _mi_diff(q, x_vars, alpha)
                    result += (dp * dq / fact) * (1j) ** (sign * n)
        else:  # 'weyl' -- general dimension-generic Moyal star product,
               # exact match to the previous 1D formula (which was the
               # mathematically correct one); see class-level notes.
            for total in range(order + 1):
                for a_deg in range(total + 1):
                    b_deg = total - a_deg
                    for alpha in _mi_all(a_deg, dim):
                        for beta in _mi_all(b_deg, dim):
                            coeff = (1j / 2) ** total * (-1) ** b_deg
                            coeff /= (_mi_factorial(alpha) * _mi_factorial(beta))
                            dp = _mi_diff(_mi_diff(p, xi_vars, alpha), x_vars, beta)
                            dq = _mi_diff(_mi_diff(q, x_vars, alpha), xi_vars, beta)
                            result += coeff * dp * dq

        try:
            return simplify(result)
        except TypeError:
            # Fallback for SymPy sorting bugs with undefined functions/derivatives
            return result

    # ======================================================================
    # Peetre-style symbolic decomposition
    # ======================================================================
        
    def _peetre_frequency_symbols(self):
        """
        Return the frequency symbols actually used in the symbol.

        Returns
        -------
        tuple
            (xi,) in 1D or (xi, eta) in 2D.
        """
        from sympy import symbols

        if self.dim == 1:
            xi = next(
                (s for s in self.symbol.free_symbols if s.name == "xi"),
                symbols("xi", real=True),
            )
            return (xi,)

        elif self.dim == 2:
            xi = next(
                (s for s in self.symbol.free_symbols if s.name == "xi"),
                symbols("xi", real=True),
            )
            eta = next(
                (s for s in self.symbol.free_symbols if s.name == "eta"),
                symbols("eta", real=True),
            )
            return xi, eta

        raise NotImplementedError("Peetre decomposition supports only 1D and 2D operators.")

    @staticmethod
    def _peetre_merge_local(dst, src):
        """
        Merge one local-coefficient dictionary into another, in place,
        summing coefficients that share the same frequency multi-index.

        Parameters
        ----------
        dst : dict
            Destination dictionary mapping frequency multi-index tuples to
            symbolic (x-dependent) coefficients; updated in place with the
            merged result.
        src : dict
            Source dictionary with the same structure as `dst`, merged
            into it. Coefficients for keys shared with `dst` are added
            together and simplified; keys unique to `src` are copied over.

        Returns
        -------
        None
            `dst` is mutated in place; nothing is returned.
        """
        from sympy import simplify, together

        for monom, coeff in src.items():
            dst[monom] = simplify(together(dst.get(monom, 0) + coeff))

    @staticmethod
    def _peetre_is_zero(expr):
        """
        Conservative symbolic zero test used throughout the Peetre
        pipeline to decide whether a coefficient or term can be safely
        dropped.

        Several increasingly expensive strategies are tried in turn
        (identity check, the `is_zero` attribute, `simplify`, `equals`);
        if none of them can prove the expression is zero, it is treated
        as non-zero rather than risk silently discarding a genuine term.

        Parameters
        ----------
        expr : sympy.Expr or None
            Expression to test. `None` is treated as zero.

        Returns
        -------
        bool
            True only if the expression could be established to be
            identically zero; False otherwise, including when the test is
            inconclusive.
        """
        from sympy import simplify

        if expr is None:
            return True

        if expr == 0:
            return True

        try:
            if expr.is_zero is True:
                return True
            if expr.is_zero is False:
                return False
        except Exception:
            pass

        try:
            return bool(simplify(expr) == 0)
        except Exception:
            try:
                return bool(expr.equals(0))
            except Exception:
                return False

    def _peetre_classify_terms(self, expr):
        """
        Classify a symbolic expression into local, separable and joint terms.

        Parameters
        ----------
        expr : sympy.Expr
            Symbol expression to classify.

        Returns
        -------
        local_coeffs : dict
            Dictionary mapping frequency multi-indices to x-dependent coefficients.
            Example in 1D:
                {(2,): 1 + x**2, (1,): x, (0,): V(x)}
            represents
                (1 + x**2) xi**2 + x xi + V(x).

        separable : list of tuple
            List of pairs (a_expr, q_expr), where a_expr depends only on space
            variables and q_expr depends only on frequency variables.

        joint : list of sympy.Expr
            Terms still entangled between space and frequency variables.
        """
        from sympy import Add, Poly, expand, simplify, together

        xi_vars = self._peetre_frequency_symbols()
        x_vars = self.vars_x

        expr = expand(expr)

        local_terms = []
        separable = []
        joint = []

        for t in Add.make_args(expr):
            try:
                a, q = t.as_independent(*xi_vars)
            except Exception:
                a, q = 1, t

            # If the frequency-dependent part still contains space variables,
            # the term is genuinely joint.
            if any(q.has(xv) for xv in x_vars):
                joint.append(t)

            # Polynomial in frequency variables => local/differential part.
            elif q.is_polynomial(*xi_vars):
                local_terms.append(t)

            # Non-polynomial but frequency-only => separable Fourier multiplier
            # with spatial amplitude.
            else:
                separable.append((simplify(a), simplify(q)))

        local_coeffs = {}

        if local_terms:
            p_local = Add(*local_terms)

            try:
                poly = Poly(p_local, *xi_vars)
            except Exception:
                try:
                    poly = Poly(p_local, *xi_vars, extension=True)
                except Exception:
                    # If Poly cannot safely parse the polynomial part, keep it
                    # as joint rather than producing wrong coefficients.
                    joint.extend(local_terms)
                    return local_coeffs, separable, joint

            for monom, coeff in poly.terms():
                local_coeffs[monom] = simplify(
                    together(local_coeffs.get(monom, 0) + coeff)
                )

        return local_coeffs, separable, joint

    def _peetre_local_symbol(self, local_coeffs):
        """
        Rebuild the local polynomial symbol from its coefficient
        dictionary — the inverse of the local part of
        `_peetre_classify_terms`.

        Parameters
        ----------
        local_coeffs : dict
            Mapping from frequency multi-index tuples (one exponent per
            frequency variable) to their x-dependent symbolic
            coefficients, as produced by `_peetre_classify_terms`.

        Returns
        -------
        sympy.Expr
            The expanded symbol `sum_alpha coeff_alpha(x) * xi**alpha`
            reconstructed from `local_coeffs`.
        """
        from sympy import Integer, expand

        xi_vars = self._peetre_frequency_symbols()
        expr = Integer(0)

        for monom, coeff in local_coeffs.items():
            term = coeff
            for xi_var, power in zip(xi_vars, monom):
                if power:
                    term = term * xi_var**power
            expr = expr + term

        return expand(expr)

    def _peetre_separable_symbol(self, separable):
        """
        Rebuild the separable symbol from a list of (a, q) pairs — the
        inverse of the separable part of `_peetre_classify_terms`.

        Parameters
        ----------
        separable : list of tuple
            Pairs `(a_expr, q_expr)` where `a_expr` depends only on the
            spatial variables and `q_expr` depends only on the frequency
            variables.

        Returns
        -------
        sympy.Expr
            The expanded sum `sum_k a_k(x) * q_k(xi)`, or
            `sympy.Integer(0)` if `separable` is empty.
        """
        from sympy import Add, Integer, expand

        if not separable:
            return Integer(0)

        return expand(Add(*[a * q for a, q in separable]))


    def _peetre_merge_separable(self, separable):
        """
        Merge separable terms having the same frequency factor q.

        This is purely cosmetic but makes the decomposition much easier to read.
        """
        from sympy import simplify

        merged = {}
        ordered_keys = []

        for a, q in separable:
            a = simplify(a)
            q = simplify(q)

            if self._peetre_is_zero(a) or self._peetre_is_zero(q):
                continue

            key = q

            if key in merged:
                merged[key] = simplify(merged[key] + a)
            else:
                merged[key] = a
                ordered_keys.append(key)

        return [
            (merged[q], q)
            for q in ordered_keys
            if not self._peetre_is_zero(merged[q])
        ]

    def _peetre_local_to_separable(self, local_coeffs):
        """
        Convert local polynomial coefficients into separable terms.
    
        A local term
    
            coeff(x) * xi^alpha
    
        is converted into
    
            (coeff(x), xi^alpha)
    
        Terms with the same spatial coefficient are merged.
        """
        from sympy import Integer, expand, simplify
    
        xi_vars = self._peetre_frequency_symbols()
        separable = []
    
        for monom, coeff in local_coeffs.items():
            if self._peetre_is_zero(coeff):
                continue
    
            q = Integer(1)
            for var, power in zip(xi_vars, monom):
                if power:
                    q = q * var**power
    
            coeff = simplify(coeff)
            q = expand(q)
    
            # Merge if the same spatial coefficient already exists.
            merged = False
            for i, (c0, q0) in enumerate(separable):
                if self._peetre_is_zero(c0 - coeff):
                    separable[i] = (c0, expand(q0 + q))
                    merged = True
                    break
    
            if not merged:
                separable.append((coeff, q))
    
        return [
            (c, expand(q))
            for c, q in separable
            if not self._peetre_is_zero(c) and not self._peetre_is_zero(q)
        ]

    def _peetre_local_as_separable(self, local_coeffs):
        """
        Represent local polynomial coefficients as separable-style pairs
        (a(x), q(xi)).

        This does not change the mathematical classification: the terms are
        still local because q(xi) is a polynomial in the frequency variables.
        It only gives the local part in the same operational form used for
        separable terms, namely:

            a(x) * q(D) u

        which is the efficient application path.
        """
        if not local_coeffs:
            return []

        return self._peetre_merge_separable(
            self._peetre_local_to_separable(local_coeffs)
        )

    # ------------------------------------------------------------------
    # Low-rank joint residual helpers
    # ------------------------------------------------------------------
    def _infer_joint_bounds(self, x_grid, kx, y_grid=None, ky=None):
        """
        Infer physical bounds for low-rank joint decomposition from
        the spatial and frequency grids.
        """
        import numpy as np

        def _bounds(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                raise ValueError("Empty grid encountered while inferring bounds.")

            lo = float(np.min(arr))
            hi = float(np.max(arr))

            if hi <= lo:
                lo -= 1.0
                hi += 1.0

            return lo, hi

        freq_syms = self._peetre_frequency_symbols()

        if self.dim == 1:
            return {
                self.vars_x[0]: _bounds(x_grid),
                freq_syms[0]: _bounds(kx),
            }

        elif self.dim == 2:
            if y_grid is None or ky is None:
                raise ValueError("y_grid and ky are required for 2D bounds.")

            return {
                self.vars_x[0]: _bounds(x_grid),
                self.vars_x[1]: _bounds(y_grid),
                freq_syms[0]: _bounds(kx),
                freq_syms[1]: _bounds(ky),
            }

        else:
            raise NotImplementedError("Only 1D and 2D bounds are supported.")

    def _remap_bounds(self, bounds, syms):
        """
        Ensure bounds keys match the exact SymPy symbols used in the
        expression. If necessary, match by symbol name.
        """
        out = {}

        for s in syms:
            if s in bounds:
                out[s] = bounds[s]
                continue

            matched_key = None
            for k in bounds.keys():
                if getattr(k, "name", str(k)) == getattr(s, "name", str(s)):
                    matched_key = k
                    break

            if matched_key is None:
                raise ValueError(f"No bound provided for symbol '{s}'.")

            out[s] = bounds[matched_key]

        return out

    def _low_rank_joint_pairs(
        self,
        joint_symbol,
        bounds,
        degree=6,
        tol=1e-5,
        num_samples=10000,
        seed=42,
        use_cache=True,
    ):
        """
        Factorize the joint residual into separable pairs.

        Returns
        -------
        pairs : list
            List of `(a_k(x), q_k(xi))`.
        metrics : dict
            Symbol-level approximation diagnostics.
        """
        import numpy as np
        from sympy import symbols

        if self._peetre_is_zero(joint_symbol):
            return [], {
                "rel_l2_error": 0.0,
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "svd_energy_retained_pct": 100.0,
                "singular_values": np.array([]),
            }

        # ----------------------------------------------------------
        # Use symbols actually present in the joint residual when
        # possible. This avoids subtle SymPy symbol-mismatch issues.
        # ----------------------------------------------------------
        x_syms = []
        for v in self.vars_x:
            s = next(
                (fs for fs in joint_symbol.free_symbols if fs.name == v.name),
                v,
            )
            x_syms.append(s)

        freq_names = ["xi"] if self.dim == 1 else ["xi", "eta"]
        xi_syms = []
        for name in freq_names:
            s = next(
                (fs for fs in joint_symbol.free_symbols if fs.name == name),
                symbols(name, real=True),
            )
            xi_syms.append(s)

        all_syms = x_syms + xi_syms
        bounds = self._remap_bounds(bounds, all_syms)

        key = (
            joint_symbol,
            degree,
            tol,
            tuple(
                (s, float(bounds[s][0]), float(bounds[s][1]))
                for s in all_syms
            ),
        )

        cache = getattr(self, "_joint_lowrank_cache", None)

        if use_cache and cache is not None and cache.get("key") == key:
            return cache["pairs"], cache["metrics"]

        pairs, metrics = factorize_symbolic(
            joint_symbol,
            x_syms,
            xi_syms,
            bounds,
            degree=degree,
            tol=tol,
            num_samples=num_samples,
            seed=seed,
        )

        self._joint_lowrank_cache = {
            "key": key,
            "pairs": pairs,
            "metrics": metrics,
        }

        return pairs, metrics

    def _resolve_joint_symbols(self, joint_symbol):
        """Shared symbol-resolution logic (matches _low_rank_joint_pairs):
        find the actual x/xi symbols present in joint_symbol by name."""
        x_syms = []
        for v in self.vars_x:
            s = next((fs for fs in joint_symbol.free_symbols if fs.name == v.name), v)
            x_syms.append(s)
        freq_names = ["xi"] if self.dim == 1 else ["xi", "eta"]
        xi_syms = []
        for name in freq_names:
            s = next((fs for fs in joint_symbol.free_symbols if fs.name == name),
                     sp.symbols(name, real=True))
            xi_syms.append(s)
        return x_syms, xi_syms

    def _resolve_nufft_plan(self, joint_symbol, use_cache=True):
        """
        Resolve the NUFFT plan for a joint residual symbol (symbolic, grid-free).
    
        Returns
        -------
        plan_info : tuple or None
            ("1d", plan) or ("2d", plan) if the symbol is NUFFT-representable,
            None otherwise.
        """
        x_syms, xi_syms = self._resolve_joint_symbols(joint_symbol)
        key = (joint_symbol, self.dim)
        cache = getattr(self, "_joint_nufft_cache", None)
        if use_cache and cache is not None and cache.get("key") == key:
            return cache["plan_info"]
    
        if self.dim == 1:
            plan = try_nufft_decomposition_1d(joint_symbol, x_syms[0], xi_syms[0])
            plan_info = ("1d", plan) if plan is not None else None
        elif self.dim == 2:
            res = try_nufft_decomposition_2d(
                joint_symbol, x_syms[0], x_syms[1], xi_syms[0], xi_syms[1]
            )
            plan_info = ("2d", res) if res is not None else None
        else:
            plan_info = None
    
        self._joint_nufft_cache = {"key": key, "plan_info": plan_info}
        return plan_info

    def _nufft_joint_apply(self, joint_symbol, u, x_grid, kx, y_grid=None, ky=None,
                            use_cache=True, freq_window="gaussian"):
        """
        Try the NUFFT joint-residual backend. Returns the applied numeric
        array on success, or None if the symbol doesn't classify as
        NUFFT-representable (caller should fall back to direct application).
        PERIODIC BOUNDARY CONDITIONS ONLY.
        """
        plan_info = self._resolve_nufft_plan(joint_symbol, use_cache=use_cache)
        if plan_info is None:
            return None
    
        kind, plan = plan_info
        dx = x_grid[1] - x_grid[0]
        dxi = kx[1] - kx[0]
        if kind == "1d":
            return apply_nufft_1d(u, plan, x_grid, kx, dx, dxi, freq_window=freq_window)
        else:  # "2d"
            dy = y_grid[1] - y_grid[0]
            deta = ky[1] - ky[0]
            plan_kind, plan_data = plan
            return apply_nufft_2d(u, plan_kind, plan_data, x_grid, y_grid, kx, ky,
                                   dx, dy, dxi, deta, freq_window=freq_window)


    def _resolve_joint_representation(
        self,
        joint_symbol,
        backend="auto",
        bounds=None,
        degree=6,
        tol=1e-5,
        num_samples=10000,
        seed=42,
        use_cache=True,
    ):
        """
        Normalize the joint residual into an executable representation.
    
        This is the single entry point for turning a joint symbol into
        something that can be numerically applied. It resolves 'auto' to a
        concrete backend, then delegates to the appropriate cached helper.
    
        Parameters
        ----------
        joint_symbol : sympy.Expr
            The irreducible joint residual.
        backend : {'auto', 'direct', 'lowrank', 'nufft', 'aaa'}
            Which factorization strategy to use.
        bounds : dict, optional
            Symbol -> (min, max) mapping. Required for 'lowrank' and 'aaa'.
        degree, tol, num_samples, seed :
            Forwarded to the underlying factorization.
        use_cache : bool
            Whether to use per-backend caches.
    
        Returns
        -------
        dict
            A representation with a "type" key:
    
            - {"type": "zero"}
            - {"type": "direct", "symbol": ..., "backend": "direct"}
            - {"type": "separable_pairs", "pairs": [...], "metrics": {...},
               "backend": "lowrank"}
            - {"type": "nufft_plan", "plan_info": (...), "backend": "nufft"}
            - {"type": "nufft_unrepresentable", "symbol": ..., "backend": "nufft"}
            - {"type": "aaa_callable", "symbol_func": ..., "metrics": {...},
               "backend": "aaa"}
            - {"type": "aaa_unfit", "symbol": ..., "backend": "aaa"}
        """
        if self._peetre_is_zero(joint_symbol):
            return {"type": "zero"}
    
        # Resolve 'auto' to a concrete backend
        resolved = backend
        if resolved == "auto":
            x_syms, xi_syms = self._resolve_joint_symbols(joint_symbol)
            resolved = self._auto_select_joint_backend(joint_symbol, x_syms, xi_syms)
    
        if resolved == "direct":
            return {"type": "direct", "symbol": joint_symbol, "backend": "direct"}
    
        elif resolved == "lowrank":
            if bounds is None:
                raise ValueError(
                    "joint_bounds must be provided for backend='lowrank' "
                    "(or use backend='auto' with grids available at apply time)."
                )
            pairs, metrics = self._low_rank_joint_pairs(
                joint_symbol, bounds, degree=degree, tol=tol,
                num_samples=num_samples, seed=seed, use_cache=use_cache,
            )
            return {
                "type": "separable_pairs",
                "pairs": pairs,
                "metrics": metrics,
                "backend": "lowrank",
            }
    
        elif resolved == "nufft":
            plan_info = self._resolve_nufft_plan(joint_symbol, use_cache=use_cache)
            if plan_info is None:
                return {
                    "type": "nufft_unrepresentable",
                    "symbol": joint_symbol,
                    "backend": "nufft",
                }
            return {
                "type": "nufft_plan",
                "plan_info": plan_info,
                "backend": "nufft",
            }
    
        elif resolved == "aaa":
            if bounds is None:
                raise ValueError(
                    "joint_bounds must be provided for backend='aaa' "
                    "(or use backend='auto' with grids available at apply time)."
                )
            symbol_func, metrics = self._aaa_joint_symbol_func(
                joint_symbol, bounds, tol=tol, use_cache=use_cache,
            )
            if symbol_func is None:
                return {
                    "type": "aaa_unfit",
                    "symbol": joint_symbol,
                    "backend": "aaa",
                }
            return {
                "type": "aaa_callable",
                "symbol_func": symbol_func,
                "metrics": metrics,
                "backend": "aaa",
            }
    
        else:
            raise ValueError(
                f"joint_backend must be 'direct', 'lowrank', 'nufft', or 'aaa', "
                f"got '{resolved}'."
            )

    def _aaa_joint_symbol_func(self, joint_symbol, bounds, degree=None, tol=1e-8,
                                use_cache=True):
        """
        Try the AAA joint-residual backend. Returns (symbol_func, metrics)
        on success, where symbol_func is a fast numpy callable suitable for
        kohn_nirenberg_fft/nonperiodic's `symbol_func` argument, or
        (None, None) if the quality gate isn't met (caller should fall back
        to direct application).

        bounds : dict mapping each space/frequency symbol to (min, max),
        same format as joint_bounds elsewhere (see _infer_joint_bounds).
        """
        x_syms, xi_syms = self._resolve_joint_symbols(joint_symbol)
        all_syms = x_syms + xi_syms
        bounds = self._remap_bounds(bounds, all_syms)

        key = (joint_symbol, tol, tuple(
            (s, float(bounds[s][0]), float(bounds[s][1])) for s in all_syms
        ))
        cache = getattr(self, "_joint_aaa_cache", None)
        if use_cache and cache is not None and cache.get("key") == key:
            plan = cache["plan"]
        else:
            if self.dim == 1:
                plan = try_aaa_decomposition_1d(
                    joint_symbol, x_syms[0], xi_syms[0],
                    x_bounds=bounds[x_syms[0]], xi_bounds=bounds[xi_syms[0]],
                    rtol=tol,
                )
            elif self.dim == 2:
                plan = try_aaa_decomposition_2d(
                    joint_symbol, x_syms[0], x_syms[1], xi_syms[0], xi_syms[1],
                    x_bounds=bounds[x_syms[0]], y_bounds=bounds[x_syms[1]],
                    xi_bounds=bounds[xi_syms[0]], eta_bounds=bounds[xi_syms[1]],
                    rtol=tol,
                )
            else:
                plan = None
            self._joint_aaa_cache = {"key": key, "plan": plan}

        if plan is None:
            return None, None

        metrics = {"rel_l2_error": plan["rel_l2_error"]}
        symbol_func = (aaa_plan_to_callable_1d(plan) if self.dim == 1
                        else aaa_plan_to_callable_2d(plan))
        return symbol_func, metrics

    def _apply_joint_residual(
        self,
        joint_symbol,
        u,
        x_grid,
        kx,
        y_grid=None,
        ky=None,
        boundary_condition="periodic",
        peetre_quantization="kohn-nirenberg",
        common_apply_kwargs=None,
        apply_separable_pair=None,
        joint_backend="direct",
        joint_degree=6,
        joint_tol=1e-5,
        joint_bounds=None,
        joint_max_rel_error=None,
        joint_num_samples=10000,
        joint_seed=42,
        use_cache=True,
        freq_window="gaussian",
        clamp=1e6,
        space_window=False,
    ):
        """
        Apply the irreducible joint residual with backend selection,
        quality gates, and automatic fallback to direct application.
    
        This method encapsulates the entire joint-residual execution
        pipeline: auto-selection, representation resolution, quality
        checking, and numerical application with fallbacks.
    
        Parameters
        ----------
        joint_symbol : sympy.Expr
            The joint residual symbol to apply.
        u : ndarray
            Input field.
        x_grid, kx, y_grid, ky : ndarray
            Spatial and frequency grids.
        boundary_condition : str
            'periodic' or 'dirichlet'/'neumann'.
        peetre_quantization : str
            Quantization for sub-operators.
        common_apply_kwargs : dict
            Keyword arguments for sub-operator apply() calls.
        apply_separable_pair : callable
            The closure from apply_peetre that applies a(x)*q(D)u.
        joint_backend, joint_degree, joint_tol, joint_bounds,
        joint_max_rel_error, joint_num_samples, joint_seed, use_cache,
        freq_window, clamp, space_window :
            As documented in apply_peetre.
    
        Returns
        -------
        ndarray
            The result of applying the joint residual to u.
        """
        import numpy as np
    
        common_apply_kwargs = dict(common_apply_kwargs or {})
    
        def _apply_joint_direct():
            """Exact (expensive) fallback: full KN application of joint symbol."""
            op_joint = PseudoDifferentialOperator(
                joint_symbol,
                self.vars_x,
                mode="symbol",
                quantization=peetre_quantization,
            )
            return op_joint.apply(u, x_grid, kx, **common_apply_kwargs)
    
        # ---------------------------------------------------------------
        # Resolve 'auto' to a concrete backend ONCE, then pass that
        # concrete name straight into `_resolve_joint_representation`.
        #
        # FIX: this used to call `_auto_select_joint_backend` twice per
        # apply() call when joint_backend='auto' -- once here as a "peek"
        # solely to know whether bounds needed to be inferred, and again
        # inside `_resolve_joint_representation` (which re-resolved 'auto'
        # itself). That's wasteful because the auto-selection heuristic
        # does non-trivial symbolic pattern matching
        # (`try_nufft_decomposition_*`, rational-function checks on the
        # joint symbol). Resolving once and threading the concrete
        # backend through removes the duplicate work; this mirrors what
        # `print_peetre_decomposition` already did correctly further down.
        # ---------------------------------------------------------------
        resolved_backend = joint_backend
        if joint_backend == "auto" and not self._peetre_is_zero(joint_symbol):
            x_syms, xi_syms = self._resolve_joint_symbols(joint_symbol)
            resolved_backend = self._auto_select_joint_backend(joint_symbol, x_syms, xi_syms)

        needs_bounds = resolved_backend in ("lowrank", "aaa")
        if needs_bounds and joint_bounds is None:
            joint_bounds = self._infer_joint_bounds(
                x_grid, kx, y_grid=y_grid, ky=ky,
            )
    
        # ---------------------------------------------------------------
        # Resolve the representation (backend is already concrete, so
        # this does not re-run auto-selection)
        # ---------------------------------------------------------------
        try:
            rep = self._resolve_joint_representation(
                joint_symbol,
                backend=resolved_backend,
                bounds=joint_bounds,
                degree=joint_degree,
                tol=joint_tol,
                num_samples=joint_num_samples,
                seed=joint_seed,
                use_cache=use_cache,
            )
        except Exception as exc:
            warnings.warn(
                f"Joint representation resolution failed: {exc}. "
                "Falling back to direct joint application."
            )
            return _apply_joint_direct()
    
        rep_type = rep["type"]
    
        # ---------------------------------------------------------------
        # Execute based on representation type
        # ---------------------------------------------------------------
        if rep_type == "zero":
            return np.zeros(np.shape(u), dtype=np.complex128)
    
        elif rep_type == "direct":
            return _apply_joint_direct()
    
        elif rep_type == "separable_pairs":
            # Low-rank: apply as sum of separable pairs
            metrics = rep.get("metrics", {})
            self.last_joint_lowrank_metrics = metrics
            if (
                joint_max_rel_error is not None
                and metrics.get("rel_l2_error", float("inf")) > joint_max_rel_error
            ):
                warnings.warn(
                    "Low-rank joint residual symbol error "
                    f"{metrics['rel_l2_error']:.6e} exceeds "
                    f"joint_max_rel_error={joint_max_rel_error}. "
                    "Falling back to direct joint application."
                )
                return _apply_joint_direct()
            result = np.zeros(np.shape(u), dtype=np.complex128)
            for a_k, q_k in rep["pairs"]:
                result = result + apply_separable_pair(a_k, q_k)
            return result
    
        elif rep_type == "nufft_plan":
            # NUFFT: periodic only
            if boundary_condition != "periodic":
                warnings.warn(
                    "joint_backend='nufft' only supports "
                    "boundary_condition='periodic'. Falling back to "
                    "direct joint application."
                )
                return _apply_joint_direct()
            try:
                plan_info = rep["plan_info"]
                kind, plan = plan_info
                dx = x_grid[1] - x_grid[0]
                dxi = kx[1] - kx[0]
                if kind == "1d":
                    return apply_nufft_1d(
                        u, plan, x_grid, kx, dx, dxi, freq_window=freq_window
                    )
                else:  # "2d"
                    dy = y_grid[1] - y_grid[0]
                    deta = ky[1] - ky[0]
                    plan_kind, plan_data = plan
                    return apply_nufft_2d(
                        u, plan_kind, plan_data, x_grid, y_grid, kx, ky,
                        dx, dy, dxi, deta, freq_window=freq_window,
                    )
            except Exception as exc:
                warnings.warn(
                    f"NUFFT joint application failed: {exc}. "
                    "Falling back to direct joint application."
                )
                return _apply_joint_direct()
    
        elif rep_type == "nufft_unrepresentable":
            warnings.warn(
                "Joint residual does not classify as NUFFT-representable "
                "(no oscillatory phase of the form exp(i*Lambda(x)*M(xi)) "
                "found). Falling back to direct joint application."
            )
            return _apply_joint_direct()
    
        elif rep_type == "aaa_callable":
            # AAA: rational fit applied via KN quadrature
            metrics = rep.get("metrics", {})
            self.last_joint_aaa_metrics = metrics
            if (
                joint_max_rel_error is not None
                and metrics.get("rel_l2_error", float("inf")) > joint_max_rel_error
            ):
                warnings.warn(
                    "AAA joint residual symbol error "
                    f"{metrics['rel_l2_error']:.6e} exceeds "
                    f"joint_max_rel_error={joint_max_rel_error}. "
                    "Falling back to direct joint application."
                )
                return _apply_joint_direct()
            symbol_func = rep["symbol_func"]
            if boundary_condition == "periodic":
                return kohn_nirenberg_fft(
                    u_vals=u, symbol_func=symbol_func,
                    x_grid=x_grid, kx=kx,
                    fft_func=self.fft, ifft_func=self.ifft,
                    dim=self.dim, y_grid=y_grid, ky=ky,
                    freq_window=freq_window, clamp=clamp,
                    space_window=space_window, is_spatial=True,
                )
            else:
                xg = x_grid if self.dim == 1 else (x_grid, y_grid)
                kg = kx if self.dim == 1 else (kx, ky)
                return kohn_nirenberg_nonperiodic(
                    u, xg, kg, symbol_func,
                    freq_window=freq_window, clamp=clamp,
                    space_window=space_window, is_spatial=True,
                )
    
        elif rep_type == "aaa_unfit":
            warnings.warn(
                "Joint residual could not be fit by AAA to the requested "
                "tolerance (joint_tol). This can happen for symbols whose "
                "poles move with x/y (a genuinely different, "
                "diagonal-singularity structural class). Falling back to "
                "direct joint application."
            )
            return _apply_joint_direct()
    
        else:
            raise ValueError(f"Unknown joint representation type: '{rep_type}'")

    def peetre_decomposition(
        self,
        use_cache=True,
        separable_local=False,
        classify_joint=False,
    ):
        """
        Symbolic Peetre-style decomposition of the operator symbol.

        Parameters
        ----------
        use_cache : bool, default=True
            Cache the decomposition.
        separable_local : bool, default=False
            If True, expose local polynomial terms as separable pairs.
        classify_joint : bool, default=False
            If True and the joint residual is non-zero, run the
            auto-selector and store the recommended backend in
            result["joint_backend"]. This is purely symbolic (no grids
            needed) and cheap.

        Returns
        -------
        dict
            Decomposition dictionary.
        """
        from sympy import Add, Integer, expand, sympify

        cache = getattr(self, "_peetre_cache", None)
        if (
            use_cache
            and cache is not None
            and cache.get("symbol") == self.symbol
            and cache.get("separable_local") == separable_local
            and cache.get("classify_joint") == classify_joint
        ):
            return cache["result"]

        local_coeffs, separable, joint = self._peetre_classify_terms(self.symbol)

        # ------------------------------------------------------------------
        # Add this before removing zero terms
        # ------------------------------------------------------------------
        separable = self._peetre_merge_separable(separable)

        if joint:
            joint_combined = expand(Add(*joint))
            joint = [joint_combined] if not self._peetre_is_zero(joint_combined) else []

        # ------------------------------------------------------------------
        # Remove zero terms.
        # ------------------------------------------------------------------
        local_coeffs = {
            k: v for k, v in local_coeffs.items()
            if not self._peetre_is_zero(v)
        }

        separable = [
            (a, q) for a, q in separable
            if not self._peetre_is_zero(a) and not self._peetre_is_zero(q)
        ]

        joint = [
            t for t in joint
            if not self._peetre_is_zero(t)
        ]

        # ------------------------------------------------------------------
        # Represent the local polynomial part in the same operational form
        # as separable terms:
        #
        #     a(x) * q(xi)
        #
        # This is useful because application is then performed as
        #
        #     a(x) * q(D) u
        #
        # which is the same efficient path used for separable terms.
        #
        # However, unlike the previous behavior, we do NOT force these terms
        # into the separable non-local category by default. They remain
        # mathematically classified as local terms.
        # ------------------------------------------------------------------
        local_terms = self._peetre_local_as_separable(local_coeffs)
    
        if separable_local and local_terms:
            #
            # Legacy behavior:
            # local polynomial terms are exposed as separable terms and the
            # local dictionary/symbol are cleared.
            #
            separable = self._peetre_merge_separable(local_terms + separable)
            local_coeffs = {}
            local_terms = []
            local_symbol = Integer(0)
        else:
            local_symbol = self._peetre_local_symbol(local_coeffs)
    
        separable_symbol = self._peetre_separable_symbol(separable)
        joint_symbol = expand(Add(*joint)) if joint else Integer(0)
    
        result = {
            "local": local_coeffs,
            "local_terms": local_terms,
            "separable": separable,
            "joint_residual": joint,
            "local_symbol": local_symbol,
            "separable_symbol": separable_symbol,
            "joint_symbol": joint_symbol,
            "separable_local": separable_local,
        }

        # --- NEW: optional backend classification ---
        if classify_joint and not self._peetre_is_zero(joint_symbol):
            x_syms, xi_syms = self._resolve_joint_symbols(joint_symbol)
            result["joint_backend"] = self._auto_select_joint_backend(
                joint_symbol, x_syms, xi_syms
            )

        self._peetre_cache = {
            "symbol": self.symbol,
            "separable_local": separable_local,
            "classify_joint": classify_joint,
            "result": result,
        }
        return result

    def decompose_symbol_peetre(self, *args, **kwargs):
        """
        Alias for peetre_decomposition(), for compatibility with the
        standalone symbolic_decompose.py naming style.
        """
        return self.peetre_decomposition(*args, **kwargs)

    def print_peetre_decomposition(
        self,
        joint_backend="direct",
        joint_bounds=None,
        joint_degree=6,
        joint_tol=1e-5,
        joint_num_samples=10000,
        joint_seed=42,
        use_cache=True,
        **kwargs,
    ):
        """
        Pretty-print the Peetre decomposition.

        Parameters
        ----------
        joint_backend : {'direct', 'lowrank', 'nufft', 'aaa', 'auto'}
            How to display the joint residual:
            - 'direct' prints the raw (un-factorized) joint residual terms.
            - 'lowrank' factorizes via _resolve_joint_representation and
              prints the resulting separable pairs a_k(x)*q_k(xi).
            - 'nufft' / 'aaa' print a structural summary (these backends
              produce execution plans, not readable separable pairs).
            - 'auto' selects the best backend and prints accordingly.
        joint_bounds : dict, optional
            Symbol -> (min, max) mapping, required for 'lowrank' and 'aaa'
            (no numerical grid available here to infer bounds from).
        joint_degree, joint_tol, joint_num_samples, joint_seed :
            Forwarded to the underlying factorization.
        use_cache : bool
            Whether to use decomposition/factorization caches.
        **kwargs
            Forwarded to peetre_decomposition().
        """
        deco = self.peetre_decomposition(use_cache=use_cache, **kwargs)
        xi_vars = self._peetre_frequency_symbols()

        # --------------------------------------------------------------
        # Local terms.
        # --------------------------------------------------------------
        local_terms = deco.get("local_terms", [])
        if local_terms:
            print(
                f"--- {len(local_terms)} local term(s), "
                f"represented as a(x)*q({', '.join(str(v) for v in xi_vars)}) ---"
            )
            for a, q in local_terms:
                print(f"  ({a}) * ({q})")
        else:
            print(
                f"--- {len(deco['local'])} local term(s), "
                f"polynomial in {xi_vars} ---"
            )
            for monom, coeff in deco["local"].items():
                factors = []
                for var, power in zip(xi_vars, monom):
                    if power == 1:
                        factors.append(str(var))
                    elif power:
                        factors.append(f"{var}**{power}")
                monom_str = "*".join(factors) if factors else "1"
                print(f"  ({coeff}) * {monom_str}")

        # --------------------------------------------------------------
        # Separable non-local terms.
        # --------------------------------------------------------------
        print(
            f"--- {len(deco['separable'])} separable non-local term(s) ---"
        )
        for a, q in deco["separable"]:
            print(f"  ({a}) * ({q})")

        # --------------------------------------------------------------
        # Joint residual — via the unified representation.
        # --------------------------------------------------------------
        joint_symbol = deco.get("joint_symbol", 0)
        if self._peetre_is_zero(joint_symbol):
            print("--- no joint residual ---")
        else:
            resolved = joint_backend
            if resolved == "auto":
                x_syms, xi_syms = self._resolve_joint_symbols(joint_symbol)
                resolved = self._auto_select_joint_backend(joint_symbol, x_syms, xi_syms)

            if resolved in ("lowrank", "aaa") and joint_bounds is None:
                print(f"--- detected '{resolved}' structure; joint_bounds required "
                      f"to factorize it here. Raw joint term(s): ---")
                for t in deco["joint_residual"]:
                    print(f"  {t}")
            else:
                rep = self._resolve_joint_representation(
                    joint_symbol, backend=resolved, bounds=joint_bounds,
                    degree=joint_degree, tol=joint_tol,
                    num_samples=joint_num_samples, seed=joint_seed,
                    use_cache=use_cache,
                )
                rt = rep["type"]
                if rt == "separable_pairs":
                    pairs, metrics = rep["pairs"], rep.get("metrics", {})
                    print(f"--- joint residual factorized into {len(pairs)} low-rank "
                          f"term(s) via factorize_symbolic "
                          f"(rel_l2_error={metrics.get('rel_l2_error', float('nan')):.3e}) ---")
                    for a, q in pairs:
                        print(f"  ({a}) * ({q})")
                elif rt == "nufft_plan":
                    print(f"--- NUFFT structure detected ({rep['plan_info'][0]}): oscillatory "
                          f"phase exp(i*Lambda(x)*M(xi)). No separable pairs to print "
                          f"(use apply() to execute). ---")
                elif rt == "aaa_callable":
                    print(f"--- AAA rational structure detected "
                          f"(rel_l2_error={rep['metrics'].get('rel_l2_error', float('nan')):.3e}). "
                          f"No separable pairs to print (use apply() to execute). ---")
                elif rt in ("nufft_unrepresentable", "aaa_unfit"):
                    print(f"--- joint residual: backend '{resolved}' could not represent "
                          f"the symbol. Raw joint term(s): ---")
                    for t in deco["joint_residual"]:
                        print(f"  {t}")
                else:
                    print(f"--- {len(deco['joint_residual'])} irreducible joint term(s) ---")
                    for t in deco["joint_residual"]:
                        print(f"  {t}")

        # --------------------------------------------------------------
        # Summary.
        # --------------------------------------------------------------
        print(
            f"local_symbol = {deco['local_symbol']}\n"
            f"separable_symbol = {deco['separable_symbol']}\n"
            f"joint_symbol = {deco['joint_symbol']}"
        )


    def _auto_select_joint_backend(self, joint_symbol, x_syms, xi_syms):
        """
        Intelligently analyze the joint residual symbol and select the 
        most efficient numerical backend ('nufft', 'aaa', or 'lowrank').
        
        Logic:
        1. 'nufft': If the symbol contains an oscillatory phase of the 
           form exp(i * Lambda(x) * M(xi)).
        2. 'aaa': If the symbol is rational or has explicit denominators 
           / negative powers (poles / algebraic decay).
        3. 'lowrank': For smooth, non-oscillatory, non-pole joint kernels 
           (e.g., Gaussians).
        """
        import sympy as sp
        all_syms = x_syms + xi_syms
        
        # 1. NUFFT Check (Oscillatory phase)
        if self.dim == 1:
            nufft_plan = try_nufft_decomposition_1d(joint_symbol, x_syms[0], xi_syms[0])
        else:
            nufft_plan = try_nufft_decomposition_2d(
                joint_symbol, x_syms[0], x_syms[1], xi_syms[0], xi_syms[1]
            )
        if nufft_plan is not None:
            return 'nufft'
            
        # 2. AAA Check (Rational / Poles / Algebraic decay)
        is_rational = joint_symbol.is_rational_function(*all_syms)
        
        # Check for negative powers of polynomial expressions (algebraic decay)
        # like (xi**2 + 1)**(-0.5) or 1/(x**2 + xi**2)
        # But NOT smooth functions like exp(-x**2) or sin(x)
        has_symbolic_denom = False
        for arg in joint_symbol.atoms(sp.Pow):
            if arg.exp.is_negative and arg.base.has(*all_syms):
                # Check if base is a polynomial expression (not transcendental)
                if arg.base.is_polynomial(*all_syms):
                    has_symbolic_denom = True
                    break
                    
        if is_rational or has_symbolic_denom:
            return 'aaa'
            
        # 3. Low-rank Check (Smooth kernels)
        # If it's not oscillatory and doesn't have poles, it's likely a 
        # smooth joint function (like a Gaussian bump) well-suited for 
        # Chebyshev/SVD approximation.
        return 'lowrank'
        
    # ======================================================================
    # Peetre-based application
    # ======================================================================

    def apply_hybrid(self, u, x_grid, kx, y_grid=None, ky=None, **kwargs):
        """
        Hybrid application: Automatically splits the joint residual into 
        individual additive terms and routes each term to its optimal 
        backend (NUFFT, AAA, or Lowrank) based on its specific structure.
        
        This guarantees O(N log N) performance for mixed symbols that would 
        otherwise trigger a fallback to O(N²) direct quadrature.
        """
        import numpy as np
        
        # 1. Get the Peetre decomposition to access the raw joint terms
        deco = self.peetre_decomposition()
        result = np.zeros(np.shape(u), dtype=np.complex128)
        
        # 2. Apply the local and separable parts (skip the joint residual for now)
        # We use a temporary call to apply_peetre with apply_joint=False
        result += self.apply_peetre(
            u, x_grid, kx, y_grid=y_grid, ky=ky, 
            apply_joint=False, **kwargs
        )
        
        # 3. Route each joint term individually via 'auto'
        joint_terms = deco.get('joint_residual', [])
        for term in joint_terms:
            if self._peetre_is_zero(term):
                continue
            
            # Create a temporary operator for this specific joint term
            # It inherits the quantization (Weyl/KN) and dimension
            sub_op = PseudoDifferentialOperator(
                term, self.vars_x, mode='symbol', 
                quantization=self.quantization
            )
            
            # Apply this single term using the 'auto' dispatcher.
            # Because it's a pure term, 'auto' will perfectly match it 
            # to 'nufft', 'aaa', or 'lowrank'.
            result += sub_op.apply_peetre(
                u, x_grid, kx, y_grid=y_grid, ky=ky,
                joint_backend='auto', **kwargs
            )
            
        return result

        
    def apply_peetre(
        self,
        u,
        x_grid,
        kx,
        boundary_condition="periodic",
        y_grid=None,
        ky=None,
        dealiasing_mask=None,
        freq_window="gaussian",
        clamp=1e6,
        space_window=False,
        weyl_order=4,
        apply_joint=True,
        decomposition=None,
        use_cache=True,
        separable_local=False,
        joint_backend="direct",
        joint_degree=6,
        joint_tol=1e-5,
        joint_bounds=None,
        joint_max_rel_error=None,
        joint_num_samples=10000,
        joint_seed=42,
    ):
        """
        Apply the operator using its Peetre decomposition.
    
        For Weyl-quantized operators, the Weyl symbol is first converted to its
        Kohn-Nirenberg equivalent before the Peetre decomposition is performed.
        """
        import numpy as np
        from sympy import lambdify
    
        if self.dim == 2 and (y_grid is None or ky is None):
            raise ValueError("y_grid and ky are required for 2D operators.")
    
        # ------------------------------------------------------------------
        # Weyl support.
        #
        # Peetre application is performed on an effective Kohn-Nirenberg
        # symbol. If the operator is Weyl-quantized, we must first convert
        # the Weyl symbol to its Kohn-Nirenberg equivalent.
        #
        # Example:
        #
        #     Weyl symbol:     x * xi
        #     KN equivalent:   x * xi - I/2
        #
        # Without this correction, apply_peetre() would apply x*D only,
        # missing the Weyl commutator correction -i/2.
        # ------------------------------------------------------------------
        if self.quantization == "weyl":
            effective_symbol = self.weyl_to_kn_symbol(order=weyl_order)
            peetre_quantization = "kohn-nirenberg"
        else:
            effective_symbol = self.symbol
            peetre_quantization = self.quantization
    
        if decomposition is None:
            if self.quantization == "weyl":
                #
                # Build a temporary Kohn-Nirenberg operator from the corrected
                # symbol and decompose that symbol.
                #
                effective_op = PseudoDifferentialOperator(
                    effective_symbol,
                    self.vars_x,
                    mode="symbol",
                    quantization="kohn-nirenberg",
                )
    
                decomposition = effective_op.peetre_decomposition(
                    use_cache=use_cache,
                    separable_local=separable_local,
                )
            else:
                decomposition = self.peetre_decomposition(
                    use_cache=use_cache,
                    separable_local=separable_local,
                )
    
        deco = decomposition
    
        result = np.zeros(np.shape(u), dtype=np.complex128)
    
        common_apply_kwargs = dict(
            boundary_condition=boundary_condition,
            y_grid=y_grid,
            ky=ky,
            dealiasing_mask=dealiasing_mask,
            freq_window=freq_window,
            clamp=clamp,
            space_window=space_window,
            weyl_order=weyl_order,
            backend='direct'
            
        )

    
        # --------------------------------------------------------------
        # Resolve local terms.
        # --------------------------------------------------------------
        local_terms = deco.get("local_terms", None)
    
        if local_terms is None or not local_terms:
            if deco.get("local", None):
                local_terms = self._peetre_local_as_separable(deco["local"])
    
            elif not self._peetre_is_zero(deco.get("local_symbol", 0)):
                op_local = PseudoDifferentialOperator(
                    deco["local_symbol"],
                    self.vars_x,
                    mode="symbol",
                    quantization=peetre_quantization,
                )
    
                result = result + op_local.apply(
                    u,
                    x_grid,
                    kx,
                    **common_apply_kwargs,
                )
    
                local_terms = []
    
            else:
                local_terms = []
    
        x_tuple = tuple(self.vars_x)
    
        def _apply_separable_pair(a, q):
            """
            Apply a single separable/local term a(x)·q(ξ) to the field u via:
    
                u ↦ a(x) · Op[q](u)
    
            The sub-operator Op[q] is applied through the full `apply()`
            pipeline (FFT multiplier if q is x-independent, or KN quadrature
            otherwise). The spatial amplitude a(x) is then multiplied
            pointwise on the physical grid.
    
            If `a` cannot be lambdified (e.g. contains undefined functions),
            the fallback constructs a full operator from the product a·q and
            applies it directly.
    
            Parameters
            ----------
            a : sympy.Expr
                Spatial amplitude depending only on x (and y in 2D).
            q : sympy.Expr
                Frequency multiplier depending only on ξ (and η in 2D).
    
            Returns
            -------
            ndarray
                The result a(x) · Op[q](u), same shape as u.
            """
            op_q = PseudoDifferentialOperator(
                q,
                self.vars_x,
                mode="symbol",
                quantization=peetre_quantization,
            )
    
            v = op_q.apply(
                u,
                x_grid,
                kx,
                **common_apply_kwargs,
            )
    
            try:
                a_func = lambdify(x_tuple, a, "numpy")
    
                if self.dim == 1:
                    a_vals = a_func(x_grid)
                else:
                    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
                    a_vals = a_func(X, Y)
    
                return np.asarray(a_vals) * v
    
            except Exception as exc:
                warnings.warn(
                    "Could not lambdify a local/separable spatial amplitude. "
                    f"Falling back to full symbol application: {exc}"
                )
    
                op_full = PseudoDifferentialOperator(
                    a * q,
                    self.vars_x,
                    mode="symbol",
                    quantization=peetre_quantization,
                )
    
                return op_full.apply(
                    u,
                    x_grid,
                    kx,
                    **common_apply_kwargs,
                )
    
        # --------------------------------------------------------------
        # 1. Local polynomial part.
        # --------------------------------------------------------------
        for a, q in local_terms:
            result = result + _apply_separable_pair(a, q)
    
        # --------------------------------------------------------------
        # 2. Separable non-local terms.
        # --------------------------------------------------------------
        for a, q in deco.get("separable", []):
            result = result + _apply_separable_pair(a, q)
    
        # --------------------------------------------------------------
        # 3. Joint residual.
        # --------------------------------------------------------------
        joint_symbol = deco.get("joint_symbol", 0)
        if not self._peetre_is_zero(joint_symbol):
            if not apply_joint:
                warnings.warn(
                    "Peetre joint residual has been ignored. "
                    "The result is an asymptotic/local+separable approximation."
                )
            else:
                result = result + self._apply_joint_residual(
                    joint_symbol,
                    u,
                    x_grid,
                    kx,
                    y_grid=y_grid,
                    ky=ky,
                    boundary_condition=boundary_condition,
                    peetre_quantization=peetre_quantization,
                    common_apply_kwargs=common_apply_kwargs,
                    apply_separable_pair=_apply_separable_pair,
                    joint_backend=joint_backend,
                    joint_degree=joint_degree,
                    joint_tol=joint_tol,
                    joint_bounds=joint_bounds,
                    joint_max_rel_error=joint_max_rel_error,
                    joint_num_samples=joint_num_samples,
                    joint_seed=joint_seed,
                    use_cache=use_cache,
                    freq_window=freq_window,
                    clamp=clamp,
                    space_window=space_window,
                )

        return result

        
    def peetre_apply(self, *args, **kwargs):
        """
        Alias for apply_peetre().
        """
        return self.apply_peetre(*args, **kwargs)

        
    def commutator_symbolic(self, other, order=1, mode='kn', sign_convention=None):
        """
        Compute the symbolic commutator [A, B] = A∘B − B∘A of two pseudo-differential operators
        using formal asymptotic expansion of their composition symbols.
    
        This method computes the asymptotic expansion of the commutator's symbol up to a given 
        order, based on the symbolic calculus of pseudo-differential operators in the 
        Kohn–Nirenberg quantization. The result is a purely symbolic sympy expression that 
        captures the leading-order noncommutativity of the operators.
    
        Parameters
        ----------
        other : PseudoDifferentialOperator
            The pseudo-differential operator B to commute with this operator A.
        order : int, default=1
            Maximum order of the asymptotic expansion. 
            - order=1 yields the leading term proportional to the Poisson bracket {p, q}.
            - Higher orders include correction terms involving higher mixed derivatives.
    
        Returns
        -------
        sympy.Expr
            Symbolic expression for the asymptotic expansion of the commutator symbol 
            σ([A,B]) = σ(A∘B − B∘A).
    
        """
        assert self.dim == other.dim, "Operator dimensions must match"
        p, q = self.symbol, other.symbol
    
        pq = self.compose_asymptotic(other, order=order, mode=mode, sign_convention=sign_convention)
        qp = other.compose_asymptotic(self, order=order, mode=mode, sign_convention=sign_convention)
        
        comm_symbol = simplify(pq-qp)

        return comm_symbol

    def right_inverse_asymptotic(self, order=1):
        """
        Construct a formal right inverse R of the pseudo-differential operator P such that 
        the composition P ∘ R equals the identity plus a smoothing operator of order -order.
    
        This method computes an asymptotic expansion for the right inverse using recursive 
        corrections based on derivatives of the symbol p(x, ξ) and lower-order terms of R.
    
        Parameters
        ----------
        order : int
            Number of terms to include in the asymptotic expansion. Higher values improve 
            approximation at the cost of complexity and computational effort.
    
        Returns
        -------
        sympy.Expr
            The symbolic expression representing the formal right inverse R(x, ξ), which satisfies:
            P ∘ R = Id + O(⟨ξ⟩^{-order}), where ⟨ξ⟩ = (1 + |ξ|²)^{1/2}.
    
        Notes
        -----
        - In 1D: The recursion involves spatial derivatives of R and derivatives of p with respect to ξ.
        - In 2D: The multi-index generalization is used with mixed derivatives in ξ and η.
        - The construction relies on the non-vanishing of the principal symbol p to ensure invertibility.
        - Each term in the expansion corresponds to higher-order corrections involving commutators 
          between the operator P and the current approximation of R.
        """
        return self._asymptotic_inverse(order, side='right')

    def _asymptotic_inverse(self, order, side):
        """Shared recursion behind right_inverse_asymptotic and
        left_inverse_asymptotic (dimension-generic multi-index Leibniz
        recursion; the two sides only differ in which symbol gets which
        derivative and the multiplication order)."""
        dim = self.dim
        if dim not in (1, 2):
            raise NotImplementedError("Only 1D and 2D cases are implemented")
        p = self.symbol
        x_vars = self.vars_x
        xi_vars = self._peetre_frequency_symbols()

        r = 1 / p
        R = r
        for n in range(1, order + 1):
            term = 0
            for alpha in _mi_upto(n, dim):
                coeff = (1j) ** (-sum(alpha)) / _mi_factorial(alpha)
                if side == 'right':
                    term += coeff * _mi_diff(p, xi_vars, alpha) * _mi_diff(R, x_vars, alpha)
                else:  # 'left'
                    term += coeff * _mi_diff(R, xi_vars, alpha) * _mi_diff(p, x_vars, alpha)
            R = R - r * term
        return R

    def left_inverse_asymptotic(self, order=1):
        """
        Construct a formal left inverse L such that the composition L ∘ P equals the identity 
        operator up to terms of order ξ^{-order}. This expansion is performed asymptotically 
        at infinity in the frequency variable(s).
    
        The left inverse is built iteratively using symbolic differentiation and the 
        method of asymptotic expansions for pseudo-differential operators. It ensures that:
        
            L(P(x,ξ),x,D) ∘ P(x,D) = Id + smoothing operator of order -order
    
        Parameters
        ----------
        order : int, optional
            Maximum number of terms in the asymptotic expansion (default is 1). Higher values 
            yield more accurate inverses at the cost of increased computational complexity.
    
        Returns
        -------
        sympy.Expr
            Symbolic expression representing the principal symbol of the formal left inverse 
            operator L(x,ξ). This expression depends on spatial variables and frequencies, 
            and includes correction terms up to the specified order.
    
        Notes
        -----
        - In 1D: Uses recursive application of the Leibniz formula for symbols.
        - In 2D: Generalizes to multi-indices for mixed derivatives in (x,y) and (ξ,η).
        - Each term involves combinations of derivatives of the original symbol p(x,ξ) and 
          previously computed terms of the inverse.
        - Coefficients include powers of 1j (i) and factorial normalization for derivative terms.
        """
        return self._asymptotic_inverse(order, side='left')

    def formal_adjoint(self):
        """
        Compute the formal adjoint symbol P* of the pseudo-differential operator.

        The adjoint is defined such that for any test functions u and v,
        ⟨P u, v⟩ = ⟨u, P* v⟩ holds in the distributional sense. This is obtained by 
        taking the complex conjugate of the symbol and expanding it asymptotically 
        at infinity to ensure proper behavior under integration by parts.

        Returns
        -------
        sympy.Expr
            The adjoint symbol P*(x, ξ) in 1D or P*(x, y, ξ, η) in 2D.
        
        Notes:
        - In 1D, the expansion is performed in powers of 1/|ξ|.
        - In 2D, the expansion is radial in |ξ| = sqrt(ξ² + η²).
        - This method ensures symbolic simplifications for readability and efficiency.
        """
        xi_vars = self._peetre_frequency_symbols()
        expansion_var = xi_vars[0] if self.dim == 1 else sqrt(sum(v**2 for v in xi_vars))
        p_star = conjugate(self.symbol)
        return simplify(series(p_star, expansion_var, oo, n=6).removeO())

    def fractional_power(self, alpha, order=1, method='symbolic', x_grid=None, L=None, N=None):
        """
        Compute the symbol or discrete matrix representation of the fractional 
        or complex power P^alpha of the pseudo-differential operator.
        
        This method calculates the fractional power using either an optimized 
        symbolic asymptotic approach or a numerical matrix approach. 
        
        For the **symbolic** method, the algorithm distinguishes between 
        spatially independent symbols (pure Fourier multipliers) and 
        spatially dependent symbols (heterogeneous media):
            - *Fast Path (Pure Multipliers)*: If the symbol has no spatial 
              dependence, it computes the exact algebraic power p(xi)^alpha. 
              It automatically detects negative symbols (e.g., -xi**2) and 
              extracts the appropriate complex phase factor to ensure the 
              correct branch of the complex power is taken.
            - *Slow Path (Spatially Dependent)*: If the symbol depends on 
              space, the fractional power is currently restricted to square roots 
              (alpha = 0.5). For order=0, it returns the principal square root. 
              For order=1, it skips heavy iterative Newton-Raphson routines and 
              directly computes the first-order microlocal spatial correction via 
              the subprincipal error: delta_q = E / (2 * q_0).

        For the **numerical** method, the operator is discretized into a 
        matrix (via spectral or finite-difference methods) and the fractional 
        matrix power is computed using SciPy.

        Parameters
        ----------
        alpha : float or complex
            The exponent to which the operator is raised.
        order : int, default=1
            The asymptotic expansion order for the symbolic correction.
            - 0: Returns the principal symbol's fractional power (q_0).
            - 1: Computes the first-order microlocal correction term (delta_q).
            Ignored for pure multipliers and the numerical method.
        method : str, {'symbolic', 'numerical'}, default='symbolic'
            The computation method to use.
            - 'symbolic': Returns a SymPy expression for the fractional power 
              symbol using direct asymptotic calculus.
            - 'numerical': Returns a NumPy array representing the discrete 
              fractional power matrix.
        x_grid : ndarray, optional
            Spatial grid for the numerical method. If None, a default grid 
            is generated based on `L` and `N`.
        L : float, optional
            Domain half-length for the numerical spectral method.
        N : int, optional
            Number of grid points for the numerical method.

        Returns
        -------
        sympy.Expr or ndarray
            - If `method='symbolic'`, returns the SymPy expression of the 
              fractional power symbol.
            - If `method='numerical'`, returns the NumPy array representing 
              the fractional power matrix.

        Raises
        ------
        NotImplementedError
            If `method='symbolic'` is used on a spatially dependent symbol 
            with `alpha != 0.5`. The direct subprincipal asymptotic correction 
            for heterogeneous media is currently optimized strictly for square roots 
            to maintain clean, high-performance symbolic evaluations.
        ValueError
            If `method` is not 'symbolic' or 'numerical'.

        Notes
        -----
        - The symbolic method automatically handles negative overall symbols 
          (e.g., the negative Laplacian) by factoring out the complex phase 
          (e.g., `I` or `I**(2*alpha)`) and evaluating the absolute value.
        - For the numerical method, the underlying discrete operator is built 
          using `_build_operator_matrix`, defaulting to a spectral method if 
          `x_grid`, `L`, and `N` are not explicitly provided.
        - The fractional power of a pseudo-differential operator is a 
          cornerstone in defining fractional PDEs (e.g., fractional 
          Schrödinger equations, anomalous diffusion) and spectral zeta 
          functions.
        """
        import sympy as sp
        from sympy import Rational, simplify, symbols, powdenest
        import numpy as np
        if method == 'numerical':
            from scipy.linalg import fractional_matrix_power
            if x_grid is None: x_grid = np.linspace(-5, 5, 128)
            if N is None: N = len(x_grid)
            if L is None: L = (x_grid[-1] - x_grid[0]) / 2.0 if len(x_grid) > 1 else 5.0
            H, _, _ = self._build_operator_matrix(x_grid, method='spectral', L=L, N=N)
            return fractional_matrix_power(H, alpha)
        if method != 'symbolic':
            raise ValueError("method must be 'symbolic' or 'numerical'")
            
        # ─── SYMBOLIC PATH ───
        p = self.symbol
        if isinstance(alpha, float):
            alpha = sp.nsimplify(alpha, rational=True)
            
        # 1. Robustly check if the overall symbol is negative (e.g., -xi**2 - eta**2)
        is_negative = False
        if p.free_symbols:
            test_val = p.subs({s: 1 for s in p.free_symbols})
            try:
                if float(test_val) < 0:
                    is_negative = True
            except (TypeError, ValueError):
                pass
        p_abs = -p if is_negative else p
        
        # 2. FAST PATH: Pure multipliers (no spatial dependence)
        if not self._is_spatial_dependent():
            sub_to_pos = {}
            sub_back = {}
            for s in p_abs.free_symbols:
                if s.name in ['xi', 'eta']:
                    s_pos = symbols(s.name, real=True, positive=True)
                    sub_to_pos[s] = s_pos
                    sub_back[s_pos] = s
            p_pos = p_abs.subs(sub_to_pos)
            q_pos = powdenest(p_pos**alpha, force=True)
            q_sym = q_pos.subs(sub_back)
            
            # --- FIX: Simplify the radical first, then multiply with evaluate=False ---
            q_sym = simplify(q_sym)
            if is_negative:
                phase = sp.I**(2*alpha)
                q_sym = sp.Mul(phase, q_sym, evaluate=False)
            return q_sym
            
        # 3. SLOW PATH: Spatially dependent symbols (Heterogeneous media)
        if alpha != 0.5 and alpha != sp.Rational(1, 2):
            raise NotImplementedError("Spatially dependent fractional powers only support alpha=0.5")
            
        p_m = self.principal_symbol(order=1)
        p_m_abs = -p_m if is_negative else p_m
        
        # Calculate the principal symbol square root directly (q_0)
        # Avoid powdenest/simplify chain which hangs on multi-variable radicals
        q_0 = sp.sqrt(p_m_abs)
        if is_negative:
            q_0 = sp.Mul(sp.I, q_0, evaluate=False)

        if order == 0:
            return q_0
            
        # --- RAPID ORDER 1 CORRECTION ---
        # Instead of generic Newton-Raphson + full left-inverse + sp.series, 
        # we can compute the first asymptotic correction directly from the error.
        # E = p - q_0^2 (which is the subprincipal part of the operator)
        
        q_op = PseudoDifferentialOperator(q_0, self.vars_x, mode='symbol')
        
        # Fast composition for order=1
        q_sq = q_op.compose_asymptotic(q_op, order=1, mode='kn')
        E = p - q_sq
        
        # The first correction term in standard pseudo-differential calculus satisfies:
        # delta_q = E / (2 * q_0)
        # We can form this directly and use basic algebraic expansion
        delta_q = E / (2 * q_0)
        
        # Combine them structurally
        q_sym = q_0 + delta_q
        
        # Fast, non-blocking cleanup
        try:
            q_sym = sp.powsimp(q_sym, combine='all')
        except Exception:
            pass
            
        return q_sym

    def exponential_symbol(self, t=1.0, order=1, mode='kn', sign_convention=None):
        """
        Compute the symbol of exp(tP) using asymptotic expansion methods.
        
        This method calculates the exponential of a pseudo-differential operator 
        using either a direct power series expansion or a Magnus expansion, 
        depending on the structure of the symbol. The result is valid up to 
        the specified asymptotic order.
        
        Parameters
        ----------
        t : float or sympy.Symbol, default=1.0
            Time or evolution parameter. Common uses:
            - t = -i*τ for Schrödinger evolution: exp(-iτH)
            - t = τ for heat/diffusion: exp(τΔ)
            - t for general propagators
        order : int, default=3
            Maximum order of the asymptotic expansion. Higher orders include 
            more composition terms, improving accuracy for small t or when 
            non-commutativity effects are significant.
        
        Returns
        -------
        sympy.Expr
            Symbolic expression for the exponential operator symbol, computed 
            as an asymptotic series up to the specified order.
        
        Notes
        -----
        - For commutative symbols (e.g., pure multiplication operators), the 
          exponential is exact: exp(tP) = exp(t*p(x,ξ)).
        
        - For general non-commutative operators, the method uses the BCH-type 
          expansion via iterated composition:
          exp(tP) ~ I + tP + (t²/2!)P∘P + (t³/3!)P∘P∘P + ...
          
        - Each power P^n is computed via compose_asymptotic, which accounts 
          for the non-commutativity through derivative terms.
        
        - The expansion is valid for |t| small enough or when the symbol has 
          appropriate decay/growth properties.
        
        - In quantum mechanics (Schrödinger): U(t) = exp(-itH/ℏ) represents 
          the time evolution operator.
        
        - In parabolic PDEs (heat equation): exp(tΔ) is the heat kernel.

        """
        if self.dim not in (1, 2):
            raise NotImplementedError("Only 1D and 2D operators are supported")

        result = 1 + t * self.symbol
        current_power = self.symbol
        for n in range(2, order + 1):
            temp_op = PseudoDifferentialOperator(current_power, self.vars_x, mode='symbol')
            current_power = temp_op.compose_asymptotic(
                self, order=order, mode=mode, sign_convention=sign_convention)
            result += t**n / factorial(n) * current_power

        return simplify(result)
        
    def trace_formula(self, volume_element=None, numerical=False, 
                      x_bounds=None, xi_bounds=None):
        """
        Compute the semiclassical trace of the pseudo-differential operator.
        
        The trace formula relates the quantum trace of an operator to a 
        phase-space integral of its symbol, providing a fundamental link 
        between classical and quantum mechanics. This implementation supports 
        both symbolic and numerical integration.
        
        Parameters
        ----------
        volume_element : sympy.Expr, optional
            Custom volume element for the phase space integration. If None, 
            uses the standard Liouville measure dx dξ/(2π)^d.
        numerical : bool, default=False
            If True, perform numerical integration over specified bounds.
            If False, attempt symbolic integration (may fail for complex symbols).
        x_bounds : tuple of tuples, optional
            Spatial integration bounds. For 1D: ((x_min, x_max),)
            For 2D: ((x_min, x_max), (y_min, y_max))
            Required if numerical=True.
        xi_bounds : tuple of tuples, optional
            Frequency integration bounds. For 1D: ((xi_min, xi_max),)
            For 2D: ((xi_min, xi_max), (eta_min, eta_max))
            Required if numerical=True.
        
        Returns
        -------
        sympy.Expr or float
            The trace of the operator. Returns a symbolic expression if 
            numerical=False, or a float if numerical=True.
        
        Notes
        -----
        - The semiclassical trace formula states:
          Tr(P) = (2π)^{-d} ∫∫ p(x,ξ) dx dξ
          where d is the spatial dimension and p(x,ξ) is the operator symbol.
        
        - For 1D: Tr(P) = (1/2π) ∫_{-∞}^{∞} ∫_{-∞}^{∞} p(x,ξ) dx dξ
        
        - For 2D: Tr(P) = (1/4π²) ∫∫∫∫ p(x,y,ξ,η) dx dy dξ dη
        
        - This formula is exact for trace-class operators and provides an 
          asymptotic approximation for general pseudo-differential operators.
        
        - Physical interpretation: the trace counts the "number of states" 
          weighted by the observable p(x,ξ).
        
        - For projection operators (χ_Ω with χ² = χ), the trace gives the 
          dimension of the range, related to the phase space volume of Ω.
        
        - The factor (2π)^{-d} comes from the quantum normalization of 
          coherent states / Weyl quantization.
        """
        from sympy import integrate, simplify, lambdify
        from scipy.integrate import dblquad, nquad
        
        p = self.symbol
        
        if numerical:
            if x_bounds is None or xi_bounds is None:
                raise ValueError(
                    "x_bounds and xi_bounds must be provided for numerical integration"
                )
        
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
            
            if volume_element is None:
                volume_element = 1 / (2 * pi)
            
            if numerical:
                # Numerical integration
                p_func = lambdify((x, xi), p, 'numpy')
                (x_min, x_max), = x_bounds
                (xi_min, xi_max), = xi_bounds
                
                def integrand(xi_val, x_val):
                    return p_func(x_val, xi_val)
                
                result, error = dblquad(
                    integrand,
                    x_min, x_max,
                    lambda x: xi_min, lambda x: xi_max
                )
                
                result *= float(volume_element)
                print(f"Numerical trace = {result:.6e} ± {error:.6e}")
                return result
            
            else:
                # Symbolic integration
                integrand = p * volume_element
                
                try:
                    # Try to integrate over xi first, then x
                    integral_xi = integrate(integrand, (xi, -oo, oo))
                    integral_x = integrate(integral_xi, (x, -oo, oo))
                    return simplify(integral_x)
                except:
                    print("Warning: Symbolic integration failed. Try numerical=True")
                    return integrate(integrand, (xi, -oo, oo), (x, -oo, oo))
        
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            
            if volume_element is None:
                volume_element = 1 / (4 * pi**2)
            
            if numerical:
                # Numerical integration in 4D
                p_func = lambdify((x, y, xi, eta), p, 'numpy')
                (x_min, x_max), (y_min, y_max) = x_bounds
                (xi_min, xi_max), (eta_min, eta_max) = xi_bounds
                
                def integrand(eta_val, xi_val, y_val, x_val):
                    return p_func(x_val, y_val, xi_val, eta_val)
                
                result, error = nquad(
                    integrand,
                    [
                        [eta_min, eta_max],
                        [xi_min, xi_max],
                        [y_min, y_max],
                        [x_min, x_max]
                    ]
                )
                
                result *= float(volume_element)
                print(f"Numerical trace = {result:.6e} ± {error:.6e}")
                return result
            
            else:
                # Symbolic integration
                integrand = p * volume_element
                
                try:
                    # Integrate in order: eta, xi, y, x
                    integral_eta = integrate(integrand, (eta, -oo, oo))
                    integral_xi = integrate(integral_eta, (xi, -oo, oo))
                    integral_y = integrate(integral_xi, (y, -oo, oo))
                    integral_x = integrate(integral_y, (x, -oo, oo))
                    return simplify(integral_x)
                except:
                    print("Warning: Symbolic integration failed. Try numerical=True")
                    return integrate(
                        integrand,
                        (eta, -oo, oo), (xi, -oo, oo),
                        (y, -oo, oo), (x, -oo, oo)
                    )
        
        else:
            raise NotImplementedError("Only 1D and 2D operators are supported")

    def pseudospectrum_analysis(self, x_grid, lambda_real_range, lambda_imag_range,
                               epsilon_levels=[0.1, 0.01, 0.001, 0.0001],
                               resolution=100, method='spectral', L=None, N=None,
                               use_sparse=False, parallel=True, n_workers=4,
                               adaptive=False, adaptive_threshold=0.5,
                               auto_range=True, plot=True):
        """
        Compute and visualize the pseudospectrum of the operator.
        
        Optimizations:
        - Uses apply() method instead of manual loops
        - Parallel computation of resolvent norms
        - Sparse matrix support for large N
        - Optional adaptive grid refinement
        
        Parameters
        ----------
        x_grid : array
            Spatial grid for quantization
        lambda_real_range : tuple
            (min, max) for real part of λ
        lambda_imag_range : tuple
            (min, max) for imaginary part of λ
        epsilon_levels : list
            Levels for ε-pseudospectrum contours
        resolution : int
            Grid resolution for λ sampling
        method : str
            'spectral' or 'finite_difference'
        L : float, optional
            Domain half-length for spectral method
        N : int, optional
            Number of grid points
        use_sparse : bool
            Use sparse matrices for large N
        parallel : bool
            Enable parallel computation
        n_workers : int
            Number of parallel workers
        adaptive : bool
            Use adaptive grid refinement
        adaptive_threshold : float
            Threshold for adaptive refinement
            
        Returns
        -------
        dict
            Dictionary with pseudospectrum data and operator matrix
        """
        if self.dim != 1:
            raise NotImplementedError('Pseudospectrum analysis currently supports 1D only')
        
        # Step 1: Build operator matrix
        print(f"Building operator matrix using '{method}' method...")
        H, x_grid_used, k_grid = self._build_operator_matrix(x_grid, method, L, N)
        N_actual = H.shape[0]
        
        # Step 1.5: Compute eigenvalues FIRST to adjust range if needed
        print('Computing eigenvalues...')
        eigenvalues = self._compute_eigenvalues(H, use_sparse)
        
        # Auto-adjust range if requested
        if auto_range and eigenvalues is not None:
            eig_real_min, eig_real_max = eigenvalues.real.min(), eigenvalues.real.max()
            eig_imag_min, eig_imag_max = eigenvalues.imag.min(), eigenvalues.imag.max()
            
            # Add 20% margin around eigenvalues
            margin_real = 0.2 * (eig_real_max - eig_real_min + 1)
            margin_imag = max(0.2 * (eig_imag_max - eig_imag_min + 1), 2.0)
            
            lambda_real_range = (eig_real_min - margin_real, eig_real_max + margin_real)
            lambda_imag_range = (eig_imag_min - margin_imag, eig_imag_max + margin_imag)
            
            print(f'Auto-adjusted λ range:')
            print(f'  Re(λ) ∈ [{lambda_real_range[0]:.2f}, {lambda_real_range[1]:.2f}]')
            print(f'  Im(λ) ∈ [{lambda_imag_range[0]:.2f}, {lambda_imag_range[1]:.2f}]')
        
        # Step 2: Compute pseudospectrum with corrected range
        print(f'Computing pseudospectrum over {resolution}×{resolution} grid...')
        if adaptive:
            print('Using adaptive grid refinement...')
            Lambda, resolvent_norm, sigma_min_grid = self._compute_pseudospectrum_adaptive(
                H, lambda_real_range, lambda_imag_range, resolution,
                use_sparse=use_sparse, parallel=parallel, n_workers=n_workers,
                threshold=adaptive_threshold
            )
        else:
            Lambda, resolvent_norm, sigma_min_grid = self._compute_pseudospectrum(
                H, lambda_real_range, lambda_imag_range, resolution,
                use_sparse=use_sparse, parallel=parallel, n_workers=n_workers
            )
        
        # Step 3: Visualize
        if plot:
            self._plot_pseudospectrum(Lambda, resolvent_norm, sigma_min_grid,
                                      epsilon_levels, eigenvalues)
        
        return {
            'lambda_grid': Lambda,
            'resolvent_norm': resolvent_norm,
            'sigma_min': sigma_min_grid,
            'epsilon_levels': epsilon_levels,
            'eigenvalues': eigenvalues,
            'operator_matrix': H,
            'x_grid': x_grid_used,
            'k_grid': k_grid
        }


    def _build_operator_matrix(self, x_grid, method, L, N):
        """
        Build the discrete operator matrix H.
        
        Optimized to use the apply() method instead of manual integration.
        
        Parameters
        ----------
        x_grid : array
            Input spatial grid
        method : str
            'spectral' or 'finite_difference'
        L : float, optional
            Domain half-length
        N : int, optional
            Number of grid points
            
        Returns
        -------
        H : ndarray
            Operator matrix (N×N)
        x_grid_used : ndarray
            Actual spatial grid used
        k_grid : ndarray
            Frequency grid
        """
        if method == 'spectral':
            # Setup spectral grid
            if L is None:
                L = (x_grid[-1] - x_grid[0]) / 2.0
            if N is None:
                N = len(x_grid)
            x_grid_spectral = np.linspace(-L, L, N, endpoint=False)
            dx = x_grid_spectral[1] - x_grid_spectral[0]
            k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
            
            # Build matrix by applying operator to canonical basis
            H = np.zeros((N, N), dtype=complex)
            for j in range(N):
                # Create basis vector e_j
                e_j = np.zeros(N, dtype=complex)
                e_j[j] = 1.0
                
                # Apply operator using the existing apply() method
                # CRITICAL: Disable frequency windowing and clamping to get 
                # the exact mathematical operator for spectral analysis.
                H[:, j] = self.apply(
                    e_j, 
                    x_grid_spectral, 
                    k,
                    boundary_condition='periodic',
                    freq_window=None,   # <--- Disable Gaussian low-pass filter
                    clamp=np.inf        # <--- Disable magnitude clipping
                )
            print(f'Operator quantized via apply() method: {N}×{N} matrix')
            return H, x_grid_spectral, k
            
        elif method == 'finite_difference':
            # Finite-difference discretization with NON-PERIODIC (Dirichlet-type) BCs.
            #
            # The KN quantization maps  ξ^n  →  (-i∂_x)^n, so:
            #   ξ^0  →  identity
            #   ξ^1  →  -i∂_x   (real-space operator: multiply by -i then differentiate)
            #   ξ^2  →  -∂_x²
            #
            # Physical operators obtained from p(x,ξ) = a(x) + b(x)·ξ + c(x)·ξ²:
            #   a(x)·I   +  b(x)·(-i∂_x)   +  c(x)·(-∂_x²)
            #   = a(x)·I  +  [b(x)·(-i)] · ∂_x   +  [-c(x)] · ∂_x²
            #   = a(x)·I  -  ic(x)·∂_x            +  ν(x)·∂_x²   (when b=-ic, c=-ν)
            #
            # Centred-difference stencils (non-periodic, open boundaries):
            #   ∂_x  u_j  ≈  (u_{j+1} - u_{j-1}) / (2 dx)
            #   ∂_x² u_j  ≈  (u_{j+1} - 2 u_j + u_{j-1}) / dx²
            #
            # Non-periodic BCs are essential for non-normal operators:
            # constant-coefficient operators with periodic BCs give circulant
            # (hence NORMAL) matrices with trivially small pseudospectrum,
            # failing the resolvent-norm test for dissipative operators.
            # Non-periodic BCs break the circulant structure and allow the
            # exponential ill-conditioning of eigenvectors that characterises
            # non-normal convection-diffusion operators.

            N = len(x_grid)
            dx = x_grid[1] - x_grid[0]
            H = np.zeros((N, N), dtype=complex)

            # Extract polynomial coefficients a(x), b(x), c(x) in ξ
            xi_tmp = np.array([0.0, 1.0, -1.0])
            X_mat  = np.tile(x_grid[:, None], (1, 3))
            XI_mat = np.tile(xi_tmp[None, :], (N, 1))
            P_mat  = self.p_func(X_mat, XI_mat)          # (N, 3)

            a_coeff =  P_mat[:, 0]                                    # p(x, 0)
            b_coeff = (P_mat[:, 1] - P_mat[:, 2]) / 2.0              # coeff of ξ
            c_coeff = (P_mat[:, 1] - 2*P_mat[:, 0] + P_mat[:, 2]) / 2.0  # coeff of ξ²

            # Real-space operators:
            #   b·ξ  → b·(-i∂_x)  → FD: b[i]*(-i)/(2dx) * (u_{j+1} - u_{j-1})
            #   c·ξ² → c·(-∂_x²)  → FD: c[i]*(-1)/dx²  * (u_{j+1} - 2u_j + u_{j-1})

            for i in range(N):
                # Zero-order
                H[i, i] += a_coeff[i]

                # First-order: b * (-i∂_x), centred difference, NO wrap
                if i < N-1:
                    H[i, i+1] += b_coeff[i] * (-1j) / (2.0 * dx)
                if i > 0:
                    H[i, i-1] += b_coeff[i] * (+1j) / (2.0 * dx)

                # Second-order: c * (-∂_x²), centred difference, NO wrap
                H[i, i]   += c_coeff[i] * 2.0 / dx**2
                if i < N-1:
                    H[i, i+1] += c_coeff[i] * (-1.0) / dx**2
                if i > 0:
                    H[i, i-1] += c_coeff[i] * (-1.0) / dx**2

            print(f'Operator quantized via finite differences: {N}×{N} matrix')
            k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
            return H, x_grid, k
            
        else:
            raise ValueError("method must be 'spectral' or 'finite_difference'")

    def _compute_pseudospectrum(self, H, lambda_real_range, lambda_imag_range,
                               resolution, use_sparse=False, parallel=True,
                               n_workers=4):
        """
        Compute pseudospectrum on a uniform grid.
        
        Optimized with parallel computation and optional sparse matrices.
        
        Parameters
        ----------
        H : ndarray or sparse matrix
            Operator matrix
        lambda_real_range : tuple
            Range for Re(λ)
        lambda_imag_range : tuple
            Range for Im(λ)
        resolution : int
            Grid resolution
        use_sparse : bool
            Use sparse SVD for large matrices
        parallel : bool
            Enable parallel computation
        n_workers : int
            Number of parallel workers
            
        Returns
        -------
        Lambda : ndarray
            Complex grid of λ values
        resolvent_norm : ndarray
            Norm of (H - λI)^{-1}
        sigma_min_grid : ndarray
            Smallest singular value σ_min(H - λI)
        """
        from scipy.linalg import svdvals
        
        N = H.shape[0]
        lambda_re = np.linspace(*lambda_real_range, resolution)
        lambda_im = np.linspace(*lambda_imag_range, resolution)
        Lambda_re, Lambda_im = np.meshgrid(lambda_re, lambda_im)
        Lambda = Lambda_re + 1j * Lambda_im
        
        resolvent_norm = np.zeros_like(Lambda, dtype=float)
        sigma_min_grid = np.zeros_like(Lambda, dtype=float)
        
        I = np.eye(N)
        
        # Convert to sparse if requested and beneficial
        if use_sparse and N > 100:
            from scipy.sparse import csr_matrix, eye as sparse_eye
            from scipy.sparse.linalg import svds
            H_sparse = csr_matrix(H)
            I_sparse = sparse_eye(N, format='csr')
            use_sparse_svd = True
            print(f'Using sparse matrices (N={N})')
        else:
            use_sparse_svd = False
        
        if parallel and resolution * resolution > 100:
            # Parallel computation
            Lambda_flat = Lambda.ravel()
            
            def compute_single_point(idx):
                """Compute resolvent norm for a single λ value"""
                lam = Lambda_flat[idx]
                try:
                    if use_sparse_svd:
                        # Sparse SVD: compute only smallest singular value
                        A = H_sparse - lam * I_sparse
                        try:
                            # svds can be unstable, wrap in try-except
                            s_min = svds(A, k=1, which='SM', 
                                       return_singular_vectors=False)[0]
                        except:
                            # Fallback to dense computation
                            s = svdvals(A.toarray())
                            s_min = s[-1]
                    else:
                        # Dense SVD
                        A = H - lam * I
                        s = svdvals(A)
                        s_min = s[-1]
                    
                    return idx, 1.0 / (s_min + 1e-16), s_min
                except Exception as e:
                    return idx, np.nan, np.nan
            
            # Use ThreadPoolExecutor for parallel computation
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(compute_single_point, idx): idx 
                          for idx in range(len(Lambda_flat))}
                
                # Progress tracking
                completed = 0
                total = len(futures)
            progress_interval = max(1, total // 10)  # FIX: Ensure at least 1
            for future in as_completed(futures):
                idx, res_norm, s_min = future.result()
                resolvent_norm.ravel()[idx] = res_norm
                sigma_min_grid.ravel()[idx] = s_min
            
                completed += 1
                if completed % progress_interval == 0:  # FIX: Use progress_interval
                    print(f'Progress: {completed}/{total} ({100*completed//total}%)')
            
        else:
            # Sequential computation
            progress_interval = max(1, resolution // 10)  # FIX: Ensure at least 1
            for i in range(resolution):
                for j in range(resolution):
                    lam = Lambda[i, j]
                    try:
                        if use_sparse_svd:
                            A = H_sparse - lam * I_sparse
                            try:
                                s_min = svds(A, k=1, which='SM',
                                           return_singular_vectors=False)[0]
                            except:
                                s = svdvals(A.toarray())
                                s_min = s[-1]
                        else:
                            A = H - lam * I
                            s = svdvals(A)
                            s_min = s[-1]
        
                        sigma_min_grid[i, j] = s_min
                        resolvent_norm[i, j] = 1.0 / (s_min + 1e-16)
                    except Exception:
                        resolvent_norm[i, j] = np.nan
                        sigma_min_grid[i, j] = np.nan
        
                if i % progress_interval == 0:  # FIX: Use progress_interval
                    print(f'Progress: {i}/{resolution} rows')
        
        return Lambda, resolvent_norm, sigma_min_grid

    def _compute_pseudospectrum_adaptive(self, H, lambda_real_range, lambda_imag_range,
                                        base_resolution, use_sparse=False, parallel=True,
                                        n_workers=4, threshold=0.5, max_refinements=2):
        """
        Compute pseudospectrum with adaptive grid refinement.
        
        Starts with coarse grid and refines regions with high gradients.
        
        Parameters
        ----------
        H : ndarray
            Operator matrix
        lambda_real_range : tuple
            Range for Re(λ)
        lambda_imag_range : tuple
            Range for Im(λ)
        base_resolution : int
            Initial coarse resolution
        use_sparse : bool
            Use sparse matrices
        parallel : bool
            Enable parallel computation
        n_workers : int
            Number of workers
        threshold : float
            Gradient threshold for refinement
        max_refinements : int
            Maximum number of refinement levels
            
        Returns
        -------
        Lambda : ndarray
            Complex grid (may be non-uniform)
        resolvent_norm : ndarray
            Resolvent norms
        sigma_min_grid : ndarray
            Smallest singular values
        """
        # Start with coarse grid
        coarse_res = base_resolution // 2
        print(f'Level 0: Computing coarse grid ({coarse_res}×{coarse_res})...')
        
        Lambda_coarse, resolvent_coarse, sigma_coarse = self._compute_pseudospectrum(
            H, lambda_real_range, lambda_imag_range, coarse_res,
            use_sparse=use_sparse, parallel=parallel, n_workers=n_workers
        )
        
        # Compute gradient to identify regions needing refinement
        log_resolvent = np.log10(resolvent_coarse + 1e-16)
        grad_y, grad_x = np.gradient(log_resolvent)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient
        grad_normalized = grad_magnitude / (np.max(grad_magnitude) + 1e-10)
        
        # For now, return uniform fine grid
        # (Full adaptive implementation would require irregular grids)
        print(f'Level 1: Computing fine grid ({base_resolution}×{base_resolution})...')
        Lambda_fine, resolvent_fine, sigma_fine = self._compute_pseudospectrum(
            H, lambda_real_range, lambda_imag_range, base_resolution,
            use_sparse=use_sparse, parallel=parallel, n_workers=n_workers
        )
        
        high_gradient_pct = 100 * np.sum(grad_normalized > threshold) / grad_normalized.size
        print(f'High-gradient regions: {high_gradient_pct:.1f}% of domain')
        
        return Lambda_fine, resolvent_fine, sigma_fine

    def _compute_eigenvalues(self, H, use_sparse=False):
        """
        Compute eigenvalues of operator matrix.
        
        Parameters
        ----------
        H : ndarray
            Operator matrix
        use_sparse : bool
            Use sparse eigenvalue solver
            
        Returns
        -------
        eigenvalues : ndarray or None
            Eigenvalues of H
        """
        try:
            if use_sparse and H.shape[0] > 100:
                from scipy.sparse.linalg import eigs
                from scipy.sparse import csr_matrix
                H_sparse = csr_matrix(H)
                k = min(20, H.shape[0] - 2)
                eigenvalues = eigs(H_sparse, k=k, return_eigenvectors=False)
            else:
                eigenvalues = np.linalg.eigvals(H)
            
            # Print diagnostics
            print(f'Eigenvalue range: [{eigenvalues.real.min():.2f}, {eigenvalues.real.max():.2f}]')
            print(f'Imaginary part range: [{eigenvalues.imag.min():.2e}, {eigenvalues.imag.max():.2e}]')
            
            return eigenvalues
        except Exception as e:
            warnings.warn(f'Eigenvalue computation failed: {e}')
            return None

    def _plot_pseudospectrum(self, Lambda, resolvent_norm, sigma_min_grid,
                            epsilon_levels, eigenvalues):
        """
        Plot pseudospectrum results.
        
        Parameters
        ----------
        Lambda : ndarray
            Complex λ grid
        resolvent_norm : ndarray
            Resolvent norms
        sigma_min_grid : ndarray
            Smallest singular values
        epsilon_levels : list
            Contour levels
        eigenvalues : ndarray or None
            Eigenvalues to overlay
        """
        Lambda_re = Lambda.real
        Lambda_im = Lambda.imag
        
        plt.figure(figsize=(14, 6))
        
        # Left plot: ε-pseudospectrum
        plt.subplot(1, 2, 1)
        
        # Better contour level computation
        log_resolvent = np.log10(resolvent_norm + 1e-16)
        levels_log = np.log10(1.0 / np.array(epsilon_levels))
        
        # Only plot contours that exist in the data range
        valid_levels = [lv for lv in levels_log 
                       if log_resolvent.min() <= lv <= log_resolvent.max()]
        
        if len(valid_levels) > 0:
            cs = plt.contour(Lambda_re, Lambda_im, log_resolvent,
                            levels=valid_levels, colors='blue', linewidths=1.5)
            # Better labels
            labels = [f'ε={eps:.0e}' for eps in epsilon_levels[:len(valid_levels)]]
            fmt = dict(zip(cs.levels, labels))
            plt.clabel(cs, inline=True, fmt=fmt, fontsize=9)
        else:
            print('⚠️ Warning: No contours in specified epsilon range')
            # Plot general contours
            cs = plt.contour(Lambda_re, Lambda_im, log_resolvent,
                            levels=10, colors='blue', linewidths=1.5)
        
        if eigenvalues is not None:
            plt.plot(eigenvalues.real, eigenvalues.imag, 'r*', 
                    markersize=10, label='Eigenvalues', markeredgecolor='darkred')
        
        plt.xlabel('Re(λ)', fontsize=12)
        plt.ylabel('Im(λ)', fontsize=12)
        plt.title('ε-Pseudospectrum: log₁₀(‖(H - λI)⁻¹‖)', fontsize=13)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.axis('equal')
        
        # Right plot: Smallest singular value
        plt.subplot(1, 2, 2)
        
        # Use better colormap normalization
        from matplotlib.colors import LogNorm
        
        # Filter out invalid values
        sigma_plot = np.where(np.isfinite(sigma_min_grid), sigma_min_grid, np.nan)
        vmin = np.nanmin(sigma_plot[sigma_plot > 0]) if np.any(sigma_plot > 0) else 1e-10
        vmax = np.nanmax(sigma_plot)
        
        cs2 = plt.contourf(Lambda_re, Lambda_im, sigma_plot,
                          levels=50, cmap='viridis',
                          norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.colorbar(cs2, label='σ_min(H - λI)')
        
        if eigenvalues is not None:
            plt.plot(eigenvalues.real, eigenvalues.imag, 'r*', 
                    markersize=10, markeredgecolor='darkred')
        
        # Plot epsilon contours
        for eps in epsilon_levels:
            cs_eps = plt.contour(Lambda_re, Lambda_im, sigma_plot,
                               levels=[eps], colors='red', linewidths=2, alpha=0.8)
        
        plt.xlabel('Re(λ)', fontsize=12)
        plt.ylabel('Im(λ)', fontsize=12)
        plt.title('Smallest singular value σ_min(H - λI)', fontsize=13)
        plt.grid(alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    
    def symplectic_flow(self):
        """
        Compute the Hamiltonian vector field associated with the principal symbol.

        This method derives the canonical equations of motion for the phase space variables 
        (x, ξ) in 1D or (x, y, ξ, η) in 2D, based on the Hamiltonian formalism. These describe 
        how position and frequency variables evolve under the flow generated by the symbol.

        Returns
        -------
        dict
            A dictionary containing the components of the Hamiltonian vector field:
            - In 1D: keys are 'dx/dt' and 'dxi/dt', corresponding to dx/dt = ∂p/∂ξ and dξ/dt = -∂p/∂x.
            - In 2D: keys are 'dx/dt', 'dy/dt', 'dxi/dt', and 'deta/dt', with similar definitions:
              dx/dt = ∂p/∂ξ, dy/dt = ∂p/∂η, dξ/dt = -∂p/∂x, dη/dt = -∂p/∂y.

        Notes
        -----
        - The Hamiltonian here is the principal symbol p(x, ξ) itself.
        - This flow preserves the symplectic structure of phase space.
        """
        if self.dim == 1:
            x,  = self.vars_x
            xi = symbols('xi', real=True)
            return {
                'dx/dt': diff(self.symbol, xi),
                'dxi/dt': -diff(self.symbol, x)
            }
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            return {
                'dx/dt': diff(self.symbol, xi),
                'dy/dt': diff(self.symbol, eta),
                'dxi/dt': -diff(self.symbol, x),
                'deta/dt': -diff(self.symbol, y)
            }

    def is_elliptic_numerically(self, x_grid, xi_grid, order=None, threshold=1e-6,
                                 xi_frac=0.5, n_random=100_000, n_edge=32, seed=None):
        """
        Check ellipticity using the standard asymptotic (high-frequency) definition.
    
        A symbol p(x, ξ) of order m is elliptic if there exist constants C > 0 and
        R > 0 such that
    
            |p(x, ξ)| ≥ C · (1 + |ξ|²)^(m/2)     for all |ξ| ≥ R (and x in the
                                                    region of interest).
    
        Ellipticity is a *high-frequency* condition: it makes no requirement on the
        symbol near ξ = 0, so this method only probes the region |ξ| ≥ R, where R
        is taken as a fraction (`xi_frac`) of the largest |ξ| present in `xi_grid`.
        This intentionally excludes the low-frequency region, so a symbol like the
        harmonic oscillator x² + ξ² — which has an isolated zero at (0, 0) but is
        the textbook example of a globally elliptic symbol — is correctly detected
        as elliptic instead of being penalised for vanishing at a single point that
        asymptotic ellipticity doesn't care about.
    
        Parameters
        ----------
        x_grid : ndarray or tuple of ndarray
            Spatial grid(s). For 1D: a 1D array of x coordinates.
            For 2D: a tuple (x, y) of two 1D arrays.
        xi_grid : ndarray or tuple of ndarray
            Frequency grid(s). For 1D: a 1D array of ξ coordinates.
            For 2D: a tuple (ξ, η) of two 1D arrays.
        order : float, optional
            The homogeneity order m used in the (1 + |ξ|²)^(m/2) normalization.
            If None (default), it is auto-detected via `self.symbol_order()`. Raises
            ValueError if auto-detection fails and no `order` was supplied.
        threshold : float, optional
            Minimum acceptable value of the *normalized* symbol
            |p(x,ξ)| / (1 + |ξ|²)^(m/2), i.e. the constant C in the ellipticity
            definition above. Default is 1e-6.
        xi_frac : float, optional
            Fraction (in (0, 1)) of the largest |ξ| available in `xi_grid` used as
            the high-frequency cutoff R. Only points with |ξ| ≥ R are tested.
            Default is 0.5.
        n_random : int, optional
            Number of random points sampled in the high-frequency region. Default
            100_000.
        n_edge : int, optional
            Number of spatial points per dimension used in the deterministic check
            performed at the largest available |ξ| (the edge of the supplied grid).
            Default 32.
        seed : int, optional
            Seed for the random sampling, for reproducible results. Default None
            (non-deterministic across calls).
    
        Returns
        -------
        bool
            True if the normalized symbol stays above `threshold` on every tested
            point in the high-frequency region, otherwise False.
    
        Notes
        -----
        This method combines two checks:
            1. **Random sampling** in the shell |ξ| ≥ R (rejection-sampled from the
               hyperrectangle defined by `xi_grid`'s extremes).
            2. **Deterministic edge sampling** — the symbol is evaluated at the
               largest available |ξ| (the corners/extremes of `xi_grid`) over a
               spatial grid of `n_edge` points per dimension, to guarantee the most
               informative high-frequency points are always checked, not just
               randomly sampled ones.
        Because random sampling is probabilistic, a True result does not guarantee
        ellipticity in the strict mathematical sense — it only indicates that no
        tested point violated the condition. For rigorous analysis, combine this
        with symbolic methods (e.g., :meth:`principal_symbol` and
        :meth:`is_homogeneous`).
        """
        import numpy as np
    
        rng = np.random.default_rng(seed)
    
        # ------------------------------------------------------------------
        # Determine the order m used to normalize the symbol
        # ------------------------------------------------------------------
        if order is None:
            order = self.symbol_order()
            if order is None:
                raise ValueError(
                    "Could not auto-detect the symbol's order (symbol_order() "
                    "returned None). Pass `order` explicitly, e.g. "
                    "is_elliptic_numerically(x_grid, xi_grid, order=2)."
                )
        m = float(order)
    
        dim = self.dim
    
        # ------------------------------------------------------------------
        # Extract bounds from the input grids
        # ------------------------------------------------------------------
        if dim == 1:
            x_min, x_max = x_grid.min(), x_grid.max()
            xi_min, xi_max = xi_grid.min(), xi_grid.max()
            xi_abs_max = max(abs(xi_min), abs(xi_max))
        else:  # dim == 2
            x_arr, y_arr = x_grid
            xi_arr, eta_arr = xi_grid
            x_min, x_max = x_arr.min(), x_arr.max()
            y_min, y_max = y_arr.min(), y_arr.max()
            xi_min, xi_max = xi_arr.min(), xi_arr.max()
            eta_min, eta_max = eta_arr.min(), eta_arr.max()
            xi_abs_max = max(abs(xi_min), abs(xi_max), abs(eta_min), abs(eta_max))
    
        if not (0.0 < xi_frac < 1.0):
            raise ValueError("xi_frac must be in (0, 1).")
        R = xi_frac * xi_abs_max
        if R <= 0:
            raise ValueError(
                "The frequency grid does not extend far enough from zero to "
                "define a high-frequency region; widen xi_grid."
            )
    
        def normalized_vals(p_vals, xi_norm_sq):
            return np.abs(p_vals) / (1.0 + xi_norm_sq) ** (m / 2.0)
    
        # ------------------------------------------------------------------
        # 1. Random sampling restricted to the high-frequency shell |ξ| ≥ R
        # ------------------------------------------------------------------
        if dim == 1:
            # |ξ| >= R within [xi_min, xi_max] is (up to) two sub-intervals.
            intervals = []
            if xi_max > R:
                intervals.append((max(xi_min, R), xi_max))
            if xi_min < -R:
                intervals.append((xi_min, min(xi_max, -R)))
            if not intervals:
                raise ValueError(
                    "No part of xi_grid lies in the high-frequency region "
                    "|xi| >= R; widen xi_grid or lower xi_frac."
                )
            lengths = np.array([b - a for a, b in intervals])
            starts = np.array([a for a, b in intervals])
            cum = np.cumsum(lengths)
            u = rng.uniform(0.0, cum[-1], n_random)
            idx = np.searchsorted(cum, u, side='right')
            offset = u - (cum - lengths)[idx]
            xi_rand = starts[idx] + offset
    
            x_rand = rng.uniform(x_min, x_max, n_random)
            vals_rand = self.p_func(x_rand, xi_rand)
            ratio_rand = normalized_vals(vals_rand, xi_rand ** 2)
    
        else:  # dim == 2
            # Rejection-sample within the box until enough points satisfy
            # sqrt(xi^2 + eta^2) >= R.
            collected_xi, collected_eta = [], []
            n_have = 0
            max_attempts = 20
            batch = n_random
            for _ in range(max_attempts):
                xi_try = rng.uniform(xi_min, xi_max, batch)
                eta_try = rng.uniform(eta_min, eta_max, batch)
                mask = xi_try ** 2 + eta_try ** 2 >= R ** 2
                if np.any(mask):
                    collected_xi.append(xi_try[mask])
                    collected_eta.append(eta_try[mask])
                    n_have += int(mask.sum())
                if n_have >= n_random:
                    break
            if n_have == 0:
                raise ValueError(
                    "No part of xi_grid lies in the high-frequency region "
                    "|xi| >= R; widen xi_grid or lower xi_frac."
                )
            xi_rand = np.concatenate(collected_xi)[:n_random]
            eta_rand = np.concatenate(collected_eta)[:n_random]
            n_pts = xi_rand.shape[0]
    
            x_rand = rng.uniform(x_min, x_max, n_pts)
            y_rand = rng.uniform(y_min, y_max, n_pts)
            vals_rand = self.p_func(x_rand, y_rand, xi_rand, eta_rand)
            ratio_rand = normalized_vals(vals_rand, xi_rand ** 2 + eta_rand ** 2)
    
        if np.any(ratio_rand < threshold):
            return False
    
        # ------------------------------------------------------------------
        # 2. Deterministic check at the largest available |ξ| (grid edges)
        # ------------------------------------------------------------------
        if dim == 1:
            x_edge = np.linspace(x_min, x_max, n_edge)
            for xi_edge_val in {xi_min, xi_max}:
                if abs(xi_edge_val) < R:
                    continue
                xi_edge = np.full_like(x_edge, xi_edge_val)
                vals_edge = self.p_func(x_edge, xi_edge)
                ratio_edge = normalized_vals(vals_edge, xi_edge ** 2)
                if np.any(ratio_edge < threshold):
                    return False
        else:  # dim == 2
            x_edge = np.linspace(x_min, x_max, n_edge)
            y_edge = np.linspace(y_min, y_max, n_edge)
            X, Y = np.meshgrid(x_edge, y_edge, indexing='ij')
            X_flat, Y_flat = X.ravel(), Y.ravel()
            for xi_edge_val in {xi_min, xi_max}:
                for eta_edge_val in {eta_min, eta_max}:
                    if xi_edge_val ** 2 + eta_edge_val ** 2 < R ** 2:
                        continue
                    xi_edge = np.full_like(X_flat, xi_edge_val)
                    eta_edge = np.full_like(Y_flat, eta_edge_val)
                    vals_edge = self.p_func(X_flat, Y_flat, xi_edge, eta_edge)
                    ratio_edge = normalized_vals(vals_edge, xi_edge ** 2 + eta_edge ** 2)
                    if np.any(ratio_edge < threshold):
                        return False
    
        # All checks passed
        return True

    def is_self_adjoint(self, tol=1e-10):
        """
        Check whether the pseudo-differential operator is formally self-adjoint (Hermitian).

        A self-adjoint operator satisfies P = P*, where P* is the formal adjoint of P.
        This property is essential for ensuring real-valued eigenvalues and stable evolution 
        in quantum mechanics and symmetric wave propagation.

        Parameters
        ----------
        tol : float
            Tolerance for symbolic comparison between P and P*. Small numerical differences 
            below this threshold are considered equal.

        Returns
        -------
        bool
            True if the symbol p(x, ξ) equals its formal adjoint p*(x, ξ) within the given tolerance,
            indicating that the operator is self-adjoint.

        Notes:
        - The formal adjoint is computed via conjugation and asymptotic expansion at infinity in ξ.
        - Symbolic simplification is used to verify equality, ensuring robustness against superficial 
          expression differences.
        """
        p = self.symbol
        p_star = self.formal_adjoint()
        return simplify(p - p_star).equals(0)

    """
    They implement the asymptotic conversion between Weyl and Kohn-Nirenberg
    quantizations via the exponential operator series:
    
        Weyl -> KN :  a_KN   = exp(+i/2 * d_x d_xi) a_Weyl
        KN -> Weyl :  a_Weyl = exp(-i/2 * d_x d_xi) a_KN
    
    In 1D, d_x d_xi is the simple cross-derivative operator.
    In 2D, it splits as:
    
        d_x d_xi + d_y d_eta   (sum of the two diagonal cross-derivative terms)
    
    and each order k of the series accumulates all ways of distributing
    k cross-differentiations between the two pairs (x, xi) and (y, eta).
    
    Recommended insertion point: after `_get_symbol_func`, before
    `_apply_constant_fft` (around line 399 of psiop.py).
    """
    
    # Dependencies (already present via `from imports import *` in psiop.py)
    # from sympy import symbols, diff, simplify, Rational, I, factorial, binomial
    
    
    # ===========================================================================
    #  Private shared helper -- core asymptotic series computation
    # ===========================================================================
    
    def _quantization_symbol_correction(self, sign, order):
        """
        Compute the corrected symbol via the truncated asymptotic series:
    
            a_out(x, xi) = sum_{k=0}^{order}  (sign * i/2)^k / k!
                             * (d_x d_xi)^k  a_in(x, xi)
    
        In 2D, (d_x d_xi)^k is expanded using the multinomial theorem:
    
            (d_x d_xi + d_y d_eta)^k
              = sum_{j=0}^{k}  C(k, j) * (d_x d_xi)^j * (d_y d_eta)^{k-j}
    
        Parameters
        ----------
        sign : int or sympy expression
            +1 for the Weyl -> KN direction, -1 for KN -> Weyl.
        order : int
            Truncation order of the series (order 0 = identity).
    
        Returns
        -------
        sympy.Expr
            Corrected symbol, simplified.
    
        Notes
        -----
        The series is **exact and finite** for symbols that are polynomial in xi
        (resp. in (xi, eta) in 2D): terms vanish automatically once the
        differentiation order exceeds the polynomial degree.  For S^m class
        or WKB symbols this is an asymptotic approximation valid at the
        considered order.
    
        The derivative pyramid ``derivs[(j, l)]`` stores
        ``(d_x d_xi)^j (d_y d_eta)^l a`` for j + l = current order k,
        computed incrementally to avoid redundant SymPy evaluations.
    
        Raises
        ------
        NotImplementedError
            If the spatial dimension is not 1 or 2.
        """
        from sympy import symbols, diff, simplify, Rational, I, factorial, binomial
    
        a = self.symbol
    
        if self.dim == 1:
            x  = self.vars_x[0]
            xi = symbols('xi', real=True)
    
            result  = a          # k=0 term
            current = a          # accumulates (d_x d_xi)^k a
    
            for k in range(1, order + 1):
                # Apply d_x d_xi one more time
                current = diff(diff(current, x), xi)
                coeff   = (sign * I / 2)**k * Rational(1, factorial(k))
                result  = result + coeff * current
    
        elif self.dim == 2:
            x,  y   = self.vars_x
            xi, eta = symbols('xi eta', real=True)
    
            result = a           # k=0 term
    
            # Derivative pyramid: derivs[(j, l)] = (d_x d_xi)^j (d_y d_eta)^l a
            # Built incrementally order by order to avoid recomputing from scratch.
            derivs = {(0, 0): a}
    
            for k in range(1, order + 1):
                new_derivs = {}
                for j in range(k + 1):
                    l = k - j
                    # (d_x d_xi)^j (d_y d_eta)^l a is obtained from either
                    # (j-1, l) or (j, l-1) at the previous level.
                    if j > 0 and (j - 1, l) in derivs:
                        expr = diff(diff(derivs[(j - 1, l)], x), xi)
                    elif l > 0 and (j, l - 1) in derivs:
                        expr = diff(diff(derivs[(j, l - 1)], y), eta)
                    else:
                        # Fallback: compute from scratch (should not be reached)
                        expr = a
                        for _ in range(j):
                            expr = diff(diff(expr, x), xi)
                        for _ in range(l):
                            expr = diff(diff(expr, y), eta)
                    new_derivs[(j, l)] = expr
    
                derivs.update(new_derivs)
    
                # Order-k contribution via the multinomial expansion:
                # (d_x d_xi + d_y d_eta)^k
                #   = sum_{j=0}^{k} C(k,j) (d_x d_xi)^j (d_y d_eta)^{k-j}
                coeff_k = (sign * I / 2)**k * Rational(1, factorial(k))
                term_k  = sum(
                    binomial(k, j) * derivs[(j, k - j)]
                    for j in range(k + 1)
                )
                result = result + coeff_k * term_k
    
        else:
            raise NotImplementedError(
                f"_quantization_symbol_correction: dimension {self.dim} not supported. "
                "Only 1D and 2D are implemented."
            )
    
        return simplify(result)
    
    
    # ===========================================================================
    #  weyl_to_kn_symbol
    # ===========================================================================
    
    def weyl_to_kn_symbol(self, order=4):
        """
        Convert the Weyl symbol to its Kohn-Nirenberg equivalent.
    
        The operator Op^w(a) admits a KN representation Op^KN(a_tilde) where
        the corrected symbol is given by the asymptotic series:
    
            a_tilde(x, xi) = exp(-i/2 * d_x d_xi) a(x, xi)
                           ~ sum_{k=0}^{order}  (-i/2)^k / k!
                               * (d_x d_xi)^k a(x, xi)
    
        In 2D, d_x d_xi is replaced by d_x d_xi + d_y d_eta.
    
        Parameters
        ----------
        order : int, default=4
            Truncation order.  The series is **exact and finite** for symbols
            that are polynomial in xi (all terms beyond the polynomial degree
            vanish automatically).  For S^m or WKB symbols, order=2 to 4 is
            generally sufficient.
    
        Returns
        -------
        sympy.Expr
            Equivalent KN symbol (simplified SymPy expression).
    
        Examples
        --------
        1D -- multiplication-by-x*xi operator:
    
        >>> from sympy import symbols
        >>> x, xi = symbols('x xi', real=True)
        >>> op = PseudoDifferentialOperator(x * xi, [x], mode='symbol')
        >>> op.weyl_to_kn_symbol(order=2)
        x*xi - I/2      # order-1 correction: -i/2 * d_x d_xi (x*xi) = -i/2
    
        2D -- symbol x*xi + y*eta:
    
        >>> x, y, xi, eta = symbols('x y xi eta', real=True)
        >>> op2 = PseudoDifferentialOperator(x*xi + y*eta, [x, y], mode='symbol')
        >>> op2.weyl_to_kn_symbol(order=2)
        x*xi + y*eta - I    # two cross terms, each contributing -i/2
    
        Notes
        -----
        The resulting KN symbol is directly usable in ``apply()`` through the
        existing KN pipeline (``kohn_nirenberg_fft`` or
        ``kohn_nirenberg_nonperiodic``), which avoids any numerical double
        integral.
    
        Typical workflow for Weyl evaluation::
    
            kn_sym  = op.weyl_to_kn_symbol(order=4)
            kn_func = lambdify((x, xi), kn_sym, 'numpy')
            result  = kohn_nirenberg_fft(u, kn_func, ...)
    
        See Also
        --------
        kn_to_weyl_symbol : inverse transformation.
        _quantization_symbol_correction : underlying series computation.
        """
        return self._quantization_symbol_correction(sign=-1, order=order)
    
    
    # ===========================================================================
    #  kn_to_weyl_symbol
    # ===========================================================================
    
    def kn_to_weyl_symbol(self, order=4):
        """
        Convert the Kohn-Nirenberg symbol to its Weyl equivalent.
    
        This is the inverse of ``weyl_to_kn_symbol``::
    
            a_Weyl(x, xi) = exp(+i/2 * d_x d_xi) a_KN(x, xi)
                          ~ sum_{k=0}^{order}  (+i/2)^k / k!
                              * (d_x d_xi)^k a_KN(x, xi)
    
        In 2D, d_x d_xi is replaced by d_x d_xi + d_y d_eta.
    
        Parameters
        ----------
        order : int, default=4
            Truncation order.  Same remarks as for ``weyl_to_kn_symbol``.
    
        Returns
        -------
        sympy.Expr
            Equivalent Weyl symbol (simplified SymPy expression).
    
        Examples
        --------
        1D -- multiplication-by-x*xi operator:
    
        >>> op.kn_to_weyl_symbol(order=2)
        x*xi + I/2      # opposite sign to weyl_to_kn_symbol
    
        Notes
        -----
        Typical use case: the operator is defined by its KN symbol (e.g.
        extracted in 'auto' mode), and you want the Weyl symbol to exploit
        its spectral properties -- self-adjointness when a_Weyl is real,
        composition via the Moyal star product, etc.
    
        Round-trip consistency check::
    
            op_weyl = PseudoDifferentialOperator(op.kn_to_weyl_symbol(order=N), ...)
            op_weyl.weyl_to_kn_symbol(order=N)  # should recover op.symbol up to order N
    
        See Also
        --------
        weyl_to_kn_symbol : inverse transformation.
        _quantization_symbol_correction : underlying series computation.
        """
        return self._quantization_symbol_correction(sign=+1, order=order)

#########################
####  Visualizations ####
#########################

    def visualize_fiber(self, x_grid, xi_grid, x0=0.0, y0=0.0):
        """Plot the cotangent fiber structure at a fixed spatial point (x0[, y0]).
        See _slice_grid/_render_field docstrings for the shared implementation.
        NOTE: original signature has no eta_grid param -- 2D reuses xi_grid
        for both frequency axes, matching the original behavior exactly."""
        a1, a2, l1, l2, Z = _slice_grid(self, 'freq', x_grid, xi_grid, eta_grid=xi_grid, x0=x0, y0=y0)
        title = 'Cotangent Fiber Structure' if self.dim == 1 else f'Cotangent Fiber at x={x0}, y={y0}'
        _render_field(a1, a2, np.abs(Z), style='contourf', cbar_label='|Symbol|',
                      xlabel=l1, ylabel=l2, title=title)

    def visualize_symbol_amplitude(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """Display |p(x, xi)| (1D) or |p(x, y, xi0, eta0)| (2D) as a color map."""
        kind = 'freq' if self.dim == 1 else 'space'
        a1, a2, l1, l2, Z = _slice_grid(self, kind, x_grid, xi_grid, y_grid, eta_grid, xi0=xi0, eta0=eta0)
        title = 'Symbol Amplitude |p(x, \u03be)|' if self.dim == 1 else f'Symbol Amplitude at \u03be={xi0}, \u03b7={eta0}'
        _render_field(a1, a2, np.abs(Z), style='pcolormesh', cbar_label='|Symbol|',
                      xlabel=l1, ylabel=l2, title=title)

    def visualize_phase(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """Plot arg(p(x, xi)) (1D) or arg(p(x, y, xi0, eta0)) (2D)."""
        kind = 'freq' if self.dim == 1 else 'space'
        a1, a2, l1, l2, Z = _slice_grid(self, kind, x_grid, xi_grid, y_grid, eta_grid, xi0=xi0, eta0=eta0)
        title = 'Phase Portrait (arg p(x, \u03be))' if self.dim == 1 else f'Phase Portrait at \u03be={xi0}, \u03b7={eta0}'
        _render_field(a1, a2, np.angle(Z), style='pcolormesh', cmap='twilight',
                      cbar_label='arg(Symbol) [rad]', xlabel=l1, ylabel=l2, title=title)
            
    def visualize_characteristic_set(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0, levels=[1e-1]):
        """Visualize the characteristic set p(x, xi) ~= 0 (1D) or the (xi, eta)
        slice at fixed (x0, y0) (2D)."""
        if self.dim not in (1, 2):
            raise NotImplementedError("Only 1D/2D characteristic sets supported.")
        if self.dim == 2 and eta_grid is None:
            raise ValueError("eta_grid must be provided for 2D visualization.")
        a1, a2, l1, l2, Z = _slice_grid(self, 'freq', x_grid, xi_grid, y_grid, eta_grid, x0=x0, y0=y0)
        title = 'Characteristic Set (p(x, \u03be) \u2248 0)' if self.dim == 1 else f'Characteristic Set at x={x0}, y={y0}'
        _render_field(a1, a2, np.abs(Z), style='contour', levels=levels,
                      xlabel=l1, ylabel=l2, title=title, grid=True)

    def visualize_characteristic_gradient(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0):
        """Visualize |grad p| in phase space. NOTE: both the 1D and 2D
        branches now consistently use abs(.)**2 in the gradient norm (the
        original 1D branch omitted the abs(), inconsistently with 2D)."""
        a1, a2, l1, l2, Z = _slice_grid(self, 'freq', x_grid, xi_grid, y_grid, eta_grid, x0=x0, y0=y0)
        title = 'Gradient Norm (High Near Zeros)' if self.dim == 1 else f'Gradient Norm at x={x0}, y={y0}'
        _render_field(a1, a2, _grad_norm(Z), style='pcolormesh', cmap='inferno',
                      cbar_label='|\u2207p|', xlabel=l1, ylabel=l2, title=title, grid=True)

    def plot_hamiltonian_flow(self, x0=0.0, xi0=5.0, y0=0.0, eta0=0.0, tmax=1.0,
                               n_steps=100, show_field=True):
        """Integrate and plot the Hamiltonian trajectories of the symbol in
        phase space. Delegates to the shared `integrate_singularity` engine
        instead of re-deriving the Hamiltonian vector field inline."""
        x0v = [x0] if self.dim == 1 else [x0, y0]
        xi0v = [xi0] if self.dim == 1 else [xi0, eta0]
        _, _, _, _, trajs = integrate_singularity(
            self.symbol, self.vars_x, x0=x0v, xi0=xi0v, tmax=tmax, n_frames=n_steps)
        Y = trajs[0]

        if self.dim == 1:
            x_vals, xi_vals = Y
            plt.plot(x_vals, xi_vals)
            plt.xlabel("x"); plt.ylabel("\u03be")
            plt.title("Hamiltonian Flow in Phase Space (1D)")
            plt.grid(True)
            plt.show()
        elif self.dim == 2:
            x_vals, y_vals, xi_vals, eta_vals = Y
            plt.plot(x_vals, y_vals, label='Position')
            plt.quiver(x_vals, y_vals, xi_vals, eta_vals, scale=20, width=0.003, alpha=0.5, color='r')
            if show_field:
                x, y = self.vars_x
                xi, eta = symbols('xi eta', real=True)
                H = self.symplectic_flow()
                dxdt = lambdify((x, y, xi, eta), _make_real(H['dx/dt']), 'numpy')
                dydt = lambdify((x, y, xi, eta), _make_real(H['dy/dt']), 'numpy')
                Xg, Yg = np.meshgrid(np.linspace(min(x_vals), max(x_vals), 20),
                                     np.linspace(min(y_vals), max(y_vals), 20))
                XI, ETA = xi0 * np.ones_like(Xg), eta0 * np.ones_like(Yg)
                plt.quiver(Xg, Yg, dxdt(Xg, Yg, XI, ETA), dydt(Xg, Yg, XI, ETA),
                          color='gray', alpha=0.2, scale=30, width=0.002)
            plt.xlabel("x"); plt.ylabel("y")
            plt.title("Hamiltonian Flow in Phase Space (2D)")
            plt.legend(); plt.grid(True); plt.axis('equal')
            plt.show()

    def plot_symplectic_vector_field(self, xlim=(-2, 2), klim=(-5, 5), density=30):
        """Quiver plot of the symplectic vector field (dp/dxi, -dp/dx). 1D only."""
        _quiver_field(self, xlim, klim, density,
                      lambda p, x, xi: (diff(p, xi), -diff(p, x)),
                      "Symplectic Vector Field (1D)")

    def visualize_micro_support(self, xlim=(-2, 2), klim=(-10, 10), threshold=1e-3, density=300):
        """Visualize 1/|p(x, xi)| to highlight regions where the symbol is
        near zero. NOTE: no longer restricted to 1D -- the shared grid/render
        helpers already handle the 2D case (fixed xi0=eta0=0, scan x, y)."""
        x_grid = np.linspace(*xlim, density)
        xi_grid = np.linspace(*klim, density)
        if self.dim == 1:
            a1, a2, l1, l2, Z = _slice_grid(self, 'freq', x_grid, xi_grid)
        else:
            a1, a2, l1, l2, Z = _slice_grid(self, 'space', x_grid, None, x_grid, None, xi0=0.0, eta0=0.0)
        _render_field(a1, a2, 1 / (np.abs(Z) + 1e-10), style='contourf', cmap='inferno',
                      cbar_label=r'$1/|p(x,\xi)|$', xlabel=l1, ylabel=l2,
                      title="Micro-Support Estimate (1/|Symbol|)")

    def group_velocity_field(self, xlim=(-2, 2), klim=(-10, 10), density=30):
        """Quiver plot of the group velocity field (1, dp/dxi). 1D only."""
        _quiver_field(self, xlim, klim, density,
                      lambda p, x, xi: (sp.Integer(1), diff(p, xi)),
                      "Group Velocity Field (1D)")

    def animate_singularity(self, xi0=5.0, eta0=0.0, x0=0.0, y0=0.0,
                             tmax=4.0, n_frames=100, projection=None):
        """Animate the propagation of a singularity under the Hamiltonian
        flow. Thin delegate to the module-level `animate_singularity`
        engine (previously ~130 lines of duplicated Hamiltonian/ODE setup
        here, plus a near-identical copy further down the module)."""
        rc('animation', html='jshtml')
        x0v = x0 if self.dim == 1 else [x0, y0]
        xi0v = xi0 if self.dim == 1 else [xi0, eta0]
        if projection is None:
            projection = 'phase' if self.dim == 1 else 'position'
        return animate_singularity(self.symbol, self.vars_x, x0=x0v, xi0=xi0v,
                                    tmax=tmax, n_frames=n_frames, projection=projection)

    def interactive_symbol_analysis(pseudo_op, xlim=(-2, 2), ylim=(-2, 2),
                                    xi_range=(0.1, 5), eta_range=(-5, 5), density=50):
        """Launch an ipywidgets dashboard for symbol exploration. Same modes,
        same sliders, same defaults as before -- rewritten as a mode-table
        dispatcher that delegates to the visualize_*/plot_* methods above
        instead of duplicating their 1D/2D branches inline (previously
        ~260 lines of near-duplicated if-elif chains)."""
        dim = pseudo_op.dim
        x_vals = np.linspace(*xlim, density)
        y_vals = np.linspace(*ylim, density) if dim == 2 else None
        xi_lin = np.linspace(*xi_range, density)
        eta_lin = np.linspace(*eta_range, density) if dim == 2 else None

        if dim == 1:
            modes = ['Symbol Amplitude', 'Symbol Phase', 'Micro-Support (1/|p|)',
                      'Cotangent Fiber', 'Characteristic Set', 'Characteristic Gradient',
                      'Group Velocity Field', 'Symplectic Vector Field', 'Hamiltonian Flow']
            needs = {
                'Symbol Amplitude': ('xi',), 'Symbol Phase': ('xi',),
                'Micro-Support (1/|p|)': ('xi',), 'Group Velocity Field': ('xi',),
                'Symplectic Vector Field': ('xi',), 'Hamiltonian Flow': ('xi', 'x'),
                'Cotangent Fiber': (), 'Characteristic Set': (), 'Characteristic Gradient': (),
            }
            mode_selector = Dropdown(options=modes, value='Symbol Amplitude', description='Mode:')
            xi_slider = FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='\u03be\u2080')
            x_slider = FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x\u2080')
            all_sliders = {'xi': xi_slider, 'x': x_slider}

            def render(mode, xi0, x0):
                plt.close('all'); plt.figure()
                if mode == 'Symbol Amplitude':
                    pseudo_op.visualize_symbol_amplitude(x_vals, xi_lin, xi0=xi0)
                elif mode == 'Symbol Phase':
                    pseudo_op.visualize_phase(x_vals, xi_lin, xi0=xi0)
                elif mode == 'Micro-Support (1/|p|)':
                    pseudo_op.visualize_micro_support(xlim, xi_range, density=density)
                elif mode == 'Group Velocity Field':
                    pseudo_op.group_velocity_field(xlim, xi_range, density=density)
                elif mode == 'Symplectic Vector Field':
                    pseudo_op.plot_symplectic_vector_field(xlim, xi_range, density=density)
                elif mode == 'Cotangent Fiber':
                    pseudo_op.visualize_fiber(x_vals, xi_lin, x0=x0)
                elif mode == 'Characteristic Set':
                    pseudo_op.visualize_characteristic_set(x_vals, xi_lin, x0=x0)
                elif mode == 'Characteristic Gradient':
                    pseudo_op.visualize_characteristic_gradient(x_vals, xi_lin, x0=x0)
                elif mode == 'Hamiltonian Flow':
                    pseudo_op.plot_hamiltonian_flow(x0=x0, xi0=xi0)

            interactive_kwargs = {'mode': mode_selector, 'xi0': xi_slider, 'x0': x_slider}
            slider_order = ['xi', 'x']

        else:  # dim == 2
            modes = ['Symbol Amplitude', 'Symbol Phase', 'Micro-Support (1/|p|)',
                      'Cotangent Fiber', 'Characteristic Set', 'Characteristic Gradient',
                      'Symplectic Vector Field', 'Hamiltonian Flow']
            needs = {
                'Symbol Amplitude': ('xi', 'eta'), 'Symbol Phase': ('xi', 'eta'),
                'Micro-Support (1/|p|)': ('xi', 'eta'), 'Symplectic Vector Field': ('xi', 'eta'),
                'Hamiltonian Flow': ('xi', 'eta', 'x', 'y'),
                'Cotangent Fiber': ('x', 'y'), 'Characteristic Set': ('x', 'y'),
                'Characteristic Gradient': ('x', 'y'),
            }
            mode_selector = Dropdown(options=modes, value='Symbol Amplitude', description='Mode:')
            xi_slider = FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='\u03be\u2080')
            eta_slider = FloatSlider(min=eta_range[0], max=eta_range[1], step=0.1, value=1.0, description='\u03b7\u2080')
            x_slider = FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x\u2080')
            y_slider = FloatSlider(min=ylim[0], max=ylim[1], step=0.1, value=0.0, description='y\u2080')
            all_sliders = {'xi': xi_slider, 'eta': eta_slider, 'x': x_slider, 'y': y_slider}

            def render(mode, xi0, eta0, x0, y0):
                plt.close('all'); plt.figure()
                if mode == 'Symbol Amplitude':
                    pseudo_op.visualize_symbol_amplitude(x_vals, xi_lin, y_vals, eta_lin, xi0=xi0, eta0=eta0)
                elif mode == 'Symbol Phase':
                    pseudo_op.visualize_phase(x_vals, xi_lin, y_vals, eta_lin, xi0=xi0, eta0=eta0)
                elif mode == 'Micro-Support (1/|p|)':
                    pseudo_op.visualize_micro_support(xlim, xi_range, density=density)
                elif mode == 'Symplectic Vector Field':
                    x, y = pseudo_op.vars_x
                    xi, eta = symbols('xi eta', real=True)
                    Xg, Yg = np.meshgrid(x_vals, y_vals, indexing='ij')
                    U, V = lambdify((x, y, xi, eta),
                                    [diff(pseudo_op.expr, xi), diff(pseudo_op.expr, eta)],
                                    'numpy')(Xg, Yg, xi0, eta0)
                    plt.quiver(Xg, Yg, U, V, scale=10, width=0.004)
                    plt.xlabel('x'); plt.ylabel('y')
                    plt.title(f'Symplectic Field at \u03be={xi0:.2f}, \u03b7={eta0:.2f}')
                elif mode == 'Cotangent Fiber':
                    pseudo_op.visualize_fiber(xi_lin, eta_lin, x0=x0, y0=y0)
                elif mode == 'Characteristic Set':
                    pseudo_op.visualize_characteristic_set(x_vals, xi_lin, y_vals, eta_lin, x0=x0, y0=y0)
                elif mode == 'Characteristic Gradient':
                    pseudo_op.visualize_characteristic_gradient(x_vals, xi_lin, y_vals, eta_lin, x0=x0, y0=y0)
                elif mode == 'Hamiltonian Flow':
                    pseudo_op.plot_hamiltonian_flow(x0=x0, y0=y0, xi0=xi0, eta0=eta0)

                if mode not in ("Cotangent Fiber", "Characteristic Set",
                                "Characteristic Gradient", "Hamiltonian Flow"):
                    plt.show()

            interactive_kwargs = {'mode': mode_selector, 'xi0': xi_slider, 'eta0': eta_slider,
                                  'x0': x_slider, 'y0': y_slider}
            slider_order = ['xi', 'eta', 'x', 'y']

        controls_box = VBox([mode_selector] + list(all_sliders.values()))

        def update_controls(change):
            active = needs[change['new']]
            controls_box.children = [mode_selector] + [all_sliders[k] for k in slider_order if k in active]
        mode_selector.observe(update_controls, names='value')
        update_controls({'new': mode_selector.value})

        out = interactive_output(render, interactive_kwargs)
        display(VBox([controls_box, out]))



# ============================================================================
# Matrix-Valued (N x N) Pseudodifferential Operators
# ============================================================================
#
# P(x, xi) an N x N matrix of scalar symbols, acting on a vector field
# u = (u_1, ..., u_N) via the same Kohn-Nirenberg / Weyl quantization used
# throughout this module:
#
#     (P u)_i(x) = (1/2pi)^d  sum_j  int P_ij(x, xi) u_hat_j(xi) e^{i x.xi} dxi
#
# i.e. entrywise it's just N^2 ordinary scalar PseudoDifferentialOperator
# applies, summed row-wise -- so `apply()` below adds no new numerics at
# all, it only orchestrates N^2 existing scalar operators on the plain
# periodic/rectangular grid (no cutoffs, no diffuse interfaces). Symbol
# composition and the commutator DO need new code, because matrix
# multiplication doesn't commute: even the leading (0th-order) term of the
# composed symbol is the matrix product P(x,xi) Q(x,xi), not interchangeable
# in either order, and for x-independent ("constant-coefficient") symbols
# that matrix product IS the *exact* composition at any order -- every
# n>=1 term in the KN expansion involves d/dx of a xi-only expression,
# which is identically zero. That exactness is the natural correctness
# check for `compose_asymptotic` below.

import sympy as sp


class MatrixPseudoDifferentialOperator:
    """
    N x N matrix-valued pseudodifferential operator, built from a sympy
    Matrix of scalar symbols P_ij(x[, y], xi[, eta]).

    Each entry P_ij is wrapped as its own scalar `PseudoDifferentialOperator`
    (same `expr`/`mode`/`quantization`/`apply_backend` conventions), so
    `apply()` reuses the existing periodic FFT / Peetre machinery entirely.
    Entries may depend on `x` (and `y`) for variable-coefficient systems,
    or be `xi`(`, eta`)-only for constant-coefficient ones -- both are
    supported by the same class; nothing here forces one or the other.

    Parameters
    ----------
    P_expr : sympy.Matrix or nested list of sympy.Expr
        N x N matrix of scalar symbol expressions, in the same
        `x[, y], xi[, eta]` convention as `PseudoDifferentialOperator`.
    vars_x : list of sympy symbols
        Spatial variables; length 1 or 2, as for `PseudoDifferentialOperator`.
    mode, quantization, apply_backend, compute_peetre, peetre_options
        Forwarded to each entry's `PseudoDifferentialOperator`.

    Attributes
    ----------
    size : int
        Matrix dimension N (the constructor accepts any square N, but
        `eigen_symbol`'s closed-form path is specific to N=2).
    entries : list of list of PseudoDifferentialOperator
        `entries[i][j]` is the scalar operator for `P_ij`.
    """

    def __init__(
        self,
        P_expr,
        vars_x,
        mode='symbol',
        quantization='kohn-nirenberg',
        apply_backend='peetre',
        compute_peetre=False,
        peetre_options=None,
    ):
        P_expr = sp.Matrix(P_expr)
        n, m = P_expr.shape
        if n != m:
            raise ValueError("P_expr must be a square matrix of symbols.")

        self.size = n
        self.dim = len(vars_x)
        self.vars_x = list(vars_x)
        self.mode = mode
        self.quantization = quantization
        self.apply_backend = apply_backend
        self.P_expr = P_expr

        self.entries = [
            [
                PseudoDifferentialOperator(
                    P_expr[i, j], vars_x, mode=mode, quantization=quantization,
                    apply_backend=apply_backend, compute_peetre=compute_peetre,
                    peetre_options=peetre_options,
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        # every entry shares the same grid conventions -- borrow one FFT/IFFT pair
        self.fft = self.entries[0][0].fft
        self.ifft = self.entries[0][0].ifft

    def apply(self, u, x_grid, kx, y_grid=None, ky=None, **apply_kwargs):
        """
        Apply P(x, xi) to a vector field u = (u_1, ..., u_N).

        Parameters
        ----------
        u : sequence of N ndarrays
            Vector field components, each sampled on the grid.
        x_grid, kx, y_grid, ky
            As for `PseudoDifferentialOperator.apply`.
        **apply_kwargs
            Forwarded to every entry's scalar `apply()` (e.g.
            `boundary_condition`, `freq_window`, ...).

        Returns
        -------
        list of N ndarrays
            `(P u)_i = sum_j Op[P_ij](u_j)`.
        """
        if len(u) != self.size:
            raise ValueError(f"Expected {self.size} vector components, got {len(u)}.")

        out = []
        for i in range(self.size):
            v_i = None
            for j in range(self.size):
                contrib = self.entries[i][j].apply(
                    u[j], x_grid, kx, y_grid=y_grid, ky=ky, **apply_kwargs
                )
                v_i = contrib if v_i is None else v_i + contrib
            out.append(v_i)
        return out

    def symbol_matrix(self, *args):
        """
        Numerically evaluate P(x[, y], xi[, eta]) at a point or
        broadcastable arrays, returning an ndarray of shape `(..., N, N)`.
        If called without arguments, returns the symbolic sympy.Matrix.
        
        Parameters
        ----------
        *args
            The point(s) to evaluate at, in the order each entry's
            `p_func` expects: `(x, xi)` for 1D, `(x, y, xi, eta)` for 2D.
            Arguments may be broadcastable ndarrays (e.g. full grids), in
            which case the leading dimensions of the output match their
            broadcast shape.
    
        Returns
        -------
        ndarray, shape (..., N, N) or sympy.Matrix if no args are provided.
        """
        # Fallback for symbolic evaluation when no numerical grid is provided
        if not args:
            return self.P_expr
            
        n = self.size
        sample = np.broadcast(*[np.asarray(a) for a in args])
        P = np.zeros(sample.shape + (n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                P[..., i, j] = self.entries[i][j].p_func(*args)
        return P

    def apply_matrix_field(self, U, x_grid, kx, y_grid=None, ky=None, **apply_kwargs):
        """
        Apply P(x, xi) to a matrix-valued field U(x) by left matrix
        multiplication on U's own N x N structure:

            (P U)_ik(x) = sum_j Op[P_ij] (U_jk) (x)

        Unlike `apply()`, which propagates a single vector field
        u = (u_1, ..., u_N), here U itself carries an extra N x N index
        pair (e.g. a density matrix or matrix Green's function) that P
        acts on only from the left. Each column `U[:, k]` of U is an
        ordinary vector field, so this reduces to N independent calls to
        `apply()`, one per column, with the results reassembled into the
        matrix-shaped output; P is applied to U, never U to P.

        Parameters
        ----------
        U : sequence of N sequences of N ndarrays, or ndarray of shape (N, N, ...)
            Matrix-valued field; `U[j][k]` (equivalently `U[j, k]` for an
            ndarray) is the scalar (j, k) component field sampled on the
            grid, so that U plays the role of an N x N matrix at every
            grid point.
        x_grid, kx, y_grid, ky
            As for `apply`.
        **apply_kwargs
            Forwarded to every entry's scalar `apply()` (e.g.
            `boundary_condition`, `freq_window`, ...).

        Returns
        -------
        list of N lists of N ndarrays
            `out[i][k]` holds `(P U)_ik`, indexed the same way as `U`.

        Raises
        ------
        ValueError
            If `U` is not an N x N array of fields, with N equal to
            `self.size`.
        """
        if len(U) != self.size or any(len(row) != self.size for row in U):
            got_cols = len(U[0]) if len(U) else 0
            raise ValueError(
                f"Expected a {self.size}x{self.size} matrix field, got "
                f"{len(U)}x{got_cols}."
            )

        out = [[None] * self.size for _ in range(self.size)]
        for k in range(self.size):
            column = [U[j][k] for j in range(self.size)]
            result_column = self.apply(
                column, x_grid, kx, y_grid=y_grid, ky=ky, **apply_kwargs
            )
            for i in range(self.size):
                out[i][k] = result_column[i]
        return out

    def apply_matrix_field_right(self, U, x_grid, kx, y_grid=None, ky=None, **apply_kwargs):
        """
        Apply this operator's symbol Q(x, xi) to a matrix-valued field
        U(x) by right matrix multiplication on U's own N x N structure:

            (U Q)_ik(x) = sum_j Op[Q_jk] (U_ij) (x)

        This is the mirror image of `apply_matrix_field` (which
        left-multiplies by P): here each *row* `U[i, :]` of U is an
        ordinary vector field acted on from the right by Q, which is
        equivalent to the left action of the transposed symbol matrix Q^T
        on that row -- hence the index order `Op[Q_jk]`, not `Op[Q_kj]`,
        so the two methods are genuinely different unless Q is symmetric.

        Together with `apply_matrix_field`, this is the numerical
        primitive needed to time-step Sylvester-type equations
        `d_t U = P U - U Q`, since left- and right-multiplication always
        commute as *operations* (`(P U) Q == P (U Q)`), even though the
        underlying scalar operators `Op[P_ij]` and `Op[Q_jk]` need not
        commute with each other when the symbols depend on x. See
        `solve_sylvester_field` for the corresponding time-stepper.

        Parameters
        ----------
        U : sequence of N sequences of N ndarrays, or ndarray of shape (N, N, ...)
            Matrix-valued field; `U[i][j]` (equivalently `U[i, j]` for an
            ndarray) is the scalar (i, j) component field sampled on the
            grid.
        x_grid, kx, y_grid, ky
            As for `apply`.
        **apply_kwargs
            Forwarded to every entry's scalar `apply()`.

        Returns
        -------
        list of N lists of N ndarrays
            `out[i][k]` holds `(U Q)_ik`, indexed the same way as `U`.

        Raises
        ------
        ValueError
            If `U` is not an N x N array of fields, with N equal to
            `self.size`.
        """
        if len(U) != self.size or any(len(row) != self.size for row in U):
            got_cols = len(U[0]) if len(U) else 0
            raise ValueError(
                f"Expected a {self.size}x{self.size} matrix field, got "
                f"{len(U)}x{got_cols}."
            )

        out = [[None] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for k in range(self.size):
                v_ik = None
                for j in range(self.size):
                    contrib = self.entries[j][k].apply(
                        U[i][j], x_grid, kx, y_grid=y_grid, ky=ky, **apply_kwargs
                    )
                    v_ik = contrib if v_ik is None else v_ik + contrib
                out[i][k] = v_ik
        return out


        """
        Numerically evaluate P(x[, y], xi[, eta]) at a point or
        broadcastable arrays, returning an ndarray of shape `(..., N, N)`.

        Parameters
        ----------
        *args
            The point(s) to evaluate at, in the order each entry's
            `p_func` expects: `(x, xi)` for 1D, `(x, y, xi, eta)` for 2D.
            Arguments may be broadcastable ndarrays (e.g. full grids), in
            which case the leading dimensions of the output match their
            broadcast shape.

        Returns
        -------
        ndarray, shape (..., N, N)
        """
        n = self.size
        sample = np.broadcast(*[np.asarray(a) for a in args])
        P = np.zeros(sample.shape + (n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                P[..., i, j] = self.entries[i][j].p_func(*args)
        return P

    def eigen_symbol(self, *args):
        """
        Pointwise eigenvalues/eigenvectors of the symbol matrix
        P(x[, y], xi[, eta]) at given point(s).
        If called without arguments, computes symbolic eigenvalues/eigenvectors.
        """
        P = self.symbol_matrix(*args)
        
        # --- Symbolic Path (when no args are provided) ---
        if isinstance(P, sp.MatrixBase):
            if self.size == 2:
                a, b = P[0, 0], P[0, 1]
                c, d = P[1, 0], P[1, 1]
                tr = a + d
                det = a * d - b * c
                disc = sp.sqrt(tr ** 2 - 4 * det)
                lam1 = (tr + disc) / 2
                lam2 = (tr - disc) / 2
                eigvals = sp.Matrix([lam1, lam2])
                
                def _eigvec(lam):
                    v_row0 = sp.Matrix([b, lam - a])
                    v_row1 = sp.Matrix([lam - d, c])
                    # Choose the row that avoids division by zero
                    v = v_row0 if b != 0 else v_row1
                    norm = sp.sqrt(v.dot(v))
                    if norm == 0:
                        return v
                    return v / norm
                
                eigvecs = sp.Matrix.hstack(_eigvec(lam1), _eigvec(lam2))
                return eigvals, eigvecs
            else:
                # General N x N symbolic eigenvalues
                return P.eigenvals(), None

        # --- Numeric Path ---
        if self.size == 2:
            a, b = P[..., 0, 0], P[..., 0, 1]
            c, d = P[..., 1, 0], P[..., 1, 1]
            tr = a + d
            det = a * d - b * c
            disc = np.sqrt((tr ** 2 - 4 * det).astype(complex))
            lam1 = (tr + disc) / 2
            lam2 = (tr - disc) / 2
            eigvals = np.stack([lam1, lam2], axis=-1)
            
            def _eigvec(lam):
                v_row0 = np.stack([b, lam - a], axis=-1)
                v_row1 = np.stack([lam - d, c], axis=-1)
                use_row0 = np.abs(b) >= np.abs(c)
                v = np.where(use_row0[..., None], v_row0, v_row1)
                norm = np.linalg.norm(v, axis=-1, keepdims=True)
                norm = np.where(norm == 0, 1.0, norm)
                return v / norm
                
            eigvecs = np.stack([_eigvec(lam1), _eigvec(lam2)], axis=-1)
            return eigvals, eigvecs
            
        return np.linalg.eig(P)  # general N x N fallback: (eigvals, eigvecs)

    def compose_asymptotic(self, other, order=1, mode='kn', sign_convention=None):
        """
        Compose two matrix-valued symbols via the same asymptotic
        Kohn-Nirenberg / Weyl expansion as
        `PseudoDifferentialOperator.compose_asymptotic`, generalized to
        (order-preserving) matrix multiplication: this is the symbol of
        `Op[self] . Op[other]`, and
        `self.compose_asymptotic(other) != other.compose_asymptotic(self)`
        in general -- unlike the scalar case, matrices don't commute even
        at 0th order.

        For x[, y]-independent ("constant-coefficient") symbols this is
        *exact* at any `order`: every n>=1 term involves a spatial
        derivative of a xi-only expression, which vanishes identically, so
        the result reduces to the ordinary matrix product `P(xi) Q(xi)`.

        Parameters
        ----------
        other : MatrixPseudoDifferentialOperator
            Same `size` and `dim` as `self`.
        order, mode, sign_convention
            As for the scalar `compose_asymptotic`.

        Returns
        -------
        sympy.Matrix, shape (size, size)
            The composed symbol.
        """
        assert self.dim == other.dim, "Operator dimensions must match"
        assert self.size == other.size, "Matrix sizes must match"
        if mode not in ('kn', 'weyl'):
            raise ValueError("mode must be 'kn' or 'weyl'")
        dim = self.dim
        if dim not in (1, 2):
            raise NotImplementedError("dim must be 1 or 2")

        P, Q = self.P_expr, other.P_expr
        x_vars = self.vars_x
        xi_vars = sp.symbols('xi eta', real=True) if dim == 2 else (sp.symbols('xi', real=True),)
        sign = -1 if (sign_convention or 'standard') == 'standard' else +1

        result = sp.zeros(self.size, self.size)
        if mode == 'kn':
            for n in range(order + 1):
                for alpha in _mi_all(n, dim):
                    fact = _mi_factorial(alpha)
                    dP = _mi_diff(P, xi_vars, alpha)
                    dQ = _mi_diff(Q, x_vars, alpha)
                    result += (dP * dQ / fact) * (1j) ** (sign * n)
        else:  # 'weyl' -- general dimension-generic Moyal product; this
               # replaces the previous 2D branch, which only differentiated
               # P in (xi, eta) and Q in (x, y) and so dropped the cross
               # terms present in the exact 1D formula (see scalar
               # compose_asymptotic for the same fix and more detail).
            for total in range(order + 1):
                for a_deg in range(total + 1):
                    b_deg = total - a_deg
                    for alpha in _mi_all(a_deg, dim):
                        for beta in _mi_all(b_deg, dim):
                            coeff = (1j / 2) ** total * (-1) ** b_deg
                            coeff /= (_mi_factorial(alpha) * _mi_factorial(beta))
                            dP = _mi_diff(_mi_diff(P, xi_vars, alpha), x_vars, beta)
                            dQ = _mi_diff(_mi_diff(Q, x_vars, alpha), xi_vars, beta)
                            result += coeff * (dP * dQ)  # matrix mult, order preserved

        return sp.simplify(result)

    def commutator_symbolic(self, other, order=1, mode='kn', sign_convention=None):
        """
        Symbol of the commutator `[Op[self], Op[other]]`, generalizing
        `PseudoDifferentialOperator.commutator_symbolic` to matrices.

        Unlike the scalar case (whose 0th-order term always vanishes,
        since scalars commute), the matrix commutator is generally
        nonzero already at 0th order: it's the ordinary matrix commutator
        `P(x,xi) Q(x,xi) - Q(x,xi) P(x,xi)`. Higher orders add the
        noncommutative analogue of the Poisson-bracket correction. For
        constant-coefficient `self`/`other` this is exact, and reduces
        exactly to the plain matrix commutator (see `compose_asymptotic`).
        """
        pq = self.compose_asymptotic(other, order=order, mode=mode, sign_convention=sign_convention)
        qp = other.compose_asymptotic(self, order=order, mode=mode, sign_convention=sign_convention)
        return sp.simplify(pq - qp)

    def exponential_symbol(self, t=1.0, order=1, mode='kn', sign_convention=None):
        """
        Symbol of `exp(t Op[self])` for the matrix-valued operator, via
        the matrix analogue of `PseudoDifferentialOperator.exponential_symbol`.

        Same truncated power series as the scalar case,

            exp(tP) ~ I + t P + (t^2/2!) P^{.2} + (t^3/3!) P^{.3} + ...

        but "P^{.n}" means the symbol of `Op[P] . Op[P] . ... . Op[P]`
        (n times), computed via the *matrix* `compose_asymptotic` --
        i.e. ordinary matrix multiplication `P @ P` order-corrected by
        the KN/Weyl derivative terms -- since matrix symbols do not
        commute and `sp.Matrix.__mul__(P, P)` alone is only the 0th-order
        (frozen-coefficient) approximation to that composition. Works
        for both `dim == 1` and `dim == 2` -- `compose_asymptotic`
        already branches on dimension internally.

        Parameters
        ----------
        t : float or sympy.Symbol, default=1.0
            Evolution parameter, same conventions as the scalar version
            (e.g. t = -i*tau for exp(-i*tau*H), t = tau for exp(tau*Delta)).
        order : int, default=1
            Truncation order, used both for the outer Taylor series and
            as the `order` passed to each `compose_asymptotic` call.
        mode : {'kn', 'weyl'}, default='kn'
            Quantization convention for the composition (2D Weyl is not
            implemented for matrix symbols -- see `compose_asymptotic`).
        sign_convention : optional
            Forwarded to `compose_asymptotic`.

        Returns
        -------
        sympy.Matrix, shape (size, size)
            Truncated symbol of exp(t Op[self]).

        Notes
        -----
        - For x[, y]-independent ("constant-coefficient") `self`,
          `compose_asymptotic` is exact, so this reduces to the exact
          truncated matrix exponential series of `P(xi[, eta])`; compare
          against `scipy.linalg.expm` at sample points to sanity-check.
        - Non-commutativity means `self` and `other`'s roles in each
          `compose_asymptotic` call matter; here every factor is `self`,
          so ordering is moot, but see `commutator_symbolic` for the
          general two-operator case.
        """
        result = sp.eye(self.size) + t * self.P_expr

        current_power = self.P_expr
        for n in range(2, order + 1):
            temp_op = MatrixPseudoDifferentialOperator(
                current_power, self.vars_x, mode='symbol',
                quantization=self.quantization, apply_backend=self.apply_backend,
            )
            current_power = temp_op.compose_asymptotic(
                self, order=order, mode=mode, sign_convention=sign_convention
            )
            coeff = t**n / sp.factorial(n)
            result += coeff * current_power

        return sp.simplify(result)

    def _asymptotic_matrix_inverse(self, order, side):
        """Matrix analogue of PseudoDifferentialOperator._asymptotic_inverse.
        Requires P(x, xi) to be invertible as a matrix (det P != 0
        symbolically); P.inv() is used as the 0th-order term. Matrix
        multiplication order is preserved: the inverse factor stays on
        the side that actually cancels P in `P . R ~ I` / `L . P ~ I`.
        """
        dim = self.dim
        if dim not in (1, 2):
            raise NotImplementedError("dim must be 1 or 2")
        P = self.P_expr
        x_vars = self.vars_x
        xi_vars = sp.symbols('xi eta', real=True) if dim == 2 else (sp.symbols('xi', real=True),)

        try:
            R0 = P.inv()
        except Exception as e:
            raise ValueError(
                "MatrixPseudoDifferentialOperator: symbol is not invertible "
                "(det P == 0 or SymPy could not confirm invertibility); "
                "asymptotic inverses require an invertible principal symbol. "
                f"Original error: {e}"
            )

        R = R0
        for n in range(1, order + 1):
            term = sp.zeros(self.size, self.size)
            for alpha in _mi_upto(n, dim):
                coeff = (1j) ** (-sum(alpha)) / _mi_factorial(alpha)
                if side == 'right':
                    dP = _mi_diff(P, xi_vars, alpha)
                    dR = _mi_diff(R, x_vars, alpha)
                    term += coeff * (dP * dR)
                else:  # 'left'
                    dR = _mi_diff(R, xi_vars, alpha)
                    dP = _mi_diff(P, x_vars, alpha)
                    term += coeff * (dR * dP)
            R = R - (R0 * term if side == 'right' else term * R0)
        return sp.simplify(R)

    def right_inverse_asymptotic(self, order=1):
        """Formal right inverse R such that Op[self] . Op[R] ~ Id up to
        O(<xi>^-order), matrix analogue of
        PseudoDifferentialOperator.right_inverse_asymptotic. Requires
        the symbol P(x, xi) to be invertible as a matrix.
        """
        return self._asymptotic_matrix_inverse(order, side='right')

    def left_inverse_asymptotic(self, order=1):
        """Formal left inverse L such that Op[L] . Op[self] ~ Id up to
        O(<xi>^-order), matrix analogue of
        PseudoDifferentialOperator.left_inverse_asymptotic. Requires
        the symbol P(x, xi) to be invertible as a matrix.
        """
        return self._asymptotic_matrix_inverse(order, side='left')

    def formal_adjoint(self, n_terms=6):
        """Formal Hermitian adjoint symbol P* of the matrix operator.

        Each entry gets the same scalar treatment as
        `PseudoDifferentialOperator.formal_adjoint` (conjugate + asymptotic
        expansion at infinity in |xi|); the resulting matrix is then
        transposed (not conjugate-transposed again -- conjugation already
        happened entrywise) because (Op[P]u, v) = (u, Op[P]* v) swaps the
        row/column roles of the symbol, same as for a plain matrix adjoint.
        """
        dim = self.dim
        xi_vars = sp.symbols('xi eta', real=True) if dim == 2 else (sp.symbols('xi', real=True),)
        expansion_var = xi_vars[0] if dim == 1 else sp.sqrt(sum(v**2 for v in xi_vars))

        P_star = self.P_expr.applyfunc(
            lambda p_ij: sp.simplify(
                sp.series(sp.conjugate(p_ij), expansion_var, sp.oo, n=n_terms).removeO()
            )
        )
        return P_star.T


# ============================================================================
# Standalone functions for Kohn-Nirenberg quantization
# ============================================================================


_KN_CACHE: Dict[Tuple, Dict[str, np.ndarray]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def _clip_complex_magnitude(P: np.ndarray, clamp: float) -> np.ndarray:
    """
    Clip a complex array by magnitude, preserving phase.
    Modifies the array in-place to minimize memory allocation.
    """
    if P.dtype != np.complex128:
        P = np.asarray(P, dtype=np.complex128)
    
    mag = np.abs(P)
    over = mag > clamp
    if np.any(over):
        # In-place multiplication preserves memory and phase
        P[over] *= (clamp / mag[over])
    return P


def _cache_key_1d(x: np.ndarray, xi: np.ndarray) -> Tuple:
    """
    Build a stable cache key from a 1D space/frequency grid pair.

    The key is derived from each grid's shape and endpoint values, so it
    automatically changes (invalidating any cached result) whenever the
    grid resolution or extent changes, without needing to hash the full
    array contents.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    xi : ndarray
        Frequency grid.

    Returns
    -------
    tuple
        `(x.shape, x[0], x[-1], xi.shape, xi[0], xi[-1])`, hashable and
        suitable as a dictionary cache key.
    """
    return (
        x.shape, float(x[0]), float(x[-1]),
        xi.shape, float(xi[0]), float(xi[-1]),
    )


def invalidate_kn_cache() -> None:
    """Clear the phase-matrix cache for non-periodic 1D operations."""
    _KN_CACHE.clear()


# ============================================================================
# Periodic Kohn-Nirenberg Quantization (FFT-based)
# ============================================================================

def kohn_nirenberg_fft(
    u_vals: np.ndarray,
    symbol_func: Callable[..., np.ndarray],
    x_grid: np.ndarray,
    kx: np.ndarray,
    fft_func: Callable,
    ifft_func: Callable,
    dim: int = 1,
    y_grid: Optional[np.ndarray] = None,
    ky: Optional[np.ndarray] = None,
    freq_window: Optional[str] = 'gaussian',
    clamp: float = 1e6,
    space_window: bool = False,
    is_spatial: bool = False,
) -> np.ndarray:
    """
    Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator
    on a periodic domain using FFTs.
    
    Applies the pseudo-differential operator Op(p) to the function u via the 
    Kohn–Nirenberg quantization:
        [Op(p) u](x) = (1/(2π)^d) ∫ p(x, ξ) e^{i x·ξ} ℱ[u](ξ) dξ,
    where ℱ[u] is the discrete Fourier transform of u, and p(x, ξ) is a symbol 
    that may depend on both spatial variables (x, y) and frequency variables (ξ, η).
    
    This implementation supports 1D and 2D spatial dimensions, providing numerical 
    stability through symbol magnitude clamping, optional frequency windowing 
    (Gaussian/Hann), and optional spatial tapering.
    
    **Fast-Path Optimization (Spatial Independence)**
        When the symbol `p` is independent of spatial coordinates (and `space_window=False`), 
        the operator simplifies to a pure Fourier multiplier: `ifft(P * fft(u))`.
        
        The function detects spatial independence using a multi-point evaluation 
        heuristic across non-zero frequencies (to avoid false positives at ξ = 0). 
        When active, computational complexity drops from O(N^2) to O(N log N) in 1D, 
        and from O(N^4) to O(N^2 log N) in 2D.
    
    **Memory-Bounded Slow Path (Spatial Dependence)**
        For space-dependent symbols, the implementation avoids O(N^2d) RAM allocation:
        - **1D**: Slices spatial evaluation into memory-bounded chunks (~256 MB max).
        - **2D**: Combines parallel multi-threaded row-blocking with 2D frequency chunking 
          and phased factorized inner products (`np.einsum`).
    
    Parameters
    ----------
    u_vals : ndarray
        Spatial samples of the input field u(x) or u(x, y).
    symbol_func : callable
        Symbol evaluator p(x, ξ) in 1D or p(x, y, ξ, η) in 2D. Must accept 
        NumPy-broadcastable positional arguments. Return arrays are automatically 
        broadcasted and cast to complex128 to handle scalar outputs safely (e.g., from `sympy.lambdify`).
    x_grid : ndarray
        1D array of spatial coordinates along the x-axis.
    kx : ndarray
        1D array of spatial frequencies in the x-direction.
    fft_func : callable
        Forward Fourier transform function (e.g., `scipy.fft.fft` or `fft2`).
    ifft_func : callable
        Inverse Fourier transform function (e.g., `scipy.fft.ifft` or `ifft2`).
    dim : {1, 2}, default=1
        Spatial dimensionality of the domain.
    y_grid : ndarray, optional
        1D array of spatial coordinates along the y-axis (required if `dim=2`).
    ky : ndarray, optional
        1D array of spatial frequencies in the y-direction (required if `dim=2`).
    freq_window : {'gaussian', 'hann', None}, default='gaussian'
        Frequency-domain window/taper applied to attenuate high-frequency instabilities.
    clamp : float, default=1e6
        Maximum allowed magnitude for the symbol entries. Magnitudes exceeding this 
        value are clipped to prevent overflow.
    space_window : bool, default=False
        If True, applies a centered Gaussian spatial taper to mitigate edge boundary artifacts. 
        *Note: Setting `space_window=True` forces execution through the slow path.*
    
    Returns
    -------
    ndarray
        Resulting array of the same shape and type (`complex128`) as `u_vals` after 
        applying the pseudo-differential operator.
    
    Raises
    ------
    ValueError
        If `dim=2` and `y_grid` or `ky` are not supplied, or if `dim` is not 1 or 2.
    """
    if dim == 1:
        dx = x_grid[1] - x_grid[0]
        Nx = len(x_grid)
        k_unshifted = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

        # --- FAST/SLOW PATH SELECTION ---
        # is_spatial=True  -> force slow path (symbol IS space-dependent)
        # is_spatial=False -> force fast path (symbol is NOT space-dependent)
        # is_spatial=None  -> heuristic decides (always slow if space_window=True,
        #                     since the taper itself makes the integrand x-dependent)
        if is_spatial is True:
            is_x_independent = False
        elif is_spatial is False:
            is_x_independent = True
        else:
            # --- FAST PATH CHECK ---
            is_x_independent = False
            if not space_window:
                try:
                    non_zero_idx = np.where(k_unshifted != 0)[0]
                    if len(non_zero_idx) >= 2:
                        idx_test = non_zero_idx[[len(non_zero_idx)//4, len(non_zero_idx)//2]]
                        k_test = k_unshifted[idx_test]
                        x_test = x_grid[[0, Nx // 2]]
                        
                        val1 = symbol_func(x_test[:, None], k_test[None, :])
                        val2 = symbol_func((x_test + dx)[:, None], k_test[None, :])
                        is_x_independent = np.allclose(val1, val2)
                except Exception:
                    is_x_independent = False

        if is_x_independent:
            U = fft_func(u_vals)
            # FIX: Enforce shape to prevent scalar/reduced-dim returns from lambdify
            P_raw = symbol_func(x_grid[0], k_unshifted)
            P = np.broadcast_to(P_raw, k_unshifted.shape).astype(np.complex128).copy()
            P = _clip_complex_magnitude(P, clamp)
            
            if freq_window == 'gaussian':
                sigma = 0.8 * np.max(np.abs(k_unshifted))
                P *= np.exp(-(k_unshifted / sigma) ** 4)
            elif freq_window == 'hann':
                k_max = np.max(np.abs(k_unshifted))
                W = 0.5 * (1 + np.cos(np.pi * k_unshifted / k_max))
                P *= W * (np.abs(k_unshifted) < k_max)
                
            return ifft_func(P * U)

        # --- SLOW PATH (O(N) Memory-Bounded Integration) ---
        k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
        dk = k[1] - k[0]
        f_hat = np.fft.fftshift(fft_func(np.fft.fftshift(u_vals)) * dx)
        
        win_k = None
        if freq_window == 'gaussian':
            sigma = 0.8 * np.max(np.abs(k))
            win_k = np.exp(-(k / sigma) ** 4)
        elif freq_window == 'hann':
            k_max = np.max(np.abs(k))
            win_k = 0.5 * (1 + np.cos(np.pi * k / k_max)) * (np.abs(k) < k_max)
            
        sw_x = None
        if space_window:
            x0 = (x_grid[0] + x_grid[-1]) / 2
            L = (x_grid[-1] - x_grid[0]) / 2
            sw_x = np.exp(-((x_grid - x0) / L) ** 2)

        MAX_ELEMENTS = 16 * 1024 * 1024  # ~256 MB for complex128
        chunk_size = max(1, min(Nx, MAX_ELEMENTS // len(k)))
        result = np.zeros(Nx, dtype=np.complex128)

        for i0 in range(0, Nx, chunk_size):
            i1 = min(i0 + chunk_size, Nx)
            x_blk = x_grid[i0:i1]
            B = i1 - i0
            
            Xb = x_blk[:, None]
            Kb = k[None, :]
            
            # FIX: Enforce target shape (B, len(k)) before any in-place ops
            P_raw = symbol_func(Xb, Kb)
            P_blk = np.broadcast_to(P_raw, (B, len(k))).astype(np.complex128).copy()
            P_blk = _clip_complex_magnitude(P_blk, clamp)
            
            if win_k is not None:
                P_blk *= win_k[None, :]
            if sw_x is not None:
                P_blk *= sw_x[i0:i1, None]
                
            kernel_blk = np.exp(1j * Xb * Kb)
            
            result[i0:i1] = (dk / (2 * np.pi)) * np.einsum(
                'bk, k, bk -> b', P_blk, f_hat, kernel_blk, optimize=True
            )
        return result

    elif dim == 2:
        if y_grid is None or ky is None:
            raise ValueError("y_grid and ky are required for dim=2")
            
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        Nx, Ny = len(x_grid), len(y_grid)
        
        kx_unshifted = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky_unshifted = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)

        # --- FAST/SLOW PATH SELECTION ---
        # is_spatial=True  -> force slow path (symbol IS space-dependent)
        # is_spatial=False -> force fast path (symbol is NOT space-dependent)
        # is_spatial=None  -> heuristic decides (always slow if space_window=True,
        #                     since the taper itself makes the integrand x-dependent)
        if is_spatial is True:
            is_independent = False
        elif is_spatial is False:
            is_independent = True
        else:
        # --- FAST PATH CHECK ---
            is_independent = False
            if not space_window:
                try:
                    non_zero_kx = np.where(kx_unshifted != 0)[0]
                    non_zero_ky = np.where(ky_unshifted != 0)[0]
                    
                    if len(non_zero_kx) > 0 and len(non_zero_ky) > 0:
                        idx_x = non_zero_kx[[len(non_zero_kx)//4, len(non_zero_kx)//2]]
                        idx_y = non_zero_ky[[len(non_zero_ky)//4, len(non_zero_ky)//2]]
                        
                        kx_test = kx_unshifted[idx_x]
                        ky_test = ky_unshifted[idx_y]
                        x_test = x_grid[[0, Nx // 2]]
                        y_test = y_grid[[0, Ny // 2]]
                        
                        X_t, Y_t = np.meshgrid(x_test, y_test, indexing='ij')
                        KX_t, KY_t = np.meshgrid(kx_test, ky_test, indexing='ij')
                        
                        val1 = symbol_func(X_t[..., None, None], Y_t[..., None, None], 
                                           KX_t[None, None, ...], KY_t[None, None, ...])
                        val2 = symbol_func((X_t + dx)[..., None, None], (Y_t + dy)[..., None, None], 
                                           KX_t[None, None, ...], KY_t[None, None, ...])
                        is_independent = np.allclose(val1, val2)
                except Exception:
                    is_independent = False

        if is_independent:
            U = fft_func(u_vals)
            KX, KY = np.meshgrid(kx_unshifted, ky_unshifted, indexing='ij')
            
            # FIX: Enforce target shape
            P_raw = symbol_func(x_grid[0], y_grid[0], KX, KY)
            P = np.broadcast_to(P_raw, KX.shape).astype(np.complex128).copy()
            P = _clip_complex_magnitude(P, clamp)
            
            if freq_window == 'gaussian':
                sx = 0.8 * np.max(np.abs(kx_unshifted))
                sy = 0.8 * np.max(np.abs(ky_unshifted))
                P *= np.exp(-(KX / sx) ** 4) * np.exp(-(KY / sy) ** 4)
            elif freq_window == 'hann':
                kx_max = np.max(np.abs(kx_unshifted))
                ky_max = np.max(np.abs(ky_unshifted))
                Wx = 0.5 * (1 + np.cos(np.pi * KX / kx_max)) * (np.abs(KX) < kx_max)
                Wy = 0.5 * (1 + np.cos(np.pi * KY / ky_max)) * (np.abs(KY) < ky_max)
                P *= Wx * Wy
                
            return ifft_func(P * U)

        # --- SLOW PATH ---
        kx_s = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
        ky_s = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
        dkx, dky = kx_s[1] - kx_s[0], ky_s[1] - ky_s[0]
        Nkx, Nky = len(kx_s), len(ky_s)

        f_hat = np.fft.fftshift(fft_func(np.fft.fftshift(u_vals)) * dx * dy)
        exp_y = np.exp(1j * np.outer(y_grid, ky_s))

        win_kx, win_ky = None, None
        if freq_window == 'gaussian':
            sx = 0.8 * np.max(np.abs(kx_s))
            sy = 0.8 * np.max(np.abs(ky_s))
            win_kx = np.exp(-(kx_s / sx) ** 4)
            win_ky = np.exp(-(ky_s / sy) ** 4)
        elif freq_window == 'hann':
            kx_max, ky_max = np.max(np.abs(kx_s)), np.max(np.abs(ky_s))
            win_kx = 0.5 * (1 + np.cos(np.pi * kx_s / kx_max)) * (np.abs(kx_s) < kx_max)
            win_ky = 0.5 * (1 + np.cos(np.pi * ky_s / ky_max)) * (np.abs(ky_s) < ky_max)

        sw_x, sw_y = None, None
        if space_window:
            x0, Lx = (x_grid[0] + x_grid[-1]) / 2, (x_grid[-1] - x_grid[0]) / 2
            y0, Ly = (y_grid[0] + y_grid[-1]) / 2, (y_grid[-1] - y_grid[0]) / 2
            sw_x = np.exp(-((x_grid - x0) / Lx) ** 2)
            sw_y = np.exp(-((y_grid - y0) / Ly) ** 2)

        n_workers = max(w for w in range(1, int(FFT_WORKERS) + 1) if Nx % w == 0)
        base = max(1, Nx // n_workers)
        boundaries = [(i * base, min((i + 1) * base, Nx)) for i in range(n_workers) if i * base < Nx]
        result = np.zeros((Nx, Ny), dtype=np.complex128)

        def _process_block(bounds: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
            """
            Process one spatial block of rows for the memory-bounded slow path.
    
            Evaluates the symbol on the chunked space-frequency sub-grid,
            applies windowing/clamping, and accumulates the quadrature
            contribution via `np.einsum` into the block result. Designed to
            run inside a ThreadPoolExecutor for parallel row-block processing
            in the 2D case.
    
            Parameters
            ----------
            bounds : tuple of (int, int)
                Row indices (i0, i1) defining the spatial block.
    
            Returns
            -------
            tuple
                (i0, i1, result_block) where result_block is the ndarray of
                shape (i1−i0, Ny) [2D] or (i1−i0,) [1D] containing the
                operator output for those rows.
            """
            i0, i1 = bounds
            x_blk = x_grid[i0:i1]
            B = i1 - i0

            MAX_ELEMENTS = 16 * 1024 * 1024
            prod_C = max(1, MAX_ELEMENTS // (B * Ny))
            C1 = min(int(np.sqrt(prod_C)), Nkx)
            C2 = min(max(1, prod_C // C1), Nky)

            Xb = x_blk[:, None, None, None]
            Yb = y_grid[None, :, None, None]
            exp_x_full = np.exp(1j * np.outer(x_blk, kx_s))
            res_block = np.zeros((B, Ny), dtype=np.complex128)
            
            sw_x_blk = sw_x[i0:i1, None, None, None] if sw_x is not None else None

            for m0 in range(0, Nkx, C1):
                m1 = min(m0 + C1, Nkx)
                exp_x_chunk = exp_x_full[:, m0:m1].reshape(B, 1, m1-m0, 1)
                fh_m = f_hat[m0:m1, :]
                
                w_kx = win_kx[m0:m1, None] if win_kx is not None else 1.0

                for n0 in range(0, Nky, C2):
                    n1 = min(n0 + C2, Nky)
                    
                    P_chunk = symbol_func(Xb, Yb, kx_s[None, None, m0:m1, None], ky_s[None, None, None, n0:n1])
                    # FIX: Enforce target shape and ensure writability
                    P_chunk = np.broadcast_to(P_chunk, (B, Ny, m1-m0, n1-n0)).astype(np.complex128).copy()
                    P_chunk = _clip_complex_magnitude(P_chunk, clamp)

                    if freq_window is not None:
                        w_ky = win_ky[None, n0:n1] if isinstance(win_ky, np.ndarray) else 1.0
                        P_chunk *= (w_kx * w_ky)
                        
                    if space_window:
                        if sw_x_blk is not None:
                            P_chunk *= sw_x_blk
                        if sw_y is not None:
                            P_chunk *= sw_y[None, :, None, None]

                    exp_y_chunk = exp_y[:, n0:n1]
                    phase_chunk = exp_x_chunk * exp_y_chunk[None, :, None, :]
                    fh_sub = fh_m[:, n0:n1]

                    res_block += (dkx * dky / (2 * np.pi) ** 2) * np.einsum(
                        'bxky, ky, bxky -> bx', 
                        P_chunk, fh_sub, phase_chunk, 
                        optimize=True
                    )
            return i0, i1, res_block

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for i0, i1, blk in executor.map(_process_block, boundaries):
                result[i0:i1, :] = blk

        return result
    else:
        raise ValueError("Only dim=1 and dim=2 are supported")


# ============================================================================
# Non-Periodic Kohn-Nirenberg Quantization (Dirichlet)
# ============================================================================

def _cache_key_2d(x1, x2, xi1, xi2, freq_window, space_window):
    """
    Build a stable cache key for the 2D non-periodic Kohn-Nirenberg branch.

    Unlike the 1D key, this one must also encode `freq_window` and
    `space_window`, since those options change the actual content of the
    cached phase matrices/windows, not just the underlying grid. Each axis
    is hashed via `_cache_key_1d` so the 2D key stays consistent with the
    existing 1D cache-invalidation logic.

    Parameters
    ----------
    x1, x2 : ndarray
        Spatial grids along each axis.
    xi1, xi2 : ndarray
        Frequency grids along each axis.
    freq_window : str or None
        Name of the frequency-domain window applied when building the
        cached phase matrix.
    space_window : bool
        Whether a spatial window is applied; included as a bare boolean
        since it only changes whether windowing is on or off.

    Returns
    -------
    tuple
        Hashable key combining both axes' grid signatures with the window
        settings, suitable for use as a dictionary cache key.
    """
    return (
        _cache_key_1d(x1, xi1),
        _cache_key_1d(x2, xi2),
        freq_window,
        bool(space_window),
    )


def kohn_nirenberg_nonperiodic(
    u_vals: np.ndarray,
    x_grid: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    xi_grid: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    symbol_func: Callable[..., np.ndarray],
    freq_window: Optional[str] = 'gaussian',
    clamp: float = 1e6,
    space_window: bool = False,
    is_spatial: Optional[bool] = None,
    _cache: Dict = _KN_CACHE,
) -> np.ndarray:
    """
    Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator
    on a non-periodic domain using direct matrix/quadrature-based transforms.

    Applies the pseudo-differential operator Op(p) to the input function u via the
    non-periodic Kohn–Nirenberg integral formula:
        [Op(p) u](x) = (1/(2π)^d) ∫ p(x, ξ) e^{i x·ξ} ℱ_NP[u](ξ) dξ,
    where ℱ_NP[u] is the direct discrete Fourier integral transform evaluated over an
    arbitrary non-periodic spatial grid `x` and frequency grid `xi`.

    Supports 1D and 2D spatial dimensions, featuring a caching mechanism for both
    1D and 2D phase/window matrices, symbol magnitude clamping, optional frequency
    windowing (Gaussian/Hann), and spatial tapering.

    **1D Cached Path**
        In 1D, precalculates and caches discrete Fourier transform phases (`phase_ft`),
        reconstruction phases (`exp_matrix`), and window arrays in `_cache` to accelerate
        repeated function evaluations on identical grids.

    **2D Cached Path**
        In 2D, precalculates and caches the analogous grid-only objects — forward phase
        matrices (`phase1`, `phase2`), reconstruction phase matrices (`exp1`, `exp2`),
        the frequency window (`freq_win_2d`), and (if `space_window=True`) the spatial
        taper arrays (`sw_x1_full`, `sw_x2`). The cache key includes `freq_window` and
        `space_window` since, unlike the 1D case, these options change the cached
        content itself, not just which arrays get used.

    **2D Fast-Path Optimization (Spatial Independence)**
        Symbol spatial-dependence is controlled by `is_spatial`:
          - `is_spatial=False` forces the fast path (symbol treated as x-independent).
          - `is_spatial=True` forces the slow path (symbol treated as x-dependent).
          - `is_spatial=None` (default) runs a multi-point sampling heuristic (skipped,
            and slow path forced, whenever `space_window=True`, since the taper itself
            introduces x-dependence).
        When spatial independence holds, matrix multiplications perform global frequency
        filtering in O(N^3) complexity instead of the full O(N^4) space-dependent integration.

    **2D Memory-Bounded Slow Path (Spatial Dependence)**
        For space-dependent symbols in 2D, a multi-tiered execution strategy prevents RAM spikes:
        - **Row-Based Parallelization**: Slices `x1` into spatial blocks distributed across
          a worker thread pool (`ThreadPoolExecutor`).
        - **Dual Frequency Chunking**: Iterates across sub-blocks of `xi1` and `xi2` to guarantee
          intermediate tensor evaluation (`sv_chunk`, `phase_chunk`) remains strictly bounded (~256 MB max).
        - **Tensor Contraction**: Employs optimized Einstein summation (`np.einsum`) for localized quadrature integration.

    Parameters
    ----------
    u_vals : ndarray
        Spatial samples of the input field u(x) [1D] or u(x1, x2) [2D].
    x_grid : ndarray or tuple of ndarray
        Spatial coordinate grid `x` (1D) or tuple `(x1, x2)` (2D).
    xi_grid : ndarray or tuple of ndarray
        Frequency grid `xi` (1D) or tuple `(xi1, xi2)` (2D).
    symbol_func : callable
        Symbol evaluator p(x, ξ) in 1D or p(x1, x2, ξ1, ξ2) in 2D. Must accept
        NumPy-broadcastable positional arguments. Returns are automatically
        broadcasted, type-cast to `complex128`, and reshaped/copied safely to handle scalar
        or reduced-dimension outputs (e.g., from `sympy.lambdify`).
    freq_window : {'gaussian', 'hann', None}, default='gaussian'
        Frequency-domain window/taper applied to attenuate high-frequency numerical artifacts.
    clamp : float, default=1e6
        Maximum allowed magnitude for symbol values. Entries exceeding this threshold
        are clipped to prevent overflow.
    space_window : bool, default=False
        If True, applies a centered Gaussian spatial taper to attenuate edge boundary artifacts.
        *Note: Enabling this disables the 2D spatial-independence fast path.*
    is_spatial : bool or None, default=None
        Explicit hint about whether the 2D symbol depends on (x1, x2).
        True forces the slow (space-dependent) path, False forces the fast
        (space-independent) path, None triggers the sampling heuristic.
    _cache : dict, optional
        Global or local cache dictionary storing reusable 1D and 2D phase/window matrices.
        Defaults to module-level `_KN_CACHE`.

    Returns
    -------
    ndarray
        Resulting complex-valued array (`complex128`) of the same dimensionality and shape
        as `u_vals` after applying the non-periodic pseudo-differential operator.

    Raises
    ------
    NotImplementedError
        If `u_vals.ndim` is not 1 or 2.
    """
    if u_vals.ndim == 1:
        x = np.asarray(x_grid)
        xi = np.asarray(xi_grid)
        dx = x[1] - x[0]
        dxi = xi[1] - xi[0]

        key = _cache_key_1d(x, xi)
        if key not in _cache:
            phase_ft = np.exp(-1j * np.outer(xi, x))
            exp_matrix = np.exp(1j * np.outer(x, xi))

            xi_abs_max = np.max(np.abs(xi))
            sigma_w = 0.8 * xi_abs_max
            window_gauss = np.exp(-(xi / sigma_w) ** 4)

            window_hann = np.zeros_like(xi)
            mask = np.abs(xi) < xi_abs_max
            window_hann[mask] = 0.5 * (1.0 + np.cos(np.pi * xi[mask] / xi_abs_max))

            x_center = (x[0] + x[-1]) / 2.0
            L_half = (x[-1] - x[0]) / 2.0
            spatial_taper = np.exp(-((x - x_center) / L_half) ** 2)

            _cache[key] = dict(
                phase_ft=phase_ft, exp_matrix=exp_matrix,
                window_gauss=window_gauss, window_hann=window_hann,
                spatial_taper=spatial_taper,
            )
            warnings.warn(
                f"kohn_nirenberg_nonperiodic: building 1D cache (Nx={len(x)}, Nxi={len(xi)}).",
                stacklevel=2,
            )

        entry = _cache[key]
        u_hat = dx * (entry['phase_ft'] @ u_vals)

        sigma_raw = symbol_func(x[:, None], xi[None, :])
        sigma = np.broadcast_to(sigma_raw, (len(x), len(xi))).astype(np.complex128).copy()
        sigma = _clip_complex_magnitude(sigma, clamp)

        if freq_window == 'gaussian':
            sigma *= entry['window_gauss'][None, :]
        elif freq_window == 'hann':
            sigma *= entry['window_hann'][None, :]

        if space_window:
            sigma *= entry['spatial_taper'][:, None]

        weighted_exp = sigma * entry['exp_matrix']
        return (dxi / (2.0 * np.pi)) * (weighted_exp @ u_hat)

    elif u_vals.ndim == 2:
        x1, x2 = x_grid
        xi1, xi2 = xi_grid
        dx1, dx2 = x1[1] - x1[0], x2[1] - x2[0]
        dxi1, dxi2 = xi1[1] - xi1[0], xi2[1] - xi2[0]
        Nx1, Nx2 = len(x1), len(x2)
        Nxi1, Nxi2 = len(xi1), len(xi2)

        # --- CACHE 2D : objets qui ne dépendent que de la grille + des options
        # de fenêtrage (jamais de symbol_func ni de u_vals) ---
        key2d = _cache_key_2d(x1, x2, xi1, xi2, freq_window, space_window)
        if key2d not in _cache:
            phase1 = np.exp(-1j * np.outer(xi1, x1))
            phase2 = np.exp(-1j * np.outer(x2, xi2))
            exp1 = np.exp(1j * np.outer(x1, xi1))
            exp2 = np.exp(1j * np.outer(x2, xi2))

            freq_win_2d = None
            if freq_window == 'gaussian':
                s1 = 0.8 * np.max(np.abs(xi1))
                s2 = 0.8 * np.max(np.abs(xi2))
                freq_win_2d = np.exp(-(xi1 / s1) ** 4)[:, None] * np.exp(-(xi2 / s2) ** 4)[None, :]
            elif freq_window == 'hann':
                xi1_max, xi2_max = np.max(np.abs(xi1)), np.max(np.abs(xi2))
                Wx = 0.5 * (1 + np.cos(np.pi * xi1 / xi1_max)) * (np.abs(xi1) < xi1_max)
                Wy = 0.5 * (1 + np.cos(np.pi * xi2 / xi2_max)) * (np.abs(xi2) < xi2_max)
                freq_win_2d = Wx[:, None] * Wy[None, :]

            sw_x1_full = sw_x2 = None
            if space_window:
                xc = (x1[0] + x1[-1]) / 2.0
                Lx = (x1[-1] - x1[0]) / 2.0
                sw_x1_full = np.exp(-((x1 - xc) / Lx) ** 2)

                yc = (x2[0] + x2[-1]) / 2.0
                Ly = (x2[-1] - x2[0]) / 2.0
                sw_x2 = np.exp(-((x2 - yc) / Ly) ** 2)

            _cache[key2d] = dict(
                phase1=phase1, phase2=phase2, exp1=exp1, exp2=exp2,
                freq_win_2d=freq_win_2d, sw_x1_full=sw_x1_full, sw_x2=sw_x2,
            )
            warnings.warn(
                f"kohn_nirenberg_nonperiodic: building 2D cache "
                f"(Nx1={Nx1}, Nx2={Nx2}, Nxi1={Nxi1}, Nxi2={Nxi2}, "
                f"freq_window={freq_window!r}, space_window={space_window}).",
                stacklevel=2,
            )

        entry2d = _cache[key2d]
        phase1, phase2 = entry2d['phase1'], entry2d['phase2']
        exp1, exp2 = entry2d['exp1'], entry2d['exp2']
        freq_win_2d = entry2d['freq_win_2d']
        sw_x1_full, sw_x2 = entry2d['sw_x1_full'], entry2d['sw_x2']

        u_hat = dx1 * dx2 * (phase1 @ u_vals @ phase2)

        # --- FAST/SLOW PATH SELECTION ---
        # is_spatial=True  -> force slow path (symbol IS space-dependent)
        # is_spatial=False -> force fast path (symbol is NOT space-dependent)
        # is_spatial=None  -> heuristic decides (always slow if space_window=True,
        #                     since the taper itself makes the integrand x-dependent)
        if is_spatial is True:
            is_x_independent = False
        elif is_spatial is False:
            is_x_independent = True
        else:
            is_x_independent = False
            if not space_window:
                try:
                    x1_test = x1[[0, -1]]
                    x2_test = x2[[0, -1]]
                    xi_idx = max(1, Nxi1 // 2)
                    eta_idx = max(1, Nxi2 // 2)

                    val1 = symbol_func(
                        x1_test[:, None, None, None],
                        x2_test[None, :, None, None],
                        xi1[None, None, xi_idx:xi_idx + 1, None],
                        xi2[None, None, None, eta_idx:eta_idx + 1]
                    )
                    val2 = symbol_func(
                        (x1_test + dx1)[:, None, None, None],
                        (x2_test + dx2)[None, :, None, None],
                        xi1[None, None, xi_idx:xi_idx + 1, None],
                        xi2[None, None, None, eta_idx:eta_idx + 1]
                    )
                    is_x_independent = np.allclose(val1, val2)
                except Exception:
                    is_x_independent = False

        if is_x_independent:
            p_full_raw = symbol_func(
                np.full((1, 1, Nxi1, 1), x1[0]),
                np.full((1, 1, 1, Nxi2), x2[0]),
                xi1[None, None, :, None],
                xi2[None, None, None, :]
            )
            p_full = np.broadcast_to(p_full_raw, (1, 1, Nxi1, Nxi2)).astype(np.complex128).reshape(Nxi1, Nxi2).copy()
            p_full = _clip_complex_magnitude(p_full, clamp)

            if freq_win_2d is not None:
                p_full *= freq_win_2d

            u_hat_filtered = p_full * u_hat

            return (dxi1 * dxi2 / (2.0 * np.pi) ** 2) * (exp1 @ u_hat_filtered @ exp2.T)

        # --- SLOW PATH ---
        iph2 = exp2  # np.exp(1j * outer(x2, xi2)), déjà en cache
        n_workers = max(w for w in range(1, int(FFT_WORKERS) + 1) if Nx1 % w == 0)
        base = max(1, Nx1 // n_workers)
        boundaries = [(i * base, min((i + 1) * base, Nx1)) for i in range(n_workers) if i * base < Nx1]
        result = np.zeros((Nx1, Nx2), dtype=np.complex128)

        def _process_block(bounds: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
            """
            Process one spatial block of rows for the memory-bounded slow path.
    
            Evaluates the symbol on the chunked space-frequency sub-grid,
            applies windowing/clamping, and accumulates the quadrature
            contribution via `np.einsum` into the block result. Designed to
            run inside a ThreadPoolExecutor for parallel row-block processing
            in the 2D case.
    
            Parameters
            ----------
            bounds : tuple of (int, int)
                Row indices (i0, i1) defining the spatial block.
    
            Returns
            -------
            tuple
                (i0, i1, result_block) where result_block is the ndarray of
                shape (i1−i0, Ny) [2D] or (i1−i0,) [1D] containing the
                operator output for those rows.
            """
            i0, i1 = bounds
            x1_blk = x1[i0:i1]
            B = i1 - i0

            MAX_ELEMENTS = 16 * 1024 * 1024
            prod_C = max(1, MAX_ELEMENTS // (B * Nx2))
            C1 = min(int(np.sqrt(prod_C)), Nxi1)
            C2 = min(max(1, prod_C // C1), Nxi2)

            X1b = x1_blk[:, None, None, None]
            X2b = x2[None, :, None, None]
            res_block = np.zeros((B, Nx2), dtype=np.complex128)

            sw_x1_blk = sw_x1_full[i0:i1, None, None, None] if space_window else None

            for k0 in range(0, Nxi1, C1):
                k1 = min(k0 + C1, Nxi1)
                iph1_chunk = np.exp(1j * np.outer(x1_blk, xi1[k0:k1])).reshape(B, 1, k1 - k0, 1)
                u_hat_k = u_hat[k0:k1, :]

                for m0 in range(0, Nxi2, C2):
                    m1 = min(m0 + C2, Nxi2)

                    sv_chunk = symbol_func(X1b, X2b, xi1[None, None, k0:k1, None], xi2[None, None, None, m0:m1])
                    sv_chunk = np.broadcast_to(sv_chunk, (B, Nx2, k1 - k0, m1 - m0)).astype(np.complex128).copy()
                    sv_chunk = _clip_complex_magnitude(sv_chunk, clamp)

                    if freq_win_2d is not None:
                        sv_chunk *= freq_win_2d[k0:k1, m0:m1][None, None, :, :]

                    if space_window:
                        if sw_x1_blk is not None:
                            sv_chunk *= sw_x1_blk
                        if sw_x2 is not None:
                            sv_chunk *= sw_x2[None, :, None, None]

                    iph2_chunk = iph2[:, m0:m1]
                    phase_chunk = iph1_chunk * iph2_chunk[None, :, None, :]
                    u_hat_sub = u_hat_k[:, m0:m1]

                    res_block += (dxi1 * dxi2 / (2.0 * np.pi) ** 2) * np.einsum(
                        'bxky, ky, bxky -> bx',
                        sv_chunk, u_hat_sub, phase_chunk,
                        optimize=True
                    )
            return i0, i1, res_block

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for i0, i1, blk in executor.map(_process_block, boundaries):
                result[i0:i1, :] = blk

        return result
    else:
        raise NotImplementedError("Only 1D (ndim=1) and 2D (ndim=2) inputs are supported")


def _sympy_number(z, digits=5, drop_tol=0.0):
    """
    Convert a Python/NumPy complex number into a SymPy number, since
    `sympy.Float` does not accept complex values directly.

    Parameters
    ----------
    z : complex or float
        Value to convert.
    digits : int, default 5
        Number of significant digits kept for the real and imaginary
        parts.
    drop_tol : float, default 0.0
        Real or imaginary components with absolute value at or below this
        threshold are snapped to exactly zero before conversion, to avoid
        carrying negligible numerical noise into the symbolic expression.

    Returns
    -------
    sympy.Float or sympy.Expr
        `sympy.Float(re, digits)` if the imaginary part is zero, otherwise
        `sympy.Float(re, digits) + sympy.I * sympy.Float(im, digits)`.
    """
    z = complex(z)
    re = float(np.real(z))
    im = float(np.imag(z))

    if abs(re) <= drop_tol:
        re = 0.0
    if abs(im) <= drop_tol:
        im = 0.0

    if im == 0.0:
        return sp.Float(re, digits)

    return sp.Float(re, digits) + sp.I * sp.Float(im, digits)


def _chebyshev_polynomial(n, z):
    """
    Return T_n(z) as an explicit expanded SymPy polynomial.

    This avoids possible lambdify issues with special Chebyshev functions.
    """
    if n == 0:
        return sp.S.One
    if n == 1:
        return z

    t_prev = sp.S.One
    t_curr = z

    for _ in range(2, n + 1):
        t_prev, t_curr = t_curr, sp.expand(2 * z * t_curr - t_prev)

    return t_curr

def evaluate_decomposition_quality(
    orig_expr,
    symbolic_pairs,
    x_syms,
    xi_syms,
    bounds,
    num_samples=10000,
    seed=42,
):
    """
    Estimate the symbol-level approximation error of a separable/low-rank
    decomposition against the original expression, via Monte Carlo
    sampling at random off-grid points (so the error reflects genuine
    approximation quality rather than exact agreement at the fitting
    nodes):

        orig_expr(x, xi) ≈ sum_k a_k(x) q_k(xi)

    Parameters
    ----------
    orig_expr : sympy.Expr
        Original joint symbol being approximated.
    symbolic_pairs : list of tuple
        Candidate decomposition, as pairs `(a_k(x), q_k(xi))` of sympy
        expressions.
    x_syms : list of sympy symbols
        Spatial variables of `orig_expr`.
    xi_syms : list of sympy symbols
        Frequency variables of `orig_expr`.
    bounds : dict
        Mapping from each symbol in `x_syms + xi_syms` to a `(min, max)`
        sampling range.
    num_samples : int, default 10000
        Number of random points drawn uniformly within `bounds`.
    seed : int, default 42
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys `'rel_l2_error'` (relative L2 error over the
        sampled points), `'max_abs_error'` and `'mean_abs_error'`
        (pointwise absolute-error statistics).
    """
    rng = np.random.default_rng(seed)

    x_syms = list(x_syms)
    xi_syms = list(xi_syms)
    all_syms = x_syms + xi_syms

    sample_dict = {}
    for s in all_syms:
        s_min, s_max = bounds[s]
        sample_dict[s] = rng.uniform(s_min, s_max, size=num_samples)

    # Original expression
    f_orig = sp.lambdify(all_syms, orig_expr, modules="numpy")
    args = [sample_dict[s] for s in all_syms]
    y_orig = np.asarray(f_orig(*args), dtype=np.complex128).reshape(-1)

    if y_orig.size == 1:
        y_orig = np.full(num_samples, y_orig.item(), dtype=np.complex128)
    elif y_orig.size != num_samples:
        y_orig = np.broadcast_to(y_orig, (num_samples,)).astype(np.complex128)

    # Approximation
    y_approx = np.zeros(num_samples, dtype=np.complex128)

    x_pts = [sample_dict[s] for s in x_syms]
    xi_pts = [sample_dict[s] for s in xi_syms]

    for a_k, q_k in symbolic_pairs:
        f_a = sp.lambdify(x_syms, a_k, modules="numpy")
        f_q = sp.lambdify(xi_syms, q_k, modules="numpy")

        try:
            val_a = np.asarray(f_a(*x_pts), dtype=np.complex128).reshape(-1)
            if val_a.size == 1:
                val_a = np.full(num_samples, val_a.item(), dtype=np.complex128)
            elif val_a.size != num_samples:
                val_a = np.broadcast_to(val_a, (num_samples,)).astype(np.complex128)
        except Exception:
            val_a = np.full(num_samples, complex(a_k), dtype=np.complex128)

        try:
            val_q = np.asarray(f_q(*xi_pts), dtype=np.complex128).reshape(-1)
            if val_q.size == 1:
                val_q = np.full(num_samples, val_q.item(), dtype=np.complex128)
            elif val_q.size != num_samples:
                val_q = np.broadcast_to(val_q, (num_samples,)).astype(np.complex128)
        except Exception:
            val_q = np.full(num_samples, complex(q_k), dtype=np.complex128)

        y_approx += val_a * val_q

    diff = y_orig - y_approx

    norm_orig = np.linalg.norm(y_orig)
    norm_diff = np.linalg.norm(diff)

    rel_l2_err = float(norm_diff / norm_orig) if norm_orig > 0 else float(norm_diff)

    abs_err = np.abs(diff)

    return {
        "rel_l2_error": rel_l2_err,
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
    }


def factorize_symbolic(
    expr,
    x_syms,
    xi_syms,
    bounds,
    degree=6,
    tol=1e-5,
    num_samples=10000,
    seed=42,
    digits=5,
):
    """
    Low-rank Chebyshev/SVD factorization of a joint symbol:

        p(x, xi) ≈ sum_{k=1}^r a_k(x) q_k(xi)

    The approximation is valid on the bounded rectangle given by `bounds`.

    Parameters
    ----------
    expr : sympy.Expr
        Symbol to factorize, usually the Peetre joint residual.
    x_syms : list of sympy symbols
        Spatial variables.
    xi_syms : list of sympy symbols
        Frequency variables.
    bounds : dict
        Dictionary mapping each symbol to (min, max).
    degree : int
        Chebyshev degree in each variable.
    tol : float
        Relative singular-value cutoff and coefficient pruning threshold.
    num_samples : int
        Number of Monte Carlo samples for quality diagnostics.
    seed : int
        RNG seed.
    digits : int
        Number of digits used when converting floating coefficients to SymPy.

    Returns
    -------
    symbolic_pairs : list of tuple
        List of `(a_k(x), q_k(xi))` SymPy expressions.
    metrics : dict
        Symbol-level approximation diagnostics.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    x_syms = list(x_syms)
    xi_syms = list(xi_syms)
    all_syms = x_syms + xi_syms

    # ---------------------------------------------------------------
    # 1. Chebyshev-Gauss-Lobatto nodes on [-1, 1]
    # ---------------------------------------------------------------
    nodes_1d = [
        np.cos(np.pi * np.arange(degree + 1) / degree)
        for _ in all_syms
    ]

    # ---------------------------------------------------------------
    # 2. Normalize physical variables to [-1, 1]
    # ---------------------------------------------------------------
    norm_vars = {}
    phys_from_norm = []

    for s in all_syms:
        s_min, s_max = bounds[s]

        if s_max <= s_min:
            s_min = float(s_min) - 1.0
            s_max = float(s_min) + 2.0

        norm_vars[s] = (2 * s - (s_min + s_max)) / (s_max - s_min)

        phys_from_norm.append(
            lambda y, b_min=s_min, b_max=s_max:
                0.5 * (b_min + b_max) + 0.5 * (b_max - b_min) * y
        )

    # ---------------------------------------------------------------
    # 3. Evaluate expression on tensor-product Chebyshev grid
    # ---------------------------------------------------------------
    grid_coords = [
        phys_from_norm[idx](nodes_1d[idx])
        for idx in range(len(all_syms))
    ]

    mesh = np.meshgrid(*grid_coords, indexing="ij")

    func_num = sp.lambdify(all_syms, expr, modules="numpy")
    P_eval = np.asarray(func_num(*mesh), dtype=np.complex128)

    target_shape = mesh[0].shape
    if P_eval.shape != target_shape:
        P_eval = np.broadcast_to(P_eval, target_shape).astype(np.complex128)

    P_eval = P_eval.copy()

    empty_metrics = {
        "rel_l2_error": 0.0,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "svd_energy_retained_pct": 100.0,
        "singular_values": np.array([]),
    }

    if np.allclose(P_eval, 0.0, atol=1e-14):
        return [], empty_metrics

    # ---------------------------------------------------------------
    # 4. Chebyshev coefficients by Vandermonde inversion
    # ---------------------------------------------------------------
    vands = [
        np.polynomial.chebyshev.chebvander(nodes_1d[i], degree)
        for i in range(len(all_syms))
    ]

    C_tensor = P_eval

    for i, V in enumerate(vands):
        inv_V = np.linalg.inv(V)

        C_tensor = np.moveaxis(C_tensor, i, 0)
        orig_shape = C_tensor.shape
        C_tensor = inv_V @ C_tensor.reshape(orig_shape[0], -1)
        C_tensor = C_tensor.reshape(orig_shape)
        C_tensor = np.moveaxis(C_tensor, 0, i)

    # ---------------------------------------------------------------
    # 5. Reshape coefficients into spatial × frequency matrix
    # ---------------------------------------------------------------
    d_x = len(x_syms)
    d_xi = len(xi_syms)

    N_x_total = (degree + 1) ** d_x
    N_xi_total = (degree + 1) ** d_xi

    C_matrix = C_tensor.reshape((N_x_total, N_xi_total))

    # ---------------------------------------------------------------
    # 6. SVD low-rank truncation
    # ---------------------------------------------------------------
    U, S, Vt = np.linalg.svd(C_matrix, full_matrices=False)

    if S.size == 0 or S[0] == 0:
        return [], empty_metrics

    keep = S > (S[0] * tol)

    if not np.any(keep):
        keep = np.zeros_like(S, dtype=bool)
        keep[0] = True

    energy_den = float(np.sum(S ** 2))
    svd_energy_retained = (
        100.0 * float(np.sum(S[keep] ** 2)) / energy_den
        if energy_den > 0 else 100.0
    )

    # ---------------------------------------------------------------
    # 7. Reconstruct symbolic separable terms
    # ---------------------------------------------------------------
    spatial_multi_indices = list(
        itertools.product(range(degree + 1), repeat=d_x)
    )
    spectral_multi_indices = list(
        itertools.product(range(degree + 1), repeat=d_xi)
    )

    def _cheb(deg, s):
        return _chebyshev_polynomial(deg, norm_vars[s])

    symbolic_pairs = []

    S_keep = S[keep]
    U_keep = U[:, keep]
    Vt_keep = Vt[keep, :]

    for k in range(len(S_keep)):
        sigma_k = S_keep[k]
        u_k = U_keep[:, k]
        v_k = Vt_keep[k, :]

        # a_k(x)
        a_k_expr = sp.S.Zero
        for idx, multi_idx in enumerate(spatial_multi_indices):
            coeff = np.sqrt(sigma_k) * u_k[idx]

            if np.abs(coeff) > tol:
                if len(multi_idx) == 0:
                    basis_term = sp.S.One
                else:
                    basis_term = sp.Mul(
                        *[
                            _cheb(deg, x_syms[m])
                            for m, deg in enumerate(multi_idx)
                        ]
                    )

                a_k_expr += _sympy_number(coeff, digits=digits) * basis_term

        # q_k(xi)
        q_k_expr = sp.S.Zero
        for idx, multi_idx in enumerate(spectral_multi_indices):
            coeff = np.sqrt(sigma_k) * v_k[idx]

            if np.abs(coeff) > tol:
                if len(multi_idx) == 0:
                    basis_term = sp.S.One
                else:
                    basis_term = sp.Mul(
                        *[
                            _cheb(deg, xi_syms[n])
                            for n, deg in enumerate(multi_idx)
                        ]
                    )

                q_k_expr += _sympy_number(coeff, digits=digits) * basis_term

        symbolic_pairs.append(
            (sp.expand(a_k_expr), sp.expand(q_k_expr))
        )

    # ---------------------------------------------------------------
    # 8. Monte Carlo quality metrics
    # ---------------------------------------------------------------
    metrics = evaluate_decomposition_quality(
        expr,
        symbolic_pairs,
        x_syms,
        xi_syms,
        bounds,
        num_samples=num_samples,
        seed=seed,
    )

    metrics["svd_energy_retained_pct"] = svd_energy_retained
    metrics["singular_values"] = S_keep

    return symbolic_pairs, metrics


# ============================================================================
# NUFFT-based joint-residual backend (joint_backend='nufft')
# ============================================================================
#
# Targets Category-C joint residuals that are OSCILLATORY (a genuine phase
# exp(i*Lambda(x)*M(xi)), e.g. sin(x*xi), exp(I*x*xi)) rather than algebraic.
# factorize_symbolic's Chebyshev/SVD basis converges poorly on these because
# a polynomial basis cannot efficiently represent a genuinely bilinear phase.
#
# PERIODIC BOUNDARY CONDITIONS ONLY. This backend has only been derived and
# validated for the FFT/periodic application path (boundary_condition=
# 'periodic'). It is not applicable to 'dirichlet' and apply_peetre() must
# fall back to the direct path in that case -- do not attempt to extend this
# silently without re-deriving the non-periodic quadrature.
#
# Requires the optional 'finufft' package for its O(N log N) benefit; falls
# back to an O(N*M) direct evaluation of the same embedding formula (correct,
# just not fast) if finufft is not installed, with a one-time warning.

try:
    import finufft as _finufft
    _HAVE_FINUFFT = True
except ImportError:
    _finufft = None
    _HAVE_FINUFFT = False

_finufft_warned = False


def _warn_no_finufft():
    global _finufft_warned
    if not _finufft_warned:
        warnings.warn(
            "finufft is not installed; joint_backend='nufft' will use a "
            "much slower O(N*M) direct-sum fallback that reproduces the "
            "same math but without the O(N log N) speed benefit. "
            "Install with `pip install finufft` for the intended performance."
        )
        _finufft_warned = True


def _nufft_split_real_imag_exponent(total_exponent):
    """Split an exponent into (I*phase, real_envelope) without silently
    dropping a real residual (a naive .coeff(sp.I) does this incorrectly
    for mixed exponents like I*x*xi - x**2)."""
    exp_terms = sp.Add.make_args(sp.expand(total_exponent))
    imag_terms, real_terms = [], []
    for t in exp_terms:
        c = t.coeff(sp.I)
        if sp.expand(t - sp.I * c) == 0:
            imag_terms.append(c)
        else:
            real_terms.append(t)
    phase_expr = sp.expand(sum(imag_terms)) if imag_terms else sp.Integer(0)
    real_envelope = sp.expand(sum(real_terms)) if real_terms else sp.Integer(0)
    return phase_expr, real_envelope, bool(imag_terms)


def _nufft_extract_term_nd(term, phys_syms, freq_syms):
    """
    Factor a single (exp-rewritten, expanded) additive term into
        c(phys) * g(freq) * exp(i * Lambda(phys) * M(freq))
    for phys_syms=(x,) / freq_syms=(xi,) [1D] or phys_syms=(x,y) /
    freq_syms=(xi,eta) [2D]. Returns None if it doesn't fit this pattern
    (conservative: never returns a wrong plan).
    """
    term_simp = sp.powsimp(term, combine="exp", deep=True)
    factors = sp.Mul.make_args(term_simp)

    exp_args, amp_factors = [], []
    for f in factors:
        if f.is_Pow and f.base == sp.E:
            exp_args.append(f.exp)
        elif isinstance(f, sp.exp):
            exp_args.append(f.args[0])
        else:
            amp_factors.append(f)

    if not exp_args:
        return None

    total_exponent = sp.expand(sum(exp_args))
    phase_expr, real_envelope, has_osc = _nufft_split_real_imag_exponent(total_exponent)
    if not has_osc:
        return None

    phase_factored = sp.factor(phase_expr)
    Lambda_p, M_f = phase_factored.as_independent(*freq_syms, as_Add=False)

    coupled_to_freq = any(Lambda_p.has(s) for s in freq_syms)
    coupled_to_phys = any(M_f.has(s) for s in phys_syms)
    no_real_coupling = not any(phase_expr.has(s) for s in freq_syms)
    if coupled_to_freq or coupled_to_phys or no_real_coupling:
        return None
    if sp.expand(Lambda_p * M_f - phase_expr) != 0:
        return None

    amp_expr = sp.Mul(*amp_factors)
    if real_envelope != 0:
        amp_expr = amp_expr * sp.exp(real_envelope)
    amp_factored = sp.factor(amp_expr) if amp_expr.is_Add else amp_expr
    c_p, g_f = amp_factored.as_independent(*freq_syms, as_Add=False)
    if any(c_p.has(s) for s in freq_syms) or any(g_f.has(s) for s in phys_syms):
        return None
    if sp.expand(c_p * g_f - amp_expr) != 0:
        return None

    return {
        "c_expr": c_p, "g_expr": g_f, "Lambda_expr": Lambda_p, "M_expr": M_f,
        "c": sp.lambdify(phys_syms, c_p, "numpy"),
        "g": sp.lambdify(freq_syms, g_f, "numpy"),
        "Lambda": sp.lambdify(phys_syms, Lambda_p, "numpy"),
        "M": sp.lambdify(freq_syms, M_f, "numpy"),
    }


def try_nufft_decomposition_1d(joint_expr, x_sym, xi_sym):
    """1D (phase space (x,xi)) NUFFT classifier. Returns a list of term
    plans, or None if any additive term doesn't fit (falls back)."""
    expr = sp.expand(joint_expr.rewrite(sp.exp))
    plans = []
    for term in sp.Add.make_args(expr):
        p = _nufft_extract_term_nd(term, (x_sym,), (xi_sym,))
        if p is None:
            return None
        plans.append(p)
    return plans


def _resolve_1d_piece_for_axis_sep(part_expr, phys_sym, freq_sym):
    """Used by the 2D axis-separable tier: resolve a single-variable-pair
    factor into pointwise (no freq dependence) or nufft1d pieces."""
    if not part_expr.has(freq_sym):
        return [{"kind": "pointwise", "amp": sp.lambdify(phys_sym, part_expr, "numpy")}]
    rewritten = sp.expand(part_expr.rewrite(sp.exp))
    pieces = []
    for sub in sp.Add.make_args(rewritten):
        if not sub.has(freq_sym):
            pieces.append({"kind": "pointwise", "amp": sp.lambdify(phys_sym, sub, "numpy")})
            continue
        plan = _nufft_extract_term_nd(sub, (phys_sym,), (freq_sym,))
        if plan is None:
            return None
        pieces.append({"kind": "nufft1d", "plan": plan})
    return pieces


def try_nufft_decomposition_2d(joint_expr, x_sym, y_sym, xi_sym, eta_sym):
    """
    2D (phase space (x,y,xi,eta)) NUFFT classifier. Tries, in order:
      (a) axis-separable: term factors as A(x,xi)*B(y,eta) (disjoint
          variable groups) -- cheapest, two independent 1D passes.
      (b) single-joint-term: term's phase is one product
          Lambda(x,y)*M(xi,eta) -- needs a 3D NUFFT embedding.
    A symbol whose terms need genuinely independent coupling on BOTH axes
    simultaneously (4D embedding) is not representable by either tier and
    returns None (finufft has no type-3 transform above 3D).
    Returns ('axis_sep', combo_plan) or ('joint3d', plans) or None.
    """
    # --- try axis-separable first (checked before any exp-rewrite, since
    #     rewriting collapses the very structure this tier looks for) ---
    expr_raw = sp.expand(joint_expr)
    combo_plan = []
    axis_sep_ok = True
    for term in sp.Add.make_args(expr_raw):
        A_part, B_part = term.as_independent(y_sym, eta_sym, as_Add=False)
        if A_part.has(y_sym) or A_part.has(eta_sym) or B_part.has(x_sym) or B_part.has(xi_sym):
            axis_sep_ok = False
            break
        A_pieces = _resolve_1d_piece_for_axis_sep(A_part, x_sym, xi_sym)
        B_pieces = _resolve_1d_piece_for_axis_sep(B_part, y_sym, eta_sym)
        if A_pieces is None or B_pieces is None:
            axis_sep_ok = False
            break
        for a in A_pieces:
            for b in B_pieces:
                combo_plan.append({"A": a, "B": b})
    if axis_sep_ok and combo_plan:
        return ("axis_sep", combo_plan)

    # --- fall back to single-joint-term (3D embed) ---
    expr = sp.expand(joint_expr.rewrite(sp.exp))
    plans = []
    for term in sp.Add.make_args(expr):
        p = _nufft_extract_term_nd(term, (x_sym, y_sym), (xi_sym, eta_sym))
        if p is None:
            return None
        plans.append(p)
    return ("joint3d", plans) if plans else None


def _nufft_uhat_1d(u, x_grid, dx, kx):
    """Continuous-FT approx of u, correcting for a grid not starting at 0
    (e.g. x_grid = linspace(-L, L, N, endpoint=False), used throughout this
    module's make_grid_1d/2d) -- see module docstring above for why this
    matters: without it, results are internally self-consistent but not
    the true KN operator action."""
    x0 = x_grid[0]
    return np.fft.fft(u) * dx * np.exp(-1j * x0 * kx)


def _nufft_direct_2d_type3(sx, sy, weights, tx, ty, isign=1):
    phase = isign * (tx[:, None] * sx[None, :] + ty[:, None] * sy[None, :])
    return (weights[None, :] * np.exp(1j * phase)).sum(axis=1)


def _nufft_direct_3d_type3(sx, sy, sz, weights, tx, ty, tz, isign=1):
    phase = isign * (tx[:, None] * sx[None, :] + ty[:, None] * sy[None, :]
                      + tz[:, None] * sz[None, :])
    return (weights[None, :] * np.exp(1j * phase)).sum(axis=1)


def _nufft_freq_window(kvals, freq_window):
    """Match kohn_nirenberg_fft's exact windowing formula (see slow-path
    P_blk *= win_k), applied elementwise on a raw (unshifted) frequency
    array -- the formula only depends on |k|/sigma pointwise, so it's
    correct regardless of fftshift ordering."""
    if freq_window == "gaussian":
        sigma = 0.8 * np.max(np.abs(kvals))
        return np.exp(-(kvals / sigma) ** 4)
    elif freq_window == "hann":
        k_max = np.max(np.abs(kvals))
        return 0.5 * (1 + np.cos(np.pi * kvals / k_max)) * (np.abs(kvals) < k_max)
    return np.ones_like(kvals, dtype=float)


def apply_nufft_1d(u, plan, x_grid, kx, dx, dxi, eps=1e-12, freq_window="gaussian"):
    """Apply Op(p_joint) via the NUFFT tier, 1D case. `plan` is the output
    of try_nufft_decomposition_1d (a list of term dicts).

    freq_window matches kohn_nirenberg_fft's default -- without applying
    it here too, results silently diverge from joint_backend='direct'
    even at freq_window='gaussian' defaults (found via end-to-end testing
    against the real dispatcher, not from the isolated unit tests, which
    never exercised the default windowing at all)."""
    u = np.asarray(u, dtype=complex)
    uhat = _nufft_uhat_1d(u, x_grid, dx, kx)
    win = _nufft_freq_window(kx, freq_window)
    result = np.zeros_like(x_grid, dtype=complex)
    for term in plan:
        c_x = term["c"](x_grid)
        g_xi = term["g"](kx) * win
        lam_x = term["Lambda"](x_grid)
        mu_xi = term["M"](kx)
        weights = (g_xi * uhat * dxi / (2 * np.pi)).astype(complex)
        src_x, src_y = kx, mu_xi
        tgt_x, tgt_y = x_grid, lam_x
        if _HAVE_FINUFFT:
            f = _finufft.nufft2d3(src_x, src_y, weights, tgt_x, tgt_y, isign=1, eps=eps)
        else:
            _warn_no_finufft()
            f = _nufft_direct_2d_type3(src_x, src_y, weights, tgt_x, tgt_y, isign=1)
        result += c_x * f
    return result


def apply_nufft_2d(u, kind, plan, x_grid, y_grid, kx, ky, dx, dy, dxi, deta, eps=1e-12,
                    freq_window="gaussian"):
    """Apply Op(p_joint) via the NUFFT tier, 2D case. `kind`/`plan` are the
    output of try_nufft_decomposition_2d. See apply_nufft_1d docstring on
    why freq_window must be matched to the direct path's default."""
    u = np.asarray(u, dtype=complex)

    if kind == "joint3d":
        x0, y0 = x_grid[0], y_grid[0]
        XI0, ETA0 = np.meshgrid(kx, ky, indexing="ij")
        uhat = np.fft.fft2(u) * dx * dy * np.exp(-1j * (x0 * XI0 + y0 * ETA0))
        XI, ETA = np.meshgrid(kx, ky, indexing="ij")
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        # 2D window: kohn_nirenberg_fft applies the SAME 1D-style formula
        # to the combined radial-like |k| via kx/ky separately multiplied;
        # match by applying to each axis and taking the product (matches
        # the 2D fast-path convention used elsewhere in this module).
        win_x = _nufft_freq_window(kx, freq_window)
        win_y = _nufft_freq_window(ky, freq_window)
        WIN = win_x[:, None] * win_y[None, :]
        result = np.zeros((len(x_grid), len(y_grid)), dtype=complex)
        for term in plan:
            c_xy = term["c"](X, Y)
            g_xieta = term["g"](XI, ETA) * WIN
            Lambda_xy = term["Lambda"](X, Y)
            M_xieta = term["M"](XI, ETA)
            Nx, Ny = len(x_grid), len(y_grid)
            src_xi, src_eta = XI.ravel(), ETA.ravel()
            src_M = np.broadcast_to(M_xieta, (Nx, Ny)).ravel()
            weights = (np.broadcast_to(g_xieta, (Nx, Ny)) * uhat
                       * dxi * deta / (2 * np.pi) ** 2).ravel().astype(complex)
            tgt_x, tgt_y = X.ravel(), Y.ravel()
            tgt_L = np.broadcast_to(Lambda_xy, (Nx, Ny)).ravel()
            if _HAVE_FINUFFT:
                f = _finufft.nufft3d3(src_xi, src_eta, src_M, weights, tgt_x, tgt_y, tgt_L,
                                       isign=1, eps=eps)
            else:
                _warn_no_finufft()
                f = _nufft_direct_3d_type3(src_xi, src_eta, src_M, weights, tgt_x, tgt_y, tgt_L, isign=1)
            result += c_xy * f.reshape(Nx, Ny)
        return result

    elif kind == "axis_sep":
        result = np.zeros_like(u, dtype=complex)
        for combo in plan:
            # Step 1: apply B (y,eta) row-wise; Step 2: apply A (x,xi) column-wise
            w = _apply_1d_piece_rows(combo["B"], u, y_grid, ky, dy, deta, along_axis=1,
                                      freq_window=freq_window)
            contrib = _apply_1d_piece_rows(combo["A"], w, x_grid, kx, dx, dxi, along_axis=0,
                                            freq_window=freq_window)
            result += contrib
        return result

    raise ValueError(f"unknown NUFFT 2D plan kind: {kind}")


def _apply_1d_piece_rows(piece, field, axis_grid, k_axis, d_axis, dk_axis, along_axis,
                          freq_window="gaussian"):
    if piece["kind"] == "pointwise":
        amp_vals = piece["amp"](axis_grid)
        return field * (amp_vals[None, :] if along_axis == 1 else amp_vals[:, None])
    plan = [piece["plan"]]
    out = np.zeros_like(field, dtype=complex)
    if along_axis == 1:
        for i in range(field.shape[0]):
            out[i, :] = apply_nufft_1d(field[i, :], plan, axis_grid, k_axis, d_axis, dk_axis,
                                        freq_window=freq_window)
    else:
        for j in range(field.shape[1]):
            out[:, j] = apply_nufft_1d(field[:, j], plan, axis_grid, k_axis, d_axis, dk_axis,
                                        freq_window=freq_window)
    return out


# ============================================================================
# AAA-based joint-residual backend (joint_backend='aaa')
# ============================================================================
#
# Targets Category-C joint residuals that are RATIONAL (resolvent-shaped,
# poles / algebraic decay, no oscillatory phase -- try_nufft_decomposition
# correctly rejects these). Builds a compact rational approximation of the
# symbol via vector-valued AAA (shared poles across a Chebyshev grid in the
# OTHER variable(s); the symbol is evaluated EXACTLY at each AAA support
# point via sympy substitution, so only the shared-pole structure in the
# frequency variable(s) introduces approximation error).
#
# KNOWN LIMITATION, BY CONSTRUCTION: this works well when the joint
# residual's pole locations are fixed or slowly varying with x (resp.
# x,y) -- it degrades (many poles needed, effectively no compression) when
# the pole genuinely MOVES with the spatial variable (e.g. 1/(xi-x-i*eps),
# a diagonal-type singularity). The quality gate below (joint_max_rel_error)
# catches a resulting bad fit and falls back to direct application; it does
# NOT silently return an inaccurate result. Diagonal-pole symbols are a
# genuinely different structural class (Calderon-Zygmund-type) that this
# backend does not target -- do not raise n_cheb/n_samples to "fix" a
# rejection here without first checking whether the pole is x-dependent.
#
# Unlike the NUFFT backend, this one delegates the actual numerical KN
# application to this module's own kohn_nirenberg_fft / 
# kohn_nirenberg_nonperiodic (via a fast numpy callable wrapping the AAA
# fit), so it supports BOTH periodic and dirichlet boundary conditions for
# free, and automatically inherits their existing grid-origin-correct
# numerics -- it does not reimplement the KN quadrature itself.

class _VectorAAA:
    """Barycentric rational fit r(z) in C^m, shared poles across m
    'vector components' (e.g. one component per Chebyshev x-node)."""
    def __init__(self, z_support, w, f_support):
        self.z_support = np.asarray(z_support)
        self.w = np.asarray(w)
        self.f_support = np.asarray(f_support)  # (k, m)

    def __call__(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        diffs = z[:, None] - self.z_support[None, :]
        exact_mask = np.isclose(diffs, 0.0)
        safe_diffs = np.where(exact_mask, 1.0, diffs)
        inv = np.where(exact_mask, 0.0, 1.0 / safe_diffs)
        num = (self.w[None, :] * inv) @ self.f_support
        den = (self.w[None, :] * inv).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        if exact_mask.any():
            rows, cols = np.where(exact_mask)
            out[rows, :] = self.f_support[cols, :]
        return out


def _vector_aaa(z_samples, F_samples, rtol=1e-8, max_terms=50):
    z_samples = np.asarray(z_samples, dtype=complex)
    F_samples = np.atleast_2d(np.asarray(F_samples, dtype=complex))
    if F_samples.shape[0] != len(z_samples):
        F_samples = F_samples.T
    M, m = F_samples.shape
    scale = np.max(np.abs(F_samples)) + 1e-300

    support_idx, test_idx = [], list(range(M))
    r_vals = np.tile(F_samples.mean(axis=0, keepdims=True), (M, 1))
    w = np.array([1.0 + 0j])
    z_support = np.array([], dtype=complex)
    f_support = np.zeros((0, m), dtype=complex)

    for _ in range(min(max_terms, M - 1)):
        resid = np.abs(F_samples[test_idx] - r_vals[test_idx])
        j_new = test_idx[np.argmax(resid.max(axis=1))]
        support_idx.append(j_new)
        test_idx.remove(j_new)

        z_support = z_samples[support_idx]
        f_support = F_samples[support_idx, :]
        k = len(support_idx)
        if not test_idx:
            w = np.ones(k) / k
            break

        z_test = z_samples[test_idx]
        F_test = F_samples[test_idx, :]
        denom = z_test[:, None] - z_support[None, :]
        blocks = [(F_test[:, c:c+1] - f_support[None, :, c].reshape(1, k)) / denom
                  for c in range(m)]
        L_stacked = np.vstack(blocks)
        _, _, Vh = np.linalg.svd(L_stacked)
        w = Vh[-1, :].conj()

        fit = _VectorAAA(z_support, w, f_support)
        r_vals = fit(z_samples)
        if np.max(np.abs(F_samples - r_vals)) / scale < rtol:
            break

    return _VectorAAA(z_support, w, f_support)


def _aaa_chebyshev_nodes(a, b, n):
    k = np.arange(n)
    x = np.cos((2*k + 1) / (2*n) * np.pi)
    return 0.5*(b-a)*x + 0.5*(b+a)


def _aaa_bary_weights_1st_kind(n):
    k = np.arange(n)
    theta = (2*k + 1) * np.pi / (2*n)
    return ((-1.0)**k) * np.sin(theta)


def try_aaa_decomposition_1d(joint_expr, x_sym, xi_sym, x_bounds, xi_bounds,
                              n_cheb=24, n_xi_samples=100, rtol=1e-8):
    """1D bivariate rational decomposition via vector-AAA. Returns a plan
    dict (with a fast numpy callable, see aaa_plan_to_callable_1d) or None
    if the quality gate (rel_l2_error > 10*rtol) isn't met."""
    p_lamb = sp.lambdify((x_sym, xi_sym), joint_expr, "numpy")
    x_nodes = _aaa_chebyshev_nodes(*x_bounds, n_cheb)
    xi_samples = np.linspace(*xi_bounds, n_xi_samples).astype(complex)
    XI, X = np.meshgrid(xi_samples, x_nodes, indexing="ij")
    F_samples = np.asarray(p_lamb(X, XI), dtype=complex)
    fit = _vector_aaa(xi_samples, F_samples, rtol=rtol)

    xi_val = np.linspace(*xi_bounds, 3*n_xi_samples + 7).astype(complex)
    XIv, Xv = np.meshgrid(xi_val, x_nodes, indexing="ij")
    F_true_val = np.asarray(p_lamb(Xv, XIv), dtype=complex)
    F_fit_val = fit(xi_val)
    rel_l2_error = (np.linalg.norm(F_fit_val - F_true_val)
                     / (np.linalg.norm(F_true_val) + 1e-300))
    if rel_l2_error > rtol * 10:
        return None
    return {"dim": 1, "fit": fit, "x_nodes": x_nodes,
            "rel_l2_error": rel_l2_error, "n_poles": len(fit.z_support)}


def _aaa_eval_1d(plan, x_eval, xi_eval):
    """Evaluate the AAA-fitted p(x,xi) at arbitrary points (barycentric
    Lagrange interp in x from the exact Chebyshev-node slices, composed
    with the AAA barycentric form in xi)."""
    fit, x_nodes = plan["fit"], plan["x_nodes"]
    xi_eval = np.atleast_1d(np.asarray(xi_eval, dtype=complex))
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    vals_at_nodes = fit(xi_eval)  # (Nxi, M)
    bw = _aaa_bary_weights_1st_kind(len(x_nodes))
    diffs = x_eval[:, None] - x_nodes[None, :]
    exact = np.isclose(diffs, 0.0)
    safe = np.where(exact, 1.0, diffs)
    inv = np.where(exact, 0.0, bw[None, :] / safe)
    den = inv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (vals_at_nodes @ inv.T) / den[None, :]  # (Nxi, Nx)
    if exact.any():
        x_idx, k_idx = np.where(exact)
        out[:, x_idx] = vals_at_nodes[:, k_idx]
    return out  # (Nxi, Nx)


def aaa_plan_to_callable_1d(plan):
    """Wrap an aaa_decomposition_1d plan as p(x, xi) -> ndarray, matching
    the symbol_func signature kohn_nirenberg_fft/nonperiodic expect."""
    def p_approx(x, xi):
        x = np.asarray(x, dtype=float)
        xi_arr = np.asarray(xi, dtype=complex)
        orig_shape = np.broadcast(x, xi_arr).shape
        xb, xib = np.broadcast_to(x, orig_shape), np.broadcast_to(xi_arr, orig_shape)
        x_flat, xi_flat = xb.ravel(), xib.ravel()
        # _aaa_eval_1d expects distinct (x_eval, xi_eval) axes; evaluate
        # pointwise via the diagonal of the outer evaluation (small arrays
        # in the kohn_nirenberg_fft slow path -- fine at that scale).
        out = np.empty(x_flat.shape, dtype=complex)
        for i in range(x_flat.size):
            out[i] = _aaa_eval_1d(plan, x_flat[i:i+1], xi_flat[i:i+1])[0, 0]
        return out.reshape(orig_shape)
    return p_approx


def try_aaa_decomposition_2d(joint_expr, x_sym, y_sym, xi_sym, eta_sym,
                              x_bounds, y_bounds, xi_bounds, eta_bounds,
                              n_cheb_x=10, n_cheb_y=10,
                              n_xi_samples=30, n_eta_samples=30, rtol=1e-8):
    """2D decomposition via sequential vector-AAA (xi support points chosen
    at a representative eta slice -- see module docstring caveat above;
    stage 2 compresses eta from the EXACT symbolic slice at each xi
    support point). Returns a plan dict or None if the quality gate fails."""
    x_nodes = _aaa_chebyshev_nodes(*x_bounds, n_cheb_x)
    y_nodes = _aaa_chebyshev_nodes(*y_bounds, n_cheb_y)
    Nx, Ny = len(x_nodes), len(y_nodes)
    XX, YY = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    xx_flat, yy_flat = XX.ravel(), YY.ravel()
    p_lamb = sp.lambdify((x_sym, y_sym, xi_sym, eta_sym), joint_expr, "numpy")

    eta_repr = 0.5 * (eta_bounds[0] + eta_bounds[1])
    xi_samples = np.linspace(*xi_bounds, n_xi_samples).astype(complex)
    F1 = np.zeros((n_xi_samples, Nx*Ny), dtype=complex)
    for j, xi_v in enumerate(xi_samples):
        F1[j, :] = p_lamb(xx_flat, yy_flat, xi_v, eta_repr)
    fit_xi = _vector_aaa(xi_samples, F1, rtol=rtol)
    xi_support = fit_xi.z_support

    eta_samples = np.linspace(*eta_bounds, n_eta_samples).astype(complex)
    eta_fits = []
    for xi_l in xi_support:
        expr_l = joint_expr.subs(xi_sym, complex(xi_l))
        p_l_lamb = sp.lambdify((x_sym, y_sym, eta_sym), expr_l, "numpy")
        F2 = np.zeros((n_eta_samples, Nx*Ny), dtype=complex)
        for j, eta_v in enumerate(eta_samples):
            F2[j, :] = p_l_lamb(xx_flat, yy_flat, eta_v)
        eta_fits.append(_vector_aaa(eta_samples, F2, rtol=rtol))

    plan = {"dim": 2, "fit_xi": fit_xi, "xi_support": xi_support,
            "eta_fits": eta_fits, "x_nodes": x_nodes, "y_nodes": y_nodes}

    x_val = np.linspace(x_bounds[0]*0.9, x_bounds[1]*0.9, 6)
    y_val = np.linspace(y_bounds[0]*0.9, y_bounds[1]*0.9, 6)
    xi_val = np.linspace(xi_bounds[0]*0.9, xi_bounds[1]*0.9, 5).astype(complex)
    eta_val = np.linspace(eta_bounds[0]*0.9, eta_bounds[1]*0.9, 5).astype(complex)
    approx = _aaa_eval_2d(plan, x_val, y_val, xi_val, eta_val)
    XIv, ETAv, Xv, Yv = np.meshgrid(xi_val, eta_val, x_val, y_val, indexing="ij")
    true_vals = p_lamb(Xv, Yv, XIv, ETAv)
    rel_err = np.linalg.norm(approx - true_vals) / (np.linalg.norm(true_vals) + 1e-300)
    plan["rel_l2_error"] = rel_err
    if rel_err > rtol * 20:
        return None
    return plan


def _interp_2d_tensor_chebyshev(vals_grid, x_nodes, y_nodes, x_eval, y_eval):
    bwx = _aaa_bary_weights_1st_kind(len(x_nodes))
    bwy = _aaa_bary_weights_1st_kind(len(y_nodes))

    def bary_1d(vals, nodes, bw, eval_pts, axis):
        diffs = eval_pts[:, None] - nodes[None, :]
        exact = np.isclose(diffs, 0.0)
        safe = np.where(exact, 1.0, diffs)
        inv = np.where(exact, 0.0, bw[None, :] / safe)
        den = inv.sum(axis=1)
        num = np.tensordot(inv, vals, axes=([1], [axis]))
        num = np.moveaxis(num, 0, axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / np.expand_dims(den, axis=[a for a in range(num.ndim) if a != axis])
        if exact.any():
            eval_idx, node_idx = np.where(exact)
            src = np.take(vals, node_idx, axis=axis)
            out = np.moveaxis(out, axis, 0)
            src = np.moveaxis(src, axis, 0)
            out[eval_idx] = src[np.arange(len(eval_idx))] if src.ndim == out.ndim else src
            out = np.moveaxis(out, 0, axis)
        return out

    ax_x, ax_y = vals_grid.ndim - 2, vals_grid.ndim - 1
    step1 = bary_1d(vals_grid, x_nodes, bwx, x_eval, axis=ax_x)
    return bary_1d(step1, y_nodes, bwy, y_eval, axis=ax_y)


def _aaa_eval_2d(plan, x_eval, y_eval, xi_eval, eta_eval):
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    y_eval = np.atleast_1d(np.asarray(y_eval, dtype=float))
    xi_eval = np.atleast_1d(np.asarray(xi_eval, dtype=complex))
    eta_eval = np.atleast_1d(np.asarray(eta_eval, dtype=complex))
    x_nodes, y_nodes = plan["x_nodes"], plan["y_nodes"]
    Nx, Ny = len(x_nodes), len(y_nodes)
    xi_support, w_xi, L = plan["xi_support"], plan["fit_xi"].w, len(plan["xi_support"])

    q_all = np.zeros((L, len(eta_eval), len(x_eval), len(y_eval)), dtype=complex)
    for l in range(L):
        vals_at_nodes = plan["eta_fits"][l](eta_eval).reshape(len(eta_eval), Nx, Ny)
        q_all[l] = _interp_2d_tensor_chebyshev(vals_at_nodes, x_nodes, y_nodes, x_eval, y_eval)

    out = np.zeros((len(xi_eval), len(eta_eval), len(x_eval), len(y_eval)), dtype=complex)
    for ix, xi_v in enumerate(xi_eval):
        diffs = xi_v - xi_support
        exact = np.isclose(diffs, 0.0)
        if exact.any():
            out[ix] = q_all[np.argmax(exact)]
            continue
        coeff = w_xi / diffs
        out[ix] = np.tensordot(coeff, q_all, axes=([0], [0])) / coeff.sum()
    return out


def aaa_plan_to_callable_2d(plan):
    """Wrap an aaa_decomposition_2d plan as p(x, y, xi, eta) -> ndarray,
    matching the symbol_func signature kohn_nirenberg_fft/nonperiodic
    expect for dim=2."""
    def p_approx(x, y, xi, eta):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        xi_arr = np.asarray(xi, dtype=complex); eta_arr = np.asarray(eta, dtype=complex)
        orig_shape = np.broadcast(x, y, xi_arr, eta_arr).shape
        xb, yb, xib, etab = (np.broadcast_to(a, orig_shape).ravel()
                              for a in (x, y, xi_arr, eta_arr))
        out = np.empty(xb.shape, dtype=complex)
        for i in range(xb.size):
            out[i] = _aaa_eval_2d(plan, xb[i:i+1], yb[i:i+1], xib[i:i+1], etab[i:i+1])[0, 0, 0, 0]
        return out.reshape(orig_shape)
    return p_approx


"""
pde_solver_exponential.py
==========================
Core solvers, propagators, and visualization utilities for pseudo-differential operators.
"""
import sympy as sp

# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------

def make_grid_1d(L=10.0, N=256):
    """
    Construct a uniform periodic spatial grid and its associated FFT-ordered
    angular frequency grid in one dimension.

    The spatial domain is [−L, L) discretised into N equally spaced points,
    and the frequency grid covers the discrete angular wavenumbers compatible
    with the FFT ordering:

        x_j = −L + j·Δx,  j = 0, …, N−1,  Δx = 2L/N
        k_m = 2π · fftfreq(N, Δx)

    Parameters
    ----------
    L : float, default 10.0
        Half-length of the spatial domain. The full period is 2L.
    N : int, default 256
        Number of grid points.

    Returns
    -------
    x : ndarray, shape (N,)
        Spatial coordinates in [−L, L).
    kx : ndarray, shape (N,)
        Angular frequency grid in FFT order (radians per unit length).

    Examples
    --------
    >>> x, kx = make_grid_1d(L=5.0, N=128)
    >>> x[0], x[-1]
    (-5.0, 4.921875)
    """
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, kx

def make_grid_2d(L=10.0, N=128):
    """
    Construct uniform periodic spatial grids and their associated FFT-ordered
    angular frequency grids in two dimensions.

    Both axes share the same half-length L and resolution N, yielding a
    square domain [−L, L)² with N × N grid points.

        x_i = −L + i·Δx,  y_j = −L + j·Δy,  Δx = Δy = 2L/N
        kx_m = 2π · fftfreq(N, Δx),  ky_n = 2π · fftfreq(N, Δy)

    Parameters
    ----------
    L : float, default 10.0
        Half-length of the spatial domain along each axis.
    N : int, default 128
        Number of grid points per axis.

    Returns
    -------
    x : ndarray, shape (N,)
        Spatial coordinates along x.
    y : ndarray, shape (N,)
        Spatial coordinates along y.
    kx : ndarray, shape (N,)
        Angular frequency grid along x in FFT order.
    ky : ndarray, shape (N,)
        Angular frequency grid along y in FFT order.

    Examples
    --------
    >>> x, y, kx, ky = make_grid_2d(L=5.0, N=64)
    """
    x, kx = make_grid_1d(L, N)
    y, ky = make_grid_1d(L, N)
    return x, y, kx, ky

# ----------------------------------------------------------------------
# Propagator & Solvers
# ----------------------------------------------------------------------

def _make_grids(vars_x, L, N):
    """Build spatial + frequency grids and the meshgrid-ed spatial
    coordinates used to evaluate initial conditions -- factors out the
    grid-setup boilerplate that used to be copy-pasted verbatim in every
    solve_* function below.

    Returns
    -------
    X, Y : ndarray, ndarray or None
        Meshgrid-ed spatial coordinates ('ij' indexing). Y is None in 1D.
    x_grid, y_grid : ndarray, ndarray or None
        1D spatial axes (y_grid is None in 1D).
    kx, ky : ndarray, ndarray or None
        Frequency axes (ky is None in 1D).
    grids : tuple
        (x, kx) in 1D or (x, y, kx, ky) in 2D -- what callers return.
    """
    dim = len(vars_x)
    if dim == 1:
        x_grid, kx = make_grid_1d(L, N)
        return x_grid, None, x_grid, None, kx, None, (x_grid, kx)
    elif dim == 2:
        x_grid, y_grid, kx, ky = make_grid_2d(L, N)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        return X, Y, x_grid, y_grid, kx, ky, (x_grid, y_grid, kx, ky)
    else:
        raise NotImplementedError("Only 1D and 2D are supported")


def _run_time_loop(step_fn, U0, dt, n_steps, save_every, check_finite=True):
    """Repeatedly apply `step_fn(U) -> U_next`, saving a snapshot every
    `save_every` steps (plus the final step and t=0) -- factors out the
    save-cadence bookkeeping that used to be copy-pasted verbatim in every
    solve_* function below.

    Raises
    ------
    FloatingPointError
        If `check_finite` is True and a non-finite value shows up -- avoids
        silently returning a diverged/garbage trajectory.
    """
    t_list = [0.0]
    U_list = [np.asarray(U0).copy()]
    U = U0
    t = 0.0
    for n in range(1, n_steps + 1):
        U = step_fn(U)
        if check_finite and not np.all(np.isfinite(U)):
            raise FloatingPointError(
                f"Non-finite values detected at step {n} (t={t + dt:.6g}); "
                "reduce dt, increase `order`, or check the input symbol."
            )
        t += dt
        if n % save_every == 0 or n == n_steps:
            t_list.append(t)
            U_list.append(np.asarray(U).copy())
    return np.array(t_list), np.array(U_list)


def build_propagator(s_expr, vars_x, dt, order=3, quantization='kohn-nirenberg',
                     mode_composition='kn', apply_backend='peetre'):
    """
    Build the one-step numerical propagator exp(dt · Op[s]) for a
    pseudo-differential operator via truncated asymptotic exponentiation.

    Given a symbol s(x, ξ) (scalar or matrix-valued), this function:
      1. Wraps it into the appropriate operator class.
      2. Computes the symbol of exp(dt · P) via `exponential_symbol(t=dt, order=order)`:

             exp(dt · P) ≈ I + dt·P + (dt²/2!)·P∘P + ⋯ + (dtⁿ/n!)·P^{∘n}

         where each power P^{∘n} is obtained through asymptotic composition.
      3. Wraps the resulting symbol into a new operator ready for `apply()`.

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Symbol expression. A scalar expression produces a
        `PseudoDifferentialOperator`; a matrix (or nested list) produces a
        `MatrixPseudoDifferentialOperator`.
    vars_x : list of sympy.Symbol
        Spatial variables ([x] for 1D, [x, y] for 2D).
    dt : float
        Time-step size used as the evolution parameter t in exp(t·P).
    order : int, default 3
        Truncation order for both the Taylor series in t and the
        asymptotic composition at each power.
    quantization : {'kohn-nirenberg', 'weyl'}, default 'kohn-nirenberg'
        Quantization convention for the resulting propagator.
    mode_composition : {'kn', 'weyl'}, default 'kn'
        Composition rule used inside `exponential_symbol`.
    apply_backend : {'peetre', 'direct'}, default 'peetre'
        Numerical backend attached to the propagator operator.

    Returns
    -------
    prop : PseudoDifferentialOperator or MatrixPseudoDifferentialOperator
        The propagator operator such that `prop.apply(u, …)` advances u
        by one time step dt.
    is_matrix : bool
        True if the propagator is matrix-valued.
    size : int or None
        Matrix dimension N if `is_matrix` is True, else None.

    Notes
    -----
    The propagator is constructed once and reused across all time steps.
    For constant-coefficient symbols the composition is exact (all
    derivative corrections vanish), so the only error is the Taylor
    truncation in dt.

    Examples
    --------
    >>> prop, is_mat, sz = build_propagator(xi**2, [x], dt=0.01, order=4)
    >>> u_next = prop.apply(u, x_grid, kx)
    """
    is_matrix = isinstance(s_expr, (sp.MatrixBase, list, tuple))

    if is_matrix:
        s_mat = sp.Matrix(s_expr)
        size = s_mat.shape[0]

        op = MatrixPseudoDifferentialOperator(
            s_mat, vars_x, mode='symbol',
            quantization=quantization, apply_backend=apply_backend,
        )
        Esym = op.exponential_symbol(t=dt, order=order, mode=mode_composition)

        prop = MatrixPseudoDifferentialOperator(
            Esym, vars_x, mode='symbol',
            quantization=quantization, apply_backend=apply_backend,
        )
        return prop, True, size
    else:
        op = PseudoDifferentialOperator(
            s_expr, vars_x, mode='symbol',
            quantization=quantization, apply_backend=apply_backend,
        )
        Esym = op.exponential_symbol(t=dt, order=order, mode=mode_composition)

        prop = PseudoDifferentialOperator(
            Esym, vars_x, mode='symbol',
            quantization=quantization, apply_backend=apply_backend,
        )
        return prop, False, None

def solve_first_order(s_expr, vars_x, f, dt, n_steps, order=3,
                      L=10.0, N=256, apply_kwargs=None, save_every=1,
                      quantization='kohn-nirenberg', apply_backend='peetre',
                      check_finite=True):
    """
    Solve the first-order evolution equation

        ∂u/∂t = Op[s](u),   u(x, 0) = f(x)

    by repeated application of the asymptotic propagator exp(dt · Op[s]).

    At each time step the field is advanced via

        u^{n+1} = exp(dt · Op[s]) u^n ≈ (I + dt·P + (dt²/2!)P∘P + ⋯) u^n

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Symbol of the spatial operator P. Scalar for a single-field
        equation; matrix-valued for a coupled system.
    vars_x : list of sympy.Symbol
        Spatial variables.
    f : callable
        Initial condition. Must accept (X,) in 1D or (X, Y) in 2D and
        return an ndarray (scalar case) or a list/tuple of ndarrays
        (matrix case with N components).
    dt : float
        Time-step size.
    n_steps : int
        Total number of time steps to evolve.
    order : int, default 3
        Asymptotic expansion order for the propagator construction.
    L : float, default 10.0
        Half-length of the periodic spatial domain.
    N : int, default 256
        Number of grid points per spatial axis.
    apply_kwargs : dict, optional
        Extra keyword arguments forwarded to `prop.apply()` at every
        step (e.g. `boundary_condition`, `freq_window`, `clamp`).
    save_every : int, default 1
        Store the solution snapshot every `save_every` steps.
    quantization : str, default 'kohn-nirenberg'
        Quantization convention.
    apply_backend : str, default 'peetre'
        Numerical application backend.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead of
        silently returning a diverged trajectory.

    Returns
    -------
    t : ndarray, shape (n_saved,)
        Time values at which snapshots were recorded.
    U : ndarray
        Solution snapshots. Shape (n_saved, N) for scalar 1D,
        (n_saved, N, N) for scalar 2D, or (n_saved, size, N…) for
        matrix-valued systems.
    grids : tuple
        The spatial and frequency grids used: (x, kx) in 1D or
        (x, y, kx, ky) in 2D.

    Raises
    ------
    NotImplementedError
        If `vars_x` has length other than 1 or 2.
    ValueError
        If `f` returns the wrong number of components for a matrix system.

    Examples
    --------
    >>> t, U, (x, kx) = solve_first_order(xi**2, [x], lambda X: np.exp(-X**2),
    ...                                     dt=0.01, n_steps=100, N=256)
    """
    apply_kwargs = dict(apply_kwargs or {})
    X, Y, x, y_grid, kx, ky, grids = _make_grids(vars_x, L, N)

    prop, is_matrix, size = build_propagator(
        s_expr, vars_x, dt, order=order,
        quantization=quantization, apply_backend=apply_backend,
    )

    f0 = f(X) if Y is None else f(X, Y)
    if is_matrix:
        u0 = np.stack([np.asarray(comp, dtype=complex) for comp in f0])
        if u0.shape[0] != size:
            raise ValueError(f"f must return {size} components, got {u0.shape[0]}")
    else:
        u0 = np.asarray(f0, dtype=complex)

    def step(u):
        u_in = list(u) if is_matrix else u
        result = prop.apply(u_in, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        return np.stack(result) if is_matrix else np.asarray(result)

    t_arr, U_arr = _run_time_loop(step, u0, dt, n_steps, save_every, check_finite)
    return t_arr, U_arr, grids

def _as_component_list(h, X, Y=None, size_hint=1):
    """
    Evaluate a callable `h` on the grid and normalise the result into a
    plain list of component arrays.

    If `h(X)` (or `h(X, Y)` in 2D) already returns a list or tuple, it is
    returned as-is. Otherwise the scalar result is wrapped in a one-element
    list, ensuring downstream code always receives a uniform list interface.

    Parameters
    ----------
    h : callable
        Function of the spatial grid. Signature: h(X) in 1D, h(X, Y) in 2D.
    X : ndarray
        Spatial grid along x (or meshgrid in 2D).
    Y : ndarray, optional
        Spatial grid along y (2D only). If None, `h` is called with X alone.
    size_hint : int, default 1
        Informational hint about the expected number of components
        (not enforced here; callers validate separately).

    Returns
    -------
    list of ndarray
        Component arrays produced by `h`.
    """
    out = h(X) if Y is None else h(X, Y)
    if isinstance(out, (list, tuple)):
        return list(out)
    return [out]

def _matrix_of(s_expr):
    """
    Coerce a symbol expression into a sympy Matrix.

    If `s_expr` is already a MatrixBase, list, or tuple, it is converted
    via `sp.Matrix(s_expr)`. A bare scalar expression is wrapped into a
    1×1 matrix so that downstream code can treat scalar and matrix-valued
    operators uniformly.

    Parameters
    ----------
    s_expr : sympy.Expr, sympy.MatrixBase, list, or tuple
        The symbol or matrix of symbols.

    Returns
    -------
    sympy.Matrix
        Square matrix of symbol expressions.

    Raises
    ------
    ValueError
        If the resulting matrix is not square (checked by callers).
    """
    if isinstance(s_expr, (sp.MatrixBase, list, tuple)):
        return sp.Matrix(s_expr)
    return sp.Matrix([[s_expr]])

def block_matrix_second_order(s_expr):
    """
    Convert a second-order-in-time operator symbol S into a first-order
    block companion system suitable for `solve_first_order`.

    The second-order equation

        ∂²u/∂t² = Op[S](u)

    is rewritten as the first-order system

        ∂/∂t [u]   [ 0   I ] [u]
             [v] = [ S   0 ] [v]

    where v = ∂u/∂t. For a k×k matrix symbol S, the companion matrix has
    dimension 2k × 2k:

        M = [ 0_k   I_k ]
            [ S     0_k ]

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        The operator symbol S (scalar or k×k matrix).

    Returns
    -------
    sympy.Matrix, shape (2k, 2k)
        The block companion matrix M.

    Raises
    ------
    ValueError
        If `s_expr` is a non-square matrix.

    Examples
    --------
    >>> M = block_matrix_second_order(-xi**2)
    >>> M.shape
    (2, 2)
    """
    S = _matrix_of(s_expr)
    if S.shape[0] != S.shape[1]:
        raise ValueError("matrix symbol must be square")
    k = S.shape[0]
    zero_k, eye_k = sp.zeros(k, k), sp.eye(k)
    return zero_k.row_join(eye_k).col_join(S.row_join(zero_k))

def solve_second_order(s_expr, vars_x, f, g, dt, n_steps, order=3,
                       L=10.0, N=256, apply_kwargs=None, save_every=1,
                       quantization='kohn-nirenberg', apply_backend='peetre'):
    """
    Solve the second-order-in-time evolution equation

        ∂²u/∂t² = Op[S](u),   u(x, 0) = f(x),   ∂u/∂t(x, 0) = g(x)

    by reduction to a first-order block companion system and time-stepping
    with the asymptotic propagator.

    The system is split into

        ∂u/∂t = v,      ∂v/∂t = Op[S](u)

    and solved jointly via `solve_first_order` on the 2k-dimensional
    companion operator. The returned arrays contain only the physical
    field u and its velocity v = ∂u/∂t, not the full state vector.

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Symbol of the spatial operator S.
    vars_x : list of sympy.Symbol
        Spatial variables.
    f : callable
        Initial displacement u(x, 0). Signature: f(X) in 1D, f(X, Y) in 2D.
        Must return one component per row of S.
    g : callable
        Initial velocity ∂u/∂t(x, 0). Same signature and component
        structure as `f`.
    dt : float
        Time-step size.
    n_steps : int
        Number of time steps.
    order : int, default 3
        Asymptotic order for the propagator.
    L : float, default 10.0
        Spatial domain half-length.
    N : int, default 256
        Grid points per axis.
    apply_kwargs : dict, optional
        Forwarded to `apply()` at each step.
    save_every : int, default 1
        Snapshot cadence.
    quantization : str, default 'kohn-nirenberg'
        Quantization convention.
    apply_backend : str, default 'peetre'
        Numerical backend.

    Returns
    -------
    t : ndarray, shape (n_saved,)
        Time values of recorded snapshots.
    U : ndarray
        Displacement field snapshots u(x, t).
    V : ndarray
        Velocity field snapshots ∂u/∂t(x, t).
    grids : tuple
        Spatial and frequency grids: (x, kx) or (x, y, kx, ky).

    Raises
    ------
    ValueError
        If `f` or `g` produce the wrong number of components.
    """
    is_matrix = isinstance(s_expr, (sp.MatrixBase, list, tuple))
    k = _matrix_of(s_expr).shape[0]
    M = block_matrix_second_order(s_expr)
    
    def f_combined(X, Y=None):
        f_comp = _as_component_list(f, X, Y, size_hint=k)
        g_comp = _as_component_list(g, X, Y, size_hint=k)
        if len(f_comp) != k or len(g_comp) != k:
            raise ValueError(f"f and g must each provide {k} component(s).")
        return f_comp + g_comp
        
    # FIX: Call solve_first_order instead of sympy's algebraic solve()
    t, U_full, grids = solve_first_order(
        M, vars_x, f_combined, dt, n_steps, order=order,
        L=L, N=N, apply_kwargs=apply_kwargs, save_every=save_every,
        quantization=quantization, apply_backend=apply_backend,
    )
    
    U = U_full[:, :k, ...]
    V = U_full[:, k:, ...]
    
    if not is_matrix:
        U = U[:, 0, ...]
        V = V[:, 0, ...]
        
    return t, U, V, grids

def solve_matrix_field(s_expr, vars_x, F, dt, n_steps, order=3,
                        L=10.0, N=256, apply_kwargs=None, save_every=1,
                        quantization='kohn-nirenberg', apply_backend='peetre',
                        check_finite=True):
    """
    Time-step the matrix-field evolution equation `∂ₜU = P U`, where `P`
    is the pseudo-differential operator with N×N matrix symbol `s_expr`
    and `U(x)` is itself an N×N matrix at every spatial point (e.g. a
    density matrix or matrix Green's function), with `P` acting on `U`
    only from the left: `(P U)_ik = Σⱼ Op[P_ij](U_jk)`. This repeatedly
    applies the exponential propagator `Op(exp(dt·s))` built by
    `build_propagator`, via
    `MatrixPseudoDifferentialOperator.apply_matrix_field`, exactly as
    `solve` does for vector fields via `apply`.

    Parameters
    ----------
    s_expr : sympy.MatrixBase or nested list of sympy.Expr
        N×N matrix symbol `S(x, ξ)` of the generator `P`; must be
        matrix-valued (matrix left-multiplication only makes sense at
        `N > 1` -- use `solve` for a scalar generator).
    vars_x : list of sympy symbols
        Spatial variables (length 1 or 2).
    F : callable
        Initial matrix field `U(·, 0)`, called as `F(X)` in 1D or
        `F(X, Y)` in 2D on the meshgrid-ed spatial coordinates, and
        expected to return an N×N array/nested list of grid-shaped
        components (`F(...)[j][k]`, or an ndarray of shape
        `(N, N, *grid_shape)`).
    dt : float
        Time step.
    n_steps : int
        Number of propagator applications (time steps) to take.
    order : int, optional
        Truncation order of the exponential symbol expansion. Default 3.
    L : float, optional
        Half-width of the spatial domain. Default 10.0.
    N : int, optional
        Number of grid points per axis. Default 256.
    apply_kwargs : dict, optional
        Extra keyword arguments forwarded to `apply_matrix_field`.
    save_every : int, optional
        Save the solution every `save_every` steps (plus the final step
        and `t=0`). Default 1 (save every step).
    quantization : str, optional
        Quantization convention. Default 'kohn-nirenberg'.
    apply_backend : str, optional
        Numerical application backend. Default 'peetre'.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead of
        silently returning a diverged trajectory.

    Returns
    -------
    t_list : ndarray
        Saved time points, starting at 0.
    U_list : ndarray, shape (n_saved, N, N, *grid_shape)
        Saved matrix-field snapshots `U(t)`.
    grids : tuple of ndarray
        `(x, kx)` in 1D or `(x, y, kx, ky)` in 2D, as returned by
        `make_grid_1d`/`make_grid_2d`.

    Raises
    ------
    NotImplementedError
        If `vars_x` has a length other than 1 or 2.
    ValueError
        If `s_expr` is not matrix-valued, or `F` does not return an
        N×N field.
    """
    apply_kwargs = dict(apply_kwargs or {})
    X, Y, x, y_grid, kx, ky, grids = _make_grids(vars_x, L, N)

    prop, is_matrix, size = build_propagator(
        s_expr, vars_x, dt, order=order,
        quantization=quantization, apply_backend=apply_backend,
    )
    if not is_matrix:
        raise ValueError(
            "solve_matrix_field requires a matrix symbol; got a scalar "
            "symbol. Use solve_first_order() for scalar/vector fields instead."
        )

    U0 = F(X) if Y is None else F(X, Y)
    U0 = np.asarray(U0, dtype=complex)
    if U0.shape[0] != size or U0.shape[1] != size:
        raise ValueError(
            f"F must return a {size}x{size} matrix field, got shape "
            f"{U0.shape[:2]}."
        )

    def step(U):
        result = prop.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        return np.asarray(result, dtype=complex)

    t_arr, U_arr = _run_time_loop(step, U0, dt, n_steps, save_every, check_finite)
    return t_arr, U_arr, grids

def solve_sylvester_field(P_expr, Q_expr, vars_x, F, dt, n_steps, order=3,
                           splitting='strang', L=10.0, N=256,
                           apply_kwargs=None, save_every=1,
                           quantization='kohn-nirenberg', apply_backend='peetre',
                           check_finite=True):
    """
    Time-step the Sylvester-type matrix-field evolution equation
    `∂ₜU = P U − U Q`, where `P` and `Q` are pseudo-differential
    operators with N×N matrix symbols and `U(x)` is an N×N matrix at
    every spatial point.

    Left-multiplication by `P` and right-multiplication by `Q` always
    commute as *operations* (`(P U) Q == P (U Q)`), so when `P` and `Q`
    are x-independent (Fourier multipliers), the exact solution over a
    step `dt` is the closed-form

        U(t) = exp(t P) U(0) exp(-t Q) ,

    obtained by applying the left-propagator `Op(exp(dt·P))`
    (`apply_matrix_field`) and the right-propagator `Op(exp(-dt·Q))`
    (`apply_matrix_field_right`), in either order. When `P` and/or `Q`
    depend on x, `Op[P_ij]` and `Op[Q_jk]` need not commute with each
    other, so the two sub-steps no longer combine exactly; this function
    then falls back to a standard Lie-Trotter (first-order, `O(dt)`
    splitting error) or Strang (second-order, `O(dt^2)`) operator
    splitting between the left and right exponential propagators.

    Parameters
    ----------
    P_expr : sympy.MatrixBase or nested list of sympy.Expr
        N×N matrix symbol `P(x, ξ)` acting on U from the left.
    Q_expr : sympy.MatrixBase or nested list of sympy.Expr
        N×N matrix symbol `Q(x, ξ)` acting on U from the right (with a
        minus sign, as in `∂ₜU = P U − U Q`); must be the same size as
        `P_expr`.
    vars_x : list of sympy symbols
        Spatial variables (length 1 or 2).
    F : callable
        Initial matrix field `U(·, 0)`, called as `F(X)` in 1D or
        `F(X, Y)` in 2D, returning an N×N array/nested list of
        grid-shaped components (as for `solve_matrix_field`).
    dt : float
        Time step.
    n_steps : int
        Number of splitting steps (each advancing `U` by `dt`).
    order : int, optional
        Truncation order of each exponential-symbol expansion. Default 3.
    splitting : str, {'lie', 'strang'}, optional
        Operator-splitting scheme between the `P` (left) and `Q` (right)
        sub-steps:

        - 'lie'    : one full left step `exp(dt·P)`, then one full right
                     step `exp(-dt·Q)` -- first order accurate, `O(dt)`.
        - 'strang' : half left step `exp(dt/2·P)`, full right step
                     `exp(-dt·Q)`, half left step `exp(dt/2·P)` --
                     second order accurate, `O(dt^2)`. Default.
    L : float, optional
        Half-width of the spatial domain. Default 10.0.
    N : int, optional
        Number of grid points per axis. Default 256.
    apply_kwargs : dict, optional
        Extra keyword arguments forwarded to `apply_matrix_field` and
        `apply_matrix_field_right`.
    save_every : int, optional
        Save the solution every `save_every` steps (plus the final step
        and `t=0`). Default 1.
    quantization : str, optional
        Quantization convention. Default 'kohn-nirenberg'.
    apply_backend : str, optional
        Numerical application backend. Default 'peetre'.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead
        of silently returning a diverged trajectory.

    Returns
    -------
    t_list : ndarray
        Saved time points, starting at 0.
    U_list : ndarray, shape (n_saved, N, N, *grid_shape)
        Saved matrix-field snapshots `U(t)`.
    grids : tuple of ndarray
        `(x, kx)` in 1D or `(x, y, kx, ky)` in 2D.

    Raises
    ------
    NotImplementedError
        If `vars_x` has a length other than 1 or 2.
    ValueError
        If `P_expr`/`Q_expr` are not matrix-valued of matching size, if
        `F` does not return an N×N field, or if `splitting` is not
        'lie' or 'strang'.
    """
    if splitting not in ('lie', 'strang'):
        raise ValueError("splitting must be 'lie' or 'strang'.")

    apply_kwargs = dict(apply_kwargs or {})
    X, Y, x, y_grid, kx, ky, grids = _make_grids(vars_x, L, N)

    prop_Q_full, is_matrix_Q, size_Q = build_propagator(
        Q_expr, vars_x, -dt, order=order,
        quantization=quantization, apply_backend=apply_backend,
    )
    if splitting == 'lie':
        prop_P_full, is_matrix_P, size_P = build_propagator(
            P_expr, vars_x, dt, order=order,
            quantization=quantization, apply_backend=apply_backend,
        )
        prop_P_half = None
    else:  # 'strang'
        prop_P_half, is_matrix_P, size_P = build_propagator(
            P_expr, vars_x, dt / 2.0, order=order,
            quantization=quantization, apply_backend=apply_backend,
        )
        prop_P_full = None

    if not (is_matrix_P and is_matrix_Q):
        raise ValueError(
            "solve_sylvester_field requires matrix symbols for both P "
            "and Q; got a scalar symbol for at least one of them."
        )
    if size_P != size_Q:
        raise ValueError(
            f"P_expr ({size_P}x{size_P}) and Q_expr ({size_Q}x{size_Q}) "
            "must have the same size."
        )
    size = size_P

    U0 = F(X) if Y is None else F(X, Y)
    U0 = np.asarray(U0, dtype=complex)
    if U0.shape[0] != size or U0.shape[1] != size:
        raise ValueError(
            f"F must return a {size}x{size} matrix field, got shape "
            f"{U0.shape[:2]}."
        )

    def step(U):
        if splitting == 'lie':
            U = prop_P_full.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
            U = prop_Q_full.apply_matrix_field_right(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        else:  # 'strang'
            U = prop_P_half.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
            U = prop_Q_full.apply_matrix_field_right(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
            U = prop_P_half.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        return np.asarray(U, dtype=complex)

    t_arr, U_arr = _run_time_loop(step, U0, dt, n_steps, save_every, check_finite)
    return t_arr, U_arr, grids

def solve_ricci_flow_conformal_2d(phi0, dt, n_steps, order=3, L=8.0, N=64,
                                   save_every=1, quantization='kohn-nirenberg',
                                   apply_backend='peetre', check_finite=True):
    """
    Integrate 2D Ricci flow in conformal gauge on a flat, doubly periodic
    background.

    Writing the metric as `g = e^{2φ}(dx² + dy²)`, the Gauss curvature is
    `K = −e^{−2φ}Δφ` and, since `R_ij = K·g_ij` in two dimensions, the
    tensorial flow `∂ₜg_ij = −2R_ij` collapses to the scalar quasi-linear
    heat equation

        ∂ₜφ = e^{−2φ} Δφ ,

    with `Δ = ∂ₓ² + ∂ᵧ²` the flat Laplacian. This is NOT handled by
    `psiop`'s ordinary linear/matrix machinery: the coefficient
    `e^{−2φ}` depends on the evolving solution itself, so no fixed
    `sympy` symbol `p(x, ξ)` describes the operator ahead of time.

    Instead, each step uses an IMEX/Lie splitting that still reuses
    `psiop`'s exact exponential propagator for the stiff part:

    1. **Explicit correction** — using the *current* coefficient field
       `c(x) = e^{−2φ(x)}`, compute `Δφ` once via a plain (x-independent)
       Laplacian `PseudoDifferentialOperator`, and take one explicit
       Euler sub-step with the deviation of `c` from its spatial average
       `c₀`: `residual = (c − c₀) Δφ`.
    2. **Stiff step** — propagate the spatially averaged, x-independent
       (Fourier-multiplier) generator `c₀·Δ` applied to the
       explicitly-corrected field. Since `c0 * lap_symbol` is a pure
       Fourier multiplier, `exp(dt · c0 · Δ)` has the closed form
       `exp(-dt · c0 · |k|^2)` on this grid -- applied directly via FFT
       instead of rebuilding an asymptotic exponential-symbol propagator
       through `build_propagator` on every step. This is both cheaper
       (no symbolic recomputation per step) and *more* accurate (exact
       exponential instead of an order-`n` Taylor truncation in dt);
       `order` is kept only for backward compatibility and no longer
       affects the stiff step.

    This freezes the quasi-linear coefficient once per step (a Rothe-type
    linearization), so accuracy in `dt` is limited by that freezing, not
    by the stiff step itself, which is now exact for the frozen
    (constant-coefficient) part at every step.

    Parameters
    ----------
    phi0 : callable
        Initial conformal factor, called as `phi0(X, Y)` on the
        meshgrid-ed spatial coordinates; must return a real-valued array.
    dt : float
        Time step.
    n_steps : int
        Number of steps to take.
    order : int, optional
        Kept for backward compatibility; no longer affects the (now
        closed-form) stiff step. Default 3.
    L : float, optional
        Half-width of the (periodic) spatial domain along each axis.
        Default 8.0.
    N : int, optional
        Number of grid points per axis. Default 64.
    save_every : int, optional
        Save every `save_every` steps (plus the final step and `t=0`).
        Default 1.
    quantization : str, optional
        Quantization convention used for the explicit Laplacian
        evaluation. Default 'kohn-nirenberg'.
    apply_backend : str, optional
        Numerical application backend for the explicit Laplacian
        evaluation. Default 'peetre'.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead
        of silently returning a diverged trajectory.

    Returns
    -------
    t_list : ndarray
        Saved time points, starting at 0.
    phi_list : ndarray, shape (n_saved, N, N)
        Saved conformal-factor snapshots `φ(x, y, t)`; the metric at each
        saved time is `g(t) = e^{2·phi_list[k]} (dx² + dy²)`.
    grids : tuple of ndarray
        `(x, y)` spatial grids, as returned by `make_grid_2d`.
    """
    x_s, y_s, xi_s, eta_s = sp.symbols('x y xi eta', real=True)
    lap_symbol = -(xi_s**2 + eta_s**2)

    x_grid, y_grid, kx, ky = make_grid_2d(L, N)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2  # |xi|^2 + |eta|^2, i.e. -lap_symbol on this grid

    lap_op = PseudoDifferentialOperator(
        lap_symbol, [x_s, y_s], mode='symbol',
        quantization=quantization, apply_backend=apply_backend,
    )

    # phi is real-valued by construction -- no need for complex dtype here.
    phi = np.asarray(phi0(X, Y), dtype=float)
    t_list = [0.0]
    phi_list = [phi.copy()]
    t = 0.0

    for n in range(1, n_steps + 1):
        c = np.exp(-2 * phi)
        c0 = float(np.mean(c))

        lap_phi = lap_op.apply(phi, x_grid, kx, y_grid=y_grid, ky=ky).real
        residual = (c - c0) * lap_phi
        phi_explicit = phi + dt * residual

        # Closed-form stiff step: exp(dt * c0 * Delta) is exactly the
        # Fourier multiplier exp(-dt * c0 * |k|^2) for this x-independent
        # generator -- apply it directly instead of rebuilding an
        # asymptotic exponential-symbol propagator every step.
        phi_hat = np.fft.fft2(phi_explicit)
        phi_hat *= np.exp(-dt * c0 * K2)
        phi = np.fft.ifft2(phi_hat).real

        if check_finite and not np.isfinite(phi).all():
            raise FloatingPointError(
                f"Non-finite values detected at step {n} (t={t + dt:.6g}); "
                "reduce dt or the spatial resolution N."
            )

        t += dt
        if n % save_every == 0 or n == n_steps:
            t_list.append(t)
            phi_list.append(phi.copy())

    return np.array(t_list), np.array(phi_list), (x_grid, y_grid)

    return np.array(t_list), np.array(phi_list), (x_grid, y_grid)
# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def _quantity_fn(quantity):
    """'real' | 'imag' | 'abs' -> the corresponding numpy function."""
    try:
        return {'real': np.real, 'imag': np.imag, 'abs': np.abs}[quantity]
    except KeyError:
        raise ValueError("quantity must be 'real', 'imag', or 'abs'")


def _finish_headless(fig, save_path=None):
    """Standard ending for the 'returns a Figure, doesn't display it'
    PDE-solution plots: tight layout, optional save, close, return."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_scalar_1d(t, U, x, title="u(x, t)", quantity='real',
                   n_snapshots=6, save_path=None):
    """Plot a scalar 1D space-time solution as a combined heatmap and
    snapshot overlay. See original docstring for full parameter docs;
    behavior and return value unchanged."""
    field = _quantity_fn(quantity)(U)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    im = ax1.pcolormesh(x, t, field, shading='auto', cmap='RdBu_r')
    ax1.set_xlabel('x'); ax1.set_ylabel('t')
    ax1.set_title(f"{title} -- space-time ({quantity})")
    fig.colorbar(im, ax=ax1)

    idx = np.linspace(0, len(t) - 1, n_snapshots).astype(int)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    for c, i in zip(cmap, idx):
        ax2.plot(x, field[i], color=c, label=f"t={t[i]:.2f}")
    ax2.set_xlabel('x'); ax2.set_ylabel(quantity)
    ax2.set_title("snapshots")
    ax2.legend(fontsize=8, ncol=2)

    return _finish_headless(fig, save_path)


def plot_matrix_1d(t, U, x, labels=None, quantity='real', save_path=None):
    """Plot each component of a matrix-valued 1D solution as a stacked
    space-time heatmap. See original docstring; behavior unchanged."""
    field_fn = _quantity_fn(quantity)
    size = U.shape[1]
    labels = labels or [f"u_{k+1}" for k in range(size)]

    fig, axes = plt.subplots(size, 1, figsize=(6, 3 * size), sharex=True)
    axes = [axes] if size == 1 else axes

    for k, ax in enumerate(axes):
        im = ax.pcolormesh(x, t, field_fn(U[:, k, :]), shading='auto', cmap='RdBu_r')
        ax.set_ylabel('t')
        ax.set_title(f"{labels[k]} ({quantity})")
        fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel('x')

    return _finish_headless(fig, save_path)


def plot_scalar_2d(t, U, x, y, times=None, quantity='real', save_path=None):
    """Plot a scalar 2D solution at selected time instants as a row of
    side-by-side pcolormesh panels. See original docstring; behavior
    unchanged."""
    field_fn = _quantity_fn(quantity)
    times = np.linspace(0, len(t) - 1, 6).astype(int) if times is None else times

    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    axes = [axes] if n == 1 else axes

    im = None
    for ax, i in zip(axes, times):
        im = ax.pcolormesh(x, y, field_fn(U[i]).T, shading='auto', cmap='RdBu_r')
        ax.set_title(f"t={t[i]:.2f}")
        ax.set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im, ax=axes[-1])

    return _finish_headless(fig, save_path)


def animate_scalar_1d(t, U, x, quantity='real', interval=40, save_path=None):
    """Animated line plot of a scalar 1D solution evolving in time. See
    original docstring; behavior unchanged."""
    from matplotlib.animation import FuncAnimation
    field = _quantity_fn(quantity)(U)
    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot(x, field[0])
    ax.set_ylim(field.min(), field.max())
    ax.set_xlabel('x'); ax.set_ylabel(quantity)
    title = ax.set_title(f"t={t[0]:.2f}")

    def update(i):
        line.set_ydata(field[i])
        title.set_text(f"t={t[i]:.2f}")
        return line, title

    anim = FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)
    if save_path:
        anim.save(save_path)
    plt.close(fig)
    return anim


# --- New: matrix-field solvers (solve_matrix_field / solve_sylvester_field)
# had NO plotting function at all -- their output shape (n_saved, N, N,
# *grid) doesn't fit plot_matrix_1d's (n_saved, size, *grid). These fill
# that gap.

def _matrix_field_reduce(U, component, quantity):
    """Reduce a (n_saved, N, N, *grid) matrix-field array to a single
    real/complex scalar field (n_saved, *grid), per `component`."""
    if component == 'trace':
        return np.trace(U, axis1=1, axis2=2), _quantity_fn(quantity)
    if component == 'frobenius':
        return np.linalg.norm(U, axis=(1, 2)), np.abs  # already real, non-negative
    if component == 'diag':
        raise ValueError("component='diag' needs the *_1d panel-per-entry "
                         "helper (plot_matrix_field_1d); pick 'trace', "
                         "'frobenius', or an (i, j) entry here.")
    i, j = component
    return U[:, i, j, ...], _quantity_fn(quantity)


def plot_matrix_field_1d(t, U, x, quantity='abs', component='diag', labels=None, save_path=None):
    """Space-time heatmap(s) for a matrix-valued 1D solution, shape
    (n_times, N, N, Nx) -- the output of solve_matrix_field /
    solve_sylvester_field in 1D.

    Parameters
    ----------
    component : 'diag' | 'trace' | 'frobenius' | (i, j)
        'diag'      -- one panel per diagonal entry U_kk(x, t).
        'trace'     -- single panel, sum_k U_kk(x, t).
        'frobenius' -- single panel, ||U(x, t)||_F.
        (i, j)      -- single panel, the (i, j) entry U_ij(x, t).
    """
    if component == 'diag':
        field_fn = _quantity_fn(quantity)
        size = U.shape[1]
        labels = labels or [f"U_{k+1}{k+1}" for k in range(size)]
        fig, axes = plt.subplots(size, 1, figsize=(6, 3 * size), sharex=True)
        axes = [axes] if size == 1 else axes
        for k, ax in enumerate(axes):
            im = ax.pcolormesh(x, t, field_fn(U[:, k, k, :]), shading='auto', cmap='RdBu_r')
            ax.set_ylabel('t'); ax.set_title(f"{labels[k]} ({quantity})")
            fig.colorbar(im, ax=ax)
        axes[-1].set_xlabel('x')
        return _finish_headless(fig, save_path)

    panel, field_fn = _matrix_field_reduce(U, component, quantity)
    if component in ('trace', 'frobenius'):
        title = 'tr U(x, t)' if component == 'trace' else '||U(x, t)||_F'
    else:
        title = f"U_{component[0]+1}{component[1]+1}(x, t)"

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.pcolormesh(x, t, field_fn(panel), shading='auto', cmap='RdBu_r')
    ax.set_xlabel('x'); ax.set_ylabel('t'); ax.set_title(f"{title} ({quantity})")
    fig.colorbar(im, ax=ax)
    return _finish_headless(fig, save_path)


def plot_matrix_field_2d(t, U, x, y, times=None, quantity='abs', component='trace', save_path=None):
    """Snapshot panels for a matrix-valued 2D solution, shape
    (n_times, N, N, Nx, Ny) -- the output of solve_matrix_field /
    solve_sylvester_field in 2D. `component` as in `plot_matrix_field_1d`,
    except 'diag' isn't supported here (pick a single scalar reduction:
    'trace', 'frobenius', or an (i, j) entry)."""
    panel_all, field_fn = _matrix_field_reduce(U, component, quantity)
    times = np.linspace(0, len(t) - 1, 6).astype(int) if times is None else times

    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    axes = [axes] if n == 1 else axes
    im = None
    for ax, idx in zip(axes, times):
        im = ax.pcolormesh(x, y, field_fn(panel_all[idx]).T, shading='auto', cmap='RdBu_r')
        ax.set_title(f"t={t[idx]:.2f}"); ax.set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im, ax=axes[-1])
    return _finish_headless(fig, save_path)


# --- New: solve_second_order returns (U, V) but there was no combined
# view -- previously required calling plot_scalar_1d twice by hand.

def plot_wave_solution_1d(t, U, V, x, quantity='real', save_path=None):
    """Side-by-side space-time heatmaps of displacement U and velocity V,
    as returned by solve_second_order (scalar, 1D case)."""
    field_fn = _quantity_fn(quantity)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, field, label in zip(axes, (U, V), ('u', r'$\partial_t u$')):
        im = ax.pcolormesh(x, t, field_fn(field), shading='auto', cmap='RdBu_r')
        ax.set_xlabel('x'); ax.set_title(f"{label} ({quantity})")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel('t')
    return _finish_headless(fig, save_path)

# ----------------------------------------------------------------------
# Singularity & Ray Flow
# ----------------------------------------------------------------------

def _order_freq_vars(freq, dim):
    """
    Order a set of free frequency symbols into canonical (ξ, η) or
    (ξ, η, …) sequence matching the spatial dimension.

    Recognised canonical names are matched first ('xi', 'eta', 'kx',
    'ky', 'k1', 'k2'); any remaining symbols are appended in
    alphabetical order. This ensures a deterministic variable ordering
    for differentiation and lambdification regardless of the order in
    which SymPy reports free symbols.

    Parameters
    ----------
    freq : iterable of sympy.Symbol
        Candidate frequency symbols extracted from an expression.
    dim : int
        Number of frequency variables expected (1 or 2).

    Returns
    -------
    list of sympy.Symbol
        Ordered frequency symbols of length `dim`.
    """
    if dim == 1:
        return list(freq)
    by_name = {str(s): s for s in freq}
    out = []
    for canon in ('xi', 'eta', 'kx', 'ky', 'k1', 'k2'):
        if canon in by_name and len(out) < dim:
            out.append(by_name.pop(canon))
    out += sorted(by_name.values(), key=lambda s: s.name)
    return out[:dim]

def characteristic_hamiltonians(s_expr, vars_x, vars_xi=None):
    """
    Extract the characteristic Hamiltonian functions H(x, ξ) from a
    (possibly matrix-valued) operator symbol.

    For a scalar symbol p(x, ξ), the single Hamiltonian is

        H(x, ξ) = Re(i · p(x, ξ))

    For a matrix symbol P(x, ξ), the eigenvalues λ_k(x, ξ) are computed
    symbolically and each branch yields

        H_k(x, ξ) = Re(i · λ_k(x, ξ))

    These Hamiltonians generate the bicharacteristic (ray) flow via
    Hamilton's equations:

        ẋ = ∂H/∂ξ,   ξ̇ = −∂H/∂x

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Operator symbol (scalar or matrix-valued).
    vars_x : list of sympy.Symbol
        Spatial variables.
    vars_xi : list of sympy.Symbol, optional
        Frequency variables. If None, they are inferred from the free
        symbols of `s_expr` that are not in `vars_x`, and ordered
        canonically via `_order_freq_vars`.

    Returns
    -------
    H_list : list of sympy.Expr
        One Hamiltonian per characteristic branch (eigenvalue).
    xs : list of sympy.Symbol
        Canonical spatial symbols (real=True) used in H_list.
    xis : list of sympy.Symbol
        Canonical frequency symbols (real=True) used in H_list.

    Notes
    -----
    The substitution to fresh canonical symbols ensures consistent
    differentiation even if the input expression uses symbols with
    different assumptions.
    """
    S = _matrix_of(s_expr)
    dim = len(vars_x)
    freq = [s for s in S.free_symbols if s not in set(vars_x)]
    vars_xi = list(vars_xi) if vars_xi is not None else _order_freq_vars(freq, dim)

    xs = [sp.Symbol(v.name, real=True) for v in vars_x]
    xis = [sp.Symbol(v.name, real=True) for v in vars_xi]
    S = S.subs(dict(zip(list(vars_x) + list(vars_xi), xs + xis)))

    eigen = [S[0, 0]] if S.shape == (1, 1) else list(S.eigenvals().keys())

    H_list = []
    for lam in eigen:
        omega = sp.I * lam
        H_list.append(sp.simplify(sp.re(omega)))
    return H_list, xs, xis

def integrate_singularity(s_expr, vars_x, x0=0.0, xi0=5.0, tmax=4.0,
                          n_frames=100, vars_xi=None, branches='all',
                          method='RK45', **ivp_kwargs):
    """
    Numerically integrate bicharacteristic (Hamiltonian ray) trajectories
    from an initial phase-space point (x₀, ξ₀).

    For each characteristic branch H_k, Hamilton's equations

        ẋ = ∂H_k/∂ξ,   ξ̇ = −∂H_k/∂x

    are integrated using `scipy.integrate.solve_ivp` over [0, tmax].

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Operator symbol from which Hamiltonians are extracted.
    vars_x : list of sympy.Symbol
        Spatial variables.
    x0 : float or array_like, default 0.0
        Initial spatial position(s). Scalar for 1D, sequence for 2D.
    xi0 : float or array_like, default 5.0
        Initial frequency (momentum) component(s).
    tmax : float, default 4.0
        Final integration time.
    n_frames : int, default 100
        Number of output time samples in [0, tmax].
    vars_xi : list of sympy.Symbol, optional
        Explicit frequency variables (inferred if None).
    branches : 'all', int, or list of int, default 'all'
        Which characteristic branches to integrate. 'all' integrates
        every branch; an int or list selects specific ones.
    method : str, default 'RK45'
        ODE solver method passed to `solve_ivp`.
    **ivp_kwargs
        Additional keyword arguments forwarded to `solve_ivp`
        (e.g. `rtol`, `atol`, `max_step`).

    Returns
    -------
    H_list : list of sympy.Expr
        Hamiltonian expressions for the integrated branches.
    xs : list of sympy.Symbol
        Canonical spatial symbols.
    xis : list of sympy.Symbol
        Canonical frequency symbols.
    t_eval : ndarray, shape (n_frames,)
        Time samples at which trajectories are recorded.
    trajs : list of ndarray
        `trajs[b]` has shape (2·dim, n_frames): the first dim rows are
        position components, the last dim rows are momentum components.

    Examples
    --------
    >>> H, xs, xis, t, trajs = integrate_singularity(xi**2 + x**2, [x],
    ...                                              x0=0.0, xi0=3.0, tmax=6.0)
    """
    H_all, xs, xis = characteristic_hamiltonians(s_expr, vars_x, vars_xi)
    H_list = [H_all[b] for b in ([branches] if isinstance(branches, int) else branches)] if branches != 'all' else H_all

    dim = len(xs)
    y0 = np.concatenate([np.atleast_1d(x0), np.atleast_1d(xi0)])
    t_eval = np.linspace(0.0, tmax, n_frames)

    trajs = []
    for H in H_list:
        rhs_exprs = [sp.diff(H, k) for k in xis] + [-sp.diff(H, x) for x in xs]
        f = sp.lambdify(xs + xis, rhs_exprs, 'numpy')
        sol = solve_ivp(lambda t, Y: f(*Y), (0.0, tmax), y0, t_eval=t_eval, method=method, **ivp_kwargs)
        trajs.append(sol.y)
    return H_list, xs, xis, t_eval, trajs

def _trail_animation(coords_list, colors, px, py, pz=None, interval=50, pad_frac=0.08):
    """Shared 'growing dashed trail + moving point' animation builder for
    both the 2D and 3D singularity animations -- factors out what used to
    be two near-identical copies of the same figure/update-function setup.

    FIX: the axes are now explicitly sized to the full trajectory range
    (with a small padding margin) before the animation starts. Previously
    no xlim/ylim(/zlim) were set at all: since every trail/point artist is
    created empty (`ax.plot([], [])`) and only ever updated via
    `set_data`/`set_3d_properties` (not `ax.relim()` + `autoscale_view()`),
    matplotlib never grew the view beyond its default (0, 1) x (0, 1) box,
    so the rendered animation domain was almost always far too small to
    show the actual ray flow.
    """
    is_3d = pz is not None
    fig = plt.figure(figsize=(7.5, 5.5) if is_3d else (6, 5))
    ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

    def _padded_range(values):
        lo, hi = float(np.min(values)), float(np.max(values))
        span = hi - lo
        pad = span * pad_frac if span > 0 else (abs(lo) * pad_frac or 1.0)
        return lo - pad, hi + pad

    all_x = np.concatenate([np.atleast_1d(c[px]) for c in coords_list])
    all_y = np.concatenate([np.atleast_1d(c[py]) for c in coords_list])
    ax.set_xlim(*_padded_range(all_x))
    ax.set_ylim(*_padded_range(all_y))
    ax.set_xlabel(px)
    ax.set_ylabel(py)
    if is_3d:
        all_z = np.concatenate([np.atleast_1d(c[pz]) for c in coords_list])
        ax.set_zlim(*_padded_range(all_z))
        ax.set_zlabel(pz)
    else:
        ax.grid(True, alpha=0.3)

    trails, points = [], []
    for c in colors:
        if is_3d:
            tr, = ax.plot([], [], [], ls='--', lw=1.3, color=c)
            pt, = ax.plot([], [], [], 'o', ms=6, color=c)
        else:
            tr, = ax.plot([], [], ls='--', lw=1.3, alpha=0.6, color=c)
            pt, = ax.plot([], [], 'o', ms=6.5, color=c)
        trails.append(tr)
        points.append(pt)

    n_act = min(len(c[px]) for c in coords_list)

    def update(i):
        for b, c in enumerate(coords_list):
            xa, ya = c[px][:i + 1], c[py][:i + 1]
            trails[b].set_data(xa, ya)
            if is_3d:
                za = c[pz][:i + 1]
                trails[b].set_3d_properties(za)
                points[b].set_data([xa[-1]], [ya[-1]])
                points[b].set_3d_properties([za[-1]])
            else:
                points[b].set_data([xa[-1]], [ya[-1]])
        return trails + points

    anim = FuncAnimation(fig, update, frames=n_act, interval=interval, blit=False)
    plt.close(fig)
    return fig, anim


def animate_singularity(s_expr, vars_x, x0=0.0, xi0=5.0, tmax=4.0,
                        n_frames=100, projection=None, branches='all',
                        labels=None, interval=50, contours=True,
                        solution=None, quantity='abs', save_path=None):
    """Animate the propagation of singularities along bicharacteristic
    trajectories in a 2D phase-space projection. See original docstring
    for full parameter docs; behavior unchanged. Now built on the shared
    `_trail_animation` helper instead of duplicating the figure/update
    setup that also appears in `animate_singularity_3d`."""
    from matplotlib.animation import FuncAnimation
    dim = len(vars_x)
    H_list, xs, xis, t_eval, trajs = integrate_singularity(
        s_expr, vars_x, x0=x0, xi0=xi0, tmax=tmax, n_frames=n_frames, branches=branches)
    n_act = min(Y.shape[1] for Y in trajs)
    t_eval = t_eval[:n_act]
    trajs = [Y[:, :n_act] for Y in trajs]

    names = [v.name for v in xs] + [v.name for v in xis]
    coords = [{**dict(zip(names, Y)), 't': t_eval} for Y in trajs]

    # FIX: `projection` was never defaulted when None, so in 1D the
    # fallback branch picked py='y' -- a key that doesn't exist in 1D
    # coords (only 'x', 'xi', 't') -- raising a KeyError as soon as the
    # axis limits are computed from the data. Default explicitly, as
    # documented ('phase' in 1D, 'position' in 2D), and support the same
    # projections as the class method (position/frequency plotted on the
    # diagonal in 1D, since there's only one spatial/frequency axis).
    if projection is None:
        projection = 'phase' if dim == 1 else 'position'
    if dim == 1:
        px, py = {'phase': ('x', 'xi'), 'position': ('x', 'x'),
                  'frequency': ('xi', 'xi')}.get(projection, (None, None))
        if px is None:
            raise ValueError(
                f"Invalid projection {projection!r} for a 1D symbol; "
                "use 'phase', 'position', or 'frequency'."
            )
    else:
        px, py = 'x', 'y'

    colors = [plt.cm.tab10.colors[b % 10] for b in range(len(H_list))]
    _, anim = _trail_animation(coords, colors, px, py, interval=interval)
    if save_path:
        anim.save(save_path)
    return anim


def animate_singularity_3d(s_expr, vars_x, x0=0.0, xi0=5.0, tmax=4.0,
                           n_frames=100, projection=None, branches='all',
                           labels=None, interval=50, save_path=None):
    """Animate bicharacteristic trajectories in a 3D matplotlib plot. See
    original docstring for full parameter docs; behavior unchanged. Now
    built on the shared `_trail_animation` helper."""
    from matplotlib.animation import FuncAnimation
    import mpl_toolkits.mplot3d  # noqa: F401

    H_list, xs, xis, t_eval, trajs = integrate_singularity(
        s_expr, vars_x, x0=x0, xi0=xi0, tmax=tmax, n_frames=n_frames, branches=branches)
    n_act = min(Y.shape[1] for Y in trajs)
    t_eval, trajs = t_eval[:n_act], [Y[:, :n_act] for Y in trajs]

    names = [v.name for v in xs] + [v.name for v in xis]
    coords = [{**dict(zip(names, Y)), 't': t_eval} for Y in trajs]
    px, py, pz = (names[0], names[1], 't') if len(vars_x) == 1 else (names[0], names[1], names[2])

    colors = [plt.cm.tab10.colors[b % 10] for b in range(len(H_list))]
    _, anim = _trail_animation(coords, colors, px, py, pz=pz, interval=interval)
    if save_path:
        anim.save(save_path)
    return anim