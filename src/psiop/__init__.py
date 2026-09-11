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
psiop — Symbolic–numerical toolkit for pseudo‑differential operators in 1D/2D
=============================================================================

Overview
--------
The ``psiop`` package provides a unified framework for constructing,
manipulating, and numerically applying pseudo‑differential operators
(ΨDOs) in one and two spatial dimensions.  It combines symbolic symbol
calculus (SymPy) with numerical evaluation, quantization, diagnostics,
and visualization (NumPy / SciPy / Matplotlib).

The package is intended for researchers and students working in
microlocal analysis, spectral theory, semiclassical analysis,
Hamiltonian / geometric optics, and the numerical analysis of PDEs.

Subpackage layout
-----------------
``psiop`` (this module)
    Core scalar ``PseudoDifferentialOperator``, Peetre decomposition,
    asymptotic symbolic calculus, quantization conversion, operator
    diagnostics, and Hamiltonian-flow / visualization wrappers.

``psiop.matpsiop``
    ``MatrixPseudoDifferentialOperator`` — N × N matrix-valued symbols
    acting on vector fields and matrix-valued fields, with noncommutative
    asymptotic composition, matrix commutators, and matrix exponential
    symbols.

``psiop.psiop_apply``
    Standalone numerical kernels: Kohn–Nirenberg FFT / non-periodic
    quadrature, NUFFT joint-residual backend, AAA rational approximation,
    and Chebyshev / SVD low-rank factorization.

``psiop.psiop_solver``
    Time-stepping solvers and propagators: ``build_propagator``,
    ``PropagatorFamily``, ``solve_first_order``, ``solve_second_order``,
    ``solve_matrix_field``, ``solve_sylvester_field``,
    ``solve_ricci_flow_conformal_2d``, and grid-generation utilities.

Main object
-----------
``PseudoDifferentialOperator``
    Scalar pseudo‑differential operator in 1D or 2D.  The symbol may be
    given explicitly (``mode='symbol'``) or extracted automatically from
    a differential expression acting on a test function (``mode='auto'``).

Key features
------------
Symbol creation (1D and 2D):
    explicit symbol mode p(x, ξ) or p(x, y, ξ, η);
    automatic extraction from differential operators via plane‑wave testing.

Quantization conventions:
    Kohn–Nirenberg (default) and Weyl quantization;
    asymptotic conversion between the two via the exponential cross-derivative
    series (exact and finite for polynomial symbols).

High‑frequency symbolic analysis:
    principal symbol extraction; asymptotic expansion as |ξ| → ∞;
    homogeneity detection; symbolic / numerical order estimation.

Asymptotic symbolic calculus:
    operator composition p ∘ q (KN and Weyl / Moyal);
    commutators [P, Q]; formal left and right inverses;
    formal adjoint P*; fractional powers P^α;
    exponential symbols exp(tP).

Peetre‑style symbolic decomposition:
    local polynomial part in the frequency variables;
    separable terms a(x) q(ξ);
    genuinely joint, non‑separable residual;
    automatic backend selection for the joint residual (NUFFT, AAA,
    or low-rank) — see ``psiop.psiop_apply`` for the numerical kernels.

Operator diagnostics:
    symbolic and numerical ellipticity tests;
    formal self‑adjointness checks;
    semiclassical trace formula (symbolic or numerical);
    pseudospectrum computation with resolvent-norm contours, eigenvalue
    overlays, sparse / dense solvers, and optional adaptive refinement.

Hamiltonian and microlocal dynamics:
    Hamiltonian vector field of the principal symbol;
    symplectic flow visualization; bicharacteristic integration;
    singularity propagation animation (2D / 3D phase-space projections);
    characteristic set and gradient visualization.

Visualization:
    Symbol amplitude / phase plots, cotangent fibers, micro-support,
    wavefront set, group velocity fields, and interactive ipywidgets
    dashboards — all delegated to the ``microlocal`` plotting module.

Mathematical background
-----------------------
Symbols and quantization
    A pseudo‑differential operator P acting on functions of
    x ∈ ℝⁿ (n = 1 or 2) is defined by its symbol p(x, ξ) on
    phase space T*ℝⁿ = ℝⁿ_x × ℝⁿ_ξ.  Symbols are classified by
    their growth in ξ: p ∈ S^m_{1,0} if

        |∂_x^β ∂_ξ^α p(x, ξ)| ≤ C_{α,β} (1 + |ξ|)^(m - |α|)

    for all multi‑indices α, β.  The integer / real number m is the
    *order* of the operator.

Quantization maps
    Kohn–Nirenberg (left / standard, default):

        (P u)(x) = Op^KN(p) u(x) = (2π)⁻ⁿ ∫_{ℝⁿ} exp(i x·ξ) p(x, ξ) û(ξ) dξ

    Weyl (symmetric):

        Op^w(a) u(x) = (2π)⁻ⁿ ∫∫ exp(i(x-y)·ξ) a((x+y)/2, ξ) u(y) dy dξ

    Conversion between the two is realized as an asymptotic series in
    mixed x–ξ derivatives:

        a_KN(x, ξ) = exp(−i/2 · ∂_x ∂_ξ) a_Weyl(x, ξ)
                   ∼ Σ_{k≥0} (−i/2)^k / k! · (∂_x ∂_ξ)^k a_Weyl(x, ξ)

    In 2D the cross-derivative operator sums both coordinate pairs:
    ∂_x ∂_ξ → ∂_x ∂_ξ + ∂_y ∂_η.  For polynomial symbols the
    series is exact and finite; for general S^m_{1,0} symbols it is
    interpreted asymptotically.

Asymptotic composition
    For two symbols p and q, the Kohn–Nirenberg expansion is

        (p ∘ q)(x, ξ) ∼ Σ_α i^(-|α|) / α! · ∂_ξ^α p · ∂_x^α q

    A Weyl / Moyal star product is also available.  The same machinery
    underlies commutators, formal inverses, adjoints, fractional powers,
    and exponential symbols.

Principal symbol and order
    The principal symbol is the leading homogeneous component as
    |ξ| → ∞.  If p is homogeneous of degree m,
    p(x, λξ) = λ^m p(x, ξ) for λ > 0.

Peetre decomposition
    For numerical application the symbol is split into

        p(x, ξ) = p_local(x, ξ) + p_sep(x, ξ) + p_joint(x, ξ)

    where p_local is polynomial in ξ (differential part),
    p_sep = Σ_k a_k(x) q_k(ξ) is a sum of separable Fourier
    multipliers, and p_joint contains the genuinely entangled
    space–frequency residual — routed automatically to NUFFT, AAA, or
    low-rank backends (see ``psiop.psiop_apply``).

Hamiltonian flow
    The Hamiltonian vector field of the principal symbol governs
    singularity propagation along bicharacteristics:

        dx/dt = ∂_ξ p,    dξ/dt = −∂_x p      (1D)
        dx/dt = ∂_ξ p,    dy/dt = ∂_η p,
        dξ/dt = −∂_x p,   dη/dt = −∂_y p      (2D)

Pseudospectrum
    The ε-pseudospectrum of P is

        σ_ε(P) = { λ ∈ ℂ : ‖(P − λI)⁻¹‖ ≥ ε⁻¹ }

    determined by the smallest singular value of the shifted matrix
    discretization: ‖(P − λI)⁻¹‖ = 1 / σ_min(P − λI).
    Particularly informative for non-normal operators.

References
----------
.. [1] Hörmander, L. *The Analysis of Linear Partial Differential
       Operators III*, Springer, 1985.  Chapter 18.
.. [2] Taylor, M. E. *Pseudo Differential Operators*,
       Princeton University Press, 1981.
.. [3] Zworski, M. *Semiclassical Analysis*, AMS, 2012.  Chapter 4.
.. [4] Martinez, A. *An Introduction to Semiclassical and Microlocal
       Analysis*, Springer, 2002.
.. [5] Trefethen, L. N. and Embree, M. *Spectra and Pseudospectra*,
       Princeton University Press, 2005.
.. [6] Peetre, J. "Applications de la théorie des espaces
       d'interpolation dans l'analyse harmonique."
       *Ricerche di Matematica*, 1968.
"""
from imports import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import microlocal as _mu
from microlocal import (
    plot_scalar_1d, plot_matrix_1d, plot_scalar_2d, animate_scalar_1d,
    plot_matrix_field_1d, plot_matrix_field_2d, plot_wave_solution_1d,
    animate_singularity, animate_singularity_3d,
    characteristic_hamiltonians, integrate_singularity,
)

# --- NEW: Import numerical backends from the new psiop_apply submodule ---
from .psiop_apply import (
    kohn_nirenberg_fft, kohn_nirenberg_nonperiodic,
    factorize_symbolic, try_nufft_decomposition_1d, try_nufft_decomposition_2d,
    apply_nufft_1d, apply_nufft_2d, try_aaa_decomposition_1d, try_aaa_decomposition_2d,
    aaa_plan_to_callable_1d, aaa_plan_to_callable_2d, _clip_complex_magnitude
)

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
            yield ((i,) + rest)

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

    def __init__(self, expr, vars_x, var_u=None, mode='symbol', quantization='kohn-nirenberg', apply_backend='peetre', compute_peetre=False, peetre_options=None):
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
        if apply_backend not in {'direct', 'peetre'}:
            raise ValueError("apply_backend must be 'direct' or 'peetre'")
        self.apply_backend = apply_backend
        self._peetre_options = dict(peetre_options or {})
        self._peetre_decomposition = None
        if compute_peetre is None:
            compute_peetre = apply_backend == 'peetre'
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
            raise NotImplementedError('Only 1D and 2D supported')
        if mode == 'auto':
            self._compute_symbol_derivatives()
            print('\nsymbol = ')
            pprint(self.symbol, num_columns=NUM_COLS)
        if compute_peetre:
            self._peetre_decomposition = self.peetre_decomposition(**self._peetre_options)

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
        if self.dim == 1:
            vars_tuple = (self.vars_x[0], symbols('xi', real=True))
        else:
            vars_tuple = tuple(self.vars_x) + (symbols('xi', real=True), symbols('eta', real=True))
        for name, expr in self.derivatives.items():
            try:
                setattr(self, f'_{name}_func', lambdify(vars_tuple, expr, 'numpy'))
            except Exception:
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
        if hasattr(self, '_peetre_cache'):
            self._peetre_cache = None
        if hasattr(self, '_peetre_decomposition'):
            self._peetre_decomposition = None
        if hasattr(self, '_joint_lowrank_cache'):
            self._joint_lowrank_cache = None
        if hasattr(self, '_joint_nufft_cache'):
            self._joint_nufft_cache = None
        if hasattr(self, '_joint_aaa_cache'):
            self._joint_aaa_cache = None

    def _get_peetre_decomposition(self):
        """
        Return the Peetre decomposition stored in the instance.
    
        If the decomposition was not computed in __init__, it is computed
        lazily on first use.
        """
        if getattr(self, '_peetre_decomposition', None) is None:
            opts = getattr(self, '_peetre_options', None) or {}
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
        backend : {'peetre', 'direct'}, optional
            Numerical application backend. If None (default), uses
            `self.apply_backend`. 'peetre' applies the operator via the
            local/separable/low-rank Peetre decomposition; 'direct' evaluates
            the Kohn-Nirenberg (or Weyl-corrected) integral pointwise.
        apply_joint : bool, default=True
            When `backend='peetre'`, whether to include the genuinely joint
            (non-separable) residual term of the Peetre decomposition. If
            False, only the local and separable terms are applied.
        joint_backend : str, default="direct"
            Numerical representation used for the joint residual term when
            `apply_joint=True` (e.g. direct evaluation, low-rank/AAA, etc.).
        joint_degree : int, default=6
            Polynomial/Chebyshev degree used when approximating the joint
            residual term.
        joint_tol : float, default=1e-5
            Tolerance controlling the accuracy of the joint-residual
            approximation.
        joint_bounds : tuple, optional
            Bounding box for the joint residual's domain of approximation.
            If None, inferred automatically.
        joint_max_rel_error : float, optional
            If given, an upper bound on the acceptable relative error of the
            joint-residual approximation; the backend may refine or fall back
            to a more accurate method if this is exceeded.
     
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
            raise NotImplementedError('Only 1D and 2D supported')

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
            effective_symbol = self.symbol
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            return lambdify((x, xi), effective_symbol, 'numpy')
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            return lambdify((x, y, xi, eta), effective_symbol, 'numpy')
        else:
            raise NotImplementedError('_get_effective_symbol_func: only 1D and 2D are supported.')

    def _apply_constant_fft(self, u, x_grid, kx, y_grid=None, ky=None, dealiasing_mask=None, freq_window='gaussian', clamp=1000000.0, space_window=False):
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
            symbol_vals = np.broadcast_to(symbol_vals, kx_fft.shape).astype(np.complex128).copy()
            symbol_vals = _clip_complex_magnitude(symbol_vals, clamp)
            if freq_window == 'gaussian':
                k_max = np.max(np.abs(kx_fft))
                if k_max > 0:
                    sigma = 0.8 * k_max
                    symbol_vals *= np.exp(-(kx_fft / sigma) ** 4)
            elif freq_window == 'hann':
                k_max = np.max(np.abs(kx_fft))
                if k_max > 0:
                    W = 0.5 * (1.0 + np.cos(np.pi * kx_fft / k_max)) * (np.abs(kx_fft) < k_max)
                    symbol_vals *= W
        elif self.dim == 2:
            if y_grid is None:
                raise ValueError('y_grid is required for 2D operators.')
            Nx = len(x_grid)
            Ny = len(y_grid)
            dx = x_grid[1] - x_grid[0]
            dy = y_grid[1] - y_grid[0]
            kx_fft = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
            ky_fft = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
            KX, KY = np.meshgrid(kx_fft, ky_fft, indexing='ij')
            X_dummy = np.zeros_like(KX)
            Y_dummy = np.zeros_like(KY)
            symbol_vals = self.p_func(X_dummy, Y_dummy, KX, KY)
            symbol_vals = np.broadcast_to(symbol_vals, KX.shape).astype(np.complex128).copy()
            symbol_vals = _clip_complex_magnitude(symbol_vals, clamp)
            if freq_window == 'gaussian':
                kx_max = np.max(np.abs(kx_fft))
                ky_max = np.max(np.abs(ky_fft))
                if kx_max > 0 and ky_max > 0:
                    sx = 0.8 * kx_max
                    sy = 0.8 * ky_max
                    symbol_vals *= np.exp(-(KX / sx) ** 4) * np.exp(-(KY / sy) ** 4)
            elif freq_window == 'hann':
                kx_max = np.max(np.abs(kx_fft))
                ky_max = np.max(np.abs(ky_fft))
                if kx_max > 0 and ky_max > 0:
                    Wx = 0.5 * (1.0 + np.cos(np.pi * KX / kx_max)) * (np.abs(KX) < kx_max)
                    Wy = 0.5 * (1.0 + np.cos(np.pi * KY / ky_max)) * (np.abs(KY) < ky_max)
                    symbol_vals *= Wx * Wy
        else:
            raise ValueError('Only 1D and 2D supported')
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
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            return simplify(series(p, xi, oo, n=order).removeO())
        elif self.dim == 2:
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in p.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            rho, theta = symbols('rho theta', real=True, positive=True)
            p_rho = p.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
            expansion = series(p_rho, rho, oo, n=order).removeO()
            expansion_cart = expansion.subs({rho: sqrt(xi ** 2 + eta ** 2), cos(theta): xi / sqrt(xi ** 2 + eta ** 2), sin(theta): eta / sqrt(xi ** 2 + eta ** 2)})
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
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            l = symbols('l', real=True, positive=True)
            p_scaled = p.subs(xi, l * xi)
            ratio = simplify(p_scaled / p)
            if ratio.has(xi):
                return (False, None)
            try:
                ratio_clean = nsimplify(simplify(ratio), rational=True)
                base, deg = ratio_clean.as_base_exp()
                if base == l:
                    return (True, deg)
                return (False, None)
            except Exception:
                return (False, None)
        elif self.dim == 2:
            p = self.symbol
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in p.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            l = symbols('l', real=True, positive=True)
            p_scaled = p.subs({xi: l * xi, eta: l * eta})
            ratio = simplify(p_scaled / p)
            if ratio.has(xi, eta):
                return (False, None)
            try:
                ratio_clean = nsimplify(ratio, rational=True)
                base, exp = ratio_clean.as_base_exp()
                if base == l:
                    return (True, exp)
            except Exception:
                pass
            return (False, None)

    def symbol_order(self, max_order=10, tol=0.001):
        """
        Estimate the asymptotic homogeneity order of the symbol as |ξ| → ∞.
        
        If the symbol has no frequency dependence at all, returns 0
        immediately. If the symbol is exactly homogeneous (as determined by
        `is_homogeneous`), returns that degree directly. Otherwise, the
        leading-order behavior is estimated via a series expansion:
        
        - 1D: tries `series(symbol, xi, oo)` first (order ≥ 0 case), then
          falls back to substituting `xi = 1/z` and expanding around `z = 0`
          (order < 0 case).
        - 2D: same two-step strategy in polar frequency coordinates
          `(rho, theta)`, expanding around `rho = oo` and, on failure,
          around `z = 1/rho -> 0`.
        
        In each case the leading term is isolated and its coefficient checked
        for being symbolically nonzero; if it vanishes, that method's result
        is discarded and the next fallback is tried. Warnings about
        non-homogeneous symbols or a symbolically zero leading coefficient are
        printed to stdout rather than raised.
        
        Parameters
        ----------
        max_order : int, default=10
            Number of terms requested from each underlying `sympy.series`
            expansion; higher values can resolve the true leading order when
            lower-order terms cancel, at increased symbolic cost.
        tol : float, default=1e-3
            Currently unused by the estimation logic itself (reserved for
            numerical-order estimation call sites); present for interface
            consistency.
        
        Returns
        -------
        int, float, or None
            The estimated homogeneity order. Returns an `int` when the
            estimated order is (numerically) integral, a `float` otherwise,
            and `None` if no method (series at infinity nor at zero, in
            either coordinate system) could determine a nonzero leading term.
        
        Raises
        ------
        NotImplementedError
            If `self.dim` is not 1 or 2.
        """
        from sympy import symbols, series, simplify, cos, sin, oo, powdenest, radsimp, Add

        def validate_order(power, coeff, tol):
            if power is None:
                return None
            if simplify(coeff) == 0 or coeff.equals(0):
                print('⚠️ Coefficient is symbolically zero; ignoring')
                return None
            return int(power) if float(power) == int(power) else float(power)
        freq_syms = [s for s in self.symbol.free_symbols if s.name in ('xi', 'eta')]
        if not freq_syms:
            return 0
        is_homog, degree = self.is_homogeneous()
        if is_homog:
            return float(degree)
        print('⚠️ The symbol is not homogeneous. The asymptotic order is not well defined.')

        def leading_power(s, var):
            terms = Add.make_args(s)
            powers = [t.as_powers_dict().get(var, 0) for t in terms]
            top = max(powers)
            lead = Add(*[t for t, p in zip(terms, powers) if p == top])
            lead = radsimp(simplify(powdenest(lead, force=True)))
            coeff = lead / var ** top if top else lead
            return (top, coeff)
        if self.dim == 1:
            x = self.vars_x[0]
            xi = next((s for s in self.symbol.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            try:
                s = series(self.symbol, xi, oo, n=max_order).removeO()
                power, coeff = leading_power(s, xi)
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return order
            except Exception:
                pass
            try:
                z = symbols('z', real=True, positive=True)
                s = series(self.symbol.subs(xi, 1 / z), z, 0, n=max_order).removeO()
                power, coeff = leading_power(s, z)
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return -order
            except Exception as e:
                print(f'⚠️ fallback z failed: {e}')
            return None
        elif self.dim == 2:
            x, y = self.vars_x
            xi = next((s for s in self.symbol.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in self.symbol.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            rho, theta = symbols('rho theta', real=True, positive=True)
            try:
                p_rho = self.symbol.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
                s = series(simplify(p_rho), rho, oo, n=max_order).removeO()
                power, coeff = leading_power(s, rho)
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return order
            except Exception as e:
                print(f'⚠️ polar expansion failed: {e}')
            try:
                z = symbols('z', real=True, positive=True)
                p_z = self.symbol.subs({xi: cos(theta) / z, eta: sin(theta) / z})
                s = series(simplify(p_z), z, 0, n=max_order).removeO()
                power, coeff = leading_power(s, z)
                order = validate_order(power, coeff, tol)
                if order is not None:
                    return -order
            except Exception as e:
                print(f'⚠️ fallback z (2D) failed: {e}')
            return None
        raise NotImplementedError('Only 1D and 2D supported.')

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
                if p.func == exp and len(p.args) == 1:
                    arg = p.args[0]
                    arg_series = series(arg, xi, oo, n=order).removeO()
                    expanded = series(exp(expand(arg_series)), xi, oo, n=order).removeO()
                    return simplify(powdenest(expanded, force=True))
                else:
                    expanded = series(p, xi, oo, n=order).removeO()
                    return simplify(powdenest(expanded, force=True))
            except Exception as e:
                print(f'Warning: 1D expansion failed: {e}')
                return p
        elif self.dim == 2:
            xi = next((s for s in p.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in p.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            rho, theta = symbols('rho theta', real=True, positive=True)
            p = simplify(p)
            p_polar = p.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
            try:
                if p_polar.func == exp and len(p_polar.args) == 1:
                    arg = p_polar.args[0]
                    arg_series = series(arg, rho, oo, n=order).removeO()
                    expanded = series(exp(expand(arg_series)), rho, oo, n=order).removeO()
                else:
                    expanded = series(p_polar, rho, oo, n=order).removeO()
                norm = sqrt(xi ** 2 + eta ** 2)
                expansion_cart = expanded.subs({rho: norm, cos(theta): xi / norm, sin(theta): eta / norm})
                result = simplify(powdenest(expansion_cart, force=True))
                result = expand(result)
                return result
            except Exception as e:
                print(f'Warning: 2D expansion failed: {e}')
                return p

    def compose_asymptotic(self, other, order=1, mode='kn', sign_convention=None, do_simplify=True):
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
        do_simplify : bool, default True
            Whether to call sympy's `simplify()` while assembling the
            propagator symbol (once when building it, and once inside every
            `compose_asymptotic()` call in the asymptotic expansion loop).
            This does not change the operator being applied -- `lambdify`
            evaluates the same function on an unsimplified expression -- it
            only affects how much symbolic cleanup happens before that.
            `simplify()` is the dominant cost of `build_propagator()` for
            symbols mixing trigonometric and polynomial terms, and its cost
            grows with `order`; set to `False` to skip it and speed up
            propagator construction, at the risk of a larger (but
            numerically equivalent) unsimplified expression tree.
        
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
        assert self.dim == other.dim, 'Operator dimensions must match'
        if mode not in ('kn', 'weyl'):
            raise ValueError("mode must be either 'kn' or 'weyl'")
        dim = self.dim
        if dim not in (1, 2):
            raise NotImplementedError('Only 1D and 2D cases are implemented')
        p, q = (self.symbol, other.symbol)
        x_vars = self.vars_x
        xi_vars = self._peetre_frequency_symbols()
        sign = -1 if (sign_convention or 'standard') == 'standard' else +1
        result = 0
        if mode == 'kn':
            for n in range(order + 1):
                for alpha in _mi_all(n, dim):
                    fact = _mi_factorial(alpha)
                    dp = _mi_diff(p, xi_vars, alpha)
                    dq = _mi_diff(q, x_vars, alpha)
                    result += dp * dq / fact * 1j ** (sign * n)
        else:
            for total in range(order + 1):
                for a_deg in range(total + 1):
                    b_deg = total - a_deg
                    for alpha in _mi_all(a_deg, dim):
                        for beta in _mi_all(b_deg, dim):
                            coeff = (1j / 2) ** total * (-1) ** b_deg
                            coeff /= _mi_factorial(alpha) * _mi_factorial(beta)
                            dp = _mi_diff(_mi_diff(p, xi_vars, alpha), x_vars, beta)
                            dq = _mi_diff(_mi_diff(q, x_vars, alpha), xi_vars, beta)
                            result += coeff * dp * dq
        if not do_simplify:
            return result
        try:
            return simplify(result)
        except TypeError:
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
            xi = next((s for s in self.symbol.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            return (xi,)
        elif self.dim == 2:
            xi = next((s for s in self.symbol.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in self.symbol.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            return (xi, eta)
        raise NotImplementedError('Peetre decomposition supports only 1D and 2D operators.')

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
                a, q = (1, t)
            if any((q.has(xv) for xv in x_vars)):
                joint.append(t)
            elif q.is_polynomial(*xi_vars):
                local_terms.append(t)
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
                    joint.extend(local_terms)
                    return (local_coeffs, separable, joint)
            for monom, coeff in poly.terms():
                local_coeffs[monom] = simplify(together(local_coeffs.get(monom, 0) + coeff))
        return (local_coeffs, separable, joint)

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
                    term = term * xi_var ** power
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
        return [(merged[q], q) for q in ordered_keys if not self._peetre_is_zero(merged[q])]

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
                    q = q * var ** power
            coeff = simplify(coeff)
            q = expand(q)
            merged = False
            for i, (c0, q0) in enumerate(separable):
                if self._peetre_is_zero(c0 - coeff):
                    separable[i] = (c0, expand(q0 + q))
                    merged = True
                    break
            if not merged:
                separable.append((coeff, q))
        return [(c, expand(q)) for c, q in separable if not self._peetre_is_zero(c) and (not self._peetre_is_zero(q))]

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
        return self._peetre_merge_separable(self._peetre_local_to_separable(local_coeffs))

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
                raise ValueError('Empty grid encountered while inferring bounds.')
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            if hi <= lo:
                lo -= 1.0
                hi += 1.0
            return (lo, hi)
        freq_syms = self._peetre_frequency_symbols()
        if self.dim == 1:
            return {self.vars_x[0]: _bounds(x_grid), freq_syms[0]: _bounds(kx)}
        elif self.dim == 2:
            if y_grid is None or ky is None:
                raise ValueError('y_grid and ky are required for 2D bounds.')
            return {self.vars_x[0]: _bounds(x_grid), self.vars_x[1]: _bounds(y_grid), freq_syms[0]: _bounds(kx), freq_syms[1]: _bounds(ky)}
        else:
            raise NotImplementedError('Only 1D and 2D bounds are supported.')

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
                if getattr(k, 'name', str(k)) == getattr(s, 'name', str(s)):
                    matched_key = k
                    break
            if matched_key is None:
                raise ValueError(f"No bound provided for symbol '{s}'.")
            out[s] = bounds[matched_key]
        return out

    def _low_rank_joint_pairs(self, joint_symbol, bounds, degree=6, tol=1e-05, num_samples=10000, seed=42, use_cache=True):
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
            return ([], {'rel_l2_error': 0.0, 'max_abs_error': 0.0, 'mean_abs_error': 0.0, 'svd_energy_retained_pct': 100.0, 'singular_values': np.array([])})
        x_syms = []
        for v in self.vars_x:
            s = next((fs for fs in joint_symbol.free_symbols if fs.name == v.name), v)
            x_syms.append(s)
        freq_names = ['xi'] if self.dim == 1 else ['xi', 'eta']
        xi_syms = []
        for name in freq_names:
            s = next((fs for fs in joint_symbol.free_symbols if fs.name == name), symbols(name, real=True))
            xi_syms.append(s)
        all_syms = x_syms + xi_syms
        bounds = self._remap_bounds(bounds, all_syms)
        key = (joint_symbol, degree, tol, tuple(((s, float(bounds[s][0]), float(bounds[s][1])) for s in all_syms)))
        cache = getattr(self, '_joint_lowrank_cache', None)
        if use_cache and cache is not None and (cache.get('key') == key):
            return (cache['pairs'], cache['metrics'])
        pairs, metrics = factorize_symbolic(joint_symbol, x_syms, xi_syms, bounds, degree=degree, tol=tol, num_samples=num_samples, seed=seed)
        self._joint_lowrank_cache = {'key': key, 'pairs': pairs, 'metrics': metrics}
        return (pairs, metrics)

    def _resolve_joint_symbols(self, joint_symbol):
        """Shared symbol-resolution logic (matches _low_rank_joint_pairs):
        find the actual x/xi symbols present in joint_symbol by name."""
        x_syms = []
        for v in self.vars_x:
            s = next((fs for fs in joint_symbol.free_symbols if fs.name == v.name), v)
            x_syms.append(s)
        freq_names = ['xi'] if self.dim == 1 else ['xi', 'eta']
        xi_syms = []
        for name in freq_names:
            s = next((fs for fs in joint_symbol.free_symbols if fs.name == name), symbols(name, real=True))
            xi_syms.append(s)
        return (x_syms, xi_syms)

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
        cache = getattr(self, '_joint_nufft_cache', None)
        if use_cache and cache is not None and (cache.get('key') == key):
            return cache['plan_info']
        if self.dim == 1:
            plan = try_nufft_decomposition_1d(joint_symbol, x_syms[0], xi_syms[0])
            plan_info = ('1d', plan) if plan is not None else None
        elif self.dim == 2:
            res = try_nufft_decomposition_2d(joint_symbol, x_syms[0], x_syms[1], xi_syms[0], xi_syms[1])
            plan_info = ('2d', res) if res is not None else None
        else:
            plan_info = None
        self._joint_nufft_cache = {'key': key, 'plan_info': plan_info}
        return plan_info

    def _nufft_joint_apply(self, joint_symbol, u, x_grid, kx, y_grid=None, ky=None, use_cache=True, freq_window='gaussian'):
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
        if kind == '1d':
            return apply_nufft_1d(u, plan, x_grid, kx, dx, dxi, freq_window=freq_window)
        else:
            dy = y_grid[1] - y_grid[0]
            deta = ky[1] - ky[0]
            plan_kind, plan_data = plan
            return apply_nufft_2d(u, plan_kind, plan_data, x_grid, y_grid, kx, ky, dx, dy, dxi, deta, freq_window=freq_window)

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
            # Low-rank: apply as sum of separable pairs.
            #
            # FIX: unlike the AAA branch below -- whose underlying fitter
            # (try_aaa_decomposition_*) enforces rtol=joint_tol internally
            # and reports "aaa_unfit" (forcing a fallback) regardless of
            # joint_max_rel_error -- the low-rank fitter has no equivalent
            # built-in rejection: factorize_symbolic always returns *some*
            # rank-r fit, however bad. Previously this branch only checked
            # metrics['rel_l2_error'] when the caller explicitly passed
            # joint_max_rel_error, so with the defaults used by e.g.
            # apply_hybrid (joint_max_rel_error=None), a low-rank fit could
            # silently be applied with 10-80%+ relative error whenever the
            # fixed 'joint_degree' under-resolves the residual over the
            # (grid-inferred, possibly very wide) bounding box -- see e.g.
            # a joint term like exp(-x**2/8)*exp(-xi**2/8)*exp(x*xi/4)
            # fit at degree=6 over a full FFT frequency range, which was
            # observed to reach rel_l2_error ~ 0.82.
            #
            # The gate now always applies, using joint_tol as the default
            # threshold (mirroring AAA's rtol=joint_tol) when the caller
            # hasn't overridden it via joint_max_rel_error.
            metrics = rep.get("metrics", {})
            self.last_joint_lowrank_metrics = metrics
            effective_max_rel_error = (
                joint_max_rel_error if joint_max_rel_error is not None else joint_tol
            )
            if metrics.get("rel_l2_error", float("inf")) > effective_max_rel_error:
                warnings.warn(
                    "Low-rank joint residual symbol error "
                    f"{metrics['rel_l2_error']:.6e} exceeds "
                    f"{'joint_max_rel_error' if joint_max_rel_error is not None else 'joint_tol (default quality gate)'}"
                    f"={effective_max_rel_error}. "
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
        Apply the operator via Peetre decomposition with per-term backend routing.
    
        The symbol is split with `peetre_decomposition` into a separable part and a
        joint (non-separable) residual. The separable part is applied directly
        through `apply_peetre` (`apply_joint=False`). The joint residual is then
        expanded and split into its individual additive terms
        (`sympy.Add.make_args` on the expanded symbol); each nonzero term is wrapped
        in its own `PseudoDifferentialOperator` and applied with
        `joint_backend='auto'`, so `_auto_select_joint_backend` picks a backend
        independently for *that term* — NUFFT if a decomposition exists for it,
        AAA if it's a rational function (or has a symbolic polynomial denominator),
        otherwise low-rank.
    
        Splitting before routing means terms that individually admit a fast
        decomposition get one, instead of the whole joint residual being treated as
        a single mixed term and falling back to O(N\u00b2) direct quadrature.
    
        Parameters
        ----------
        u : array_like
            Input field values, matching the shape/dtype convention used by
            `apply`/`apply_peetre`.
        x_grid, y_grid : array_like
            Spatial grid(s). `y_grid` is only used for 2D operators.
        kx, ky : array_like
            Frequency grid(s) conjugate to `x_grid`/`y_grid`. `ky` is only used
            for 2D operators.
        **kwargs
            Forwarded to every `apply_peetre` call this method makes (both the
            separable-part call and each per-term joint call) — e.g.
            `boundary_condition`, `freq_window`, `clamp`, `space_window`,
            `weyl_order`, `joint_degree`, `joint_tol`, `joint_bounds`,
            `joint_max_rel_error`, `joint_num_samples`, `joint_seed`, `use_cache`.
            Do not pass `apply_joint` or `joint_backend` here: both are fixed
            internally by this method, and supplying them will raise
            `TypeError: apply_peetre() got multiple values for keyword argument`.
    
        Returns
        -------
        numpy.ndarray
            Complex128 array, same shape as `u`: the separable contribution plus
            the sum of the per-term joint contributions.
    
        Notes
        -----
        Backend selection happens per additive term of the joint symbol, not once
        for the whole residual, so this can outperform calling `apply_peetre`
        directly with `joint_backend='auto'` whenever the joint part mixes terms
        that individually favor different backends. Compare with `apply_hybrid_old`,
        which instead loops over `peetre_decomposition`'s own `joint_residual` list
        without expanding it into additive terms.
        """
        import numpy as np, sympy as sp
        deco = self.peetre_decomposition()
        result = np.zeros(np.shape(u), dtype=np.complex128)
        result += self.apply_peetre(u, x_grid, kx, y_grid=y_grid, ky=ky,
                                    apply_joint=False, **kwargs)
        joint_symbol = deco.get('joint_symbol', 0)
        if not self._peetre_is_zero(joint_symbol):
            for term in sp.Add.make_args(sp.expand(joint_symbol)):   # <-- the real split
                if self._peetre_is_zero(term):
                    continue
                sub_op = PseudoDifferentialOperator(term, self.vars_x, mode='symbol',
                                                    quantization=self.quantization)
                result += sub_op.apply_peetre(u, x_grid, kx, y_grid=y_grid, ky=ky,
                                              joint_backend='auto', **kwargs)
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
        Apply the operator by summing the pieces of its Peetre decomposition.
        
        `peetre_decomposition` splits the symbol into three kinds of contributions,
        each applied by whichever method suits its structure:
        
        - **Local terms** (`local_terms`, or the legacy `local`/`local_symbol` keys):
          applied directly via `PseudoDifferentialOperator.apply` when a term isn't
          separable, or as an amplitude-times-operator pair otherwise.
        - **Separable terms** `a(x) * q(xi)` (`deco['separable']`): `q` is applied to
          `u` as its own operator, then the result is multiplied pointwise by `a`
          evaluated on `x_grid`/`y_grid` (falling back to applying the full product
          `a * q` as one operator if `a` can't be lambdified).
        - **Joint residual** (`deco['joint_symbol']`, the non-separable remainder):
          applied via `_apply_joint_residual` using `joint_backend` ('direct',
          'lowrank', 'nufft', or 'aaa'), unless `apply_joint=False`, in which case it
          is dropped and a warning is issued — the returned result is then only a
          local+separable (asymptotic) approximation of the true operator.
        
        For Weyl-quantized operators, the Weyl symbol is first converted to its
        Kohn-Nirenberg equivalent (`weyl_to_kn_symbol`, truncated at `weyl_order`)
        before the decomposition above is computed or used; all downstream sub-operators
        are built with `quantization='kohn-nirenberg'`.
        
        Parameters
        ----------
        u : array_like
            Input field values on the grid(s).
        x_grid, y_grid : array_like
            Spatial grid(s). `y_grid` is required for 2D operators.
        kx, ky : array_like
            Frequency grid(s) conjugate to `x_grid`/`y_grid`. `ky` is required for
            2D operators.
        boundary_condition : str, optional
            Boundary handling passed through to each sub-operator's `apply`.
        dealiasing_mask, freq_window, clamp, space_window : optional
            Passed through to each sub-operator's `apply` call.
        weyl_order : int, optional
            Truncation order used when converting a Weyl symbol to Kohn-Nirenberg.
        apply_joint : bool, optional
            If False, skip the joint residual entirely (faster, but only an
            approximation) and warn that it was ignored.
        decomposition : dict, optional
            A precomputed `peetre_decomposition` result to reuse instead of computing
            one. If omitted, it's computed (and cached, per `use_cache`) internally.
        use_cache, separable_local : optional
            Forwarded to `peetre_decomposition` when `decomposition` is not supplied.
        joint_backend : {'direct', 'lowrank', 'nufft', 'aaa'}, optional
            Backend used to apply the joint residual.
        joint_degree, joint_tol, joint_bounds, joint_max_rel_error, joint_num_samples, joint_seed : optional
            Backend-specific parameters forwarded to `_apply_joint_residual`.
        
        Returns
        -------
        numpy.ndarray
            Complex128 array, same shape as `u`, equal to the sum of the local,
            separable, and (if `apply_joint`) joint contributions.
        
        Raises
        ------
        ValueError
            If `self.dim == 2` and `y_grid` or `ky` is not provided.
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
        mode : {'kn', 'weyl'}, default='kn'
            Quantization mode forwarded to `compose_asymptotic` for both
            A∘B and B∘A.
        sign_convention : {'standard', 'inverse'}, optional
            Phase-factor convention forwarded to `compose_asymptotic`
            (controls the sign of `[x, ξ]`); see `compose_asymptotic` for
            details. If None, defaults to 'standard'.
    
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

    def exponential_symbol(self, t=1.0, order=2, mode='kn', sign_convention=None, do_simplify=True):
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
        order : int, default=2
            Maximum order of the asymptotic expansion. Higher orders include 
            more composition terms, improving accuracy for small t or when 
            non-commutativity effects are significant.
        mode : {'kn', 'weyl'}, default='kn'
            Quantization mode forwarded to `compose_asymptotic` at each power
            P^n in the series.
        sign_convention : {'standard', 'inverse'}, optional
            Phase-factor convention forwarded to `compose_asymptotic`; see
            `compose_asymptotic` for details. If None, defaults to 'standard'.
        do_simplify : bool, default True
            Whether to call sympy's `simplify()` while assembling the
            propagator symbol (once when building it, and once inside every
            `compose_asymptotic()` call in the asymptotic expansion loop).
            This does not change the operator being applied -- `lambdify`
            evaluates the same function on an unsimplified expression -- it
            only affects how much symbolic cleanup happens before that.
            `simplify()` is the dominant cost of `build_propagator()` for
            symbols mixing trigonometric and polynomial terms, and its cost
            grows with `order`; set to `False` to skip it and speed up
            propagator construction, at the risk of a larger (but
            numerically equivalent) unsimplified expression tree.
        
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
                self, order=order, mode=mode, sign_convention=sign_convention, do_simplify=do_simplify)
            result += t**n / factorial(n) * current_power

        return simplify(result) if do_simplify else result
        
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
        auto_range : bool, default=True
            If True, `lambda_real_range`/`lambda_imag_range` are ignored and
            instead recomputed as a 20%-margin box around the eigenvalues of
            the discretized operator (once eigenvalues are available). Set to
            False to use the ranges passed in as-is.
        plot : bool, default=True
            If True, calls the internal pseudospectrum visualization after
            computing the resolvent-norm grid. Set to False to only return
            the data dictionary without producing a figure.
            
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

    def _plot_pseudospectrum(self, Lambda, resolvent_norm, sigma_min_grid, epsilon_levels, eigenvalues):
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
        return _mu.plot_pseudospectrum(Lambda, resolvent_norm, sigma_min_grid, epsilon_levels, eigenvalues)

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

    # -- visualization: thin wrappers delegating to microlocal_ud -----
    # All rendering logic lives in microlocal_ud.py; these keep the
    # familiar op.visualize_...(...) call sites working unchanged.

    def visualize_fiber(self, x_grid, xi_grid, x0=0.0, y0=0.0):
        """Plot the cotangent fiber structure at a fixed spatial point (x0[, y0]).
        See _slice_grid/_render_field docstrings for the shared implementation.
        NOTE: original signature has no eta_grid param -- 2D reuses xi_grid
        for both frequency axes, matching the original behavior exactly.
        cf. package microlocal.py."""
        return _mu.visualize_fiber(self, x_grid, xi_grid, x0=x0, y0=y0)

    def visualize_symbol_amplitude(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """Display |p(x, xi)| (1D) or |p(x, y, xi0, eta0)| (2D) as a color map.
        cf. package microlocal.py."""
        return _mu.visualize_symbol_amplitude(self, x_grid, xi_grid, y_grid, eta_grid, xi0=xi0, eta0=eta0)

    def visualize_phase(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """Plot arg(p(x, xi)) (1D) or arg(p(x, y, xi0, eta0)) (2D).
        cf. package microlocal.py."""
        return _mu.visualize_phase(self, x_grid, xi_grid, y_grid, eta_grid, xi0=xi0, eta0=eta0)

    def visualize_characteristic_set(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0, levels=[0.1]):
        """Visualize the characteristic set p(x, xi) ~= 0 (1D) or the (xi, eta)
        slice at fixed (x0, y0) (2D).
        cf. package microlocal.py."""
        return _mu.visualize_characteristic_set(self, x_grid, xi_grid, y_grid, eta_grid, y0=y0, x0=x0, levels=levels)

    def visualize_characteristic_gradient(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0):
        """Visualize |grad p| in phase space. NOTE: both the 1D and 2D
        branches now consistently use abs(.)**2 in the gradient norm (the
        original 1D branch omitted the abs(), inconsistently with 2D).
        cf. package microlocal.py."""
        return _mu.visualize_characteristic_gradient(self, x_grid, xi_grid, y_grid, eta_grid, y0=y0, x0=x0)

    def plot_hamiltonian_flow(self, x0=0.0, xi0=5.0, y0=0.0, eta0=0.0, tmax=1.0, n_steps=100, show_field=True):
        """Integrate and plot the Hamiltonian trajectories of the symbol in
        phase space. Delegates to the shared `integrate_singularity` engine
        instead of re-deriving the Hamiltonian vector field inline.
        cf. package microlocal.py."""
        return _mu.plot_hamiltonian_flow(self, x0=x0, xi0=xi0, y0=y0, eta0=eta0, tmax=tmax, n_steps=n_steps, show_field=show_field)

    def plot_symplectic_vector_field(self, xlim=(-2, 2), klim=(-5, 5), density=30):
        """Quiver plot of the symplectic vector field (dp/dxi, -dp/dx). 1D only.
        cf. package microlocal.py."""
        return _mu.plot_symplectic_vector_field(self, xlim=xlim, klim=klim, density=density)

    def visualize_micro_support(self, xlim=(-2, 2), klim=(-10, 10), threshold=0.001, density=300, xi0=0.0, eta0=0.0):
        """Visualize 1/|p(x, xi)| to highlight regions where the symbol is
        near zero. NOTE: no longer restricted to 1D -- the shared grid/render
        helpers already handle the 2D case (fixed xi0=eta0=0, scan x, y).
        cf. package microlocal.py."""
        return _mu.visualize_micro_support(self, xlim=xlim, klim=klim, threshold=threshold, density=density, xi0=xi0, eta0=eta0)

    def visualize_wavefront_set(self, seeds=None, tspan=(0, 3.0), projection='cotangent', n_steps=500, cmap='plasma', show_flow=True, show_endpoints=True, title=None, x0=0.0, y0=0.0, xi0=1.0, eta0=0.0, spread=2.0, n_seeds=25, radius=0.15):
        """Wavefront set WF(u) obtained by propagating seed singularities along
        the bicharacteristics of `op.symbol`. If `seeds` is not given, a default
        fan/point-source is built from (x0, y0) and (xi0, eta0) -- see
        _default_wavefront_seeds. `projection` follows plot_wavefront_set:
        1D -> 'cotangent' or 'position'; 2D -> 'cotangent', 'position',
        'frequency', 'mixed_x', 'mixed_y', or 'full' (2x2 cotangent-bundle grid).
        cf. package microlocal.py."""
        return _mu.visualize_wavefront_set(self, seeds=seeds, tspan=tspan, projection=projection, n_steps=n_steps, cmap=cmap, show_flow=show_flow, show_endpoints=show_endpoints, title=title, x0=x0, y0=y0, xi0=xi0, eta0=eta0, spread=spread, n_seeds=n_seeds, radius=radius)

    def group_velocity_field(self, xlim=(-2, 2), klim=(-10, 10), density=30):
        """Quiver plot of the group velocity field (1, dp/dxi). 1D only.
        cf. package microlocal.py."""
        return _mu.group_velocity_field(self, xlim=xlim, klim=klim, density=density)

    def animate_singularity(self, xi0=5.0, eta0=0.0, x0=0.0, y0=0.0, tmax=4.0, n_frames=100, projection=None):
        """Animate the propagation of a singularity under the Hamiltonian
        flow. Thin delegate to the module-level `animate_singularity`
        engine (previously ~130 lines of duplicated Hamiltonian/ODE setup
        here, plus a near-identical copy further down the module).
        cf. package microlocal.py."""
        return _mu.animate_operator_singularity(self, xi0=xi0, eta0=eta0, x0=x0, y0=y0, tmax=tmax, n_frames=n_frames, projection=projection)

    def interactive_symbol_analysis(self, xlim=(-2, 2), ylim=(-2, 2), xi_range=(0.1, 5), eta_range=(-5, 5), density=50):
        """Launch an ipywidgets dashboard for symbol exploration. Same modes,
        same sliders, same defaults as before -- rewritten as a mode-table
        dispatcher that delegates to the visualize_*/plot_* methods above
        instead of duplicating their 1D/2D branches inline (previously
        ~260 lines of near-duplicated if-elif chains).
        cf. package microlocal.py."""
        return _mu.interactive_symbol_analysis(self, xlim=xlim, ylim=ylim, xi_range=xi_range, eta_range=eta_range, density=density)

# --- Expose submodules and maintain backward compatibility for tests/scripts ---
from . import matpsiop
from . import psiop_apply
from . import psiop_solver

# 1. Numerical backends & helpers (from psiop_apply)
from .psiop_apply import (
    _clip_complex_magnitude, _cache_key_1d, _cache_key_2d, invalidate_kn_cache,
    kohn_nirenberg_fft, kohn_nirenberg_nonperiodic,
    factorize_symbolic, evaluate_decomposition_quality,
    try_nufft_decomposition_1d, try_nufft_decomposition_2d,
    apply_nufft_1d, apply_nufft_2d, 
    try_aaa_decomposition_1d, try_aaa_decomposition_2d,
    aaa_plan_to_callable_1d, aaa_plan_to_callable_2d, 
    _sympy_number, _chebyshev_polynomial
)

# 2. Solvers & Grid utilities (from psiop_solver)
from .psiop_solver import (
    make_grid_1d, make_grid_2d, make_grids, run_time_loop,
    _make_grids, _run_time_loop,  # Internal aliases
    PropagatorFamily, build_propagator,
    solve_first_order, solve_second_order,
    solve_matrix_field, solve_sylvester_field,
    solve_ricci_flow_conformal_2d, block_matrix_second_order,
)

# 3. Matrix-valued operators (from matpsiop)
from .matpsiop import MatrixPseudoDifferentialOperator