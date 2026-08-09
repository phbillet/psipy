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
psiop.py — Symbolic‑numerical toolkit for pseudo‑differential operators in 1D/2D
=======================================================================================

Overview
--------
The `psiop` module provides a unified framework for constructing, manipulating, and numerically applying pseudo‑differential operators (ΨDOs) in one and two spatial dimensions. It combines **symbolic** construction and calculus of operator symbols (using SymPy) with **numerical** evaluation, quantization, and visualisation (using NumPy/SciPy/Matplotlib). The package is designed for researchers and students working in microlocal analysis, spectral theory, and the numerical analysis of PDEs.

Key features include:

* Symbol creation either from an explicit expression (symbol mode) or by automatic extraction from a differential operator (auto mode), in both Kohn‑Nirenberg and Weyl quantisation, with symbolic conversion between the two conventions.
* Computation of asymptotic expansions of symbols at high frequencies, and determination of the operator order (homogeneity degree) via symbolic/numerical heuristics.
* Asymptotic symbolic calculus: composition, commutators, formal left/right inverses, the formal adjoint, fractional powers, and the symbol of the exponential `exp(tP)` via series expansion.
* **Peetre decomposition**: automatic classification of a symbol into local (polynomial‑in‑ξ), separable, and joint (entangled) parts, with Chebyshev/SVD low‑rank factorization (`factorize_symbolic`) of residual joint terms into short sums of separable pairs `a_k(x) q_k(ξ)`, together with Monte Carlo quality diagnostics.
* Two complementary numerical application backends: a direct pointwise‑symbol evaluation on the space‑frequency grid, and a Peetre‑decomposition‑based backend that applies local, separable, and low‑rank joint terms through periodic (FFT) or non‑periodic Kohn‑Nirenberg quantization, with windowing, dealiasing, magnitude clamping, and result caching.
* Ellipticity and self‑adjointness checks, evaluated both symbolically and numerically on user‑supplied grids.
* Semiclassical trace formula (symbolic or numerical) and pseudospectrum computation with adaptive grid refinement, sparse/dense eigenvalue solvers, and parallelisation.
* Hamiltonian flow associated with the principal symbol (symplectic phase‑space dynamics), including the symplectic vector field and singularity‑propagation animation along bicharacteristics.
* Rich visualisation suite: symbol amplitude/phase, cotangent‑fiber structure, characteristic set and its gradient, group velocity field, micro‑support, Hamiltonian trajectories, and pseudospectrum contour plots.
* Interactive dashboard (ipywidgets) for real‑time exploration of the symbol and its associated phase‑space structures.

Mathematical background
-----------------------
A pseudo‑differential operator `P` acting on functions of `x ∈ ℝⁿ` is formally defined by its **symbol** `p(x,ξ)`, a function on phase space `T*ℝⁿ`. The action on a function `u` is given by the Kohn‑Nirenberg quantisation

    (P u)(x) = (2π)^{-n} ∫_{ℝⁿ} e^{i x·ξ} p(x,ξ) û(ξ) dξ ,

where `û` is the Fourier transform of `u`; the Weyl quantisation instead evaluates the symbol at the midpoint of interacting spatial points, and the two conventions are related by an explicit, order‑truncated correction that the module computes symbolically. If the symbol does not depend on `x`, the operator reduces to a Fourier multiplier. For a general spatially varying symbol, the above representation provides a rigorous extension of differential operators.

The **asymptotic behaviour** of the symbol as `|ξ| → ∞` determines many properties of the operator. The **principal symbol** `pₘ` is the homogeneous component of highest order `m`. When the symbol is non‑homogeneous, the module attempts to estimate the effective order by expanding in inverse powers of `|ξ|` (or in a radial variable for 2D).

**Symbolic calculus** allows one to compose operators asymptotically. For two symbols `p` and `q`, the symbol of the composition `P ∘ Q` is given by an asymptotic series

    (p ∘ q)(x,ξ) ~ Σ_{α} (i)^{-|α|}/α! ∂_ξ^α p(x,ξ) ∂_x^α q(x,ξ)

in the Kohn‑Nirenberg convention (a similar expansion exists for the Weyl star product). Truncating this series yields approximate compositions valid for high frequencies or slowly varying symbols, and the same machinery underlies the module's commutator, formal inverse, adjoint, fractional‑power, and matrix‑exponential‑symbol constructions, assuming the principal symbol never vanishes.

For numerical application, a joint symbol `p(x,ξ)` is further split — via the **Peetre decomposition** — into a local polynomial part (differential operators, applied directly), separable Fourier‑multiplier terms `a(x) q(ξ)` (applied as pointwise multiplication sandwiched around an FFT), and a residual joint part that is not exactly separable. This joint residual is approximated on a bounded phase‑space window by a short Chebyshev/SVD low‑rank expansion `sum_k a_k(x) q_k(ξ)`, reducing the numerical application of a general ΨDO to a small number of FFT‑based separable applications, at high frequencies or over spatial domains where the decomposition is accurate.

The **Hamiltonian flow** generated by the principal symbol describes the propagation of singularities along bicharacteristics – a cornerstone of microlocal analysis.

The **pseudospectrum** `σ_ε(P)` is the set of `λ ∈ ℂ` for which `‖(P-λI)^{-1}‖ ≥ ε^{-1}`. It captures the near‑spectral behaviour of non‑normal operators and is visualised via contour plots of the resolvent norm, computed on a matrix discretisation of `P` with optional adaptive grid refinement.

References
----------
.. [1] Hörmander, L.  *The Analysis of Linear Partial Differential Operators III*, Springer, 1985.  Chapter 18: Pseudo‑differential Operators.
.. [2] Taylor, M. E.  *Pseudo Differential Operators*, Princeton University Press, 1981.
.. [3] Zworski, M.  *Semiclassical Analysis*, American Mathematical Society, 2012.  Chapter 4: Pseudo‑differential Operators.
.. [4] Martinez, A.  *An Introduction to Semiclassical and Microlocal Analysis*, Springer, 2002.
.. [5] Trefethen, L. N. & Embree, M.  *Spectra and Pseudospectra*, Princeton University Press, 2005.  (For pseudospectrum methods.)
.. [6] Peetre, J.  "Applications de la théorie des espaces d'interpolation dans l'analyse harmonique."  *Ricerche di Matematica*, 1968.  (For separable/local decomposition ideas underlying the Peetre backend.)
"""

from imports import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

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
        """
        Clear cached symbol evaluations and Peetre decompositions.
        """
        self.symbol_cached = None

        if hasattr(self, "_peetre_cache"):
            self._peetre_cache = None

        if hasattr(self, "_peetre_decomposition"):
            self._peetre_decomposition = None

        if hasattr(self, "_joint_lowrank_cache"):
            self._joint_lowrank_cache = None

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
        """
        Estimate the homogeneity order of the pseudo-differential symbol in high-frequency asymptotics.
    
        This method attempts to determine the leading-order behavior of the symbol p(x, ξ) or p(x, y, ξ, η)
        as |ξ| → ∞ (in 1D) or |(ξ, η)| → ∞ (in 2D). The returned value represents the asymptotic growth or decay rate,
        which is essential for understanding the regularity and mapping properties of the corresponding operator.
    
        The function uses symbolic preprocessing to ensure proper factorization of frequency variables,
        especially in sqrt and power expressions, to avoid erroneous order detection (e.g., due to hidden scaling).
    
        Parameters
        ----------
        max_order : int, optional
            Maximum number of terms to consider in the series expansion. Default is 10.
        tol : float, optional
            Tolerance threshold for evaluating the coefficient magnitude. If the coefficient is too small,
            the detected order may be discarded. Default is 1e-3.
    
        Returns
        -------
        float or None
            - If the symbol is homogeneous, returns its exact homogeneity degree as a float.
            - Otherwise, estimates the dominant asymptotic order from leading terms in the expansion.
            - Returns None if no valid order could be determined.
    
        Notes
        -----
        - In 1D:
            Two strategies are used:
                1. Expand directly in xi at infinity.
                2. Substitute xi = 1/z and expand around z = 0.
    
        - In 2D:
            - Transform the symbol into polar coordinates: (xi, eta) = rho*(cos(theta), sin(theta)).
            - Expand in rho at infinity, then extract the leading term's power.
            - An alternative substitution using 1/z is also tried if the first method fails.
    
        - Preprocessing steps:
            - Sqrt expressions involving frequencies are rewritten to isolate the leading variable.
            - Power expressions are factored explicitly to ensure correct symbolic scaling.
    
        - If the symbol is not homogeneous, a warning is issued, and the result should be interpreted with care.
        
        - For non-homogeneous symbols, only the principal asymptotic term is considered.
    
        Raises
        ------
        NotImplementedError
            If the spatial dimension is neither 1 nor 2.
        """
        from sympy import (
            symbols, series, simplify, sqrt, cos, sin, oo, powdenest, radsimp,
            expand, expand_power_base
        )
    
        def preprocess_sqrt(expr, freq):
            return expr.replace(
                lambda e: e.func == sqrt and freq in e.free_symbols,
                lambda e: freq * sqrt(1 + (e.args[0] - freq**2) / freq**2)
            )
    
        def preprocess_power(expr, freq):
            return expr.replace(
                lambda e: e.is_Pow and freq in e.free_symbols,
                lambda e: freq**e.exp * (1 + e.base / freq**e.base.as_powers_dict().get(freq, 0))**e.exp
            )
    
        def validate_order(power, coeff, vars_x, tol):
            if power is None:
                return None
#            if any(v in coeff.free_symbols for v in vars_x):
#                print("⚠️ Coefficient depends on spatial variables; ignoring")
#                return None
                
            if simplify(coeff) == 0 or coeff.equals(0):
                print("⚠️ Coefficient is symbolically zero; ignoring")
                return None
                
            return int(power) if power == int(power) else float(power)
    
        # Homogeneity check
        is_homog, degree = self.is_homogeneous()
        if is_homog:
            return float(degree)
        else:
            print("⚠️ The symbol is not homogeneous. The asymptotic order is not well defined.")
    
        if self.dim == 1:
            x = self.vars_x[0]
            # Reuse the xi symbol actually present in self.symbol (see the same
            # fix in is_homogeneous) instead of a fresh, mismatched positive one.
            xi = next((s for s in self.symbol.free_symbols if s.name == 'xi'), symbols('xi', real=True))
    
            try:
                print("1D symbol_order - method 1")
                expr = preprocess_sqrt(self.symbol, xi)
                s = series(expr, xi, oo, n=max_order).removeO()
                lead = simplify(powdenest(s.as_leading_term(xi), force=True))
                power = lead.as_powers_dict().get(xi, None)
                coeff = lead / xi**power if power is not None else 0
                print("lead =", lead)
                print("power =", power)
                print("coeff =", coeff)
                order = validate_order(power, coeff, [x], tol)
                if order is not None:
                    return order
            except Exception:
                pass
    
            try:
                print("1D symbol_order - method 2")
                z = symbols('z', real=True, positive=True)
                expr_z = preprocess_sqrt(self.symbol.subs(xi, 1/z), 1/z)
                s = series(expr_z, z, 0, n=max_order).removeO()
                lead = simplify(powdenest(s.as_leading_term(z), force=True))
                power = lead.as_powers_dict().get(z, None)
                coeff = lead / z**power if power is not None else 0
                print("lead =", lead)
                print("power =", power)
                print("coeff =", coeff)
                order = validate_order(power, coeff, [x], tol)
                if order is not None:
                    return -order
            except Exception as e:
                print(f"⚠️ fallback z failed: {e}")
            return None
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi = next((s for s in self.symbol.free_symbols if s.name == 'xi'), symbols('xi', real=True))
            eta = next((s for s in self.symbol.free_symbols if s.name == 'eta'), symbols('eta', real=True))
            rho, theta = symbols('rho theta', real=True, positive=True)
    
            try:
                print("2D symbol_order - method 1")
                p_rho = self.symbol.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
                p_rho = preprocess_power(preprocess_sqrt(p_rho, rho), rho)
                s = series(simplify(p_rho), rho, oo, n=max_order).removeO()
                lead = radsimp(simplify(powdenest(s.as_leading_term(rho), force=True)))
                power = lead.as_powers_dict().get(rho, None)
                coeff = lead / rho**power if power is not None else 0
                print("lead =", lead)
                print("power =", power)
                print("coeff =", coeff)
                order = validate_order(power, coeff, [x, y], tol)
                if order is not None:
                    return order
            except Exception as e:
                print(f"⚠️ polar expansion failed: {e}")
    
            try:
                print("2D symbol_order - method 2")
                z = symbols('z', real=True, positive=True)
                xi_eta = {xi: (1/z) * cos(theta), eta: (1/z) * sin(theta)}
                p_rho = preprocess_sqrt(self.symbol.subs(xi_eta), 1/z)
                s = series(simplify(p_rho), z, 0, n=max_order).removeO()
                lead = radsimp(simplify(powdenest(s.as_leading_term(z), force=True)))
                power = lead.as_powers_dict().get(z, None)
                coeff = lead / z**power if power is not None else 0
                print("lead =", lead)
                print("power =", power)
                print("coeff =", coeff)
                order = validate_order(power, coeff, [x, y], tol)
                if order is not None:
                    return -order
            except Exception as e:
                print(f"⚠️ fallback z (2D) failed: {e}")
            return None
    
        else:
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
    
        from sympy import diff, factorial, simplify, symbols
    
        assert self.dim == other.dim, "Operator dimensions must match"
        p, q = self.symbol, other.symbol
    
        # Default sign convention
        if sign_convention is None:
            sign_convention = 'standard'
        sign = -1 if sign_convention == 'standard' else +1
    
        # --- 1D case ---
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            result = 0
    
            if mode == 'kn':  # Kohn–Nirenberg
                for n in range(order + 1):
                    term = (1 / factorial(n)) * diff(p, xi, n) * diff(q, x, n) * (1j) ** (sign * n)
                    result += term
    
            elif mode == 'weyl':  # Weyl symmetric composition
                # Weyl star product: exp((i/2)(∂_ξ^p ∂_x^q - ∂_x^p ∂_ξ^q))
                result = 0
                for n in range(order + 1):
                    for k in range(n + 1):
                        # k derivatives acting as (∂_ξ^k p)(∂_x^(n−k) q)
                        coeff = (1 / (factorial(k) * factorial(n - k))) * ((1j / 2) ** n) * ((-1) ** (n - k))
                        term = coeff * diff(p, xi, k, x, n - k, evaluate=True) * diff(q, x, k, xi, n - k, evaluate=True)
                        result += term
    
            else:
                raise ValueError("mode must be either 'kn' or 'weyl'")
    
            # Replace: return simplify(result)
            try:
                return simplify(result)
            except TypeError:
                # Fallback for SymPy sorting bugs with undefined functions/derivatives
                return result 
    
        # --- 2D case ---
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            result = 0
    
            if mode == 'kn':
                for n in range(order + 1):
                    for i in range(n + 1):
                        j = n - i
                        term = (1 / (factorial(i) * factorial(j))) * \
                               diff(p, xi, i, eta, j) * diff(q, x, i, y, j) * (1j) ** (sign * n)
                        result += term
    
            elif mode == 'weyl':
                for n in range(order + 1):
                    for i in range(n + 1):
                        j = n - i
                        coeff = (1 / (factorial(i) * factorial(j))) * ((1j / 2) ** n) * ((-1) ** (n - i))
                        term = coeff * diff(p, xi, i, eta, j, x, 0, y, 0) * diff(q, x, i, y, j, xi, 0, eta, 0)
                        result += term
            else:
                raise ValueError("mode must be either 'kn' or 'weyl'")
    
            # Replace: return simplify(result)
            try:
                return simplify(result)
            except TypeError:
                # Fallback for SymPy sorting bugs with undefined functions/derivatives
                return result 
    
        else:
            raise NotImplementedError("Only 1D and 2D cases are implemented")

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
        
    def peetre_decomposition(
        self,
        use_cache=True,
        separable_local=False,
    ):
        """
        Symbolic Peetre-style decomposition of the operator symbol.

        The symbol is split into:

            local:
                Polynomial part in the frequency variables.

            separable:
                Sum of terms a(x) q(xi), with q frequency-only.

            joint_residual:
                Remaining genuinely entangled terms.


        Parameters
        ----------

        use_cache : bool, default=True
            Cache the decomposition.

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

        self._peetre_cache = {
            "symbol": self.symbol,
            "separable_local": separable_local,
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

        Remaining keyword arguments are passed to peetre_decomposition().

        Parameters
        ----------
        joint_backend : {'direct', 'lowrank'}, default='direct'
            'direct' prints the raw (un-factorized) joint residual terms,
            same as before. 'lowrank' factorizes the joint residual via
            factorize_symbolic() (through `_low_rank_joint_pairs`, the
            same helper used by `apply_peetre`) and prints the resulting
            short sum of separable pairs a_k(x) * q_k(xi) instead, along
            with the Monte Carlo approximation quality.
        joint_bounds : dict, optional
            Mapping from each space/frequency symbol to a (min, max)
            sampling range, required when joint_backend='lowrank' since
            no numerical grid is available here to infer bounds from.
        joint_degree, joint_tol, joint_num_samples, joint_seed :
            Forwarded to `_low_rank_joint_pairs` / `factorize_symbolic`.
            Ignored when joint_backend='direct'.

        Examples
        --------
        op.print_peetre_decomposition()
        op.print_peetre_decomposition(
            joint_backend="lowrank",
            joint_bounds={x: (-5, 5), xi: (-50, 50)},
        )

        """
        deco = self.peetre_decomposition(use_cache=use_cache, **kwargs)
        xi_vars = self._peetre_frequency_symbols()

        # --------------------------------------------------------------
        # Local terms.
        #
        # If local_terms is available, print it in the same style as
        # separable terms:
        #
        #     (a(x)) * (q(xi))
        #
        # This is more natural for variable-coefficient polynomial symbols.
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
        # Joint residual.
        #
        # If joint_backend='lowrank', factorize it via factorize_symbolic
        # (through the same `_low_rank_joint_pairs` helper apply_peetre()
        # uses) and print the resulting a_k(x)*q_k(xi) pairs instead of
        # the raw irreducible terms.
        # --------------------------------------------------------------
        joint_symbol = deco.get("joint_symbol", 0)

        if joint_backend == "lowrank" and not self._peetre_is_zero(joint_symbol):
            if joint_bounds is None:
                raise ValueError(
                    "joint_bounds must be provided when "
                    "joint_backend='lowrank' (no numerical grid is "
                    "available here to infer bounds from)."
                )

            pairs, metrics = self._low_rank_joint_pairs(
                joint_symbol,
                joint_bounds,
                degree=joint_degree,
                tol=joint_tol,
                num_samples=joint_num_samples,
                seed=joint_seed,
                use_cache=use_cache,
            )

            print(
                f"--- joint residual factorized into {len(pairs)} low-rank "
                f"term(s) via factorize_symbolic "
                f"(rel_l2_error={metrics.get('rel_l2_error', float('nan')):.3e}) ---"
            )
            for a, q in pairs:
                print(f"  ({a}) * ({q})")
        else:
            print(
                f"--- {len(deco['joint_residual'])} irreducible joint term(s) ---"
            )
            for t in deco["joint_residual"]:
                print(f"  {t}")

        print(
            f"local_symbol = {deco['local_symbol']}\n"
            f"separable_symbol = {deco['separable_symbol']}\n"
            f"joint_symbol = {deco['joint_symbol']}"
        )

        
    # ======================================================================
    # Peetre-based application
    # ======================================================================
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
            Apply a term a(x) q(xi) as:
    
                a(x) * Op(q) u
    
            The suboperator Op(q) is applied using the effective quantization.
            For Weyl operators, this is Kohn-Nirenberg because the Weyl symbol
            has already been converted before decomposition.
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
            def _apply_joint_direct():
                op_joint = PseudoDifferentialOperator(
                    joint_symbol,
                    self.vars_x,
                    mode="symbol",
                    quantization=peetre_quantization,
                )
                return op_joint.apply(
                    u,
                    x_grid,
                    kx,
                    **common_apply_kwargs,
                )
    
            if not apply_joint:
                warnings.warn(
                    "Peetre joint residual has been ignored. "
                    "The result is an asymptotic/local+separable approximation."
                )
    
            elif joint_backend == "direct":
                result = result + _apply_joint_direct()
    
            elif joint_backend == "lowrank":
                if joint_bounds is None:
                    joint_bounds = self._infer_joint_bounds(
                        x_grid,
                        kx,
                        y_grid=y_grid,
                        ky=ky,
                    )
    
                try:
                    pairs, metrics = self._low_rank_joint_pairs(
                        joint_symbol,
                        joint_bounds,
                        degree=joint_degree,
                        tol=joint_tol,
                        num_samples=joint_num_samples,
                        seed=joint_seed,
                        use_cache=use_cache,
                    )
    
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
                        result = result + _apply_joint_direct()
                    else:
                        for a_k, q_k in pairs:
                            result = result + _apply_separable_pair(a_k, q_k)
    
                except Exception as exc:
                    warnings.warn(
                        "Low-rank joint decomposition failed: "
                        f"{exc}. Falling back to direct joint application."
                    )
                    result = result + _apply_joint_direct()
    
            else:
                raise ValueError(
                    "joint_backend must be 'direct' or 'lowrank'."
                )
    
        return result

        
    def peetre_apply(self, *args, **kwargs):
        """
        Alias for apply_peetre().
        """
        return self.apply_peetre(*args, **kwargs)

    def apply_subdomain(
        self,
        u,
        x_grid,
        kx,
        g,
        f,
        h=None,
        y_grid=None,
        ky=None,
        sigma=None,
        kappa_f=1.0,
        kappa_h=1.0,
        n_ext=None,
        correction='gain',
        relax=0.5,
        tol=1e-6,
        max_iter=10,
        boundary_condition='periodic',
        assume_local=False,
        **apply_kwargs,
    ):
        """
        Apply Op_KN(s) to `u`, restricted to an implicitly-defined subdomain
        Omega = {g <= 0} bounded by C = {g = 0}, with Dirichlet data `f`
        (and optional Neumann data `h`) injected on C.

        Implements the Kohn-Nirenberg Subdomain Boundary Integration
        Pipeline, Revision 2: boundary data is approximated via localized
        diffuse-interface injection (single-/double-layer shells built from
        a smoothed indicator of Omega), the operator is applied unchanged
        through the existing fast `apply()` path, and a defect-correction
        loop (Step 7) iterates the injected data against the measured trace
        residual `v_Omega|_C - f`.

        Scope
        -----
        This is only justified for **local or weakly-nonlocal** `Op_KN(s)`
        (differential operators, elliptic-type multipliers, rapidly-decaying
        kernels): the solution's behaviour near `C` must be controlled by
        data in a neighbourhood of `C`. It is **not** justified as-is for
        genuinely nonlocal symbols (e.g. fractional-Laplacian `|xi|^s`
        type), where the correct boundary-value problem needs data on all
        of the exterior of Omega, not just on `C`. A warning is emitted
        unless `assume_local=True`; treat results for nonlocal symbols as
        an unvalidated heuristic (benchmark against a known nonlocal
        reference solution first).

        Parameters
        ----------
        u : ndarray
            Interior field, sampled on the full grid.
        x_grid, kx : ndarray
            Spatial / frequency grid along x, as for `apply()`.
        g : ndarray
            Samples of the implicit level-set function on the grid;
            `Omega = {g <= 0}`, `C = {g = 0}`.
        f : ndarray
            Dirichlet data, sampled on the full grid. Only its values near
            `C` matter, since the injection is supported in a thin collar.
        h : ndarray, optional
            Neumann data, sampled on the full grid. If given, the
            double-layer shell is injected and its trace residual tracked
            alongside the Dirichlet one.
        y_grid, ky : ndarray, optional
            Spatial / frequency grid along y (2D only).
        sigma : float, optional
            Gaussian smoothing width for the geometry. Defaults to
            `2.5 * dx` (auto-tuned); should resolve to 5-10 grid points
            across the transition collar.
        kappa_f, kappa_h : float, default 1.0
            Initial single-/double-layer gains.
        n_ext : tuple of ndarray, optional
            Outward-normal extension contracted with `n_delta` for the
            Neumann injection. Defaults to the geometric outward unit
            normal `-n_delta / rho_delta`.
        correction : {'gain', 'full', None}, default 'gain'
            Defect-correction strategy (Step 7). `'gain'` takes secant
            updates of the scalar gains `kappa_f`/`kappa_h`, driven by a
            `rho_delta`-weighted *mean* of the trace residual -- a linear
            functional that is exactly affine in `kappa` (Steps 2-6 are
            linear), so this converges in one secant step once two probe
            points are available. `'full'` does a relaxed fixed-point
            update of `f`/`h` directly (Picard iteration for the boundary
            integral equation this pipeline implicitly approximates); it
            is not guaranteed to be a contraction for every operator/
            geometry, so it is damped by `relax` and guarded against
            blow-up (see `relax`). `None` performs a single forward pass,
            no correction.
        relax : float, default 0.5
            Damping factor for the `'full'` correction's fixed-point
            update, `f_ext <- f_ext - relax * residual_field`. Lower it
            (e.g. 0.1-0.3) if the residual grows across iterations.
            Ignored for `'gain'` and `None`. Regardless of `relax`, the
            iteration is monitored: if the residual grows past 10x its
            best value so far, it stops and the best iterate found is
            returned rather than a diverged one.
        tol : float, default 1e-6
            Stop once the `rho_delta`-weighted Dirichlet (and Neumann, if
            `h` is given) trace residual falls below this.
        max_iter : int, default 10
            Iteration budget for the defect-correction loop.
        boundary_condition, **apply_kwargs
            Forwarded to `self.apply()` for Step 5 (Op_KN(s) evaluation) --
            the pipeline only changes what is fed into the operator, not
            how the operator itself is applied.

        Returns
        -------
        dict
            `v_Omega` (output field restricted to Omega), `chi_Omega`
            (smooth indicator), `residual_D` / `residual_N` (final
            weighted trace residual norms, `residual_N` is `None` if `h`
            was not given), `kappa_f`, `kappa_h` (final gains), `n_iter`,
            `converged`. Report `residual_D`/`residual_N` alongside
            `v_Omega` as a trust diagnostic, the same role
            `rel_l2_error` plays for the low-rank Peetre fallback.
        """
        if not assume_local:
            warnings.warn(
                "apply_subdomain approximates boundary data via localized "
                "diffuse-interface injection, which is only justified for "
                "local or weakly-nonlocal Op_KN(s). For genuinely nonlocal "
                "symbols (e.g. fractional-Laplacian |xi|^s type) this is an "
                "unvalidated heuristic -- see the 'Scope restriction' in "
                "kn_subdomain_boundary_pipeline_v2.md. Pass "
                "assume_local=True to silence this warning.",
                stacklevel=2,
            )
        if correction not in ('gain', 'full', None):
            raise ValueError("correction must be 'gain', 'full' or None")

        chi_Omega, n_delta, rho_delta, sigma = subdomain_masks(
            g, x_grid, self.fft, self.ifft, y_grid=y_grid, sigma=sigma, dim=self.dim
        )

        if n_ext is None:
            eps = np.finfo(rho_delta.dtype).tiny
            n_ext = tuple(-c / (rho_delta + eps) for c in n_delta)

        f_ext = np.asarray(f)
        h_ext = None if h is None else np.asarray(h)
        u_Omega = chi_Omega * u

        prev_f = prev_h = None  # (kappa, signed residual) pairs, for secant updates
        best_score = float('inf')
        best = None  # (v_Omega, residual_D, residual_N, kappa_f, kappa_h)

        for it in range(1, max_iter + 1):
            F_shell = kappa_f * f_ext * rho_delta
            H_shell = 0.0
            if h_ext is not None:
                n_dot_n = sum(ne * nd for ne, nd in zip(n_ext, n_delta))
                H_shell = kappa_h * h_ext * n_dot_n

            u_tot = u_Omega + F_shell + H_shell

            v_global = self.apply(
                u_tot, x_grid, kx,
                boundary_condition=boundary_condition,
                y_grid=y_grid, ky=ky,
                **apply_kwargs,
            )
            v_Omega = chi_Omega * v_global

            residual_field, residual_D = subdomain_trace_residual(v_Omega, f_ext, rho_delta)

            residual_field_N = residual_N = None
            if h_ext is not None:
                grad_v = _spectral_gradient(v_global, x_grid, y_grid, self.fft, self.ifft, self.dim)
                residual_field_N, residual_N = subdomain_trace_residual(
                    v_Omega, h_ext, rho_delta, grad_v=grad_v, n_ext=n_ext
                )

            # Track the best iterate seen (lowest residual): a diverging
            # 'full' Picard iteration -- not guaranteed to be a contraction
            # for every operator/geometry -- must never be allowed to
            # silently hand back a blown-up result.
            score = residual_D if residual_N is None else max(residual_D, residual_N)
            diverging = score > 10.0 * best_score
            if score < best_score:
                best_score = score
                best = (v_Omega, residual_D, residual_N, kappa_f, kappa_h)

            converged = residual_D <= tol and (residual_N is None or residual_N <= tol)
            if correction is None or converged or it == max_iter or diverging:
                break

            if correction == 'gain':
                signed_D = float(np.real(_weighted_mean(residual_field, rho_delta)))
                kappa_f, prev_f = _secant_update(kappa_f, signed_D, prev_f)
                if h_ext is not None:
                    signed_N = float(np.real(_weighted_mean(residual_field_N, rho_delta)))
                    kappa_h, prev_h = _secant_update(kappa_h, signed_N, prev_h)
            else:  # 'full'
                f_ext = f_ext - relax * residual_field
                if h_ext is not None:
                    h_ext = h_ext - relax * residual_field_N

        v_Omega, residual_D, residual_N, kappa_f, kappa_h = best
        converged = residual_D <= tol and (residual_N is None or residual_N <= tol)

        return {
            "v_Omega": v_Omega,
            "chi_Omega": chi_Omega,
            "residual_D": residual_D,
            "residual_N": residual_N,
            "kappa_f": kappa_f,
            "kappa_h": kappa_h,
            "n_iter": it,
            "converged": converged,
        }

        
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
        p = self.symbol
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            r = 1 / p.subs(xi, xi)  # r0
            R = r
            for n in range(1, order + 1):
                term = 0
                for k in range(1, n + 1):
                    coeff = (1j)**(-k) / factorial(k)
                    inner = diff(p, xi, k) * diff(R, x, k)
                    term += coeff * inner
                R = R - r * term
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            r = 1 / p.subs({xi: xi, eta: eta})
            R = r
            for n in range(1, order + 1):
                term = 0
                for k1 in range(n + 1):
                    for k2 in range(n + 1 - k1):
                        if k1 + k2 == 0: continue
                        coeff = (1j)**(-(k1 + k2)) / (factorial(k1) * factorial(k2))
                        dp = diff(p, xi, k1, eta, k2)
                        dR = diff(R, x, k1, y, k2)
                        term += coeff * dp * dR
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
        p = self.symbol
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            l = 1 / p.subs(xi, xi)
            L = l
            for n in range(1, order + 1):
                term = 0
                for k in range(1, n + 1):
                    coeff = (1j)**(-k) / factorial(k)
                    inner = diff(L, xi, k) * diff(p, x, k)
                    term += coeff * inner
                L = L - term * l
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            l = 1 / p.subs({xi: xi, eta: eta})
            L = l
            for n in range(1, order + 1):
                term = 0
                for k1 in range(n + 1):
                    for k2 in range(n + 1 - k1):
                        if k1 + k2 == 0: continue
                        coeff = (1j)**(-(k1 + k2)) / (factorial(k1) * factorial(k2))
                        dp = diff(p, x, k1, y, k2)
                        dL = diff(L, xi, k1, eta, k2)
                        term += coeff * dL * dp
                L = L - term * l
        return L

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
        p = self.symbol
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
            p_star = conjugate(p)
            p_star = simplify(series(p_star, xi, oo, n=6).removeO())
            return p_star
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            p_star = conjugate(p)
            p_star = simplify(series(p_star, sqrt(xi**2 + eta**2), oo, n=6).removeO())
            return p_star

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
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            
            # Initialize with identity
            result = 1
            
            # First order term: tP
            current_power = self.symbol
            result += t * current_power
            
            # Higher order terms: (t^n/n!) P^n computed via composition
            for n in range(2, order + 1):
                # Compute P^n = P^(n-1) ∘ P via asymptotic composition
                # We use a temporary operator for composition
                temp_op = PseudoDifferentialOperator(
                    current_power, [x], mode='symbol'
                )
                current_power = temp_op.compose_asymptotic(self, order=order, mode=mode, sign_convention=sign_convention)
                
                # Add term (t^n/n!) * P^n
                coeff = t**n / factorial(n)
                result += coeff * current_power
            
            return simplify(result)
        
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            
            # Initialize with identity
            result = 1
            
            # First order term: tP
            current_power = self.symbol
            result += t * current_power
            
            # Higher order terms: (t^n/n!) P^n computed via composition
            for n in range(2, order + 1):
                # Compute P^n = P^(n-1) ∘ P via asymptotic composition
                temp_op = PseudoDifferentialOperator(
                    current_power, [x, y], mode='symbol'
                )
                current_power = temp_op.compose_asymptotic(self, order=order, mode=mode, sign_convention=sign_convention)
                
                # Add term (t^n/n!) * P^n
                coeff = t**n / factorial(n)
                result += coeff * current_power
            
            return simplify(result)
        
        else:
            raise NotImplementedError("Only 1D and 2D operators are supported")
        
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
        """
        Plot the cotangent fiber structure at a fixed spatial point (x₀[, y₀]).
    
        This visualization shows how the symbol p(x, ξ) behaves on the cotangent fiber 
        above a fixed spatial point. In microlocal analysis, this provides insight into 
        the frequency content of the operator at that location.
    
        Parameters
        ----------
        x_grid : ndarray
            Spatial grid values (1D) for evaluation in 1D case.
        xi_grid : ndarray
            Frequency grid values (1D) for evaluation in both 1D and 2D cases.
        x0 : float, optional
            Fixed x-coordinate of the base point in space (1D or 2D).
        y0 : float, optional
            Fixed y-coordinate of the base point in space (2D only).
    
        Notes
        -----
        - In 1D: Displays |p(x, ξ)| over the (x, ξ) phase plane near the fixed point.
        - In 2D: Fixes (x₀, y₀) and evaluates p(x₀, y₀, ξ, η), showing the fiber over that point.
        - The color map represents the magnitude of the symbol, highlighting regions where it vanishes or becomes singular.
    
        Raises
        ------
        NotImplementedError
            If called in 2D with missing or improperly formatted grids.
        """
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI)
            plt.contourf(X, XI, np.abs(symbol_vals), levels=50, cmap='viridis')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('x (position)')
            plt.ylabel('ξ (frequency)')
            plt.title('Cotangent Fiber Structure')
            plt.show()
        elif self.dim == 2:
            xi_grid2, eta_grid2 = np.meshgrid(xi_grid, xi_grid)
            symbol_vals = self.p_func(x0, y0, xi_grid2, eta_grid2)
            plt.contourf(xi_grid, xi_grid, np.abs(symbol_vals), levels=50, cmap='viridis')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('ξ')
            plt.ylabel('η')
            plt.title(f'Cotangent Fiber at x={x0}, y={y0}')
            plt.show()

    def visualize_symbol_amplitude(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """
        Display the modulus |p(x, ξ)| or |p(x, y, ξ₀, η₀)| as a color map.
    
        This method visualizes the amplitude of the pseudodifferential operator's symbol 
        in either 1D or 2D spatial configuration. In 2D, the frequency variables are fixed 
        to specified values (ξ₀, η₀) for visualization purposes.
    
        Parameters
        ----------
        x_grid, y_grid : ndarray
            Spatial grids over which to evaluate the symbol. y_grid is optional and used only in 2D.
        xi_grid, eta_grid : ndarray
            Frequency grids. In 2D, these define the domain over which the symbol is evaluated,
            but the visualization fixes ξ = ξ₀ and η = η₀.
        xi0, eta0 : float, optional
            Fixed frequency values for slicing in 2D visualization. Defaults to zero.
    
        Notes
        -----
        - In 1D: Visualizes |p(x, ξ)| over the (x, ξ) grid.
        - In 2D: Visualizes |p(x, y, ξ₀, η₀)| at fixed frequencies ξ₀ and η₀.
        - The color intensity represents the magnitude of the symbol, highlighting regions where the symbol is large or small.
        """
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI) 
            plt.pcolormesh(X, XI, np.abs(symbol_vals), shading='auto')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Symbol Amplitude |p(x, ξ)|')
            plt.show()
        elif self.dim == 2:
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            XI = np.full_like(X, xi0)
            ETA = np.full_like(Y, eta0)
            symbol_vals = self.p_func(X, Y, XI, ETA)
            plt.pcolormesh(X, Y, np.abs(symbol_vals), shading='auto')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Symbol Amplitude at ξ={xi0}, η={eta0}')
            plt.show()

    def visualize_phase(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """
        Plot the phase (argument) of the pseudodifferential operator's symbol p(x, ξ) or p(x, y, ξ, η).

        This visualization helps in understanding the oscillatory behavior and regularity 
        properties of the operator in phase space. The phase is displayed modulo 2π using 
        a cyclic colormap ('twilight') to emphasize its periodic nature.

        Parameters
        ----------
        x_grid : ndarray
            1D array of spatial coordinates (x).
        xi_grid : ndarray
            1D array of frequency coordinates (ξ).
        y_grid : ndarray, optional
            2D spatial grid for y-coordinate (in 2D problems). Default is None.
        eta_grid : ndarray, optional
            2D frequency grid for η (in 2D problems). Not used directly but kept for API consistency.
        xi0 : float, optional
            Fixed value of ξ for slicing in 2D visualization. Default is 0.0.
        eta0 : float, optional
            Fixed value of η for slicing in 2D visualization. Default is 0.0.

        Notes:
        - In 1D: Displays arg(p(x, ξ)) over the (x, ξ) phase plane.
        - In 2D: Displays arg(p(x, y, ξ₀, η₀)) for fixed frequency values (ξ₀, η₀).
        - Uses plt.pcolormesh with 'twilight' colormap to represent angles from -π to π.

        Raises:
        - NotImplementedError: If the spatial dimension is not 1D or 2D.
        """
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI) 
            plt.pcolormesh(X, XI, np.angle(symbol_vals), shading='auto', cmap='twilight')
            plt.colorbar(label='arg(Symbol) [rad]')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Phase Portrait (arg p(x, ξ))')
            plt.show()
        elif self.dim == 2:
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            XI = np.full_like(X, xi0)
            ETA = np.full_like(Y, eta0)
            symbol_vals = self.p_func(X, Y, XI, ETA)
            plt.pcolormesh(X, Y, np.angle(symbol_vals), shading='auto', cmap='twilight')
            plt.colorbar(label='arg(Symbol) [rad]')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Phase Portrait at ξ={xi0}, η={eta0}')
            plt.show()
            
    def visualize_characteristic_set(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0, levels=[1e-1]):
        """
        Visualize the characteristic set of the pseudo-differential symbol, defined as the approximate zero set p(x, ξ) ≈ 0.
    
        In microlocal analysis, the characteristic set is the locus of points in phase space (x, ξ) where the symbol p(x, ξ) vanishes,
        playing a key role in understanding propagation of singularities.
    
        Parameters
        ----------
        x_grid : ndarray
            Spatial grid values (1D array) for plotting in 1D or evaluation point in 2D.
        xi_grid : ndarray
            Frequency variable grid values (1D array) used to construct the frequency domain.
        x0 : float, optional
            Fixed spatial coordinate in 2D case for evaluating the symbol at a specific x position.
        y0 : float, optional
            Fixed spatial coordinate in 2D case for evaluating the symbol at a specific y position.
    
        Notes
        -----
        - For 1D, this method plots the contour of |p(x, ξ)| = ε with ε = 1e-5 over the (x, ξ) plane.
        - For 2D, it evaluates the symbol at fixed (x₀, y₀) and plots the characteristic set in the (ξ, η) frequency plane.
        - This visualization helps identify directions of degeneracy or hypoellipticity of the operator.
    
        Raises
        ------
        NotImplementedError
            If called on a solver with dimensionality other than 1D or 2D.
    
        Displays
        ------
        A matplotlib contour plot showing either:
            - The characteristic curve in the (x, ξ) phase plane (1D),
            - The characteristic surface slice in the (ξ, η) frequency plane at (x₀, y₀) (2D).
        """
        if self.dim == 1:
            x_grid = np.asarray(x_grid)
            xi_grid = np.asarray(xi_grid)
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI) 
            plt.contour(X, XI, np.abs(symbol_vals), levels=levels, colors='red')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Characteristic Set (p(x, ξ) ≈ 0)')
            plt.grid(True)
            plt.show()
        elif self.dim == 2:
            if eta_grid is None:
                raise ValueError("eta_grid must be provided for 2D visualization.")
            xi_grid = np.asarray(xi_grid)
            eta_grid = np.asarray(eta_grid)
            xi_grid2, eta_grid2 = np.meshgrid(xi_grid, eta_grid, indexing='ij')
            symbol_vals = self.p_func(x0, y0, xi_grid2, eta_grid2)
            plt.contour(xi_grid, eta_grid, np.abs(symbol_vals), levels=levels, colors='red')
            plt.xlabel('ξ')
            plt.ylabel('η')
            plt.title(f'Characteristic Set at x={x0}, y={y0}')
            plt.grid(True)
            plt.show()
        else:
            raise NotImplementedError("Only 1D/2D characteristic sets supported.")

    def visualize_characteristic_gradient(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0):
        """
        Visualize the norm of the gradient of the symbol in phase space.
        
        This method computes the magnitude of the gradient |∇p| of a pseudo-differential 
        symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D. The resulting colormap reveals 
        regions where the symbol varies rapidly or remains nearly stationary, 
        which is particularly useful for analyzing characteristic sets.
        
        Parameters
        ----------
        x_grid : numpy.ndarray
            1D array of spatial coordinates for the x-direction.
        xi_grid : numpy.ndarray
            1D array of frequency coordinates (ξ).
        y_grid : numpy.ndarray, optional
            1D array of spatial coordinates for the y-direction (used in 2D mode). Default is None.
        eta_grid : numpy.ndarray, optional
            1D array of frequency coordinates (η) for the 2D case. Default is None.
        x0 : float, optional
            Fixed x-coordinate for evaluating the symbol in 2D. Default is 0.0.
        y0 : float, optional
            Fixed y-coordinate for evaluating the symbol in 2D. Default is 0.0.
        
        Returns
        -------
        None
            Displays a 2D colormap of |∇p| over the relevant phase-space domain.
        
        Notes
        -----
        - In 1D, the full gradient ∇p = (∂ₓp, ∂ξp) is computed over the (x, ξ) grid.
        - In 2D, the gradient ∇p = (∂ξp, ∂ηp) is computed at a fixed spatial point (x₀, y₀) over the (ξ, η) grid.
        - Numerical differentiation is performed using `np.gradient`.
        - High values of |∇p| indicate rapid variation of the symbol, while low values typically suggest characteristic regions.
        """
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI)
            grad_x = np.gradient(symbol_vals, axis=0)
            grad_xi = np.gradient(symbol_vals, axis=1)
            grad_norm = np.sqrt(grad_x**2 + grad_xi**2)
            plt.pcolormesh(X, XI, grad_norm, cmap='inferno', shading='auto')
            plt.colorbar(label='|∇p|')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Gradient Norm (High Near Zeros)')
            plt.grid(True)
            plt.show()
        elif self.dim == 2:
            xi_grid2, eta_grid2 = np.meshgrid(xi_grid, eta_grid, indexing='ij')
            symbol_vals = self.p_func(x0, y0, xi_grid2, eta_grid2)
            grad_xi = np.gradient(symbol_vals, axis=0)
            grad_eta = np.gradient(symbol_vals, axis=1)
            grad_norm = np.sqrt(np.abs(grad_xi)**2 + np.abs(grad_eta)**2)
            plt.pcolormesh(xi_grid, eta_grid, grad_norm, cmap='inferno', shading='auto')
            plt.colorbar(label='|∇p|')
            plt.xlabel('ξ')
            plt.ylabel('η')
            plt.title(f'Gradient Norm at x={x0}, y={y0}')
            plt.grid(True)
            plt.show()

    def plot_hamiltonian_flow(self, x0=0.0, xi0=5.0, y0=0.0, eta0=0.0, tmax=1.0, n_steps=100, show_field=True):
        """
        Integrate and plot the Hamiltonian trajectories of the symbol in phase space.

        This method numerically integrates the Hamiltonian vector field derived from 
        the operator's symbol to visualize how singularities propagate under the flow. 
        It supports both 1D and 2D problems.

        Parameters
        ----------
        x0, xi0 : float
            Initial position and frequency (momentum) in 1D.
        y0, eta0 : float, optional
            Initial position and frequency in 2D; defaults to zero.
        tmax : float
            Final integration time for the ODE solver.
        n_steps : int
            Number of time steps used in the integration.

        Notes
        -----
        - The Hamiltonian vector field is obtained from the symplectic flow of the symbol.
        - If the field is complex-valued, only its real part is used for integration.
        - In 1D, the trajectory is plotted in (x, ξ) phase space.
        - In 2D, the spatial trajectory (x(t), y(t)) is shown along with instantaneous 
          momentum vectors (ξ(t), η(t)) using a quiver plot.

        Raises
        ------
        NotImplementedError
            If the spatial dimension is not 1D or 2D.

        Displays
        --------
        matplotlib plot
            Phase space trajectory(ies) showing the evolution of position and momentum 
            under the Hamiltonian dynamics.
        """
        def make_real(expr):
            from sympy import re, simplify
            expr = expr.doit(deep=True)
            return simplify(re(expr))
    
        H = self.symplectic_flow()
    
        if any(im(H[k]) != 0 for k in H):
            print("⚠️ The Hamiltonian field is complex. Only the real part is used for integration.")
    
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
    
            dxdt_expr = make_real(H['dx/dt'])
            dxidt_expr = make_real(H['dxi/dt'])
    
            dxdt = lambdify((x, xi), dxdt_expr, 'numpy')
            dxidt = lambdify((x, xi), dxidt_expr, 'numpy')
    
            def hamilton(t, Y):
                x, xi = Y
                return [dxdt(x, xi), dxidt(x, xi)]
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, xi0], t_eval=np.linspace(0, tmax, n_steps))

            if sol.status != 0:
                print(f"⚠️ Integration warning: {sol.message}")
            
            n_points = sol.y.shape[1]
            if n_points < n_steps:
                print(f"⚠️ Only {n_points} frames computed. Adjusting animation.")
                n_steps = n_points

            x_vals, xi_vals = sol.y
    
            plt.plot(x_vals, xi_vals)
            plt.xlabel("x")
            plt.ylabel("ξ")
            plt.title("Hamiltonian Flow in Phase Space (1D)")
            plt.grid(True)
            plt.show()
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
    
            dxdt = lambdify((x, y, xi, eta), make_real(H['dx/dt']), 'numpy')
            dydt = lambdify((x, y, xi, eta), make_real(H['dy/dt']), 'numpy')
            dxidt = lambdify((x, y, xi, eta), make_real(H['dxi/dt']), 'numpy')
            detadt = lambdify((x, y, xi, eta), make_real(H['deta/dt']), 'numpy')
    
            def hamilton(t, Y):
                x, y, xi, eta = Y
                return [
                    dxdt(x, y, xi, eta),
                    dydt(x, y, xi, eta),
                    dxidt(x, y, xi, eta),
                    detadt(x, y, xi, eta)
                ]
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, y0, xi0, eta0], t_eval=np.linspace(0, tmax, n_steps))

            if sol.status != 0:
                print(f"⚠️ Integration warning: {sol.message}")
            
            n_points = sol.y.shape[1]
            if n_points < n_steps:
                print(f"⚠️ Only {n_points} frames computed. Adjusting animation.")
                n_steps = n_points

            x_vals, y_vals, xi_vals, eta_vals = sol.y
    
            plt.plot(x_vals, y_vals, label='Position')
            plt.quiver(x_vals, y_vals, xi_vals, eta_vals, scale=20, width=0.003, alpha=0.5, color='r')
            
            # Vector field of the flow (optional)
            if show_field:
                X, Y = np.meshgrid(np.linspace(min(x_vals), max(x_vals), 20),
                                   np.linspace(min(y_vals), max(y_vals), 20))
                XI, ETA = xi0 * np.ones_like(X), eta0 * np.ones_like(Y)
                U = dxdt(X, Y, XI, ETA)
                V = dydt(X, Y, XI, ETA)
                plt.quiver(X, Y, U, V, color='gray', alpha=0.2, scale=30, width=0.002)

            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Hamiltonian Flow in Phase Space (2D)")
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()

    def plot_symplectic_vector_field(self, xlim=(-2, 2), klim=(-5, 5), density=30):
        """
        Visualize the symplectic vector field (Hamiltonian vector field) associated with the operator's symbol.

        The plotted vector field corresponds to (∂_ξ p, -∂_x p), where p(x, ξ) is the principal symbol 
        of the pseudo-differential operator. This field governs the bicharacteristic flow in phase space.

        Parameters
        ----------
        xlim : tuple of float
            Range for spatial variable x, as (x_min, x_max).
        klim : tuple of float
            Range for frequency variable ξ, as (ξ_min, ξ_max).
        density : int
            Number of grid points per axis for the visualization grid.

        Raises
        ------
        NotImplementedError
            If called on a 2D operator (currently only 1D implementation available).

        Notes
        -----
        - Only supports one-dimensional operators.
        - Uses symbolic differentiation to compute ∂_ξ p and ∂_x p.
        - Numerical evaluation is done via lambdify with NumPy backend.
        - Visualization uses matplotlib quiver plot to show vector directions.
        """
        x_vals = np.linspace(*xlim, density)
        xi_vals = np.linspace(*klim, density)
        X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')

        if self.dim != 1:
            raise NotImplementedError("Only 1D version implemented.")

        x, = self.vars_x
        xi = symbols('xi', real=True)
        H = self.symplectic_flow()
        dxdt = lambdify((x, xi), simplify(H['dx/dt']), 'numpy')
        dxidt = lambdify((x, xi), simplify(H['dxi/dt']), 'numpy')

        U = dxdt(X, XI)
        V = dxidt(X, XI)

        plt.quiver(X, XI, U, V, scale=10, width=0.005)
        plt.xlabel('x')
        plt.ylabel(r'$\xi$')
        plt.title("Symplectic Vector Field (1D)")
        plt.grid(True)
        plt.show()

    def visualize_micro_support(self, xlim=(-2, 2), klim=(-10, 10), threshold=1e-3, density=300):
        """
        Visualize the micro-support of the operator by plotting the inverse of the symbol magnitude 1 / |p(x, ξ)|.
    
        The micro-support provides insight into the singularities of a pseudo-differential operator 
        in phase space (x, ξ). Regions where |p(x, ξ)| is small correspond to large values in 1/|p(x, ξ)|,
        highlighting areas of significant operator influence or singularity.
    
        Parameters
        ----------
        xlim : tuple
            Spatial domain limits (x_min, x_max).
        klim : tuple
            Frequency domain limits (ξ_min, ξ_max).
        threshold : float
            Threshold below which |p(x, ξ)| is considered effectively zero; used for numerical stability.
        density : int
            Number of grid points along each axis for visualization resolution.
    
        Raises
        ------
        NotImplementedError
            If called on a solver with dimension greater than 1 (only 1D visualization is supported).
    
        Notes
        -----
        - This method evaluates the symbol p(x, ξ) over a grid and plots its reciprocal to emphasize 
          regions where the symbol is near zero.
        - A small constant (1e-10) is added to the denominator to avoid division by zero.
        - The resulting plot helps identify characteristic sets.
        """
        if self.dim != 1:
            raise NotImplementedError("Only 1D micro-support visualization implemented.")

        x_vals = np.linspace(*xlim, density)
        xi_vals = np.linspace(*klim, density)
        X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
        Z = np.abs(self.p_func(X, XI))

        plt.contourf(X, XI, 1 / (Z + 1e-10), levels=100, cmap='inferno')
        plt.colorbar(label=r'$1/|p(x,\xi)|$')
        plt.xlabel('x')
        plt.ylabel(r'$\xi$')
        plt.title("Micro-Support Estimate (1/|Symbol|)")
        plt.show()

    def group_velocity_field(self, xlim=(-2, 2), klim=(-10, 10), density=30):
        """
        Plot the group velocity field ∇_ξ p(x, ξ) for 1D pseudo-differential operators.

        The group velocity represents the speed at which waves of different frequencies propagate 
        in a dispersive medium. It is defined as the gradient of the symbol p(x, ξ) with respect 
        to the frequency variable ξ.

        Parameters
        ----------
        xlim : tuple of float
            Spatial domain limits (x-axis).
        klim : tuple of float
            Frequency domain limits (ξ-axis).
        density : int
            Number of grid points per axis used for visualization.

        Raises
        ------
        NotImplementedError
            If called on a 2D operator, since this visualization is only implemented for 1D.

        Notes
        -----
        - This method visualizes the vector field (∂p/∂ξ) in phase space.
        - Used for analyzing wave propagation properties and dispersion relations.
        - Requires symbolic expression self.expr depending on x and ξ.
        """
        if self.dim != 1:
            raise NotImplementedError("Only 1D group velocity visualization implemented.")

        x, = self.vars_x
        xi = symbols('xi', real=True)
        dp_dxi = diff(self.symbol, xi)
        grad_func = lambdify((x, xi), dp_dxi, 'numpy')

        x_vals = np.linspace(*xlim, density)
        xi_vals = np.linspace(*klim, density)
        X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
        V = grad_func(X, XI)

        plt.quiver(X, XI, np.ones_like(V), V, scale=10, width=0.004)
        plt.xlabel('x')
        plt.ylabel(r'$\xi$')
        plt.title("Group Velocity Field (1D)")
        plt.grid(True)
        plt.show()

    def animate_singularity(self, xi0=5.0, eta0=0.0, x0=0.0, y0=0.0,
                            tmax=4.0, n_frames=100, projection=None):
        """
        Animate the propagation of a singularity under the Hamiltonian flow.

        This method visualizes how a singularity (x₀, y₀, ξ₀, η₀) evolves in phase space 
        according to the Hamiltonian dynamics induced by the principal symbol of the operator.
        The animation integrates the Hamiltonian equations of motion and supports various projections:
        position (x-y), frequency (ξ-η), or mixed phase space coordinates.

        Parameters
        ----------
        xi0, eta0 : float
            Initial frequency components (ξ₀, η₀).
        x0, y0 : float
            Initial spatial coordinates (x₀, y₀).
        tmax : float
            Total time of integration (final animation time).
        n_frames : int
            Number of frames in the resulting animation.
        projection : str or None
            Type of projection to display:
                - 'position' : x vs y (or x alone in 1D)
                - 'frequency': ξ vs η (or ξ alone in 1D)
                - 'phase'    : mixed coordinates like x vs ξ or x vs η
                If None, defaults to 'phase' in 1D and 'position' in 2D.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object that can be displayed interactively in Jupyter notebooks or saved as a video.

        Notes
        -----
        - In 1D, only one spatial and one frequency variable are used.
        - Complex-valued Hamiltonian fields are truncated to their real parts for integration.
        - Trajectories are shown with both instantaneous position (dot) and full path (dashed line).
        """
        rc('animation', html='jshtml')
    
        def make_real(expr):
            from sympy import re, simplify
            expr = expr.doit(deep=True)
            return simplify(re(expr))
  
        H = self.symplectic_flow()

        H = {k: v.doit(deep=True) for k, v in H.items()}

        print("H = ", H)
    
        if any(im(H[k]) != 0 for k in H):
            print("⚠️ The Hamiltonian field is complex. Only the real part is used for integration.")
    
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
    
            dxdt = lambdify((x, xi), make_real(H['dx/dt']), 'numpy')
            dxidt = lambdify((x, xi), make_real(H['dxi/dt']), 'numpy')
    
            def hamilton(t, Y):
                x, xi = Y
                return [dxdt(x, xi), dxidt(x, xi)]
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, xi0],
                            t_eval=np.linspace(0, tmax, n_frames))
            
            if sol.status != 0:
                print(f"⚠️ Integration warning: {sol.message}")
            
            n_points = sol.y.shape[1]
            if n_points < n_frames:
                print(f"⚠️ Only {n_points} frames computed. Adjusting animation.")
                n_frames = n_points

            x_vals, xi_vals = sol.y
    
            if projection is None:
                projection = 'phase'
    
            fig, ax = plt.subplots()
            point, = ax.plot([], [], 'ro')
            traj, = ax.plot([], [], 'b--', lw=1, alpha=0.5)
    
            if projection == 'phase':
                ax.set_xlabel('x')
                ax.set_ylabel(r'$\xi$')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [xi_vals[i]])
                    traj.set_data(x_vals[:i+1], xi_vals[:i+1])
                    return point, traj
    
            elif projection == 'position':
                ax.set_xlabel('x')
                ax.set_ylabel('x')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(x_vals) - 1, np.max(x_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [x_vals[i]])
                    traj.set_data(x_vals[:i+1], x_vals[:i+1])
                    return point, traj
    
            elif projection == 'frequency':
                ax.set_xlabel(r'$\xi$')
                ax.set_ylabel(r'$\xi$')
                ax.set_xlim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
                ax.set_ylim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
    
                def update(i):
                    point.set_data([xi_vals[i]], [xi_vals[i]])
                    traj.set_data(xi_vals[:i+1], xi_vals[:i+1])
                    return point, traj
    
            else:
                raise ValueError("Invalid projection mode")
    
            ax.set_title(f"1D Singularity Flow ({projection})")
            ax.grid(True)
            ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
            plt.close(fig)
            return ani
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
    
            dxdt = lambdify((x, y, xi, eta), make_real(H['dx/dt']), 'numpy')
            dydt = lambdify((x, y, xi, eta), make_real(H['dy/dt']), 'numpy')
            dxidt = lambdify((x, y, xi, eta), make_real(H['dxi/dt']), 'numpy')
            detadt = lambdify((x, y, xi, eta), make_real(H['deta/dt']), 'numpy')
    
            def hamilton(t, Y):
                x, y, xi, eta = Y
                return [
                    dxdt(x, y, xi, eta),
                    dydt(x, y, xi, eta),
                    dxidt(x, y, xi, eta),
                    detadt(x, y, xi, eta)
                ]
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, y0, xi0, eta0],
                            t_eval=np.linspace(0, tmax, n_frames))

            if sol.status != 0:
                print(f"⚠️ Integration warning: {sol.message}")
            
            n_points = sol.y.shape[1]
            if n_points < n_frames:
                print(f"⚠️ Only {n_points} frames computed. Adjusting animation.")
                n_frames = n_points
                
            x_vals, y_vals, xi_vals, eta_vals = sol.y
    
            if projection is None:
                projection = 'position'
    
            fig, ax = plt.subplots()
            point, = ax.plot([], [], 'ro')
            traj, = ax.plot([], [], 'b--', lw=1, alpha=0.5)
    
            if projection == 'position':
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(y_vals) - 1, np.max(y_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [y_vals[i]])
                    traj.set_data(x_vals[:i+1], y_vals[:i+1])
                    return point, traj
    
            elif projection == 'frequency':
                ax.set_xlabel(r'$\xi$')
                ax.set_ylabel(r'$\eta$')
                ax.set_xlim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
                ax.set_ylim(np.min(eta_vals) - 1, np.max(eta_vals) + 1)
    
                def update(i):
                    point.set_data([xi_vals[i]], [eta_vals[i]])
                    traj.set_data(xi_vals[:i+1], eta_vals[:i+1])
                    return point, traj
    
            elif projection == 'phase':
                ax.set_xlabel('x')
                ax.set_ylabel(r'$\eta$')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(eta_vals) - 1, np.max(eta_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [eta_vals[i]])
                    traj.set_data(x_vals[:i+1], eta_vals[:i+1])
                    return point, traj
    
            else:
                raise ValueError("Invalid projection mode")
    
            ax.set_title(f"2D Singularity Flow ({projection})")
            ax.grid(True)
            ax.axis('equal')
            ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
            plt.close(fig)
            return ani

    def interactive_symbol_analysis(pseudo_op,
                                    xlim=(-2, 2), ylim=(-2, 2),
                                    xi_range=(0.1, 5), eta_range=(-5, 5),
                                    density=50):
        """
        Launch an interactive dashboard for symbol exploration using ipywidgets.
    
        This function provides a user-friendly interface to visualize various aspects of the pseudo-differential operator's symbol.
        It supports multiple visualization modes in both 1D and 2D, including group velocity fields, micro-support estimates,
        symplectic vector fields, symbol amplitude/phase, cotangent fiber structure, characteristic sets and Hamiltonian flows.
    
        Parameters
        ----------
        pseudo_op : PseudoDifferentialOperator
            The pseudo-differential operator whose symbol is to be analyzed interactively.
        xlim, ylim : tuple of float
            Spatial domain limits along x and y axes respectively.
        xi_range, eta_range : tuple
            Frequency domain limits along ξ and η axes respectively.
        density : int
            Number of points per axis used to construct the evaluation grid. Controls resolution.
    
        Notes
        -----
        - In 1D mode, sliders control the fixed frequency (ξ₀) and spatial position (x₀).
        - In 2D mode, additional sliders control the second frequency component (η₀) and second spatial coordinate (y₀).
        - Visualization updates dynamically as parameters are adjusted via sliders or dropdown menus.
        - Supported visualization modes:
            - 'Symbol Amplitude'           : |p(x,ξ)| or |p(x,y,ξ,η)|
            - 'Symbol Phase'               : arg(p(x,ξ)) or similar in 2D
            - 'Micro-Support (1/|p|)'      : Reciprocal of symbol magnitude
            - 'Cotangent Fiber'            : Structure of symbol over frequency space at fixed x
            - 'Characteristic Set'         : Zero set approximation {p ≈ 0}
            - 'Characteristic Gradient'    : |∇p(x, ξ)| or |∇p(x₀, y₀, ξ, η)|
            - 'Group Velocity Field'       : ∇_ξ p(x,ξ) or ∇_{ξ,η} p(x,y,ξ,η)
            - 'Symplectic Vector Field'    : (∇_ξ p, -∇_x p) or similar in 2D
            - 'Hamiltonian Flow'           : Trajectories generated by the Hamiltonian vector field
    
        Raises
        ------
        NotImplementedError
            If the spatial dimension is not 1D or 2D.
    
        Prints
        ------
        Interactive matplotlib figures with dynamic updates based on widget inputs.
        """
        dim = pseudo_op.dim
        expr = pseudo_op.expr
        vars_x = pseudo_op.vars_x
    
        mode_selector_1D = Dropdown(
            options=[
                'Symbol Amplitude',
                'Symbol Phase',
                'Micro-Support (1/|p|)',
                'Cotangent Fiber',
                'Characteristic Set',
                'Characteristic Gradient',
                'Group Velocity Field',
                'Symplectic Vector Field',
                'Hamiltonian Flow',
            ],
            value='Symbol Amplitude',
            description='Mode:'
        )

        mode_selector_2D = Dropdown(
            options=[
                'Symbol Amplitude',
                'Symbol Phase',
                'Micro-Support (1/|p|)',
                'Cotangent Fiber',
                'Characteristic Set',
                'Characteristic Gradient',
                'Symplectic Vector Field',
                'Hamiltonian Flow',
            ],
            value='Symbol Amplitude',
            description='Mode:'
        )
    
        x_vals = np.linspace(*xlim, density)
        if dim == 2:
            y_vals = np.linspace(*ylim, density)
    
        if dim == 1:
            x, = vars_x
            xi = symbols('xi', real=True)
            grad_func = lambdify((x, xi), diff(expr, xi), 'numpy')
            symplectic_func = lambdify((x, xi), [diff(expr, xi), -diff(expr, x)], 'numpy')
            symbol_func = lambdify((x, xi), expr, 'numpy')

            xi_slider = FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='ξ₀')
            x_slider = FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x₀')
    
            def plot_1d(mode, xi0, x0):

                plt.close('all')
                plt.figure()
                X = x_vals[:, None]
    
                if mode == 'Group Velocity Field':
                    V = grad_func(X, xi0)
                    plt.quiver(X, V, np.ones_like(V), V, scale=10, width=0.004)
                    plt.xlabel('x')
                    plt.title(f'Group Velocity Field at ξ={xi0:.2f}')
    
                elif mode == 'Micro-Support (1/|p|)':
                    Z = 1 / (np.abs(symbol_func(X, xi0)) + 1e-10)
                    plt.plot(x_vals, Z)
                    plt.xlabel('x')
                    plt.title(f'Micro-Support (1/|p|) at ξ={xi0:.2f}')
    
                elif mode == 'Symplectic Vector Field':
                    U, V = symplectic_func(X, xi0)
                    plt.quiver(X, V, U, V, scale=10, width=0.004)
                    plt.xlabel('x')
                    plt.title(f'Symplectic Field at ξ={xi0:.2f}')
    
                elif mode == 'Symbol Amplitude':
                    Z = np.abs(symbol_func(X, xi0))
                    plt.plot(x_vals, Z)
                    plt.xlabel('x')
                    plt.title(f'Symbol Amplitude |p(x,ξ)| at ξ={xi0:.2f}')
    
                elif mode == 'Symbol Phase':
                    Z = np.angle(symbol_func(X, xi0))
                    plt.plot(x_vals, Z)
                    plt.xlabel('x')
                    plt.title(f'Symbol Phase arg(p(x,ξ)) at ξ={xi0:.2f}')
    
                elif mode == 'Cotangent Fiber':
                    pseudo_op.visualize_fiber(x_vals, np.linspace(*xi_range, density), x0=x0)
    
                elif mode == 'Characteristic Set':
                    pseudo_op.visualize_characteristic_set(x_vals, np.linspace(*xi_range, density), x0=x0)
    
                elif mode == 'Characteristic Gradient':
                    pseudo_op.visualize_characteristic_gradient(x_vals, np.linspace(*xi_range, density), x0=x0)
    
                elif mode == 'Hamiltonian Flow':
                    pseudo_op.plot_hamiltonian_flow(x0=x0, xi0=xi0)
                    
                if mode not in (
                    "Cotangent Fiber",
                    "Characteristic Set",
                    "Characteristic Gradient",
                    "Hamiltonian Flow",
                ):
                    plt.show()

    
            # --- Dynamic container for sliders ---
            controls_box = VBox([mode_selector_1D, xi_slider, x_slider])
            # --- Function to adjust visible sliders based on mode ---
            def update_controls(change):
                mode = change['new']
                # modes that depend only on xi and eta
                if mode in ['Symbol Amplitude', 'Symbol Phase', 'Micro-Support (1/|p|)',
                            'Group Velocity Field', 'Symplectic Vector Field']:
                    controls_box.children = [mode_selector_1D, xi_slider]
                # modes that require xi and x
                elif mode in ['Hamiltonian Flow']:
                    controls_box.children = [mode_selector_1D, xi_slider, x_slider]
                # modes that require nothing
                elif mode in ['Cotangent Fiber', 'Characteristic Set', 'Characteristic Gradient']:
                    controls_box.children = [mode_selector_1D]
            mode_selector_1D.observe(update_controls, names='value')
            update_controls({'new': mode_selector_1D.value}) 
            # --- Interactive binding ---
            out = interactive_output(plot_1d, {'mode': mode_selector_1D, 'xi0': xi_slider, 'x0': x_slider})
            display(VBox([controls_box, out]))

        elif dim == 2:
            x, y = vars_x
            xi, eta = symbols('xi eta', real=True)
            symplectic_func = lambdify((x, y, xi, eta), [diff(expr, xi), diff(expr, eta)], 'numpy')
            symbol_func = lambdify((x, y, xi, eta), expr, 'numpy')

            xi_slider=FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='ξ₀')
            eta_slider=FloatSlider(min=eta_range[0], max=eta_range[1], step=0.1, value=1.0, description='η₀')
            x_slider=FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x₀')
            y_slider=FloatSlider(min=ylim[0], max=ylim[1], step=0.1, value=0.0, description='y₀')
    
            def plot_2d(mode, xi0, eta0, x0, y0):

                plt.close('all')
                plt.figure()
                X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
                if mode == 'Micro-Support (1/|p|)':
                    Z = 1 / (np.abs(symbol_func(X, Y, xi0, eta0)) + 1e-10)
                    plt.pcolormesh(X, Y, Z, shading='auto', cmap='inferno')
                    plt.colorbar(label='1/|p|')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'Micro-Support at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Symplectic Vector Field':
                    U, V = symplectic_func(X, Y, xi0, eta0)
                    plt.quiver(X, Y, U, V, scale=10, width=0.004)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'Symplectic Field at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Symbol Amplitude':
                    Z = np.abs(symbol_func(X, Y, xi0, eta0))
                    plt.pcolormesh(X, Y, Z, shading='auto')
                    plt.colorbar(label='|p(x,y,ξ,η)|')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'Symbol Amplitude at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Symbol Phase':
                    Z = np.angle(symbol_func(X, Y, xi0, eta0))
                    plt.pcolormesh(X, Y, Z, shading='auto', cmap='twilight')
                    plt.colorbar(label='arg(p)')
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.title(f'Symbol Phase at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Cotangent Fiber':
                    pseudo_op.visualize_fiber(np.linspace(*xi_range, density), np.linspace(*eta_range, density),
                                              x0=x0, y0=y0)
    
                elif mode == 'Characteristic Set':
                    pseudo_op.visualize_characteristic_set(x_grid=x_vals, xi_grid=np.linspace(*xi_range, density),
                                                  y_grid=y_vals, eta_grid=np.linspace(*eta_range, density), x0=x0, y0=y0)
    
                elif mode == 'Characteristic Gradient':
                    pseudo_op.visualize_characteristic_gradient(x_grid=x_vals, xi_grid=np.linspace(*xi_range, density),
                                                  y_grid=y_vals, eta_grid=np.linspace(*eta_range, density), x0=x0, y0=y0)
    
                elif mode == 'Hamiltonian Flow':
                    pseudo_op.plot_hamiltonian_flow(x0=x0, y0=y0, xi0=xi0, eta0=eta0)

                if mode not in (
                    "Cotangent Fiber",
                    "Characteristic Set",
                    "Characteristic Gradient",
                    "Hamiltonian Flow",
                ):
                    plt.show()
                    
            # --- Dynamic container for sliders ---
            controls_box = VBox([mode_selector_2D, xi_slider, eta_slider, x_slider, y_slider])
            # --- Function to adjust visible sliders based on mode ---
            def update_controls(change):
                mode = change['new']
                # modes that depend only on xi
                if mode in ['Symbol Amplitude', 'Symbol Phase', 'Micro-Support (1/|p|)', 'Symplectic Vector Field']:
                    controls_box.children = [mode_selector_2D, xi_slider, eta_slider]
                # modes that require xi, eta, x and y
                elif mode in ['Hamiltonian Flow']:
                    controls_box.children = [mode_selector_2D, xi_slider, eta_slider, x_slider, y_slider]
                # modes that require x and y
                elif mode in ['Cotangent Fiber', 'Characteristic Set', 'Characteristic Gradient']:
                    controls_box.children = [mode_selector_2D, x_slider, y_slider]
            mode_selector_2D.observe(update_controls, names='value')
            update_controls({'new': mode_selector_2D.value}) 
            # --- Interactive binding ---
            out = interactive_output(plot_2d, {'mode': mode_selector_2D, 'xi0': xi_slider, 'eta0': eta_slider, 'x0': x_slider, 'y0': y_slider})
            display(VBox([controls_box, out]))


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

import itertools
import numpy as np
import sympy as sp


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


# ============================================================================
# Kohn-Nirenberg Subdomain Boundary Integration Pipeline (Revision 2)
# ============================================================================
#
# Diffuse-interface / penalty injection of Dirichlet & Neumann data on an
# implicitly-defined subdomain boundary C = dOmega = {g(x) = 0}, entirely
# within the periodic Fourier framework used by kohn_nirenberg_fft. See
# kn_subdomain_boundary_pipeline_v2.md for the derivation, scope
# restriction and validation protocol. Driven by
# `PseudoDifferentialOperator.apply_subdomain`.


def _fft_freqs(x_grid, y_grid=None, dim=1):
    """Native FFT-ordered angular frequency grid(s) matching x_grid[, y_grid]."""
    Nx = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    if dim == 1:
        return (kx,)
    Ny = len(y_grid)
    dy = y_grid[1] - y_grid[0]
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return (KX, KY)


def _spectral_gradient(field, x_grid, y_grid, fft_func, ifft_func, dim):
    """Componentwise gradient of a periodic field via FFT differentiation."""
    freqs = _fft_freqs(x_grid, y_grid, dim=dim)
    field_hat = fft_func(field)
    return tuple(ifft_func(1j * k * field_hat) for k in freqs)


def _weighted_mean(field, weight):
    """rho_delta-weighted mean of a field, restricted to where weight is non-negligible."""
    denom = np.sum(weight)
    return np.sum(weight * field) / denom if denom > 0 else np.mean(field)


def _secant_update(kappa, signed_residual, prev, bootstrap=0.1):
    """
    One secant (or bootstrap) step for a scalar gain kappa against a
    *signed* scalar residual functional (Step 7, "Simplest form (gain
    calibration)").

    Because the pipeline (Steps 2-6) is linear in `kappa`, any linear
    functional of the trace residual field -- e.g. the `rho_delta`-weighted
    mean returned by `subdomain_trace_residual`'s companion, not the
    (non-negative, non-linear) RMS norm -- is *exactly* affine in `kappa`.
    Driving that signed functional to zero by secant is therefore not just
    "approximately linear near the current estimate": it is the correct
    linear equation for `kappa`, solved in one step once two points are
    available. The RMS norm itself must not be used here, since it has no
    zero to find in general (a single scalar gain generally cannot drive
    the *pointwise* residual to exactly zero) and secant iteration on it
    does not converge.

    Parameters
    ----------
    kappa : float
        Current gain.
    signed_residual : float
        Signed scalar residual functional measured at `kappa` (e.g. a
        weighted mean of the trace residual field).
    prev : tuple or None
        `(kappa_prev, signed_residual_prev)` from the previous iteration,
        or `None` on the first call.
    bootstrap : float, default 0.1
        Relative perturbation used to probe the sensitivity on the first
        call, when no previous point is available for a secant slope.

    Returns
    -------
    kappa_new : float
        Updated gain.
    prev_new : tuple
        `(kappa, signed_residual)`, to pass back in on the next call.
    """
    if prev is None:
        step = bootstrap if abs(kappa) < 1e-12 else bootstrap * kappa
        return kappa + step, (kappa, signed_residual)

    kappa0, residual0 = prev
    dk = kappa - kappa0
    slope = (signed_residual - residual0) / dk if dk != 0 else None
    kappa_new = kappa - signed_residual / slope if slope else kappa
    return kappa_new, (kappa, signed_residual)


def subdomain_masks(g_vals, x_grid, fft_func, ifft_func, y_grid=None, sigma=None, dim=1):
    """
    Pre-computation phase of the subdomain boundary pipeline: build the
    smooth indicator, normal shell and single-layer shell of an implicitly
    defined subdomain `Omega = {g <= 0}`.

    The single-layer shell is derived directly from the normal shell
    (`rho_delta = |n_delta|`) rather than built independently, so both
    scale the same way as `sigma -> 0` (Revision-2 fix: the amplitude-1,
    width-`sigma` bump `4*chi*(1-chi)` used in Revision 1 has an integral
    that vanishes as `sigma -> 0`, unlike `|grad(chi_Omega)|`).

    Parameters
    ----------
    g_vals : ndarray
        Samples of the implicit level-set function `g(x)` [`g(x, y)`] on
        the grid; `Omega = {g <= 0}`, boundary `C = {g = 0}`.
    x_grid, y_grid : ndarray
        Spatial grids (`y_grid` required for `dim=2`).
    fft_func, ifft_func : callable
        Periodic FFT/IFFT matching `dim` (e.g. `op.fft`, `op.ifft`).
    sigma : float, optional
        Gaussian smoothing width. Defaults to `2.5 * dx` -- target 5-10
        grid points resolving the transition collar, coupled to the grid
        exactly as for the rest of the periodization/regularization scheme.
    dim : {1, 2}
        Spatial dimension.

    Returns
    -------
    chi_Omega : ndarray
        Smooth indicator of `Omega` (~1 inside, ~0 outside).
    n_delta : tuple of ndarray
        Normal shell components, `grad(chi_Omega)`; as `sigma -> 0`,
        `integral(phi * |n_delta|) -> integral_C(phi) dS` (coarea formula).
    rho_delta : ndarray
        Single-layer shell `|n_delta|`: a proper, `sigma`-independent-
        integral approximate surface delta on `C`.
    sigma : float
        Smoothing width actually used.
    """
    dx = x_grid[1] - x_grid[0]
    if sigma is None:
        sigma = 2.5 * dx

    freqs = _fft_freqs(x_grid, y_grid, dim=dim)
    k2 = sum(k ** 2 for k in freqs)
    gauss = np.exp(-0.5 * sigma ** 2 * k2)

    indicator = (np.asarray(g_vals) <= 0).astype(np.float64)
    chi_Omega = np.real(ifft_func(fft_func(indicator) * gauss))

    n_delta = tuple(
        np.real(c) for c in _spectral_gradient(chi_Omega, x_grid, y_grid, fft_func, ifft_func, dim)
    )
    rho_delta = np.sqrt(sum(c ** 2 for c in n_delta))

    return chi_Omega, n_delta, rho_delta, sigma


def subdomain_trace_residual(v_Omega, target, rho_delta, grad_v=None, n_ext=None):
    """
    Weighted trace residual (Step 7): how well a field matches prescribed
    boundary data, localized to the boundary collar by the single-layer
    weight `rho_delta`.

    Used for both the Dirichlet residual (`v_Omega` vs. `f`, the default)
    and the Neumann residual (`n_ext . grad_v` vs. `h`, by passing `grad_v`
    and `n_ext`).

    Parameters
    ----------
    v_Omega : ndarray
        Field to evaluate the trace of (`v_Omega` for Dirichlet; ignored
        in favor of `grad_v`/`n_ext` for Neumann).
    target : ndarray
        Prescribed boundary data (`f` or `h`), sampled on the full grid.
    rho_delta : ndarray
        Single-layer shell weight from `subdomain_masks`, localizing the
        comparison to a collar around `C`.
    grad_v, n_ext : tuple of ndarray, optional
        Gradient components of the field and the outward-normal extension
        to contract them with, for the Neumann residual. If either is
        `None`, the Dirichlet residual (`v_Omega - target`) is returned.

    Returns
    -------
    residual_field : ndarray
        Pointwise deviation from `target`; only meaningful where
        `rho_delta` is non-negligible (i.e. near `C`).
    residual_norm : float
        `rho_delta`-weighted RMS residual -- the scalar trust diagnostic
        reported alongside the output.
    """
    if grad_v is not None and n_ext is not None:
        sampled = sum(ne * gv for ne, gv in zip(n_ext, grad_v))
    else:
        sampled = v_Omega

    residual_field = sampled - target
    w = rho_delta
    denom = np.sum(w)
    if denom > 0:
        residual_norm = float(np.sqrt(np.sum(w * np.abs(residual_field) ** 2) / denom))
    else:
        residual_norm = float(np.linalg.norm(residual_field))

    return residual_field, residual_norm


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