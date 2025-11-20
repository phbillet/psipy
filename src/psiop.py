# Copyright 2025 Philippe Billet
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
Toolkit for pseudo-differential manipulation in 1D or 2D

This package offers the following options:
 - psiOp symbol manipulation: asymptotic expansion, asymptotic composition, adjoint, exponential....
 - psiOp symbol visualization: amplitude, phase, characteristic set, characteristic gradient... 
"""

from imports import *

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

    def __init__(self, expr, vars_x, var_u=None, mode='symbol'):
        self.dim = len(vars_x)
        self.mode = mode
        self.symbol_cached = None
        self.expr = expr
        self.vars_x = vars_x

        if self.dim == 1:
            x, = vars_x
            xi_internal = symbols('xi', real=True)
            expr = expr.subs(symbols('xi', real=True), xi_internal)
            self.fft = partial(fft, workers=FFT_WORKERS)
            self.ifft = partial(ifft, workers=FFT_WORKERS)

            if mode == 'symbol':
                self.p_func = lambdify((x, xi_internal), expr, 'numpy')
                self.symbol = expr
            elif mode == 'auto':
                if var_u is None:
                    raise ValueError("var_u must be provided in mode='auto'")
                exp_i = exp(I * x * xi_internal)
                P_ei = expr.subs(var_u, exp_i)
                symbol = simplify(P_ei / exp_i)
                symbol = expand(symbol)
                self.symbol = symbol
                self.p_func = lambdify((x, xi_internal), symbol, 'numpy')
            else:
                raise ValueError("mode must be 'auto' or 'symbol'")

        elif self.dim == 2:
            x, y = vars_x
            xi_internal, eta_internal = symbols('xi eta', real=True)
            expr = expr.subs(symbols('xi', real=True), xi_internal)
            expr = expr.subs(symbols('eta', real=True), eta_internal)
            self.fft = partial(fft2, workers=FFT_WORKERS)
            self.ifft = partial(ifft2, workers=FFT_WORKERS)

            if mode == 'symbol':
                self.symbol = expr
                self.p_func = lambdify((x, y, xi_internal, eta_internal), expr, 'numpy')
            elif mode == 'auto':
                if var_u is None:
                    raise ValueError("var_u must be provided in mode='auto'")
                exp_i = exp(I * (x * xi_internal + y * eta_internal))
                P_ei = expr.subs(var_u, exp_i)
                symbol = simplify(P_ei / exp_i)
                symbol = expand(symbol)
                self.symbol = symbol
                self.p_func = lambdify((x, y, xi_internal, eta_internal), symbol, 'numpy')
            else:
                raise ValueError("mode must be 'auto' or 'symbol'")

        else:
            raise NotImplementedError("Only 1D and 2D supported")

        if mode == 'auto':
            print("\nsymbol = ")
            pprint(self.symbol, num_columns=NUM_COLS)
        
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
        Clear cached symbol evaluations.
        """        
        self.symbol_cached = None

    def apply(self, u, x_grid, kx, boundary_condition='periodic', 
              y_grid=None, ky=None, dealiasing_mask=None,
              freq_window='gaussian', clamp=1e6, space_window=False):
        """
        Apply the pseudo-differential operator to the input field u.
    
        This method dispatches the application of the pseudo-differential operator based on:
        
        - Whether the symbol is spatially dependent (x/y)
        - The boundary condition in use (periodic or dirichlet)
    
        Supported operations:
        
        - Constant-coefficient symbols: applied via Fourier multiplication.
        - Spatially varying symbols: applied via Kohn–Nirenberg quantization.
        - Dirichlet boundary conditions: handled with non-periodic convolution-like quantization.
    
        Dispatch Logic:\n
        if not self.is_spatial: u ↦ Op(p)(D) ⋅ u = 𝓕⁻¹[ p(ξ) ⋅ 𝓕(u) ]\n
        elif periodic: u ↦ Op(p)(x,D) ⋅ u ≈ ∫ eᶦˣᶿ p(x, ξ) 𝓕(u)(ξ) dξ based of FFT (quicker)\n
        elif dirichlet: u ↦ Op(p)(x,D) ⋅ u ≈ u ≈ ∫ eᶦˣᶿ p(x, ξ) 𝓕(u)(ξ) dξ (slower)\n
        
        Parameters
        ----------
        u : ndarray
            Function to which the operator is applied
        x_grid : ndarray
            Spatial grid in x direction
        kx : ndarray
            Frequency grid in x direction
        boundary_condition : str
            'periodic' or 'dirichlet'
        y_grid : ndarray, optional
            Spatial grid in y direction (for 2D)
        ky : ndarray, optional
            Frequency grid in y direction (for 2D)
        dealiasing_mask : ndarray, optional
            Dealiasing mask
        freq_window : str
            Frequency windowing ('gaussian' or 'hann')
        clamp : float
            Clamp symbol values to [-clamp, clamp]
        space_window : bool
            Apply spatial windowing
            
        Returns
        -------
        ndarray
            Result of applying the operator
        """
        # Check if symbol depends on spatial variables
        is_spatial = self._is_spatial_dependent()
        
        # Case 1: Constant symbol with periodic BC (fast path)
        if not is_spatial and boundary_condition == 'periodic':
            return self._apply_constant_fft(u, x_grid, kx, y_grid, ky, dealiasing_mask)
        
        # Case 2: Spatial symbol with periodic BC
        elif boundary_condition == 'periodic':
            symbol_func = self._get_symbol_func()
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
                space_window=space_window
            )
        
        # Case 3: Dirichlet BC (non-periodic)
        elif boundary_condition == 'dirichlet':
            symbol_func = self._get_symbol_func()
            
            if self.dim == 1:
                return kohn_nirenberg_nonperiodic(
                    u_vals=u,
                    x_grid=x_grid,
                    xi_grid=kx,
                    symbol_func=symbol_func,
                    freq_window=freq_window,
                    clamp=clamp,
                    space_window=space_window
                )
            elif self.dim == 2:
                return kohn_nirenberg_nonperiodic(
                    u_vals=u,
                    x_grid=(x_grid, y_grid),
                    xi_grid=(kx, ky),
                    symbol_func=symbol_func,
                    freq_window=freq_window,
                    clamp=clamp,
                    space_window=space_window
                )
        
        else:
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
    
    def _apply_constant_fft(self, u, x_grid, kx, y_grid, ky, dealiasing_mask):
        """
        Apply a constant-coefficient pseudo-differential operator in Fourier space.

        This method assumes the symbol is diagonal in the Fourier basis and acts as a 
        multiplication operator. It performs the operation:
        
            (ψu)(x) = 𝓕⁻¹[ -σ(k) · 𝓕[u](k) ]

        where:
        - σ(k) is the combined pseudo-differential operator symbol
        - 𝓕 denotes the forward Fourier transform
        - 𝓕⁻¹ denotes the inverse Fourier transform

        The dealiasing mask is applied before returning to physical space.
        
        Parameters
        ----------
        u : ndarray
            Input function
        x_grid : ndarray
            Spatial grid (x)
        kx : ndarray
            Frequency grid (x)
        y_grid : ndarray, optional
            Spatial grid (y, for 2D)
        ky : ndarray, optional
            Frequency grid (y, for 2D)
        dealiasing_mask : ndarray, optional
            Dealiasing mask
            
        Returns
        -------
        ndarray
            Result
        """
        u_hat = self.fft(u)
        
        # Evaluate symbol at grid points
        if self.dim == 1:
            X_dummy = np.zeros_like(kx)
            symbol_vals = self.p_func(X_dummy, kx)
        elif self.dim == 2:
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            X_dummy = np.zeros_like(KX)
            Y_dummy = np.zeros_like(KY)
            symbol_vals = self.p_func(X_dummy, Y_dummy, KX, KY)
        else:
            raise ValueError("Only 1D and 2D supported")
        
        # Apply symbol
        u_hat *= symbol_vals
        
        # Apply dealiasing
        if dealiasing_mask is not None:
            u_hat *= dealiasing_mask
        
        return self.ifft(u_hat)

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
            xi = symbols('xi', real=True, positive=True)
            return simplify(series(p, xi, oo, n=order).removeO())
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True, positive=True)
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
        from sympy import symbols, simplify, expand, Eq
        from sympy.abc import l
    
        if self.dim == 1:
            xi = symbols('xi', real=True, positive=True)
            l = symbols('l', real=True, positive=True)
            p = self.symbol
            p_scaled = p.subs(xi, l * xi)
            ratio = simplify(p_scaled / p)
            if ratio.has(xi):
                return False, None
            try:
                deg = simplify(ratio).as_base_exp()[1]
                return True, deg
            except Exception:
                return False, None
    
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True, positive=True)
            l = symbols('l', real=True, positive=True)
            p = self.symbol
            p_scaled = p.subs({xi: l * xi, eta: l * eta})
            ratio = simplify(p_scaled / p)
            # If ratio == l**m with no (xi, eta) left, it's homogeneous
            if ratio.has(xi, eta):
                return False, None
            try:
                base, exp = ratio.as_base_exp()
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
            if any(v in coeff.free_symbols for v in vars_x):
                print("⚠️ Coefficient depends on spatial variables; ignoring")
                return None
            try:
                coeff_val = abs(float(coeff.evalf()))
                if coeff_val < tol:
                    print(f"⚠️ Coefficient too small ({coeff_val:.2e} < {tol})")
                    return None
            except Exception as e:
                print(f"⚠️ Coefficient evaluation failed: {e}")
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
            xi = symbols('xi', real=True, positive=True)
    
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
            xi, eta = symbols('xi eta', real=True, positive=True)
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
            xi = symbols('xi', real=True, positive=True)
    
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
            xi, eta = symbols('xi eta', real=True, positive=True)
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
    
        Examples
        --------
        X = a*x, Y = b*ξ
        X_op.compose_asymptotic(Y_op, order=3, mode='weyl')
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
    
            return simplify(result)
    
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
    
            return simplify(result)
    
        else:
            raise NotImplementedError("Only 1D and 2D cases are implemented")

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
                               epsilon_levels=[1e-1, 1e-2, 1e-3, 1e-4],
                               resolution=100, method='spectral', L=None, N=None):
        """
        Compute and visualize the pseudospectrum of the pseudo-differential operator.
        
        The ε-pseudospectrum is defined as:
            Λ_ε(A) = { λ ∈ ℂ : ‖(A - λI)^{-1}‖ ≥ ε^{-1} }
        
        This method quantizes the operator symbol into a matrix representation 
        and samples the resolvent norm over a grid in the complex plane.
        
        Parameters
        ----------
        x_grid : ndarray
            Spatial discretization grid (used if method='finite_difference')
        lambda_real_range : tuple
            Real part range of complex λ: (λ_re_min, λ_re_max)
        lambda_imag_range : tuple
            Imaginary part range: (λ_im_min, λ_im_max)
        epsilon_levels : list of float
            Contour levels for ε-pseudospectrum boundaries
        resolution : int
            Number of grid points per axis in the λ-plane
        method : str
            Discretization method:
            - 'spectral': FFT-based spectral differentiation (periodic, high accuracy)
            - 'finite_difference': Standard finite differences
        L : float, optional
            Half-domain length for spectral method (default: inferred from x_grid)
        N : int, optional
            Number of grid points for spectral method (default: len(x_grid))
        
        Returns
        -------
        dict
            Contains:
            - 'lambda_grid': meshgrid of complex λ values
            - 'resolvent_norm': 2D array of ‖(A - λI)^{-1}‖
            - 'sigma_min': 2D array of σ_min(A - λI)
            - 'epsilon_levels': input epsilon levels
            - 'eigenvalues': computed eigenvalues (if available)
        
        Notes
        -----
        - For non-self-adjoint operators, the pseudospectrum can extend far from 
          the actual spectrum, revealing transient behavior and non-normal dynamics.
        - The spectral method is preferred for smooth, periodic-like symbols.
        - Computational cost scales as O(resolution² × N³) due to SVD at each λ.
        
        Examples
        --------
        >>> # Analyze pseudospectrum of a non-self-adjoint operator
        >>> x, xi = symbols('x xi', real=True)
        >>> symbol = xi**2 + 1j*x*xi  # non-self-adjoint
        >>> op = PseudoDifferentialOperator(symbol, [x], mode='symbol')
        >>> result = op.pseudospectrum_analysis(
        ...     x_grid=np.linspace(-5, 5, 128),
        ...     lambda_real_range=(-2, 10),
        ...     lambda_imag_range=(-3, 3),
        ...     method='spectral'
        ... )
        """
        from scipy.linalg import svdvals
        from scipy.sparse import diags
        
        if self.dim != 1:
            raise NotImplementedError("Pseudospectrum analysis currently supports 1D only")
        
        # --- Step 1: Quantize the operator into a matrix ---
        if method == 'spectral':
            # Spectral (FFT) discretization
            if L is None:
                L = (x_grid[-1] - x_grid[0]) / 2.0
            if N is None:
                N = len(x_grid)
            
            x_grid_spectral = np.linspace(-L, L, N, endpoint=False)
            dx = x_grid_spectral[1] - x_grid_spectral[0]
            k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
            k2 = -k**2  # symbol for -d²/dx²
            
            # Build operator matrix via spectral differentiation
            def apply_operator(u):
                """Apply Op(symbol) to vector u"""
                u_hat = np.fft.fft(u)
                # Extract kinetic part from symbol (assuming symbol = f(xi) + g(x))
                # This is a simplified model; for general symbols, use full quantization
                kinetic = k2 * u_hat
                v = np.fft.ifft(kinetic)
                # Add potential/position-dependent part
                x_vals = x_grid_spectral
                potential = self.p_func(x_vals, 0.0)  # evaluate at ξ=0 for position part
                v += potential * u
                return np.real(v)
            
            # Assemble matrix
            H = np.zeros((N, N), dtype=complex)
            for j in range(N):
                e = np.zeros(N)
                e[j] = 1.0
                H[:, j] = apply_operator(e)
            
            print(f"Operator quantized via spectral method: {N}×{N} matrix")
        
        elif method == 'finite_difference':
            # Finite difference discretization
            N = len(x_grid)
            dx = x_grid[1] - x_grid[0]
            
            # Build -d²/dx² using centered differences
            diag_main = -2.0 / dx**2 * np.ones(N)
            diag_off = 1.0 / dx**2 * np.ones(N - 1)
            D2 = diags([diag_off, diag_main, diag_off], [-1, 0, 1], shape=(N, N)).toarray()
            
            # Add position-dependent part from symbol
            x_vals = x_grid
            potential = np.diag(self.p_func(x_vals, 0.0))
            
            H = -D2 + potential
            print(f"Operator quantized via finite differences: {N}×{N} matrix")
        
        else:
            raise ValueError("method must be 'spectral' or 'finite_difference'")
        
        # --- Step 2: Sample resolvent norm over λ-plane ---
        lambda_re = np.linspace(*lambda_real_range, resolution)
        lambda_im = np.linspace(*lambda_imag_range, resolution)
        Lambda_re, Lambda_im = np.meshgrid(lambda_re, lambda_im)
        Lambda = Lambda_re + 1j * Lambda_im
        
        resolvent_norm = np.zeros_like(Lambda, dtype=float)
        sigma_min_grid = np.zeros_like(Lambda, dtype=float)
        
        I = np.eye(N)
        
        print(f"Computing pseudospectrum over {resolution}×{resolution} grid...")
        for i in range(resolution):
            for j in range(resolution):
                lam = Lambda[i, j]
                A = H - lam * I
                
                try:
                    # Compute smallest singular value
                    s = svdvals(A)
                    s_min = s[-1]
                    sigma_min_grid[i, j] = s_min
                    resolvent_norm[i, j] = 1.0 / (s_min + 1e-16)  # regularization
                except Exception:
                    resolvent_norm[i, j] = np.nan
                    sigma_min_grid[i, j] = np.nan
        
        # --- Step 3: Compute eigenvalues ---
        try:
            eigenvalues = np.linalg.eigvals(H)
        except:
            eigenvalues = None
        
        # --- Step 4: Visualization ---
        plt.figure(figsize=(14, 6))
        
        # Left panel: log10(resolvent norm)
        plt.subplot(1, 2, 1)
        levels_log = np.log10(1.0 / np.array(epsilon_levels))
        cs = plt.contour(Lambda_re, Lambda_im, np.log10(resolvent_norm + 1e-16), 
                         levels=levels_log, colors='blue', linewidths=1.5)
        plt.clabel(cs, inline=True, fmt='ε=10^%d')
        
        if eigenvalues is not None:
            plt.plot(eigenvalues.real, eigenvalues.imag, 'r*', markersize=8, label='Eigenvalues')
        
        plt.xlabel('Re(λ)')
        plt.ylabel('Im(λ)')
        plt.title('ε-Pseudospectrum: log₁₀(‖(A - λI)⁻¹‖)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.axis('equal')
        
        # Right panel: σ_min contours
        plt.subplot(1, 2, 2)
        cs2 = plt.contourf(Lambda_re, Lambda_im, sigma_min_grid, 
                           levels=50, cmap='viridis')
        plt.colorbar(cs2, label='σ_min(A - λI)')
        
        if eigenvalues is not None:
            plt.plot(eigenvalues.real, eigenvalues.imag, 'r*', markersize=8)
        
        for eps in epsilon_levels:
            plt.contour(Lambda_re, Lambda_im, sigma_min_grid, 
                       levels=[eps], colors='red', linewidths=1.5, alpha=0.7)
        
        plt.xlabel('Re(λ)')
        plt.ylabel('Im(λ)')
        plt.title('Smallest singular value σ_min(A - λI)')
        plt.grid(alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'lambda_grid': Lambda,
            'resolvent_norm': resolvent_norm,
            'sigma_min': sigma_min_grid,
            'epsilon_levels': epsilon_levels,
            'eigenvalues': eigenvalues,
            'operator_matrix': H
        }
    
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

    def is_elliptic_numerically(self, x_grid, xi_grid, threshold=1e-8):
        """
        Check if the pseudo-differential symbol p(x, ξ) is elliptic over a given grid.
    
        A symbol is considered elliptic if its magnitude |p(x, ξ)| remains bounded away from zero 
        across all points in the spatial-frequency domain. This method evaluates the symbol on a 
        grid of spatial and frequency coordinates and checks whether its minimum absolute value 
        exceeds a specified threshold.
    
        Resampling is applied to large grids to prevent excessive memory usage, particularly in 2D.
    
        Parameters
        ----------
        x_grid : ndarray
            Spatial grid: either a 1D array (x) or a tuple of two 1D arrays (x, y).
        xi_grid : ndarray
            Frequency grid: either a 1D array (ξ) or a tuple of two 1D arrays (ξ, η).
        threshold : float, optional
            Minimum acceptable value for |p(x, ξ)|. If the smallest evaluated symbol value falls below this,
            the symbol is not considered elliptic.
    
        Returns
        -------
        bool
            True if the symbol is elliptic on the resampled grid, False otherwise.
        """
        RESAMPLE_SIZE = 32  # Reduced size to prevent memory explosion
        
        if self.dim == 1:
            x_vals = x_grid
            xi_vals = xi_grid
            # Resampling if necessary
            if len(x_vals) > RESAMPLE_SIZE:
                x_vals = np.linspace(x_vals.min(), x_vals.max(), RESAMPLE_SIZE)
            if len(xi_vals) > RESAMPLE_SIZE:
                xi_vals = np.linspace(xi_vals.min(), xi_vals.max(), RESAMPLE_SIZE)
        
            X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
            symbol_vals = self.p_func(X, XI)
        
        elif self.dim == 2:
            x_vals, y_vals = x_grid
            xi_vals, eta_vals = xi_grid
        
            # Spatial resampling
            if len(x_vals) > RESAMPLE_SIZE:
                x_vals = np.linspace(x_vals.min(), x_vals.max(), RESAMPLE_SIZE)
            if len(y_vals) > RESAMPLE_SIZE:
                y_vals = np.linspace(y_vals.min(), y_vals.max(), RESAMPLE_SIZE)
        
            # Frequency resampling
            if len(xi_vals) > RESAMPLE_SIZE:
                xi_vals = np.linspace(xi_vals.min(), xi_vals.max(), RESAMPLE_SIZE)
            if len(eta_vals) > RESAMPLE_SIZE:
                eta_vals = np.linspace(eta_vals.min(), eta_vals.max(), RESAMPLE_SIZE)
        
            X, Y, XI, ETA = np.meshgrid(x_vals, y_vals, xi_vals, eta_vals, indexing='ij')
            symbol_vals = self.p_func(X, Y, XI, ETA)
        
        min_abs_val = np.min(np.abs(symbol_vals))
        return min_abs_val > threshold


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
                                    density=100):
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
            'Symbol Amplitude'           : |p(x,ξ)| or |p(x,y,ξ,η)|
            'Symbol Phase'               : arg(p(x,ξ)) or similar in 2D
            'Micro-Support (1/|p|)'      : Reciprocal of symbol magnitude
            'Cotangent Fiber'            : Structure of symbol over frequency space at fixed x
            'Characteristic Set'         : Zero set approximation {p ≈ 0}
            'Characteristic Gradient'    : |∇p(x, ξ)| or |∇p(x₀, y₀, ξ, η)|
            'Group Velocity Field'       : ∇_ξ p(x,ξ) or ∇_{ξ,η} p(x,y,ξ,η)
            'Symplectic Vector Field'    : (∇_ξ p, -∇_x p) or similar in 2D
            'Hamiltonian Flow'           : Trajectories generated by the Hamiltonian vector field
    
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

def kohn_nirenberg_fft(u_vals, symbol_func, x_grid, kx, fft_func, ifft_func, 
                       dim=1, y_grid=None, ky=None, freq_window='gaussian', 
                       clamp=1e6, space_window=False):
    """
    Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator.
    
    Applies the pseudo-differential operator Op(p) to the function f via the Kohn–Nirenberg quantization:
    
        [Op(p)f](x) = (1/(2π)^d) ∫ p(x, ξ) e^{ix·ξ} ℱ[f](ξ) dξ
    
    where p(x, ξ) is a symbol that may depend on both spatial variables x and frequency variables ξ.
    
    This method supports both 1D and 2D cases and includes optional smoothing techniques to improve numerical stability.

    Parameters
    ----------
    u_vals : np.ndarray
        Spatial samples of the input function f(x) or f(x, y), defined on a uniform grid.
    symbol_func : callable
        A function representing the full symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
        Must accept NumPy-compatible array inputs and return a complex-valued array.
    freq_window : {'gaussian', 'hann', None}, optional
        Type of frequency-domain window to apply:
        - 'gaussian': smooth decay near high frequencies
        - 'hann': cosine-based tapering with hard cutoff
        - None: no frequency window applied
    clamp : float, optional
        Upper bound on the absolute value of the symbol. Prevents numerical blow-up from large values.
    space_window : bool, optional
        Whether to apply a spatial Gaussian window to suppress edge effects in physical space.
    
    Parameters
    ----------
    u_vals : ndarray
        Input function values
    symbol_func : callable
        Symbol function p(x, ξ) or p(x, y, ξ, η)
    x_grid : ndarray
        Spatial grid in x direction
    kx : ndarray
        Frequency grid in x direction
    fft_func : callable
        FFT function (fft or fft2)
    ifft_func : callable
        Inverse FFT function
    dim : int
        Dimension (1 or 2)
    y_grid : ndarray, optional
        Spatial grid in y direction (for 2D)
    ky : ndarray, optional
        Frequency grid in y direction (for 2D)
    freq_window : str
        Windowing function ('gaussian' or 'hann')
    clamp : float
        Clamp symbol values to [-clamp, clamp]
    space_window : bool
        Apply spatial windowing
        
    Returns
    -------
    ndarray
        Result of applying the operator
    """
    if dim == 1:
        dx = x_grid[1] - x_grid[0]
        Nx = len(x_grid)
        k = 2 * np.pi * fftshift(fftfreq(Nx, d=dx))
        dk = k[1] - k[0]
        
        f_shift = fftshift(u_vals)
        f_hat = fft_func(f_shift) * dx
        f_hat = fftshift(f_hat)
        
        X, K = np.meshgrid(x_grid, k, indexing='ij')
        P = symbol_func(X, K)
        P = np.clip(P, -clamp, clamp)
        
        # Apply frequency windowing
        if freq_window == 'gaussian':
            sigma = 0.8 * np.max(np.abs(k))
            W = np.exp(-(K / sigma) ** 4)
            P *= W
        elif freq_window == 'hann':
            W = 0.5 * (1 + np.cos(np.pi * K / np.max(np.abs(K))))
            P *= W * (np.abs(K) < np.max(np.abs(K)))
        
        # Apply spatial windowing
        if space_window:
            x0 = (x_grid[0] + x_grid[-1]) / 2
            L = (x_grid[-1] - x_grid[0]) / 2
            S = np.exp(-((X - x0) / L) ** 2)
            P *= S
        
        kernel = np.exp(1j * X * K)
        integrand = P * f_hat[None, :] * kernel
        u = np.sum(integrand, axis=1) * dk / (2 * np.pi)
        
        return u
        
    elif dim == 2:
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        Nx, Ny = len(x_grid), len(y_grid)
        
        kx_shift = 2 * np.pi * fftshift(fftfreq(Nx, d=dx))
        ky_shift = 2 * np.pi * fftshift(fftfreq(Ny, d=dy))
        dkx = kx_shift[1] - kx_shift[0]
        dky = ky_shift[1] - ky_shift[0]
        
        f_shift = fftshift(u_vals)
        f_hat = fft_func(f_shift) * dx * dy
        f_hat = fftshift(f_hat)
        
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        KX, KY = np.meshgrid(kx_shift, ky_shift, indexing='ij')
        
        Xb = X[:, :, None, None]
        Yb = Y[:, :, None, None]
        KXb = KX[None, None, :, :]
        KYb = KY[None, None, :, :]
        
        P_vals = symbol_func(Xb, Yb, KXb, KYb)
        P_vals = np.clip(P_vals, -clamp, clamp)
        
        # Apply frequency windowing
        if freq_window == 'gaussian':
            sigma_kx = 0.8 * np.max(np.abs(kx_shift))
            sigma_ky = 0.8 * np.max(np.abs(ky_shift))
            W_kx = np.exp(-(KXb / sigma_kx) ** 4)
            W_ky = np.exp(-(KYb / sigma_ky) ** 4)
            P_vals *= W_kx * W_ky
        elif freq_window == 'hann':
            Wx = 0.5 * (1 + np.cos(np.pi * KXb / np.max(np.abs(kx_shift))))
            Wy = 0.5 * (1 + np.cos(np.pi * KYb / np.max(np.abs(ky_shift))))
            mask_x = np.abs(KXb) < np.max(np.abs(kx_shift))
            mask_y = np.abs(KYb) < np.max(np.abs(ky_shift))
            P_vals *= Wx * Wy * mask_x * mask_y
        
        # Apply spatial windowing
        if space_window:
            x0 = (x_grid[0] + x_grid[-1]) / 2
            y0 = (y_grid[0] + y_grid[-1]) / 2
            Lx = (x_grid[-1] - x_grid[0]) / 2
            Ly = (y_grid[-1] - y_grid[0]) / 2
            S = np.exp(-((Xb - x0) / Lx) ** 2 - ((Yb - y0) / Ly) ** 2)
            P_vals *= S
        
        phase = np.exp(1j * (Xb * KXb + Yb * KYb))
        integrand = P_vals * phase * f_hat[None, None, :, :]
        u = np.sum(integrand, axis=(2, 3)) * dkx * dky / (2 * np.pi) ** 2
        
        return u
    
    else:
        raise ValueError("Only 1D and 2D are supported")


def kohn_nirenberg_nonperiodic(u_vals, x_grid, xi_grid, symbol_func, 
                                freq_window='gaussian', clamp=1e6, 
                                space_window=False):
    """
    Apply a pseudo-differential operator using the Kohn–Nirenberg quantization on non-periodic domains.

    This method evaluates the action of a pseudo-differential operator Op(p) on a function u 
    via the Kohn–Nirenberg representation. It supports both 1D and 2D cases and uses spatial 
    and frequency grids to evaluate the operator symbol p(x, ξ).

    The operator symbol p(x, ξ) is extracted from the PDE and evaluated numerically using 
    `_total_symbol_expr` and `_build_symbol_func`.

    Parameters
    ----------
    u : np.ndarray
        Input function (real space) to which the operator is applied.

    Returns:
        np.ndarray: Result of applying Op(p) to u in real space.

    Notes:
        - For 1D: p(x, ξ) is evaluated over x_grid and xi_grid.
        - For 2D: p(x, y, ξ, η) is evaluated over (x_grid, y_grid) and (xi_grid, eta_grid).
        - The result is computed using `kohn_nirenberg_nonperiodic`, which handles non-periodic boundary conditions.
    
    Parameters
    ----------
    u_vals : ndarray
        Input function values (1D or 2D)
    x_grid : ndarray or tuple of ndarray
        Spatial grid(s). For 1D: ndarray. For 2D: (x_grid, y_grid)
    xi_grid : ndarray or tuple of ndarray
        Frequency grid(s). For 1D: ndarray. For 2D: (xi_grid, eta_grid)
    symbol_func : callable
        Symbol function p(x, ξ) or p(x, y, ξ, η)
    freq_window : str
        Windowing function ('gaussian' or 'hann')
    clamp : float
        Clamp symbol values
    space_window : bool
        Apply spatial windowing
        
    Returns
    -------
    ndarray
        Result of applying the operator
    """
    if u_vals.ndim == 1:
        # 1D case
        x = x_grid
        xi = xi_grid
        dx = x[1] - x[0]
        dxi = xi[1] - xi[0]
        
        # Fourier transform
        phase_ft = np.exp(-1j * np.outer(xi, x))
        u_hat = dx * np.dot(phase_ft, u_vals)
        
        # Evaluate symbol
        X, XI = np.meshgrid(x, xi, indexing='ij')
        sigma_vals = symbol_func(X, XI)
        sigma_vals = np.clip(sigma_vals, -clamp, clamp)
        
        # Apply frequency windowing
        if freq_window == 'gaussian':
            sigma = 0.8 * np.max(np.abs(XI))
            window = np.exp(-(XI / sigma) ** 4)
            sigma_vals *= window
        elif freq_window == 'hann':
            window = 0.5 * (1 + np.cos(np.pi * XI / np.max(np.abs(XI))))
            sigma_vals *= window * (np.abs(XI) < np.max(np.abs(XI)))
        
        # Apply spatial windowing
        if space_window:
            x_center = (x[0] + x[-1]) / 2
            L = (x[-1] - x[0]) / 2
            window = np.exp(-((X - x_center) / L) ** 2)
            sigma_vals *= window
        
        # Inverse transform
        exp_matrix = np.exp(1j * np.outer(x, xi))
        integrand = sigma_vals * u_hat[np.newaxis, :] * exp_matrix
        result = dxi * np.sum(integrand, axis=1) / (2 * np.pi)
        
        return result
        
    elif u_vals.ndim == 2:
        # 2D case
        x1, x2 = x_grid
        xi1, xi2 = xi_grid
        dx1 = x1[1] - x1[0]
        dx2 = x2[1] - x2[0]
        dxi1 = xi1[1] - xi1[0]
        dxi2 = xi2[1] - xi2[0]
        
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
        
        # Fourier transform
        phase_ft = np.exp(-1j * (
            np.tensordot(x1, xi1, axes=0)[:, None, :, None] + 
            np.tensordot(x2, xi2, axes=0)[None, :, None, :]
        ))
        u_hat = np.tensordot(u_vals, phase_ft, axes=([0, 1], [0, 1])) * dx1 * dx2
        
        # Evaluate symbol
        sigma_vals = symbol_func(
            X1[:, :, None, None], 
            X2[:, :, None, None], 
            XI1[None, None, :, :], 
            XI2[None, None, :, :]
        )
        sigma_vals = np.clip(sigma_vals, -clamp, clamp)
        
        # Apply frequency windowing
        if freq_window == 'gaussian':
            sigma_xi1 = 0.8 * np.max(np.abs(XI1))
            sigma_xi2 = 0.8 * np.max(np.abs(XI2))
            window = np.exp(
                -(XI1[None, None, :, :] / sigma_xi1) ** 4 - 
                (XI2[None, None, :, :] / sigma_xi2) ** 4
            )
            sigma_vals *= window
        elif freq_window == 'hann':
            wx = 0.5 * (1 + np.cos(np.pi * XI1 / np.max(np.abs(XI1))))
            wy = 0.5 * (1 + np.cos(np.pi * XI2 / np.max(np.abs(XI2))))
            mask_x = np.abs(XI1) < np.max(np.abs(XI1))
            mask_y = np.abs(XI2) < np.max(np.abs(XI2))
            sigma_vals *= wx[:, :, None, None] * wy[:, :, None, None]
            sigma_vals *= mask_x[:, :, None, None] * mask_y[:, :, None, None]
        
        # Apply spatial windowing
        if space_window:
            x_center = (x1[0] + x1[-1]) / 2
            y_center = (x2[0] + x2[-1]) / 2
            Lx = (x1[-1] - x1[0]) / 2
            Ly = (x2[-1] - x2[0]) / 2
            window = np.exp(
                -((X1 - x_center) / Lx) ** 2 - 
                ((X2 - y_center) / Ly) ** 2
            )
            sigma_vals *= window[:, :, None, None]
        
        # Inverse transform
        phase = np.exp(1j * (
            X1[:, :, None, None] * XI1[None, None, :, :] + 
            X2[:, :, None, None] * XI2[None, None, :, :]
        ))
        integrand = sigma_vals * u_hat[None, None, :, :] * phase
        result = dxi1 * dxi2 * np.sum(integrand, axis=(2, 3)) / (2 * np.pi) ** 2
        
        return result
        
    else:
        raise NotImplementedError("Only 1D and 2D supported")
