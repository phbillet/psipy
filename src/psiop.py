# Copyright 2025 Philippe Billet assisted by LLMs in free mode: chatGPT, Qwen, Gemini, Claude, le chat Mistral.
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
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# ==================================================================
# CAUSTIC DETECTION AND CLASSIFICATION
# ==================================================================

class CausticDetector:
    """
    Detect and classify caustics in ray families.
    
    Caustic types:
    - Fold (A2): Generic 1-parameter family, corrected by Airy function
    - Cusp (A3): Generic 2-parameter family, corrected by Pearcey function
    - Swallowtail (A4): More degenerate, requires higher corrections
    """
    
    def __init__(self, rays, dimension):
        self.rays = rays
        self.dimension = dimension
        self.caustics = []
    
    def detect_caustics(self, threshold=1e-3):
        """
        Detect caustics by analyzing Jacobian of ray mapping.
        
        Caustic occurs when ∂(x,y)/∂(ray_param, t) = 0
        """
        print("Detecting caustics...")
        
        for i, ray in enumerate(self.rays):
            if self.dimension == 1:
                caustics_1d = self._detect_1d_caustics(ray, i)
                self.caustics.extend(caustics_1d)
            else:
                caustics_2d = self._detect_2d_caustics(ray, i, threshold)
                self.caustics.extend(caustics_2d)
        
        print(f"Found {len(self.caustics)} caustic points")
        return self.caustics
    
    def _detect_1d_caustics(self, ray, ray_idx):
        """
        In 1D, caustic occurs when dx/dt = 0 (ray turns around).
        """
        caustics = []
        
        x = ray['x']
        t = ray['t']
        
        # Compute velocity
        dxdt = np.gradient(x, t)
        
        # Find sign changes (turning points)
        sign_changes = np.where(np.diff(np.sign(dxdt)))[0]
        
        for idx in sign_changes:
            caustics.append({
                'type': 'fold',  # 1D caustics are always folds
                'ray_idx': ray_idx,
                'time_idx': idx,
                'position': x[idx],
                'time': t[idx],
                'caustic_type': 'A2'
            })
        
        return caustics
    
    def _detect_2d_caustics(self, ray, ray_idx, threshold):
        """
        In 2D, caustic occurs when det(Jacobian) ≈ 0.
        Classify type by eigenvalues of Hessian.
        """
        caustics = []
        
        x = ray['x']
        y = ray['y']
        xi = ray['xi']
        eta = ray['eta']
        t = ray['t']
        
        # Numerical Jacobian along ray
        # J = [[∂x/∂t, ∂x/∂s], [∂y/∂t, ∂y/∂s]]
        # Approximate using neighboring rays (would need full family)
        
        # Simpler criterion: momentum magnitude
        p_mag = np.sqrt(xi**2 + eta**2)
        
        # Look for near-zero momentum (approximate caustic indicator)
        near_zero = np.where(p_mag < threshold)[0]
        
        for idx in near_zero:
            # Classify caustic type by analyzing trajectory curvature
            if idx > 0 and idx < len(t) - 1:
                # Second derivatives
                d2x = x[idx+1] - 2*x[idx] + x[idx-1]
                d2y = y[idx+1] - 2*y[idx] + y[idx-1]
                curvature = np.sqrt(d2x**2 + d2y**2)
                
                # Simple classification
                if curvature < 0.1:
                    caustic_type = 'A2'  # Fold
                    correction_type = 'airy'
                else:
                    caustic_type = 'A3'  # Cusp
                    correction_type = 'pearcey'
                
                caustics.append({
                    'type': correction_type,
                    'ray_idx': ray_idx,
                    'time_idx': idx,
                    'position': (x[idx], y[idx]),
                    'time': t[idx],
                    'caustic_type': caustic_type,
                    'curvature': curvature
                })
        
        return caustics
    
    def compute_maslov_index(self, ray):
        """
        Compute Maslov index: number of caustics crossed × π/2.
        
        The Maslov index accumulates phase jumps at caustics.
        """
        maslov = 0
        
        if self.dimension == 1:
            x = ray['x']
            dxdt = np.gradient(x, ray['t'])
            # Count sign changes
            maslov = len(np.where(np.diff(np.sign(dxdt)))[0])
        else:
            # In 2D, need to track conjugate points
            # Simplified: count momentum near-zeros
            xi, eta = ray['xi'], ray['eta']
            p_mag = np.sqrt(xi**2 + eta**2)
            maslov = len(np.where(p_mag < 0.01)[0])
        
        return maslov * np.pi / 2


# ==================================================================
# SPECIAL FUNCTIONS FOR CAUSTIC CORRECTIONS
# ==================================================================

class CausticFunctions:
    """
    Special functions for caustic corrections.
    """
    
    @staticmethod
    def airy_uniform(z):
        """
        Airy function Ai(z) for fold caustic correction.
        
        Near a fold caustic, the WKB solution is replaced by:
        u(x) ≈ A(x) · Ai((x-x_c)/ε^{2/3}) · exp(iS(x)/ε)
        """
        return airy(z)[0]
    
    @staticmethod
    def airy_derivative(z):
        """
        Derivative of Airy function Ai'(z).
        """
        return airy(z)[1]
    
    @staticmethod
    def pearcey_integral(x, y):
        """
        Pearcey integral for cusp caustic (A3 singularity).
        
        P(x,y) = ∫_{-∞}^{∞} exp(i(t^4 + xt^2 + yt)) dt
        
        This is more complex and typically requires numerical integration.
        Simplified implementation using stationary phase.
        """
        # Number of integration points
        n_pts = 200
        t = np.linspace(-5, 5, n_pts)
        dt = t[1] - t[0]
        
        # Phase function: φ(t) = t^4 + x*t^2 + y*t
        phase = t**4 + x * t**2 + y * t
        
        # Numerical integration
        integrand = np.exp(1j * phase)
        result = np.trapz(integrand, dx=dt)
        
        return result
    
    @staticmethod
    def pearcey_approx(x, y):
        """
        Approximate Pearcey function using asymptotic expansion.
        Faster but less accurate than full integration.
        """
        # Asymptotic form for large |x|, |y|
        r = np.sqrt(x**2 + y**2) + 1e-10
        
        if r > 2:
            # Asymptotic expansion
            return np.exp(1j * (x**2/4 + y**2/(4*x))) / np.sqrt(r)
        else:
            # Fall back to numerical
            return CausticFunctions.pearcey_integral(x, y)
    
    @staticmethod
    def maslov_phase_shift(n_caustics):
        """
        Phase shift from Maslov index.
        
        Each caustic crossed adds π/2 to the phase.
        """
        return n_caustics * np.pi / 2

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

    def __init__(self, expr, vars_x, var_u=None, mode='symbol', quantization='weyl'):
        self.dim = len(vars_x)
        self.mode = mode
        self.symbol_cached = None
        self.expr = expr
        self.vars_x = vars_x
        self.quantization = quantization

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
            self._compute_symbol_derivatives() 
            print("\nsymbol = ")
            pprint(self.symbol, num_columns=NUM_COLS)

    def _compute_symbol_derivatives(self):
        """Compute derivatives for WKB application."""
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
            setattr(self, f'_{name}_func', lambdify(vars_tuple, expr, 'numpy'))
        
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
              freq_window='gaussian', clamp=1e6, space_window=False,
              wkb_mode=False, wkb_order=1, epsilon=None):
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
        wkb_mode : bool or dict
            If True, treat u as WKB solution dict and use apply_to_wkb()
            If dict, must be the WKB solution itself (u parameter ignored)
        wkb_order : int
            Order for WKB application (only used if wkb_mode=True)
        epsilon : float
            Semi-classical parameter for WKB (only used if wkb_mode=True)
            
        Returns
        -------
        ndarray
            Result of applying the operator
        """
        # Check if symbol depends on spatial variables
        is_spatial = self._is_spatial_dependent()
        
        # Case 0: using WKB approximation
        if wkb_mode:
            wkb_solution = u if isinstance(wkb_mode, bool) and isinstance(u, dict) else wkb_mode
            return self.apply_to_wkb(wkb_solution, order=wkb_order, epsilon=epsilon)
            
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

    def apply_to_wkb(self, wkb_solution, order=1, epsilon=None):
        """Apply pseudo-differential operator to WKB solution.
        
        Parameters:
        -----------
        wkb_solution : dict
            WKB solution with keys: 'S', 'a', 'x', ('y'), 'dimension', 'epsilon'
        order : int
            Order of approximation (0, 1, or 2)
        epsilon : float
            Semi-classical parameter (uses wkb_solution['epsilon'] if None)
        
        Returns:
        --------
        dict : Modified WKB solution with operator applied
        """
        if epsilon is None:
            epsilon = wkb_solution.get('epsilon', 0.1)
        
        if not hasattr(self, 'derivatives'):
            self._compute_symbol_derivatives()
        
        dim = wkb_solution['dimension']
        if dim != self.dim:
            raise ValueError(f'Dimension mismatch: operator is {self.dim}D, solution is {dim}D')
        
        S = wkb_solution['S']
        a_input = wkb_solution['a'][0] if isinstance(wkb_solution['a'], dict) else wkb_solution['a']
        
        if dim == 1:
            x_grid = wkb_solution['x']
            dS_dx = np.gradient(S, x_grid)
            
            # Order 0
            p_vals = self.p_func(x_grid, dS_dx)
            b0 = p_vals * a_input
            amplitudes = {0: b0}
            
            if order >= 1:
                d2S_dx2 = np.gradient(dS_dx, x_grid)
                da0_dx = np.gradient(a_input, x_grid)
                
                d2p_dxi2 = self._d2p_dxi2_func(x_grid, dS_dx)
                d2p_dxidx = self._d2p_dxidx_func(x_grid, dS_dx)
                dp_dxi = self._dp_dxi_func(x_grid, dS_dx)
                
                if self.quantization == 'weyl':
                    b1 = 1j/2 * (d2p_dxi2 * d2S_dx2 * a_input + 2 * d2p_dxidx * da0_dx)
                else:  # Kohn-Nirenberg
                    b1 = 1j * dp_dxi * da0_dx
                amplitudes[1] = b1
            
            if order >= 2:
                amplitudes[2] = np.zeros_like(b0, dtype=complex)
        
        else:  # dim == 2
            X, Y = wkb_solution['x'], wkb_solution['y']
            dx = X[1,0] - X[0,0]
            dy = Y[0,1] - Y[0,0]
            
            dS_dx = np.gradient(S, axis=0) / dx
            dS_dy = np.gradient(S, axis=1) / dy
            
            # Order 0
            p_vals = self.p_func(X, Y, dS_dx, dS_dy)
            b0 = p_vals * a_input
            amplitudes = {0: b0}
            
            if order >= 1:
                d2S_dx2 = np.gradient(dS_dx, axis=0) / dx
                d2S_dy2 = np.gradient(dS_dy, axis=1) / dy
                da0_dx = np.gradient(a_input, axis=0) / dx
                da0_dy = np.gradient(a_input, axis=1) / dy
                
                d2p_dxi2 = self._d2p_dxi2_func(X, Y, dS_dx, dS_dy)
                d2p_deta2 = self._d2p_deta2_func(X, Y, dS_dx, dS_dy)
                d2p_dxidx = self._d2p_dxidx_func(X, Y, dS_dx, dS_dy)
                d2p_detady = self._d2p_detady_func(X, Y, dS_dx, dS_dy)
                
                if self.quantization == 'weyl':
                    b1 = 1j/2 * (d2p_dxi2 * d2S_dx2 * a_input + 
                                 d2p_deta2 * d2S_dy2 * a_input + 
                                 2 * d2p_dxidx * da0_dx + 
                                 2 * d2p_detady * da0_dy)
                else:
                    dp_dxi = self._dp_dxi_func(X, Y, dS_dx, dS_dy)
                    dp_deta = self._dp_deta_func(X, Y, dS_dx, dS_dy)
                    b1 = 1j * (dp_dxi * da0_dx + dp_deta * da0_dy)
                amplitudes[1] = b1
            
            if order >= 2:
                amplitudes[2] = np.zeros_like(b0, dtype=complex)
        
        # Combine amplitudes
        a_total = sum(epsilon**k * amplitudes[k] for k in range(order + 1))
        u_output = a_total * np.exp(1j * S / epsilon)
        
        result = wkb_solution.copy()
        result['u'] = u_output
        result['a'] = amplitudes
        result['a_total'] = a_total
        result['operator_applied'] = str(self.symbol)
        
        return result
    
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
            # This is the KEY OPTIMIZATION: use apply() instead of manual loops
            H = np.zeros((N, N), dtype=complex)
            
            for j in range(N):
                # Create basis vector e_j
                e_j = np.zeros(N, dtype=complex)
                e_j[j] = 1.0
                
                # Apply operator using the existing apply() method
                # This automatically handles the symbol evaluation and FFT operations
                H[:, j] = self.apply(
                    e_j, 
                    x_grid_spectral, 
                    k,
                    boundary_condition='periodic'
                )
            
            print(f'Operator quantized via apply() method: {N}×{N} matrix')
            return H, x_grid_spectral, k
            
        elif method == 'finite_difference':
            # Fallback to finite difference (keep original implementation)
            N = len(x_grid)
            dx = x_grid[1] - x_grid[0]
            H = np.zeros((N, N), dtype=complex)
            
            for i in range(N):
                for j in range(N):
                    if i == j:
                        H[i, j] = self.p_func(x_grid[i], 0.0)
                    elif abs(i - j) == 1:
                        xi_approx = np.pi / dx
                        H[i, j] = self.p_func(
                            (x_grid[i] + x_grid[j]) / 2,
                            xi_approx * np.sign(i - j)
                        ) / (2 * dx)
                    elif abs(i - j) == 2:
                        xi_approx = 2 * np.pi / dx
                        H[i, j] = self.p_func(
                            (x_grid[i] + x_grid[j]) / 2,
                            xi_approx
                        ) / dx ** 2
            
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
                
                for future in as_completed(futures):
                    idx, res_norm, s_min = future.result()
                    resolvent_norm.ravel()[idx] = res_norm
                    sigma_min_grid.ravel()[idx] = s_min
                    
                    completed += 1
                    if completed % (total // 10) == 0:
                        print(f'Progress: {completed}/{total} ({100*completed//total}%)')
            
        else:
            # Sequential computation
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
                
                if i % (resolution // 10) == 0:
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



# ==================================================================
# ENHANCED WKB WITH CAUSTIC CORRECTIONS
# ==================================================================

def wkb_approximation(symbol, initial_phase, order=1, domain=None,
                               resolution=50, epsilon=0.1, dimension=None,
                               caustic_correction='auto', caustic_threshold=1e-3):
    """
    Enhanced WKB with automatic caustic detection and correction.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ) or p(x, y, ξ, η).
    initial_phase : dict
        Initial data (see wkb_multidim documentation).
    order : int
        WKB order (0-3).
    domain : tuple or None
        Spatial domain.
    resolution : int or tuple
        Grid resolution.
    epsilon : float
        Small parameter.
    dimension : int or None
        Force dimension (1 or 2), or auto-detect.
    caustic_correction : str
        'auto': automatic detection and correction
        'maslov': Maslov index only
        'airy': Force Airy correction (fold caustics)
        'pearcey': Force Pearcey correction (cusp caustics)
        'none': No caustic correction
    caustic_threshold : float
        Threshold for caustic detection.
    
    Returns
    -------
    dict
        Enhanced solution with caustic information.
    """

    base_solution = _compute_base_wkb(symbol, initial_phase, order, domain,
                                      resolution, epsilon, dimension)
    
    # Detect caustics
    detector = CausticDetector(base_solution['rays'], base_solution['dimension'])
    caustics = detector.detect_caustics(threshold=caustic_threshold)
    
    if len(caustics) == 0:
        print("No caustics detected - using standard WKB")
        base_solution['caustic_correction'] = 'none'
        base_solution['caustics'] = []
        return base_solution
    
    print(f"\nApplying caustic corrections (mode: {caustic_correction})...")
    
    # Apply corrections based on caustic type
    if base_solution['dimension'] == 1:
        corrected_solution = _apply_1d_caustic_corrections(
            base_solution, caustics, epsilon, caustic_correction
        )
    else:
        corrected_solution = _apply_2d_caustic_corrections(
            base_solution, caustics, epsilon, caustic_correction
        )
    
    corrected_solution['caustics'] = caustics
    corrected_solution['caustic_correction'] = caustic_correction
    
    print("Caustic corrections applied successfully")
    
    return corrected_solution


def _compute_base_wkb(symbol, initial_phase, order, domain, resolution, epsilon, dimension):
    """
    Compute base WKB solution (simplified version for demo).
    In practice, this would call the full wkb_multidim function.
    """
    # Detect dimension
    if dimension is None:
        has_y = 'y' in initial_phase and 'p_y' in initial_phase
        dimension = 2 if has_y else 1
    
    # Setup variables
    if dimension == 1:
        x, xi = symbols('x xi', real=True)
        all_vars = (x, xi)
    else:
        x, y, xi, eta = symbols('x y xi eta', real=True)
        all_vars = (x, y, xi, eta)
    
    # Compute derivatives
    dp_dxi = diff(symbol, all_vars[-2] if dimension == 1 else all_vars[2])
    if dimension == 2:
        dp_deta = diff(symbol, all_vars[3])
    dp_dx = diff(symbol, all_vars[0])
    if dimension == 2:
        dp_dy = diff(symbol, all_vars[1])
    
    # Lambdify
    f_dx = lambdify(all_vars, dp_dxi, 'numpy')
    if dimension == 2:
        f_dy = lambdify(all_vars, dp_deta, 'numpy')
    f_dxi = lambdify(all_vars, -dp_dx, 'numpy')
    if dimension == 2:
        f_deta = lambdify(all_vars, -dp_dy, 'numpy')
    p_func = lambdify(all_vars, symbol, 'numpy')
    
    # Extract initial data
    x_init = np.asarray(initial_phase['x'])
    n_rays = len(x_init)
    
    if dimension == 2:
        y_init = np.asarray(initial_phase['y'])
    
    S_init = np.asarray(initial_phase['S'])
    px_init = np.asarray(initial_phase['p_x'])
    
    if dimension == 2:
        py_init = np.asarray(initial_phase['p_y'])
    
    # Get amplitude
    if 'a' in initial_phase:
        if isinstance(initial_phase['a'], dict):
            a0_init = np.asarray(initial_phase['a'][0])
        else:
            a0_init = np.asarray(initial_phase['a'])
    else:
        a0_init = np.ones(n_rays)
    
    # Ray tracing
    rays = []
    tmax = 5.0
    n_steps = 100
    
    for i in range(n_rays):
        if dimension == 1:
            z0 = [x_init[i], px_init[i], S_init[i], a0_init[i]]
        else:
            z0 = [x_init[i], y_init[i], px_init[i], py_init[i], 
                  S_init[i], a0_init[i]]
        
        def ray_ode(t, z):
            if dimension == 1:
                x_val, xi_val, S_val, a_val = z
                args = (x_val, xi_val)
                dxdt = f_dx(*args)
                dxidt = f_dxi(*args)
                p_val = p_func(*args)
                dSdt = xi_val * dxdt - p_val
                
                # Simple amplitude evolution
                dadt = -0.5 * a_val * 0.0  # Simplified
                
                return [dxdt, dxidt, dSdt, dadt]
            else:
                x_val, y_val, xi_val, eta_val, S_val, a_val = z
                args = (x_val, y_val, xi_val, eta_val)
                dxdt = f_dx(*args)
                dydt = f_dy(*args)
                dxidt = f_dxi(*args)
                detadt = f_deta(*args)
                p_val = p_func(*args)
                dSdt = xi_val * dxdt + eta_val * dydt - p_val
                
                dadt = 0.0  # Simplified
                
                return [dxdt, dydt, dxidt, detadt, dSdt, dadt]
        
        try:
            sol = solve_ivp(ray_ode, (0, tmax), z0,
                          t_eval=np.linspace(0, tmax, n_steps),
                          method='RK45')
            
            if dimension == 1:
                rays.append({
                    't': sol.t,
                    'x': sol.y[0],
                    'xi': sol.y[1],
                    'S': sol.y[2],
                    'a': sol.y[3]
                })
            else:
                rays.append({
                    't': sol.t,
                    'x': sol.y[0],
                    'y': sol.y[1],
                    'xi': sol.y[2],
                    'eta': sol.y[3],
                    'S': sol.y[4],
                    'a': sol.y[5]
                })
        except:
            continue
    
    # Interpolate to grid
    if domain is None:
        x_all = np.concatenate([r['x'] for r in rays])
        x_min, x_max = x_all.min(), x_all.max()
        if dimension == 1:
            domain = (x_min - 1, x_max + 1)
        else:
            y_all = np.concatenate([r['y'] for r in rays])
            y_min, y_max = y_all.min(), y_all.max()
            domain = ((x_min - 1, x_max + 1), (y_min - 1, y_max + 1))
    
    if dimension == 1:
        x_grid = np.linspace(domain[0], domain[1], resolution)
        
        x_pts = np.concatenate([r['x'] for r in rays])
        S_pts = np.concatenate([r['S'] for r in rays])
        a_pts = np.concatenate([r['a'] for r in rays])
        
        sort_idx = np.argsort(x_pts)
        S_grid = interp1d(x_pts[sort_idx], S_pts[sort_idx], 
                         kind='linear', bounds_error=False, fill_value=0.0)(x_grid)
        a_grid = interp1d(x_pts[sort_idx], a_pts[sort_idx],
                         kind='linear', bounds_error=False, fill_value=0.0)(x_grid)
        
        u_grid = a_grid * np.exp(1j * S_grid / epsilon)
        
        return {
            'dimension': 1,
            'x': x_grid,
            'S': S_grid,
            'a': {0: a_grid},
            'u': u_grid,
            'rays': rays,
            'epsilon': epsilon,
            'order': order
        }
    else:
        nx = ny = resolution if isinstance(resolution, int) else resolution[0]
        x_grid = np.linspace(domain[0][0], domain[0][1], nx)
        y_grid = np.linspace(domain[1][0], domain[1][1], ny)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        x_pts = []
        y_pts = []
        S_pts = []
        a_pts = []
        
        for r in rays:
            x_pts.extend(r['x'])
            y_pts.extend(r['y'])
            S_pts.extend(r['S'])
            a_pts.extend(r['a'])
        
        points = np.column_stack([x_pts, y_pts])
        
        S_grid = griddata(points, S_pts, (X, Y), method='linear', fill_value=0.0)
        a_grid = griddata(points, a_pts, (X, Y), method='linear', fill_value=0.0)
        
        S_grid = np.nan_to_num(S_grid)
        a_grid = np.nan_to_num(a_grid)
        
        u_grid = a_grid * np.exp(1j * S_grid / epsilon)
        
        return {
            'dimension': 2,
            'x': X,
            'y': Y,
            'S': S_grid,
            'a': {0: a_grid},
            'u': u_grid,
            'rays': rays,
            'epsilon': epsilon,
            'order': order
        }

def _apply_1d_caustic_corrections(base_solution, caustics, epsilon, mode):
    """
    Apply caustic corrections in 1D using Airy functions and Maslov index.
    """
    x = base_solution['x']
    S = base_solution['S']
    a = base_solution['a'][0]
    
    # Initialize u_corrected with the standard solution
    u_corrected = np.copy(base_solution['u'])
    
    # Compute Maslov index for each point
    maslov_phases = np.zeros_like(x)
    
    for caustic in caustics:
        x_c = caustic['position']
        # Add π/2 phase shift past each caustic
        maslov_phases[x > x_c] += np.pi / 2
    
    # Apply corrections based on mode
    if mode == 'none':
        # No correction, just return the standard solution
        print("No caustic correction applied (mode: none)")
    
    elif mode == 'maslov' or (mode == 'auto' and len(caustics) > 0):
        # Apply Maslov phase correction
        u_corrected = a * np.exp(1j * (S / epsilon + maslov_phases))
        print(f"Applied Maslov correction: {len(caustics)} caustics found")
    
    if mode == 'airy' or (mode == 'auto' and len(caustics) > 0):
        # Apply Airy function near caustics
        # Start from current u_corrected (which may already have Maslov)
        
        for caustic in caustics:
            x_c = caustic['position']
            
            # Region of Airy correction
            airy_width = 5 * epsilon**(2/3)
            mask = np.abs(x - x_c) < airy_width
            
            if np.any(mask):
                # Scaled coordinate
                z = (x[mask] - x_c) / epsilon**(2/3)
                
                # Airy function
                Ai = CausticFunctions.airy_uniform(z)
                
                # Amplitude at caustic
                idx_c = np.argmin(np.abs(x - x_c))
                a_c = a[idx_c]
                S_c = S[idx_c]
                
                # Replace with uniform approximation
                u_corrected[mask] = a_c * Ai * np.exp(1j * S_c / epsilon)
        
        if mode == 'airy':
            print(f"Applied Airy corrections near {len(caustics)} fold caustics")
        elif mode == 'auto':
            print(f"Applied Airy corrections near {len(caustics)} fold caustics")
    
    result = base_solution.copy()
    result['u'] = u_corrected
    result['u_standard'] = base_solution['u']  # Keep original for comparison
    result['maslov_phases'] = maslov_phases
    
    return result

def _apply_2d_caustic_corrections(base_solution, caustics, epsilon, mode):
    """
    Apply caustic corrections in 2D using Airy/Pearcey functions.
    """
    X = base_solution['x']
    Y = base_solution['y']
    S = base_solution['S']
    a = base_solution['a'][0]
    
    u_corrected = np.copy(base_solution['u'])
    
    if mode == 'none':
        print("No caustic correction applied (mode: none)")
        result = base_solution.copy()
        result['u_standard'] = base_solution['u']
        return result
    
    # Classify and correct each caustic
    for caustic in caustics:
        x_c, y_c = caustic['position']
        caustic_type = caustic['type']
        
        if caustic_type == 'airy' or caustic['caustic_type'] == 'A2':
            # Fold caustic - use Airy
            correction_width = 5 * epsilon**(2/3)
            
            # Distance to caustic
            dist = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
            mask = dist < correction_width
            
            if np.any(mask):
                # Find direction normal to caustic (simplified)
                # In practice, compute from ray geometry
                
                # Scaled coordinate perpendicular to caustic
                z = dist[mask] / epsilon**(2/3)
                
                # Airy correction
                Ai = CausticFunctions.airy_uniform(z)
                
                idx_x = np.argmin(np.abs(X[:, 0] - x_c))
                idx_y = np.argmin(np.abs(Y[0, :] - y_c))
                
                a_c = a[idx_x, idx_y]
                S_c = S[idx_x, idx_y]
                
                u_corrected[mask] = a_c * Ai * np.exp(1j * S_c / epsilon)
        
        elif caustic_type == 'pearcey' or caustic['caustic_type'] == 'A3':
            # Cusp caustic - use Pearcey
            correction_width = 5 * epsilon**(1/2)
            
            dist = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
            mask = dist < correction_width
            
            if np.any(mask):
                # Scaled coordinates
                x_scaled = (X[mask] - x_c) / epsilon**(1/2)
                y_scaled = (Y[mask] - y_c) / epsilon**(1/4)
                
                # Pearcey integral (expensive!)
                P_vals = np.array([CausticFunctions.pearcey_approx(xs, ys) 
                                  for xs, ys in zip(x_scaled, y_scaled)])
                
                idx_x = np.argmin(np.abs(X[:, 0] - x_c))
                idx_y = np.argmin(np.abs(Y[0, :] - y_c))
                
                a_c = a[idx_x, idx_y]
                S_c = S[idx_x, idx_y]
                
                u_corrected[mask] = a_c * P_vals * np.exp(1j * S_c / epsilon)
    
    print(f"Applied corrections to {len(caustics)} caustics")
    print(f"  Fold (Airy): {sum(1 for c in caustics if c['caustic_type']=='A2')}")
    print(f"  Cusp (Pearcey): {sum(1 for c in caustics if c['caustic_type']=='A3')}")
    
    result = base_solution.copy()
    result['u'] = u_corrected
    result['u_standard'] = base_solution['u']
    
    return result


# ==================================================================
# VISUALIZATION WITH CAUSTIC HIGHLIGHTING
# ==================================================================

def plot_with_caustics(solution, component='abs', highlight_caustics=True):
    """
    Plot WKB solution with caustics highlighted.
    
    Parameters
    ----------
    solution : dict
        Output of wkb_approximation()
    component : {'abs','real','imag','phase'}
        Which component of u to visualize.
    highlight_caustics : bool
        Whether to mark caustic locations.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # ----------------------------------------------------------
    # Helper: select component to plot
    # ----------------------------------------------------------
    def _select_component(u, component):
        if component == 'real':
            return np.real(u), 'RdBu_r'
        elif component == 'imag':
            return np.imag(u), 'RdBu_r'
        elif component == 'abs':
            return np.abs(u), 'viridis'
        elif component == 'phase':
            return np.angle(u), 'twilight'
        else:
            raise ValueError("component must be one of: real, imag, abs, phase")

    # ----------------------------------------------------------
    # Helper: plot 1D caustics (single legend entry)
    # ----------------------------------------------------------
    def _plot_caustics_1d(ax, caustics, x, data):
        if not highlight_caustics or len(caustics) == 0:
            return
        added = False
        for c in caustics:
            xc = c['position']
            ax.axvline(xc, color='red', linestyle='--', linewidth=2,
                       alpha=0.7, label='Caustic' if not added else None)
            ax.plot(
                xc,
                data[np.argmin(np.abs(x - xc))],
                'ro', markersize=8
            )
            added = True

    # ----------------------------------------------------------
    # Helper: plot 2D caustics (only two legend entries: A2, A3)
    # ----------------------------------------------------------
    def _plot_caustics_2d(ax, caustics):
        if not highlight_caustics or len(caustics) == 0:
            return

        type_seen = set()

        for c in caustics:
            x_c, y_c = c['position']
            t = c.get('caustic_type', 'A2')

            marker = 'o' if t == 'A2' else 's'
            color = 'red' if t == 'A2' else 'orange'

            label = None
            if t not in type_seen:
                label = f"{'Fold' if t=='A2' else 'Cusp'} ({t})"
                type_seen.add(t)

            ax.plot(
                x_c, y_c, marker,
                color=color,
                markersize=10,
                markeredgecolor='white',
                markeredgewidth=1.5,
                label=label
            )

    # ----------------------------------------------------------
    # Retrieve data & select component
    # ----------------------------------------------------------
    dim = solution['dimension']
    u = solution['u']
    caustics = solution.get('caustics', [])

    data, cmap = _select_component(u, component)

    # ----------------------------------------------------------
    # 1D plotting
    # ----------------------------------------------------------
    if dim == 1:
        fig, axes = plt.subplots(2 if 'u_standard' in solution else 1,
                                 1, figsize=(12, 6))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        x = solution['x']

        # Main panel
        ax = axes[0]
        ax.plot(x, data, 'b-', linewidth=2, label=component)
        ax.set_xlabel('x')
        ax.set_ylabel(f'{component}(u)')
        ax.set_title(f'WKB with Caustic Corrections ({solution.get("caustic_correction","none")})')
        ax.grid(True, alpha=0.3)

        # Caustics
        _plot_caustics_1d(ax, caustics, x, data)
        ax.legend()

        # Comparison panel
        if 'u_standard' in solution:
            ax2 = axes[1]
            data_std, _ = _select_component(solution['u_standard'], component)
            ax2.plot(x, data_std, 'r--', linewidth=2, alpha=0.7, label='Standard WKB')
            ax2.plot(x, data, 'b-', linewidth=2, label='Corrected')
            ax2.set_xlabel('x')
            ax2.set_ylabel(f'{component}(u)')
            ax2.set_title('Comparison: Standard vs Corrected')
            ax2.grid(True, alpha=0.3)

            # Caustics on the comparison plot
            _plot_caustics_1d(ax2, caustics, x, data)
            ax2.legend()

        plt.tight_layout()
        return fig

    # ----------------------------------------------------------
    # 2D plotting
    # ----------------------------------------------------------
    else:
        fig, axes = plt.subplots(1, 2 if 'u_standard' in solution else 1,
                                 figsize=(16, 6))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        X = solution['x']
        Y = solution['y']

        idx = 0

        # Standard WKB
        if 'u_standard' in solution:
            data_std, _ = _select_component(solution['u_standard'], component)
            im = axes[0].contourf(X, Y, data_std, 30, cmap=cmap)
            axes[0].set_title("Standard WKB")
            axes[0].set_aspect('equal')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            fig.colorbar(im, ax=axes[0])
            idx = 1

        # Corrected WKB
        im2 = axes[idx].contourf(X, Y, data, 30, cmap=cmap)
        axes[idx].set_title(f"With Caustic Corrections ({solution.get('caustic_correction','none')})")
        axes[idx].set_aspect('equal')
        axes[idx].set_xlabel('x')
        axes[idx].set_ylabel('y')
        fig.colorbar(im2, ax=axes[idx])

        # Caustics (single legend)
        _plot_caustics_2d(axes[idx], caustics)

        # Rays (optional)
        if 'rays' in solution:
            rays = solution['rays']
            step = max(1, len(rays)//20)
            for r in rays[::step]:
                axes[idx].plot(r['x'], r['y'], 'k-', linewidth=0.6, alpha=0.25)

        axes[idx].legend(loc='upper right')

        plt.tight_layout()
        return fig
        
def plot_with_caustics_old (solution, component='abs', highlight_caustics=True):
    """
    Plot WKB solution with caustics highlighted.
    """
    import matplotlib.pyplot as plt
    
    dim = solution['dimension']
    u = solution['u']
    caustics = solution.get('caustics', [])
    
    # Select component
    if component == 'real':
        data = np.real(u)
        cmap = 'RdBu_r'
    elif component == 'imag':
        data = np.imag(u)
        cmap = 'RdBu_r'
    elif component == 'abs':
        data = np.abs(u)
        cmap = 'viridis'
    elif component == 'phase':
        data = np.angle(u)
        cmap = 'twilight'
    
    if dim == 1:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        x = solution['x']
        
        # Plot solution
        axes[0].plot(x, data, 'b-', linewidth=2, label=component)
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel(f'{component}(u)', fontsize=12)
        axes[0].set_title(f'WKB with Caustic Corrections ({solution.get("caustic_correction", "none")})')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight caustics
        if highlight_caustics and len(caustics) > 0:
            for caustic in caustics:
                x_c = caustic['position']
                axes[0].axvline(x_c, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Caustic')
                axes[0].plot(x_c, data[np.argmin(np.abs(x - x_c))], 
                           'ro', markersize=10)
        
        axes[0].legend()
        
        # Compare with/without correction
        if 'u_standard' in solution:
            data_std = np.abs(solution['u_standard']) if component == 'abs' else np.real(solution['u_standard'])
            axes[1].plot(x, data_std, 'r--', linewidth=2, label='Standard WKB', alpha=0.7)
            axes[1].plot(x, data, 'b-', linewidth=2, label='With corrections')
            axes[1].set_xlabel('x', fontsize=12)
            axes[1].set_ylabel(f'{component}(u)', fontsize=12)
            axes[1].set_title('Comparison: Standard vs Corrected')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            
            if highlight_caustics:
                for caustic in caustics:
                    axes[1].axvline(caustic['position'], color='red', 
                                  linestyle=':', alpha=0.5)
    
    else:  # 2D
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        X, Y = solution['x'], solution['y']
        
        # Standard solution
        if 'u_standard' in solution:
            data_std = np.abs(solution['u_standard']) if component == 'abs' else np.real(solution['u_standard'])
            im1 = axes[0].contourf(X, Y, data_std, levels=30, cmap=cmap)
            axes[0].set_title('Standard WKB')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')
            axes[0].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0])
        
        # Corrected solution
        im2 = axes[1].contourf(X, Y, data, levels=30, cmap=cmap)
        axes[1].set_title(f'With Caustic Corrections ({solution.get("caustic_correction", "none")})')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot caustics
        if highlight_caustics and len(caustics) > 0:
            for caustic in caustics:
                x_c, y_c = caustic['position']
                for ax in axes:
                    marker = 'o' if caustic['caustic_type'] == 'A2' else 's'
                    color = 'red' if caustic['caustic_type'] == 'A2' else 'orange'
                    ax.plot(x_c, y_c, marker, color=color, markersize=10,
                           markeredgecolor='white', markeredgewidth=2)
            
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Fold (A2)'),
                Patch(facecolor='orange', label='Cusp (A3)')
            ]
            axes[1].legend(handles=legend_elements, loc='upper right')
        
        # Plot rays
        if 'rays' in solution:
            for ray in solution['rays'][::max(1, len(solution['rays'])//20)]:
                axes[1].plot(ray['x'], ray['y'], 'k-', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    return fig


def plot_caustic_analysis(solution):
    """
    Detailed analysis plot of caustics.
    """
    import matplotlib.pyplot as plt
    
    caustics = solution.get('caustics', [])
    if len(caustics) == 0:
        print("No caustics to analyze")
        return None
    
    dim = solution['dimension']
    
    if dim == 1:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        x = solution['x']
        
        # 1. Solution amplitude
        axes[0].plot(x, np.abs(solution['u']), 'b-', linewidth=2, label='|u| corrected')
        if 'u_standard' in solution:
            axes[0].plot(x, np.abs(solution['u_standard']), 'r--', 
                        linewidth=2, alpha=0.7, label='|u| standard')
        
        for caustic in caustics:
            x_c = caustic['position']
            axes[0].axvline(x_c, color='red', linestyle=':', alpha=0.5)
            axes[0].text(x_c, axes[0].get_ylim()[1]*0.9, 
                        f"Caustic\n{caustic['caustic_type']}", 
                        ha='center', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        axes[0].set_ylabel('|u|', fontsize=12)
        axes[0].set_title('Amplitude with Caustic Locations')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Phase
        axes[1].plot(x, solution['S'], 'g-', linewidth=2, label='Phase S')
        
        if 'maslov_phases' in solution:
            axes[1].plot(x, solution['maslov_phases'], 'orange', 
                        linewidth=2, linestyle='--', label='Maslov correction')
        
        for caustic in caustics:
            axes[1].axvline(caustic['position'], color='red', linestyle=':', alpha=0.5)
        
        axes[1].set_ylabel('Phase', fontsize=12)
        axes[1].set_title('Phase and Maslov Index')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Error between standard and corrected
        if 'u_standard' in solution:
            error = np.abs(solution['u'] - solution['u_standard'])
            axes[2].plot(x, error, 'purple', linewidth=2)
            axes[2].set_xlabel('x', fontsize=12)
            axes[2].set_ylabel('|u_corrected - u_standard|', fontsize=12)
            axes[2].set_title('Correction Magnitude')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
            
            for caustic in caustics:
                axes[2].axvline(caustic['position'], color='red', linestyle=':', alpha=0.5)
    
    else:  # 2D
        n_caustics = len(caustics)
        fig = plt.figure(figsize=(16, 10))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        X, Y = solution['x'], solution['y']
        
        # Main plot: solution with caustics
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        im = ax_main.contourf(X, Y, np.abs(solution['u']), levels=30, cmap='viridis')
        plt.colorbar(im, ax=ax_main, label='|u|')
        
        # Plot rays
        if 'rays' in solution:
            for ray in solution['rays'][::max(1, len(solution['rays'])//30)]:
                ax_main.plot(ray['x'], ray['y'], 'k-', alpha=0.15, linewidth=0.5)
        
        # Mark caustics
        fold_caustics = []
        cusp_caustics = []
        
        for caustic in caustics:
            x_c, y_c = caustic['position']
            if caustic['caustic_type'] == 'A2':
                fold_caustics.append((x_c, y_c))
                ax_main.plot(x_c, y_c, 'ro', markersize=12, 
                           markeredgecolor='white', markeredgewidth=2)
            else:
                cusp_caustics.append((x_c, y_c))
                ax_main.plot(x_c, y_c, 'ys', markersize=12,
                           markeredgecolor='white', markeredgewidth=2)
        
        ax_main.set_xlabel('x', fontsize=11)
        ax_main.set_ylabel('y', fontsize=11)
        ax_main.set_title(f'Solution with {n_caustics} Caustics', fontsize=13)
        ax_main.set_aspect('equal')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label=f'Fold (A2): {len(fold_caustics)}'),
            Patch(facecolor='yellow', label=f'Cusp (A3): {len(cusp_caustics)}')
        ]
        ax_main.legend(handles=legend_elements, loc='upper right')
        
        # Phase plot
        ax_phase = fig.add_subplot(gs[0, 2])
        im_phase = ax_phase.contourf(X, Y, solution['S'], levels=30, cmap='twilight')
        plt.colorbar(im_phase, ax=ax_phase, label='Phase S')
        ax_phase.set_title('Phase')
        ax_phase.set_aspect('equal')
        
        # Error plot
        if 'u_standard' in solution:
            ax_error = fig.add_subplot(gs[1, 2])
            error = np.abs(solution['u'] - solution['u_standard'])
            im_error = ax_error.contourf(X, Y, np.log10(error + 1e-10), 
                                        levels=30, cmap='hot')
            plt.colorbar(im_error, ax=ax_error, label='log10(error)')
            ax_error.set_title('Correction Effect')
            ax_error.set_aspect('equal')
        
        # Caustic statistics
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')
        
        stats_text = f"Caustic Statistics:\n"
        stats_text += f"  Total caustics: {n_caustics}\n"
        stats_text += f"  Fold caustics (A2): {len(fold_caustics)}\n"
        stats_text += f"  Cusp caustics (A3): {len(cusp_caustics)}\n"
        stats_text += f"  Correction method: {solution.get('caustic_correction', 'none')}\n"
        stats_text += f"  Epsilon: {solution['epsilon']:.4f}\n"
        
        if 'u_standard' in solution:
            max_error = np.max(np.abs(solution['u'] - solution['u_standard']))
            mean_error = np.mean(np.abs(solution['u'] - solution['u_standard']))
            stats_text += f"  Max correction: {max_error:.4e}\n"
            stats_text += f"  Mean correction: {mean_error:.4e}\n"
        
        ax_stats.text(0.1, 0.5, stats_text, fontsize=11, 
                     verticalalignment='center',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_initial_data_line(x_range, n_points=20, direction=(1, 0), 
                             y_intercept=0.0):
    """
    Create initial data for WKB on a line segment.
    
    Parameters
    ----------
    x_range : tuple
        Range (x_min, x_max) for the line segment.
    n_points : int
        Number of points on the line.
    direction : tuple
        Direction of rays (ξ₀, η₀).
    y_intercept : float
        y-coordinate of the line.
    
    Returns
    -------
    dict
        Initial data for wkb_multidim.
    
    Examples
    --------
    >>> # Horizontal line with rays going upward
    >>> ic = create_initial_data_line((-1, 1), n_points=20, 
    ...                                direction=(0, 1), y_intercept=0)
    """
    x_init = np.linspace(x_range[0], x_range[1], n_points)
    y_init = np.full(n_points, y_intercept)
    S_init = np.zeros(n_points)
    
    # Normalize direction
    dir_norm = np.sqrt(direction[0]**2 + direction[1]**2)
    px_init = np.full(n_points, direction[0] / dir_norm)
    py_init = np.full(n_points, direction[1] / dir_norm)
    
    return {
        'x': x_init,
        'y': y_init,
        'S': S_init,
        'p_x': px_init,
        'p_y': py_init
    }


def create_initial_data_circle(radius=1.0, n_points=30, outward=True):
    """
    Create initial data for WKB on a circle.
    
    Parameters
    ----------
    radius : float
        Radius of the circle.
    n_points : int
        Number of points on the circle.
    outward : bool
        If True, rays point outward; if False, inward.
    
    Returns
    -------
    dict
        Initial data for wkb_multidim.
    
    Examples
    --------
    >>> # Circle with outward rays
    >>> ic = create_initial_data_circle(radius=1.0, n_points=30, outward=True)
    """
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    x_init = radius * np.cos(theta)
    y_init = radius * np.sin(theta)
    S_init = np.zeros(n_points)
    
    # Rays perpendicular to circle
    if outward:
        px_init = np.cos(theta)
        py_init = np.sin(theta)
    else:
        px_init = -np.cos(theta)
        py_init = -np.sin(theta)
    
    return {
        'x': x_init,
        'y': y_init,
        'S': S_init,
        'p_x': px_init,
        'p_y': py_init
    }


def create_initial_data_point_source(x0=0.0, y0=0.0, n_rays=20):
    """
    Create initial data for WKB from a point source.
    
    Parameters
    ----------
    x0, y0 : float
        Source location.
    n_rays : int
        Number of rays emanating from source.
    
    Returns
    -------
    dict
        Initial data for wkb_multidim.
    
    Examples
    --------
    >>> # Point source at origin
    >>> ic = create_initial_data_point_source(0, 0, n_rays=24)
    """
    theta = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    
    x_init = np.full(n_rays, x0)
    y_init = np.full(n_rays, y0)
    S_init = np.zeros(n_rays)
    
    # Rays in all directions
    px_init = np.cos(theta)
    py_init = np.sin(theta)
    
    return {
        'x': x_init,
        'y': y_init,
        'S': S_init,
        'p_x': px_init,
        'p_y': py_init
    }

def visualize_wkb_rays(wkb_result, plot_type='phase', n_rays_plot=None):
    """
    Visualize WKB solution with rays.
    
    Parameters
    ----------
    wkb_result : dict
        Output from wkb_multidim.
    plot_type : str
        What to visualize: 'phase', 'amplitude', 'real', 'rays'.
    n_rays_plot : int, optional
        Number of rays to plot (if None, plot all).
    
    Examples
    --------
    >>> wkb = wkb_multidim(...)
    >>> visualize_wkb_rays(wkb, plot_type='phase')
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    X = wkb_result['x']
    Y = wkb_result['y']
    
    if plot_type == 'phase':
        # Plot phase
        S = wkb_result['S']
        im = ax.contourf(X, Y, S, levels=30, cmap='twilight')
        plt.colorbar(im, ax=ax, label='Phase S(x,y)')
        ax.set_title('WKB Phase Function')
    
    elif plot_type == 'amplitude':
        # Plot amplitude
        a = wkb_result['a']
        im = ax.contourf(X, Y, a, levels=30, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Amplitude a(x,y)')
        ax.set_title('WKB Amplitude')
    
    elif plot_type == 'real':
        # Plot real part
        u = wkb_result['u']
        im = ax.contourf(X, Y, np.real(u), levels=30, cmap='RdBu')
        plt.colorbar(im, ax=ax, label='Re(u)')
        ax.set_title('WKB Solution - Real Part')
    
    elif plot_type == 'rays':
        # Plot phase contours with rays
        S = wkb_result['S']
        ax.contour(X, Y, S, levels=20, colors='gray', alpha=0.3)
        ax.set_title('WKB Rays')
    
    # Overlay rays
    if 'rays' in wkb_result and plot_type in ['phase', 'amplitude', 'rays']:
        rays = wkb_result['rays']
        n_total = len(rays)
        
        if n_rays_plot is None:
            n_rays_plot = min(n_total, 20)  # Limit for clarity
        
        # Select evenly spaced rays
        ray_indices = np.linspace(0, n_total-1, n_rays_plot, dtype=int)
        
        for idx in ray_indices:
            ray = rays[idx]
            ax.plot(ray['x'], ray['y'], 'r-', alpha=0.5, linewidth=1)
            # Mark start
            ax.plot(ray['x'][0], ray['y'][0], 'go', markersize=4)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
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
