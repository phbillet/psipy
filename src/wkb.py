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
wkb.py — Multidimensional WKB approximation with caustic corrections
==========================================================================

Overview
--------
The `wkb` module provides a comprehensive implementation of the Wentzel–Kramers–Brillouin (WKB) method for constructing asymptotic solutions to linear partial differential equations of the form

    P(x, –iε∇) u(x) = 0,  ε → 0,

where `P(x,ξ)` is the (pseudo‑differential) symbol of the operator.  The solution is sought as a sum over rays (bicharacteristics):

    u(x) ≈ ∑_j A_j(x) e^{i S_j(x)/ε},

with the phase `S` satisfying the eikonal equation `P(x,∇S)=0` and the amplitudes `A_j` determined by transport equations along the rays.

Key features:

* **Automatic dimension detection** – works for both 1D and 2D problems without changing the calling interface.
* **Ray tracing** – integrates Hamilton’s equations for the bicharacteristics, including the variational equations for the stability matrix `J` (used to detect caustics).
* **Multi‑order amplitude transport** – computes amplitudes up to arbitrary order (0,1,2,3,…) by solving ODEs along each ray.
* **Caustic detection and correction** – monitors `det(J)` to locate caustics (folds, cusps) and applies:
    * Maslov phase shifts (π/2 per caustic),
    * Uniform Airy (fold) or Pearcey (cusp) corrections near caustics.
* **Interpolation onto regular grids** – uses `scipy.interpolate` (linear, `griddata`) to map the ray‑based solution to a uniform spatial grid.
* **Rich visualisation suite** – phase portraits, amplitude decomposition, caustic analysis, ray plots, comparison of WKB orders.
* **Utilities for generating initial data** – line segments, circles, point sources.

Mathematical background
-----------------------
The WKB ansatz `u = e^{iS/ε} (a₀ + ε a₁ + ε² a₂ + …)` is inserted into the equation `P(x,–iε∇)u = 0`.  Expanding in powers of ε yields:

* **Order ε⁰ (eikonal equation):**  `P(x, ∇S) = 0`.  This is a Hamilton–Jacobi equation solved by the method of characteristics (rays).  The rays satisfy

        dx/dt = ∂P/∂ξ,  dξ/dt = –∂P/∂x,  dS/dt = ξ·∂P/∂ξ – P.

* **Order ε¹ (transport equation for a₀):**  `(∂P/∂ξ)·∇a₀ + ½ (∇_ξ·∇_x P) a₀ = 0`.  Along a ray this becomes an ODE for the amplitude.

* **Higher orders:**  Similar ODEs for `a₁, a₂, …` involving derivatives of `P` up to order three.

The **stability matrix** `J = ∂x(t)/∂q` (where `q` parametrises the initial data) measures the focusing of nearby rays.  Its determinant vanishes at **caustics**.  Near a fold caustic the standard WKB amplitude blows up; the correct uniform approximation involves the Airy function.  Near a cusp, the Pearcey integral is required.  The module implements both corrections using the companion `caustics` module.


References
----------
.. [1] Maslov, V. P. & Fedoriuk, M. V.  *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
.. [2] Duistermaat, J. J.  “Oscillatory integrals, Lagrange immersions and unfolding of singularities”, *Comm. Pure Appl. Math.* 27, 207–281, 1974.
.. [3] Berry, M. V. & Howls, C. J.  “High orders of the Weyl expansion for quantum billiards”, *Phys. Rev. E* 50(5), 3577–3595, 1994.
.. [4] Ludwig, D.  “Uniform asymptotic expansions at a caustic”, *Comm. Pure Appl. Math.* 19, 215–250, 1966.
.. [5] Kravtsov, Yu. A. & Orlov, Yu. I.  *Caustics, Catastrophes and Wave Fields*, Springer, 1999.
"""

from imports import *
from caustics import * 
# ==================================================================
# ENHANCED WKB WITH CAUSTIC CORRECTIONS
# ==================================================================

def wkb_approximation(symbol, initial_phase, order=1, domain=None,
                               resolution=50, epsilon=0.1, dimension=None,
                               caustic_correction='auto', caustic_threshold=1e-3):
    """
    Compute multidimensional WKB approximation (1D or 2D).

    u(x) ≈ exp(iS/ε) · [a₀ + ε·a₁ + ε²·a₂ + ...]
    
    Automatically detects dimension from initial_phase or uses dimension parameter.
    
    Parameters  
    ----------  
    symbol : sympy expression  
        Principal symbol p(x, ξ) for 1D or p(x, y, ξ, η) for 2D.
    initial_phase : dict  
        Initial data on a curve/point:
        
        1D: Keys 'x', 'S', 'p_x', optionally 'a' (dict or array)
        2D: Keys 'x', 'y', 'S', 'p_x', 'p_y', optionally 'a' (dict or array)
        
    order : int  
        WKB order (0, 1, 2, or 3).
    domain : tuple or None
        1D: (x_min, x_max)
        2D: ((x_min, x_max), (y_min, y_max))
        If None, inferred from initial data.
    resolution : int or tuple
        Grid resolution (single int or (nx, ny) for 2D).
    epsilon : float
        Small parameter for asymptotic expansion.
    dimension : int or None
        Force dimension (1 or 2). If None, auto-detect.
      
    Returns  
    -------  
    dict  
        WKB solution with keys adapted to dimension.
        
    Examples
    --------
    >>> from sympy import symbols, sqrt
    >>> 
    >>> # 1D harmonic oscillator
    >>> x, xi = symbols('x xi', real=True)
    >>> symbol = xi**2 + x**2  # p(x,ξ) = ξ² + x²
    >>> 
    >>> # Initial conditions: Gaussian wave packet
    >>> n_rays = 20
    >>> x0 = np.linspace(-2, 2, n_rays)
    >>> initial = {
    ...     'x': x0,
    ...     'p_x': np.ones(n_rays),  # momentum = 1
    ...     'S': 0.5 * x0**2,         # phase
    ...     'a': np.exp(-x0**2)       # Gaussian amplitude
    ... }
    >>> 
    >>> result = wkb_approximation(
    ...     symbol, initial, order=2, epsilon=0.05, resolution=100
    ... )
    >>> 
    >>> # Extract solution
    >>> x_grid = result['x']
    >>> u = result['u']
    >>> plt.plot(x_grid, np.abs(u))
    
    >>> # 2D wave equation
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> c = 1.0  # wave speed
    >>> symbol = xi**2 + eta**2 - c**2
    >>> 
    >>> # Point source initial conditions
    >>> theta = np.linspace(0, 2*np.pi, 30, endpoint=False)
    >>> r0 = 0.5
    >>> initial = {
    ...     'x': r0 * np.cos(theta),
    ...     'y': r0 * np.sin(theta),
    ...     'p_x': np.cos(theta),
    ...     'p_y': np.sin(theta),
    ...     'S': np.zeros(30),
    ...     'a': np.ones(30)
    ... }
    >>> 
    >>> result = wkb_approximation(
    ...     symbol, initial, order=1, epsilon=0.1, resolution=(80, 80)
    ... )
    >>> 
    >>> # Visualize 2D solution
    >>> X, Y = result['x'], result['y']
    >>> U = result['u']
    >>> plt.pcolormesh(X, Y, np.abs(U))
    """  
    base_solution = _compute_base_wkb(symbol, initial_phase, order, domain,
                                      resolution, epsilon, dimension)
    
    # Detect caustics
    detector = RayCausticDetector(base_solution['rays'], base_solution['dimension'], det_threshold=caustic_threshold)
    caustics = detector.detect()
    
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

def _compute_base_wkb(symbol, initial_phase, order=1, domain=None,
                resolution=50, epsilon=0.1, dimension=None):
    """
    Compute multidimensional WKB (Wentzel-Kramers-Brillouin) approximation for wave propagation.
    
    Constructs an asymptotic solution to a pseudo-differential equation using the WKB method,
    which represents the solution as an oscillatory phase multiplied by a slowly varying amplitude:
    
        u(x, ε) = exp(iS(x)/ε) · Σₖ₌₀ⁿ εᵏ aₖ(x)
    
    where S(x) is the eikonal (phase function), aₖ(x) are transport amplitudes, and ε is a 
    small parameter representing the inverse wavelength. The method traces bicharacteristic rays
    through phase space using Hamilton's equations and solves transport equations along these
    rays to determine amplitude evolution.
    
    This implementation supports both 1D and 2D spatial domains and can compute multi-order
    corrections (order 0 through 3+) to improve accuracy beyond the standard semiclassical limit.
    
    The stability matrix J(t) is co-integrated alongside each ray via:
    
        dJ/dt = H_px · J,   J(0) = I
    
    where H_px[i,j] = ∂²p/∂ξᵢ∂xⱼ is the mixed Hessian of the Hamiltonian.
    det(J) → 0 signals a caustic; the ray dict exposes J11 (..J22 in 2D) for
    downstream use by RayCausticDetector.

    Mathematical Framework
    ----------------------
    Given a symbol p(x, ξ) defining the pseudo-differential operator, the WKB method solves:
    
    1. **Eikonal equation** (determines phase):
       p(x, ∇S(x)) = 0
       
    2. **Transport equations** (determine amplitudes):
       ∇ₓp·∇a₀ + (1/2)a₀∇ξ·∇ₓp = 0  (order 0)
       [Higher-order corrections for k ≥ 1]
    
    The rays are computed via Hamilton's equations:
       dx/dt = ∂p/∂ξ,    dξ/dt = -∂p/∂x
       dS/dt = ξ·∂p/∂ξ - p
    
    Parameters
    ----------
    symbol : sympy.Expr
        Symbolic expression for the principal symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
        Must be written in terms of SymPy symbols and represent the dispersion relation
        of the wave equation. Common examples:
        - Schrödinger: ξ² + V(x)
        - Wave equation: ξ² + η² - ω²
        - Helmholtz: ξ² - k²(x)
        
    initial_phase : dict
        Initial conditions for ray tracing at t=0. Must contain:
        
        **Required keys (1D)**:
            - 'x' : array_like, shape (n_rays,)
                Initial spatial positions
            - 'p_x' : array_like, shape (n_rays,)
                Initial momenta (ξ values)
            - 'S' : array_like, shape (n_rays,)
                Initial phase values
                
        **Required keys (2D)**:
            - 'x', 'y' : array_like, shape (n_rays,)
                Initial spatial positions
            - 'p_x', 'p_y' : array_like, shape (n_rays,)
                Initial momenta (ξ, η values)
            - 'S' : array_like, shape (n_rays,)
                Initial phase values
                
        **Optional key**:
            - 'a' : array_like or dict
                Initial amplitude values. Can be:
                - Single array (n_rays,) → interpreted as a₀
                - Dict {0: a₀, 1: a₁, ...} → multi-order amplitudes
                If omitted, a₀ defaults to ones, higher orders to zeros.
    
    order : int, default=1
        Maximum order of the asymptotic expansion. Higher orders include progressively
        smaller corrections proportional to εᵏ:
        - 0: Leading-order WKB (geometric optics)
        - 1: First correction (includes caustic pre-factors)
        - 2: Second correction (third-derivative terms)
        - 3+: Higher corrections (simplified expressions)
        
    domain : tuple or None, default=None
        Spatial domain for the output grid:
        - 1D: (x_min, x_max)
        - 2D: ((x_min, x_max), (y_min, y_max))
        If None, automatically determined from ray extent with 10% margin.
        
    resolution : int or tuple, default=50
        Grid resolution for interpolated output:
        - int: Same resolution in all dimensions
        - tuple: (nx,) for 1D or (nx, ny) for 2D
        
    epsilon : float, default=0.1
        Small parameter ε representing the inverse wavelength or semiclassical parameter.
        The solution is accurate when ε → 0. Typical values: 0.01 to 0.5.
        
    dimension : int or None, default=None
        Spatial dimension (1 or 2). If None, automatically detected from initial_phase:
        - Presence of 'y' and 'p_y' keys → dimension = 2
        - Otherwise → dimension = 1
    
    Returns
    -------
    result : dict
        Dictionary containing the computed WKB solution and diagnostic information:
        
        **Solution fields**:
            - 'u' : ndarray, shape (nx,) or (nx, ny), complex
                Complete WKB approximation: u = exp(iS/ε) · Σₖ εᵏaₖ
            - 'S' : ndarray, shape (nx,) or (nx, ny), float
                Interpolated phase function (eikonal)
            - 'a' : dict of ndarrays
                Individual amplitude orders: {0: a₀, 1: a₁, ..., order: aₙ}
            - 'a_total' : ndarray, complex
                Total amplitude: Σₖ εᵏaₖ (without phase factor)
                
        **Grid information**:
            - 'x' : ndarray
                Spatial grid in x direction (1D: shape (nx,), 2D: shape (nx, ny))
            - 'y' : ndarray (2D only)
                Spatial grid in y direction, shape (nx, ny)
                
        **Ray tracing data**:
            - 'rays' : list of dict
                Each dict contains traced ray data with keys:
                't', 'x', ['y'], 'xi', ['eta'], 'S', 'a0', 'a1', ...,
                'J11' (all dims), 'J12', 'J21', 'J22' (2D only).
            - 'n_rays' : int
                Number of successfully traced rays
                
        **Metadata**:
            - 'dimension' : int
                Spatial dimension used (1 or 2)
            - 'order' : int
                Order of asymptotic expansion
            - 'epsilon' : float
                Small parameter value
            - 'domain' : tuple
                Spatial domain used for output grid
    
    Raises
    ------
    ValueError
        - If dimension is not 1 or 2
        - If required keys are missing from initial_phase
        - If array lengths are inconsistent (e.g., len(x) ≠ len(y) in 2D)
    RuntimeError
        - If all rays fail to integrate (numerical instability)
    
    Notes
    -----
    **Validity and Limitations**:
    
    1. **Small ε assumption**: The WKB method is asymptotic in ε → 0. Results become
       inaccurate for ε > 1 or near caustics where rays focus.
       
    2. **Caustics**: At caustics (where rays cross), det(J) → 0 and the amplitude formally
       diverges. The stability matrix J is tracked per-ray for caustic detection by
       RayCausticDetector.
       
    3. **Turning points**: Classical turning points (where p = 0) require connection
       formulas not implemented here. The method works best away from such points.
       
    4. **Interpolation artifacts**: The solution is interpolated from discrete rays
       onto a regular grid. Ensure sufficient ray density (n_rays) and resolution
       to avoid aliasing or gaps.
    
    **Numerical Implementation**:
    
    - Ray integration uses scipy.integrate.solve_ivp with RK45 (4th/5th order Runge-Kutta)
    - Default tolerances: rtol=1e-6, atol=1e-9
    - Integration time: t ∈ [0, 5] with 100 evaluation points per ray
    - 1D interpolation: linear (interp1d)
    - 2D interpolation: linear scattered data (griddata)
    
    **Computational Complexity**:
    
    - Ray tracing: O(n_rays × n_steps × n_derivatives)
    - Grid interpolation: O(n_rays × n_steps × resolution^dimension)
    - Memory: O(resolution^dimension) for output arrays
    
    References
    ----------
    .. [1] Maslov, V.P. and Fedoriuk, M.V. (1981). "Semi-Classical Approximation 
           in Quantum Mechanics". Springer.
    .. [2] Ralston, J. (1982). "Gaussian beams and the propagation of singularities". 
           Studies in PDE, MAA Studies in Mathematics, 23, 206-248.
    .. [3] Hörmander, L. (1985). "The Analysis of Linear Partial Differential Operators III".
           Springer-Verlag.
    
    See Also
    --------
    scipy.integrate.solve_ivp : ODE solver used for ray tracing
    scipy.interpolate.griddata : 2D interpolation method
    numpy.fft : For comparison with spectral methods
    """  
    from scipy.integrate import solve_ivp  
    from scipy.interpolate import griddata, interp1d
    
    # ==================================================================
    # DETECT DIMENSION
    # ==================================================================
    
    if dimension is None:
        # Auto-detect from initial_phase
        has_y = 'y' in initial_phase and 'p_y' in initial_phase
        dimension = 2 if has_y else 1
    
    if dimension not in [1, 2]:
        raise ValueError(f"Dimension must be 1 or 2, got {dimension}")
    
    print(f"WKB approximation in {dimension}D (order {order})")
    
    # ==================================================================
    # SETUP SYMBOLIC VARIABLES
    # ==================================================================
    
    if dimension == 1:
        x = symbols('x', real=True)
        xi = symbols('xi', real=True)
        spatial_vars = [x]
        momentum_vars = [xi]
        spatial_symbols = (x,)
        momentum_symbols = (xi,)
        all_vars = (x, xi)
    else:  # dimension == 2
        x, y = symbols('x y', real=True)
        xi, eta = symbols('xi eta', real=True)
        spatial_vars = [x, y]
        momentum_vars = [xi, eta]
        spatial_symbols = (x, y)
        momentum_symbols = (xi, eta)
        all_vars = (x, y, xi, eta)
    
    # ==================================================================
    # VALIDATE AND EXTRACT INITIAL DATA
    # ==================================================================
    
    required_keys_1d = ['x', 'S', 'p_x']
    required_keys_2d = ['x', 'y', 'S', 'p_x', 'p_y']
    required_keys = required_keys_2d if dimension == 2 else required_keys_1d
    
    if not all(k in initial_phase for k in required_keys):
        raise ValueError(f"initial_phase must contain: {required_keys}")
    
    # Extract spatial coordinates
    x_init = np.asarray(initial_phase['x'])
    n_rays = len(x_init)
    
    if dimension == 2:
        y_init = np.asarray(initial_phase['y'])
        if len(y_init) != n_rays:
            raise ValueError("x and y must have same length")
    
    # Extract phase and momentum
    S_init = np.asarray(initial_phase['S'])
    px_init = np.asarray(initial_phase['p_x'])
    
    if dimension == 2:
        py_init = np.asarray(initial_phase['p_y'])
    
    # Extract amplitudes for each order
    a_init = {}
    
    if 'a' in initial_phase:
        if isinstance(initial_phase['a'], dict):
            for k, v in initial_phase['a'].items():
                a_init[k] = np.asarray(v)
        else:
            a_init[0] = np.asarray(initial_phase['a'])
    else:
        a_init[0] = np.ones(n_rays)
    
    # Initialize missing orders to zero
    for k in range(order + 1):
        if k not in a_init:
            a_init[k] = np.zeros(n_rays)
    
    # ==================================================================
    # COMPUTE SYMBOLIC DERIVATIVES
    # ==================================================================
    
    print("Computing symbolic derivatives...")
    
    derivatives = {}
    
    # First derivatives (Hamilton equations)
    for i, mom_var in enumerate(momentum_vars):
        derivatives[f'dp_d{mom_var.name}'] = diff(symbol, mom_var)
    
    for i, space_var in enumerate(spatial_vars):
        derivatives[f'dp_d{space_var.name}'] = diff(symbol, space_var)
    
    # Second derivatives (transport equations)
    for mom_var in momentum_vars:
        derivatives[f'd2p_d{mom_var.name}2'] = diff(symbol, mom_var, 2)
    
    for space_var in spatial_vars:
        for mom_var in momentum_vars:
            derivatives[f'd2p_d{mom_var.name}d{space_var.name}'] = \
                diff(diff(symbol, mom_var), space_var)
    
    if len(momentum_vars) == 2:
        derivatives['d2p_dxideta'] = diff(diff(symbol, momentum_vars[0]), 
                                          momentum_vars[1])

    # ----------------------------------------------------------------
    # STABILITY MATRIX: variational equations of the Hamiltonian flow
    #
    # J = dx(t)/dq, where q is the parameter along the initial curve.
    # Its ODE comes from differentiating Hamilton's equations w.r.t. q:
    #
    #   1D:
    #     d/dt(dx/dq)  =  (d2p/dxi2)*dxi0/dq  +  (d2p/dxi/dx)*dx0/dq
    #     d/dt(dxi/dq) = -(d2p/dx2) *dx0/dq   -  (d2p/dx/dxi)*dxi0/dq
    #
    #   We track the POSITION component J = dx/dq only, initialised from
    #   the geometry of the initial data curve (see ray-loop below).
    #   The coupled variable K = dxi/dq is also integrated so that J
    #   can evolve correctly through the (d2p/dxi2)*K term.
    #
    # Required second derivatives:
    #   d2p/dxi2    (already present as d2p_dxi2)
    #   d2p/dxi/dx  (already present as d2p_dxidx)
    #   d2p/dx2     NEW
    # For 2D, the full 2x2 block equivalents are needed.
    # ----------------------------------------------------------------

    # Spatial second derivatives (variational equations)
    for space_var in spatial_vars:
        derivatives[f'd2p_d{space_var.name}2'] = diff(symbol, space_var, 2)

    if dimension == 2:
        derivatives['d2p_dxidy']  = diff(diff(symbol, xi),  y)   # d2p/dxi/dy
        derivatives['d2p_detadx'] = diff(diff(symbol, eta), x)   # d2p/deta/dx
        derivatives['d2p_dxy']    = diff(diff(symbol, x),   y)   # d2p/dx/dy
        # d2p_dxidx  -> d2p/dxi/dx  (already present)
        # d2p_detady -> d2p/deta/dy (already present)
    
    # Third derivatives (higher-order corrections)
    if order >= 2:
        for mom_var in momentum_vars:
            derivatives[f'd3p_d{mom_var.name}3'] = diff(symbol, mom_var, 3)
        
        if dimension == 2:
            derivatives['d3p_dxi2deta'] = diff(diff(symbol, xi, 2), eta)
            derivatives['d3p_dxideta2'] = diff(diff(symbol, xi), eta, 2)
            derivatives['d3p_dxi2dx'] = diff(diff(symbol, xi, 2), x)
            derivatives['d3p_deta2dy'] = diff(diff(symbol, eta, 2), y)
    
    # Lambdify all derivatives
    print(f"Lambdifying {len(derivatives)} derivatives...")
    funcs = {}
    for name, expr in derivatives.items():
        funcs[name] = lambdify(all_vars, expr, 'numpy')
    
    # Principal symbol
    funcs['p'] = lambdify(all_vars, symbol, 'numpy')
    
    # ==================================================================
    # HELPER FUNCTIONS FOR DERIVATIVES EVALUATION
    # ==================================================================
    
    def eval_func(name, *args):
        """Safely evaluate a function, handling dimension differences."""
        if name in funcs:
            return funcs[name](*args)
        return 0.0
    
    def compute_geometric_spreading(*args):
        """
        Compute divergence of momentum gradient.
        1D: d²p/dξ²
        2D: d²p/dξ² + d²p/dη²
        """
        if dimension == 1:
            return eval_func('d2p_dxi2', *args)
        else:
            return (eval_func('d2p_dxi2', *args) + 
                   eval_func('d2p_deta2', *args))
    
    def compute_spatial_momentum_coupling(*args):
        """
        Compute ∇_x · ∇_ξ p
        1D: ∂²p/∂x∂ξ
        2D: ∂²p/∂x∂ξ + ∂²p/∂y∂η
        """
        if dimension == 1:
            return eval_func('d2p_dxidx', *args)
        else:
            return (eval_func('d2p_dxidx', *args) + 
                   eval_func('d2p_detady', *args))
    
    # ==================================================================
    # STATE VECTOR LAYOUT
    # ==================================================================
    #
    # 1D: [x, xi, S,  J11, K11,  a0, a1, ...]
    #      0   1   2   3    4    5   6
    #   J11 = dx/dq,   K11 = dxi/dq
    #
    # 2D: [x, y, xi, eta, S,  J11,J12,J21,J22,  K11,K12,K21,K22,  a0, a1, ...]
    #      0  1   2   3   4    5   6   7   8      9  10  11  12    13  14
    #   Jij = dxi(t)/dqj,   Kij = dxii(t)/dqj  (i=spatial, xi=momentum)
    # ==================================================================

    if dimension == 1:
        idx_x, idx_xi, idx_S = 0, 1, 2
        idx_J11 = 3
        idx_K11 = 4
        idx_a_start = 5
    else:
        idx_x, idx_y, idx_xi, idx_eta, idx_S = 0, 1, 2, 3, 4
        idx_J11, idx_J12, idx_J21, idx_J22 = 5, 6, 7, 8
        idx_K11, idx_K12, idx_K21, idx_K22 = 9, 10, 11, 12
        idx_a_start = 13

    idx_a = {k: idx_a_start + k for k in range(order + 1)}
    
    # ==================================================================
    # ==================================================================
    # RAY TRACING  —  vectorised batch integration
    # ==================================================================
    # Instead of n_rays separate solve_ivp calls, we pack ALL rays into a
    # single flat state vector and call solve_ivp ONCE.  The lambdified
    # functions already accept numpy arrays, so every derivative evaluation
    # operates on all rays simultaneously (numpy broadcasting).
    #
    # State: Z  shape (n_rays, ns)  where ns = components per ray.
    # Flat:  z_flat = Z.ravel(),  reconstructed as Z = z_flat.reshape(n_rays, ns).
    # On exit we slice Z back into per-ray dicts for the rest of the pipeline.
    # ==================================================================

    print(f"Ray tracing {n_rays} rays (vectorised batch)...")

    tmax = 5.0
    n_steps_per_ray = 100

    # ------------------------------------------------------------------
    # Build the initial batch state matrix Z0  (n_rays × ns)
    # ------------------------------------------------------------------
    if dimension == 1:
        ns = idx_a_start + order + 1
        Z0 = np.zeros((n_rays, ns))
        Z0[:, idx_x]   = x_init
        Z0[:, idx_xi]  = px_init
        Z0[:, idx_S]   = S_init
        Z0[:, idx_J11] = 1.0
        if n_rays > 1:
            dpx = np.empty(n_rays); dx_ = np.empty(n_rays)
            dpx[1:-1] = px_init[2:] - px_init[:-2]
            dx_[1:-1] = x_init[2:]  - x_init[:-2]
            dpx[0]  = px_init[1]  - px_init[0];  dx_[0]  = x_init[1]  - x_init[0]
            dpx[-1] = px_init[-1] - px_init[-2]; dx_[-1] = x_init[-1] - x_init[-2]
            with np.errstate(invalid='ignore', divide='ignore'):
                Z0[:, idx_K11] = np.where(dx_ != 0, dpx / dx_, 0.0)
        for k in range(order + 1):
            Z0[:, idx_a[k]] = a_init[k]
    else:
        ns = idx_a_start + order + 1
        Z0 = np.zeros((n_rays, ns))
        Z0[:, idx_x]   = x_init;  Z0[:, idx_y]   = y_init
        Z0[:, idx_xi]  = px_init; Z0[:, idx_eta] = py_init
        Z0[:, idx_S]   = S_init
        Z0[:, idx_J11] = 1.0;     Z0[:, idx_J22] = 1.0
        if n_rays > 1:
            dpx0 = np.empty(n_rays); dpy0 = np.empty(n_rays)
            dx0  = np.empty(n_rays); dy0  = np.empty(n_rays)
            for arr_in, arr_out in [(px_init, dpx0), (py_init, dpy0),
                                    (x_init, dx0),   (y_init, dy0)]:
                arr_out[1:-1] = arr_in[2:]  - arr_in[:-2]
                arr_out[0]    = arr_in[1]   - arr_in[0]
                arr_out[-1]   = arr_in[-1]  - arr_in[-2]
            arc = np.hypot(dx0, dy0)
            m = arc > 1e-14
            Z0[m, idx_K11] = dpx0[m] / arc[m]
            Z0[m, idx_K21] = dpy0[m] / arc[m]
        for k in range(order + 1):
            Z0[:, idx_a[k]] = a_init[k]

    # ------------------------------------------------------------------
    # Vectorised ODE right-hand side
    # ------------------------------------------------------------------
    def ray_ode_batch(t, z_flat):
        Z = z_flat.reshape(n_rays, ns)

        if dimension == 1:
            xv  = Z[:, idx_x];   xiv = Z[:, idx_xi]
            args_ = (xv, xiv)
            dxdt   =  eval_func('dp_dxi', *args_)
            dxidt  = -eval_func('dp_dx',  *args_)
            dSdt   = xiv * dxdt - eval_func('p', *args_)
            pxixi  = eval_func('d2p_dxi2',  *args_)
            pxxi   = eval_func('d2p_dxidx', *args_)
            pxx    = eval_func('d2p_dx2',   *args_)
            Jv = Z[:, idx_J11]; Kv = Z[:, idx_K11]
            dJdt =  pxixi*Kv + pxxi*Jv
            dKdt = -pxx*Jv   - pxxi*Kv
            geom = pxixi; coupling = pxxi
            dZ = np.empty_like(Z)
            dZ[:, idx_x]   = dxdt;  dZ[:, idx_xi]  = dxidt
            dZ[:, idx_S]   = dSdt
            dZ[:, idx_J11] = dJdt;  dZ[:, idx_K11] = dKdt
            a0v = Z[:, idx_a[0]]
            dZ[:, idx_a[0]] = -0.5*a0v*geom
            if order >= 1:
                a1v = Z[:, idx_a[1]]
                dZ[:, idx_a[1]] = -0.5*a1v*geom - 0.5*a0v*coupling
            if order >= 2:
                a2v = Z[:, idx_a[2]]
                d3  = eval_func('d3p_dxi3', *args_)
                dZ[:, idx_a[2]] = (-0.5*a2v*geom - 0.125*a0v*d3*dxidt
                                   - 0.25*Z[:, idx_a[1]]*coupling)
            if order >= 3:
                a3v = Z[:, idx_a[3]]
                d3  = eval_func('d3p_dxi3', *args_)
                dZ[:, idx_a[3]] = -0.5*a3v*geom - 0.1*Z[:, idx_a[1]]*d3*dxidt

        else:  # dimension == 2
            xv   = Z[:, idx_x];   yv   = Z[:, idx_y]
            xiv  = Z[:, idx_xi];  etav = Z[:, idx_eta]
            args_ = (xv, yv, xiv, etav)
            dxdt   =  eval_func('dp_dxi',  *args_)
            dydt   =  eval_func('dp_deta', *args_)
            dxidt  = -eval_func('dp_dx',   *args_)
            detadt = -eval_func('dp_dy',   *args_)
            dSdt   = xiv*dxdt + etav*dydt - eval_func('p', *args_)
            pxx    = eval_func('d2p_dx2',    *args_)
            pyy    = eval_func('d2p_dy2',    *args_)
            pxy    = eval_func('d2p_dxy',    *args_)
            pxix   = eval_func('d2p_dxidx',  *args_)
            pxiy   = eval_func('d2p_dxidy',  *args_)
            petax  = eval_func('d2p_detadx', *args_)
            petay  = eval_func('d2p_detady', *args_)
            pxixi    = eval_func('d2p_dxi2',     *args_)
            petapeta = eval_func('d2p_deta2',    *args_)
            pxipeta  = eval_func('d2p_dxideta',  *args_)
            J11=Z[:,idx_J11]; J12=Z[:,idx_J12]
            J21=Z[:,idx_J21]; J22=Z[:,idx_J22]
            K11=Z[:,idx_K11]; K12=Z[:,idx_K12]
            K21=Z[:,idx_K21]; K22=Z[:,idx_K22]
            dZ = np.empty_like(Z)
            dZ[:,idx_x]=dxdt; dZ[:,idx_y]=dydt
            dZ[:,idx_xi]=dxidt; dZ[:,idx_eta]=detadt; dZ[:,idx_S]=dSdt
            dZ[:,idx_J11] = pxixi*K11+pxipeta*K21+pxix*J11+pxiy*J21
            dZ[:,idx_J12] = pxixi*K12+pxipeta*K22+pxix*J12+pxiy*J22
            dZ[:,idx_J21] = pxipeta*K11+petapeta*K21+petax*J11+petay*J21
            dZ[:,idx_J22] = pxipeta*K12+petapeta*K22+petax*J12+petay*J22
            dZ[:,idx_K11] = -(pxx*J11+pxy*J21)-(pxix*K11+petax*K21)
            dZ[:,idx_K12] = -(pxx*J12+pxy*J22)-(pxix*K12+petax*K22)
            dZ[:,idx_K21] = -(pxy*J11+pyy*J21)-(pxiy*K11+petay*K21)
            dZ[:,idx_K22] = -(pxy*J12+pyy*J22)-(pxiy*K12+petay*K22)
            geom = pxixi + petapeta; coupling = pxix + petay
            a0v = Z[:, idx_a[0]]
            dZ[:, idx_a[0]] = -0.5*a0v*geom
            if order >= 1:
                a1v = Z[:, idx_a[1]]
                cross = eval_func('d2p_dxideta', *args_)
                dZ[:, idx_a[1]] = (-0.5*a1v*geom - 0.5*a0v*coupling
                                   - 0.25*a0v*cross*(dxidt+detadt))
            if order >= 2:
                a2v = Z[:, idx_a[2]]
                d3xi   = eval_func('d3p_dxi3',     *args_)
                d3eta  = eval_func('d3p_deta3',    *args_)
                d3mix1 = eval_func('d3p_dxi2deta', *args_)
                d3mix2 = eval_func('d3p_dxideta2', *args_)
                correction = (d3xi*dxidt + d3eta*detadt
                              + d3mix1*(dxidt+detadt) + d3mix2*(dxidt+detadt))
                dZ[:, idx_a[2]] = (-0.5*a2v*geom - 0.125*a0v*correction
                                   - 0.25*Z[:,idx_a[1]]*coupling)
            if order >= 3:
                a3v = Z[:, idx_a[3]]
                d3t = eval_func('d3p_dxi3',*args_)+eval_func('d3p_deta3',*args_)
                dZ[:, idx_a[3]] = (-0.5*a3v*geom
                                   - 0.1*Z[:,idx_a[1]]*d3t*(dxidt+detadt))
        return dZ.ravel()

    # ------------------------------------------------------------------
    # Single solve_ivp integrating all rays at once
    # ------------------------------------------------------------------
    sol = solve_ivp(
        ray_ode_batch,
        (0, tmax),
        Z0.ravel(),
        method='RK45',
        t_eval=np.linspace(0, tmax, n_steps_per_ray),
        rtol=1e-6,
        atol=1e-9
    )

    if not sol.success:
        print(f"Warning: batch ray integration: {sol.message}")

    # Unpack: sol.y has shape (n_rays*ns, n_steps)
    Y = sol.y.reshape(n_rays, ns, -1)   # (n_rays, ns, n_steps)

    rays = []
    for i in range(n_rays):
        rd = {'t': sol.t}
        if dimension == 1:
            rd['x']   = Y[i, idx_x,  :]
            rd['xi']  = Y[i, idx_xi, :]
            rd['S']   = Y[i, idx_S,  :]
            J11 = Y[i, idx_J11, :]
            rd['J11'] = J11; rd['J'] = J11
        else:
            rd['x']   = Y[i, idx_x,   :]; rd['y']   = Y[i, idx_y,   :]
            rd['xi']  = Y[i, idx_xi,  :]; rd['eta'] = Y[i, idx_eta, :]
            rd['S']   = Y[i, idx_S,   :]
            J11=Y[i,idx_J11,:]; J12=Y[i,idx_J12,:]
            J21=Y[i,idx_J21,:]; J22=Y[i,idx_J22,:]
            rd['J11']=J11; rd['J12']=J12; rd['J21']=J21; rd['J22']=J22
            rd['J'] = J11*J22 - J12*J21
        for k in range(order + 1):
            rd[f'a{k}'] = Y[i, idx_a[k], :]
        rays.append(rd)

    print(f"Successfully traced {len(rays)} rays")
    
    # ==================================================================
    # INTERPOLATION ONTO REGULAR GRID
    # ==================================================================
    
    print("Interpolating solution onto grid...")
    
    # Determine domain
    if domain is None:
        x_all = np.concatenate([ray['x'] for ray in rays])
        x_min, x_max = x_all.min(), x_all.max()
        margin = 0.1 * (x_max - x_min)
        
        if dimension == 1:
            domain = (x_min - margin, x_max + margin)
        else:
            y_all = np.concatenate([ray['y'] for ray in rays])
            y_min, y_max = y_all.min(), y_all.max()
            margin_y = 0.1 * (y_max - y_min)
            domain = ((x_min - margin, x_max + margin),
                     (y_min - margin_y, y_max + margin_y))
    
    # Create grid
    if dimension == 1:
        if isinstance(resolution, tuple):
            resolution = resolution[0]
        
        x_grid = np.linspace(domain[0], domain[1], resolution)
        
        # Collect ray data
        x_points = np.concatenate([ray['x'] for ray in rays])
        S_points = np.concatenate([ray['S'] for ray in rays])
        a_points = {k: np.concatenate([ray[f'a{k}'] for ray in rays]) 
                   for k in range(order + 1)}
        
        # Sort for interpolation
        sort_idx = np.argsort(x_points)
        x_points = x_points[sort_idx]
        S_points = S_points[sort_idx]
        for k in range(order + 1):
            a_points[k] = a_points[k][sort_idx]
        
        # Interpolate
        S_grid = interp1d(x_points, S_points, kind='linear', 
                         bounds_error=False, fill_value=0.0)(x_grid)
        
        a_grids = {}
        for k in range(order + 1):
            a_grids[k] = interp1d(x_points, a_points[k], kind='linear',
                                 bounds_error=False, fill_value=0.0)(x_grid)
        
        grid_coords = {'x': x_grid}
        
    else:  # dimension == 2
        if isinstance(resolution, int):
            nx = ny = resolution
        else:
            nx, ny = resolution
        
        (x_min, x_max), (y_min, y_max) = domain
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='ij')
        
        # Collect ray data
        x_points = []
        y_points = []
        S_points = []
        a_points = {k: [] for k in range(order + 1)}
        
        for ray in rays:
            x_points.extend(ray['x'])
            y_points.extend(ray['y'])
            S_points.extend(ray['S'])
            for k in range(order + 1):
                a_points[k].extend(ray[f'a{k}'])
        
        points = np.column_stack([x_points, y_points])
        if np.std(y_points) < 1e-12:
            # degenerate case: 1D interpolation
            S_grid = np.interp(X_grid[:,0], x_points, S_points)
            S_grid = np.tile(S_grid[:, None], (1, Y_grid.shape[1]))
        else:
            S_grid = griddata(points, S_points, (X_grid, Y_grid),
                              method='linear', fill_value=0.0,
                              rescale=True)       
        # Interpolate
        S_grid = np.nan_to_num(S_grid, nan=0.0)
        
        # Amplitude interpolations are independent — run in parallel with threads.
        # (griddata spends most time in C code for Delaunay triangulation, so
        #  threads get real concurrency despite the GIL.)
        from concurrent.futures import ThreadPoolExecutor as _TPool
        a_grids = {}
        def _interp_order(k):
            arr = griddata(points, a_points[k], (X_grid, Y_grid),
                           method='linear', fill_value=0.0)
            return k, np.nan_to_num(arr, nan=0.0)
        with _TPool() as _pool:
            for k, arr in _pool.map(_interp_order, range(order + 1)):
                a_grids[k] = arr
        
        grid_coords = {'x': X_grid, 'y': Y_grid}
    
    # ==================================================================
    # CONSTRUCT WKB SOLUTION
    # ==================================================================
    
    phase_factor = np.exp(1j * S_grid / epsilon)
    
    # Sum asymptotic series
    a_total = np.zeros_like(a_grids[0], dtype=complex)
    epsilon_power = 1.0
    
    for k in range(order + 1):
        a_total += epsilon_power * a_grids[k]
        epsilon_power *= epsilon
        print(f"  Order {k}: max|a_{k}| = {np.max(np.abs(a_grids[k])):.6f}")
    
    u_grid = phase_factor * a_total
    
    print(f"\nWKB solution computed (order {order}, dim={dimension})")
    print(f"Max |u| = {np.max(np.abs(u_grid)):.6f}")
    
    # ==================================================================
    # RETURN RESULTS
    # ==================================================================
    
    result = {
        'dimension': dimension,
        'order': order,
        'epsilon': epsilon,
        'domain': domain,
        'S': S_grid,
        'a': a_grids,
        'a_total': a_total,
        'u': u_grid,
        'rays': rays,
        'n_rays': len(rays)
    }
    
    result.update(grid_coords)
    
    return result

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
        x_c = caustic.position[0]
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
            x_c = caustic.position[0]
            
            # Region of Airy correction
            airy_width = 5 * epsilon**(2/3)
            mask = np.abs(x - x_c) < airy_width
            
            if np.any(mask):
                # Scaled coordinate
                z = (x[mask] - x_c) / epsilon**(2/3)
                
                # Airy function via scipy
                from scipy.special import airy as _airy
                Ai, _, _, _ = _airy(z)
                
                # Amplitude at caustic
                idx_c = np.argmin(np.abs(x - x_c))
                a_c = a[idx_c]
                S_c = S[idx_c]
                
                # Replace with uniform approximation
                u_corrected[mask] = a_c * np.pi * Ai * np.exp(1j * S_c / epsilon)
        
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
        x_c, y_c = caustic.position[0], caustic.position[1]
        caustic_type = caustic.arnold_type
        
        if caustic.arnold_type == 'A2':
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
                from scipy.special import airy as _airy
                Ai, _, _, _ = _airy(z)
                Ai = np.pi * Ai
                
                idx_x = np.argmin(np.abs(X[:, 0] - x_c))
                idx_y = np.argmin(np.abs(Y[0, :] - y_c))
                
                a_c = a[idx_x, idx_y]
                S_c = S[idx_x, idx_y]
                
                u_corrected[mask] = a_c * Ai * np.exp(1j * S_c / epsilon)
        
        elif caustic.arnold_type == 'A3':
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
    print(f"  Fold (Airy): {sum(1 for c in caustics if c.arnold_type=='A2')}")
    print(f"  Cusp (Pearcey): {sum(1 for c in caustics if c.arnold_type=='A3')}")
    
    result = base_solution.copy()
    result['u'] = u_corrected
    result['u_standard'] = base_solution['u']
    
    return result

def compare_orders(symbol, initial_phase, max_order=3, **kwargs):
    import matplotlib.pyplot as plt

    solutions = {}
    for order in range(max_order + 1):
        print(f"\n{'='*60}\nComputing order {order}\n{'='*60}")
        sol = wkb_approximation(symbol, initial_phase, order=order, **kwargs)
        solutions[order] = sol

    # Determine dimension by looking at the shape of the 'x' grid
    if solutions[0]['x'].ndim == 1:
        dim = 1
    else:
        dim = 2

    n_orders = max_order + 1

    if dim == 1:
        fig, axes = plt.subplots(n_orders, 1, figsize=(12, 3*n_orders))
        if n_orders == 1:
            axes = [axes]
        for order, ax in enumerate(axes):
            sol = solutions[order]
            x = sol['x']
            u = sol['u']
            ax.plot(x, np.real(u), 'b-', label='Re(u)', linewidth=2)
            ax.plot(x, np.imag(u), 'r--', label='Im(u)', linewidth=2)
            ax.plot(x, np.abs(u), 'g:', label='|u|', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title(f'Order {order} (ε={sol["epsilon"]:.3f})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.text(0.02, 0.98, f'max|u| = {np.max(np.abs(u)):.4f}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        fig, axes = plt.subplots(2, n_orders, figsize=(5*n_orders, 9))
        if n_orders == 1:
            axes = axes.reshape(2, 1)
        for order in range(n_orders):
            sol = solutions[order]
            X, Y = sol['x'], sol['y']
            u = sol['u']
            im1 = axes[0, order].contourf(X, Y, np.abs(u), levels=30, cmap='viridis')
            axes[0, order].set_title(f'Order {order}: |u|')
            axes[0, order].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0, order])
            im2 = axes[1, order].contourf(X, Y, np.real(u), levels=30, cmap='RdBu_r')
            axes[1, order].set_title(f'Order {order}: Re(u)')
            axes[1, order].set_aspect('equal')
            plt.colorbar(im2, ax=axes[1, order])
            if 'rays' in sol:
                for ray in sol['rays'][::max(1, len(sol['rays'])//15)]:
                    axes[0, order].plot(ray['x'], ray['y'], 'k-', alpha=0.2, lw=0.5)
                    axes[1, order].plot(ray['x'], ray['y'], 'k-', alpha=0.2, lw=0.5)
    plt.tight_layout()

    # Convergence analysis
    print("\n" + "="*60 + "\nCONVERGENCE ANALYSIS\n" + "="*60)
    if dim == 1:
        idx = len(solutions[0]['x']) // 2
        xc = float(solutions[0]['x'][idx])
        print(f"\nAt x = {xc:.3f}:")
        for order in range(n_orders):
            u_val = complex(solutions[order]['u'][idx])
            print(f"  Order {order}: u = {u_val:.6f}, |u| = {abs(u_val):.6f}")
    else:
        nx, ny = solutions[0]['x'].shape
        ix, iy = nx//2, ny//2
        xc = float(solutions[0]['x'][ix, iy])
        yc = float(solutions[0]['y'][ix, iy])
        print(f"\nAt (x,y) = ({xc:.3f}, {yc:.3f}):")
        for order in range(n_orders):
            u_val = complex(solutions[order]['u'][ix, iy])
            print(f"  Order {order}: u = {u_val:.6f}, |u| = {abs(u_val):.6f}")

    print("\nRelative differences between orders:")
    for order in range(1, n_orders):
        u_prev = solutions[order-1]['u']
        u_curr = solutions[order]['u']
        diff = np.linalg.norm(u_curr - u_prev) / (np.linalg.norm(u_prev) + 1e-10)
        print(f"  ||u_{order} - u_{order-1}|| / ||u_{order-1}|| = {diff:.6e}")

    return solutions, fig

def plot_phase_space(solution, time_slice=None):
    """
    Plot phase space (position-momentum) trajectories.
    
    Parameters
    ----------
    solution : dict
        Output from wkb_approximation
    time_slice : float or None
        Time at which to sample (None = final time)
    """
    import matplotlib.pyplot as plt
    
    dim = solution['dimension']
    rays = solution['rays']
    
    if time_slice is None:
        time_idx = -1
    else:
        time_idx = np.argmin(np.abs(rays[0]['t'] - time_slice))
    
    if dim == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Position-momentum plot
        for ray in rays:
            x = ray['x']
            xi = ray['xi']
            ax1.plot(x, xi, 'b-', alpha=0.5, linewidth=1)
            ax1.plot(x[0], xi[0], 'go', markersize=6)
            ax1.plot(x[time_idx], xi[time_idx], 'ro', markersize=4)
        
        ax1.set_xlabel('x (position)', fontsize=12)
        ax1.set_ylabel('ξ (momentum)', fontsize=12)
        ax1.set_title('Phase Space Trajectories', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Phase evolution
        for i, ray in enumerate(rays[::max(1, len(rays)//10)]):
            ax2.plot(ray['t'], ray['S'], alpha=0.7, label=f'Ray {i}')
        
        ax2.set_xlabel('t (time)', fontsize=12)
        ax2.set_ylabel('S (phase)', fontsize=12)
        ax2.set_title('Phase Evolution', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8)
        
    else:  # dim == 2
        fig = plt.figure(figsize=(16, 5))
        
        # 3D phase space (x, y, |p|)
        ax1 = fig.add_subplot(131, projection='3d')
        for ray in rays[::max(1, len(rays)//20)]:
            x = ray['x']
            y = ray['y']
            p_mag = np.sqrt(ray['xi']**2 + ray['eta']**2)
            ax1.plot(x, y, p_mag, alpha=0.6, linewidth=1)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('|p|')
        ax1.set_title('Phase Space (x, y, |p|)')
        
        # Momentum plane (ξ, η)
        ax2 = fig.add_subplot(132)
        for ray in rays:
            xi = ray['xi']
            eta = ray['eta']
            ax2.plot(xi, eta, 'b-', alpha=0.4, linewidth=0.8)
            ax2.plot(xi[0], eta[0], 'go', markersize=4)
            ax2.plot(xi[time_idx], eta[time_idx], 'ro', markersize=3)
        
        ax2.set_xlabel('ξ', fontsize=12)
        ax2.set_ylabel('η', fontsize=12)
        ax2.set_title('Momentum Space', fontsize=13)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        
        # Phase evolution
        ax3 = fig.add_subplot(133)
        for i, ray in enumerate(rays[::max(1, len(rays)//15)]):
            ax3.plot(ray['t'], ray['S'], alpha=0.6)
        
        ax3.set_xlabel('t (time)', fontsize=12)
        ax3.set_ylabel('S (phase)', fontsize=12)
        ax3.set_title('Phase Evolution', fontsize=13)
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==================================================================
# ADVANCED: Amplitude decomposition
# ==================================================================

def plot_amplitude_decomposition(solution):
    """
    Plot individual amplitude orders aₖ and their contributions.
    """
    import matplotlib.pyplot as plt
    
    dim = solution['dimension']
    order = solution['order']
    eps = solution['epsilon']
    
    if dim == 1:
        fig, axes = plt.subplots(order + 2, 1, figsize=(12, 3*(order+2)))
        if order == 0:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        x = solution['x']
        
        # Plot each amplitude order
        for k in range(order + 1):
            ak = solution['a'][k]
            weight = eps**k
            
            axes[k].plot(x, ak, 'b-', linewidth=2, label=f'$a_{k}$')
            axes[k].plot(x, weight * ak, 'r--', linewidth=2, 
                        label=f'$\\varepsilon^{k} a_{k}$')
            
            axes[k].set_xlabel('x')
            axes[k].set_ylabel(f'$a_{k}$')
            axes[k].set_title(f'Amplitude order {k} (weight = ε^{k} = {weight:.4f})')
            axes[k].grid(True, alpha=0.3)
            axes[k].legend()
        
        # Plot total amplitude
        axes[order + 1].plot(x, np.real(solution['a_total']), 'b-', 
                            linewidth=2, label='Re($a_{total}$)')
        axes[order + 1].plot(x, np.imag(solution['a_total']), 'r--', 
                            linewidth=2, label='Im($a_{total}$)')
        axes[order + 1].plot(x, np.abs(solution['a_total']), 'g:', 
                            linewidth=2, label='$|a_{total}|$')
        
        axes[order + 1].set_xlabel('x')
        axes[order + 1].set_ylabel('Total amplitude')
        axes[order + 1].set_title('Total Amplitude (sum of all orders)')
        axes[order + 1].grid(True, alpha=0.3)
        axes[order + 1].legend()
        
    else:  # dim == 2
        fig, axes = plt.subplots(2, order + 2, figsize=(5*(order+2), 9))
        if order == 0:
            axes = axes.reshape(2, 1)
        
        X, Y = solution['x'], solution['y']
        
        # Plot each amplitude order
        for k in range(order + 1):
            ak = solution['a'][k]
            weight = eps**k
            
            # Top: ak
            im1 = axes[0, k].contourf(X, Y, ak, levels=30, cmap='viridis')
            axes[0, k].set_title(f'$a_{k}$')
            axes[0, k].set_xlabel('x')
            axes[0, k].set_ylabel('y')
            axes[0, k].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0, k])
            
            # Bottom: weighted
            im2 = axes[1, k].contourf(X, Y, weight * ak, levels=30, cmap='viridis')
            axes[1, k].set_title(f'$\\varepsilon^{k} a_{k}$ (ε={eps:.3f})')
            axes[1, k].set_xlabel('x')
            axes[1, k].set_ylabel('y')
            axes[1, k].set_aspect('equal')
            plt.colorbar(im2, ax=axes[1, k])
        
        # Plot total amplitude
        a_total_abs = np.abs(solution['a_total'])
        a_total_real = np.real(solution['a_total'])
        
        im3 = axes[0, order+1].contourf(X, Y, a_total_abs, levels=30, cmap='viridis')
        axes[0, order+1].set_title('$|a_{total}|$')
        axes[0, order+1].set_xlabel('x')
        axes[0, order+1].set_ylabel('y')
        axes[0, order+1].set_aspect('equal')
        plt.colorbar(im3, ax=axes[0, order+1])
        
        im4 = axes[1, order+1].contourf(X, Y, a_total_real, levels=30, cmap='RdBu_r')
        axes[1, order+1].set_title('Re($a_{total}$)')
        axes[1, order+1].set_xlabel('x')
        axes[1, order+1].set_ylabel('y')
        axes[1, order+1].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, order+1])
    
    plt.tight_layout()
    return fig

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
            xc = c.position[0]           # CausticEvent uses attribute access, not dict
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
            x_c, y_c = c.position[0], c.position[1]
            t = getattr(c, 'arnold_type', 'A2')

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
            x_c = caustic.position[0]
            axes[0].axvline(x_c, color='red', linestyle=':', alpha=0.5)
            axes[0].text(x_c, axes[0].get_ylim()[1]*0.9, 
                        f"Caustic\n{caustic.arnold_type}", 
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
            axes[1].axvline(caustic.position[0], color='red', linestyle=':', alpha=0.5)
        
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
                axes[2].axvline(caustic.position[0], color='red', linestyle=':', alpha=0.5)
    
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
            x_c, y_c = caustic.position[0], caustic.position[1]
            if caustic.arnold_type == 'A2':
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
        Initial data for wkb_approximation.
    
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
        Initial data for wkb_approximation.
    
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
        Initial data for wkb_approximation.
    
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
        Output from wkb_approximation.
    plot_type : str
        What to visualize: 'phase', 'amplitude', 'real', 'rays'.
    n_rays_plot : int, optional
        Number of rays to plot (if None, plot all).
    
    Examples
    --------
    >>> wkb = wkb_approximation(...)
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
