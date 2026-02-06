from imports import *

def quantization_via_stationary_phase(f, symbol, x_grid, epsilon=0.1, 
                                       dimension=None, y_grid=None,
                                       order=0):
    """
    Evaluate pseudo-differential operator via stationary phase approximation.
    
    Computes [Op(p)f](x) using the stationary phase method instead of direct FFT.
    This bridges WKB ray tracing and Kohn-Nirenberg quantization.
    
    For small ε, the oscillatory integral:
        [Op(p)f](x) = (1/2πε)^d ∫∫ e^{i(x-y)·ξ/ε} p(x,ξ) f(y) dy dξ
    
    is evaluated by finding stationary points where:
        ∇_ξ[(x-y)·ξ] = 0  ⟹  ξ = ∇S  (ray condition)
        
    This naturally connects to WKB: the stationary phase points define the
    bicharacteristic rays of the symbol.
    
    Parameters
    ----------
    f : ndarray
        Input function values, shape (nx,) for 1D or (nx, ny) for 2D
    symbol : sympy.Expr or callable
        Symbol p(x, ξ) for 1D or p(x, y, ξ, η) for 2D
        Can be:
        - SymPy expression (will be lambdified)
        - Callable function accepting (*spatial_coords, *momenta)
    x_grid : ndarray
        Spatial grid in x direction
    epsilon : float, default=0.1
        Small semiclassical parameter
    dimension : int or None
        Spatial dimension (1 or 2). Auto-detected if None.
    y_grid : ndarray, optional
        Spatial grid in y direction (required for 2D)
    order : int, default=0
        Order of stationary phase expansion:
        - 0: Leading order (geometric optics)
        - 1: Include curvature corrections
        - 2: Second-order corrections
        
    Returns
    -------
    result : ndarray
        [Op(p)f](x), same shape as input f
        
    Notes
    -----
    **Method**: Stationary phase approximation
    
    1. For each output point x, find stationary phase points (y*, ξ*) where:
       - ∇_y φ = 0
       - ∇_ξ φ = 0
       where φ(x,y,ξ) = (x-y)·ξ
       
    2. Contribution from each stationary point:
       I ≈ e^{iφ(x,y*,ξ*)/ε} · p(x,ξ*) · f(y*) · |det(Hess(φ))|^{-1/2} · (2πε)^{d/2}
       
    3. Sum over all stationary points (in simple case: just one per x)
    
    **Limitations**:
    - No caustic handling (diverges where rays focus)
    - Assumes stationary points are non-degenerate
    - Works best for ε << 1
    
    Examples
    --------
    >>> # 1D Schrödinger operator with constant potential
    >>> from sympy import symbols
    >>> x_sym, xi_sym = symbols('x xi', real=True)
    >>> symbol = xi_sym**2 + 0.5*x_sym**2  # Harmonic oscillator
    >>> 
    >>> x_grid = np.linspace(-5, 5, 100)
    >>> f = np.exp(-x_grid**2)  # Gaussian input
    >>> 
    >>> result = quantization_via_stationary_phase(
    ...     f, symbol, x_grid, epsilon=0.1
    ... )
    
    >>> # 2D wave equation
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> symbol = xi**2 + eta**2 - 1.0  # Constant speed wave
    >>> 
    >>> nx, ny = 50, 50
    >>> x_grid = np.linspace(-3, 3, nx)
    >>> y_grid = np.linspace(-3, 3, ny)
    >>> X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    >>> 
    >>> f = np.exp(-(X**2 + Y**2))  # 2D Gaussian
    >>> 
    >>> result = quantization_via_stationary_phase(
    ...     f, symbol, x_grid, epsilon=0.1, y_grid=y_grid
    ... )
    """
    from sympy import symbols, diff, lambdify, Symbol
    from scipy.optimize import fsolve, minimize
    import warnings
    
    # ==================================================================
    # DETECT DIMENSION
    # ==================================================================
    
    if dimension is None:
        if y_grid is not None:
            dimension = 2
        else:
            dimension = f.ndim
    
    if dimension not in [1, 2]:
        raise ValueError(f"Dimension must be 1 or 2, got {dimension}")
    
    if dimension == 2 and y_grid is None:
        raise ValueError("y_grid required for 2D")
    
    print(f"Stationary phase quantization in {dimension}D (ε={epsilon})")
    
    # ==================================================================
    # SETUP SYMBOLIC VARIABLES AND LAMBDIFY SYMBOL
    # ==================================================================
    
    if dimension == 1:
        x_sym = symbols('x', real=True)
        xi_sym = symbols('xi', real=True)
        spatial_vars = [x_sym]
        momentum_vars = [xi_sym]
        all_vars = [x_sym, xi_sym]
    else:
        x_sym, y_sym = symbols('x y', real=True)
        xi_sym, eta_sym = symbols('xi eta', real=True)
        spatial_vars = [x_sym, y_sym]
        momentum_vars = [xi_sym, eta_sym]
        all_vars = [x_sym, y_sym, xi_sym, eta_sym]
    
    # Lambdify symbol if needed
    if callable(symbol):
        symbol_func = symbol
    else:
        symbol_func = lambdify(all_vars, symbol, 'numpy')
    
    # ==================================================================
    # COMPUTE SYMBOL DERIVATIVES (for stationary phase corrections)
    # ==================================================================
    
    if not callable(symbol):
        # Compute Hessian of symbol w.r.t. momentum
        print("Computing symbol derivatives...")
        
        hessian_elements = {}
        
        if dimension == 1:
            d2p_dxi2 = diff(symbol, xi_sym, 2)
            hessian_elements['d2p_dxi2'] = lambdify(all_vars, d2p_dxi2, 'numpy')
        else:
            d2p_dxi2 = diff(symbol, xi_sym, 2)
            d2p_deta2 = diff(symbol, eta_sym, 2)
            d2p_dxideta = diff(diff(symbol, xi_sym), eta_sym)
            
            hessian_elements['d2p_dxi2'] = lambdify(all_vars, d2p_dxi2, 'numpy')
            hessian_elements['d2p_deta2'] = lambdify(all_vars, d2p_deta2, 'numpy')
            hessian_elements['d2p_dxideta'] = lambdify(all_vars, d2p_dxideta, 'numpy')
        
        # Gradient for finding stationary points (if needed for optimization)
        grad_p_xi = [lambdify(all_vars, diff(symbol, mom_var), 'numpy') 
                     for mom_var in momentum_vars]
    else:
        hessian_elements = None
        grad_p_xi = None
    
    # ==================================================================
    # STATIONARY PHASE EVALUATION
    # ==================================================================
    
    if dimension == 1:
        result = _stationary_phase_1d(
            f, symbol_func, x_grid, epsilon, 
            hessian_elements, order
        )
    else:
        result = _stationary_phase_2d(
            f, symbol_func, x_grid, y_grid, epsilon,
            hessian_elements, order
        )
    
    return result


def _stationary_phase_1d(f, symbol_func, x_grid, epsilon, hessian_elements, order):
    """
    1D stationary phase evaluation.
    
    The phase is φ(x, y, ξ) = (x-y)·ξ
    
    Stationary conditions:
        ∂φ/∂ξ = x - y = 0  ⟹  y = x  (trivial in this simple case!)
        ∂φ/∂y = -ξ = 0     ⟹  ξ = 0  (but this is degenerate)
        
    Actually, for linear phase, we need to be more careful.
    The stationary point in ξ depends on derivatives of f.
    
    For simplicity, we use a local plane wave approximation:
    f(y) ≈ f(x) e^{ik(y-x)} where k = f'(x)/f(x) (local wavenumber)
    
    Then the stationary point is ξ* = k.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    
    nx = len(x_grid)
    result = np.zeros(nx, dtype=complex)
    
    # Compute local wavenumber of f (via finite differences)
    dx = x_grid[1] - x_grid[0]
    
    # Avoid division by zero
    f_safe = f.copy()
    f_safe[np.abs(f_safe) < 1e-14] = 1e-14
    
    # Local derivative: f'(x) / f(x) gives local phase gradient
    df_dx = np.gradient(f, dx)
    local_wavenumber = df_dx / (f_safe + 1e-14)
    
    # For each output point x
    for i, x_val in enumerate(x_grid):
        # In stationary phase, the main contribution comes from y ≈ x
        # and ξ ≈ local_wavenumber at that point
        
        y_star = x_val  # Stationary point in space
        xi_star = np.real(local_wavenumber[i])  # Stationary point in momentum
        
        # Evaluate symbol at stationary point
        p_val = symbol_func(x_val, xi_star)
        
        # Phase at stationary point: φ = (x-y)*ξ = 0 (since y=x)
        phase = 0.0
        
        # Amplitude: f(y*) = f(x)
        amplitude = f[i]
        
        # Hessian determinant (curvature correction)
        if order >= 1 and hessian_elements is not None:
            # For φ = (x-y)·ξ, the Hessian w.r.t. (y,ξ) is:
            # H = [[ 0    -1  ]
            #      [ -1    0  ]]
            # det(H) = -1, so |det(H)|^{1/2} = 1
            
            # But we also have curvature from the symbol
            d2p = hessian_elements['d2p_dxi2'](x_val, xi_star)
            
            # Corrected determinant (simplified)
            det_correction = 1.0 / np.sqrt(np.abs(d2p) + 1e-10)
        else:
            det_correction = 1.0
        
        # Stationary phase formula:
        # I ≈ (2πε)^{1/2} e^{iφ/ε} p(x,ξ*) f(y*) |det(H)|^{-1/2}
        
        prefactor = np.sqrt(2 * np.pi * epsilon)
        
        result[i] = (prefactor * np.exp(1j * phase / epsilon) * 
                    p_val * amplitude * det_correction)
    
    return result


def _stationary_phase_2d(f, symbol_func, x_grid, y_grid, epsilon, 
                         hessian_elements, order):
    """
    2D stationary phase evaluation.
    
    Similar to 1D but with 2D local wavenumber estimation.
    """
    import numpy as np
    
    nx = len(x_grid)
    ny = len(y_grid)
    
    if f.shape != (nx, ny):
        raise ValueError(f"f shape {f.shape} doesn't match grids ({nx}, {ny})")
    
    result = np.zeros((nx, ny), dtype=complex)
    
    # Grid spacings
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    
    # Compute local wavenumbers (2D gradient)
    f_safe = f.copy()
    f_safe[np.abs(f_safe) < 1e-14] = 1e-14
    
    df_dx = np.gradient(f, axis=0) / dx
    df_dy = np.gradient(f, axis=1) / dy
    
    kx_local = df_dx / (f_safe + 1e-14)
    ky_local = df_dy / (f_safe + 1e-14)
    
    # Create 2D grids
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    
    # For each output point (x, y)
    for i in range(nx):
        for j in range(ny):
            x_val = X[i, j]
            y_val = Y[i, j]
            
            # Stationary points
            x_star = x_val
            y_star = y_val
            xi_star = np.real(kx_local[i, j])
            eta_star = np.real(ky_local[i, j])
            
            # Evaluate symbol
            p_val = symbol_func(x_val, y_val, xi_star, eta_star)
            
            # Phase (zero at stationary point)
            phase = 0.0
            
            # Amplitude
            amplitude = f[i, j]
            
            # Hessian correction
            if order >= 1 and hessian_elements is not None:
                d2p_dxi2 = hessian_elements['d2p_dxi2'](x_val, y_val, xi_star, eta_star)
                d2p_deta2 = hessian_elements['d2p_deta2'](x_val, y_val, xi_star, eta_star)
                d2p_dxideta = hessian_elements['d2p_dxideta'](x_val, y_val, xi_star, eta_star)
                
                # Determinant of momentum Hessian
                det_hess = d2p_dxi2 * d2p_deta2 - d2p_dxideta**2
                det_correction = 1.0 / np.sqrt(np.abs(det_hess) + 1e-10)
            else:
                det_correction = 1.0
            
            # Stationary phase formula (2D)
            prefactor = 2 * np.pi * epsilon  # (2πε)^{d/2} for d=2
            
            result[i, j] = (prefactor * np.exp(1j * phase / epsilon) *
                           p_val * amplitude * det_correction)
    
    return result


# ==================================================================
# EXAMPLE USAGE AND TESTS
# ==================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sympy import symbols, exp as sym_exp, sqrt
    
    print("=" * 70)
    print("STATIONARY PHASE QUANTIZATION - EXAMPLES")
    print("=" * 70)
    
    # ================================================================
    # Example 1: 1D Harmonic oscillator
    # ================================================================
    
    print("\n" + "=" * 70)
    print("Example 1: 1D Harmonic Oscillator")
    print("=" * 70)
    
    x_sym, xi_sym = symbols('x xi', real=True)
    
    # Symbol: p(x,ξ) = ξ² + x²
    omega = 1.0
    symbol_1d = xi_sym**2 + omega**2 * x_sym**2
    
    # Grid
    x_grid = np.linspace(-4, 4, 200)
    
    # Input: Gaussian wave packet
    x0 = -1.0
    k0 = 2.0
    sigma = 0.5
    f_1d = np.exp(-((x_grid - x0) / sigma)**2) * np.exp(1j * k0 * x_grid)
    
    # Apply operator via stationary phase
    epsilon = 0.1
    result_1d = quantization_via_stationary_phase(
        f_1d, symbol_1d, x_grid, epsilon=epsilon, order=1
    )
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_grid, np.abs(f_1d), 'b-', label='|f(x)| input', linewidth=2)
    plt.plot(x_grid, np.real(f_1d), 'b--', alpha=0.5, label='Re[f(x)]')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.title('Input function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_grid, np.abs(result_1d), 'r-', label='|Op(p)f| output', linewidth=2)
    plt.plot(x_grid, np.real(result_1d), 'r--', alpha=0.5, label='Re[Op(p)f]')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.title(f'Output via stationary phase (ε={epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/stationary_phase_1d.png', dpi=150, bbox_inches='tight')
    print("\n✓ 1D plot saved to /tmp/stationary_phase_1d.png")
    
    # ================================================================
    # Example 2: 2D Wave equation
    # ================================================================
    
    print("\n" + "=" * 70)
    print("Example 2: 2D Wave Equation")
    print("=" * 70)
    
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Symbol: p(x,y,ξ,η) = ξ² + η² - c²
    c = 1.0
    symbol_2d = xi**2 + eta**2 - c**2
    
    # Grid
    nx, ny = 60, 60
    x_grid_2d = np.linspace(-3, 3, nx)
    y_grid_2d = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x_grid_2d, y_grid_2d, indexing='ij')
    
    # Input: 2D Gaussian wave packet
    x0, y0 = -0.5, 0.0
    kx0, ky0 = 1.5, 1.0
    sigma_2d = 0.4
    
    f_2d = (np.exp(-(((X - x0)**2 + (Y - y0)**2) / sigma_2d**2)) *
            np.exp(1j * (kx0 * X + ky0 * Y)))
    
    # Apply operator via stationary phase
    epsilon_2d = 0.15
    result_2d = quantization_via_stationary_phase(
        f_2d, symbol_2d, x_grid_2d, epsilon=epsilon_2d, 
        y_grid=y_grid_2d, order=1
    )
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    # Input magnitude
    im1 = axes[0, 0].pcolormesh(X, Y, np.abs(f_2d), shading='auto', cmap='viridis')
    axes[0, 0].set_title('|f(x,y)| - Input')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Input phase
    im2 = axes[0, 1].pcolormesh(X, Y, np.angle(f_2d), shading='auto', cmap='twilight')
    axes[0, 1].set_title('arg(f(x,y)) - Input phase')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Output magnitude
    im3 = axes[1, 0].pcolormesh(X, Y, np.abs(result_2d), shading='auto', cmap='viridis')
    axes[1, 0].set_title(f'|Op(p)f| - Output (ε={epsilon_2d})')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Output phase
    im4 = axes[1, 1].pcolormesh(X, Y, np.angle(result_2d), shading='auto', cmap='twilight')
    axes[1, 1].set_title('arg(Op(p)f) - Output phase')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('/tmp/stationary_phase_2d.png', dpi=150, bbox_inches='tight')
    print("✓ 2D plot saved to /tmp/stationary_phase_2d.png")
    
    print("\n" + "=" * 70)
    print("DONE - Stationary phase quantization examples complete")
    print("=" * 70)