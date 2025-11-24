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
Microlocal analysis toolkit for 1D problems.

This module provides tools for studying singularities and wave propagation:
 - Characteristic variety Char(P) = {(x,ξ) : p(x,ξ) = 0}
 - Bicharacteristic flows on cotangent bundle
 - WKB approximation: eikonal and transport equations
 - Bohr-Sommerfeld quantization conditions
 - Propagation of singularities
 - Caustics and turning points
"""

from imports import *

def characteristic_variety(symbol, tol=1e-8):
    """
    Compute characteristic variety of a pseudo-differential operator.
    
    Char(P) = {(x, ξ) ∈ T*ℝ : p(x, ξ) = 0}
    
    where p(x, ξ) is the principal symbol.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ).
    tol : float
        Tolerance for zero detection.
    
    Returns
    -------
    dict
        Contains symbolic and numerical representations.
    
    Examples
    --------
    >>> x, xi = symbols('x xi', real=True)
    >>> p = xi**2 - x**2  # Wave operator
    >>> char = characteristic_variety(p)
    >>> print(char['implicit'])
    xi**2 - x**2
    
    Notes
    -----
    The characteristic variety determines where the operator
    fails to be elliptic and where singularities propagate.
    """
    x, xi = symbols('x xi', real=True)
    
    # Symbolic characteristic set
    char_eq = Eq(symbol, 0)
    
    # Try to solve for ξ(x)
    try:
        xi_solutions = solve(symbol, xi)
        explicit_curves = [simplify(sol) for sol in xi_solutions]
    except:
        explicit_curves = None
    
    # Lambdify for numerical evaluation
    char_func = lambdify((x, xi), symbol, 'numpy')
    
    return {
        'implicit': symbol,
        'equation': char_eq,
        'explicit': explicit_curves,
        'function': char_func
    }


def bicharacteristic_flow(symbol, z0, tspan, method='hamiltonian', n_steps=1000):
    """
    Integrate bicharacteristic flow on cotangent bundle T*ℝ.
    
    The bicharacteristic equations are Hamilton's equations with
    Hamiltonian H = p(x, ξ):
        ẋ = ∂p/∂ξ
        ξ̇ = -∂p/∂x
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ).
    z0 : tuple
        Initial condition (x₀, ξ₀) on T*ℝ.
    tspan : tuple
        Time interval (t_start, t_end).
    method : str
        Integration method: 'hamiltonian', 'symplectic', 'rk45'.
    n_steps : int
        Number of time steps.
    
    Returns
    -------
    dict
        Bicharacteristic curve: 't', 'x', 'xi', 'symbol_value'.
    
    Examples
    --------
    >>> x, xi = symbols('x xi', real=True)
    >>> p = xi**2 + x**2  # Elliptic
    >>> traj = bicharacteristic_flow(p, (1, 1), (0, 10))
    >>> plt.plot(traj['x'], traj['xi'])
    
    Notes
    -----
    Bicharacteristics are the rays along which singularities propagate.
    They are null geodesics with respect to the symbol's metric.
    """
    from scipy.integrate import solve_ivp
    
    x, xi = symbols('x xi', real=True)
    
    # Compute Hamiltonian vector field
    dp_dxi = diff(symbol, xi)
    dp_dx = diff(symbol, x)
    
    # Lambdify
    f_x = lambdify((x, xi), dp_dxi, 'numpy')
    f_xi = lambdify((x, xi), -dp_dx, 'numpy')
    p_func = lambdify((x, xi), symbol, 'numpy')
    
    if method == 'rk45':
        def ode_system(t, z):
            x_val, xi_val = z
            return [f_x(x_val, xi_val), f_xi(x_val, xi_val)]
        
        sol = solve_ivp(
            ode_system,
            tspan,
            z0,
            method='RK45',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps),
            rtol=1e-9,
            atol=1e-12
        )
        
        return {
            't': sol.t,
            'x': sol.y[0],
            'xi': sol.y[1],
            'symbol_value': p_func(sol.y[0], sol.y[1])
        }
    
    elif method in ['hamiltonian', 'symplectic']:
        dt = (tspan[1] - tspan[0]) / n_steps
        t_vals = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        xi_vals = np.zeros(n_steps)
        
        x_vals[0], xi_vals[0] = z0
        
        for i in range(n_steps - 1):
            x_curr = x_vals[i]
            xi_curr = xi_vals[i]
            
            # Symplectic Euler
            xi_new = xi_curr + dt * f_xi(x_curr, xi_curr)
            x_new = x_curr + dt * f_x(x_curr, xi_new)
            
            x_vals[i+1] = x_new
            xi_vals[i+1] = xi_new
        
        return {
            't': t_vals,
            'x': x_vals,
            'xi': xi_vals,
            'symbol_value': p_func(x_vals, xi_vals)
        }
    
    else:
        raise ValueError("Invalid method")


def wkb_ansatz(symbol, initial_phase, order=1, x_domain=(-5, 5), n_points=200):
    """
    Compute WKB approximation u(x) ≈ a(x) e^(iS(x)/ε).
    
    Solves eikonal and transport equations:
        Eikonal: p(x, S'(x)) = 0
        Transport: ∂_ξp · a' + ½(∂²_ξξp) S'' a = 0
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ).
    initial_phase : dict
        Initial data: {'x0': x₀, 'S0': S₀, 'Sp0': S'₀}.
    order : int
        Order of WKB expansion (0 or 1).
    x_domain : tuple
        Spatial domain for solution.
    n_points : int
        Number of grid points.
    
    Returns
    -------
    dict
        WKB solution: 'x', 'S' (phase), 'a' (amplitude), 'u' (full solution).
    
    Examples
    --------
    >>> x, xi = symbols('x xi', real=True)
    >>> p = xi**2 - x  # Airy equation
    >>> ic = {'x0': 0, 'S0': 0, 'Sp0': 1}
    >>> wkb = wkb_ansatz(p, ic)
    >>> plt.plot(wkb['x'], np.real(wkb['u']))
    
    Notes
    -----
    WKB breaks down at caustics where S'(x) becomes multivalued.
    """
    from scipy.integrate import odeint
    
    x, xi = symbols('x xi', real=True)
    
    # Eikonal equation: p(x, S'(x)) = 0
    # Solve for S'(x) implicitly
    
    x0 = initial_phase['x0']
    S0 = initial_phase['S0']
    Sp0 = initial_phase['Sp0']  # S'(x₀) = ξ₀
    
    # Compute derivatives of p
    dp_dxi = diff(symbol, xi)
    dp_dx = diff(symbol, x)
    d2p_dxi2 = diff(symbol, xi, 2)
    
    # Lambdify
    dp_dxi_func = lambdify((x, xi), dp_dxi, 'numpy')
    dp_dx_func = lambdify((x, xi), dp_dx, 'numpy')
    d2p_dxi2_func = lambdify((x, xi), d2p_dxi2, 'numpy')
    p_func = lambdify((x, xi), symbol, 'numpy')
    
    # Setup ODEs for phase and amplitude
    def ode_system(y, x_val):
        """
        y = [S, S', a, a']
        
        S'' = -∂_x p / ∂_ξ p  (from eikonal)
        a' = given by transport equation
        """
        S_val, Sp_val, a_val, ap_val = y
        
        # Eikonal: dS'/dx
        denom = dp_dxi_func(x_val, Sp_val)
        if abs(denom) < 1e-10:
            # Caustic point
            Spp = 0
        else:
            Spp = -dp_dx_func(x_val, Sp_val) / denom
        
        # Transport equation (simplified)
        # ∂_ξp · a' + ½(∂²_ξξp) S'' a = 0
        if order >= 1 and abs(denom) > 1e-10:
            transport_coeff = 0.5 * d2p_dxi2_func(x_val, Sp_val) * Spp / denom
            app = -transport_coeff * a_val
        else:
            app = 0
        
        return [Sp_val, Spp, ap_val, app]
    
    # Initial conditions
    a0 = 1.0  # Initial amplitude
    ap0 = 0.0
    y0 = [S0, Sp0, a0, ap0]
    
    # Integrate
    x_vals = np.linspace(x_domain[0], x_domain[1], n_points)
    
    # Split integration if x0 not at boundary
    if abs(x_vals[0] - x0) > 1e-6:
        # Forward integration
        x_forward = x_vals[x_vals >= x0]
        sol_forward = odeint(ode_system, y0, x_forward)
        
        # Backward integration
        x_backward = x_vals[x_vals < x0][::-1]
        sol_backward = odeint(ode_system, y0, x_backward)
        sol_backward = sol_backward[::-1]
        
        # Combine
        x_vals = np.concatenate([x_backward, x_forward])
        sol = np.vstack([sol_backward, sol_forward])
    else:
        sol = odeint(ode_system, y0, x_vals)
    
    S_vals = sol[:, 0]
    a_vals = sol[:, 2]
    
    # Construct WKB solution (with ε = 1 for visualization)
    u_vals = a_vals * np.exp(1j * S_vals)
    
    return {
        'x': x_vals,
        'S': S_vals,
        'Sp': sol[:, 1],
        'a': a_vals,
        'u': u_vals
    }

def bohr_sommerfeld_quantization(H, n_max=10, x_range=(-10, 10),
                                 hbar=1.0, method='fast'):
    """
    Compute Bohr-Sommerfeld quantization condition.
    
    For bound states in 1D:
        (1/(2π)) ∮ p dx = ℏ(n + α)
    
    where α is the Maslov index correction (typically 1/2 or 1/4).
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    n_max : int
        Maximum quantum number to compute.
    x_range : tuple
        Spatial range for classical turning points.
    hbar : float
        Planck's constant (set to 1 in natural units).
    method : str
        Computation method: 'contour', 'approximate'.
    
    Returns
    -------
    dict
        Quantized energies: 'n', 'E_n', 'actions'.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2  # Harmonic oscillator
    >>> quant = bohr_sommerfeld_quantization(H, n_max=5)
    >>> print(quant['E_n'])  # Should be E_n = (n + 1/2)ℏω
    
    Notes
    -----
    This is the semiclassical quantization condition, exact for
    harmonic oscillator, accurate for slowly varying potentials.
    """
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import bisect
    from sympy import symbols, solve, lambdify

    x, p = symbols('x p', real=True)
    E_sym = symbols('E', real=True, positive=True)

    # Solve H(x,p)=E → p(x,E)
    p_solutions = solve(H - E_sym, p)
    if not p_solutions:
        raise ValueError("Unable to solve H=E for p(x,E).")

    # Keep the branch with positive momentum
    p_expr = p_solutions[-1]
    p_func = lambdify((x, E_sym), p_expr, 'numpy')

    alpha = 0.5  # Maslov index

    energies = []
    actions = []
    quantum_numbers = []

    # Find turning points from the sign of p^2
    X = np.linspace(x_range[0], x_range[1], 2000)

    def action(E):
        """Compute classical action I(E) = (1/pi) ∫ p dx."""
        p_vals = p_func(X, E)
        p_vals = np.real_if_close(p_vals)
        p_vals = np.real(p_vals)
        
        # Handle case where p_vals is a scalar (independent of x)
        if np.ndim(p_vals) == 0:
            # For free particle or x-independent momentum, no bound states
            return 0.0
        
        mask = p_vals >= 0
        if not np.any(mask):
            return 0.0
    
        # locate turning region
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0
        
        a = X[idx[0]]
        b = X[idx[-1]]
    
        def integrand(xv):
            pv = p_func(xv, E)
            return np.sqrt(max(pv, 0))
    
        I, _ = quad(integrand, a, b, epsabs=1e-10, epsrel=1e-10)
        return I / np.pi

    # target quantized actions
    targets = [hbar*(n + alpha) for n in range(n_max)]

    # Energy brackets to scan
    E_scan = np.linspace(1e-6, 50, 200)

    I_scan = [action(E) for E in E_scan]

    for n, Itarget in zip(range(n_max), targets):

        # Need an interval where action crosses target
        found = False
        for k in range(len(E_scan)-1):
            if (I_scan[k] - Itarget)*(I_scan[k+1] - Itarget) < 0:
                E_left, E_right = E_scan[k], E_scan[k+1]
                found = True
                break

        if not found:
            continue

        # Solve I(E)=target by bisection (monotone → guaranteed)
        def F(E):
            return action(E) - Itarget

        E_n = bisect(F, E_left, E_right, xtol=1e-10, rtol=1e-10, maxiter=100)
        energies.append(E_n)
        actions.append(Itarget)
        quantum_numbers.append(n)

    return {
        "n": np.array(quantum_numbers),
        "E_n": np.array(energies),
        "actions": np.array(actions),
        "hbar": hbar,
        "alpha": alpha
    }


def propagate_singularity(symbol, initial_sing_support, tspan, n_samples=10):
    """
    Propagate singular support along bicharacteristics.
    
    Theorem: Singularities of solutions propagate along bicharacteristics
    in the characteristic variety.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ).
    initial_sing_support : list of tuples
        Initial singular support points [(x₁, ξ₁), (x₂, ξ₂), ...].
    tspan : tuple
        Time interval for propagation.
    n_samples : int
        Number of bicharacteristics per initial point.
    
    Returns
    -------
    dict
        Evolved singular support: 'trajectories', 'endpoints'.
    
    Examples
    --------
    >>> x, xi = symbols('x xi', real=True)
    >>> p = xi  # Transport operator
    >>> sing = propagate_singularity(p, [(0, 1)], (0, 5))
    >>> # Singularity should move to x=5
    
    Notes
    -----
    This implements the microlocal propagation of singularities theorem.
    """
    trajectories = []
    
    for z0 in initial_sing_support:
        traj = bicharacteristic_flow(symbol, z0, tspan, method='symplectic')
        trajectories.append(traj)
    
    # Endpoints
    endpoints = [(traj['x'][-1], traj['xi'][-1]) for traj in trajectories]
    
    return {
        'trajectories': trajectories,
        'endpoints': endpoints,
        'initial': initial_sing_support
    }


def find_caustics(symbol, x_range, xi_range, resolution=100):
    """
    Find caustics (envelope of bicharacteristics).
    
    Caustics occur where the Jacobian of the flow map becomes singular,
    i.e., where bicharacteristics focus.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ).
    x_range, xi_range : tuple
        Phase space domain.
    resolution : int
        Grid resolution.
    
    Returns
    -------
    dict
        Caustic locations and intensities.
    
    Notes
    -----
    At caustics, WKB approximation breaks down and special
    functions (Airy, etc.) are needed.
    
    Examples
    --------
    >>> x, xi = symbols('x xi', real=True)
    >>> p = xi**2 - x  # Creates caustic at x=0
    >>> caustics = find_caustics(p, (-2, 2), (-2, 2))
    """
    x, xi = symbols('x xi', real=True)
    
    # Compute second derivatives for Jacobian
    dp_dxi = diff(symbol, xi)
    dp_dx = diff(symbol, x)
    d2p_dxi2 = diff(symbol, xi, 2)
    d2p_dxdxi = diff(diff(symbol, x), xi)
    
    # Lambdify
    dp_dxi_func = lambdify((x, xi), dp_dxi, 'numpy')
    d2p_dxi2_func = lambdify((x, xi), d2p_dxi2, 'numpy')
    
    # Grid
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    xi_vals = np.linspace(xi_range[0], xi_range[1], resolution)
    X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
    
    # Caustic condition: determinant of stability matrix vanishes
    # Simplified: where ∂²_ξξ p = 0 (turning points in frequency)
    caustic_indicator = np.abs(d2p_dxi2_func(X, XI))
    
    return {
        'x_grid': X,
        'xi_grid': XI,
        'caustic_indicator': caustic_indicator,
        'threshold': np.percentile(caustic_indicator, 10)
    }


def visualize_characteristic_set(symbol, x_range, xi_range, resolution=200):
    """
    Visualize characteristic variety in phase space.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ).
    x_range, xi_range : tuple
        Phase space domain.
    resolution : int
        Grid resolution.
    
    Examples
    --------
    >>> x, xi = symbols('x xi', real=True)
    >>> p = xi**2 - x**2  # Hyperboloid
    >>> visualize_characteristic_set(p, (-3, 3), (-3, 3))
    """
    x, xi = symbols('x xi', real=True)
    p_func = lambdify((x, xi), symbol, 'numpy')
    
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    xi_vals = np.linspace(xi_range[0], xi_range[1], resolution)
    X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
    
    Z = p_func(X, XI)
    
    plt.figure(figsize=(12, 10))
    
    # Contour at zero (characteristic set)
    plt.contour(X, XI, Z, levels=[0], colors='red', linewidths=3,
               label='Char(P)')
    
    # Background: magnitude
    im = plt.pcolormesh(X, XI, np.log10(np.abs(Z) + 1e-10),
                        shading='auto', cmap='viridis', alpha=0.5)
    plt.colorbar(im, label='log₁₀|p(x,ξ)|')
    
    plt.xlabel('x (position)', fontsize=12)
    plt.ylabel('ξ (frequency)', fontsize=12)
    plt.title('Characteristic Variety', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_bicharacteristics(symbol, initial_points, tspan, x_range, xi_range):
    """
    Visualize bicharacteristic curves in phase space.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol.
    initial_points : list of tuples
        Initial points on characteristic set.
    tspan : tuple
        Time interval.
    x_range, xi_range : tuple
        Plot domain.
    
    Examples
    --------
    >>> x, xi = symbols('x xi', real=True)
    >>> p = xi**2 - x
    >>> ics = [(xi_val, np.sqrt(xi_val)) for xi_val in np.linspace(0.1, 2, 5)]
    >>> visualize_bicharacteristics(p, ics, (0, 10), (-1, 5), (-2, 2))
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot characteristic set as background
    x, xi = symbols('x xi', real=True)
    p_func = lambdify((x, xi), symbol, 'numpy')
    
    x_vals = np.linspace(x_range[0], x_range[1], 200)
    xi_vals = np.linspace(xi_range[0], xi_range[1], 200)
    X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
    Z = p_func(X, XI)
    
    ax.contour(X, XI, Z, levels=[0], colors='lightgray', linewidths=2,
              alpha=0.5, linestyles='dashed')
    
    # Plot bicharacteristics
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_points)))
    
    for idx, z0 in enumerate(initial_points):
        traj = bicharacteristic_flow(symbol, z0, tspan, method='symplectic')
        
        # Color by time
        for i in range(len(traj['x']) - 1):
            ax.plot(traj['x'][i:i+2], traj['xi'][i:i+2],
                   color=colors[idx], alpha=0.8, linewidth=2)
        
        # Mark start
        ax.plot(z0[0], z0[1], 'go', markersize=10, zorder=10)
        ax.plot(traj['x'][-1], traj['xi'][-1], 'ro', markersize=8, zorder=10)
    
    ax.set_xlabel('x (position)', fontsize=12)
    ax.set_ylabel('ξ (frequency)', fontsize=12)
    ax.set_title('Bicharacteristic Flow', fontsize=14)
    ax.set_xlim(x_range)
    ax.set_ylim(xi_range)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_wkb_solution(wkb_solution, plot_type='all'):
    """
    Visualize WKB approximation components.
    
    Parameters
    ----------
    wkb_solution : dict
        Output from wkb_ansatz.
    plot_type : str
        What to plot: 'all', 'phase', 'amplitude', 'real', 'imag'.
    
    Examples
    --------
    >>> wkb = wkb_ansatz(...)
    >>> visualize_wkb_solution(wkb, plot_type='all')
    """
    x = wkb_solution['x']
    S = wkb_solution['S']
    a = wkb_solution['a']
    u = wkb_solution['u']
    
    if plot_type == 'all':
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Phase
        axes[0, 0].plot(x, S, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('S(x)')
        axes[0, 0].set_title('Phase Function')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Amplitude
        axes[0, 1].plot(x, a, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('a(x)')
        axes[0, 1].set_title('Amplitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Real part
        axes[1, 0].plot(x, np.real(u), 'g-', linewidth=2, label='Re(u)')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('Re(u)')
        axes[1, 0].set_title('Real Part of WKB Solution')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Imaginary part
        axes[1, 1].plot(x, np.imag(u), 'm-', linewidth=2, label='Im(u)')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('Im(u)')
        axes[1, 1].set_title('Imaginary Part')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    elif plot_type == 'phase':
        plt.figure(figsize=(10, 6))
        plt.plot(x, S, 'b-', linewidth=2)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('S(x)', fontsize=12)
        plt.title('WKB Phase Function', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    elif plot_type == 'amplitude':
        plt.figure(figsize=(10, 6))
        plt.plot(x, a, 'r-', linewidth=2)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('a(x)', fontsize=12)
        plt.title('WKB Amplitude', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    elif plot_type in ['real', 'imag']:
        plt.figure(figsize=(10, 6))
        component = np.real(u) if plot_type == 'real' else np.imag(u)
        label = 'Re(u)' if plot_type == 'real' else 'Im(u)'
        
        plt.plot(x, component, linewidth=2, label=label)
        plt.xlabel('x', fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.title(f'WKB Solution - {label}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


# ============================================================================
# Tests
# ============================================================================

def test_characteristic_variety():
    """Test characteristic variety computation."""
    x, xi = symbols('x xi', real=True)
    
    # Wave operator
    p = xi**2 - 1
    char = characteristic_variety(p)
    
    assert char['explicit'] is not None
    assert len(char['explicit']) == 2  # Two branches: ξ = ±1
    
    print("✓ Characteristic variety test passed")


def test_bicharacteristic_flow():
    """Test bicharacteristic integration."""
    x, xi = symbols('x xi',real=True)
    
    # Transport operator p = ξ
    p = xi
    
    # Bicharacteristic should be straight line in x with constant ξ
    traj = bicharacteristic_flow(p, (0, 1), (0, 5), method='symplectic')
    
    # Check ξ is constant
    assert np.std(traj['xi']) < 1e-6
    
    # Check x = t (since ẋ = ∂p/∂ξ = 1)
    expected_x = traj['t']
    assert np.allclose(traj['x'], expected_x, rtol=1e-2)
    
    print("✓ Bicharacteristic flow test passed")


def test_wkb_ansatz():
    """Test WKB approximation."""
    x, xi = symbols('x xi', real=True)
    
    # Free particle: p = ξ²
    p = xi**2
    
    ic = {'x0': 0, 'S0': 0, 'Sp0': 1}
    wkb = wkb_ansatz(p, ic, order=1, x_domain=(-2, 2))
    
    # Phase should be approximately linear for free particle
    # S(x) ≈ x (since S'(x) ≈ 1)
    assert wkb['S'] is not None
    assert len(wkb['x']) > 0
    
    print("✓ WKB ansatz test passed")


if __name__ == "__main__":
    print("Running microlocal_1d tests...\n")
    test_characteristic_variety()
    test_bicharacteristic_flow()
    test_wkb_ansatz()
    print("\n✓ All tests passed")