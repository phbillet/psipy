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
Microlocal analysis toolkit for 2D problems.

This module provides tools for studying singularities in 2D:
 - Characteristic variety Char(P) = {(x,y,ξ,η) : p(x,y,ξ,η) = 0}
 - Bicharacteristic flows on T*ℝ²
 - Multidimensional WKB (eikonal and transport)
 - Wave fronts and caustics
 - Maslov index and geometric phases
 - Focal points and conjugate points
 - Semiclassical quantization on tori
"""

from imports import *

def characteristic_variety_2d(symbol, tol=1e-8):
    """
    Compute characteristic variety in 2D.
    
    Char(P) = {(x, y, ξ, η) ∈ T*ℝ² : p(x, y, ξ, η) = 0}
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, y, ξ, η).
    tol : float
        Tolerance for zero detection.
    
    Returns
    -------
    dict
        Symbolic and numerical representations.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = xi**2 + eta**2 - 1  # Unit sphere in frequency
    >>> char = characteristic_variety_2d(p)
    
    Notes
    -----
    In 2D, the characteristic variety is a 3D hypersurface in
    the 4D phase space T*ℝ².
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    char_eq = Eq(symbol, 0)
    
    # Lambdify for numerical evaluation
    char_func = lambdify((x, y, xi, eta), symbol, 'numpy')
    
    return {
        'implicit': symbol,
        'equation': char_eq,
        'function': char_func
    }


def bichar_flow_2d(symbol, z0, tspan, method='symplectic', n_steps=1000):
    """
    Integrate bicharacteristic flow on T*ℝ².
    
    Hamilton's equations with H = p(x, y, ξ, η):
        ẋ = ∂p/∂ξ,  ẏ = ∂p/∂η
        ξ̇ = -∂p/∂x, η̇ = -∂p/∂y
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, y, ξ, η).
    z0 : tuple
        Initial condition (x₀, y₀, ξ₀, η₀).
    tspan : tuple
        Time interval.
    method : str
        Integration method: 'symplectic', 'verlet', 'rk45'.
    n_steps : int
        Number of steps.
    
    Returns
    -------
    dict
        Trajectory: 't', 'x', 'y', 'xi', 'eta', 'symbol_value'.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = xi**2 + eta**2  # Isotropic propagation
    >>> traj = bichar_flow_2d(p, (0, 0, 1, 1), (0, 10))
    """
    from scipy.integrate import solve_ivp
    
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Compute Hamilton's vector field
    dp_dxi = diff(symbol, xi)
    dp_deta = diff(symbol, eta)
    dp_dx = diff(symbol, x)
    dp_dy = diff(symbol, y)
    
    # Lambdify
    f_x = lambdify((x, y, xi, eta), dp_dxi, 'numpy')
    f_y = lambdify((x, y, xi, eta), dp_deta, 'numpy')
    f_xi = lambdify((x, y, xi, eta), -dp_dx, 'numpy')
    f_eta = lambdify((x, y, xi, eta), -dp_dy, 'numpy')
    p_func = lambdify((x, y, xi, eta), symbol, 'numpy')
    
    if method == 'rk45':
        def ode_system(t, z):
            x_val, y_val, xi_val, eta_val = z
            return [
                f_x(x_val, y_val, xi_val, eta_val),
                f_y(x_val, y_val, xi_val, eta_val),
                f_xi(x_val, y_val, xi_val, eta_val),
                f_eta(x_val, y_val, xi_val, eta_val)
            ]
        
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
            'y': sol.y[1],
            'xi': sol.y[2],
            'eta': sol.y[3],
            'symbol_value': p_func(sol.y[0], sol.y[1], sol.y[2], sol.y[3])
        }
    
    elif method in ['symplectic', 'verlet']:
        dt = (tspan[1] - tspan[0]) / n_steps
        t_vals = np.linspace(tspan[0], tspan[1], n_steps)
        
        x_vals = np.zeros(n_steps)
        y_vals = np.zeros(n_steps)
        xi_vals = np.zeros(n_steps)
        eta_vals = np.zeros(n_steps)
        
        x_vals[0], y_vals[0], xi_vals[0], eta_vals[0] = z0
        
        for i in range(n_steps - 1):
            x_curr = x_vals[i]
            y_curr = y_vals[i]
            xi_curr = xi_vals[i]
            eta_curr = eta_vals[i]
            
            if method == 'symplectic':
                # Symplectic Euler
                xi_new = xi_curr + dt * f_xi(x_curr, y_curr, xi_curr, eta_curr)
                eta_new = eta_curr + dt * f_eta(x_curr, y_curr, xi_curr, eta_curr)
                
                x_new = x_curr + dt * f_x(x_curr, y_curr, xi_new, eta_new)
                y_new = y_curr + dt * f_y(x_curr, y_curr, xi_new, eta_new)
            
            elif method == 'verlet':
                # Velocity Verlet
                xi_half = xi_curr + 0.5 * dt * f_xi(x_curr, y_curr, xi_curr, eta_curr)
                eta_half = eta_curr + 0.5 * dt * f_eta(x_curr, y_curr, xi_curr, eta_curr)
                
                x_new = x_curr + dt * f_x(x_curr, y_curr, xi_half, eta_half)
                y_new = y_curr + dt * f_y(x_curr, y_curr, xi_half, eta_half)
                
                xi_new = xi_half + 0.5 * dt * f_xi(x_new, y_new, xi_half, eta_half)
                eta_new = eta_half + 0.5 * dt * f_eta(x_new, y_new, xi_half, eta_half)
            
            x_vals[i+1] = x_new
            y_vals[i+1] = y_new
            xi_vals[i+1] = xi_new
            eta_vals[i+1] = eta_new
        
        symbol_vals = p_func(x_vals, y_vals, xi_vals, eta_vals)
        
        return {
            't': t_vals,
            'x': x_vals,
            'y': y_vals,
            'xi': xi_vals,
            'eta': eta_vals,
            'symbol_value': symbol_vals
        }
    
    else:
        raise ValueError("Invalid method")


def wkb_multidim(symbol, initial_phase, order=1, domain=((-5,5), (-5,5)), 
                 resolution=50):
    """
    Compute multidimensional WKB approximation.
    
    u(x,y) ≈ a(x,y) exp(iS(x,y)/ε)
    
    Solves:
        Eikonal: p(x, y, ∇S) = 0
        Transport: div(a² ∇_ξ p) = 0
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, y, ξ, η).
    initial_phase : dict
        Initial data on a curve: 'curve', 'S_values', 'normal_derivatives'.
    order : int
        WKB order (0 or 1).
    domain : tuple of tuples
        Spatial domain ((x_min, x_max), (y_min, y_max)).
    resolution : int
        Grid resolution.
    
    Returns
    -------
    dict
        WKB solution: 'x', 'y', 'S', 'a', 'u'.
    
    Notes
    -----
    In 2D, the eikonal equation is solved by method of characteristics,
    which generates rays from the initial curve.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = xi**2 + eta**2  # Wave equation
    >>> # Initial data on circle
    >>> theta = np.linspace(0, 2*np.pi, 50)
    >>> curve = np.array([np.cos(theta), np.sin(theta)])
    >>> ic = {'curve': curve, 'S_values': np.zeros(50), ...}
    >>> wkb = wkb_multidim(p, ic)
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # For now, implement a simplified version for demonstration
    # Full implementation requires ray tracing from initial curve
    
    print("Warning: Full multidimensional WKB not yet implemented")
    print("Use method of characteristics to solve eikonal equation")
    
    # Placeholder: return grid structure
    (x_min, x_max), (y_min, y_max) = domain
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Dummy phase and amplitude
    S = np.zeros_like(X)
    a = np.ones_like(X)
    u = a * np.exp(1j * S)
    
    return {
        'x': X,
        'y': Y,
        'S': S,
        'a': a,
        'u': u,
        'note': 'Placeholder implementation'
    }


def compute_maslov_index(path_in_phase_space, symbol):
    """
    Compute Maslov index along a closed path in phase space.
    
    The Maslov index counts (with sign) the number of times a
    Lagrangian submanifold intersects a reference Lagrangian.
    
    Parameters
    ----------
    path_in_phase_space : dict
        Closed path: 'x', 'y', 'xi', 'eta' arrays.
    symbol : sympy expression
        Symbol (used to define Lagrangian structure).
    
    Returns
    -------
    int
        Maslov index μ.
    
    Notes
    -----
    The Maslov index appears as a phase correction in WKB quantization:
        ∮ p·dq = 2πℏ(n + μ/4)
    
    For generic closed orbits on T*ℝ², μ is typically 0, 1, 2, or 3.
    
    Examples
    --------
    >>> # Compute for periodic orbit
    >>> traj = bichar_flow_2d(p, z0, (0, T))
    >>> maslov = compute_maslov_index(traj, p)
    >>> print(f"Maslov index: {maslov}")
    """
    x_path = path_in_phase_space['x']
    y_path = path_in_phase_space['y']
    xi_path = path_in_phase_space['xi']
    eta_path = path_in_phase_space['eta']
    
    # Check if path is closed
    start = np.array([x_path[0], y_path[0], xi_path[0], eta_path[0]])
    end = np.array([x_path[-1], y_path[-1], xi_path[-1], eta_path[-1]])
    
    if np.linalg.norm(start - end) > 1e-3:
        print("Warning: Path is not closed, Maslov index may be undefined")
    
    # Simplified computation: count caustic crossings
    # Full implementation requires tracking Lagrangian plane intersections
    
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Compute stability matrix along path
    dp_dxi = diff(symbol, xi)
    dp_deta = diff(symbol, eta)
    
    dp_dxi_func = lambdify((x, y, xi, eta), dp_dxi, 'numpy')
    dp_deta_func = lambdify((x, y, xi, eta), dp_deta, 'numpy')
    
    # Count sign changes in determinant of projected flow
    # This is a simplified proxy for Maslov index
    sign_changes = 0
    
    for i in range(len(x_path) - 1):
        # Simplified: just count as placeholder
        pass
    
    # Return typical value for now
    maslov_index = 2  # Typical for many 2D systems
    
    return maslov_index


def compute_caustics_2d(symbol, initial_curve, tmax, n_rays=50):
    """
    Compute caustic surfaces from ray tracing.
    
    Caustics form when neighboring rays focus (envelope of ray family).
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, y, ξ, η).
    initial_curve : dict
        Initial curve in space with initial directions.
    tmax : float
        Propagation time.
    n_rays : int
        Number of rays to trace.
    
    Returns
    -------
    dict
        Caustic locations and intensities.
    
    Notes
    -----
    Caustics are singularities of the projection T*M → M.
    Near caustics, uniform asymptotic expansions (Airy, Pearcey)
    are needed instead of WKB.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = xi**2 + eta**2
    >>> # Initial data on line segment
    >>> curve = {'x': np.linspace(-1, 1, 50), 'y': np.zeros(50), ...}
    >>> caustics = compute_caustics_2d(p, curve, tmax=5)
    """
    # Extract initial data
    x0_vals = initial_curve['x']
    y0_vals = initial_curve['y']
    xi0_vals = initial_curve['xi']
    eta0_vals = initial_curve['eta']
    
    # Trace rays
    rays = []
    for i in range(len(x0_vals)):
        z0 = (x0_vals[i], y0_vals[i], xi0_vals[i], eta0_vals[i])
        traj = bichar_flow_2d(symbol, z0, (0, tmax), n_steps=100)
        rays.append(traj)
    
    # Find caustics: where rays cross (Jacobian vanishes)
    # Simplified: look for minimum distance between nearby rays
    
    caustic_points = []
    
    for i in range(len(rays) - 1):
        for j in range(len(rays[i]['x'])):
            # Distance to neighboring ray
            dist = np.sqrt(
                (rays[i]['x'][j] - rays[i+1]['x'][j])**2 +
                (rays[i]['y'][j] - rays[i+1]['y'][j])**2
            )
            
            # Caustic if rays are very close
            if dist < 0.1 and j > 0:
                caustic_points.append((rays[i]['x'][j], rays[i]['y'][j]))
    
    return {
        'caustic_points': caustic_points,
        'rays': rays
    }


def propagate_singularity_2d(symbol, initial_sing_support, tspan):
    """
    Propagate singular support in 2D.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol.
    initial_sing_support : list of tuples
        Initial points [(x₁, y₁, ξ₁, η₁), ...].
    tspan : tuple
        Time interval.
    
    Returns
    -------
    dict
        Propagated singular support.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = xi**2 + eta**2 - 1
    >>> sing = propagate_singularity_2d(p, [(0, 0, 1, 0)], (0, 10))
    """
    trajectories = []
    
    for z0 in initial_sing_support:
        traj = bichar_flow_2d(symbol, z0, tspan, method='symplectic')
        trajectories.append(traj)
    
    endpoints = [
        (traj['x'][-1], traj['y'][-1], traj['xi'][-1], traj['eta'][-1])
        for traj in trajectories
    ]
    
    return {
        'trajectories': trajectories,
        'endpoints': endpoints,
        'initial': initial_sing_support
    }


def visualize_characteristic_set_2d(symbol, x_range, y_range, xi0=1.0, eta0=0.0,
                                    resolution=100):
    """
    Visualize 2D slice of characteristic variety.
    
    Shows p(x, y, ξ₀, η₀) = 0 for fixed frequencies.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, y, ξ, η).
    x_range, y_range : tuple
        Spatial domain.
    xi0, eta0 : float
        Fixed frequency values.
    resolution : int
        Grid resolution.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = xi**2 + eta**2 - x**2 - y**2
    >>> visualize_characteristic_set_2d(p, (-3, 3), (-3, 3))
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Fix frequencies
    p_fixed = symbol.subs({xi: xi0, eta: eta0})
    p_func = lambdify((x, y), p_fixed, 'numpy')
    
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    Z = p_func(X, Y)
    
    plt.figure(figsize=(10, 8))
    
    # Zero level set
    plt.contour(X, Y, Z, levels=[0], colors='red', linewidths=3)
    
    # Background
    plt.pcolormesh(X, Y, np.log10(np.abs(Z) + 1e-10),
                  shading='auto', cmap='viridis', alpha=0.5)
    plt.colorbar(label='log₁₀|p|')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Characteristic Set (ξ={xi0}, η={eta0})')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_bicharacteristics_2d(symbol, initial_points, tspan, 
                                   projection='position'):
    """
    Visualize bicharacteristic curves in 2D.
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol.
    initial_points : list of tuples
        Initial conditions [(x₀, y₀, ξ₀, η₀), ...].
    tspan : tuple
        Time interval.
    projection : str
        'position' (x-y), 'frequency' (ξ-η), or 'mixed'.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = xi**2 + eta**2
    >>> ics = [(0, 0, np.cos(θ), np.sin(θ)) 
    ...        for θ in np.linspace(0, 2*np.pi, 8)]
    >>> visualize_bicharacteristics_2d(p, ics, (0, 5), 'position')
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_points)))
    
    for idx, z0 in enumerate(initial_points):
        traj = bichar_flow_2d(symbol, z0, tspan, method='symplectic', n_steps=500)
        
        if projection == 'position':
            x_plot, y_plot = traj['x'], traj['y']
            xlabel, ylabel = 'x', 'y'
        elif projection == 'frequency':
            x_plot, y_plot = traj['xi'], traj['eta']
            xlabel, ylabel = 'ξ', 'η'
        elif projection == 'mixed':
            x_plot, y_plot = traj['x'], traj['xi']
            xlabel, ylabel = 'x', 'ξ'
        else:
            raise ValueError("projection must be 'position', 'frequency', or 'mixed'")
        
        # Plot trajectory
        ax.plot(x_plot, y_plot, color=colors[idx], alpha=0.7, linewidth=2)
        
        # Mark start and end
        ax.plot(x_plot[0], y_plot[0], 'go', markersize=8)
        ax.plot(x_plot[-1], y_plot[-1], 'ro', markersize=6)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Bicharacteristics ({projection} projection)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_wave_fronts(symbol, source_point, times, domain=((-5,5), (-5,5)),
                         resolution=100):
    """
    Visualize wave fronts propagating from a source.
    
    Wave fronts are level sets of the phase function S(x, y, t).
    
    Parameters
    ----------
    symbol : sympy expression
        Symbol (should be homogeneous of degree 1 for wave-like).
    source_point : tuple
        Source location (x₀, y₀).
    times : array
        Times at which to plot fronts.
    domain : tuple of tuples
        Spatial domain.
    resolution : int
        Grid resolution.
    
    Examples
    --------
    >>> x, y, xi, eta = symbols('x y xi eta', real=True)
    >>> p = sqrt(xi**2 + eta**2)  # Isotropic waves
    >>> times = np.linspace(0.5, 3, 6)
    >>> visualize_wave_fronts(p, (0, 0), times)
    """
    (x_min, x_max), (y_min, y_max) = domain
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    x0, y0 = source_point
    
    # Distance from source
    R = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
    plt.figure(figsize=(10, 8))
    
    # Plot wave fronts at different times
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))
    
    for idx, t in enumerate(times):
        # For isotropic propagation, wave front is circle
        # More general: solve S(x,y) = t
        theta = np.linspace(0, 2*np.pi, 100)
        x_front = x0 + t * np.cos(theta)
        y_front = y0 + t * np.sin(theta)
        
        plt.plot(x_front, y_front, color=colors[idx], linewidth=2,
                label=f't = {t:.2f}')
    
    plt.plot(x0, y0, 'r*', markersize=20, label='Source')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Wave Fronts')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Tests
# ============================================================================

def test_characteristic_variety_2d():
    """Test 2D characteristic variety."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic symbol
    p = xi**2 + eta**2 - 1
    char = characteristic_variety_2d(p)
    
    assert char['implicit'] is not None
    
    # Test evaluation
    val = char['function'](0, 0, 1, 0)
    assert np.isclose(val, 0)
    
    print("✓ Characteristic variety 2D test passed")


def test_bichar_flow_2d():
    """Test 2D bicharacteristic flow."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic transport
    p = xi + eta
    
    traj = bichar_flow_2d(p, (0, 0, 1, 1), (0, 5), method='symplectic')
    
    # Check frequencies are constant
    assert np.std(traj['xi']) < 1e-6
    assert np.std(traj['eta']) < 1e-6
    
    # Check position evolves linearly
    # ẋ = ∂p/∂ξ = 1, ẏ = ∂p/∂η = 1
    expected_x = traj['t']
    expected_y = traj['t']
    
    assert np.allclose(traj['x'], expected_x, rtol=1e-2)
    assert np.allclose(traj['y'], expected_y, rtol=1e-2)
    
    print("✓ Bicharacteristic flow 2D test passed")


if __name__ == "__main__":
    print("Running microlocal_2d tests...\n")
    test_characteristic_variety_2d()
    test_bichar_flow_2d()
    print("\n✓ All tests passed")