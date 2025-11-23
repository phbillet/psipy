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
Riemannian geometry toolkit for 1D manifolds.

This module provides tools for working with Riemannian metrics on 1D manifolds:
 - Metric representation: g₁₁(x), g¹¹(x), |g|, √|g|
 - Christoffel symbols: Γ¹₁₁ = ½(log g₁₁)'
 - Geodesic equations: ẍ + Γ¹₁₁ ẋ² = 0
 - Laplace-Beltrami operator with explicit transport terms
 - Riemannian volume: dVg = √g₁₁ dx
 - Connections to Sturm-Liouville theory
"""

from imports import *

class Metric1D:
    """
    Riemannian metric on a 1D manifold.
    
    Represents a metric tensor g₁₁(x) and provides methods for computing
    geometric quantities: inverse metric, Christoffel symbols, curvature,
    and associated operators.
    
    Parameters
    ----------
    g_expr : sympy expression
        Symbolic expression for the metric component g₁₁(x).
    var_x : sympy symbol
        Spatial coordinate variable.
    
    Attributes
    ----------
    g_expr : sympy expression
        Metric tensor component g₁₁(x).
    g_inv_expr : sympy expression
        Inverse metric g¹¹(x) = 1/g₁₁(x).
    sqrt_det_expr : sympy expression
        Square root of determinant √|g| = √g₁₁.
    christoffel_expr : sympy expression
        Christoffel symbol Γ¹₁₁ = ½(log g₁₁)'.
    
    Examples
    --------
    >>> # Flat metric
    >>> x = symbols('x', real=True)
    >>> metric = Metric1D(1, x)
    
    >>> # Hyperbolic metric
    >>> metric = Metric1D(1/x**2, x)
    >>> print(metric.gauss_curvature())
    
    >>> # From Hamiltonian
    >>> p = symbols('p', real=True)
    >>> H = p**2 / (2*x**2)  # Kinetic term
    >>> metric = Metric1D.from_hamiltonian(H, x, p)
    """
    
    def __init__(self, g_expr, var_x):
        self.var_x = var_x
        self.g_expr = simplify(g_expr)
        self.g_inv_expr = simplify(1 / self.g_expr)
        self.sqrt_det_expr = simplify(sqrt(abs(self.g_expr)))
        
        # Christoffel symbol: Γ¹₁₁ = ½(log g₁₁)'
        log_g = log(abs(self.g_expr))
        self.christoffel_expr = simplify(diff(log_g, var_x) / 2)
        
        # Lambdify for numerical evaluation
        self.g_func = lambdify(var_x, self.g_expr, 'numpy')
        self.g_inv_func = lambdify(var_x, self.g_inv_expr, 'numpy')
        self.sqrt_det_func = lambdify(var_x, self.sqrt_det_expr, 'numpy')
        self.christoffel_func = lambdify(var_x, self.christoffel_expr, 'numpy')
    
    @classmethod
    def from_hamiltonian(cls, H_expr, var_x, var_p):
        """
        Extract metric from Hamiltonian kinetic term.
        
        For a Hamiltonian H = g¹¹(x) p²/2 + V(x), extract the inverse
        metric g¹¹ = ∂²H/∂p².
        
        Parameters
        ----------
        H_expr : sympy expression
            Hamiltonian expression H(x, p).
        var_x : sympy symbol
            Position variable.
        var_p : sympy symbol
            Momentum variable.
        
        Returns
        -------
        Metric1D
            Metric object with g₁₁ = 1/g¹¹.
        
        Examples
        --------
        >>> x, p = symbols('x p', real=True)
        >>> H = p**2/(2*x**2) + x**2/2
        >>> metric = Metric1D.from_hamiltonian(H, x, p)
        >>> print(metric.g_expr)
        x**2
        """
        # Extract g¹¹ from kinetic term
        g_inv = diff(H_expr, var_p, 2)
        g = simplify(1 / g_inv)
        return cls(g, var_x)
    
    def eval(self, x_vals):
        """
        Evaluate metric components at given points.
        
        Parameters
        ----------
        x_vals : float or ndarray
            Spatial coordinates.
        
        Returns
        -------
        dict
            Dictionary containing 'g', 'g_inv', 'sqrt_det', 'christoffel'.
        """
        return {
            'g': self.g_func(x_vals),
            'g_inv': self.g_inv_func(x_vals),
            'sqrt_det': self.sqrt_det_func(x_vals),
            'christoffel': self.christoffel_func(x_vals)
        }
    
    def gauss_curvature(self):
        """
        Compute Gaussian curvature K(x).
        
        In 1D (curves in higher-dimensional space), intrinsic curvature
        vanishes. This returns the extrinsic curvature if embedded.
        For surfaces, use riemannian_2d.
        
        Returns
        -------
        sympy expression
            Curvature K(x) = 0 for intrinsic 1D geometry.
        
        Notes
        -----
        For a curve parametrized by arc length, the curvature measures
        how much the curve deviates from being a straight line.
        """
        # Intrinsic curvature is zero for 1D
        return sympify(0)
    
    def ricci_scalar(self):
        """
        Compute Ricci scalar R(x).
        
        Returns
        -------
        sympy expression
            Ricci scalar R = 0 (1D manifold).
        """
        return sympify(0)
    
    def laplace_beltrami_symbol(self):
        """
        Compute symbol of the Laplace-Beltrami operator.
        
        The Laplace-Beltrami operator in 1D is:
            Δg f = (1/√g) d/dx(√g g¹¹ df/dx)
                 = g¹¹ d²f/dx² + (√g)'/√g · g¹¹ df/dx
        
        Returns
        -------
        dict
            Dictionary with 'principal' (g¹¹ ξ²) and 'subprincipal' 
            (first-order transport term).
        
        Examples
        --------
        >>> x, xi = symbols('x xi', real=True)
        >>> metric = Metric1D(x**2, x)
        >>> lb = metric.laplace_beltrami_symbol()
        >>> print(lb['principal'])
        xi**2/x**2
        """
        x = self.var_x
        xi = symbols('xi', real=True)
        
        # Principal symbol: g¹¹(x) ξ²
        principal = self.g_inv_expr * xi**2
        
        # Subprincipal symbol (transport term)
        # Coefficient of first derivative: d(log√g)/dx · g¹¹
        log_sqrt_g = log(self.sqrt_det_expr)
        transport_coeff = simplify(diff(log_sqrt_g, x) * self.g_inv_expr)
        subprincipal = transport_coeff * xi
        
        return {
            'principal': simplify(principal),
            'subprincipal': simplify(subprincipal),
            'full': simplify(principal + 1j * subprincipal)
        }
    
    def riemannian_volume(self, x_min, x_max, method='symbolic'):
        """
        Compute Riemannian volume of interval [x_min, x_max].
        
        Vol([a,b]) = ∫ₐᵇ √g₁₁(x) dx
        
        Parameters
        ----------
        x_min, x_max : float
            Interval endpoints.
        method : {'symbolic', 'numerical'}
            Integration method.
        
        Returns
        -------
        float or sympy expression
            Volume of the interval.
        
        Examples
        --------
        >>> x = symbols('x', real=True)
        >>> metric = Metric1D(1, x)  # Flat
        >>> vol = metric.riemannian_volume(0, 1)
        >>> print(vol)
        1
        """
        if method == 'symbolic':
            return integrate(self.sqrt_det_expr, (self.var_x, x_min, x_max))
        elif method == 'numerical':
            from scipy.integrate import quad
            integrand = lambda x: self.sqrt_det_func(x)
            result, error = quad(integrand, x_min, x_max)
            return result
        else:
            raise ValueError("method must be 'symbolic' or 'numerical'")
    
    def arc_length(self, x_min, x_max, method='numerical'):
        """
        Compute arc length between two points.
        
        L = ∫ₐᵇ √g₁₁(x) dx
        
        Parameters
        ----------
        x_min, x_max : float
            Endpoints.
        method : {'symbolic', 'numerical'}
            Computation method.
        
        Returns
        -------
        float
            Arc length.
        """
        return self.riemannian_volume(x_min, x_max, method=method)


def christoffel(metric):
    """
    Compute Christoffel symbol as a callable function.
    
    Parameters
    ----------
    metric : Metric1D
        Riemannian metric.
    
    Returns
    -------
    callable
        Function Γ(x) returning Γ¹₁₁(x).
    
    Examples
    --------
    >>> x = symbols('x', real=True)
    >>> metric = Metric1D(x**2, x)
    >>> Gamma = christoffel(metric)
    >>> print(Gamma(2.0))
    """
    return metric.christoffel_func


def geodesic_integrator(metric, x0, v0, tspan, method='rk4', n_steps=1000):
    """
    Integrate geodesic equations.
    
    Solves: ẍ + Γ¹₁₁(x) ẋ² = 0
    
    Converted to first-order system:
        ẋ = v
        v̇ = -Γ¹₁₁(x) v²
    
    Parameters
    ----------
    metric : Metric1D
        Riemannian metric.
    x0 : float
        Initial position.
    v0 : float
        Initial velocity dx/dt.
    tspan : tuple
        Time interval (t_start, t_end).
    method : {'rk4', 'symplectic', 'adaptive'}
        Integration method.
    n_steps : int
        Number of time steps.
    
    Returns
    -------
    dict
        Dictionary with 't', 'x', 'v' arrays.
    
    Examples
    --------
    >>> x = symbols('x', real=True)
    >>> metric = Metric1D(1, x)  # Flat
    >>> traj = geodesic_integrator(metric, 0.0, 1.0, (0, 10))
    >>> plt.plot(traj['t'], traj['x'])
    
    Notes
    -----
    - For flat metric, geodesics are straight lines.
    - Symplectic integrators preserve energy better for long integrations.
    """
    from scipy.integrate import solve_ivp
    
    Gamma_func = metric.christoffel_func
    
    def geodesic_ode(t, y):
        x, v = y
        dxdt = v
        dvdt = -Gamma_func(x) * v**2
        return [dxdt, dvdt]
    
    if method == 'rk4' or method == 'adaptive':
        sol = solve_ivp(
            geodesic_ode, 
            tspan, 
            [x0, v0],
            method='RK45' if method == 'adaptive' else 'RK23',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps)
        )
        return {
            't': sol.t,
            'x': sol.y[0],
            'v': sol.y[1]
        }
    
    elif method == 'symplectic':
        # Symplectic Euler for Hamiltonian formulation
        # H = g¹¹(x)/2 · p²
        # ẋ = g¹¹ p
        # ṗ = -½ (∂g¹¹/∂x) p²
        
        dt = (tspan[1] - tspan[0]) / n_steps
        t_vals = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        p_vals = np.zeros(n_steps)
        
        # Initial momentum: p = v / g¹¹
        g_inv_0 = metric.g_inv_func(x0)
        p0 = v0 / g_inv_0
        
        x_vals[0] = x0
        p_vals[0] = p0
        
        # Prepare derivative of g¹¹
        g_inv_prime = lambdify(
            metric.var_x,
            diff(metric.g_inv_expr, metric.var_x),
            'numpy'
        )
        
        for i in range(n_steps - 1):
            x = x_vals[i]
            p = p_vals[i]
            
            # Symplectic Euler step
            g_inv = metric.g_inv_func(x)
            p_new = p - dt * 0.5 * g_inv_prime(x) * p**2
            x_new = x + dt * g_inv * p_new
            
            x_vals[i+1] = x_new
            p_vals[i+1] = p_new
        
        # Convert momentum back to velocity
        v_vals = np.array([
            metric.g_inv_func(x) * p 
            for x, p in zip(x_vals, p_vals)
        ])
        
        return {
            't': t_vals,
            'x': x_vals,
            'v': v_vals,
            'p': p_vals
        }
    
    else:
        raise ValueError("method must be 'rk4', 'symplectic', or 'adaptive'")


def geodesic_hamiltonian_flow(metric, x0, p0, tspan, method='verlet', n_steps=1000):
    """
    Integrate geodesic flow in Hamiltonian formulation.
    
    Hamiltonian: H(x,p) = g¹¹(x) p² / 2
    
    Hamilton's equations:
        ẋ = ∂H/∂p = g¹¹(x) p
        ṗ = -∂H/∂x = -½ (∂g¹¹/∂x) p²
    
    Parameters
    ----------
    metric : Metric1D
        Riemannian metric.
    x0 : float
        Initial position.
    p0 : float
        Initial momentum.
    tspan : tuple
        Time interval (t_start, t_end).
    method : {'verlet', 'stormer', 'symplectic_euler'}
        Symplectic integration method.
    n_steps : int
        Number of steps.
    
    Returns
    -------
    dict
        Trajectory with 't', 'x', 'p', 'energy' arrays.
    
    Notes
    -----
    Symplectic integrators preserve the Hamiltonian structure and
    conserve energy up to small oscillations (no secular drift).
    """
    dt = (tspan[1] - tspan[0]) / n_steps
    t_vals = np.linspace(tspan[0], tspan[1], n_steps)
    x_vals = np.zeros(n_steps)
    p_vals = np.zeros(n_steps)
    
    x_vals[0] = x0
    p_vals[0] = p0
    
    g_inv_func = metric.g_inv_func
    g_inv_prime = lambdify(
        metric.var_x,
        diff(metric.g_inv_expr, metric.var_x),
        'numpy'
    )
    
    if method == 'verlet' or method == 'stormer':
        # Velocity Verlet / Störmer-Verlet
        for i in range(n_steps - 1):
            x = x_vals[i]
            p = p_vals[i]
            
            # Half-step momentum
            force = -0.5 * g_inv_prime(x) * p**2
            p_half = p + 0.5 * dt * force
            
            # Full-step position
            x_new = x + dt * g_inv_func(x) * p_half
            
            # Half-step momentum (second half)
            force_new = -0.5 * g_inv_prime(x_new) * p_half**2
            p_new = p_half + 0.5 * dt * force_new
            
            x_vals[i+1] = x_new
            p_vals[i+1] = p_new
    
    elif method == 'symplectic_euler':
        for i in range(n_steps - 1):
            x = x_vals[i]
            p = p_vals[i]
            
            # Update p first
            force = -0.5 * g_inv_prime(x) * p**2
            p_new = p + dt * force
            
            # Update x with new p
            x_new = x + dt * g_inv_func(x) * p_new
            
            x_vals[i+1] = x_new
            p_vals[i+1] = p_new
    
    else:
        raise ValueError("method must be 'verlet', 'stormer', or 'symplectic_euler'")
    
    # Compute energy
    energy = np.array([
        0.5 * g_inv_func(x) * p**2 
        for x, p in zip(x_vals, p_vals)
    ])
    
    return {
        't': t_vals,
        'x': x_vals,
        'p': p_vals,
        'energy': energy
    }


def laplace_beltrami(metric):
    """
    Construct Laplace-Beltrami operator as a pseudo-differential operator.
    
    Returns a symbol compatible with psiop.PseudoDifferentialOperator.
    
    Parameters
    ----------
    metric : Metric1D
        Riemannian metric.
    
    Returns
    -------
    dict
        Symbol components for use with PseudoDifferentialOperator.
    
    Examples
    --------
    >>> from psiop import PseudoDifferentialOperator
    >>> x = symbols('x', real=True)
    >>> metric = Metric1D(x**2, x)
    >>> lb_symbol = laplace_beltrami(metric)
    >>> op = PseudoDifferentialOperator(
    ...     lb_symbol['full'], [x], mode='symbol'
    ... )
    """
    return metric.laplace_beltrami_symbol()


def sturm_liouville_reduce(metric, potential_expr=None):
    """
    Reduce Laplace-Beltrami to Sturm-Liouville form.
    
    The eigenvalue problem:
        -Δg u + V(x) u = λ u
    
    can be written in Sturm-Liouville form:
        -(p(x) u')' + q(x) u = λ w(x) u
    
    Parameters
    ----------
    metric : Metric1D
        Riemannian metric.
    potential_expr : sympy expression, optional
        Potential function V(x).
    
    Returns
    -------
    dict
        Dictionary with 'p', 'q', 'w' (coefficient functions).
    
    Notes
    -----
    For the Laplace-Beltrami:
        p(x) = √g g¹¹
        q(x) = V(x) √g
        w(x) = √g
    
    Examples
    --------
    >>> x = symbols('x', real=True)
    >>> metric = Metric1D(x**2, x)
    >>> sl = sturm_liouville_reduce(metric)
    >>> print(sl['p'])
    1/x
    """
    x = metric.var_x
    sqrt_g = metric.sqrt_det_expr
    g_inv = metric.g_inv_expr
    
    p_expr = simplify(sqrt_g * g_inv)
    w_expr = sqrt_g
    
    if potential_expr is None:
        q_expr = sympify(0)
    else:
        q_expr = simplify(potential_expr * sqrt_g)
    
    return {
        'p': p_expr,
        'q': q_expr,
        'w': w_expr,
        'p_func': lambdify(x, p_expr, 'numpy'),
        'q_func': lambdify(x, q_expr, 'numpy'),
        'w_func': lambdify(x, w_expr, 'numpy')
    }


def visualize_geodesics(metric, initial_conditions, tspan, x_range=None, 
                        colorby='speed', n_steps=500):
    """
    Visualize geodesics on 1D manifold with curvature coloring.
    
    Parameters
    ----------
    metric : Metric1D
        Riemannian metric.
    initial_conditions : list of tuples
        List of (x0, v0) initial conditions.
    tspan : tuple
        Time interval.
    x_range : tuple, optional
        Plot range for x-axis.
    colorby : {'speed', 'curvature', 'time'}
        Coloring scheme for trajectories.
    n_steps : int
        Number of integration steps.
    
    Examples
    --------
    >>> x = symbols('x', real=True)
    >>> metric = Metric1D(1 + x**2, x)
    >>> ics = [(0, 1), (0, 2), (1, 1)]
    >>> visualize_geodesics(metric, ics, (0, 10))
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot metric
    if x_range is None:
        x_vals_all = []
        for x0, v0 in initial_conditions:
            traj = geodesic_integrator(metric, x0, v0, tspan, n_steps=n_steps)
            x_vals_all.extend(traj['x'])
        x_min, x_max = np.min(x_vals_all), np.max(x_vals_all)
        x_range = (x_min - 0.5, x_max + 0.5)
    
    x_plot = np.linspace(x_range[0], x_range[1], 200)
    g_plot = metric.g_func(x_plot)
    
    ax1.plot(x_plot, g_plot, 'k-', linewidth=2, label='g₁₁(x)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('g₁₁(x)')
    ax1.set_title('Metric Component')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot geodesics
    for idx, (x0, v0) in enumerate(initial_conditions):
        traj = geodesic_integrator(metric, x0, v0, tspan, n_steps=n_steps)
        
        if colorby == 'speed':
            colors = np.abs(traj['v'])
            label = f'IC: x₀={x0:.2f}, v₀={v0:.2f}'
        elif colorby == 'time':
            colors = traj['t']
            label = f'IC: x₀={x0:.2f}, v₀={v0:.2f}'
        elif colorby == 'curvature':
            Gamma = metric.christoffel_func(traj['x'])
            colors = np.abs(Gamma)
            label = f'IC: x₀={x0:.2f}, v₀={v0:.2f}'
        else:
            colors = None
            label = f'IC: x₀={x0:.2f}, v₀={v0:.2f}'
        
        if colors is not None:
            scatter = ax2.scatter(traj['t'], traj['x'], c=colors, 
                                 s=10, cmap='viridis', alpha=0.6)
        else:
            ax2.plot(traj['t'], traj['x'], alpha=0.7, label=label)
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('x(t)')
    ax2.set_title('Geodesic Trajectories')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    if colorby and colors is not None:
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label(colorby.capitalize())
    
    plt.tight_layout()
    plt.show()


def test_metric_geometry_and_sl():
    """
    Test geometric quantities and Sturm-Liouville reduction.
    Metric: g(x) = x^2 (Polar radial part-like or cone)
    """
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(x**2, x)
    
    # 1. Check Inverse and Determinant
    # g_inv = 1/x^2, sqrt_det = x
    assert simplify(metric.g_inv_expr - x**(-2)) == 0
    assert simplify(metric.sqrt_det_expr - x) == 0
    
    # 2. Check Christoffel Symbol
    # Γ = 0.5 * (ln(x^2))' = 0.5 * 2/x = 1/x
    assert simplify(metric.christoffel_expr - 1/x) == 0
    
    # 3. Check Sturm-Liouville Reduction
    # p = √g * g^11 = x * 1/x^2 = 1/x
    # w = √g = x
    sl = sturm_liouville_reduce(metric)
    assert simplify(sl['p'] - 1/x) == 0
    assert simplify(sl['w'] - x) == 0
    
    print("✓ Geometric properties and Sturm-Liouville test passed")


def test_integration_methods_consistency():
    """
    Test consistency between symbolic and numerical integration for volume.
    Metric: g(x) = 1/x^2 (Hyperbolic)
    Interval: [1, e] -> Volume should be 1.
    """
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(1/x**2, x)
    
    # Symbolic: ∫(1/x) dx from 1 to e = ln(e) - ln(1) = 1
    vol_sym = metric.riemannian_volume(1, np.e, method='symbolic')
    
    # Numerical
    vol_num = metric.riemannian_volume(1, np.e, method='numerical')
    
    assert abs(float(vol_sym) - 1.0) < 1e-9
    assert abs(vol_num - 1.0) < 1e-5
    
    print("✓ Integration methods consistency test passed")


def test_geodesic_integrators_accuracy():
    """
    Test accuracy of different geodesic integrator backends (RK4, Adaptive, Symplectic).
    Metric: Flat g(x) = 1
    Exact solution: x(t) = x0 + v0*t
    """
    x = symbols('x', real=True)
    metric = Metric1D(1, x)
    
    x0, v0 = 0.0, 2.0
    t_end = 2.0
    expected_x = x0 + v0 * t_end
    
    methods = ['rk4', 'adaptive', 'symplectic']
    
    for method in methods:
        traj = geodesic_integrator(metric, x0, v0, (0, t_end), method=method, n_steps=100)
        final_x = traj['x'][-1]
        
        # Symplectic Euler is first order, RK4 is 4th order, Adaptive is variable
        tol = 1e-1 if method == 'symplectic' else 1e-5
        
        assert abs(final_x - expected_x) < tol, \
            f"Method {method} failed accuracy test. Got {final_x}, expected {expected_x}"

    print("✓ Geodesic integrator backends test passed")


def test_hamiltonian_flow_conservation():
    """
    Test energy conservation in Hamiltonian flow solvers.
    Metric: g(x) = x^2
    Hamiltonian: H = p^2 / (2x^2)
    """
    x, p_sym = symbols('x p', real=True)
    metric = Metric1D(x**2, x)
    
    x0 = 2.0
    p0 = 10.0 # High momentum
    tspan = (0, 10.0)
    n_steps = 2000
    
    # Initial Energy
    H_func = lambdify((x, p_sym), p_sym**2 / (2*x**2))
    E0 = H_func(x0, p0)
    
    solvers = ['verlet', 'stormer', 'symplectic_euler']
    
    for method in solvers:
        res = geodesic_hamiltonian_flow(metric, x0, p0, tspan, method=method, n_steps=n_steps)
        
        energies = res['energy']
        
        # Energy should not drift significantly (bounded oscillation)
        # Verify that the standard deviation of energy is small relative to E0
        energy_variation = np.std(energies) / E0
        
        # Verlet is 2nd order symplectic, Euler is 1st order.
        # Adjusted tolerance: 
        # Symplectic Euler needs ~5% (5e-2)
        # Verlet/Stormer needs ~0.5% (5e-3) given n_steps=2000 and high p0
        tol = 5e-2 if method == 'symplectic_euler' else 5e-3
        
        assert energy_variation < tol, \
            f"Energy drift too high for {method}: {energy_variation:.2e}"
        
        # Ensure we actually moved
        assert abs(res['x'][-1] - x0) > 0.1
        
    print("✓ Hamiltonian energy conservation test passed")


def test_laplace_beltrami_properties():
    """
    Test Laplace-Beltrami symbol generation.
    Metric: g(x) = x^2
    LB Operator: Δ = x^-2 ∂² + (...)∂
    """
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(x**2, x)
    
    lb = metric.laplace_beltrami_symbol()
    
    # Principal symbol g^11 * xi^2 = (1/x^2) * xi^2
    xi = symbols('xi', real=True)
    expected_principal = xi**2 / x**2
    
    assert simplify(lb['principal'] - expected_principal) == 0
    
    # Subprincipal calculation check:
    # d(log√g)/dx * g^11 * xi = d(ln x)/dx * x^-2 * xi = (1/x) * x^-2 * xi = xi / x^3
    expected_sub = xi / x**3
    assert simplify(lb['subprincipal'] - expected_sub) == 0
    
    print("✓ Laplace-Beltrami symbol test passed")


def test_error_handling():
    """Test that invalid inputs raise appropriate errors."""
    x = symbols('x', real=True)
    metric = Metric1D(1, x)
    
    # Invalid volume method
    try:
        metric.riemannian_volume(0, 1, method='monte_carlo')
        assert False, "Should have raised ValueError for invalid method"
    except ValueError:
        pass
    
    # Invalid integrator method
    try:
        geodesic_integrator(metric, 0, 1, (0, 1), method='magic')
        assert False, "Should have raised ValueError for invalid integrator"
    except ValueError:
        pass
        
    print("✓ Error handling test passed")


if __name__ == "__main__":
    print("Running improved riemannian_1d tests...\n")
    test_metric_geometry_and_sl()
    test_integration_methods_consistency()
    test_geodesic_integrators_accuracy()
    test_hamiltonian_flow_conservation()
    test_laplace_beltrami_properties()
    test_error_handling()
    print("\n✓ All improved tests passed")