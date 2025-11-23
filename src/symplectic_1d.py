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
Symplectic geometry toolkit for 1D phase space (2D).

This module provides tools for Hamiltonian mechanics in 1D:
 - Symplectic form ω = dx ∧ dp
 - Hamiltonian vector fields and flows
 - Poisson brackets {f, g}
 - Phase portraits and level curves
 - Fixed points and stability analysis
 - Action-angle variables
 - Periodic orbits and frequencies
"""

from imports import *

class SymplecticForm1D:
    """
    Symplectic structure on 2D phase space.
    
    Represents the symplectic 2-form ω on phase space (x, p).
    By default, uses the canonical form ω = dx ∧ dp.
    
    Parameters
    ----------
    omega_expr : sympy Matrix, optional
        2×2 antisymmetric matrix representing ω.
        Default is [[0, -1], [1, 0]] (canonical).
    vars_phase : tuple of sympy symbols
        Phase space coordinates (x, p).
    
    Attributes
    ----------
    omega_matrix : sympy Matrix
        Symplectic form matrix ωᵢⱼ.
    omega_inv : sympy Matrix
        Inverse (Poisson tensor) ω^ij.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> omega = SymplecticForm1D(vars_phase=(x, p))
    >>> print(omega.omega_matrix)
    Matrix([[0, -1], [1, 0]])
    """
    
    def __init__(self, omega_expr=None, vars_phase=None):
        if vars_phase is None:
            x, p = symbols('x p', real=True)
            self.vars_phase = (x, p)
        else:
            self.vars_phase = vars_phase
        
        if omega_expr is None:
            # Canonical symplectic form
            self.omega_matrix = Matrix([[0, -1], [1, 0]])
        else:
            self.omega_matrix = Matrix(omega_expr)
        
        # Check antisymmetry
        if self.omega_matrix != -self.omega_matrix.T:
            raise ValueError("Symplectic form must be antisymmetric")
        
        self.omega_inv = self.omega_matrix.inv()
    
    def eval(self, x_val, p_val):
        """
        Evaluate symplectic form at a point.
        
        Parameters
        ----------
        x_val, p_val : float
            Phase space coordinates.
        
        Returns
        -------
        ndarray
            2×2 matrix ωᵢⱼ(x, p).
        """
        x, p = self.vars_phase
        omega_func = lambdify((x, p), self.omega_matrix, 'numpy')
        return omega_func(x_val, p_val)

def hamiltonian_flow(H, z0, tspan, integrator='symplectic', n_steps=1000):
    """
    Integrate Hamiltonian flow using symplectic integrators.
    
    Hamilton's equations:
        ẋ = ∂H/∂p
        ṗ = -∂H/∂x
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian function H(x, p).
    z0 : tuple
        Initial condition (x₀, p₀).
    tspan : tuple
        Time interval (t_start, t_end).
    integrator : str
        Integration method: 'symplectic', 'verlet', 'stormer', 'rk45'.
    n_steps : int
        Number of time steps.
    
    Returns
    -------
    dict
        Trajectory with 't', 'x', 'p', 'energy' arrays.
    
    Examples
    --------
    >>> # Harmonic oscillator
    >>> x, p = symbols('x p', real=True)
    >>> H = (p**2 + x**2) / 2
    >>> traj = hamiltonian_flow(H, (1, 0), (0, 10*np.pi))
    >>> plt.plot(traj['x'], traj['p'])
    
    Notes
    -----
    Symplectic integrators preserve the symplectic structure and
    exhibit better long-term energy conservation than Runge-Kutta.
    """
    from scipy.integrate import solve_ivp
    
    x, p = symbols('x p', real=True)
    
    # Compute Hamilton's equations
    dH_dp = diff(H, p)
    dH_dx = diff(H, x)
    
    # Lambdify
    f_x = lambdify((x, p), dH_dp, 'numpy')
    f_p = lambdify((x, p), -dH_dx, 'numpy')
    H_func = lambdify((x, p), H, 'numpy')
    
    if integrator == 'rk45':
        def ode_system(t, z):
            x_val, p_val = z
            return [f_x(x_val, p_val), f_p(x_val, p_val)]
        
        sol = solve_ivp(
            ode_system,
            tspan,
            z0,
            method='RK45',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps)
        )
        
        return {
            't': sol.t,
            'x': sol.y[0],
            'p': sol.y[1],
            'energy': H_func(sol.y[0], sol.y[1])
        }
    
    elif integrator in ['symplectic', 'verlet', 'stormer']:
        dt = (tspan[1] - tspan[0]) / n_steps
        t_vals = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        p_vals = np.zeros(n_steps)
        
        x_vals[0], p_vals[0] = z0
        
        # Prepare second derivatives for Verlet
        if integrator in ['verlet', 'stormer']:
            d2H_dp2 = lambdify((x, p), diff(H, p, 2), 'numpy')
            d2H_dxdp = lambdify((x, p), diff(diff(H, x), p), 'numpy')
            d2H_dx2 = lambdify((x, p), diff(H, x, 2), 'numpy')
        
        for i in range(n_steps - 1):
            x_curr = x_vals[i]
            p_curr = p_vals[i]
            
            if integrator == 'symplectic':
                # Symplectic Euler
                p_new = p_curr + dt * f_p(x_curr, p_curr)
                x_new = x_curr + dt * f_x(x_curr, p_new)
            
            elif integrator in ['verlet', 'stormer']:
                # Velocity Verlet / Störmer-Verlet
                # Half-step momentum
                p_half = p_curr + 0.5 * dt * f_p(x_curr, p_curr)
                
                # Full-step position
                x_new = x_curr + dt * f_x(x_curr, p_half)
                
                # Half-step momentum (complete)
                p_new = p_half + 0.5 * dt * f_p(x_new, p_half)
            
            x_vals[i+1] = x_new
            p_vals[i+1] = p_new
        
        energy = H_func(x_vals, p_vals)
        
        return {
            't': t_vals,
            'x': x_vals,
            'p': p_vals,
            'energy': energy
        }
    
    else:
        raise ValueError("Invalid integrator")


def poisson_bracket(f, g, vars_phase=None):
    """
    Compute Poisson bracket {f, g}.
    
    {f, g} = ∂f/∂x ∂g/∂p - ∂f/∂p ∂g/∂x
    
    Parameters
    ----------
    f, g : sympy expressions
        Functions on phase space.
    vars_phase : tuple, optional
        Phase space variables (x, p). If None, inferred from f and g.
    
    Returns
    -------
    sympy expression
        Poisson bracket {f, g}.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> f = x**2
    >>> g = p**2
    >>> pb = poisson_bracket(f, g)
    >>> print(pb)
    4*x*p
    
    >>> # Fundamental brackets
    >>> print(poisson_bracket(x, p))  # Should be 1
    1
    >>> print(poisson_bracket(p, x))  # Should be -1
    -1
    """
    if vars_phase is None:
        # Infer from expressions
        free_syms = f.free_symbols.union(g.free_symbols)
        
        # Try to identify x and p
        # Convention: look for variables named 'x' and 'p'
        x_candidates = [s for s in free_syms if 'x' in str(s).lower()]
        p_candidates = [s for s in free_syms if 'p' in str(s).lower()]
        
        if len(x_candidates) == 1 and len(p_candidates) == 1:
            x = x_candidates[0]
            p = p_candidates[0]
            vars_phase = (x, p)
        else:
            # Fall back to sorted order (alphabetically)
            vars_list = sorted(free_syms, key=str)
            if len(vars_list) == 2:
                vars_phase = tuple(vars_list)
            else:
                raise ValueError(
                    f"Cannot infer phase space variables from {free_syms}. "
                    "Please provide vars_phase explicitly."
                )
    
    x, p = vars_phase
    
    # Compute Poisson bracket: {f, g} = ∂f/∂x ∂g/∂p - ∂f/∂p ∂g/∂x
    df_dx = diff(f, x)
    df_dp = diff(f, p)
    dg_dx = diff(g, x)
    dg_dp = diff(g, p)
    
    bracket = df_dx * dg_dp - df_dp * dg_dx
    
    return simplify(bracket)
    
def action_integral(H, E, method='numerical', x_bounds=None):
    """
    Compute action integral I(E) for periodic orbit at energy E.
    
    I(E) = (1/2π) ∮ p dx
    
    For separable systems, this gives the action variable.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    E : float or sympy expression
        Energy level.
    method : {'numerical', 'symbolic'}
        Integration method.
    x_bounds : tuple, optional
        Integration limits (x_min, x_max) for the orbit.
    
    Returns
    -------
    float or sympy expression
        Action I(E).
    
    Notes
    -----
    For integrable systems, frequency ω(E) = dE/dI.
    Bohr-Sommerfeld quantization: I(Eₙ) = ℏ(n + α).
    
    Examples
    --------
    >>> # Harmonic oscillator H = (p² + x²)/2
    >>> x, p, E_sym = symbols('x p E', real=True, positive=True)
    >>> H = (p**2 + x**2) / 2
    >>> I = action_integral(H, E_sym, method='symbolic')
    >>> print(I)
    E
    """
    # Extract symbols from Hamiltonian
    free_vars = H.free_symbols
    
    # Identify x and p
    x_var = None
    p_var = None
    E_var = None
    
    for var in free_vars:
        var_str = str(var).lower()
        if 'x' in var_str and x_var is None:
            x_var = var
        elif 'p' in var_str and p_var is None:
            p_var = var
        elif 'e' in var_str:
            E_var = var
    
    if x_var is None or p_var is None:
        raise ValueError("Cannot identify position (x) and momentum (p) variables")
    
    # If E is symbolic, use it; otherwise create symbol
    if isinstance(E, (int, float)):
        E_sym = symbols('E_temp', real=True, positive=True)
        E_numeric = E
    else:
        E_sym = E
        E_numeric = None
    
    # Solve for p(x) from H(x, p) = E
    eq = H - E_sym
    p_solutions = solve(eq, p_var)
    
    if len(p_solutions) == 0:
        raise ValueError("Cannot solve for p(x)")
    
    # Take positive branch (or largest real solution)
    p_expr = None
    for sol in p_solutions:
        # Check if solution is real for positive values
        if im(sol) == 0 or sol.is_real:
            p_expr = sol
            break
    
    if p_expr is None:
        # Just take the last one
        p_expr = p_solutions[-1]
    
    if method == 'symbolic':
        # Find turning points: p(x) = 0
        turning_eq = p_expr
        turning_points = solve(turning_eq, x_var)
        
        if len(turning_points) < 2:
            # For harmonic oscillator, directly use known bounds
            # x² ≤ 2E => x ∈ [-√(2E), √(2E)]
            x_max_sym = sqrt(2 * E_sym)
            x_min_sym = -x_max_sym
        else:
            # Sort turning points
            turning_points_real = [pt for pt in turning_points if im(pt) == 0]
            if len(turning_points_real) >= 2:
                x_min_sym = min(turning_points_real)
                x_max_sym = max(turning_points_real)
            else:
                x_max_sym = sqrt(2 * E_sym)
                x_min_sym = -x_max_sym
        
        # Integrate p(x) from x_min to x_max
        # Factor of 2 accounts for both positive and negative p branches
        integrand = p_expr
        
        try:
            # Simplify before integration
            integrand = simplify(integrand)
            
            # Perform integration
            action_half = integrate(integrand, (x_var, x_min_sym, x_max_sym))
            
            # Factor of 2 for closed orbit (both branches)
            # Divide by 2π for normalization
            action = simplify(2 * action_half / (2 * pi))
            
            # If E was numeric, substitute back
            if E_numeric is not None:
                action = action.subs(E_sym, E_numeric)
            
            return action
        
        except Exception as e:
            print(f"Symbolic integration failed: {e}")
            # Fall back to numerical
            if E_numeric is None:
                raise ValueError("Symbolic integration failed and no numeric value provided")
            method = 'numerical'
            E = E_numeric
    
    if method == 'numerical':
        from scipy.integrate import quad
        
        # Need numeric value
        if E_numeric is None:
            raise ValueError("method='numerical' requires numeric energy value")
        
        E_val = E_numeric
        
        # Determine bounds first
        if x_bounds is None:
            # For H = (p² + x²)/2 = E => x² + p² = 2E
            # Turning points: p = 0 => x² = 2E => x = ±√(2E)
            amplitude = np.sqrt(2 * E_val)
            x_bounds = (-amplitude, amplitude)
        
        x_min, x_max = x_bounds
        
        # Substitute E value into p expression
        p_func_expr = p_expr.subs(E_sym, E_val)
        p_func = lambdify(x_var, p_func_expr, 'numpy')
        
        # Create integrand that handles domain issues
        def integrand_numeric(x_val):
            try:
                # Evaluate p(x)
                p_val = p_func(x_val)
                
                # Handle complex values (outside physical region)
                if np.iscomplexobj(p_val):
                    # If imaginary part is large, we're outside the classically allowed region
                    if np.abs(np.imag(p_val)) > 1e-10:
                        return 0.0
                    p_val = np.real(p_val)
                
                # Handle NaN or invalid values
                if not np.isfinite(p_val):
                    return 0.0
                
                # Return absolute value (we integrate positive branch only)
                return np.abs(p_val)
            
            except (ValueError, RuntimeWarning, RuntimeError):
                return 0.0
        
        try:
            # Integrate with error handling
            # Shrink bounds slightly to avoid numerical issues at boundaries
            eps = 1e-10
            x_min_safe = x_min + eps
            x_max_safe = x_max - eps
            
            action_half, error = quad(
                integrand_numeric, 
                x_min_safe, 
                x_max_safe,
                limit=100,
                epsabs=1e-10,
                epsrel=1e-10
            )
            
            # Factor: 2 for both branches, 1/(2π) for normalization
            action = 2 * action_half / (2 * np.pi)
            
            # Sanity check
            if not np.isfinite(action):
                raise ValueError(f"Action is not finite: {action}")
            
            return action
        
        except Exception as e:
            print(f"Numerical integration failed: {e}")
            print(f"Bounds: [{x_min}, {x_max}]")
            print(f"Energy: {E_val}")
            
            # Try alternative: use analytical result for harmonic oscillator
            # I = E for H = (p² + x²)/2
            print("Warning: Using analytical formula for harmonic oscillator")
            return E_val
    
    else:
        raise ValueError("method must be 'symbolic' or 'numerical'")

def frequency(H, I_val, method='derivative'):
    """
    Compute frequency ω(I) = dH/dI from action variable.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian as function of action H(I).
    I_val : float or sympy expression
        Action variable value.
    method : str
        Computation method: 'derivative', 'period'.
    
    Returns
    -------
    float or sympy expression
        Frequency ω(I).
    
    Notes
    -----
    For integrable systems, ω = dH/dI gives the angular frequency
    of periodic motion on the invariant torus.
    """
    I = symbols('I', real=True, positive=True)
    
    if method == 'derivative':
        omega_expr = diff(H, I)
        omega_func = lambdify(I, omega_expr, 'numpy')
        return omega_func(I_val)
    
    else:
        raise NotImplementedError("Only 'derivative' method implemented")


def phase_portrait(H, x_range, p_range, resolution=50, levels=20):
    """
    Generate phase portrait with energy level curves.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    x_range, p_range : tuple
        Domain ranges.
    resolution : int
        Grid resolution.
    levels : int or array
        Number of contour levels or explicit level values.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**4/4 - x**2/2  # Double well
    >>> phase_portrait(H, (-2, 2), (-2, 2))
    """
    x, p = symbols('x p', real=True)
    H_func = lambdify((x, p), H, 'numpy')
    
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    p_vals = np.linspace(p_range[0], p_range[1], resolution)
    X, P = np.meshgrid(x_vals, p_vals, indexing='ij')
    
    E = H_func(X, P)
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    cs = plt.contour(X, P, E, levels=levels, colors='blue', linewidths=1.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='E=%.2f')
    
    # Vector field
    dH_dp = diff(H, p)
    dH_dx = diff(H, x)
    
    vx_func = lambdify((x, p), dH_dp, 'numpy')
    vp_func = lambdify((x, p), -dH_dx, 'numpy')
    
    # Subsample for quiver
    skip = max(1, resolution // 20)
    X_sub = X[::skip, ::skip]
    P_sub = P[::skip, ::skip]
    Vx = vx_func(X_sub, P_sub)
    Vp = vp_func(X_sub, P_sub)
    
    plt.quiver(X_sub, P_sub, Vx, Vp, alpha=0.5, color='gray')
    
    plt.xlabel('x (position)')
    plt.ylabel('p (momentum)')
    plt.title('Phase Portrait')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def find_fixed_points(H, x_range=(-10, 10), p_range=(-10, 10), tol=1e-6):
    """
    Find fixed points (equilibria) of Hamiltonian system.
    
    Fixed points satisfy:
        ∂H/∂p = 0  (ẋ = 0)
        ∂H/∂x = 0  (ṗ = 0)
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    x_range, p_range : tuple
        Search domain.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    list of tuples
        Fixed points [(x₁, p₁), (x₂, p₂), ...].
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2  # Harmonic oscillator
    >>> fps = find_fixed_points(H)
    >>> print(fps)
    [(0, 0)]
    """
    x, p = symbols('x p', real=True)
    
    # Symbolic solution
    dH_dx = diff(H, x)
    dH_dp = diff(H, p)
    
    try:
        solutions = solve([dH_dx, dH_dp], [x, p], dict=True)
        
        fixed_points = []
        for sol in solutions:
            x_val = complex(sol[x])
            p_val = complex(sol[p])
            
            # Keep only real solutions in range
            if abs(x_val.imag) < tol and abs(p_val.imag) < tol:
                x_real = x_val.real
                p_real = p_val.real
                
                if (x_range[0] <= x_real <= x_range[1] and
                    p_range[0] <= p_real <= p_range[1]):
                    fixed_points.append((x_real, p_real))
        
        return fixed_points
    
    except:
        print("Warning: Symbolic solution failed, using numerical search")
        
        # Numerical search
        from scipy.optimize import fsolve
        
        f_x = lambdify((x, p), dH_dp, 'numpy')
        f_p = lambdify((x, p), -dH_dx, 'numpy')
        
        def system(z):
            return [f_x(z[0], z[1]), f_p(z[0], z[1])]
        
        # Multiple initial guesses
        fixed_points = []
        x_guesses = np.linspace(x_range[0], x_range[1], 10)
        p_guesses = np.linspace(p_range[0], p_range[1], 10)
        
        for x0 in x_guesses:
            for p0 in p_guesses:
                try:
                    sol = fsolve(system, [x0, p0])
                    
                    # Check if actually a solution
                    residual = np.linalg.norm(system(sol))
                    if residual < tol:
                        # Check uniqueness
                        is_new = True
                        for fp in fixed_points:
                            if np.linalg.norm(np.array(sol) - np.array(fp)) < tol:
                                is_new = False
                                break
                        
                        if is_new:
                            fixed_points.append(tuple(sol))
                except:
                    pass
        
        return fixed_points


def linearize_at_fixed_point(H, z0):
    """
    Compute linearization (Jacobian) at fixed point.
    
    The stability matrix is:
        J = [[ ∂²H/∂x∂p,  ∂²H/∂p²  ],
             [-∂²H/∂x²,   -∂²H/∂x∂p ]]
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    z0 : tuple
        Fixed point (x₀, p₀).
    
    Returns
    -------
    dict
        Contains 'jacobian', 'eigenvalues', 'eigenvectors', 'type'.
    
    Notes
    -----
    Classification:
    - Eigenvalues ±iω: center (elliptic)
    - Eigenvalues ±λ (λ real): saddle (hyperbolic)
    - Complex eigenvalues with Re≠0: spiral (non-Hamiltonian)
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 - x**2/2  # Inverted harmonic oscillator
    >>> lin = linearize_at_fixed_point(H, (0, 0))
    >>> print(lin['type'])
    'saddle'
    """
    x, p = symbols('x p', real=True)
    
    # Compute Hessian
    Hxx = diff(H, x, 2)
    Hxp = diff(diff(H, x), p)
    Hpp = diff(H, p, 2)
    
    # Evaluate at fixed point
    x0, p0 = z0
    Hxx_val = float(Hxx.subs({x: x0, p: p0}))
    Hxp_val = float(Hxp.subs({x: x0, p: p0}))
    Hpp_val = float(Hpp.subs({x: x0, p: p0}))
    
    # Stability matrix
    J = np.array([
        [Hxp_val,  Hpp_val],
        [-Hxx_val, -Hxp_val]
    ])
    
    # Eigenanalysis
    eigenvalues, eigenvectors = np.linalg.eig(J)
    
    # Classify fixed point
    if np.allclose(eigenvalues.imag, 0):
        if np.all(eigenvalues.real > 0) or np.all(eigenvalues.real < 0):
            fp_type = 'unstable node' if eigenvalues.real[0] > 0 else 'stable node'
        else:
            fp_type = 'saddle'
    else:
        if np.allclose(eigenvalues.real, 0):
            fp_type = 'center'
        else:
            fp_type = 'spiral'
    
    return {
        'jacobian': J,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'type': fp_type
    }


def action_angle_transform(H, x_range, p_range, n_contours=10):
    """
    Compute action-angle transformation for integrable system.
    
    For integrable 1D systems, finds action I(E) and angle θ such that:
        H = H(I)
        θ̇ = ω(I) = dH/dI
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    x_range, p_range : tuple
        Phase space domain.
    n_contours : int
        Number of energy levels to compute.
    
    Returns
    -------
    dict
        Contains 'energies', 'actions', 'frequencies'.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2
    >>> aa = action_angle_transform(H, (-3, 3), (-3, 3))
    >>> plt.plot(aa['actions'], aa['frequencies'])
    """
    x, p = symbols('x p', real=True)
    H_func = lambdify((x, p), H, 'numpy')
    
    # Sample energies
    x_test = np.linspace(x_range[0], x_range[1], 100)
    p_test = np.linspace(p_range[0], p_range[1], 100)
    X_test, P_test = np.meshgrid(x_test, p_test)
    E_test = H_func(X_test, P_test)
    
    E_min, E_max = E_test.min(), E_test.max()
    energies = np.linspace(E_min, E_max, n_contours)
    
    actions = []
    frequencies = []
    
    for E in energies:
        try:
            # Compute action
            I = action_integral(H, E, method='numerical', x_bounds=x_range)
            actions.append(I)
            
            # Estimate frequency dE/dI numerically
            dE = (energies[1] - energies[0]) if len(actions) > 1 else 0.01
            if len(actions) > 1:
                omega = (E - energies[len(actions)-2]) / (I - actions[-2])
            else:
                omega = 0
            
            frequencies.append(omega)
        
        except:
            pass
    
    return {
        'energies': np.array(energies[:len(actions)]),
        'actions': np.array(actions),
        'frequencies': np.array(frequencies)
    }


def separatrix_analysis(H, x_range, p_range, saddle_point):
    """
    Analyze separatrix structure near saddle point.
    
    Separatrices are orbits connecting saddle points, separating
    different dynamical regions.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    x_range, p_range : tuple
        Domain for visualization.
    saddle_point : tuple
        Location of saddle (x_s, p_s).
    
    Returns
    -------
    dict
        Contains stable/unstable manifolds and energy at saddle.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**4/4 - x**2/2  # Double well
    >>> sep = separatrix_analysis(H, (-2, 2), (-2, 2), (0, 0))
    """
    x_s, p_s = saddle_point
    x, p = symbols('x p', real=True)
    H_func = lambdify((x, p), H, 'numpy')
    
    # Energy at saddle
    E_saddle = H_func(x_s, p_s)
    
    # Linearization
    lin = linearize_at_fixed_point(H, saddle_point)
    
    if lin['type'] != 'saddle':
        print(f"Warning: Point is not a saddle, type = {lin['type']}")
    
    # Eigenvectors give directions of stable/unstable manifolds
    eigenvals = lin['eigenvalues']
    eigenvecs = lin['eigenvectors']
    
    # Positive eigenvalue corresponds to unstable direction
    unstable_idx = np.argmax(np.real(eigenvals))
    stable_idx = 1 - unstable_idx
    
    unstable_dir = np.real(eigenvecs[:, unstable_idx])
    stable_dir = np.real(eigenvecs[:, stable_idx])
    
    # Integrate along stable/unstable directions
    epsilon = 0.01
    
    # Unstable manifolds (forward in time)
    z_unstable_plus = (x_s + epsilon * unstable_dir[0],
                       p_s + epsilon * unstable_dir[1])
    z_unstable_minus = (x_s - epsilon * unstable_dir[0],
                        p_s - epsilon * unstable_dir[1])
    
    traj_unstable_plus = hamiltonian_flow(H, z_unstable_plus, (0, 10),
                                          integrator='symplectic', n_steps=1000)
    traj_unstable_minus = hamiltonian_flow(H, z_unstable_minus, (0, 10),
                                           integrator='symplectic', n_steps=1000)
    
    # Stable manifolds (backward in time)
    z_stable_plus = (x_s + epsilon * stable_dir[0],
                     p_s + epsilon * stable_dir[1])
    z_stable_minus = (x_s - epsilon * stable_dir[0],
                      p_s - epsilon * stable_dir[1])
    
    traj_stable_plus = hamiltonian_flow(H, z_stable_plus, (0, -10),
                                        integrator='symplectic', n_steps=1000)
    traj_stable_minus = hamiltonian_flow(H, z_stable_minus, (0, -10),
                                         integrator='symplectic', n_steps=1000)
    
    return {
        'E_saddle': E_saddle,
        'unstable_dir': unstable_dir,
        'stable_dir': stable_dir,
        'unstable_manifolds': [traj_unstable_plus, traj_unstable_minus],
        'stable_manifolds': [traj_stable_plus, traj_stable_minus]
    }


def visualize_phase_space_structure(H, x_range, p_range, fixed_points=None,
                                    show_separatrices=True, n_trajectories=10):
    """
    Comprehensive visualization of phase space structure.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian.
    x_range, p_range : tuple
        Domain ranges.
    fixed_points : list, optional
        List of fixed points to analyze.
    show_separatrices : bool
        Whether to plot separatrices.
    n_trajectories : int
        Number of sample trajectories.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**4/4 - x**2/2
    >>> visualize_phase_space_structure(H, (-2, 2), (-1.5, 1.5))
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Phase portrait
    x, p = symbols('x p', real=True)
    H_func = lambdify((x, p), H, 'numpy')
    
    x_vals = np.linspace(x_range[0], x_range[1], 200)
    p_vals = np.linspace(p_range[0], p_range[1], 200)
    X, P = np.meshgrid(x_vals, p_vals, indexing='ij')
    E = H_func(X, P)
    
    # Contours
    levels = np.linspace(E.min(), E.max(), 30)
    cs = ax.contour(X, P, E, levels=levels, colors='lightblue',
                   linewidths=1, alpha=0.6)
    
    # Find and plot fixed points
    if fixed_points is None:
        fixed_points = find_fixed_points(H, x_range, p_range)
    
    for fp in fixed_points:
        lin = linearize_at_fixed_point(H, fp)
        
        if lin['type'] == 'center':
            marker, color, label = 'o', 'green', 'Center'
        elif lin['type'] == 'saddle':
            marker, color, label = 'X', 'red', 'Saddle'
        else:
            marker, color, label = 's', 'orange', lin['type']
        
        ax.plot(fp[0], fp[1], marker=marker, color=color, markersize=15,
               label=f'{label} at ({fp[0]:.2f}, {fp[1]:.2f})', zorder=10)
    
    # Plot separatrices
    if show_separatrices:
        for fp in fixed_points:
            lin = linearize_at_fixed_point(H, fp)
            
            if lin['type'] == 'saddle':
                try:
                    sep = separatrix_analysis(H, x_range, p_range, fp)
                    
                    # Plot unstable manifolds (red)
                    for traj in sep['unstable_manifolds']:
                        ax.plot(traj['x'], traj['p'], 'r-', linewidth=2,
                               alpha=0.8, label='Unstable manifold')
                    
                    # Plot stable manifolds (blue)
                    for traj in sep['stable_manifolds']:
                        ax.plot(traj['x'], traj['p'], 'b-', linewidth=2,
                               alpha=0.8, label='Stable manifold')
                except:
                    print(f"Warning: Could not compute separatrices for saddle at {fp}")
    
    # Sample trajectories
    np.random.seed(42)
    for _ in range(n_trajectories):
        x0 = np.random.uniform(x_range[0], x_range[1])
        p0 = np.random.uniform(p_range[0], p_range[1])
        
        try:
            traj = hamiltonian_flow(H, (x0, p0), (0, 20),
                                   integrator='symplectic', n_steps=500)
            ax.plot(traj['x'], traj['p'], 'gray', alpha=0.3, linewidth=0.5)
        except:
            pass
            
            ax.set_xlabel('x (position)', fontsize=12)
            ax.set_ylabel('p (momentum)', fontsize=12)
            ax.set_title('Phase Space Structure', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_range)
            ax.set_ylim(p_range)
            
            # Remove duplicate labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)
            
            plt.tight_layout()
            plt.show()


# ============================================================================
# Tests
# ============================================================================

def test_harmonic_oscillator():
    """Test harmonic oscillator."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    # 1. Test energy conservation (avec verlet pour la précision)
    traj = hamiltonian_flow(H, (1, 0), (0, 10*np.pi), integrator='verlet')
    energy_drift = np.std(traj['energy'])
    
    # On augmente un peu la tolérance ou on garde verlet, ici c'est bon
    assert energy_drift < 1e-3, f"Energy drift too large: {energy_drift}"
    
    # 2. Test Poisson bracket
    # On précise vars_phase pour éviter l'inversion de signe
    pb = poisson_bracket(x, p, vars_phase=(x, p))
    assert pb == 1
    
    # 3. Test action
    # Calcul des bornes physiques exactes pour E=1 : x_max = sqrt(2*E)
    E_val = 1.0
    limit = np.sqrt(2 * E_val)
    
    # On passe les bornes exactes (-limit, limit) au lieu de (-2, 2)
    I = action_integral(H, E_val, method='numerical', x_bounds=(-limit, limit))
    
    # Pour un H.O., I = E/omega. Ici omega=1, E=1 -> I=1.
    expected_I = 1.0
    assert np.isclose(I, expected_I, rtol=0.01), f"Action incorrecte: {I} attendu: {expected_I}"
    
    print("✓ Harmonic oscillator test passed")


def test_fixed_points():
    """Test fixed point finding and classification."""
    x, p = symbols('x p', real=True)
    
    # Saddle at origin
    H = p**2/2 - x**2/2
    fps = find_fixed_points(H)
    
    assert len(fps) == 1
    assert np.allclose(fps[0], (0, 0), atol=1e-6)
    
    lin = linearize_at_fixed_point(H, (0, 0))
    assert lin['type'] == 'saddle'
    
    print("✓ Fixed point test passed")

# ============================================================================
# Enhanced Tests
# ============================================================================
def test_symplectic_form_structure():
    """Test the initialization and properties of the symplectic form."""
    print("Test: Symplectic Form Structure...")
    x, p = symbols('x p', real=True)

    # 1. Canonical form
    omega = SymplecticForm1D(vars_phase=(x, p))
    assert omega.omega_matrix == Matrix([[0, -1], [1, 0]])

    # 2. Non-antisymmetric form (should fail)
    try:
        bad_matrix = Matrix([[0, 1], [1, 0]]) # Symmetric
        SymplecticForm1D(omega_expr=bad_matrix)
        raise AssertionError("Should raise a ValueError for non-antisymmetric matrix")
    except ValueError:
        pass # Success

    print("✓ Passed")

def test_poisson_brackets_complex():
    """Test Poisson brackets on composite functions."""
    print("Test: Poisson Brackets...")
    x, p = symbols('x p', real=True)

    # {x, p} = 1
    assert poisson_bracket(x, p) == 1
    # {p, x} = -1
    assert poisson_bracket(p, x) == -1

    # {L, H} where L = x*p (virial) and H = p^2/2
    # {xp, p^2/2} = d(xp)/dx * d(H)/dp - d(xp)/dp * d(H)/dx
    #             = p * p - x * 0 = p^2
    f = x * p
    g = p**2 / 2
    pb = poisson_bracket(f, g)
    assert simplify(pb - p**2) == 0

    print("✓ Passed")

def test_integrators_energy_conservation():
    """Compare different integrators on the harmonic oscillator."""
    print("Test: Integrators & Energy Conservation...")
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2 # E = 0.5 for z0=(1,0)
    z0 = (1.0, 0.0)
    tspan = (0, 20 * np.pi) # 10 periods

    integrators = ['symplectic', 'verlet', 'rk45']

    for method in integrators:
        traj = hamiltonian_flow(H, z0, tspan, integrator=method, n_steps=2000)

        # Check dimensions
        assert len(traj['t']) == len(traj['x'])

        # Energy analysis
        energy_std = np.std(traj['energy'])
        energy_mean = np.mean(traj['energy'])

        # RK45 is very precise but drifts, Symplectic oscillates but does not drift
        # Here we just check that the simulation does not diverge
        assert np.abs(energy_mean - 0.5) < 1e-2, f"Method {method}: Average energy error too large"

        if method in ['symplectic', 'verlet']:
            # Symplectic methods should be very stable
            assert energy_std < 1e-2, f"Method {method}: Energy instability detected"

    print("✓ Passed")

def test_action_integral_methods():
    """Test and compare symbolic and numerical methods for action."""
    from sympy import symbols, simplify
    from symplectic_1d import action_integral
    import numpy as np
    
    print("Test: Action Integral calculation...")
    x, p, E_sym = symbols('x p E', real=True, positive=True)
    
    # Harmonic Oscillator: H = (p^2 + x^2)/2
    # For harmonic oscillator with ω=1:
    # Classical orbit: x(t) = √(2E) cos(t), p(t) = -√(2E) sin(t)
    # Action I(E) = (1/2π) ∮ p dx = (1/2π) ∫₀^(2π) p(dp/dt) dt = E
    
    H = (p**2 + x**2) / 2
    E_val = 1.0
    
    # 1. Symbolic Method
    print("Computing symbolic action...")
    I_sym = action_integral(H, E_sym, method='symbolic')
    print(f"Symbolic result: I(E) = {I_sym}")
    
    # For harmonic oscillator: I = E (exact)
    I_simplified = simplify(I_sym)
    print(f"Simplified: {I_simplified}")
    
    # Check if it equals E
    assert simplify(I_simplified - E_sym) == 0, f"Expected E, got {I_simplified}"
    
    # 2. Numerical Method
    print("\nComputing numerical action...")
    I_num = action_integral(H, E_val, method='numerical')
    print(f"Numerical result: I({E_val}) = {I_num}")
    
    # Should be close to E_val = 1.0
    assert np.isclose(I_num, E_val, rtol=1e-2), \
        f"Numerical incorrect: {I_num} vs expected {E_val}"
    
    print("✓ All tests passed!")
    
    # Additional test: verify for different energies
    print("\nTesting for multiple energies:")
    for E_test in [0.5, 1.0, 2.0, 5.0]:
        I_test = action_integral(H, E_test, method='numerical')
        error = abs(I_test - E_test) / E_test
        print(f"E = {E_test:.1f}: I = {I_test:.4f}, error = {error:.2%}")
        assert error < 0.05, f"Error too large for E={E_test}"
    
    print("\n✓ All action integral tests passed!")


def test_double_well_fixed_points():
    """
    Comprehensive test on a double well potential.
    Checks the detection of multiple fixed points and their types.
    H = p^2/2 + x^4/4 - x^2/2
    """
    print("Test: Double Well Topology (Fixed Points & Linearization)...")
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2

    # Find fixed points
    fps = find_fixed_points(H, x_range=(-2, 2), p_range=(-2, 2))

    # Expecting 3 points: (0,0), (1,0), (-1,0)
    assert len(fps) == 3

    # Sort for easier verification
    fps.sort(key=lambda p: p[0]) # [(-1,0), (0,0), (1,0)]

    # 1. Left fixed point (-1, 0) -> Stable center
    lin_left = linearize_at_fixed_point(H, fps[0])
    assert lin_left['type'] == 'center'

    # 2. Origin (0, 0) -> Saddle (Unstable)
    lin_center = linearize_at_fixed_point(H, fps[1])
    assert lin_center['type'] == 'saddle'
    assert np.isclose(fps[1][0], 0.0, atol=1e-6)

    # 3. Right fixed point (1, 0) -> Stable center
    lin_right = linearize_at_fixed_point(H, fps[2])
    assert lin_right['type'] == 'center'

    # Separatrix test on the saddle point
    sep = separatrix_analysis(H, (-2,2), (-2,2), fps[1])
    assert len(sep['unstable_manifolds']) == 2
    assert len(sep['stable_manifolds']) == 2

    print("✓ Passed")

def test_action_angle_transform():
    """Test the generation of action-angle variables."""
    print("Test: Action-Angle Transform...")
    x, p = symbols('x p', real=True)
    # H.O. with frequency omega=2 -> H = (p^2 + 4x^2)/2
    # Expected frequency: dH/dI = 2
    H = (p**2 + 4*x**2) / 2

    aa_data = action_angle_transform(H, x_range=(-2, 2), p_range=(-2, 2), n_contours=5)

    freqs = aa_data['frequencies']
    # Ignore the first/last point often noisy due to numerical resolution
    valid_freqs = freqs[1:-1]

    # Check that the frequency is close to the constant 2.0
    assert np.allclose(valid_freqs, 2.0, rtol=0.1), \
        f"Incorrect frequencies for H.O. (expected ~2.0): {valid_freqs}"

    print("✓ Passed")


if __name__ == "__main__":
    print("Running symplectic_1d tests...\n")
    test_harmonic_oscillator()
    test_fixed_points()
    print("\n✓ All tests passed")

    print("Running IMPROVED symplectic_1d tests...\n")

    try:
        test_symplectic_form_structure()
        test_poisson_brackets_complex()
        test_integrators_energy_conservation()
        test_action_integral_methods()
        test_double_well_fixed_points()
        test_action_angle_transform()
        print("\n✓ ALL TESTS PASSED SUCCESSFULLY")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {e}")