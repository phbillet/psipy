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
Symplectic geometry toolkit for 2D phase space (4D).

This module provides tools for Hamiltonian mechanics in 4D phase space:
 - Symplectic form ω = dx₁∧dp₁ + dx₂∧dp₂
 - Hamiltonian flows in 4D
 - Poincaré sections and return maps
 - Monodromy matrix and Floquet analysis
 - Lyapunov exponents
 - Invariant tori and resonances
 - KAM theory applications
"""

from imports import *

class SymplecticForm2D:
    """
    Symplectic structure on 4D phase space.
    
    Represents the symplectic 2-form ω on phase space (x₁, p₁, x₂, p₂).
    By default, uses canonical form ω = dx₁∧dp₁ + dx₂∧dp₂.
    
    Parameters
    ----------
    omega_matrix : 4×4 sympy Matrix, optional
        Antisymmetric matrix representing ω.
    vars_phase : tuple of sympy symbols
        Phase space coordinates (x₁, p₁, x₂, p₂).
    
    Examples
    --------
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> omega = SymplecticForm2D(vars_phase=(x1, p1, x2, p2))
    """
    
    def __init__(self, omega_matrix=None, vars_phase=None):
        if vars_phase is None:
            x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
            self.vars_phase = (x1, p1, x2, p2)
        else:
            self.vars_phase = vars_phase
        
        if omega_matrix is None:
            # Canonical symplectic form
            self.omega_matrix = Matrix([
                [0, -1,  0,  0],
                [1,  0,  0,  0],
                [0,  0,  0, -1],
                [0,  0,  1,  0]
            ])
        else:
            self.omega_matrix = Matrix(omega_matrix)
        
        # Check antisymmetry
        if self.omega_matrix != -self.omega_matrix.T:
            raise ValueError("Symplectic form must be antisymmetric")
        
        self.omega_inv = self.omega_matrix.inv()


def hamiltonian_flow_4d(H, z0, tspan, integrator='symplectic', n_steps=1000):
    """
    Integrate Hamiltonian flow in 4D phase space.
    
    Hamilton's equations:
        ẋᵢ = ∂H/∂pᵢ
        ṗᵢ = -∂H/∂xᵢ
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x₁, p₁, x₂, p₂).
    z0 : tuple or array
        Initial condition (x₁, p₁, x₂, p₂).
    tspan : tuple
        Time interval (t_start, t_end).
    integrator : str
        Integration method: 'symplectic', 'verlet', 'rk45'.
    n_steps : int
        Number of time steps.
    
    Returns
    -------
    dict
        Trajectory with 't', 'x1', 'p1', 'x2', 'p2', 'energy' arrays.
    
    Examples
    --------
    >>> # Coupled oscillators
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> H = (p1**2 + p2**2)/2 + (x1**2 + x2**2)/2 + 0.1*x1*x2
    >>> traj = hamiltonian_flow_4d(H, (1, 0, 0.5, 0), (0, 50))
    """
    from scipy.integrate import solve_ivp
    
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    
    # Hamilton's equations
    dH_dp1 = diff(H, p1)
    dH_dp2 = diff(H, p2)
    dH_dx1 = diff(H, x1)
    dH_dx2 = diff(H, x2)
    
    # Lambdify
    f_x1 = lambdify((x1, p1, x2, p2), dH_dp1, 'numpy')
    f_x2 = lambdify((x1, p1, x2, p2), dH_dp2, 'numpy')
    f_p1 = lambdify((x1, p1, x2, p2), -dH_dx1, 'numpy')
    f_p2 = lambdify((x1, p1, x2, p2), -dH_dx2, 'numpy')
    H_func = lambdify((x1, p1, x2, p2), H, 'numpy')
    
    if integrator == 'rk45':
        def ode_system(t, z):
            x1_val, p1_val, x2_val, p2_val = z
            return [
                f_x1(x1_val, p1_val, x2_val, p2_val),
                f_p1(x1_val, p1_val, x2_val, p2_val),
                f_x2(x1_val, p1_val, x2_val, p2_val),
                f_p2(x1_val, p1_val, x2_val, p2_val)
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
            'x1': sol.y[0],
            'p1': sol.y[1],
            'x2': sol.y[2],
            'p2': sol.y[3],
            'energy': H_func(sol.y[0], sol.y[1], sol.y[2], sol.y[3])
        }
    
    elif integrator in ['symplectic', 'verlet']:
        dt = (tspan[1] - tspan[0]) / n_steps
        t_vals = np.linspace(tspan[0], tspan[1], n_steps)
        
        x1_vals = np.zeros(n_steps)
        p1_vals = np.zeros(n_steps)
        x2_vals = np.zeros(n_steps)
        p2_vals = np.zeros(n_steps)
        
        x1_vals[0], p1_vals[0], x2_vals[0], p2_vals[0] = z0
        
        for i in range(n_steps - 1):
            x1_curr = x1_vals[i]
            p1_curr = p1_vals[i]
            x2_curr = x2_vals[i]
            p2_curr = p2_vals[i]
            
            if integrator == 'symplectic':
                # Symplectic Euler
                p1_new = p1_curr + dt * f_p1(x1_curr, p1_curr, x2_curr, p2_curr)
                p2_new = p2_curr + dt * f_p2(x1_curr, p1_curr, x2_curr, p2_curr)
                
                x1_new = x1_curr + dt * f_x1(x1_curr, p1_new, x2_curr, p2_new)
                x2_new = x2_curr + dt * f_x2(x1_curr, p1_new, x2_curr, p2_new)
            
            elif integrator == 'verlet':
                # Velocity Verlet
                p1_half = p1_curr + 0.5 * dt * f_p1(x1_curr, p1_curr, x2_curr, p2_curr)
                p2_half = p2_curr + 0.5 * dt * f_p2(x1_curr, p1_curr, x2_curr, p2_curr)
                
                x1_new = x1_curr + dt * f_x1(x1_curr, p1_half, x2_curr, p2_half)
                x2_new = x2_curr + dt * f_x2(x1_curr, p1_half, x2_curr, p2_half)
                
                p1_new = p1_half + 0.5 * dt * f_p1(x1_new, p1_half, x2_new, p2_half)
                p2_new = p2_half + 0.5 * dt * f_p2(x1_new, p1_half, x2_new, p2_half)
            
            x1_vals[i+1] = x1_new
            p1_vals[i+1] = p1_new
            x2_vals[i+1] = x2_new
            p2_vals[i+1] = p2_new
        
        energy = H_func(x1_vals, p1_vals, x2_vals, p2_vals)
        
        return {
            't': t_vals,
            'x1': x1_vals,
            'p1': p1_vals,
            'x2': x2_vals,
            'p2': p2_vals,
            'energy': energy
        }
    
    else:
        raise ValueError("Invalid integrator")


def poincare_section(H, Sigma_def, z0, tmax, n_returns=1000, 
                     integrator='symplectic'):
    """
    Compute Poincaré section (surface of section).
    
    A Poincaré section Σ is a codimension-1 surface in phase space.
    Records points where trajectory intersects Σ.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x₁, p₁, x₂, p₂).
    Sigma_def : dict
        Section definition with 'variable', 'value', 'direction'.
        Example: {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0 : tuple
        Initial condition.
    tmax : float
        Maximum integration time.
    n_returns : int
        Maximum number of returns to section.
    integrator : str
        Integration method.
    
    Returns
    -------
    dict
        Section points: 't_crossings', 'section_points'.
    
    Examples
    --------
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    >>> section = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    >>> ps = poincare_section(H, section, (1, 0, 0, 0.5), tmax=100)
    """
    # Integrate trajectory
    n_steps = 10000
    traj = hamiltonian_flow_4d(H, z0, (0, tmax), integrator=integrator, 
                               n_steps=n_steps)
    
    # Extract section variable
    var_name = Sigma_def['variable']
    var_values = traj[var_name]
    var_threshold = Sigma_def['value']
    direction = Sigma_def.get('direction', 'positive')
    
    # Find crossings
    crossings = []
    section_points = []
    
    for i in range(len(var_values) - 1):
        v_curr = var_values[i]
        v_next = var_values[i+1]
        
        # Check crossing
        if direction == 'positive':
            crosses = (v_curr < var_threshold) and (v_next >= var_threshold)
        elif direction == 'negative':
            crosses = (v_curr > var_threshold) and (v_next <= var_threshold)
        else:  # 'both'
            crosses = (v_curr - var_threshold) * (v_next - var_threshold) < 0
        
        if crosses:
            # Linear interpolation for crossing time
            alpha = (var_threshold - v_curr) / (v_next - v_curr)
            t_cross = traj['t'][i] + alpha * (traj['t'][i+1] - traj['t'][i])
            
            # Interpolate all variables
            point = {}
            for key in ['x1', 'p1', 'x2', 'p2']:
                point[key] = traj[key][i] + alpha * (traj[key][i+1] - traj[key][i])
            
            crossings.append(t_cross)
            section_points.append(point)
            
            if len(crossings) >= n_returns:
                break
    
    return {
        't_crossings': np.array(crossings),
        'section_points': section_points
    }


def first_return_map(section_points, plot_variables=('x1', 'p1')):
    """
    Compute first return map from Poincaré section.
    
    Maps each section point to the next: Pₙ₊₁ = P(Pₙ).
    
    Parameters
    ----------
    section_points : list of dict
        Points on Poincaré section.
    plot_variables : tuple
        Variables to plot for return map.
    
    Returns
    -------
    dict
        Contains arrays for current and next points.
    
    Examples
    --------
    >>> ps = poincare_section(...)
    >>> rm = first_return_map(ps['section_points'])
    >>> plt.plot(rm['current'], rm['next'], 'o')
    """
    if len(section_points) < 2:
        raise ValueError("Need at least 2 section points")
    
    var1, var2 = plot_variables
    
    current = np.array([[p[var1], p[var2]] for p in section_points[:-1]])
    next_pts = np.array([[p[var1], p[var2]] for p in section_points[1:]])
    
    return {
        'current': current,
        'next': next_pts,
        'variables': plot_variables
    }


def monodromy_matrix(H, periodic_orbit, method='finite_difference'):
    """
    Compute monodromy matrix for periodic orbit.
    
    The monodromy matrix M is the linearization of the Poincaré
    return map around a periodic orbit.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian.
    periodic_orbit : dict
        Periodic trajectory.
    method : str
        Computation method: 'finite_difference', 'variational'.
    
    Returns
    -------
    dict
        Contains 'M', 'eigenvalues', 'floquet_multipliers'.
    
    Notes
    -----
    Eigenvalues of M (Floquet multipliers) determine stability:
    - |λ| < 1: stable
    - |λ| > 1: unstable
    - |λ| = 1: neutral (for Hamiltonian systems, always has λ=1)
    
    Examples
    --------
    >>> # Find periodic orbit first
    >>> orbit = find_periodic_orbit(H, ...)
    >>> mono = monodromy_matrix(H, orbit)
    >>> print(mono['floquet_multipliers'])
    """
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    
    # Extract period
    T = periodic_orbit['t'][-1]
    z_orbit = np.array([
        periodic_orbit['x1'],
        periodic_orbit['p1'],
        periodic_orbit['x2'],
        periodic_orbit['p2']
    ])
    
    if method == 'finite_difference':
        # Perturb initial conditions
        z0 = z_orbit[:, 0]
        epsilon = 1e-6
        
        M = np.zeros((4, 4))
        
        for i in range(4):
            # Forward perturbation
            z_pert = z0.copy()
            z_pert[i] += epsilon
            
            traj_pert = hamiltonian_flow_4d(
                H, tuple(z_pert), (0, T), n_steps=1000
            )
            
            z_final_pert = np.array([
                traj_pert['x1'][-1],
                traj_pert['p1'][-1],
                traj_pert['x2'][-1],
                traj_pert['p2'][-1]
            ])
            
            z_final_ref = z_orbit[:, -1]
            
            M[:, i] = (z_final_pert - z_final_ref) / epsilon
        
        eigenvalues = np.linalg.eigvals(M)
        
        return {
            'M': M,
            'eigenvalues': eigenvalues,
            'floquet_multipliers': eigenvalues,
            'stable': np.all(np.abs(eigenvalues) <= 1.0 + 1e-6)
        }
    
    elif method == 'variational':
        # Solve variational equations
        # More accurate but more complex
        raise NotImplementedError("Variational method not yet implemented")
    
    else:
        raise ValueError("method must be 'finite_difference' or 'variational'")


def lyapunov_exponents(trajectory, dt, n_vectors=4, renorm_interval=10):
    """
    Compute Lyapunov exponents from trajectory.
    
    Lyapunov exponents measure exponential divergence of nearby trajectories.
    Positive exponents indicate chaos.
    
    Parameters
    ----------
    trajectory : dict
        Phase space trajectory.
    dt : float
        Time step.
    n_vectors : int
        Number of Lyapunov vectors to compute.
    renorm_interval : int
        Steps between Gram-Schmidt orthonormalization.
    
    Returns
    -------
    ndarray
        Lyapunov exponents λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄.
    
    Notes
    -----
    For Hamiltonian systems:
    - λᵢ + λ₅₋ᵢ = 0 (symmetry)
    - One exponent is always zero (time translation)
    
    Examples
    --------
    >>> traj = hamiltonian_flow_4d(H, z0, (0, 1000), n_steps=10000)
    >>> exponents = lyapunov_exponents(traj, dt=0.1)
    >>> print(f"Largest Lyapunov exponent: {exponents[0]:.4f}")
    """
    n_steps = len(trajectory['t'])
    
    # Initialize perturbation vectors
    Q = np.eye(n_vectors)
    running_sum = np.zeros(n_vectors)
    
    # Extract state vectors
    z = np.array([
        trajectory['x1'],
        trajectory['p1'],
        trajectory['x2'],
        trajectory['p2']
    ])
    
    for step in range(1, n_steps):
        # Finite difference Jacobian
        epsilon = 1e-8
        J = np.zeros((4, 4))
        
        for i in range(4):
            dz = (z[:, step] - z[:, step-1]) / dt
            J[:, i] = dz  # Simplified; proper Jacobian needed
        
        # Evolve perturbations
        Q = J @ Q
        
        # Renormalize periodically
        if step % renorm_interval == 0:
            Q, R = np.linalg.qr(Q)
            
            for i in range(n_vectors):
                running_sum[i] += np.log(abs(R[i, i]))
    
    # Compute exponents
    exponents = running_sum / (trajectory['t'][-1])
    
    return np.sort(exponents)[::-1]  # Descending order


def project(trajectory, plane='xy'):
    """
    Project 4D trajectory onto 2D plane.
    
    Parameters
    ----------
    trajectory : dict
        4D phase space trajectory.
    plane : str
        Projection plane: 'xy', 'xp', 'pp', 'config', 'momentum'.
    
    Returns
    -------
    tuple
        (x_coords, y_coords, labels)
    
    Examples
    --------
    >>> traj = hamiltonian_flow_4d(H, z0, (0, 50))
    >>> x, y, labels = project(traj, 'xy')
    >>> plt.plot(x, y)
    >>> plt.xlabel(labels[0])
    >>> plt.ylabel(labels[1])
    """
    if plane == 'xy' or plane == 'config':
        return trajectory['x1'], trajectory['x2'], ('x₁', 'x₂')
    elif plane == 'xp':
        return trajectory['x1'], trajectory['p1'], ('x₁', 'p₁')
    elif plane == 'pp' or plane == 'momentum':
        return trajectory['p1'], trajectory['p2'], ('p₁', 'p₂')
    elif plane == 'x1p2':
        return trajectory['x1'], trajectory['p2'], ('x₁', 'p₂')
    elif plane == 'x2p1':
        return trajectory['x2'], trajectory['p1'], ('x₂', 'p₁')
    else:
        raise ValueError(f"Unknown projection plane: {plane}")


def visualize_poincare_section(H, z0_list, Sigma_def, tmax=100,
                                n_returns=500, plot_vars=('x1', 'p1')):
    """
    Visualize Poincaré section for multiple initial conditions.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian.
    z0_list : list of tuples
        Initial conditions.
    Sigma_def : dict
        Section definition.
    tmax : float
        Integration time per IC.
    n_returns : int
        Maximum returns per IC.
    plot_vars : tuple
        Variables to plot.
    
    Examples
    --------
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> H = (p1**2 + p2**2 + x1**2 + x2**2)/2 + 0.1*x1**2*x2**2
    >>> section = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    >>> z0_list = [(E, 0, 0, np.sqrt(2*E)) for E in np.linspace(0.1, 2, 10)]
    >>> visualize_poincare_section(H, z0_list, section)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(z0_list)))
    
    for idx, z0 in enumerate(z0_list):
        try:
            ps = poincare_section(H, Sigma_def, z0, tmax, n_returns)
            
            if len(ps['section_points']) > 0:
                var1, var2 = plot_vars
                x_vals = [p[var1] for p in ps['section_points']]
                y_vals = [p[var2] for p in ps['section_points']]
                
                ax.plot(x_vals, y_vals, 'o', markersize=2, color=colors[idx],
                       alpha=0.6, label=f'IC {idx+1}')
        except:
            print(f"Warning: Failed for IC {idx}")
    
    ax.set_xlabel(f'{plot_vars[0]}', fontsize=12)
    ax.set_ylabel(f'{plot_vars[1]}', fontsize=12)
    ax.set_title('Poincaré Section', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.show()


def detect_resonances(H, action_range, n_samples=50):
    """
    Detect resonant tori in integrable or near-integrable system.
    
    Resonances occur when frequency ratio ω₁/ω₂ is rational.
    
    Parameters
    ----------
    H : sympy expression
        Hamiltonian (should be integrable or nearly so).
    action_range : tuple
        Range of action variables to scan.
    n_samples : int
        Number of sample points.
    
    Returns
    -------
    dict
        Resonance information: actions, frequencies, ratios.
    
    Notes
    -----
    KAM theory: Irrational tori persist under small perturbations,
    while resonant tori break up into chaotic zones.
    """
    # This is a simplified placeholder
    # Full implementation requires action-angle transformation
    
    print("Note: detect_resonances requires action-angle coordinates")
    print("This is a placeholder. Use action_angle_transform first.")
    
    return {
        'message': 'Not yet fully implemented',
        'suggestion': 'Transform to action-angle variables first'
    }


# ============================================================================
# Tests
# ============================================================================

def test_coupled_oscillators():
    """Test coupled harmonic oscillators."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    
    # Uncoupled case
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 0, 1)
    traj = hamiltonian_flow_4d(H, z0, (0, 10*np.pi), integrator='symplectic')
    
    # Check energy conservation
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 1e-3, f"Energy drift: {energy_drift}"
    
    print("✓ Coupled oscillators test passed")


def test_poincare_section():
    """Test Poincaré section computation."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    section_def = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0 = (1, 0, 0, 0.5)
    
    ps = poincare_section(H, section_def, z0, tmax=50, n_returns=10)
    
    assert len(ps['section_points']) > 0
    
    print("✓ Poincaré section test passed")


if __name__ == "__main__":
    print("Running symplectic_2d tests...\n")
    test_coupled_oscillators()
    test_poincare_section()
    print("\n✓ All tests passed")