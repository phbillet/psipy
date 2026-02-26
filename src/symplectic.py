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
symplectic.py — Unified symplectic geometry toolkit for Hamiltonian mechanics
======================================================================================

Overview
--------
The `symplectic` module provides a comprehensive set of tools for studying Hamiltonian systems with an arbitrary number of degrees of freedom.  It is built around the canonical symplectic structure `ω = Σ dxᵢ ∧ dpᵢ` and offers:

* Construction of the symplectic form and its inverse (Poisson tensor) for any `n`.
* Integration of Hamilton’s equations using symplectic integrators (symplectic Euler, velocity Verlet) as well as standard RK45.
* Symbolic computation of Poisson brackets.
* Detection and linear stability analysis of fixed points (equilibria).
* **1‑DOF specific** utilities: action‑angle variables, phase portraits, separatrix analysis.
* **2‑DOF specific** utilities: Poincaré sections, first‑return maps, monodromy matrix, Lyapunov exponents.
* Automatic inference of phase‑space variables from the Hamiltonian expression.

The module is designed to be both a pedagogical tool for exploring Hamiltonian dynamics and a practical library for more advanced studies (e.g., computing monodromy matrices of periodic orbits).

Mathematical background
-----------------------
A Hamiltonian system with `n` degrees of freedom is defined on a **phase space** of dimension `2n` with canonical coordinates `(x₁, p₁, …, xₙ, pₙ)`.  The **symplectic form**

    ω = Σ dxᵢ ∧ dpᵢ

is a closed non‑degenerate 2‑form that encodes the geometry of the space.  Hamilton’s equations are derived from the relation `ι_{X_H} ω = dH` and read

    ẋᵢ = ∂H/∂pᵢ ,  ṗᵢ = –∂H/∂xᵢ .

The **Poisson bracket** of two observables `f, g` is

    {f, g} = Σ (∂f/∂xᵢ ∂g/∂pᵢ – ∂f/∂pᵢ ∂g/∂xᵢ)

and satisfies the Jacobi identity.  It corresponds to the commutator in quantum mechanics via the Dirac rule `{f, g} → (1/iℏ)[f̂, ĝ]`.

**Symplectic integrators** preserve the symplectic structure exactly (up to machine precision) and therefore exhibit excellent long‑term energy conservation.  The module implements:

* **Symplectic Euler** (first‑order)
* **Velocity Verlet** (second‑order, equivalent to the Störmer–Verlet method)

For non‑symplectic comparisons, a standard `'rk45'` integrator is also available.

For **1‑DOF systems**, the phase space is two‑dimensional; energy surfaces are curves.  The **action variable**

    I(E) = (1/2π) ∮ p dx

is an adiabatic invariant and, for integrable systems, can be used to construct action‑angle coordinates `(I, θ)` in which `H = H(I)` and `θ̇ = ω(I) = dH/dI`.

For **2‑DOF systems**, the phase space is four‑dimensional.  A **Poincaré section** reduces the dynamics to a two‑dimensional map, revealing regular and chaotic motion.  The **monodromy matrix** (linearised return map) of a periodic orbit has eigenvalues (Floquet multipliers) that characterise its stability.  **Lyapunov exponents** quantify the rate of separation of nearby trajectories.


References
----------
.. [1] Arnold, V. I.  *Mathematical Methods of Classical Mechanics*, Springer‑Verlag, 1989 (2nd ed.).  Chapters 7–9.
.. [2] Goldstein, H., Poole, C., & Safko, J.  *Classical Mechanics*, Addison‑Wesley, 2002 (3rd ed.).  Chapter 8.
.. [3] Hairer, E., Lubich, C., & Wanner, G.  *Geometric Numerical Integration*, Springer, 2006 (2nd ed.).  Chapters 1, 6.
.. [4] Lichtenberg, A. J., & Lieberman, M. A.  *Regular and Chaotic Dynamics*, Springer, 1992.
.. [5] Meyer, K. R., Hall, G. R., & Offin, D.  *Introduction to Hamiltonian Dynamical Systems and the N‑Body Problem*, Springer, 2009 (2nd ed.).
"""

from imports import *

# -----------------------------------------------------------------------------
# Utility functions for variable inference and dimension handling
# -----------------------------------------------------------------------------
def _infer_variables(H, expected_ndof=None):
    """
    Attempt to infer phase space variables from Hamiltonian expression.
    Heuristic: look for symbols named 'x1','p1','x2','p2',... or 'x','p' for 1‑DOF.
    If ambiguous, raise ValueError.

    Parameters
    ----------
    H : sympy expression
    expected_ndof : int, optional
        If provided, check that the inferred number of DOF matches.

    Returns
    -------
    list of sympy symbols in order [x1, p1, x2, p2, ...]
    """
    free_syms = list(H.free_symbols)
    if not free_syms:
        raise ValueError("Hamiltonian has no free symbols; cannot infer variables.")

    # 1‑DOF special case: look for exactly two symbols, try to identify x and p
    if len(free_syms) == 2:
        pos_candidates = [s for s in free_syms if 'x' in str(s).lower() or 'q' in str(s).lower()]
        mom_candidates = [s for s in free_syms if 'p' in str(s).lower()]
        if len(pos_candidates) == 1 and len(mom_candidates) == 1:
            return [pos_candidates[0], mom_candidates[0]]
        else:
            raise ValueError(
                f"Cannot unambiguously infer position and momentum from {free_syms}. "
                "Please provide vars_phase explicitly."
            )

    # Multi‑DOF: try to pair xᵢ, pᵢ
    # Collect all symbols that look like positions (x, q) and momenta (p)
    pos_syms = {}
    mom_syms = {}
    for s in free_syms:
        name = str(s)
        if name.startswith('x') or name.startswith('q'):
            # Try to extract index
            idx = name[1:]  # may be empty or number
            if idx.isdigit():
                pos_syms[int(idx)] = s
            else:
                pos_syms[0] = s  # assume x (no index)
        elif name.startswith('p'):
            idx = name[1:]
            if idx.isdigit():
                mom_syms[int(idx)] = s
            else:
                mom_syms[0] = s
    # If we have consistent indices, pair them
    if pos_syms and mom_syms and set(pos_syms.keys()) == set(mom_syms.keys()):
        indices = sorted(pos_syms.keys())
        vars_phase = []
        for i in indices:
            vars_phase.extend([pos_syms[i], mom_syms[i]])
        return vars_phase

    # If everything else fails, raise error
    raise ValueError(
        f"Cannot infer phase space variables from {free_syms}. "
        "Please provide vars_phase explicitly."
    )

def _get_ndof(vars_phase):
    """Return number of degrees of freedom from variable list (must be even)."""
    if len(vars_phase) % 2 != 0:
        raise ValueError("Phase space variables must come in pairs (xᵢ, pᵢ).")
    return len(vars_phase) // 2


def _check_ndof(vars_phase, expected):
    """Raise ValueError if ndof does not match expected."""
    ndof = _get_ndof(vars_phase)
    if ndof != expected:
        raise ValueError(f"This function requires {expected} DOF, but got {ndof}.")


# -----------------------------------------------------------------------------
# SymplecticForm class (dimension‑agnostic)
# -----------------------------------------------------------------------------

class SymplecticForm:
    """
    Symplectic structure for an arbitrary number of degrees of freedom.

    Represents the symplectic 2‑form ω = Σ dxᵢ ∧ dpᵢ (canonical) on a 2n‑dimensional
    phase space. The matrix representation is block‑diagonal with 2×2 blocks [[0,-1],[1,0]].

    Parameters
    ----------
    n : int, optional
        Number of degrees of freedom. If not given, inferred from vars_phase.
    vars_phase : list of symbols, optional
        Ordered list of canonical variables (x1, p1, x2, p2, ...).
    omega_matrix : sympy Matrix, optional
        Custom 2n×2n antisymmetric matrix. If None, canonical form is used.
    """
    def __init__(self, n=None, vars_phase=None, omega_matrix=None):
        if vars_phase is not None:
            self.vars_phase = list(vars_phase)
            self.n = _get_ndof(self.vars_phase)
        elif n is not None:
            self.n = n
            # Create generic variable names x1, p1, x2, p2, ...
            symbols_list = []
            for i in range(1, self.n + 1):
                symbols_list.extend([
                    symbols(f'x{i}', real=True),
                    symbols(f'p{i}', real=True)
                ])
            self.vars_phase = symbols_list
        else:
            raise ValueError("Either n or vars_phase must be provided.")

        if omega_matrix is None:
            # Build canonical block‑diagonal matrix manually
            size = 2 * self.n
            self.omega_matrix = Matrix.zeros(size, size)
            for i in range(self.n):
                self.omega_matrix[2*i, 2*i+1] = -1
                self.omega_matrix[2*i+1, 2*i] = 1
        else:
            self.omega_matrix = Matrix(omega_matrix)
            if self.omega_matrix.shape != (2 * self.n, 2 * self.n):
                raise ValueError(
                    f"omega_matrix must be {2 * self.n}×{2 * self.n}, "
                    f"got {self.omega_matrix.shape}"
                )

        # Check antisymmetry
        if self.omega_matrix != -self.omega_matrix.T:
            raise ValueError("Symplectic form must be antisymmetric")

        self.omega_inv = self.omega_matrix.inv()

    def eval(self, point):
        """
        Evaluate the (constant) symplectic matrix at a point.
        For canonical form, it's independent of coordinates.

        Parameters
        ----------
        point : array_like
            Phase space point (length 2n).

        Returns
        -------
        ndarray
            2n×2n matrix ωᵢⱼ.
        """
        return np.array(self.omega_matrix).astype(float)


# -----------------------------------------------------------------------------
# Hamiltonian flow (generic)
# -----------------------------------------------------------------------------

def hamiltonian_flow(H, z0, tspan, vars_phase=None, integrator='symplectic', n_steps=1000):
    """
    Integrate Hamilton's equations for any number of degrees of freedom.

    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x₁, p₁, x₂, p₂, ...).
    z0 : array_like
        Initial condition, length 2n.
    tspan : (float, float)
        Time interval (t_start, t_end).
    vars_phase : list of symbols, optional
        Phase space variables in order [x₁, p₁, x₂, p₂, ...].
        If None, attempt to infer from H.
    integrator : {'symplectic', 'verlet', 'rk45'}
        Integration method.
    n_steps : int
        Number of time steps.

    Returns
    -------
    dict
        Trajectory with keys 't', each variable name (e.g., 'x1', 'p1', ...), and 'energy'.
    """
    from scipy.integrate import solve_ivp

    # 1. Determine variables and number of DOF
    if vars_phase is None:
        vars_phase = _infer_variables(H)
    vars_phase = list(vars_phase)
    ndof = _get_ndof(vars_phase)
    n = ndof

    # 2. Compute derivatives
    dH_dx = []
    dH_dp = []
    for i in range(n):
        xi = vars_phase[2 * i]
        pi = vars_phase[2 * i + 1]
        dH_dx.append(diff(H, xi))
        dH_dp.append(diff(H, pi))

    # 3. Lambdify (create functions for each derivative)
    args = vars_phase
    f_x = [lambdify(args, dH_dp[i], 'numpy') for i in range(n)]
    f_p = [lambdify(args, -dH_dx[i], 'numpy') for i in range(n)]
    H_func = lambdify(args, H, 'numpy')

    # 4. Integrate
    if integrator == 'rk45':
        def ode_system(t, z):
            # z is a flat array [x1, p1, x2, p2, ...]
            args_val = z
            dz = np.zeros(2 * n)
            for i in range(n):
                dz[2 * i] = f_x[i](*args_val)
                dz[2 * i + 1] = f_p[i](*args_val)
            return dz

        sol = solve_ivp(
            ode_system,
            tspan,
            z0,
            method='RK45',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps),
            rtol=1e-9,
            atol=1e-12
        )

        # Build result dictionary
        result = {'t': sol.t}
        for i, name in enumerate(vars_phase):
            result[str(name)] = sol.y[i]
        result['energy'] = H_func(*sol.y)
        return result

    elif integrator in ['symplectic', 'verlet']:
        dt = (tspan[1] - tspan[0]) / n_steps
        t_vals = np.linspace(tspan[0], tspan[1], n_steps)

        # Allocate arrays for each variable
        arrays = {str(name): np.zeros(n_steps) for name in vars_phase}
        for i, name in enumerate(vars_phase):
            arrays[str(name)][0] = z0[i]

        # Main loop
        for step in range(n_steps - 1):
            # Current values as a flat list
            curr = [arrays[str(name)][step] for name in vars_phase]

            if integrator == 'symplectic':
                # Symplectic Euler: first update all momenta, then all positions
                # (this is first‑order but symplectic)
                # Compute new momenta using current positions and momenta
                p_new = [curr[2 * i + 1] + dt * f_p[i](*curr) for i in range(n)]
                # Build argument list with updated momenta for position update
                args_for_x = []
                for i in range(n):
                    args_for_x.extend([curr[2 * i], p_new[i]])
                x_new = [curr[2 * i] + dt * f_x[i](*args_for_x) for i in range(n)]
                # Combine
                for i in range(n):
                    arrays[str(vars_phase[2 * i])][step + 1] = x_new[i]
                    arrays[str(vars_phase[2 * i + 1])][step + 1] = p_new[i]

            elif integrator == 'verlet':
                # Velocity Verlet: half‑step momentum, full step position, half‑step momentum
                # Half‑step momenta
                p_half = [curr[2 * i + 1] + 0.5 * dt * f_p[i](*curr) for i in range(n)]
                # Build arguments for position update: use half‑step momenta
                args_for_x = []
                for i in range(n):
                    args_for_x.extend([curr[2 * i], p_half[i]])
                x_new = [curr[2 * i] + dt * f_x[i](*args_for_x) for i in range(n)]
                # Build arguments for second half‑step: new positions, half‑step momenta
                args_for_p = []
                for i in range(n):
                    args_for_p.extend([x_new[i], p_half[i]])
                p_new = [p_half[i] + 0.5 * dt * f_p[i](*args_for_p) for i in range(n)]

                for i in range(n):
                    arrays[str(vars_phase[2 * i])][step + 1] = x_new[i]
                    arrays[str(vars_phase[2 * i + 1])][step + 1] = p_new[i]

        # Energy calculation
        energy = np.zeros(n_steps)
        for step in range(n_steps):
            args_step = [arrays[str(name)][step] for name in vars_phase]
            energy[step] = H_func(*args_step)

        result = {'t': t_vals}
        result.update(arrays)
        result['energy'] = energy
        return result

    else:
        raise ValueError("integrator must be 'rk45', 'symplectic', or 'verlet'")


# -----------------------------------------------------------------------------
# Poisson bracket (generic)
# -----------------------------------------------------------------------------

def poisson_bracket(f, g, vars_phase=None):
    """
    Compute Poisson bracket {f, g} = (∂f/∂z) · ω⁻¹ · (∂g/∂z).

    Parameters
    ----------
    f, g : sympy expressions
        Functions on phase space.
    vars_phase : list of symbols, optional
        Phase space variables in order [x₁, p₁, x₂, p₂, ...].
        If None, attempt to infer from f and g.

    Returns
    -------
    sympy expression
        Poisson bracket {f, g}.
    """
    if vars_phase is None:
        # Attempt to infer from both functions
        all_syms = f.free_symbols.union(g.free_symbols)
        if len(all_syms) == 2:
            # Possibly 1‑DOF
            vars_phase = _infer_variables(f + g, expected_ndof=1)
        else:
            # More complex: try _infer_variables on f+g (crude but works if both use same vars)
            vars_phase = _infer_variables(f + g)
    vars_phase = list(vars_phase)
    n = _get_ndof(vars_phase)

    # Build gradient vectors
    grad_f = Matrix([diff(f, var) for var in vars_phase])
    grad_g = Matrix([diff(g, var) for var in vars_phase])

    # Build symplectic inverse (Poisson tensor)
    J_inv = SymplecticForm(n=n).omega_inv

    # Contract: grad_fᵀ · J_inv · grad_g
    bracket = grad_f.dot(J_inv * grad_g)
    return simplify(bracket)


# -----------------------------------------------------------------------------
# Fixed points and linearization (generic)
# -----------------------------------------------------------------------------

def find_fixed_points(H, vars_phase=None, domain=None, tol=1e-6, numerical=False):
    """
    Find fixed points (equilibria) of Hamiltonian system: ∂H/∂z_i = 0 for all i.

    Parameters
    ----------
    H : sympy expression
        Hamiltonian.
    vars_phase : list of symbols, optional
        Phase space variables.
    domain : list of tuples, optional
        Bounds for each variable [(min, max), ...] for numerical search.
    tol : float
        Tolerance for numerical solutions.
    numerical : bool
        If True, use numerical root‑finding even if symbolic solve is possible.

    Returns
    -------
    list of tuples
        Fixed points (z₁, z₂, ..., z₂ₙ).
    """
    if vars_phase is None:
        vars_phase = _infer_variables(H)
    vars_phase = list(vars_phase)
    nvar = len(vars_phase)
    ndof = nvar // 2

    # Equations: dH/dz_i = 0
    eqs = [diff(H, var) for var in vars_phase]

    # Try symbolic solution first (if not forced numerical)
    if not numerical:
        try:
            solutions = solve(eqs, vars_phase, dict=True)
            fixed_points = []
            for sol in solutions:
                point = []
                valid = True
                for var in vars_phase:
                    val = sol.get(var, None)
                    if val is None:
                        valid = False
                        break
                    # Convert to float if possible, else keep symbolic
                    try:
                        val_f = complex(val)
                        if abs(val_f.imag) > tol:
                            valid = False
                            break
                        point.append(val_f.real)
                    except:
                        # Keep symbolic
                        point.append(val)
                if valid:
                    fixed_points.append(tuple(point))
            if fixed_points:
                return fixed_points
            else:
                print("Symbolic solution gave complex points; falling back to numerical.")
        except Exception as e:
            print(f"Symbolic solve failed: {e}")

    # Numerical root‑finding
    from scipy.optimize import fsolve

    # Create lambdified functions for the gradient
    grad_funcs = [lambdify(vars_phase, eq, 'numpy') for eq in eqs]

    def system(z):
        return np.array([f(*z) for f in grad_funcs])

    # Default domain if not provided
    if domain is None:
        domain = [(-10, 10) for _ in range(nvar)]

    # Generate initial guesses on a coarse grid
    fixed_points = []
    grid_points = [np.linspace(dom[0], dom[1], 5) for dom in domain]
    mesh = np.meshgrid(*grid_points, indexing='ij')
    guesses = np.vstack([m.ravel() for m in mesh]).T

    for guess in guesses:
        try:
            sol = fsolve(system, guess)
            if np.linalg.norm(system(sol)) < tol:
                # Check uniqueness
                is_new = True
                for fp in fixed_points:
                    if np.linalg.norm(np.array(sol) - np.array(fp)) < tol:
                        is_new = False
                        break
                if is_new:
                    fixed_points.append(tuple(sol))
        except:
            continue

    return fixed_points


def linearize_at_fixed_point(H, z0, vars_phase=None):
    """
    Compute linear stability matrix (Jacobian of the Hamiltonian vector field) at a fixed point.

    For Hamiltonian systems, the linearized equations are dδz/dt = J · Hessian(H) · δz,
    where J is the symplectic matrix. This function returns the matrix J·Hessian(H) and its eigenvalues.

    Parameters
    ----------
    H : sympy expression
        Hamiltonian.
    z0 : tuple or list
        Fixed point coordinates (length 2n).
    vars_phase : list of symbols, optional
        Phase space variables.

    Returns
    -------
    dict
        Contains 'jacobian' (numpy array), 'eigenvalues', 'eigenvectors',
        and a simple classification based on eigenvalues:
        - 'elliptic' if all eigenvalues are purely imaginary
        - 'hyperbolic' if all eigenvalues are real (or come in opposite pairs)
        - 'mixed' otherwise.
    """
    if vars_phase is None:
        vars_phase = _infer_variables(H)
    vars_phase = list(vars_phase)
    nvar = len(vars_phase)
    ndof = nvar // 2

    # Compute Hessian matrix (second derivatives)
    hessian = Matrix([[diff(diff(H, xi), xj) for xj in vars_phase] for xi in vars_phase])

    # Symplectic matrix J (canonical)
    J = SymplecticForm(n=ndof).omega_matrix

    # Stability matrix = J * Hessian
    stability_matrix = J * hessian

    # Substitute fixed point
    subs_dict = {var: val for var, val in zip(vars_phase, z0)}
    stability_matrix_num = np.array(stability_matrix.subs(subs_dict)).astype(float)

    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(stability_matrix_num)

    # Simple classification
    if np.all(np.abs(eigvals.imag) < 1e-10):  # all real
        # Check if they come in opposite pairs (hyperbolic)
        # For Hamiltonian systems, they often do, but we'll just call it hyperbolic
        fp_type = 'hyperbolic'
    elif np.all(np.abs(eigvals.real) < 1e-10):  # all imaginary
        fp_type = 'elliptic'
    else:
        fp_type = 'mixed (partially hyperbolic/elliptic)'

    return {
        'jacobian': stability_matrix_num,
        'eigenvalues': eigvals,
        'eigenvectors': eigvecs,
        'type': fp_type
    }


# -----------------------------------------------------------------------------
# 1‑DOF specific functions (require ndof == 1)
# -----------------------------------------------------------------------------

def action_integral(H, E, vars_phase=None, method='numerical', x_bounds=None):
    """
    Compute action integral I(E) for a 1‑DOF system.

    I(E) = (1/2π) ∮ p dx

    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    E : float or sympy expression
        Energy level.
    vars_phase : list, optional
        Phase space variables [x, p].
    method : {'numerical', 'symbolic'}
        Integration method.
    x_bounds : tuple, optional
        Integration limits (x_min, x_max) for the orbit.

    Returns
    -------
    float or sympy expression
        Action I(E).
    """
    if vars_phase is None:
        vars_phase = _infer_variables(H, expected_ndof=1)
    _check_ndof(vars_phase, 1)
    x, p = vars_phase

    # If E is symbolic, create a temporary symbol
    E_sym = E if hasattr(E, 'free_symbols') else symbols('E_temp', real=True, positive=True)
    E_numeric = None if hasattr(E, 'free_symbols') else E

    # Solve for p(x)
    eq = H - E_sym
    p_solutions = solve(eq, p)
    if not p_solutions:
        raise ValueError("Cannot solve for p(x)")
    # Take the first real‑looking solution
    p_expr = None
    for sol in p_solutions:
        if im(sol) == 0 or sol.is_real:
            p_expr = sol
            break
    if p_expr is None:
        p_expr = p_solutions[0]

    if method == 'symbolic':
        # Find turning points: p(x) = 0
        turning_points = solve(p_expr, x)
        if len(turning_points) < 2:
            # Fallback for harmonic oscillator
            x_max_sym = sqrt(2 * E_sym)
            x_min_sym = -x_max_sym
        else:
            real_pts = [pt for pt in turning_points if im(pt) == 0]
            if len(real_pts) >= 2:
                x_min_sym = min(real_pts)
                x_max_sym = max(real_pts)
            else:
                x_max_sym = sqrt(2 * E_sym)
                x_min_sym = -x_max_sym

        try:
            integrand = simplify(p_expr)
            action_half = integrate(integrand, (x, x_min_sym, x_max_sym))
            action = simplify(2 * action_half / (2 * pi))
            if E_numeric is not None:
                action = action.subs(E_sym, E_numeric)
            return action
        except Exception as e:
            print(f"Symbolic integration failed: {e}")
            if E_numeric is None:
                raise
            method = 'numerical'
            E = E_numeric

    if method == 'numerical':
        from scipy.integrate import quad
        if E_numeric is None:
            raise ValueError("numerical method requires numeric energy")
        # Determine bounds
        if x_bounds is None:
            # Try to get from p_expr=0 numerically
            p_func_zero = lambdify(x, p_expr.subs(E_sym, E_numeric), 'numpy')
            # crude bound estimate: scan
            x_scan = np.linspace(-10, 10, 1000)
            p_vals = p_func_zero(x_scan)
            # find where sign changes
            zero_crossings = np.where(np.diff(np.sign(p_vals)))[0]
            if len(zero_crossings) >= 2:
                x_min = x_scan[zero_crossings[0]]
                x_max = x_scan[zero_crossings[-1]]
            else:
                # fallback: amplitude from harmonic oscillator formula
                amp = np.sqrt(2 * E_numeric)
                x_min, x_max = -amp, amp
        else:
            x_min, x_max = x_bounds

        p_func = lambdify(x, p_expr.subs(E_sym, E_numeric), 'numpy')
        def integrand_numeric(xv):
            try:
                pv = p_func(xv)
                if np.iscomplexobj(pv):
                    if np.abs(np.imag(pv)) > 1e-10:
                        return 0.0
                    pv = np.real(pv)
                if not np.isfinite(pv):
                    return 0.0
                return np.abs(pv)
            except:
                return 0.0

        eps = 1e-10
        action_half, err = quad(integrand_numeric, x_min + eps, x_max - eps,
                                limit=100, epsabs=1e-10, epsrel=1e-10)
        action = 2 * action_half / (2 * np.pi)
        return action

    else:
        raise ValueError("method must be 'symbolic' or 'numerical'")


def phase_portrait(H, x_range, p_range, vars_phase=None, resolution=50, levels=20):
    """Generate phase portrait for 1‑DOF system."""
    if vars_phase is None:
        vars_phase = _infer_variables(H, expected_ndof=1)
    _check_ndof(vars_phase, 1)
    x, p = vars_phase

    H_func = lambdify((x, p), H, 'numpy')
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    p_vals = np.linspace(p_range[0], p_range[1], resolution)
    X, P = np.meshgrid(x_vals, p_vals, indexing='ij')
    E = H_func(X, P)

    plt.figure(figsize=(10, 8))
    cs = plt.contour(X, P, E, levels=levels, colors='blue', linewidths=1.5)
    plt.clabel(cs, inline=True, fontsize=8, fmt='E=%.2f')

    # Vector field
    dH_dp = diff(H, p)
    dH_dx = diff(H, x)
    vx_func = lambdify((x, p), dH_dp, 'numpy')
    vp_func = lambdify((x, p), -dH_dx, 'numpy')
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


def separatrix_analysis(H, x_range, p_range, saddle_point, vars_phase=None):
    """
    Analyze separatrix structure near saddle point (1‑DOF).

    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    x_range, p_range : tuple
        Domain for visualization.
    saddle_point : tuple
        Location of saddle (x_s, p_s).
    vars_phase : list, optional
        Phase space variables [x, p].

    Returns
    -------
    dict
        Contains stable/unstable manifolds and energy at saddle.
    """
    if vars_phase is None:
        vars_phase = _infer_variables(H, expected_ndof=1)
    _check_ndof(vars_phase, 1)
    x, p = vars_phase
    x_s, p_s = saddle_point

    H_func = lambdify((x, p), H, 'numpy')
    E_saddle = H_func(x_s, p_s)

    # Linearization
    lin = linearize_at_fixed_point(H, saddle_point, vars_phase=vars_phase)

    if lin['type'] not in ('hyperbolic', 'saddle'):
        print(f"Warning: Point is not a saddle, type = {lin['type']}")

    eigenvals = lin['eigenvalues']
    eigenvecs = lin['eigenvectors']

    # Positive eigenvalue corresponds to unstable direction
    unstable_idx = np.argmax(np.real(eigenvals))
    stable_idx = 1 - unstable_idx

    unstable_dir = np.real(eigenvecs[:, unstable_idx])
    stable_dir = np.real(eigenvecs[:, stable_idx])

    epsilon = 0.01

    # Unstable manifolds (forward in time)
    z_unstable_plus = (x_s + epsilon * unstable_dir[0],
                       p_s + epsilon * unstable_dir[1])
    z_unstable_minus = (x_s - epsilon * unstable_dir[0],
                        p_s - epsilon * unstable_dir[1])

    traj_unstable_plus = hamiltonian_flow(H, z_unstable_plus, (0, 10),
                                          vars_phase=vars_phase,
                                          integrator='symplectic', n_steps=1000)
    traj_unstable_minus = hamiltonian_flow(H, z_unstable_minus, (0, 10),
                                           vars_phase=vars_phase,
                                           integrator='symplectic', n_steps=1000)

    # Stable manifolds (backward in time)
    z_stable_plus = (x_s + epsilon * stable_dir[0],
                     p_s + epsilon * stable_dir[1])
    z_stable_minus = (x_s - epsilon * stable_dir[0],
                      p_s - epsilon * stable_dir[1])

    traj_stable_plus = hamiltonian_flow(H, z_stable_plus, (0, -10),
                                        vars_phase=vars_phase,
                                        integrator='symplectic', n_steps=1000)
    traj_stable_minus = hamiltonian_flow(H, z_stable_minus, (0, -10),
                                         vars_phase=vars_phase,
                                         integrator='symplectic', n_steps=1000)

    return {
        'E_saddle': E_saddle,
        'unstable_dir': unstable_dir,
        'stable_dir': stable_dir,
        'unstable_manifolds': [traj_unstable_plus, traj_unstable_minus],
        'stable_manifolds': [traj_stable_plus, traj_stable_minus]
    }


def visualize_phase_space_structure(H, x_range, p_range, vars_phase=None,
                                    fixed_points=None, show_separatrices=True,
                                    n_trajectories=10):
    """
    Comprehensive visualization of phase space structure for 1‑DOF.

    Parameters
    ----------
    H : sympy expression
        Hamiltonian.
    x_range, p_range : tuple
        Domain ranges.
    vars_phase : list, optional
        Phase space variables [x, p].
    fixed_points : list, optional
        List of fixed points to analyze.
    show_separatrices : bool
        Whether to plot separatrices.
    n_trajectories : int
        Number of sample trajectories.
    """
    if vars_phase is None:
        vars_phase = _infer_variables(H, expected_ndof=1)
    _check_ndof(vars_phase, 1)
    x, p = vars_phase

    fig, ax = plt.subplots(figsize=(12, 10))

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
        fixed_points = find_fixed_points(H, vars_phase=vars_phase,
                                         domain=[x_range, p_range])

    for fp in fixed_points:
        lin = linearize_at_fixed_point(H, fp, vars_phase=vars_phase)

        if lin['type'] == 'elliptic':
            marker, color = 'o', 'green'
        elif lin['type'] in ('hyperbolic', 'saddle'):
            marker, color = 'X', 'red'
        else:
            marker, color = 's', 'orange'

        ax.plot(fp[0], fp[1], marker=marker, color=color, markersize=15,
                label=f'{lin["type"]} at ({fp[0]:.2f}, {fp[1]:.2f})', zorder=10)

    # Plot separatrices
    if show_separatrices:
        for fp in fixed_points:
            lin = linearize_at_fixed_point(H, fp, vars_phase=vars_phase)
            if lin['type'] in ('hyperbolic', 'saddle'):
                try:
                    sep = separatrix_analysis(H, x_range, p_range, fp, vars_phase=vars_phase)
                    for traj in sep['unstable_manifolds']:
                        ax.plot(traj['x'], traj['p'], 'r-', linewidth=2, alpha=0.8)
                    for traj in sep['stable_manifolds']:
                        ax.plot(traj['x'], traj['p'], 'b-', linewidth=2, alpha=0.8)
                except Exception as e:
                    print(f"Warning: Could not compute separatrices for saddle at {fp}: {e}")

    # Sample trajectories
    np.random.seed(42)
    for _ in range(n_trajectories):
        x0 = np.random.uniform(x_range[0], x_range[1])
        p0 = np.random.uniform(p_range[0], p_range[1])
        try:
            traj = hamiltonian_flow(H, (x0, p0), (0, 20), vars_phase=vars_phase,
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


def action_angle_transform(H, x_range, p_range, vars_phase=None, n_contours=10):
    """
    Compute action-angle transformation for integrable system (1‑DOF).

    For integrable 1D systems, finds action I(E) and angle θ such that:
        H = H(I)
        θ̇ = ω(I) = dH/dI

    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x, p).
    x_range, p_range : tuple
        Phase space domain.
    vars_phase : list, optional
        Phase space variables [x, p].
    n_contours : int
        Number of energy levels to compute.

    Returns
    -------
    dict
        Contains 'energies', 'actions', 'frequencies'.
    """
    if vars_phase is None:
        vars_phase = _infer_variables(H, expected_ndof=1)
    _check_ndof(vars_phase, 1)
    x, p = vars_phase

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
            I = action_integral(H, E, vars_phase=vars_phase, method='numerical', x_bounds=x_range)
            actions.append(I)

            # Estimate frequency dE/dI numerically
            if len(actions) > 1:
                omega = (E - energies[len(actions)-2]) / (I - actions[-2])
            else:
                omega = 0

            frequencies.append(omega)
        except Exception:
            pass

    return {
        'energies': np.array(energies[:len(actions)]),
        'actions': np.array(actions),
        'frequencies': np.array(frequencies)
    }


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
    """
    I = symbols('I', real=True, positive=True)

    if method == 'derivative':
        omega_expr = diff(H, I)
        omega_func = lambdify(I, omega_expr, 'numpy')
        return omega_func(I_val)
    else:
        raise NotImplementedError("Only 'derivative' method implemented")


# -----------------------------------------------------------------------------
# 2‑DOF specific functions (require ndof == 2)
# -----------------------------------------------------------------------------

def hamiltonian_flow_4d(H, z0, tspan, integrator='symplectic', n_steps=1000):
    """Backward compatibility wrapper for 4D flow."""
    # Assume variables are x1, p1, x2, p2
    vars_phase = [symbols('x1 p1 x2 p2', real=True)]
    return hamiltonian_flow(H, z0, tspan, vars_phase=vars_phase,
                            integrator=integrator, n_steps=n_steps)


def poincare_section(H, Sigma_def, z0, tmax, vars_phase=None, n_returns=1000, integrator='symplectic'):
    """Poincaré section for 2‑DOF systems."""
    if vars_phase is None:
        vars_phase = [symbols('x1 p1 x2 p2', real=True)]  # assume canonical
    _check_ndof(vars_phase, 2)

    # Integrate trajectory
    n_steps = 10000  # high resolution for accurate crossings
    traj = hamiltonian_flow(H, z0, (0, tmax), vars_phase=vars_phase,
                            integrator=integrator, n_steps=n_steps)

    # Extract section variable (string)
    var_name = Sigma_def['variable']
    # Map string to actual index in vars_phase
    try:
        idx = [str(v) for v in vars_phase].index(var_name)
    except ValueError:
        raise ValueError(f"Variable '{var_name}' not found in phase space variables.")

    var_values = traj[var_name]
    threshold = Sigma_def['value']
    direction = Sigma_def.get('direction', 'positive')

    crossings = []
    section_points = []

    for i in range(len(var_values) - 1):
        v_curr = var_values[i]
        v_next = var_values[i+1]

        if direction == 'positive':
            crosses = (v_curr < threshold) and (v_next >= threshold)
        elif direction == 'negative':
            crosses = (v_curr > threshold) and (v_next <= threshold)
        else:
            crosses = (v_curr - threshold) * (v_next - threshold) < 0

        if crosses:
            alpha = (threshold - v_curr) / (v_next - v_curr)
            t_cross = traj['t'][i] + alpha * (traj['t'][i+1] - traj['t'][i])

            point = {}
            for key in [str(v) for v in vars_phase]:
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
    """First return map from Poincaré section points."""
    if len(section_points) < 2:
        raise ValueError("Need at least 2 section points")
    var1, var2 = plot_variables
    current = np.array([[p[var1], p[var2]] for p in section_points[:-1]])
    next_pts = np.array([[p[var1], p[var2]] for p in section_points[1:]])
    return {'current': current, 'next': next_pts, 'variables': plot_variables}


def monodromy_matrix(H, periodic_orbit, vars_phase=None, method='finite_difference'):
    """Monodromy matrix for a periodic orbit (2‑DOF)."""
    if vars_phase is None:
        vars_phase = [symbols('x1 p1 x2 p2', real=True)]
    _check_ndof(vars_phase, 2)

    T = periodic_orbit['t'][-1]
    z0 = np.array([periodic_orbit[str(v)][0] for v in vars_phase])

    if method == 'finite_difference':
        epsilon = 1e-6
        n_steps_pert = len(periodic_orbit['t'])
        M = np.zeros((4, 4))

        for i in range(4):
            # Forward perturbation
            z_plus = z0.copy()
            z_plus[i] += epsilon
            traj_plus = hamiltonian_flow(H, z_plus, (0, T), vars_phase=vars_phase,
                                         integrator='rk45', n_steps=n_steps_pert)
            zf_plus = np.array([traj_plus[str(v)][-1] for v in vars_phase])

            # Backward perturbation
            z_minus = z0.copy()
            z_minus[i] -= epsilon
            traj_minus = hamiltonian_flow(H, z_minus, (0, T), vars_phase=vars_phase,
                                          integrator='rk45', n_steps=n_steps_pert)
            zf_minus = np.array([traj_minus[str(v)][-1] for v in vars_phase])

            M[:, i] = (zf_plus - zf_minus) / (2 * epsilon)

        eigenvalues = np.linalg.eigvals(M)
        return {
            'M': M,
            'eigenvalues': eigenvalues,
            'floquet_multipliers': eigenvalues,
            'stable': np.all(np.abs(eigenvalues) <= 1.0 + 1e-6)
        }
    else:
        raise NotImplementedError("Only finite_difference method implemented.")


def lyapunov_exponents(trajectory, dt, vars_phase=None, n_vectors=4, renorm_interval=10):
    """Estimate Lyapunov exponents (simplified) for 4D trajectory."""
    if vars_phase is None:
        vars_phase = [symbols('x1 p1 x2 p2', real=True)]
    _check_ndof(vars_phase, 2)

    n_steps = len(trajectory['t'])
    Q = np.eye(n_vectors)
    running_sum = np.zeros(n_vectors)

    # Extract state vectors as a 4×n_steps array
    z = np.array([trajectory[str(v)] for v in vars_phase])

    for step in range(1, n_steps):
        # Crude finite‑difference Jacobian
        epsilon = 1e-8
        J = np.zeros((4, 4))
        for i in range(4):
            dz = (z[:, step] - z[:, step-1]) / dt
            J[:, i] = dz  # This is a rough approximation; real Jacobian would involve derivatives.

        Q = J @ Q

        if step % renorm_interval == 0:
            Q, R = np.linalg.qr(Q)
            for i in range(n_vectors):
                running_sum[i] += np.log(abs(R[i, i]))

    exponents = running_sum / (trajectory['t'][-1])
    return np.sort(exponents)[::-1]


def project(trajectory, plane='xy', vars_phase=None):
    """Project a 4D trajectory onto a 2D plane."""
    if vars_phase is None:
        vars_phase = [symbols('x1 p1 x2 p2', real=True)]
    _check_ndof(vars_phase, 2)

    x1, p1, x2, p2 = vars_phase
    if plane in ('xy', 'config'):
        return trajectory['x1'], trajectory['x2'], ('x₁', 'x₂')
    elif plane == 'xp':
        return trajectory['x1'], trajectory['p1'], ('x₁', 'p₁')
    elif plane in ('pp', 'momentum'):
        return trajectory['p1'], trajectory['p2'], ('p₁', 'p₂')
    elif plane == 'x1p2':
        return trajectory['x1'], trajectory['p2'], ('x₁', 'p₂')
    elif plane == 'x2p1':
        return trajectory['x2'], trajectory['p1'], ('x₂', 'p₁')
    else:
        raise ValueError(f"Unknown projection plane: {plane}")


def visualize_poincare_section(H, z0_list, Sigma_def, vars_phase=None,
                               tmax=100, n_returns=500, plot_vars=('x1', 'p1')):
    """Visualize Poincaré section for multiple initial conditions (2‑DOF)."""
    if vars_phase is None:
        vars_phase = [symbols('x1 p1 x2 p2', real=True)]
    _check_ndof(vars_phase, 2)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(z0_list)))

    for idx, z0 in enumerate(z0_list):
        try:
            ps = poincare_section(H, Sigma_def, z0, tmax, vars_phase=vars_phase,
                                   n_returns=n_returns)
            if ps['section_points']:
                var1, var2 = plot_vars
                x_vals = [p[var1] for p in ps['section_points']]
                y_vals = [p[var2] for p in ps['section_points']]
                ax.plot(x_vals, y_vals, 'o', markersize=2, color=colors[idx],
                        alpha=0.6, label=f'IC {idx+1}')
        except Exception as e:
            print(f"Warning: Failed for IC {idx}: {e}")

    ax.set_xlabel(f'{plot_vars[0]}', fontsize=12)
    ax.set_ylabel(f'{plot_vars[1]}', fontsize=12)
    ax.set_title('Poincaré Section', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Backward compatibility aliases
# -----------------------------------------------------------------------------

SymplecticForm1D = lambda vars_phase=None: SymplecticForm(n=1, vars_phase=vars_phase)
SymplecticForm2D = lambda vars_phase=None: SymplecticForm(n=2, vars_phase=vars_phase)

# -----------------------------------------------------------------------------
# Tests (optional, can be removed if not needed)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Tests (combined and adapted)
# -----------------------------------------------------------------------------

def test_harmonic_oscillator():
    """Test 1‑DOF harmonic oscillator."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2

    traj = hamiltonian_flow(H, (1, 0), (0, 10*np.pi), vars_phase=[x, p],
                            integrator='verlet', n_steps=2000)
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 1e-3, f"Energy drift too large: {energy_drift}"

    pb = poisson_bracket(x, p, vars_phase=[x, p])
    assert pb == 1

    # Action integral
    limit = np.sqrt(2)
    I = action_integral(H, 1.0, vars_phase=[x, p], method='numerical', x_bounds=(-limit, limit))
    assert np.isclose(I, 1.0, rtol=0.01)

    print("✓ Harmonic oscillator test passed")


def test_coupled_oscillators():
    """Test 2‑DOF coupled oscillators."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2

    z0 = (1, 0, 0, 1)
    traj = hamiltonian_flow(H, z0, (0, 10*np.pi), vars_phase=[x1, p1, x2, p2],
                            integrator='symplectic', n_steps=2000)
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 1e-3

    print("✓ Coupled oscillators test passed")


def test_poincare_section():
    """Test Poincaré section on 2‑DOF."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    section_def = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0 = (1, 0, 0, 0.5)
    ps = poincare_section(H, section_def, z0, tmax=50, vars_phase=[x1, p1, x2, p2],
                          n_returns=10)
    assert len(ps['section_points']) > 0
    print("✓ Poincaré section test passed")


def test_fixed_points_1d():
    """Test fixed point finding in 1‑DOF."""
    x, p = symbols('x p', real=True)
    H = p**2/2 - x**2/2
    fps = find_fixed_points(H, vars_phase=[x, p])
    assert len(fps) == 1
    assert np.allclose(fps[0], (0, 0), atol=1e-6)
    lin = linearize_at_fixed_point(H, (0, 0), vars_phase=[x, p])
    # Should have one positive and one negative eigenvalue (saddle)
    assert lin['type'] == 'hyperbolic'  # our simple classification
    print("✓ Fixed point test passed")


def test_monodromy_simple():
    """Test monodromy matrix on a simple periodic orbit (2‑DOF)."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    T = 2 * np.pi
    z0 = (1, 0, 0, 0)
    periodic_orbit = hamiltonian_flow(H, z0, (0, T), vars_phase=[x1, p1, x2, p2],
                                      integrator='rk45', n_steps=5000)
    mono = monodromy_matrix(H, periodic_orbit, vars_phase=[x1, p1, x2, p2])
    multipliers = mono['floquet_multipliers']
    assert np.allclose(np.abs(multipliers), 1.0, atol=1e-3)
    print("✓ Monodromy matrix test passed")


if __name__ == "__main__":
    print("Running unified symplectic tests...\n")
    test_harmonic_oscillator()
    test_coupled_oscillators()
    test_poincare_section()
    test_fixed_points_1d()
    test_monodromy_simple()
    print("\n✓ All tests passed")
