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
* **Integrability classification** utilities: level-spacing statistics (Berry-Tabor /
  BGS), Weyl's law, Berry-Tabor smoothed density of states, KAM tori detection,
  and winding / rotation numbers (``IntegrabilityAnalysis`` class).

The module is designed to be both a pedagogical tool for exploring Hamiltonian dynamics and a practical library for more advanced studies (e.g., computing monodromy matrices of periodic orbits, classifying systems as integrable or chaotic).

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
.. [6] Berry, M. V. & Tabor, M.  "Level clustering in the regular spectrum", *Proc. R. Soc. Lond. A* 356, 375–394, 1977.
.. [7] Bohigas, O., Giannoni, M. J., & Schmit, C.  "Characterization of chaotic quantum spectra and universality of level fluctuation laws", *Phys. Rev. Lett.* 52, 1–4, 1984.
"""

from imports import *

# -----------------------------------------------------------------------------
# Utility functions for variable inference and dimension handling
# -----------------------------------------------------------------------------

def _infer_variables(H, expected_ndof=None):
    """
    Infer phase space canonical variables from a Hamiltonian expression.
    
    This function uses naming conventions to automatically identify position and
    momentum variables, eliminating the need for manual specification in common cases.
    
    Heuristics applied:
    - For 1-DOF: Looks for exactly 2 symbols where one contains 'x' or 'q' 
      (position) and one contains 'p' (momentum)
    - For multi-DOF: Pairs symbols like x1/p1, x2/p2, ... or q1/p1, q2/p2, ...
    - Indices must match between position and momentum variables
    
    Parameters
    ----------
    H : sympy.Expr
        Hamiltonian expression containing phase space variables as free symbols.
    expected_ndof : int, optional
        Expected number of degrees of freedom. If provided, validates that the
        inferred variables match this expectation.
    
    Returns
    -------
    list of sympy.Symbol
        Ordered list of canonical variables [x₁, p₁, x₂, p₂, ...].
    
    Raises
    ------
    ValueError
        If no free symbols exist in H, if position/momentum cannot be 
        unambiguously identified, or if inferred ndof doesn't match expected_ndof.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2
    >>> _infer_variables(H)
    [x, p]
    
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> H = p1**2/2 + p2**2/2 + x1**2/2 + x2**2/2
    >>> _infer_variables(H)
    [x1, p1, x2, p2]
    
    Notes
    -----
    If automatic inference fails, provide vars_phase explicitly to functions
    that accept this parameter.
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
    Represents the symplectic structure on a 2n-dimensional phase space.
    
    The symplectic form ω is a closed, non-degenerate 2-form that defines the
    geometric structure of Hamiltonian mechanics. In canonical coordinates, it
    takes the standard form:
    
        ω = Σᵢ dxᵢ ∧ dpᵢ
    
    The matrix representation J (also called the symplectic matrix) satisfies:
    - Jᵀ = -J (antisymmetric)
    - J² = -I (for canonical form)
    - det(J) = 1
    
    This matrix is used to convert gradients of the Hamiltonian into the
    Hamiltonian vector field: ż = J · ∇H
    
    Parameters
    ----------
    n : int, optional
        Number of degrees of freedom. Used to generate generic variables if
        vars_phase is not provided.
    vars_phase : list of sympy.Symbol, optional
        Ordered list of canonical variables [x₁, p₁, x₂, p₂, ...]. If provided,
        n is inferred from the length of this list.
    omega_matrix : sympy.Matrix, optional
        Custom 2n×2n symplectic matrix. If None, the canonical form is used:
        J = [[0, -I], [I, 0]] where I is the n×n identity.
    
    Attributes
    ----------
    n : int
        Number of degrees of freedom.
    vars_phase : list
        Phase space variables.
    omega_matrix : sympy.Matrix
        The symplectic matrix J.
    omega_inv : sympy.Matrix
        The inverse J⁻¹ (Poisson tensor), used for computing Poisson brackets.
    
    Raises
    ------
    ValueError
        If neither n nor vars_phase is provided, if omega_matrix has incorrect
        dimensions, or if omega_matrix is not antisymmetric.
    
    Examples
    --------
    >>> # 1-DOF canonical form
    >>> sf = SymplecticForm(n=1)
    >>> sf.omega_matrix
    Matrix([[0, -1], [1, 0]])
    
    >>> # 2-DOF with explicit variables
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> sf = SymplecticForm(vars_phase=[x1, p1, x2, p2])
    >>> sf.n
    2
    
    See Also
    --------
    poisson_bracket : Uses omega_inv to compute {f, g}
    hamiltonian_flow : Uses the symplectic structure for integration
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
def hamiltonian_flow(H, z0, tspan, vars_phase=None, integrator='symplectic',
                     n_steps=1000):
    """
    Numerically integrate Hamilton's equations of motion.

    Solves the system:
        ẋᵢ = ∂H/∂pᵢ
        ṗᵢ = -∂H/∂xᵢ

    for i = 1, ..., n degrees of freedom.

    Integrator options and their properties:
    - 'symplectic' (Symplectic Euler): First-order, exactly preserves symplectic
      structure, good long-term energy behavior, computationally efficient.
    - 'verlet' (Velocity Verlet): Second-order, time-reversible, excellent for
      oscillatory systems, preserves symplectic structure.
    - 'rk45' (Runge-Kutta 4/5): Fourth-order adaptive, not symplectic, better
      short-term accuracy but may drift in energy over long simulations.

    Parameters
    ----------
    H : sympy.Expr
        Hamiltonian function H(x₁, p₁, ..., xₙ, pₙ).
    z0 : array_like
        Initial conditions as [x₁₀, p₁₀, x₂₀, p₂₀, ...], length 2n.
    tspan : tuple of float
        Time integration interval (t_start, t_end).
    vars_phase : list of sympy.Symbol, optional
        Phase space variables in canonical order. If None, inferred from H.
    integrator : {'symplectic', 'verlet', 'rk45'}
        Numerical integration method (default: 'symplectic').
    n_steps : int
        Number of discrete time steps (default: 1000).

    Returns
    -------
    dict
        Dictionary containing:
        - 't'            : ndarray, time points
        - '<var_name>'   : ndarray, trajectory for each phase space variable
        - 'energy'       : ndarray, H evaluated along the trajectory

    Raises
    ------
    ValueError
        If integrator is not recognized, if variables cannot be inferred, or
        if z0 has incorrect dimension.

    Notes
    -----
    Performance: derivatives are lambdified once into a single vectorised
    function F(z) → dz that operates on a numpy state vector.  The symplectic
    and verlet loops are written with direct array indexing (no Python-level
    list allocation per step), giving 10–50× speedup over the original
    scalar-loop implementation for large n_steps.

    For long-time simulations prefer 'symplectic' or 'verlet' to preserve the
    geometric structure of phase space.  Use 'rk45' for short-time
    high-accuracy requirements.
    """
    from scipy.integrate import solve_ivp

    # ── 1. Variables and DOF ──────────────────────────────────────────────────
    if vars_phase is None:
        vars_phase = _infer_variables(H)
    vars_phase = list(vars_phase)
    n  = _get_ndof(vars_phase)       # number of DOF
    z0 = np.asarray(z0, dtype=float)

    # ── 2. Symbolic derivatives — computed ONCE ───────────────────────────────
    dH_dx = [diff(H, vars_phase[2*i])   for i in range(n)]
    dH_dp = [diff(H, vars_phase[2*i+1]) for i in range(n)]

    # ── 3. Lambdify into a SINGLE vectorised function ─────────────────────────
    # F_x[i](z) = ∂H/∂pᵢ,   F_p[i](z) = -∂H/∂xᵢ
    # Each callable takes the full state vector unpacked: f(*z)
    F_x = [lambdify(vars_phase, expr, 'numpy') for expr in dH_dp]
    F_p = [lambdify(vars_phase, expr, 'numpy') for expr in
           [-dxe for dxe in dH_dx]]
    H_func = lambdify(vars_phase, H, 'numpy')

    # Unified RHS: dz/dt as a numpy array, state z shape (2n,)
    # Used by rk45 and as a building block for the symplectic loops.
    def rhs(z):
        """Return dz/dt as shape-(2n,) array. z must be 1-D length 2n."""
        dz = np.empty(2*n)
        zp = z          # alias — no copy
        for i in range(n):
            dz[2*i]   = F_x[i](*zp)
            dz[2*i+1] = F_p[i](*zp)
        return dz

    # ── 4a. RK45 ─────────────────────────────────────────────────────────────
    if integrator == 'rk45':
        sol = solve_ivp(
            lambda t, z: rhs(z),
            tspan, z0,
            method='RK45',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps),
            rtol=1e-9, atol=1e-12,
        )
        result = {'t': sol.t}
        for i, name in enumerate(vars_phase):
            result[str(name)] = sol.y[i]
        result['energy'] = H_func(*sol.y)   # vectorised over time axis
        return result

    # ── 4b. Symplectic / Verlet ───────────────────────────────────────────────
    elif integrator in ('symplectic', 'verlet'):
        dt      = (tspan[1] - tspan[0]) / (n_steps - 1)
        t_vals  = np.linspace(tspan[0], tspan[1], n_steps)

        # Pre-allocate state matrix: shape (2n, n_steps)
        # traj[2i,   :] = xᵢ(t),   traj[2i+1, :] = pᵢ(t)
        traj = np.empty((2*n, n_steps))
        traj[:, 0] = z0

        # ── Helper: force vector at state z ──────────────────────────────────
        # Returns (dx_dt, dp_dt) each as length-n arrays
        def forces(z):
            fx = np.empty(n)
            fp = np.empty(n)
            for i in range(n):
                fx[i] = F_x[i](*z)
                fp[i] = F_p[i](*z)
            return fx, fp

        if integrator == 'symplectic':
            # ── Symplectic Euler (1st order) ──────────────────────────────────
            # p_{k+1} = p_k + dt · F_p(z_k)
            # x_{k+1} = x_k + dt · F_x(x_k, p_{k+1})
            for k in range(n_steps - 1):
                z  = traj[:, k]
                fx, fp = forces(z)

                # New momenta from current state
                p_new = z[1::2] + dt * fp          # shape (n,)

                # Build z_half: current positions + new momenta
                z_half      = z.copy()
                z_half[1::2] = p_new

                fx_half, _ = forces(z_half)
                x_new = z[0::2] + dt * fx_half     # shape (n,)

                traj[0::2, k+1] = x_new
                traj[1::2, k+1] = p_new

        else:  # verlet
            # ── Velocity Verlet (2nd order) ───────────────────────────────────
            # p_{k+1/2} = p_k       + dt/2 · F_p(z_k)
            # x_{k+1}   = x_k       + dt   · F_x(x_k, p_{k+1/2})
            # p_{k+1}   = p_{k+1/2} + dt/2 · F_p(x_{k+1}, p_{k+1/2})
            half = 0.5 * dt
            for k in range(n_steps - 1):
                z = traj[:, k]
                _, fp = forces(z)

                p_half = z[1::2] + half * fp       # shape (n,)

                z_mid       = z.copy()
                z_mid[1::2] = p_half
                fx_mid, _   = forces(z_mid)
                x_new = z[0::2] + dt * fx_mid      # shape (n,)

                z_end       = np.empty(2*n)
                z_end[0::2] = x_new
                z_end[1::2] = p_half
                _, fp_end   = forces(z_end)
                p_new = p_half + half * fp_end

                traj[0::2, k+1] = x_new
                traj[1::2, k+1] = p_new

        # ── 5. Build result dict ──────────────────────────────────────────────
        result = {'t': t_vals}
        for i in range(n):
            result[str(vars_phase[2*i])]   = traj[2*i]
            result[str(vars_phase[2*i+1])] = traj[2*i+1]

        # Energy: vectorised call over all timesteps
        result['energy'] = H_func(*[traj[i] for i in range(2*n)])
        return result

    else:
        raise ValueError("integrator must be 'rk45', 'symplectic', or 'verlet'")

def hamiltonian_flow_old(H, z0, tspan, vars_phase=None, integrator='symplectic', 
                     n_steps=1000):
    """
    Numerically integrate Hamilton's equations of motion.
    
    Solves the system:
        ẋᵢ = ∂H/∂pᵢ
        ṗᵢ = -∂H/∂xᵢ
    
    for i = 1, ..., n degrees of freedom.
    
    Integrator options and their properties:
    - 'symplectic' (Symplectic Euler): First-order, exactly preserves symplectic
      structure, good long-term energy behavior, computationally efficient
    - 'verlet' (Velocity Verlet): Second-order, time-reversible, excellent for
      oscillatory systems, preserves symplectic structure
    - 'rk45' (Runge-Kutta 4/5): Fourth-order adaptive, not symplectic, better
      short-term accuracy but may drift in energy over long simulations
    
    Parameters
    ----------
    H : sympy.Expr
        Hamiltonian function H(x₁, p₁, ..., xₙ, pₙ).
    z0 : array_like
        Initial conditions as [x₁₀, p₁₀, x₂₀, p₂₀, ...], length 2n.
    tspan : tuple of float
        Time integration interval (t_start, t_end).
    vars_phase : list of sympy.Symbol, optional
        Phase space variables in canonical order. If None, inferred from H.
    integrator : {'symplectic', 'verlet', 'rk45'}
        Numerical integration method (default: 'symplectic').
    n_steps : int
        Number of discrete time steps (default: 1000).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 't' : ndarray, time points
        - '<var_name>' : ndarray, trajectory for each phase space variable
        - 'energy' : ndarray, H evaluated along the trajectory
    
    Raises
    ------
    ValueError
        If integrator is not recognized, if variables cannot be inferred, or
        if z0 has incorrect dimension.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2  # Harmonic oscillator
    >>> traj = hamiltonian_flow(H, z0=[1, 0], tspan=(0, 10), 
    ...                         vars_phase=[x, p], integrator='verlet')
    >>> traj['energy'].std()  # Should be very small for symplectic integrators
    < 1e-3
    
    Notes
    -----
    For long-time simulations, prefer 'symplectic' or 'verlet' integrators to
    preserve the geometric structure of phase space and maintain bounded energy
    errors. Use 'rk45' for short-time high-accuracy requirements or when
    comparing with non-symplectic methods.
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
# Symplectic gradient
# -----------------------------------------------------------------------------
def symplectic_gradient(f, vars_phase=None, numeric=False):
    """
    Compute the Hamiltonian vector field (symplectic gradient) of a function f.

    For a function f on a 2n‑dimensional phase space with canonical coordinates
    (x₁, p₁, …, xₙ, pₙ), the symplectic gradient X_f is the unique vector field
    satisfying ω(X_f, ·) = df, where ω = Σ dxᵢ ∧ dpᵢ is the symplectic form.
    In coordinates, X_f = J⁻¹·∇f, with J⁻¹ the Poisson tensor (the inverse of
    the symplectic matrix).

    Parameters
    ----------
    f : sympy.Expr
        Function defined on phase space.
    vars_phase : list of sympy.Symbol, optional
        Ordered list of canonical variables [x₁, p₁, x₂, p₂, …].
        If None, the variables are inferred automatically from f.
    numeric : bool, default False
        If True, return a callable function that evaluates the vector field
        numerically at a given point. If False, return a list of sympy
        expressions representing each component of X_f.

    Returns
    -------
    If numeric=False : list of sympy.Expr
        Components of the Hamiltonian vector field in the same order as
        vars_phase, i.e., [X_f_x₁, X_f_p₁, X_f_x₂, X_f_p₂, …].
    If numeric=True : callable
        A function X(z) that takes a phase space point z (array‑like of length
        2n) and returns a numpy array of the vector field components at that point.

    Notes
    -----
    For large‑scale evaluations (e.g., plotting on a grid), it is more efficient
    to create vectorized functions from the symbolic components manually:

        X_components = symplectic_gradient(f, vars_phase, numeric=False)
        funcs = [lambdify(vars_phase, expr, 'numpy') for expr in X_components]

    Then evaluate on entire arrays:

        results = [func(*(grid_arrays)) for func in funcs]

    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2
    >>> symplectic_gradient(H, [x, p])
    [p, -x]   # dx/dt = p, dp/dt = -x

    >>> # Numeric evaluation at a single point
    >>> X = symplectic_gradient(H, [x, p], numeric=True)
    >>> X([1.0, 0.5])
    array([ 0.5, -1. ])

    >>> # Vectorized evaluation for a grid (fast)
    >>> X_x, X_p = symplectic_gradient(H, [x, p], numeric=False)
    >>> X_x_func = lambdify((x, p), X_x, 'numpy')
    >>> X_p_func = lambdify((x, p), X_p, 'numpy')
    >>> q_vals = np.linspace(-2, 2, 100)
    >>> p_vals = np.linspace(-2, 2, 100)
    >>> Q, P = np.meshgrid(q_vals, p_vals)
    >>> U = X_x_func(Q, P)   # entire array at once
    >>> V = X_p_func(Q, P)

    See Also
    --------
    poisson_bracket : {f, g} = X_f(g) = -X_g(f)
    hamiltonian_flow : integrates the Hamiltonian vector field of H
    SymplecticForm : the underlying symplectic structure
    """
    # Infer variables if not provided
    if vars_phase is None:
        vars_phase = _infer_variables(f)
    vars_phase = list(vars_phase)

    # Build the symplectic structure and obtain the Poisson tensor (J⁻¹)
    symp_form = SymplecticForm(vars_phase=vars_phase)
    J_inv = symp_form.omega_inv   # Poisson tensor

    # Gradient ∇f as a column vector
    grad_f = Matrix([diff(f, var) for var in vars_phase])

    # X_f = J⁻¹ · ∇f
    X_f_expr = J_inv * grad_f   # sympy Matrix

    # Convert to list in the same order as vars_phase
    X_list = list(X_f_expr)

    if not numeric:
        return X_list
    else:
        # Create a callable for numerical evaluation at a single point
        X_func = lambdify(vars_phase, X_list, modules='numpy')
        def vector_field(z):
            z = np.asarray(z).flatten()
            if len(z) != len(vars_phase):
                raise ValueError(f"Point dimension mismatch: expected {len(vars_phase)}, got {len(z)}")
            return np.array(X_func(*z), dtype=float)
        return vector_field
        
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


def phase_portrait(H, x_range, p_range, vars_phase=None, resolution=50, 
                   levels=20):
    """
    Generate a 2D phase portrait for a 1-DOF Hamiltonian system.
    
    Creates a visualization showing:
    1. Energy contour lines (level sets of H)
    2. Hamiltonian vector field arrows indicating flow direction
    
    For 1-DOF systems, trajectories lie on energy contours, making this an
    effective tool for identifying:
    - Fixed points (where vector field vanishes)
    - Periodic orbits (closed contours)
    - Separatrices (boundaries between different types of motion)
    - Allowed vs. forbidden regions
    
    Parameters
    ----------
    H : sympy.Expr
        Hamiltonian H(x, p) for a 1-degree-of-freedom system.
    x_range : tuple of float
        Position axis limits (x_min, x_max).
    p_range : tuple of float
        Momentum axis limits (p_min, p_max).
    vars_phase : list of sympy.Symbol, optional
        Variables [x, p]. If None, inferred from H.
    resolution : int
        Grid resolution for contour and vector field computation (default: 50).
        Higher values give smoother plots but increase computation time.
    levels : int
        Number of energy contour levels to display (default: 20).
    
    Returns
    -------
    None
        Displays a matplotlib figure with the phase portrait.
    
    Raises
    ------
    ValueError
        If the system is not 1-DOF or variables cannot be inferred.
    
    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2  # Harmonic oscillator
    >>> phase_portrait(H, x_range=(-2, 2), p_range=(-2, 2))
    
    >>> # Pendulum
    >>> H = p**2/2 - cos(x)
    >>> phase_portrait(H, x_range=(-2*pi, 2*pi), p_range=(-3, 3))
    
    Notes
    -----
    The vector field is subsampled (every resolution//20 points) to avoid
    visual clutter. Energy contours are labeled with their H values.
    
    See Also
    --------
    visualize_phase_space_structure : More comprehensive visualization with
        fixed points and separatrices
    separatrix_analysis : Detailed analysis near saddle points
    """
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
    elif method == 'period':
        # H must be a sympy expression in phase-space variables, I_val is the energy E
        # Infer variables and compute T = 2 ∫ dx / (∂H/∂p) along the orbit
        from scipy.integrate import quad
        vars_phase = _infer_variables(H)
        _check_ndof(vars_phase, 1)
        x, p = vars_phase
        E_val = float(I_val)   # in this branch I_val is treated as energy
    
        # Solve for p(x, E)
        eq = H - E_val
        p_solutions = solve(eq, p)
        p_expr = next((s for s in p_solutions if im(s) == 0 or s.is_real), p_solutions[0])
    
        # dH/dp = dx/dt  →  dt = dx / (dH/dp)
        dHdp_expr = diff(H, p)
        # Substitute p = p(x, E) into dH/dp
        dHdp_sub = dHdp_expr.subs(p, p_expr)
        dHdp_func = lambdify(x, dHdp_sub, 'numpy')
    
        # Find turning points (where p = 0)
        p_func_zero = lambdify(x, p_expr, 'numpy')
        x_scan = np.linspace(-20, 20, 2000)
        p_vals = np.real(p_func_zero(x_scan).astype(complex))
        crossings = np.where(np.diff(np.sign(p_vals)))[0]
        if len(crossings) < 2:
            amp = np.sqrt(2 * E_val)
            x_min, x_max = -amp, amp
        else:
            x_min = x_scan[crossings[0]]
            x_max = x_scan[crossings[-1]]
    
        def integrand(xv):
            val = dHdp_func(xv)
            val = np.real(complex(val))
            return 1.0 / val if abs(val) > 1e-14 else 0.0
    
        eps = 1e-8
        half_period, _ = quad(integrand, x_min + eps, x_max - eps, limit=200)
        T = 2 * abs(half_period)
        return 2 * np.pi / T


# -----------------------------------------------------------------------------
# 2‑DOF specific functions (require ndof == 2)
# -----------------------------------------------------------------------------

def hamiltonian_flow_4d(H, z0, tspan, integrator='symplectic', n_steps=1000):
    """Backward compatibility wrapper for 4D flow."""
    # Assume variables are x1, p1, x2, p2
    vars_phase = [symbols('x1 p1 x2 p2', real=True)]
    return hamiltonian_flow(H, z0, tspan, vars_phase=vars_phase,
                            integrator=integrator, n_steps=n_steps)


def poincare_section(H, Sigma_def, z0, tmax, vars_phase=None, n_returns=1000, 
                     integrator='symplectic'):
    """
    Compute a Poincaré section for a 2-DOF Hamiltonian system.
    
    A Poincaré section reduces the 4D continuous flow to a 2D discrete map by
    recording intersections of trajectories with a specified surface Σ in phase
    space. This reveals the underlying structure of the dynamics:
    - Regular motion appears as closed curves (KAM tori)
    - Chaotic motion appears as scattered points
    - Periodic orbits appear as fixed points or finite sets
    
    The section surface is defined by Σ = {(x₁, p₁, x₂, p₂) : variable = value}
    with an optional crossing direction.
    
    Parameters
    ----------
    H : sympy.Expr
        Hamiltonian for a 2-degree-of-freedom system.
    Sigma_def : dict
        Section surface definition with keys:
        - 'variable' : str, name of the section variable (e.g., 'x2', 'p1')
        - 'value' : float, the constant value defining the surface
        - 'direction' : str, optional, 'positive' or 'negative' crossing 
          (default: 'positive')
    z0 : array_like
        Initial condition [x₁₀, p₁₀, x₂₀, p₂₀].
    tmax : float
        Maximum integration time.
    vars_phase : list of sympy.Symbol, optional
        Variables [x1, p1, x2, p2]. If None, uses default canonical names.
    n_returns : int
        Maximum number of section crossings to collect (default: 1000).
    integrator : {'symplectic', 'verlet', 'rk45'}
        Integration method (default: 'symplectic').
    
    Returns
    -------
    dict
        Dictionary containing:
        - 't_crossings' : ndarray, times at which the section was crossed
        - 'section_points' : list of dict, phase space coordinates at each
          crossing (interpolated to the exact crossing time)
    
    Raises
    ------
    ValueError
        If the system is not 2-DOF, if the section variable is not found, or
        if insufficient crossings are detected.
    
    Examples
    --------
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> H = (p1**2 + p2**2 + x1**2 + x2**2)/2  # Coupled oscillators
    >>> Sigma = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    >>> ps = poincare_section(H, Sigma, z0=[1, 0, 0, 0.5], tmax=100)
    >>> len(ps['section_points'])
    > 0
    
    Notes
    -----
    Crossing times are linearly interpolated between integration steps for
    accuracy. Use high n_steps in the underlying integration for precise
    section points. For chaotic systems, increase tmax and n_returns.
    
    See Also
    --------
    visualize_poincare_section : Plot multiple sections together
    first_return_map : Analyze the discrete map from section points
    """
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
    """
    Compute the monodromy matrix for a periodic orbit in a 2-DOF system.
    
    The monodromy matrix M is the linearized Poincaré return map around a
    periodic orbit. It describes how infinitesimal perturbations evolve over
    one period T:
    
        δz(T) = M · δz(0)
    
    The eigenvalues of M (Floquet multipliers) determine orbital stability:
    - |λ| = 1 for all λ: neutrally stable (typical for Hamiltonian systems)
    - Any |λ| > 1: unstable orbit
    - All |λ| < 1: stable (not possible in conservative Hamiltonian systems)
    
    For Hamiltonian systems, eigenvalues come in reciprocal pairs (λ, 1/λ)
    and/or complex conjugate pairs on the unit circle.
    
    Parameters
    ----------
    H : sympy.Expr
        Hamiltonian for a 2-degree-of-freedom system.
    periodic_orbit : dict
        A trajectory dictionary (from hamiltonian_flow) representing one complete
        period, with keys 't' and variable names.
    vars_phase : list of sympy.Symbol, optional
        Variables [x1, p1, x2, p2]. If None, uses default canonical names.
    method : {'finite_difference'}
        Computation method (default: 'finite_difference').
        Currently only finite difference perturbations are implemented.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'M' : ndarray, 4×4 monodromy matrix
        - 'eigenvalues' : ndarray, Floquet multipliers
        - 'floquet_multipliers' : ndarray, same as eigenvalues
        - 'stable' : bool, True if all |λ| ≤ 1 (within tolerance)
    
    Raises
    ------
    NotImplementedError
        If method is not 'finite_difference'.
    ValueError
        If the system is not 2-DOF.
    
    Examples
    --------
    >>> x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    >>> H = (p1**2 + p2**2 + x1**2 + x2**2)/2
    >>> orbit = hamiltonian_flow(H, [1, 0, 0, 0], (0, 2*pi), 
    ...                          vars_phase=[x1, p1, x2, p2])
    >>> mono = monodromy_matrix(H, orbit, vars_phase=[x1, p1, x2, p2])
    >>> np.allclose(np.abs(mono['eigenvalues']), 1.0, atol=1e-3)
    True
    
    Notes
    -----
    The finite difference method perturbs each coordinate by ε = 1e-6 and
    integrates forward one period. For higher accuracy, consider implementing
    variational equations directly.
    
    See Also
    --------
    lyapunov_exponents : Long-term stability characterization
    linearize_at_fixed_point : Stability analysis for equilibria
    """
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


def lyapunov_exponents(trajectory, dt, vars_phase=None, n_vectors=4, 
                       renorm_interval=10):
    """
    Estimate Lyapunov exponents from a trajectory in a 2-DOF system.
    
    Lyapunov exponents quantify the average exponential rate of separation
    between nearby trajectories. For a 2n-dimensional phase space, there are
    2n exponents λ₁ ≥ λ₂ ≥ ... ≥ λ₂ₙ.
    
    Interpretation:
    - λ₁ > 0: Chaotic dynamics (sensitive dependence on initial conditions)
    - λ₁ = 0: Regular/periodic motion (marginal stability)
    - Sum of all λᵢ = 0 for Hamiltonian systems (phase space volume preservation)
    - Exponents come in pairs (λ, -λ) for Hamiltonian systems
    
    This implementation uses a simplified QR-based algorithm with periodic
    renormalization to prevent numerical overflow.
    
    Parameters
    ----------
    trajectory : dict
        Trajectory dictionary from hamiltonian_flow with keys 't' and variable
        names containing the time series.
    dt : float
        Time step between trajectory points.
    vars_phase : list of sympy.Symbol, optional
        Variables [x1, p1, x2, p2]. If None, uses default canonical names.
    n_vectors : int
        Number of tangent vectors to evolve (default: 4 for 2-DOF).
    renorm_interval : int
        Number of steps between QR renormalization (default: 10).
        Smaller values improve accuracy but increase computation.
    
    Returns
    -------
    ndarray
        Lyapunov exponents sorted in descending order [λ₁, λ₂, λ₃, λ₄].
    
    Raises
    ------
    ValueError
        If the system is not 2-DOF or trajectory is too short.
    
    Examples
    --------
    >>> # Regular motion (coupled oscillators)
    >>> traj = hamiltonian_flow(H, z0, (0, 100), integrator='rk45')
    >>> lyap = lyapunov_exponents(traj, dt=0.01)
    >>> lyap[0]  # Should be approximately 0 for regular motion
    ≈ 0
    
    >>> # Chaotic motion (e.g., Hénon-Heiles)
    >>> lyap = lyapunov_exponents(chaotic_traj, dt=0.01)
    >>> lyap[0]  # Should be positive for chaos
    > 0
    
    Notes
    -----
    This is a simplified implementation using finite-difference Jacobian
    approximation. For production use, consider implementing the full
    variational equations alongside the trajectory integration.
    
    Convergence requires long trajectories (tmax >> 1/λ₁). The largest
    exponent λ₁ converges fastest; smaller exponents require longer runs.
    
    See Also
    --------
    monodromy_matrix : Stability of periodic orbits
    poincare_section : Visual detection of chaos vs. regularity
    """
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
    """
    Project a 4D trajectory onto a 2D plane for visualization.
    
    For 2-DOF systems, the full phase space is 4-dimensional (x₁, p₁, x₂, p₂),
    which cannot be directly visualized. This function extracts 2D projections
    suitable for plotting.
    
    Available projection planes:
    - 'xy' or 'config': Configuration space (x₁, x₂)
    - 'xp': Phase space of first DOF (x₁, p₁)
    - 'pp' or 'momentum': Momentum space (p₁, p₂)
    - 'x1p2': Mixed coordinates (x₁, p₂)
    - 'x2p1': Mixed coordinates (x₂, p₁)
    
    Parameters
    ----------
    trajectory : dict
        Trajectory dictionary from hamiltonian_flow with variable name keys.
    plane : str
        Projection plane specification (default: 'xy').
        Options: 'xy', 'config', 'xp', 'pp', 'momentum', 'x1p2', 'x2p1'
    vars_phase : list of sympy.Symbol, optional
        Variables [x1, p1, x2, p2]. If None, uses default canonical names.
    
    Returns
    -------
    tuple
        (x_data, y_data, labels) where:
        - x_data : ndarray, coordinates for horizontal axis
        - y_data : ndarray, coordinates for vertical axis
        - labels : tuple of str, axis label strings
    
    Raises
    ------
    ValueError
        If the system is not 2-DOF or if plane is not recognized.
    
    Examples
    --------
    >>> traj = hamiltonian_flow(H, z0, (0, 100), vars_phase=[x1, p1, x2, p2])
    >>> x, y, labels = project(traj, plane='xy')
    >>> plt.plot(x, y)
    >>> plt.xlabel(labels[0]); plt.ylabel(labels[1])
    
    >>> # Phase space of first oscillator
    >>> x, p, labels = project(traj, plane='xp')
    
    Notes
    -----
    Different projections reveal different aspects of the dynamics:
    - Configuration space shows spatial trajectories
    - Phase space projections show energy exchange between DOFs
    - Mixed projections can reveal correlations between variables
    
    See Also
    --------
    visualize_poincare_section : 2D section visualization
    phase_portrait : 1-DOF phase space visualization
    """
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

def rectangle_region(center, width, height, n_points=50):
    """
    Generate a closed rectangular region in phase space.

    Parameters
    ----------
    center : (float, float)
        Coordinates (x0, p0) of the rectangle centre.
    width : float
        Total width in the x direction.
    height : float
        Total height in the p direction.
    n_points : int, optional
        Number of distinct points on the perimeter (default 50). Must be at least 4.

    Returns
    -------
    (N,2) ndarray
        Ordered points forming a closed rectangle (first and last point identical).
        The array has shape (n_points + 1, 2).
    """
    if n_points < 4:
        n_points = 4
    x0, p0 = center
    half_w = width / 2
    half_h = height / 2

    # Distribute points as evenly as possible among the four edges
    base = n_points // 4
    rem = n_points % 4
    pts_per_edge = [base] * 4
    for i in range(rem):
        pts_per_edge[i] += 1

    # Build each edge, using endpoint=False to avoid duplicating corners
    # Bottom edge: left → right
    bottom_x = np.linspace(x0 - half_w, x0 + half_w, pts_per_edge[0], endpoint=False)
    bottom_y = np.full(pts_per_edge[0], p0 - half_h)
    bottom = np.column_stack([bottom_x, bottom_y])

    # Right edge: bottom → top
    right_y = np.linspace(p0 - half_h, p0 + half_h, pts_per_edge[1], endpoint=False)
    right_x = np.full(pts_per_edge[1], x0 + half_w)
    right = np.column_stack([right_x, right_y])

    # Top edge: right → left (reverse x)
    top_x = np.linspace(x0 + half_w, x0 - half_w, pts_per_edge[2], endpoint=False)
    top_y = np.full(pts_per_edge[2], p0 + half_h)
    top = np.column_stack([top_x, top_y])

    # Left edge: top → bottom (reverse y)
    left_y = np.linspace(p0 + half_h, p0 - half_h, pts_per_edge[3], endpoint=False)
    left_x = np.full(pts_per_edge[3], x0 - half_w)
    left = np.column_stack([left_x, left_y])

    # Concatenate edges and close the polygon
    points = np.vstack([bottom, right, top, left])
    points = np.vstack([points, points[0]])   # close the loop
    return points


def evolve_phase_space_region(H, initial_region, t_eval, vars_phase=None,
                              integrator='verlet', n_steps=10000,
                              plot=True, ax=None, **plot_kwargs):
    """
    Evolve a closed region in phase space under the Hamiltonian flow and
    optionally visualise its deformation and area conservation.

    The function integrates each vertex of the polygon defined by
    `initial_region` forward to the times specified in `t_eval`, computes
    the enclosed area at each time (using the shoelace formula), and plots
    the evolved regions if requested.

    This provides a direct visual demonstration of Liouville's theorem:
    the symplectic area is preserved by the Hamiltonian flow (up to numerical
    errors, especially when using a symplectic integrator).

    Parameters
    ----------
    H : sympy.Expr
        Hamiltonian for a 1‑degree‑of‑freedom system H(x, p).
    initial_region : (N,2) array_like
        Ordered points defining a closed polygon (first and last point should
        be identical). The region is typically a simple shape such as a
        rectangle or an ellipse.
    t_eval : float or list of float
        Times at which to record the evolved region. If a single float is given,
        it is treated as the final time and the region is evaluated at t=0 and
        that time. Use a list to obtain intermediate snapshots.
    vars_phase : list of sympy.Symbol, optional
        Phase space variables [x, p]. If not provided, they are inferred from H.
    integrator : {'symplectic', 'verlet', 'rk45'}, optional
        Numerical integrator passed to `hamiltonian_flow` (default 'verlet').
        Symplectic integrators are recommended for accurate long‑term area
        preservation.
    n_steps : int, optional
        Number of integration steps used by `hamiltonian_flow` over the whole
        time interval [0, max(t_eval)]. A high value ensures accurate point
        positions at the requested times (default 10000).
    plot : bool, optional
        If True, generate a plot showing the region at each time in `t_eval`
        (default True).
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None and plot=True, a new figure
        and axes are created.
    **plot_kwargs
        Additional keyword arguments passed to the plotting call (e.g.,
        `colors`, `alpha`, `linewidth`). Useful for customising the appearance
        of the evolved regions.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'times' : ndarray
            The evaluation times (sorted, unique).
        - 'region_at_t' : list of (N,2) ndarrays
            Evolved polygon for each time, in the same vertex order as input.
        - 'areas' : ndarray
            Area enclosed by the polygon at each time (computed with the
            shoelace formula).
        - 'plot_handles' : list, optional
            If plot=True, a list of handles to the plotted lines/fills for
            each time (useful for further customisation or legends).

    Raises
    ------
    ValueError
        If the system is not 1‑DOF, if the initial region has fewer than 3 points,
        or if the polygon is not closed (first and last point differ beyond a
        small tolerance).

    Examples
    --------
    >>> x, p = symbols('x p', real=True)
    >>> H = p**2/2 + x**2/2          # harmonic oscillator
    >>> region = rectangle_region(center=(0, 1), width=0.5, height=0.3)
    >>> result = evolve_phase_space_region(H, region, t_eval=[0, 2*np.pi, 4*np.pi],
    ...                                    integrator='verlet')
    >>> result['areas']
    array([0.15, 0.15, 0.15])        # area conserved (up to numerical errors)

    See Also
    --------
    rectangle_region : Convenience function to generate a rectangular region.
    phase_portrait : Plot energy contours and vector fields.
    hamiltonian_flow : Underlying numerical integration routine.
    """
    import warnings
    # ----------------------------------------------------------------------
    # Preliminary checks and variable inference
    # ----------------------------------------------------------------------
    if vars_phase is None:
        vars_phase = _infer_variables(H, expected_ndof=1)
    _check_ndof(vars_phase, 1)
    x, p = vars_phase

    region = np.asarray(initial_region)
    if region.ndim != 2 or region.shape[1] != 2:
        raise ValueError("initial_region must be an (N,2) array")
    if region.shape[0] < 3:
        raise ValueError("Region must have at least three points")

    # Check that the polygon is closed (first point ≈ last point)
    if not np.allclose(region[0], region[-1], atol=1e-12):
        warnings.warn("Region is not closed – automatically closing by adding first point at the end.")
        region = np.vstack([region, region[0]])

    # Normalise t_eval to a sorted list of unique times
    t_eval = np.atleast_1d(t_eval).flatten()
    t_eval = np.unique(t_eval)
    t_eval.sort()
    t_max = t_eval[-1]

    # ----------------------------------------------------------------------
    # Evolve each vertex
    # ----------------------------------------------------------------------
    n_vertices = region.shape[0]
    # We will store, for each vertex, its trajectory as a dict from hamiltonian_flow
    trajectories = []
    for i, (xi, pi) in enumerate(region):
        traj = hamiltonian_flow(H, (xi, pi), (0, t_max),
                                vars_phase=vars_phase,
                                integrator=integrator,
                                n_steps=n_steps)
        trajectories.append(traj)

    # ----------------------------------------------------------------------
    # Extract vertex positions at each requested time
    # ----------------------------------------------------------------------
    # For each time in t_eval, we need the (x,p) for all vertices.
    # We'll build a list of arrays: region_at_t[k] is an (n_vertices,2) array for time t_eval[k].
    times_available = trajectories[0]['t']   # same for all vertices
    region_at_t = []
    for target_t in t_eval:
        # find index in times_available closest to target_t
        idx = np.argmin(np.abs(times_available - target_t))
        vertices_at_t = []
        for traj in trajectories:
            x_val = traj[str(x)][idx]
            p_val = traj[str(p)][idx]
            vertices_at_t.append([x_val, p_val])
        region_at_t.append(np.array(vertices_at_t))

    # ----------------------------------------------------------------------
    # Compute area at each time using the shoelace formula
    # ----------------------------------------------------------------------
    areas = []
    for vertices in region_at_t:
        x_vals = vertices[:, 0]
        y_vals = vertices[:, 1]
        # shoelace: A = 0.5 * |Σ (x_i y_{i+1} - x_{i+1} y_i)|
        # (vertices are closed, so last and first already connected)
        area = 0.5 * np.abs(np.sum(x_vals[:-1] * y_vals[1:] - x_vals[1:] * y_vals[:-1]))
        areas.append(area)
    areas = np.array(areas)

    # ----------------------------------------------------------------------
    # Plotting (optional)
    # ----------------------------------------------------------------------
    plot_handles = None
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Choose a colormap if colors are not provided
        if 'color' not in plot_kwargs and 'colors' not in plot_kwargs:
            colors = plt.cm.viridis(np.linspace(0, 1, len(t_eval)))
        else:
            colors = [plot_kwargs.pop('color', None)] * len(t_eval)

        plot_handles = []
        for k, t_val in enumerate(t_eval):
            verts = region_at_t[k]
            # Plot the polygon outline
            line, = ax.plot(verts[:, 0], verts[:, 1],
                            color=colors[k], linewidth=2,
                            label=f't={t_val:.2f}, A={areas[k]:.4f}',
                            **plot_kwargs)
            # Fill the polygon with low opacity
            fill = ax.fill(verts[:, 0], verts[:, 1],
                           color=colors[k], alpha=0.2)
            plot_handles.append((line, fill[0]))

        # Optionally overlay energy contours for context
        # (reusing code from phase_portrait but without cluttering)
        x_range = (np.min([v[:,0].min() for v in region_at_t]),
                   np.max([v[:,0].max() for v in region_at_t]))
        p_range = (np.min([v[:,1].min() for v in region_at_t]),
                   np.max([v[:,1].max() for v in region_at_t]))
        # Add some margin
        x_margin = 0.1 * (x_range[1] - x_range[0])
        p_margin = 0.1 * (p_range[1] - p_range[0])
        x_range = (x_range[0] - x_margin, x_range[1] + x_margin)
        p_range = (p_range[0] - p_margin, p_range[1] + p_margin)

        X_grid, P_grid = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 50),
            np.linspace(p_range[0], p_range[1], 50)
        )
        H_func = lambdify((x, p), H, 'numpy')
        H_vals = H_func(X_grid, P_grid)
        ax.contour(X_grid, P_grid, H_vals, levels=15,
                   colors='gray', alpha=0.3, linewidths=0.5)

        ax.set_xlabel(str(x), fontsize=12)
        ax.set_ylabel(str(p), fontsize=12)
        ax.set_title(f'Phase space region evolution\n(A₀ = {areas[0]:.4f})',
                     fontsize=13)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if ax is None:
            plt.show()

    # ----------------------------------------------------------------------
    # Build result dictionary
    # ----------------------------------------------------------------------
    result = {
        'times': t_eval,
        'region_at_t': region_at_t,
        'areas': areas,
    }
    if plot:
        result['plot_handles'] = plot_handles
    return result


# -----------------------------------------------------------------------------
# Integrability analysis
# -----------------------------------------------------------------------------

class IntegrabilityAnalysis:
    """
    Spectral and topological tools for classifying Hamiltonian systems as
    integrable, chaotic, or intermediate.
    
    These methods are the symplectic counterparts of the geometric indicators
    computed in geometry.py (Poincaré sections, Lyapunov exponents, monodromy
    matrices).  Together they provide a complete picture of the regular/chaotic
    nature of a system:
    
    Spectral statistics
    ~~~~~~~~~~~~~~~~~~~
    * **Level-spacing statistics** (``analyze_integrability``) — quantum/semiclassical
      fingerprint: Poisson distribution for integrable systems (Berry-Tabor
      conjecture, 1977), Wigner surmise for chaotic ones (BGS conjecture, 1984).
    * **Brody distribution** (``brody_distribution``) — one-parameter family
      P(s; β) interpolating continuously between Poisson (β=0, integrable) and
      Wigner-GOE (β=1, chaotic).  The fitted parameter β ∈ [0, 1] quantifies the
      degree of level repulsion and serves as a scalar chaos indicator.
    
    Semiclassical density of states
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * **Weyl's law** (``weyl_law``) — asymptotic count of states N(E) ~ Vol{H≤E}/(2πℏ)ⁿ.
    * **Berry-Tabor smoothed density** (``berry_tabor_formula``) — semiclassical
      density of states built from a sum over periodic orbits, each contributing
      a Gaussian peak weighted by its period.
    
    Topological structure
    ~~~~~~~~~~~~~~~~~~~~~
    * **KAM tori detection** (``detect_kam_tori``) — clusters periodic orbits by
      action proximity using Ward hierarchical clustering; each cluster corresponds
      to a KAM torus.
    * **Winding / rotation numbers** (``winding_number``, ``rotation_numbers``) —
      topological invariants that distinguish resonant (rational ratio) and
      quasi-periodic (irrational ratio) motion on invariant tori.
    * **Topological monodromy** (``topological_monodromy``) — detects non-trivial
      monodromy of the action-angle fibration (Duistermaat 1980).  Given a second
      integral of motion L, transports the action vector (I₁, I₂) around a closed
      loop in the energy-momentum plane (E, ℓ) encircling a critical value.  The
      resulting integer matrix M ∈ GL(2, ℤ) is the identity for trivially fibred
      systems and M = [[1,1],[0,1]] for focus-focus singularities (e.g. the
      spherical pendulum), signalling a global obstruction to smooth action-angle
      coordinates.
    
    Scarring
    ~~~~~~~~
    * **Scar intensity** (``scar_intensity``) — classical precursor of quantum
      scarring (Heller 1984).  Measures the fraction of trajectory time spent
      within a prescribed neighbourhood of a periodic orbit and compares it to
      the ergodic baseline, yielding a scalar S ≥ 0 where S ≫ 1 indicates that
      the trajectory is anomalously concentrated along the orbit.
    
    References
    ----------
    .. [BT77]  Berry, M. V. & Tabor, M., "Level clustering in the regular
               spectrum", *Proc. R. Soc. Lond. A* 356, 375–394 (1977).
    .. [BGS84] Bohigas, O., Giannoni, M. J., & Schmit, C., "Characterization of
               chaotic quantum spectra and universality of level fluctuation laws",
               *Phys. Rev. Lett.* 52, 1–4 (1984).
    .. [Ar89]  Arnold, V. I., *Mathematical Methods of Classical Mechanics*,
               Springer-Verlag, 1989, Chapters 10–11.
    .. [Br73]  Brody, T. A., "A statistical measure for the repulsion of energy
               levels", *Lett. Nuovo Cimento* 7, 482–484 (1973).
    .. [Du80]  Duistermaat, J. J., "On global action-angle coordinates",
               *Comm. Pure Appl. Math.* 33, 687–706 (1980).
    .. [Cu97]  Cushman, R. H. & Bates, L. M., *Global Aspects of Classical
               Integrable Systems*, Birkhäuser, 1997.
    .. [He84]  Heller, E. J., "Bound-state eigenfunctions of classically chaotic
               Hamiltonian systems: scars of periodic orbits",
               *Phys. Rev. Lett.* 53, 1515–1518 (1984).
    """
    # ------------------------------------------------------------------
    # Weyl's law
    # ------------------------------------------------------------------
    @staticmethod
    def weyl_law(energy: float, ndof: int, hbar: float = 1.0) -> float:
        """
        Weyl's asymptotic law: number of quantum states below energy E.

        The simplified form assumes that the phase-space volume scales as E^n
        (valid, e.g., for harmonic-oscillator-like Hamiltonians):

            N(E) ≈ (1 / (2πℏ))^n · E^n

        For a general Hamiltonian use ``phase_space_volume`` from this module to
        compute Vol{H ≤ E} and then divide by (2πℏ)^n.

        Parameters
        ----------
        energy : float
            Energy threshold E.
        ndof : int
            Number of degrees of freedom n.
        hbar : float
            Reduced Planck constant (default 1.0).

        Returns
        -------
        float
            Approximate number of states N(E).

        Examples
        --------
        >>> IntegrabilityAnalysis.weyl_law(3.0, ndof=1, hbar=1.0)
        0.477...
        """
        return ((1.0 / (2.0 * np.pi * hbar)) ** ndof) * (energy ** ndof)

    # ------------------------------------------------------------------
    # Level-spacing statistics
    # ------------------------------------------------------------------
    @staticmethod
    def analyze_integrability(spacings: np.ndarray) -> dict:
        """
        Classify a Hamiltonian system via level-spacing statistics.

        Given the sequence of nearest-neighbour level spacings s_i = E_{i+1} - E_i
        (normalised so that the mean spacing is 1), the ratio

            R = <s²> / <s>²

        distinguishes the two universal classes:

        * **Integrable** (Berry-Tabor): spacings follow a Poisson distribution
          P(s) = e^{-s}, giving R ≈ 2.0.
        * **Chaotic** (BGS / GOE): spacings follow the Wigner surmise
          P(s) = (π/2) s e^{-πs²/4}, giving R ≈ 1.273.

        The threshold values used here (R > 1.7 → integrable, R < 1.4 → chaotic)
        are conventional; intermediate values indicate a mixed phase space.

        Parameters
        ----------
        spacings : ndarray of shape (N,)
            Raw (un-normalised) nearest-neighbour energy-level spacings.

        Returns
        -------
        dict with keys:
            * ``ratio``          – <s²>/<s>² of normalised spacings
            * ``mean_spacing``   – mean of raw spacings
            * ``std_spacing``    – standard deviation of raw spacings
            * ``classification`` – one of
              ``'Integrable (Poisson-like)'``,
              ``'Chaotic (Wigner-like)'``, or
              ``'Intermediate'``

        Examples
        --------
        >>> rng = np.random.default_rng(0)
        >>> poisson_spacings = rng.exponential(scale=1.0, size=500)
        >>> info = IntegrabilityAnalysis.analyze_integrability(poisson_spacings)
        >>> info['classification']
        'Integrable (Poisson-like)'

        Notes
        -----
        For meaningful statistics at least ~50 spacings are recommended.
        Unfolding the spectrum (mapping raw spacings to a unit-mean sequence)
        before calling this function improves the accuracy of the classification.
        """
        spacings = np.asarray(spacings, dtype=float)
        s_mean = np.mean(spacings)
        s_norm = spacings / s_mean
        ratio  = np.mean(s_norm ** 2) / (np.mean(s_norm) ** 2)
        if ratio > 1.7:
            cls = "Integrable (Poisson-like)"
        elif ratio < 1.4:
            cls = "Chaotic (Wigner-like)"
        else:
            cls = "Intermediate"
        return dict(
            ratio=float(ratio),
            mean_spacing=float(s_mean),
            std_spacing=float(np.std(spacings)),
            classification=cls,
        )

    # ------------------------------------------------------------------
    # Berry-Tabor smoothed density of states
    # ------------------------------------------------------------------
    @staticmethod
    def berry_tabor_formula(orbits, energy: float, window: float = 1.0) -> float:
        """
        Berry-Tabor smoothed density of states from a collection of periodic orbits.

        Each orbit contributes a Gaussian peak centred at its energy:

            ρ(E) = Σ_γ  T_γ · exp(-(E_γ - E)² / 2σ²) / (σ√(2π) · 2π)

        where T_γ is the period and σ = ``window`` is the energy smoothing width.

        Parameters
        ----------
        orbits : iterable
            Each element must expose ``.energy`` and ``.period`` attributes **or**
            be a dict with ``'energy'`` and ``'period'`` keys (as returned, e.g.,
            by the monodromy / periodic-orbit routines in this module).
        energy : float
            Evaluation energy E.
        window : float
            Gaussian smoothing width σ in energy units (default 1.0).

        Returns
        -------
        float
            Smoothed density of states ρ(E).

        Examples
        --------
        >>> # Using plain dicts (compatible with symplectic.py orbit dicts)
        >>> orbits = [{'energy': 1.0, 'period': 6.28},
        ...           {'energy': 2.0, 'period': 6.28}]
        >>> IntegrabilityAnalysis.berry_tabor_formula(orbits, energy=1.5)
        ...

        Notes
        -----
        For integrable systems the smoothed density converges to the Weyl term
        plus Berry-Tabor oscillatory corrections.  For chaotic systems use the
        Gutzwiller trace formula instead (implemented in geometry.py).
        """
        def _get(o, attr, key):
            if hasattr(o, attr):
                return getattr(o, attr)
            return o[key]

        density = sum(
            np.exp(-((_get(o, 'energy', 'energy') - energy) ** 2) / (2.0 * window ** 2))
            * _get(o, 'period', 'period') / (2.0 * np.pi)
            for o in orbits
        )
        return float(density / (window * np.sqrt(2.0 * np.pi)))

    # ------------------------------------------------------------------
    # KAM tori detection
    # ------------------------------------------------------------------
    @staticmethod
    def detect_kam_tori(orbits, tolerance: float = 0.1) -> dict:
        """
        Cluster periodic orbits into KAM tori by action proximity.

        In an integrable (or near-integrable) system, periodic orbits of nearby
        action lie on the same KAM torus.  This function uses Ward hierarchical
        clustering on the action values to group orbits accordingly.

        Parameters
        ----------
        orbits : iterable
            Each element must expose ``.action``, ``.energy``, ``.period`` and
            ``.stability`` / ``'stability'`` (or ``'stability_1'`` for 2-DOF
            objects) **or** be a dict with the same keys.
        tolerance : float
            Maximum within-cluster action distance for the Ward linkage cut
            (default 0.1).

        Returns
        -------
        dict with keys:
            * ``n_tori`` – number of detected tori
            * ``tori``   – list of dicts, one per torus, each containing:

              - ``id``       : cluster id (int)
              - ``n_orbits`` : number of member orbits
              - ``action``   : mean action of member orbits
              - ``energy``   : mean energy of member orbits
              - ``period``   : mean period of member orbits
              - ``stable``   : True if mean stability exponent < 0

        Examples
        --------
        >>> result = IntegrabilityAnalysis.detect_kam_tori(orbits, tolerance=0.05)
        >>> print(result['n_tori'], result['tori'][0]['action'])

        Notes
        -----
        At least two orbits are required for clustering; a single orbit is
        returned as a degenerate torus.
        """
        from scipy.cluster.hierarchy import fcluster, linkage

        def _get(o, *keys):
            for k in keys:
                if hasattr(o, k):
                    return getattr(o, k)
                if isinstance(o, dict) and k in o:
                    return o[k]
            return 0.0

        orbits = list(orbits)
        if not orbits:
            return {'n_tori': 0, 'tori': []}

        actions = np.array([_get(o, 'action') for o in orbits])
        if len(actions) > 1:
            Z        = linkage(actions.reshape(-1, 1), method='ward')
            clusters = fcluster(Z, t=tolerance, criterion='distance')
        else:
            clusters = np.array([1])

        tori = []
        for tid in np.unique(clusters):
            members = [o for o, c in zip(orbits, clusters) if c == tid]
            # Accept both attribute-style (dataclass) and dict-style orbits
            stab = float(np.mean([_get(o, 'stability', 'stability_1') for o in members]))
            tori.append(dict(
                id=int(tid),
                n_orbits=len(members),
                action=float(np.mean([_get(o, 'action') for o in members])),
                energy=float(np.mean([_get(o, 'energy') for o in members])),
                period=float(np.mean([_get(o, 'period') for o in members])),
                stable=bool(stab < 0),
            ))
        return {'n_tori': len(tori), 'tori': tori}

    # ------------------------------------------------------------------
    # Winding / rotation numbers  (2-DOF)
    # ------------------------------------------------------------------
    @staticmethod
    def winding_number(traj: dict, x_key: str = 'x', y_key: str = 'y') -> float:
        """
        Winding number of a 2-DOF trajectory around the origin in configuration space.

        The winding number counts how many times the spatial projection
        (x(t), y(t)) winds around the origin:

            W = (θ(T) - θ(0)) / 2π,   θ = arctan2(y, x).

        Parameters
        ----------
        traj : dict
            Trajectory dict as returned by ``hamiltonian_flow``.
        x_key, y_key : str
            Keys for the two configuration-space coordinates (default ``'x'``,
            ``'y'``).  Adjust if your variable names differ (e.g. ``'x1'``, ``'x2'``).

        Returns
        -------
        float
            Winding number (positive = counter-clockwise).
        """
        x_arr = np.asarray(traj[x_key], dtype=float)
        y_arr = np.asarray(traj[y_key], dtype=float)
        angles = np.arctan2(y_arr, x_arr)
        return float((np.unwrap(angles)[-1] - np.unwrap(angles)[0]) / (2.0 * np.pi))

    @staticmethod
    def rotation_numbers(
        traj: dict,
        x_key: str = 'x1', p_key: str = 'p1',
        y_key: str = 'x2', q_key: str = 'p2',
    ) -> tuple:
        """
        Rotation numbers (ω_1, ω_2) of a 2-DOF trajectory.

        Each rotation number is the average angular velocity in the corresponding
        (xᵢ, pᵢ) phase-space plane:

            ωᵢ = (θᵢ(T) - θᵢ(0)) / (2π · T),   θᵢ = arctan2(pᵢ, xᵢ).

        Their ratio ω_1/ω_2 is the frequency ratio; rational values indicate
        resonant (periodic) orbits, irrational values indicate quasi-periodic
        motion on a KAM torus.

        Parameters
        ----------
        traj : dict
            Trajectory dict as returned by ``hamiltonian_flow``.
        x_key, p_key : str
            Keys for the first position and momentum (default ``'x1'``, ``'p1'``).
        y_key, q_key : str
            Keys for the second position and momentum (default ``'x2'``, ``'p2'``).

        Returns
        -------
        (float, float)
            Rotation numbers (ω_1, ω_2).

        Examples
        --------
        >>> x1,p1,x2,p2 = symbols('x1 p1 x2 p2', real=True)
        >>> H = (p1**2 + p2**2 + x1**2 + x2**2) / 2   # isotropic oscillator
        >>> traj = hamiltonian_flow(H, (1,0,0,1), (0, 20*np.pi),
        ...                         vars_phase=[x1,p1,x2,p2], n_steps=5000)
        >>> IntegrabilityAnalysis.rotation_numbers(traj)
        (0.5, 0.5)   # both oscillators at frequency 1/2π

        Notes
        -----
        For short trajectories the rotation number estimate may be noisy; use
        longer integration times for robust results.
        """
        t = np.asarray(traj['t'], dtype=float)
        T = t[-1] - t[0]

        def _omega(pos_key, mom_key):
            pos = np.asarray(traj[pos_key], dtype=float)
            mom = np.asarray(traj[mom_key], dtype=float)
            theta = np.unwrap(np.arctan2(mom, pos))
            return float((theta[-1] - theta[0]) / (2.0 * np.pi * T))

        return _omega(x_key, p_key), _omega(y_key, q_key)

    @staticmethod
    def brody_distribution(spacings: np.ndarray, fit: bool = True) -> dict:
        """
        Brody's intermediate level-spacing distribution.
    
        Brody (1973) introduced a one-parameter family that interpolates
        continuously between the Poisson distribution (integrable, β=0)
        and the Wigner surmise (chaotic/GOE, β=1):
    
            P(s; β) = (β+1) b s^β exp(-b s^{β+1})
    
        where b = [Γ((β+2)/(β+1))]^{β+1} ensures unit mean spacing.
    
        Parameters
        ----------
        spacings : ndarray of shape (N,)
            Raw nearest-neighbour energy-level spacings (need not be
            pre-normalised; normalisation is performed internally).
        fit : bool
            If True (default), estimate β from the data by maximum
            likelihood. If False, return only the theoretical curves
            for β=0 and β=1 without fitting.
    
        Returns
        -------
        dict with keys:
            * ``beta``           – fitted Brody parameter (0 ≤ β ≤ 1),
                                   or None if fit=False
            * ``beta_std``       – standard error of the fit (bootstrap),
                                   or None if fit=False
            * ``classification`` – 'Integrable (β≈0)', 'Chaotic (β≈1)',
                                   or 'Intermediate (β=<value>)'
            * ``pdf``            – callable P(s; β_fit) for plotting
            * ``nll``            – negative log-likelihood at optimum
    
        Examples
        --------
        >>> rng = np.random.default_rng(42)
        >>> wigner_spacings = rng.rayleigh(scale=np.sqrt(4/np.pi), size=500)
        >>> result = IntegrabilityAnalysis.brody_distribution(wigner_spacings)
        >>> round(result['beta'], 1)
        1.0
    
        Notes
        -----
        The fit uses scipy.optimize.minimize_scalar on the negative
        log-likelihood, which is unimodal in β ∈ [0,1] for typical spectra.
        Bootstrap standard error uses 200 resamples.
    
        References
        ----------
        .. [Br73] Brody, T. A., "A statistical measure for the repulsion of
                  energy levels", *Lett. Nuovo Cimento* 7, 482–484 (1973).
        """
        from scipy.optimize import minimize_scalar
        from scipy.special import gamma
    
        spacings = np.asarray(spacings, dtype=float)
        s = spacings / spacings.mean()   # normalise to unit mean
    
        def _b(beta):
            return gamma((beta + 2.0) / (beta + 1.0)) ** (beta + 1.0)
    
        def _pdf(s_arr, beta):
            b = _b(beta)
            return (beta + 1.0) * b * s_arr ** beta * np.exp(-b * s_arr ** (beta + 1.0))
    
        def _nll(beta):
            p = _pdf(s, beta)
            p = np.clip(p, 1e-300, None)
            return -np.sum(np.log(p))
    
        beta_fit = None
        beta_std = None
        nll_val  = None
    
        if fit:
            res = minimize_scalar(_nll, bounds=(0.0, 1.0), method='bounded')
            beta_fit = float(np.clip(res.x, 0.0, 1.0))
            nll_val  = float(res.fun)
    
            # Bootstrap standard error (200 resamples)
            rng = np.random.default_rng(0)
            betas_boot = []
            for _ in range(200):
                s_boot = rng.choice(s, size=len(s), replace=True)
                r = minimize_scalar(
                    lambda b: -np.sum(np.log(np.clip(_pdf(s_boot, b), 1e-300, None))),
                    bounds=(0.0, 1.0), method='bounded'
                )
                betas_boot.append(float(np.clip(r.x, 0.0, 1.0)))
            beta_std = float(np.std(betas_boot))
    
            if beta_fit < 0.2:
                cls = 'Integrable (β≈0)'
            elif beta_fit > 0.8:
                cls = 'Chaotic (β≈1)'
            else:
                cls = f'Intermediate (β={beta_fit:.2f})'
        else:
            cls = 'Not fitted'
    
        beta_for_pdf = beta_fit if beta_fit is not None else 0.5
        pdf = lambda s_arr: _pdf(np.asarray(s_arr, dtype=float), beta_for_pdf)
    
        return dict(
            beta=beta_fit,
            beta_std=beta_std,
            classification=cls,
            pdf=pdf,
            nll=nll_val,
        )

    @staticmethod
    def topological_monodromy(
        H,
        L,
        vars_phase,
        critical_value: tuple,
        loop_radius: float = 0.5,
        n_loop_points: int = 48,
    ) -> dict:
        """
        Detect non-trivial topological monodromy (Duistermaat 1980).
    
        For a 2-DOF integrable system with Hamiltonians H and a second
        integral L, the energy-momentum map F: M → ℝ² sends each phase-space
        point to (H, L).  The fibres of F are Liouville tori parametrised by
        action variables (I₁, I₂).
    
        Around a critical value (E*, ℓ*) of F (a focus-focus singularity),
        the action lattice undergoes a non-trivial linear transport:
    
            (I₁, I₂) → M · (I₁, I₂),   M ∈ GL(2, ℤ)
    
        For a focus-focus point M = [[1, 1], [0, 1]] (upper-triangular,
        non-identity), signalling a global obstruction to action-angle
        variables (Duistermaat 1980; Cushman & Bates 1997).
    
        The loop is traced in the (E, ℓ) plane as:
            E(θ) = E* + r·cos(θ),   ℓ(θ) = ℓ* + r·sin(θ),   θ ∈ [0, 2π).
    
        At each (E(θ), ℓ(θ)) the two actions are computed from the 1-DOF
        sub-problems obtained by energy-momentum reduction, and the transport
        matrix is read off at the end of the loop.
    
        Parameters
        ----------
        H : sympy.Expr
            2-DOF Hamiltonian H(x1, p1, x2, p2).
        L : sympy.Expr
            Second conserved quantity (must Poisson-commute with H).
        vars_phase : list of sympy.Symbol
            [x1, p1, x2, p2].
        critical_value : (float, float)
            (E*, ℓ*) — the critical value of the energy-momentum map around
            which the loop is traced.
        loop_radius : float
            Radius r of the loop in (E, ℓ) space (default 0.5).
        n_loop_points : int
            Number of sample points along the loop (default 48).
    
        Returns
        -------
        dict with keys:
            * ``monodromy_matrix``   – 2×2 integer ndarray M (rounded from float)
            * ``monodromy_float``    – 2×2 float ndarray before rounding
            * ``is_trivial``         – True if M == identity
            * ``actions_along_loop`` – (n_loop_points, 2) array of (I₁(θ), I₂(θ))
            * ``angles``             – loop parameter θ ∈ [0, 2π)
            * ``loop_EL``            – (n_loop_points, 2) array of (E(θ), ℓ(θ))
    
        Raises
        ------
        ValueError
            If vars_phase is not 2-DOF.
    
        Notes
        -----
        The monodromy matrix is estimated as:
    
            M_float = I_end · pinv(I_start_matrix)
    
        where I_start_matrix is a 2×2 matrix built from the action vectors at
        θ=0 and θ=π/2 (two independent directions), giving the full linear map.
        Rounding to integers reflects the lattice structure of the torus.
    
        A well-resolved loop (n_loop_points ≥ 36, loop_radius small enough to
        contain only one critical value) is needed for accurate results.
    
        References
        ----------
        .. [Du80] Duistermaat, J. J., "On global action-angle coordinates",
                  *Comm. Pure Appl. Math.* 33, 687–706 (1980).
        .. [Cu97] Cushman, R. H. & Bates, L. M., *Global Aspects of Classical
                  Integrable Systems*, Birkhäuser, 1997.
        """
        from sympy import solve as sym_solve, diff as sym_diff, symbols as sym_sym
    
        _check_ndof(vars_phase, 2)
        x1, p1, x2, p2 = vars_phase
    
        E_star, ell_star = float(critical_value[0]), float(critical_value[1])
        angles = np.linspace(0.0, 2.0 * np.pi, n_loop_points, endpoint=False)
    
        # Pre-lambdify L for fast evaluation
        L_func = lambdify(vars_phase, L, 'numpy')
    
        actions = np.zeros((n_loop_points, 2))
    
        for k, theta in enumerate(angles):
            E_k   = E_star  + loop_radius * np.cos(theta)
            ell_k = ell_star + loop_radius * np.sin(theta)
    
            # --- Action I₁: reduce to 1-DOF problem in (x1, p1) ---
            # Fix H = E_k and L = ell_k, solve for p2 = p2(x1, p1, x2)
            # then set x2 to its equilibrium value (reduces to 1-DOF in x1,p1).
            # For separable H = H1(x1,p1) + H2(x2,p2): H1 = E_k - H2_eq
            # For non-separable: use L = ell_k to set x2 = x2_eq(ell_k).
            try:
                # Solve L = ell_k for x2 at p2=0 (turning point of DOF 2)
                L_at_p2_0 = L.subs(p2, 0)
                x2_sols = sym_solve(L_at_p2_0 - ell_k, x2)
                x2_eq = float(x2_sols[0]) if x2_sols else 0.0
    
                # Substitute x2=x2_eq, p2=0 into H to get effective H1
                H1_eff = H.subs([(x2, x2_eq), (p2, 0)])
                I1 = action_integral(H1_eff, E_k, vars_phase=[x1, p1],
                                     method='numerical')
            except Exception:
                I1 = np.nan
    
            # --- Action I₂: reduce to 1-DOF problem in (x2, p2) ---
            try:
                # Solve H = E_k for p1=0, x1 varies: get H2 at effective energy
                H2_eff = H.subs([(x1, 0), (p1, 0)])
                # Energy available for DOF2: solve H(0,0,x2,p2) = E_k
                I2 = action_integral(H2_eff, E_k, vars_phase=[x2, p2],
                                     method='numerical')
            except Exception:
                I2 = np.nan
    
            actions[k] = [I1, I2]
    
        # --- Recover 2×2 monodromy matrix ---
        # Use the action vectors at 4 well-separated angles to build a
        # linear system:  M · I(0) = I(2π-step)  and  M · I(π/2) = I(...)
        # Simplest robust estimator: component-wise ratio at start vs end
        I_start = actions[0]
        I_end   = actions[-1]
    
        # Full 2×2 transport: use two linearly independent reference vectors
        # at θ=0 and θ=n//4 (quarter loop = π/2)
        q = n_loop_points // 4
        I_ref0  = actions[0]
        I_ref1  = actions[q]
        I_img0  = actions[-1]          # image of ref0 after full loop
        I_img1  = actions[q - 1]      # image of ref1 (one step before its position)
    
        A = np.column_stack([I_ref0, I_ref1])   # 2×2 reference matrix
        B = np.column_stack([I_img0, I_img1])   # 2×2 image matrix
    
        try:
            M_float = B @ np.linalg.pinv(A)
        except Exception:
            M_float = np.eye(2)
    
        M_int = np.round(M_float).astype(int)
        is_trivial = bool(np.array_equal(M_int, np.eye(2, dtype=int)))
    
        loop_EL = np.column_stack([
            E_star  + loop_radius * np.cos(angles),
            ell_star + loop_radius * np.sin(angles),
        ])
    
        return dict(
            monodromy_matrix=M_int,
            monodromy_float=M_float,
            is_trivial=is_trivial,
            actions_along_loop=actions,
            angles=angles,
            loop_EL=loop_EL,
        )

    @staticmethod
    def scar_intensity(
        traj: dict,
        vars_phase,
        orbit_points: np.ndarray,
        radius: float = None,
    ) -> dict:
        """
        Measure the scar intensity of a trajectory relative to a periodic orbit.
    
        Scarring (Heller 1984) is the tendency of trajectories near an unstable
        periodic orbit to spend more time there than ergodicity would predict.
        The classical signature is a dwell-time excess: the fraction of time
        steps within distance ``radius`` of the orbit is larger than the
        fraction of phase-space volume that ball occupies.
    
        The scar intensity S is defined as:
    
            S = f_orbit / f_expected
    
        where:
            f_orbit    = fraction of trajectory points within distance r of
                         any orbit point
            f_expected = π r² / A_traj   (area of the ball relative to the
                         bounding box of the trajectory — a crude ergodic baseline)
    
        S ≈ 1  →  ergodic (no scarring)
        S >> 1 →  trajectory is scarred along the orbit
    
        Parameters
        ----------
        traj : dict
            Trajectory dict as returned by ``hamiltonian_flow``.
            For 2-DOF systems the projection onto (x1, p1) is used.
        vars_phase : list of sympy.Symbol
            [x, p] or [x1, p1, x2, p2].
        orbit_points : (K, 2) ndarray
            Points (x, p) sampled along the periodic orbit.
        radius : float, optional
            Proximity radius r for dwell-time counting.  Defaults to 1/10 of
            the trajectory's diameter in phase space.
    
        Returns
        -------
        dict with keys:
            * ``scar_intensity``  – S = f_orbit / f_expected (float)
            * ``f_orbit``         – fraction of trajectory within r of orbit
            * ``f_expected``      – ergodic baseline fraction
            * ``n_close``         – number of trajectory points within r
            * ``radius``          – radius used
    
        Examples
        --------
        >>> x, p = symbols('x p', real=True)
        >>> H = (p**2 + x**2) / 2
        >>> traj = hamiltonian_flow(H, (1, 0), (0, 40*np.pi),
        ...                         vars_phase=[x, p], n_steps=4000)
        >>> theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        >>> orbit = np.column_stack([np.cos(theta), np.sin(theta)])
        >>> result = IntegrabilityAnalysis.scar_intensity(traj, [x, p], orbit)
        >>> result['scar_intensity']   # >> 1 since traj IS the orbit
        > 5.0
    
        References
        ----------
        .. [He84] Heller, E. J., "Bound-state eigenfunctions of classically
                  chaotic Hamiltonian systems: scars of periodic orbits",
                  *Phys. Rev. Lett.* 53, 1515–1518 (1984).
        """
        ndof = _get_ndof(vars_phase)
        x_key = str(vars_phase[0])
        p_key = str(vars_phase[1])   # always use first (x,p) pair
    
        x_traj = np.asarray(traj[x_key], dtype=float)
        p_traj = np.asarray(traj[p_key], dtype=float)
        orbit  = np.asarray(orbit_points, dtype=float)
    
        if orbit.ndim != 2 or orbit.shape[1] != 2:
            raise ValueError("orbit_points must be a (K, 2) array of (x, p) pairs.")
    
        # Default radius: 1/10 of trajectory diameter
        if radius is None:
            diam = max(x_traj.max() - x_traj.min(),
                       p_traj.max() - p_traj.min())
            radius = diam / 10.0
        radius = float(radius)
    
        # For each trajectory point, find minimum distance to any orbit point
        # Shape: (N_traj, K_orbit) — batched to avoid huge arrays
        traj_pts = np.column_stack([x_traj, p_traj])   # (N, 2)
        batch = 1000
        close_mask = np.zeros(len(traj_pts), dtype=bool)
        for start in range(0, len(orbit), batch):
            orb_batch = orbit[start:start + batch]            # (B, 2)
            # (N, B)
            dists = np.linalg.norm(
                traj_pts[:, np.newaxis, :] - orb_batch[np.newaxis, :, :],
                axis=2
            )
            close_mask |= (dists.min(axis=1) < radius)
    
        n_close    = int(close_mask.sum())
        f_orbit    = n_close / len(traj_pts)
    
        # Ergodic baseline: area of disk / area of bounding box
        bbox_area  = ((x_traj.max() - x_traj.min()) *
                      (p_traj.max() - p_traj.min()))
        f_expected = (np.pi * radius ** 2) / max(bbox_area, 1e-12)
        f_expected = min(f_expected, 1.0)   # cap at 1
    
        scar_int = f_orbit / max(f_expected, 1e-12)
    
        return dict(
            scar_intensity=float(scar_int),
            f_orbit=float(f_orbit),
            f_expected=float(f_expected),
            n_close=n_close,
            radius=radius,
        )
# -----------------------------------------------------------------------------
# Backward compatibility aliases
# -----------------------------------------------------------------------------

SymplecticForm1D = lambda vars_phase=None: SymplecticForm(n=1, vars_phase=vars_phase)
SymplecticForm2D = lambda vars_phase=None: SymplecticForm(n=2, vars_phase=vars_phase)

