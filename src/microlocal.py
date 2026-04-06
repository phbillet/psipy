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
microlocal.py — Unified microlocal analysis toolkit for 1D and 2D problems
============================================================================

Overview
--------
The `microlocal` module provides a high‑level interface for studying the propagation of singularities and constructing semiclassical approximations for linear partial differential equations.  It builds upon the dedicated `wkb` (WKB approximations) and `caustics` (catastrophe classification and ray caustic detection) modules, adding dimension‑agnostic functions for the core concepts of microlocal analysis:

* **Characteristic variety** `Char(P) = {(x,ξ) : p(x,ξ)=0}` – the set of phase‑space points where the principal symbol vanishes, indicating where singularities can propagate.
* **Bicharacteristic flow** – Hamilton’s equations for the principal symbol, whose integral curves (bicharacteristics) govern the propagation of wave‑front sets.
* **Wavefront set** `WF(u)` – a refinement of the singular support that also records the directions (frequencies) in which the singularity occurs.  The module visualises how an initial wavefront set evolves under the flow.
* **WKB approximation** (re‑exported from `wkb`) – asymptotic solutions of the form `u ≈ A e^{iS/ε}`.
* **Caustics and Maslov index** – detection of caustics (where rays focus) and computation of the associated Maslov phase shifts, crucial for correct semiclassical quantisation.
* **Bohr–Sommerfeld quantisation** (1D) – semiclassical energy levels for bound states.

All functions automatically detect the spatial dimension (1 or 2) from the input data, making the module usable for both one‑dimensional and two‑dimensional problems without changing the calling syntax.

Mathematical background
-----------------------
In microlocal analysis, a linear partial differential operator `P` is studied via its **principal symbol** `p(x,ξ)`, a function on the cotangent bundle `T*ℝⁿ`.  The **characteristic variety** is the zero set of `p`.  Singularities of a distribution `u` satisfying `P u ≈ 0` are confined to the characteristic variety and propagate along **bicharacteristics** – integral curves of the Hamiltonian vector field

    X_p = ( ∂p/∂ξ , –∂p/∂x ).

The **wavefront set** `WF(u)` is a closed conic subset of `T*ℝⁿ \ {0}` that records both the location `x` and the direction `ξ` of the singularity.  If `(x₀,ξ₀) ∉ WF(u)`, then `u` is smooth in a neighbourhood of `x₀` in the direction `ξ₀`.  The fundamental theorem of microlocal analysis states that `WF(Pu) ⊆ WF(u)` and that `WF(u) \ WF(Pu)` is contained in the characteristic variety and is invariant under the bicharacteristic flow.

The **WKB method** seeks solutions of the form `u(x) = e^{iS(x)/ε} (a₀(x) + ε a₁(x) + …)`.  The phase `S` satisfies the eikonal equation `p(x,∇S)=0`, and the amplitudes `a_k` satisfy transport equations along bicharacteristics.  This construction breaks down at **caustics**, where rays focus; the Maslov index `μ` (a signed count of caustic crossings) provides a phase correction `e^{iμπ/2}` that restores uniformity.

The module integrates these concepts into a coherent toolkit, allowing the user to:

* Symbolically compute the characteristic variety.
* Numerically integrate bicharacteristics with symplectic integrators.
* Visualise the evolution of wavefront sets.
* Obtain semiclassical spectra via Bohr–Sommerfeld quantisation (1D).
* Detect caustics and compute the Maslov index (using the `caustics` module).

References
----------
.. [1] Hörmander, L.  *The Analysis of Linear Partial Differential Operators I*, Springer, 1983.  Chapter 8: Wave Front Sets.
.. [2] Duistermaat, J. J.  *Fourier Integral Operators*, Courant Institute Lecture Notes, 1996.
.. [3] Maslov, V. P. & Fedoriuk, M. V.  *Semi‑Classical Approximation in Quantum Mechanics*, Reidel, 1981.
.. [4] Zworski, M.  *Semiclassical Analysis*, American Mathematical Society, 2012.  Chapter 3: Propagation of Singularities.
.. [5] Taylor, M. E.  *Partial Differential Equations II*, Springer, 2011.  Chapter 8: Microlocal Analysis.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint, quad
from scipy.optimize import bisect
from scipy.interpolate import griddata, interp1d
from wkb import *
from caustics import * 

# ----------------------------------------------------------------------
# Dimension detection helper (used internally by microlocal functions)
# ----------------------------------------------------------------------
def _infer_dim(symbol, dim=None):
    """
    Infer the dimension (1 or 2) from the symbol or a provided dim.
    Relies on the presence of common variable names: 'x','xi' for 1D,
    and 'x','y','xi','eta' for 2D.
    """
    if dim is not None:
        if dim not in (1, 2):
            raise ValueError("dim must be 1 or 2")
        return dim
    free_vars = set(symbol.free_symbols)
    # Heuristic: if 'y' or 'eta' present, assume 2D
    if any(str(v) in ('y', 'eta') for v in free_vars):
        return 2
    return 1

# ----------------------------------------------------------------------
# Characteristic variety
# ----------------------------------------------------------------------
def characteristic_variety(symbol, dim=None, tol=1e-8):
    """
    Compute the characteristic variety of a pseudo-differential operator.
    
    Char(P) = { (x,ξ) in T*ℝ : p(x,ξ)=0 }   (1D)
    or
    Char(P) = { (x,y,ξ,η) in T*ℝ² : p(x,y,ξ,η)=0 }   (2D)
    
    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p.
    dim : int, optional
        Dimension (1 or 2). If None, inferred from symbol.
    tol : float
        Tolerance for zero detection (unused, kept for compatibility).
    
    Returns
    -------
    dict
        A dictionary containing the following keys:
        
        - 'implicit' : sympy expression
            The symbol expression.
        - 'equation' : sympy Eq
            The equation `symbol = 0`.
        - 'explicit' : list or None
            List of explicit solutions ξ(x) (1D only), or None if no explicit solution exists.
        - 'function' : callable
            A callable that evaluates the symbol.
    """
    dim = _infer_dim(symbol, dim)
    if dim == 1:
        x, xi = sp.symbols('x xi', real=True)
        char_eq = sp.Eq(symbol, 0)
        try:
            xi_solutions = sp.solve(symbol, xi)
            explicit = [sp.simplify(sol) for sol in xi_solutions]
        except:
            explicit = None
        func = sp.lambdify((x, xi), symbol, 'numpy')
        return {
            'implicit': symbol,
            'equation': char_eq,
            'explicit': explicit,
            'function': func
        }
    else:  # dim == 2
        x, y, xi, eta = sp.symbols('x y xi eta', real=True)
        char_eq = sp.Eq(symbol, 0)
        func = sp.lambdify((x, y, xi, eta), symbol, 'numpy')
        return {
            'implicit': symbol,
            'equation': char_eq,
            'explicit': None,   # no explicit form in 2D in general
            'function': func
        }


# ----------------------------------------------------------------------
# Bicharacteristic flow
# ----------------------------------------------------------------------
def bicharacteristic_flow(symbol, z0, tspan, dim=None,
                          method='symplectic', n_steps=1000):
    """
    Integrate the bicharacteristic flow on the cotangent bundle.

    Hamilton's equations:  ẋ = ∂p/∂ξ,  ξ̇ = -∂p/∂x   (1D)
    or  ẋ = ∂p/∂ξ,  ẏ = ∂p/∂η,  ξ̇ = -∂p/∂x,  η̇ = -∂p/∂y   (2D)

    Parameters
    ----------
    symbol : sympy expression
        Principal symbol.
    z0 : tuple
        Initial condition on T*M. For 1D: (x₀, ξ₀); for 2D: (x₀, y₀, ξ₀, η₀).
    tspan : tuple
        (t_start, t_end).
    dim : int, optional
        Dimension. If None, inferred from length of z0.
    method : str
        Integration method: 'rk45', 'symplectic', or 'verlet' (2D only).
    n_steps : int
        Number of time steps.

    Returns
    -------
    dict
        Trajectory data with keys 't', 'x', 'xi' (1D) and also 'y','eta' (2D),
        plus 'symbol_value'.
    """
    if dim is None:
        dim = 2 if len(z0) == 4 else 1
    if dim == 1:
        return _bichar_flow_1d(symbol, z0, tspan, method, n_steps)
    else:
        return _bichar_flow_2d(symbol, z0, tspan, method, n_steps)

def _bichar_flow_1d(symbol, z0, tspan, method, n_steps):
    """
    Integrate the 1D bicharacteristic flow and the associated stability matrix.

    For a Hamiltonian system defined by the principal symbol p(x,ξ), the
    bicharacteristics (Hamiltonian flow) satisfy

        dx/dt = ∂p/∂ξ,   dξ/dt = –∂p/∂x.

    Simultaneously, the 2×2 Jacobian matrix

        J(t) = [[ ∂x/∂x₀ , ∂x/∂ξ₀ ],
                [ ∂ξ/∂x₀ , ∂ξ/∂ξ₀ ]]

    obeys the linearised variational equation

        dJ/dt = A(t) J,   J(0) = I₂,

    where

        A(t) = [[ ∂²p/(∂ξ∂x) , ∂²p/∂ξ² ],
                [ –∂²p/∂x²  , –∂²p/(∂x∂ξ) ]]

    evaluated along the trajectory.

    Parameters
    ----------
    symbol : sympy.Expr
        Principal symbol p(x,ξ). Must depend on the real symbols `x` and `xi`.
    z0 : tuple of float
        Initial condition (x₀, ξ₀).
    tspan : tuple of float
        Integration interval (t_start, t_end).
    method : {'rk45', 'symplectic', 'hamiltonian'}
        Integration scheme:
        - 'rk45'      : adaptive Runge‑Kutta 4(5) from `scipy.integrate.solve_ivp`.
                        The state is augmented with the four entries of J.
        - 'symplectic': fixed‑step symplectic Euler (position half‑step,
                        momentum full‑step). The linearised map is derived
                        from the exact discretisation of the variational
                        equations, preserving the symplectic structure up to
                        the order of the integrator.
        - 'hamiltonian': alias for 'symplectic' (kept for backward compatibility).
    n_steps : int
        Number of output points (including the initial state). For fixed‑step
        methods this equals the number of integration steps; for 'rk45' it
        specifies the number of equally spaced time values at which the
        solution is evaluated.

    Returns
    -------
    dict
        A dictionary containing the following fields:
        - 't'            : 1D ndarray of shape (n_steps,) – time points.
        - 'x'            : 1D ndarray – position trajectory.
        - 'xi'           : 1D ndarray – momentum trajectory.
        - 'symbol_value' : 1D ndarray – p(x,ξ) evaluated along the trajectory.
        - 'J11', 'J12', 'J21', 'J22' : 1D ndarrays – components of the 2×2
          stability matrix J(t) at each time point.

    Notes
    -----
    - The function uses symbolic differentiation (`sympy.diff`) and generates
      NumPy callables via `lambdify`. The symbols must be named exactly `x` and
      `xi` (case‑sensitive).
    - For the symplectic Euler method, the linearised map M that advances the
      Jacobian from step i to i+1 is:

          M = [[ 1 + Δt·a11 + Δt²·a12·a21 ,  Δt·a12·(1 + Δt·a22) ],
               [ Δt·a21                  ,  1 + Δt·a22            ]]

      where a11, a12, a21, a22 are the entries of A(t) evaluated at the
      beginning of the step. This map is applied as J_{i+1} = M · J_i.
    - This function is intended for internal use and is not part of the public
      API of the `microlocal` module.
    """
    x, xi = sp.symbols('x xi', real=True)
    
    # First derivatives (Hamilton's equations)
    dp_dxi = sp.diff(symbol, xi)
    dp_dx  = sp.diff(symbol, x)
    f_x   = sp.lambdify((x, xi), dp_dxi, 'numpy')
    f_xi  = sp.lambdify((x, xi), -dp_dx, 'numpy')
    
    # Second derivatives for the stability matrix
    d2p_dx2   = sp.diff(symbol, x, x)
    d2p_dxi2  = sp.diff(symbol, xi, xi)
    d2p_dxdxi = sp.diff(symbol, x, xi)
    
    A11 = sp.diff(symbol, xi, x)        # ∂²p/∂ξ∂x
    A12 = d2p_dxi2                      # ∂²p/∂ξ²
    A21 = -d2p_dx2                      # -∂²p/∂x²
    A22 = -d2p_dxdxi                    # -∂²p/∂x∂ξ
    
    A_func = sp.lambdify((x, xi), (A11, A12, A21, A22), 'numpy')
    p_func = sp.lambdify((x, xi), symbol, 'numpy')
    
    if method == 'rk45':
        # Augment state: [x, xi, J11, J12, J21, J22]
        def ode(t, z):
            xv, xiv = z[0], z[1]
            J11, J12, J21, J22 = z[2], z[3], z[4], z[5]
            a11, a12, a21, a22 = A_func(xv, xiv)
            dJ11 = a11*J11 + a12*J21
            dJ12 = a11*J12 + a12*J22
            dJ21 = a21*J11 + a22*J21
            dJ22 = a21*J12 + a22*J22
            return [f_x(xv, xiv), f_xi(xv, xiv), dJ11, dJ12, dJ21, dJ22]
        
        z0_aug = [z0[0], z0[1], 1.0, 0.0, 0.0, 1.0]   # J(0)=I₂
        sol = solve_ivp(ode, tspan, z0_aug, method='RK45',
                        t_eval=np.linspace(tspan[0], tspan[1], n_steps),
                        rtol=1e-9, atol=1e-12)
        return {
            't': sol.t,
            'x': sol.y[0],
            'xi': sol.y[1],
            'symbol_value': p_func(sol.y[0], sol.y[1]),
            'J11': sol.y[2], 'J12': sol.y[3],
            'J21': sol.y[4], 'J22': sol.y[5],
        }
    
    elif method in ('symplectic', 'hamiltonian'):
        dt = (tspan[1] - tspan[0]) / n_steps
        t = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        xi_vals = np.zeros(n_steps)
        J11_vals = np.zeros(n_steps)
        J12_vals = np.zeros(n_steps)
        J21_vals = np.zeros(n_steps)
        J22_vals = np.zeros(n_steps)
    
        x_vals[0], xi_vals[0] = z0
        J11_vals[0], J12_vals[0] = 1.0, 0.0
        J21_vals[0], J22_vals[0] = 0.0, 1.0
    
        for i in range(n_steps - 1):
            xv, xiv = x_vals[i], xi_vals[i]
    
            # Second derivatives at the current point
            a11, a12, a21, a22 = A_func(xv, xiv)
    
            # Symplectic Euler update for position and momentum
            xi_new = xiv + dt * f_xi(xv, xiv)
            x_new = xv + dt * f_x(xv, xi_new)
    
            # Linearised symplectic Euler map M:
            #   M = [[1 + dt*a11 + dt²*a12*a21,  dt*a12*(1 + dt*a22)],
            #        [dt*a21,                   1 + dt*a22]]
            M11 = 1.0 + dt * a11 + dt * dt * a12 * a21
            M12 = dt * a12 * (1.0 + dt * a22)
            M21 = dt * a21
            M22 = 1.0 + dt * a22
    
            # Apply M to the current Jacobian
            J11_new = M11 * J11_vals[i] + M12 * J21_vals[i]
            J12_new = M11 * J12_vals[i] + M12 * J22_vals[i]
            J21_new = M21 * J11_vals[i] + M22 * J21_vals[i]
            J22_new = M21 * J12_vals[i] + M22 * J22_vals[i]
    
            x_vals[i+1] = x_new
            xi_vals[i+1] = xi_new
            J11_vals[i+1] = J11_new
            J12_vals[i+1] = J12_new
            J21_vals[i+1] = J21_new
            J22_vals[i+1] = J22_new
    
        return {
            't': t,
            'x': x_vals,
            'xi': xi_vals,
            'symbol_value': p_func(x_vals, xi_vals),
            'J11': J11_vals, 'J12': J12_vals,
            'J21': J21_vals, 'J22': J22_vals,
        }
    else:
        raise ValueError("Invalid method for 1D flow")


def _bichar_flow_2d(symbol, z0, tspan, method, n_steps):
    """
    Compute the bicharacteristic (Hamiltonian) flow for a 2D symbol
    and the associated stability matrix.

    The Hamiltonian system is defined by the symbol p(x, y, ξ, η):

        dx/dt =  ∂p/∂ξ,    dξ/dt = -∂p/∂x,
        dy/dt =  ∂p/∂η,    dη/dt = -∂p/∂y.

    Simultaneously, the 2×2 stability matrix J(t) = ∂(x,y)/∂(x₀,y₀)
    (the Jacobian of the spatial part of the flow with respect to initial
    positions) is propagated according to

        dJ/dt = H_px · J,   J(0) = I₂,

    where H_px is the 2×2 matrix of mixed derivatives:

        H_px = [[ ∂²p/(∂ξ∂x), ∂²p/(∂ξ∂y) ],
                [ ∂²p/(∂η∂x), ∂²p/(∂η∂y) ]].

    Parameters
    ----------
    symbol : sympy.Expr
        Symbolic expression for the Hamiltonian p(x, y, ξ, η). Must depend
        on the real variables x, y, xi, eta (the names are fixed in the
        function body).
    z0 : tuple of float
        Initial condition (x0, y0, xi0, eta0).
    tspan : tuple of float
        Time interval (t0, t1).
    method : {'rk45', 'symplectic', 'verlet'}
        Integration method:
        - 'rk45'      : adaptive Runge‑Kutta 4(5) from `scipy.integrate.solve_ivp`.
                        The state is augmented with the four entries of J.
        - 'symplectic' : fixed‑step symplectic Euler (position half‑step,
                        momentum full‑step). Stability matrix uses forward Euler.
        - 'verlet'     : fixed‑step Störmer‑Verlet (leapfrog). Stability matrix
                        uses the midpoint value of H_px for the update.
    n_steps : int
        Number of output points (including the initial state). For fixed‑step
        methods this equals the number of integration steps; for 'rk45' it is
        the number of equally spaced time values at which the solution is
        evaluated.

    Returns
    -------
    dict
        A dictionary containing:
        - 't'       : 1D ndarray of shape (n_steps,) – time points.
        - 'x','y'   : 1D ndarrays – trajectory in position.
        - 'xi','eta': 1D ndarrays – trajectory in momentum.
        - 'symbol_value' : 1D ndarray – p(x,y,ξ,η) evaluated along the trajectory.
        - 'J11','J12','J21','J22' : 1D ndarrays – components of the 2×2
          stability matrix J(t) at each time point.

    Notes
    -----
    - The function uses symbolic differentiation (`sympy.diff`) and generates
      NumPy callables via `lambdify`. The symbols *must* be named exactly
      `x`, `y`, `xi`, `eta` (case‑sensitive).
    - For 'rk45', high relative/absolute tolerances (1e‑9, 1e‑12) are used to
      ensure accurate stability propagation.
    - For the symplectic and Verlet schemes, the stability update is a simple
      first‑order method (forward Euler or midpoint) that is consistent with
      the overall integrator order but may not preserve symplecticity of the
      linearised flow.
    - This function is intended for internal use in pseudo‑differential
      operator construction and is not part of the public API.
    """
    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    dp_dxi  = sp.diff(symbol, xi)
    dp_deta = sp.diff(symbol, eta)
    dp_dx   = sp.diff(symbol, x)
    dp_dy   = sp.diff(symbol, y)
    f_x   = sp.lambdify((x,y,xi,eta), dp_dxi,  'numpy')
    f_y   = sp.lambdify((x,y,xi,eta), dp_deta, 'numpy')
    f_xi  = sp.lambdify((x,y,xi,eta), -dp_dx,  'numpy')
    f_eta = sp.lambdify((x,y,xi,eta), -dp_dy,  'numpy')
    p_func = sp.lambdify((x,y,xi,eta), symbol, 'numpy')

    # Build H_px = [[∂²H/∂ξ∂x, ∂²H/∂ξ∂y],
    #               [∂²H/∂η∂x, ∂²H/∂η∂y]]
    # used by the stability ODE  dJ/dt = H_px · J,  J(0) = I₂
    H_px_sym = sp.Matrix([
        [sp.diff(symbol, xi,  x), sp.diff(symbol, xi,  y)],
        [sp.diff(symbol, eta, x), sp.diff(symbol, eta, y)],
    ])
    H_px_func = sp.lambdify((x, y, xi, eta), H_px_sym, 'numpy')

    def _H_px(xv, yv, xiv, etav):
        """Return H_px as a (2,2) float array."""
        return np.asarray(H_px_func(xv, yv, xiv, etav), dtype=float).reshape(2, 2)

    if method == 'rk45':
        # Augment the state with the 4 entries of J (row-major: J11,J12,J21,J22).
        # Full state: [x, y, xi, eta, J11, J12, J21, J22]
        z0_aug = list(z0) + [1.0, 0.0, 0.0, 1.0]   # J(0) = I₂

        def ode(t, z):
            xv, yv, xiv, etav = z[0], z[1], z[2], z[3]
            J = z[4:].reshape(2, 2)
            dJ = _H_px(xv, yv, xiv, etav) @ J
            return [
                f_x  (xv, yv, xiv, etav),
                f_y  (xv, yv, xiv, etav),
                f_xi (xv, yv, xiv, etav),
                f_eta(xv, yv, xiv, etav),
                *dJ.ravel(),
            ]

        sol = solve_ivp(ode, tspan, z0_aug, method='RK45',
                        t_eval=np.linspace(tspan[0], tspan[1], n_steps),
                        rtol=1e-9, atol=1e-12)
        return {
            't':            sol.t,
            'x':            sol.y[0],
            'y':            sol.y[1],
            'xi':           sol.y[2],
            'eta':          sol.y[3],
            'symbol_value': p_func(sol.y[0], sol.y[1], sol.y[2], sol.y[3]),
            'J11':          sol.y[4],
            'J12':          sol.y[5],
            'J21':          sol.y[6],
            'J22':          sol.y[7],
        }

    elif method in ('symplectic', 'verlet'):
        dt = (tspan[1] - tspan[0]) / n_steps
        t = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals   = np.zeros(n_steps)
        y_vals   = np.zeros(n_steps)
        xi_vals  = np.zeros(n_steps)
        eta_vals = np.zeros(n_steps)
        J_vals   = np.zeros((n_steps, 2, 2))   # stability matrix at each step

        x_vals[0], y_vals[0], xi_vals[0], eta_vals[0] = z0
        J_vals[0] = np.eye(2)                  # J(0) = I₂

        if method == 'symplectic':
            for i in range(n_steps - 1):
                xv, yv   = x_vals[i],  y_vals[i]
                xiv, etav = xi_vals[i], eta_vals[i]

                # --- momentum half-step (symplectic Euler) ---
                xi_new  = xiv  + dt * f_xi (xv, yv, xiv, etav)
                eta_new = etav + dt * f_eta(xv, yv, xiv, etav)

                # --- position full-step ---
                x_new = xv + dt * f_x(xv, yv, xi_new, eta_new)
                y_new = yv + dt * f_y(xv, yv, xi_new, eta_new)

                x_vals[i+1]   = x_new
                y_vals[i+1]   = y_new
                xi_vals[i+1]  = xi_new
                eta_vals[i+1] = eta_new

                # --- stability matrix: forward Euler on dJ/dt = H_px · J ---
                J_vals[i+1] = J_vals[i] + dt * (_H_px(xv, yv, xiv, etav) @ J_vals[i])

        else:  # verlet
            for i in range(n_steps - 1):
                xv, yv    = x_vals[i],  y_vals[i]
                xiv, etav = xi_vals[i], eta_vals[i]

                # --- Störmer-Verlet / leapfrog ---
                xi_half  = xiv  + 0.5*dt * f_xi (xv, yv, xiv, etav)
                eta_half = etav + 0.5*dt * f_eta(xv, yv, xiv, etav)
                x_new  = xv + dt * f_x(xv, yv, xi_half, eta_half)
                y_new  = yv + dt * f_y(xv, yv, xi_half, eta_half)
                xi_new  = xi_half  + 0.5*dt * f_xi (x_new, y_new, xi_half, eta_half)
                eta_new = eta_half + 0.5*dt * f_eta(x_new, y_new, xi_half, eta_half)

                x_vals[i+1]   = x_new
                y_vals[i+1]   = y_new
                xi_vals[i+1]  = xi_new
                eta_vals[i+1] = eta_new

                # --- stability matrix: midpoint H_px for consistency with Verlet ---
                Hpx_mid = 0.5 * (_H_px(xv,    yv,    xiv,    etav) +
                                  _H_px(x_new, y_new, xi_new, eta_new))
                J_vals[i+1] = J_vals[i] + dt * (Hpx_mid @ J_vals[i])

        return {
            't':            t,
            'x':            x_vals,
            'y':            y_vals,
            'xi':           xi_vals,
            'eta':          eta_vals,
            'symbol_value': p_func(x_vals, y_vals, xi_vals, eta_vals),
            'J11':          J_vals[:, 0, 0],
            'J12':          J_vals[:, 0, 1],
            'J21':          J_vals[:, 1, 0],
            'J22':          J_vals[:, 1, 1],
        }

    else:
        raise ValueError("Invalid method for 2D flow")

# ----------------------------------------------------------------------
# 1D‑specific functions (Bohr–Sommerfeld, caustic detection)
# ----------------------------------------------------------------------
def bohr_sommerfeld_quantization(H, n_max=10, x_range=(-10,10),
                                 hbar=1.0, E_range=(1e-6, 50.0)):
    """
    Bohr–Sommerfeld quantisation for 1D bound states.

    (1/(2π)) ∮ p dx = ℏ(n + α)   with α = 1/2 (Maslov index).

    Parameters
    ----------
    H : sympy expression
        Hamiltonian H(x,p).
    n_max : int
        Maximum quantum number.
    x_range : tuple
        Spatial range for turning points.
    hbar : float
        Planck constant.
    method : str
        Ignored, kept for compatibility.

    Returns
    -------
    dict
        Quantised energies and actions.
    """
    x, p = sp.symbols('x p', real=True)
    E_sym = sp.symbols('E', real=True, positive=True)
    solutions = sp.solve(H - E_sym, p)
    if not solutions:
        raise ValueError("Cannot solve H=E for p(x,E)")
    p_expr = solutions[-1]   # positive branch
    p_func = sp.lambdify((x, E_sym), p_expr, 'numpy')

    alpha = 0.5

    def action(E):
        X = np.linspace(x_range[0], x_range[1], 2000)
        p_vals = p_func(X, E)
        p_vals = np.real_if_close(p_vals)
        p_vals = np.real(p_vals)
        # --- NEW GUARD ---
        if np.ndim(p_vals) == 0:
            return 0.0
        # -----------------
        mask = p_vals >= 0
        if not np.any(mask):
            return 0.0
        idx = np.where(mask)[0]
        a, b = X[idx[0]], X[idx[-1]]
        def integrand(xv):
            return p_func(xv, E)
        I, _ = quad(integrand, a, b, epsabs=1e-10, epsrel=1e-10)
        return I / np.pi

    targets = [hbar*(n + alpha) for n in range(n_max)]
    E_scan = np.linspace(E_range[0], E_range[1], 200) 
    I_scan = [action(E) for E in E_scan]

    energies = []
    actions = []
    quantum_numbers = []
    for n, Itarget in zip(range(n_max), targets):
        found = False
        for k in range(len(E_scan)-1):
            if (I_scan[k]-Itarget)*(I_scan[k+1]-Itarget) < 0:
                E_left, E_right = E_scan[k], E_scan[k+1]
                found = True
                break
        if not found:
            continue
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

def find_caustics_1d(symbol, x_range, xi_range, resolution=100):
    """
    Find caustics (envelope of bicharacteristics) in 1D.

    Simplified condition: where d²p/dξ² ≈ 0 (turning points in frequency).
    """
    x, xi = sp.symbols('x xi', real=True)
    d2p = sp.diff(symbol, xi, 2)
    func = sp.lambdify((x, xi), d2p, 'numpy')
    xv = np.linspace(x_range[0], x_range[1], resolution)
    xiv = np.linspace(xi_range[0], xi_range[1], resolution)
    X, XI = np.meshgrid(xv, xiv, indexing='ij')
    Z = np.abs(func(X, XI))
    return {
        'x_grid': X,
        'xi_grid': XI,
        'caustic_indicator': Z,
        'threshold': np.percentile(Z, 10)
    }

def propagate_singularity(symbol, initial_sing_support, tspan,
                          dim=None, n_samples=None):
    """
    Propagate singular support along bicharacteristics.
    """
    dim = _infer_dim(symbol, dim)
    trajectories = []
    for z0 in initial_sing_support:
        traj = bicharacteristic_flow(symbol, z0, tspan, dim=dim,
                                     method='symplectic')
        trajectories.append(traj)
    endpoints = [ (traj['x'][-1], traj.get('y',[None])[-1],
                   traj['xi'][-1], traj.get('eta',[None])[-1])
                  for traj in trajectories ]
    return {'trajectories': trajectories, 'endpoints': endpoints,
            'initial': initial_sing_support}

# ----------------------------------------------------------------------
# Visualisation functions (non‑WKB)
# ----------------------------------------------------------------------
def plot_characteristic_set(symbol, x_range, xi_range, dim=None,
                            resolution=200, **kwargs):
    """
    Plot the characteristic variety.

    For 1D: contour p(x,ξ)=0 in the (x,ξ) plane.
    For 2D: a slice with fixed (ξ,η); you must provide xi0, eta0 as kwargs.
    """
    dim = _infer_dim(symbol, dim)
    if dim == 1:
        x, xi = sp.symbols('x xi', real=True)
        p_func = sp.lambdify((x, xi), symbol, 'numpy')
        xv = np.linspace(x_range[0], x_range[1], resolution)
        xiv = np.linspace(xi_range[0], xi_range[1], resolution)
        X, XI = np.meshgrid(xv, xiv, indexing='ij')
        Z = p_func(X, XI)
        if np.isscalar(Z):
            Z = np.full_like(X, Z)
        plt.figure(figsize=(8,6))
        plt.contour(X, XI, Z, levels=[0], colors='red', linewidths=3)
        plt.pcolormesh(X, XI, np.log10(np.abs(Z)+1e-10),
                       shading='auto', cmap='viridis', alpha=0.5)
        plt.colorbar(label='log₁₀|p|')
        plt.xlabel('x'); plt.ylabel('ξ')
        plt.title('Characteristic variety (1D)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        # 2D slice
        xi0 = kwargs.get('xi0', 1.0)
        eta0 = kwargs.get('eta0', 0.0)
        x, y, xi, eta = sp.symbols('x y xi eta', real=True)
        p_fixed = symbol.subs({xi: xi0, eta: eta0})
        p_func = sp.lambdify((x, y), p_fixed, 'numpy')
        xv = np.linspace(x_range[0], x_range[1], resolution)
        yv = np.linspace(xi_range[0], xi_range[1], resolution)
        X, Y = np.meshgrid(xv, yv, indexing='ij')
        Z = p_func(X, Y)
        if np.isscalar(Z):
            Z = np.full_like(X, Z)
        plt.figure(figsize=(8,6))
        plt.contour(X, Y, Z, levels=[0], colors='red', linewidths=3)
        plt.pcolormesh(X, Y, np.log10(np.abs(Z)+1e-10),
                       shading='auto', cmap='viridis', alpha=0.5)
        plt.colorbar(label='log₁₀|p|')
        plt.xlabel('x'); plt.ylabel('y')
        plt.title(f'Characteristic set (ξ={xi0}, η={eta0})')
        plt.axis('equal')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_bicharacteristics(symbol, initial_points, tspan, dim=None,
                           projection='position', **kwargs):
    """
    Plot bicharacteristic curves.

    For 1D: plots in (x,ξ) plane.
    For 2D: projection can be 'position' (x-y), 'frequency' (ξ-η) or 'mixed' (x-ξ).
    """
    dim = _infer_dim(symbol, dim)
    fig, ax = plt.subplots(figsize=(10,8))
    colors = plt.cm.viridis(np.linspace(0,1,len(initial_points)))

    for idx, z0 in enumerate(initial_points):
        traj = bicharacteristic_flow(symbol, z0, tspan, dim=dim,
                                     method='symplectic', n_steps=500)
        if dim == 1:
            ax.plot(traj['x'], traj['xi'], color=colors[idx], alpha=0.7, lw=2)
            ax.plot(traj['x'][0], traj['xi'][0], 'go', ms=8)
            ax.plot(traj['x'][-1], traj['xi'][-1], 'ro', ms=6)
        else:
            if projection == 'position':
                ax.plot(traj['x'], traj['y'], color=colors[idx], alpha=0.7, lw=2)
                ax.plot(traj['x'][0], traj['y'][0], 'go', ms=8)
                ax.plot(traj['x'][-1], traj['y'][-1], 'ro', ms=6)
            elif projection == 'frequency':
                ax.plot(traj['xi'], traj['eta'], color=colors[idx], alpha=0.7, lw=2)
                ax.plot(traj['xi'][0], traj['eta'][0], 'go', ms=8)
                ax.plot(traj['xi'][-1], traj['eta'][-1], 'ro', ms=6)
            elif projection == 'mixed':
                ax.plot(traj['x'], traj['xi'], color=colors[idx], alpha=0.7, lw=2)
                ax.plot(traj['x'][0], traj['xi'][0], 'go', ms=8)
                ax.plot(traj['x'][-1], traj['xi'][-1], 'ro', ms=6)
            else:
                raise ValueError("projection must be 'position', 'frequency', or 'mixed'")

    if dim == 1:
        ax.set_xlabel('x'); ax.set_ylabel('ξ')
    else:
        if projection == 'position':
            ax.set_xlabel('x'); ax.set_ylabel('y')
        elif projection == 'frequency':
            ax.set_xlabel('ξ'); ax.set_ylabel('η')
        else:
            ax.set_xlabel('x'); ax.set_ylabel('ξ')
        ax.axis('equal')
    ax.set_title('Bicharacteristics')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_wavefront_set(symbol, initial_sing_support, tspan,
                      dim=None, projection='cotangent',
                      n_steps=500, cmap='plasma',
                      show_flow=True, show_endpoints=True,
                      title=None, ax=None):
    """
    Plot the wavefront set WF(u) of a distribution u whose singularities
    propagate along bicharacteristics of the operator with symbol p.

    The wavefront set is represented as a subset of the cotangent bundle T*ℝⁿ.
    Each initial point (x₀, ξ₀) seeds a bicharacteristic strip; the union of
    these strips in phase space approximates WF(u) at time tspan[1].

    Parameters
    ----------
    symbol : sympy expression
        Principal symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
    initial_sing_support : list of tuples
        Seed points on the wavefront set at t=0.
        1D: list of (x₀, ξ₀) pairs.
        2D: list of (x₀, y₀, ξ₀, η₀) quadruples.
    tspan : tuple
        (t_start, t_end) for bicharacteristic integration.
    dim : int, optional
        Dimension (1 or 2). Inferred from seed points if None.
    projection : str
        Which subspace to visualise (dimension-dependent):
        - 1D:
            'cotangent'  – (x, ξ) phase-space portrait  [default]
            'position'   – x(t) projected onto ℝ (singular support)
        - 2D:
            'cotangent'  – (x, ξ) slice (ignoring y, η)
            'position'   – (x, y) projection (singular support in ℝ²)
            'frequency'  – (ξ, η) projection (directions of non-smoothness)
            'full'       – 2×2 grid: position, frequency, (x,ξ), (y,η)
    n_steps : int
        Number of integration steps per bicharacteristic.
    cmap : str
        Matplotlib colormap used to colour individual bicharacteristics
        (colour encodes the index of the seed point, i.e. which part of the
        initial singular support it originated from).
    show_flow : bool
        If True, draw the full bicharacteristic strip (trajectory in phase
        space). If False, only show the endpoint scatter.
    show_endpoints : bool
        If True, mark the *initial* point (green •) and *final* point (red •)
        of each bicharacteristic.
    title : str, optional
        Figure title. A sensible default is generated if None.
    ax : matplotlib Axes or array of Axes, optional
        Axes to draw into. If None a new figure is created. For
        projection='full' (2D) pass an array of 4 Axes or leave None.

    Returns
    -------
    fig : matplotlib Figure
    axes : Axes or array of Axes

    Examples
    --------
    1D example – Schrödinger-type operator, horizontal line singularity::

        x, xi = sp.symbols('x xi', real=True)
        p = xi**2 - (1 - x**2)           # simple potential well symbol
        seeds = [(xi_val, float(xi_val)) for xi_val in np.linspace(-1, 1, 12)]
        fig, ax = plot_wavefront_set(p, seeds, tspan=(0, 4), dim=1)
        plt.show()

    2D example – wave operator, outward circular wavefront::

        x, y, xi, eta = sp.symbols('x y xi eta', real=True)
        p = xi**2 + eta**2 - 1
        seeds = [(np.cos(t), np.sin(t), np.cos(t), np.sin(t))
                 for t in np.linspace(0, 2*np.pi, 24, endpoint=False)]
        fig, axes = plot_wavefront_set(p, seeds, tspan=(0, 2), dim=2,
                                       projection='full')
        plt.show()
    """
    # ------------------------------------------------------------------ setup
    if dim is None:
        dim = 2 if len(initial_sing_support[0]) == 4 else 1
    n_seeds = len(initial_sing_support)
    colors = plt.get_cmap(cmap)(np.linspace(0.05, 0.95, n_seeds))

    # ------------------------------------------------- integrate all trajectories
    trajs = []
    for z0 in initial_sing_support:
        traj = bicharacteristic_flow(symbol, z0, tspan, dim=dim,
                                     method='symplectic', n_steps=n_steps)
        trajs.append(traj)

    # ------------------------------------------------- build figure / axes
    if projection == 'full' and dim == 2:
        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
        else:
            axes = np.asarray(ax).flatten()
            fig = axes[0].get_figure()
        ax_list = [
            (axes[0], 'x', 'y',   'position',  'Position space  (x, y)'),
            (axes[1], 'xi', 'eta', 'frequency', 'Frequency space  (ξ, η)'),
            (axes[2], 'x', 'xi',  'mixed_x',   'Cotangent slice  (x, ξ)'),
            (axes[3], 'y', 'eta', 'mixed_y',   'Cotangent slice  (y, η)'),
        ]
        for idx, traj in enumerate(trajs):
            c = colors[idx]
            x0, y0   = traj['x'][0],  traj['y'][0]
            xi0, eta0 = traj['xi'][0], traj['eta'][0]
            xf, yf   = traj['x'][-1],  traj['y'][-1]
            xif, etaf = traj['xi'][-1], traj['eta'][-1]
            pairs = [
                (traj['x'],  traj['y'],   x0, y0,   xf, yf),
                (traj['xi'], traj['eta'], xi0, eta0, xif, etaf),
                (traj['x'],  traj['xi'],  x0, xi0,  xf, xif),
                (traj['y'],  traj['eta'], y0, eta0,  yf, etaf),
            ]
            for (ax_obj, kx, ky, proj_label, panel_title), (u, v, u0, v0, uf, vf) \
                    in zip(ax_list, pairs):
                if show_flow:
                    ax_obj.plot(u, v, color=c, alpha=0.6, lw=1.2)
                if show_endpoints:
                    ax_obj.plot(u0, v0, 'o', color='limegreen', ms=5,
                                zorder=5, markeredgewidth=0)
                    ax_obj.plot(uf, vf, 's', color='crimson', ms=4,
                                zorder=5, markeredgewidth=0)
        for (ax_obj, kx, ky, proj_label, panel_title) in ax_list:
            ax_obj.set_xlabel({'x': 'x', 'xi': 'ξ', 'y': 'y', 'eta': 'η'}[kx])
            ax_obj.set_ylabel({'x': 'x', 'xi': 'ξ', 'y': 'y', 'eta': 'η'}[ky])
            ax_obj.set_title(panel_title, fontsize=10)
            ax_obj.grid(alpha=0.25)
            ax_obj.set_aspect('equal', adjustable='datalim')
        fig.suptitle(
            title or 'Wavefront set  WF(u)  [2D, full cotangent bundle]',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig, axes

    # --------------------------------- single-panel projections (1D or 2D)
    if ax is None:
        fig, axes = plt.subplots(figsize=(8, 6))
    else:
        axes = ax
        fig = axes.get_figure()

    # Determine what to plot on each axis
    if dim == 1:
        if projection in ('cotangent', 'phase'):
            get_u = lambda t: t['x']
            get_v = lambda t: t['xi']
            xlabel, ylabel = 'x', 'ξ'
            equal_aspect = False
        elif projection == 'position':
            get_u = lambda t: t['t']
            get_v = lambda t: t['x']
            xlabel, ylabel = 't', 'x  (singular support)'
            equal_aspect = False
        else:
            raise ValueError(
                f"Unknown projection '{projection}' for dim=1. "
                "Choose 'cotangent' or 'position'.")
    else:  # dim == 2
        if projection in ('cotangent', 'mixed_x'):
            get_u = lambda t: t['x']
            get_v = lambda t: t['xi']
            xlabel, ylabel = 'x', 'ξ'
            equal_aspect = True
        elif projection == 'position':
            get_u = lambda t: t['x']
            get_v = lambda t: t['y']
            xlabel, ylabel = 'x', 'y'
            equal_aspect = True
        elif projection == 'frequency':
            get_u = lambda t: t['xi']
            get_v = lambda t: t['eta']
            xlabel, ylabel = 'ξ', 'η'
            equal_aspect = True
        elif projection == 'mixed_y':
            get_u = lambda t: t['y']
            get_v = lambda t: t['eta']
            xlabel, ylabel = 'y', 'η'
            equal_aspect = True
        else:
            raise ValueError(
                f"Unknown projection '{projection}' for dim=2. "
                "Choose 'cotangent', 'position', 'frequency', 'mixed_x', "
                "'mixed_y', or 'full'.")

    for idx, traj in enumerate(trajs):
        c = colors[idx]
        u = get_u(traj)
        v = get_v(traj)
        if show_flow:
            axes.plot(u, v, color=c, alpha=0.65, lw=1.5)
        if show_endpoints:
            axes.plot(u[0],  v[0],  'o', color='limegreen', ms=7,
                      zorder=5, label='initial' if idx == 0 else '')
            axes.plot(u[-1], v[-1], 's', color='crimson',   ms=5,
                      zorder=5, label='final'   if idx == 0 else '')

    axes.set_xlabel(xlabel, fontsize=12)
    axes.set_ylabel(ylabel, fontsize=12)
    if equal_aspect:
        axes.set_aspect('equal', adjustable='datalim')

    if title is None:
        dim_str = f'{dim}D'
        title = f'Wavefront set  WF(u)  [{dim_str}, projection: {projection}]'
    axes.set_title(title, fontsize=12)
    axes.grid(alpha=0.3)
    if show_endpoints:
        handles, labels = axes.get_legend_handles_labels()
        if labels:
            axes.legend(fontsize=9)
    plt.tight_layout()
    return fig, axes


def compute_maslov_index(traj):
    """
    Compute the Maslov index for a single trajectory.
    
    Parameters
    ----------
    traj : dict
        Returned by bicharacteristic_flow. Must contain J11..J22 components.
    
    Returns
    -------
    int
        Maslov index (number of sign changes of det(J) or caustic crossings).
    """
    # Detect dimension: 2D if 'y' is present, otherwise 1D
    dim = 2 if 'y' in traj else 1
    detector = RayCausticDetector([traj], dimension=dim, det_threshold=0.05)
    detector.detect()
    return detector.maslov_index(0)

    
def compute_caustics_2d(p, initial_curve, tmax, n_rays=None, **kwargs):
    """
    Compute caustics for a 2D Hamiltonian given an initial curve.

    Parameters
    ----------
    p : sympy expression
        Hamiltonian symbol p(x, y, xi, eta).
    initial_curve : dict
        Must contain keys 'x', 'y', 'xi', 'eta' with array values of equal length.
    tmax : float
        Maximum integration time.
    n_rays : int, optional
        Number of rays to use. If None, use all points in the initial curve.
    **kwargs : additional arguments passed to bicharacteristic_flow
               (e.g., method='symplectic', n_steps=200).

    Returns
    -------
    list of CausticEvent
        Each event contains position, time, momentum, Arnold type, etc.
    """
    # Extract initial data
    x_init = np.asarray(initial_curve['x'])
    y_init = np.asarray(initial_curve['y'])
    xi_init = np.asarray(initial_curve['xi'])
    eta_init = np.asarray(initial_curve['eta'])

    n_pts = len(x_init)
    if n_rays is None:
        n_rays = n_pts
    # If fewer rays requested, subsample evenly
    if n_rays < n_pts:
        indices = np.linspace(0, n_pts-1, n_rays, dtype=int)
        x_init = x_init[indices]
        y_init = y_init[indices]
        xi_init = xi_init[indices]
        eta_init = eta_init[indices]
    elif n_rays > n_pts:
        # Not enough points; just use all
        pass

    rays = []
    # Default integration parameters
    default_kwargs = {'method': 'symplectic', 'n_steps': 200}
    flow_kwargs = {**default_kwargs, **kwargs}

    for i in range(len(x_init)):
        z0 = (x_init[i], y_init[i], xi_init[i], eta_init[i])
        traj = bicharacteristic_flow(p, z0, (0, tmax), dim=2, **flow_kwargs)
        # bicharacteristic_flow must now return J11, J12, J21, J22 (stability matrix)
        rays.append(traj)

    from caustics import RayCausticDetector
    detector = RayCausticDetector(rays, dimension=2, det_threshold=0.05)
    events = detector.detect()
    return events
