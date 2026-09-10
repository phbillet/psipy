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
* **Operator visualisation and interactive analysis** – a comprehensive suite for plotting symbol amplitude, phase, micro‑support, characteristic gradients, and launching interactive `ipywidgets` dashboards.
* **Matrix-valued symbols and pseudospectra** – extraction of characteristic branches via eigenvalue computation for matrix operators, and visualisation of ε‑pseudospectra.
* **PDE solution rendering** – rendering and animation tools for scalar, matrix, and wave equation fields in 1D and 2D space-time.

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
* Visualise the evolution of wavefront sets, including 2D and 3D singularity animations.
* Obtain semiclassical spectra via Bohr–Sommerfeld quantisation (1D).
* Detect caustics and compute the Maslov index (using the `caustics` module).
* Interactively explore symbol properties such as amplitude, phase, and micro-support.
* Analyse matrix-valued symbols by computing their characteristic branches.
* Plot ε-pseudospectra and overlay complex eigenvalues.
* Render scalar, matrix, and wave PDE solutions in space-time.

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
from sympy import symbols, lambdify, simplify, diff
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp, odeint, quad
from scipy.optimize import bisect
from scipy.interpolate import griddata, interp1d
from IPython.display import display
from ipywidgets import Dropdown, FloatSlider, VBox, interactive_output
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
            raise ValueError('dim must be 1 or 2')
        return dim
    free_vars = set(symbol.free_symbols)
    if any((str(v) in ('y', 'eta') for v in free_vars)):
        return 2
    return 1
    
# ----------------------------------------------------------------------
# Characteristic variety
# ----------------------------------------------------------------------
def characteristic_variety(symbol, dim=None, tol=1e-08):
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
        return {'implicit': symbol, 'equation': char_eq, 'explicit': explicit, 'function': func}
    else:
        x, y, xi, eta = sp.symbols('x y xi eta', real=True)
        char_eq = sp.Eq(symbol, 0)
        func = sp.lambdify((x, y, xi, eta), symbol, 'numpy')
        return {'implicit': symbol, 'equation': char_eq, 'explicit': None, 'function': func}

# ----------------------------------------------------------------------
# Bicharacteristic flow
# ----------------------------------------------------------------------
def bicharacteristic_flow(symbol, z0, tspan, dim=None, method='symplectic', n_steps=1000):
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
    dp_dxi = sp.diff(symbol, xi)
    dp_dx = sp.diff(symbol, x)
    f_x = sp.lambdify((x, xi), dp_dxi, 'numpy')
    f_xi = sp.lambdify((x, xi), -dp_dx, 'numpy')
    d2p_dx2 = sp.diff(symbol, x, x)
    d2p_dxi2 = sp.diff(symbol, xi, xi)
    d2p_dxdxi = sp.diff(symbol, x, xi)
    A11 = sp.diff(symbol, xi, x)
    A12 = d2p_dxi2
    A21 = -d2p_dx2
    A22 = -d2p_dxdxi
    A_func = sp.lambdify((x, xi), (A11, A12, A21, A22), 'numpy')
    p_func = sp.lambdify((x, xi), symbol, 'numpy')
    if method == 'rk45':

        def ode(t, z):
            xv, xiv = (z[0], z[1])
            J11, J12, J21, J22 = (z[2], z[3], z[4], z[5])
            a11, a12, a21, a22 = A_func(xv, xiv)
            dJ11 = a11 * J11 + a12 * J21
            dJ12 = a11 * J12 + a12 * J22
            dJ21 = a21 * J11 + a22 * J21
            dJ22 = a21 * J12 + a22 * J22
            return [f_x(xv, xiv), f_xi(xv, xiv), dJ11, dJ12, dJ21, dJ22]
        z0_aug = [z0[0], z0[1], 1.0, 0.0, 0.0, 1.0]
        sol = solve_ivp(ode, tspan, z0_aug, method='RK45', t_eval=np.linspace(tspan[0], tspan[1], n_steps), rtol=1e-09, atol=1e-12)
        return {'t': sol.t, 'x': sol.y[0], 'xi': sol.y[1], 'symbol_value': p_func(sol.y[0], sol.y[1]), 'J11': sol.y[2], 'J12': sol.y[3], 'J21': sol.y[4], 'J22': sol.y[5]}
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
        J11_vals[0], J12_vals[0] = (1.0, 0.0)
        J21_vals[0], J22_vals[0] = (0.0, 1.0)
        for i in range(n_steps - 1):
            xv, xiv = (x_vals[i], xi_vals[i])
            a11, a12, a21, a22 = A_func(xv, xiv)
            xi_new = xiv + dt * f_xi(xv, xiv)
            x_new = xv + dt * f_x(xv, xi_new)
            M11 = 1.0 + dt * a11 + dt * dt * a12 * a21
            M12 = dt * a12 * (1.0 + dt * a22)
            M21 = dt * a21
            M22 = 1.0 + dt * a22
            J11_new = M11 * J11_vals[i] + M12 * J21_vals[i]
            J12_new = M11 * J12_vals[i] + M12 * J22_vals[i]
            J21_new = M21 * J11_vals[i] + M22 * J21_vals[i]
            J22_new = M21 * J12_vals[i] + M22 * J22_vals[i]
            x_vals[i + 1] = x_new
            xi_vals[i + 1] = xi_new
            J11_vals[i + 1] = J11_new
            J12_vals[i + 1] = J12_new
            J21_vals[i + 1] = J21_new
            J22_vals[i + 1] = J22_new
        return {'t': t, 'x': x_vals, 'xi': xi_vals, 'symbol_value': p_func(x_vals, xi_vals), 'J11': J11_vals, 'J12': J12_vals, 'J21': J21_vals, 'J22': J22_vals}
    else:
        raise ValueError('Invalid method for 1D flow')

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
    dp_dxi = sp.diff(symbol, xi)
    dp_deta = sp.diff(symbol, eta)
    dp_dx = sp.diff(symbol, x)
    dp_dy = sp.diff(symbol, y)
    f_x = sp.lambdify((x, y, xi, eta), dp_dxi, 'numpy')
    f_y = sp.lambdify((x, y, xi, eta), dp_deta, 'numpy')
    f_xi = sp.lambdify((x, y, xi, eta), -dp_dx, 'numpy')
    f_eta = sp.lambdify((x, y, xi, eta), -dp_dy, 'numpy')
    p_func = sp.lambdify((x, y, xi, eta), symbol, 'numpy')
    H_px_sym = sp.Matrix([[sp.diff(symbol, xi, x), sp.diff(symbol, xi, y)], [sp.diff(symbol, eta, x), sp.diff(symbol, eta, y)]])
    H_px_func = sp.lambdify((x, y, xi, eta), H_px_sym, 'numpy')

    def _H_px(xv, yv, xiv, etav):
        return np.asarray(H_px_func(xv, yv, xiv, etav), dtype=float).reshape(2, 2)
    if method == 'rk45':
        z0_aug = list(z0) + [1.0, 0.0, 0.0, 1.0]

        def ode(t, z):
            xv, yv, xiv, etav = (z[0], z[1], z[2], z[3])
            J = z[4:].reshape(2, 2)
            dJ = _H_px(xv, yv, xiv, etav) @ J
            return [f_x(xv, yv, xiv, etav), f_y(xv, yv, xiv, etav), f_xi(xv, yv, xiv, etav), f_eta(xv, yv, xiv, etav), *dJ.ravel()]
        sol = solve_ivp(ode, tspan, z0_aug, method='RK45', t_eval=np.linspace(tspan[0], tspan[1], n_steps), rtol=1e-09, atol=1e-12)
        return {'t': sol.t, 'x': sol.y[0], 'y': sol.y[1], 'xi': sol.y[2], 'eta': sol.y[3], 'symbol_value': p_func(sol.y[0], sol.y[1], sol.y[2], sol.y[3]), 'J11': sol.y[4], 'J12': sol.y[5], 'J21': sol.y[6], 'J22': sol.y[7]}
    elif method in ('symplectic', 'verlet'):
        dt = (tspan[1] - tspan[0]) / n_steps
        t = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        y_vals = np.zeros(n_steps)
        xi_vals = np.zeros(n_steps)
        eta_vals = np.zeros(n_steps)
        J_vals = np.zeros((n_steps, 2, 2))
        x_vals[0], y_vals[0], xi_vals[0], eta_vals[0] = z0
        J_vals[0] = np.eye(2)
        if method == 'symplectic':
            for i in range(n_steps - 1):
                xv, yv = (x_vals[i], y_vals[i])
                xiv, etav = (xi_vals[i], eta_vals[i])
                xi_new = xiv + dt * f_xi(xv, yv, xiv, etav)
                eta_new = etav + dt * f_eta(xv, yv, xiv, etav)
                x_new = xv + dt * f_x(xv, yv, xi_new, eta_new)
                y_new = yv + dt * f_y(xv, yv, xi_new, eta_new)
                x_vals[i + 1] = x_new
                y_vals[i + 1] = y_new
                xi_vals[i + 1] = xi_new
                eta_vals[i + 1] = eta_new
                J_vals[i + 1] = J_vals[i] + dt * (_H_px(xv, yv, xiv, etav) @ J_vals[i])
        else:
            for i in range(n_steps - 1):
                xv, yv = (x_vals[i], y_vals[i])
                xiv, etav = (xi_vals[i], eta_vals[i])
                xi_half = xiv + 0.5 * dt * f_xi(xv, yv, xiv, etav)
                eta_half = etav + 0.5 * dt * f_eta(xv, yv, xiv, etav)
                x_new = xv + dt * f_x(xv, yv, xi_half, eta_half)
                y_new = yv + dt * f_y(xv, yv, xi_half, eta_half)
                xi_new = xi_half + 0.5 * dt * f_xi(x_new, y_new, xi_half, eta_half)
                eta_new = eta_half + 0.5 * dt * f_eta(x_new, y_new, xi_half, eta_half)
                x_vals[i + 1] = x_new
                y_vals[i + 1] = y_new
                xi_vals[i + 1] = xi_new
                eta_vals[i + 1] = eta_new
                Hpx_mid = 0.5 * (_H_px(xv, yv, xiv, etav) + _H_px(x_new, y_new, xi_new, eta_new))
                J_vals[i + 1] = J_vals[i] + dt * (Hpx_mid @ J_vals[i])
        return {'t': t, 'x': x_vals, 'y': y_vals, 'xi': xi_vals, 'eta': eta_vals, 'symbol_value': p_func(x_vals, y_vals, xi_vals, eta_vals), 'J11': J_vals[:, 0, 0], 'J12': J_vals[:, 0, 1], 'J21': J_vals[:, 1, 0], 'J22': J_vals[:, 1, 1]}
    else:
        raise ValueError('Invalid method for 2D flow')

# ----------------------------------------------------------------------
# 1D‑specific functions (Bohr–Sommerfeld, caustic detection)
# ----------------------------------------------------------------------
def bohr_sommerfeld_quantization(H, n_max=10, x_range=(-10, 10), hbar=1.0, E_range=(1e-06, 50.0)):
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
        raise ValueError('Cannot solve H=E for p(x,E)')
    p_expr = solutions[-1]
    p_func = sp.lambdify((x, E_sym), p_expr, 'numpy')
    alpha = 0.5

    def action(E):
        X = np.linspace(x_range[0], x_range[1], 2000)
        p_vals = p_func(X, E)
        p_vals = np.real_if_close(p_vals)
        p_vals = np.real(p_vals)
        if np.ndim(p_vals) == 0:
            return 0.0
        mask = p_vals >= 0
        if not np.any(mask):
            return 0.0
        idx = np.where(mask)[0]
        a, b = (X[idx[0]], X[idx[-1]])

        def integrand(xv):
            return p_func(xv, E)
        I, _ = quad(integrand, a, b, epsabs=1e-10, epsrel=1e-10)
        return I / np.pi
    targets = [hbar * (n + alpha) for n in range(n_max)]
    E_scan = np.linspace(E_range[0], E_range[1], 200)
    I_scan = [action(E) for E in E_scan]
    energies = []
    actions = []
    quantum_numbers = []
    for n, Itarget in zip(range(n_max), targets):
        found = False
        for k in range(len(E_scan) - 1):
            if (I_scan[k] - Itarget) * (I_scan[k + 1] - Itarget) < 0:
                E_left, E_right = (E_scan[k], E_scan[k + 1])
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
    return {'n': np.array(quantum_numbers), 'E_n': np.array(energies), 'actions': np.array(actions), 'hbar': hbar, 'alpha': alpha}

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
    return {'x_grid': X, 'xi_grid': XI, 'caustic_indicator': Z, 'threshold': np.percentile(Z, 10)}

def propagate_singularity(symbol, initial_sing_support, tspan, dim=None, n_samples=None):
    """
    Propagate singular support along bicharacteristics.
    """
    dim = _infer_dim(symbol, dim)
    trajectories = []
    for z0 in initial_sing_support:
        traj = bicharacteristic_flow(symbol, z0, tspan, dim=dim, method='symplectic')
        trajectories.append(traj)
    endpoints = [(traj['x'][-1], traj.get('y', [None])[-1], traj['xi'][-1], traj.get('eta', [None])[-1]) for traj in trajectories]
    return {'trajectories': trajectories, 'endpoints': endpoints, 'initial': initial_sing_support}

# ----------------------------------------------------------------------
# Visualisation functions (non‑WKB)
# ----------------------------------------------------------------------
def plot_characteristic_set(symbol, x_range, xi_range, dim=None, resolution=200, **kwargs):
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
        plt.figure(figsize=(8, 6))
        plt.contour(X, XI, Z, levels=[0], colors='red', linewidths=3)
        plt.pcolormesh(X, XI, np.log10(np.abs(Z) + 1e-10), shading='auto', cmap='viridis', alpha=0.5)
        plt.colorbar(label='log₁₀|p|')
        plt.xlabel('x')
        plt.ylabel('ξ')
        plt.title('Characteristic variety (1D)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
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
        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=[0], colors='red', linewidths=3)
        plt.pcolormesh(X, Y, np.log10(np.abs(Z) + 1e-10), shading='auto', cmap='viridis', alpha=0.5)
        plt.colorbar(label='log₁₀|p|')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Characteristic set (ξ={xi0}, η={eta0})')
        plt.axis('equal')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_bicharacteristics(symbol, initial_points, tspan, dim=None, projection='position', **kwargs):
    """
    Plot bicharacteristic curves.

    For 1D: plots in (x,ξ) plane.
    For 2D: projection can be 'position' (x-y), 'frequency' (ξ-η) or 'mixed' (x-ξ).
    """
    dim = _infer_dim(symbol, dim)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_points)))
    for idx, z0 in enumerate(initial_points):
        traj = bicharacteristic_flow(symbol, z0, tspan, dim=dim, method='symplectic', n_steps=500)
        if dim == 1:
            ax.plot(traj['x'], traj['xi'], color=colors[idx], alpha=0.7, lw=2)
            ax.plot(traj['x'][0], traj['xi'][0], 'go', ms=8)
            ax.plot(traj['x'][-1], traj['xi'][-1], 'ro', ms=6)
        elif projection == 'position':
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
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
    else:
        if projection == 'position':
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif projection == 'frequency':
            ax.set_xlabel('ξ')
            ax.set_ylabel('η')
        else:
            ax.set_xlabel('x')
            ax.set_ylabel('ξ')
        ax.axis('equal')
    ax.set_title('Bicharacteristics')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_wavefront_set(symbol, initial_sing_support, tspan, dim=None, projection='cotangent', n_steps=500, cmap='plasma', show_flow=True, show_endpoints=True, title=None, ax=None):
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
    if dim is None:
        dim = 2 if len(initial_sing_support[0]) == 4 else 1
    n_seeds = len(initial_sing_support)
    colors = plt.get_cmap(cmap)(np.linspace(0.05, 0.95, n_seeds))
    trajs = []
    for z0 in initial_sing_support:
        traj = bicharacteristic_flow(symbol, z0, tspan, dim=dim, method='symplectic', n_steps=n_steps)
        trajs.append(traj)
    if projection == 'full' and dim == 2:
        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
        else:
            axes = np.asarray(ax).flatten()
            fig = axes[0].get_figure()
        ax_list = [(axes[0], 'x', 'y', 'position', 'Position space  (x, y)'), (axes[1], 'xi', 'eta', 'frequency', 'Frequency space  (ξ, η)'), (axes[2], 'x', 'xi', 'mixed_x', 'Cotangent slice  (x, ξ)'), (axes[3], 'y', 'eta', 'mixed_y', 'Cotangent slice  (y, η)')]
        for idx, traj in enumerate(trajs):
            c = colors[idx]
            x0, y0 = (traj['x'][0], traj['y'][0])
            xi0, eta0 = (traj['xi'][0], traj['eta'][0])
            xf, yf = (traj['x'][-1], traj['y'][-1])
            xif, etaf = (traj['xi'][-1], traj['eta'][-1])
            pairs = [(traj['x'], traj['y'], x0, y0, xf, yf), (traj['xi'], traj['eta'], xi0, eta0, xif, etaf), (traj['x'], traj['xi'], x0, xi0, xf, xif), (traj['y'], traj['eta'], y0, eta0, yf, etaf)]
            for (ax_obj, kx, ky, proj_label, panel_title), (u, v, u0, v0, uf, vf) in zip(ax_list, pairs):
                if show_flow:
                    ax_obj.plot(u, v, color=c, alpha=0.6, lw=1.2)
                if show_endpoints:
                    ax_obj.plot(u0, v0, 'o', color='limegreen', ms=5, zorder=5, markeredgewidth=0)
                    ax_obj.plot(uf, vf, 's', color='crimson', ms=4, zorder=5, markeredgewidth=0)
        for ax_obj, kx, ky, proj_label, panel_title in ax_list:
            ax_obj.set_xlabel({'x': 'x', 'xi': 'ξ', 'y': 'y', 'eta': 'η'}[kx])
            ax_obj.set_ylabel({'x': 'x', 'xi': 'ξ', 'y': 'y', 'eta': 'η'}[ky])
            ax_obj.set_title(panel_title, fontsize=10)
            ax_obj.grid(alpha=0.25)
            ax_obj.set_aspect('equal', adjustable='datalim')
        fig.suptitle(title or 'Wavefront set  WF(u)  [2D, full cotangent bundle]', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return (fig, axes)
    if ax is None:
        fig, axes = plt.subplots(figsize=(8, 6))
    else:
        axes = ax
        fig = axes.get_figure()
    if dim == 1:
        if projection in ('cotangent', 'phase'):
            get_u = lambda t: t['x']
            get_v = lambda t: t['xi']
            xlabel, ylabel = ('x', 'ξ')
            equal_aspect = False
        elif projection == 'position':
            get_u = lambda t: t['t']
            get_v = lambda t: t['x']
            xlabel, ylabel = ('t', 'x  (singular support)')
            equal_aspect = False
        else:
            raise ValueError(f"Unknown projection '{projection}' for dim=1. Choose 'cotangent' or 'position'.")
    elif projection in ('cotangent', 'mixed_x'):
        get_u = lambda t: t['x']
        get_v = lambda t: t['xi']
        xlabel, ylabel = ('x', 'ξ')
        equal_aspect = True
    elif projection == 'position':
        get_u = lambda t: t['x']
        get_v = lambda t: t['y']
        xlabel, ylabel = ('x', 'y')
        equal_aspect = True
    elif projection == 'frequency':
        get_u = lambda t: t['xi']
        get_v = lambda t: t['eta']
        xlabel, ylabel = ('ξ', 'η')
        equal_aspect = True
    elif projection == 'mixed_y':
        get_u = lambda t: t['y']
        get_v = lambda t: t['eta']
        xlabel, ylabel = ('y', 'η')
        equal_aspect = True
    else:
        raise ValueError(f"Unknown projection '{projection}' for dim=2. Choose 'cotangent', 'position', 'frequency', 'mixed_x', 'mixed_y', or 'full'.")
    for idx, traj in enumerate(trajs):
        c = colors[idx]
        u = get_u(traj)
        v = get_v(traj)
        if show_flow:
            axes.plot(u, v, color=c, alpha=0.65, lw=1.5)
        if show_endpoints:
            axes.plot(u[0], v[0], 'o', color='limegreen', ms=7, zorder=5, label='initial' if idx == 0 else '')
            axes.plot(u[-1], v[-1], 's', color='crimson', ms=5, zorder=5, label='final' if idx == 0 else '')
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
    return (fig, axes)

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
    x_init = np.asarray(initial_curve['x'])
    y_init = np.asarray(initial_curve['y'])
    xi_init = np.asarray(initial_curve['xi'])
    eta_init = np.asarray(initial_curve['eta'])
    n_pts = len(x_init)
    if n_rays is None:
        n_rays = n_pts
    if n_rays < n_pts:
        indices = np.linspace(0, n_pts - 1, n_rays, dtype=int)
        x_init = x_init[indices]
        y_init = y_init[indices]
        xi_init = xi_init[indices]
        eta_init = eta_init[indices]
    elif n_rays > n_pts:
        pass
    rays = []
    default_kwargs = {'method': 'symplectic', 'n_steps': 200}
    flow_kwargs = {**default_kwargs, **kwargs}
    for i in range(len(x_init)):
        z0 = (x_init[i], y_init[i], xi_init[i], eta_init[i])
        traj = bicharacteristic_flow(p, z0, (0, tmax), dim=2, **flow_kwargs)
        rays.append(traj)
    from caustics import RayCausticDetector
    detector = RayCausticDetector(rays, dimension=2, det_threshold=0.05)
    events = detector.detect()
    return events
    
# ======================================================================
# Plotting system moved here from psiop_ud.py to keep that module
# focused on operator algebra / numerics. See psiop_ud.py for the
# thin wrapper methods on PseudoDifferentialOperator that delegate here.
# ======================================================================

# ---- low-level rendering helpers (shared by the op-visualization and
#      characteristic-set/wavefront plots below) --------------------

def _slice_grid(op, kind, x_grid=None, xi_grid=None, y_grid=None, eta_grid=None, x0=0.0, y0=0.0, xi0=0.0, eta0=0.0):
    """Evaluate op.p_func over one canonical 2D slice of phase space.

    kind='freq'  : 1D -> vary (x, xi).   2D -> fix (x0, y0), vary (xi, eta).
                   (used by: fiber, characteristic set/gradient, micro-support)
    kind='space' : 1D -> vary (x, xi) (1D has only one slice, same as above).
                   2D -> fix (xi0, eta0), vary (x, y).
                   (used by: symbol amplitude, phase)

    Returns (axis1_vals, axis2_vals, axis1_label, axis2_label, Z).
    """
    if op.dim == 1:
        A, B = np.meshgrid(x_grid, xi_grid, indexing='ij')
        return (x_grid, xi_grid, 'x', '$\\xi$', op.p_func(A, B))
    if kind == 'space':
        A, B = np.meshgrid(x_grid, y_grid, indexing='ij')
        Z = op.p_func(A, B, np.full_like(A, xi0), np.full_like(B, eta0))
        return (x_grid, y_grid, 'x', 'y', Z)
    A, B = np.meshgrid(xi_grid, eta_grid, indexing='ij')
    Z = op.p_func(x0, y0, A, B)
    return (xi_grid, eta_grid, '$\\xi$', '$\\eta$', Z)

def _render_field(ax1, ax2, Z, style='pcolormesh', cmap='viridis', cbar_label=None, xlabel='x', ylabel='$\\xi$', title='', levels=50, contour_color='red', grid=False, show=True):
    """One shared renderer for pcolormesh / contourf / contour panels."""
    if style == 'pcolormesh':
        im_ = plt.pcolormesh(ax1, ax2, Z, shading='auto', cmap=cmap)
        plt.colorbar(im_, label=cbar_label)
    elif style == 'contourf':
        im_ = plt.contourf(ax1, ax2, Z, levels=levels, cmap=cmap)
        plt.colorbar(im_, label=cbar_label)
    elif style == 'contour':
        plt.contour(ax1, ax2, Z, levels=levels, colors=contour_color)
    else:
        raise ValueError(f'unknown style {style!r}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if grid:
        plt.grid(True)
    if show:
        plt.show()

def _grad_norm(Z):
    """|grad Z| over a 2D array, using a consistent abs()**2 convention on
    both axes (see visualize_characteristic_gradient's note: this unifies
    an inconsistency that used to exist between its 1D and 2D branches)."""
    gx = np.gradient(Z, axis=0)
    gy = np.gradient(Z, axis=1)
    return np.sqrt(np.abs(gx) ** 2 + np.abs(gy) ** 2)

def _quiver_colored(ax, X, Y, U, V, cmap='viridis', scale=25, width=0.004, colorbar=True, cbar_label='|field|', **quiver_kwargs):
    """Draw a quiver field with fixed-length arrows, colored by magnitude.

    Plain `plt.quiver(X, Y, U, V)` sizes each arrow by its own (U, V)
    magnitude, which is what makes raw vector-field plots look like a
    tangle of dark arrows of wildly different lengths whenever the field
    spans more than about one order of magnitude (e.g. near a symbol's
    zero set). Here every arrow is normalized to the same unit length
    (direction only), and the original magnitude is instead mapped to
    color via matplotlib's `quiver(X, Y, U, V, C)` form -- the same trick
    used for wind/flow-field plots.

    Parameters
    ----------
    ax : matplotlib Axes or the `matplotlib.pyplot` module
        Target to draw on; anything exposing `.quiver(...)` (an Axes) or
        module-level `quiver(...)` (`plt` itself) works.
    X, Y : ndarray
        Arrow base positions (same shape as U, V).
    U, V : ndarray
        Raw (unnormalized) vector field components.
    cmap : str, default='viridis'
        Colormap used for the magnitude.
    scale : float, default=25
        Passed to `quiver`; larger values shrink the (now uniform) arrow
        length. Tune this once the field is normalized -- it no longer
        needs to be re-tuned per symbol the way a magnitude-scaled plot
        would.
    width : float, default=0.004
        Arrow shaft width, passed to `quiver`.
    colorbar : bool, default=True
        If True, attach a colorbar labelled `cbar_label` showing the
        magnitude scale. Skipped automatically if `ax` has no attached
        figure to draw it on (e.g. `ax=plt`, when called more than once
        on the same axes -- pass `colorbar=False` for background/overlay
        fields to avoid stacking colorbars).
    **quiver_kwargs
        Extra keyword arguments forwarded to `quiver` (e.g. `alpha`).
        `color` is ignored if passed here, since color is used to encode
        magnitude; use `cmap` instead.

    Returns
    -------
    matplotlib.quiver.Quiver
        The artist returned by the underlying `quiver` call, so callers
        can attach their own colorbar/legend if `colorbar=False`.
    """
    quiver_kwargs.pop('color', None)
    mag = np.hypot(np.abs(U), np.abs(V))
    safe_mag = np.where(mag == 0, 1.0, mag)
    Un, Vn = (np.real(U) / safe_mag, np.real(V) / safe_mag)
    q = ax.quiver(X, Y, Un, Vn, mag, cmap=cmap, scale=scale, width=width, **quiver_kwargs)
    if colorbar:
        fig = ax.figure if hasattr(ax, 'figure') else plt.gcf()
        fig.colorbar(q, ax=ax if hasattr(ax, 'figure') else plt.gca(), label=cbar_label)
    return q

def _quiver_field(op, xlim, klim, density, vec_exprs_fn, title, scale=25, width=0.004, cmap='viridis', cbar_label='|field|'):
    """1D-only quiver plot of a vector field derived from the symbol.

    Arrows are drawn at fixed length; the field's local magnitude is
    encoded by color instead (see `_quiver_colored`).

    vec_exprs_fn(p, x, xi) -> (U_expr, V_expr)
    """
    if op.dim != 1:
        raise NotImplementedError('Only 1D version implemented.')
    x, = op.vars_x
    xi = symbols('xi', real=True)
    x_vals = np.linspace(*xlim, density)
    xi_vals = np.linspace(*klim, density)
    X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
    U_expr, V_expr = vec_exprs_fn(op.symbol, x, xi)
    U = lambdify((x, xi), simplify(U_expr), 'numpy')(X, XI)
    V = lambdify((x, xi), simplify(V_expr), 'numpy')(X, XI)
    if np.isscalar(U):
        U = np.full_like(X, U, dtype=float)
    if np.isscalar(V):
        V = np.full_like(X, V, dtype=float)
    _quiver_colored(plt.gca(), X, XI, U, V, cmap=cmap, scale=scale, width=width, cbar_label=cbar_label)
    plt.xlabel('x')
    plt.ylabel('$\\xi$')
    plt.title(title)
    plt.grid(True)
    plt.show()

def _make_real(expr):
    """Re(expr), fully evaluated -- used when a Hamiltonian field may come
    out complex-valued from sympy but only the real part is physically
    meaningful for the flow."""
    return simplify(sp.re(expr.doit(deep=True)))

# ---- PseudoDifferentialOperator visualizations, as free functions
#      taking the operator `op` as the first argument (duck-typed on
#      op.p_func / op.dim / op.vars_x / op.symbol / op.symplectic_flow /
#      op.expr) -------------------------------------------------------

def visualize_fiber(op, x_grid, xi_grid, x0=0.0, y0=0.0):
    """Plot the cotangent fiber structure at a fixed spatial point (x0[, y0]).
    See _slice_grid/_render_field docstrings for the shared implementation.
    NOTE: original signature has no eta_grid param -- 2D reuses xi_grid
    for both frequency axes, matching the original behavior exactly."""
    a1, a2, l1, l2, Z = _slice_grid(op, 'freq', x_grid, xi_grid, eta_grid=xi_grid, x0=x0, y0=y0)
    title = 'Cotangent Fiber Structure' if op.dim == 1 else f'Cotangent Fiber at x={x0}, y={y0}'
    _render_field(a1, a2, np.abs(Z), style='contourf', cbar_label='|Symbol|', xlabel=l1, ylabel=l2, title=title)

def visualize_symbol_amplitude(op, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
    """Display |p(x, xi)| (1D) or |p(x, y, xi0, eta0)| (2D) as a color map."""
    kind = 'freq' if op.dim == 1 else 'space'
    a1, a2, l1, l2, Z = _slice_grid(op, kind, x_grid, xi_grid, y_grid, eta_grid, xi0=xi0, eta0=eta0)
    title = 'Symbol Amplitude |p(x, ξ)|' if op.dim == 1 else f'Symbol Amplitude at ξ={xi0}, η={eta0}'
    _render_field(a1, a2, np.abs(Z), style='pcolormesh', cbar_label='|Symbol|', xlabel=l1, ylabel=l2, title=title)

def visualize_phase(op, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
    """Plot arg(p(x, xi)) (1D) or arg(p(x, y, xi0, eta0)) (2D)."""
    kind = 'freq' if op.dim == 1 else 'space'
    a1, a2, l1, l2, Z = _slice_grid(op, kind, x_grid, xi_grid, y_grid, eta_grid, xi0=xi0, eta0=eta0)
    title = 'Phase Portrait (arg p(x, ξ))' if op.dim == 1 else f'Phase Portrait at ξ={xi0}, η={eta0}'
    _render_field(a1, a2, np.angle(Z), style='pcolormesh', cmap='twilight', cbar_label='arg(Symbol) [rad]', xlabel=l1, ylabel=l2, title=title)

def visualize_characteristic_set(op, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0, levels=[0.1]):
    """Visualize the characteristic set p(x, xi) ~= 0 (1D) or the (xi, eta)
    slice at fixed (x0, y0) (2D)."""
    if op.dim not in (1, 2):
        raise NotImplementedError('Only 1D/2D characteristic sets supported.')
    if op.dim == 2 and eta_grid is None:
        raise ValueError('eta_grid must be provided for 2D visualization.')
    a1, a2, l1, l2, Z = _slice_grid(op, 'freq', x_grid, xi_grid, y_grid, eta_grid, x0=x0, y0=y0)
    title = 'Characteristic Set (p(x, ξ) ≈ 0)' if op.dim == 1 else f'Characteristic Set at x={x0}, y={y0}'
    _render_field(a1, a2, np.abs(Z), style='contour', levels=levels, xlabel=l1, ylabel=l2, title=title, grid=True)

def visualize_characteristic_gradient(op, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0):
    """Visualize |grad p| in phase space. NOTE: both the 1D and 2D
    branches now consistently use abs(.)**2 in the gradient norm (the
    original 1D branch omitted the abs(), inconsistently with 2D)."""
    a1, a2, l1, l2, Z = _slice_grid(op, 'freq', x_grid, xi_grid, y_grid, eta_grid, x0=x0, y0=y0)
    title = 'Gradient Norm (High Near Zeros)' if op.dim == 1 else f'Gradient Norm at x={x0}, y={y0}'
    _render_field(a1, a2, _grad_norm(Z), style='pcolormesh', cmap='inferno', cbar_label='|∇p|', xlabel=l1, ylabel=l2, title=title, grid=True)

def plot_hamiltonian_flow(op, x0=0.0, xi0=5.0, y0=0.0, eta0=0.0, tmax=1.0, n_steps=100, show_field=True):
    """Integrate and plot the Hamiltonian trajectories of the symbol in
    phase space. Delegates to the shared `integrate_singularity` engine
    instead of re-deriving the Hamiltonian vector field inline."""
    x0v = [x0] if op.dim == 1 else [x0, y0]
    xi0v = [xi0] if op.dim == 1 else [xi0, eta0]
    _, _, _, _, trajs = integrate_singularity(op.symbol, op.vars_x, x0=x0v, xi0=xi0v, tmax=tmax, n_frames=n_steps)
    Y = trajs[0]
    if op.dim == 1:
        x_vals, xi_vals = Y
        plt.plot(x_vals, xi_vals)
        plt.xlabel('x')
        plt.ylabel('ξ')
        plt.title('Hamiltonian Flow in Phase Space (1D)')
        plt.grid(True)
        plt.show()
    elif op.dim == 2:
        x_vals, y_vals, xi_vals, eta_vals = Y
        plt.plot(x_vals, y_vals, label='Position')
        _quiver_colored(plt.gca(), x_vals, y_vals, xi_vals, eta_vals, cmap='autumn', scale=20, width=0.003, alpha=0.7, cbar_label='$|(\\xi,\\eta)|$')
        if show_field:
            x, y = op.vars_x
            xi, eta = symbols('xi eta', real=True)
            H = op.symplectic_flow()
            dxdt = lambdify((x, y, xi, eta), _make_real(H['dx/dt']), 'numpy')
            dydt = lambdify((x, y, xi, eta), _make_real(H['dy/dt']), 'numpy')
            Xg, Yg = np.meshgrid(np.linspace(min(x_vals), max(x_vals), 20), np.linspace(min(y_vals), max(y_vals), 20))
            XI, ETA = (xi0 * np.ones_like(Xg), eta0 * np.ones_like(Yg))
            Ub, Vb = (dxdt(Xg, Yg, XI, ETA), dydt(Xg, Yg, XI, ETA))
            mag_b = np.hypot(Ub, Vb)
            safe_b = np.where(mag_b == 0, 1.0, mag_b)
            plt.quiver(Xg, Yg, Ub / safe_b, Vb / safe_b, color='gray', alpha=0.25, scale=30, width=0.002)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Hamiltonian Flow in Phase Space (2D)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

def plot_symplectic_vector_field(op, xlim=(-2, 2), klim=(-5, 5), density=30):
    """Quiver plot of the symplectic vector field (dp/dxi, -dp/dx). 1D only."""
    _quiver_field(op, xlim, klim, density, lambda p, x, xi: (diff(p, xi), -diff(p, x)), 'Symplectic Vector Field (1D)')

def visualize_micro_support(op, xlim=(-2, 2), klim=(-10, 10), threshold=0.001, density=300, xi0=0.0, eta0=0.0):
    """Visualize 1/|p(x, xi)| to highlight regions where the symbol is
    near zero. NOTE: no longer restricted to 1D -- the shared grid/render
    helpers already handle the 2D case (fixed xi0=eta0=0, scan x, y)."""
    x_grid = np.linspace(*xlim, density)
    xi_grid = np.linspace(*klim, density)
    if op.dim == 1:
        a1, a2, l1, l2, Z = _slice_grid(op, 'freq', x_grid, xi_grid)
    else:
        a1, a2, l1, l2, Z = _slice_grid(op, 'space', x_grid, None, x_grid, None, xi0=xi0, eta0=eta0)
    title = 'Micro-Support Estimate (1/|Symbol|)' if op.dim == 1 else f'Micro-Support Estimate at ξ={xi0}, η={eta0}'
    _render_field(a1, a2, 1 / (np.abs(Z) + 1e-10), style='contourf', cmap='inferno', cbar_label='$1/|p(x,\\xi)|$', xlabel=l1, ylabel=l2, title=title)

def group_velocity_field(op, xlim=(-2, 2), klim=(-10, 10), density=30):
    """Quiver plot of the group velocity field (1, dp/dxi). 1D only."""
    _quiver_field(op, xlim, klim, density, lambda p, x, xi: (sp.Integer(1), diff(p, xi)), 'Group Velocity Field (1D)')

def animate_operator_singularity(op, xi0=5.0, eta0=0.0, x0=0.0, y0=0.0, tmax=4.0, n_frames=100, projection=None):
    """Animate the propagation of a singularity under the Hamiltonian
    flow. Thin delegate to the module-level `animate_singularity`
    engine (previously ~130 lines of duplicated Hamiltonian/ODE setup
    here, plus a near-identical copy further down the module)."""
    rc('animation', html='jshtml')
    x0v = x0 if op.dim == 1 else [x0, y0]
    xi0v = xi0 if op.dim == 1 else [xi0, eta0]
    if projection is None:
        projection = 'phase' if op.dim == 1 else 'position'
    return animate_singularity(op.symbol, op.vars_x, x0=x0v, xi0=xi0v, tmax=tmax, n_frames=n_frames, projection=projection)

def interactive_symbol_analysis(pseudo_op, xlim=(-2, 2), ylim=(-2, 2), xi_range=(0.1, 5), eta_range=(-5, 5), density=50):
    """Launch an ipywidgets dashboard for symbol exploration. Same modes,
    same sliders, same defaults as before -- rewritten as a mode-table
    dispatcher that delegates to the visualize_*/plot_* methods above
    instead of duplicating their 1D/2D branches inline (previously
    ~260 lines of near-duplicated if-elif chains)."""
    dim = pseudo_op.dim
    x_vals = np.linspace(*xlim, density)
    y_vals = np.linspace(*ylim, density) if dim == 2 else None
    xi_lin = np.linspace(*xi_range, density)
    eta_lin = np.linspace(*eta_range, density) if dim == 2 else None
    if dim == 1:
        modes = ['Symbol Amplitude', 'Symbol Phase', 'Micro-Support (1/|p|)',
                  'Cotangent Fiber', 'Characteristic Set', 'Characteristic Gradient',
                  'Group Velocity Field', 'Symplectic Vector Field', 'Hamiltonian Flow']
        needs = {
            'Symbol Amplitude': (), 'Symbol Phase': (), 'Micro-Support (1/|p|)': (), 
            'Group Velocity Field': (), 'Symplectic Vector Field': (), 
            'Hamiltonian Flow': ('xi', 'x'), 'Cotangent Fiber': (), 'Characteristic Set': (),
            'Characteristic Gradient': ()
        }
        mode_selector = Dropdown(options=modes, value='Symbol Amplitude', description='Mode:')
        xi_slider = FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='ξ₀')
        x_slider = FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x₀')
        all_sliders = {'xi': xi_slider, 'x': x_slider}

        def render(mode, xi0, x0):
            plt.close('all')
            plt.figure()
            if mode == 'Symbol Amplitude':
                visualize_symbol_amplitude(pseudo_op, x_vals, xi_lin, xi0=xi0)
            elif mode == 'Symbol Phase':
                visualize_phase(pseudo_op, x_vals, xi_lin, xi0=xi0)
            elif mode == 'Micro-Support (1/|p|)':
                visualize_micro_support(pseudo_op, xlim, xi_range, density=density)
            elif mode == 'Group Velocity Field':
                group_velocity_field(pseudo_op, xlim, xi_range, density=density)
            elif mode == 'Symplectic Vector Field':
                plot_symplectic_vector_field(pseudo_op, xlim, xi_range, density=density)
            elif mode == 'Cotangent Fiber':
                visualize_fiber(pseudo_op, x_vals, xi_lin, x0=x0)
            elif mode == 'Characteristic Set':
                visualize_characteristic_set(pseudo_op, x_vals, xi_lin, x0=x0)
            elif mode == 'Characteristic Gradient':
                visualize_characteristic_gradient(pseudo_op, x_vals, xi_lin, x0=x0)
            elif mode == 'Hamiltonian Flow':
                plot_hamiltonian_flow(pseudo_op, x0=x0, xi0=xi0)
        interactive_kwargs = {'mode': mode_selector, 'xi0': xi_slider, 'x0': x_slider}
        slider_order = ['xi', 'x']
    else:
        modes = ['Symbol Amplitude', 'Symbol Phase', 'Micro-Support (1/|p|)',
                  'Cotangent Fiber', 'Characteristic Set', 'Characteristic Gradient',
                  'Symplectic Vector Field', 'Hamiltonian Flow']
        needs = {
            'Symbol Amplitude': ('xi', 'eta'), 'Symbol Phase': ('xi', 'eta'),
            'Micro-Support (1/|p|)': ('xi', 'eta'), 'Symplectic Vector Field': ('xi', 'eta'),
            'Hamiltonian Flow': ('xi', 'eta', 'x', 'y'),
            'Cotangent Fiber': ('x', 'y'), 'Characteristic Set': ('x', 'y'),
            'Characteristic Gradient': ('x', 'y'),
        }
        mode_selector = Dropdown(options=modes, value='Symbol Amplitude', description='Mode:')
        xi_slider = FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='ξ₀')
        eta_slider = FloatSlider(min=eta_range[0], max=eta_range[1], step=0.1, value=1.0, description='η₀')
        x_slider = FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x₀')
        y_slider = FloatSlider(min=ylim[0], max=ylim[1], step=0.1, value=0.0, description='y₀')
        all_sliders = {'xi': xi_slider, 'eta': eta_slider, 'x': x_slider, 'y': y_slider}

        def render(mode, xi0, eta0, x0, y0):
            plt.close('all')
            plt.figure()
            if mode == 'Symbol Amplitude':
                visualize_symbol_amplitude(pseudo_op, x_vals, xi_lin, y_vals, eta_lin, xi0=xi0, eta0=eta0)
            elif mode == 'Symbol Phase':
                visualize_phase(pseudo_op, x_vals, xi_lin, y_vals, eta_lin, xi0=xi0, eta0=eta0)
            elif mode == 'Micro-Support (1/|p|)':
                visualize_micro_support(pseudo_op, xlim, xi_range, density=density, xi0=xi0, eta0=eta0)
            elif mode == 'Symplectic Vector Field':
                x, y = pseudo_op.vars_x
                xi, eta = symbols('xi eta', real=True)
                Xg, Yg = np.meshgrid(x_vals, y_vals, indexing='ij')
                U, V = lambdify((x, y, xi, eta), [diff(pseudo_op.expr, xi), diff(pseudo_op.expr, eta)], 'numpy')(Xg, Yg, xi0, eta0)
                _quiver_colored(plt.gca(), Xg, Yg, U, V, scale=10, width=0.004, cbar_label='|(dp/dξ, dp/dη)|')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Symplectic Field at ξ={xi0:.2f}, η={eta0:.2f}')
            elif mode == 'Cotangent Fiber':
                visualize_fiber(pseudo_op, xi_lin, eta_lin, x0=x0, y0=y0)
            elif mode == 'Characteristic Set':
                visualize_characteristic_set(pseudo_op, x_vals, xi_lin, y_vals, eta_lin, x0=x0, y0=y0)
            elif mode == 'Characteristic Gradient':
                visualize_characteristic_gradient(pseudo_op, x_vals, xi_lin, y_vals, eta_lin, x0=x0, y0=y0)
            elif mode == 'Hamiltonian Flow':
                plot_hamiltonian_flow(pseudo_op, x0=x0, y0=y0, xi0=xi0, eta0=eta0)
            if mode not in ('Cotangent Fiber', 'Characteristic Set', 'Characteristic Gradient', 'Hamiltonian Flow'):
                plt.show()
        interactive_kwargs = {'mode': mode_selector, 'xi0': xi_slider, 'eta0': eta_slider, 'x0': x_slider, 'y0': y_slider}
        slider_order = ['xi', 'eta', 'x', 'y']
    controls_box = VBox([mode_selector] + list(all_sliders.values()))

    def update_controls(change):
        active = needs[change['new']]
        controls_box.children = [mode_selector] + [all_sliders[k] for k in slider_order if k in active]
    mode_selector.observe(update_controls, names='value')
    update_controls({'new': mode_selector.value})
    out = interactive_output(render, interactive_kwargs)
    display(VBox([controls_box, out]))

def plot_pseudospectrum(Lambda, resolvent_norm, sigma_min_grid, epsilon_levels, eigenvalues):
    """
    Plot pseudospectrum results.
    
    Parameters
    ----------
    Lambda : ndarray
        Complex λ grid
    resolvent_norm : ndarray
        Resolvent norms
    sigma_min_grid : ndarray
        Smallest singular values
    epsilon_levels : list
        Contour levels
    eigenvalues : ndarray or None
        Eigenvalues to overlay
    """
    Lambda_re = Lambda.real
    Lambda_im = Lambda.imag
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    log_resolvent = np.log10(resolvent_norm + 1e-16)
    levels_log = np.log10(1.0 / np.array(epsilon_levels))
    valid_levels = [lv for lv in levels_log if log_resolvent.min() <= lv <= log_resolvent.max()]
    if len(valid_levels) > 0:
        cs = plt.contour(Lambda_re, Lambda_im, log_resolvent, levels=valid_levels, colors='blue', linewidths=1.5)
        labels = [f'ε={eps:.0e}' for eps in epsilon_levels[:len(valid_levels)]]
        fmt = dict(zip(cs.levels, labels))
        plt.clabel(cs, inline=True, fmt=fmt, fontsize=9)
    else:
        print('⚠️ Warning: No contours in specified epsilon range')
        cs = plt.contour(Lambda_re, Lambda_im, log_resolvent, levels=10, colors='blue', linewidths=1.5)
    if eigenvalues is not None:
        plt.plot(eigenvalues.real, eigenvalues.imag, 'r*', markersize=10, label='Eigenvalues', markeredgecolor='darkred')
    plt.xlabel('Re(λ)', fontsize=12)
    plt.ylabel('Im(λ)', fontsize=12)
    plt.title('ε-Pseudospectrum: log₁₀(‖(H - λI)⁻¹‖)', fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    from matplotlib.colors import LogNorm
    sigma_plot = np.where(np.isfinite(sigma_min_grid), sigma_min_grid, np.nan)
    vmin = np.nanmin(sigma_plot[sigma_plot > 0]) if np.any(sigma_plot > 0) else 1e-10
    vmax = np.nanmax(sigma_plot)
    cs2 = plt.contourf(Lambda_re, Lambda_im, sigma_plot, levels=50, cmap='viridis', norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(cs2, label='σ_min(H - λI)')
    if eigenvalues is not None:
        plt.plot(eigenvalues.real, eigenvalues.imag, 'r*', markersize=10, markeredgecolor='darkred')
    for eps in epsilon_levels:
        cs_eps = plt.contour(Lambda_re, Lambda_im, sigma_plot, levels=[eps], colors='red', linewidths=2, alpha=0.8)
    plt.xlabel('Re(λ)', fontsize=12)
    plt.ylabel('Im(λ)', fontsize=12)
    plt.title('Smallest singular value σ_min(H - λI)', fontsize=13)
    plt.grid(alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ---- solver-output plotting/animation (scalar & matrix fields from
#      the time-stepping solvers, and bicharacteristic/singularity
#      trajectory animations) ---------------------------------------

def _matrix_of(s_expr):
    """
    Coerce a symbol expression into a sympy Matrix.

    If `s_expr` is already a MatrixBase, list, or tuple, it is converted
    via `sp.Matrix(s_expr)`. A bare scalar expression is wrapped into a
    1×1 matrix so that downstream code can treat scalar and matrix-valued
    operators uniformly.

    Parameters
    ----------
    s_expr : sympy.Expr, sympy.MatrixBase, list, or tuple
        The symbol or matrix of symbols.

    Returns
    -------
    sympy.Matrix
        Square matrix of symbol expressions.

    Raises
    ------
    ValueError
        If the resulting matrix is not square (checked by callers).
    """
    if isinstance(s_expr, (sp.MatrixBase, list, tuple)):
        return sp.Matrix(s_expr)
    return sp.Matrix([[s_expr]])

def _quantity_fn(quantity):
    """'real' | 'imag' | 'abs' -> the corresponding numpy function."""
    try:
        return {'real': np.real, 'imag': np.imag, 'abs': np.abs}[quantity]
    except KeyError:
        raise ValueError("quantity must be 'real', 'imag', or 'abs'")


def _finish_headless(fig, save_path=None):
    """Standard ending for the 'returns a Figure, doesn't display it'
    PDE-solution plots: tight layout, optional save, close, return."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return fig


def plot_scalar_1d(t, U, x, title="u(x, t)", quantity='real',
                   n_snapshots=6, save_path=None):
    """
    Plot a scalar 1D space-time solution as a combined heatmap and
    snapshot overlay.
    
    Parameters
    ----------
    t : ndarray, shape (n_times,)
        Time samples.
    U : ndarray, shape (n_times, Nx)
        Solution values u(x, t) sampled on the grid.
    x : ndarray, shape (Nx,)
        Spatial grid.
    title : str, default="u(x, t)"
        Base title used for the heatmap panel.
    quantity : {'real', 'imag', 'abs'}, default='real'
        Which part of U to plot.
    n_snapshots : int, default=6
        Number of time slices drawn as line overlays in the second panel.
    save_path : str, optional
        If given, the figure is saved to this path (dpi=150) before closing.
    
    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (already closed via `_finish_headless`, so it
        will not display inline; use `save_path` or re-show it explicitly).
    """
    field = _quantity_fn(quantity)(U)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    im = ax1.pcolormesh(x, t, field, shading='auto', cmap='RdBu_r')
    ax1.set_xlabel('x'); ax1.set_ylabel('t')
    ax1.set_title(f"{title} -- space-time ({quantity})")
    fig.colorbar(im, ax=ax1)

    idx = np.linspace(0, len(t) - 1, n_snapshots).astype(int)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    for c, i in zip(cmap, idx):
        ax2.plot(x, field[i], color=c, label=f"t={t[i]:.2f}")
    ax2.set_xlabel('x'); ax2.set_ylabel(quantity)
    ax2.set_title("snapshots")
    ax2.legend(fontsize=8, ncol=2)

    return _finish_headless(fig, save_path)


def plot_matrix_1d(t, U, x, labels=None, quantity='real', save_path=None):
    """
    Plot each component of a matrix-valued 1D solution as a stacked
    space-time heatmap.
    
    Parameters
    ----------
    t : ndarray, shape (n_times,)
        Time samples.
    U : ndarray, shape (n_times, size, Nx)
        Diagonal (or otherwise reduced) matrix solution components, one
        row of panels per index k = 0, ..., size-1.
    x : ndarray, shape (Nx,)
        Spatial grid.
    labels : list of str, optional
        One label per component; defaults to `["u_1", ..., "u_size"]`.
    quantity : {'real', 'imag', 'abs'}, default='real'
        Which part of U to plot.
    save_path : str, optional
        If given, the figure is saved to this path (dpi=150) before closing.
    
    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (already closed via `_finish_headless`).
    """
    field_fn = _quantity_fn(quantity)
    size = U.shape[1]
    labels = labels or [f"u_{k+1}" for k in range(size)]

    fig, axes = plt.subplots(size, 1, figsize=(6, 3 * size), sharex=True)
    axes = [axes] if size == 1 else axes

    for k, ax in enumerate(axes):
        im = ax.pcolormesh(x, t, field_fn(U[:, k, :]), shading='auto', cmap='RdBu_r')
        ax.set_ylabel('t')
        ax.set_title(f"{labels[k]} ({quantity})")
        fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel('x')

    return _finish_headless(fig, save_path)


def plot_scalar_2d(t, U, x, y, times=None, quantity='real', save_path=None):
    """
    Plot a scalar 2D solution at selected time instants as a row of
    side-by-side pcolormesh panels.
    
    Parameters
    ----------
    t : ndarray, shape (n_times,)
        Time samples.
    U : ndarray, shape (n_times, Nx, Ny)
        Solution values u(x, y, t) sampled on the grid.
    x, y : ndarray
        Spatial grids along each axis.
    times : array_like of int, optional
        Indices into `t` selecting which snapshots to plot. Defaults to 6
        indices evenly spaced across the whole time range.
    quantity : {'real', 'imag', 'abs'}, default='real'
        Which part of U to plot.
    save_path : str, optional
        If given, the figure is saved to this path (dpi=150) before closing.
    
    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (already closed via `_finish_headless`).
    """
    field_fn = _quantity_fn(quantity)
    times = np.linspace(0, len(t) - 1, 6).astype(int) if times is None else times

    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    axes = [axes] if n == 1 else axes

    im = None
    for ax, i in zip(axes, times):
        im = ax.pcolormesh(x, y, field_fn(U[i]).T, shading='auto', cmap='RdBu_r')
        ax.set_title(f"t={t[i]:.2f}")
        ax.set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im, ax=axes[-1])

    return _finish_headless(fig, save_path)


def animate_scalar_1d(t, U, x, quantity='real', interval=40, save_path=None):
    """
    Animate a scalar 1D solution as a line plot evolving in time.
    
    Parameters
    ----------
    t : ndarray, shape (n_times,)
        Time samples.
    U : ndarray, shape (n_times, Nx)
        Solution values u(x, t) sampled on the grid.
    x : ndarray, shape (Nx,)
        Spatial grid.
    quantity : {'real', 'imag', 'abs'}, default='real'
        Which part of U to plot; also sets the fixed y-axis limits from
        the min/max of that quantity over the whole trajectory.
    interval : int, default=40
        Delay between animation frames, in milliseconds.
    save_path : str, optional
        If given, the animation is saved to this path via `anim.save`.
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object (figure is not shown automatically).
"""
    from matplotlib.animation import FuncAnimation
    field = _quantity_fn(quantity)(U)
    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot(x, field[0])
    ax.set_ylim(field.min(), field.max())
    ax.set_xlabel('x'); ax.set_ylabel(quantity)
    title = ax.set_title(f"t={t[0]:.2f}")

    def update(i):
        line.set_ydata(field[i])
        title.set_text(f"t={t[i]:.2f}")
        return line, title

    anim = FuncAnimation(fig, update, frames=len(t), interval=interval, blit=False)
    if save_path:
        anim.save(save_path)
    plt.close(fig)
    return anim


# --- New: matrix-field solvers (solve_matrix_field / solve_sylvester_field)
# had NO plotting function at all -- their output shape (n_saved, N, N,
# *grid) doesn't fit plot_matrix_1d's (n_saved, size, *grid). These fill
# that gap.

def _matrix_field_reduce(U, component, quantity):
    """Reduce a (n_saved, N, N, *grid) matrix-field array to a single
    real/complex scalar field (n_saved, *grid), per `component`."""
    if component == 'trace':
        return np.trace(U, axis1=1, axis2=2), _quantity_fn(quantity)
    if component == 'frobenius':
        return np.linalg.norm(U, axis=(1, 2)), np.abs  # already real, non-negative
    if component == 'diag':
        raise ValueError("component='diag' needs the *_1d panel-per-entry "
                         "helper (plot_matrix_field_1d); pick 'trace', "
                         "'frobenius', or an (i, j) entry here.")
    i, j = component
    return U[:, i, j, ...], _quantity_fn(quantity)


def plot_matrix_field_1d(t, U, x, quantity='abs', component='diag', labels=None, save_path=None):
    """
    Space-time heatmap(s) for a matrix-valued 1D solution.
    
    Parameters
    ----------
    t : ndarray, shape (n_times,)
        Time samples.
    U : ndarray, shape (n_times, N, N, Nx)
        Matrix-valued solution field, as returned by `solve_matrix_field`
        / `solve_sylvester_field` in 1D.
    x : ndarray, shape (Nx,)
        Spatial grid.
    quantity : {'real', 'imag', 'abs'}, default='abs'
        Which part of the (reduced) field to plot.
    component : 'diag' | 'trace' | 'frobenius' | (i, j), default='diag'
        'diag'      -- one panel per diagonal entry U_kk(x, t).
        'trace'     -- single panel, sum_k U_kk(x, t).
        'frobenius' -- single panel, ||U(x, t)||_F.
        (i, j)      -- single panel, the (i, j) entry U_ij(x, t).
    labels : list of str, optional
        Panel labels used when `component='diag'`; defaults to
        `["U_11", "U_22", ...]`.
    save_path : str, optional
        If given, the figure is saved to this path (dpi=150) before closing.
    
    Returns
    -------
    matplotlib.figure.Figure
        The completed figure (already closed via `_finish_headless`).
    """
    if component == 'diag':
        field_fn = _quantity_fn(quantity)
        size = U.shape[1]
        labels = labels or [f"U_{k+1}{k+1}" for k in range(size)]
        fig, axes = plt.subplots(size, 1, figsize=(6, 3 * size), sharex=True)
        axes = [axes] if size == 1 else axes
        for k, ax in enumerate(axes):
            im = ax.pcolormesh(x, t, field_fn(U[:, k, k, :]), shading='auto', cmap='RdBu_r')
            ax.set_ylabel('t'); ax.set_title(f"{labels[k]} ({quantity})")
            fig.colorbar(im, ax=ax)
        axes[-1].set_xlabel('x')
        return _finish_headless(fig, save_path)

    panel, field_fn = _matrix_field_reduce(U, component, quantity)
    if component in ('trace', 'frobenius'):
        title = 'tr U(x, t)' if component == 'trace' else '||U(x, t)||_F'
    else:
        title = f"U_{component[0]+1}{component[1]+1}(x, t)"

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.pcolormesh(x, t, field_fn(panel), shading='auto', cmap='RdBu_r')
    ax.set_xlabel('x'); ax.set_ylabel('t'); ax.set_title(f"{title} ({quantity})")
    fig.colorbar(im, ax=ax)
    return _finish_headless(fig, save_path)


def plot_matrix_field_2d(t, U, x, y, times=None, quantity='abs', component='trace', save_path=None):
    """Snapshot panels for a matrix-valued 2D solution, shape
    (n_times, N, N, Nx, Ny) -- the output of solve_matrix_field /
    solve_sylvester_field in 2D. `component` as in `plot_matrix_field_1d`,
    except 'diag' isn't supported here (pick a single scalar reduction:
    'trace', 'frobenius', or an (i, j) entry)."""
    panel_all, field_fn = _matrix_field_reduce(U, component, quantity)
    times = np.linspace(0, len(t) - 1, 6).astype(int) if times is None else times

    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), sharey=True)
    axes = [axes] if n == 1 else axes
    im = None
    for ax, idx in zip(axes, times):
        im = ax.pcolormesh(x, y, field_fn(panel_all[idx]).T, shading='auto', cmap='RdBu_r')
        ax.set_title(f"t={t[idx]:.2f}"); ax.set_xlabel('x')
    axes[0].set_ylabel('y')
    fig.colorbar(im, ax=axes[-1])
    return _finish_headless(fig, save_path)


# --- New: solve_second_order returns (U, V) but there was no combined
# view -- previously required calling plot_scalar_1d twice by hand.

def plot_wave_solution_1d(t, U, V, x, quantity='real', save_path=None):
    """Side-by-side space-time heatmaps of displacement U and velocity V,
    as returned by solve_second_order (scalar, 1D case)."""
    field_fn = _quantity_fn(quantity)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, field, label in zip(axes, (U, V), ('u', r'$\partial_t u$')):
        im = ax.pcolormesh(x, t, field_fn(field), shading='auto', cmap='RdBu_r')
        ax.set_xlabel('x'); ax.set_title(f"{label} ({quantity})")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel('t')
    return _finish_headless(fig, save_path)

# ----------------------------------------------------------------------
# Singularity & Ray Flow
# ----------------------------------------------------------------------

def _order_freq_vars(freq, dim):
    """
    Order a set of free frequency symbols into canonical (ξ, η) or
    (ξ, η, …) sequence matching the spatial dimension.

    Recognised canonical names are matched first ('xi', 'eta', 'kx',
    'ky', 'k1', 'k2'); any remaining symbols are appended in
    alphabetical order. This ensures a deterministic variable ordering
    for differentiation and lambdification regardless of the order in
    which SymPy reports free symbols.

    Parameters
    ----------
    freq : iterable of sympy.Symbol
        Candidate frequency symbols extracted from an expression.
    dim : int
        Number of frequency variables expected (1 or 2).

    Returns
    -------
    list of sympy.Symbol
        Ordered frequency symbols of length `dim`.
    """
    if dim == 1:
        return list(freq)
    by_name = {str(s): s for s in freq}
    out = []
    for canon in ('xi', 'eta', 'kx', 'ky', 'k1', 'k2'):
        if canon in by_name and len(out) < dim:
            out.append(by_name.pop(canon))
    out += sorted(by_name.values(), key=lambda s: s.name)
    return out[:dim]

def characteristic_hamiltonians(s_expr, vars_x, vars_xi=None):
    """
    Extract the characteristic Hamiltonian functions H(x, ξ) from a
    (possibly matrix-valued) operator symbol.

    For a scalar symbol p(x, ξ), the single Hamiltonian is

        H(x, ξ) = Re(p(x, ξ))

    For a matrix symbol P(x, ξ), the eigenvalues λ_k(x, ξ) are computed
    symbolically and each branch yields

        H_k(x, ξ) = Re(λ_k(x, ξ))

    These Hamiltonians generate the bicharacteristic (ray) flow via
    Hamilton's equations:

        ẋ = ∂H/∂ξ,   ξ̇ = −∂H/∂x

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Operator symbol (scalar or matrix-valued).
    vars_x : list of sympy.Symbol
        Spatial variables.
    vars_xi : list of sympy.Symbol, optional
        Frequency variables. If None, they are inferred from the free
        symbols of `s_expr` that are not in `vars_x`, and ordered
        canonically via `_order_freq_vars`.

    Returns
    -------
    H_list : list of sympy.Expr
        One Hamiltonian per characteristic branch (eigenvalue).
    xs : list of sympy.Symbol
        Canonical spatial symbols (real=True) used in H_list.
    xis : list of sympy.Symbol
        Canonical frequency symbols (real=True) used in H_list.

    Notes
    -----
    The classical Hamiltonian governing bicharacteristic flow is the
    (real part of the) principal symbol itself.  A previous version of
    this function erroneously computed ``Re(i * p)``, which vanishes
    identically for any real-valued symbol and therefore produced
    trivial (stationary) trajectories in the flow visualization.

    The substitution to fresh canonical symbols ensures consistent
    differentiation even if the input expression uses symbols with
    different assumptions.
    """
    S = _matrix_of(s_expr)
    dim = len(vars_x)
    freq = [s for s in S.free_symbols if s not in set(vars_x)]
    vars_xi = list(vars_xi) if vars_xi is not None else _order_freq_vars(freq, dim)

    xs = [sp.Symbol(v.name, real=True) for v in vars_x]
    xis = [sp.Symbol(v.name, real=True) for v in vars_xi]
    S = S.subs(dict(zip(list(vars_x) + list(vars_xi), xs + xis)))

    eigen = [S[0, 0]] if S.shape == (1, 1) else list(S.eigenvals().keys())

    H_list = []
    for lam in eigen:
        H_list.append(sp.simplify(sp.re(lam)))
    return H_list, xs, xis

def integrate_singularity(s_expr, vars_x, x0=0.0, xi0=5.0, tmax=4.0,
                          n_frames=100, vars_xi=None, branches='all',
                          method='RK45', **ivp_kwargs):
    """
    Numerically integrate bicharacteristic (Hamiltonian ray) trajectories
    from an initial phase-space point (x₀, ξ₀).

    For each characteristic branch H_k, Hamilton's equations

        ẋ = ∂H_k/∂ξ,   ξ̇ = −∂H_k/∂x

    are integrated using `scipy.integrate.solve_ivp` over [0, tmax].

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Operator symbol from which Hamiltonians are extracted.
    vars_x : list of sympy.Symbol
        Spatial variables.
    x0 : float or array_like, default 0.0
        Initial spatial position(s). Scalar for 1D, sequence for 2D.
    xi0 : float or array_like, default 5.0
        Initial frequency (momentum) component(s).
    tmax : float, default 4.0
        Final integration time.
    n_frames : int, default 100
        Number of output time samples in [0, tmax].
    vars_xi : list of sympy.Symbol, optional
        Explicit frequency variables (inferred if None).
    branches : 'all', int, or list of int, default 'all'
        Which characteristic branches to integrate. 'all' integrates
        every branch; an int or list selects specific ones.
    method : str, default 'RK45'
        ODE solver method passed to `solve_ivp`.
    **ivp_kwargs
        Additional keyword arguments forwarded to `solve_ivp`
        (e.g. `rtol`, `atol`, `max_step`).

    Returns
    -------
    H_list : list of sympy.Expr
        Hamiltonian expressions for the integrated branches.
    xs : list of sympy.Symbol
        Canonical spatial symbols.
    xis : list of sympy.Symbol
        Canonical frequency symbols.
    t_eval : ndarray, shape (n_frames,)
        Time samples at which trajectories are recorded.
    trajs : list of ndarray
        `trajs[b]` has shape (2·dim, n_frames): the first dim rows are
        position components, the last dim rows are momentum components.

    Examples
    --------
    >>> H, xs, xis, t, trajs = integrate_singularity(xi**2 + x**2, [x],
    ...                                              x0=0.0, xi0=3.0, tmax=6.0)
    """
    H_all, xs, xis = characteristic_hamiltonians(s_expr, vars_x, vars_xi)
    H_list = [H_all[b] for b in ([branches] if isinstance(branches, int) else branches)] if branches != 'all' else H_all

    dim = len(xs)
    y0 = np.concatenate([np.atleast_1d(x0), np.atleast_1d(xi0)])
    t_eval = np.linspace(0.0, tmax, n_frames)

    trajs = []
    for H in H_list:
        rhs_exprs = [sp.diff(H, k) for k in xis] + [-sp.diff(H, x) for x in xs]
        f = sp.lambdify(xs + xis, rhs_exprs, 'numpy')
        sol = solve_ivp(lambda t, Y: f(*Y), (0.0, tmax), y0, t_eval=t_eval, method=method, **ivp_kwargs)
        trajs.append(sol.y)
    return H_list, xs, xis, t_eval, trajs

def _trail_animation(coords_list, colors, px, py, pz=None, interval=50, pad_frac=0.08):
    """Shared 'growing dashed trail + moving point' animation builder for
    both the 2D and 3D singularity animations -- factors out what used to
    be two near-identical copies of the same figure/update-function setup.

    FIX: the axes are now explicitly sized to the full trajectory range
    (with a small padding margin) before the animation starts. Previously
    no xlim/ylim(/zlim) were set at all: since every trail/point artist is
    created empty (`ax.plot([], [])`) and only ever updated via
    `set_data`/`set_3d_properties` (not `ax.relim()` + `autoscale_view()`),
    matplotlib never grew the view beyond its default (0, 1) x (0, 1) box,
    so the rendered animation domain was almost always far too small to
    show the actual ray flow.
    """
    is_3d = pz is not None
    fig = plt.figure(figsize=(7.5, 5.5) if is_3d else (6, 5))
    ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)

    def _padded_range(values):
        lo, hi = float(np.min(values)), float(np.max(values))
        span = hi - lo
        pad = span * pad_frac if span > 0 else (abs(lo) * pad_frac or 1.0)
        return lo - pad, hi + pad

    all_x = np.concatenate([np.atleast_1d(c[px]) for c in coords_list])
    all_y = np.concatenate([np.atleast_1d(c[py]) for c in coords_list])
    ax.set_xlim(*_padded_range(all_x))
    ax.set_ylim(*_padded_range(all_y))
    ax.set_xlabel(px)
    ax.set_ylabel(py)
    if is_3d:
        all_z = np.concatenate([np.atleast_1d(c[pz]) for c in coords_list])
        ax.set_zlim(*_padded_range(all_z))
        ax.set_zlabel(pz)
    else:
        ax.grid(True, alpha=0.3)

    trails, points = [], []
    for c in colors:
        if is_3d:
            tr, = ax.plot([], [], [], ls='--', lw=1.3, color=c)
            pt, = ax.plot([], [], [], 'o', ms=6, color=c)
        else:
            tr, = ax.plot([], [], ls='--', lw=1.3, alpha=0.6, color=c)
            pt, = ax.plot([], [], 'o', ms=6.5, color=c)
        trails.append(tr)
        points.append(pt)

    n_act = min(len(c[px]) for c in coords_list)

    def update(i):
        for b, c in enumerate(coords_list):
            xa, ya = c[px][:i + 1], c[py][:i + 1]
            trails[b].set_data(xa, ya)
            if is_3d:
                za = c[pz][:i + 1]
                trails[b].set_3d_properties(za)
                points[b].set_data([xa[-1]], [ya[-1]])
                points[b].set_3d_properties([za[-1]])
            else:
                points[b].set_data([xa[-1]], [ya[-1]])
        return trails + points

    anim = FuncAnimation(fig, update, frames=n_act, interval=interval, blit=False)
    plt.close(fig)
    return fig, anim


def animate_singularity(s_expr, vars_x, x0=0.0, xi0=5.0, tmax=4.0,
                        n_frames=100, projection=None, branches='all',
                        labels=None, interval=50, contours=True,
                        solution=None, quantity='abs', save_path=None):
    """
    Animate the propagation of singularities along bicharacteristic
    trajectories, projected onto a 2D phase-space plane.
    
    Internally calls `integrate_singularity` to obtain the trajectories,
    then draws a growing dashed trail plus a moving point per branch via
    the shared `_trail_animation` helper.
    
    Parameters
    ----------
    s_expr : sympy.Expr
        Principal symbol used to build the Hamiltonian(s).
    vars_x : list of sympy.Symbol
        Spatial variables.
    x0 : float or array_like, default=0.0
        Initial spatial position(s).
    xi0 : float or array_like, default=5.0
        Initial frequency (momentum) component(s).
    tmax : float, default=4.0
        Final integration time.
    n_frames : int, default=100
        Number of time samples used for the trajectory and the animation.
    projection : {'phase', 'position', 'frequency'} or None, default=None
        Which plane to draw. In 1D, defaults to 'phase' (x vs xi); in 2D,
        defaults to 'position' (x vs y) and ('frequency'/'phase' are not
        selectable in 2D -- the projection is always (x, y)).
    branches : 'all', int, or list of int, default='all'
        Which characteristic branches to animate.
    labels : list of str, optional
        Currently unused (reserved for per-branch legend labels).
    interval : int, default=50
        Delay between animation frames, in milliseconds.
    contours : bool, default=True
        Currently unused (reserved for background contour overlays).
    solution : optional
        Currently unused (reserved for overlaying a PDE solution field).
    quantity : str, default='abs'
        Currently unused (reserved alongside `solution`).
    save_path : str, optional
        If given, the animation is saved to this path via `anim.save`.
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.
    
    Raises
    ------
    ValueError
        If `projection` is not one of the supported values for a 1D symbol.
    """
    from matplotlib.animation import FuncAnimation
    dim = len(vars_x)
    H_list, xs, xis, t_eval, trajs = integrate_singularity(
        s_expr, vars_x, x0=x0, xi0=xi0, tmax=tmax, n_frames=n_frames, branches=branches)
    n_act = min(Y.shape[1] for Y in trajs)
    t_eval = t_eval[:n_act]
    trajs = [Y[:, :n_act] for Y in trajs]

    names = [v.name for v in xs] + [v.name for v in xis]
    coords = [{**dict(zip(names, Y)), 't': t_eval} for Y in trajs]

    # FIX: `projection` was never defaulted when None, so in 1D the
    # fallback branch picked py='y' -- a key that doesn't exist in 1D
    # coords (only 'x', 'xi', 't') -- raising a KeyError as soon as the
    # axis limits are computed from the data. Default explicitly, as
    # documented ('phase' in 1D, 'position' in 2D), and support the same
    # projections as the class method (position/frequency plotted on the
    # diagonal in 1D, since there's only one spatial/frequency axis).
    if projection is None:
        projection = 'phase' if dim == 1 else 'position'
    if dim == 1:
        px, py = {'phase': ('x', 'xi'), 'position': ('x', 'x'),
                  'frequency': ('xi', 'xi')}.get(projection, (None, None))
        if px is None:
            raise ValueError(
                f"Invalid projection {projection!r} for a 1D symbol; "
                "use 'phase', 'position', or 'frequency'."
            )
    else:
        px, py = 'x', 'y'

    colors = [plt.cm.tab10.colors[b % 10] for b in range(len(H_list))]
    _, anim = _trail_animation(coords, colors, px, py, interval=interval)
    if save_path:
        anim.save(save_path)
    return anim


def animate_singularity_3d(s_expr, vars_x, x0=0.0, xi0=5.0, tmax=4.0,
                           n_frames=100, projection=None, branches='all',
                           labels=None, interval=50, save_path=None):
    """
    Animate bicharacteristic trajectories in a 3D matplotlib plot.
    
    Internally calls `integrate_singularity` to obtain the trajectories,
    then draws a growing dashed trail plus a moving point per branch via
    the shared `_trail_animation` helper. In 1D, the third axis is time
    `t`; in 2D (or higher), the first three phase-space coordinates
    `(x, y, ...)` are used directly.
    
    Parameters
    ----------
    s_expr : sympy.Expr
        Principal symbol used to build the Hamiltonian(s).
    vars_x : list of sympy.Symbol
        Spatial variables.
    x0 : float or array_like, default=0.0
        Initial spatial position(s).
    xi0 : float or array_like, default=5.0
        Initial frequency (momentum) component(s).
    tmax : float, default=4.0
        Final integration time.
    n_frames : int, default=100
        Number of time samples used for the trajectory and the animation.
    projection : optional
        Currently unused for the 3D case (reserved for API parity with
        `animate_singularity`); the axes are always chosen as described
        above.
    branches : 'all', int, or list of int, default='all'
        Which characteristic branches to animate.
    labels : list of str, optional
        Currently unused (reserved for per-branch legend labels).
    interval : int, default=50
        Delay between animation frames, in milliseconds.
    save_path : str, optional
        If given, the animation is saved to this path via `anim.save`.
    
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.
    """
    from matplotlib.animation import FuncAnimation
    import mpl_toolkits.mplot3d  # noqa: F401

    H_list, xs, xis, t_eval, trajs = integrate_singularity(
        s_expr, vars_x, x0=x0, xi0=xi0, tmax=tmax, n_frames=n_frames, branches=branches)
    n_act = min(Y.shape[1] for Y in trajs)
    t_eval, trajs = t_eval[:n_act], [Y[:, :n_act] for Y in trajs]

    names = [v.name for v in xs] + [v.name for v in xis]
    coords = [{**dict(zip(names, Y)), 't': t_eval} for Y in trajs]
    px, py, pz = (names[0], names[1], 't') if len(vars_x) == 1 else (names[0], names[1], names[2])

    colors = [plt.cm.tab10.colors[b % 10] for b in range(len(H_list))]
    _, anim = _trail_animation(coords, colors, px, py, pz=pz, interval=interval)
    if save_path:
        anim.save(save_path)
    return anim