# microlocal.py
# Copyright 2025 Philippe Billet assisted by LLMs in free mode: chatGPT, Qwen, Gemini, Claude, le chat Mistral.
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
Microlocal analysis toolkit for 1D and 2D problems (unified).

This module provides tools for studying singularities and wave propagation:
 - Characteristic variety Char(P) = {(x,ξ) : p(x,ξ)=0} or {(x,y,ξ,η) : p=0}
 - Bicharacteristic flows on the cotangent bundle
 - WKB approximation: eikonal and transport equations (multiple orders)
 - Bohr–Sommerfeld quantisation (1D)
 - Propagation of singularities
 - Caustics and turning points
 - Maslov index and wave fronts (2D)

All functions automatically detect the dimension of the problem from the
symbol or from the supplied initial data.

WKB‑related functionality (wkb_approximation, visualisation, initial‑data
generators) is imported from the dedicated `wkb` module to keep the code
DRY and maintainable.
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
        Contains:
        - 'implicit': the symbol expression
        - 'equation': sympy Eq(symbol,0)
        - 'explicit': list of explicit solutions ξ(x) (1D only) or None
        - 'function': a callable that evaluates the symbol
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
    x, xi = sp.symbols('x xi', real=True)
    dp_dxi = sp.diff(symbol, xi)
    dp_dx  = sp.diff(symbol, x)
    f_x   = sp.lambdify((x, xi), dp_dxi, 'numpy')
    f_xi  = sp.lambdify((x, xi), -dp_dx, 'numpy')
    p_func = sp.lambdify((x, xi), symbol, 'numpy')

    if method == 'rk45':
        def ode(t, z):
            xv, xiv = z
            return [f_x(xv, xiv), f_xi(xv, xiv)]
        sol = solve_ivp(ode, tspan, z0, method='RK45',
                        t_eval=np.linspace(tspan[0], tspan[1], n_steps),
                        rtol=1e-9, atol=1e-12)
        return {
            't': sol.t,
            'x': sol.y[0],
            'xi': sol.y[1],
            'symbol_value': p_func(sol.y[0], sol.y[1])
        }
    elif method in ('hamiltonian', 'symplectic'):
        dt = (tspan[1]-tspan[0])/n_steps
        t = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        xi_vals = np.zeros(n_steps)
        x_vals[0], xi_vals[0] = z0
        for i in range(n_steps-1):
            # symplectic Euler
            xi_new = xi_vals[i] + dt * f_xi(x_vals[i], xi_vals[i])
            x_new  = x_vals[i]  + dt * f_x(x_vals[i], xi_new)
            x_vals[i+1] = x_new
            xi_vals[i+1] = xi_new
        return {
            't': t,
            'x': x_vals,
            'xi': xi_vals,
            'symbol_value': p_func(x_vals, xi_vals)
        }
    else:
        raise ValueError("Invalid method for 1D flow")

def _bichar_flow_2d(symbol, z0, tspan, method, n_steps):
    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    dp_dxi  = sp.diff(symbol, xi)
    dp_deta = sp.diff(symbol, eta)
    dp_dx   = sp.diff(symbol, x)
    dp_dy   = sp.diff(symbol, y)
    f_x   = sp.lambdify((x,y,xi,eta), dp_dxi, 'numpy')
    f_y   = sp.lambdify((x,y,xi,eta), dp_deta, 'numpy')
    f_xi  = sp.lambdify((x,y,xi,eta), -dp_dx, 'numpy')
    f_eta = sp.lambdify((x,y,xi,eta), -dp_dy, 'numpy')
    p_func = sp.lambdify((x,y,xi,eta), symbol, 'numpy')

    if method == 'rk45':
        def ode(t, z):
            xv, yv, xiv, etav = z
            return [f_x(xv,yv,xiv,etav),
                    f_y(xv,yv,xiv,etav),
                    f_xi(xv,yv,xiv,etav),
                    f_eta(xv,yv,xiv,etav)]
        sol = solve_ivp(ode, tspan, z0, method='RK45',
                        t_eval=np.linspace(tspan[0], tspan[1], n_steps),
                        rtol=1e-9, atol=1e-12)
        return {
            't': sol.t,
            'x': sol.y[0],
            'y': sol.y[1],
            'xi': sol.y[2],
            'eta': sol.y[3],
            'symbol_value': p_func(sol.y[0], sol.y[1], sol.y[2], sol.y[3])
        }
    elif method in ('symplectic', 'verlet'):
        dt = (tspan[1]-tspan[0])/n_steps
        t = np.linspace(tspan[0], tspan[1], n_steps)
        x_vals = np.zeros(n_steps)
        y_vals = np.zeros(n_steps)
        xi_vals = np.zeros(n_steps)
        eta_vals = np.zeros(n_steps)
        x_vals[0], y_vals[0], xi_vals[0], eta_vals[0] = z0

        if method == 'symplectic':
            for i in range(n_steps-1):
                xi_new = xi_vals[i] + dt * f_xi(x_vals[i], y_vals[i], xi_vals[i], eta_vals[i])
                eta_new = eta_vals[i] + dt * f_eta(x_vals[i], y_vals[i], xi_vals[i], eta_vals[i])
                x_new = x_vals[i] + dt * f_x(x_vals[i], y_vals[i], xi_new, eta_new)
                y_new = y_vals[i] + dt * f_y(x_vals[i], y_vals[i], xi_new, eta_new)
                x_vals[i+1] = x_new
                y_vals[i+1] = y_new
                xi_vals[i+1] = xi_new
                eta_vals[i+1] = eta_new
        else:  # verlet
            for i in range(n_steps-1):
                xi_half  = xi_vals[i]  + 0.5*dt * f_xi(x_vals[i], y_vals[i], xi_vals[i], eta_vals[i])
                eta_half = eta_vals[i] + 0.5*dt * f_eta(x_vals[i], y_vals[i], xi_vals[i], eta_vals[i])
                x_new = x_vals[i] + dt * f_x(x_vals[i], y_vals[i], xi_half, eta_half)
                y_new = y_vals[i] + dt * f_y(x_vals[i], y_vals[i], xi_half, eta_half)
                xi_new  = xi_half  + 0.5*dt * f_xi(x_new, y_new, xi_half, eta_half)
                eta_new = eta_half + 0.5*dt * f_eta(x_new, y_new, xi_half, eta_half)
                x_vals[i+1] = x_new
                y_vals[i+1] = y_new
                xi_vals[i+1] = xi_new
                eta_vals[i+1] = eta_new

        return {
            't': t,
            'x': x_vals,
            'y': y_vals,
            'xi': xi_vals,
            'eta': eta_vals,
            'symbol_value': p_func(x_vals, y_vals, xi_vals, eta_vals)
        }
    else:
        raise ValueError("Invalid method for 2D flow")

# ----------------------------------------------------------------------
# 1D‑specific functions (Bohr–Sommerfeld, caustic detection)
# ----------------------------------------------------------------------
def bohr_sommerfeld_quantization(H, n_max=10, x_range=(-10,10),
                                 hbar=1.0, method='fast'):
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
    E_scan = np.linspace(1e-6, 50, 200)
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


def compute_maslov_index(traj, p=None):
    """
    Compute the Maslov index for a single trajectory.
    traj : dict returned by bicharacteristic_flow (must contain J11..J22 or J)
    p    : unused (kept for compatibility)
    """
    dim = 2 if 'J22' in traj else 1
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

# ----------------------------------------------------------------------
# Tests (unchanged, now using imported WKB functions)
# ----------------------------------------------------------------------
def test_characteristic_variety():
    x, xi = sp.symbols('x xi', real=True)
    p = xi**2 - 1
    char = characteristic_variety(p)
    assert char['explicit'] is not None and len(char['explicit']) == 2
    print("✓ characteristic_variety (1D)")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p2 = xi**2 + eta**2 - 1
    char2 = characteristic_variety(p2)
    assert char2['explicit'] is None
    assert np.isclose(char2['function'](0,0,1,0), 0)
    print("✓ characteristic_variety (2D)")

def test_bicharacteristic_flow():
    x, xi = sp.symbols('x xi', real=True)
    p = xi
    traj = bicharacteristic_flow(p, (0,1), (0,5), dim=1)
    assert np.std(traj['xi']) < 1e-6
    assert np.allclose(traj['x'], traj['t'], rtol=1e-2)
    print("✓ bicharacteristic_flow (1D)")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p2 = xi + eta
    traj2 = bicharacteristic_flow(p2, (0,0,1,1), (0,5), dim=2)
    assert np.std(traj2['xi']) < 1e-6 and np.std(traj2['eta']) < 1e-6
    assert np.allclose(traj2['x'], traj2['t'], rtol=1e-2)
    assert np.allclose(traj2['y'], traj2['t'], rtol=1e-2)
    print("✓ bicharacteristic_flow (2D)")

def test_wkb():
    x, xi = sp.symbols('x xi', real=True)
    p = xi**2 - (1 + 0.1*x)**2
    ic = {'x': [0.0], 'S': [0.0], 'p_x': [1.0], 'a': {0:[1.0]}}
    sol = wkb_approximation(p, ic, order=2, domain=(-5,5), epsilon=0.1)
    assert sol['dimension'] == 1
    assert sol['u'] is not None
    print("✓ wkb_approximation (1D)")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p2 = xi**2 + eta**2 - 1
    ic2 = create_initial_data_circle(radius=1.0, n_points=20, outward=True)
    sol2 = wkb_approximation(p2, ic2, order=1, domain=((-3,3),(-3,3)),
                              resolution=50, epsilon=0.1)
    assert sol2['dimension'] == 2
    print("✓ wkb_approximation (2D)")

def test_bohr_sommerfeld():
    x, p = sp.symbols('x p', real=True)
    H = p**2/2 + x**2/2
    quant = bohr_sommerfeld_quantization(H, n_max=5)
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.allclose(quant['E_n'], expected, rtol=1e-2)
    print("✓ bohr_sommerfeld_quantization")

if __name__ == "__main__":
    print("Running unified microlocal tests...\n")
    test_characteristic_variety()
    test_bicharacteristic_flow()
    test_wkb()
    test_bohr_sommerfeld()
    print("\n✓ All tests passed")

# ----------------------------------------------------------------------
# Testing and examples
# ----------------------------------------------------------------------
def test_characteristic_variety():
    x, xi = sp.symbols('x xi', real=True)
    p = xi**2 - 1
    char = characteristic_variety(p)
    assert char['explicit'] is not None and len(char['explicit']) == 2
    print("✓ characteristic_variety (1D)")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p2 = xi**2 + eta**2 - 1
    char2 = characteristic_variety(p2)
    assert char2['explicit'] is None
    assert np.isclose(char2['function'](0,0,1,0), 0)
    print("✓ characteristic_variety (2D)")

def test_bicharacteristic_flow():
    x, xi = sp.symbols('x xi', real=True)
    p = xi
    traj = bicharacteristic_flow(p, (0,1), (0,5), dim=1)
    assert np.std(traj['xi']) < 1e-6
    assert np.allclose(traj['x'], traj['t'], rtol=1e-2)
    print("✓ bicharacteristic_flow (1D)")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p2 = xi + eta
    traj2 = bicharacteristic_flow(p2, (0,0,1,1), (0,5), dim=2)
    assert np.std(traj2['xi']) < 1e-6 and np.std(traj2['eta']) < 1e-6
    assert np.allclose(traj2['x'], traj2['t'], rtol=1e-2)
    assert np.allclose(traj2['y'], traj2['t'], rtol=1e-2)
    print("✓ bicharacteristic_flow (2D)")

def test_wkb():
    x, xi = sp.symbols('x xi', real=True)
    p = xi**2 - (1 + 0.1*x)**2
    ic = {'x': [0.0], 'S': [0.0], 'p_x': [1.0], 'a': {0:[1.0]}}
    sol = wkb_approximation(p, ic, order=2, domain=(-5,5), epsilon=0.1)
    assert sol['dimension'] == 1
    assert sol['u'] is not None
    print("✓ wkb_approximation (1D)")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p2 = xi**2 + eta**2 - 1
    # Use a circular source to avoid collinear rays
    ic2 = create_initial_data_circle(radius=1.0, n_points=20, outward=True)
    sol2 = wkb_approximation(p2, ic2, order=1, domain=((-3,3),(-3,3)),
                              resolution=50, epsilon=0.1)
    assert sol2['dimension'] == 2
    print("✓ wkb_approximation (2D)")

def test_bohr_sommerfeld():
    x, p = sp.symbols('x p', real=True)
    H = p**2/2 + x**2/2
    quant = bohr_sommerfeld_quantization(H, n_max=5)
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.allclose(quant['E_n'], expected, rtol=1e-2)
    print("✓ bohr_sommerfeld_quantization")

if __name__ == "__main__":
    print("Running unified microlocal tests...\n")
    test_characteristic_variety()
    test_bicharacteristic_flow()
    test_wkb()
    test_bohr_sommerfeld()
    print("\n✓ All tests passed")

    # ... (after all function definitions, before the tests)
    print("Running unified microlocal tests with visualisation...\n")

    # Test 1: characteristic variety
    print("\nTest: characteristic_variety")
    x, xi = sp.symbols('x xi', real=True)
    p1 = xi**2 - 1
    char1 = characteristic_variety(p1)
    assert char1['explicit'] is not None and len(char1['explicit']) == 2
    plot_characteristic_set(p1, x_range=(-2,2), xi_range=(-2,2), dim=1)
    print("✓ characteristic_variety (1D) – plot shown")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p2 = xi**2 + eta**2 - 1
    char2 = characteristic_variety(p2)
    assert np.isclose(char2['function'](0,0,1,0), 0)
    plot_characteristic_set(p2, x_range=(-2,2), xi_range=(-2,2), dim=2, xi0=1.0, eta0=0.0)
    print("✓ characteristic_variety (2D) – plot shown")

    # Test 2: bicharacteristic flow
    print("\nTest: bicharacteristic_flow")
    p_flow1 = xi
    traj1 = bicharacteristic_flow(p_flow1, (0,1), (0,5), dim=1)
    assert np.std(traj1['xi']) < 1e-6
    assert np.allclose(traj1['x'], traj1['t'], rtol=1e-2)
    plot_bicharacteristics(p_flow1, [(0,1)], (0,5), dim=1)
    print("✓ bicharacteristic_flow (1D) – plot shown")

    p_flow2 = xi + eta
    traj2 = bicharacteristic_flow(p_flow2, (0,0,1,1), (0,5), dim=2)
    assert np.std(traj2['xi']) < 1e-6 and np.std(traj2['eta']) < 1e-6
    assert np.allclose(traj2['x'], traj2['t'], rtol=1e-2)
    assert np.allclose(traj2['y'], traj2['t'], rtol=1e-2)
    plot_bicharacteristics(p_flow2, [(0,0,1,1), (0,0,1,0)], (0,5), dim=2, projection='position')
    print("✓ bicharacteristic_flow (2D) – plot shown")

    # Test 3: WKB approximation
    print("\nTest: wkb_approximation")
    x, xi = sp.symbols('x xi', real=True)
    p_wkb1 = xi**2 - (1 + 0.1*x)**2
    ic1 = {'x': [0.0], 'S': [0.0], 'p_x': [1.0], 'a': {0:[1.0]}}
    sol1 = wkb_approximation(p_wkb1, ic1, order=2, domain=(-5,5), epsilon=0.1)
    assert sol1['dimension'] == 1
    # Use plot_with_caustics instead of the missing plot_wkb_solution
    fig1 = plot_with_caustics(sol1, component='real', highlight_caustics=False)
    plt.show()  # ensure the plot is displayed
    print("✓ wkb_approximation (1D) – plot shown")

    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p_wkb2 = xi**2 + eta**2 - 1
    ic2 = create_initial_data_circle(radius=1.0, n_points=20, outward=True)
    sol2 = wkb_approximation(p_wkb2, ic2, order=1, domain=((-3,3),(-3,3)),
                              resolution=50, epsilon=0.1)
    assert sol2['dimension'] == 2
    fig2 = plot_with_caustics(sol2, component='abs', highlight_caustics=False)
    plt.show()
    print("✓ wkb_approximation (2D) – plot shown")

    # Test 4: Bohr–Sommerfeld quantisation
    print("\nTest: bohr_sommerfeld_quantization")
    x, p = sp.symbols('x p', real=True)
    H = p**2/2 + x**2/2
    quant = bohr_sommerfeld_quantization(H, n_max=5)
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.allclose(quant['E_n'], expected, rtol=1e-2)
    # Plot the quantised energies
    plt.figure()
    plt.plot(quant['n'], quant['E_n'], 'bo-', label='Bohr–Sommerfeld')
    plt.plot(quant['n'], expected, 'r--', label='Exact (n+½)')
    plt.xlabel('Quantum number n')
    plt.ylabel('Energy E_n')
    plt.title('Bohr–Sommerfeld quantisation for harmonic oscillator')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    print("✓ bohr_sommerfeld_quantization – plot shown")


    print("\nTest: 1D wavefront set")
    x, xi = sp.symbols('x xi', real=True)
    p = xi**2 - (1 - x**2)
    seeds = [(x0, 1.0) for x0 in np.linspace(-1, 1, 12)]
    
    fig, ax = plot_wavefront_set(p, seeds, tspan=(0, 4), dim=1,
                                  projection='cotangent')   # (x, ξ) portrait
    plt.show()
    print("✓ 1D plot_wavefront_set – plot shown")

    print("\nTest: 2D wavefront set")
    x, y, xi, eta = sp.symbols('x y xi eta', real=True)
    p = xi**2 + eta**2 - 1
    seeds = [(np.cos(t), np.sin(t), np.cos(t), np.sin(t))
             for t in np.linspace(0, 2*np.pi, 24, endpoint=False)]
    
    fig, axes = plot_wavefront_set(p, seeds, tspan=(0, 2), dim=2,
                                    projection='full')   # 2×2 grid
    plt.show()
    print("✓ 2D plot_wavefront_set – plot shown")

    print("\n✓ All tests passed and plots displayed")