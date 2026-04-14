"""
pde_data_generator.py
=====================
Generates (u_in, u_out) training pairs for pseudo-differential operator
identification.  All solvers are pseudo-spectral (FFT-based) with a
4th-order Runge-Kutta time integrator.  Only NumPy / SciPy are required.

The symbol is assumed **time-independent** throughout.  Each generator
maps an initial condition u_in = u(·, 0) to u_out = u(·, T) for a fixed
final time T.  This makes (u_in, u_out) a pair for the **solution
operator** G_T, whose symbol is identified by the estimator.

For small T, the solution operator is close to  I + T * L  where L is
the PDE generator, so its symbol is close to  1 + T * a_L(x, xi).
The generators therefore also return `true_symbol_func`, the symbol of
the PDE **generator** L (not of G_T), which is what one usually wants
after dividing out the identity part.

Supported equations
-------------------
1D : Advection, Heat, Schrodinger
2D : Advection, Heat

All grids are uniform and periodic.  Dealiasing uses the standard 2/3
rule (modes above N/3 are zeroed before each physical-space evaluation).

Dependencies: numpy, scipy.fft
"""

import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2


# ===========================================================================
#  Random initial conditions
# ===========================================================================

def _random_ic_1d(x_grid, rng, n_modes_range=(3, 12), amp_range=(0.5, 1.5)):
    """
    Smooth random 1D initial condition: superposition of low-frequency
    sinusoids with random amplitudes and phases.

    Parameters
    ----------
    x_grid        : (Nx,) periodic spatial grid
    rng           : numpy.random.Generator
    n_modes_range : (min, max) number of Fourier modes
    amp_range     : (min, max) amplitude per mode

    Returns
    -------
    u0 : (Nx,) real array
    """
    Nx      = len(x_grid)
    n_modes = int(rng.integers(*n_modes_range))
    freqs   = rng.integers(1, max(2, Nx // 8), size=n_modes)
    phases  = rng.uniform(0.0, 2 * np.pi, size=n_modes)
    ampls   = rng.uniform(*amp_range,       size=n_modes)
    return sum(a * np.sin(k * x_grid + p)
               for a, k, p in zip(ampls, freqs, phases))


def _random_ic_2d(x_grid, y_grid, rng,
                  n_modes_range=(2, 6), amp_range=(0.5, 1.5)):
    """
    Smooth random 2D initial condition.

    Returns
    -------
    u0 : (Nx, Ny) real array
    """
    Nx, Ny  = len(x_grid), len(y_grid)
    X, Y    = np.meshgrid(x_grid, y_grid, indexing='ij')
    n_modes = int(rng.integers(*n_modes_range))
    u0      = np.zeros((Nx, Ny))
    for _ in range(n_modes):
        kx  = int(rng.integers(1, max(2, Nx // 8)))
        ky  = int(rng.integers(1, max(2, Ny // 8)))
        phi = float(rng.uniform(0.0, 2 * np.pi))
        amp = float(rng.uniform(*amp_range))
        u0 += amp * np.sin(kx * X + ky * Y + phi)
    return u0


# ===========================================================================
#  Dealiasing masks  (2/3 rule)
# ===========================================================================

def _dealias_1d(Nx):
    """1D dealiasing mask: zeros modes above index N//3."""
    mask             = np.ones(Nx)
    cutoff           = Nx // 3
    mask[cutoff: Nx - cutoff] = 0.0
    return mask


def _dealias_2d(Nx, Ny):
    """2D dealiasing mask: outer product of two 1D masks."""
    return np.outer(_dealias_1d(Nx), _dealias_1d(Ny))


# ===========================================================================
#  Generic RK4 integrator (works for any shape via NumPy broadcasting)
# ===========================================================================

def _rk4_integrate(u0, rhs, T, dt):
    """
    Integrate u from t=0 to t=T using RK4 with fixed step dt.

    Parameters
    ----------
    u0  : ndarray of any shape
    rhs : callable  f(u) -> du/dt, same shape as u
    T   : float     final time
    dt  : float     time step

    Returns
    -------
    uT : ndarray  solution at t=T, same shape as u0
    """
    u       = u0.copy().astype(complex)
    n_steps = max(1, int(np.ceil(T / dt)))
    dt_eff  = T / n_steps
    for _ in range(n_steps):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt_eff * k1)
        k3 = rhs(u + 0.5 * dt_eff * k2)
        k4 = rhs(u +       dt_eff * k3)
        u  = u + (dt_eff / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


# ===========================================================================
#  1D — Advection   du/dt + c(x) du/dx = 0
# ===========================================================================

def generate_advection_1d(
    N_samples    = 500,
    Nx           = 128,
    T            = 0.5,
    L            = 2 * np.pi,
    c_func       = None,
    cfl          = 0.4,
    seed         = 42,
):
    """
    Training pairs for the 1D variable-coefficient advection equation:

        du/dt + c(x) du/dx = 0,   x in [0, L) periodic

    The PDE generator is  L u = -c(x) du/dx,
    with KN symbol  a(x, xi) = -c(x) * i * xi.

    Parameters
    ----------
    N_samples : int
    Nx        : int     spatial resolution
    T         : float   integration time (keep small for near-linear regime)
    L         : float   domain length
    c_func    : callable  c(x), default 1 + 0.5*sin(x)
    cfl       : float   CFL number for automatic time-step selection
    seed      : int

    Returns
    -------
    x_grid           : (Nx,)
    xi_grid          : (Nx,)   frequency grid
    U_in             : (N_samples, Nx)  initial conditions
    U_out            : (N_samples, Nx)  solutions at t=T
    true_symbol_func : callable  a(x, xi) = -c(x) * 1j * xi
    """
    rng  = np.random.default_rng(seed)
    x    = np.linspace(0.0, L, Nx, endpoint=False)
    dx   = x[1] - x[0]
    xi   = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    mask = _dealias_1d(Nx)

    if c_func is None:
        c_func = lambda xv: 1.0 + 0.5 * np.sin(xv)
    c_vals = np.asarray(c_func(x), dtype=float)

    dt = cfl * dx / (np.max(np.abs(c_vals)) + 1e-14)

    def rhs(u):
        dudx = np.real(ifft(1j * xi * fft(u) * mask))
        return -c_vals * dudx

    U_in  = np.zeros((N_samples, Nx))
    U_out = np.zeros((N_samples, Nx))
    for i in range(N_samples):
        u0        = _random_ic_1d(x, rng)
        U_in[i]   = u0
        U_out[i]  = np.real(_rk4_integrate(u0, rhs, T, dt))

    true_symbol_func = lambda xv, xiv: -c_func(xv) * 1j * xiv

    return x, xi, U_in, U_out, true_symbol_func


# ===========================================================================
#  1D — Heat   du/dt = kappa(x) d²u/dx²
# ===========================================================================

def generate_heat_1d(
    N_samples   = 500,
    Nx          = 128,
    T           = 0.1,
    L           = 2 * np.pi,
    kappa_func  = None,
    dt          = 5e-4,
    seed        = 42,
):
    """
    Training pairs for the 1D variable-coefficient heat equation:

        du/dt = kappa(x) d²u/dx²,   x in [0, L) periodic

    KN symbol of the generator:  a(x, xi) = -kappa(x) * xi².

    Parameters
    ----------
    kappa_func : callable  kappa(x) > 0, default 0.5 + 0.3*cos(x)
    dt         : float     time step (explicit scheme — keep small)

    Returns
    -------
    x_grid, xi_grid, U_in, U_out, true_symbol_func
    """
    rng  = np.random.default_rng(seed)
    x    = np.linspace(0.0, L, Nx, endpoint=False)
    dx   = x[1] - x[0]
    xi   = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    mask = _dealias_1d(Nx)

    if kappa_func is None:
        kappa_func = lambda xv: 0.5 + 0.3 * np.cos(xv)
    k_vals = np.asarray(kappa_func(x), dtype=float)

    def rhs(u):
        d2u = np.real(ifft(-xi ** 2 * fft(u) * mask))
        return k_vals * d2u

    U_in  = np.zeros((N_samples, Nx))
    U_out = np.zeros((N_samples, Nx))
    for i in range(N_samples):
        u0        = _random_ic_1d(x, rng)
        U_in[i]   = u0
        U_out[i]  = np.real(_rk4_integrate(u0, rhs, T, dt))

    true_symbol_func = lambda xv, xiv: -kappa_func(xv) * xiv ** 2

    return x, xi, U_in, U_out, true_symbol_func


# ===========================================================================
#  1D — Schrodinger   i du/dt = (-d²/dx² + V(x)) u
# ===========================================================================

def generate_schrodinger_1d(
    N_samples  = 500,
    Nx         = 128,
    T          = 0.2,
    L          = 2 * np.pi,
    V_func     = None,
    dt         = 1e-3,
    seed       = 42,
):
    """
    Training pairs for the 1D Schrodinger equation:

        i du/dt = (-d²/dx² + V(x)) u,   x in [0, L) periodic

    The Weyl symbol of the generator is the Hamiltonian:

        a_Weyl(x, xi) = xi² + V(x)

    Note: U_in and U_out are **complex-valued** (Nx,) arrays.

    Parameters
    ----------
    V_func : callable  V(x), default cos(x)

    Returns
    -------
    x_grid, xi_grid, U_in (complex), U_out (complex), true_symbol_func
        true_symbol_func returns the Weyl symbol a_W(x, xi) = xi² + V(x).
    """
    rng  = np.random.default_rng(seed)
    x    = np.linspace(0.0, L, Nx, endpoint=False)
    dx   = x[1] - x[0]
    xi   = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    mask = _dealias_1d(Nx)

    if V_func is None:
        V_func = lambda xv: np.cos(xv)
    V_vals = np.asarray(V_func(x), dtype=float)

    def rhs(u):
        d2u = ifft(-xi ** 2 * fft(u) * mask)
        return -1j * (-d2u + V_vals * u)

    U_in  = np.zeros((N_samples, Nx), dtype=complex)
    U_out = np.zeros((N_samples, Nx), dtype=complex)
    for i in range(N_samples):
        re       = _random_ic_1d(x, rng)
        im       = _random_ic_1d(x, rng)
        u0       = (re + 1j * im) / np.sqrt(2.0)
        U_in[i]  = u0
        U_out[i] = _rk4_integrate(u0, rhs, T, dt)

    # The Weyl symbol is the Hamiltonian H = xi² + V(x)
    true_symbol_func = lambda xv, xiv: xiv ** 2 + V_func(xv)

    return x, xi, U_in, U_out, true_symbol_func


# ===========================================================================
#  2D — Advection   du/dt + cx(x,y) du/dx + cy(x,y) du/dy = 0
# ===========================================================================

def generate_advection_2d(
    N_samples  = 200,
    Nx         = 64,
    Ny         = 64,
    T          = 0.3,
    Lx         = 2 * np.pi,
    Ly         = 2 * np.pi,
    cx_func    = None,
    cy_func    = None,
    cfl        = 0.4,
    seed       = 42,
):
    """
    Training pairs for the 2D variable-coefficient advection equation.

    KN symbol: a(x, y, xi, eta) = -cx(x,y)*i*xi - cy(x,y)*i*eta.

    Returns
    -------
    (x_grid, y_grid)   : tuple of (Nx,) and (Ny,) arrays
    (xi_grid, eta_grid): tuple of frequency arrays
    U_in               : (N_samples, Nx, Ny)
    U_out              : (N_samples, Nx, Ny)
    true_symbol_func   : callable  a(x, y, xi, eta)
    """
    rng  = np.random.default_rng(seed)
    x    = np.linspace(0.0, Lx, Nx, endpoint=False)
    y    = np.linspace(0.0, Ly, Ny, endpoint=False)
    dx   = x[1] - x[0]
    dy   = y[1] - y[0]
    xi   = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    eta  = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    XI, ETA = np.meshgrid(xi, eta, indexing='ij')
    mask    = _dealias_2d(Nx, Ny)
    X, Y    = np.meshgrid(x,  y,   indexing='ij')

    if cx_func is None:
        cx_func = lambda xv, yv: 1.0 + 0.3 * np.sin(xv)
    if cy_func is None:
        cy_func = lambda xv, yv: 0.5 + 0.3 * np.cos(yv)

    cx_vals = np.asarray(cx_func(X, Y), dtype=float)
    cy_vals = np.asarray(cy_func(X, Y), dtype=float)
    c_max   = np.max(np.sqrt(cx_vals ** 2 + cy_vals ** 2))
    dt      = cfl * min(dx, dy) / (c_max + 1e-14)

    def rhs(u):
        u_hat = fft2(u)
        dudx  = np.real(ifft2(1j * XI  * u_hat * mask))
        dudy  = np.real(ifft2(1j * ETA * u_hat * mask))
        return -(cx_vals * dudx + cy_vals * dudy)

    U_in  = np.zeros((N_samples, Nx, Ny))
    U_out = np.zeros((N_samples, Nx, Ny))
    for i in range(N_samples):
        u0        = _random_ic_2d(x, y, rng)
        U_in[i]   = u0
        U_out[i]  = np.real(_rk4_integrate(u0, rhs, T, dt))

    def true_symbol_func(xv, yv, xiv, etav):
        return -cx_func(xv, yv) * 1j * xiv - cy_func(xv, yv) * 1j * etav

    return (x, y), (xi, eta), U_in, U_out, true_symbol_func


# ===========================================================================
#  2D — Heat   du/dt = kappa(x,y) * Laplacian(u)
# ===========================================================================

def generate_heat_2d(
    N_samples  = 200,
    Nx         = 64,
    Ny         = 64,
    T          = 0.05,
    Lx         = 2 * np.pi,
    Ly         = 2 * np.pi,
    kappa_func = None,
    dt         = 1e-4,
    seed       = 42,
):
    """
    Training pairs for the 2D variable-coefficient heat equation:

        du/dt = kappa(x,y) * (d²u/dx² + d²u/dy²)

    KN symbol: a(x, y, xi, eta) = -kappa(x,y) * (xi² + eta²).

    Returns
    -------
    (x_grid, y_grid), (xi_grid, eta_grid), U_in, U_out, true_symbol_func
    """
    rng  = np.random.default_rng(seed)
    x    = np.linspace(0.0, Lx, Nx, endpoint=False)
    y    = np.linspace(0.0, Ly, Ny, endpoint=False)
    dx   = x[1] - x[0]
    dy   = y[1] - y[0]
    xi   = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    eta  = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    XI, ETA = np.meshgrid(xi, eta, indexing='ij')
    mask    = _dealias_2d(Nx, Ny)
    X, Y    = np.meshgrid(x,  y,   indexing='ij')

    if kappa_func is None:
        kappa_func = lambda xv, yv: 0.5 + 0.3 * np.cos(xv) * np.sin(yv)
    k_vals = np.asarray(kappa_func(X, Y), dtype=float)

    def rhs(u):
        lap = np.real(ifft2(-(XI ** 2 + ETA ** 2) * fft2(u) * mask))
        return k_vals * lap

    U_in  = np.zeros((N_samples, Nx, Ny))
    U_out = np.zeros((N_samples, Nx, Ny))
    for i in range(N_samples):
        u0        = _random_ic_2d(x, y, rng)
        U_in[i]   = u0
        U_out[i]  = np.real(_rk4_integrate(u0, rhs, T, dt))

    def true_symbol_func(xv, yv, xiv, etav):
        return -kappa_func(xv, yv) * (xiv ** 2 + etav ** 2)

    return (x, y), (xi, eta), U_in, U_out, true_symbol_func

import numpy as np
from sympy import lambdify, Symbol
from scipy.fft import fft, ifft

def generate_from_symbol(
    a_expr,              # SymPy expression in x and xi
    N_samples=100,
    Nx=128,
    L=2*np.pi,
    seed=42,
    return_numeric_symbol=False,
):
    """
    Generate (u_in, u_out) pairs for the pseudo-differential operator
    with KN symbol a(x, xi).

    Parameters
    ----------
    a_expr : SymPy expression
        Symbol a(x, xi). Must contain symbols 'x' and 'xi'.
    N_samples : int
        Number of training pairs.
    Nx : int
        Spatial grid resolution.
    L : float
        Domain length (periodic).
    seed : int
        Random seed.
    return_numeric_symbol : bool
        If True, also return the numeric symbol matrix (Nx, Nx).

    Returns
    -------
    x_grid : (Nx,)
    xi_grid : (Nx,)
    U_in : (N_samples, Nx) real
    U_out : (N_samples, Nx) real
    true_symbol_func : callable a(x, xi) (vectorised)
    (optional) symbol_matrix : (Nx, Nx) complex
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, Nx, endpoint=False)
    dx = x[1] - x[0]
    xi = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi

    # Vectorised numeric function for the symbol
    x_sym = Symbol('x')
    xi_sym = Symbol('xi')
    a_numeric = lambdify((x_sym, xi_sym), a_expr, 'numpy')

    # Precompute symbol matrix on the grid (Nx, Nx)
    X, XI = np.meshgrid(x, xi, indexing='ij')
    symbol_matrix = a_numeric(X, XI).astype(complex)

    # Generate random initial conditions
    def random_ic():
        # smooth random combination of low-frequency sinusoids
        n_modes = rng.integers(3, 12)
        freqs = rng.integers(1, Nx//8, size=n_modes)
        phases = rng.uniform(0, 2*np.pi, size=n_modes)
        amps = rng.uniform(0.5, 1.5, size=n_modes)
        u = np.zeros(Nx)
        for a, k, p in zip(amps, freqs, phases):
            u += a * np.sin(k * x + p)
        return u

    U_in = np.array([random_ic() for _ in range(N_samples)])
    U_out = np.zeros_like(U_in)

    # Apply operator: u_out(x) = (1/2π) ∫ a(x, ξ) û(ξ) e^{i x ξ} dξ
    for i in range(N_samples):
        u_hat = fft(U_in[i])
        u_out = np.zeros(Nx, dtype=complex)
        for j, xj in enumerate(x):
            integrand = symbol_matrix[j, :] * u_hat * np.exp(1j * xi * xj)
            u_out[j] = np.sum(integrand) * (xi[1] - xi[0]) / (2 * np.pi)
        U_out[i] = np.real(u_out)

    true_symbol_func = a_numeric

    if return_numeric_symbol:
        return x, xi, U_in, U_out, true_symbol_func, symbol_matrix
    return x, xi, U_in, U_out, true_symbol_func