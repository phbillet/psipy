"""
symbol_estimator.py
===================
STFT-based numerical estimator for the symbol of a pseudo-differential
operator from input/output function pairs.

Given N_samples pairs (u_in^(i), u_out^(i)) satisfying

    u_out^(i)(x) = (1/2pi) int a(x, xi) u_in_hat^(i)(xi) e^{ix xi} dxi  [KN]

this module estimates a(x, xi) on a phase-space grid via the Gabor
(short-time Fourier transform) method.

The key approximation is that a(x, xi) is slowly varying on the spatial
support of the analysis window g.  Under this assumption:

    V_{u_out}(x0, xi) ~= a(x0, xi) * V_{u_in}(x0, xi)

where V_u(x0, xi) = int u(x) g(x - x0) e^{-ix xi} dx  is the Gabor
transform of u centred at x0.

Averaging over N_samples gives a robust least-squares estimate at each
phase-space point (x0_k, xi_j):

    a(x0_k, xi_j) =  sum_i  V_out^(i) * conj(V_in^(i))
                     ----------------------------------------
                     sum_i  |V_in^(i)|^2  +  epsilon * scale

Dependencies: numpy, scipy  (no external packages required)
"""

import numpy as np
from scipy.signal import get_window


# ===========================================================================
#  Window builder
# ===========================================================================

def _make_window(window_type, window_width, Nx, dx):
    """
    Build a normalised analysis window on the spatial grid.

    The window is zero-padded to length Nx and rolled so that its peak
    sits at index 0 (ready for the circular-shift / roll trick used in
    the Gabor batch).

    Parameters
    ----------
    window_type  : str
        'gaussian' uses window_width as sigma in physical units.
        Any other string is forwarded to scipy.signal.get_window with
        window_width converted to a sample count.
    window_width : float
        Characteristic width in physical (x) units.
    Nx           : int     number of spatial grid points
    dx           : float   spatial step

    Returns
    -------
    win : (Nx,) ndarray   window, unit L2-norm on the grid (i.e. ||win||_2 * sqrt(dx) = 1)
    """
    sigma_samples = max(1, int(window_width / dx))
    n_win = min(Nx, 6 * sigma_samples)   # truncate Gaussian at ±3 sigma
    if n_win % 2 == 0:
        n_win += 1                        # keep odd so the peak is centred

    if window_type == 'gaussian':
        t    = np.arange(n_win) - n_win // 2
        core = np.exp(-0.5 * (t / sigma_samples) ** 2)
    else:
        core = get_window(window_type, n_win)

    # Zero-pad into a length-Nx array with the peak at the centre
    win           = np.zeros(Nx)
    start         = Nx // 2 - n_win // 2
    win[start: start + n_win] = core

    # Roll peak to index 0 so np.roll(win, k) centres the window at x_k
    win = np.roll(win, -(Nx // 2))

    # L2-normalise: integral approximated as sum * dx
    norm = np.sqrt(np.sum(win ** 2) * dx)
    if norm > 0:
        win /= norm
    return win


# ===========================================================================
#  Gabor batch  (memory-efficient, vectorised over samples)
# ===========================================================================

def _gabor_batch(U, win, dx):
    """
    Compute the Gabor transform of a batch of signals for all centre
    positions x0_k = k * dx simultaneously.

    Algorithm
    ---------
    For each centre k, the windowed signal is  u(x) * g(x - x0_k),
    which in discrete form is  U[i, :] * roll(win, k).
    Its DFT gives V[i, k, :].

    We loop over centres (one FFT per centre per sample batch) which is
    O(Nx) FFTs of length Nx — i.e. O(Nx^2 log Nx) total.  Memory is
    O(N_samples * Nx) per step, never O(N_samples * Nx^2).

    Parameters
    ----------
    U   : (N_samples, Nx) complex array   signals
    win : (Nx,) real array                window with peak at index 0
    dx  : float                           spatial step

    Returns
    -------
    V : (N_samples, Nx, Nx) complex array
        V[i, k, j] = Gabor transform of U[i] centred at x0_k, frequency xi_j.
    """
    N_samples, Nx = U.shape
    V = np.zeros((N_samples, Nx, Nx), dtype=complex)
    for k in range(Nx):
        win_k       = np.roll(win, k)           # window centred at x0_k
        windowed    = U * win_k[None, :]        # (N_samples, Nx)
        V[:, k, :] = np.fft.fft(windowed, axis=1) * dx
    return V


# ===========================================================================
#  Main estimator
# ===========================================================================

def estimate_symbol_stft(
    U_in,
    U_out,
    x_grid,
    window_type  = 'gaussian',
    window_width = None,
    epsilon      = 1e-6,
    regularize   = 'tikhonov',
    xi_max       = None,
):
    """
    Estimate the KN symbol a(x, xi) of a pseudo-differential operator
    from input/output sample pairs via the Gabor (STFT) method.

    Parameters
    ----------
    U_in  : (N_samples, Nx) array
        Input function samples on x_grid.  May be real or complex.
    U_out : (N_samples, Nx) array
        Output samples: Op^KN(a) applied to each row of U_in.
    x_grid : (Nx,) array
        Uniform spatial grid (periodic domain).
    window_type : str, default='gaussian'
        Analysis window.  'gaussian' gives optimal time-frequency
        localisation (minimum uncertainty).
    window_width : float or None
        Window width in physical units.
        Default: L / sqrt(Nx), which balances spatial and frequency
        resolution (geometric mean of the two extreme choices).
    epsilon : float, default=1e-6
        Regularisation strength relative to the median input energy.
        Prevents division by near-zero in low-energy phase-space regions.
    regularize : {'tikhonov', 'threshold'}, default='tikhonov'
        'tikhonov'  : denominator += epsilon * scale  (smooth, always safe)
        'threshold' : set estimate to 0 where input energy < epsilon * scale
    xi_max : float or None
        Discard frequencies |xi| > xi_max.  Useful to suppress
        high-frequency noise in the estimated symbol.

    Returns
    -------
    xi_grid       : (Nxi,) array    frequency grid in physical units
    symbol_matrix : (Nx, Nxi) complex array   estimated symbol a(x, xi)

    Notes
    -----
    The symbol is assumed **time-independent**.  For time-dependent
    operators, call this function on pairs (u(t_k), du/dt(t_k)) at
    each time snapshot.

    The estimate is exact for constant-coefficient symbols regardless
    of window width (Parseval / shift-invariance argument).
    For spatially varying symbols, accuracy improves with N_samples and
    with the spatial smoothness of a(x, xi).

    Examples
    --------
    >>> import numpy as np
    >>> from symbol_estimator import estimate_symbol_stft
    >>> Nx, Ns = 128, 300
    >>> x  = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    >>> dx = x[1] - x[0]
    >>> xi = np.fft.fftfreq(Nx, d=dx) * 2*np.pi
    >>> # True symbol: a(x, xi) = -(1 + 0.5*sin(x)) * 1j * xi
    >>> rng = np.random.default_rng(0)
    >>> U_in  = rng.standard_normal((Ns, Nx))
    >>> c     = 1 + 0.5 * np.sin(x)
    >>> U_out = np.real(np.fft.ifft(
    ...     1j * xi[None, :] * np.fft.fft(U_in, axis=1), axis=1
    ... )) * (-c[None, :])
    >>> xi_est, sym_est = estimate_symbol_stft(U_in, U_out, x)
    """
    U_in  = np.asarray(U_in,  dtype=complex)
    U_out = np.asarray(U_out, dtype=complex)

    if U_in.shape != U_out.shape:
        raise ValueError(
            f"U_in and U_out must have the same shape, "
            f"got {U_in.shape} vs {U_out.shape}"
        )

    N_samples, Nx = U_in.shape
    dx = float(x_grid[1] - x_grid[0])
    L  = Nx * dx

    if window_width is None:
        window_width = L / np.sqrt(Nx)

    win   = _make_window(window_type, window_width, Nx, dx)

    # Gabor transforms: (N_samples, Nx_centres, Nx_freqs)
    V_in  = _gabor_batch(U_in,  win, dx)
    V_out = _gabor_batch(U_out, win, dx)

    # Least-squares estimate: cross-correlation / auto-correlation
    numerator   = np.sum(V_out * np.conj(V_in), axis=0)  # (Nx, Nx)
    denominator = np.sum(np.abs(V_in) ** 2,     axis=0)  # (Nx, Nx)

    # Regularisation scale = median input energy (robust to outliers)
    pos_denom = denominator[denominator > 0]
    scale     = float(np.median(pos_denom)) if len(pos_denom) > 0 else 1.0

    if regularize == 'tikhonov':
        symbol_matrix = numerator / (denominator + epsilon * scale)

    elif regularize == 'threshold':
        safe_denom    = np.where(denominator >= epsilon * scale,
                                 denominator, 1.0)
        symbol_matrix = np.where(denominator >= epsilon * scale,
                                 numerator / safe_denom,
                                 0.0 + 0.0j)
    else:
        raise ValueError(
            f"Unknown regularize='{regularize}'. "
            "Choose 'tikhonov' or 'threshold'."
        )

    # Full frequency grid in physical units
    xi_grid = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi   # (Nx,)

    # Optional truncation to |xi| <= xi_max
    if xi_max is not None:
        keep          = np.abs(xi_grid) <= xi_max
        xi_grid       = xi_grid[keep]
        symbol_matrix = symbol_matrix[:, keep]

    return xi_grid, symbol_matrix


# ===========================================================================
#  Reconstruction quality diagnostics
# ===========================================================================

def symbol_estimation_diagnostics(U_in, U_out, U_out_reconstructed, x_grid):
    """
    Compute reconstruction quality metrics after symbol estimation.

    Parameters
    ----------
    U_in               : (N_samples, Nx)
    U_out              : (N_samples, Nx)   ground-truth output
    U_out_reconstructed: (N_samples, Nx)   output from estimated symbol
    x_grid             : (Nx,)

    Returns
    -------
    dict with keys:
        'relative_l2_error'    : mean relative L2 error over samples
        'relative_l2_std'      : std  of per-sample relative L2 errors
        'max_pointwise_error'  : max absolute pointwise error
        'snr_db'               : signal-to-noise ratio in dB
    """
    dx       = float(x_grid[1] - x_grid[0])
    err      = U_out - U_out_reconstructed

    norm_err = np.sqrt(np.sum(np.abs(err)    ** 2, axis=1) * dx)
    norm_sig = np.sqrt(np.sum(np.abs(U_out)  ** 2, axis=1) * dx)
    rel_err  = norm_err / (norm_sig + 1e-14)

    snr_db   = 20.0 * np.log10(
        np.mean(norm_sig) / (np.mean(norm_err) + 1e-14)
    )

    return {
        'relative_l2_error'   : float(np.mean(rel_err)),
        'relative_l2_std'     : float(np.std(rel_err)),
        'max_pointwise_error' : float(np.max(np.abs(err))),
        'snr_db'              : float(snr_db),
    }