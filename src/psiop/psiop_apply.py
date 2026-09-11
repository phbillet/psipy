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
psiop_apply.py — Numerical backends for pseudo‑differential operator application
================================================================================

Overview
--------
The ``psiop_apply`` module provides the heavy‑lifting numerical kernels for
applying pseudo‑differential operators (ΨDOs) to spatial fields. While the
core ``psiop`` module handles symbolic calculus, asymptotic expansions, and
the Peetre decomposition, this module is responsible for the actual
evaluation of the resulting integrals on discrete spatial and frequency
grids.

It implements both periodic (FFT‑based) and non‑periodic (direct
quadrature) Kohn–Nirenberg quantization, alongside specialized
factorization and approximation backends for the genuinely joint
space‑frequency residuals that arise in the Peetre decomposition.

Main objects and workflows
--------------------------
Periodic and Non‑Periodic Kohn–Nirenberg Application
    ``kohn_nirenberg_fft``: Applies the operator on a periodic domain using
    FFTs. Features an automatic fast‑path for spatially independent symbols
    (pure Fourier multipliers) and a memory‑bounded, multi‑threaded slow
    path for spatially dependent symbols.

    ``kohn_nirenberg_nonperiodic``: Applies the operator on a non‑periodic
    (Dirichlet‑type) domain via direct discrete Fourier quadrature.
    Utilizes aggressive caching of phase matrices and windowing arrays to
    minimize redundant computations across repeated calls.

Low‑Rank Chebyshev/SVD Factorization
    ``factorize_symbolic``: Approximates a joint space‑frequency symbol
    ``p(x, ξ)`` as a sum of separable terms ``∑ₖ aₖ(x) qₖ(ξ)`` using
    Chebyshev interpolation followed by Singular Value Decomposition (SVD)
    truncation. Includes Monte‑Carlo quality diagnostics.

NUFFT‑Based Joint‑Residual Backend
    ``try_nufft_decomposition_*`` / ``apply_nufft_*``: Targets joint
    residuals with genuinely oscillatory phases of the form
    ``exp(i · Λ(x) · M(ξ))``. Extracts the phase and amplitude, and
    evaluates the resulting non‑uniform FFT (Type 3) via the optional
    ``finufft`` library, with a pure‑NumPy O(N·M) direct‑sum fallback.

AAA Rational Approximation Backend
    ``try_aaa_decomposition_*`` / ``aaa_plan_to_callable_*``: Targets joint
    residuals that are rational functions or exhibit explicit poles /
    algebraic decay. Uses a vector‑valued Adaptive Antoulas‑Algorithm (AAA)
    barycentric rational interpolation to build a compact, fast‑evaluating
    surrogate for the symbol.

Key features
------------
Memory‑bounded execution:
    The 1D and 2D slow paths avoid O(N²ᵈ) RAM allocation by evaluating the
    symbol on chunked space‑frequency sub‑grids (~256 MB max per block) and
    accumulating via optimized Einstein summation (``np.einsum``).

Multi‑threaded row‑blocking:
    The 2D slow path distributes spatial row‑blocks across a
    ``ThreadPoolExecutor``, achieving near‑linear speedup on multi‑core
    machines for spatially dependent symbols.

Automatic fast‑path detection:
    Before executing the expensive quadrature, the periodic and
    non‑periodic kernels probe the symbol at a few test frequencies. If the
    symbol is spatially independent, the code bypasses the quadrature
    entirely and applies the symbol as a pure Fourier multiplier, reducing
    complexity from O(N²) to O(N log N) in 1D, and O(N⁴) to O(N² log N) in 2D.

Phase‑matrix caching:
    Non‑periodic transforms precalculate and cache discrete Fourier
    transform phases, reconstruction phases, and window arrays. The cache
    keys are derived from grid shapes and endpoints, ensuring automatic
    invalidation when grid resolution changes.

Quality‑gated approximations:
    The low‑rank, NUFFT, and AAA backends all compute relative L2 errors
    against the exact symbol. If the approximation error exceeds the
    requested tolerance, the backend gracefully falls back to the exact
    (but slower) direct Kohn–Nirenberg quadrature.

Mathematical background and numerical design
--------------------------------------------
Kohn–Nirenberg quantization (Periodic)
    The operator ``Op(p)`` is applied to a periodic function ``u`` via:

        [Op(p) u](x) = (2π)⁻ᵈ ∫ p(x, ξ) e^{i x·ξ} ℱ[u](ξ) dξ

    where ``ℱ[u]`` is the discrete Fourier transform. 
    Fast‑path: If ``p(x, ξ) = p(ξ)``, the ``x``‑dependence drops out, and
    the integral collapses to the pure multiplier:

        [Op(p) u](x) = ℱ⁻¹[ p(ξ) · ℱ[u](ξ) ]

    Slow‑path: For space‑dependent ``p(x, ξ)``, the integral is evaluated
    directly. To prevent memory exhaustion, the spatial domain is split
    into blocks of size ``B``, and the frequency domain into chunks of size
    ``C``. The quadrature is accumulated block‑by‑block:

        result[i₀:i₁] = (Δξ / 2π) ∑_{k‑chunk} P_{blk} · ℱ[u]_{chunk} · e^{i x_{blk} · ξ_{chunk}}

Kohn–Nirenberg quantization (Non‑Periodic)
    On a non‑periodic domain, the continuous Fourier transform is replaced
    by a direct discrete quadrature:

        [Op(p) u](x) = (2π)⁻ᵈ ∫ p(x, ξ) e^{i x·ξ} [ ∫ e^{-i y·ξ} u(y) dy ] dξ

    The inner integral (forward transform) and outer integral (reconstruction)
    are represented as dense matrix‑vector products using precomputed phase
    matrices ``Φ_{ft} = e^{-i ξ xᵀ}`` and ``Φ_{rec} = e^{i x ξᵀ}``. These
    matrices are cached globally. The fast‑path logic is identical to the
    periodic case, bypassing the matrix multiplications when ``p`` is
    independent of ``x``.

Peetre Joint Residual Factorization
    When the Peetre decomposition yields a genuinely joint residual
    ``p_joint(x, ξ)`` that cannot be written as ``a(x)q(ξ)``, it is routed
    to one of three specialized backends based on its algebraic structure:

    1. Low‑Rank (Chebyshev/SVD):
       For smooth, non‑oscillatory kernels. The symbol is interpolated on
       a tensor‑product Chebyshev grid, reshaped into a matrix
       ``C ∈ ℂ^{N_x × N_ξ}``, and truncated via SVD:

           C ≈ U_r Σ_r V_r^H   ⇒   p_joint(x, ξ) ≈ ∑_{k=1}^r aₖ(x) qₖ(ξ)

       The basis functions are explicit Chebyshev polynomials mapped to the
       physical bounding box.

    2. NUFFT (Oscillatory):
       For residuals containing a bilinear phase ``exp(i Λ(x) M(ξ))``.
       The symbol is factored as:

           p_joint(x, ξ) = c(x) g(ξ) exp(i Λ(x) M(ξ))

       The application is reformulated as a Type 3 Non‑Uniform FFT,
       evaluating the sum:

           f(x) = ∑_{j} w_j exp(i (x · Λ(x) + μ_j · M(ξ_j)))

       where ``w_j`` are the weighted Fourier coefficients of ``u``. This
       achieves O(N log N) complexity via ``finufft``, avoiding the
       polynomial basis convergence issues of the low‑rank method.

    3. AAA (Rational / Poles):
       For residuals with explicit poles or algebraic decay (e.g.,
       resolvent‑like structures). A vector‑valued AAA barycentric rational
       interpolant is constructed:

           r(ξ) = ∑_{k} wₖ fₖ / (ξ - zₖ)  /  ∑_{k} wₖ / (ξ - zₖ)

       where the support points ``zₖ`` and weights ``wₖ`` are selected
       adaptively to minimize the residual. The spatial dependence is
       handled by building a separate AAA fit for each Chebyshev node in
       ``x``, followed by barycentric Lagrange interpolation in ``x``.

Numerical stability
-------------------
All application kernels enforce numerical stability through:
    - Magnitude clamping: Symbol values exceeding ``clamp`` (default 10⁶)
      are scaled down while preserving their complex phase.
    - Frequency windowing: Optional Gaussian or Hann tapers in the
      frequency domain to attenuate high‑frequency numerical artifacts.
    - Spatial tapering: Optional centered Gaussian tapers in the spatial
      domain to mitigate edge boundary artifacts in non‑periodic settings.
"""
import numpy as np
import sympy as sp
import warnings
import itertools
from typing import Callable, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

# Import global constants (FFT_WORKERS is used in the chunking logic)
from imports import FFT_WORKERS 

# ============================================================================
# Standalone functions for Kohn-Nirenberg quantization
# ============================================================================


_KN_CACHE: Dict[Tuple, Dict[str, np.ndarray]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def _clip_complex_magnitude(P: np.ndarray, clamp: float) -> np.ndarray:
    """
    Clip a complex array by magnitude, preserving phase.
    Modifies the array in-place to minimize memory allocation.
    """
    if P.dtype != np.complex128:
        P = np.asarray(P, dtype=np.complex128)
    
    mag = np.abs(P)
    over = mag > clamp
    if np.any(over):
        # In-place multiplication preserves memory and phase
        P[over] *= (clamp / mag[over])
    return P


def _cache_key_1d(x: np.ndarray, xi: np.ndarray) -> Tuple:
    """
    Build a stable cache key from a 1D space/frequency grid pair.

    The key is derived from each grid's shape and endpoint values, so it
    automatically changes (invalidating any cached result) whenever the
    grid resolution or extent changes, without needing to hash the full
    array contents.

    Parameters
    ----------
    x : ndarray
        Spatial grid.
    xi : ndarray
        Frequency grid.

    Returns
    -------
    tuple
        `(x.shape, x[0], x[-1], xi.shape, xi[0], xi[-1])`, hashable and
        suitable as a dictionary cache key.
    """
    return (
        x.shape, float(x[0]), float(x[-1]),
        xi.shape, float(xi[0]), float(xi[-1]),
    )


def invalidate_kn_cache() -> None:
    """Clear the phase-matrix cache for non-periodic 1D operations."""
    _KN_CACHE.clear()


# ============================================================================
# Periodic Kohn-Nirenberg Quantization (FFT-based)
# ============================================================================

def kohn_nirenberg_fft(
    u_vals: np.ndarray,
    symbol_func: Callable[..., np.ndarray],
    x_grid: np.ndarray,
    kx: np.ndarray,
    fft_func: Callable,
    ifft_func: Callable,
    dim: int = 1,
    y_grid: Optional[np.ndarray] = None,
    ky: Optional[np.ndarray] = None,
    freq_window: Optional[str] = 'gaussian',
    clamp: float = 1e6,
    space_window: bool = False,
    is_spatial: bool = False,
) -> np.ndarray:
    """
    Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator
    on a periodic domain using FFTs.
    
    Applies the pseudo-differential operator Op(p) to the function u via the 
    Kohn–Nirenberg quantization:
        [Op(p) u](x) = (1/(2π)^d) ∫ p(x, ξ) e^{i x·ξ} ℱ[u](ξ) dξ,
    where ℱ[u] is the discrete Fourier transform of u, and p(x, ξ) is a symbol 
    that may depend on both spatial variables (x, y) and frequency variables (ξ, η).
    
    This implementation supports 1D and 2D spatial dimensions, providing numerical 
    stability through symbol magnitude clamping, optional frequency windowing 
    (Gaussian/Hann), and optional spatial tapering.
    
    **Fast-Path Optimization (Spatial Independence)**
        When the symbol `p` is independent of spatial coordinates (and `space_window=False`), 
        the operator simplifies to a pure Fourier multiplier: `ifft(P * fft(u))`.
        
        The function detects spatial independence using a multi-point evaluation 
        heuristic across non-zero frequencies (to avoid false positives at ξ = 0). 
        When active, computational complexity drops from O(N^2) to O(N log N) in 1D, 
        and from O(N^4) to O(N^2 log N) in 2D.
    
    **Memory-Bounded Slow Path (Spatial Dependence)**
        For space-dependent symbols, the implementation avoids O(N^2d) RAM allocation:
        - **1D**: Slices spatial evaluation into memory-bounded chunks (~256 MB max).
        - **2D**: Combines parallel multi-threaded row-blocking with 2D frequency chunking 
          and phased factorized inner products (`np.einsum`).
    
    Parameters
    ----------
    u_vals : ndarray
        Spatial samples of the input field u(x) or u(x, y).
    symbol_func : callable
        Symbol evaluator p(x, ξ) in 1D or p(x, y, ξ, η) in 2D. Must accept 
        NumPy-broadcastable positional arguments. Return arrays are automatically 
        broadcasted and cast to complex128 to handle scalar outputs safely (e.g., from `sympy.lambdify`).
    x_grid : ndarray
        1D array of spatial coordinates along the x-axis.
    kx : ndarray
        1D array of spatial frequencies in the x-direction.
    fft_func : callable
        Forward Fourier transform function (e.g., `scipy.fft.fft` or `fft2`).
    ifft_func : callable
        Inverse Fourier transform function (e.g., `scipy.fft.ifft` or `ifft2`).
    dim : {1, 2}, default=1
        Spatial dimensionality of the domain.
    y_grid : ndarray, optional
        1D array of spatial coordinates along the y-axis (required if `dim=2`).
    ky : ndarray, optional
        1D array of spatial frequencies in the y-direction (required if `dim=2`).
    freq_window : {'gaussian', 'hann', None}, default='gaussian'
        Frequency-domain window/taper applied to attenuate high-frequency instabilities.
    clamp : float, default=1e6
        Maximum allowed magnitude for the symbol entries. Magnitudes exceeding this 
        value are clipped to prevent overflow.
    space_window : bool, default=False
        If True, applies a centered Gaussian spatial taper to mitigate edge boundary artifacts. 
        *Note: Setting `space_window=True` forces execution through the slow path.*
    is_spatial : bool or None, default=None
        Explicit hint about whether the 2D symbol depends on (x1, x2).
        True forces the slow (space-dependent) path, False forces the fast
        (space-independent) path, None triggers the sampling heuristic.
    
    Returns
    -------
    ndarray
        Resulting array of the same shape and type (`complex128`) as `u_vals` after 
        applying the pseudo-differential operator.
    
    Raises
    ------
    ValueError
        If `dim=2` and `y_grid` or `ky` are not supplied, or if `dim` is not 1 or 2.
    """
    if dim == 1:
        dx = x_grid[1] - x_grid[0]
        Nx = len(x_grid)
        k_unshifted = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)

        # --- FAST/SLOW PATH SELECTION ---
        # is_spatial=True  -> force slow path (symbol IS space-dependent)
        # is_spatial=False -> force fast path (symbol is NOT space-dependent)
        # is_spatial=None  -> heuristic decides (always slow if space_window=True,
        #                     since the taper itself makes the integrand x-dependent)
        if is_spatial is True:
            is_x_independent = False
        elif is_spatial is False:
            is_x_independent = True
        else:
            # --- FAST PATH CHECK ---
            is_x_independent = False
            if not space_window:
                try:
                    non_zero_idx = np.where(k_unshifted != 0)[0]
                    if len(non_zero_idx) >= 2:
                        idx_test = non_zero_idx[[len(non_zero_idx)//4, len(non_zero_idx)//2]]
                        k_test = k_unshifted[idx_test]
                        x_test = x_grid[[0, Nx // 2]]
                        
                        val1 = symbol_func(x_test[:, None], k_test[None, :])
                        val2 = symbol_func((x_test + dx)[:, None], k_test[None, :])
                        is_x_independent = np.allclose(val1, val2)
                except Exception:
                    is_x_independent = False

        if is_x_independent:
            U = fft_func(u_vals)
            # FIX: Enforce shape to prevent scalar/reduced-dim returns from lambdify
            P_raw = symbol_func(x_grid[0], k_unshifted)
            P = np.broadcast_to(P_raw, k_unshifted.shape).astype(np.complex128).copy()
            P = _clip_complex_magnitude(P, clamp)
            
            if freq_window == 'gaussian':
                sigma = 0.8 * np.max(np.abs(k_unshifted))
                P *= np.exp(-(k_unshifted / sigma) ** 4)
            elif freq_window == 'hann':
                k_max = np.max(np.abs(k_unshifted))
                W = 0.5 * (1 + np.cos(np.pi * k_unshifted / k_max))
                P *= W * (np.abs(k_unshifted) < k_max)
                
            return ifft_func(P * U)

        # --- SLOW PATH (O(N) Memory-Bounded Integration) ---
        k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
        dk = k[1] - k[0]
        f_hat = np.fft.fftshift(fft_func(np.fft.fftshift(u_vals)) * dx)
        
        win_k = None
        if freq_window == 'gaussian':
            sigma = 0.8 * np.max(np.abs(k))
            win_k = np.exp(-(k / sigma) ** 4)
        elif freq_window == 'hann':
            k_max = np.max(np.abs(k))
            win_k = 0.5 * (1 + np.cos(np.pi * k / k_max)) * (np.abs(k) < k_max)
            
        sw_x = None
        if space_window:
            x0 = (x_grid[0] + x_grid[-1]) / 2
            L = (x_grid[-1] - x_grid[0]) / 2
            sw_x = np.exp(-((x_grid - x0) / L) ** 2)

        MAX_ELEMENTS = 16 * 1024 * 1024  # ~256 MB for complex128
        chunk_size = max(1, min(Nx, MAX_ELEMENTS // len(k)))
        result = np.zeros(Nx, dtype=np.complex128)

        for i0 in range(0, Nx, chunk_size):
            i1 = min(i0 + chunk_size, Nx)
            x_blk = x_grid[i0:i1]
            B = i1 - i0
            
            Xb = x_blk[:, None]
            Kb = k[None, :]
            
            # FIX: Enforce target shape (B, len(k)) before any in-place ops
            P_raw = symbol_func(Xb, Kb)
            P_blk = np.broadcast_to(P_raw, (B, len(k))).astype(np.complex128).copy()
            P_blk = _clip_complex_magnitude(P_blk, clamp)
            
            if win_k is not None:
                P_blk *= win_k[None, :]
            if sw_x is not None:
                P_blk *= sw_x[i0:i1, None]
                
            kernel_blk = np.exp(1j * Xb * Kb)
            
            result[i0:i1] = (dk / (2 * np.pi)) * np.einsum(
                'bk, k, bk -> b', P_blk, f_hat, kernel_blk, optimize=True
            )
        return result

    elif dim == 2:
        if y_grid is None or ky is None:
            raise ValueError("y_grid and ky are required for dim=2")
            
        dx = x_grid[1] - x_grid[0]
        dy = y_grid[1] - y_grid[0]
        Nx, Ny = len(x_grid), len(y_grid)
        
        kx_unshifted = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
        ky_unshifted = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)

        # --- FAST/SLOW PATH SELECTION ---
        # is_spatial=True  -> force slow path (symbol IS space-dependent)
        # is_spatial=False -> force fast path (symbol is NOT space-dependent)
        # is_spatial=None  -> heuristic decides (always slow if space_window=True,
        #                     since the taper itself makes the integrand x-dependent)
        if is_spatial is True:
            is_independent = False
        elif is_spatial is False:
            is_independent = True
        else:
        # --- FAST PATH CHECK ---
            is_independent = False
            if not space_window:
                try:
                    non_zero_kx = np.where(kx_unshifted != 0)[0]
                    non_zero_ky = np.where(ky_unshifted != 0)[0]
                    
                    if len(non_zero_kx) > 0 and len(non_zero_ky) > 0:
                        idx_x = non_zero_kx[[len(non_zero_kx)//4, len(non_zero_kx)//2]]
                        idx_y = non_zero_ky[[len(non_zero_ky)//4, len(non_zero_ky)//2]]
                        
                        kx_test = kx_unshifted[idx_x]
                        ky_test = ky_unshifted[idx_y]
                        x_test = x_grid[[0, Nx // 2]]
                        y_test = y_grid[[0, Ny // 2]]
                        
                        X_t, Y_t = np.meshgrid(x_test, y_test, indexing='ij')
                        KX_t, KY_t = np.meshgrid(kx_test, ky_test, indexing='ij')
                        
                        val1 = symbol_func(X_t[..., None, None], Y_t[..., None, None], 
                                           KX_t[None, None, ...], KY_t[None, None, ...])
                        val2 = symbol_func((X_t + dx)[..., None, None], (Y_t + dy)[..., None, None], 
                                           KX_t[None, None, ...], KY_t[None, None, ...])
                        is_independent = np.allclose(val1, val2)
                except Exception:
                    is_independent = False

        if is_independent:
            U = fft_func(u_vals)
            KX, KY = np.meshgrid(kx_unshifted, ky_unshifted, indexing='ij')
            
            # FIX: Enforce target shape
            P_raw = symbol_func(x_grid[0], y_grid[0], KX, KY)
            P = np.broadcast_to(P_raw, KX.shape).astype(np.complex128).copy()
            P = _clip_complex_magnitude(P, clamp)
            
            if freq_window == 'gaussian':
                sx = 0.8 * np.max(np.abs(kx_unshifted))
                sy = 0.8 * np.max(np.abs(ky_unshifted))
                P *= np.exp(-(KX / sx) ** 4) * np.exp(-(KY / sy) ** 4)
            elif freq_window == 'hann':
                kx_max = np.max(np.abs(kx_unshifted))
                ky_max = np.max(np.abs(ky_unshifted))
                Wx = 0.5 * (1 + np.cos(np.pi * KX / kx_max)) * (np.abs(KX) < kx_max)
                Wy = 0.5 * (1 + np.cos(np.pi * KY / ky_max)) * (np.abs(KY) < ky_max)
                P *= Wx * Wy
                
            return ifft_func(P * U)

        # --- SLOW PATH ---
        kx_s = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
        ky_s = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
        dkx, dky = kx_s[1] - kx_s[0], ky_s[1] - ky_s[0]
        Nkx, Nky = len(kx_s), len(ky_s)

        f_hat = np.fft.fftshift(fft_func(np.fft.fftshift(u_vals)) * dx * dy)
        exp_y = np.exp(1j * np.outer(y_grid, ky_s))

        win_kx, win_ky = None, None
        if freq_window == 'gaussian':
            sx = 0.8 * np.max(np.abs(kx_s))
            sy = 0.8 * np.max(np.abs(ky_s))
            win_kx = np.exp(-(kx_s / sx) ** 4)
            win_ky = np.exp(-(ky_s / sy) ** 4)
        elif freq_window == 'hann':
            kx_max, ky_max = np.max(np.abs(kx_s)), np.max(np.abs(ky_s))
            win_kx = 0.5 * (1 + np.cos(np.pi * kx_s / kx_max)) * (np.abs(kx_s) < kx_max)
            win_ky = 0.5 * (1 + np.cos(np.pi * ky_s / ky_max)) * (np.abs(ky_s) < ky_max)

        sw_x, sw_y = None, None
        if space_window:
            x0, Lx = (x_grid[0] + x_grid[-1]) / 2, (x_grid[-1] - x_grid[0]) / 2
            y0, Ly = (y_grid[0] + y_grid[-1]) / 2, (y_grid[-1] - y_grid[0]) / 2
            sw_x = np.exp(-((x_grid - x0) / Lx) ** 2)
            sw_y = np.exp(-((y_grid - y0) / Ly) ** 2)

        n_workers = max(w for w in range(1, int(FFT_WORKERS) + 1) if Nx % w == 0)
        base = max(1, Nx // n_workers)
        boundaries = [(i * base, min((i + 1) * base, Nx)) for i in range(n_workers) if i * base < Nx]
        result = np.zeros((Nx, Ny), dtype=np.complex128)

        def _process_block(bounds: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
            """
            Process one spatial block of rows for the memory-bounded slow path.
    
            Evaluates the symbol on the chunked space-frequency sub-grid,
            applies windowing/clamping, and accumulates the quadrature
            contribution via `np.einsum` into the block result. Designed to
            run inside a ThreadPoolExecutor for parallel row-block processing
            in the 2D case.
    
            Parameters
            ----------
            bounds : tuple of (int, int)
                Row indices (i0, i1) defining the spatial block.
    
            Returns
            -------
            tuple
                (i0, i1, result_block) where result_block is the ndarray of
                shape (i1−i0, Ny) [2D] or (i1−i0,) [1D] containing the
                operator output for those rows.
            """
            i0, i1 = bounds
            x_blk = x_grid[i0:i1]
            B = i1 - i0

            MAX_ELEMENTS = 16 * 1024 * 1024
            prod_C = max(1, MAX_ELEMENTS // (B * Ny))
            C1 = min(int(np.sqrt(prod_C)), Nkx)
            C2 = min(max(1, prod_C // C1), Nky)

            Xb = x_blk[:, None, None, None]
            Yb = y_grid[None, :, None, None]
            exp_x_full = np.exp(1j * np.outer(x_blk, kx_s))
            res_block = np.zeros((B, Ny), dtype=np.complex128)
            
            sw_x_blk = sw_x[i0:i1, None, None, None] if sw_x is not None else None

            for m0 in range(0, Nkx, C1):
                m1 = min(m0 + C1, Nkx)
                exp_x_chunk = exp_x_full[:, m0:m1].reshape(B, 1, m1-m0, 1)
                fh_m = f_hat[m0:m1, :]
                
                w_kx = win_kx[m0:m1, None] if win_kx is not None else 1.0

                for n0 in range(0, Nky, C2):
                    n1 = min(n0 + C2, Nky)
                    
                    P_chunk = symbol_func(Xb, Yb, kx_s[None, None, m0:m1, None], ky_s[None, None, None, n0:n1])
                    # FIX: Enforce target shape and ensure writability
                    P_chunk = np.broadcast_to(P_chunk, (B, Ny, m1-m0, n1-n0)).astype(np.complex128).copy()
                    P_chunk = _clip_complex_magnitude(P_chunk, clamp)

                    if freq_window is not None:
                        w_ky = win_ky[None, n0:n1] if isinstance(win_ky, np.ndarray) else 1.0
                        P_chunk *= (w_kx * w_ky)
                        
                    if space_window:
                        if sw_x_blk is not None:
                            P_chunk *= sw_x_blk
                        if sw_y is not None:
                            P_chunk *= sw_y[None, :, None, None]

                    exp_y_chunk = exp_y[:, n0:n1]
                    phase_chunk = exp_x_chunk * exp_y_chunk[None, :, None, :]
                    fh_sub = fh_m[:, n0:n1]

                    res_block += (dkx * dky / (2 * np.pi) ** 2) * np.einsum(
                        'bxky, ky, bxky -> bx', 
                        P_chunk, fh_sub, phase_chunk, 
                        optimize=True
                    )
            return i0, i1, res_block

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for i0, i1, blk in executor.map(_process_block, boundaries):
                result[i0:i1, :] = blk

        return result
    else:
        raise ValueError("Only dim=1 and dim=2 are supported")


# ============================================================================
# Non-Periodic Kohn-Nirenberg Quantization (Dirichlet)
# ============================================================================

def _cache_key_2d(x1, x2, xi1, xi2, freq_window, space_window):
    """
    Build a stable cache key for the 2D non-periodic Kohn-Nirenberg branch.

    Unlike the 1D key, this one must also encode `freq_window` and
    `space_window`, since those options change the actual content of the
    cached phase matrices/windows, not just the underlying grid. Each axis
    is hashed via `_cache_key_1d` so the 2D key stays consistent with the
    existing 1D cache-invalidation logic.

    Parameters
    ----------
    x1, x2 : ndarray
        Spatial grids along each axis.
    xi1, xi2 : ndarray
        Frequency grids along each axis.
    freq_window : str or None
        Name of the frequency-domain window applied when building the
        cached phase matrix.
    space_window : bool
        Whether a spatial window is applied; included as a bare boolean
        since it only changes whether windowing is on or off.

    Returns
    -------
    tuple
        Hashable key combining both axes' grid signatures with the window
        settings, suitable for use as a dictionary cache key.
    """
    return (
        _cache_key_1d(x1, xi1),
        _cache_key_1d(x2, xi2),
        freq_window,
        bool(space_window),
    )


def kohn_nirenberg_nonperiodic(
    u_vals: np.ndarray,
    x_grid: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    xi_grid: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    symbol_func: Callable[..., np.ndarray],
    freq_window: Optional[str] = 'gaussian',
    clamp: float = 1e6,
    space_window: bool = False,
    is_spatial: Optional[bool] = None,
    _cache: Dict = _KN_CACHE,
) -> np.ndarray:
    """
    Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator
    on a non-periodic domain using direct matrix/quadrature-based transforms.

    Applies the pseudo-differential operator Op(p) to the input function u via the
    non-periodic Kohn–Nirenberg integral formula:
        [Op(p) u](x) = (1/(2π)^d) ∫ p(x, ξ) e^{i x·ξ} ℱ_NP[u](ξ) dξ,
    where ℱ_NP[u] is the direct discrete Fourier integral transform evaluated over an
    arbitrary non-periodic spatial grid `x` and frequency grid `xi`.

    Supports 1D and 2D spatial dimensions, featuring a caching mechanism for both
    1D and 2D phase/window matrices, symbol magnitude clamping, optional frequency
    windowing (Gaussian/Hann), and spatial tapering.

    **1D Cached Path**
        In 1D, precalculates and caches discrete Fourier transform phases (`phase_ft`),
        reconstruction phases (`exp_matrix`), and window arrays in `_cache` to accelerate
        repeated function evaluations on identical grids.

    **2D Cached Path**
        In 2D, precalculates and caches the analogous grid-only objects — forward phase
        matrices (`phase1`, `phase2`), reconstruction phase matrices (`exp1`, `exp2`),
        the frequency window (`freq_win_2d`), and (if `space_window=True`) the spatial
        taper arrays (`sw_x1_full`, `sw_x2`). The cache key includes `freq_window` and
        `space_window` since, unlike the 1D case, these options change the cached
        content itself, not just which arrays get used.

    **2D Fast-Path Optimization (Spatial Independence)**
        Symbol spatial-dependence is controlled by `is_spatial`:
          - `is_spatial=False` forces the fast path (symbol treated as x-independent).
          - `is_spatial=True` forces the slow path (symbol treated as x-dependent).
          - `is_spatial=None` (default) runs a multi-point sampling heuristic (skipped,
            and slow path forced, whenever `space_window=True`, since the taper itself
            introduces x-dependence).
        When spatial independence holds, matrix multiplications perform global frequency
        filtering in O(N^3) complexity instead of the full O(N^4) space-dependent integration.

    **2D Memory-Bounded Slow Path (Spatial Dependence)**
        For space-dependent symbols in 2D, a multi-tiered execution strategy prevents RAM spikes:
        - **Row-Based Parallelization**: Slices `x1` into spatial blocks distributed across
          a worker thread pool (`ThreadPoolExecutor`).
        - **Dual Frequency Chunking**: Iterates across sub-blocks of `xi1` and `xi2` to guarantee
          intermediate tensor evaluation (`sv_chunk`, `phase_chunk`) remains strictly bounded (~256 MB max).
        - **Tensor Contraction**: Employs optimized Einstein summation (`np.einsum`) for localized quadrature integration.

    Parameters
    ----------
    u_vals : ndarray
        Spatial samples of the input field u(x) [1D] or u(x1, x2) [2D].
    x_grid : ndarray or tuple of ndarray
        Spatial coordinate grid `x` (1D) or tuple `(x1, x2)` (2D).
    xi_grid : ndarray or tuple of ndarray
        Frequency grid `xi` (1D) or tuple `(xi1, xi2)` (2D).
    symbol_func : callable
        Symbol evaluator p(x, ξ) in 1D or p(x1, x2, ξ1, ξ2) in 2D. Must accept
        NumPy-broadcastable positional arguments. Returns are automatically
        broadcasted, type-cast to `complex128`, and reshaped/copied safely to handle scalar
        or reduced-dimension outputs (e.g., from `sympy.lambdify`).
    freq_window : {'gaussian', 'hann', None}, default='gaussian'
        Frequency-domain window/taper applied to attenuate high-frequency numerical artifacts.
    clamp : float, default=1e6
        Maximum allowed magnitude for symbol values. Entries exceeding this threshold
        are clipped to prevent overflow.
    space_window : bool, default=False
        If True, applies a centered Gaussian spatial taper to attenuate edge boundary artifacts.
        *Note: Enabling this disables the 2D spatial-independence fast path.*
    is_spatial : bool or None, default=None
        Explicit hint about whether the 2D symbol depends on (x1, x2).
        True forces the slow (space-dependent) path, False forces the fast
        (space-independent) path, None triggers the sampling heuristic.
    _cache : dict, optional
        Global or local cache dictionary storing reusable 1D and 2D phase/window matrices.
        Defaults to module-level `_KN_CACHE`.

    Returns
    -------
    ndarray
        Resulting complex-valued array (`complex128`) of the same dimensionality and shape
        as `u_vals` after applying the non-periodic pseudo-differential operator.

    Raises
    ------
    NotImplementedError
        If `u_vals.ndim` is not 1 or 2.
    """
    if u_vals.ndim == 1:
        x = np.asarray(x_grid)
        xi = np.asarray(xi_grid)
        dx = x[1] - x[0]
        dxi = xi[1] - xi[0]

        key = _cache_key_1d(x, xi)
        if key not in _cache:
            phase_ft = np.exp(-1j * np.outer(xi, x))
            exp_matrix = np.exp(1j * np.outer(x, xi))

            xi_abs_max = np.max(np.abs(xi))
            sigma_w = 0.8 * xi_abs_max
            window_gauss = np.exp(-(xi / sigma_w) ** 4)

            window_hann = np.zeros_like(xi)
            mask = np.abs(xi) < xi_abs_max
            window_hann[mask] = 0.5 * (1.0 + np.cos(np.pi * xi[mask] / xi_abs_max))

            x_center = (x[0] + x[-1]) / 2.0
            L_half = (x[-1] - x[0]) / 2.0
            spatial_taper = np.exp(-((x - x_center) / L_half) ** 2)

            _cache[key] = dict(
                phase_ft=phase_ft, exp_matrix=exp_matrix,
                window_gauss=window_gauss, window_hann=window_hann,
                spatial_taper=spatial_taper,
            )
            warnings.warn(
                f"kohn_nirenberg_nonperiodic: building 1D cache (Nx={len(x)}, Nxi={len(xi)}).",
                stacklevel=2,
            )

        entry = _cache[key]
        u_hat = dx * (entry['phase_ft'] @ u_vals)

        sigma_raw = symbol_func(x[:, None], xi[None, :])
        sigma = np.broadcast_to(sigma_raw, (len(x), len(xi))).astype(np.complex128).copy()
        sigma = _clip_complex_magnitude(sigma, clamp)

        if freq_window == 'gaussian':
            sigma *= entry['window_gauss'][None, :]
        elif freq_window == 'hann':
            sigma *= entry['window_hann'][None, :]

        if space_window:
            sigma *= entry['spatial_taper'][:, None]

        weighted_exp = sigma * entry['exp_matrix']
        return (dxi / (2.0 * np.pi)) * (weighted_exp @ u_hat)

    elif u_vals.ndim == 2:
        x1, x2 = x_grid
        xi1, xi2 = xi_grid
        dx1, dx2 = x1[1] - x1[0], x2[1] - x2[0]
        dxi1, dxi2 = xi1[1] - xi1[0], xi2[1] - xi2[0]
        Nx1, Nx2 = len(x1), len(x2)
        Nxi1, Nxi2 = len(xi1), len(xi2)

        # --- CACHE 2D : objets qui ne dépendent que de la grille + des options
        # de fenêtrage (jamais de symbol_func ni de u_vals) ---
        key2d = _cache_key_2d(x1, x2, xi1, xi2, freq_window, space_window)
        if key2d not in _cache:
            phase1 = np.exp(-1j * np.outer(xi1, x1))
            phase2 = np.exp(-1j * np.outer(x2, xi2))
            exp1 = np.exp(1j * np.outer(x1, xi1))
            exp2 = np.exp(1j * np.outer(x2, xi2))

            freq_win_2d = None
            if freq_window == 'gaussian':
                s1 = 0.8 * np.max(np.abs(xi1))
                s2 = 0.8 * np.max(np.abs(xi2))
                freq_win_2d = np.exp(-(xi1 / s1) ** 4)[:, None] * np.exp(-(xi2 / s2) ** 4)[None, :]
            elif freq_window == 'hann':
                xi1_max, xi2_max = np.max(np.abs(xi1)), np.max(np.abs(xi2))
                Wx = 0.5 * (1 + np.cos(np.pi * xi1 / xi1_max)) * (np.abs(xi1) < xi1_max)
                Wy = 0.5 * (1 + np.cos(np.pi * xi2 / xi2_max)) * (np.abs(xi2) < xi2_max)
                freq_win_2d = Wx[:, None] * Wy[None, :]

            sw_x1_full = sw_x2 = None
            if space_window:
                xc = (x1[0] + x1[-1]) / 2.0
                Lx = (x1[-1] - x1[0]) / 2.0
                sw_x1_full = np.exp(-((x1 - xc) / Lx) ** 2)

                yc = (x2[0] + x2[-1]) / 2.0
                Ly = (x2[-1] - x2[0]) / 2.0
                sw_x2 = np.exp(-((x2 - yc) / Ly) ** 2)

            _cache[key2d] = dict(
                phase1=phase1, phase2=phase2, exp1=exp1, exp2=exp2,
                freq_win_2d=freq_win_2d, sw_x1_full=sw_x1_full, sw_x2=sw_x2,
            )
            warnings.warn(
                f"kohn_nirenberg_nonperiodic: building 2D cache "
                f"(Nx1={Nx1}, Nx2={Nx2}, Nxi1={Nxi1}, Nxi2={Nxi2}, "
                f"freq_window={freq_window!r}, space_window={space_window}).",
                stacklevel=2,
            )

        entry2d = _cache[key2d]
        phase1, phase2 = entry2d['phase1'], entry2d['phase2']
        exp1, exp2 = entry2d['exp1'], entry2d['exp2']
        freq_win_2d = entry2d['freq_win_2d']
        sw_x1_full, sw_x2 = entry2d['sw_x1_full'], entry2d['sw_x2']

        u_hat = dx1 * dx2 * (phase1 @ u_vals @ phase2)

        # --- FAST/SLOW PATH SELECTION ---
        # is_spatial=True  -> force slow path (symbol IS space-dependent)
        # is_spatial=False -> force fast path (symbol is NOT space-dependent)
        # is_spatial=None  -> heuristic decides (always slow if space_window=True,
        #                     since the taper itself makes the integrand x-dependent)
        if is_spatial is True:
            is_x_independent = False
        elif is_spatial is False:
            is_x_independent = True
        else:
            is_x_independent = False
            if not space_window:
                try:
                    x1_test = x1[[0, -1]]
                    x2_test = x2[[0, -1]]
                    xi_idx = max(1, Nxi1 // 2)
                    eta_idx = max(1, Nxi2 // 2)

                    val1 = symbol_func(
                        x1_test[:, None, None, None],
                        x2_test[None, :, None, None],
                        xi1[None, None, xi_idx:xi_idx + 1, None],
                        xi2[None, None, None, eta_idx:eta_idx + 1]
                    )
                    val2 = symbol_func(
                        (x1_test + dx1)[:, None, None, None],
                        (x2_test + dx2)[None, :, None, None],
                        xi1[None, None, xi_idx:xi_idx + 1, None],
                        xi2[None, None, None, eta_idx:eta_idx + 1]
                    )
                    is_x_independent = np.allclose(val1, val2)
                except Exception:
                    is_x_independent = False

        if is_x_independent:
            p_full_raw = symbol_func(
                np.full((1, 1, Nxi1, 1), x1[0]),
                np.full((1, 1, 1, Nxi2), x2[0]),
                xi1[None, None, :, None],
                xi2[None, None, None, :]
            )
            p_full = np.broadcast_to(p_full_raw, (1, 1, Nxi1, Nxi2)).astype(np.complex128).reshape(Nxi1, Nxi2).copy()
            p_full = _clip_complex_magnitude(p_full, clamp)

            if freq_win_2d is not None:
                p_full *= freq_win_2d

            u_hat_filtered = p_full * u_hat

            return (dxi1 * dxi2 / (2.0 * np.pi) ** 2) * (exp1 @ u_hat_filtered @ exp2.T)

        # --- SLOW PATH ---
        iph2 = exp2  # np.exp(1j * outer(x2, xi2)), déjà en cache
        n_workers = max(w for w in range(1, int(FFT_WORKERS) + 1) if Nx1 % w == 0)
        base = max(1, Nx1 // n_workers)
        boundaries = [(i * base, min((i + 1) * base, Nx1)) for i in range(n_workers) if i * base < Nx1]
        result = np.zeros((Nx1, Nx2), dtype=np.complex128)

        def _process_block(bounds: Tuple[int, int]) -> Tuple[int, int, np.ndarray]:
            """
            Process one spatial block of rows for the memory-bounded slow path.
    
            Evaluates the symbol on the chunked space-frequency sub-grid,
            applies windowing/clamping, and accumulates the quadrature
            contribution via `np.einsum` into the block result. Designed to
            run inside a ThreadPoolExecutor for parallel row-block processing
            in the 2D case.
    
            Parameters
            ----------
            bounds : tuple of (int, int)
                Row indices (i0, i1) defining the spatial block.
    
            Returns
            -------
            tuple
                (i0, i1, result_block) where result_block is the ndarray of
                shape (i1−i0, Ny) [2D] or (i1−i0,) [1D] containing the
                operator output for those rows.
            """
            i0, i1 = bounds
            x1_blk = x1[i0:i1]
            B = i1 - i0

            MAX_ELEMENTS = 16 * 1024 * 1024
            prod_C = max(1, MAX_ELEMENTS // (B * Nx2))
            C1 = min(int(np.sqrt(prod_C)), Nxi1)
            C2 = min(max(1, prod_C // C1), Nxi2)

            X1b = x1_blk[:, None, None, None]
            X2b = x2[None, :, None, None]
            res_block = np.zeros((B, Nx2), dtype=np.complex128)

            sw_x1_blk = sw_x1_full[i0:i1, None, None, None] if space_window else None

            for k0 in range(0, Nxi1, C1):
                k1 = min(k0 + C1, Nxi1)
                iph1_chunk = np.exp(1j * np.outer(x1_blk, xi1[k0:k1])).reshape(B, 1, k1 - k0, 1)
                u_hat_k = u_hat[k0:k1, :]

                for m0 in range(0, Nxi2, C2):
                    m1 = min(m0 + C2, Nxi2)

                    sv_chunk = symbol_func(X1b, X2b, xi1[None, None, k0:k1, None], xi2[None, None, None, m0:m1])
                    sv_chunk = np.broadcast_to(sv_chunk, (B, Nx2, k1 - k0, m1 - m0)).astype(np.complex128).copy()
                    sv_chunk = _clip_complex_magnitude(sv_chunk, clamp)

                    if freq_win_2d is not None:
                        sv_chunk *= freq_win_2d[k0:k1, m0:m1][None, None, :, :]

                    if space_window:
                        if sw_x1_blk is not None:
                            sv_chunk *= sw_x1_blk
                        if sw_x2 is not None:
                            sv_chunk *= sw_x2[None, :, None, None]

                    iph2_chunk = iph2[:, m0:m1]
                    phase_chunk = iph1_chunk * iph2_chunk[None, :, None, :]
                    u_hat_sub = u_hat_k[:, m0:m1]

                    res_block += (dxi1 * dxi2 / (2.0 * np.pi) ** 2) * np.einsum(
                        'bxky, ky, bxky -> bx',
                        sv_chunk, u_hat_sub, phase_chunk,
                        optimize=True
                    )
            return i0, i1, res_block

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for i0, i1, blk in executor.map(_process_block, boundaries):
                result[i0:i1, :] = blk

        return result
    else:
        raise NotImplementedError("Only 1D (ndim=1) and 2D (ndim=2) inputs are supported")


def _sympy_number(z, digits=5, drop_tol=0.0):
    """
    Convert a Python/NumPy complex number into a SymPy number, since
    `sympy.Float` does not accept complex values directly.

    Parameters
    ----------
    z : complex or float
        Value to convert.
    digits : int, default 5
        Number of significant digits kept for the real and imaginary
        parts.
    drop_tol : float, default 0.0
        Real or imaginary components with absolute value at or below this
        threshold are snapped to exactly zero before conversion, to avoid
        carrying negligible numerical noise into the symbolic expression.

    Returns
    -------
    sympy.Float or sympy.Expr
        `sympy.Float(re, digits)` if the imaginary part is zero, otherwise
        `sympy.Float(re, digits) + sympy.I * sympy.Float(im, digits)`.
    """
    z = complex(z)
    re = float(np.real(z))
    im = float(np.imag(z))

    if abs(re) <= drop_tol:
        re = 0.0
    if abs(im) <= drop_tol:
        im = 0.0

    if im == 0.0:
        return sp.Float(re, digits)

    return sp.Float(re, digits) + sp.I * sp.Float(im, digits)


def _chebyshev_polynomial(n, z):
    """
    Return T_n(z) as an explicit expanded SymPy polynomial.

    This avoids possible lambdify issues with special Chebyshev functions.
    """
    if n == 0:
        return sp.S.One
    if n == 1:
        return z

    t_prev = sp.S.One
    t_curr = z

    for _ in range(2, n + 1):
        t_prev, t_curr = t_curr, sp.expand(2 * z * t_curr - t_prev)

    return t_curr

def evaluate_decomposition_quality(
    orig_expr,
    symbolic_pairs,
    x_syms,
    xi_syms,
    bounds,
    num_samples=10000,
    seed=42,
):
    """
    Estimate the symbol-level approximation error of a separable/low-rank
    decomposition against the original expression, via Monte Carlo
    sampling at random off-grid points (so the error reflects genuine
    approximation quality rather than exact agreement at the fitting
    nodes):

        orig_expr(x, xi) ≈ sum_k a_k(x) q_k(xi)

    Parameters
    ----------
    orig_expr : sympy.Expr
        Original joint symbol being approximated.
    symbolic_pairs : list of tuple
        Candidate decomposition, as pairs `(a_k(x), q_k(xi))` of sympy
        expressions.
    x_syms : list of sympy symbols
        Spatial variables of `orig_expr`.
    xi_syms : list of sympy symbols
        Frequency variables of `orig_expr`.
    bounds : dict
        Mapping from each symbol in `x_syms + xi_syms` to a `(min, max)`
        sampling range.
    num_samples : int, default 10000
        Number of random points drawn uniformly within `bounds`.
    seed : int, default 42
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys `'rel_l2_error'` (relative L2 error over the
        sampled points), `'max_abs_error'` and `'mean_abs_error'`
        (pointwise absolute-error statistics).
    """
    rng = np.random.default_rng(seed)

    x_syms = list(x_syms)
    xi_syms = list(xi_syms)
    all_syms = x_syms + xi_syms

    sample_dict = {}
    for s in all_syms:
        s_min, s_max = bounds[s]
        sample_dict[s] = rng.uniform(s_min, s_max, size=num_samples)

    # Original expression
    f_orig = sp.lambdify(all_syms, orig_expr, modules="numpy")
    args = [sample_dict[s] for s in all_syms]
    y_orig = np.asarray(f_orig(*args), dtype=np.complex128).reshape(-1)

    if y_orig.size == 1:
        y_orig = np.full(num_samples, y_orig.item(), dtype=np.complex128)
    elif y_orig.size != num_samples:
        y_orig = np.broadcast_to(y_orig, (num_samples,)).astype(np.complex128)

    # Approximation
    y_approx = np.zeros(num_samples, dtype=np.complex128)

    x_pts = [sample_dict[s] for s in x_syms]
    xi_pts = [sample_dict[s] for s in xi_syms]

    for a_k, q_k in symbolic_pairs:
        f_a = sp.lambdify(x_syms, a_k, modules="numpy")
        f_q = sp.lambdify(xi_syms, q_k, modules="numpy")

        try:
            val_a = np.asarray(f_a(*x_pts), dtype=np.complex128).reshape(-1)
            if val_a.size == 1:
                val_a = np.full(num_samples, val_a.item(), dtype=np.complex128)
            elif val_a.size != num_samples:
                val_a = np.broadcast_to(val_a, (num_samples,)).astype(np.complex128)
        except Exception:
            val_a = np.full(num_samples, complex(a_k), dtype=np.complex128)

        try:
            val_q = np.asarray(f_q(*xi_pts), dtype=np.complex128).reshape(-1)
            if val_q.size == 1:
                val_q = np.full(num_samples, val_q.item(), dtype=np.complex128)
            elif val_q.size != num_samples:
                val_q = np.broadcast_to(val_q, (num_samples,)).astype(np.complex128)
        except Exception:
            val_q = np.full(num_samples, complex(q_k), dtype=np.complex128)

        y_approx += val_a * val_q

    diff = y_orig - y_approx

    norm_orig = np.linalg.norm(y_orig)
    norm_diff = np.linalg.norm(diff)

    rel_l2_err = float(norm_diff / norm_orig) if norm_orig > 0 else float(norm_diff)

    abs_err = np.abs(diff)

    return {
        "rel_l2_error": rel_l2_err,
        "max_abs_error": float(np.max(abs_err)),
        "mean_abs_error": float(np.mean(abs_err)),
    }


def factorize_symbolic(
    expr,
    x_syms,
    xi_syms,
    bounds,
    degree=6,
    tol=1e-5,
    num_samples=10000,
    seed=42,
    digits=5,
):
    """
    Low-rank Chebyshev/SVD factorization of a joint symbol:

        p(x, xi) ≈ sum_{k=1}^r a_k(x) q_k(xi)

    The approximation is valid on the bounded rectangle given by `bounds`.

    Parameters
    ----------
    expr : sympy.Expr
        Symbol to factorize, usually the Peetre joint residual.
    x_syms : list of sympy symbols
        Spatial variables.
    xi_syms : list of sympy symbols
        Frequency variables.
    bounds : dict
        Dictionary mapping each symbol to (min, max).
    degree : int
        Chebyshev degree in each variable.
    tol : float
        Relative singular-value cutoff and coefficient pruning threshold.
    num_samples : int
        Number of Monte Carlo samples for quality diagnostics.
    seed : int
        RNG seed.
    digits : int
        Number of digits used when converting floating coefficients to SymPy.

    Returns
    -------
    symbolic_pairs : list of tuple
        List of `(a_k(x), q_k(xi))` SymPy expressions.
    metrics : dict
        Symbol-level approximation diagnostics.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    x_syms = list(x_syms)
    xi_syms = list(xi_syms)
    all_syms = x_syms + xi_syms

    # ---------------------------------------------------------------
    # 1. Chebyshev-Gauss-Lobatto nodes on [-1, 1]
    # ---------------------------------------------------------------
    nodes_1d = [
        np.cos(np.pi * np.arange(degree + 1) / degree)
        for _ in all_syms
    ]

    # ---------------------------------------------------------------
    # 2. Normalize physical variables to [-1, 1]
    # ---------------------------------------------------------------
    norm_vars = {}
    phys_from_norm = []

    for s in all_syms:
        s_min, s_max = bounds[s]

        if s_max <= s_min:
            s_min = float(s_min) - 1.0
            s_max = float(s_min) + 2.0

        norm_vars[s] = (2 * s - (s_min + s_max)) / (s_max - s_min)

        phys_from_norm.append(
            lambda y, b_min=s_min, b_max=s_max:
                0.5 * (b_min + b_max) + 0.5 * (b_max - b_min) * y
        )

    # ---------------------------------------------------------------
    # 3. Evaluate expression on tensor-product Chebyshev grid
    # ---------------------------------------------------------------
    grid_coords = [
        phys_from_norm[idx](nodes_1d[idx])
        for idx in range(len(all_syms))
    ]

    mesh = np.meshgrid(*grid_coords, indexing="ij")

    func_num = sp.lambdify(all_syms, expr, modules="numpy")
    P_eval = np.asarray(func_num(*mesh), dtype=np.complex128)

    target_shape = mesh[0].shape
    if P_eval.shape != target_shape:
        P_eval = np.broadcast_to(P_eval, target_shape).astype(np.complex128)

    P_eval = P_eval.copy()

    empty_metrics = {
        "rel_l2_error": 0.0,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "svd_energy_retained_pct": 100.0,
        "singular_values": np.array([]),
    }

    if np.allclose(P_eval, 0.0, atol=1e-14):
        return [], empty_metrics

    # ---------------------------------------------------------------
    # 4. Chebyshev coefficients by Vandermonde inversion
    # ---------------------------------------------------------------
    vands = [
        np.polynomial.chebyshev.chebvander(nodes_1d[i], degree)
        for i in range(len(all_syms))
    ]

    C_tensor = P_eval

    for i, V in enumerate(vands):
        inv_V = np.linalg.inv(V)

        C_tensor = np.moveaxis(C_tensor, i, 0)
        orig_shape = C_tensor.shape
        C_tensor = inv_V @ C_tensor.reshape(orig_shape[0], -1)
        C_tensor = C_tensor.reshape(orig_shape)
        C_tensor = np.moveaxis(C_tensor, 0, i)

    # ---------------------------------------------------------------
    # 5. Reshape coefficients into spatial × frequency matrix
    # ---------------------------------------------------------------
    d_x = len(x_syms)
    d_xi = len(xi_syms)

    N_x_total = (degree + 1) ** d_x
    N_xi_total = (degree + 1) ** d_xi

    C_matrix = C_tensor.reshape((N_x_total, N_xi_total))

    # ---------------------------------------------------------------
    # 6. SVD low-rank truncation
    # ---------------------------------------------------------------
    U, S, Vt = np.linalg.svd(C_matrix, full_matrices=False)

    if S.size == 0 or S[0] == 0:
        return [], empty_metrics

    keep = S > (S[0] * tol)

    if not np.any(keep):
        keep = np.zeros_like(S, dtype=bool)
        keep[0] = True

    energy_den = float(np.sum(S ** 2))
    svd_energy_retained = (
        100.0 * float(np.sum(S[keep] ** 2)) / energy_den
        if energy_den > 0 else 100.0
    )

    # ---------------------------------------------------------------
    # 7. Reconstruct symbolic separable terms
    # ---------------------------------------------------------------
    spatial_multi_indices = list(
        itertools.product(range(degree + 1), repeat=d_x)
    )
    spectral_multi_indices = list(
        itertools.product(range(degree + 1), repeat=d_xi)
    )

    def _cheb(deg, s):
        return _chebyshev_polynomial(deg, norm_vars[s])

    symbolic_pairs = []

    S_keep = S[keep]
    U_keep = U[:, keep]
    Vt_keep = Vt[keep, :]

    for k in range(len(S_keep)):
        sigma_k = S_keep[k]
        u_k = U_keep[:, k]
        v_k = Vt_keep[k, :]

        # a_k(x)
        a_k_expr = sp.S.Zero
        for idx, multi_idx in enumerate(spatial_multi_indices):
            coeff = np.sqrt(sigma_k) * u_k[idx]

            if np.abs(coeff) > tol:
                if len(multi_idx) == 0:
                    basis_term = sp.S.One
                else:
                    basis_term = sp.Mul(
                        *[
                            _cheb(deg, x_syms[m])
                            for m, deg in enumerate(multi_idx)
                        ]
                    )

                a_k_expr += _sympy_number(coeff, digits=digits) * basis_term

        # q_k(xi)
        q_k_expr = sp.S.Zero
        for idx, multi_idx in enumerate(spectral_multi_indices):
            coeff = np.sqrt(sigma_k) * v_k[idx]

            if np.abs(coeff) > tol:
                if len(multi_idx) == 0:
                    basis_term = sp.S.One
                else:
                    basis_term = sp.Mul(
                        *[
                            _cheb(deg, xi_syms[n])
                            for n, deg in enumerate(multi_idx)
                        ]
                    )

                q_k_expr += _sympy_number(coeff, digits=digits) * basis_term

        symbolic_pairs.append(
            (sp.expand(a_k_expr), sp.expand(q_k_expr))
        )

    # ---------------------------------------------------------------
    # 8. Monte Carlo quality metrics
    # ---------------------------------------------------------------
    metrics = evaluate_decomposition_quality(
        expr,
        symbolic_pairs,
        x_syms,
        xi_syms,
        bounds,
        num_samples=num_samples,
        seed=seed,
    )

    metrics["svd_energy_retained_pct"] = svd_energy_retained
    metrics["singular_values"] = S_keep

    return symbolic_pairs, metrics


# ============================================================================
# NUFFT-based joint-residual backend (joint_backend='nufft')
# ============================================================================
#
# Targets Category-C joint residuals that are OSCILLATORY (a genuine phase
# exp(i*Lambda(x)*M(xi)), e.g. sin(x*xi), exp(I*x*xi)) rather than algebraic.
# factorize_symbolic's Chebyshev/SVD basis converges poorly on these because
# a polynomial basis cannot efficiently represent a genuinely bilinear phase.
#
# PERIODIC BOUNDARY CONDITIONS ONLY. This backend has only been derived and
# validated for the FFT/periodic application path (boundary_condition=
# 'periodic'). It is not applicable to 'dirichlet' and apply_peetre() must
# fall back to the direct path in that case -- do not attempt to extend this
# silently without re-deriving the non-periodic quadrature.
#
# Requires the optional 'finufft' package for its O(N log N) benefit; falls
# back to an O(N*M) direct evaluation of the same embedding formula (correct,
# just not fast) if finufft is not installed, with a one-time warning.

try:
    import finufft as _finufft
    _HAVE_FINUFFT = True
except ImportError:
    _finufft = None
    _HAVE_FINUFFT = False

_finufft_warned = False


def _warn_no_finufft():
    global _finufft_warned
    if not _finufft_warned:
        warnings.warn(
            "finufft is not installed; joint_backend='nufft' will use a "
            "much slower O(N*M) direct-sum fallback that reproduces the "
            "same math but without the O(N log N) speed benefit. "
            "Install with `pip install finufft` for the intended performance."
        )
        _finufft_warned = True


def _nufft_split_real_imag_exponent(total_exponent):
    """Split an exponent into (I*phase, real_envelope) without silently
    dropping a real residual (a naive .coeff(sp.I) does this incorrectly
    for mixed exponents like I*x*xi - x**2)."""
    exp_terms = sp.Add.make_args(sp.expand(total_exponent))
    imag_terms, real_terms = [], []
    for t in exp_terms:
        c = t.coeff(sp.I)
        if sp.expand(t - sp.I * c) == 0:
            imag_terms.append(c)
        else:
            real_terms.append(t)
    phase_expr = sp.expand(sum(imag_terms)) if imag_terms else sp.Integer(0)
    real_envelope = sp.expand(sum(real_terms)) if real_terms else sp.Integer(0)
    return phase_expr, real_envelope, bool(imag_terms)


def _nufft_extract_term_nd(term, phys_syms, freq_syms):
    """
    Factor a single (exp-rewritten, expanded) additive term into
        c(phys) * g(freq) * exp(i * Lambda(phys) * M(freq))
    for phys_syms=(x,) / freq_syms=(xi,) [1D] or phys_syms=(x,y) /
    freq_syms=(xi,eta) [2D]. Returns None if it doesn't fit this pattern
    (conservative: never returns a wrong plan).
    """
    term_simp = sp.powsimp(term, combine="exp", deep=True)
    factors = sp.Mul.make_args(term_simp)

    exp_args, amp_factors = [], []
    for f in factors:
        if f.is_Pow and f.base == sp.E:
            exp_args.append(f.exp)
        elif isinstance(f, sp.exp):
            exp_args.append(f.args[0])
        else:
            amp_factors.append(f)

    if not exp_args:
        return None

    total_exponent = sp.expand(sum(exp_args))
    phase_expr, real_envelope, has_osc = _nufft_split_real_imag_exponent(total_exponent)
    if not has_osc:
        return None

    phase_factored = sp.factor(phase_expr)
    Lambda_p, M_f = phase_factored.as_independent(*freq_syms, as_Add=False)

    coupled_to_freq = any(Lambda_p.has(s) for s in freq_syms)
    coupled_to_phys = any(M_f.has(s) for s in phys_syms)
    no_real_coupling = not any(phase_expr.has(s) for s in freq_syms)
    if coupled_to_freq or coupled_to_phys or no_real_coupling:
        return None
    if sp.expand(Lambda_p * M_f - phase_expr) != 0:
        return None

    amp_expr = sp.Mul(*amp_factors)
    if real_envelope != 0:
        amp_expr = amp_expr * sp.exp(real_envelope)
    amp_factored = sp.factor(amp_expr) if amp_expr.is_Add else amp_expr
    c_p, g_f = amp_factored.as_independent(*freq_syms, as_Add=False)
    if any(c_p.has(s) for s in freq_syms) or any(g_f.has(s) for s in phys_syms):
        return None
    if sp.expand(c_p * g_f - amp_expr) != 0:
        return None

    return {
        "c_expr": c_p, "g_expr": g_f, "Lambda_expr": Lambda_p, "M_expr": M_f,
        "c": sp.lambdify(phys_syms, c_p, "numpy"),
        "g": sp.lambdify(freq_syms, g_f, "numpy"),
        "Lambda": sp.lambdify(phys_syms, Lambda_p, "numpy"),
        "M": sp.lambdify(freq_syms, M_f, "numpy"),
    }


def try_nufft_decomposition_1d(joint_expr, x_sym, xi_sym):
    """1D (phase space (x,xi)) NUFFT classifier. Returns a list of term
    plans, or None if any additive term doesn't fit (falls back)."""
    expr = sp.expand(joint_expr.rewrite(sp.exp))
    plans = []
    for term in sp.Add.make_args(expr):
        p = _nufft_extract_term_nd(term, (x_sym,), (xi_sym,))
        if p is None:
            return None
        plans.append(p)
    return plans


def _resolve_1d_piece_for_axis_sep(part_expr, phys_sym, freq_sym):
    """Used by the 2D axis-separable tier: resolve a single-variable-pair
    factor into pointwise (no freq dependence) or nufft1d pieces."""
    if not part_expr.has(freq_sym):
        return [{"kind": "pointwise", "amp": sp.lambdify(phys_sym, part_expr, "numpy")}]
    rewritten = sp.expand(part_expr.rewrite(sp.exp))
    pieces = []
    for sub in sp.Add.make_args(rewritten):
        if not sub.has(freq_sym):
            pieces.append({"kind": "pointwise", "amp": sp.lambdify(phys_sym, sub, "numpy")})
            continue
        plan = _nufft_extract_term_nd(sub, (phys_sym,), (freq_sym,))
        if plan is None:
            return None
        pieces.append({"kind": "nufft1d", "plan": plan})
    return pieces


def try_nufft_decomposition_2d(joint_expr, x_sym, y_sym, xi_sym, eta_sym):
    """
    2D (phase space (x,y,xi,eta)) NUFFT classifier. Tries, in order:
      (a) axis-separable: term factors as A(x,xi)*B(y,eta) (disjoint
          variable groups) -- cheapest, two independent 1D passes.
      (b) single-joint-term: term's phase is one product
          Lambda(x,y)*M(xi,eta) -- needs a 3D NUFFT embedding.
    A symbol whose terms need genuinely independent coupling on BOTH axes
    simultaneously (4D embedding) is not representable by either tier and
    returns None (finufft has no type-3 transform above 3D).
    Returns ('axis_sep', combo_plan) or ('joint3d', plans) or None.
    """
    # --- try axis-separable first (checked before any exp-rewrite, since
    #     rewriting collapses the very structure this tier looks for) ---
    expr_raw = sp.expand(joint_expr)
    combo_plan = []
    axis_sep_ok = True
    for term in sp.Add.make_args(expr_raw):
        A_part, B_part = term.as_independent(y_sym, eta_sym, as_Add=False)
        if A_part.has(y_sym) or A_part.has(eta_sym) or B_part.has(x_sym) or B_part.has(xi_sym):
            axis_sep_ok = False
            break
        A_pieces = _resolve_1d_piece_for_axis_sep(A_part, x_sym, xi_sym)
        B_pieces = _resolve_1d_piece_for_axis_sep(B_part, y_sym, eta_sym)
        if A_pieces is None or B_pieces is None:
            axis_sep_ok = False
            break
        for a in A_pieces:
            for b in B_pieces:
                combo_plan.append({"A": a, "B": b})
    if axis_sep_ok and combo_plan:
        return ("axis_sep", combo_plan)

    # --- fall back to single-joint-term (3D embed) ---
    expr = sp.expand(joint_expr.rewrite(sp.exp))
    plans = []
    for term in sp.Add.make_args(expr):
        p = _nufft_extract_term_nd(term, (x_sym, y_sym), (xi_sym, eta_sym))
        if p is None:
            return None
        plans.append(p)
    return ("joint3d", plans) if plans else None


def _nufft_uhat_1d(u, x_grid, dx, kx):
    """Continuous-FT approx of u, correcting for a grid not starting at 0
    (e.g. x_grid = linspace(-L, L, N, endpoint=False), used throughout this
    module's make_grid_1d/2d) -- see module docstring above for why this
    matters: without it, results are internally self-consistent but not
    the true KN operator action."""
    x0 = x_grid[0]
    return np.fft.fft(u) * dx * np.exp(-1j * x0 * kx)


def _nufft_direct_2d_type3(sx, sy, weights, tx, ty, isign=1):
    phase = isign * (tx[:, None] * sx[None, :] + ty[:, None] * sy[None, :])
    return (weights[None, :] * np.exp(1j * phase)).sum(axis=1)


def _nufft_direct_3d_type3(sx, sy, sz, weights, tx, ty, tz, isign=1):
    phase = isign * (tx[:, None] * sx[None, :] + ty[:, None] * sy[None, :]
                      + tz[:, None] * sz[None, :])
    return (weights[None, :] * np.exp(1j * phase)).sum(axis=1)


def _nufft_freq_window(kvals, freq_window):
    """Match kohn_nirenberg_fft's exact windowing formula (see slow-path
    P_blk *= win_k), applied elementwise on a raw (unshifted) frequency
    array -- the formula only depends on |k|/sigma pointwise, so it's
    correct regardless of fftshift ordering."""
    if freq_window == "gaussian":
        sigma = 0.8 * np.max(np.abs(kvals))
        return np.exp(-(kvals / sigma) ** 4)
    elif freq_window == "hann":
        k_max = np.max(np.abs(kvals))
        return 0.5 * (1 + np.cos(np.pi * kvals / k_max)) * (np.abs(kvals) < k_max)
    return np.ones_like(kvals, dtype=float)


def apply_nufft_1d(u, plan, x_grid, kx, dx, dxi, eps=1e-12, freq_window="gaussian"):
    """Apply Op(p_joint) via the NUFFT tier, 1D case. `plan` is the output
    of try_nufft_decomposition_1d (a list of term dicts).

    freq_window matches kohn_nirenberg_fft's default -- without applying
    it here too, results silently diverge from joint_backend='direct'
    even at freq_window='gaussian' defaults (found via end-to-end testing
    against the real dispatcher, not from the isolated unit tests, which
    never exercised the default windowing at all)."""
    u = np.asarray(u, dtype=complex)
    uhat = _nufft_uhat_1d(u, x_grid, dx, kx)
    win = _nufft_freq_window(kx, freq_window)
    result = np.zeros_like(x_grid, dtype=complex)
    for term in plan:
        c_x = term["c"](x_grid)
        g_xi = term["g"](kx) * win
        lam_x = term["Lambda"](x_grid)
        mu_xi = term["M"](kx)
        weights = (g_xi * uhat * dxi / (2 * np.pi)).astype(complex)
        src_x, src_y = kx, mu_xi
        tgt_x, tgt_y = x_grid, lam_x
        if _HAVE_FINUFFT:
            f = _finufft.nufft2d3(src_x, src_y, weights, tgt_x, tgt_y, isign=1, eps=eps)
        else:
            _warn_no_finufft()
            f = _nufft_direct_2d_type3(src_x, src_y, weights, tgt_x, tgt_y, isign=1)
        result += c_x * f
    return result


def apply_nufft_2d(u, kind, plan, x_grid, y_grid, kx, ky, dx, dy, dxi, deta, eps=1e-12,
                    freq_window="gaussian"):
    """Apply Op(p_joint) via the NUFFT tier, 2D case. `kind`/`plan` are the
    output of try_nufft_decomposition_2d. See apply_nufft_1d docstring on
    why freq_window must be matched to the direct path's default."""
    u = np.asarray(u, dtype=complex)

    if kind == "joint3d":
        x0, y0 = x_grid[0], y_grid[0]
        XI0, ETA0 = np.meshgrid(kx, ky, indexing="ij")
        uhat = np.fft.fft2(u) * dx * dy * np.exp(-1j * (x0 * XI0 + y0 * ETA0))
        XI, ETA = np.meshgrid(kx, ky, indexing="ij")
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        # 2D window: kohn_nirenberg_fft applies the SAME 1D-style formula
        # to the combined radial-like |k| via kx/ky separately multiplied;
        # match by applying to each axis and taking the product (matches
        # the 2D fast-path convention used elsewhere in this module).
        win_x = _nufft_freq_window(kx, freq_window)
        win_y = _nufft_freq_window(ky, freq_window)
        WIN = win_x[:, None] * win_y[None, :]
        result = np.zeros((len(x_grid), len(y_grid)), dtype=complex)
        for term in plan:
            c_xy = term["c"](X, Y)
            g_xieta = term["g"](XI, ETA) * WIN
            Lambda_xy = term["Lambda"](X, Y)
            M_xieta = term["M"](XI, ETA)
            Nx, Ny = len(x_grid), len(y_grid)
            src_xi, src_eta = XI.ravel(), ETA.ravel()
            src_M = np.broadcast_to(M_xieta, (Nx, Ny)).ravel()
            weights = (np.broadcast_to(g_xieta, (Nx, Ny)) * uhat
                       * dxi * deta / (2 * np.pi) ** 2).ravel().astype(complex)
            tgt_x, tgt_y = X.ravel(), Y.ravel()
            tgt_L = np.broadcast_to(Lambda_xy, (Nx, Ny)).ravel()
            if _HAVE_FINUFFT:
                f = _finufft.nufft3d3(src_xi, src_eta, src_M, weights, tgt_x, tgt_y, tgt_L,
                                       isign=1, eps=eps)
            else:
                _warn_no_finufft()
                f = _nufft_direct_3d_type3(src_xi, src_eta, src_M, weights, tgt_x, tgt_y, tgt_L, isign=1)
            result += c_xy * f.reshape(Nx, Ny)
        return result

    elif kind == "axis_sep":
        result = np.zeros_like(u, dtype=complex)
        for combo in plan:
            # Step 1: apply B (y,eta) row-wise; Step 2: apply A (x,xi) column-wise
            w = _apply_1d_piece_rows(combo["B"], u, y_grid, ky, dy, deta, along_axis=1,
                                      freq_window=freq_window)
            contrib = _apply_1d_piece_rows(combo["A"], w, x_grid, kx, dx, dxi, along_axis=0,
                                            freq_window=freq_window)
            result += contrib
        return result

    raise ValueError(f"unknown NUFFT 2D plan kind: {kind}")


def _apply_1d_piece_rows(piece, field, axis_grid, k_axis, d_axis, dk_axis, along_axis,
                          freq_window="gaussian"):
    if piece["kind"] == "pointwise":
        amp_vals = piece["amp"](axis_grid)
        return field * (amp_vals[None, :] if along_axis == 1 else amp_vals[:, None])
    plan = [piece["plan"]]
    out = np.zeros_like(field, dtype=complex)
    if along_axis == 1:
        for i in range(field.shape[0]):
            out[i, :] = apply_nufft_1d(field[i, :], plan, axis_grid, k_axis, d_axis, dk_axis,
                                        freq_window=freq_window)
    else:
        for j in range(field.shape[1]):
            out[:, j] = apply_nufft_1d(field[:, j], plan, axis_grid, k_axis, d_axis, dk_axis,
                                        freq_window=freq_window)
    return out

# ============================================================================
# AAA-based joint-residual backend (joint_backend='aaa')
# ============================================================================
#
# Targets Category-C joint residuals that are RATIONAL (resolvent-shaped,
# poles / algebraic decay, no oscillatory phase -- try_nufft_decomposition
# correctly rejects these). Builds a compact rational approximation of the
# symbol via vector-valued AAA (shared poles across a Chebyshev grid in the
# OTHER variable(s); the symbol is evaluated EXACTLY at each AAA support
# point via sympy substitution, so only the shared-pole structure in the
# frequency variable(s) introduces approximation error).
#
# KNOWN LIMITATION, BY CONSTRUCTION: this works well when the joint
# residual's pole locations are fixed or slowly varying with x (resp.
# x,y) -- it degrades (many poles needed, effectively no compression) when
# the pole genuinely MOVES with the spatial variable (e.g. 1/(xi-x-i*eps),
# a diagonal-type singularity). The quality gate below (joint_max_rel_error)
# catches a resulting bad fit and falls back to direct application; it does
# NOT silently return an inaccurate result. Diagonal-pole symbols are a
# genuinely different structural class (Calderon-Zygmund-type) that this
# backend does not target -- do not raise n_cheb/n_samples to "fix" a
# rejection here without first checking whether the pole is x-dependent.
#
# Unlike the NUFFT backend, this one delegates the actual numerical KN
# application to this module's own kohn_nirenberg_fft / 
# kohn_nirenberg_nonperiodic (via a fast numpy callable wrapping the AAA
# fit), so it supports BOTH periodic and dirichlet boundary conditions for
# free, and automatically inherits their existing grid-origin-correct
# numerics -- it does not reimplement the KN quadrature itself.

class _VectorAAA:
    """Barycentric rational fit r(z) in C^m, shared poles across m
    'vector components' (e.g. one component per Chebyshev x-node)."""
    def __init__(self, z_support, w, f_support):
        self.z_support = np.asarray(z_support)
        self.w = np.asarray(w)
        self.f_support = np.asarray(f_support)  # (k, m)

    def __call__(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        diffs = z[:, None] - self.z_support[None, :]
        exact_mask = np.isclose(diffs, 0.0)
        safe_diffs = np.where(exact_mask, 1.0, diffs)
        inv = np.where(exact_mask, 0.0, 1.0 / safe_diffs)
        num = (self.w[None, :] * inv) @ self.f_support
        den = (self.w[None, :] * inv).sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        if exact_mask.any():
            rows, cols = np.where(exact_mask)
            out[rows, :] = self.f_support[cols, :]
        return out


def _vector_aaa(z_samples, F_samples, rtol=1e-8, max_terms=50):
    z_samples = np.asarray(z_samples, dtype=complex)
    F_samples = np.atleast_2d(np.asarray(F_samples, dtype=complex))
    if F_samples.shape[0] != len(z_samples):
        F_samples = F_samples.T
    M, m = F_samples.shape
    scale = np.max(np.abs(F_samples)) + 1e-300

    support_idx, test_idx = [], list(range(M))
    r_vals = np.tile(F_samples.mean(axis=0, keepdims=True), (M, 1))
    w = np.array([1.0 + 0j])
    z_support = np.array([], dtype=complex)
    f_support = np.zeros((0, m), dtype=complex)

    for _ in range(min(max_terms, M - 1)):
        resid = np.abs(F_samples[test_idx] - r_vals[test_idx])
        j_new = test_idx[np.argmax(resid.max(axis=1))]
        support_idx.append(j_new)
        test_idx.remove(j_new)

        z_support = z_samples[support_idx]
        f_support = F_samples[support_idx, :]
        k = len(support_idx)
        if not test_idx:
            w = np.ones(k) / k
            break

        z_test = z_samples[test_idx]
        F_test = F_samples[test_idx, :]
        denom = z_test[:, None] - z_support[None, :]
        blocks = [(F_test[:, c:c+1] - f_support[None, :, c].reshape(1, k)) / denom
                  for c in range(m)]
        L_stacked = np.vstack(blocks)
        _, _, Vh = np.linalg.svd(L_stacked)
        w = Vh[-1, :].conj()

        fit = _VectorAAA(z_support, w, f_support)
        r_vals = fit(z_samples)
        if np.max(np.abs(F_samples - r_vals)) / scale < rtol:
            break

    return _VectorAAA(z_support, w, f_support)


def _aaa_chebyshev_nodes(a, b, n):
    k = np.arange(n)
    x = np.cos((2*k + 1) / (2*n) * np.pi)
    return 0.5*(b-a)*x + 0.5*(b+a)


def _aaa_bary_weights_1st_kind(n):
    k = np.arange(n)
    theta = (2*k + 1) * np.pi / (2*n)
    return ((-1.0)**k) * np.sin(theta)


def try_aaa_decomposition_1d(joint_expr, x_sym, xi_sym, x_bounds, xi_bounds,
                              n_cheb=24, n_xi_samples=100, rtol=1e-8):
    """1D bivariate rational decomposition via vector-AAA. Returns a plan
    dict (with a fast numpy callable, see aaa_plan_to_callable_1d) or None
    if the quality gate (rel_l2_error > 10*rtol) isn't met."""
    p_lamb = sp.lambdify((x_sym, xi_sym), joint_expr, "numpy")
    x_nodes = _aaa_chebyshev_nodes(*x_bounds, n_cheb)
    xi_samples = np.linspace(*xi_bounds, n_xi_samples).astype(complex)
    XI, X = np.meshgrid(xi_samples, x_nodes, indexing="ij")
    F_samples = np.asarray(p_lamb(X, XI), dtype=complex)
    fit = _vector_aaa(xi_samples, F_samples, rtol=rtol)

    xi_val = np.linspace(*xi_bounds, 3*n_xi_samples + 7).astype(complex)
    XIv, Xv = np.meshgrid(xi_val, x_nodes, indexing="ij")
    F_true_val = np.asarray(p_lamb(Xv, XIv), dtype=complex)
    F_fit_val = fit(xi_val)
    rel_l2_error = (np.linalg.norm(F_fit_val - F_true_val)
                     / (np.linalg.norm(F_true_val) + 1e-300))
    if rel_l2_error > rtol * 10:
        return None
    return {"dim": 1, "fit": fit, "x_nodes": x_nodes,
            "rel_l2_error": rel_l2_error, "n_poles": len(fit.z_support)}


def _aaa_eval_1d(plan, x_eval, xi_eval):
    """Evaluate the AAA-fitted p(x,xi) at arbitrary points (barycentric
    Lagrange interp in x from the exact Chebyshev-node slices, composed
    with the AAA barycentric form in xi)."""
    fit, x_nodes = plan["fit"], plan["x_nodes"]
    xi_eval = np.atleast_1d(np.asarray(xi_eval, dtype=complex))
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    vals_at_nodes = fit(xi_eval)  # (Nxi, M)
    bw = _aaa_bary_weights_1st_kind(len(x_nodes))
    diffs = x_eval[:, None] - x_nodes[None, :]
    exact = np.isclose(diffs, 0.0)
    safe = np.where(exact, 1.0, diffs)
    inv = np.where(exact, 0.0, bw[None, :] / safe)
    den = inv.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (vals_at_nodes @ inv.T) / den[None, :]  # (Nxi, Nx)
    if exact.any():
        x_idx, k_idx = np.where(exact)
        out[:, x_idx] = vals_at_nodes[:, k_idx]
    return out  # (Nxi, Nx)


def aaa_plan_to_callable_1d(plan):
    """Wrap an aaa_decomposition_1d plan as p(x, xi) -> ndarray, matching
    the symbol_func signature kohn_nirenberg_fft/nonperiodic expect."""
    def p_approx(x, xi):
        x = np.asarray(x, dtype=float)
        xi_arr = np.asarray(xi, dtype=complex)
        orig_shape = np.broadcast(x, xi_arr).shape
        xb, xib = np.broadcast_to(x, orig_shape), np.broadcast_to(xi_arr, orig_shape)
        x_flat, xi_flat = xb.ravel(), xib.ravel()
        # _aaa_eval_1d expects distinct (x_eval, xi_eval) axes; evaluate
        # pointwise via the diagonal of the outer evaluation (small arrays
        # in the kohn_nirenberg_fft slow path -- fine at that scale).
        out = np.empty(x_flat.shape, dtype=complex)
        for i in range(x_flat.size):
            out[i] = _aaa_eval_1d(plan, x_flat[i:i+1], xi_flat[i:i+1])[0, 0]
        return out.reshape(orig_shape)
    return p_approx


def try_aaa_decomposition_2d(joint_expr, x_sym, y_sym, xi_sym, eta_sym,
                              x_bounds, y_bounds, xi_bounds, eta_bounds,
                              n_cheb_x=10, n_cheb_y=10,
                              n_xi_samples=30, n_eta_samples=30, rtol=1e-8):
    """2D decomposition via sequential vector-AAA (xi support points chosen
    at a representative eta slice -- see module docstring caveat above;
    stage 2 compresses eta from the EXACT symbolic slice at each xi
    support point). Returns a plan dict or None if the quality gate fails."""
    x_nodes = _aaa_chebyshev_nodes(*x_bounds, n_cheb_x)
    y_nodes = _aaa_chebyshev_nodes(*y_bounds, n_cheb_y)
    Nx, Ny = len(x_nodes), len(y_nodes)
    XX, YY = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    xx_flat, yy_flat = XX.ravel(), YY.ravel()
    p_lamb = sp.lambdify((x_sym, y_sym, xi_sym, eta_sym), joint_expr, "numpy")

    eta_repr = 0.5 * (eta_bounds[0] + eta_bounds[1])
    xi_samples = np.linspace(*xi_bounds, n_xi_samples).astype(complex)
    F1 = np.zeros((n_xi_samples, Nx*Ny), dtype=complex)
    for j, xi_v in enumerate(xi_samples):
        F1[j, :] = p_lamb(xx_flat, yy_flat, xi_v, eta_repr)
    fit_xi = _vector_aaa(xi_samples, F1, rtol=rtol)
    xi_support = fit_xi.z_support

    eta_samples = np.linspace(*eta_bounds, n_eta_samples).astype(complex)
    eta_fits = []
    for xi_l in xi_support:
        expr_l = joint_expr.subs(xi_sym, complex(xi_l))
        p_l_lamb = sp.lambdify((x_sym, y_sym, eta_sym), expr_l, "numpy")
        F2 = np.zeros((n_eta_samples, Nx*Ny), dtype=complex)
        for j, eta_v in enumerate(eta_samples):
            F2[j, :] = p_l_lamb(xx_flat, yy_flat, eta_v)
        eta_fits.append(_vector_aaa(eta_samples, F2, rtol=rtol))

    plan = {"dim": 2, "fit_xi": fit_xi, "xi_support": xi_support,
            "eta_fits": eta_fits, "x_nodes": x_nodes, "y_nodes": y_nodes}

    x_val = np.linspace(x_bounds[0]*0.9, x_bounds[1]*0.9, 6)
    y_val = np.linspace(y_bounds[0]*0.9, y_bounds[1]*0.9, 6)
    xi_val = np.linspace(xi_bounds[0]*0.9, xi_bounds[1]*0.9, 5).astype(complex)
    eta_val = np.linspace(eta_bounds[0]*0.9, eta_bounds[1]*0.9, 5).astype(complex)
    approx = _aaa_eval_2d(plan, x_val, y_val, xi_val, eta_val)
    XIv, ETAv, Xv, Yv = np.meshgrid(xi_val, eta_val, x_val, y_val, indexing="ij")
    true_vals = p_lamb(Xv, Yv, XIv, ETAv)
    rel_err = np.linalg.norm(approx - true_vals) / (np.linalg.norm(true_vals) + 1e-300)
    plan["rel_l2_error"] = rel_err
    if rel_err > rtol * 20:
        return None
    return plan


def _interp_2d_tensor_chebyshev(vals_grid, x_nodes, y_nodes, x_eval, y_eval):
    bwx = _aaa_bary_weights_1st_kind(len(x_nodes))
    bwy = _aaa_bary_weights_1st_kind(len(y_nodes))

    def bary_1d(vals, nodes, bw, eval_pts, axis):
        diffs = eval_pts[:, None] - nodes[None, :]
        exact = np.isclose(diffs, 0.0)
        safe = np.where(exact, 1.0, diffs)
        inv = np.where(exact, 0.0, bw[None, :] / safe)
        den = inv.sum(axis=1)
        num = np.tensordot(inv, vals, axes=([1], [axis]))
        num = np.moveaxis(num, 0, axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / np.expand_dims(den, axis=[a for a in range(num.ndim) if a != axis])
        if exact.any():
            eval_idx, node_idx = np.where(exact)
            src = np.take(vals, node_idx, axis=axis)
            out = np.moveaxis(out, axis, 0)
            src = np.moveaxis(src, axis, 0)
            out[eval_idx] = src[np.arange(len(eval_idx))] if src.ndim == out.ndim else src
            out = np.moveaxis(out, 0, axis)
        return out

    ax_x, ax_y = vals_grid.ndim - 2, vals_grid.ndim - 1
    step1 = bary_1d(vals_grid, x_nodes, bwx, x_eval, axis=ax_x)
    return bary_1d(step1, y_nodes, bwy, y_eval, axis=ax_y)


def _aaa_eval_2d(plan, x_eval, y_eval, xi_eval, eta_eval):
    x_eval = np.atleast_1d(np.asarray(x_eval, dtype=float))
    y_eval = np.atleast_1d(np.asarray(y_eval, dtype=float))
    xi_eval = np.atleast_1d(np.asarray(xi_eval, dtype=complex))
    eta_eval = np.atleast_1d(np.asarray(eta_eval, dtype=complex))
    x_nodes, y_nodes = plan["x_nodes"], plan["y_nodes"]
    Nx, Ny = len(x_nodes), len(y_nodes)
    xi_support, w_xi, L = plan["xi_support"], plan["fit_xi"].w, len(plan["xi_support"])

    q_all = np.zeros((L, len(eta_eval), len(x_eval), len(y_eval)), dtype=complex)
    for l in range(L):
        vals_at_nodes = plan["eta_fits"][l](eta_eval).reshape(len(eta_eval), Nx, Ny)
        q_all[l] = _interp_2d_tensor_chebyshev(vals_at_nodes, x_nodes, y_nodes, x_eval, y_eval)

    out = np.zeros((len(xi_eval), len(eta_eval), len(x_eval), len(y_eval)), dtype=complex)
    for ix, xi_v in enumerate(xi_eval):
        diffs = xi_v - xi_support
        exact = np.isclose(diffs, 0.0)
        if exact.any():
            out[ix] = q_all[np.argmax(exact)]
            continue
        coeff = w_xi / diffs
        out[ix] = np.tensordot(coeff, q_all, axes=([0], [0])) / coeff.sum()
    return out


def aaa_plan_to_callable_2d(plan):
    """Wrap an aaa_decomposition_2d plan as p(x, y, xi, eta) -> ndarray,
    matching the symbol_func signature kohn_nirenberg_fft/nonperiodic
    expect for dim=2."""
    def p_approx(x, y, xi, eta):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        xi_arr = np.asarray(xi, dtype=complex); eta_arr = np.asarray(eta, dtype=complex)
        orig_shape = np.broadcast(x, y, xi_arr, eta_arr).shape
        xb, yb, xib, etab = (np.broadcast_to(a, orig_shape).ravel()
                              for a in (x, y, xi_arr, eta_arr))
        out = np.empty(xb.shape, dtype=complex)
        for i in range(xb.size):
            out[i] = _aaa_eval_2d(plan, xb[i:i+1], yb[i:i+1], xib[i:i+1], etab[i:i+1])[0, 0, 0, 0]
        return out.reshape(orig_shape)
    return p_approx

