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
psiop_solver — Time-stepping solvers, propagators, and grid utilities
=====================================================================

Overview
--------
The ``psiop_solver`` submodule orchestrates the numerical application of 
scalar and matrix-valued pseudo-differential operators (from ``psiop`` and 
``psiop.matpsiop``) to evolve fields in time. It provides high-level 
time-stepping routines, amortized exponential propagators, operator-splitting 
schemes for coupled matrix systems, and specialized quasi-linear solvers, 
along with the underlying periodic spatial and frequency grid generation.

The submodule is designed to seamlessly bridge the symbolic asymptotic 
calculus of ``psiop`` with efficient, memory-bounded numerical integration, 
handling both simple linear evolution and complex non-commutative or 
quasi-linear systems.

Main objects and workflows
--------------------------
`PropagatorFamily` and `build_propagator`
    Construct the one-step approximate propagator ``exp(dt · Op[s])`` for a 
    given symbol ``s``. ``PropagatorFamily`` amortizes the cost by building 
    the asymptotic exponential-symbol expansion *once* with ``dt`` left as 
    a free symbolic parameter, allowing subsequent propagators for different 
    step sizes to be generated via a single, cheap ``sympy.subs`` call 
    rather than repeating the full ``compose_asymptotic`` recursion.

`solve_first_order`
    Time-steps the first-order evolution equation ``∂ₜu = Op(s)(u)`` by 
    repeated application of the asymptotic propagator.

`solve_second_order`
    Handles second-order-in-time equations ``∂²ₜu = Op(S)(u)`` by reducing 
    them to a first-order block companion system of double the dimension, 
    then time-stepping via `solve_first_order`.

`solve_matrix_field`
    Extends the propagator machinery to matrix-valued fields ``U(x)``, 
    evolving ``∂ₜU = Op(P)U`` where ``P`` acts on ``U`` from the left 
    (e.g., density matrices or matrix Green's functions).

`solve_sylvester_field`
    Time-steps Sylvester-type equations ``∂ₜU = Op(P)U − U·Op(Q)`` using 
    Lie-Trotter or Strang operator splitting between the independent 
    left-acting ``P`` and right-acting ``Q`` symbols.

`solve_ricci_flow_conformal_2d`
    Integrates the 2D conformal Ricci flow ``∂ₜφ = e⁻²ᵠΔφ``. Because the 
    coefficient depends on the evolving solution, this quasi-linear equation 
    cannot be described by a fixed symbol. It uses an IMEX/Lie splitting: 
    an explicit Euler correction for the deviation from the spatial average, 
    followed by an exact stiff step for the averaged part applied directly 
    via FFT.

Grid utilities
    `make_grid_1d` and `make_grid_2d` construct uniform periodic spatial 
    grids and their associated FFT-ordered angular frequency grids.

Key features
------------
Amortized propagator construction:
    `PropagatorFamily` computes the symbolic ``exp(t · Op[s])`` once; 
    `build_propagator` caches these families process-wide, making parameter 
    sweeps, adaptive stepping, and dt-convergence studies extremely cheap.

Block companion reduction:
    Automatic conversion of scalar or matrix second-order operators into 
    first-order ``2k × 2k`` block systems ``[[0, I], [S, 0]]``.

Operator splitting for non-commutative matrix fields:
    Lie (``𝒪(dt)``) and Strang (``𝒪(dt²)``) splitting for Sylvester-type 
    equations, leveraging the fact that left and right operator actions 
    commute as operations even when the underlying symbols do not.

Quasi-linear IMEX integration:
    Rothe-type linearization for the conformal Ricci flow, freezing the 
    non-linear coefficient once per step and applying the stiff, 
    constant-coefficient remainder exactly via Fourier multipliers.

Robust time-loop management:
    Built-in snapshot cadence, automatic finite-value checking to prevent 
    silent divergence, and unified handling of scalar vs. matrix-valued 
    initial conditions.

Mathematical background and numerical design
--------------------------------------------
Asymptotic exponential symbol
    The propagator ``exp(dt · P)`` is constructed via the truncated 
    asymptotic series:
        exp(tP) ~ I + tP + (t²/2!)P∘P + (t³/3!)P∘P∘P + ...
    where ``∘`` denotes the asymptotic composition (Kohn–Nirenberg or Weyl). 
    For constant-coefficient symbols, all derivative corrections in the 
    composition vanish, making the series exact at any truncation order 
    (reducing to the standard matrix/scalar exponential of the symbol).

Amortized propagator families
    Evaluating the asymptotic series for ``exp(tP)`` requires repeated 
    calls to ``compose_asymptotic``, which involves heavy symbolic 
    differentiation. `PropagatorFamily` performs this expensive recursion 
    exactly once, leaving ``t`` as a symbolic parameter ``_dt_family``. 
    Requesting a propagator for a specific numeric ``dt`` then requires 
    only a single ``sympy.subs`` operation, bypassing the recursion 
    entirely. This is cached process-wide by `build_propagator`.

Second-order block companion reduction
    The equation ``∂²ₜu = Op(S)(u)`` is rewritten as a first-order system 
    for the state vector ``[u, v]ᵀ`` where ``v = ∂ₜu``:
        ∂ₜ [u]   [ 0   I ] [u]
           [v] = [ S   0 ] [v]
    For a ``k × k`` matrix symbol ``S``, the companion matrix is ``2k × 2k``. 
    This allows the full machinery of `solve_first_order` to be reused 
    without deriving a separate second-order integrator.

Sylvester-type operator splitting
    For ``∂ₜU = Op(P)U − U·Op(Q)``, the left action ``L_P(U) = Op(P)U`` 
    and right action ``R_Q(U) = U·Op(Q)`` commute as operations: 
    ``L_P(R_Q(U)) = R_Q(L_P(U))``. 
    If ``P`` and ``Q`` are constant-coefficient (Fourier multipliers), the 
    exact solution over a step ``dt`` is ``U(t) = exp(tP) U(0) exp(-tQ)``. 
    If they depend on ``x``, the underlying scalar operators ``Op[P_ij]`` 
    and ``Op[Q_jk]`` may not commute. The solver then falls back to:
    - Lie-Trotter: ``U^{n+1} = exp(dt·P) exp(-dt·Q) U^n``  (``𝒪(dt)`` error)
    - Strang: ``U^{n+1} = exp(dt/2·P) exp(-dt·Q) exp(dt/2·P) U^n`` (``𝒪(dt²)``)

Quasi-linear IMEX for conformal Ricci flow
    The 2D Ricci flow in conformal gauge ``g = e²ᵠ(dx² + dy²)`` reduces to 
    the scalar quasi-linear heat equation ``∂ₜφ = e⁻²ᵠΔφ``. 
    Because the coefficient ``c(x) = e⁻²ᵠ`` depends on the solution, no 
    fixed symbol describes the operator ahead of time. Each step uses an 
    IMEX/Lie splitting:
    1. Explicit correction: Compute ``Δφ`` via a plain Laplacian. Take one 
       explicit Euler sub-step using the deviation of ``c`` from its spatial 
       average ``c₀``: ``φ_explicit = φ + dt · (c − c₀)Δφ``.
    2. Stiff step: Propagate the spatially averaged, x-independent generator 
       ``c₀·Δ``. Since ``c₀·Δ`` is a pure Fourier multiplier, 
       ``exp(dt · c₀ · Δ)`` has the closed form ``exp(-dt · c₀ · |k|²)`` 
       on the grid. This is applied directly via FFT, avoiding the need to 
       rebuild an asymptotic exponential-symbol propagator at every step. 
       This makes the stiff step exact for the frozen coefficient and 
       significantly cheaper.

Numerical design notes
----------------------
- Grid generation strictly follows FFT ordering (``np.fft.fftfreq``) to 
  ensure compatibility with the fast-path FFT multipliers in ``psiop_apply``.
- Time-stepping loops include automatic ``np.isfinite`` checks to raise 
  ``FloatingPointError`` immediately upon divergence, rather than silently 
  returning ``NaN``-filled arrays.
- Snapshot saving is decoupled from the integration step via a ``save_every`` 
  parameter, allowing high-resolution temporal integration with low-resolution 
  storage.
- Matrix-valued solvers rigorously distinguish between left-action 
  (``apply_matrix_field``) and right-action (``apply_matrix_field_right``), 
  ensuring correct index contraction for density matrices and Green's functions.

References
----------
.. [1] Hairer, E., Lubich, C., and Wanner, G. 
       *Geometric Numerical Integration: Structure-Preserving Algorithms 
       for Ordinary Differential Equations*, Springer, 2006. 
       (For Lie-Trotter and Strang operator splitting).
.. [2] Trefethen, L. N. 
       *Spectral Methods in MATLAB*, SIAM, 2000. 
       (For FFT-based grid generation and Fourier multiplier application).
.. [3] Chorin, A. J., and Marsden, J. E. 
       *A Mathematical Introduction to Fluid Dynamics*, Springer, 1990. 
       (For IMEX and Rothe-type linearization strategies).
"""
import numpy as np
import sympy as sp

# Import operators from the parent package
from . import PseudoDifferentialOperator
from .matpsiop import MatrixPseudoDifferentialOperator

import sympy as sp

# ----------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------

def make_grid_1d(L=10.0, N=256):
    """
    Construct a uniform periodic spatial grid and its associated FFT-ordered
    angular frequency grid in one dimension.

    The spatial domain is [−L, L) discretised into N equally spaced points,
    and the frequency grid covers the discrete angular wavenumbers compatible
    with the FFT ordering:

        x_j = −L + j·Δx,  j = 0, …, N−1,  Δx = 2L/N
        k_m = 2π · fftfreq(N, Δx)

    Parameters
    ----------
    L : float, default 10.0
        Half-length of the spatial domain. The full period is 2L.
    N : int, default 256
        Number of grid points.

    Returns
    -------
    x : ndarray, shape (N,)
        Spatial coordinates in [−L, L).
    kx : ndarray, shape (N,)
        Angular frequency grid in FFT order (radians per unit length).

    Examples
    --------
    >>> x, kx = make_grid_1d(L=5.0, N=128)
    >>> x[0], x[-1]
    (-5.0, 4.921875)
    """
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, kx

def make_grid_2d(L=10.0, N=128):
    """
    Construct uniform periodic spatial grids and their associated FFT-ordered
    angular frequency grids in two dimensions.

    Both axes share the same half-length L and resolution N, yielding a
    square domain [−L, L)² with N × N grid points.

        x_i = −L + i·Δx,  y_j = −L + j·Δy,  Δx = Δy = 2L/N
        kx_m = 2π · fftfreq(N, Δx),  ky_n = 2π · fftfreq(N, Δy)

    Parameters
    ----------
    L : float, default 10.0
        Half-length of the spatial domain along each axis.
    N : int, default 128
        Number of grid points per axis.

    Returns
    -------
    x : ndarray, shape (N,)
        Spatial coordinates along x.
    y : ndarray, shape (N,)
        Spatial coordinates along y.
    kx : ndarray, shape (N,)
        Angular frequency grid along x in FFT order.
    ky : ndarray, shape (N,)
        Angular frequency grid along y in FFT order.

    Examples
    --------
    >>> x, y, kx, ky = make_grid_2d(L=5.0, N=64)
    """
    x, kx = make_grid_1d(L, N)
    y, ky = make_grid_1d(L, N)
    return x, y, kx, ky

# ----------------------------------------------------------------------
# Propagator & Solvers
# ----------------------------------------------------------------------

def make_grids(vars_x, L, N):
    """Build spatial + frequency grids and the meshgrid-ed spatial
    coordinates used to evaluate initial conditions -- factors out the
    grid-setup boilerplate that used to be copy-pasted verbatim in every
    solve_* function below.

    Returns
    -------
    X, Y : ndarray, ndarray or None
        Meshgrid-ed spatial coordinates ('ij' indexing). Y is None in 1D.
    x_grid, y_grid : ndarray, ndarray or None
        1D spatial axes (y_grid is None in 1D).
    kx, ky : ndarray, ndarray or None
        Frequency axes (ky is None in 1D).
    grids : tuple
        (x, kx) in 1D or (x, y, kx, ky) in 2D -- what callers return.
    """
    dim = len(vars_x)
    if dim == 1:
        x_grid, kx = make_grid_1d(L, N)
        return x_grid, None, x_grid, None, kx, None, (x_grid, kx)
    elif dim == 2:
        x_grid, y_grid, kx, ky = make_grid_2d(L, N)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
        return X, Y, x_grid, y_grid, kx, ky, (x_grid, y_grid, kx, ky)
    else:
        raise NotImplementedError("Only 1D and 2D are supported")


def run_time_loop(step_fn, U0, dt, n_steps, save_every, check_finite=True):
    """Repeatedly apply `step_fn(U) -> U_next`, saving a snapshot every
    `save_every` steps (plus the final step and t=0) -- factors out the
    save-cadence bookkeeping that used to be copy-pasted verbatim in every
    solve_* function below.

    Raises
    ------
    FloatingPointError
        If `check_finite` is True and a non-finite value shows up -- avoids
        silently returning a diverged/garbage trajectory.
    """
    t_list = [0.0]
    U_list = [np.asarray(U0).copy()]
    U = U0
    t = 0.0
    for n in range(1, n_steps + 1):
        U = step_fn(U)
        if check_finite and not np.all(np.isfinite(U)):
            raise FloatingPointError(
                f"Non-finite values detected at step {n} (t={t + dt:.6g}); "
                "reduce dt, increase `order`, or check the input symbol."
            )
        t += dt
        if n % save_every == 0 or n == n_steps:
            t_list.append(t)
            U_list.append(np.asarray(U).copy())
    return np.array(t_list), np.array(U_list)


class PropagatorFamily:
    """
    A one-step propagator family exp(dt · Op[s]) for a *fixed* symbol `s`
    but *variable* dt, built so that changing dt is cheap.

    `build_propagator` rebuilds the full truncated exponential series
    (via `exponential_symbol`, i.e. `order` rounds of symbolic
    differentiation and `compose_asymptotic`) from scratch every time it
    is called -- expensive, and wasteful if the same symbol is re-solved
    at several different step sizes (a dt-convergence study, a parameter
    sweep, or an adaptive/embedded stepper that changes dt every few
    steps).

    `exponential_symbol` already accepts `t` as a sympy Symbol, so this
    class builds it ONCE with dt left symbolic, and produces a concrete
    propagator for any numeric dt via a single cheap `sympy.subs` call --
    skipping the `compose_asymptotic` recursion entirely on every
    subsequent request. `build_propagator` below uses this internally
    (via a small process-wide cache), so existing call sites benefit
    automatically with no code changes required.

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Symbol expression (scalar or matrix-valued), same as
        `build_propagator`.
    vars_x : list of sympy.Symbol
        Spatial variables.
    order : int, default 3
        Truncation order for both the Taylor series in t and the
        asymptotic composition at each power.
    quantization : {'kohn-nirenberg', 'weyl'}, default 'kohn-nirenberg'
        Quantization convention for the resulting propagator.
    mode_composition : {'kn', 'weyl'}, default 'kn'
        Composition rule used inside `exponential_symbol`.
    apply_backend : {'peetre', 'direct'}, default 'peetre'
        Numerical backend attached to the propagator operator.
    do_simplify : bool, default True
        Whether to call sympy's `simplify()` while assembling the
        propagator symbol (once when building it, and once inside every
        `compose_asymptotic()` call in the asymptotic expansion loop).
        This does not change the operator being applied -- `lambdify`
        evaluates the same function on an unsimplified expression -- it
        only affects how much symbolic cleanup happens before that.
        `simplify()` is the dominant cost of `build_propagator()` for
        symbols mixing trigonometric and polynomial terms, and its cost
        grows with `order`; set to `False` to skip it and speed up
        propagator construction, at the risk of a larger (but
        numerically equivalent) unsimplified expression tree.

    Examples
    --------
    >>> family = PropagatorFamily(xi**2, [x], order=4)
    >>> prop_1, _, _ = family.propagator_for(0.01)
    >>> prop_2, _, _ = family.propagator_for(0.02)   # cheap: no recomposition
    >>> u1 = prop_1.apply(u, x_grid, kx)
    """

    def __init__(self, s_expr, vars_x, order=3, quantization='kohn-nirenberg',
                 mode_composition='kn', apply_backend='peetre', do_simplify=True):
        self.vars_x = vars_x
        self.order = order
        self.quantization = quantization
        self.mode_composition = mode_composition
        self.apply_backend = apply_backend
        self.is_matrix = isinstance(s_expr, (sp.MatrixBase, list, tuple))
        self.size = None

        self._dt_sym = sp.Symbol('_dt_family', positive=True)

        if self.is_matrix:
            s_mat = sp.Matrix(s_expr)
            self.size = s_mat.shape[0]
            op = MatrixPseudoDifferentialOperator(
                s_mat, vars_x, mode='symbol',
                quantization=quantization, apply_backend=apply_backend,
            )
        else:
            op = PseudoDifferentialOperator(
                s_expr, vars_x, mode='symbol',
                quantization=quantization, apply_backend=apply_backend,
            )

        # The expensive step -- performed exactly ONCE, with dt symbolic.
        self._Esym_generic = op.exponential_symbol(
            t=self._dt_sym, order=order, mode=mode_composition, do_simplify=do_simplify
        )

    def propagator_for(self, dt):
        """
        Return the concrete one-step propagator for a numeric `dt`.

        Cheap: a single `sympy.subs` call -- no `compose_asymptotic`
        recursion. Returns the same `(prop, is_matrix, size)` triple as
        `build_propagator`, so it is a drop-in replacement at the call
        site.
        """
        Esym = self._Esym_generic.subs(self._dt_sym, dt)

        if self.is_matrix:
            prop = MatrixPseudoDifferentialOperator(
                Esym, self.vars_x, mode='symbol',
                quantization=self.quantization, apply_backend=self.apply_backend,
            )
        else:
            prop = PseudoDifferentialOperator(
                Esym, self.vars_x, mode='symbol',
                quantization=self.quantization, apply_backend=self.apply_backend,
            )
        return prop, self.is_matrix, self.size


# Process-wide cache of PropagatorFamily objects, keyed by everything that
# determines the *symbolic* composition (i.e. everything except dt itself).
# `build_propagator` consults this before doing any symbolic work, so
# repeated calls with the same symbol/order/quantization/backend but a
# different `dt` reuse the expensive composition automatically.
_propagator_family_cache = {}


def _propagator_family_key(s_expr, vars_x, order, quantization,
                            mode_composition, apply_backend, do_simplify=True):
    if isinstance(s_expr, (sp.MatrixBase, list, tuple)):
        expr_key = sp.srepr(sp.Matrix(s_expr))
    else:
        expr_key = sp.srepr(s_expr)
    return (expr_key, tuple(str(v) for v in vars_x), order,
            quantization, mode_composition, apply_backend, do_simplify)


def build_propagator(s_expr, vars_x, dt, order=3, quantization='kohn-nirenberg',
                     mode_composition='kn', apply_backend='peetre', do_simplify=True):
    """
    Build the one-step numerical propagator exp(dt · Op[s]) for a
    pseudo-differential operator via truncated asymptotic exponentiation.

    Given a symbol s(x, ξ) (scalar or matrix-valued), this function:
      1. Wraps it into the appropriate operator class.
      2. Computes the symbol of exp(dt · P) via `exponential_symbol(t=dt, order=order)`:

             exp(dt · P) ≈ I + dt·P + (dt²/2!)·P∘P + ⋯ + (dtⁿ/n!)·P^{∘n}

         where each power P^{∘n} is obtained through asymptotic composition.
      3. Wraps the resulting symbol into a new operator ready for `apply()`.

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Symbol expression. A scalar expression produces a
        `PseudoDifferentialOperator`; a matrix (or nested list) produces a
        `MatrixPseudoDifferentialOperator`.
    vars_x : list of sympy.Symbol
        Spatial variables ([x] for 1D, [x, y] for 2D).
    dt : float
        Time-step size used as the evolution parameter t in exp(t·P).
    order : int, default 3
        Truncation order for both the Taylor series in t and the
        asymptotic composition at each power.
    quantization : {'kohn-nirenberg', 'weyl'}, default 'kohn-nirenberg'
        Quantization convention for the resulting propagator.
    mode_composition : {'kn', 'weyl'}, default 'kn'
        Composition rule used inside `exponential_symbol`.
    apply_backend : {'peetre', 'direct'}, default 'peetre'
        Numerical backend attached to the propagator operator.
    do_simplify : bool, default True
        Whether to call sympy's `simplify()` while assembling the
        propagator symbol (once when building it, and once inside every
        `compose_asymptotic()` call in the asymptotic expansion loop).
        This does not change the operator being applied -- `lambdify`
        evaluates the same function on an unsimplified expression -- it
        only affects how much symbolic cleanup happens before that.
        `simplify()` is the dominant cost of `build_propagator()` for
        symbols mixing trigonometric and polynomial terms, and its cost
        grows with `order`; set to `False` to skip it and speed up
        propagator construction, at the risk of a larger (but
        numerically equivalent) unsimplified expression tree.

    Returns
    -------
    prop : PseudoDifferentialOperator or MatrixPseudoDifferentialOperator
        The propagator operator such that `prop.apply(u, …)` advances u
        by one time step dt.
    is_matrix : bool
        True if the propagator is matrix-valued.
    size : int or None
        Matrix dimension N if `is_matrix` is True, else None.

    Notes
    -----
    The propagator is constructed once and reused across all time steps.
    For constant-coefficient symbols the composition is exact (all
    derivative corrections vanish), so the only error is the Taylor
    truncation in dt.

    This function is backed by a process-wide cache of `PropagatorFamily`
    objects (see above), keyed on everything except `dt`. So calling it
    repeatedly with the *same* `s_expr`/`vars_x`/`order`/`quantization`/
    `mode_composition`/`apply_backend` but a *different* `dt` (e.g. a
    step-size sweep, or an adaptive stepper) reuses the expensive
    `exponential_symbol`/`compose_asymptotic` recursion instead of
    redoing it from scratch -- only a cheap `sympy.subs` is performed per
    new `dt`. The very first call for a given symbol still pays the full
    cost.

    Examples
    --------
    >>> prop, is_mat, sz = build_propagator(xi**2, [x], dt=0.01, order=4)
    >>> u_next = prop.apply(u, x_grid, kx)
    """
    key = _propagator_family_key(s_expr, vars_x, order, quantization,
                                  mode_composition, apply_backend, do_simplify)
    family = _propagator_family_cache.get(key)
    if family is None:
        family = PropagatorFamily(
            s_expr, vars_x, order=order, quantization=quantization,
            mode_composition=mode_composition, apply_backend=apply_backend, do_simplify=do_simplify
        )
        _propagator_family_cache[key] = family

    return family.propagator_for(dt)

def solve_first_order(s_expr, vars_x, f, dt, n_steps, order=3,
                      L=10.0, N=256, apply_kwargs=None, save_every=1,
                      quantization='kohn-nirenberg', apply_backend='peetre',
                      check_finite=True, do_simplify=True):
    """
    Solve the first-order evolution equation

        ∂u/∂t = Op[s](u),   u(x, 0) = f(x)

    by repeated application of the asymptotic propagator exp(dt · Op[s]).

    At each time step the field is advanced via

        u^{n+1} = exp(dt · Op[s]) u^n ≈ (I + dt·P + (dt²/2!)P∘P + ⋯) u^n

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Symbol of the spatial operator P. Scalar for a single-field
        equation; matrix-valued for a coupled system.
    vars_x : list of sympy.Symbol
        Spatial variables.
    f : callable
        Initial condition. Must accept (X,) in 1D or (X, Y) in 2D and
        return an ndarray (scalar case) or a list/tuple of ndarrays
        (matrix case with N components).
    dt : float
        Time-step size.
    n_steps : int
        Total number of time steps to evolve.
    order : int, default 3
        Asymptotic expansion order for the propagator construction.
    L : float, default 10.0
        Half-length of the periodic spatial domain.
    N : int, default 256
        Number of grid points per spatial axis.
    apply_kwargs : dict, optional
        Extra keyword arguments forwarded to `prop.apply()` at every
        step (e.g. `boundary_condition`, `freq_window`, `clamp`).
    save_every : int, default 1
        Store the solution snapshot every `save_every` steps.
    quantization : str, default 'kohn-nirenberg'
        Quantization convention.
    apply_backend : str, default 'peetre'
        Numerical application backend.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead of
        silently returning a diverged trajectory.
    do_simplify : bool, default True
        Whether to call sympy's `simplify()` while assembling the
        propagator symbol (once when building it, and once inside every
        `compose_asymptotic()` call in the asymptotic expansion loop).
        This does not change the operator being applied -- `lambdify`
        evaluates the same function on an unsimplified expression -- it
        only affects how much symbolic cleanup happens before that.
        `simplify()` is the dominant cost of `build_propagator()` for
        symbols mixing trigonometric and polynomial terms, and its cost
        grows with `order`; set to `False` to skip it and speed up
        propagator construction, at the risk of a larger (but
        numerically equivalent) unsimplified expression tree.

    Returns
    -------
    t : ndarray, shape (n_saved,)
        Time values at which snapshots were recorded.
    U : ndarray
        Solution snapshots. Shape (n_saved, N) for scalar 1D,
        (n_saved, N, N) for scalar 2D, or (n_saved, size, N…) for
        matrix-valued systems.
    grids : tuple
        The spatial and frequency grids used: (x, kx) in 1D or
        (x, y, kx, ky) in 2D.

    Raises
    ------
    NotImplementedError
        If `vars_x` has length other than 1 or 2.
    ValueError
        If `f` returns the wrong number of components for a matrix system.

    Examples
    --------
    >>> t, U, (x, kx) = solve_first_order(xi**2, [x], lambda X: np.exp(-X**2),
    ...                                     dt=0.01, n_steps=100, N=256)
    """
    apply_kwargs = dict(apply_kwargs or {})
    X, Y, x, y_grid, kx, ky, grids = _make_grids(vars_x, L, N)

    prop, is_matrix, size = build_propagator(
        s_expr, vars_x, dt, order=order,
        quantization=quantization, apply_backend=apply_backend, do_simplify=do_simplify
    )

    f0 = f(X) if Y is None else f(X, Y)
    if is_matrix:
        u0 = np.stack([np.asarray(comp, dtype=complex) for comp in f0])
        if u0.shape[0] != size:
            raise ValueError(f"f must return {size} components, got {u0.shape[0]}")
    else:
        u0 = np.asarray(f0, dtype=complex)

    def step(u):
        u_in = list(u) if is_matrix else u
        result = prop.apply(u_in, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        return np.stack(result) if is_matrix else np.asarray(result)

    t_arr, U_arr = _run_time_loop(step, u0, dt, n_steps, save_every, check_finite)
    return t_arr, U_arr, grids

def _as_component_list(h, X, Y=None, size_hint=1):
    """
    Evaluate a callable `h` on the grid and normalise the result into a
    plain list of component arrays.

    If `h(X)` (or `h(X, Y)` in 2D) already returns a list or tuple, it is
    returned as-is. Otherwise the scalar result is wrapped in a one-element
    list, ensuring downstream code always receives a uniform list interface.

    Parameters
    ----------
    h : callable
        Function of the spatial grid. Signature: h(X) in 1D, h(X, Y) in 2D.
    X : ndarray
        Spatial grid along x (or meshgrid in 2D).
    Y : ndarray, optional
        Spatial grid along y (2D only). If None, `h` is called with X alone.
    size_hint : int, default 1
        Informational hint about the expected number of components
        (not enforced here; callers validate separately).

    Returns
    -------
    list of ndarray
        Component arrays produced by `h`.
    """
    out = h(X) if Y is None else h(X, Y)
    if isinstance(out, (list, tuple)):
        return list(out)
    return [out]

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

def block_matrix_second_order(s_expr):
    """
    Convert a second-order-in-time operator symbol S into a first-order
    block companion system suitable for `solve_first_order`.

    The second-order equation

        ∂²u/∂t² = Op[S](u)

    is rewritten as the first-order system

        ∂/∂t [u]   [ 0   I ] [u]
             [v] = [ S   0 ] [v]

    where v = ∂u/∂t. For a k×k matrix symbol S, the companion matrix has
    dimension 2k × 2k:

        M = [ 0_k   I_k ]
            [ S     0_k ]

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        The operator symbol S (scalar or k×k matrix).

    Returns
    -------
    sympy.Matrix, shape (2k, 2k)
        The block companion matrix M.

    Raises
    ------
    ValueError
        If `s_expr` is a non-square matrix.

    Examples
    --------
    >>> M = block_matrix_second_order(-xi**2)
    >>> M.shape
    (2, 2)
    """
    S = _matrix_of(s_expr)
    if S.shape[0] != S.shape[1]:
        raise ValueError("matrix symbol must be square")
    k = S.shape[0]
    zero_k, eye_k = sp.zeros(k, k), sp.eye(k)
    return zero_k.row_join(eye_k).col_join(S.row_join(zero_k))

def solve_second_order(s_expr, vars_x, f, g, dt, n_steps, order=3,
                       L=10.0, N=256, apply_kwargs=None, save_every=1,
                       quantization='kohn-nirenberg', apply_backend='peetre', do_simplify=True):
    """
    Solve the second-order-in-time evolution equation

        ∂²u/∂t² = Op[S](u),   u(x, 0) = f(x),   ∂u/∂t(x, 0) = g(x)

    by reduction to a first-order block companion system and time-stepping
    with the asymptotic propagator.

    The system is split into

        ∂u/∂t = v,      ∂v/∂t = Op[S](u)

    and solved jointly via `solve_first_order` on the 2k-dimensional
    companion operator. The returned arrays contain only the physical
    field u and its velocity v = ∂u/∂t, not the full state vector.

    Parameters
    ----------
    s_expr : sympy.Expr or sympy.Matrix
        Symbol of the spatial operator S.
    vars_x : list of sympy.Symbol
        Spatial variables.
    f : callable
        Initial displacement u(x, 0). Signature: f(X) in 1D, f(X, Y) in 2D.
        Must return one component per row of S.
    g : callable
        Initial velocity ∂u/∂t(x, 0). Same signature and component
        structure as `f`.
    dt : float
        Time-step size.
    n_steps : int
        Number of time steps.
    order : int, default 3
        Asymptotic order for the propagator.
    L : float, default 10.0
        Spatial domain half-length.
    N : int, default 256
        Grid points per axis.
    apply_kwargs : dict, optional
        Forwarded to `apply()` at each step.
    save_every : int, default 1
        Snapshot cadence.
    quantization : str, default 'kohn-nirenberg'
        Quantization convention.
    apply_backend : str, default 'peetre'
        Numerical backend.
    do_simplify : bool, default True
        Whether to call sympy's `simplify()` while assembling the
        propagator symbol (once when building it, and once inside every
        `compose_asymptotic()` call in the asymptotic expansion loop).
        This does not change the operator being applied -- `lambdify`
        evaluates the same function on an unsimplified expression -- it
        only affects how much symbolic cleanup happens before that.
        `simplify()` is the dominant cost of `build_propagator()` for
        symbols mixing trigonometric and polynomial terms, and its cost
        grows with `order`; set to `False` to skip it and speed up
        propagator construction, at the risk of a larger (but
        numerically equivalent) unsimplified expression tree.

    Returns
    -------
    t : ndarray, shape (n_saved,)
        Time values of recorded snapshots.
    U : ndarray
        Displacement field snapshots u(x, t).
    V : ndarray
        Velocity field snapshots ∂u/∂t(x, t).
    grids : tuple
        Spatial and frequency grids: (x, kx) or (x, y, kx, ky).

    Raises
    ------
    ValueError
        If `f` or `g` produce the wrong number of components.
    """
    is_matrix = isinstance(s_expr, (sp.MatrixBase, list, tuple))
    k = _matrix_of(s_expr).shape[0]
    M = block_matrix_second_order(s_expr)
    
    def f_combined(X, Y=None):
        f_comp = _as_component_list(f, X, Y, size_hint=k)
        g_comp = _as_component_list(g, X, Y, size_hint=k)
        if len(f_comp) != k or len(g_comp) != k:
            raise ValueError(f"f and g must each provide {k} component(s).")
        return f_comp + g_comp
        
    # FIX: Call solve_first_order instead of sympy's algebraic solve()
    t, U_full, grids = solve_first_order(
        M, vars_x, f_combined, dt, n_steps, order=order,
        L=L, N=N, apply_kwargs=apply_kwargs, save_every=save_every,
        quantization=quantization, apply_backend=apply_backend, do_simplify=do_simplify
    )
    
    U = U_full[:, :k, ...]
    V = U_full[:, k:, ...]
    
    if not is_matrix:
        U = U[:, 0, ...]
        V = V[:, 0, ...]
        
    return t, U, V, grids

def solve_matrix_field(s_expr, vars_x, F, dt, n_steps, order=3,
                        L=10.0, N=256, apply_kwargs=None, save_every=1,
                        quantization='kohn-nirenberg', apply_backend='peetre',
                        check_finite=True, do_simplify=True):
    """
    Time-step the matrix-field evolution equation `∂ₜU = P U`, where `P`
    is the pseudo-differential operator with N×N matrix symbol `s_expr`
    and `U(x)` is itself an N×N matrix at every spatial point (e.g. a
    density matrix or matrix Green's function), with `P` acting on `U`
    only from the left: `(P U)_ik = Σⱼ Op[P_ij](U_jk)`. This repeatedly
    applies the exponential propagator `Op(exp(dt·s))` built by
    `build_propagator`, via
    `MatrixPseudoDifferentialOperator.apply_matrix_field`, exactly as
    `solve` does for vector fields via `apply`.

    Parameters
    ----------
    s_expr : sympy.MatrixBase or nested list of sympy.Expr
        N×N matrix symbol `S(x, ξ)` of the generator `P`; must be
        matrix-valued (matrix left-multiplication only makes sense at
        `N > 1` -- use `solve` for a scalar generator).
    vars_x : list of sympy symbols
        Spatial variables (length 1 or 2).
    F : callable
        Initial matrix field `U(·, 0)`, called as `F(X)` in 1D or
        `F(X, Y)` in 2D on the meshgrid-ed spatial coordinates, and
        expected to return an N×N array/nested list of grid-shaped
        components (`F(...)[j][k]`, or an ndarray of shape
        `(N, N, *grid_shape)`).
    dt : float
        Time step.
    n_steps : int
        Number of propagator applications (time steps) to take.
    order : int, optional
        Truncation order of the exponential symbol expansion. Default 3.
    L : float, optional
        Half-width of the spatial domain. Default 10.0.
    N : int, optional
        Number of grid points per axis. Default 256.
    apply_kwargs : dict, optional
        Extra keyword arguments forwarded to `apply_matrix_field`.
    save_every : int, optional
        Save the solution every `save_every` steps (plus the final step
        and `t=0`). Default 1 (save every step).
    quantization : str, optional
        Quantization convention. Default 'kohn-nirenberg'.
    apply_backend : str, optional
        Numerical application backend. Default 'peetre'.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead of
        silently returning a diverged trajectory.
    do_simplify : bool, default True
        Whether to call sympy's `simplify()` while assembling the
        propagator symbol (once when building it, and once inside every
        `compose_asymptotic()` call in the asymptotic expansion loop).
        This does not change the operator being applied -- `lambdify`
        evaluates the same function on an unsimplified expression -- it
        only affects how much symbolic cleanup happens before that.
        `simplify()` is the dominant cost of `build_propagator()` for
        symbols mixing trigonometric and polynomial terms, and its cost
        grows with `order`; set to `False` to skip it and speed up
        propagator construction, at the risk of a larger (but
        numerically equivalent) unsimplified expression tree.

    Returns
    -------
    t_list : ndarray
        Saved time points, starting at 0.
    U_list : ndarray, shape (n_saved, N, N, *grid_shape)
        Saved matrix-field snapshots `U(t)`.
    grids : tuple of ndarray
        `(x, kx)` in 1D or `(x, y, kx, ky)` in 2D, as returned by
        `make_grid_1d`/`make_grid_2d`.

    Raises
    ------
    NotImplementedError
        If `vars_x` has a length other than 1 or 2.
    ValueError
        If `s_expr` is not matrix-valued, or `F` does not return an
        N×N field.
    """
    apply_kwargs = dict(apply_kwargs or {})
    X, Y, x, y_grid, kx, ky, grids = _make_grids(vars_x, L, N)

    prop, is_matrix, size = build_propagator(
        s_expr, vars_x, dt, order=order,
        quantization=quantization, apply_backend=apply_backend, do_simplify=do_simplify
    )
    if not is_matrix:
        raise ValueError(
            "solve_matrix_field requires a matrix symbol; got a scalar "
            "symbol. Use solve_first_order() for scalar/vector fields instead."
        )

    U0 = F(X) if Y is None else F(X, Y)
    U0 = np.asarray(U0, dtype=complex)
    if U0.shape[0] != size or U0.shape[1] != size:
        raise ValueError(
            f"F must return a {size}x{size} matrix field, got shape "
            f"{U0.shape[:2]}."
        )

    def step(U):
        result = prop.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        return np.asarray(result, dtype=complex)

    t_arr, U_arr = _run_time_loop(step, U0, dt, n_steps, save_every, check_finite)
    return t_arr, U_arr, grids

def solve_sylvester_field(P_expr, Q_expr, vars_x, F, dt, n_steps, order=3,
                           splitting='strang', L=10.0, N=256,
                           apply_kwargs=None, save_every=1,
                           quantization='kohn-nirenberg', apply_backend='peetre',
                           check_finite=True, do_simplify=True):
    """
    Time-step the Sylvester-type matrix-field evolution equation
    `∂ₜU = P U − U Q`, where `P` and `Q` are pseudo-differential
    operators with N×N matrix symbols and `U(x)` is an N×N matrix at
    every spatial point.

    Left-multiplication by `P` and right-multiplication by `Q` always
    commute as *operations* (`(P U) Q == P (U Q)`), so when `P` and `Q`
    are x-independent (Fourier multipliers), the exact solution over a
    step `dt` is the closed-form

        U(t) = exp(t P) U(0) exp(-t Q) ,

    obtained by applying the left-propagator `Op(exp(dt·P))`
    (`apply_matrix_field`) and the right-propagator `Op(exp(-dt·Q))`
    (`apply_matrix_field_right`), in either order. When `P` and/or `Q`
    depend on x, `Op[P_ij]` and `Op[Q_jk]` need not commute with each
    other, so the two sub-steps no longer combine exactly; this function
    then falls back to a standard Lie-Trotter (first-order, `O(dt)`
    splitting error) or Strang (second-order, `O(dt^2)`) operator
    splitting between the left and right exponential propagators.

    Parameters
    ----------
    P_expr : sympy.MatrixBase or nested list of sympy.Expr
        N×N matrix symbol `P(x, ξ)` acting on U from the left.
    Q_expr : sympy.MatrixBase or nested list of sympy.Expr
        N×N matrix symbol `Q(x, ξ)` acting on U from the right (with a
        minus sign, as in `∂ₜU = P U − U Q`); must be the same size as
        `P_expr`.
    vars_x : list of sympy symbols
        Spatial variables (length 1 or 2).
    F : callable
        Initial matrix field `U(·, 0)`, called as `F(X)` in 1D or
        `F(X, Y)` in 2D, returning an N×N array/nested list of
        grid-shaped components (as for `solve_matrix_field`).
    dt : float
        Time step.
    n_steps : int
        Number of splitting steps (each advancing `U` by `dt`).
    order : int, optional
        Truncation order of each exponential-symbol expansion. Default 3.
    splitting : str, {'lie', 'strang'}, optional
        Operator-splitting scheme between the `P` (left) and `Q` (right)
        sub-steps:

        - 'lie'    : one full left step `exp(dt·P)`, then one full right
                     step `exp(-dt·Q)` -- first order accurate, `O(dt)`.
        - 'strang' : half left step `exp(dt/2·P)`, full right step
                     `exp(-dt·Q)`, half left step `exp(dt/2·P)` --
                     second order accurate, `O(dt^2)`. Default.
    L : float, optional
        Half-width of the spatial domain. Default 10.0.
    N : int, optional
        Number of grid points per axis. Default 256.
    apply_kwargs : dict, optional
        Extra keyword arguments forwarded to `apply_matrix_field` and
        `apply_matrix_field_right`.
    save_every : int, optional
        Save the solution every `save_every` steps (plus the final step
        and `t=0`). Default 1.
    quantization : str, optional
        Quantization convention. Default 'kohn-nirenberg'.
    apply_backend : str, optional
        Numerical application backend. Default 'peetre'.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead
        of silently returning a diverged trajectory.
    do_simplify : bool, default True
        Whether to call sympy's `simplify()` while assembling the
        propagator symbol (once when building it, and once inside every
        `compose_asymptotic()` call in the asymptotic expansion loop).
        This does not change the operator being applied -- `lambdify`
        evaluates the same function on an unsimplified expression -- it
        only affects how much symbolic cleanup happens before that.
        `simplify()` is the dominant cost of `build_propagator()` for
        symbols mixing trigonometric and polynomial terms, and its cost
        grows with `order`; set to `False` to skip it and speed up
        propagator construction, at the risk of a larger (but
        numerically equivalent) unsimplified expression tree.

    Returns
    -------
    t_list : ndarray
        Saved time points, starting at 0.
    U_list : ndarray, shape (n_saved, N, N, *grid_shape)
        Saved matrix-field snapshots `U(t)`.
    grids : tuple of ndarray
        `(x, kx)` in 1D or `(x, y, kx, ky)` in 2D.

    Raises
    ------
    NotImplementedError
        If `vars_x` has a length other than 1 or 2.
    ValueError
        If `P_expr`/`Q_expr` are not matrix-valued of matching size, if
        `F` does not return an N×N field, or if `splitting` is not
        'lie' or 'strang'.
    """
    if splitting not in ('lie', 'strang'):
        raise ValueError("splitting must be 'lie' or 'strang'.")

    apply_kwargs = dict(apply_kwargs or {})
    X, Y, x, y_grid, kx, ky, grids = _make_grids(vars_x, L, N)

    prop_Q_full, is_matrix_Q, size_Q = build_propagator(
        Q_expr, vars_x, -dt, order=order,
        quantization=quantization, apply_backend=apply_backend, do_simplify=do_simplify
    )
    if splitting == 'lie':
        prop_P_full, is_matrix_P, size_P = build_propagator(
            P_expr, vars_x, dt, order=order,
            quantization=quantization, apply_backend=apply_backend, do_simplify=do_simplify
        )
        prop_P_half = None
    else:  # 'strang'
        prop_P_half, is_matrix_P, size_P = build_propagator(
            P_expr, vars_x, dt / 2.0, order=order,
            quantization=quantization, apply_backend=apply_backend, do_simplify=do_simplify
        )
        prop_P_full = None

    if not (is_matrix_P and is_matrix_Q):
        raise ValueError(
            "solve_sylvester_field requires matrix symbols for both P "
            "and Q; got a scalar symbol for at least one of them."
        )
    if size_P != size_Q:
        raise ValueError(
            f"P_expr ({size_P}x{size_P}) and Q_expr ({size_Q}x{size_Q}) "
            "must have the same size."
        )
    size = size_P

    U0 = F(X) if Y is None else F(X, Y)
    U0 = np.asarray(U0, dtype=complex)
    if U0.shape[0] != size or U0.shape[1] != size:
        raise ValueError(
            f"F must return a {size}x{size} matrix field, got shape "
            f"{U0.shape[:2]}."
        )

    def step(U):
        if splitting == 'lie':
            U = prop_P_full.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
            U = prop_Q_full.apply_matrix_field_right(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        else:  # 'strang'
            U = prop_P_half.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
            U = prop_Q_full.apply_matrix_field_right(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
            U = prop_P_half.apply_matrix_field(U, x, kx, y_grid=y_grid, ky=ky, **apply_kwargs)
        return np.asarray(U, dtype=complex)

    t_arr, U_arr = _run_time_loop(step, U0, dt, n_steps, save_every, check_finite)
    return t_arr, U_arr, grids

def solve_ricci_flow_conformal_2d(phi0, dt, n_steps, order=3, L=8.0, N=64,
                                   save_every=1, quantization='kohn-nirenberg',
                                   apply_backend='peetre', check_finite=True):
    """
    Integrate 2D Ricci flow in conformal gauge on a flat, doubly periodic
    background.

    Writing the metric as `g = e^{2φ}(dx² + dy²)`, the Gauss curvature is
    `K = −e^{−2φ}Δφ` and, since `R_ij = K·g_ij` in two dimensions, the
    tensorial flow `∂ₜg_ij = −2R_ij` collapses to the scalar quasi-linear
    heat equation

        ∂ₜφ = e^{−2φ} Δφ ,

    with `Δ = ∂ₓ² + ∂ᵧ²` the flat Laplacian. This is NOT handled by
    `psiop`'s ordinary linear/matrix machinery: the coefficient
    `e^{−2φ}` depends on the evolving solution itself, so no fixed
    `sympy` symbol `p(x, ξ)` describes the operator ahead of time.

    Instead, each step uses an IMEX/Lie splitting that still reuses
    `psiop`'s exact exponential propagator for the stiff part:

    1. **Explicit correction** — using the *current* coefficient field
       `c(x) = e^{−2φ(x)}`, compute `Δφ` once via a plain (x-independent)
       Laplacian `PseudoDifferentialOperator`, and take one explicit
       Euler sub-step with the deviation of `c` from its spatial average
       `c₀`: `residual = (c − c₀) Δφ`.
    2. **Stiff step** — propagate the spatially averaged, x-independent
       (Fourier-multiplier) generator `c₀·Δ` applied to the
       explicitly-corrected field. Since `c0 * lap_symbol` is a pure
       Fourier multiplier, `exp(dt · c0 · Δ)` has the closed form
       `exp(-dt · c0 · |k|^2)` on this grid -- applied directly via FFT
       instead of rebuilding an asymptotic exponential-symbol propagator
       through `build_propagator` on every step. This is both cheaper
       (no symbolic recomputation per step) and *more* accurate (exact
       exponential instead of an order-`n` Taylor truncation in dt);
       `order` is kept only for backward compatibility and no longer
       affects the stiff step.

    This freezes the quasi-linear coefficient once per step (a Rothe-type
    linearization), so accuracy in `dt` is limited by that freezing, not
    by the stiff step itself, which is now exact for the frozen
    (constant-coefficient) part at every step.

    Parameters
    ----------
    phi0 : callable
        Initial conformal factor, called as `phi0(X, Y)` on the
        meshgrid-ed spatial coordinates; must return a real-valued array.
    dt : float
        Time step.
    n_steps : int
        Number of steps to take.
    order : int, optional
        Kept for backward compatibility; no longer affects the (now
        closed-form) stiff step. Default 3.
    L : float, optional
        Half-width of the (periodic) spatial domain along each axis.
        Default 8.0.
    N : int, optional
        Number of grid points per axis. Default 64.
    save_every : int, optional
        Save every `save_every` steps (plus the final step and `t=0`).
        Default 1.
    quantization : str, optional
        Quantization convention used for the explicit Laplacian
        evaluation. Default 'kohn-nirenberg'.
    apply_backend : str, optional
        Numerical application backend for the explicit Laplacian
        evaluation. Default 'peetre'.
    check_finite : bool, default True
        Raise `FloatingPointError` as soon as a NaN/Inf appears, instead
        of silently returning a diverged trajectory.

    Returns
    -------
    t_list : ndarray
        Saved time points, starting at 0.
    phi_list : ndarray, shape (n_saved, N, N)
        Saved conformal-factor snapshots `φ(x, y, t)`; the metric at each
        saved time is `g(t) = e^{2·phi_list[k]} (dx² + dy²)`.
    grids : tuple of ndarray
        `(x, y)` spatial grids, as returned by `make_grid_2d`.
    """
    x_s, y_s, xi_s, eta_s = sp.symbols('x y xi eta', real=True)
    lap_symbol = -(xi_s**2 + eta_s**2)

    x_grid, y_grid, kx, ky = make_grid_2d(L, N)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2  # |xi|^2 + |eta|^2, i.e. -lap_symbol on this grid

    lap_op = PseudoDifferentialOperator(
        lap_symbol, [x_s, y_s], mode='symbol',
        quantization=quantization, apply_backend=apply_backend,
    )

    # phi is real-valued by construction -- no need for complex dtype here.
    phi = np.asarray(phi0(X, Y), dtype=float)
    t_list = [0.0]
    phi_list = [phi.copy()]
    t = 0.0

    for n in range(1, n_steps + 1):
        c = np.exp(-2 * phi)
        c0 = float(np.mean(c))

        lap_phi = lap_op.apply(phi, x_grid, kx, y_grid=y_grid, ky=ky).real
        residual = (c - c0) * lap_phi
        phi_explicit = phi + dt * residual

        # Closed-form stiff step: exp(dt * c0 * Delta) is exactly the
        # Fourier multiplier exp(-dt * c0 * |k|^2) for this x-independent
        # generator -- apply it directly instead of rebuilding an
        # asymptotic exponential-symbol propagator every step.
        phi_hat = np.fft.fft2(phi_explicit)
        phi_hat *= np.exp(-dt * c0 * K2)
        phi = np.fft.ifft2(phi_hat).real

        if check_finite and not np.isfinite(phi).all():
            raise FloatingPointError(
                f"Non-finite values detected at step {n} (t={t + dt:.6g}); "
                "reduce dt or the spatial resolution N."
            )

        t += dt
        if n % save_every == 0 or n == n_steps:
            t_list.append(t)
            phi_list.append(phi.copy())

    return np.array(t_list), np.array(phi_list), (x_grid, y_grid)

# ======================================================================
# Internal aliases used by the solvers (solve_first_order, etc. call 
# _make_grids and _run_time_loop, but the public functions lack the underscore)
# ======================================================================
_make_grids = make_grids
_run_time_loop = run_time_loop