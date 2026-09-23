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
physics.py — Lagrangian and Hamiltonian toolkit: Legendre transforms and symbolic PDEs

Overview
--------
The ``physics`` module provides a unified interface for transforming between
Lagrangian and Hamiltonian descriptions of physical systems, both symbolically
and numerically.

It implements:

- the classical Legendre transform for smooth, invertible momentum maps
  ``p ↦ ∂L/∂p``;
- the Legendre–Fenchel transform, i.e. the convex conjugate

    H(x, u, ξ) = sup_p (⟨ξ, p⟩ − L(x, u, p)),

  for non-quadratic or non-invertible cases;
- a numeric Fenchel path for 1D and 2D problems, using either a deterministic
  grid search or SciPy local optimisation when available;
- a heuristic decomposition of a Hamiltonian symbol into polynomial/local and
  non-polynomial/non-local parts;
- generation of formal symbolic PDEs using the placeholder ``ψOp(H, u)`` for a
  pseudo-differential operator with symbol ``H``.

Main classes
------------
LagrangianHamiltonianConverter
    Converts ``L(x, u, p)`` to ``H(x, u, ξ)`` and back.

HamiltonianSymbolicConverter
    Converts a Hamiltonian symbol into a formal PDE representation.

Supported dimensions
--------------------
The public API currently supports one- and two-dimensional coordinate spaces.

For one coordinate, the conjugate momentum is ``ξ``.

For two coordinates, the conjugate momenta are ``ξ`` and ``η``.

Transformation methods
----------------------
``LagrangianHamiltonianConverter.L_to_H`` accepts the following methods:

``method="legendre"``
    Classical symbolic Legendre transform. Preferred when ``L`` is smooth and
    the Hessian ``∂²L/∂p²`` is invertible.

``method="fenchel_symbolic"``
    Symbolic Legendre–Fenchel attempt. Useful for some convex non-quadratic
    cases, but it rejects manifestly nonsmooth expressions such as ``Abs`` or
    ``sign``.

``method="fenchel_numeric"``
    Numerical approximation of the convex conjugate. Returns a formal SymPy
    placeholder together with a callable that evaluates the approximation.

PDE modes
---------
``HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde`` supports:

``"stationary"``

    ψOp(H, u) = E u

``"heat"``

    ∂_t u = −ψOp(H, u)

``"schrodinger"``

    i ∂_t u = ψOp(H, u)

``"wave"``

    ∂_{tt} u + ψOp(H, u) = 0

Mathematical background
-----------------------
In classical mechanics, a Lagrangian ``L(x, u, p)`` depends on position ``x``,
an optional field variable ``u``, and generalised velocities ``p``. The
conjugate momenta are defined by

    ξ = ∂L/∂p.

When this relation can be inverted, the Hamiltonian is obtained from the
Legendre transform

    H(x, u, ξ) = ⟨ξ, p⟩ − L(x, u, p),

with ``p`` expressed in terms of ``ξ``.

If the Hessian ``∂²L/∂p²`` is singular, or if ``L`` is not strictly convex,
the classical transform may fail. The Legendre–Fenchel conjugate

    H(x, u, ξ) = sup_p (⟨ξ, p⟩ − L(x, u, p))

always produces a convex function of ``ξ``, when the supremum is well defined.
This module provides both symbolic and numerical treatments of this supremum.

In pseudo-differential operator formalism, a Hamiltonian ``H(x, ξ)`` can be
viewed as the symbol of an operator. The module splits ``H`` into a polynomial,
local part and a non-polynomial, non-local part, for example terms containing
``√(1 + ξ²)``. This decomposition is heuristic and is used to assemble formal
PDEs involving ``ψOp(H, u)``.

Notes
-----
- The numeric Fenchel transform is computed over a bounded search domain
  ``p ∈ p_bounds``. Therefore it approximates a constrained convex conjugate.
  If the true maximiser lies outside the supplied bounds, the result may be
  inaccurate.

- The symbolic paths may fail when SymPy cannot solve the momentum relations.
  Setting ``force=True`` allows the implementation to attempt recovery, but it
  does not guarantee a mathematically valid transform.

- ``ψOp`` is a formal placeholder. It records the intended operator action but
  does not implement quantisation, composition rules, boundary conditions, or
  functional calculus.

- SciPy is optional. If unavailable, numeric Fenchel methods fall back to grid
  search.

- Matplotlib is imported optionally for possible plotting helpers, but the core
  conversion API does not require it.

Examples
--------
Construct a Hamiltonian from a quadratic Lagrangian:

>>> from sympy import symbols
>>> from physics import LagrangianHamiltonianConverter
>>> x, u = symbols("x u")
>>> p = symbols("p", real=True)
>>> L = p**2 / 2
>>> H, xi = LagrangianHamiltonianConverter.L_to_H(L, (x,), u, (p,))
>>> H  # doctest: +SKIP
xi**2/2

Build a formal Schrödinger-type PDE from a Hamiltonian symbol:

>>> from sympy import Function, symbols
>>> from physics import HamiltonianSymbolicConverter
>>> x, t = symbols("x t")
>>> xi = symbols("xi", real=True)
>>> u = Function("u")(x, t)
>>> H = xi**2
>>> result = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
...     H, (x,), t, u, mode="schrodinger"
... )
>>> result["formal_string"]  # doctest: +SKIP
'i ∂_t u = ψOp(H, u)   (H = H(x; xi))'

References
----------
.. [1] Arnold, V. I. Mathematical Methods of Classical Mechanics,
       Springer-Verlag, 1989 (2nd ed.). §14: Legendre Transform.

.. [2] Rockafellar, R. T. Convex Analysis, Princeton University Press, 1970.
       Chapter 12: Conjugate Functions.

.. [3] Evans, L. C. Partial Differential Equations, American Mathematical
       Society, 2010 (2nd ed.). §4.3: Hamilton-Jacobi Equations.

.. [4] Folland, G. B. Quantum Field Theory: A Tourist Guide for
       Mathematicians, American Mathematical Society, 2008. §1: Legendre
       Transform and Quantisation.
"""
# ==============================================================================
# Explicit Imports (Required for Type Checking)
# ==============================================================================
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload
)
import math as _math

import numpy as np
from numpy import ndarray

from sympy import (
    Expr, Symbol, Function, Matrix, Eq, Derivative, I,
    diff, simplify, degree, hessian, solve, expand, sqrt, Abs, sign, Max, lambdify, N
)

# Optional SciPy for numeric optimization
try:
    from scipy import optimize as _optimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Optional plotting
try:
    import matplotlib.pyplot as _plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ==============================================================================
# Type Aliases
# ==============================================================================
# A callable that takes a float or numpy array and returns a float or numpy array
NumericCallable = Callable[[Union[float, ndarray]], Union[float, ndarray]]


# ==============================================================================
# Lagrangian <-> Hamiltonian
# ==============================================================================
class LagrangianHamiltonianConverter:
    """
    Bidirectional converter between Lagrangian and Hamiltonian descriptions.

    This class implements symbolic and numeric transformations between a
    Lagrangian `L(x, u, p)` and a Hamiltonian `H(x, u, ξ)`, where `p` denotes
    generalised velocities and `ξ` denotes conjugate momenta.

    The classical Legendre transform is used when the momentum map

        p ↦ ∂L/∂p

    can be inverted. When inversion fails, or when `L` is non-quadratic or
    only convex, a Legendre–Fenchel conjugate can be used:

        H(x, u, ξ) = sup_p (⟨ξ, p⟩ − L(x, u, p)).

    Supported transformation methods are:

    - `"legendre"`: classical symbolic Legendre transform.
    - `"fenchel_symbolic"`: symbolic convex-conjugate attempt.
    - `"fenchel_numeric"`: numeric convex-conjugate approximation.

    Only one-dimensional and two-dimensional coordinate sets are currently
    supported by the high-level API.

    Attributes
    ----------
    _numeric_cache : Dict[int, NumericCallable]
        Internal cache mapping symbolic numeric Hamiltonian placeholders to
        their numerical callables. The cache is keyed by object identity and
        is intended for internal use only.

    Notes
    -----
    The numeric Fenchel path returns a SymPy placeholder such as
    `H_numeric(ξ)` together with a callable that evaluates the same object
    numerically. The placeholder is formal; use the returned callable for
    numerical evaluation.

    See Also
    --------
    HamiltonianSymbolicConverter : Converts Hamiltonians into formal PDEs.
    """
    
    _numeric_cache: Dict[int, NumericCallable] = {}

    # --------------------
    # Utilities
    # --------------------
    @staticmethod
    def _is_quadratic_in_p(L_expr: Expr, p_vars: Tuple[Symbol, ...]) -> bool:
        """
        Check whether a Lagrangian is at most quadratic in momentum variables.

        Parameters
        ----------
        L_expr : Expr
            Symbolic Lagrangian expression.
        p_vars : Tuple[Symbol, ...]
            Momentum/velocity symbols with respect to which the degree is
            checked.

        Returns
        -------
        bool
            `True` if `L_expr` is polynomial of degree less than or equal to 2
            in each variable of `p_vars`; otherwise `False`.

        Notes
        -----
        The test is performed variable by variable. Therefore mixed terms such
        as `p_x * p_y` are accepted because each variable appears with degree
        one. Expressions containing non-polynomial objects such as `sqrt`,
        `Abs`, or `sign` are rejected.

        This predicate is used to select the fast analytic Legendre path for
        quadratic Lagrangians.
        """
        for p in p_vars:
            if not L_expr.is_polynomial(p):
                return False
            try:
                deg = degree(L_expr, p)
            except Exception:
                return False
            if deg is None or deg > 2:
                return False
        return True

    @staticmethod
    def _quadratic_legendre(
        L_expr: Expr,
        p_vars: Tuple[Symbol, ...],
        xi_vars: Tuple[Symbol, ...]
    ) -> Tuple[Expr, Dict[Symbol, Expr]]:
        """
        Compute the analytic Legendre transform of a quadratic Lagrangian.

        Parameters
        ----------
        L_expr : Expr
            Symbolic Lagrangian expression, assumed to be quadratic in the
            variables `p_vars`.
        p_vars : Tuple[Symbol, ...]
            Velocity/momentum variables appearing in `L_expr`.
        xi_vars : Tuple[Symbol, ...]
            Conjugate momentum variables used in the returned Hamiltonian.

        Returns
        -------
        H_expr : Expr
            Symbolic Hamiltonian obtained from the quadratic Legendre transform.
        sol_map : Dict[Symbol, Expr]
            Mapping from each velocity variable `p_i` to its expression in
            terms of the conjugate momenta `ξ_i`.

        Raises
        ------
        ValueError
            If the Hessian matrix `∂²L/∂p²` is singular and therefore cannot
            be inverted.

        Notes
        -----
        The method assumes a quadratic form

            L = ½ pᵀ A p + bᵀ p + c,

        where

            A = ∂²L/∂p².

        The conjugate momenta satisfy

            ξ = ∂L/∂p = A p + b,

        and the Hamiltonian is computed as

            H = ⟨ξ, p⟩ − L.
        """
        A = Matrix([[diff(diff(L_expr, p_i), p_j) for p_j in p_vars] for p_i in p_vars])
        grad = Matrix([diff(L_expr, p) for p in p_vars])
        try:
            A_inv = A.inv()
        except Exception:
            raise ValueError("Quadratic analytic path: Hessian A is singular (non-invertible).")
        
        subs_zero = {p: 0 for p in p_vars}
        b_vec = grad.subs(subs_zero)
        xi_vec = Matrix(xi_vars)
        p_solution_vec = A_inv * (xi_vec - b_vec)
        sol = {p_vars[i]: simplify(p_solution_vec[i]) for i in range(len(p_vars))}
        
        H_expr = sum(xi_vars[i] * sol[p_vars[i]] for i in range(len(p_vars))) - simplify(L_expr.subs(sol))
        return simplify(H_expr), sol

    # ----------------------------
    # Numeric Legendre-Fenchel helpers
    # ----------------------------
    @staticmethod
    def _legendre_fenchel_1d_numeric_callable(
        L_func: Callable[[float], float],
        p_bounds: Tuple[float, float] = (-10.0, 10.0),
        n_grid: int = 2001,
        mode: Literal["auto", "scipy", "grid"] = "auto",
        scipy_multistart: int = 5
    ) -> NumericCallable:
        """
        Build a numerical callable for the one-dimensional Legendre–Fenchel transform.

        The returned callable approximates

            H(ξ) = sup_p (ξ p − L(p))

        over the finite interval `p ∈ [p_min, p_max]`.

        Parameters
        ----------
        L_func : Callable[[float], float]
            Scalar numerical Lagrangian `p ↦ L(p)`.
        p_bounds : Tuple[float, float], optional
            Finite search interval `(p_min, p_max)` for the supremum.
            Default is `(-10.0, 10.0)`.
        n_grid : int, optional
            Number of grid points used by the fallback grid search.
            Default is `2001`.
        mode : {"auto", "scipy", "grid"}, optional
            Numerical strategy:

            - `"auto"`: use SciPy when available, otherwise use the grid method.
            - `"scipy"`: request the SciPy optimiser, with grid fallback if
              SciPy is unavailable or optimisation fails.
            - `"grid"`: use only the deterministic grid search.

            Default is `"auto"`.
        scipy_multistart : int, optional
            Number of initial points used by the SciPy L-BFGS-B multistart
            search. Default is `5`.

        Returns
        -------
        NumericCallable
            Callable `H_numeric(xi)` evaluating the approximate convex
            conjugate. It accepts a scalar or an array-like input and returns
            a scalar for scalar input, or a NumPy array for array input.

        Notes
        -----
        Because the supremum is restricted to a bounded interval, the result
        approximates the constrained convex conjugate. If the true maximiser
        lies outside `p_bounds`, the returned value may be inaccurate.

        The grid method has complexity `O(n_grid)` per evaluation and is
        deterministic. The SciPy method uses local optimisation from several
        starting points and may be faster or more accurate for smooth
        problems, but it can fail for highly non-convex or nonsmooth
        Lagrangians.
        """
        pmin, pmax = float(p_bounds[0]), float(p_bounds[1])
        
        def _compute_by_grid(xi: float) -> Tuple[float, float]:
            grid = np.linspace(pmin, pmax, int(n_grid))
            Lvals = np.array([float(L_func(p)) for p in grid], dtype=float)
            S = xi * grid - Lvals
            idx = int(np.argmax(S))
            return float(S[idx]), float(grid[idx])

        def _compute_by_scipy(xi: float) -> Tuple[float, float]:
            if not _HAS_SCIPY:
                return _compute_by_grid(xi)
            def negS(p: List[float]) -> float:
                p0 = float(p[0])
                return -(xi * p0 - float(L_func(p0)))
            
            best_val = -_math.inf
            best_p = None
            inits = np.linspace(pmin, pmax, max(3, int(scipy_multistart)))
            for x0 in inits:
                try:
                    res = _optimize.minimize(negS, x0=[float(x0)], bounds=[(pmin, pmax)], method="L-BFGS-B")
                    if res.success:
                        pstar = float(res.x[0])
                        sval = float(xi * pstar - float(L_func(pstar)))
                        if sval > best_val:
                            best_val = sval
                            best_p = pstar
                except Exception:
                    continue
            if best_p is None:
                return _compute_by_grid(xi)
            return best_val, best_p

        compute = _compute_by_scipy if (_HAS_SCIPY and mode != "grid") else _compute_by_grid

        def H_numeric(xi_in: Union[float, ndarray]) -> Union[float, ndarray]:
            xi_arr = np.atleast_1d(xi_in).astype(float)
            out = np.empty_like(xi_arr, dtype=float)
            for i, xi in enumerate(xi_arr):
                val, _ = compute(float(xi))
                out[i] = val
            if np.isscalar(xi_in):
                return float(out[0])
            return out

        return H_numeric

    @staticmethod
    def _legendre_fenchel_nd_numeric_callable(
        L_func: Callable[[ndarray], float],
        dim: int,
        p_bounds: Tuple[Tuple[float, float], ...],
        n_grid_per_dim: int = 41,
        mode: Literal["auto", "scipy", "grid"] = "auto",
        scipy_multistart: int = 10,
        multistart_restarts: int = 8
    ) -> NumericCallable:
        """
        Build a numerical callable for the multidimensional Legendre–Fenchel transform.

        The returned callable approximates

            H(ξ) = sup_p (⟨ξ, p⟩ − L(p))

        over the hyperrectangle defined by `p_bounds`.

        Parameters
        ----------
        L_func : Callable[[ndarray], float]
            Numerical Lagrangian `p ↦ L(p)`, where `p` is a one-dimensional
            array of length `dim`.
        dim : int
            Dimension of the velocity/momentum space. The high-level API
            currently uses this helper for `dim = 2`.
        p_bounds : Tuple[Tuple[float, float], ...]
            Sequence of `dim` intervals `(p_min_d, p_max_d)` defining the
            bounded search domain.
        n_grid_per_dim : int, optional
            Number of grid points per dimension for the fallback grid search.
            Default is `41`.
        mode : {"auto", "scipy", "grid"}, optional
            Numerical strategy:

            - `"auto"`: use SciPy when available, otherwise use the grid method.
            - `"scipy"`: request the SciPy optimiser, with grid fallback if
              SciPy is unavailable or optimisation fails.
            - `"grid"`: use only the deterministic grid search.

            Default is `"auto"`.
        scipy_multistart : int, optional
            Accepted for API compatibility. The current multidimensional SciPy
            implementation uses `multistart_restarts` instead.
        multistart_restarts : int, optional
            Number of random restarts added to the centre point in the SciPy
            multistart search. Default is `8`.

        Returns
        -------
        NumericCallable
            Callable `H_numeric(xi)` evaluating the approximate convex
            conjugate. For a single momentum vector it returns a scalar. For a
            batch of momentum vectors it returns a one-dimensional NumPy array.

        Notes
        -----
        The grid search scales exponentially with dimension:

            O(n_grid_per_dim^dim).

        Therefore it is practical only for low-dimensional problems.

        The SciPy path minimises the negative objective

            p ↦ −(⟨ξ, p⟩ − L(p))

        using L-BFGS-B from the domain centre and from randomly generated
        starting points. The random generator uses a fixed seed, so repeated
        calls are deterministic for identical inputs.

        As in one dimension, the computed transform is a constrained
        approximation of the true Legendre–Fenchel conjugate.
        """
        pmin_list, pmax_list = zip(*p_bounds)
        pmin = [float(v) for v in pmin_list]
        pmax = [float(v) for v in pmax_list]

        def compute_by_grid(xi_vec: ndarray) -> Tuple[float, ndarray]:
            import itertools
            grids = [np.linspace(pmin[d], pmax[d], int(n_grid_per_dim)) for d in range(dim)]
            best = -_math.inf
            best_p = None
            for pt in itertools.product(*grids):
                pt_arr = np.array(pt, dtype=float)
                sval = float(np.dot(xi_vec, pt_arr) - L_func(pt_arr))
                if sval > best:
                    best = sval
                    best_p = pt_arr
            return best, best_p

        def compute_by_scipy(xi_vec: ndarray) -> Tuple[float, ndarray]:
            if not _HAS_SCIPY:
                return compute_by_grid(xi_vec)
            def negS(p: ndarray) -> float:
                p = np.asarray(p, dtype=float)
                return - (float(np.dot(xi_vec, p)) - float(L_func(p)))
            
            best_val = -_math.inf
            best_p = None
            center = np.array([(pmin[d] + pmax[d]) / 2.0 for d in range(dim)], dtype=float)
            rng = np.random.default_rng(123456)
            inits = [center]
            for _ in range(multistart_restarts):
                r = rng.random(dim)
                start = np.array([pmin[d] + r[d] * (pmax[d] - pmin[d]) for d in range(dim)], dtype=float)
                inits.append(start)
                
            for x0 in inits:
                try:
                    res = _optimize.minimize(
                        negS, x0=x0, 
                        bounds=tuple((pmin[d], pmax[d]) for d in range(dim)),
                        method="L-BFGS-B"
                    )
                    if res.success:
                        pstar = np.asarray(res.x, dtype=float)
                        sval = float(np.dot(xi_vec, pstar) - L_func(pstar))
                        if sval > best_val:
                            best_val = sval
                            best_p = pstar
                except Exception:
                    continue
            if best_p is None:
                return compute_by_grid(xi_vec)
            return best_val, best_p

        compute = compute_by_scipy if (_HAS_SCIPY and mode != "grid") else compute_by_grid

        def H_numeric(xi_in: Union[float, ndarray]) -> Union[float, ndarray]:
            xi_arr = np.atleast_2d(xi_in).astype(float)
            if xi_arr.shape[-1] != dim:
                xi_arr = xi_arr.reshape(-1, dim)
            out = np.empty((xi_arr.shape[0],), dtype=float)
            for i, xivec in enumerate(xi_arr):
                val, _ = compute(xivec)
                out[i] = val
            if out.shape[0] == 1:
                return float(out[0])
            return out

        return H_numeric

    # ----------------------------
    # Main methods
    # ----------------------------
    
    # Overload 1: Symbolic methods return (Expr, Tuple[Symbol, ...])
    @overload
    @staticmethod
    def L_to_H(
        L_expr: Expr, coords: Tuple[Symbol, ...], u: Expr, p_vars: Tuple[Symbol, ...],
        return_symbol_only: bool = False, force: bool = False,
        method: Literal["legendre", "fenchel_symbolic"] = "legendre",
        fenchel_opts: Optional[Dict[str, Any]] = None
    ) -> Tuple[Expr, Tuple[Symbol, ...]]: ...
    """
    Type overload for symbolic Lagrangian-to-Hamiltonian conversion.
    
    Parameters
    ----------
    L_expr : Expr
        Symbolic Lagrangian `L(x, u, p)`.
    coords : Tuple[Symbol, ...]
        Coordinate symbols. Must contain one or two symbols.
    u : Expr
        Field symbol or expression.
    p_vars : Tuple[Symbol, ...]
        Velocity/momentum variables.
    return_symbol_only : bool, optional
        If `True`, remove explicit field dependence by substituting `u = 0`.
    force : bool, optional
        If `True`, attempt to continue after recoverable symbolic failures.
    method : {"legendre", "fenchel_symbolic"}, optional
        Symbolic transformation method.
    fenchel_opts : dict, optional
        Unused for symbolic methods, accepted for API symmetry.
    
    Returns
    -------
    H_expr : Expr
        Symbolic Hamiltonian `H(x, u, ξ)`.
    xi_vars : Tuple[Symbol, ...]
        Conjugate momentum symbols.
    
    Notes
    -----
    This overload documents the return type when `method` is `"legendre"`
    or `"fenchel_symbolic"`. See the implementation docstring for full
    semantics.
    """

    # Overload 2: Numeric method returns (Expr, Tuple[Symbol, ...], NumericCallable)
    @overload
    @staticmethod
    def L_to_H(
        L_expr: Expr, coords: Tuple[Symbol, ...], u: Expr, p_vars: Tuple[Symbol, ...],
        return_symbol_only: bool = False, force: bool = False,
        method: Literal["fenchel_numeric"] = "fenchel_numeric",
        fenchel_opts: Optional[Dict[str, Any]] = None
    ) -> Tuple[Expr, Tuple[Symbol, ...], NumericCallable]: ...
    """
    Type overload for numeric Lagrangian-to-Hamiltonian conversion.
    
    Parameters
    ----------
    L_expr : Expr
        Symbolic Lagrangian `L(x, u, p)`.
    coords : Tuple[Symbol, ...]
        Coordinate symbols. Must contain one or two symbols.
    u : Expr
        Field symbol or expression.
    p_vars : Tuple[Symbol, ...]
        Velocity/momentum variables.
    return_symbol_only : bool, optional
        Does not affect the numerical callable.
    force : bool, optional
        Does not affect the numerical Fenchel path.
    method : {"fenchel_numeric"}, optional
        Numeric convex-conjugate method.
    fenchel_opts : dict, optional
        Options controlling bounds, grid size, and optimisation mode.
    
    Returns
    -------
    H_repr : Expr
        Formal SymPy placeholder such as `H_numeric(ξ)` or
        `H_numeric(ξ, η)`.
    xi_vars : Tuple[Symbol, ...]
        Conjugate momentum symbols.
    H_numeric : NumericCallable
        Numerical evaluator for the approximate Hamiltonian.
    
    Notes
    -----
    This overload documents the return type when `method` is
    `"fenchel_numeric"`. See the implementation docstring for full
    semantics.
    """

    # Actual Implementation
    @staticmethod
    def L_to_H(
        L_expr: Expr, coords: Tuple[Symbol, ...], u: Expr, p_vars: Tuple[Symbol, ...],
        return_symbol_only: bool = False, force: bool = False,
        method: str = "legendre", fenchel_opts: Optional[Dict[str, Any]] = None
    ) -> Union[Tuple[Expr, Tuple[Symbol, ...]], Tuple[Expr, Tuple[Symbol, ...], NumericCallable]]:
        """
        Convert a Lagrangian `L(x, u, p)` into a Hamiltonian `H(x, u, ξ)`.
    
        The transformation can be classical Legendre, symbolic Fenchel, or
        numeric Fenchel. For one coordinate, the conjugate momentum variable is
        `ξ`; for two coordinates, the conjugate momentum variables are `ξ` and
        `η`.
    
        Parameters
        ----------
        L_expr : Expr
            Symbolic Lagrangian expression.
        coords : Tuple[Symbol, ...]
            Spatial or configuration coordinate symbols. Must contain one or
            two symbols.
        u : Expr
            Field symbol or field expression. If `return_symbol_only=True`,
            occurrences of `u` are replaced by zero in symbolic outputs.
        p_vars : Tuple[Symbol, ...]
            Velocity/momentum variables appearing in `L_expr`. This tuple
            should have the same length as `coords`.
        return_symbol_only : bool, optional
            If `True`, remove explicit field dependence by substituting `u = 0`
            in symbolic results. This option does not change the numerical
            callable returned by `"fenchel_numeric"`.
        force : bool, optional
            If `True`, attempt to continue after recoverable symbolic failures,
            for example when the Hessian determinant cannot be verified or when
            the classical Legendre inversion is problematic.
        method : {"legendre", "fenchel_symbolic", "fenchel_numeric"}, optional
            Transformation strategy:
    
            - `"legendre"`: classical symbolic Legendre transform.
            - `"fenchel_symbolic"`: symbolic Legendre–Fenchel attempt.
            - `"fenchel_numeric"`: numerical Legendre–Fenchel approximation.
    
            Default is `"legendre"`.
        fenchel_opts : dict, optional
            Options for the numeric Fenchel transform. Recognised keys depend
            on the spatial dimension.
    
            For one-dimensional problems:
    
            - `p_bounds` : Tuple[float, float]
                Search interval for `p`. Default is `(-10.0, 10.0)`.
            - `n_grid` : int
                Number of grid points for the fallback grid search.
                Default is `2001`.
            - `mode` : {"auto", "scipy", "grid"}
                Numerical strategy. Default is `"auto"`.
            - `scipy_multistart` : int
                Number of SciPy starting points. Default is `8`.
    
            For two-dimensional problems:
    
            - `p_bounds` : Sequence[Tuple[float, float]]
                Search intervals for each momentum component. Default is
                `[(-10.0, 10.0), (-10.0, 10.0)]`.
            - `n_grid_per_dim` : int
                Number of grid points per dimension for the fallback grid
                search. Default is `41`.
            - `mode` : {"auto", "scipy", "grid"}
                Numerical strategy. Default is `"auto"`.
            - `scipy_multistart` : int
                Accepted for API compatibility. Default is `20`.
            - `multistart_restarts` : int
                Number of random restarts for the SciPy multistart search.
                Default is `8`.
    
        Returns
        -------
        H_expr : Expr
            For `"legendre"` and `"fenchel_symbolic"`, the symbolic Hamiltonian.
            For `"fenchel_numeric"`, a formal placeholder such as
            `H_numeric(ξ)` or `H_numeric(ξ, η)`.
        xi_vars : Tuple[Symbol, ...]
            Conjugate momentum symbols: `(ξ,)` in one dimension, or
            `(ξ, η)` in two dimensions.
        H_numeric : NumericCallable
            Numerical evaluator returned only when `method="fenchel_numeric"`.
    
        Raises
        ------
        ValueError
            If `coords` does not have length one or two.
        ValueError
            If the classical Legendre transform is not invertible and
            `force=False`.
        ValueError
            If symbolic solving fails and `force=False`.
        ValueError
            If nonsmooth expressions such as `Abs` or `sign` are supplied to
            the symbolic Fenchel method.
        ValueError
            If `method` is not one of `"legendre"`, `"fenchel_symbolic"`, or
            `"fenchel_numeric"`.
    
        Notes
        -----
        For quadratic Lagrangians, the implementation first tries the analytic
        path based on the Hessian
    
            A = ∂²L/∂p².
    
        If `A` is invertible, the transform is computed from
    
            ξ = A p + b.
    
        The numeric Fenchel transform approximates
    
            H(ξ) = sup_p (⟨ξ, p⟩ − L(p))
    
        over a bounded domain. It is therefore an approximation of the convex
        conjugate, not an exact symbolic object.
    
        When `method="fenchel_numeric"` is used, the returned SymPy expression
        is only a placeholder. The actual numerical evaluation is performed by
        the returned callable. The placeholder is stored in an internal cache
        keyed by object identity.
        """
        dim = len(coords)
        if dim == 1:
            xi_vars = (Symbol('xi', real=True),)
        elif dim == 2:
            xi_vars = (Symbol('xi', real=True), Symbol('eta', real=True))
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")

        # Quadratic fast-path (symbolic)
        if method in ("legendre", "fenchel_symbolic") and LagrangianHamiltonianConverter._is_quadratic_in_p(L_expr, p_vars):
            try:
                H_expr, sol = LagrangianHamiltonianConverter._quadratic_legendre(L_expr, p_vars, xi_vars)
                if return_symbol_only:
                    H_expr = H_expr.subs(u, 0)
                return H_expr, xi_vars
            except Exception:
                if not force and method == "legendre":
                    raise

        # CLASSICAL LEGENDRE
        if method == "legendre":
            H_p = None
            try:
                H_p = hessian(L_expr, p_vars)
                det_H = simplify(H_p.det())
            except Exception:
                det_H = None
                
            if det_H is not None and det_H == 0 and not force:
                raise ValueError("Legendre transform not invertible: Hessian singular. Use force=True or Fenchel method.")
            if det_H is None and not force:
                raise ValueError("Unable to verify Hessian determinant symbolically. Use force=True to attempt solve().")
                
            eqs = [Eq(diff(L_expr, p_vars[i]), xi_vars[i]) for i in range(dim)]
            sol_list = solve(eqs, p_vars, dict=True)
            
            if not sol_list:
                if not force:
                    raise ValueError("Unable to solve symbolic Legendre relations. Use force=True or Fenchel fallback.")
                    
            if sol_list:
                sol = sol_list[0]
                if isinstance(sol, tuple) and len(sol) == len(p_vars):
                    sol = {p_vars[i]: sol[i] for i in range(len(p_vars))}
                H_expr = sum(xi_vars[i]*sol[p_vars[i]] for i in range(dim)) - L_expr.subs(sol)
                H_expr = simplify(H_expr)
                if return_symbol_only:
                    H_expr = H_expr.subs(u, 0)
                return H_expr, xi_vars
            raise ValueError("Legendre inversion failed even with solve().")

        # FENCHEL: symbolic attempt
        if method == "fenchel_symbolic":
            if L_expr.has(Abs) or L_expr.has(sign) or any(
                diff(L_expr, p).has(sign, Abs) for p in p_vars
            ):
                raise ValueError(
                    "Symbolic Fenchel not possible for nonsmooth L (Abs, sign). "
                    "Use method='fenchel_numeric' instead."
                )
                
            eqs = [Eq(diff(L_expr, p_vars[i]), xi_vars[i]) for i in range(dim)]
            sol_list = solve(eqs, p_vars, dict=True)
            
            if sol_list:
                candidates = []
                for sol in sol_list:
                    if isinstance(sol, tuple) and len(sol) == len(p_vars):
                        sol = {p_vars[i]: sol[i] for i in range(len(p_vars))}
                    S_expr = sum(xi_vars[i] * sol[p_vars[i]] for i in range(dim)) - L_expr.subs(sol)
                    candidates.append(simplify(S_expr))
                    
                H_candidates = simplify(Max(*candidates)) if len(candidates) > 1 else candidates[0]
                if return_symbol_only:
                    H_candidates = H_candidates.subs(u, 0)
                return H_candidates, xi_vars
            raise ValueError("Symbolic Fenchel conjugate not found; use method='fenchel_numeric' for numeric computation.")

        # FENCHEL: numeric path
        if method == "fenchel_numeric":
            if fenchel_opts is None:
                fenchel_opts = {}
                
            if dim == 1:
                p_bounds = fenchel_opts.get("p_bounds", (-10.0, 10.0))
                n_grid = int(fenchel_opts.get("n_grid", 2001))
                mode = fenchel_opts.get("mode", "auto")
                scipy_multistart = int(fenchel_opts.get("scipy_multistart", 8))
                
                try:
                    f_lamb = lambdify((p_vars[0],), L_expr, "numpy")
                    def L_func_scalar(p: float) -> float:
                        return float(f_lamb(p))
                except Exception:
                    try:
                        f_lamb = lambdify(p_vars[0], L_expr, "numpy")
                        def L_func_scalar(p: float) -> float:
                            return float(f_lamb(p))
                    except Exception:
                        def L_func_scalar(p: float) -> float:
                            return float(N(L_expr.subs({p_vars[0]: p})))
                            
                H_numeric = LagrangianHamiltonianConverter._legendre_fenchel_1d_numeric_callable(
                    L_func_scalar, p_bounds=p_bounds, n_grid=n_grid, mode=mode,
                    scipy_multistart=scipy_multistart
                )
                H_func = Function("H_numeric")
                H_repr = H_func(xi_vars[0])
                LagrangianHamiltonianConverter._numeric_cache[id(H_repr)] = H_numeric
                return H_repr, xi_vars, H_numeric
                
            else: # dim == 2
                p_bounds = fenchel_opts.get("p_bounds", [(-10.0, 10.0), (-10.0, 10.0)])
                n_grid_per_dim = int(fenchel_opts.get("n_grid_per_dim", 41))
                mode = fenchel_opts.get("mode", "auto")
                scipy_multistart = int(fenchel_opts.get("scipy_multistart", 20))
                multistart_restarts = int(fenchel_opts.get("multistart_restarts", 8))
                
                try:
                    f_lamb = lambdify((p_vars[0], p_vars[1]), L_expr, "numpy")
                    def L_func_nd(p: ndarray) -> float:
                        return float(f_lamb(float(p[0]), float(p[1])))
                except Exception:
                    try:
                        f_lamb = lambdify((p_vars,), L_expr, "numpy")
                        def L_func_nd(p: ndarray) -> float:
                            return float(f_lamb(tuple(float(v) for v in p)))
                    except Exception:
                        def L_func_nd(p: ndarray) -> float:
                            subs_map = {p_vars[i]: float(p[i]) for i in range(2)}
                            return float(N(L_expr.subs(subs_map)))
                            
                H_numeric = LagrangianHamiltonianConverter._legendre_fenchel_nd_numeric_callable(
                    L_func_nd, dim=2, p_bounds=(p_bounds[0], p_bounds[1]),
                    n_grid_per_dim=n_grid_per_dim, mode=mode,
                    scipy_multistart=scipy_multistart, multistart_restarts=multistart_restarts
                )
                H_func = Function("H_numeric")
                H_repr = H_func(*xi_vars)
                LagrangianHamiltonianConverter._numeric_cache[id(H_repr)] = H_numeric
                return H_repr, xi_vars, H_numeric

        raise ValueError(f"Unknown method '{method}'. Choose 'legendre', 'fenchel_symbolic' or 'fenchel_numeric'.")

    @staticmethod
    def H_to_L(
        H_expr: Expr, coords: Tuple[Symbol, ...], u: Expr, xi_vars: Tuple[Symbol, ...], force: bool = False
    ) -> Tuple[Expr, Tuple[Symbol, ...]]:
        """
        Perform the inverse classical Legendre transform from Hamiltonian to Lagrangian.

        Given a Hamiltonian `H(x, u, ξ)`, this method solves

            p = ∂H/∂ξ

        for `ξ = ξ(p)`, then computes

            L(x, u, p) = ⟨p, ξ(p)⟩ − H(x, u, ξ(p)).

        Parameters
        ----------
        H_expr : Expr
            Symbolic Hamiltonian expression.
        coords : Tuple[Symbol, ...]
            Spatial or configuration coordinate symbols. Must contain one or
            two symbols.
        u : Expr
            Field symbol or field expression.
        xi_vars : Tuple[Symbol, ...]
            Conjugate momentum symbols appearing in `H_expr`. In one dimension
            this is usually `(ξ,)`; in two dimensions `(ξ, η)`.
        force : bool, optional
            If `True`, attempt to continue after some symbolic solving
            failures. Default is `False`.

        Returns
        -------
        L_expr : Expr
            Symbolic Lagrangian expression.
        p_vars : Tuple[Symbol, ...]
            Velocity/momentum symbols introduced by the inverse transform:
            `(p,)` in one dimension, or `(p_x, p_y)` in two dimensions.

        Raises
        ------
        ValueError
            If `coords` does not have length one or two.
        ValueError
            If the system `p = ∂H/∂ξ` cannot be solved symbolically and
            `force=False`.
        ValueError
            If the inverse Legendre transform fails because no suitable
            expression `ξ(p)` can be constructed.

        Notes
        -----
        This routine implements only the classical inverse Legendre transform.
        It does not compute a Fenchel biconjugate and does not attempt to
        recover a Lagrangian from a nonsmooth or nonconvex Hamiltonian.

        The inverse transform is well behaved when `H` is smooth and strictly
        convex in `ξ`. In other cases, symbolic solving may fail or return a
        branch-dependent result.
        """
        dim = len(coords)
        if dim == 1:
            p_vars = (Symbol('p', real=True),)
        elif dim == 2:
            p_vars = (Symbol('p_x', real=True), Symbol('p_y', real=True))
        else:
            raise ValueError("Only 1D and 2D are supported.")

        eqs = [Eq(diff(H_expr, xi_vars[i]), p_vars[i]) for i in range(dim)]
        sol = solve(eqs, xi_vars, dict=True)
        
        if not sol:
            if not force:
                raise ValueError("Unable to symbolically solve p = ∂H/∂ξ for ξ. Use force=True.")
            sol = solve(eqs, xi_vars)
            
        if not sol:
            raise ValueError("Inverse Legendre transform failed; cannot find ξ(p).")
            
        sol = sol[0] if isinstance(sol, list) else sol
        if isinstance(sol, tuple) and len(sol) == len(xi_vars):
            sol = {xi_vars[i]: sol[i] for i in range(len(xi_vars))}
            
        if not isinstance(sol, dict):
            if isinstance(sol, list) and sol and isinstance(sol[0], dict):
                sol = sol[0]
            else:
                raise ValueError("Unexpected output from solve(); cannot construct ξ(p).")
                
        L_expr = sum(sol[xi_vars[i]] * p_vars[i] for i in range(dim)) - H_expr.subs(sol)
        return simplify(L_expr), p_vars


# ==============================================================================
# HamiltonianSymbolicConverter
# ==============================================================================
class HamiltonianSymbolicConverter:
    """
    Symbolic converter between Hamiltonians and formal pseudo-differential PDEs.

    This class treats a Hamiltonian `H(x, ξ)` as the symbol of a formal
    pseudo-differential operator. The operator action on a field `u` is
    represented by the placeholder `ψOp(H, u)`.

    The converter can split a Hamiltonian into a polynomial, local part and a
    non-polynomial, non-local part, and then assemble formal evolution or
    eigenvalue equations such as

        ψOp(H, u) = E u,

        i ∂ₜ u = ψOp(H, u),

        ∂ₜ u + ψOp(H, u) = 0.

    Notes
    -----
    The symbol `ψOp` is formal. It encodes the intended pseudo-differential
    operator action but does not by itself provide numerical evaluation,
    functional calculus, or rigorous operator-domain information.

    See Also
    --------
    LagrangianHamiltonianConverter : Constructs Hamiltonians from Lagrangians.
    """

    @staticmethod
    def decompose_hamiltonian(H_expr: Expr, xi_vars: Sequence[Symbol]) -> Tuple[Expr, Expr]:
        """
        Decompose a Hamiltonian into polynomial/local and non-polynomial/non-local parts.

        The decomposition is heuristic. Terms that are polynomial in the
        momentum variables are collected as the local part. Terms containing
        objects such as `sqrt`, `Abs`, or `sign`, or terms that are not
        recognised as polynomial in the momentum variables, are collected as
        the non-local part.

        Parameters
        ----------
        H_expr : Expr
            Symbolic Hamiltonian expression.
        xi_vars : Sequence[Symbol]
            Momentum variables with respect to which polynomiality is tested.

        Returns
        -------
        H_poly : Expr
            Polynomial, local part of the Hamiltonian.
        H_nonlocal : Expr
            Non-polynomial, non-local remainder.

        Notes
        -----
        The method first expands `H_expr` and then inspects ordered terms.
        This is useful for identifying principal polynomial symbols and
        lower-order or non-local remainders, but it is not a rigorous
        pseudo-differential operator calculus.

        In particular, the classification depends on SymPy's notion of
        polynomiality and on the chosen momentum variables. Coefficients
        depending on coordinates or fields are not automatically treated as
        non-local; only the momentum dependence is inspected.
        """
        xi = xi_vars if isinstance(xi_vars, (tuple, list)) else (xi_vars,)
        poly_terms, nonlocal_terms = 0, 0
        H_expand = expand(H_expr)
        
        for term in H_expand.as_ordered_terms():
            if any(func in term.free_symbols for func in [sqrt, Abs, sign]) or \
               term.has(sqrt) or term.has(Abs) or term.has(sign):
                nonlocal_terms += term
            elif all(term.is_polynomial(xi_i) for xi_i in xi):
                poly_terms += term
            else:
                nonlocal_terms += term
                
        return simplify(poly_terms), simplify(nonlocal_terms)

    @classmethod
    def hamiltonian_to_symbolic_pde(
        cls,
        H_expr: Expr,
        coords: Tuple[Symbol, ...],
        t: Symbol,
        u: Expr,
        mode: Literal["stationary", "heat", "schrodinger", "wave"] = "schrodinger"
    ) -> Dict[str, Any]:
        """
        Build a formal symbolic PDE from a Hamiltonian symbol.

        The Hamiltonian is first decomposed into local and non-local parts.
        The full Hamiltonian is then inserted into the placeholder operator
        `ψOp(H, u)`, and a formal equation is constructed according to `mode`.

        Parameters
        ----------
        H_expr : Expr
            Symbolic Hamiltonian `H(x, ξ)` or `H(x, u, ξ)`.
        coords : Tuple[Symbol, ...]
            Spatial coordinate symbols. Must contain one or two symbols.
        t : Symbol
            Time symbol used by evolutionary modes.
        u : Expr
            Unknown field symbol or expression.
        mode : {"stationary", "heat", "schrodinger", "wave"}, optional
            Type of formal PDE to generate:

            - `"stationary"`:

                ψOp(H, u) = E u

            - `"heat"`:

                ∂ₜ u = −ψOp(H, u)

            - `"schrodinger"`:

                i ∂ₜ u = ψOp(H, u)

            - `"wave"`:

                ∂ₜₜ u + ψOp(H, u) = 0

            Default is `"schrodinger"`.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - `"pde"` : Expr
                Formal SymPy equation representing the selected PDE.
            - `"H_poly"` : Expr
                Polynomial/local part of the Hamiltonian.
            - `"H_nonlocal"` : Expr
                Non-polynomial/non-local part of the Hamiltonian.
            - `"formal_string"` : str
                Human-readable formal expression, including the coordinate and
                momentum variable names.
            - `"mode"` : str
                Selected mode.

        Raises
        ------
        ValueError
            If `coords` does not have length one or two.
        ValueError
            If `mode` is not one of `"stationary"`, `"heat"`, `"schrodinger"`,
            or `"wave"`.

        Notes
        -----
        In one spatial dimension, the conjugate momentum variable is named
        `ξ`. In two spatial dimensions, the momentum variables are named `ξ`
        and `η`.

        The returned PDE is formal. The placeholder `ψOp` represents the
        action of a pseudo-differential operator with symbol `H`, but it does
        not perform operator composition, quantisation, boundary-condition
        handling, or functional calculus.
        """
        dim = len(coords)
        if dim == 1:
            xi_vars = (Symbol("xi", real=True),)
        elif dim == 2:
            xi_vars = (Symbol("xi", real=True), Symbol("eta", real=True))
        else:
            raise ValueError("Only 1D and 2D Hamiltonians are supported.")

        H_poly, H_nonlocal = cls.decompose_hamiltonian(H_expr, xi_vars)
        H_total = H_poly + H_nonlocal
        psiOp_H_u = Function("psiOp")(H_total, u)

        if mode == "stationary":
            E = Symbol("E", real=True)
            pde = Eq(psiOp_H_u, E * u)
            formal = "ψOp(H, u) = E u"
        elif mode == "heat":
            pde = Eq(Derivative(u, t), -psiOp_H_u)
            formal = "∂_t u = -ψOp(H, u)"
        elif mode == "schrodinger":
            pde = Eq(I * Derivative(u, t), psiOp_H_u)
            formal = "i ∂_t u = ψOp(H, u)"
        elif mode == "wave":
            pde = Eq(Derivative(u, (t, 2)), -psiOp_H_u)
            formal = "∂_{tt} u + ψOp(H, u) = 0"
        else:
            # The Literal type hint already prevents this at static analysis time, 
            # but we keep the runtime check for safety.
            raise ValueError("mode must be one of: 'stationary', 'heat', 'schrodinger' or 'wave'.")

        coord_str = ", ".join(str(c) for c in coords)
        xi_str = ", ".join(str(x) for x in xi_vars)
        formal += f"   (H = H({coord_str}; {xi_str}))"

        return {
            "pde": simplify(pde),
            "H_poly": H_poly,
            "H_nonlocal": H_nonlocal,
            "formal_string": formal,
            "mode": mode
        }