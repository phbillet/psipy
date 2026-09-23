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
matpsiop — N × N matrix-valued pseudo-differential operators
============================================================

Overview
--------
The `matpsiop` submodule extends the scalar pseudo-differential
framework to systems of coupled partial differential equations,
Dirac-type equations, and matrix-valued fields (such as density
matrices or matrix Green's functions).

It provides the :class:`MatrixPseudoDifferentialOperator` class, which
wraps an N × N matrix of scalar symbols Pᵢⱼ(x, ξ)
and orchestrates their numerical application and symbolic calculus.
Because matrix multiplication is inherently non-commutative, this
module implements specialized asymptotic expansions for composition,
commutators, and exponentials that preserve matrix ordering.

Main object
-----------
:class:`MatrixPseudoDifferentialOperator`
    N × N matrix-valued pseudo-differential operator. Each entry
    Pᵢⱼ is internally wrapped as its own scalar
    :class:`PseudoDifferentialOperator`. This design allows the matrix
    operator to seamlessly inherit all scalar numerical backends
    (FFT, Peetre decomposition, NUFFT, AAA, low-rank) entrywise,
    without requiring separate matrix-specific numerical kernels.

Key features
------------
Vector and matrix field application:
    Entrywise application to vector fields u = (u₁, …, uₙ);
    left and right matrix-multiplication actions on N × N
    matrix-valued fields (essential for Sylvester-type equations
    ∂ₜU = P U - U Q).

Symbol matrix evaluation and spectral analysis:
    Pointwise numerical evaluation of the full (…, N, N) symbol
    on spatial/frequency grids;
    pointwise eigenvalues and eigenvectors of the symbol matrix
    (closed-form for N=2, used to build per-branch Hamiltonians
    for coupled systems).

Non-commutative asymptotic symbolic calculus:
    Matrix composition P ∘ Q (exact for constant-coefficient
    entries, non-commutative even at 0th order);
    matrix commutators [P, Q];
    formal left and right asymptotic inverses (requiring an
    invertible principal symbol);
    formal Hermitian adjoint P*;
    matrix exponential symbols exp(t · Op[P]) for
    propagators of coupled/vector-valued PDE systems.

Mathematical background
-----------------------
Matrix-valued symbols and quantization
    A matrix pseudo-differential operator P acting on a
    vector field u(x) = (u₁(x), …, uₙ(x))ᵀ is defined by
    an N × N matrix of scalar symbols P(x, ξ). The
    Kohn–Nirenberg quantization is applied entrywise:

        (P u)ᵢ(x) = (2π)⁻ᵈ Σⱼ ∫ exp(i x·ξ) Pᵢⱼ(x, ξ) ûⱼ(ξ) dξ

    For matrix-valued fields U(x) (e.g., density matrices), the 
    operator can act from the left (P U) or from the 
    right (U Q), corresponding to the row-wise or 
    column-wise application of the scalar entries.

Asymptotic matrix composition
    The symbol of the composition P ∘ Q
    admits the Kohn–Nirenberg asymptotic expansion:

        (P ∘ Q)ᵢₖ(x, ξ) ∼ Σⱼ Σ_α [ i^(-|α|) / α! ] ∂_ξ^α Pᵢⱼ(x, ξ) · ∂_x^α Qⱼₖ(x, ξ)

    Unlike the scalar case, matrices do not commute. Therefore, the 
    0th-order term of the composition is the standard matrix product 
    P(x, ξ) Q(x, ξ), and P ∘ Q ≠ Q ∘ P even 
    for constant-coefficient symbols. For x-independent symbols, 
    all higher-order derivative terms vanish identically, making the 
    0th-order matrix product the *exact* composition symbol.

Matrix exponential and propagators
    The symbol of the exponential operator exp(t P) is
    computed via a truncated Taylor series using the matrix
    composition rule:

        exp(t P) ∼ I + t P + (t²/2!) (P ∘ P) + …

    This is the fundamental building block for time-stepping coupled 
    systems ∂ₜu = P u. Because the composition 
    is exact for constant-coefficient matrices, the propagator symbol 
    for such systems reduces to the standard matrix exponential 
    exp(t P(ξ)).

Formal adjoints and inverses
    The formal Hermitian adjoint P* is obtained by taking
    the formal adjoint of each scalar entry (complex conjugation +
    asymptotic expansion in |ξ| → ∞), and then
    transposing the resulting matrix.

    Formal asymptotic inverses R (such that 
    P ∘ R ∼ I) require the principal 
    symbol matrix to be invertible (det P ≠ 0). The recursion 
    preserves matrix multiplication order, ensuring the inverse factor 
    remains on the correct side to cancel P.

Numerical design notes
----------------------
Entrywise numerical orchestration:
    :class:`MatrixPseudoDifferentialOperator` introduces no new numerical
    kernels for application. The `apply()` method simply orchestrates
    N² independent calls to the scalar :meth:`PseudoDifferentialOperator.apply`.
    Consequently, it automatically inherits the Peetre decomposition,
    NUFFT, AAA, and low-rank backends for each entry.

Memory and performance:
    Because each of the N² scalar operators is constructed and
    cached independently, memory usage scales linearly with N².
    For large systems, constant-coefficient entries benefit from the
    scalar fast-path FFT multiplier, making matrix application highly
    efficient.

Sylvester-type equations:
    The `apply_matrix_field` (left action) and
    `apply_matrix_field_right` (right action) methods are designed
    to support operator splitting for equations of the form
    ∂ₜU = P U - U Q. Left and right
    actions commute as operations ((P U) Q =
    P (U Q)), enabling efficient Lie-Trotter or
    Strang splitting schemes even when the underlying scalar operators
    do not commute.

Examples
--------
Define a 2×2 constant-coefficient system (e.g., a simplified Dirac 
or coupled wave system) and apply it to a vector field:

>>> import sympy as sp
>>> import numpy as np
>>> from matpsiop import MatrixPseudoDifferentialOperator
>>> 
>>> # Define symbols
>>> x, xi = sp.symbols('x xi', real=True)
>>> 
>>> # Define a 2x2 symbol matrix P(xi)
>>> P_expr = sp.Matrix([[0, 1j*xi], 
...                     [1j*xi, 0]])
>>> 
>>> # Create the operator
>>> P = MatrixPseudoDifferentialOperator(P_expr, vars_x=[x])
>>> 
>>> # Apply to a vector field u = [u1, u2] on a grid
>>> x_grid = np.linspace(0, 2*np.pi, 64, endpoint=False)
>>> kx = np.fft.fftfreq(64, d=2*np.pi/64) * 2 * np.pi
>>> u1 = np.sin(x_grid)
>>> u2 = np.cos(x_grid)
>>> 
>>> # Compute P(u)
>>> v1, v2 = P.apply([u1, u2], x_grid, kx)
"""
from imports import *
# Import core components from the parent package
from . import PseudoDifferentialOperator
from . import _mi_all, _mi_diff, _mi_factorial, _mi_upto
SymbolicMatrix = Union[Matrix, MatrixBase]
ScalarExpr = Expr
VectorField = Sequence[np.ndarray]
MatrixField = Sequence[Sequence[np.ndarray]]
GridArray = np.ndarray


# ============================================================================
# Matrix-Valued (N x N) Pseudodifferential Operators
# ============================================================================
#
# P(x, xi) an N x N matrix of scalar symbols, acting on a vector field
# u = (u_1, ..., u_N) via the same Kohn-Nirenberg / Weyl quantization used
# throughout this module:
#
#     (P u)_i(x) = (1/2pi)^d  sum_j  int P_ij(x, xi) u_hat_j(xi) e^{i x.xi} dxi
#
# i.e. entrywise it's just N^2 ordinary scalar PseudoDifferentialOperator
# applies, summed row-wise -- so `apply()` below adds no new numerics at
# all, it only orchestrates N^2 existing scalar operators on the plain
# periodic/rectangular grid (no cutoffs, no diffuse interfaces). Symbol
# composition and the commutator DO need new code, because matrix
# multiplication doesn't commute: even the leading (0th-order) term of the
# composed symbol is the matrix product P(x,xi) Q(x,xi), not interchangeable
# in either order, and for x-independent ("constant-coefficient") symbols
# that matrix product IS the *exact* composition at any order -- every
# n>=1 term in the KN expansion involves d/dx of a xi-only expression,
# which is identically zero. That exactness is the natural correctness
# check for `compose_asymptotic` below.


class MatrixPseudoDifferentialOperator:
    """
    N x N matrix-valued pseudodifferential operator, built from a sympy
    Matrix of scalar symbols P_ij(x[, y], xi[, eta]).

    Each entry P_ij is wrapped as its own scalar `PseudoDifferentialOperator`
    (same `expr`/`mode`/`quantization`/`apply_backend` conventions), so
    `apply()` reuses the existing periodic FFT / Peetre machinery entirely.
    Entries may depend on `x` (and `y`) for variable-coefficient systems,
    or be `xi`(`, eta`)-only for constant-coefficient ones -- both are
    supported by the same class; nothing here forces one or the other.

    Parameters
    ----------
    P_expr : sympy.Matrix or nested list of sympy.Expr
        N x N matrix of scalar symbol expressions, in the same
        `x[, y], xi[, eta]` convention as `PseudoDifferentialOperator`.
    vars_x : list of sympy symbols
        Spatial variables; length 1 or 2, as for `PseudoDifferentialOperator`.
    mode, quantization, apply_backend, compute_peetre, peetre_options
        Forwarded to each entry's `PseudoDifferentialOperator`.

    Attributes
    ----------
    size : int
        Matrix dimension N (the constructor accepts any square N, but
        `eigen_symbol`'s closed-form path is specific to N=2).
    entries : list of list of PseudoDifferentialOperator
        `entries[i][j]` is the scalar operator for `P_ij`.
    """

    def __init__(
            self,
            P_expr: Union[SymbolicMatrix, Sequence[Sequence[ScalarExpr]]],
            vars_x: Sequence[Symbol],
            mode: str = 'symbol',
            quantization: str = 'kohn-nirenberg',
            apply_backend: str = 'peetre',
            compute_peetre: bool = False,
            peetre_options: Optional[Dict[str, Any]] = None,
        ) -> None:
        P_expr = Matrix(P_expr)
        n, m = P_expr.shape
        if n != m:
            raise ValueError("P_expr must be a square matrix of symbols.")

        self.size = n
        self.dim = len(vars_x)
        self.vars_x = list(vars_x)
        self.mode = mode
        self.quantization = quantization
        self.apply_backend = apply_backend
        self.P_expr = P_expr

        self.entries = [
            [
                PseudoDifferentialOperator(
                    P_expr[i, j], vars_x, mode=mode, quantization=quantization,
                    apply_backend=apply_backend, compute_peetre=compute_peetre,
                    peetre_options=peetre_options,
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        # every entry shares the same grid conventions -- borrow one FFT/IFFT pair
        self.fft = self.entries[0][0].fft
        self.ifft = self.entries[0][0].ifft

    def apply(
            self,
            u: VectorField,
            x_grid: GridArray,
            kx: GridArray,
            y_grid: Optional[GridArray] = None,
            ky: Optional[GridArray] = None,
            **apply_kwargs: Any
        ) -> List[np.ndarray]:
        """
        Apply P(x, xi) to a vector field u = (u_1, ..., u_N).

        Parameters
        ----------
        u : sequence of N ndarrays
            Vector field components, each sampled on the grid.
        x_grid, kx, y_grid, ky
            As for `PseudoDifferentialOperator.apply`.
        **apply_kwargs
            Forwarded to every entry's scalar `apply()` (e.g.
            `boundary_condition`, `freq_window`, ...).

        Returns
        -------
        list of N ndarrays
            `(P u)_i = sum_j Op[P_ij](u_j)`.
        """
        if len(u) != self.size:
            raise ValueError(f"Expected {self.size} vector components, got {len(u)}.")

        out = []
        for i in range(self.size):
            v_i = None
            for j in range(self.size):
                contrib = self.entries[i][j].apply(
                    u[j], x_grid, kx, y_grid=y_grid, ky=ky, **apply_kwargs
                )
                v_i = contrib if v_i is None else v_i + contrib
            out.append(v_i)
        return out

    def symbol_matrix(
            self,
            *args: Union[float, complex, GridArray]
        ) -> Union[SymbolicMatrix, np.ndarray]:
        """
        Numerically evaluate P(x[, y], xi[, eta]) at a point or
        broadcastable arrays, returning an ndarray of shape `(..., N, N)`.
        If called without arguments, returns the symbolic sympy.Matrix.
        
        Parameters
        ----------
        *args
            The point(s) to evaluate at, in the order each entry's
            `p_func` expects: `(x, xi)` for 1D, `(x, y, xi, eta)` for 2D.
            Arguments may be broadcastable ndarrays (e.g. full grids), in
            which case the leading dimensions of the output match their
            broadcast shape.
    
        Returns
        -------
        ndarray, shape (..., N, N) or sympy.Matrix if no args are provided.
        """
        # Fallback for symbolic evaluation when no numerical grid is provided
        if not args:
            return self.P_expr
            
        n = self.size
        sample = np.broadcast(*[np.asarray(a) for a in args])
        P = np.zeros(sample.shape + (n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                P[..., i, j] = self.entries[i][j].p_func(*args)
        return P

    def apply_matrix_field(
            self,
            U: MatrixField,
            x_grid: GridArray,
            kx: GridArray,
            y_grid: Optional[GridArray] = None,
            ky: Optional[GridArray] = None,
            **apply_kwargs: Any
        ) -> List[List[np.ndarray]]:
        """
        Apply P(x, xi) to a matrix-valued field U(x) by left matrix
        multiplication on U's own N x N structure:

            (P U)_ik(x) = sum_j Op[P_ij] (U_jk) (x)

        Unlike `apply()`, which propagates a single vector field
        u = (u_1, ..., u_N), here U itself carries an extra N x N index
        pair (e.g. a density matrix or matrix Green's function) that P
        acts on only from the left. Each column `U[:, k]` of U is an
        ordinary vector field, so this reduces to N independent calls to
        `apply()`, one per column, with the results reassembled into the
        matrix-shaped output; P is applied to U, never U to P.

        Parameters
        ----------
        U : sequence of N sequences of N ndarrays, or ndarray of shape (N, N, ...)
            Matrix-valued field; `U[j][k]` (equivalently `U[j, k]` for an
            ndarray) is the scalar (j, k) component field sampled on the
            grid, so that U plays the role of an N x N matrix at every
            grid point.
        x_grid, kx, y_grid, ky
            As for `apply`.
        **apply_kwargs
            Forwarded to every entry's scalar `apply()` (e.g.
            `boundary_condition`, `freq_window`, ...).

        Returns
        -------
        list of N lists of N ndarrays
            `out[i][k]` holds `(P U)_ik`, indexed the same way as `U`.

        Raises
        ------
        ValueError
            If `U` is not an N x N array of fields, with N equal to
            `self.size`.
        """
        if len(U) != self.size or any(len(row) != self.size for row in U):
            got_cols = len(U[0]) if len(U) else 0
            raise ValueError(
                f"Expected a {self.size}x{self.size} matrix field, got "
                f"{len(U)}x{got_cols}."
            )

        out = [[None] * self.size for _ in range(self.size)]
        for k in range(self.size):
            column = [U[j][k] for j in range(self.size)]
            result_column = self.apply(
                column, x_grid, kx, y_grid=y_grid, ky=ky, **apply_kwargs
            )
            for i in range(self.size):
                out[i][k] = result_column[i]
        return out

    def apply_matrix_field_right(
            self,
            U: MatrixField,
            x_grid: GridArray,
            kx: GridArray,
            y_grid: Optional[GridArray] = None,
            ky: Optional[GridArray] = None,
            **apply_kwargs: Any
        ) -> List[List[np.ndarray]]:
        """
        Apply this operator's symbol Q(x, xi) to a matrix-valued field
        U(x) by right matrix multiplication on U's own N x N structure:

            (U Q)_ik(x) = sum_j Op[Q_jk] (U_ij) (x)

        This is the mirror image of `apply_matrix_field` (which
        left-multiplies by P): here each *row* `U[i, :]` of U is an
        ordinary vector field acted on from the right by Q, which is
        equivalent to the left action of the transposed symbol matrix Q^T
        on that row -- hence the index order `Op[Q_jk]`, not `Op[Q_kj]`,
        so the two methods are genuinely different unless Q is symmetric.

        Together with `apply_matrix_field`, this is the numerical
        primitive needed to time-step Sylvester-type equations
        `d_t U = P U - U Q`, since left- and right-multiplication always
        commute as *operations* (`(P U) Q == P (U Q)`), even though the
        underlying scalar operators `Op[P_ij]` and `Op[Q_jk]` need not
        commute with each other when the symbols depend on x. See
        `solve_sylvester_field` for the corresponding time-stepper.

        Parameters
        ----------
        U : sequence of N sequences of N ndarrays, or ndarray of shape (N, N, ...)
            Matrix-valued field; `U[i][j]` (equivalently `U[i, j]` for an
            ndarray) is the scalar (i, j) component field sampled on the
            grid.
        x_grid, kx, y_grid, ky
            As for `apply`.
        **apply_kwargs
            Forwarded to every entry's scalar `apply()`.

        Returns
        -------
        list of N lists of N ndarrays
            `out[i][k]` holds `(U Q)_ik`, indexed the same way as `U`.

        Raises
        ------
        ValueError
            If `U` is not an N x N array of fields, with N equal to
            `self.size`.
        """
        if len(U) != self.size or any(len(row) != self.size for row in U):
            got_cols = len(U[0]) if len(U) else 0
            raise ValueError(
                f"Expected a {self.size}x{self.size} matrix field, got "
                f"{len(U)}x{got_cols}."
            )

        out = [[None] * self.size for _ in range(self.size)]
        for i in range(self.size):
            for k in range(self.size):
                v_ik = None
                for j in range(self.size):
                    contrib = self.entries[j][k].apply(
                        U[i][j], x_grid, kx, y_grid=y_grid, ky=ky, **apply_kwargs
                    )
                    v_ik = contrib if v_ik is None else v_ik + contrib
                out[i][k] = v_ik
        return out


        """
        Numerically evaluate P(x[, y], xi[, eta]) at a point or
        broadcastable arrays, returning an ndarray of shape `(..., N, N)`.

        Parameters
        ----------
        *args
            The point(s) to evaluate at, in the order each entry's
            `p_func` expects: `(x, xi)` for 1D, `(x, y, xi, eta)` for 2D.
            Arguments may be broadcastable ndarrays (e.g. full grids), in
            which case the leading dimensions of the output match their
            broadcast shape.

        Returns
        -------
        ndarray, shape (..., N, N)
        """
        n = self.size
        sample = np.broadcast(*[np.asarray(a) for a in args])
        P = np.zeros(sample.shape + (n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                P[..., i, j] = self.entries[i][j].p_func(*args)
        return P

    def eigen_symbol(
            self,
            *args: Union[float, complex, GridArray]
        ) -> Tuple[
            Union[np.ndarray, SymbolicMatrix, Dict[Any, Any]],
            Optional[Union[np.ndarray, SymbolicMatrix]]
        ]:
        """
        Compute pointwise eigenvalues and eigenvectors of the symbol matrix.

        Evaluates the eigenvalues and eigenvectors of the N×N symbol matrix
        P(x, ξ) at given spatial/frequency points. If called without arguments,
        computes the exact symbolic eigenvalues and eigenvectors using SymPy.

        Parameters
        ----------
        *args : tuple of float or numpy.ndarray
            The point(s) to evaluate at, in the order each entry's `p_func`
            expects: `(x, ξ)` for 1D, `(x, y, ξ, η)` for 2D. Arguments may be
            broadcastable ndarrays. If omitted, returns symbolic expressions.

        Returns
        -------
        eigvals : numpy.ndarray or sympy.Matrix
            If `*args` is provided, an ndarray of shape `(..., N)` containing
            the eigenvalues at each point. If omitted, a `sympy.Matrix` of
            shape `(N, 1)` (for N=2) or a dictionary of eigenvalues (for N>2).
        eigvecs : numpy.ndarray or sympy.Matrix or None
            If `*args` is provided, an ndarray of shape `(..., N, N)` containing
            the normalized eigenvectors as columns. If omitted and N=2, a
            `sympy.Matrix` of shape `(N, N)`. If omitted and N>2, returns `None`.

        Notes
        -----
        For N=2, closed-form analytical expressions are used for both the
        symbolic and numerical paths to avoid the overhead of general eigenvalue
        solvers. For N>2, the numerical path falls back to `numpy.linalg.eig`,
        and the symbolic path returns `sympy.Matrix.eigenvals()`.
        """
        P = self.symbol_matrix(*args)
        
        # --- Symbolic Path (when no args are provided) ---
        if isinstance(P, MatrixBase):
            if self.size == 2:
                a, b = P[0, 0], P[0, 1]
                c, d = P[1, 0], P[1, 1]
                tr = a + d
                det = a * d - b * c
                disc = sqrt(tr ** 2 - 4 * det)
                lam1 = (tr + disc) / 2
                lam2 = (tr - disc) / 2
                eigvals = Matrix([lam1, lam2])
                
                def _eigvec(lam):
                    v_row0 = Matrix([b, lam - a])
                    v_row1 = Matrix([lam - d, c])
                    # Choose the row that avoids division by zero
                    v = v_row0 if b != 0 else v_row1
                    norm = sqrt(v.dot(v))
                    if norm == 0:
                        return v
                    return v / norm
                
                eigvecs = Matrix.hstack(_eigvec(lam1), _eigvec(lam2))
                return eigvals, eigvecs
            else:
                # General N x N symbolic eigenvalues
                return P.eigenvals(), None

        # --- Numeric Path ---
        if self.size == 2:
            a, b = P[..., 0, 0], P[..., 0, 1]
            c, d = P[..., 1, 0], P[..., 1, 1]
            tr = a + d
            det = a * d - b * c
            disc = np.sqrt((tr ** 2 - 4 * det).astype(complex))
            lam1 = (tr + disc) / 2
            lam2 = (tr - disc) / 2
            eigvals = np.stack([lam1, lam2], axis=-1)
            
            def _eigvec(lam):
                v_row0 = np.stack([b, lam - a], axis=-1)
                v_row1 = np.stack([lam - d, c], axis=-1)
                use_row0 = np.abs(b) >= np.abs(c)
                v = np.where(use_row0[..., None], v_row0, v_row1)
                norm = np.linalg.norm(v, axis=-1, keepdims=True)
                norm = np.where(norm == 0, 1.0, norm)
                return v / norm
                
            eigvecs = np.stack([_eigvec(lam1), _eigvec(lam2)], axis=-1)
            return eigvals, eigvecs
            
        return np.linalg.eig(P)  # general N x N fallback: (eigvals, eigvecs)

    def compose_asymptotic(
            self,
            other: "MatrixPseudoDifferentialOperator",
            order: int = 1,
            mode: Literal['kn', 'weyl'] = 'kn',
            sign_convention: Optional[str] = None,
            do_simplify: bool = True
        ) -> SymbolicMatrix:
        """
        Compose two matrix-valued symbols via the same asymptotic
        Kohn-Nirenberg / Weyl expansion as
        `PseudoDifferentialOperator.compose_asymptotic`, generalized to
        (order-preserving) matrix multiplication: this is the symbol of
        `Op[self] . Op[other]`, and
        `self.compose_asymptotic(other) != other.compose_asymptotic(self)`
        in general -- unlike the scalar case, matrices don't commute even
        at 0th order.

        For x[, y]-independent ("constant-coefficient") symbols this is
        *exact* at any `order`: every n>=1 term involves a spatial
        derivative of a xi-only expression, which vanishes identically, so
        the result reduces to the ordinary matrix product `P(xi) Q(xi)`.

        Parameters
        ----------
        other : MatrixPseudoDifferentialOperator
            Same `size` and `dim` as `self`.
        order, mode, sign_convention and do_simplify
            As for the scalar `compose_asymptotic`.

        Returns
        -------
        sympy.Matrix, shape (size, size)
            The composed symbol.
        """
        assert self.dim == other.dim, "Operator dimensions must match"
        assert self.size == other.size, "Matrix sizes must match"
        if mode not in ('kn', 'weyl'):
            raise ValueError("mode must be 'kn' or 'weyl'")
        dim = self.dim
        if dim not in (1, 2):
            raise NotImplementedError("dim must be 1 or 2")

        P, Q = self.P_expr, other.P_expr
        x_vars = self.vars_x
        xi_vars = symbols('xi eta', real=True) if dim == 2 else (symbols('xi', real=True),)
        sign = -1 if (sign_convention or 'standard') == 'standard' else +1

        result = zeros(self.size, self.size)
        if mode == 'kn':
            for n in range(order + 1):
                for alpha in _mi_all(n, dim):
                    fact = _mi_factorial(alpha)
                    dP = _mi_diff(P, xi_vars, alpha)
                    dQ = _mi_diff(Q, x_vars, alpha)
                    result += (dP * dQ / fact) * (1j) ** (sign * n)
        else:  # 'weyl' -- general dimension-generic Moyal product; this
               # replaces the previous 2D branch, which only differentiated
               # P in (xi, eta) and Q in (x, y) and so dropped the cross
               # terms present in the exact 1D formula (see scalar
               # compose_asymptotic for the same fix and more detail).
            for total in range(order + 1):
                for a_deg in range(total + 1):
                    b_deg = total - a_deg
                    for alpha in _mi_all(a_deg, dim):
                        for beta in _mi_all(b_deg, dim):
                            coeff = (1j / 2) ** total * (-1) ** b_deg
                            coeff /= (_mi_factorial(alpha) * _mi_factorial(beta))
                            dP = _mi_diff(_mi_diff(P, xi_vars, alpha), x_vars, beta)
                            dQ = _mi_diff(_mi_diff(Q, x_vars, alpha), xi_vars, beta)
                            result += coeff * (dP * dQ)  # matrix mult, order preserved

        return simplify(result) if do_simplify else result


    def commutator_symbolic(
            self,
            other: "MatrixPseudoDifferentialOperator",
            order: int = 1,
            mode: Literal['kn', 'weyl'] = 'kn',
            sign_convention: Optional[str] = None
        ) -> SymbolicMatrix:
        """
        Compute the symbol of the commutator [Op[self], Op[other]].

        Generalizes the scalar commutator to matrix-valued pseudo-differential
        operators. Unlike the scalar case, the matrix commutator is generally
        nonzero at 0th order, reducing to the ordinary matrix commutator
        P(x, ξ)Q(x, ξ) - Q(x, ξ)P(x, ξ). Higher orders add the noncommutative
        analogue of the Poisson-bracket correction.

        Parameters
        ----------
        other : MatrixPseudoDifferentialOperator
            The right-hand operator. Must have the same `size` and `dim` as `self`.
        order : int, default=1
            Truncation order for the asymptotic expansion.
        mode : {'kn', 'weyl'}, default='kn'
            Quantization convention ('kohn-nirenberg' or 'weyl').
        sign_convention : str, optional
            Sign convention for the Fourier transform / quantization.
            Forwarded to `compose_asymptotic`.

        Returns
        -------
        sympy.Matrix
            The symbolic matrix representing the commutator [P, Q].
        """
        pq = self.compose_asymptotic(
            other, order=order, mode=mode, sign_convention=sign_convention
        )
        qp = other.compose_asymptotic(
            self, order=order, mode=mode, sign_convention=sign_convention
        )
        return simplify(pq - qp)

    def exponential_symbol(
            self,
            t: Union[float, complex, Symbol] = 1.0,
            order: int = 2,
            mode: Literal['kn', 'weyl'] = 'kn',
            sign_convention: Optional[str] = None,
            do_simplify: bool = True
        ) -> SymbolicMatrix:
        """
        Symbol of `exp(t Op[self])` for the matrix-valued operator, via
        the matrix analogue of `PseudoDifferentialOperator.exponential_symbol`.

        Same truncated power series as the scalar case,

            exp(tP) ~ I + t P + (t^2/2!) P^{.2} + (t^3/3!) P^{.3} + ...

        but "P^{.n}" means the symbol of `Op[P] . Op[P] . ... . Op[P]`
        (n times), computed via the *matrix* `compose_asymptotic` --
        i.e. ordinary matrix multiplication `P @ P` order-corrected by
        the KN/Weyl derivative terms -- since matrix symbols do not
        commute and `Matrix.__mul__(P, P)` alone is only the 0th-order
        (frozen-coefficient) approximation to that composition. Works
        for both `dim == 1` and `dim == 2` -- `compose_asymptotic`
        already branches on dimension internally.

        Parameters
        ----------
        t : float or sympy.Symbol, default=1.0
            Evolution parameter, same conventions as the scalar version
            (e.g. t = -i*tau for exp(-i*tau*H), t = tau for exp(tau*Delta)).
        order : int, default=2
            Truncation order, used both for the outer Taylor series and
            as the `order` passed to each `compose_asymptotic` call.
        mode : {'kn', 'weyl'}, default='kn'
            Quantization convention for the composition (2D Weyl is not
            implemented for matrix symbols -- see `compose_asymptotic`).
        sign_convention : optional
            Forwarded to `compose_asymptotic`.
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
        sympy.Matrix, shape (size, size)
            Truncated symbol of exp(t Op[self]).

        Notes
        -----
        - For x[, y]-independent ("constant-coefficient") `self`,
          `compose_asymptotic` is exact, so this reduces to the exact
          truncated matrix exponential series of `P(xi[, eta])`; compare
          against `scipy.linalg.expm` at sample points to sanity-check.
        - Non-commutativity means `self` and `other`'s roles in each
          `compose_asymptotic` call matter; here every factor is `self`,
          so ordering is moot, but see `commutator_symbolic` for the
          general two-operator case.
        """
        result = eye(self.size) + t * self.P_expr

        current_power = self.P_expr
        for n in range(2, order + 1):
            temp_op = MatrixPseudoDifferentialOperator(
                current_power, self.vars_x, mode='symbol',
                quantization=self.quantization, apply_backend=self.apply_backend,
            )
            current_power = temp_op.compose_asymptotic(
                self, order=order, mode=mode, sign_convention=sign_convention, do_simplify=do_simplify
            )
            coeff = t**n / factorial(n)
            result += coeff * current_power

        return simplify(result) if do_simplify else result

    def _asymptotic_matrix_inverse(
            self,
            order: int,
            side: Literal['left', 'right']
        ) -> SymbolicMatrix:
        """
        Internal engine for computing left or right asymptotic matrix inverses.

        Requires the symbol P(x, ξ) to be invertible as a matrix (det P ≠ 0
        symbolically); P⁻¹ is used as the 0th-order term. Matrix multiplication
        order is preserved: the inverse factor stays on the side that actually
        cancels P in P ∘ R ∼ I (right) or L ∘ P ∼ I (left).

        Parameters
        ----------
        order : int
            The truncation order for the asymptotic expansion.
        side : {'left', 'right'}
            Specifies whether to compute the left or right inverse.

        Returns
        -------
        sympy.Matrix
            The symbolic matrix representing the asymptotic inverse.

        Raises
        ------
        ValueError
            If the symbol is not invertible (det P = 0).
        NotImplementedError
            If the spatial dimension `dim` is not 1 or 2.
        """
        dim = self.dim
        if dim not in (1, 2):
            raise NotImplementedError("dim must be 1 or 2")
        P = self.P_expr
        x_vars = self.vars_x
        xi_vars = symbols('xi eta', real=True) if dim == 2 else (symbols('xi', real=True),)

        try:
            R0 = P.inv()
        except Exception as e:
            raise ValueError(
                "MatrixPseudoDifferentialOperator: symbol is not invertible "
                "(det P == 0 or SymPy could not confirm invertibility); "
                "asymptotic inverses require an invertible principal symbol. "
                f"Original error: {e}"
            )

        R = R0
        for n in range(1, order + 1):
            term = zeros(self.size, self.size)
            for alpha in _mi_upto(n, dim):
                coeff = (1j) ** (-sum(alpha)) / _mi_factorial(alpha)
                if side == 'right':
                    dP = _mi_diff(P, xi_vars, alpha)
                    dR = _mi_diff(R, x_vars, alpha)
                    term += coeff * (dP * dR)
                else:  # 'left'
                    dR = _mi_diff(R, xi_vars, alpha)
                    dP = _mi_diff(P, x_vars, alpha)
                    term += coeff * (dR * dP)
            R = R - (R0 * term if side == 'right' else term * R0)
        return simplify(R)

    def right_inverse_asymptotic(self, order: int = 1) -> SymbolicMatrix:
        """
        Compute the formal right asymptotic inverse R.

        Finds a formal right inverse R such that Op[self] ∘ Op[R] ∼ Id up to
        O(⟨ξ⟩⁻ᵒʳᵈᵉʳ). This is the matrix analogue of the scalar asymptotic
        inverse. Matrix multiplication order is strictly preserved.

        Parameters
        ----------
        order : int, default=1
            The truncation order for the asymptotic expansion.

        Returns
        -------
        sympy.Matrix
            The symbolic matrix representing the right inverse R.

        Raises
        ------
        ValueError
            If the principal symbol matrix P(x, ξ) is not invertible
            (i.e., det P = 0 or SymPy cannot confirm invertibility).
        """
        return self._asymptotic_matrix_inverse(order, side='right')

    def left_inverse_asymptotic(self, order: int = 1) -> SymbolicMatrix:
        """
        Compute the formal left asymptotic inverse L.

        Finds a formal left inverse L such that Op[L] ∘ Op[self] ∼ Id up to
        O(⟨ξ⟩⁻ᵒʳᵈᵉʳ). This is the matrix analogue of the scalar asymptotic
        inverse. Matrix multiplication order is strictly preserved.

        Parameters
        ----------
        order : int, default=1
            The truncation order for the asymptotic expansion.

        Returns
        -------
        sympy.Matrix
            The symbolic matrix representing the left inverse L.

        Raises
        ------
        ValueError
            If the principal symbol matrix P(x, ξ) is not invertible
            (i.e., det P = 0 or SymPy cannot confirm invertibility).
        """
        return self._asymptotic_matrix_inverse(order, side='left')

    def formal_adjoint(self, n_terms: int = 6) -> SymbolicMatrix:
        """
        Compute the formal Hermitian adjoint symbol P*.

        Each entry Pᵢⱼ receives the same scalar treatment as
        `PseudoDifferentialOperator.formal_adjoint` (complex conjugation +
        asymptotic expansion as |ξ| → ∞). The resulting matrix is then
        transposed (P* = (Pᵢⱼ*)ᵀ) because the adjoint operation swaps the
        row and column roles of the symbol, analogous to a plain matrix adjoint.

        Parameters
        ----------
        n_terms : int, default=6
            Number of terms to include in the asymptotic expansion at infinity.

        Returns
        -------
        sympy.Matrix
            The symbolic matrix representing the formal adjoint P*.
        """
        dim = self.dim
        xi_vars = symbols('xi eta', real=True) if dim == 2 else (symbols('xi', real=True),)
        expansion_var = xi_vars[0] if dim == 1 else sqrt(sum(v**2 for v in xi_vars))

        P_star = self.P_expr.applyfunc(
            lambda p_ij: simplify(
                series(conjugate(p_ij), expansion_var, oo, n=n_terms).removeO()
            )
        )
        return P_star.T

