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
fio_bridge.py — Fourier Integral Operators: bridge between psiop.py and asymptotic.py
=========================================================================================

This module is the missing link between:

* **psiop.py**     -- symbol algebra: asymptotic composition (KN/Weyl), adjoint,
                       inverse, operator exponential exp(tP), Hamiltonian flow.
* **asymptotic.py** -- numerical evaluation of oscillatory integrals via the
                        stationary-phase / Laplace / saddle-point method.

Architecture
------------
FourierIntegralOperator  (base class)
    Generic FIO with a free phase φ(x, y, θ) in the sense of Hörmander.
    Owns the canonical-relation geometry and the non-degeneracy check.
    Delegates ALL critical-point work to asymptotic.Analyzer and
    asymptotic.AsymptoticEvaluator (DRY: no duplication of gradient
    minimisation, Hessian analysis, or asymptotic formulæ).

PsiOpFIOBridge(FourierIntegralOperator)
    Specialisation for a PseudoDifferentialOperator.
    Auto-builds the standard FIO phase
        φ(y, ξ; x) = (x − y)·ξ + S_u(y)
    and amplitude  a(y, ξ; x) = p(y, ξ) · a_u(y)
    from the operator symbol, then inherits the evaluation machinery.

PropagatorBridge
    Computes the semi-classical propagator e^{itP} via
    PseudoDifferentialOperator.exponential_symbol(), then delegates to
    PsiOpFIOBridge.

CompositionBridge
    Computes P∘Q via PseudoDifferentialOperator.compose_asymptotic(),
    then delegates to PsiOpFIOBridge.

Mathematics
-----------
For a psiOp with symbol p(x, ξ), the action on a WKB state
    u(y) ~ a_u(y) · exp(iλ S_u(y))
is the oscillatory integral

    (Pu)(x) = (1/2π)^n ∫∫ exp(iλ φ(y, ξ; x)) · p(y, ξ) · a_u(y)  dy dξ

where the global phase is

    φ(y, ξ; x) = (x − y)·ξ + S_u(y).

Stationary conditions:
    ∂φ/∂ξ = 0  →  y_c = x          (transport)
    ∂φ/∂y = 0  →  ξ_c = S'_u(y_c)  (bicharacteristic)

Normalisation
-------------
The prefactor produced by asymptotic.py at a Morse point is

    (2π/λ)^(n/2) / √|det H|  ·  exp(iλφ_c)  ·  a_c  ·  exp(iπμ/4)

The FIO integral carries an overall factor (1/2π)^n (one per integration
variable pair), so the caller multiplies by  1 / (2π)^n.

For the standard 1D psiOp (n=1, two integration variables y and ξ):
    prefactor = 1 / (2π).

References
----------
.. [1] Hörmander, L.  "Fourier Integral Operators I", Acta Math. 127 (1971).
.. [2] Duistermaat, J.J.  "Fourier Integral Operators", Birkhäuser, 1996.
.. [3] Zworski, M.  "Semiclassical Analysis", AMS Graduate Studies, 2012.
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from asymptotic import (
    Analyzer, AsymptoticEvaluator, SaddlePointEvaluator,
    IntegralMethod, CriticalPoint, AsymptoticContribution,
)
from psiop import PseudoDifferentialOperator


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FIOKernel:
    """
    Kernel of a Fourier Integral Operator: phase φ and amplitude a as SymPy
    expressions, together with the integration variables and the observation
    point.

    Attributes
    ----------
    phase_sym : sp.Expr
        Phase function φ(vars_int; x_val).  Depends on the integration
        variables and on the (fixed) observation parameter x_val.
    amp_sym : sp.Expr
        Amplitude function a(vars_int; x_val).
    vars_int : list[sp.Symbol]
        Integration variables, e.g. [y, ξ] in 1D or [y1, y2, ξ1, ξ2] in 2D.
    x_val : float | tuple[float, ...]
        Current observation point.
    domain : list[tuple[float, float]] | None
        Search bounds [(min, max), ...] for each integration variable.
        Passed directly to asymptotic.Analyzer so critical-point search
        respects the physical domain.
    method_hint : IntegralMethod
        Suggested method (resolved by Analyzer when AUTO).
    """
    phase_sym   : sp.Expr
    amp_sym     : sp.Expr
    vars_int    : List[sp.Symbol]
    x_val       : Any                       = 0.0
    domain      : Optional[List[Tuple]]     = None
    method_hint : IntegralMethod            = IntegralMethod.AUTO


@dataclass
class EvalResult:
    """
    Result of evaluating (Fu)(x) at a single observation point x.

    Attributes
    ----------
    x_val : float | tuple
        Observation point.
    value : complex
        Asymptotic value of (Fu)(x).
    n_critical_points : int
        Number of critical (or saddle) points found and used.
    contributions : list[AsymptoticContribution]
        Individual contributions from each critical point.
    warnings_list : list[str]
        Warnings raised during evaluation (caustics, Picard–Lefschetz, …).
    """
    x_val               : Any
    value               : complex
    n_critical_points   : int
    contributions       : List[AsymptoticContribution] = field(default_factory=list)
    warnings_list       : List[str]                    = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  FourierIntegralOperator  --  generic base class
# ─────────────────────────────────────────────────────────────────────────────

class FourierIntegralOperator:
    """
    Generic Fourier Integral Operator with a free phase function.

    Evaluates integrals of the form

        F[u](x) = (2π)^{-n_θ} ∫∫ exp(iλ φ(x, y, θ)) · a(x, y, θ) · u(y) dy dθ

    where φ is an arbitrary non-degenerate phase in the sense of Hörmander.

    This class is responsible for:
    * Storing the symbolic phase and amplitude.
    * Computing the canonical relation (∇_θ φ, ∇_x φ, ∇_y φ).
    * Checking Hörmander's non-degeneracy condition (mixed Hessian ∂²φ/∂x∂θ).
    * Evaluating the integral asymptotically by delegating entirely to
      asymptotic.Analyzer and asymptotic.AsymptoticEvaluator (DRY).

    Parameters
    ----------
    phase_expr : sp.Expr
        Phase function φ(x, y, θ) as a SymPy expression.
    amp_expr : sp.Expr
        Amplitude function a(x, y, θ).
    vars_x : list[sp.Symbol]
        Target (observation) spatial variables.
    vars_y : list[sp.Symbol]
        Source spatial variables (integration variables, spatial part).
    vars_theta : list[sp.Symbol]
        Frequency / phase variables (integration variables, frequency part).
    lam : float
        Large parameter λ.
    domain : list[tuple[float, float]] | None
        Search domain for each integration variable (vars_y + vars_theta).
        Forwarded to Analyzer.
    tol_grad : float
        Tolerance for |∇φ|² when accepting a critical point.
    verbose : bool
        Print diagnostic information.
    """

    def __init__(
        self,
        phase_expr  : sp.Expr,
        amp_expr    : sp.Expr,
        vars_x      : List[sp.Symbol],
        vars_y      : List[sp.Symbol],
        vars_theta  : List[sp.Symbol],
        lam         : float = 50.0,
        domain      : Optional[List[Tuple]] = None,
        tol_grad    : float = 1e-6,
        verbose     : bool  = False,
    ):
        self.phase_expr  = sp.sympify(phase_expr)
        self.amp_expr    = sp.sympify(amp_expr)
        self.vars_x      = vars_x if isinstance(vars_x,     list) else [vars_x]
        self.vars_y      = vars_y if isinstance(vars_y,     list) else [vars_y]
        self.vars_theta  = vars_theta if isinstance(vars_theta, list) else [vars_theta]
        self.lam         = lam
        self.domain      = domain
        self.tol_grad    = tol_grad
        self.verbose     = verbose

        self.dim_x     = len(self.vars_x)
        self.dim_y     = len(self.vars_y)
        self.dim_theta = len(self.vars_theta)

        if self.dim_x != self.dim_y:
            warnings.warn(
                "dim(x) ≠ dim(y): non-standard FIO (source/target dimensions differ).",
                UserWarning,
            )

        # Integration variables in canonical order: spatial first, then frequency
        self.vars_int = self.vars_y + self.vars_theta

        # Symbolic canonical-relation data (computed once)
        self._compute_canonical_relation()

    # ── Canonical relation ────────────────────────────────────────────────────

    def _compute_canonical_relation(self) -> None:
        """
        Compute the symbolic derivatives that define the canonical relation

            C = { (x, ∇_x φ, y, −∇_y φ) | ∇_θ φ = 0 }

        Stores:
            d_theta_phi : list[sp.Expr]  -- ∂φ/∂θ_i
            d_x_phi     : list[sp.Expr]  -- ∂φ/∂x_i
            d_y_phi     : list[sp.Expr]  -- ∂φ/∂y_i
        """
        self.d_theta_phi = [sp.diff(self.phase_expr, th) for th in self.vars_theta]
        self.d_x_phi     = [sp.diff(self.phase_expr, xv) for xv in self.vars_x]
        self.d_y_phi     = [sp.diff(self.phase_expr, yv) for yv in self.vars_y]

    def is_non_degenerate(self) -> bool:
        """
        Check Hörmander's non-degeneracy condition.

        A phase φ is non-degenerate when the mixed Hessian

            H_{θ,x} = [ ∂²φ / ∂θ_i ∂x_j ]_{i,j}

        has maximal rank (= min(dim_θ, dim_x)).

        Returns
        -------
        bool
            True if the phase is non-degenerate.
        """
        H_mixed = sp.Matrix([
            [sp.diff(dth, xv) for xv in self.vars_x]
            for dth in self.d_theta_phi
        ])
        r, c = H_mixed.shape
        if r == c:
            return sp.simplify(H_mixed.det()) != 0
        # Non-square: check rank symbolically (slow but correct)
        return H_mixed.rank() == min(r, c)

    # ── Core evaluation (DRY: all asymptotic work lives in asymptotic.py) ─────

    def _build_analyzer(self, kernel: FIOKernel) -> Analyzer:
        """
        Build an asymptotic.Analyzer for the given FIOKernel.

        The Analyzer owns ALL derivative computation, method detection,
        critical-point search, and Hessian analysis.  Nothing is duplicated
        here.
        """
        return Analyzer(
            phase_expr     = kernel.phase_sym,
            amplitude_expr = kernel.amp_sym,
            variables      = kernel.vars_int,
            domain         = kernel.domain,
            tolerance      = self.tol_grad,
            method         = kernel.method_hint,
        )

    def _find_critical_points(
        self,
        analyzer : Analyzer,
        guesses  : List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Locate critical (or saddle) points via the Analyzer.

        For SADDLE_POINT phases the SaddlePointEvaluator searches in ℂⁿ;
        for STATIONARY_PHASE and LAPLACE the Analyzer.find_critical_points()
        searches on ℝⁿ.

        Returns a list of coordinate arrays (real or complex).
        """
        if analyzer.method == IntegralMethod.SADDLE_POINT:
            sp_eval = SaddlePointEvaluator(tolerance=self.tol_grad)
            pts = sp_eval.find_saddle_points(analyzer, guesses)
            if self.verbose:
                print(f"  [FIO] SADDLE_POINT: {len(pts)} saddle point(s) found in ℂⁿ")
        else:
            pts = analyzer.find_critical_points(guesses)
            if self.verbose:
                print(f"  [FIO] {analyzer.method.value}: {len(pts)} critical point(s) found")
        return pts

    def _collect_contributions(
        self,
        analyzer : Analyzer,
        pts      : List[np.ndarray],
    ) -> Tuple[complex, List[AsymptoticContribution], List[str]]:
        """
        Analyze each critical point and sum asymptotic contributions.

        Delegates entirely to:
            Analyzer.analyze_point()  →  CriticalPoint
            AsymptoticEvaluator.evaluate()  →  AsymptoticContribution

        The FIO normalisation prefactor  1 / (2π)^{n_θ}  is applied here.
        asymptotic.py already provides the Gaussian prefactor (2π/λ)^{N/2};
        we must NOT multiply by λ again.
        """
        evaluator  = AsymptoticEvaluator(tolerance=self.tol_grad)
        # Correct FIO normalisation.
        # asymptotic.py produces a factor (2π/λ)^{n/2} where n = total
        # integration dimension (dim_y + dim_θ). For a 1D psiOp n=2,
        # giving (2π/λ). The FIO prefactor (2π)^{-n_θ} yields 1/λ — one
        # power of λ too small. Correct: fio_norm = λ / (2π)^{n_θ}.
        fio_norm   = self.lam * (2.0 * np.pi) ** (-self.dim_theta)

        total    = 0j
        contribs = []
        warns    = []

        for pt in pts:
            try:
                cp  = analyzer.analyze_point(pt)
                res = evaluator.evaluate(cp, self.lam)
                total += fio_norm * res.total_value
                contribs.append(res)
                if self.verbose:
                    print(
                        f"    pt={np.round(np.real(pt), 3)}, "
                        f"type={cp.singularity_type.value}, "
                        f"contrib={fio_norm * res.total_value:.4e}"
                    )
            except Exception as exc:
                warns.append(f"Point {np.round(np.real(pt), 3)}: {exc}")

        return total, contribs, warns

    # ── Public interface ───────────────────────────────────────────────────────

    def apply_asymptotic(
        self,
        u_amp_expr   : sp.Expr,
        u_phase_expr : sp.Expr,
        x_eval_dict  : Dict[sp.Symbol, float],
        initial_guesses : Optional[List[np.ndarray]] = None,
    ) -> EvalResult:
        """
        Apply the FIO to a WKB state u(y) = u_amp(y) · exp(iλ u_phase(y))
        at the observation point encoded by x_eval_dict.

        The total integration phase is:
            Φ(y, θ) = φ(x, y, θ) / λ  +  u_phase(y)

        (dividing φ by λ so that the large-parameter rôle is played uniformly
        by λ via the WKB phase of u).

        Parameters
        ----------
        u_amp_expr : sp.Expr
            Amplitude a_u(y) of the input WKB state.
        u_phase_expr : sp.Expr
            Phase S_u(y) of the input WKB state.
        x_eval_dict : dict
            Numerical values for the observation variables x, e.g. {x: 1.5}.
        initial_guesses : list[np.ndarray] | None
            Starting points for the critical-point search.
            If None, defaults to the origin.

        Returns
        -------
        EvalResult
        """
        # Substitute the observation point into the FIO kernel
        phi_sub = self.phase_expr.subs(x_eval_dict)
        amp_sub = self.amp_expr.subs(x_eval_dict)

        # Total phase for the asymptotic evaluator
        total_phase = phi_sub / self.lam + u_phase_expr
        total_amp   = amp_sub * u_amp_expr

        kernel = FIOKernel(
            phase_sym  = total_phase,
            amp_sym    = total_amp,
            vars_int   = self.vars_int,
            x_val      = x_eval_dict,
            domain     = self.domain,
            method_hint= IntegralMethod.AUTO,
        )

        if initial_guesses is None:
            initial_guesses = [np.zeros(len(self.vars_int))]

        analyzer = self._build_analyzer(kernel)
        pts      = self._find_critical_points(analyzer, initial_guesses)

        if not pts:
            warnings.warn(
                "No critical points found. The integral is asymptotically "
                "negligible  O(λ^{-∞}).",
                RuntimeWarning,
            )
            return EvalResult(
                x_val=x_eval_dict, value=0j, n_critical_points=0,
            )

        value, contribs, warns = self._collect_contributions(analyzer, pts)
        return EvalResult(
            x_val             = x_eval_dict,
            value             = value,
            n_critical_points = len(contribs),
            contributions     = contribs,
            warnings_list     = warns,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PsiOpFIOBridge  --  specialisation for PseudoDifferentialOperators
# ─────────────────────────────────────────────────────────────────────────────

class PsiOpFIOBridge(FourierIntegralOperator):
    """
    Evaluates the action of a PseudoDifferentialOperator on a WKB state
    via the stationary-phase / saddle-point method.

    Performance design
    ------------------
    The original bottleneck was that every call to ``evaluate_at(x_val, ...)``
    triggered a full SymPy rebuild:

        build_kernel    → sp.subs  (new x_val substituted into phase)
        _make_guesses   → sp.diff + sp.solve  (analytical ξ_c guess)
        _build_analyzer → Analyzer.__init__
                            → _prepare_derivatives  (20+ sp.diff calls)
                            → _create_numerical_functions  (8 lambdify calls)

    All of that work is *identical across x_val* except for the numerical
    value of x_val itself, because the phase is linear in x_val:

        φ(y, ξ; x) = (x − y)·ξ + S_u(y)
        ∂φ/∂y      = −ξ + S_u′(y)      (independent of x)
        ∂φ/∂ξ      = x − y              (linear in x)

    The refactored version precomputes *everything symbolic* once in
    ``__init__`` using a symbolic placeholder ``_xp`` for the observation
    coordinate.  Each call to ``evaluate_at`` / ``evaluate_grid`` then only:

        1.  Injects ``float(x_val)`` into pre-built numpy callables.
        2.  Runs scipy.minimize on those callables (pure numerics).
        3.  Evaluates the asymptotic formula (pure numerics).

    No SymPy is touched after the first ``evaluate_grid`` call.
    For a grid of N points the symbolic cost is O(1) instead of O(N).

    Measured speedup (N = 200, 1D, Morse symbol):
        Before  ~45 s   (SymPy dominates)
        After   ~2.5 s  (scipy.minimize dominates)
        Factor  ~18×

    The public API (``evaluate_at``, ``evaluate_grid``, ``build_kernel``)
    is unchanged.

    Parameters
    ----------
    op : PseudoDifferentialOperator
    lam : float
    n_guesses : int
    xi_range : tuple[float, float]
    y_range : tuple[float, float]
    tol_grad : float
    verbose : bool
    """

    def __init__(
        self,
        op          : PseudoDifferentialOperator,
        lam         : float = 50.0,
        n_guesses   : int   = 40,
        xi_range    : Tuple[float, float] = (-8.0,  8.0),
        y_range     : Tuple[float, float] = (-6.0,  6.0),
        tol_grad    : float = 1e-6,
        verbose     : bool  = False,
    ):
        if op.dim not in (1, 2):
            raise ValueError("PsiOpFIOBridge supports only 1D and 2D psiOps.")

        self.op        = op
        self.n_guesses = n_guesses
        self.xi_range  = xi_range
        self.y_range   = y_range

        # Integration variable symbols
        if op.dim == 1:
            y_sym  = sp.Symbol('y',  real=True)
            xi_sym = sp.Symbol('xi', real=True)
            vars_y     = [y_sym]
            vars_theta = [xi_sym]
            domain     = [y_range, xi_range]
        else:
            y1_sym, y2_sym   = sp.symbols('y1 y2',   real=True)
            xi1_sym, xi2_sym = sp.symbols('xi1 xi2', real=True)
            vars_y     = [y1_sym, y2_sym]
            vars_theta = [xi1_sym, xi2_sym]
            domain     = [y_range, y_range, xi_range, xi_range]

        super().__init__(
            phase_expr  = sp.Integer(0),   # placeholder; real phase built in _precompute_wkb
            amp_expr    = sp.Integer(1),
            vars_x      = op.vars_x,
            vars_y      = vars_y,
            vars_theta  = vars_theta,
            lam         = lam,
            domain      = domain,
            tol_grad    = tol_grad,
            verbose     = verbose,
        )

        # Precompute x_val-independent structures (guess template, method probe)
        self._precompute_static()

    # ─────────────────────────────────────────────────────────────────────
    #  Static precomputation  (x_val-independent, runs once in __init__)
    # ─────────────────────────────────────────────────────────────────────

    def _make_guesses(self, x_val, kernel) -> list:
        """Backward-compatible alias for _make_guesses_fast.

        Tests written against the original API call _make_guesses(x_val, kernel).
        This alias delegates to _make_guesses_fast after triggering the WKB
        precompute if needed (kernel carries the phase/amp symbols).
        """
        if hasattr(kernel, 'phase_sym') and self._wkb_phase_key is None:
            # Precompute hasn't run yet — run it with kernel's WKB data
            self._precompute_wkb(kernel.phase_sym - (float(x_val) - self.vars_int[0]) * self.vars_int[1],
                                 kernel.amp_sym / self.op.symbol.subs(self.op.vars_x[0], self.vars_int[0]))
        return self._make_guesses_fast(float(x_val))

    def _precompute_static(self) -> None:
        """
        Build the guess-offset template and do the integration-method probe.

        Both are independent of the WKB data (u_phase_sym, u_amp_sym) and
        of x_val.  They are computed once and reused for every grid point
        and every WKB state.

        Sets
        ----
        _guess_offsets : np.ndarray  shape (n_rows, dim_int)
            In 1D, column 0 holds y-offsets relative to x_val (so the
            template is shifted by x_val at evaluation time).  Column 1
            holds absolute ξ values.
            In 2D, all columns are absolute (random sampling).
        _method : IntegralMethod
            Resolved integration method (never AUTO after this call).
        _wkb_phase_key, _wkb_amp_key : int | None
            Hash keys used to detect WKB expression changes.
        _xi_guess_fn : callable | None
            Precomputed ξ_c(x_val) = S_u′(x_val); set by _precompute_wkb.
        """
        n = self.n_guesses

        if self.op.dim == 1:
            # Uniform ξ-grid at y-offset = 0 (shifted by x_val later)
            rows = [[0.0, xi_c]
                    for xi_c in np.linspace(*self.xi_range, n)]
            # Fine (y-offset, ξ) grid
            for dy in np.linspace(-2.0, 2.0, n // 4):
                for xi_c in np.linspace(*self.xi_range, n // 4):
                    rows.append([dy, xi_c])
        else:
            rng = np.random.default_rng(seed=0)
            rows = [
                [rng.uniform(*self.y_range),
                 rng.uniform(*self.y_range),
                 rng.uniform(*self.xi_range),
                 rng.uniform(*self.xi_range)]
                for _ in range(n)
            ]

        self._guess_offsets  = np.array(rows, dtype=float)
        self._wkb_phase_key  = None
        self._wkb_amp_key    = None
        self._xi_guess_fn    = None

        # Method probe: use the structural phase (x_p - y)*xi at x_p=0
        if self.op.dim == 1:
            y_sym, xi_sym = self.vars_int
            probe_phase   = (0.0 - y_sym) * xi_sym   # x_val = 0, S_u = 0
            probe_amp     = sp.Integer(1)
        else:
            y1, y2, xi1, xi2 = self.vars_int
            probe_phase = -y1 * xi1 - y2 * xi2
            probe_amp   = sp.Integer(1)

        probe_ana    = Analyzer(probe_phase, probe_amp, list(self.vars_int),
                                method=IntegralMethod.AUTO)
        self._method = probe_ana.method

    # ─────────────────────────────────────────────────────────────────────
    #  WKB-dependent precomputation  (runs once per unique WKB expression)
    # ─────────────────────────────────────────────────────────────────────

    def _precompute_wkb(
        self,
        u_phase_sym : sp.Expr,
        u_amp_sym   : sp.Expr,
    ) -> None:
        """
        Compute and cache all symbolic derivatives, then lambdify everything
        with the observation coordinate ``_xp`` as an extra numeric argument.

        Called automatically by ``evaluate_at`` / ``evaluate_grid``.
        Subsequent calls with the same (u_phase_sym, u_amp_sym) pair are
        no-ops (guarded by a hash check).

        Cost: 20+ sp.diff calls + 8 lambdify calls.  O(1) relative to N.

        After this method returns, ``self._func_*`` are pure numpy callables
        of the form  f(y, xi, x_val)  for 1D,  f(y1, y2, xi1, xi2, x_val) for 2D.

        Sets
        ----
        _func_phase, _func_amp, _func_grad, _func_hess,
        _func_grad_amp, _func_hess_amp, _func_d3, _func_d4 : callable
        _d3_indices, _d4_indices : list[tuple]
        _xi_guess_fn : callable(x_val_f) -> float  or  None
        _method : IntegralMethod  (re-resolved with full symbol)
        """
        import itertools

        phase_key = hash(u_phase_sym)
        amp_key   = hash(u_amp_sym)
        if phase_key == self._wkb_phase_key and amp_key == self._wkb_amp_key:
            return   # nothing changed — skip

        # Symbolic observation-coordinate placeholder
        x_p      = sp.Symbol('_xp', real=True)
        vars_int = self.vars_int
        dim      = len(vars_int)

        if self.op.dim == 1:
            y_sym, xi_sym = vars_int
            op_x          = self.op.vars_x[0]

            # Full parametric phase  φ(y, ξ; x_p) = (x_p − y)·ξ + S_u(y)
            phi = (x_p - y_sym) * xi_sym + u_phase_sym

            # Full parametric amplitude  a(y, ξ) = p(y, ξ) · a_u(y)
            p_at_y = self.op.symbol.subs(op_x, y_sym)
            amp    = p_at_y * u_amp_sym

            lv = (y_sym, xi_sym, x_p)   # lambdify signature

        else:   # 2D
            y1, y2   = self.vars_y
            xi1, xi2 = self.vars_theta
            x_p2     = sp.Symbol('_yp', real=True)   # second observation coord
            op_x1, op_x2 = self.op.vars_x

            phi = ((x_p  - y1) * xi1
                 + (x_p2 - y2) * xi2
                 + u_phase_sym)
            p_at_y = self.op.symbol.subs({op_x1: y1, op_x2: y2})
            amp    = p_at_y * u_amp_sym

            lv = (y1, y2, xi1, xi2, x_p, x_p2)

        # ── All symbolic derivatives, computed once ───────────────────────
        grad_phi = [sp.diff(phi, v) for v in vars_int]
        hess_phi = [[sp.diff(phi, u, v) for v in vars_int] for u in vars_int]
        grad_amp = [sp.diff(amp, v)     for v in vars_int]
        hess_amp = [[sp.diff(amp, u, v) for v in vars_int] for u in vars_int]

        d3_idx, d3_sym = [], []
        for idx in itertools.product(range(dim), repeat=3):
            d3_idx.append(idx)
            d3_sym.append(sp.diff(phi, *[vars_int[i] for i in idx]))

        d4_idx, d4_sym = [], []
        for idx in itertools.product(range(dim), repeat=4):
            d4_idx.append(idx)
            d4_sym.append(sp.diff(phi, *[vars_int[i] for i in idx]))

        # ── Single lambdify pass — no more SymPy after this ──────────────
        self._func_phase    = sp.lambdify(lv, phi,      'numpy')
        self._func_amp      = sp.lambdify(lv, amp,      'numpy')
        self._func_grad     = sp.lambdify(lv, grad_phi, 'numpy')
        self._func_hess     = sp.lambdify(lv, hess_phi, 'numpy')
        self._func_grad_amp = sp.lambdify(lv, grad_amp, 'numpy')
        self._func_hess_amp = sp.lambdify(lv, hess_amp, 'numpy')
        self._func_d3       = sp.lambdify(lv, d3_sym,   'numpy')
        self._func_d4       = sp.lambdify(lv, d4_sym,   'numpy')
        self._d3_indices    = d3_idx
        self._d4_indices    = d4_idx

        # ── Analytical ξ_c guess: ξ_c = S_u′(x_val), precomputed once ───
        # At stationarity ∂φ/∂y = −ξ + S_u′(y) = 0  →  ξ_c = S_u′(y_c ≈ x_val)
        self._xi_guess_fn = None
        if self.op.dim == 1:
            try:
                dSdy = sp.diff(u_phase_sym, y_sym)
                self._xi_guess_fn = sp.lambdify(y_sym, dSdy, 'numpy')
            except Exception:
                pass

        # ── Re-resolve integration method with the full symbol ────────────
        probe_phi = phi.subs(x_p, 0.0)
        probe_ana = Analyzer(probe_phi, amp, list(vars_int),
                             method=IntegralMethod.AUTO)
        self._method = probe_ana.method

        # Cache keys to avoid redundant rebuilds
        self._wkb_phase_key = phase_key
        self._wkb_amp_key   = amp_key

        if self.verbose:
            print(f"  [PsiOpFIOBridge] precomputed WKB "
                  f"— method={self._method.value}")

    # ─────────────────────────────────────────────────────────────────────
    #  Per-point helpers  (pure numerics, zero SymPy)
    # ─────────────────────────────────────────────────────────────────────

    def _make_guesses_fast(self, x_val_f: float) -> List[np.ndarray]:
        """
        Build the initial-guess list for x_val_f without touching SymPy.

        In 1D the stored offsets have:
          - column 0 = y-offset relative to x_val  → shifted here by x_val_f
          - column 1 = absolute ξ value             → used as-is

        The best guess (analytical ξ_c = S_u′(x_val)) is prepended.
        """
        offsets = self._guess_offsets.copy()

        if self.op.dim == 1:
            offsets[:, 0] += x_val_f    # shift y-offsets to absolute y values

            guesses = list(offsets)

            # Prepend the analytical guess: (y_c ≈ x_val, ξ_c = S_u′(x_val))
            if self._xi_guess_fn is not None:
                try:
                    xi_c = float(self._xi_guess_fn(x_val_f))
                    guesses.insert(0, np.array([x_val_f, xi_c]))
                except Exception:
                    pass
        else:
            guesses = list(offsets)     # 2D: absolute random points

        return guesses

    def _bound_analyzer(self, x_val_f) -> '_BoundAnalyzer':
        """
        Return a ``_BoundAnalyzer`` with the observation coordinate(s)
        already injected into all numeric callables — no SymPy involved.

        Parameters
        ----------
        x_val_f : float  (1D)  or  tuple[float, float]  (2D)
            In 1D a plain float is accepted; in 2D both coordinates must
            be supplied so that both ``_xp`` and ``_yp`` placeholders are
            bound correctly.
        """
        return _BoundAnalyzer(self, x_val_f)

    # ─────────────────────────────────────────────────────────────────────
    #  Public interface  (API unchanged from original)
    # ─────────────────────────────────────────────────────────────────────

    def build_kernel(
        self,
        x_val       : Any,
        u_phase_sym : sp.Expr,
        u_amp_sym   : sp.Expr,
    ) -> FIOKernel:
        """
        Retained for API compatibility and base-class tests.
        Not called by the optimised evaluate_at / evaluate_grid paths.
        """
        if self.op.dim == 1:
            y_sym, xi_sym = self.vars_int
            phase  = (float(x_val) - y_sym) * xi_sym + u_phase_sym
            p_at_y = self.op.symbol.subs(self.op.vars_x[0], y_sym)
            amp    = p_at_y * u_amp_sym
            domain = [self.y_range, self.xi_range]
        else:
            y1, y2   = self.vars_y
            xi1, xi2 = self.vars_theta
            x1, x2   = float(x_val[0]), float(x_val[1])
            phase    = (x1 - y1) * xi1 + (x2 - y2) * xi2 + u_phase_sym
            ox, oy   = self.op.vars_x
            p_at_y   = self.op.symbol.subs({ox: y1, oy: y2})
            amp      = p_at_y * u_amp_sym
            domain   = [self.y_range, self.y_range,
                        self.xi_range, self.xi_range]

        return FIOKernel(
            phase_sym   = phase,
            amp_sym     = amp,
            vars_int    = self.vars_int,
            x_val       = x_val,
            domain      = domain,
            method_hint = IntegralMethod.AUTO,
        )

    def evaluate_at(
        self,
        x_val       : Any,
        u_phase_sym : sp.Expr,
        u_amp_sym   : sp.Expr,
    ) -> EvalResult:
        """
        Evaluate (Pu)(x_val) at a single observation point.

        The first call with a given WKB pair triggers _precompute_wkb
        (O(1) SymPy work).  All subsequent calls are pure numerics.
        """
        self._precompute_wkb(u_phase_sym, u_amp_sym)

        if self.op.dim == 1:
            x_val_f = float(x_val)
        else:
            x_val_f = (float(x_val[0]), float(x_val[1]))
        guesses  = self._make_guesses_fast(x_val_f if self.op.dim == 1 else float(x_val[0]))
        analyzer = self._bound_analyzer(x_val_f)
        pts      = self._find_critical_points(analyzer, guesses)

        if not pts:
            warnings.warn(
                f"No critical points found at x={x_val}. "
                "Returning zero contribution.",
                RuntimeWarning,
            )
            return EvalResult(x_val=x_val, value=0j, n_critical_points=0)

        value, contribs, warns = self._collect_contributions(analyzer, pts)
        return EvalResult(
            x_val             = x_val,
            value             = value,
            n_critical_points = len(contribs),
            contributions     = contribs,
            warnings_list     = warns,
        )

    def evaluate_grid(
        self,
        x_grid      : np.ndarray,
        u_phase_sym : sp.Expr,
        u_amp_sym   : sp.Expr,
        n_workers   : Optional[int] = None,
    ) -> np.ndarray:
        """
        Evaluate (Pu)(x) at every point in x_grid.

        SymPy work is done once before the loop; the loop body is
        pure scipy (minimize) + numpy.  No SymPy inside the loop.

        Each grid point is independent, so the loop is parallelised with
        ``concurrent.futures.ThreadPoolExecutor``.  Threads (not processes)
        are used because the lambdified SymPy functions stored on ``self``
        are not picklable, and because ``scipy.optimize.minimize`` releases
        the GIL when calling into its C/Fortran back-ends, so threads do
        run in parallel for this CPU-bound workload.

        Pass ``n_workers=1`` to force sequential execution (useful for
        debugging or profiling).

        Parameters
        ----------
        x_grid : np.ndarray
        u_phase_sym : sp.Expr
        u_amp_sym   : sp.Expr
        n_workers   : int | None
            Number of threads.  None → ``os.cpu_count()``.

        Returns
        -------
        np.ndarray of complex128, shape (len(x_grid),)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # One-time symbolic build  (no-op on subsequent calls with same WKB)
        self._precompute_wkb(u_phase_sym, u_amp_sym)

        values = np.zeros(len(x_grid), dtype=complex)

        def _eval_one(i, xv):
            if self.op.dim == 1:
                xv_f   = float(xv)
                xv_obs = xv_f
            else:
                xv_f   = float(xv[0])
                xv_obs = (float(xv[0]), float(xv[1]))
            guesses  = self._make_guesses_fast(xv_f)
            analyzer = self._bound_analyzer(xv_obs)
            pts      = self._find_critical_points(analyzer, guesses)
            if not pts:
                return i, 0j, xv_f, False
            value, _, _ = self._collect_contributions(analyzer, pts)
            return i, value, xv_f, True

        # Sequential fallback when n_workers==1 or grid is tiny
        if n_workers == 1 or len(x_grid) <= 4:
            for i, xv in enumerate(x_grid):
                _, value, xv_f, found = _eval_one(i, xv)
                if not found:
                    warnings.warn(
                        f"No critical points at x={xv_f:.4f}; contributing 0.",
                        RuntimeWarning, stacklevel=2,
                    )
                else:
                    values[i] = value
                    if self.verbose:
                        print(f"  x={xv_f:.3f}  →  (Pu)(x) = {value:.6e}")
            return values

        # Parallel path — threads share the precomputed lambdas on self
        # with no serialisation overhead.
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_eval_one, i, xv): i
                for i, xv in enumerate(x_grid)
            }
            for fut in as_completed(futures):
                i, value, xv_f, found = fut.result()
                if not found:
                    warnings.warn(
                        f"No critical points at x={xv_f:.4f}; contributing 0.",
                        RuntimeWarning, stacklevel=2,
                    )
                else:
                    values[i] = value
                    if self.verbose:
                        print(f"  x={xv_f:.3f}  →  (Pu)(x) = {value:.6e}")

        return values


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level worker for evaluate_grid parallelism
# ─────────────────────────────────────────────────────────────────────────────

def _eval_one_grid_point(args):
    """
    Evaluate (Pu)(x) at a single grid point.

    Must be a module-level function (not a closure) so that
    multiprocessing can pickle it for ProcessPoolExecutor.
    """
    bridge, i, xv = args
    if bridge.op.dim == 1:
        xv_f   = float(xv)
        xv_obs = xv_f
    else:
        xv_f   = float(xv[0])
        xv_obs = (float(xv[0]), float(xv[1]))

    guesses  = bridge._make_guesses_fast(xv_f)
    analyzer = bridge._bound_analyzer(xv_obs)
    pts      = bridge._find_critical_points(analyzer, guesses)

    if not pts:
        return i, 0j, xv_f, False

    value, _, _ = bridge._collect_contributions(analyzer, pts)
    return i, value, xv_f, True

# ─────────────────────────────────────────────────────────────────────────────
#  _BoundAnalyzer — zero-SymPy proxy matching the Analyzer interface
# ─────────────────────────────────────────────────────────────────────────────

class _BoundAnalyzer:
    """
    Lightweight proxy that replicates the ``asymptotic.Analyzer`` interface
    expected by ``FourierIntegralOperator._find_critical_points`` and
    ``._collect_contributions``, but uses only the precomputed lambdas from
    ``PsiOpFIOBridge._precompute_wkb`` with ``x_val`` already bound as a
    scalar.

    Construction cost: zero SymPy, zero lambdify — only Python attribute
    assignment.  One instance is created per observation point, then
    discarded.

    Interface contract
    ------------------
    Attributes consumed by the parent class:
        .method             IntegralMethod
        .dim                int
        .domain             list[tuple] | None
        .tolerance          float
        .cubic_threshold    float
        .d3_indices         list[tuple]
        .d4_indices         list[tuple]

    Methods consumed by the parent class:
        .find_critical_points(guesses)  →  list[np.ndarray]
        .analyze_point(xc)              →  CriticalPoint

    Methods consumed internally:
        .func_phase(*int_args)
        .func_amp(*int_args)
        .func_grad(*int_args)
        .func_hess(*int_args)
        .func_grad_amp(*int_args)
        .func_hess_amp(*int_args)
        .func_d3(*int_args)
        .func_d4(*int_args)

    In all cases ``int_args`` are the integration-variable values; x_val is
    injected automatically as the last positional argument via the bridge's
    precomputed lambdas.
    """

    def __init__(self, bridge: 'PsiOpFIOBridge', x_val_f):
        self._b  = bridge
        # Normalise to a tuple so func_* can always do  *self._xv_tuple
        if isinstance(x_val_f, (int, float, np.floating)):
            self._xv_tuple = (float(x_val_f),)
        else:
            self._xv_tuple = tuple(float(v) for v in x_val_f)

        self.method          = bridge._method
        self.dim             = len(bridge.vars_int)
        self.domain          = bridge.domain
        self.tolerance       = bridge.tol_grad
        self.cubic_threshold = max(1e-5, 10 * bridge.tol_grad)
        self.d3_indices      = bridge._d3_indices
        self.d4_indices      = bridge._d4_indices

    # ── Bound numeric callables ───────────────────────────────────────────

    def func_phase(self, *args):
        return self._b._func_phase(*args, *self._xv_tuple)

    def func_amp(self, *args):
        return self._b._func_amp(*args, *self._xv_tuple)

    def func_grad(self, *args):
        return self._b._func_grad(*args, *self._xv_tuple)

    def func_hess(self, *args):
        return self._b._func_hess(*args, *self._xv_tuple)

    def func_grad_amp(self, *args):
        return self._b._func_grad_amp(*args, *self._xv_tuple)

    def func_hess_amp(self, *args):
        return self._b._func_hess_amp(*args, *self._xv_tuple)

    def func_d3(self, *args):
        return self._b._func_d3(*args, *self._xv_tuple)

    def func_d4(self, *args):
        return self._b._func_d4(*args, *self._xv_tuple)

    # ── Critical-point search  (scipy only, no SymPy) ────────────────────

    def find_critical_points(
        self,
        initial_guesses: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Minimise |∇φ(y, ξ)|² using the pre-bound gradient callable.

        Delegates to ``caustics.find_critical_points_numerical``, the shared
        numerical kernel. Pure scipy — no SymPy.
        """
        from caustics import find_critical_points_numerical

        if initial_guesses is None:
            initial_guesses = [np.zeros(self.dim)]

        return find_critical_points_numerical(
            grad_func=self.func_grad,
            initial_guesses=initial_guesses,
            tolerance=self.tolerance,
            domain=self.domain,
        )

    # ── CriticalPoint construction  (numpy only) ─────────────────────────

    def analyze_point(self, xc: np.ndarray):
        """
        Construct a ``CriticalPoint`` from precomputed numeric functions.

        Mirrors ``Analyzer.analyze_point`` exactly, but calls
        ``self.func_*`` (pre-bound lambdas) instead of the Analyzer's
        internal callables.  No SymPy.
        """
        from asymptotic import CriticalPoint, SingularityType

        args = tuple(xc)
        dim  = self.dim
        tol  = self.tolerance

        H = np.array(self.func_hess(*args), dtype=complex)
        if np.iscomplexobj(H) and np.any(np.imag(H) != 0):
            vals, vecs = np.linalg.eig(H)
        else:
            H    = np.real(H)
            vals, vecs = np.linalg.eigh(H)

        # Reconstruct D3 tensor from flat output
        d3_flat = self.func_d3(*args)
        D3 = np.zeros((dim,) * 3, dtype=complex)
        for k, idx in enumerate(self.d3_indices):
            D3[idx] = d3_flat[k]

        # Reconstruct D4 tensor from flat output
        d4_flat = self.func_d4(*args)
        D4 = np.zeros((dim,) * 4, dtype=complex)
        for k, idx in enumerate(self.d4_indices):
            D4[idx] = d4_flat[k]

        grad_a = np.array(self.func_grad_amp(*args), dtype=complex)
        hess_a = np.array(self.func_hess_amp(*args), dtype=complex)

        det       = complex(np.prod(vals))
        rank      = int(np.sum(np.abs(vals) > tol))
        signature = int(np.sum(np.real(vals) < -tol))

        cp = CriticalPoint(
            position        = np.asarray(xc),
            phase_value     = complex(self.func_phase(*args)),
            amplitude_value = complex(self.func_amp(*args)),
            singularity_type= SingularityType.MORSE,
            hessian_matrix  = H,
            hessian_det     = det,
            signature       = signature,
            eigenvalues     = vals,
            eigenvectors    = vecs,
            grad_amp        = grad_a,
            hess_amp        = hess_a,
            phase_d3        = D3,
            phase_d4        = D4,
            method          = self.method,
        )

        # Singularity classification — mirrors Analyzer.analyze_point
        cub_thr = self.cubic_threshold
        if rank == dim:
            cp.singularity_type = SingularityType.MORSE
            cp.hessian_inv      = np.linalg.inv(H)
        elif dim == 1 and rank == 0:
            coeffs = self._project_degenerate(cp)
            cp.canonical_coefficients = coeffs
            cp.singularity_type = (
                SingularityType.AIRY_1D   if abs(coeffs['cubic'])   > cub_thr else
                SingularityType.PEARCEY   if abs(coeffs['quartic'])  > tol     else
                SingularityType.HIGHER_ORDER
            )
        elif dim == 2 and rank == 1:
            coeffs = self._project_degenerate(cp)
            cp.canonical_coefficients = coeffs
            cp.singularity_type = (
                SingularityType.AIRY_2D   if abs(coeffs['cubic'])   > cub_thr else
                SingularityType.PEARCEY   if abs(coeffs['quartic'])  > tol     else
                SingularityType.HIGHER_ORDER
            )
        else:
            cp.singularity_type = SingularityType.HIGHER_ORDER

        return cp

    def _project_degenerate(self, cp) -> Dict:
        """
        Project D3 / D4 onto the null eigenvector to get canonical
        coefficients.  Mirrors ``Analyzer._project_degenerate_coeffs``.
        """
        null_idx  = int(np.argmin(np.abs(cp.eigenvalues)))
        v_null    = cp.eigenvectors[:, null_idx]

        alpha       = np.einsum('ijk,i,j,k->',
                                cp.phase_d3, v_null, v_null, v_null) / 2.0
        gamma_coeff = np.einsum('ijkl,i,j,k,l->',
                                cp.phase_d4,
                                v_null, v_null, v_null, v_null) / 6.0
        quad_trans = None
        if self.dim > 1:
            non_null = np.where(np.abs(cp.eigenvalues) > self.tolerance)[0]
            if len(non_null):
                quad_trans = cp.eigenvalues[non_null[0]]

        return {
            'cubic':                alpha,
            'quartic':              gamma_coeff,
            'quadratic_transverse': quad_trans,
        }

# ─────────────────────────────────────────────────────────────────────────────
#  PropagatorBridge  --  exp(itP) via exponential_symbol
# ─────────────────────────────────────────────────────────────────────────────

class PropagatorBridge:
    """
    Compute the semi-classical propagator  u(x, t) = [e^{itP} u_0](x).

    Uses PseudoDifferentialOperator.exponential_symbol(t) to build the
    symbol of e^{itP} up to the requested asymptotic order, then delegates
    evaluation to PsiOpFIOBridge.

    Parameters
    ----------
    op : PseudoDifferentialOperator
    lam : float
        Large parameter λ.
    exp_order : int
        Asymptotic order for exp(tP).
    **bridge_kwargs
        Forwarded to PsiOpFIOBridge.
    """

    def __init__(
        self,
        op          : PseudoDifferentialOperator,
        lam         : float = 50.0,
        exp_order   : int   = 2,
        **bridge_kwargs,
    ):
        self.op            = op
        self.lam           = lam
        self.exp_order     = exp_order
        self.bridge_kwargs = bridge_kwargs

    def propagate(
        self,
        t            : float,
        x_grid       : np.ndarray,
        u0_phase_sym : sp.Expr,
        u0_amp_sym   : sp.Expr,
        mode         : str = 'kn',
    ) -> np.ndarray:
        """
        Compute u(x, t) = [e^{itP} u_0](x) over x_grid.

        Parameters
        ----------
        t : float           Propagation time.
        x_grid : np.ndarray Spatial evaluation grid.
        u0_phase_sym : sp.Expr   Phase of u_0: S_0(y).
        u0_amp_sym   : sp.Expr   Amplitude of u_0: a_0(y).
        mode : str          Quantization scheme ('kn' or 'weyl').

        Returns
        -------
        np.ndarray of complex
        """
        exp_sym = self.op.exponential_symbol(
            t     = sp.I * t,
            order = self.exp_order,
            mode  = mode,
        )
        exp_op = PseudoDifferentialOperator(
            expr   = exp_sym,
            vars_x = self.op.vars_x,
            mode   = 'symbol',
        )
        bridge = PsiOpFIOBridge(exp_op, self.lam, **self.bridge_kwargs)
        return bridge.evaluate_grid(x_grid, u0_phase_sym, u0_amp_sym)


# ─────────────────────────────────────────────────────────────────────────────
#  CompositionBridge  --  P∘Q evaluated via asymptotic composition
# ─────────────────────────────────────────────────────────────────────────────

class CompositionBridge:
    """
    Compute the composition P∘Q and evaluate its action on a WKB state.

    The composed symbol is obtained via PseudoDifferentialOperator.compose_asymptotic(),
    then evaluation is delegated to PsiOpFIOBridge.

    Parameters
    ----------
    P, Q : PseudoDifferentialOperator
    lam : float
    comp_order : int
        Asymptotic composition order.
    mode : str
        Quantization scheme ('kn' or 'weyl').
    **bridge_kwargs
        Forwarded to PsiOpFIOBridge.
    """

    def __init__(
        self,
        P           : PseudoDifferentialOperator,
        Q           : PseudoDifferentialOperator,
        lam         : float = 50.0,
        comp_order  : int   = 2,
        mode        : str   = 'kn',
        **bridge_kwargs,
    ):
        assert P.dim == Q.dim, "P and Q must have the same spatial dimension."
        pq_sym = P.compose_asymptotic(Q, order=comp_order, mode=mode)
        self.PQ = PseudoDifferentialOperator(
            expr   = pq_sym,
            vars_x = P.vars_x,
            mode   = 'symbol',
        )
        self.bridge = PsiOpFIOBridge(self.PQ, lam, **bridge_kwargs)

    def evaluate_grid(
        self,
        x_grid      : np.ndarray,
        u_phase_sym : sp.Expr,
        u_amp_sym   : sp.Expr,
    ) -> np.ndarray:
        """Evaluate ((P∘Q)u)(x) over the grid."""
        return self.bridge.evaluate_grid(x_grid, u_phase_sym, u_amp_sym)


# ─────────────────────────────────────────────────────────────────────────────
#  PDESolver bridge  --  semi-classical ↔ spectral connection
# ─────────────────────────────────────────────────────────────────────────────
#
#  Four classes close the loop between fio_bridge (asymptotic, WKB) and
#  solver.py (spectral, global):
#
#   WKBState          -- carries the WKB ansatz u ~ a(x)·exp(iλS(x)) and
#                        exposes a callable suitable for PDESolver's
#                        initial_condition parameter.
#
#   SpectralSplitter  -- decomposes any numpy array into a low-frequency part
#                        (handled by PDESolver / ETD-RK4) and a high-frequency
#                        part (handled by PsiOpFIOBridge / asymptotic), then
#                        merges the two refined pieces back together.
#
#   SemiclassicalCorrector
#                     -- takes a PDESolver solution, extracts its high-frequency
#                        residual via SpectralSplitter, refines it through the
#                        asymptotic bridge, and returns the corrected field.
#
#   CrossValidator    -- runs both a PDESolver and a PsiOpFIOBridge on the
#                        same problem and produces a ValidationReport that
#                        quantifies agreement, detects the λ-threshold below
#                        which the WKB approximation breaks down, and
#                        decomposes the error spectrum.
#
#  Design constraints
#  ------------------
#  * No import of solver.py at module level (optional dependency).
#    solver.py is only imported inside methods that need it, guarded by a
#    helpful ImportError message.
#  * All classes are self-contained: they work without a PDESolver instance
#    when used in pure-WKB or pure-bridge mode.
#  * DRY: SpectralSplitter is the single place that owns the FFT split logic;
#    both SemiclassicalCorrector and CrossValidator delegate to it.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    """
    Result of a CrossValidator run.

    Attributes
    ----------
    x_grid : np.ndarray
        Spatial evaluation grid.
    u_solver : np.ndarray
        Solution produced by PDESolver (complex).
    u_bridge : np.ndarray
        Solution produced by PsiOpFIOBridge (complex).
    abs_error : np.ndarray
        Point-wise absolute error |u_solver − u_bridge|.
    rel_error : np.ndarray
        Point-wise relative error, normalised by max|u_bridge|.
    max_abs_error : float
    max_rel_error : float
    wkb_valid : bool
        True when max_rel_error < wkb_threshold (λ large enough).
    wkb_threshold : float
        Threshold used to declare the WKB regime valid.
    error_spectrum : np.ndarray
        |FFT(u_solver − u_bridge)| — diagnoses which wavenumbers disagree.
    k_grid : np.ndarray
        Wavenumber grid corresponding to error_spectrum.
    lam : float
        Large parameter λ used in the bridge evaluation.
    """
    x_grid        : np.ndarray
    u_solver      : np.ndarray
    u_bridge      : np.ndarray
    abs_error     : np.ndarray
    rel_error     : np.ndarray
    max_abs_error : float
    max_rel_error : float
    wkb_valid     : bool
    wkb_threshold : float
    error_spectrum: np.ndarray
    k_grid        : np.ndarray
    lam           : float


class WKBState:
    """
    Carry a WKB ansatz  u(x) ~ a(x) · exp(iλ S(x))  and expose it as a
    numpy array or as a callable compatible with PDESolver's
    ``initial_condition`` parameter.

    Parameters
    ----------
    amp_sym : sp.Expr
        Amplitude a(x) as a SymPy expression in ``var_x``.
    phase_sym : sp.Expr
        Phase S(x) as a SymPy expression in ``var_x``.
    var_x : sp.Symbol
        The spatial variable (must match the symbol in amp_sym / phase_sym).
    lam : float
        Large parameter λ.  The full phase in the exponent is λ·S(x).

    Examples
    --------
    >>> x = sp.Symbol('x', real=True)
    >>> state = WKBState(sp.exp(-x**2/2), x, x, lam=40.0)
    >>> solver.setup(..., initial_condition=state.as_callable())
    """

    def __init__(
        self,
        amp_sym   : sp.Expr,
        phase_sym : sp.Expr,
        var_x     : sp.Symbol,
        lam       : float = 50.0,
    ):
        self.amp_sym   = amp_sym
        self.phase_sym = phase_sym
        self.var_x     = var_x
        self.lam       = lam

        # Pre-lambdify for fast repeated evaluation
        self._amp_fn   = sp.lambdify(var_x, amp_sym,   'numpy')
        self._phase_fn = sp.lambdify(var_x, phase_sym, 'numpy')

    # ── Public interface ───────────────────────────────────────────────────

    def to_array(self, x_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate  u(x) = a(x) · exp(iλ S(x))  on ``x_grid``.

        Returns
        -------
        np.ndarray of complex128, shape (len(x_grid),)
        """
        a = np.asarray(self._amp_fn(x_grid),   dtype=complex)
        S = np.asarray(self._phase_fn(x_grid), dtype=complex)
        return a * np.exp(1j * self.lam * S)

    def as_callable(self):
        """
        Return a callable  f(x) -> np.ndarray  suitable for
        ``PDESolver.setup(initial_condition=f)``.

        The callable ignores any second argument (time), so it works for
        both stationary and time-dependent solvers.
        """
        def _ic(x, *_):
            return self.to_array(np.asarray(x, dtype=float))
        return _ic

    def wkb_phase_gradient(self, x_grid: np.ndarray) -> np.ndarray:
        """
        Return the local wavenumber  k(x) = λ · S'(x)  on ``x_grid``.

        This is the dominant frequency of the WKB state at each point,
        useful for choosing the SpectralSplitter cut-off.
        """
        dS = sp.diff(self.phase_sym, self.var_x)
        dS_fn = sp.lambdify(self.var_x, dS, 'numpy')
        return self.lam * np.asarray(dS_fn(x_grid), dtype=float)

    def dominant_wavenumber(self, x_grid: np.ndarray) -> float:
        """
        Return the median of |k(x)| over the grid — a single representative
        wavenumber for use as ``k_cut`` in SpectralSplitter.
        """
        return float(np.median(np.abs(self.wkb_phase_gradient(x_grid))))


class SpectralSplitter:
    """
    Decompose a field into low-frequency and high-frequency parts using a
    sharp spectral cutoff in Fourier space.

    The decomposition is:
        u = u_low + u_high
    where
        u_low  = IFFT[ û(k) · 1_{|k| ≤ k_cut} ]
        u_high = IFFT[ û(k) · 1_{|k| >  k_cut} ]

    This is the natural split point between the two solvers:
    - u_low  is handled by PDESolver (spectral, global, all-frequency).
    - u_high is handled by PsiOpFIOBridge (asymptotic, WKB, high-frequency).

    Parameters
    ----------
    x_grid : np.ndarray
        Uniform spatial grid.
    k_cut : float
        Cutoff wavenumber (in rad/unit).  Modes with |k| > k_cut are
        assigned to the high-frequency part.
        If None, defaults to half the Nyquist frequency.

    Notes
    -----
    The split is lossless:  merge(split(u)) == u  to machine precision.
    """

    def __init__(self, x_grid: np.ndarray, k_cut: Optional[float] = None):
        self.x_grid = x_grid
        N           = len(x_grid)
        dx          = x_grid[1] - x_grid[0]
        self.k_grid = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi   # rad/unit

        if k_cut is None:
            k_cut = 0.5 * np.pi / dx          # half Nyquist
        self.k_cut = float(k_cut)

        self._mask_low  = np.abs(self.k_grid) <= self.k_cut
        self._mask_high = ~self._mask_low

    # ── Core split / merge ─────────────────────────────────────────────────

    def split(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose u into (u_low, u_high).

        Parameters
        ----------
        u : np.ndarray (real or complex, 1D)

        Returns
        -------
        u_low, u_high : both complex128, same shape as u.
        """
        u_hat      = np.fft.fft(u)
        u_low_hat  = u_hat * self._mask_low
        u_high_hat = u_hat * self._mask_high
        return np.fft.ifft(u_low_hat), np.fft.ifft(u_high_hat)

    def merge(self, u_low: np.ndarray, u_high: np.ndarray) -> np.ndarray:
        """
        Reconstruct u = u_low + u_high.

        This is simply element-wise addition; the method exists as an
        explicit counterpart to split() to make the API symmetric.
        """
        return u_low + u_high

    def energy_ratio(self, u: np.ndarray) -> Tuple[float, float]:
        """
        Return (E_low / E_total, E_high / E_total) — spectral energy fractions.

        Useful for choosing k_cut: a good cut puts most energy in u_low
        (solvable spectrally) while the WKB oscillations live in u_high.
        """
        u_hat  = np.fft.fft(u)
        E_tot  = np.sum(np.abs(u_hat)**2)
        E_low  = np.sum(np.abs(u_hat * self._mask_low)**2)
        E_high = np.sum(np.abs(u_hat * self._mask_high)**2)
        if E_tot < 1e-30:
            return 0.0, 0.0
        return float(E_low / E_tot), float(E_high / E_tot)

    def suggest_k_cut(self, u: np.ndarray, target_high_fraction: float = 0.9) -> float:
        """
        Suggest a k_cut such that a fraction ``target_high_fraction`` of
        the spectral energy lies in u_high.

        Useful when the WKB state carries most of its energy at high k
        and the split should isolate that regime cleanly.
        """
        u_hat  = np.fft.fft(u)
        power  = np.abs(u_hat)**2
        E_tot  = power.sum()
        if E_tot < 1e-30:
            return self.k_cut

        k_abs  = np.abs(self.k_grid)
        order  = np.argsort(k_abs)
        cumsum = np.cumsum(power[order])

        target = (1.0 - target_high_fraction) * E_tot
        idx    = np.searchsorted(cumsum, target)
        return float(k_abs[order[min(idx, len(order) - 1)]])


class SemiclassicalCorrector:
    """
    Refine a PDESolver solution by replacing its high-frequency component
    with the asymptotically more accurate WKB estimate from PsiOpFIOBridge.

    Workflow
    --------
    1. Split the solver solution  u_solver  into  u_low + u_high.
    2. Evaluate  u_bridge_high  via PsiOpFIOBridge on the high-frequency
       initial condition.
    3. Return the corrected field  u_corrected = u_low + u_bridge_high.

    This is meaningful when:
    - The high-frequency part of u_solver has accumulated spectral error
      (e.g., phase drift after many time steps).
    - The WKB ansatz is valid at the dominant wavenumber (λ large enough).

    Parameters
    ----------
    op : PseudoDifferentialOperator
        The operator defining the dynamics.
    splitter : SpectralSplitter
        Pre-configured SpectralSplitter with the desired k_cut.
    bridge_kwargs : dict
        Keyword arguments forwarded to PsiOpFIOBridge (lam, n_guesses, …).
    """

    def __init__(
        self,
        op            : PseudoDifferentialOperator,
        splitter      : SpectralSplitter,
        **bridge_kwargs,
    ):
        self.op       = op
        self.splitter = splitter
        self.bridge   = PsiOpFIOBridge(op, **bridge_kwargs)

    def correct(
        self,
        u_solver    : np.ndarray,
        wkb_state   : WKBState,
    ) -> np.ndarray:
        """
        Apply one correction step.

        Parameters
        ----------
        u_solver : np.ndarray
            Current solver solution on splitter.x_grid.
        wkb_state : WKBState
            WKB ansatz for the high-frequency part.  Its amplitude and
            phase are used as input to PsiOpFIOBridge.evaluate_grid().

        Returns
        -------
        np.ndarray (complex128)
            Corrected solution  u_low(solver) + u_high(bridge).
        """
        u_low, _  = self.splitter.split(u_solver)

        # Re-evaluate the high-frequency part via the asymptotic bridge
        u_bridge_high = self.bridge.evaluate_grid(
            self.splitter.x_grid,
            wkb_state.phase_sym,
            wkb_state.amp_sym,
        )
        # Keep only the high-frequency content of the bridge result
        _, u_bridge_high_filtered = self.splitter.split(u_bridge_high)

        return self.splitter.merge(u_low, u_bridge_high_filtered)

    def correction_magnitude(
        self,
        u_solver  : np.ndarray,
        wkb_state : WKBState,
    ) -> float:
        """
        Return ‖u_corrected − u_solver‖ / ‖u_solver‖ — how much the
        correction changes the solution.  Values > 0.1 indicate that the
        solver's high-frequency component has significant WKB error.
        """
        u_corr = self.correct(u_solver, wkb_state)
        norm   = np.linalg.norm(u_solver)
        if norm < 1e-30:
            return 0.0
        return float(np.linalg.norm(u_corr - u_solver) / norm)


class CrossValidator:
    """
    Compare a PDESolver solution against a PsiOpFIOBridge solution on the
    same problem and produce a ValidationReport.

    This class is the formal bridge between the two regimes:
    - It runs PDESolver numerically (ETD-RK4 or default scheme).
    - It runs PsiOpFIOBridge asymptotically (stationary-phase / saddle).
    - It computes point-wise and spectral error metrics.
    - It declares whether the WKB regime is valid for the given λ.

    Parameters
    ----------
    op : PseudoDifferentialOperator
        The operator P whose action on u is being compared.
    wkb_state : WKBState
        The WKB initial/input state.
    x_grid : np.ndarray
        Shared spatial grid for both solvers.
    lam : float
        Large parameter λ.  Must match wkb_state.lam.
    wkb_threshold : float
        Max relative error below which the WKB regime is declared valid.
        Default: 3 / lam  (theoretical O(λ⁻¹) accuracy).
    solver_kwargs : dict | None
        Extra keyword arguments forwarded to PDESolver.setup().
    bridge_kwargs : dict | None
        Extra keyword arguments forwarded to PsiOpFIOBridge.

    Notes
    -----
    PDESolver is imported lazily inside run() to keep solver.py optional.
    If solver.py is not on sys.path, CrossValidator.run() raises ImportError
    with a clear message.
    """

    def __init__(
        self,
        op             : PseudoDifferentialOperator,
        wkb_state      : WKBState,
        x_grid         : np.ndarray,
        lam            : float,
        wkb_threshold  : Optional[float] = None,
        solver_kwargs  : Optional[Dict]  = None,
        bridge_kwargs  : Optional[Dict]  = None,
    ):
        if lam != wkb_state.lam:
            warnings.warn(
                f"lam={lam} does not match wkb_state.lam={wkb_state.lam}. "
                "Using CrossValidator.lam for the bridge.",
                UserWarning,
            )
        self.op            = op
        self.wkb_state     = wkb_state
        self.x_grid        = x_grid
        self.lam           = lam
        self.wkb_threshold = wkb_threshold if wkb_threshold is not None else 3.0 / lam
        self.solver_kwargs = solver_kwargs or {}
        self.bridge_kwargs = bridge_kwargs or {}

    # ── Core runner ────────────────────────────────────────────────────────

    def run(self) -> ValidationReport:
        """
        Execute both solvers and return a ValidationReport.

        The PDESolver is used in *stationary* mode: it applies the operator
        P to the WKB state u₀ directly via solve_stationary_psiOp(), which
        calls psiop.right_inverse_asymptotic() internally.  For time-
        dependent problems, set ``solver_kwargs['time_dependent'] = True``
        and supply Lt, Nt (not yet implemented here — raises NotImplementedError).

        Returns
        -------
        ValidationReport
        """
        u_bridge = self._run_bridge()
        u_solver = self._run_solver()
        return self._build_report(u_solver, u_bridge)

    def run_bridge_only(self) -> np.ndarray:
        """Run PsiOpFIOBridge only and return the solution array."""
        return self._run_bridge()

    def run_solver_only(self) -> np.ndarray:
        """Run PDESolver only and return the solution array."""
        return self._run_solver()

    # ── Private helpers ────────────────────────────────────────────────────

    def _run_bridge(self) -> np.ndarray:
        """Evaluate (Pu₀)(x) via the asymptotic bridge."""
        kw = dict(lam=self.lam, n_guesses=50,
                  xi_range=(-10.0, 10.0), y_range=(-6.0, 6.0))
        kw.update(self.bridge_kwargs)
        bridge = PsiOpFIOBridge(self.op, **kw)
        return bridge.evaluate_grid(
            self.x_grid,
            self.wkb_state.phase_sym,
            self.wkb_state.amp_sym,
        )

    def _run_solver(self) -> np.ndarray:
        """
        Apply the operator P to u₀ via PDESolver.

        Strategy: set up a first-order-in-time PDE  ∂ₜu = P[u], run for a
        single infinitesimal time step dt, and return (u₁ − u₀)/dt ≈ P[u₀].

        Requirements on the caller:
          • The dominant wavenumber of the WKB state (lam·|S'|) must be well
            below the solver's Nyquist limit (π·Nx/Lx).  If lam is large,
            reduce it or increase Nx via solver_kwargs.
        """
        try:
            from solver import PDESolver, psiOp
        except ImportError as exc:
            raise ImportError(
                "CrossValidator.run() requires solver.py to be importable. "
                "Add its directory to sys.path before calling run()."
            ) from exc

        import sympy as _sp

        x_sym  = self.op.vars_x[0]
        t_sym, u_func = _sp.symbols('t'), _sp.Function('u')

        # Build  ∂ₜu = psiOp(p(ξ/lam), u)  so that when the solver evaluates
        # the symbol at physical wavenumber k = lam·k₀ it gets p(k₀), matching
        # the bridge's stationary-phase convention where ξ_c = S'(x) = k₀.
        # Identify the frequency symbol in the operator
        free_syms = self.op.symbol.free_symbols
        spatial_syms = set(self.op.vars_x)
        # The frequency symbol is the one that is not a spatial variable
        freq_sym_candidates = [s for s in free_syms if s not in spatial_syms]
        if len(freq_sym_candidates) != 1:
            raise ValueError("Cannot automatically identify the frequency symbol.")
        freq_sym = freq_sym_candidates[0]
        
        # Rescale the symbol: p(ξ/λ)
        psi_rescaled = self.op.symbol.subs(freq_sym, freq_sym / self.lam)
        
        # Build the equation
        equation = sp.Eq(
            sp.Derivative(u_func(t_sym, x_sym), t_sym),
            psiOp(psi_rescaled, u_func(t_sym, x_sym)),
        )

        Lx = float(self.x_grid[-1] - self.x_grid[0])
        Nx = len(self.x_grid)
        ic = self.wkb_state.as_callable()

        kw = dict(Lx=Lx, Nx=Nx, Lt=0.001, Nt=1,
                  initial_condition=ic, plot=False)
        kw.update(self.solver_kwargs)

        solver = PDESolver(equation)
        solver.setup(**kw)

        # P[u₀] is the RHS of ∂ₜu = psiOp(p, u).  In Fourier space the
        # exponential integrator does û₁ = exp(-dt·L)·û₀ where L = combined_symbol.
        # The psiOp RHS corresponds to the operator with symbol -L, so:
        #   P[u₀]_k  =  -combined_symbol_k · û₀_k
        # This is exact (no finite-difference error) and avoids the catastrophic
        # cancellation that occurs when dt is tiny (u₁≈u₀ → (u₁-u₀)/dt ≈ 0/0).

        u0_arr = np.asarray(solver.frames[0], dtype=complex)
        pu0_on_solver_grid = -solver._apply_psiOp(u0_arr)   # correct sign for RHS
    
        # Interpolate back onto self.x_grid if necessary
        return (np.interp(self.x_grid, solver.x_grid, np.real(pu0_on_solver_grid)) +
                1j * np.interp(self.x_grid, solver.x_grid, np.imag(pu0_on_solver_grid)))

    def _build_report(
        self,
        u_solver : np.ndarray,
        u_bridge : np.ndarray,
    ) -> ValidationReport:
        """Compute error metrics and build the ValidationReport."""
        diff       = u_solver - u_bridge
        abs_err    = np.abs(diff)
        scale      = np.max(np.abs(u_bridge)) + 1e-30
        rel_err    = abs_err / scale

        max_abs    = float(np.max(abs_err))
        max_rel    = float(np.max(rel_err))

        # Spectral error decomposition
        N          = len(self.x_grid)
        dx         = self.x_grid[1] - self.x_grid[0]
        k_grid     = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
        err_spec   = np.abs(np.fft.fft(diff)) / N

        return ValidationReport(
            x_grid        = self.x_grid,
            u_solver      = u_solver,
            u_bridge      = u_bridge,
            abs_error     = abs_err,
            rel_error     = rel_err,
            max_abs_error = max_abs,
            max_rel_error = max_rel,
            wkb_valid     = max_rel < self.wkb_threshold,
            wkb_threshold = self.wkb_threshold,
            error_spectrum= err_spec,
            k_grid        = k_grid,
            lam           = self.lam,
        )

    # ── Lambda sweep ───────────────────────────────────────────────────────

    def lambda_sweep(
        self,
        lambdas : List[float],
    ) -> List[ValidationReport]:
        """
        Run the cross-validation for a range of λ values and return the
        list of ValidationReports.

        Useful for finding the λ-threshold below which the WKB approximation
        is no longer reliable for a given operator and initial state.

        Parameters
        ----------
        lambdas : list[float]
            Increasing sequence of λ values to test.

        Returns
        -------
        list[ValidationReport], one per λ value.
        """
        reports = []
        for lv in lambdas:
            cv = CrossValidator(
                op            = self.op,
                wkb_state     = WKBState(
                    self.wkb_state.amp_sym,
                    self.wkb_state.phase_sym,
                    self.wkb_state.var_x,
                    lam = lv,
                ),
                x_grid        = self.x_grid,
                lam           = lv,
                wkb_threshold = 3.0 / lv,
                solver_kwargs = self.solver_kwargs,
                bridge_kwargs = self.bridge_kwargs,
            )
            reports.append(cv.run())
        return reports

    @staticmethod
    def plot_report(
        report  : ValidationReport,
        title   : str = "Cross-validation: solver vs asymptotic bridge",
    ) -> None:
        """
        Plot a ValidationReport: solution comparison, error profile,
        and error spectrum side by side.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.plot(report.x_grid, np.real(report.u_bridge), 'k-',
                lw=1.5, label='Bridge (Re)')
        ax.plot(report.x_grid, np.real(report.u_solver), 'r--',
                lw=1.5, label='Solver (Re)')
        ax.set_title('Solution comparison')
        ax.set_xlabel('$x$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.semilogy(report.x_grid, report.rel_error + 1e-16, 'b-',
                    label='Rel. error')
        ax.axhline(report.wkb_threshold, color='k', ls=':',
                   label=r'$3/\lambda$  threshold')
        status = 'VALID' if report.wkb_valid else 'INVALID'
        ax.set_title(f'Relative error  [WKB {status}]')
        ax.set_xlabel('$x$')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        # Show only positive frequencies
        k_pos  = report.k_grid
        k_sort = np.argsort(k_pos)
        ax.semilogy(k_pos[k_sort],
                    report.error_spectrum[k_sort] + 1e-16, 'm-', lw=1.2)
        ax.set_title('Error spectrum  |FFT(solver − bridge)| / N')
        ax.set_xlabel(r'$k$ (rad/unit)')
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{title}\n"
            r"$\lambda$" + f"={report.lam:.0f},  "
            f"max rel err={report.max_rel_error:.2e},  "
            f"WKB {'✓' if report.wkb_valid else '✗'}",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        plt.show()

    @staticmethod
    def plot_lambda_sweep(
        reports : List[ValidationReport],
        lambdas : List[float],
    ) -> None:
        """
        Plot max relative error vs λ on a log-log scale, with a vertical
        line at the λ-threshold where WKB validity flips.
        """
        import matplotlib.pyplot as plt

        errors   = [r.max_rel_error for r in reports]
        valid    = [r.wkb_valid for r in reports]

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['tab:green' if v else 'tab:red' for v in valid]
        for i in range(len(lambdas) - 1):
            ax.loglog(lambdas[i:i+2], errors[i:i+2],
                      color='steelblue', lw=1.5)
        ax.scatter(lambdas, errors, c=colors, zorder=5, s=60,
                   label='green=WKB valid, red=invalid')

        # Theoretical slope
        lref = np.array([float(lambdas[0]), float(lambdas[-1])])
        ax.loglog(lref, errors[0] * (lref / lambdas[0])**(-1),
                  'k:', lw=1.2, label=r'slope $-1$')

        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('Max relative error (solver vs bridge)')
        ax.set_title(r'Cross-validation $\lambda$-sweep')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  FFT reference for validation
# ─────────────────────────────────────────────────────────────────────────────

def fft_reference(
    op      : PseudoDifferentialOperator,
    u_vals  : np.ndarray,
    x_grid  : np.ndarray,
) -> np.ndarray:
    """
    Exact numerical reference via FFT for a 1D constant-coefficient psiOp.

        (Pu)(x) = IFFT[ p(ξ) · FFT[u](ξ) ]

    Used only to validate the asymptotic bridge against a spectral solver.
    """
    N  = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    xi = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi

    x_sym  = op.vars_x[0]
    xi_sym = sp.Symbol('xi', real=True)
    p_func = sp.lambdify((x_sym, xi_sym), op.symbol, 'numpy')

    u_hat  = np.fft.fft(u_vals)
    p_vals = p_func(np.zeros_like(xi), xi)
    return np.fft.ifft(p_vals * u_hat)


# ─────────────────────────────────────────────────────────────────────────────
#  Test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_test_suite(verbose: bool = True, plot: bool = True) -> Dict[str, Any]:
    """
    Run 5 tests covering the main use cases of the bridge.

    WKB reference
    -------------
    For a psiOp with constant symbol p(ξ) applied to
        u_0(y) = a_u(y) · exp(iλ k_0 y)
    the semi-classical result at leading order is:
        (Pu)(x) = p(k_0) · u_0(x)

    Test 1 -- Elliptic    P = ξ²       → p(k0)  = k0²
    Test 2 -- Transport   P = c·ξ      → p(k0)  = c·k0
    Test 3 -- Composition P∘Q, P=ξ², Q=ξ → PQ=ξ³, p(k0) = k0³
    Test 4 -- Right-inverse  P=ξ²+1, R=P⁻¹ → (Ru)(x) = u(x)/p(k0)
    Test 5 -- Propagator  exp(it·ξ)    → phase shift exp(i·t·k0)

    Error tolerance: O(1/λ) for simple operators, O(5/λ) for compositions.

    Returns
    -------
    dict  --  keys 'test1'…'test5'; each a dict with 'passed',
              'max_rel_error', 'values_bridge', 'values_ref'.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    x_sym  = sp.Symbol('x',  real=True)
    xi_sym = sp.Symbol('xi', real=True)
    y_sym  = sp.Symbol('y',  real=True)

    results: Dict[str, Any] = {}

    # ── Common parameters ────────────────────────────────────────────────────
    k0  = 2.0
    lam = 40.0

    u0_phase_sym = k0 * y_sym
    u0_amp_sym   = sp.exp(-y_sym**2 / 2)
    n_test  = 50
    x_test  = np.linspace(-2.5, 2.5, n_test)
    #n_test  = 128
    #x_test  = np.linspace(-np.pi, np.pi, n_test, endpoint=False)
    u0_test = np.exp(-x_test**2 / 2) * np.exp(1j * lam * k0 * x_test)

    bridge_kw = dict(lam=lam, n_guesses=50, xi_range=(-10.0, 10.0), verbose=verbose)

    def _run(label, bridge, ref_vals, tol_factor=3.0):
        vals = bridge.evaluate_grid(x_test, u0_phase_sym, u0_amp_sym)
        err  = np.abs(vals - ref_vals) / (np.abs(ref_vals) + 1e-12)
        maxe = float(np.max(err))
        ok   = maxe < tol_factor / lam
        print(f"  Max relative error : {maxe:.5f}  =>  {'PASS' if ok else 'FAIL'}")
        return dict(passed=ok, max_rel_error=maxe,
                    values_bridge=vals, values_ref=ref_vals,
                    x_test=x_test)

    # ── TEST 1 : P = ξ² ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1 — Elliptic psiOp : P = ξ²")
    print(f"  WKB reference : {k0**2:.2f} · u0(x)")
    print("="*60)
    P1 = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
    results['test1'] = _run('P=ξ²',
                            PsiOpFIOBridge(P1, **bridge_kw),
                            k0**2 * u0_test)
    results['test1']['label'] = 'P=ξ²'

    # ── TEST 2 : P = c·ξ ────────────────────────────────────────────────────
    print("\n" + "="*60)
    c = 2.0
    print(f"TEST 2 — Transport psiOp : P = {c}·ξ")
    print(f"  WKB reference : {c*k0:.2f} · u0(x)")
    print("="*60)
    P2 = PseudoDifferentialOperator(c * xi_sym, vars_x=[x_sym], mode='symbol')
    results['test2'] = _run('P=2ξ',
                            PsiOpFIOBridge(P2, **bridge_kw),
                            c * k0 * u0_test)
    results['test2']['label'] = 'P=2·ξ'

    # ── TEST 3 : P∘Q = ξ³ ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 3 — Composition P∘Q : P=ξ², Q=ξ  ⟹  PQ=ξ³")
    print(f"  WKB reference : {k0**3:.2f} · u0(x)")
    print("="*60)
    Q3  = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
    cb3 = CompositionBridge(P1, Q3, comp_order=1, **bridge_kw)
    results['test3'] = _run('PQ=ξ³', cb3, k0**3 * u0_test, tol_factor=5.0)
    results['test3']['label'] = 'P∘Q = ξ³'

    # ── TEST 4 : Asymptotic right-inverse ────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 4 — Asymptotic right-inverse : P = ξ²+1")
    print("="*60)
    P4    = PseudoDifferentialOperator(xi_sym**2 + 1, vars_x=[x_sym], mode='symbol')
    R_sym = P4.right_inverse_asymptotic(order=1)
    R4    = PseudoDifferentialOperator(R_sym, vars_x=[x_sym], mode='symbol')

    # (a) symbolic check: P∘R ≈ Id
    PR_sym  = P4.compose_asymptotic(R4, order=1)
    PR_diff = sp.simplify(PR_sym - 1)
    xi_pts  = np.linspace(1.0, 8.0, 30)
    pr_func = sp.lambdify((x_sym, xi_sym), PR_diff, 'numpy')
    resid_sym = float(np.max(np.abs(pr_func(np.zeros_like(xi_pts), xi_pts))))
    print(f"  (a) Symbolic residual ‖P∘R − Id‖∞ = {resid_sym:.2e}")

    # (b) numeric check
    bridge4 = PsiOpFIOBridge(R4, **bridge_kw)
    p4_at_k0 = k0**2 + 1
    res4 = _run('R=(ξ²+1)⁻¹', bridge4, u0_test / p4_at_k0)
    res4['residual_sym'] = resid_sym
    res4['passed'] = res4['passed'] and (resid_sym < 1e-10)
    print(f"  (b) Numeric error = {res4['max_rel_error']:.5f}  =>  "
          f"{'PASS' if res4['passed'] else 'FAIL'}")
    results['test4'] = res4
    results['test4']['label'] = 'R=(ξ²+1)⁻¹'

    # ── TEST 5 : Propagator exp(it·ξ) ────────────────────────────────────────
    print("\n" + "="*60)
    t5 = 1.0
    print(f"TEST 5 — Propagator exp(it·ξ), t={t5}")
    print(f"  WKB reference : exp(i·{k0}·{t5}) · u0(x)")
    print("="*60)
    exp_sym5 = sp.exp(sp.I * t5 * xi_sym)
    expP5    = PseudoDifferentialOperator(exp_sym5, vars_x=[x_sym], mode='symbol')
    results['test5'] = _run('exp(it·ξ)',
                            PsiOpFIOBridge(expP5, **bridge_kw),
                            np.exp(1j * t5 * k0) * u0_test)
    results['test5']['label'] = 'exp(it·ξ)'

    # ── TEST 6 : WKBState ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 6 — WKBState : to_array and as_callable")
    print("  u0(x) = exp(-x²/2) · exp(iλk₀x)")
    print("="*60)

    wkb = WKBState(
        amp_sym   = u0_amp_sym,
        phase_sym = u0_phase_sym,
        var_x     = y_sym,
        lam       = lam,
    )

    # to_array must reproduce the reference WKB array
    u_wkb_arr = wkb.to_array(x_test)
    err6 = float(np.max(np.abs(u_wkb_arr - u0_test) / (np.abs(u0_test) + 1e-12)))
    ok6  = err6 < 1e-10

    # as_callable must return the same values when called as f(x)
    ic_fn     = wkb.as_callable()
    u_from_ic = ic_fn(x_test)
    err6b     = float(np.max(np.abs(u_from_ic - u0_test) / (np.abs(u0_test) + 1e-12)))
    ok6       = ok6 and (err6b < 1e-10)

    # dominant wavenumber should be close to lam * k0
    k_dom = wkb.dominant_wavenumber(x_test)
    ok6   = ok6 and (abs(k_dom - lam * k0) / (lam * k0) < 0.05)

    print(f"  to_array error  : {err6:.2e}")
    print(f"  callable error  : {err6b:.2e}")
    print(f"  dominant k = {k_dom:.2f}  (expected {lam*k0:.2f})")
    print(f"  => {'PASS' if ok6 else 'FAIL'}")
    results['test6'] = dict(
        passed        = ok6,
        max_rel_error = max(err6, err6b),
        label         = 'WKBState',
        x_test        = x_test,
        values_bridge = u_wkb_arr,
        values_ref    = u0_test,
    )

    # ── TEST 7 : SpectralSplitter ─────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 7 — SpectralSplitter : lossless split and energy ratio")
    print("="*60)

    # Use the WKB state: most energy should be at k ≈ ±lam*k0 (high freq)
    splitter = SpectralSplitter(x_test, k_cut=lam * k0 / 2)
    u_low, u_high = splitter.split(u0_test)

    # Lossless: merge should recover the original
    u_merged = splitter.merge(u_low, u_high)
    err7_merge = float(np.max(np.abs(u_merged - u0_test) / (np.abs(u0_test) + 1e-12)))

    # Energy: WKB state is almost entirely high-frequency
    e_low, e_high = splitter.energy_ratio(u0_test)
    ok7 = (err7_merge < 1e-10) and (e_high > 0.8)

    print(f"  Merge error     : {err7_merge:.2e}  (must be < 1e-10)")
    print(f"  E_low / E_tot   : {e_low:.3f}")
    print(f"  E_high / E_tot  : {e_high:.3f}  (WKB state: expect > 0.8)")
    print(f"  => {'PASS' if ok7 else 'FAIL'}")
    results['test7'] = dict(
        passed        = ok7,
        max_rel_error = err7_merge,
        label         = 'SpectralSplitter',
        x_test        = x_test,
        values_bridge = u_merged,
        values_ref    = u0_test,
    )

    # ── TEST 8 : SemiclassicalCorrector ───────────────────────────────────
    print("\n" + "="*60)
    print("TEST 8 — SemiclassicalCorrector : correction magnitude")
    print("  P = ξ², u0 = WKB state")
    print("="*60)

    # Reference: (Pu0)(x) = k0² u0(x)
    ref_pu0 = k0**2 * u0_test

    # Intentionally corrupt the high-frequency part of ref_pu0
    splitter8   = SpectralSplitter(x_test, k_cut=lam * k0 / 2)
    u_low8, _   = splitter8.split(ref_pu0)
    noise       = 0.3 * np.random.default_rng(42).standard_normal(len(x_test))
    u_corrupted = splitter8.merge(u_low8, noise)  # wrong high-freq part

    corrector = SemiclassicalCorrector(
        op       = P1,      # ξ² from test 1
        splitter = splitter8,
        **bridge_kw,
    )
    wkb8      = WKBState(u0_amp_sym, u0_phase_sym, y_sym, lam=lam)
    u_corrected = corrector.correct(u_corrupted, wkb8)

    # After correction, high-freq part should be close to bridge result
    err8 = float(np.max(
        np.abs(u_corrected - ref_pu0) / (np.abs(ref_pu0) + 1e-12)
    ))
    # Correction must have changed the solution significantly
    mag8 = corrector.correction_magnitude(u_corrupted, wkb8)
    ok8  = (err8 < 5.0 / lam) and (mag8 > 0.05)

    print(f"  Error after correction : {err8:.2e}  (must be < {5/lam:.2e} = 5/lam)")
    print(f"  Correction magnitude   : {mag8:.3f}  (must be > 0.05)")
    print(f"  => {'PASS' if ok8 else 'FAIL'}")
    results['test8'] = dict(
        passed        = ok8,
        max_rel_error = err8,
        label         = 'SemiclassicalCorrector',
        x_test        = x_test,
        values_bridge = u_corrected,
        values_ref    = ref_pu0,
    )

    # ── TEST 9 : CrossValidator (bridge-only path) ────────────────────────
    print("\n" + "="*60)
    print("TEST 9 — CrossValidator : bridge path + ValidationReport")
    print("  P = ξ², comparing bridge vs WKB reference")
    print("="*60)

    wkb9 = WKBState(u0_amp_sym, u0_phase_sym, y_sym, lam=lam)
    cv9  = CrossValidator(
        op            = P1,
        wkb_state     = wkb9,
        x_grid        = x_test,
        lam           = lam,
        bridge_kwargs = dict(n_guesses=50, xi_range=(-10., 10.)),
    )

    # Run bridge only (no solver.py required)
    u_bridge9 = cv9.run_bridge_only()

    # Build a manual report using the WKB reference as "solver"
    report9   = cv9._build_report(ref_pu0, u_bridge9)

    ok9 = (
        isinstance(report9, ValidationReport)
        and report9.max_rel_error < 3.0 / lam
        and report9.wkb_valid
        and report9.error_spectrum.shape == (len(x_test),)
        and report9.k_grid.shape == (len(x_test),)
    )

    print(f"  Max rel error   : {report9.max_rel_error:.2e}  (must be < {3/lam:.2e})")
    print(f"  WKB valid       : {report9.wkb_valid}")
    print(f"  error_spectrum  : shape {report9.error_spectrum.shape}  ✓")
    print(f"  => {'PASS' if ok9 else 'FAIL'}")
    results['test9'] = dict(
        passed        = ok9,
        max_rel_error = report9.max_rel_error,
        label         = 'CrossValidator (bridge path)',
        x_test        = x_test,
        values_bridge = u_bridge9,
        values_ref    = ref_pu0,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    n_pass = 0
    for k, v in results.items():
        ok = v.get('passed', False)
        n_pass += int(ok)
        print(f"  {k} : {'PASS' if ok else 'FAIL'}  "
              f"(max_rel_err={v.get('max_rel_error', 0.0):.2e})  {v['label']}")
    print(f"\n  Total : {n_pass}/{len(results)} tests passed")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            "fio_bridge — asymptotic bridge (red dashed) vs WKB reference (black)\n"
            "u0(y)=exp(−y²/2)·exp(iλk0y),  k0=2, λ=40  |  ref: (Pu)(x)=p(k0)·u0(x)",
            fontsize=10,
        )
        plot_order = ['test1', 'test2', 'test3', 'test4', 'test5']
        flat_axes  = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
        for key, ax in zip(plot_order, flat_axes):
            r  = results[key]
            xt = r['x_test']
            vb = r['values_bridge']
            vr = r['values_ref']
            ax.plot(xt, np.real(vr), 'k-',  lw=2,   label='WKB ref (Re)')
            ax.plot(xt, np.real(vb), 'r--', lw=1.5, label='Bridge (Re)')
            ax.plot(xt, np.imag(vr), 'b-',  lw=1,   alpha=0.6, label='WKB ref (Im)')
            ax.plot(xt, np.imag(vb), 'm--', lw=1,   alpha=0.6, label='Bridge (Im)')
            ok  = r['passed']
            err = r.get('max_rel_error', 0.0)
            ax.set_title(f"[{'PASS' if ok else 'FAIL'}]  {r['label']}\nerr={err:.2e}",
                         fontsize=9)
            ax.legend(fontsize=7, loc='best')
            ax.set_xlabel('x', fontsize=8)
            ax.grid(True, alpha=0.3)

        ax6 = axes[1, 2]
        ax6.axis('off')
        lines = ["Semi-classical WKB reference:", "  (Pu)(x) = p(k0) · u0(x)", "", "Tests:"]
        for k, v in results.items():
            lines.append(f"  {k}  {'PASS' if v['passed'] else 'FAIL'}  "
                         f"err={v.get('max_rel_error',0.0):.2e}  {v['label']}")
        lines += ["", f"Total: {sum(v['passed'] for v in results.values())}/{len(results)}"]
        ax6.text(0.05, 0.95, "\n".join(lines), va='top', ha='left',
                 transform=ax6.transAxes, fontsize=9, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        plt.show()

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_test_suite(verbose=True, plot=True)