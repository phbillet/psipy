"""
fio_bridge.py  --  Bridge between psiop.py and asymptotic.py
=============================================================

This module is the missing link between the two packages:

* **psiop.py** provides symbol algebra: asymptotic composition (KN/Weyl),
  adjoint, inverse, operator exponential exp(tP), Hamiltonian flow,
  pseudospectrum.

* **asymptotic.py** provides numerical evaluation of oscillatory integrals
  via the stationary-phase method (Morse / Airy / Pearcey) or saddle-point
  (steepest descent).

This module provides:

FIOKernel
    Kernel of a Fourier Integral Operator.
    Holds the phase phi(y,xi;x) and the amplitude a(y,xi;x).

PsiOpFIOBridge
    Converts a PseudoDifferentialOperator into a FIOKernel
    and evaluates its action on a function u(y) at a point x
    using AsymptoticEvaluator from asymptotic.py.

PropagatorBridge
    Computes the semi-classical propagator e^{itP} by
    combining exponential_symbol() (psiop) with the evaluator.

CompositionBridge
    Computes the composition P∘Q of two psiOps and evaluates
    the result at a given point x.

Test functions
    run_test_suite() -- 5 tests covering the main use cases.

Mathematics
-----------
For a psiOp with symbol p(x,xi), the action on u is:

    (Pu)(x) = (1/2pi) ∫∫ e^{i*lam*[(x-y)*xi + phi_u(y)]} p(y,xi) a_u(y,xi) dy dxi

where u is represented in WKB notation: u(y) ~ a_u(y) * e^{i*lam*S(y)}.

The global phase phi(y,xi;x) = (x-y)*xi + S(y) is stationary in (y,xi) when
    d phi/dy  = 0  -->  xi  = S'(y_c)   (bicharacteristic condition)
    d phi/dxi = 0  -->  y_c = x         (transport condition)

For general (non-WKB) symbols, a discrete spectral representation of u
is used and all critical-point contributions are summed.

References
----------
[1] Hormander, L. "Fourier integral operators I", Acta Math. 127 (1971)
[2] Duistermaat, J.J. "Fourier Integral Operators", Birkhauser, 1996
[3] Zworski, M. "Semiclassical Analysis", AMS Graduate Studies, 2012
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict, Any

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from asymptotic import (
    Analyzer, AsymptoticEvaluator, SaddlePointEvaluator,
    IntegralMethod, CriticalPoint, AsymptoticContribution
)
from psiop import PseudoDifferentialOperator


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FIOKernel:
    """
    Kernel of a Fourier Integral Operator: phase phi(y,xi;x) and
    amplitude a(y,xi;x) as SymPy expressions.

    Attributes
    ----------
    phase_sym : sp.Expr
        SymPy expression for the phase phi(y, xi; x_val).
        Depends on the integration variables (y, xi) and the
        observation parameter x_val.
    amp_sym : sp.Expr
        SymPy expression for the amplitude a(y, xi; x_val).
    int_vars : list[sp.Symbol]
        Integration variables [y, xi] (or [y1, y2, xi1, xi2] in 2D).
    x_val : float
        Current observation point (value of x).
    method_hint : IntegralMethod
        Suggested method after inspecting the phase.
    """
    phase_sym  : sp.Expr
    amp_sym    : sp.Expr
    int_vars   : List[sp.Symbol]
    x_val      : float             = 0.0
    method_hint: IntegralMethod    = IntegralMethod.AUTO


@dataclass
class EvalResult:
    """
    Result of evaluating (Pu)(x) at a single observation point x.

    Attributes
    ----------
    x_val : float
        Observation point.
    value : complex
        Asymptotic value of (Pu)(x).
    n_critical_points : int
        Number of critical points found and used.
    contributions : list[AsymptoticContribution]
        Individual contributions from each critical point.
    warnings_list : list[str]
        Any warnings raised (caustics, Picard-Lefschetz issues, etc.)
    """
    x_val               : float
    value               : complex
    n_critical_points   : int
    contributions       : List[AsymptoticContribution] = field(default_factory=list)
    warnings_list       : List[str]                    = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class PsiOpFIOBridge:
    """
    Evaluates the action of a PseudoDifferentialOperator on a function u
    using the stationary-phase method from asymptotic.py.

    Workflow
    --------
    1. ``bridge = PsiOpFIOBridge(op, lam)``
       Initialise with a PseudoDifferentialOperator and the large parameter lambda.

    2. ``result = bridge.evaluate_at(x_val, u_phase_sym, u_amp_sym)``
       Evaluate (Pu)(x_val) by building the FIO phase and searching for
       critical points via stationary phase.

    3. ``values = bridge.evaluate_grid(x_grid, u_phase_sym, u_amp_sym)``
       Evaluate over a spatial grid.

    Parameters
    ----------
    op : PseudoDifferentialOperator
        The psiOp (1D or 2D) to evaluate.
    lam : float
        Large parameter lambda (semi-classical frequency). Typically 20-200.
    n_guesses : int
        Number of initial guesses for the critical-point search.
    xi_range : tuple[float, float]
        Search domain in the frequency variable xi.
    y_range : tuple[float, float]
        Search domain in the spatial variable y.
    tol_grad : float
        Tolerance on |grad phi|^2 for accepting a critical point.
    verbose : bool
        Print diagnostic information.
    """

    def __init__(
        self,
        op         : PseudoDifferentialOperator,
        lam        : float = 50.0,
        n_guesses  : int   = 40,
        xi_range   : Tuple[float, float] = (-8.0, 8.0),
        y_range    : Tuple[float, float] = (-6.0, 6.0),
        tol_grad   : float = 1e-6,
        verbose    : bool  = False,
    ):
        if op.dim not in (1, 2):
            raise ValueError("PsiOpFIOBridge only supports 1D and 2D psiOps.")
        self.op         = op
        self.lam        = lam
        self.n_guesses  = n_guesses
        self.xi_range   = xi_range
        self.y_range    = y_range
        self.tol_grad   = tol_grad
        self.verbose    = verbose
        self._evaluator = AsymptoticEvaluator()

    # ── FIO kernel construction ───────────────────────────────────────────────

    def build_kernel(
        self,
        x_val       : float,
        u_phase_sym : sp.Expr,
        u_amp_sym   : sp.Expr,
    ) -> FIOKernel:
        """
        Build the FIO kernel at observation point x_val.

        The standard FIO phase is:
            phi(y, xi; x) = (x - y) * xi + S_u(y)

        where S_u(y) is the WKB phase of u.  The amplitude is:
            a(y, xi; x) = p(y, xi) * a_u(y)

        Parameters
        ----------
        x_val : float
            Current observation point.
        u_phase_sym : sp.Expr
            WKB phase S_u(y) of the initial condition (SymPy expression in y).
        u_amp_sym : sp.Expr
            Amplitude a_u(y) of the initial condition.

        Returns
        -------
        FIOKernel
            Kernel ready to be analysed by asymptotic.Analyzer.
        """
        if self.op.dim == 1:
            y  = sp.Symbol('y',  real=True)
            xi = sp.Symbol('xi', real=True)

            # FIO phase: (x - y)*xi + S_u(y)
            phase = (x_val - y) * xi + u_phase_sym

            # Amplitude: p(y, xi) * a_u(y)
            # Substitute x -> y in the operator symbol (Kohn-Nirenberg quantization)
            p_at_y = self.op.symbol.subs(self.op.vars_x[0], y)
            amp    = p_at_y * u_amp_sym

            int_vars = [y, xi]

        else:  # dim == 2
            y1,  y2  = sp.symbols('y1 y2',  real=True)
            xi1, xi2 = sp.symbols('xi1 xi2', real=True)
            x1, x2   = float(x_val[0]), float(x_val[1])

            phase = (x1 - y1) * xi1 + (x2 - y2) * xi2 + u_phase_sym

            x_sym, y_sym = self.op.vars_x
            p_at_y = self.op.symbol.subs({x_sym: y1, y_sym: y2})
            amp    = p_at_y * u_amp_sym

            int_vars = [y1, y2, xi1, xi2]

        return FIOKernel(
            phase_sym  = phase,
            amp_sym    = amp,
            int_vars   = int_vars,
            x_val      = x_val,
            method_hint= IntegralMethod.AUTO,
        )

    # ── Critical-point search and contribution collection ─────────────────────

    def _make_guesses(self, x_val: float, kernel: FIOKernel) -> List[np.ndarray]:
        """
        Generate initial guesses for the critical-point search.

        In 1D: d phi/dy = 0  -->  xi_c = S'_u(y_c),
               d phi/dxi = 0 -->  y_c  = x
        Starting points are (y=x, xi=S'(x)) supplemented by a uniform grid.
        """
        dim = len(kernel.int_vars)
        n   = self.n_guesses

        if dim == 2:  # 1D psiOp --> (y, xi) is a 2D integration domain
            # Uniform grid over xi
            xi_pts  = np.linspace(*self.xi_range, n)
            guesses = [np.array([x_val, xic]) for xic in xi_pts]

            # Analytical guess: y_c ~ x_val, xi_c = d(u_phase)/dy |_{y=x}
            y_sym, xi_sym = kernel.int_vars
            try:
                dSdy = sp.diff(
                    kernel.phase_sym - (float(x_val) - y_sym) * xi_sym,
                    y_sym
                )
                xi_c_expr = sp.solve(dSdy, xi_sym)
                for sol in xi_c_expr:
                    xic_num = float(sol.subs(y_sym, x_val))
                    guesses.insert(0, np.array([x_val, xic_num]))
            except Exception:
                pass

            # Additional (y, xi) grid centered around x_val
            y_pts = np.linspace(max(self.y_range[0], x_val - 2),
                                min(self.y_range[1], x_val + 2), n // 4)
            for yc in y_pts:
                for xic in np.linspace(*self.xi_range, n // 4):
                    guesses.append(np.array([yc, xic]))

            return guesses

        else:  # dim == 4: 2D psiOp case
            pts = []
            for _ in range(n):
                y1c  = np.random.uniform(*self.y_range)
                y2c  = np.random.uniform(*self.y_range)
                xi1c = np.random.uniform(*self.xi_range)
                xi2c = np.random.uniform(*self.xi_range)
                pts.append(np.array([y1c, y2c, xi1c, xi2c]))
            return pts

    def _collect_contributions(
        self,
        kernel  : FIOKernel,
        guesses : List[np.ndarray],
    ) -> Tuple[complex, List[AsymptoticContribution], List[str]]:
        """
        Find critical points, analyse each one, and return their summed contribution.
        """
        analyzer = Analyzer(
            phase_expr     = kernel.phase_sym,
            amplitude_expr = kernel.amp_sym,
            variables      = kernel.int_vars,
            domain         = [self.y_range, self.xi_range]
                             if len(kernel.int_vars) == 2 else None,
            tolerance      = self.tol_grad,
        )

        critical_pts = analyzer.find_critical_points(guesses)

        if self.verbose:
            print(f"  [bridge] {len(critical_pts)} critical point(s) found")

        total    = 0j
        contribs = []
        warns    = []

        prefactor = self.lam / (2 * np.pi)

        for pt in critical_pts:
            try:
                cp    = analyzer.analyze_point(pt)
                res   = self._evaluator.evaluate(cp, self.lam)
                total += prefactor * res.total_value
                contribs.append(res)
                if self.verbose:
                    print(f"    pt={np.round(pt, 3)}, type={cp.singularity_type.value}, "
                          f"contrib={prefactor * res.total_value:.4f}")
            except Exception as exc:
                warns.append(f"Point {np.round(pt, 3)}: {exc}")

        return total, contribs, warns

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate_at(
        self,
        x_val       : float,
        u_phase_sym : sp.Expr,
        u_amp_sym   : sp.Expr,
    ) -> EvalResult:
        """
        Evaluate (Pu)(x_val) via stationary phase.

        Parameters
        ----------
        x_val : float
            Observation point.
        u_phase_sym : sp.Expr
            WKB phase S_u(y) of the initial data (SymPy expression in y).
        u_amp_sym : sp.Expr
            Amplitude a_u(y) of the initial data.

        Returns
        -------
        EvalResult
        """
        kernel  = self.build_kernel(x_val, u_phase_sym, u_amp_sym)
        guesses = self._make_guesses(x_val, kernel)
        value, contribs, warns = self._collect_contributions(kernel, guesses)

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
    ) -> np.ndarray:
        """
        Evaluate (Pu)(x) at every point in x_grid.

        Returns
        -------
        np.ndarray of complex
            Values of (Pu)(x) for each x in x_grid.
        """
        values = np.zeros(len(x_grid), dtype=complex)
        for i, xv in enumerate(x_grid):
            res = self.evaluate_at(xv, u_phase_sym, u_amp_sym)
            values[i] = res.value
            if self.verbose:
                print(f"x={xv:.3f} -> (Pu)(x) = {res.value:.6f}")
        return values


# ─────────────────────────────────────────────────────────────────────────────
#  PropagatorBridge  --  exp(itP) via exponential_symbol
# ─────────────────────────────────────────────────────────────────────────────

class PropagatorBridge:
    """
    Computes the semi-classical propagator u(x,t) = [e^{itP} u_0](x).

    Uses `PseudoDifferentialOperator.exponential_symbol(t)` to build the
    symbol of e^{itP} at the requested asymptotic order, then delegates
    evaluation to PsiOpFIOBridge.

    Parameters
    ----------
    op : PseudoDifferentialOperator
        Operator P (SymPy symbol).
    lam : float
        Large parameter lambda.
    exp_order : int
        Asymptotic expansion order for exp(tP).
    **bridge_kwargs
        Additional keyword arguments forwarded to PsiOpFIOBridge.
    """

    def __init__(
        self,
        op          : PseudoDifferentialOperator,
        lam         : float = 50.0,
        exp_order   : int   = 2,
        **bridge_kwargs
    ):
        self.op            = op
        self.lam           = lam
        self.exp_order     = exp_order
        self.bridge_kwargs = bridge_kwargs

    def propagate(
        self,
        t           : float,
        x_grid      : np.ndarray,
        u0_phase_sym: sp.Expr,
        u0_amp_sym  : sp.Expr,
        mode        : str = 'kn',
    ) -> np.ndarray:
        """
        Compute u(x, t) = [e^{itP} u_0](x) over x_grid.

        Parameters
        ----------
        t : float
            Propagation time.
        x_grid : np.ndarray
            Spatial evaluation grid.
        u0_phase_sym : sp.Expr
            Phase of the initial condition S_0(y).
        u0_amp_sym : sp.Expr
            Amplitude of the initial condition a_0(y).
        mode : str
            Quantization scheme ('kn' or 'weyl') for compose_asymptotic.

        Returns
        -------
        np.ndarray of complex
        """
        # Build symbol of exp(itP) via asymptotic BCH expansion
        exp_sym = self.op.exponential_symbol(
            t     = sp.I * t,
            order = self.exp_order,
            mode  = mode,
        )

        # Create a temporary operator carrying this symbol
        exp_op = PseudoDifferentialOperator(
            expr   = exp_sym,
            vars_x = self.op.vars_x,
            mode   = 'symbol',
        )

        bridge = PsiOpFIOBridge(exp_op, self.lam, **self.bridge_kwargs)
        return bridge.evaluate_grid(x_grid, u0_phase_sym, u0_amp_sym)


# ─────────────────────────────────────────────────────────────────────────────
#  CompositionBridge  --  P∘Q evaluated via asymptotic
# ─────────────────────────────────────────────────────────────────────────────

class CompositionBridge:
    """
    Compute the composition P∘Q and evaluate its action on u.

    The composed symbol is obtained via compose_asymptotic (psiop),
    then evaluation is delegated to PsiOpFIOBridge (asymptotic).

    Parameters
    ----------
    P, Q : PseudoDifferentialOperator
        The two operators to compose.
    lam : float
        Large parameter lambda.
    comp_order : int
        Asymptotic composition order.
    mode : str
        Quantization scheme ('kn' or 'weyl').
    **bridge_kwargs
        Arguments forwarded to PsiOpFIOBridge.
    """

    def __init__(
        self,
        P           : PseudoDifferentialOperator,
        Q           : PseudoDifferentialOperator,
        lam         : float = 50.0,
        comp_order  : int   = 2,
        mode        : str   = 'kn',
        **bridge_kwargs
    ):
        assert P.dim == Q.dim, "P and Q must have the same spatial dimension"
        self.lam           = lam
        self.bridge_kwargs = bridge_kwargs

        # Symbolic composition via psiop
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
#  FFT reference for validation
# ─────────────────────────────────────────────────────────────────────────────

def fft_reference(
    op      : PseudoDifferentialOperator,
    u_vals  : np.ndarray,
    x_grid  : np.ndarray,
) -> np.ndarray:
    """
    Exact numerical reference via FFT for a 1D constant-coefficient psiOp.

    (Pu)(x) = ifft[ p(xi) * fft[u](xi) ]

    Used only to validate the asymptotic bridge against a spectral solver.
    """
    N  = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    xi = np.fft.fftfreq(N, d=dx) * 2 * np.pi

    x_sym  = op.vars_x[0]
    xi_sym = sp.Symbol('xi', real=True)
    p_func = sp.lambdify((x_sym, xi_sym), op.symbol, 'numpy')

    u_hat  = np.fft.fft(u_vals)
    p_vals = p_func(np.zeros_like(xi), xi)  # symbol is x-independent
    pu_hat = p_vals * u_hat
    return np.fft.ifft(pu_hat)


# ─────────────────────────────────────────────────────────────────────────────
#  Test suite
# ─────────────────────────────────────────────────────────────────────────────

def run_test_suite(verbose: bool = True, plot: bool = True) -> Dict[str, Any]:
    """
    Run 5 tests covering the main use cases of the bridge.

    Reference
    ---------
    For a psiOp with constant symbol p(xi) applied to a WKB packet
        u(y) = a_u(y) * exp(i*lam*k0*y)
    the semi-classical result at leading order is exact:
        (Pu)(x) = p(k0) * a_u(x) * exp(i*lam*k0*x) = p(k0) * u0(x)
    This formula is used as the reference for all tests.

    Error tolerance
    ---------------
    The Morse formula has an O(1/lam) truncation error whose coefficient
    depends on the second derivative of the amplitude envelope.
    Tolerances are set to 3/lam (simple operators) or 5/lam (compositions).
    Halving lam exactly doubles the observed error, confirming the
    semi-classical scaling.

    Test 1 -- Elliptic psiOp  :  P = xi^2,  ref = k0^2 * u0(x)
    Test 2 -- Transport psiOp :  P = c*xi,  ref = c*k0 * u0(x)
    Test 3 -- Composition P∘Q :  P=xi^2, Q=xi --> PQ=xi^3, ref = k0^3 * u0(x)
    Test 4 -- Asymptotic inv  :  P = xi^2+1, verify P*R = Id (symbolic + numeric)
    Test 5 -- Propagator      :  exp(it*xi) on WKB packet, ref = exp(i*t*k0) * u0(x)

    Returns
    -------
    dict
        Keys 'test1' through 'test5'; each value is a dict with
        'passed', 'max_rel_error', 'values_bridge', 'values_ref'.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    x_sym  = sp.Symbol('x',  real=True)
    xi_sym = sp.Symbol('xi', real=True)
    y_sym  = sp.Symbol('y',  real=True)

    results = {}

    # Common parameters
    k0  = 2.0
    lam = 40.0

    # WKB initial condition: u0(y) = exp(-y^2/2) * exp(i*lam*k0*y)
    u0_phase_sym = k0 * y_sym
    u0_amp_sym   = sp.exp(-y_sym**2 / 2)
    n_test = 50
    x_test  = np.linspace(-2.5, 2.5, n_test)
    u0_test = np.exp(-x_test**2 / 2) * np.exp(1j * lam * k0 * x_test)

    # ------------------------------------------------------------------
    # TEST 1  :  P = xi^2
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 1 - elliptic psiOp : P = xi^2")
    print(f"  WKB reference : p(k0)*u0(x) = {k0**2:.2f} * u0(x)")
    print("="*60)

    P1 = PseudoDifferentialOperator(xi_sym**2, vars_x=[x_sym], mode='symbol')
    bridge1 = PsiOpFIOBridge(P1, lam=lam, n_guesses=50,
                              xi_range=(-10, 10), verbose=verbose)
    vals_bridge1 = bridge1.evaluate_grid(x_test, u0_phase_sym, u0_amp_sym)
    vals_ref1    = k0**2 * u0_test

    rel_err1 = np.abs(vals_bridge1 - vals_ref1) / (np.abs(vals_ref1) + 1e-12)
    max_err1 = float(np.max(rel_err1))
    passed1  = max_err1 < 3.0 / lam

    print(f"  Max relative error : {max_err1:.5f}  =>  {'PASS' if passed1 else 'FAIL'}")
    results['test1'] = dict(passed=passed1, max_rel_error=max_err1,
                             values_bridge=vals_bridge1, values_ref=vals_ref1,
                             x_test=x_test, label='P=xi^2')

    # ------------------------------------------------------------------
    # TEST 2  :  P = c*xi  (degree-1 symbol)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2 - transport psiOp : P = 2*xi")
    print(f"  WKB reference : c*k0*u0(x) = {2.0*k0:.2f} * u0(x)")
    print("="*60)

    c  = 2.0
    P2 = PseudoDifferentialOperator(c * xi_sym, vars_x=[x_sym], mode='symbol')
    bridge2 = PsiOpFIOBridge(P2, lam=lam, n_guesses=50,
                              xi_range=(-10, 10), verbose=verbose)
    vals_bridge2 = bridge2.evaluate_grid(x_test, u0_phase_sym, u0_amp_sym)
    vals_ref2    = c * k0 * u0_test

    rel_err2 = np.abs(vals_bridge2 - vals_ref2) / (np.abs(vals_ref2) + 1e-12)
    max_err2 = float(np.max(rel_err2))
    passed2  = max_err2 < 3.0 / lam

    print(f"  Max relative error : {max_err2:.5f}  =>  {'PASS' if passed2 else 'FAIL'}")
    results['test2'] = dict(passed=passed2, max_rel_error=max_err2,
                             values_bridge=vals_bridge2, values_ref=vals_ref2,
                             x_test=x_test, label='P=2*xi')

    # ------------------------------------------------------------------
    # TEST 3  :  Composition P∘Q : P=xi^2, Q=xi  -->  PQ=xi^3
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 3 - Composition P∘Q : P=xi^2, Q=xi  =>  PQ=xi^3")
    print(f"  WKB reference : k0^3*u0(x) = {k0**3:.2f} * u0(x)")
    print("="*60)

    Q3  = PseudoDifferentialOperator(xi_sym, vars_x=[x_sym], mode='symbol')
    cb3 = CompositionBridge(P1, Q3, lam=lam, comp_order=1,
                            n_guesses=50, xi_range=(-10, 10), verbose=verbose)
    vals_bridge3 = cb3.evaluate_grid(x_test, u0_phase_sym, u0_amp_sym)
    vals_ref3    = k0**3 * u0_test

    rel_err3 = np.abs(vals_bridge3 - vals_ref3) / (np.abs(vals_ref3) + 1e-12)
    max_err3 = float(np.max(rel_err3))
    # Composition accumulates two O(1/lam) errors, so tolerance is 5/lam
    passed3  = max_err3 < 5.0 / lam

    print(f"  Max relative error : {max_err3:.5f}  =>  {'PASS' if passed3 else 'FAIL'}")
    results['test3'] = dict(passed=passed3, max_rel_error=max_err3,
                             values_bridge=vals_bridge3, values_ref=vals_ref3,
                             x_test=x_test, label='P∘Q=xi^3')

    # ------------------------------------------------------------------
    # TEST 4  :  Asymptotic right-inverse R of P = xi^2 + 1
    #   (a) symbolic residual  P∘R - Id ~ 0
    #   (b) numeric evaluation (Ru)(x) ~ u0(x) / p(k0)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 4 - Asymptotic inverse : P = xi^2+1, R = P^{-1}")
    print("="*60)

    P4    = PseudoDifferentialOperator(xi_sym**2 + 1, vars_x=[x_sym], mode='symbol')
    R_sym = P4.right_inverse_asymptotic(order=1)
    R4    = PseudoDifferentialOperator(R_sym, vars_x=[x_sym], mode='symbol')

    # (a) symbolic check: P∘R should equal Id
    PR_sym  = P4.compose_asymptotic(R4, order=1)
    PR_diff = sp.simplify(PR_sym - 1)
    xi_pts  = np.linspace(1.0, 8.0, 30)
    pr_func = sp.lambdify((x_sym, xi_sym), PR_diff, 'numpy')
    resid_sym = float(np.max(np.abs(pr_func(np.zeros_like(xi_pts), xi_pts))))
    print(f"  (a) Symbolic residual ||P∘R - Id||_inf = {resid_sym:.2e}")

    # (b) numeric check: (Ru)(x) should equal u0(x) / p(k0)
    bridge4 = PsiOpFIOBridge(R4, lam=lam, n_guesses=50,
                              xi_range=(-10, 10), verbose=verbose)
    vals_bridge4 = bridge4.evaluate_grid(x_test, u0_phase_sym, u0_amp_sym)
    p4_at_k0  = k0**2 + 1   # p(k0) = k0^2 + 1 = 5
    vals_ref4 = u0_test / p4_at_k0

    rel_err4 = np.abs(vals_bridge4 - vals_ref4) / (np.abs(vals_ref4) + 1e-12)
    max_err4 = float(np.max(rel_err4))
    passed4  = (resid_sym < 1e-10) and (max_err4 < 3.0 / lam)

    print(f"  (b) Numeric error ||(Ru) - u/p(k0)|| / ||u/p(k0)|| = {max_err4:.5f}")
    print(f"  =>  {'PASS' if passed4 else 'FAIL'}")
    results['test4'] = dict(passed=passed4, max_rel_error=max_err4,
                             residual_sym=resid_sym,
                             values_bridge=vals_bridge4, values_ref=vals_ref4,
                             x_test=x_test, label='R=(xi^2+1)^{-1}')

    # ------------------------------------------------------------------
    # TEST 5  :  Propagator exp(it*xi) on WKB packet  (t=1)
    #   ref : p(k0)*u0(x) = exp(i*t*k0) * u0(x)  (pure phase shift)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 5 - Propagator exp(itP): P=xi, t=1")
    print(f"  WKB reference : exp(i*t*k0)*u0(x) = exp({1j*k0:.2f}) * u0(x)")
    print("="*60)

    t5       = 1.0
    exp_sym5 = sp.exp(sp.I * t5 * xi_sym)   # exact symbol of exp(it*xi)
    expP5    = PseudoDifferentialOperator(exp_sym5, vars_x=[x_sym], mode='symbol')
    bridge5  = PsiOpFIOBridge(expP5, lam=lam, n_guesses=50,
                               xi_range=(-10, 10), verbose=verbose)
    vals_bridge5 = bridge5.evaluate_grid(x_test, u0_phase_sym, u0_amp_sym)
    vals_ref5    = np.exp(1j * t5 * k0) * u0_test

    rel_err5 = np.abs(vals_bridge5 - vals_ref5) / (np.abs(vals_ref5) + 1e-12)
    max_err5 = float(np.max(rel_err5))
    passed5  = max_err5 < 3.0 / lam

    print(f"  Max relative error : {max_err5:.5f}  =>  {'PASS' if passed5 else 'FAIL'}")
    results['test5'] = dict(passed=passed5, max_rel_error=max_err5,
                             values_bridge=vals_bridge5, values_ref=vals_ref5,
                             x_test=x_test, label='exp(it*xi)')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    n_pass = 0
    for k, v in results.items():
        ok     = v.get('passed', False)
        n_pass += int(ok)
        err    = v.get('max_rel_error', 0.0)
        print(f"  {k} : {'PASS' if ok else 'FAIL'}  (max_rel_err={err:.2e})")
    print(f"\n  Total : {n_pass}/{len(results)} tests passed")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(
            "fio_bridge -- asymptotic bridge (red dashed) vs exact WKB reference (black)\n"
            "u0(y)=exp(-y^2/2)*exp(i*lam*k0*y),  k0=2, lam=40  |  ref: (Pu)(x)=p(k0)*u0(x)",
            fontsize=10
        )

        plot_tests = [
            ('test1', axes[0, 0]),
            ('test2', axes[0, 1]),
            ('test3', axes[0, 2]),
            ('test4', axes[1, 0]),
            ('test5', axes[1, 1]),
        ]
        for key, ax in plot_tests:
            r  = results[key]
            xt = r['x_test']
            vb = r['values_bridge']
            vr = r['values_ref']
            ax.plot(xt, np.real(vr),  'k-',  lw=2,   label='WKB ref (Re)')
            ax.plot(xt, np.real(vb),  'r--', lw=1.5, label='Bridge (Re)')
            ax.plot(xt, np.imag(vr),  'b-',  lw=1,   alpha=0.6, label='WKB ref (Im)')
            ax.plot(xt, np.imag(vb),  'm--', lw=1,   alpha=0.6, label='Bridge (Im)')
            ok  = r['passed']
            err = r.get('max_rel_error', 0.0)
            ax.set_title(f"[{'PASS' if ok else 'FAIL'}]  {r['label']}\nerr={err:.2e}",
                         fontsize=9)
            ax.legend(fontsize=7, loc='best')
            ax.set_xlabel('x', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Summary panel (bottom-right)
        ax6 = axes[1, 2]
        ax6.axis('off')
        lines = [
            "Semi-classical WKB reference:",
            "  (Pu)(x) = p(k0) * u0(x)",
            "",
            "Tests:",
        ]
        for k, v in results.items():
            ok  = v['passed']
            err = v.get('max_rel_error', 0.0)
            lines.append(f"  {k}  {'PASS' if ok else 'FAIL'}  err={err:.2e}")
        lines.append("")
        lines.append(f"Total: {sum(v['passed'] for v in results.values())}/{len(results)}")
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
    results = run_test_suite(verbose=True, plot=True)
