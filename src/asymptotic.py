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
asymptotic.py — Large-parameter asymptotics for oscillatory and Laplace-type integrals
====================================================================================

Overview
--------
This module provides symbolic-numerical tools for computing the asymptotic
behaviour of parameter-dependent integrals of the form

    I(λ) = ∫ a(x) exp(iλφ(x)) dx,   λ → +∞

as the large parameter λ grows without bound.  The nature of the phase
function φ determines which asymptotic method applies:

+------------------+-------------------------+-----------------------------------+
| φ                | Integral form           | Method                            |
+==================+=========================+===================================+
| real             | oscillatory             | Stationary phase                  |
+------------------+-------------------------+-----------------------------------+
| purely imaginary | exponentially damped    | Laplace                           |
+------------------+-------------------------+-----------------------------------+
| genuinely complex| oscillatory + damped    | Saddle-point (method of steepest  |
|                  |                         | descent)                          |
+------------------+-------------------------+-----------------------------------+

The correct method is selected **automatically** from the symbolic expression
of φ when the analyzer is initialised (``method=IntegralMethod.AUTO``, the
default).  It can also be set explicitly.


Mathematical background
-----------------------
All three methods share the same underlying idea: the dominant contribution
to I(λ) as λ → ∞ comes from a small neighbourhood of a *critical point*
x_c where ∇φ(x_c) = 0.  Away from x_c, rapid oscillations (or exponential
decay) make the integrand self-cancelling.

**Stationary phase** (φ real)
    The leading term at a non-degenerate critical point (det ∇²φ ≠ 0,
    *Morse* point) is

        I(λ) ≈ (2π/λ)^(n/2) · a(x_c) · exp(iλφ(x_c)) · exp(iπμ/4) / √|det ∇²φ(x_c)|

    where n is the dimension and μ = n − 2σ is the Maslov index (σ =
    number of negative eigenvalues of ∇²φ).  Degenerate critical points
    require special treatment: corank-1 singularities with a non-zero
    cubic term yield *Airy* integrals (decay O(λ^(−1/3)) in 1D,
    O(λ^(−5/6)) in 2D); those with a vanishing cubic but non-zero quartic
    term yield *Pearcey* integrals (decay O(λ^(−3/4))).

**Laplace's method** (φ = iψ, ψ real)
    The integrand concentrates exponentially around the minimum x_c of ψ.
    The leading term is identical to the stationary-phase formula with the
    oscillatory factor replaced by a real Gaussian:

        I(λ) ≈ (2π/λ)^(n/2) · a(x_c) · exp(−λψ(x_c)) / √|det ∇²ψ(x_c)|

    A second-order correction O(λ^(−n/2−1)) involving amplitude
    derivatives and phase anharmonicity (cubic/quartic tensors) is also
    computed.

**Saddle-point / steepest descent** (φ complex)
    The integration contour is deformed into ℂⁿ to pass through a saddle
    point z_c ∈ ℂⁿ satisfying ∇φ(z_c) = 0.  On the steepest-descent
    contour through z_c the phase Im(λφ) is stationary and Re(λφ) grows
    as fast as possible, making the integrand a complex Gaussian.  The
    asymptotic formula is formally identical to the Morse case:

        I(λ) ≈ (2π/λ)^(n/2) · a(z_c) · exp(iλφ(z_c)) / √det ∇²φ(z_c)

    where the square root is taken on the principal branch.
    **Limitation:** this implementation uses a naive continuation strategy
    (minimising |∇φ(z)|² over ℝ^(2n)) and does NOT verify that the
    original contour can be deformed through the found saddle (Picard-
    Lefschetz theory).  A RuntimeWarning is always emitted.

References
----------
.. [1] Hörmander, L.  *The Analysis of Linear Partial Differential
       Operators I*, Springer, 1983.  Chapter 7: Oscillatory Integrals.
.. [2] Olver, F. W. J.  *Asymptotics and Special Functions*, Academic Press,
       1974 (reprinted A K Peters, 1997).
.. [3] Wong, R.  *Asymptotic Approximations of Integrals*, Academic Press,
       1989.
.. [4] Bleistein, N. & Handelsman, R.  *Asymptotic Expansions of Integrals*,
       Holt, Rinehart & Winston, 1975.
.. [5] Berry, M. V. & Howls, C. J.  "High orders of the Weyl expansion for
       quantum billiards", *Physical Review E* 50(5), 3577–3595, 1994.
.. [6] Delabaere, E. & Howls, C. J.  "Global asymptotics for multiple
       integrals with boundaries", *Duke Mathematical Journal* 112(2),
       199–264, 2002.
"""

import numpy as np
import sympy as sp
from scipy.special import airy, gamma
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import warnings

# --- Types and Enums ---

class IntegralMethod(Enum):
    """
    Selects which asymptotic method to apply to the integral I(λ).

    The three concrete methods correspond to the three possible natures of
    the phase function φ(x) appearing in the exponential exp(iλφ(x)):

    - STATIONARY_PHASE: φ is purely real.
        I(λ) = ∫ a(x) exp(iλφ(x)) dx,  φ ∈ ℝ,  λ → +∞
      The integrand oscillates with unit modulus; contributions arise from
      stationary points ∇φ = 0.  Decay rate: O(λ^(-n/2)).
      Typical applications: wave optics, quantum mechanics, Fourier integrals.

    - LAPLACE: φ is purely imaginary, i.e. φ(x) = i·ψ(x) with ψ ∈ ℝ.
        I(λ) = ∫ a(x) exp(iλ·iψ(x)) dx = ∫ a(x) exp(-λψ(x)) dx,  λ → +∞
      The integrand is real and exponentially damped; contributions arise
      from minima of ψ where ∇ψ = 0 and ∇²ψ > 0.
      Typical applications: large deviations, Bayesian inference,
      statistical mechanics partition functions.

    - SADDLE_POINT: φ is genuinely complex, φ = φ_R + i·φ_I with both
      φ_R ≠ 0 and φ_I ≠ 0.
        I(λ) = ∫ a(x) exp(iλφ_R(x)) exp(-λφ_I(x)) dx,  λ → +∞
      The integrand both oscillates and is exponentially modulated.
      Contributions come from saddle points in ℂⁿ found by analytically
      continuing ∇φ(z) = 0 into the complex plane.  The integration contour
      must be deformed to pass through these saddle points along the
      steepest-descent direction.
      Note: this implementation uses a naive continuation strategy; see
      SaddlePointEvaluator for limitations.

    - AUTO: automatic detection (default).
      The analyzer inspects φ symbolically (via sympy.im / sympy.re) and
      falls back to a numerical test if the symbolic check is inconclusive.
      The detected method is stored back in Analyzer.method
      after __init__ so the user can always query which method was chosen.

    Hierarchy
    ---------
    SADDLE_POINT is the general case; the other two are special cases:
        SADDLE_POINT with φ_I ≡ 0  →  STATIONARY_PHASE
        SADDLE_POINT with φ_R ≡ 0  →  LAPLACE
    """
    STATIONARY_PHASE = "stationary_phase"
    LAPLACE          = "laplace"
    SADDLE_POINT     = "saddle_point"
    AUTO             = "auto"


class SingularityType(Enum):
    """
    Classification of critical points based on Hessian rank and higher-order derivatives.
    
    The type determines which asymptotic formula applies to the stationary phase integral:
    
    - MORSE: Non-degenerate critical point (det H ≠ 0)
      Contribution scales as O(λ^(-n/2)) where n is the dimension
      
    - AIRY_1D: Corank-1 singularity with non-zero cubic term (1D case)
      Contribution scales as O(λ^(-1/3))
      
    - AIRY_2D: Corank-1 singularity with non-zero cubic term (2D case)
      Contribution scales as O(λ^(-5/6)) = O(λ^(-1/3-1/2))
      
    - PEARCEY: Corank-1 singularity with vanishing cubic but non-zero quartic term
      Contribution scales as O(λ^(-3/4)) = O(λ^(-1/4-1/2))
      
    - HIGHER_ORDER: More degenerate cases requiring special treatment
      Not implemented in this code
    """
    MORSE = "morse"             # Non-degenerate (det H != 0)
    AIRY_1D = "airy_1d"         # Corank 1, cubic term != 0 (1D)
    AIRY_2D = "airy_2d"         # Corank 1, cubic term != 0 (2D)
    PEARCEY = "pearcey"         # Corank 1, cubic = 0, quartic != 0
    HIGHER_ORDER = "higher_order"

@dataclass
class CriticalPoint:
    """
    Stores all geometric and analytical properties of a critical point.
    
    A critical point x_c satisfies ∇φ(x_c) = 0. This class contains all the data
    needed to compute its asymptotic contribution to the stationary phase integral
    ∫ a(x) exp(iλφ(x)) dx as λ → ∞.
    
    Attributes:
        position (np.ndarray): Coordinates of the critical point x_c.
        phase_value (complex): Value of phase function φ(x_c).
        amplitude_value (complex): Value of amplitude function a(x_c).
        singularity_type (SingularityType): Classification determining which formula to use.
        hessian_matrix (np.ndarray): The Hessian matrix ∇²φ at x_c.
        hessian_inv (Optional[np.ndarray]): Inverse of the Hessian (for Morse points only).
        hessian_det (float): Determinant of the Hessian.
        signature (int): Number of negative eigenvalues of the Hessian (Morse index).
        eigenvalues (np.ndarray): Eigenvalues of the Hessian matrix.
        eigenvectors (np.ndarray): Eigenvectors of the Hessian matrix.
        grad_amp (Optional[np.ndarray]): Gradient of amplitude ∇a at x_c.
        hess_amp (Optional[np.ndarray]): Hessian of amplitude ∇²a at x_c.
        phase_d3 (Optional[np.ndarray]): Rank-3 tensor of 3rd derivatives of φ.
        phase_d4 (Optional[np.ndarray]): Rank-4 tensor of 4th derivatives of φ.
        canonical_coefficients (Optional[Dict]): Coefficients for normal forms 
            (Airy/Pearcey canonical representations). Contains keys like 'cubic', 
            'quartic', 'quadratic_transverse' depending on singularity type.
    """
    position: np.ndarray
    phase_value: complex
    amplitude_value: complex
    singularity_type: SingularityType
    hessian_matrix: np.ndarray
    hessian_inv: Optional[np.ndarray] = None
    hessian_det: float = 0.0
    signature: int = 0
    eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    eigenvectors: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Higher order derivatives (stored as numpy tensors)
    grad_amp: Optional[np.ndarray] = None      
    hess_amp: Optional[np.ndarray] = None      
    phase_d3: Optional[np.ndarray] = None      
    phase_d4: Optional[np.ndarray] = None      
    
    canonical_coefficients: Optional[Dict] = None
    # Integration method that produced this critical point
    method: 'IntegralMethod' = None  # set by the analyzer; forward ref resolved at runtime

@dataclass
class AsymptoticContribution:
    """
    Represents the calculated asymptotic contribution from a specific critical point.
    
    The total asymptotic expansion typically has the form:
        I(λ) ≈ leading_term + correction_term + O(λ^(-order_leading - 2))
    
    For Morse points, the correction term is of order λ^(-n/2-1) relative to λ^(-n/2).
    For degenerate singularities (Airy, Pearcey), typically only the leading term
    is computed, as correction terms require more sophisticated analysis.
    
    Attributes:
        leading_term (complex): The dominant term, O(λ^(-order_leading)).
            For Morse in 2D: O(λ^(-1))
            For Airy 1D: O(λ^(-1/3))
            For Airy 2D: O(λ^(-5/6))
            For Pearcey: O(λ^(-3/4))
        correction_term (complex): The next-order correction term.
            For Morse: O(λ^(-n/2-1))
            For degenerate cases: typically 0j (not computed)
        total_value (complex): Sum of leading_term + correction_term.
        point (CriticalPoint): The source critical point for this contribution.
        order_leading (float): The exponent p in the scaling λ^(-p) of the leading term.
        method (IntegralMethod): The asymptotic method used to compute this contribution
            (STATIONARY_PHASE, LAPLACE or SADDLE_POINT).
    """
    leading_term: complex
    correction_term: complex 
    total_value: complex
    point: CriticalPoint
    order_leading: float
    method: IntegralMethod = IntegralMethod.STATIONARY_PHASE  # default for backward compat

# --- Analyzer (Symbolic -> Numerical) ---

class Analyzer:
    """
    Handles symbolic analysis of phase and amplitude functions.

    Supports three integration paradigms selected via ``method``:

    - ``IntegralMethod.STATIONARY_PHASE``:
        Oscillatory integrals  I(λ) = ∫ a(x) exp(iλφ(x)) dx,  φ real.
        All singularity types (Morse, Airy, Pearcey) are supported.

    - ``IntegralMethod.LAPLACE``:
        Exponentially damped integrals  I(λ) = ∫ a(x) exp(-λφ(x)) dx,  φ real.
        Only non-degenerate minima of φ are relevant.

    - ``IntegralMethod.SADDLE_POINT``:
        Mixed integrals  I(λ) = ∫ a(x) exp(iλφ(x)) dx,  φ genuinely complex.
        Saddle points are searched in ℂⁿ by continuation from real guesses.

    - ``IntegralMethod.AUTO`` (default):
        The analyzer inspects φ symbolically and chooses automatically among
        the three concrete methods.  The resolved method is written back to
        ``self.method`` after __init__ completes.

    The main workflow is:
        1. Initialize with symbolic SymPy expressions (method=AUTO by default).
        2. Find critical / saddle points using find_critical_points().
        3. Analyze each point using analyze_point().
        4. Use AsymptoticEvaluator (unified façade) to compute contributions.

    Attributes:
        phase_expr: SymPy expression for the phase function φ(x).
        amplitude_expr: SymPy expression for the amplitude function a(x).
        variables: List of SymPy symbols representing the integration variables.
        dim (int): Dimension of the integration domain.
        domain (Optional[List[Tuple]]): Optional bounds [(min, max), ...] for each variable.
        tolerance (float): Numerical tolerance for detecting zeros and critical points.
        cubic_threshold (float): Absolute threshold for distinguishing Airy vs Pearcey
            singularities. Default: max(1e-5, 10*tolerance). Unused for LAPLACE.
        method (IntegralMethod): Resolved integration method (never AUTO after __init__).
    """

    def __init__(self, phase_expr, amplitude_expr, variables, domain=None,
                 tolerance=1e-6, cubic_threshold=None,
                 method: IntegralMethod = IntegralMethod.AUTO):
        """
        Initialize the analyzer.

        Args:
            phase_expr: SymPy expression for phase φ(x).
                The nature of φ (real / imaginary / complex) determines which
                asymptotic method is appropriate; use method=AUTO to detect it
                automatically.
            amplitude_expr: SymPy expression for amplitude a(x). Can be complex.
            variables: List of SymPy symbols [x, y, ...] or a single symbol for 1D.
            domain: Optional list of tuples [(min, max), ...] specifying search bounds
                for each variable when finding critical points.
            tolerance: Numerical tolerance for zero-detection and optimization
                (default: 1e-6).
            cubic_threshold: Absolute threshold for classifying cubic terms
                (STATIONARY_PHASE only). If None, defaults to max(1e-5, 10*tolerance).
            method: One of IntegralMethod.{AUTO, STATIONARY_PHASE, LAPLACE,
                SADDLE_POINT}.  AUTO (default) inspects φ symbolically and
                resolves to one of the three concrete values.
        """
        self.phase_expr = phase_expr
        self.amplitude_expr = amplitude_expr
        self.variables = list(variables) if isinstance(variables, (list, tuple)) else [variables]
        self.dim = len(self.variables)
        self.domain = domain
        self.tolerance = tolerance
        self.cubic_threshold = cubic_threshold if cubic_threshold is not None else max(1e-5, 10 * tolerance)

        # Resolve AUTO before preparing derivatives so that _detect_method
        # can inspect the raw expression before any lambdification.
        if method == IntegralMethod.AUTO:
            self.method = self._detect_method(phase_expr)
        else:
            self.method = method

        self._prepare_derivatives()
        self._create_numerical_functions()

    # ------------------------------------------------------------------
    # Method auto-detection
    # ------------------------------------------------------------------

    def _detect_method(self, phase_expr) -> IntegralMethod:
        """
        Inspect the phase expression symbolically to select the integration method.

        Strategy
        --------
        1. Symbolic test (fast, exact when SymPy can simplify):
           - Compute re_part = sp.re(φ)  and  im_part = sp.im(φ)  after
             assuming all variables are real (sp.refine with Q.real).
           - If re_part simplifies to zero   → LAPLACE
           - If im_part simplifies to zero   → STATIONARY_PHASE
           - Otherwise                       → SADDLE_POINT

        2. Numerical fallback (if symbolic test is inconclusive, i.e. SymPy
           cannot decide):
           - Sample n_samples random real points in [-2, 2]^dim.
           - Evaluate φ numerically at each sample.
           - If max|Re φ| < tol * max|Im φ|   → LAPLACE
           - If max|Im φ| < tol * max|Re φ|   → STATIONARY_PHASE
           - Otherwise                         → SADDLE_POINT

        The numerical threshold uses tol = 1e-6 relative to the dominant part.

        Returns
        -------
        IntegralMethod
            One of STATIONARY_PHASE, LAPLACE, or SADDLE_POINT (never AUTO).
        """
        # --- Step 1: symbolic test ---
        real_assumptions = {v: True for v in self.variables}
        # Replace variables with real-stamped symbols for sp.re / sp.im
        real_vars = [sp.Symbol(str(v), real=True) for v in self.variables]
        expr_real = phase_expr.subs(dict(zip(self.variables, real_vars)))

        try:
            re_part = sp.simplify(sp.re(expr_real))
            im_part = sp.simplify(sp.im(expr_real))

            re_is_zero = (re_part == sp.S.Zero)
            im_is_zero = (im_part == sp.S.Zero)

            if re_is_zero and not im_is_zero:
                return IntegralMethod.LAPLACE
            if im_is_zero and not re_is_zero:
                return IntegralMethod.STATIONARY_PHASE
            if re_is_zero and im_is_zero:
                # Constant zero phase — stationary phase is the safest default
                return IntegralMethod.STATIONARY_PHASE
            # Both parts are symbolically non-zero → fall through to numeric
        except Exception:
            pass  # SymPy failed to simplify; proceed to numerical fallback

        # --- Step 2: numerical fallback ---
        n_samples = 12
        rng = np.random.default_rng(seed=0)  # deterministic seed for reproducibility
        samples = rng.uniform(-2.0, 2.0, size=(n_samples, self.dim))

        func_phase_num = sp.lambdify(tuple(self.variables), phase_expr, 'numpy')
        re_magnitudes = []
        im_magnitudes = []
        for pt in samples:
            try:
                val = complex(func_phase_num(*pt))
                re_magnitudes.append(abs(val.real))
                im_magnitudes.append(abs(val.imag))
            except Exception:
                pass

        if not re_magnitudes:
            # Could not evaluate at all — default to stationary phase
            warnings.warn(
                "AUTO method detection: could not evaluate φ numerically. "
                "Defaulting to STATIONARY_PHASE.",
                RuntimeWarning,
            )
            return IntegralMethod.STATIONARY_PHASE

        max_re = max(re_magnitudes)
        max_im = max(im_magnitudes)
        scale = max(max_re, max_im, 1e-30)
        rel_tol = 1e-6

        if max_re < rel_tol * scale:
            detected = IntegralMethod.LAPLACE
        elif max_im < rel_tol * scale:
            detected = IntegralMethod.STATIONARY_PHASE
        else:
            detected = IntegralMethod.SADDLE_POINT

        return detected
    
    def _prepare_derivatives(self):
        """
        Symbolically compute all necessary derivatives of phase and amplitude functions.
        
        Computes and stores as SymPy expressions:
        1. Gradient ∇φ and Hessian ∇²φ of phase
        2. Gradient ∇a and Hessian ∇²a of amplitude
        3. Third-order tensor D3[i,j,k] = ∂³φ/∂xi∂xj∂xk (for Airy classification 
           and second-order Morse correction)
        4. Fourth-order tensor D4[i,j,k,l] = ∂⁴φ/∂xi∂xj∂xk∂xl (for Pearcey 
           classification and second-order Morse correction)
        
        These symbolic derivatives are later converted to numerical functions via lambdify.
        """
        # 1. Phase Gradient & Hessian
        self.grad_sym = [sp.diff(self.phase_expr, v) for v in self.variables]
        self.hess_sym = [[sp.diff(self.phase_expr, v1, v2) for v2 in self.variables] 
                         for v1 in self.variables]
        
        # 2. Amplitude Gradient & Hessian
        self.grad_amp_sym = [sp.diff(self.amplitude_expr, v) for v in self.variables]
        self.hess_amp_sym = [[sp.diff(self.amplitude_expr, v1, v2) for v2 in self.variables] 
                             for v1 in self.variables]
        
        # 3. Higher order tensors for Phase
        # D3 Tensor (Rank 3)
        self.d3_indices = []
        self.d3_sym = []
        import itertools
        for idx in itertools.product(range(self.dim), repeat=3):
            self.d3_indices.append(idx)
            var_seq = [self.variables[i] for i in idx]
            self.d3_sym.append(sp.diff(self.phase_expr, *var_seq))
            
        # D4 Tensor (Rank 4) - Required for 2nd order Morse correction
        self.d4_indices = []
        self.d4_sym = []
        for idx in itertools.product(range(self.dim), repeat=4):
            self.d4_indices.append(idx)
            var_seq = [self.variables[i] for i in idx]
            self.d4_sym.append(sp.diff(self.phase_expr, *var_seq))

    def _create_numerical_functions(self):
        """
        Convert symbolic expressions to fast numerical functions using SymPy's lambdify.
        
        Creates NumPy-compatible functions for efficient numerical evaluation:
        - func_phase, func_amp: Evaluate φ(x) and a(x)
        - func_grad, func_hess: Evaluate ∇φ and ∇²φ
        - func_grad_amp, func_hess_amp: Evaluate ∇a and ∇²a
        - func_d3, func_d4: Evaluate third and fourth order derivatives of φ
        
        These lambdified functions are much faster than evaluating SymPy expressions directly.
        """
        vars_tuple = tuple(self.variables)
        self.func_phase = sp.lambdify(vars_tuple, self.phase_expr, 'numpy')
        self.func_amp = sp.lambdify(vars_tuple, self.amplitude_expr, 'numpy')
        self.func_grad = sp.lambdify(vars_tuple, self.grad_sym, 'numpy')
        self.func_hess = sp.lambdify(vars_tuple, self.hess_sym, 'numpy')
        
        self.func_grad_amp = sp.lambdify(vars_tuple, self.grad_amp_sym, 'numpy')
        self.func_hess_amp = sp.lambdify(vars_tuple, self.hess_amp_sym, 'numpy')
        
        self.func_d3 = sp.lambdify(vars_tuple, self.d3_sym, 'numpy')
        self.func_d4 = sp.lambdify(vars_tuple, self.d4_sym, 'numpy')

    def find_critical_points(self, initial_guesses=None) -> List[np.ndarray]:
        """
        Locate critical points where ∇φ(x) = 0.

        Uses numerical minimization of |∇φ|² starting from provided initial guesses.
        Multiple guesses can help find multiple critical points if they exist.
        
        The method deduplicates found points (within tolerance 1e-4) and optionally
        filters by domain bounds if specified.
        
        Args:
            initial_guesses: List of starting coordinate arrays for optimization.
                If None, uses [0, ...] and domain center (if domain is specified).
                Provide multiple guesses to search for multiple critical points.
            
        Returns:
            List of unique critical point coordinates (as numpy arrays) found within 
            the specified tolerance. Empty list if no critical points are found.
        """
        points = []
        if initial_guesses is None:
            initial_guesses = [np.zeros(self.dim)]
            if self.domain:
                # Add center of domain as a guess
                centers = [0.5*(d[0]+d[1]) for d in self.domain]
                initial_guesses.append(centers)

        def objective(x):
            g = np.array(self.func_grad(*x))
            return np.sum(g**2)

        for guess in initial_guesses:
            try:
                res = minimize(objective, guess, tol=self.tolerance)
                if res.success and res.fun < self.tolerance:
                    xc = res.x
                    # Check for duplicates
                    if not any(np.linalg.norm(xc - p) < 1e-4 for p in points):
                        # Check domain bounds
                        if self.domain:
                            if all(d[0] <= xi <= d[1] for xi, d in zip(xc, self.domain)):
                                points.append(xc)
                        else:
                            points.append(xc)
            except Exception: 
                pass
        return points

    def analyze_point(self, xc) -> CriticalPoint:
        """
        Perform complete analysis of a critical point (real or complex).

        For STATIONARY_PHASE and LAPLACE the coordinates xc are real (numpy
        float array).  For SADDLE_POINT, xc may be complex (numpy complex
        array produced by SaddlePointEvaluator.find_saddle_points).

        Computes all geometric and analytical properties needed for asymptotic
        evaluation:
        - Phase value φ(x_c) and amplitude value a(x_c)
        - Hessian matrix ∇²φ and its properties (determinant, eigenvalues,
          signature)
        - Higher-order derivatives: D3 and D4 tensors of φ, gradients and
          Hessians of a
        - Classification of singularity type (Morse, Airy, Pearcey, etc.)
        - Canonical coefficients for degenerate cases
        - Hessian inverse (for Morse points only)

        Args:
            xc: Coordinates of the critical point.  Real numpy array for
                STATIONARY_PHASE / LAPLACE; complex numpy array for
                SADDLE_POINT.

        Returns:
            CriticalPoint object containing all computed properties.
        """
        args = tuple(xc)

        # Evaluate Hessian — complex-safe: cast to complex array so that
        # complex saddle-point coordinates do not silently drop imaginary parts.
        H = np.array(self.func_hess(*args), dtype=complex)

        # Eigendecomposition — use np.linalg.eig for complex matrices
        # (eigh requires Hermitian; a complex Hessian at a saddle point is
        # symmetric but not necessarily Hermitian).
        if np.iscomplexobj(H) and np.any(np.imag(H) != 0):
            vals, vecs = np.linalg.eig(H)
        else:
            H = np.real(H)
            vals, vecs = np.linalg.eigh(H)

        # Reconstruct higher order tensors from flattened symbolic output
        d3_flat = self.func_d3(*args)
        D3 = np.zeros((self.dim,)*3, dtype=complex)
        for k, idx in enumerate(self.d3_indices):
            D3[idx] = d3_flat[k]

        d4_flat = self.func_d4(*args)
        D4 = np.zeros((self.dim,)*4, dtype=complex)
        for k, idx in enumerate(self.d4_indices):
            D4[idx] = d4_flat[k]

        grad_a = np.array(self.func_grad_amp(*args), dtype=complex)
        hess_a = np.array(self.func_hess_amp(*args), dtype=complex)

        det = np.prod(vals)
        # Rank: number of eigenvalues with non-negligible magnitude
        rank = int(np.sum(np.abs(vals) > self.tolerance))
        # Signature: number of eigenvalues with strictly negative real part
        signature = int(np.sum(np.real(vals) < -self.tolerance))

        cp = CriticalPoint(
            position=np.asarray(xc),
            phase_value=complex(self.func_phase(*args)),
            amplitude_value=complex(self.func_amp(*args)),
            singularity_type=SingularityType.MORSE,  # Default, may be overridden below
            hessian_matrix=H,
            hessian_det=complex(det),
            signature=signature,
            eigenvalues=vals,
            eigenvectors=vecs,
            grad_amp=grad_a,
            hess_amp=hess_a,
            phase_d3=D3,
            phase_d4=D4,
            method=self.method,
        )

        if rank == self.dim:
            cp.singularity_type = SingularityType.MORSE
            cp.hessian_inv = np.linalg.inv(H)
            # For Laplace's method the Hessian must be positive definite (minimum of φ).
            if self.method == IntegralMethod.LAPLACE and np.any(np.real(vals) <= self.tolerance):
                warnings.warn(
                    f"Laplace method: critical point at {xc} has a non-positive Hessian "
                    "eigenvalue (it may be a saddle point or maximum). "
                    "The Laplace approximation requires a strict minimum of φ.",
                    RuntimeWarning,
                )
        elif self.dim == 1 and rank == 0:  # 1D degenerate
            coeffs = self._project_degenerate_coeffs(cp)
            cp.canonical_coefficients = coeffs
            if abs(coeffs['cubic']) > self.cubic_threshold:
                cp.singularity_type = SingularityType.AIRY_1D
            elif abs(coeffs['quartic']) > self.tolerance:
                cp.singularity_type = SingularityType.PEARCEY
            else:
                cp.singularity_type = SingularityType.HIGHER_ORDER
        elif self.dim == 2 and rank == 1:  # 2D corank 1
            coeffs = self._project_degenerate_coeffs(cp)
            cp.canonical_coefficients = coeffs
            if abs(coeffs['cubic']) > self.cubic_threshold:
                cp.singularity_type = SingularityType.AIRY_2D
            elif abs(coeffs['quartic']) > self.tolerance:
                cp.singularity_type = SingularityType.PEARCEY
            else:
                cp.singularity_type = SingularityType.HIGHER_ORDER
        else:
            cp.singularity_type = SingularityType.HIGHER_ORDER

        return cp
        
    def _project_degenerate_coeffs(self, cp: CriticalPoint) -> Dict[str, float]:
        """
        Projects derivatives onto the eigenvectors to find canonical form coefficients.
        
        This handles cases where the singularity is not aligned with the axes 
        (e.g., phi = (x+y)^3). It identifies the 'null' direction and computes
        directional derivatives along it.
        
        Returns:
            Dictionary with 'cubic', 'quartic' and 'quadratic_transverse' coefficients.
        """
        null_idx = np.argmin(np.abs(cp.eigenvalues))
        v_null = cp.eigenvectors[:, null_idx]
        
        # AIRY CORRECTION: alpha = D^3/2 for phi ~ alpha * u^3/3 (since d^3(alpha * u^3/3)/du^3 = 2 * alpha)
        alpha = np.einsum('ijk,i,j,k->', cp.phase_d3, v_null, v_null, v_null) / 2.0
        
        # PEARCEY CORRECTION: gamma = D^4/24 for phi ~ gamma * u^4/4 
        # (since d^4(gamma * u^4/4)/du^4 = 6 * gamma -> gamma = D^4/6,
        # but the canonical normal form uses u^4/4 -> coefficient = D^4/24)
        # gamma_coeff = np.einsum('ijkl,i,j,k,l->', cp.phase_d4, v_null, v_null, v_null, v_null) / 24.0
        gamma_coeff = np.einsum('ijkl,i,j,k,l->', cp.phase_d4, v_null, v_null, v_null, v_null) / 6.0
        
        # Transverse quadratic term (2D only)
        quadratic_transverse = None
        if self.dim > 1:
            non_null_idxs = np.where(np.abs(cp.eigenvalues) > self.tolerance)[0]
            if len(non_null_idxs) > 0:
                # The eigenvalue IS the coefficient beta in phi ~ beta*v^2/2
                # (since d²phi/dv² = beta for this form)
                quadratic_transverse = cp.eigenvalues[non_null_idxs[0]]
        
        return {
            'cubic': alpha,               # alpha for phi = alpha * u^3/3
            'quartic': gamma_coeff,       # gamma for phi = gamma * u^4/4
            'quadratic_transverse': quadratic_transverse
        }

# --- Evaluator (Asymptotic Calculation) ---

class StationaryPhaseEvaluator:
    """
    Computes asymptotic contributions from critical points for large parameter λ.
    
    This class implements the standard stationary phase formulas for different types
    of critical points:
    - Morse points: Standard stationary phase with second-order corrections
    - Airy singularities (1D and 2D): Catastrophe integrals with exact formulas
    - Pearcey singularities: Quartic catastrophe integrals
    
    The evaluation includes both leading-order terms and next-order corrections where
    applicable (primarily for Morse points). Each method returns an AsymptoticContribution
    object containing the computed terms.
    
    Reference: 
        - Wong, "Asymptotic Approximations of Integrals" (1989)
        - Olver, "Asymptotics and Special Functions" (1997)
    
    Attributes:
        tolerance (float): Numerical tolerance for detecting near-zero coefficients.
    """
    def __init__(self, tolerance=1e-8):
        self.tolerance = tolerance  # ← Addition required
            
    def evaluate(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Dispatch evaluation to the appropriate method based on singularity type.
        
        Args:
            cp: CriticalPoint object with all necessary geometric data.
            lam: Large parameter λ in the oscillatory integral I(λ).
            
        Returns:
            AsymptoticContribution containing leading term, correction (if computed),
            and total value. For HIGHER_ORDER or unknown types, returns zero contribution
            with a warning.
        """
        if cp.singularity_type == SingularityType.MORSE:
            result = self._eval_morse_order2(cp, lam)
        elif cp.singularity_type == SingularityType.AIRY_1D:
            result = self._eval_airy_1d(cp, lam)
        elif cp.singularity_type == SingularityType.AIRY_2D:
            result = self._eval_airy_2d(cp, lam)
        elif cp.singularity_type == SingularityType.PEARCEY:
            result = self._eval_pearcey(cp, lam)
        else:  # HIGHER_ORDER or unknown type
            warnings.warn(
                f"Unhandled singularity type {cp.singularity_type.value} at {cp.position}. "
                f"Returning zero contribution (no asymptotic formula available).",
                RuntimeWarning
            )
            result = AsymptoticContribution(
                leading_term=0j,
                correction_term=0j,
                total_value=0j,
                point=cp,
                order_leading=float('inf'),  # Indicates negligible contribution
                method=IntegralMethod.STATIONARY_PHASE,
            )
        result.method = IntegralMethod.STATIONARY_PHASE
        return result

    def _eval_morse_order2(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
            """
            Evaluate the asymptotic contribution for a non-degenerate (Morse) critical point.
            
            This method implements the standard stationary phase formula with second-order 
            correction terms for oscillatory integrals of the form:
            
                I(λ) = ∫ a(x) exp(iλφ(x)) dx
            
            as λ → ∞. The asymptotic expansion is:
            
                I(λ) ≈ I₀(λ) + I₁(λ) + O(λ^(-n/2-2))
            
            where I₀ is the leading term (order λ^(-n/2)) and I₁ is the first correction 
            (order λ^(-n/2-1)).
            
            Leading Term (Order 0)
            ----------------------
            The dominant contribution from the critical point x_c where ∇φ(x_c) = 0:
            
                I₀(λ) = (2π)^(n/2) / (λ^(n/2) √|det H|) × exp(iλφ(x_c)) × a(x_c) × exp(iπμ/4)
            
            Components:
                - (2π/λ)^(n/2): Gaussian prefactor from the quadratic approximation
                - √|det H|: Determinant of the Hessian matrix H = ∇²φ(x_c)
                - exp(iλφ(x_c)): Rapid oscillation at the critical point
                - a(x_c): Amplitude function evaluated at the critical point
                - exp(iπμ/4): Maslov phase correction, where μ = n - 2σ is the Morse index
                  (n = dimension, σ = signature = number of negative eigenvalues of H)
            
            Correction Term (Order 1)
            -------------------------
            The next-order contribution accounts for:
            1. Non-constant amplitude (amplitude derivatives)
            2. Cubic phase anharmonicity (third derivatives of φ)
            3. Quartic phase anharmonicity (fourth derivatives of φ)
            
                I₁(λ) = I₀(λ) / (iλ) × C
            
            where the correction factor C is:
            
                C = (1/2) Tr(H⁻¹ ∇²a) - (1/2) ⟨H⁻¹∇a, V⟩ + (a(x_c)/24) (5S₃ - 3S₄)
            
            Term breakdown:
            
            1. Amplitude Laplacian term: (1/2) Tr(H⁻¹ ∇²a)
               - Captures the effect of amplitude curvature at the critical point
               - H⁻¹ "twists" the Laplacian by the phase geometry
            
            2. Mixed amplitude-phase term: -(1/2) ⟨H⁻¹∇a, V⟩
               - Couples amplitude gradient with cubic phase nonlinearity
               - V_k = Σᵢⱼ (H⁻¹)ᵢⱼ ∂³φ/∂xᵢ∂xⱼ∂xₖ
            
            3. Pure phase anharmonicity: (a(x_c)/24) (5S₃ - 3S₄)
               - S₄: Quartic term = Σᵢⱼₖₗ (H⁻¹)ᵢⱼ (H⁻¹)ₖₗ ∂⁴φ/∂xᵢ∂xⱼ∂xₖ∂xₗ
               - S₃: Cubic term = Σᵢⱼₖₗₘₙ (H⁻¹)ᵢⱼ (H⁻¹)ₖₗ (H⁻¹)ₘₙ D³φᵢₖₘ D³φⱼₗₙ
               - The coefficients 5 and 3 come from Feynman diagram combinatorics
            
            Mathematical Background
            -----------------------
            The correction terms arise from expanding the integrand to higher orders in 
            (x - x_c) around the critical point and performing Gaussian integrals. The 
            coefficients are determined by the topology of Feynman diagrams:
            - S₃ corresponds to "theta graph" diagrams (three-loop)
            - S₄ corresponds to "sunset" diagrams (two-loop with quartic vertex)
            
            The factors of 1/2 in the amplitude terms come from the expansion of the 
            Gaussian measure, while the 5 and 3 in the phase term arise from diagram 
            symmetry factors.
            
            Parameters
            ----------
            cp : CriticalPoint
                Critical point with non-zero Hessian determinant (det H ≠ 0).
                Must contain: position, phase_value, amplitude_value, hessian_matrix,
                hessian_inv, signature, grad_amp, hess_amp, phase_d3, phase_d4.
            lam : float
                Large frequency parameter λ. The asymptotic approximation improves 
                as λ → ∞. Typically valid for λ ≳ 10.
                
            Returns
            -------
            AsymptoticContribution
                Object containing:
                - leading_term: I₀(λ), the dominant O(λ^(-n/2)) contribution
                - correction_term: I₁(λ), the O(λ^(-n/2-1)) correction
                - total_value: I₀(λ) + I₁(λ)
                - point: Reference to the input critical point
                - order_leading: n/2 (the decay exponent)
            
            Notes
            -----
            The correction term becomes negligible for large λ. The ratio
            |I₁/I₀| ~ O(λ⁻¹) should decrease linearly on a log-log plot, which
            can be verified using the convergence diagnostic tools.
            
            For dimension n=2, the leading term scales as O(λ⁻¹) and the correction 
            as O(λ⁻²), providing rapid asymptotic convergence.
            
            References
            ----------
            .. [1] Hörmander, L. "The Analysis of Linear Partial Differential Operators I" 
                   (1983), Chapter 7: Oscillatory Integrals
            .. [2] Berry, M.V. & Howls, C.J. "High orders of the Weyl expansion for quantum 
                   billiards" Physical Review E 50.5 (1994): 3577-3595
            .. [3] Wong, R. "Asymptotic Approximations of Integrals" (1989), Chapter 2
            
            Examples
            --------
            >>> # For a Gaussian phase φ = x²/2 + y²/2 with constant amplitude a = 1
            >>> # at the critical point (0, 0), the leading term is:
            >>> # I₀(λ) = 2π/λ (exact for Gaussian)
            >>> evaluator = StationaryPhaseEvaluator()
            >>> contribution = evaluator._eval_morse_order2(cp, lam=100)
            >>> print(f"Leading: {contribution.leading_term:.4e}")
            >>> print(f"Correction: {contribution.correction_term:.4e}")
            >>> print(f"Ratio: {abs(contribution.correction_term/contribution.leading_term):.2%}")
            """
            # dim = cp.position.shape[0]
            dim = len(cp.position)
            
            # ============================================================================
            # LEADING TERM (Order λ^(-n/2))
            # ============================================================================
            # Compute the dominant Gaussian contribution from the quadratic approximation
            # of the phase near the critical point.
            
            # Gaussian prefactor: (2π/λ)^(n/2)
            # This comes from the n-dimensional Gaussian integral formula
            prefactor = (2 * np.pi / lam) ** (dim / 2.0)
            
            # Maslov phase: exp(iπμ/4) where μ = n - 2σ (Morse index)
            # Accounts for the topology of the phase function at the critical point
            # σ = signature = number of negative eigenvalues of the Hessian
            maslov = np.exp(1j * np.pi / 4 * (dim - 2 * cp.signature))
            
            # Rapid oscillatory factor: exp(iλφ(x_c))
            # This is the phase evaluated at the critical point
            phase_osc = np.exp(1j * lam * cp.phase_value)
            
            # Geometric factor: 1/√|det H|
            # The Hessian determinant measures the "curvature volume" at the critical point
            denom = np.sqrt(np.abs(cp.hessian_det))
            
            # Amplitude at critical point: a(x_c)
            leading_amp = cp.amplitude_value
            
            # Combine all factors for the leading term
            term_0 = (prefactor / denom) * phase_osc * maslov * leading_amp
            
            # ============================================================================
            # CORRECTION TERM (Order λ^(-n/2-1))
            # ============================================================================
            # Compute next-order corrections from amplitude derivatives and phase 
            # anharmonicity (cubic and quartic terms in the Taylor expansion).
            
            # Inverse Hessian matrix: H⁻¹ = (∇²φ)⁻¹
            # Used to "propagate" corrections through the phase geometry
            H_inv = cp.hessian_inv
            
            # ------------------------------------------------------------------------
            # Term 1: Amplitude Laplacian Contribution
            # ------------------------------------------------------------------------
            # Measures how the amplitude curvature affects the integral
            # Formula: (1/2) Tr(H⁻¹ ∇²a)
            # 
            # Physical interpretation: If the amplitude has negative curvature along
            # directions where the phase is flat (small eigenvalues of H), this term
            # can become significant.
            term_amp = 0.5 * np.einsum('ij,ij->', H_inv, cp.hess_amp)
            
            # ------------------------------------------------------------------------
            # Term 2: Mixed Amplitude-Phase Contribution
            # ------------------------------------------------------------------------
            # Couples the amplitude gradient with cubic phase terms
            # 
            # Step 1: Contract H⁻¹ with D³φ to get effective vector V_k
            # V_k = Σᵢⱼ (H⁻¹)ᵢⱼ ∂³φ/∂xᵢ∂xⱼ∂xₖ
            # This vector represents the "cubic force" felt by the amplitude gradient
            V = np.einsum('ij,ijk->k', H_inv, cp.phase_d3)
            
            # Step 2: Inner product of (H⁻¹∇a) with V, scaled by -1/2
            # Formula: -(1/2) ⟨H⁻¹∇a, V⟩
            # 
            # Physical interpretation: If amplitude increases along directions where 
            # phase has strong cubic nonlinearity, this coupling enhances the contribution
            term_mix = -0.5 * np.dot(np.dot(H_inv, cp.grad_amp), V)
            
            # ------------------------------------------------------------------------
            # Term 3: Pure Phase Anharmonicity
            # ------------------------------------------------------------------------
            # Captures corrections from non-quadratic phase terms (cubic and quartic)
            # These arise from expanding exp(iλφ(x)) beyond the Gaussian approximation
            
            # S₄: Quartic contraction
            # Contract the fourth derivative tensor D⁴φ with two copies of H⁻¹
            # Formula: Σᵢⱼₖₗ (H⁻¹)ᵢⱼ (H⁻¹)ₖₗ ∂⁴φ/∂xᵢ∂xⱼ∂xₖ∂xₗ
            # 
            # Corresponds to Feynman diagrams with a single quartic vertex
            S4 = np.einsum('ij,kl,ijkl->', H_inv, H_inv, cp.phase_d4)
            
            # S₃: Cubic contraction (Theta graph)
            # Contract two copies of D³φ with three copies of H⁻¹
            # Formula: Σᵢⱼₖₗₘₙ (H⁻¹)ᵢⱼ (H⁻¹)ₖₗ (H⁻¹)ₘₙ D³φᵢₖₘ D³φⱼₗₙ
            # 
            # Corresponds to Feynman diagrams with two cubic vertices connected in a loop
            # (the "theta graph" topology)
            S3 = np.einsum('ij,kl,mn,ikm,jln->', H_inv, H_inv, H_inv, cp.phase_d3, cp.phase_d3)
            
            # Combine cubic and quartic contributions with diagram symmetry factors
            # Formula: (a₀/24) (5S₃ - 3S₄)
            # 
            # The coefficients 5 and 3 arise from:
            # - Combinatorial factors in the Taylor expansion
            # - Symmetry factors of Feynman diagrams (vertex permutations)
            # - Wick's theorem for Gaussian integrals
            term_phase = (cp.amplitude_value / 24.0) * (5.0 * S3 - 3.0 * S4)
            
            # Sum all three correction contributions
            correction_factor = term_amp + term_mix + term_phase
            
            # Scale correction by 1/(iλ) relative to leading term
            # The factor of i comes from ∫ x² exp(iλφ) dx ∝ -i/λ ∫ ∂²/∂λ² exp(iλφ) dx
            val_correction = (prefactor / denom) * phase_osc * maslov * (correction_factor / (1j * lam))
            
            # ============================================================================
            # RETURN ASYMPTOTIC CONTRIBUTION
            # ============================================================================
            return AsymptoticContribution(
                leading_term=term_0,
                correction_term=val_correction,
                total_value=term_0 + val_correction,
                point=cp,
                order_leading=dim/2.0
            )
    

    def _eval_airy_1d(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Evaluate the 1D Airy catastrophe integral contribution.
        
        For a canonical Airy integral of the form:
            ∫ exp(iλ α x³/3) dx
        
        The exact asymptotic formula is:
            I(λ) = 2π Ai(0) × (3λ|α|)^(-1/3) × exp(iπ/6 × sign(α))
        
        where:
            - Ai(0) ≈ 0.355028... is the Airy function at zero
            - α is the cubic coefficient in the canonical form φ ~ α x³/3
            - The scaling is O(λ^(-1/3)), which is slower decay than Morse O(λ^(-1/2))
        
        The Maslov phase exp(iπ/6 × sign(α)) accounts for the orientation of the
        integration contour in the complex plane:
            ∫ exp(i t³/3) dt = 2π Ai(0)
            ∫ exp(-i t³/3) dt = 2π Ai(0) × exp(-iπ/3)
        
        Reference: 
            - Olver, "Asymptotics and Special Functions" (1997), §7.3
            - Hörmander, "The Analysis of Linear Partial Differential Operators I" (1983)
        
        Args:
            cp: CriticalPoint classified as AIRY_1D with canonical coefficients.
            lam: Large parameter λ.
            
        Returns:
            AsymptoticContribution with exact Airy scaling O(λ^(-1/3)).
            No correction term is computed (set to 0j).
        """
        coeffs = cp.canonical_coefficients
        alpha = coeffs['cubic']  # Coefficient of x³/3 in the normal form
        
        if abs(alpha) < self.tolerance:
            warnings.warn("Cubic coefficient near zero in Airy evaluation")
            return AsymptoticContribution(0j, 0j, 0j, cp, 1/3)
        
        # Exact value of Ai(0)
        Ai0 = airy(0)[0]  # ≈ 0.3550280538878172
        
        # Scale factor: (λ|α|)^(-1/3)
        # Derivation: sub t = (λ|α|)^(1/3)·x → (λ|α|)^(-1/3) ∫ exp(it³/3) dt
        #             = (λ|α|)^(-1/3) · 2πAi(0)   [Olver, "Asymptotics & Special Functions", §7.3]
        # NOTE: (3λ|α|)^(-1/3) is WRONG — there is no factor of 3 here.
        scale = (lam * abs(alpha)) ** (-1/3)
        
        # No Maslov phase: ∫ exp(iλαx³/3) dx over ℝ is purely real for any sign of α,
        # because cos(λαx³/3) is even (contributes) and sin(λαx³/3) is odd (cancels).
        
        # Total contribution
        val = 2 * np.pi * Ai0 * scale * cp.amplitude_value
        
        return AsymptoticContribution(
            leading_term=val,
            correction_term=0j,
            total_value=val,
            point=cp,
            order_leading=1/3  # λ^(-1/3)
        )
        
    def _eval_airy_2d(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Evaluate the 2D Airy catastrophe integral contribution (corank 1).
        
        For a canonical 2D Airy integral of the form:
            ∫∫ exp(iλ [α u³/3 + β v²/2]) du dv
        
        The asymptotic formula is obtained by combining a 1D Airy integral
        with a transverse 1D Gaussian integral:
            I(λ) = [2π Ai(0) (3λ|α|)^(-1/3) e^{iπ/6 sign(α)}] * [sqrt(2π/(λ|β|)) e^{iπ/4 sign(β)}]
        
        where:
            - α is the cubic coefficient of the degenerate direction.
            - β is the quadratic coefficient of the transverse (non-degenerate) direction.
            - The scaling is O(λ^(-5/6)), combining O(λ^(-1/3)) and O(λ^(-1/2)).
        
        Args:
            cp: CriticalPoint classified as AIRY_2D with canonical coefficients.
            lam: Large parameter λ.
            
        Returns:
            AsymptoticContribution with scaling O(λ^(-5/6)).
            No correction term is computed (set to 0j).
        """
        coeffs = cp.canonical_coefficients
        alpha = coeffs['cubic']
        beta = coeffs['quadratic_transverse']
        
        # Transverse Gaussian Integral
        # For ∫ exp(iλβv²/2) dv = √(2π/(λβ)) exp(iπ/4 sign(β))
        scale_v = np.sqrt(2 * np.pi / (lam * np.abs(beta)))
        phase_v = np.exp(1j * np.pi/4 * np.sign(beta))
        
        # Degenerate Airy Integral (REAL, no Maslov phase):
        # ∫ exp(iλα u³/3) du = 2π Ai(0) · (λ|α|)^(-1/3)
        # Sub t = (λ|α|)^(1/3)·u → (λ|α|)^(-1/3) ∫ exp(it³/3) dt = (λ|α|)^(-1/3)·2πAi(0)
        # The integral over ℝ is REAL: cos(λu³/3) is even → contributes; sin is odd → cancels.
        # NOTE: (3λ|α|)^(-1/3) would be wrong — there is NO factor of 3.
        Ai0 = airy(0)[0]
        scale_u = 2 * np.pi * Ai0 * (lam * np.abs(alpha))**(-1.0/3.0)
        # No phase_u: integral over ℝ of exp(iλαu³/3) is purely real for any sign of α.
        
        val = (cp.amplitude_value * np.exp(1j * lam * cp.phase_value) * scale_u * scale_v * phase_v)
        
        return AsymptoticContribution(
            leading_term=val,
            correction_term=0j,
            total_value=val,
            point=cp,
            order_leading=5.0/6.0  # 1/3 (Airy) + 1/2 (Transverse Gaussian)
        )

    def _eval_pearcey(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Evaluate the Pearcey catastrophe integral contribution (corank 1, quartic).
        
        For a canonical Pearcey integral of the form:
            ∫∫ exp(iλ [γ u⁴/4 + β v²/2]) du dv
            
        The asymptotic formula uses the specific value of the Pearcey integral at the origin:
            I(λ) = [0.5 Γ(1/4) (λ|γ|)^(-1/4) e^{iπ/8 sign(γ)}] * [sqrt(2π/(λ|β|)) e^{iπ/4 sign(β)}]
            
        where:
            - γ is the quartic coefficient in the canonical form φ ~ γ u⁴/4.
            - β is the quadratic coefficient of the transverse direction.
            - The scaling is O(λ^(-3/4)), combining O(λ^(-1/4)) and O(λ^(-1/2)).
            
        Note:
            This evaluates the "cusp" catastrophe (A3) at its singular point.
            
        Args:
            cp: CriticalPoint classified as PEARCEY with canonical coefficients.
            lam: Large parameter λ.
            
        Returns:
            AsymptoticContribution with scaling O(λ^(-3/4)).
            No correction term is computed (set to 0j).
        """
        coeffs = cp.canonical_coefficients
        gamma_coeff = coeffs['quartic']
        beta_coeff = coeffs['quadratic_transverse']
        
        if abs(gamma_coeff) < self.tolerance or abs(beta_coeff) < self.tolerance:
            warnings.warn("Near-zero coefficients in Pearcey evaluation")
            return AsymptoticContribution(0j, 0j, 0j, cp, 0.75)
        
        # Exact asymptotic constant for ∫ exp(iλγu⁴/4) du:
        # Sub t = (λ|γ|)^{1/4}·u → (λ|γ|)^{-1/4} ∫ exp(it⁴/4) dt
        # ∫_{-∞}^∞ exp(it⁴/4) dt = 4^{1/4} · ∫ exp(iv⁴) dv  [sub v = t/4^{1/4}]
        #                         = 4^{1/4} · (1/2)·Γ(1/4)·exp(iπ/8)
        # Therefore: ∫ exp(iλγu⁴/4) du = (4/(λ|γ|))^{1/4} · (1/2)·Γ(1/4)·exp(iπ sign(γ)/8)
        # NOTE: (1/(λ|γ|))^{1/4} is WRONG — the correct factor is (4/(λ|γ|))^{1/4} = √2/(λ|γ|)^{1/4}
        pearcey_factor = (4.0 / (lam * abs(gamma_coeff)))**0.25 * 0.5 * gamma(0.25)
        
        # Transverse Gaussian factor: ∫ exp(iλβv²/2) dv = √(2π/(λ|β|)) exp(iπ sign(β)/4)
        gaussian_factor = np.sqrt(2.0 * np.pi / (lam * abs(beta_coeff)))
        
        # Maslov phases
        maslov_degen = np.exp(1j * np.pi * np.sign(gamma_coeff) / 8.0)
        maslov_trans = np.exp(1j * np.pi * np.sign(beta_coeff) / 4.0)
        
        leading = (cp.amplitude_value * np.exp(1j * lam * cp.phase_value) *
                   pearcey_factor * gaussian_factor * maslov_degen * maslov_trans)
        
        return AsymptoticContribution(
            leading_term=leading,
            correction_term=0j,
            total_value=leading,
            point=cp,
            order_leading=0.75
        )


class LaplaceEvaluator:
    """
    Evaluator for exponentially damped integrals of the form:
        I(λ) = ∫ a(x) exp(-λ φ(x)) dx,  λ → +∞

    Uses Laplace's method with second-order asymptotic corrections O(λ^(-n/2-1)).

    The critical point must be a strict minimum of φ (positive definite Hessian).
    Saddle points and maxima are not supported: the Laplace method relies on the
    Gaussian concentration of the integrand around the minimum.

    Returns an AsymptoticContribution with method=IntegralMethod.LAPLACE for
    consistency with StationaryPhaseEvaluator.
    """

    def evaluate(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Standard Laplace formula for n dimensions with anharmonic corrections.

        Leading term (Order 0):
            I₀(λ) = a(x_c) · exp(-λ φ(x_c)) · (2π/λ)^(n/2) · |det H|^(-1/2)

        Correction term (Order 1, relative order O(1/λ)):
            I₁(λ) = I₀(λ) · (1/λ) · C

        where the real correction factor C is:
            C = (1/2) Tr(H⁻¹ ∇²a)
              − (1/2) ⟨∇a, (H⁻¹ ⊗ H⁻¹) D³φ⟩
              − (1/8) (H⁻¹ ⊗ H⁻¹) : D⁴φ
              + (5/24) (H⁻¹ ⊗ H⁻¹ ⊗ H⁻¹) : (D³φ ⊗ D³φ)

        Args:
            cp: CriticalPoint with a non-degenerate, positive definite Hessian.
                Must contain hessian_inv (or hessian_matrix), hess_amp, grad_amp,
                phase_d3, phase_d4.
            lam: Large parameter λ. The approximation improves as λ → +∞.

        Returns:
            AsymptoticContribution with:
                - leading_term:    I₀(λ), real for real a and φ.
                - correction_term: I₁(λ), the O(λ^(-n/2-1)) correction.
                - total_value:     I₀ + I₁.
                - order_leading:   n/2.
                - method:          IntegralMethod.LAPLACE.

        Raises:
            ValueError: If the Hessian is singular (det H ≈ 0).
        """
        n = len(cp.position)
        inv_lam = 1.0 / lam

        # --- Leading term (Order 0) ---
        # I₀ = a(x_c) · exp(-λ φ(x_c)) · (2π/λ)^(n/2) / √|det H|
        exponent = np.exp(-lam * np.real(cp.phase_value))

        det_h_abs = np.abs(cp.hessian_det)
        if det_h_abs < 1e-15:
            raise ValueError(
                "Hessian is singular at the critical point: Laplace method fails. "
                "Use a higher-order evaluator for degenerate critical points."
            )

        prefactor = (2.0 * np.pi * inv_lam) ** (n / 2.0) / np.sqrt(det_h_abs)
        term0 = cp.amplitude_value * exponent * prefactor

        # --- Second-order corrections (O(1/λ)) ---
        # Retrieve or recompute the inverse Hessian.
        h_inv = cp.hessian_inv if cp.hessian_inv is not None else np.linalg.inv(cp.hessian_matrix)

        # 1. Amplitude curvature: (1/2) Tr(H⁻¹ ∇²a)
        term_lap_a = 0.5 * np.einsum('ij,ij', h_inv, cp.hess_amp)

        # 2. Amplitude-gradient / cubic-phase coupling:
        #    −(1/2) Σᵢⱼₖ (∇a)ᵢ (H⁻¹)ⱼₖ (H⁻¹)ₖₗ D³φⱼₖₗ
        #    Rewritten as −(1/2) ⟨∇a, [(H⁻¹ ⊗ H⁻¹) : D³φ]⟩
        t_as3 = np.einsum('i,jkl,ij,kl', cp.grad_amp, cp.phase_d3, h_inv, h_inv)

        # 3. Quartic phase anharmonicity:
        #    −(1/8) Σᵢⱼₖₗ (H⁻¹)ᵢⱼ (H⁻¹)ₖₗ D⁴φᵢⱼₖₗ
        t_s4 = np.einsum('ijkl,ij,kl', cp.phase_d4, h_inv, h_inv)

        # 4. Squared cubic anharmonicity (theta-graph Feynman diagram):
        #    +(5/24) Σᵢⱼₖₗₘₙ (H⁻¹)ᵢₗ (H⁻¹)ⱼₘ (H⁻¹)ₖₙ D³φᵢⱼₖ D³φₗₘₙ
        t_s3s3 = np.einsum('ijk,lmn,il,jm,kn', cp.phase_d3, cp.phase_d3, h_inv, h_inv, h_inv)

        # Combine correction terms (all real for real-valued a and φ).
        correction_factor = term_lap_a - 0.5 * t_as3 - 0.125 * t_s4 + (5.0 / 24.0) * t_s3s3
        correction_val = term0 * inv_lam * correction_factor

        total_value = term0 + correction_val

        return AsymptoticContribution(
            leading_term=complex(term0),
            correction_term=complex(correction_val),
            total_value=complex(total_value),
            point=cp,
            order_leading=n / 2.0,
            method=IntegralMethod.LAPLACE,
        )
        


# --- Saddle-Point Evaluator ---

class SaddlePointEvaluator:
    """
    Naive saddle-point evaluator for integrals with a genuinely complex phase.

    Handles integrals of the form:
        I(λ) = ∫ a(x) exp(iλφ(x)) dx,  φ = φ_R + i·φ_I,  λ → +∞

    where both the real part φ_R and the imaginary part φ_I are non-trivial.
    The integrand simultaneously oscillates (φ_R) and is exponentially damped
    (φ_I).  The asymptotic contribution is dominated by saddle points in ℂⁿ,
    i.e. solutions of ∇φ(z) = 0 with z ∈ ℂⁿ.

    Strategy (naive continuation)
    ------------------------------
    1. Start from real initial guesses x₀ ∈ ℝⁿ (supplied by the caller).
    2. Analytically continue into ℂⁿ by minimising |∇φ(z)|² over the 2n
       real degrees of freedom (Re z, Im z), using scipy.optimize.minimize.
    3. Accept a point z_c if |∇φ(z_c)|² < tolerance.
    4. Apply the standard Morse formula with the complex Hessian:

       I(λ) ≈ (2π/λ)^(n/2) · a(z_c) · exp(iλφ(z_c))
                             · 1/√det(∇²φ(z_c))

       where the complex square root is chosen with positive real part
       (principal branch convention).

    Limitations and warnings
    ------------------------
    - Contour validity is NOT checked.  The naive continuation finds a
      saddle point algebraically but does NOT verify that the original
      real integration contour can be deformed through that saddle without
      crossing other singularities (Picard-Lefschetz theory).  A
      RuntimeWarning is always emitted to remind the user of this.

    - Branch choice.  The complex square root √det H is multi-valued.
      This implementation uses numpy's principal branch (argument in
      (-π, π]).  The correct branch depends on the global topology of
      the steepest-descent contour.

    - Multiple saddles.  When several saddle points are found, ALL
      contributions are returned; their relative signs (Stokes phenomena)
      are not resolved.

    - Degenerate saddles (det H ≈ 0) are not supported; a warning is
      issued and a zero contribution is returned.

    References
    ----------
    .. [1] Bleistein & Handelsman, "Asymptotic Expansions of Integrals" (1975)
    .. [2] Delabaere & Howls, "Global asymptotics for multiple integrals with
           boundaries" (2002)
    """

    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Saddle-point search in ℂⁿ
    # ------------------------------------------------------------------

    def find_saddle_points(
        self,
        analyzer: 'Analyzer',
        initial_guesses: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Search for saddle points z_c ∈ ℂⁿ satisfying ∇φ(z_c) = 0.

        The search minimises the real function

            F(u, v) = |∇φ(u + iv)|²,   u, v ∈ ℝⁿ

        starting from (u₀, v₀) = (x₀, 0) for each real guess x₀.

        Parameters
        ----------
        analyzer : Analyzer
            Analyzer whose lambdified gradient func_grad is used.
            The phase must accept complex-valued arguments.
        initial_guesses : list of ndarray
            Real starting points in ℝⁿ.  Each is lifted to ℂⁿ by setting
            the imaginary part to zero.

        Returns
        -------
        list of complex ndarray
            Unique saddle points found (deduplicated within 1e-6).
            Each array has shape (dim,) and dtype complex128.
        """
        dim = analyzer.dim
        func_grad = analyzer.func_grad

        def objective(uv: np.ndarray) -> float:
            """Real objective: |∇φ(u + iv)|²."""
            z = uv[:dim] + 1j * uv[dim:]
            try:
                g = np.array(func_grad(*z), dtype=complex)
                return float(np.real(np.dot(g.conj(), g)))
            except Exception:
                return 1e30

        saddle_points: List[np.ndarray] = []

        for guess in initial_guesses:
            # Initial point: real guess, zero imaginary part
            uv0 = np.concatenate([np.real(guess), np.zeros(dim)])
            try:
                res = minimize(objective, uv0, method='L-BFGS-B',
                               tol=self.tolerance,
                               options={'maxiter': 2000, 'ftol': self.tolerance**2})
                if res.fun < self.tolerance:
                    z_c = res.x[:dim] + 1j * res.x[dim:]
                    # Deduplicate: reject if too close to an existing saddle
                    if not any(np.linalg.norm(z_c - s) < 1e-6 for s in saddle_points):
                        saddle_points.append(z_c)
            except Exception:
                pass

        return saddle_points

    # ------------------------------------------------------------------
    # Asymptotic formula at a single saddle point
    # ------------------------------------------------------------------

    def evaluate(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Evaluate the leading-order saddle-point contribution.

        Uses the standard multi-dimensional Morse formula extended to a
        complex critical point z_c:

            I(λ) ≈ (2π/λ)^(n/2) · a(z_c) · exp(iλφ(z_c))
                                 · 1/√det(∇²φ(z_c))

        The complex determinant det(∇²φ(z_c)) is evaluated with numpy's
        principal square root.

        .. warning::
            This contribution is valid only if the original integration
            contour can be deformed through z_c along a steepest-descent
            path.  This is NOT verified here (Picard-Lefschetz theory).
            Always examine the result critically.

        Parameters
        ----------
        cp : CriticalPoint
            Saddle point with method=SADDLE_POINT, produced by
            Analyzer.analyze_point() called with a complex
            coordinate returned by find_saddle_points().
        lam : float
            Large parameter λ > 0.

        Returns
        -------
        AsymptoticContribution
            leading_term  : saddle-point formula value.
            correction_term : 0j (not implemented for saddle points).
            order_leading : n/2.
            method        : IntegralMethod.SADDLE_POINT.
        """
        warnings.warn(
            "SaddlePointEvaluator: contour validity (Picard-Lefschetz) is NOT "
            "checked.  The contribution is correct only if the integration contour "
            "can be deformed through this saddle point.  Verify independently.",
            RuntimeWarning,
            stacklevel=2,
        )

        dim = len(cp.position)

        # Degenerate saddle: det H ≈ 0 → formula undefined
        if abs(cp.hessian_det) < self.tolerance:
            warnings.warn(
                f"SaddlePointEvaluator: det(∇²φ) ≈ 0 at saddle {cp.position}. "
                "Degenerate saddle points are not supported; returning zero contribution.",
                RuntimeWarning,
                stacklevel=2,
            )
            return AsymptoticContribution(
                leading_term=0j,
                correction_term=0j,
                total_value=0j,
                point=cp,
                order_leading=dim / 2.0,
                method=IntegralMethod.SADDLE_POINT,
            )

        # (2π/λ)^(n/2)
        prefactor = (2.0 * np.pi / lam) ** (dim / 2.0)

        # exp(iλφ(z_c)) — complex phase, possibly with exponential decay
        phase_osc = np.exp(1j * lam * cp.phase_value)

        # 1/√det(∇²φ(z_c)) — complex square root, principal branch.
        # This encodes both the Gaussian curvature and the Maslov-like phase
        # at the complex saddle point.
        sqrt_det = np.sqrt(complex(cp.hessian_det))  # principal branch

        leading = prefactor * cp.amplitude_value * phase_osc / sqrt_det

        return AsymptoticContribution(
            leading_term=leading,
            correction_term=0j,
            total_value=leading,
            point=cp,
            order_leading=dim / 2.0,
            method=IntegralMethod.SADDLE_POINT,
        )


# --- Unified Façade ---

class AsymptoticEvaluator:
    """
    Unified façade that dispatches asymptotic evaluation to the appropriate
    evaluator based on the integration method stored in a CriticalPoint.

    This is the recommended entry point for end users.  Routing table:

        cp.method == STATIONARY_PHASE  →  StationaryPhaseEvaluator
        cp.method == LAPLACE           →  LaplaceEvaluator
        cp.method == SADDLE_POINT      →  SaddlePointEvaluator

    The method is determined automatically when the analyzer is constructed
    with method=AUTO (the default).

    Usage
    -----
    >>> analyzer = Analyzer(phi, amp, [x, y])  # AUTO by default
    >>> print(analyzer.method)   # e.g. IntegralMethod.SADDLE_POINT
    >>> pts = analyzer.find_critical_points([np.array([0., 0.])])
    >>> cp  = analyzer.analyze_point(pts[0])
    >>> result = AsymptoticEvaluator().evaluate(cp, lam=100)
    >>> print(result.method, result.total_value)

    For SADDLE_POINT, use SaddlePointEvaluator.find_saddle_points() to
    obtain complex coordinates before calling analyze_point():

    >>> sp_eval = SaddlePointEvaluator()
    >>> saddles = sp_eval.find_saddle_points(analyzer, real_guesses)
    >>> cp = analyzer.analyze_point(saddles[0])
    >>> result = AsymptoticEvaluator().evaluate(cp, lam=100)

    Attributes
    ----------
    sp_evaluator : StationaryPhaseEvaluator
    laplace_evaluator : LaplaceEvaluator
    saddle_evaluator : SaddlePointEvaluator
    """

    def __init__(self, tolerance: float = 1e-8):
        self.sp_evaluator      = StationaryPhaseEvaluator(tolerance=tolerance)
        self.laplace_evaluator = LaplaceEvaluator()
        self.saddle_evaluator  = SaddlePointEvaluator(tolerance=tolerance)

    def evaluate(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Evaluate the asymptotic contribution at parameter λ.

        The evaluation method is selected from ``cp.method``:
        - STATIONARY_PHASE → full singularity-type dispatch (Morse, Airy, Pearcey …)
        - LAPLACE          → Laplace formula with O(1/λ) corrections
        - SADDLE_POINT     → complex Morse formula (naive; see SaddlePointEvaluator)

        Parameters
        ----------
        cp : CriticalPoint
            Critical/saddle point from Analyzer.analyze_point().
        lam : float
            Large asymptotic parameter λ > 0.

        Returns
        -------
        AsymptoticContribution

        Raises
        ------
        ValueError
            If cp.method is None, AUTO, or unrecognised.
        """
        if cp.method is None or cp.method == IntegralMethod.AUTO:
            raise ValueError(
                "CriticalPoint.method is None or AUTO. "
                "Make sure the point was produced by Analyzer "
                "after method resolution (AUTO should have been replaced by a "
                "concrete method during __init__)."
            )
        if cp.method == IntegralMethod.STATIONARY_PHASE:
            return self.sp_evaluator.evaluate(cp, lam)
        elif cp.method == IntegralMethod.LAPLACE:
            return self.laplace_evaluator.evaluate(cp, lam)
        elif cp.method == IntegralMethod.SADDLE_POINT:
            return self.saddle_evaluator.evaluate(cp, lam)
        else:
            raise ValueError(f"Unknown IntegralMethod: {cp.method!r}")


import matplotlib.pyplot as plt
import matplotlib.cm as cm

class AsymptoticVisualizer:
    """
    Visualization toolkit for asymptotic analysis — supports all three integration
    methods (STATIONARY_PHASE, LAPLACE, SADDLE_POINT).

    Provides three diagnostic plots, each adapted to the nature of the phase φ:

    plot_phase_landscape
        2D contour map of φ with critical / saddle-point overlay.
        - φ real   (STATIONARY_PHASE) : single panel, Re(φ).
        - φ imag   (LAPLACE)          : single panel, Im(φ) = ψ (the damping potential).
        - φ complex (SADDLE_POINT)    : two panels side-by-side, Re(φ) and Im(φ).
        The ``display`` parameter overrides the automatic choice
        ('real', 'imag', 'both', 'abs', 'arg').

    plot_integrand
        2D map of the integrand f(x,y) = a(x,y)·exp(iλφ(x,y)) at a given λ.
        - STATIONARY_PHASE : single panel, Re(f)  — pure oscillation, |f| = const.
        - LAPLACE          : single panel, Re(f) = a·exp(-λψ)  — exponential envelope.
        - SADDLE_POINT     : two panels, Re(f) and |f| = |a|·exp(-λ Im φ), revealing
                             both the oscillation pattern and the exponential damping.

    plot_asymptotic_convergence
        Log-log plot of |I₀(λ)| and |I₁(λ)| vs λ for any dimension and any method.
        Overlays the theoretical decay slope λ^(-p) for verification.

    Notes
    -----
    plot_phase_landscape and plot_integrand require dim = 2.
    plot_asymptotic_convergence works for any dimension.
    """

    # ------------------------------------------------------------------
    # Marker / colour convention for critical-point overlay (shared)
    # ------------------------------------------------------------------
    _MARKER_STYLE: Dict[SingularityType, Tuple] = {
        SingularityType.MORSE       : ('o', 'red',    'Morse'),
        SingularityType.AIRY_1D     : ('*', 'orange', 'Airy'),
        SingularityType.AIRY_2D     : ('*', 'orange', 'Airy'),
        SingularityType.PEARCEY     : ('D', 'magenta','Pearcey'),
        SingularityType.HIGHER_ORDER: ('s', 'gray',   'Higher-order'),
    }

    def __init__(self, analyzer: 'Analyzer'):
        """
        Parameters
        ----------
        analyzer : Analyzer
            Analyzer instance (any method, any dimension).
            dim = 2 is required for plot_phase_landscape and plot_integrand.
        """
        self.analyzer = analyzer
        if analyzer.dim != 2:
            warnings.warn(
                f"plot_phase_landscape and plot_integrand require dim=2 "
                f"(received dim={analyzer.dim}). "
                "plot_asymptotic_convergence works for any dimension.",
                UserWarning,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_grid(
        self,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        n: int,
    ):
        """Return (X, Y, phi_val, amp_val) on a regular n×n grid."""
        x_range = np.linspace(bounds[0][0], bounds[0][1], n)
        y_range = np.linspace(bounds[1][0], bounds[1][1], n)
        X, Y = np.meshgrid(x_range, y_range)
        phi_val = np.asarray(self.analyzer.func_phase(X, Y), dtype=complex)
        amp_val = np.asarray(self.analyzer.func_amp(X, Y),   dtype=complex)
        return X, Y, phi_val, amp_val

    def _overlay_critical_points(
        self,
        ax,
        critical_points: List[CriticalPoint],
        offset: float = 0.12,
    ) -> None:
        """Scatter-plot critical / saddle points on an existing Axes."""
        plotted_labels: set = set()
        for cp in critical_points:
            marker, color, label = self._MARKER_STYLE.get(
                cp.singularity_type,
                ('s', 'gray', 'Unknown'),
            )
            # For complex (saddle) positions, project to real part for 2D display
            px = float(np.real(cp.position[0]))
            py = float(np.real(cp.position[1]))

            kw = dict(c=color, s=150, marker=marker,
                      edgecolors='white', linewidths=1.5, zorder=10)
            if label not in plotted_labels:
                ax.scatter(px, py, label=label, **kw)
                plotted_labels.add(label)
            else:
                ax.scatter(px, py, **kw)

            ax.text(
                px + offset, py + offset,
                cp.singularity_type.value,
                fontsize=8, color='white', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=1),
            )

    @staticmethod
    def _contour_panel(
        fig, ax, X, Y, Z: np.ndarray,
        title: str, label: str,
        cmap: str = 'viridis', n_levels: int = 40,
    ) -> None:
        """Draw a filled contour panel with a colour bar."""
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        if z_min == z_max:          # constant field — avoid degenerate levels
            z_min -= 1e-10
            z_max += 1e-10
        levels = np.linspace(z_min, z_max, n_levels)
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.88)
        ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.35, alpha=0.25)
        fig.colorbar(cf, ax=ax, label=label, shrink=0.92)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('$x$', fontsize=11)
        ax.set_ylabel('$y$', fontsize=11)
        ax.set_aspect('equal', adjustable='box')

    @staticmethod
    def _imshow_panel(
        fig, ax, data: np.ndarray,
        bounds: Tuple, title: str, label: str,
        cmap: str = 'RdBu_r', symmetric: bool = True,
    ) -> None:
        """Draw an imshow panel with a colour bar."""
        vmax = np.nanmax(np.abs(data))
        vmin = -vmax if symmetric else 0.0
        im = ax.imshow(
            data,
            extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
            origin='lower', cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation='bilinear',
        )
        fig.colorbar(im, ax=ax, label=label, shrink=0.92)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('$x$', fontsize=11)
        ax.set_ylabel('$y$', fontsize=11)

    # ------------------------------------------------------------------
    # Public plots
    # ------------------------------------------------------------------

    def plot_phase_landscape(
        self,
        critical_points: List[CriticalPoint],
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-3, 3), (-3, 3)),
        points_per_axis: int = 120,
        display: Optional[str] = None,
    ) -> None:
        """
        Visualize the phase function φ(x,y) with critical/saddle-point overlay.

        The panels shown depend on the integration method (or on ``display``):

        +--------------------+---------------------------------------------------+
        | Method / display   | Panels                                            |
        +====================+===================================================+
        | STATIONARY_PHASE   | Re(φ)                                             |
        | display='real'     |                                                   |
        +--------------------+---------------------------------------------------+
        | LAPLACE            | Im(φ) = ψ  (the damping potential)                |
        | display='imag'     |                                                   |
        +--------------------+---------------------------------------------------+
        | SADDLE_POINT       | Re(φ)  |  Im(φ)  side-by-side                    |
        | display='both'     |                                                   |
        +--------------------+---------------------------------------------------+
        | display='abs'      | |φ|                                               |
        +--------------------+---------------------------------------------------+
        | display='arg'      | arg(φ)                                            |
        +--------------------+---------------------------------------------------+

        Parameters
        ----------
        critical_points : list of CriticalPoint
            Points to overlay (real or complex coordinates accepted).
        bounds : pair of (min, max) pairs
            Spatial domain  ((x_min, x_max), (y_min, y_max)).
        points_per_axis : int
            Grid resolution (default 120).
        display : str or None
            Override automatic panel selection.  One of
            'real', 'imag', 'both', 'abs', 'arg'.

        Notes
        -----
        Marker conventions (shared with plot_integrand):
        ○ red    — Morse (non-degenerate)
        ★ orange — Airy  (corank 1, cubic)
        ◆ magenta — Pearcey (corank 1, quartic)
        □ gray   — Higher-order / unclassified
        """
        if self.analyzer.dim != 2:
            warnings.warn("plot_phase_landscape requires dim=2.", UserWarning)
            return

        X, Y, phi_val, _ = self._make_grid(bounds, points_per_axis)

        # Determine which panels to draw
        method = self.analyzer.method
        if display is None:
            if method == IntegralMethod.STATIONARY_PHASE:
                display = 'real'
            elif method == IntegralMethod.LAPLACE:
                display = 'imag'
            else:                          # SADDLE_POINT or unknown
                display = 'both'

        panel_map = {
            'real': [(np.real(phi_val), r'$\operatorname{Re}(\phi)$',
                      r'$\operatorname{Re}(\phi(x,y))$', 'viridis')],
            'imag': [(np.imag(phi_val), r'$\operatorname{Im}(\phi) = \psi$',
                      r'$\operatorname{Im}(\phi(x,y))$', 'plasma')],
            'abs' : [(np.abs(phi_val),  r'$|\phi|$',
                      r'$|\phi(x,y)|$',  'magma')],
            'arg' : [(np.angle(phi_val),r'$\arg(\phi)$',
                      r'$\arg(\phi(x,y))$', 'hsv')],
            'both': [(np.real(phi_val), r'$\operatorname{Re}(\phi)$',
                      r'$\operatorname{Re}(\phi(x,y))$', 'viridis'),
                     (np.imag(phi_val), r'$\operatorname{Im}(\phi) = \psi$',
                      r'$\operatorname{Im}(\phi(x,y))$', 'plasma')],
        }
        if display not in panel_map:
            raise ValueError(f"display must be one of {list(panel_map)}; got {display!r}")

        panels = panel_map[display]
        ncols  = len(panels)
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6),
                                 squeeze=False)

        for ax, (data, cbar_label, title, cmap) in zip(axes[0], panels):
            self._contour_panel(fig, ax, X, Y, data, title, cbar_label, cmap)
            self._overlay_critical_points(ax, critical_points)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.85)

        method_label = method.value.replace('_', ' ').title()
        fig.suptitle(
            f'Phase Landscape — {method_label}',
            fontsize=14, fontweight='bold', y=0.98,
        )
        plt.tight_layout()
        plt.show()

    def plot_integrand(
        self,
        lam_value: float,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-3, 3), (-3, 3)),
        points_per_axis: int = 200,
    ) -> None:
        """
        Visualize the structure of the integrand f = a(x,y)·exp(iλφ(x,y)) at
        a given parameter λ.

        The panels shown depend on the integration method:

        +--------------------+---------------------------------------------------+
        | Method             | Panels                                            |
        +====================+===================================================+
        | STATIONARY_PHASE   | Re(f)  — oscillation with unit-modulus envelope   |
        +--------------------+---------------------------------------------------+
        | LAPLACE            | f = a·exp(-λψ)  — real exponential concentration  |
        +--------------------+---------------------------------------------------+
        | SADDLE_POINT       | Re(f)  |  |f| = |a|·exp(-λ Im φ) side-by-side:   |
        |                    | left shows oscillations, right shows the          |
        |                    | exponential damping envelope                      |
        +--------------------+---------------------------------------------------+

        Parameters
        ----------
        lam_value : float
            Parameter λ.  Larger values produce finer oscillations / sharper
            concentration.
        bounds : pair of (min, max) pairs
            Spatial domain  ((x_min, x_max), (y_min, y_max)).
        points_per_axis : int
            Grid resolution (default 200).  Increase for large λ to resolve
            fine oscillations.

        Notes
        -----
        For STATIONARY_PHASE, as λ increases the oscillations become finer
        everywhere *except* near stationary points (∇φ = 0), where the phase
        is locally flat — this is the geometric core of the method.

        For LAPLACE, the integrand concentrates sharply around the minimum of
        Im(φ) = ψ, illustrating why only a small neighbourhood contributes.

        For SADDLE_POINT, |f| reveals the exponential ridge structure while
        Re(f) shows the additional rapid oscillations along the ridge.
        """
        if self.analyzer.dim != 2:
            warnings.warn("plot_integrand requires dim=2.", UserWarning)
            return

        X, Y, phi_val, amp_val = self._make_grid(bounds, points_per_axis)
        method = self.analyzer.method

        # Full complex integrand  f = a · exp(iλφ)
        f = amp_val * np.exp(1j * lam_value * phi_val)
        re_f  = np.real(f)
        abs_f = np.abs(f)

        if method == IntegralMethod.STATIONARY_PHASE:
            # Pure oscillation: |f| = |a| is independent of λ; show Re(f).
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self._imshow_panel(
                fig, ax, re_f, bounds,
                title=rf'$\operatorname{{Re}}[a\, e^{{i\lambda\phi}}]$'
                      rf' — Stationary Phase  ($\lambda={lam_value}$)',
                label=r'$\operatorname{Re}[f]$',
                cmap='RdBu_r', symmetric=True,
            )

        elif method == IntegralMethod.LAPLACE:
            # Real exponential: f = a·exp(-λψ); always real for real a and ψ.
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self._imshow_panel(
                fig, ax, re_f, bounds,
                title=rf'$a\, e^{{-\lambda\psi}}$'
                      rf' — Laplace  ($\lambda={lam_value}$)',
                label=r'$a\,e^{-\lambda\psi}$',
                cmap='hot', symmetric=False,
            )

        else:  # SADDLE_POINT: show oscillation + damping envelope side-by-side
            fig, (ax_re, ax_abs) = plt.subplots(1, 2, figsize=(15, 6))
            self._imshow_panel(
                fig, ax_re, re_f, bounds,
                title=rf'$\operatorname{{Re}}[a\, e^{{i\lambda\phi}}]$'
                      rf'  ($\lambda={lam_value}$)',
                label=r'$\operatorname{Re}[f]$',
                cmap='RdBu_r', symmetric=True,
            )
            self._imshow_panel(
                fig, ax_abs, abs_f, bounds,
                title=rf'$|a\, e^{{i\lambda\phi}}| = |a|\,e^{{-\lambda\,\operatorname{{Im}}\phi}}$'
                      rf'  ($\lambda={lam_value}$)',
                label=r'$|f|$',
                cmap='hot', symmetric=False,
            )

        plt.tight_layout()
        plt.show()

    def plot_asymptotic_convergence(
        self,
        cp: CriticalPoint,
        lambda_start: float = 10,
        lambda_end: float = 1000,
        num_points: int = 50,
    ) -> None:
        """
        Log-log convergence diagnostic for any method and any dimension.

        Plots |I₀(λ)| and |I₁(λ)| vs λ and compares the empirical slope
        with the theoretical decay exponent −p:

        +--------------------+------------------------------------------+
        | Method / type      | Theoretical slope  −p                   |
        +====================+==========================================+
        | Morse (any dim n)  | −n/2                                     |
        | Airy 1D            | −1/3                                     |
        | Airy 2D            | −5/6                                     |
        | Pearcey            | −3/4                                     |
        | Laplace (any n)    | −n/2                                     |
        | Saddle-point       | −n/2  (complex Morse)                    |
        +--------------------+------------------------------------------+

        Parameters
        ----------
        cp : CriticalPoint
            Critical / saddle point (any method, any dimension).
        lambda_start : float
            Minimum λ for the convergence sweep (default 10).
        lambda_end : float
            Maximum λ (default 1000).
        num_points : int
            Number of log-spaced λ samples (default 50).

        Notes
        -----
        A straight line on the log-log plot confirms the asymptotic regime.
        Deviations at small λ indicate pre-asymptotic behaviour.
        The correction term |I₁| should be parallel to |I₀| but shifted
        down by slope −1 for Morse / Laplace points.
        """
        # Use the unified evaluator so all three methods are handled.
        evaluator = AsymptoticEvaluator()
        lams = np.logspace(np.log10(lambda_start), np.log10(lambda_end), num_points)

        abs_leading    = []
        abs_correction = []

        for lam in lams:
            with warnings.catch_warnings():
                # Suppress the Picard-Lefschetz warning during the sweep;
                # it has already been emitted when the saddle was found.
                warnings.simplefilter('ignore', RuntimeWarning)
                res = evaluator.evaluate(cp, lam)
            abs_leading.append(np.abs(res.leading_term))
            abs_correction.append(np.abs(res.correction_term))

        abs_leading    = np.array(abs_leading)
        abs_correction = np.array(abs_correction)

        # Theoretical decay exponent (negative slope on log-log)
        dim = len(cp.position)
        theoretical_order: Optional[float] = {
            SingularityType.MORSE       : dim / 2.0,
            SingularityType.AIRY_1D     : 1.0 / 3.0,
            SingularityType.AIRY_2D     : 5.0 / 6.0,
            SingularityType.PEARCEY     : 3.0 / 4.0,
            SingularityType.HIGHER_ORDER: None,
        }.get(cp.singularity_type, dim / 2.0)
        # For Laplace and Saddle-point the Morse formula applies (n/2).
        if cp.method in (IntegralMethod.LAPLACE, IntegralMethod.SADDLE_POINT):
            theoretical_order = dim / 2.0

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.loglog(lams, abs_leading, 'o-',
                  label=r'Leading term $|I_0(\lambda)|$',
                  linewidth=2.5, markersize=4, alpha=0.85)

        # Show correction term only if it is numerically significant
        if np.any(abs_correction > 1e-15 * np.max(abs_leading)):
            ax.loglog(lams, abs_correction, 's--',
                      label=r'Correction term $|I_1(\lambda)|$',
                      linewidth=2.0, markersize=3, alpha=0.80)

        # Empirical slope via linear regression on log-log data
        valid = abs_leading > 0
        if valid.sum() >= 2:
            slope_lead = np.polyfit(np.log(lams[valid]),
                                    np.log(abs_leading[valid]), 1)[0]
        else:
            slope_lead = float('nan')

        # Overlay theoretical reference line  λ^(-p)
        if theoretical_order is not None:
            ref = abs_leading[0] * (lams / lams[0]) ** (-theoretical_order)
            ax.loglog(lams, ref, 'k:', linewidth=1.4, alpha=0.6,
                      label=rf'Ref. slope $\lambda^{{-{theoretical_order:.3g}}}$')

        # Annotation box with empirical vs theoretical slope
        annotation = f'Empirical slope: {slope_lead:.3f}'
        if theoretical_order is not None:
            annotation += f'\nTheoretical: −{theoretical_order:.3g}'
        ax.text(lams[4], abs_leading[4] * 2.0, annotation,
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        ax.grid(True, which='both', ls=':', alpha=0.6)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlabel(r'Parameter $\lambda$ (log scale)', fontsize=12)
        ax.set_ylabel(r'Magnitude $|I(\lambda)|$ (log scale)', fontsize=12)

        # Build a title that works for any dimension
        method_label = cp.method.value.replace('_', ' ').title()
        pos_str = ', '.join(f'{np.real(v):.3g}' for v in cp.position)
        ax.set_title(
            f'Asymptotic Convergence — {method_label} / '
            f'{cp.singularity_type.value.capitalize()}\n'
            rf'$x_c = ({pos_str})$,  '
            rf'$\phi(x_c) = {cp.phase_value:.4g}$',
            fontsize=12,
        )
        plt.tight_layout()
        plt.show()


# Backward-compatible alias so that existing code using StationaryPhaseVisualizer
# continues to work without modification.
StationaryPhaseVisualizer = AsymptoticVisualizer
        
# --- Execution Example ---

if __name__ == "__main__":
    x, y = sp.symbols('x y')

    # =========================================================================
    # Helper: shared report printer (works for all three methods)
    # =========================================================================
    def print_report(label: str, points, analyzer,
                     lam_values=(10, 100, 1000),
                     saddle_points=None):
        """
        Print an asymptotic analysis report for the first critical/saddle point.

        For SADDLE_POINT, pass the complex coordinates via `saddle_points`
        (produced by SaddlePointEvaluator.find_saddle_points).
        """
        evaluator = AsymptoticEvaluator()
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  Detected method: {analyzer.method.value}")
        print(f"{'='*60}")

        # Select coordinate list: real points or complex saddle points
        coords = saddle_points if saddle_points is not None else points
        if not coords:
            print("  No critical/saddle points found.")
            return

        cp = analyzer.analyze_point(coords[0])
        print(f"  Critical point : {cp.position}")
        print(f"  Singularity    : {cp.singularity_type.value}")
        print(f"  Hessian det    : {cp.hessian_det:.4g}")

        for lam in lam_values:
            try:
                res = evaluator.evaluate(cp, lam)
            except RuntimeWarning:
                pass  # saddle-point warning already emitted; continue
            ratio = (np.abs(res.correction_term) / np.abs(res.leading_term)
                     if np.abs(res.leading_term) > 0 else float('nan'))
            print(f"\n  λ = {lam}")
            print(f"    Leading term      : {res.leading_term:.4e}")
            print(f"    Correction term   : {res.correction_term:.4e}")
            print(f"    Total             : {res.total_value:.4e}")
            print(f"    |correction/lead| : {ratio:.2%}")

    # =========================================================================
    # 1. φ purely real → AUTO detects STATIONARY_PHASE
    #    I(λ) = ∫∫ (1 + x²) exp(iλ (x²/2 + y²/2 + x³/10)) dx dy
    # =========================================================================
    phi_real = x**2/2 + y**2/2 + x**3/10
    amp_real = 1 + x**2

    analyzer_sp = Analyzer(phi_real, amp_real, [x, y])
    pts_sp = analyzer_sp.find_critical_points([np.array([0., 0.])])
    print_report("Case 1 — φ real (AUTO → STATIONARY_PHASE)", pts_sp, analyzer_sp)

    # =========================================================================
    # 2. φ purely imaginary → AUTO detects LAPLACE
    #    φ = i·ψ  with ψ = x²/2 + y²/2 + x³/20  (ψ real, minimum at origin)
    #    The integral becomes ∫∫ exp(-λ ψ(x,y)) dx dy
    # =========================================================================
    phi_imag = sp.I * (x**2/2 + y**2/2 + x**3/20)
    amp_imag = sp.Integer(1)

    analyzer_lap = Analyzer(phi_imag, amp_imag, [x, y])
    pts_lap = analyzer_lap.find_critical_points([np.array([0., 0.])])
    print_report("Case 2 — φ imaginary (AUTO → LAPLACE)", pts_lap, analyzer_lap)

    # =========================================================================
    # 3. φ genuinely complex → AUTO detects SADDLE_POINT
    #    φ = (x²/2 + y²/2) + i·(x²/4 + y²/4)
    #      = (1/2 + i/4)(x² + y²)
    #    Saddle point at origin; contribution is a complex Gaussian.
    # =========================================================================
    phi_cplx = (x**2/2 + y**2/2) + sp.I*(x**2/4 + y**2/4)
    amp_cplx = sp.Integer(1)

    analyzer_sdl = Analyzer(phi_cplx, amp_cplx, [x, y])

    # For SADDLE_POINT we use SaddlePointEvaluator to search in ℂⁿ
    sdl_eval = SaddlePointEvaluator()
    saddles  = sdl_eval.find_saddle_points(analyzer_sdl, [np.array([0., 0.])])

    print_report("Case 3 — φ complex (AUTO → SADDLE_POINT)", [],
                 analyzer_sdl, saddle_points=saddles)

    # =========================================================================
    # 4. Visualisations — one visualizer per case
    # =========================================================================
    print("\n--- Generating Visualizations ---")
    bounds2d = ((-2, 2), (-2, 2))

    # Case 1: STATIONARY_PHASE
    if pts_sp:
        cp_sp = analyzer_sp.analyze_point(pts_sp[0])
        viz_sp = AsymptoticVisualizer(analyzer_sp)
        viz_sp.plot_phase_landscape([cp_sp], bounds=bounds2d)
        viz_sp.plot_integrand(lam_value=50, bounds=bounds2d)
        viz_sp.plot_asymptotic_convergence(cp_sp)

    # Case 2: LAPLACE
    if pts_lap:
        cp_lap = analyzer_lap.analyze_point(pts_lap[0])
        viz_lap = AsymptoticVisualizer(analyzer_lap)
        viz_lap.plot_phase_landscape([cp_lap], bounds=bounds2d)
        viz_lap.plot_integrand(lam_value=10, bounds=bounds2d)
        viz_lap.plot_asymptotic_convergence(cp_lap)

    # Case 3: SADDLE_POINT
    if saddles:
        cp_sdl = analyzer_sdl.analyze_point(saddles[0])
        viz_sdl = AsymptoticVisualizer(analyzer_sdl)
        viz_sdl.plot_phase_landscape([cp_sdl], bounds=bounds2d)
        viz_sdl.plot_integrand(lam_value=20, bounds=bounds2d)
        viz_sdl.plot_asymptotic_convergence(cp_sdl)
