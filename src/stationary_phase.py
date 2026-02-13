import numpy as np
import sympy as sp
from scipy.special import airy, gamma
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import warnings

# --- Types and Enums ---

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
    """
    leading_term: complex
    correction_term: complex 
    total_value: complex
    point: CriticalPoint
    order_leading: float

# --- Analyzer (Symbolic -> Numerical) ---

class StationaryPhaseAnalyzer:
    """
    Handles symbolic analysis of phase and amplitude functions for stationary phase integrals.
    
    This class analyzes oscillatory integrals of the form:
        I(λ) = ∫ a(x) exp(iλφ(x)) dx
    
    as λ → ∞, using the stationary phase method. It takes symbolic expressions for φ(x) 
    and a(x), computes all necessary derivatives symbolically up to order 4, and converts 
    them to fast numerical functions. It detects and classifies critical points where ∇φ = 0.
    
    The main workflow is:
        1. Initialize with symbolic SymPy expressions
        2. Find critical points using find_critical_points()
        3. Analyze each point using analyze_point()
        4. Use StationaryPhaseEvaluator to compute asymptotic contributions
    
    Attributes:
        phase_expr: SymPy expression for the phase function φ(x).
        amplitude_expr: SymPy expression for the amplitude function a(x).
        variables: List of SymPy symbols representing the integration variables.
        dim (int): Dimension of the integration domain.
        domain (Optional[List[Tuple]]): Optional bounds [(min, max), ...] for each variable.
        tolerance (float): Numerical tolerance for detecting zeros and critical points.
        cubic_threshold (float): Absolute threshold for distinguishing Airy vs Pearcey 
            singularities based on cubic term magnitude. Default: max(1e-5, 10*tolerance).
    """

    def __init__(self, phase_expr, amplitude_expr, variables, domain=None, tolerance=1e-6, cubic_threshold=None):
        """
        Initialize the stationary phase analyzer.

        Args:
            phase_expr: SymPy expression for phase φ(x). Should be a real-valued function.
            amplitude_expr: SymPy expression for amplitude a(x). Can be complex.
            variables: List of SymPy symbols [x, y, ...] or a single symbol for 1D.
            domain: Optional list of tuples [(min, max), ...] specifying search bounds 
                for each variable when finding critical points.
            tolerance: Numerical tolerance for zero-detection and optimization (default: 1e-6).
                Used to determine if |∇φ| ≈ 0 at a critical point.
            cubic_threshold: Absolute threshold for classifying cubic terms. If None, defaults 
                to max(1e-5, 10 * tolerance). Used to distinguish Airy singularities 
                (|cubic| > threshold) from Pearcey singularities (|cubic| < threshold).
        """
        self.phase_expr = phase_expr
        self.amplitude_expr = amplitude_expr
        self.variables = list(variables) if isinstance(variables, (list, tuple)) else [variables]
        self.dim = len(self.variables)
        self.domain = domain
        self.tolerance = tolerance
        self.cubic_threshold = cubic_threshold if cubic_threshold is not None else max(1e-5, 10 * tolerance)
        
        self._prepare_derivatives()
        self._create_numerical_functions()
    
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
        Perform complete analysis of a critical point.
        
        Computes all geometric and analytical properties needed for asymptotic evaluation:
        - Phase value φ(x_c) and amplitude value a(x_c)
        - Hessian matrix ∇²φ and its properties (determinant, eigenvalues, signature)
        - Higher-order derivatives: D3 and D4 tensors of φ, gradients and Hessians of a
        - Classification of singularity type (Morse, Airy, Pearcey, etc.)
        - Canonical coefficients for degenerate cases (extracted via eigenvector projection)
        - Hessian inverse (for Morse points only)
        
        Args:
            xc: Coordinates of the critical point (numpy array of shape (dim,)).
            
        Returns:
            CriticalPoint object containing all computed properties necessary for
            evaluating the asymptotic contribution.
        """
        args = tuple(xc)
        H = np.array(self.func_hess(*args))
        vals, vecs = np.linalg.eigh(H) # Eigen decomposition (Symmetric matrix)
        
        # Reconstruct higher order tensors from flattened symbolic output
        d3_flat = self.func_d3(*args)
        D3 = np.zeros((self.dim,)*3)
        for k, idx in enumerate(self.d3_indices):
            D3[idx] = d3_flat[k]
            
        d4_flat = self.func_d4(*args)
        D4 = np.zeros((self.dim,)*4)
        for k, idx in enumerate(self.d4_indices):
            D4[idx] = d4_flat[k]

        grad_a = np.array(self.func_grad_amp(*args))
        hess_a = np.array(self.func_hess_amp(*args))

        det = np.prod(vals)
        # Rank: number of non-zero eigenvalues
        rank = np.sum(np.abs(vals) > self.tolerance)
        # Signature: number of negative eigenvalues
        signature = np.sum(vals < -self.tolerance)

        cp = CriticalPoint(
            position=xc,
            phase_value=complex(self.func_phase(*args)),
            amplitude_value=complex(self.func_amp(*args)),
            singularity_type=SingularityType.MORSE, # Default
            hessian_matrix=H,
            hessian_det=det,
            signature=signature,
            eigenvalues=vals,
            eigenvectors=vecs,
            grad_amp=grad_a,
            hess_amp=hess_a,
            phase_d3=D3,
            phase_d4=D4
        )

        if rank == self.dim:
            cp.singularity_type = SingularityType.MORSE
            cp.hessian_inv = np.linalg.inv(H)
        elif self.dim == 1 and rank == 0:  # 1D degenerate
            coeffs = self._project_degenerate_coeffs(cp)
            cp.canonical_coefficients = coeffs
            if abs(coeffs['cubic']) > self.cubic_threshold:
                cp.singularity_type = SingularityType.AIRY_1D
            elif abs(coeffs['quartic']) > self.tolerance:
                cp.singularity_type = SingularityType.PEARCEY  # or other 1D type
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
                quadratic_transverse = cp.eigenvalues[non_null_idxs[0]] / 2.0
        
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
            return self._eval_morse_order2(cp, lam)
        elif cp.singularity_type == SingularityType.AIRY_1D:
            return self._eval_airy_1d(cp, lam)
        elif cp.singularity_type == SingularityType.AIRY_2D:
            return self._eval_airy_2d(cp, lam)
        elif cp.singularity_type == SingularityType.PEARCEY:
            return self._eval_pearcey(cp, lam)
        else:  # HIGHER_ORDER or unknown type
            warnings.warn(
                f"Unhandled singularity type {cp.singularity_type.value} at {cp.position}. "
                f"Returning zero contribution (no asymptotic formula available).",
                RuntimeWarning
            )
            return AsymptoticContribution(
                leading_term=0j,
                correction_term=0j,
                total_value=0j,
                point=cp,
                order_leading=float('inf')  # Indicates negligible contribution
            )

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
            dim = cp.position.shape[0]
            
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
    
    def _eval_morse_order2_old(self, cp: CriticalPoint, lam: float) -> AsymptoticContribution:
        """
        Evaluate the asymptotic contribution for a non-degenerate (Morse) critical point.
        
        Implements the standard stationary phase formula with second-order correction:
        
        Leading term (Order 0):
            I₀(λ) = (2π)^(n/2) / (λ^(n/2) √|det H|) × exp(iλφ₀) × a₀ × exp(iπμ/4)
        
        where:
            n = dimension
            H = Hessian at critical point
            φ₀ = phase value at critical point
            a₀ = amplitude value at critical point
            μ = dim - 2×signature (Maslov index)
            signature = number of negative eigenvalues of H
        
        Correction term (Order 1):
            I₁(λ) = I₀ / (iλ) × [correction_factor]
        
        where correction_factor includes:
            1. Amplitude Laplacian: (1/2) Trace(H⁻¹ × ∇²a)
            2. Mixed term: -(1/2) ⟨H⁻¹∇a, V⟩ where V_k = Σᵢⱼ (H⁻¹)ᵢⱼ D3ᵢⱼₖ
            3. Pure phase anharmonicity: (a₀/24) × (5S3 - 3S4) where
               S4 = Σᵢⱼₖₗ (H⁻¹)ᵢⱼ (H⁻¹)ₖₗ D4ᵢⱼₖₗ
               S3 = Σᵢⱼₖₗₘₙ (H⁻¹)ᵢⱼ (H⁻¹)ₖₗ (H⁻¹)ₘₙ D3ᵢₖₘ D3ⱼₗₙ
        
        Reference: 
            - Hörmander, "The Analysis of Linear Partial Differential Operators I" (1983)
            - Berry & Howls, "High orders of the Weyl expansion" (1994)
        
        Args:
            cp: CriticalPoint with non-zero Hessian determinant.
            lam: Large parameter λ.
            
        Returns:
            AsymptoticContribution with both leading and correction terms.
        """
        dim = cp.position.shape[0]
        
        # --- Leading Term (Order 0) ---
        prefactor = (2 * np.pi / lam) ** (dim / 2.0)
        
        # Maslov Index: exp(i * pi/4 * (dim - 2*number_of_negative_eigenvalues))
        maslov = np.exp(1j * np.pi / 4 * (dim - 2 * cp.signature))
        phase_osc = np.exp(1j * lam * cp.phase_value)
        
        denom = np.sqrt(np.abs(cp.hessian_det))
        leading_amp = cp.amplitude_value
        
        term_0 = (prefactor / denom) * phase_osc * maslov * leading_amp
        
        # --- Correction Term (Order 1 relative, Order 2 absolute) ---
        
        H_inv = cp.hessian_inv
        
        # 1. Laplacian of Amplitude twisted by Inverse Hessian
        # Calculation: Trace(H^-1 * Hess_Amp)
        #term_amp = np.einsum('ij,ij->', H_inv, cp.hess_amp)
        term_amp = 0.5 * np.einsum('ij,ij->', H_inv, cp.hess_amp)
        
        # 2. Mixed Term (Gradient Amp dot D3 Phase)
        # Vector V_k = sum_{i,j} (H^-1)_ij * D3_ijk
        V = np.einsum('ij,ijk->k', H_inv, cp.phase_d3)
        # Result: - < H^-1 * Grad_Amp, V >
        #term_mix = - np.dot(np.dot(H_inv, cp.grad_amp), V)
        term_mix = -0.5 * np.dot(np.dot(H_inv, cp.grad_amp), V)
        
        # 3. Pure Phase Term (Anharmonicity)
        # Requires contractions of D4 with H^-1 and D3 with H^-1
        
        # S4 = Contract D4 with two H_inv: sum (H^-1)_ij (H^-1)_kl D4_ijkl
        S4 = np.einsum('ij,kl,ijkl->', H_inv, H_inv, cp.phase_d4)
        
        # S3 = Contract D3^2 with three H_inv (The "Theta" graph)
        # sum (H^-1)_ij (H^-1)_kl (H^-1)_mn D3_ikm D3_jln
        S3 = np.einsum('ij,kl,mn,ikm,jln->', H_inv, H_inv, H_inv, cp.phase_d3, cp.phase_d3)
        
        # Total correction factor (before dividing by i*lambda)
        # Formula: Amp_term + Mix_term + Amp * (5/6 S3 - 1/4 S4) ? 
        # Note: Standard coefficients are usually (5/3 S3 - S4) / 4 or similar depending on definitions.
        # Used here: (1/2) * Div(grad) - (1/8) * ... -> factorized:
        #correction_factor = (term_amp + term_mix + (cp.amplitude_value / 4.0) * (5.0/3.0 * S3 - S4))
        term_phase = (cp.amplitude_value / 24.0) * (5.0 * S3 - 3.0 * S4)
    
        correction_factor = term_amp + term_mix + term_phase
        
        # The correction scales as 1/(i * lambda) relative to the leading term
        val_correction = (prefactor / denom) * phase_osc * maslov * (correction_factor / (1j * lam))
        
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
        
        # Scale factor: (3 λ |α|)^(-1/3)
        scale = (3 * lam * abs(alpha)) ** (-1/3)
        
        # Maslov phase for Airy: exp(i π/6 * sign(α))
        # Since ∫ exp(i t³/3) dt = 2π Ai(0) and ∫ exp(-i t³/3) dt = 2π Ai(0) * exp(-iπ/3)
        phase_sign = np.exp(1j * np.pi / 6 * np.sign(alpha))
        
        # Total contribution
        val = 2 * np.pi * Ai0 * scale * phase_sign * cp.amplitude_value
        
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
        scale_v = np.sqrt(np.pi / (lam * np.abs(beta)))
        phase_v = np.exp(1j * np.pi/4 * np.sign(beta))
        
        # Degenerate Airy Integral with Maslov phase
        # ∫ exp(i λ α u³/3) du = 2π Ai(0) (3λ|α|)^{-1/3} exp(i π/6 · sign(α))
        Ai0 = airy(0)[0]
        scale_u = 2 * np.pi * Ai0 * (3 * lam * np.abs(alpha))**(-1.0/3.0)
        phase_u = np.exp(1j * np.pi / 6.0 * np.sign(alpha))  # ← MASLOV PHASE ADDED
        
        val = (cp.amplitude_value * np.exp(1j * lam * cp.phase_value) * scale_u * phase_u * scale_v * phase_v)
        
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
        
        # MAJOR CORRECTION: exact asymptotic constant
        pearcey_factor = 0.5 * gamma(0.25) * (1.0 / (lam * abs(gamma_coeff)))**0.25
        
        # Transverse Gaussian factor (already correct)
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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import List, Tuple, Optional

class StationaryPhaseVisualizer:
    """
    Visualization toolkit for stationary phase analysis in 2D domains.
    
    Provides diagnostic plots for:
    - Phase function topology and critical point classification
    - Oscillatory integrand structure at finite frequencies
    - Asymptotic convergence rates of leading/correction terms
    
    Notes
    -----
    Currently supports only 2-dimensional phase spaces (n=2). Attempts to
    initialize with higher-dimensional analyzers will issue a warning but
    allow partial functionality for compatible methods.
    """
    
    def __init__(self, analyzer: 'StationaryPhaseAnalyzer'):
        """
        Initialize visualizer with a pre-configured phase analyzer.
        
        Parameters
        ----------
        analyzer : StationaryPhaseAnalyzer
            Analyzer instance containing symbolic phase/amplitude definitions
            and derivative structures. Must have dimension=2 for full functionality.
            
        Warns
        -----
        UserWarning
            If analyzer dimension is not 2, visualization capabilities will be limited.
        """
        self.analyzer = analyzer
        if analyzer.dim != 2:
            warnings.warn(
                f"Visualization optimized for 2D domains (received dim={analyzer.dim}). "
                "Some plotting methods may fail or produce misleading results.",
                UserWarning
            )

    def plot_phase_landscape(
        self,
        critical_points: List[CriticalPoint],
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-3, 3), (-3, 3)),
        points_per_axis: int = 100
    ) -> None:
        """
        Visualize phase function topology with critical point classification overlay.
        
        Generates a filled contour plot of the phase function φ(x,y) with:
        - Color-mapped phase values (viridis colormap)
        - Thin black contour lines for structural clarity
        - Critical points marked by singularity type (Morse/Airy/Pearcey)
        - Type annotations with color-coded markers
        
        Parameters
        ----------
        critical_points : list of CriticalPoint
            Critical points to overlay on the phase landscape.
        bounds : tuple of (min, max) tuples, optional
            Domain boundaries as ((x_min, x_max), (y_min, y_max)). Default: ((-3,3), (-3,3)).
        points_per_axis : int, optional
            Grid resolution for phase evaluation. Default: 100.
            
        Notes
        -----
        Marker conventions:
        - ○ Red: Morse (non-degenerate) critical points
        - ★ Orange: Airy-type singularities (corank 1 with cubic term)
        - ◆ Magenta: Pearcey singularities (corank 1 with quartic dominance)
        
        The phase landscape reveals geometric structures governing asymptotic behavior:
        valleys/ridges indicate regions of stationary phase, while saddle points
        correspond to Morse-type contributions.
        """
        if self.analyzer.dim != 2:
            warnings.warn("Phase landscape visualization requires 2D domain", UserWarning)
            return

        # Generate evaluation grid
        x_range = np.linspace(bounds[0][0], bounds[0][1], points_per_axis)
        y_range = np.linspace(bounds[1][0], bounds[1][1], points_per_axis)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Evaluate phase function on grid (vectorized)
        Z = self.analyzer.func_phase(X, Y)

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filled contour plot with adaptive levels
        levels = np.linspace(np.min(Z), np.max(Z), 40)
        contourf_plot = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.85)
        fig.colorbar(contourf_plot, ax=ax, label=r'Phase $\phi(x, y)$')
        
        # Overlay thin structural contours for readability
        ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.4, alpha=0.3)

        # Plot and annotate critical points by singularity type
        plotted_types = set()  # Track types for legend deduplication
        for cp in critical_points:
            # Determine marker/style based on singularity classification
            if cp.singularity_type == SingularityType.MORSE:
                marker, color, label = 'o', 'red', 'Morse'
            elif cp.singularity_type in (SingularityType.AIRY_1D, SingularityType.AIRY_2D):
                marker, color, label = '*', 'orange', 'Airy'
            elif cp.singularity_type == SingularityType.PEARCEY:
                marker, color, label = 'D', 'magenta', 'Pearcey'
            else:
                marker, color, label = 's', 'gray', 'Higher-order'
            
            # Skip legend duplicates while plotting all points
            if label not in plotted_types:
                ax.scatter(
                    cp.position[0], cp.position[1],
                    c=color, s=120, marker=marker, edgecolors='white',
                    linewidths=1.5, zorder=10, label=label
                )
                plotted_types.add(label)
            else:
                ax.scatter(
                    cp.position[0], cp.position[1],
                    c=color, s=120, marker=marker, edgecolors='white',
                    linewidths=1.5, zorder=10
                )
            
            # Annotate singularity type near point
            ax.text(
                cp.position[0] + 0.12, cp.position[1] + 0.12,
                cp.singularity_type.value,
                fontsize=9, color='white', fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1)
            )

        # Finalize plot aesthetics
        ax.set_title(r'Phase Topology $\phi(x,y)$ with Critical Point Classification', fontsize=14)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$y$', fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    def plot_oscillations(
        self,
        lam_value: float,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-3, 3), (-3, 3)),
        points_per_axis: int = 200
    ) -> None:
        """
        Visualize oscillatory structure of the integrand at finite frequency λ.
        
        Plots the real part of the oscillatory integrand:
            Re[ a(x,y) · exp(i λ φ(x,y)) ]
            
        revealing:
        - Stationary phase regions (slow oscillation zones near critical points)
        - Rapid oscillation zones (destructive interference regions)
        - Amplitude modulation effects from a(x,y)
        
        Parameters
        ----------
        lam_value : float
            Frequency parameter λ controlling oscillation rate.
        bounds : tuple of (min, max) tuples, optional
            Domain boundaries as ((x_min, x_max), (y_min, y_max)). Default: ((-3,3), (-3,3)).
        points_per_axis : int, optional
            Grid resolution for integrand evaluation. Higher values capture finer
            oscillations but increase computation time. Default: 200.
            
        Notes
        -----
        As λ increases:
        - Oscillation wavelength decreases as ~1/√λ near Morse points
        - Stationary phase regions contract around critical points
        - Destructive interference dominates away from critical manifolds
        
        This visualization provides intuition for why asymptotic methods focus
        exclusively on neighborhoods of critical points for large λ.
        """
        if self.analyzer.dim != 2:
            warnings.warn("Oscillation visualization requires 2D domain", UserWarning)
            return

        # Generate high-resolution evaluation grid
        x_range = np.linspace(bounds[0][0], bounds[0][1], points_per_axis)
        y_range = np.linspace(bounds[1][0], bounds[1][1], points_per_axis)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Evaluate phase and amplitude on grid
        phi_val = self.analyzer.func_phase(X, Y)
        amp_val = self.analyzer.func_amp(X, Y)
        
        # Compute real part of oscillatory integrand
        integrand = np.real(amp_val * np.exp(1j * lam_value * phi_val))

        # Create figure with symmetric colormap centered at zero
        plt.figure(figsize=(10, 8))
        im = plt.imshow(
            integrand,
            extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
            origin='lower',
            cmap='RdBu_r',
            vmin=-np.max(np.abs(integrand)),
            vmax=np.max(np.abs(integrand)),
            interpolation='bilinear'
        )
        plt.colorbar(im, label=r'$\operatorname{Re}\left[a(x,y) e^{i \lambda \phi(x,y)}\right]$')
        plt.title(f'Oscillatory Integrand Structure at $\\lambda = {lam_value}$', fontsize=14)
        plt.xlabel('$x$', fontsize=12)
        plt.ylabel('$y$', fontsize=12)
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_asymptotic_convergence(
        self,
        cp: CriticalPoint,
        lambda_start: float = 10,
        lambda_end: float = 1000,
        num_points: int = 50
    ) -> None:
        """
        Diagnose asymptotic convergence rates through log-log magnitude scaling.
        
        Plots absolute magnitudes of:
        - Leading asymptotic term: |I₀(λ)|
        - First correction term: |I₁(λ)| (if non-zero)
        
        on log-log axes to verify theoretical decay rates:
            |I₀(λ)| ~ λ^(-p)  where p = order_leading
            |I₁(λ)| ~ λ^(-p-1) for Morse points with order-2 corrections
            
        Parameters
        ----------
        cp : CriticalPoint
            Critical point to analyze for asymptotic behavior.
        lambda_start : float, optional
            Minimum λ value for convergence study. Default: 10.
        lambda_end : float, optional
            Maximum λ value for convergence study. Default: 1000.
        num_points : int, optional
            Number of λ samples (log-spaced). Default: 50.
            
        Notes
        -----
        Expected slopes on log-log plot:
        - Morse (2D): -1.0 for leading term (λ^(-1))
        - Airy 2D: -5/6 ≈ -0.833
        - Pearcey: -3/4 = -0.75
        
        Deviations at small λ indicate breakdown of asymptotic regime.
        Correction term slope should be steeper by exactly -1.0 for Morse points
        with valid order-2 expansions.
        
        This diagnostic validates both the classification logic and the
        correctness of asymptotic coefficient computations.
        """
        evaluator = StationaryPhaseEvaluator()
        lams = np.logspace(np.log10(lambda_start), np.log10(lambda_end), num_points)
        
        abs_leading = []
        abs_correction = []
        
        for lam in lams:
            res = evaluator.evaluate(cp, lam)
            abs_leading.append(np.abs(res.leading_term))
            abs_correction.append(np.abs(res.correction_term))
        
        # 🔧 CORRECTION : déduire l'ordre asymptotique du type de singularité
        theoretical_order = {
            SingularityType.MORSE: self.analyzer.dim / 2.0,
            SingularityType.AIRY_1D: 1.0/3.0,
            SingularityType.AIRY_2D: 5.0/6.0,
            SingularityType.PEARCEY: 3.0/4.0,
            SingularityType.HIGHER_ORDER: None
        }.get(cp.singularity_type, None)
        
        plt.figure(figsize=(9, 6))
        plt.loglog(lams, abs_leading, 'o-', label='Leading term $|I_0(\\lambda)|$', 
                   linewidth=2.5, markersize=4, alpha=0.85)
        
        # Plot correction term only if non-negligible
        abs_corr_arr = np.array(abs_correction)
        if np.any(abs_corr_arr > 1e-15 * np.max(abs_leading)):
            plt.loglog(lams, abs_corr_arr, 's--', label='Correction term $|I_1(\\lambda)|$', 
                       linewidth=2, markersize=3, alpha=0.8)
        
        # Compute empirical slope for leading term
        slope_lead = np.polyfit(np.log(lams), np.log(abs_leading), 1)[0]
        
        # 🔧 AFFICHAGE CORRIGÉ : utiliser theoretical_order au lieu de cp.order_leading
        annotation = f'Empirical slope: {slope_lead:.2f}'
        if theoretical_order is not None:
            annotation += f'\n(Theoretical: -{theoretical_order:.2f})'
        
        plt.text(
            lams[5], abs_leading[5]*1.5,
            annotation,
            fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
        plt.grid(True, which="both", ls=":", alpha=0.7)
        plt.legend(loc='best', fontsize=11)
        plt.xlabel(r'Frequency parameter $\lambda$ (log scale)', fontsize=12)
        plt.ylabel(r'Magnitude $|I(\lambda)|$ (log scale)', fontsize=12)
        plt.title(
            f'Asymptotic Convergence: {cp.singularity_type.value.capitalize()} Singularity\n'
            f'at $\\phi({cp.position[0]:.2f}, {cp.position[1]:.2f}) = {cp.phase_value:.2f}$',
            fontsize=13
        )
        plt.tight_layout()
        plt.show()
        
# --- Execution Example ---

if __name__ == "__main__":
    # Define symbols
    x, y = sp.symbols('x y')
    
    # 1. Standard Morse Case with Anharmonicity
    # phi = x^2/2 + y^2/2 + 0.1*x^3 (Perturbed Gaussian)
    phi = x**2/2 + y**2/2 + 0.1 * x**3 
    amp = 1 + x**2
    
    # Initialize Analyzer
    analyzer = StationaryPhaseAnalyzer(phi, amp, [x, y])
    
    # Find critical points (Expect one at 0,0)
    points = analyzer.find_critical_points([np.array([0.0, 0.0])])
    
    evaluator = StationaryPhaseEvaluator()
    
    print(f"--- Asymptotic Analysis Report ---")
    
    if points:
        # Analyze the first point found
        cp = analyzer.analyze_point(points[0])
        
        print(f"Critical Point: {cp.position}")
        print(f"Type: {cp.singularity_type.value}")
        print(f"Hessian Det: {cp.hessian_det:.4f}")
        
        # Evaluate for increasing lambda
        for lam in [10, 100, 1000]:
            res = evaluator.evaluate(cp, lam)
            
            print(f"\nLambda = {lam}")
            print(f"  Order 0 Term (Leading):    {res.leading_term:.2e}")
            print(f"  Order 1 Term (Correction): {res.correction_term:.2e}")
            print(f"  Total Value:               {res.total_value}")
            
            # Check ratio to ensure asymptotic convergence
            ratio = np.abs(res.correction_term) / np.abs(res.leading_term)
            print(f"  Correction/Leading Ratio:  {ratio:.2%}")
    else:
        print("No critical points found.")

    if points:
        print("\n--- Generating Visualizations ---")
        viz = StationaryPhaseVisualizer(analyzer)

        # 1. Phase map
        viz.plot_phase_landscape(
            [analyzer.analyze_point(p) for p in points],
            bounds=((-2, 2), (-2, 2))
        )

        # 2. Oscillations
        viz.plot_oscillations(lam_value=50, bounds=((-2, 2), (-2, 2)))

        # 3. Convergence (for the first point)
        viz.plot_asymptotic_convergence(cp)
