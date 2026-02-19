# Copyright 2025
# Fourier Integral Operators (FIO) built on top of asymptotic.py and psiop.py

import sympy as sp
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple
from asymptotic import Analyzer, AsymptoticEvaluator, IntegralMethod, CriticalPoint

class FourierIntegralOperator:
    """
    Evaluates Fourier Integral Operators (FIO) of the form:
        F[u](x) = (2π)^(-n) ∫∫ e^{i φ(x,y,θ)} a(x,y,θ) u(y) dy dθ
        
    This class bridges the gap between symbolic pseudo-differential 
    manipulations and high-frequency asymptotic evaluation.
    """
    
    def __init__(self, phase_expr, amp_expr, vars_x: List[sp.Symbol], 
                 vars_y: List[sp.Symbol], vars_theta: List[sp.Symbol]):
        """
        Initialize the FIO.
        
        Args:
            phase_expr: SymPy expression for the phase function φ(x,y,θ).
            amp_expr: SymPy expression for the amplitude function a(x,y,θ).
            vars_x: List of target spatial variables.
            vars_y: List of source spatial variables.
            vars_theta: List of frequency/phase variables.
        """
        self.phase_expr = sp.sympify(phase_expr)
        self.amp_expr = sp.sympify(amp_expr)

        # self.phase_expr = phase_expr
        # self.amp_expr = amp_expr
        self.vars_x = vars_x if isinstance(vars_x, (list, tuple)) else [vars_x]
        self.vars_y = vars_y if isinstance(vars_y, (list, tuple)) else [vars_y]
        self.vars_theta = vars_theta if isinstance(vars_theta, (list, tuple)) else [vars_theta]
        
        self.dim_x = len(self.vars_x)
        self.dim_y = len(self.vars_y)
        self.dim_theta = len(self.vars_theta)
        
        if self.dim_x != self.dim_y:
            warnings.warn("Dimension of x and y spaces usually match in standard FIOs.")
            
        self._compute_canonical_relation()
        
    def _compute_canonical_relation(self):
        """
        Computes the symbolic derivatives defining the canonical relation C:
        C = { (x, ∇_x φ, y, -∇_y φ) | ∇_θ φ = 0 }
        """
        self.d_theta_phi = [sp.diff(self.phase_expr, th) for th in self.vars_theta]
        self.d_x_phi = [sp.diff(self.phase_expr, x) for x in self.vars_x]
        self.d_y_phi = [sp.diff(self.phase_expr, y) for y in self.vars_y]
        
    def is_non_degenerate(self) -> bool:
        """
        Checks Hörmander's non-degeneracy condition.
        For a phase function to be non-degenerate, the mixed Hessian matrix 
        H_{x, θ} = ∂²φ / ∂x_i ∂θ_j must have maximal rank.
        
        Returns:
            bool: True if the phase is non-degenerate (determinant != 0).
        """
        # Build the mixed Hessian matrix
        H_mixed = sp.Matrix([
            [sp.diff(dth, x) for x in self.vars_x] 
            for dth in self.d_theta_phi
        ])
        
        if H_mixed.shape[0] == H_mixed.shape[1]:
            det = sp.simplify(H_mixed.det())
            return det != 0
        else:
            # If dimensions don't match, we need to check if it has maximal rank.
            # Simplified for square matrices here.
            return False

    def apply_asymptotic(self, u_amp_expr, u_phase_expr, lam_val: float, 
                         x_eval_dict: Dict[sp.Symbol, float], 
                         initial_guesses: Optional[List[np.ndarray]] = None):
        """
        Applies the FIO to a highly oscillatory function:
            u(y) = u_amp(y) * exp(i * λ * u_phase(y))
            
        Evaluates the integral asymptotically for λ → ∞ at a specific point x.
        
        Args:
            u_amp_expr: Amplitude of the input function.
            u_phase_expr: Phase of the input function.
            lam_val: The large parameter λ.
            x_eval_dict: Dictionary mapping x variables to their numerical evaluation point.
            initial_guesses: Initial guesses for the critical points in (y, θ) space.
            
        Returns:
            complex: The asymptotic value of F[u](x).
        """
        # 1. Substitute the evaluation point x into the FIO phase and amplitude
        phi_sub = self.phase_expr.subs(x_eval_dict)
        amp_sub = self.amp_expr.subs(x_eval_dict)
        
        # 2. Construct the total integrand
        # Total integral: ∫ a(x,y,θ) u_amp(y) exp(i * λ * [φ(x,y,θ)/λ + u_phase(y)]) dy dθ
        # Note: Usually, FIOs have a large parameter built into theta. 
        # Here we scale the FIO phase by 1/λ so everything is governed by the same λ.
        total_phase = (phi_sub / lam_val) + u_phase_expr
        total_amp = amp_sub * u_amp_expr
        
        integration_vars = self.vars_y + self.vars_theta
        
        # 3. Instantiate Analyzer to find critical points
        analyzer = Analyzer(total_phase, total_amp, integration_vars)
        
        if initial_guesses is None:
            # Default to origin of (y, θ) space
            initial_guesses = [np.zeros(len(integration_vars))]
            
        pts = analyzer.find_critical_points(initial_guesses)
        
        if not pts:
            warnings.warn("No critical points found. The integral is asymptotically negligible (O(λ^-∞)).")
            return 0j
            
        # 4. Evaluate contributions
        evaluator = AsymptoticEvaluator()
        total_value = 0j
        
        # Pre-factor (2π)^(-n) typical for FIOs, assuming n = dim_theta
        # Note: asymptotic.py already includes (2π/λ)^(N/2) where N is integration dim.
        # We might need to adjust normalization depending on FIO conventions, 
        # but for this generic evaluator, we return the raw asymptotic sum.
        fio_normalization = (2 * np.pi) ** (-self.dim_theta)
        
        for pt in pts:
            cp = analyzer.analyze_point(pt)
            res = evaluator.evaluate(cp, lam_val)
            total_value += res.total_value
            
        return total_value * fio_normalization