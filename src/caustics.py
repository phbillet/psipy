from imports import *
# ==================================================================
# CAUSTIC DETECTION AND CLASSIFICATION
# ==================================================================

class CausticDetector:
    """
    Detect and classify caustics in ray families.
    
    Caustic types:
    - Fold (A2): Generic 1-parameter family, corrected by Airy function
    - Cusp (A3): Generic 2-parameter family, corrected by Pearcey function
    - Swallowtail (A4): More degenerate, requires higher corrections
    """
    
    def __init__(self, rays, dimension):
        self.rays = rays
        self.dimension = dimension
        self.caustics = []
    
    def detect_caustics(self, threshold=1e-3):
        """
        Detect caustics by analyzing Jacobian of ray mapping.
        
        Caustic occurs when ∂(x,y)/∂(ray_param, t) = 0
        """
        print("Detecting caustics...")
        
        for i, ray in enumerate(self.rays):
            if self.dimension == 1:
                caustics_1d = self._detect_1d_caustics(ray, i)
                self.caustics.extend(caustics_1d)
            else:
                caustics_2d = self._detect_2d_caustics(ray, i, threshold)
                self.caustics.extend(caustics_2d)
        
        print(f"Found {len(self.caustics)} caustic points")
        return self.caustics
    
    def _detect_1d_caustics(self, ray, ray_idx):
        """
        In 1D, caustic occurs when dx/dt = 0 (ray turns around).
        """
        caustics = []
        
        x = ray['x']
        t = ray['t']
        
        # Compute velocity
        dxdt = np.gradient(x, t)
        
        # Find sign changes (turning points)
        sign_changes = np.where(np.diff(np.sign(dxdt)))[0]
        
        for idx in sign_changes:
            caustics.append({
                'type': 'fold',  # 1D caustics are always folds
                'ray_idx': ray_idx,
                'time_idx': idx,
                'position': x[idx],
                'time': t[idx],
                'caustic_type': 'A2'
            })
        
        return caustics
    
    def _detect_2d_caustics(self, ray, ray_idx, threshold):
        """
        In 2D, caustic occurs when det(Jacobian) ≈ 0.
        Classify type by eigenvalues of Hessian.
        """
        caustics = []
        
        x = ray['x']
        y = ray['y']
        xi = ray['xi']
        eta = ray['eta']
        t = ray['t']
        
        # Numerical Jacobian along ray
        # J = [[∂x/∂t, ∂x/∂s], [∂y/∂t, ∂y/∂s]]
        # Approximate using neighboring rays (would need full family)
        
        # Simpler criterion: momentum magnitude
        p_mag = np.sqrt(xi**2 + eta**2)
        
        # Look for near-zero momentum (approximate caustic indicator)
        near_zero = np.where(p_mag < threshold)[0]
        
        for idx in near_zero:
            # Classify caustic type by analyzing trajectory curvature
            if idx > 0 and idx < len(t) - 1:
                # Second derivatives
                d2x = x[idx+1] - 2*x[idx] + x[idx-1]
                d2y = y[idx+1] - 2*y[idx] + y[idx-1]
                curvature = np.sqrt(d2x**2 + d2y**2)
                
                # Simple classification
                if curvature < 0.1:
                    caustic_type = 'A2'  # Fold
                    correction_type = 'airy'
                else:
                    caustic_type = 'A3'  # Cusp
                    correction_type = 'pearcey'
                
                caustics.append({
                    'type': correction_type,
                    'ray_idx': ray_idx,
                    'time_idx': idx,
                    'position': (x[idx], y[idx]),
                    'time': t[idx],
                    'caustic_type': caustic_type,
                    'curvature': curvature
                })
        
        return caustics
    
    def compute_maslov_index(self, ray):
        """
        Compute Maslov index: number of caustics crossed × π/2.
        
        The Maslov index accumulates phase jumps at caustics.
        """
        maslov = 0
        
        if self.dimension == 1:
            x = ray['x']
            dxdt = np.gradient(x, ray['t'])
            # Count sign changes
            maslov = len(np.where(np.diff(np.sign(dxdt)))[0])
        else:
            # In 2D, need to track conjugate points
            # Simplified: count momentum near-zeros
            xi, eta = ray['xi'], ray['eta']
            p_mag = np.sqrt(xi**2 + eta**2)
            maslov = len(np.where(p_mag < 0.01)[0])
        
        return maslov * np.pi / 2


# ==================================================================
# SPECIAL FUNCTIONS FOR CAUSTIC CORRECTIONS
# ==================================================================

class CausticFunctions:
    """
    Special functions for caustic corrections.
    """
    
    @staticmethod
    def airy_uniform(z):
        """
        Airy function Ai(z) for fold caustic correction.
        
        Near a fold caustic, the WKB solution is replaced by:
        u(x) ≈ A(x) · Ai((x-x_c)/ε^{2/3}) · exp(iS(x)/ε)
        """
        return airy(z)[0]
    
    @staticmethod
    def airy_derivative(z):
        """
        Derivative of Airy function Ai'(z).
        """
        return airy(z)[1]
    
    @staticmethod
    def pearcey_integral(x, y):
        """
        Pearcey integral for cusp caustic (A3 singularity).
        
        P(x,y) = ∫_{-∞}^{∞} exp(i(t^4 + xt^2 + yt)) dt
        
        This is more complex and typically requires numerical integration.
        Simplified implementation using stationary phase.
        """
        # Number of integration points
        n_pts = 200
        t = np.linspace(-5, 5, n_pts)
        dt = t[1] - t[0]
        
        # Phase function: φ(t) = t^4 + x*t^2 + y*t
        phase = t**4 + x * t**2 + y * t
        
        # Numerical integration
        integrand = np.exp(1j * phase)
        result = np.trapz(integrand, dx=dt)
        
        return result
    
    @staticmethod
    def pearcey_approx(x, y):
        """
        Approximate Pearcey function using asymptotic expansion.
        Faster but less accurate than full integration.
        """
        # Asymptotic form for large |x|, |y|
        r = np.sqrt(x**2 + y**2) + 1e-10
        
        if r > 2:
            # Asymptotic expansion
            return np.exp(1j * (x**2/4 + y**2/(4*x))) / np.sqrt(r)
        else:
            # Fall back to numerical
            return CausticFunctions.pearcey_integral(x, y)
    
    @staticmethod
    def maslov_phase_shift(n_caustics):
        """
        Phase shift from Maslov index.
        
        Each caustic crossed adds π/2 to the phase.
        """
        return n_caustics * np.pi / 2