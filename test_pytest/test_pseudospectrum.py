import pytest
from psiop import *

class TestPseudospectrumAnalysis:
    """Test suite for pseudospectrum analysis with known spectra."""
    
    def setup_method(self):
        """Setup common test parameters."""
        self.x, self.xi = symbols('x xi', real=True)
        self.tolerance = 1e-6
        self.N = 32  # Small N for fast tests
        
    # ========================================================================
    # TEST 1: Laplacian (corrected)
    # ========================================================================
    def test_laplacian_periodic_spectrum(self):
        """
        Test: -∂²/∂x² on [0, 2π) with periodic BC
        
        Known spectrum: λ_n = n² for n = 0, 1, 2, ..., N-1
        where k_n = n (frequency modes)
        
        For domain [0, 2π) with N points:
        - dx = 2π/N
        - k_n = n for n in [0, N-1]
        - λ_n = k_n² = n²
        """
        symbol = self.xi**2
        op = PseudoDifferentialOperator(symbol, [self.x], mode='symbol')
        
        # Domain [0, 2π) to get integer frequencies k_n = n
        L = np.pi  # Half-period
        x_grid = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        
        result = op.pseudospectrum_analysis(
            x_grid=x_grid,
            lambda_real_range=(-5, self.N**2 + 10),
            lambda_imag_range=(-2, 2),
            resolution=50,
            method='spectral',
            L=np.pi,  # Specify domain half-length
            N=self.N,
            parallel=False,
            plot=False
        )
        
        eigenvalues = result['eigenvalues']
        
        # Check: All eigenvalues should be real
        max_imag = np.max(np.abs(eigenvalues.imag))
        assert max_imag < 1e-10, \
            f"Laplacian should have purely real eigenvalues (max imag = {max_imag})"
        
        # Check: Eigenvalues should match n² pattern
        eigs_sorted = np.sort(eigenvalues.real)
        
        # For FFT with N points, frequencies are [0, 1, 2, ..., N/2-1, -N/2, ..., -1]
        # After squaring: [0, 1, 4, 9, ..., (N/2)²] repeated (positive and negative freqs)
        n_half = self.N // 2
        expected_unique = sorted(set([n**2 for n in range(-n_half, n_half)]))
        
        # Compare first few eigenvalues (allowing duplicates for ±n)
        # The spectrum should contain: 0 (once), 1 (twice for ±1), 4 (twice), etc.
        assert np.abs(eigs_sorted[0]) < 1e-10, \
            f"First eigenvalue should be ~0, got {eigs_sorted[0]}"
        
        # Check second eigenvalue is close to 1
        assert np.abs(eigs_sorted[1] - 1.0) < 0.1, \
            f"Second eigenvalue should be ~1, got {eigs_sorted[1]}"
        
        # Check that eigenvalues grow roughly quadratically
        # For large n, λ_n ≈ n²
        assert eigs_sorted[-1] > (self.N/2)**2 * 0.5, \
            f"Largest eigenvalue should be ~(N/2)² = {(self.N/2)**2}, got {eigs_sorted[-1]}"
    
    # ========================================================================
    # TEST 2: Identity operator (simplest case)
    # ========================================================================
    def test_scaled_identity(self):
        """
        Test: Constant symbol p(x,ξ) = c (identity operator)
        
        Known spectrum: All eigenvalues = c
        """
        c = 5.0
        symbol = c + 0*self.xi  # Constant symbol
        op = PseudoDifferentialOperator(symbol, [self.x], mode='symbol')
        
        x_grid = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        
        result = op.pseudospectrum_analysis(
            x_grid=x_grid,
            lambda_real_range=(c-2, c+2),
            lambda_imag_range=(-1, 1),
            resolution=40,
            method='spectral',
            parallel=False,
            plot=False
        )
        
        eigenvalues = result['eigenvalues']
        
        # Check: All eigenvalues should equal c
        assert np.allclose(eigenvalues.real, c, atol=1e-6), \
            f"Identity operator: all eigenvalues should be {c}, got range [{eigenvalues.real.min()}, {eigenvalues.real.max()}]"
        
        assert np.max(np.abs(eigenvalues.imag)) < 1e-10, \
            "Identity operator should have real eigenvalues"
    
    # ========================================================================
    # TEST 3: Linear symbol (first-order derivative)
    # ========================================================================
    def test_first_order_derivative(self):
        """
        Test: First derivative i∂/∂x (anti-Hermitian)
        
        Symbol: p(x,ξ) = iξ
        Known spectrum: λ_n = i·k_n where k_n ∈ [-N/2, N/2]
        
        Expected: Purely imaginary eigenvalues
        """
        symbol = I * self.xi
        op = PseudoDifferentialOperator(symbol, [self.x], mode='symbol')
        
        x_grid = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        
        result = op.pseudospectrum_analysis(
            x_grid=x_grid,
            lambda_real_range=(-2, 2),
            lambda_imag_range=(-self.N//2 - 5, self.N//2 + 5),
            resolution=50,
            method='spectral',
            parallel=False,
            plot=False
        )
        
        eigenvalues = result['eigenvalues']
        
        # Check: Eigenvalues should be purely imaginary
        max_real = np.max(np.abs(eigenvalues.real))
        assert max_real < 1e-8, \
            f"Anti-Hermitian operator should have purely imaginary eigenvalues (max real = {max_real})"
        
        # Check: Range of imaginary parts
        imag_min, imag_max = eigenvalues.imag.min(), eigenvalues.imag.max()
        assert imag_min < -5 and imag_max > 5, \
            f"Imaginary eigenvalues should span positive and negative values, got [{imag_min}, {imag_max}]"
    
    # ========================================================================
    # TEST 4: Harmonic oscillator (simplified check)
    # ========================================================================
    def test_harmonic_oscillator_positivity(self):
        """
        Test: Harmonic oscillator H = ξ² + x²
        
        This is positive definite, so all eigenvalues should be positive.
        We don't check exact values due to discretization.
        """
        symbol = self.xi**2 + self.x**2
        op = PseudoDifferentialOperator(symbol, [self.x], mode='symbol')
        
        x_grid = np.linspace(-5, 5, self.N, endpoint=False)
        
        result = op.pseudospectrum_analysis(
            x_grid=x_grid,
            lambda_real_range=(-2, 100),
            lambda_imag_range=(-2, 2),
            resolution=50,
            method='spectral',
            parallel=False,
            plot=False
        )
        
        eigenvalues = result['eigenvalues']
        
        # Check: All eigenvalues should be positive (positive definite operator)
        assert np.min(eigenvalues.real) > -0.1, \
            f"Harmonic oscillator should have positive eigenvalues, got min = {np.min(eigenvalues.real)}"
        
        # Check: Self-adjoint (real eigenvalues)
        assert np.max(np.abs(eigenvalues.imag)) < 1e-8, \
            "Harmonic oscillator should have real eigenvalues"
    
    # ========================================================================
    # TEST 5: Convection-diffusion (non-normal)
    # ========================================================================
#    def test_convection_diffusion_dissipation(self):
#        """
#        Test: Convection-diffusion -iν∂_x - ν∂²_x
#        
#        Symbol: p(x,ξ) = -ic·ξ - ν·ξ²
#        
#        Expected: 
#        - Negative real parts (dissipation from diffusion)
#       - Large pseudospectrum (non-normality)
#        """
#        c = 5.0   # Convection
#        nu = 0.5  # Diffusion
#       
#        symbol = -I*c*self.xi - nu*self.xi**2
#        op = PseudoDifferentialOperator(symbol, [self.x], mode='symbol')
#       
#        x_grid = np.linspace(0, 2*np.pi, self.N, endpoint=False)
#       
#        result = op.pseudospectrum_analysis(
#            x_grid=x_grid,
#            lambda_real_range=(-50, 10),
#            lambda_imag_range=(-30, 30),
#            resolution=60,
#            method='finite_difference',
#            parallel=False,
#            plot=False
#        )
#       
#        eigenvalues = result['eigenvalues']
#       
#        # Check: All eigenvalues should have non-positive real part (dissipation)
#        max_real = np.max(eigenvalues.real)
#        assert max_real < 1.0, \
#            f"Dissipative operator should have Re(λ) ≤ 0, got max = {max_real}"
#        
#        # Check: Non-trivial imaginary parts (from convection)
#        imag_range = np.max(eigenvalues.imag) - np.min(eigenvalues.imag)
#        assert imag_range > 5.0, \
#            f"Convection should produce spread in Im(λ), got range = {imag_range}"
#       
#        # Check: Non-normality (large resolvent norms)
#        resolvent_norm = result['resolvent_norm']
#        max_resolvent = np.nanmax(resolvent_norm)
#        
#        assert max_resolvent > 50, \
#            f"Non-normal operator should have large resolvent norm (got {max_resolvent})"
    
    # ========================================================================
    # TEST 6: Matrix consistency
    # ========================================================================
    def test_matrix_dimensions(self):
        """Test that operator matrix has correct dimensions."""
        symbol = self.xi**2
        op = PseudoDifferentialOperator(symbol, [self.x], mode='symbol')
        
        N_test = 48
        x_grid = np.linspace(0, 2*np.pi, N_test, endpoint=False)
        
        result = op.pseudospectrum_analysis(
            x_grid=x_grid,
            lambda_real_range=(0, 50),
            lambda_imag_range=(-2, 2),
            resolution=30,
            method='spectral',
            plot=False
        )
        
        H = result['operator_matrix']
        
        assert H.shape == (N_test, N_test), \
            f"Operator matrix should be {N_test}×{N_test}, got {H.shape}"
        
        assert len(result['eigenvalues']) == N_test, \
            f"Should have {N_test} eigenvalues, got {len(result['eigenvalues'])}"
    
    # ========================================================================
    # TEST 7: Self-adjointness detection
    # ========================================================================
    def test_self_adjoint_real_spectrum(self):
        """
        Test that self-adjoint operators have real spectra.
        
        Test several self-adjoint operators:
        - ξ² (Laplacian)
        - ξ² + c (shifted Laplacian)
        - ξ⁴ (biharmonic)
        """
        test_cases = [
            (self.xi**2, "Laplacian"),
            (self.xi**2 + 5, "Shifted Laplacian"),
            (self.xi**4, "Biharmonic")
        ]
        
        x_grid = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        
        for symbol, name in test_cases:
            op = PseudoDifferentialOperator(symbol, [self.x], mode='symbol')
            
            result = op.pseudospectrum_analysis(
                x_grid=x_grid,
                lambda_real_range=(-10, 200),
                lambda_imag_range=(-5, 5),
                resolution=30,
                method='spectral',
                parallel=False,
                plot=False
            )
            
            eigenvalues = result['eigenvalues']
            max_imag = np.max(np.abs(eigenvalues.imag))
            
            assert max_imag < 1e-8, \
                f"{name} should have real spectrum, got max |Im(λ)| = {max_imag}"


# ============================================================================
# Parametrized tests
# ============================================================================
@pytest.mark.parametrize("N", [16, 32, 64])
def test_eigenvalue_count(N):
    """Test that we get N eigenvalues for N grid points."""
    x, xi = symbols('x xi', real=True)
    symbol = xi**2
    op = PseudoDifferentialOperator(symbol, [x], mode='symbol')
    
    x_grid = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    result = op.pseudospectrum_analysis(
        x_grid=x_grid,
        lambda_real_range=(0, N**2),
        lambda_imag_range=(-1, 1),
        resolution=30,
        method='spectral',
        plot=False
    )
    
    assert len(result['eigenvalues']) == N, \
        f"Expected {N} eigenvalues, got {len(result['eigenvalues'])}"


@pytest.mark.parametrize("method", ['spectral', 'finite_difference'])
def test_methods_give_real_spectrum_for_laplacian(method):
    """Test that both methods recognize Laplacian as self-adjoint."""
    x, xi = symbols('x xi', real=True)
    symbol = xi**2
    op = PseudoDifferentialOperator(symbol, [x], mode='symbol')
    
    x_grid = np.linspace(0, 2*np.pi, 32, endpoint=False)
    
    result = op.pseudospectrum_analysis(
        x_grid=x_grid,
        lambda_real_range=(0, 50),
        lambda_imag_range=(-2, 2),
        resolution=30,
        method=method,
        plot=False
    )
    
    eigenvalues = result['eigenvalues']
    max_imag = np.max(np.abs(eigenvalues.imag))
    
    assert max_imag < 1e-6, \
        f"Method '{method}' should give real spectrum for Laplacian, got max |Im| = {max_imag}"


# ============================================================================
# Run tests
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])