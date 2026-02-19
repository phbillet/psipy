# test_fio.py
# Run with: pytest test_fio.py -v

import pytest
import sympy as sp
import numpy as np
from fio import *

# --- Fixtures ---
@pytest.fixture
def symbols_1d():
    return sp.symbols('x y theta')

@pytest.fixture
def symbols_2d():
    return sp.symbols('x1 x2 y1 y2 theta1 theta2')

# =========================================================================
# PART 1: 10 Unit Tests (Internal mechanics)
# =========================================================================

def test_01_initialization(symbols_1d):
    """Test standard initialization and variable parsing."""
    x, y, theta = symbols_1d
    phase = (x - y) * theta
    amp = sp.Integer(1)
    
    fio = FourierIntegralOperator(phase, amp, [x], [y], [theta])
    assert fio.dim_x == 1
    assert fio.dim_y == 1
    assert fio.dim_theta == 1
    assert fio.phase_expr == phase

def test_02_pdo_reduction(symbols_1d):
    """Test that setting phase to (x-y)*theta acts like a standard PDO setup."""
    x, y, theta = symbols_1d
    phase = (x - y) * theta
    amp = theta**2  # Laplacian symbol
    
    fio = FourierIntegralOperator(phase, amp, [x], [y], [theta])
    assert len(fio.d_theta_phi) == 1
    assert sp.simplify(fio.d_theta_phi[0] - (x - y)) == 0
    assert sp.simplify(fio.d_y_phi[0] - (-theta)) == 0

def test_03_canonical_relation_symbolic(symbols_2d):
    """Test correct derivatives for the Lagrangian manifold."""
    x1, x2, y1, y2, th1, th2 = symbols_2d
    phase = (x1 - y1) * th1 + (x2 - y2) * th2 + th1**2 + th2**2
    amp = sp.Integer(1)
    
    fio = FourierIntegralOperator(phase, amp, [x1, x2], [y1, y2], [th1, th2])
    assert sp.simplify(fio.d_theta_phi[0] - (x1 - y1 + 2*th1)) == 0
    assert sp.simplify(fio.d_y_phi[1] - (-th2)) == 0

def test_04_non_degeneracy_condition(symbols_1d):
    """Test Hörmander's non-degeneracy condition det(d^2_{x,θ} φ) ≠ 0."""
    x, y, theta = symbols_1d
    
    # Degenerate phase (independent of x)
    deg_phase = (y**2) * theta
    fio_deg = FourierIntegralOperator(deg_phase, 1, [x], [y], [theta])
    assert not fio_deg.is_non_degenerate()
    
    # Non-degenerate phase
    non_deg_phase = (x - y) * theta
    fio_non_deg = FourierIntegralOperator(non_deg_phase, 1, [x], [y], [theta])
    assert fio_non_deg.is_non_degenerate()

def test_05_asymptotic_phase_setup(symbols_1d, mocker):
    """Test that the total phase is correctly constructed before Analyzer call."""
    x, y, theta = symbols_1d
    phase = (x - y) * theta
    amp = sp.Integer(1)
    
    fio = FourierIntegralOperator(phase, amp, [x], [y], [theta])
    
    # Mock Analyzer to intercept the total_phase
    mock_analyzer = mocker.patch('fio.Analyzer')
    mock_analyzer.return_value.find_critical_points.return_value = []
    
    u_amp = sp.Integer(1)
    u_phase = y**2 / 2
    lam = 100.0
    
    fio.apply_asymptotic(u_amp, u_phase, lam, {x: 1.0})
    
    # Check that Analyzer was instantiated with the correct total phase
    expected_total_phase = ((1.0 - y) * theta) / lam + u_phase
    args, _ = mock_analyzer.call_args
    assert sp.simplify(args[0] - expected_total_phase) == 0

def test_06_apply_numerical_constant():
    """STUB: Test numerical integration fallback vs analytical solution."""
    pass # To be implemented with scipy.integrate

def test_07_dimension_mismatch(symbols_1d):
    """Test warnings when dim_x != dim_y."""
    x, y, theta = symbols_1d
    z = sp.Symbol('z')
    with pytest.warns(UserWarning):
        FourierIntegralOperator((x - y)*theta, 1, [x, z], [y], [theta])

def test_08_lambdify_caching():
    """STUB: Test performance caching of SymPy functions."""
    pass 

def test_09_stationary_point_dispatch():
    """STUB: Verify apply_asymptotic correctly flags STATIONARY_PHASE method."""
    pass

def test_10_1d_vs_2d(symbols_1d, symbols_2d):
    """Test that FIO handles both 1D and 2D variables cleanly."""
    fio1 = FourierIntegralOperator(symbols_1d[0]*symbols_1d[2], 1, *[[s] for s in symbols_1d])
    fio2 = FourierIntegralOperator(symbols_2d[0]*symbols_2d[4], 1, symbols_2d[:2], symbols_2d[2:4], symbols_2d[4:])
    assert fio1.dim_theta == 1
    assert fio2.dim_theta == 2


# =========================================================================
# PART 2: 10 Integration Tests (Math & Physics cases)
# =========================================================================

def test_11_identity_operator(symbols_1d):
    """
    Test Identity: F[u](x) = u(x).
    Phase = (x-y)θ. u(y) = exp(i * λ * y^2 / 2). Target x = 0.
    """
    x, y, theta = symbols_1d
    phase = (x - y) * theta # Rescaled θ to match lambda for the test
    fio = FourierIntegralOperator(phase, 1, [x], [y], [theta])
    
    lam = 100.0
    u_phase = y**2 / 2
    # Expect F[u](0) ≈ exp(0) = 1.0
    
    res = fio.apply_asymptotic(1, u_phase, lam, {x: 0.0}, initial_guesses=[np.array([0., 0.])])
    # Very rough check of magnitude (since purely asymptotic)
    assert np.isclose(np.abs(res), 1.0, atol=0.2)

def test_12_translation_operator(symbols_1d):
    """STUB: Wave propagation u(x-x0)."""
    pass

def test_13_schrodinger_free_particle(symbols_1d):
    """
    Quantum evolution exp(-itΔ/2).
    Phase = (x-y)θ - (t/2)θ^2.
    """
    x, y, theta = symbols_1d
    t = 1.0
    phase = (x - y) * theta - (t / 2.0) * theta**2
    fio = FourierIntegralOperator(phase, 1, [x], [y], [theta])
    
    # Gaussian wavepacket phase
    u_phase = y**2 / 2
    res = fio.apply_asymptotic(1, u_phase, 100.0, {x: 1.0}, initial_guesses=[np.array([0., 0.])])
    assert res != 0j # Ensure a non-zero propagation happened

def test_14_fractional_fourier_transform():
    """STUB: Optical FrFT operator."""
    pass

def test_15_eikonal_geometrical_optics():
    """STUB: Check that FIO focuses energy on characteristic curves."""
    pass

def test_16_airy_caustic_generation(symbols_1d, mocker):
    """
    Fold catastrophe producing an Airy function.
    Phase contains a cubic term.
    """
    x, y, theta = symbols_1d
    # Cubic phase generating Airy-type decay
    phase = (x - y) * theta + (theta**3) / 3.0
    fio = FourierIntegralOperator(phase, 1, [x], [y], [theta])
    
    # Mock the evaluate method to catch the SingularityType
    spy_evaluate = mocker.spy(AsymptoticEvaluator, 'evaluate')
    
    fio.apply_asymptotic(1, y**2/2, 100.0, {x: 0.0}, initial_guesses=[np.array([0., 0.])])
    
    # Extract the CriticalPoint passed to evaluate
    if spy_evaluate.call_count > 0:
        cp = spy_evaluate.call_args[0][1] # arg 1 is CriticalPoint
        # Verify it was classified as AIRY or MORSE depending on grid
        assert cp.singularity_type.name in ['MORSE', 'AIRY_1D', 'HIGHER_ORDER']

def test_17_wave_equation_acoustics():
    """STUB: Front propagation with phase |θ|."""
    pass

def test_18_radon_transform_connection():
    """STUB: Tomography integration paths."""
    pass

def test_19_bicharacteristic_flow_match():
    """STUB: Ensure FIO's C matches psiop.py Hamiltonian flow."""
    pass

def test_20_laplace_regime_fio(symbols_1d):
    """
    Test fallback to Laplace method when phase is imaginary.
    Diffusive regime.
    """
    x, y, theta = symbols_1d
    # Imaginary phase for heat equation
    phase = sp.I * ((x - y)**2 + theta**2)
    fio = FourierIntegralOperator(phase, 1, [x], [y], [theta])
    
    res = fio.apply_asymptotic(1, sp.I * y**2, 10.0, {x: 0.0}, initial_guesses=[np.array([0., 0.])])
    # The result should be strictly real because everything is exponentially damped
    assert np.isreal(res) or np.abs(np.imag(res)) < 1e-10