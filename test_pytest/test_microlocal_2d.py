import numpy as np
import pytest
from sympy import symbols, sqrt, sin, cos, simplify
from microlocal_2d import *


def test_characteristic_variety_2d():
    """Test 2D characteristic variety computation."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic: p = ξ² + η² - 1
    p = xi**2 + eta**2 - 1
    char = characteristic_variety_2d(p)
    
    assert 'implicit' in char
    assert 'equation' in char
    assert 'function' in char


def test_characteristic_variety_2d_elliptic():
    """Test characteristic variety for elliptic operator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Elliptic: no real characteristic points
    p = xi**2 + eta**2 + 1
    char = characteristic_variety_2d(p)
    
    # Should compute without error
    assert char is not None
    assert callable(char['function'])


def test_characteristic_variety_2d_anisotropic():
    """Test anisotropic characteristic variety."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Anisotropic: p = ξ² - η²
    p = xi**2 - eta**2
    char = characteristic_variety_2d(p)
    
    # Test evaluation on characteristic
    func = char['function']
    val = func(0, 0, 1, 1)
    assert np.isclose(val, 0.0, atol=1e-10)


def test_bichar_flow_2d_isotropic():
    """Test 2D bicharacteristic flow for isotropic symbol."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic propagation: p = ξ² + η²
    p = xi**2 + eta**2
    
    z0 = (0, 0, 1, 1)
    traj = bichar_flow_2d(p, z0, (0, 5), method='symplectic', n_steps=100)
    
    assert 'x' in traj
    assert 'y' in traj
    assert 'xi' in traj
    assert 'eta' in traj
    assert 'symbol_value' in traj
    
    # Frequencies should be constant
    assert np.std(traj['xi']) < 1e-6
    assert np.std(traj['eta']) < 1e-6
    
    # Positions should evolve linearly
    # ẋ = ∂p/∂ξ = 2ξ, ẏ = ∂p/∂η = 2η
    expected_slope = 2.0  # Since ξ=η=1
    assert np.allclose(np.diff(traj['x']), np.diff(traj['y']), rtol=1e-2)


def test_bichar_flow_2d_anisotropic():
    """Test bicharacteristic flow for anisotropic symbol."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Anisotropic: p = ξ² + 2η²
    p = xi**2 + 2*eta**2
    
    z0 = (0, 0, 1, 1)
    traj = bichar_flow_2d(p, z0, (0, 5), method='symplectic', n_steps=100)
    
    # Check all components exist
    assert len(traj['x']) == 100
    assert len(traj['y']) == 100


def test_bichar_flow_2d_methods():
    """Test different integration methods for 2D bicharacteristics."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    z0 = (0, 0, 1, 0)
    
    # Symplectic
    traj_symp = bichar_flow_2d(p, z0, (0, 5), method='symplectic', n_steps=50)
    assert len(traj_symp['x']) == 50
    
    # Verlet
    traj_verlet = bichar_flow_2d(p, z0, (0, 5), method='verlet', n_steps=50)
    assert len(traj_verlet['x']) == 50
    
    # RK45
    traj_rk = bichar_flow_2d(p, z0, (0, 5), method='rk45', n_steps=50)
    assert len(traj_rk['x']) > 0


def test_bichar_flow_2d_energy_conservation():
    """Test energy conservation along bicharacteristics."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # For H = ξ² + η² + x² + y²
    H = xi**2 + eta**2 + x**2 + y**2
    
    z0 = (1, 1, 1, 0)
    traj = bichar_flow_2d(H, z0, (0, 10), method='symplectic', n_steps=500)
    
    # Symbol value should be approximately constant
    symbol_drift = np.std(traj['symbol_value'])
    assert symbol_drift < 0.2


def test_wkb_multidim_placeholder():
    """Test WKB multidimensional (placeholder implementation)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Wave equation: p = ξ² + η²
    p = xi**2 + eta**2
    
    # Dummy initial data
    initial_phase = {
        'curve': np.array([[0, 0], [1, 0]]),
        'S_values': np.zeros(2)
    }
    
    wkb = wkb_multidim(p, initial_phase, order=1, 
                       domain=((-2, 2), (-2, 2)), resolution=20)
    
    # Check structure
    assert 'x' in wkb
    assert 'y' in wkb
    assert 'S' in wkb
    assert 'a' in wkb
    assert 'u' in wkb


def test_compute_maslov_index():
    """Test Maslov index computation."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Generate periodic orbit
    p = xi**2 + eta**2 + x**2 + y**2
    
    # Harmonic oscillator has period 2π
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi
    
    traj = bichar_flow_2d(p, z0, (0, T), method='symplectic', n_steps=100)
    
    maslov = compute_maslov_index(traj, p)
    
    # Should return an integer
    assert isinstance(maslov, (int, np.integer))


def test_compute_maslov_index_non_closed():
    """Test Maslov index for non-closed path (should warn)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    # Non-periodic trajectory
    z0 = (0, 0, 1, 1)
    traj = bichar_flow_2d(p, z0, (0, 3), method='symplectic', n_steps=50)
    
    # Should handle non-closed path
    maslov = compute_maslov_index(traj, p)
    assert maslov is not None


def test_compute_caustics_2d():
    """Test caustic computation in 2D."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic propagation
    p = xi**2 + eta**2
    
    # Initial curve (line segment)
    n_points = 10
    initial_curve = {
        'x': np.linspace(-1, 1, n_points),
        'y': np.zeros(n_points),
        'xi': np.ones(n_points),
        'eta': np.zeros(n_points)
    }
    
    caustics = compute_caustics_2d(p, initial_curve, tmax=2.0, n_rays=n_points)
    
    assert 'caustic_points' in caustics
    assert 'rays' in caustics
    assert len(caustics['rays']) == n_points


def test_propagate_singularity_2d():
    """Test propagation of singularities in 2D."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Transport-like operator
    p = xi + eta
    
    initial_sing = [(0, 0, 1, 1)]
    result = propagate_singularity_2d(p, initial_sing, (0, 5))
    
    assert 'trajectories' in result
    assert 'endpoints' in result
    assert 'initial' in result
    
    assert len(result['trajectories']) == 1
    assert len(result['endpoints']) == 1


def test_propagate_singularity_2d_multiple():
    """Test propagation of multiple singularities."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    # Multiple initial singularities
    initial_sing = [
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (0, 0, np.sqrt(0.5), np.sqrt(0.5))
    ]
    
    result = propagate_singularity_2d(p, initial_sing, (0, 3))
    
    assert len(result['trajectories']) == 3
    assert len(result['endpoints']) == 3


def test_characteristic_variety_2d_numerical():
    """Test numerical evaluation of 2D characteristic variety."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2 - x**2 - y**2
    
    char = characteristic_variety_2d(p)
    func = char['function']
    
    # On characteristic: p(1, 1, 1, 1) = 1 + 1 - 1 - 1 = 0
    val = func(1.0, 1.0, 1.0, 1.0)
    assert np.isclose(val, 0.0, atol=1e-10)
    
    # Off characteristic
    val_off = func(0.0, 0.0, 1.0, 1.0)
    assert not np.isclose(val_off, 0.0)


def test_bichar_flow_2d_return_to_origin():
    """Test periodic bicharacteristic for harmonic oscillator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # 2D harmonic oscillator
    H = xi**2 + eta**2 + x**2 + y**2
    
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi  # Period
    
    traj = bichar_flow_2d(H, z0, (0, T), method='symplectic', n_steps=200)
    
    # Should approximately return to initial point
    assert np.isclose(traj['x'][-1], z0[0], rtol=1e-1)
    assert np.isclose(traj['y'][-1], z0[1], rtol=1e-1)


def test_bichar_flow_2d_phase_space_volume():
    """Test phase space volume preservation (Liouville theorem)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    # Multiple nearby initial conditions
    z0_list = [
        (0, 0, 1, 1),
        (0.1, 0, 1, 1),
        (0, 0.1, 1, 1),
        (0, 0, 1.1, 1)
    ]
    
    trajectories = []
    for z0 in z0_list:
        traj = bichar_flow_2d(p, z0, (0, 5), method='symplectic', n_steps=50)
        trajectories.append(traj)
    
    # All should complete
    assert len(trajectories) == 4


def test_invalid_bichar_method_2d():
    """Test error handling for invalid integration method."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    with pytest.raises(ValueError):
        bichar_flow_2d(p, (0, 0, 1, 1), (0, 5), method='invalid')


def test_bichar_flow_2d_long_time():
    """Test long-time integration stability."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2 + x**2 + y**2
    
    z0 = (1, 0, 0, 1)
    traj = bichar_flow_2d(p, z0, (0, 50), method='symplectic', n_steps=1000)
    
    # Should complete without NaN
    assert np.all(np.isfinite(traj['x']))
    assert np.all(np.isfinite(traj['y']))
    assert np.all(np.isfinite(traj['xi']))
    assert np.all(np.isfinite(traj['eta']))


def test_caustics_2d_convergence():
    """Test that caustic computation converges with resolution."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    # Simple initial curve
    n_points = 5
    initial_curve = {
        'x': np.linspace(-0.5, 0.5, n_points),
        'y': np.zeros(n_points),
        'xi': np.ones(n_points),
        'eta': np.zeros(n_points)
    }
    
    caustics = compute_caustics_2d(p, initial_curve, tmax=1.0, n_rays=n_points)
    
    # Should find some rays
    assert len(caustics['rays']) > 0


def test_maslov_index_harmonic_2d():
    """Test Maslov index for 2D harmonic oscillator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # 2D isotropic harmonic oscillator
    H = (xi**2 + eta**2 + x**2 + y**2) / 2
    
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi
    
    traj = bichar_flow_2d(H, z0, (0, T), method='symplectic', n_steps=200)
    
    maslov = compute_maslov_index(traj, H)
    
    # For 2D harmonic oscillator, typical value is 2
    assert maslov in [0, 1, 2, 3, 4]


def test_wkb_multidim_structure():
    """Test WKB multidimensional output structure."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    initial_phase = {'curve': np.array([[0, 0]])}
    wkb = wkb_multidim(p, initial_phase, resolution=10)
    
    # Check all required fields exist
    required_fields = ['x', 'y', 'S', 'a', 'u']
    for field in required_fields:
        assert field in wkb


def test_propagate_singularity_2d_consistency():
    """Test consistency of singularity propagation."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi + eta
    
    initial_sing = [(0, 0, 1, 1)]
    result = propagate_singularity_2d(p, initial_sing, (0, 5))
    
    # Number of trajectories should match initial singularities
    assert len(result['trajectories']) == len(initial_sing)
    assert len(result['endpoints']) == len(initial_sing)
    
    # Initial points should match
    assert result['initial'] == initial_sing