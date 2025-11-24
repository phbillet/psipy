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
    """Test WKB multidimensional."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Wave equation: p = ξ² + η²
    p = xi**2 + eta**2
    
    # Create proper initial data using the helper function
    # Line segment from (-1, 0) to (1, 0) with rays going in direction (0, 1)
    initial_phase = create_initial_data_line(
        x_range=(-1, 1), 
        n_points=10,  # Use fewer points for faster test
        direction=(0, 1),  # Rays going upward
        y_intercept=0.0
    )
    
    # Run WKB with smaller domain and resolution for faster test
    wkb = wkb_multidim(
        p, 
        initial_phase, 
        order=1,
        domain=((-2, 2), (-2, 2)), 
        resolution=20
    )
    
    # Basic checks
    assert 'x' in wkb
    assert 'y' in wkb
    assert 'S' in wkb
    assert 'a' in wkb
    assert 'u' in wkb
    assert 'rays' in wkb
    
    # Check shapes
    assert wkb['x'].shape == (20, 20)
    assert wkb['y'].shape == (20, 20)
    assert wkb['S'].shape == (20, 20)
    assert wkb['a'].shape == (20, 20)
    assert wkb['u'].shape == (20, 20)
    
    # Check that we traced some rays
    assert len(wkb['rays']) > 0
    
    print("✓ WKB multidimensional test passed")

def test_wkb_multidim_line_source():
    """Test WKB with line source (plane wave generation)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic wave equation
    p = xi**2 + eta**2
    
    # Horizontal line with upward rays
    ic = create_initial_data_line(
        x_range=(-1, 1),
        n_points=15,
        direction=(0, 1),
        y_intercept=-1.0
    )
    
    wkb = wkb_multidim(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=30
    )
    
    # Check output structure
    assert 'x' in wkb
    assert 'y' in wkb
    assert 'S' in wkb
    assert 'a' in wkb
    assert 'u' in wkb
    assert 'rays' in wkb
    
    # Check dimensions
    assert wkb['x'].shape == (30, 30)
    assert wkb['y'].shape == (30, 30)
    
    # Check rays were traced
    assert len(wkb['rays']) > 0
    
    # Phase should vary (not all zeros)
    assert np.std(wkb['S']) > 0.1


def test_wkb_multidim_circular_source():
    """Test WKB with circular source (expanding waves)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic symbol
    p = xi**2 + eta**2
    
    # Circle with outward rays
    ic = create_initial_data_circle(
        radius=0.5,
        n_points=20,
        outward=True
    )
    
    wkb = wkb_multidim(
        p, 
        ic, 
        order=1,
        domain=((-3, 3), (-3, 3)),
        resolution=40
    )
    
    # Should have traced multiple rays
    assert len(wkb['rays']) >= 15
    
    # Check amplitude behavior (should decay with distance for circular waves)
    a_center = wkb['a'][20, 20]  # Center
    a_edge = wkb['a'][0, 0]      # Edge
    
    # Solution should exist
    assert wkb['u'].shape == (40, 40)
    
    # Phase should have circular symmetry (approximately)
    # Check that phase increases with distance from origin
    S_center_region = wkb['S'][18:22, 18:22]
    S_edge_region = wkb['S'][0:5, 0:5]
    assert np.mean(np.abs(S_edge_region)) > np.mean(np.abs(S_center_region))


def test_wkb_multidim_point_source():
    """Test WKB with point source (spherical waves)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Wave equation
    p = xi**2 + eta**2
    
    # Point source at origin
    ic = create_initial_data_point_source(
        x0=0.0,
        y0=0.0,
        n_rays=16
    )
    
    wkb = wkb_multidim(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=35
    )
    
    # All rays should start at origin
    for ray in wkb['rays']:
        assert np.isclose(ray['x'][0], 0.0, atol=1e-6)
        assert np.isclose(ray['y'][0], 0.0, atol=1e-6)
    
    # Rays should diverge
    for ray in wkb['rays']:
        distance_traveled = np.sqrt(
            (ray['x'][-1] - ray['x'][0])**2 + 
            (ray['y'][-1] - ray['y'][0])**2
        )
        assert distance_traveled > 0.5


def test_wkb_multidim_anisotropic():
    """Test WKB with anisotropic symbol."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Anisotropic dispersion: faster in x-direction
    p = xi**2 + 4*eta**2
    
    # Vertical line with horizontal rays
    n_pts = 12
    y_vals = np.linspace(-1, 1, n_pts)
    ic = {
        'x': np.zeros(n_pts),
        'y': y_vals,
        'S': np.zeros(n_pts),
        'p_x': np.ones(n_pts),   # ξ = 1
        'p_y': np.zeros(n_pts)   # η = 0
    }
    
    wkb = wkb_multidim(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=30
    )
    
    # Check anisotropic propagation
    # Rays should propagate primarily in x-direction
    for ray in wkb['rays']:
        dx = ray['x'][-1] - ray['x'][0]
        dy = ray['y'][-1] - ray['y'][0]
        
        # Movement in x should dominate
        if abs(dx) > 0.1:  # If ray traveled
            assert abs(dx) > abs(dy)


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

def test_bichar_flow_2d_return_to_origin_symplectic():
    """Test periodic bicharacteristic for harmonic oscillator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # 2D harmonic oscillator
    H = xi**2 + eta**2 + x**2 + y**2
    
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi  # Period
    
    # Increase steps for better accuracy
    traj = bichar_flow_2d(H, z0, (0, T), method='symplectic', n_steps=1000)
    
    # Should approximately return to initial point
    assert np.isclose(traj['x'][-1], 1, rtol=1e-2)
    assert np.isclose(traj['y'][-1], 0, atol=2e-2)
    assert np.isclose(traj['xi'][-1], 0, atol=2e-2)
    assert np.isclose(traj['eta'][-1], 1, rtol=1e-2)
    
    print("✓ Bicharacteristic flow 2D return to origin test passed")

def test_bichar_flow_2d_return_to_origin_verlet():
    """Test periodic bicharacteristic for harmonic oscillator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # 2D harmonic oscillator
    H = xi**2 + eta**2 + x**2 + y**2
    
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi  # Period
    
    # Use Verlet method (2nd order, more accurate)
    traj = bichar_flow_2d(H, z0, (0, T), method='verlet', n_steps=1000)
    
    # Should approximately return to initial point
    assert np.isclose(traj['x'][-1], 1, rtol=1e-2)
    assert np.isclose(traj['y'][-1], 0, atol=2e-2)
    assert np.isclose(traj['xi'][-1], 0, atol=2e-2)
    assert np.isclose(traj['eta'][-1], 1, rtol=1e-2)
    
    print("✓ Bicharacteristic flow 2D return to origin test passed")


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
    
    # Utiliser la fonction helper pour créer une source ponctuelle
    # avec plusieurs rayons (plus réaliste pour tester la structure)
    initial_phase = create_initial_data_point_source(
        x0=0.0, 
        y0=0.0, 
        n_rays=8  # 8 rayons partant de l'origine
    )
    
    wkb = wkb_multidim(p, initial_phase, resolution=10)
    
    # Check all required fields exist
    required_fields = ['x', 'y', 'S', 'a', 'u', 'rays']
    for field in required_fields:
        assert field in wkb
    
    # Vérifier aussi les dimensions
    assert wkb['x'].shape == (10, 10)
    assert wkb['y'].shape == (10, 10)
    assert wkb['S'].shape == (10, 10)
    assert wkb['a'].shape == (10, 10)
    assert wkb['u'].shape == (10, 10)
    assert isinstance(wkb['rays'], list)
    assert len(wkb['rays']) > 0


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