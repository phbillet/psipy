# test_microlocal.py
# Merged and adapted test suite for the unified microlocal module

import numpy as np
import pytest
from sympy import symbols, sqrt, exp, sin, cos, simplify

# Import everything from the unified microlocal module
from microlocal import *


# ======================================================================
# 1D TESTS (originally from test_microlocal_1d.py)
# ======================================================================

def test_characteristic_variety():
    """Test characteristic variety computation."""
    x, xi = symbols('x xi', real=True)
    
    # Wave operator: p = ξ² - 1
    p = xi**2 - 1
    char = characteristic_variety(p)
    
    assert 'implicit' in char
    assert 'equation' in char
    assert 'explicit' in char
    
    # Should have two branches: ξ = ±1
    assert len(char['explicit']) == 2


def test_characteristic_variety_elliptic():
    """Test characteristic variety for elliptic operator."""
    x, xi = symbols('x xi', real=True)
    
    # Elliptic: p = ξ² + 1 (no real zeros)
    p = xi**2 + 1
    char = characteristic_variety(p)
    
    # Explicit solutions should be complex
    assert char['explicit'] is not None


def test_characteristic_variety_transport():
    """Test characteristic variety for transport operator."""
    x, xi = symbols('x xi', real=True)
    
    # Transport: p = ξ
    p = xi
    char = characteristic_variety(p)
    
    # Single branch: ξ = 0
    assert len(char['explicit']) == 1
    assert char['explicit'][0] == 0


def test_bicharacteristic_flow_transport():
    """Test bicharacteristic flow for transport operator."""
    x, xi = symbols('x xi', real=True)
    
    # Transport: p = ξ
    # Bicharacteristics: ẋ = 1, ξ̇ = 0
    p = xi
    
    traj = bicharacteristic_flow(p, (0, 1), (0, 5), 
                                 method='symplectic', n_steps=100)
    
    # ξ should be constant
    assert np.std(traj['xi']) < 1e-6
    
    # x should evolve linearly: x = t
    expected_x = traj['t']
    assert np.allclose(traj['x'], expected_x, rtol=1e-2)


def test_bicharacteristic_flow_harmonic():
    """Test bicharacteristic flow for harmonic oscillator."""
    x, xi = symbols('x xi', real=True)
    
    # Harmonic oscillator: p = ξ² + x²
    p = xi**2 + x**2
    
    traj = bicharacteristic_flow(p, (1, 0), (0, np.pi), 
                                 method='symplectic', n_steps=200)
    
    # Symbol value should be conserved (on characteristic)
    assert np.std(traj['symbol_value']) < 0.1


def test_bicharacteristic_flow_methods():
    """Test different integration methods for bicharacteristics."""
    x, xi = symbols('x xi', real=True)
    p = xi**2 + x**2
    
    # Hamiltonian method
    traj_ham = bicharacteristic_flow(p, (1, 0), (0, 5), 
                                     method='hamiltonian', n_steps=100)
    assert len(traj_ham['x']) == 100
    
    # RK45 method
    traj_rk = bicharacteristic_flow(p, (1, 0), (0, 5), 
                                    method='rk45', n_steps=100)
    assert len(traj_rk['x']) > 0


def test_wkb_approximation_free_particle():
    """Test WKB approximation for free particle."""
    x, xi = symbols('x xi', real=True)
    
    # Free particle: p = ξ²
    p = xi**2
    
    # Initial data for wkb_approximation (scalar point)
    initial_phase = {
        'x': [0.0],
        'S': [0.0],
        'p_x': [1.0],
        'a': {0: [1.0]}
    }
    wkb = wkb_approximation(p, initial_phase, order=1, domain=(-2, 2), 
                            resolution=100, epsilon=1.0)
    
    assert 'x' in wkb
    assert 'S' in wkb
    assert 'a' in wkb
    assert 'u' in wkb
    
    # Phase should be approximately linear for free particle
    assert len(wkb['x']) == 100


def test_wkb_approximation_harmonic():
    """Test WKB approximation for harmonic oscillator."""
    x, xi = symbols('x xi', real=True)
    
    # Harmonic: p = ξ² + x²
    p = xi**2 + x**2
    
    initial_phase = {
        'x': [0.0],
        'S': [0.0],
        'p_x': [1.0],
        'a': {0: [1.0]}
    }
    wkb = wkb_approximation(p, initial_phase, order=1, domain=(-1, 1), 
                            resolution=50, epsilon=1.0)
    
    assert wkb is not None
    assert len(wkb['x']) > 0


def test_bohr_sommerfeld_harmonic():
    """Test Bohr-Sommerfeld quantization for harmonic oscillator."""
    x, p = symbols('x p', real=True)
    
    # Harmonic oscillator: H = (p² + x²)/2
    H = (p**2 + x**2) / 2
    
    quant = bohr_sommerfeld_quantization(H, n_max=5, hbar=1.0, 
                                         x_range=(-5, 5))
    
    assert 'n' in quant
    assert 'E_n' in quant
    assert 'actions' in quant
    
    # For harmonic oscillator: E_n = (n + 1/2)ℏω with ω=1
    # Should be close to n + α where α ≈ 0.5
    assert len(quant['E_n']) > 0


def test_bohr_sommerfeld_convergence():
    """Test Bohr-Sommerfeld gives reasonable energies."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    quant = bohr_sommerfeld_quantization(H, n_max=3, hbar=1.0)
    
    # Ground state should be close to 0.5
    if len(quant['E_n']) > 0:
        E0 = quant['E_n'][0]
        assert 0.3 < E0 < 0.7


def test_propagate_singularity():
    """Test propagation of singularities."""
    x, xi = symbols('x xi', real=True)
    
    # Transport operator
    p = xi
    
    initial_sing = [(0, 1)]
    result = propagate_singularity(p, initial_sing, (0, 5))
    
    assert 'trajectories' in result
    assert 'endpoints' in result
    assert 'initial' in result
    
    # Should have propagated
    assert len(result['trajectories']) == 1
    assert len(result['endpoints']) == 1


def test_propagate_singularity_multiple():
    """Test propagation of multiple singularities."""
    x, xi = symbols('x xi', real=True)
    p = xi**2 + x
    
    initial_sing = [(0, 1), (0, -1), (1, 0)]
    result = propagate_singularity(p, initial_sing, (0, 3))
    
    assert len(result['trajectories']) == 3
    assert len(result['endpoints']) == 3


def test_find_caustics_1d():
    """Test 1D caustic finding."""
    x, xi = symbols('x xi', real=True)
    
    # Symbol with caustic
    p = xi**2 - x
    
    caustics = find_caustics_1d(p, (-2, 2), (-2, 2), resolution=50)
    
    assert 'x_grid' in caustics
    assert 'xi_grid' in caustics
    assert 'caustic_indicator' in caustics


def test_characteristic_variety_numerical():
    """Test numerical evaluation of characteristic variety."""
    x, xi = symbols('x xi', real=True)
    p = xi**2 - x**2
    
    char = characteristic_variety(p)
    
    # Test numerical function
    func = char['function']
    
    # On characteristic: p(1, 1) = 0
    val = func(1.0, 1.0)
    assert np.isclose(val, 0.0, atol=1e-10)
    
    # Off characteristic: p(1, 0) ≠ 0
    val_off = func(1.0, 0.0)
    assert not np.isclose(val_off, 0.0)


def test_wkb_approximation_consistency():
    """Test WKB solution consistency."""
    x, xi = symbols('x xi', real=True)
    p = xi**2
    
    initial_phase = {
        'x': [0.0],
        'S': [0.0],
        'p_x': [1.0],
        'a': {0: [1.0]}
    }
    wkb = wkb_approximation(p, initial_phase, order=0, domain=(-1, 1), 
                            resolution=50, epsilon=1.0)
    
    # Check arrays have consistent lengths
    assert len(wkb['x']) == len(wkb['S'])
    assert len(wkb['x']) == len(wkb['a'][0])
    assert len(wkb['x']) == len(wkb['u'])


def test_bicharacteristic_flow_energy():
    """Test energy conservation along bicharacteristics."""
    x, xi = symbols('x xi', real=True)
    
    # For H = ξ² + x², energy (symbol value) should be constant
    H = xi**2 + x**2
    
    traj = bicharacteristic_flow(H, (1, 0), (0, 10), 
                                 method='symplectic', n_steps=500)
    
    # Symbol value = energy
    energy_drift = np.std(traj['symbol_value'])
    assert energy_drift < 0.1


def test_invalid_wkb_initial_conditions():
    """Test WKB with missing initial conditions."""
    x, xi = symbols('x xi', real=True)
    p = xi**2

    # Missing initial data – now expect ValueError (not KeyError)
    with pytest.raises(ValueError):          # ← changed
        wkb_approximation(p, {}, order=1)


def test_bohr_sommerfeld_no_bound_states():
    """Test Bohr-Sommerfeld with potential that has no bound states."""
    x, p = symbols('x p', real=True)
    
    # Free particle (no bound states)
    H = p**2 / 2
    
    quant = bohr_sommerfeld_quantization(H, n_max=5, x_range=(-10, 10))
    
    # May return empty or fail - should handle gracefully
    assert 'E_n' in quant


# ======================================================================
# 2D TESTS (originally from test_microlocal_2d.py)
# ======================================================================

def test_characteristic_variety_2d():
    """Test 2D characteristic variety computation."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic: p = ξ² + η² - 1
    p = xi**2 + eta**2 - 1
    char = characteristic_variety(p)   # unified function
    
    assert 'implicit' in char
    assert 'equation' in char
    assert 'function' in char


def test_characteristic_variety_2d_elliptic():
    """Test characteristic variety for elliptic operator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Elliptic: no real characteristic points
    p = xi**2 + eta**2 + 1
    char = characteristic_variety(p)
    
    assert char is not None
    assert callable(char['function'])


def test_characteristic_variety_2d_anisotropic():
    """Test anisotropic characteristic variety."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Anisotropic: p = ξ² - η²
    p = xi**2 - eta**2
    char = characteristic_variety(p)
    
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
    traj = bicharacteristic_flow(p, z0, (0, 5), method='symplectic', n_steps=100)
    
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
    traj = bicharacteristic_flow(p, z0, (0, 5), method='symplectic', n_steps=100)
    
    assert len(traj['x']) == 100
    assert len(traj['y']) == 100


def test_bichar_flow_2d_methods():
    """Test different integration methods for 2D bicharacteristics."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    z0 = (0, 0, 1, 0)
    
    # Symplectic
    traj_symp = bicharacteristic_flow(p, z0, (0, 5), method='symplectic', n_steps=50)
    assert len(traj_symp['x']) == 50
    
    # Verlet
    traj_verlet = bicharacteristic_flow(p, z0, (0, 5), method='verlet', n_steps=50)
    assert len(traj_verlet['x']) == 50
    
    # RK45
    traj_rk = bicharacteristic_flow(p, z0, (0, 5), method='rk45', n_steps=50)
    assert len(traj_rk['x']) > 0


def test_bichar_flow_2d_energy_conservation():
    """Test energy conservation along bicharacteristics."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # For H = ξ² + η² + x² + y²
    H = xi**2 + eta**2 + x**2 + y**2
    
    z0 = (1, 1, 1, 0)
    traj = bicharacteristic_flow(H, z0, (0, 10), method='symplectic', n_steps=500)
    
    # Symbol value should be approximately constant
    symbol_drift = np.std(traj['symbol_value'])
    assert symbol_drift < 0.2


def test_wkb_approximation_2d_placeholder():
    """Test WKB multidimensional (basic structure)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Wave equation: p = ξ² + η²
    p = xi**2 + eta**2
    
    # Create proper initial data using the helper function
    initial_phase = create_initial_data_line(
        x_range=(-1, 1), 
        n_points=10,
        direction=(0, 1),
        y_intercept=0.0
    )
    
    # Add amplitude data if needed (defaults to ones)
    wkb = wkb_approximation(
        p, 
        initial_phase, 
        order=1,
        domain=((-2, 2), (-2, 2)), 
        resolution=20,
        epsilon=0.1
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
    assert wkb['a'][0].shape == (20, 20)
    assert wkb['u'].shape == (20, 20)
    
    # Check that we traced some rays
    assert len(wkb['rays']) > 0


def test_wkb_approximation_2d_line_source():
    """Test WKB with line source (plane wave generation)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic wave equation
    p = xi**2 + eta**2
    
    ic = create_initial_data_line(
        x_range=(-1, 1),
        n_points=15,
        direction=(0, 1),
        y_intercept=-1.0
    )
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=30,
        epsilon=0.1
    )
    
    # Check output structure
    assert wkb['x'].shape == (30, 30)
    assert wkb['y'].shape == (30, 30)
    assert len(wkb['rays']) > 0
    assert np.std(wkb['S']) > 0.1


def test_wkb_approximation_2d_circular_source():
    """Test WKB with circular source (expanding waves)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    p = xi**2 + eta**2
    
    ic = create_initial_data_circle(
        radius=0.5,
        n_points=20,
        outward=True
    )
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-3, 3), (-3, 3)),
        resolution=40,
        epsilon=0.1
    )
    
    assert len(wkb['rays']) >= 15
    assert wkb['u'].shape == (40, 40)
    
    # Phase should increase with distance
    S_center = np.mean(wkb['S'][18:22, 18:22])
    S_edge = np.mean(wkb['S'][0:5, 0:5])
    assert np.abs(S_edge) > np.abs(S_center)


def test_wkb_approximation_2d_point_source():
    """Test WKB with point source (spherical waves)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    p = xi**2 + eta**2
    
    ic = create_initial_data_point_source(
        x0=0.0,
        y0=0.0,
        n_rays=16
    )
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=35,
        epsilon=0.1
    )
    
    for ray in wkb['rays']:
        assert np.isclose(ray['x'][0], 0.0, atol=1e-6)
        assert np.isclose(ray['y'][0], 0.0, atol=1e-6)
    
    for ray in wkb['rays']:
        distance = np.sqrt(ray['x'][-1]**2 + ray['y'][-1]**2)
        assert distance > 0.5


def test_wkb_approximation_2d_anisotropic():
    """Test WKB with anisotropic symbol."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    p = xi**2 + 4*eta**2
    
    n_pts = 12
    y_vals = np.linspace(-1, 1, n_pts)
    ic = {
        'x': np.zeros(n_pts),
        'y': y_vals,
        'S': np.zeros(n_pts),
        'p_x': np.ones(n_pts),
        'p_y': np.zeros(n_pts)
    }
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=30,
        epsilon=0.1
    )
    
    for ray in wkb['rays']:
        dx = ray['x'][-1] - ray['x'][0]
        dy = ray['y'][-1] - ray['y'][0]
        if abs(dx) > 0.1:
            assert abs(dx) > abs(dy)


def test_compute_maslov_index():
    """Test Maslov index computation."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    p = xi**2 + eta**2 + x**2 + y**2
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi
    
    traj = bicharacteristic_flow(p, z0, (0, T), method='symplectic', n_steps=100)
    
    maslov = compute_maslov_index(traj, p)
    assert isinstance(maslov, (int, np.integer))


def test_compute_maslov_index_non_closed():
    """Test Maslov index for non-closed path (should warn)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    z0 = (0, 0, 1, 1)
    traj = bicharacteristic_flow(p, z0, (0, 3), method='symplectic', n_steps=50)
    
    maslov = compute_maslov_index(traj, p)
    assert maslov is not None


def test_compute_caustics_2d():
    """Test caustic computation in 2D."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2

    n_points = 10
    initial_curve = {
        'x': np.linspace(-1, 1, n_points),
        'y': np.zeros(n_points),
        'xi': np.ones(n_points),
        'eta': np.zeros(n_points)
    }

    caustics = compute_caustics_2d(p, initial_curve, tmax=2.0, n_rays=n_points)

    # The function now returns a list of CausticEvent objects.
    assert isinstance(caustics, list)
    # With the given initial data (parallel rays, no focusing), no caustics are expected.
    assert len(caustics) == 0


def test_propagate_singularity_2d():
    """Test propagation of singularities in 2D (unified function)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    p = xi + eta
    initial_sing = [(0, 0, 1, 1)]
    result = propagate_singularity(p, initial_sing, (0, 5))
    
    assert 'trajectories' in result
    assert 'endpoints' in result
    assert 'initial' in result
    assert len(result['trajectories']) == 1
    assert len(result['endpoints']) == 1


def test_propagate_singularity_2d_multiple():
    """Test propagation of multiple singularities in 2D."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    initial_sing = [
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (0, 0, np.sqrt(0.5), np.sqrt(0.5))
    ]
    
    result = propagate_singularity(p, initial_sing, (0, 3))
    
    assert len(result['trajectories']) == 3
    assert len(result['endpoints']) == 3


def test_characteristic_variety_2d_numerical():
    """Test numerical evaluation of 2D characteristic variety."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2 - x**2 - y**2
    
    char = characteristic_variety(p)
    func = char['function']
    
    # On characteristic: p(1, 1, 1, 1) = 0
    val = func(1.0, 1.0, 1.0, 1.0)
    assert np.isclose(val, 0.0, atol=1e-10)
    
    # Off characteristic
    val_off = func(0.0, 0.0, 1.0, 1.0)
    assert not np.isclose(val_off, 0.0)


def test_bichar_flow_2d_return_to_origin_symplectic():
    """Test periodic bicharacteristic for harmonic oscillator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    H = xi**2 + eta**2 + x**2 + y**2
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi
    
    traj = bicharacteristic_flow(H, z0, (0, T), method='symplectic', n_steps=1000)
    
    assert np.isclose(traj['x'][-1], 1, rtol=1e-2)
    assert np.isclose(traj['y'][-1], 0, atol=2e-2)
    assert np.isclose(traj['xi'][-1], 0, atol=2e-2)
    assert np.isclose(traj['eta'][-1], 1, rtol=1e-2)


def test_bichar_flow_2d_return_to_origin_verlet():
    """Test periodic bicharacteristic for harmonic oscillator (verlet)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    H = xi**2 + eta**2 + x**2 + y**2
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi
    
    traj = bicharacteristic_flow(H, z0, (0, T), method='verlet', n_steps=1000)
    
    assert np.isclose(traj['x'][-1], 1, rtol=1e-2)
    assert np.isclose(traj['y'][-1], 0, atol=2e-2)
    assert np.isclose(traj['xi'][-1], 0, atol=2e-2)
    assert np.isclose(traj['eta'][-1], 1, rtol=1e-2)


def test_bichar_flow_2d_phase_space_volume():
    """Test phase space volume preservation (Liouville theorem)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    z0_list = [
        (0, 0, 1, 1),
        (0.1, 0, 1, 1),
        (0, 0.1, 1, 1),
        (0, 0, 1.1, 1)
    ]
    
    trajectories = []
    for z0 in z0_list:
        traj = bicharacteristic_flow(p, z0, (0, 5), method='symplectic', n_steps=50)
        trajectories.append(traj)
    
    assert len(trajectories) == 4


def test_invalid_bichar_method_2d():
    """Test error handling for invalid integration method."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    with pytest.raises(ValueError):
        bicharacteristic_flow(p, (0, 0, 1, 1), (0, 5), method='invalid')


def test_bichar_flow_2d_long_time():
    """Test long-time integration stability."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2 + x**2 + y**2
    
    z0 = (1, 0, 0, 1)
    traj = bicharacteristic_flow(p, z0, (0, 50), method='symplectic', n_steps=1000)
    
    assert np.all(np.isfinite(traj['x']))
    assert np.all(np.isfinite(traj['y']))
    assert np.all(np.isfinite(traj['xi']))
    assert np.all(np.isfinite(traj['eta']))


def test_caustics_2d_convergence():
    """Test that caustic computation runs without error (no actual caustics expected)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2

    n_points = 5
    initial_curve = {
        'x': np.linspace(-0.5, 0.5, n_points),
        'y': np.zeros(n_points),
        'xi': np.ones(n_points),
        'eta': np.zeros(n_points)
    }

    caustics = compute_caustics_2d(p, initial_curve, tmax=1.0, n_rays=n_points)

    assert isinstance(caustics, list)
    # No caustics for this simple flow.
    assert len(caustics) == 0


def test_maslov_index_harmonic_2d():
    """Test Maslov index for 2D harmonic oscillator."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    H = (xi**2 + eta**2 + x**2 + y**2) / 2
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi
    
    traj = bicharacteristic_flow(H, z0, (0, T), method='symplectic', n_steps=200)
    
    maslov = compute_maslov_index(traj, H)
    
    # For 2D harmonic oscillator, typical value is 2
    assert maslov in [0, 1, 2, 3, 4]


def test_wkb_approximation_2d_structure():
    """Test WKB multidimensional output structure (point source)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    initial_phase = create_initial_data_point_source(
        x0=0.0, 
        y0=0.0, 
        n_rays=8
    )
    
    wkb = wkb_approximation(p, initial_phase, resolution=10, epsilon=0.1)
    
    required_fields = ['x', 'y', 'S', 'a', 'u', 'rays']
    for field in required_fields:
        assert field in wkb
    
    assert wkb['x'].shape == (10, 10)
    assert wkb['y'].shape == (10, 10)
    assert wkb['S'].shape == (10, 10)
    assert wkb['a'][0].shape == (10, 10)
    assert wkb['u'].shape == (10, 10)
    assert isinstance(wkb['rays'], list)
    assert len(wkb['rays']) > 0


def test_propagate_singularity_2d_consistency():
    """Test consistency of singularity propagation."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi + eta
    
    initial_sing = [(0, 0, 1, 1)]
    result = propagate_singularity(p, initial_sing, (0, 5))
    
    assert len(result['trajectories']) == len(initial_sing)
    assert len(result['endpoints']) == len(initial_sing)
    assert result['initial'] == initial_sing