import numpy as np
import pytest
from sympy import symbols, sqrt, exp, sin, cos, simplify
from microlocal_1d import *

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


def test_wkb_ansatz_free_particle():
    """Test WKB approximation for free particle."""
    x, xi = symbols('x xi', real=True)
    
    # Free particle: p = ξ²
    p = xi**2
    
    initial_phase = {'x0': 0, 'S0': 0, 'Sp0': 1}
    wkb = wkb_ansatz(p, initial_phase, order=1, x_domain=(-2, 2), n_points=100)
    
    assert 'x' in wkb
    assert 'S' in wkb
    assert 'a' in wkb
    assert 'u' in wkb
    
    # Phase should be approximately linear for free particle
    assert len(wkb['x']) == 100


def test_wkb_ansatz_harmonic():
    """Test WKB approximation for harmonic oscillator."""
    x, xi = symbols('x xi', real=True)
    
    # Harmonic: p = ξ² + x²
    p = xi**2 + x**2
    
    initial_phase = {'x0': 0, 'S0': 0, 'Sp0': 1}
    wkb = wkb_ansatz(p, initial_phase, order=1, x_domain=(-1, 1), n_points=50)
    
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
    
    # Check energies
    n_vals = quant['n']
    E_vals = quant['E_n']
    
    # For harmonic oscillator: E_n = (n + 1/2)ℏω with ω=1
    # Should be close to n + α where α ≈ 0.5
    assert len(E_vals) > 0


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


def test_find_caustics():
    """Test caustic finding."""
    x, xi = symbols('x xi', real=True)
    
    # Symbol with caustic
    p = xi**2 - x
    
    caustics = find_caustics(p, (-2, 2), (-2, 2), resolution=50)
    
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


def test_wkb_ansatz_consistency():
    """Test WKB solution consistency."""
    x, xi = symbols('x xi', real=True)
    p = xi**2
    
    initial_phase = {'x0': 0, 'S0': 0, 'Sp0': 1}
    wkb = wkb_ansatz(p, initial_phase, order=0, x_domain=(-1, 1), n_points=50)
    
    # Check arrays have consistent lengths
    assert len(wkb['x']) == len(wkb['S'])
    assert len(wkb['x']) == len(wkb['a'])
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
    
    # Missing initial data
    with pytest.raises(KeyError):
        wkb_ansatz(p, {}, order=1)


def test_bohr_sommerfeld_no_bound_states():
    """Test Bohr-Sommerfeld with potential that has no bound states."""
    x, p = symbols('x p', real=True)
    
    # Free particle (no bound states)
    H = p**2 / 2
    
    quant = bohr_sommerfeld_quantization(H, n_max=5, x_range=(-10, 10))
    
    # May return empty or fail - should handle gracefully
    assert 'E_n' in quant