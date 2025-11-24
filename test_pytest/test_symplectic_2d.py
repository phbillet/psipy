import numpy as np
import pytest
from sympy import symbols, simplify
from symplectic_2d import (
    SymplecticForm2D, hamiltonian_flow_4d, poincare_section,
    first_return_map, monodromy_matrix, lyapunov_exponents, project
)


def test_symplectic_form_2d():
    """Test 2D symplectic form initialization."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    omega = SymplecticForm2D(vars_phase=(x1, p1, x2, p2))
    
    # Check canonical form structure
    assert omega.omega_matrix.shape == (4, 4)
    
    # Check antisymmetry
    assert omega.omega_matrix == -omega.omega_matrix.T


def test_hamiltonian_flow_4d_uncoupled():
    """Test 4D flow for uncoupled oscillators."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    
    # Two independent harmonic oscillators
    H = (p1**2 + x1**2 + p2**2 + x2**2) / 2
    
    z0 = (1, 0, 0, 1)
    traj = hamiltonian_flow_4d(H, z0, (0, 2*np.pi), 
                               integrator='symplectic', n_steps=1000)
    
    # Check energy conservation
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 1e-2
    
    # Should be periodic
    assert np.isclose(traj['x1'][-1], 1.0, rtol=1e-2)


def test_hamiltonian_flow_4d_coupled():
    """Test 4D flow for coupled oscillators."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    
    # Coupled system
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2 + 0.1 * x1 * x2
    
    z0 = (1, 0, 0.5, 0)
    traj = hamiltonian_flow_4d(H, z0, (0, 20), 
                               integrator='symplectic', n_steps=500)
    
    assert len(traj['x1']) == 500
    assert len(traj['x2']) == 500
    
    # Energy should be conserved
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 0.1


def test_hamiltonian_flow_4d_methods():
    """Test different integration methods for 4D flow."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 1, 0)
    
    # Symplectic
    traj_symp = hamiltonian_flow_4d(H, z0, (0, 10), 
                                    integrator='symplectic', n_steps=100)
    assert len(traj_symp['x1']) == 100
    
    # Verlet
    traj_verlet = hamiltonian_flow_4d(H, z0, (0, 10), 
                                      integrator='verlet', n_steps=100)
    assert len(traj_verlet['x1']) == 100
    
    # RK45
    traj_rk = hamiltonian_flow_4d(H, z0, (0, 10), 
                                  integrator='rk45', n_steps=100)
    assert len(traj_rk['x1']) > 0


def test_poincare_section():
    """Test Poincaré section computation."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    # Define section: x2 = 0
    section_def = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    
    z0 = (1, 0, 0, 1)
    ps = poincare_section(H, section_def, z0, tmax=50, n_returns=20)
    
    assert 't_crossings' in ps
    assert 'section_points' in ps
    
    # Should have found some crossings
    assert len(ps['t_crossings']) > 0
    assert len(ps['section_points']) > 0


def test_poincare_section_multiple_directions():
    """Test Poincaré section with different crossing directions."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 0, 1)
    
    # Positive crossings
    section_pos = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    ps_pos = poincare_section(H, section_pos, z0, tmax=30, n_returns=10)
    
    # Both directions
    section_both = {'variable': 'x2', 'value': 0, 'direction': 'both'}
    ps_both = poincare_section(H, section_both, z0, tmax=30, n_returns=10)
    
    # Both should have more crossings than positive only
    assert len(ps_both['t_crossings']) >= len(ps_pos['t_crossings'])


def test_first_return_map():
    """Test first return map computation."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    section_def = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0 = (1, 0, 0, 1)
    
    ps = poincare_section(H, section_def, z0, tmax=50, n_returns=20)
    
    if len(ps['section_points']) >= 2:
        rm = first_return_map(ps['section_points'], plot_variables=('x1', 'p1'))
        
        assert 'current' in rm
        assert 'next' in rm
        assert 'variables' in rm
        
        # Should have one less point than crossings
        assert len(rm['current']) == len(ps['section_points']) - 1


def test_monodromy_matrix():
    """Test monodromy matrix computation."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    # Generate a periodic orbit
    z0 = (1, 0, 0, 1)
    T = 2 * np.pi  # Period for harmonic oscillator
    
    traj = hamiltonian_flow_4d(H, z0, (0, T), 
                               integrator='symplectic', n_steps=100)
    
    mono = monodromy_matrix(H, traj, method='finite_difference')
    
    assert 'M' in mono
    assert 'eigenvalues' in mono
    assert 'floquet_multipliers' in mono
    
    # Matrix should be 4x4
    assert mono['M'].shape == (4, 4)
    
    # For stable periodic orbit, eigenvalues should be on unit circle
    eigs = mono['floquet_multipliers']
    assert len(eigs) == 4


def test_lyapunov_exponents():
    """Test Lyapunov exponent computation."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 0.5, 0)
    traj = hamiltonian_flow_4d(H, z0, (0, 50), 
                               integrator='symplectic', n_steps=500)
    
    dt = traj['t'][1] - traj['t'][0]
    exponents = lyapunov_exponents(traj, dt, n_vectors=4)
    
    # Should return 4 exponents
    assert len(exponents) == 4
    
    # For Hamiltonian systems: λᵢ + λ₅₋ᵢ = 0 (pairing)
    # At least one should be zero (time translation)


def test_project_position():
    """Test trajectory projection onto position space."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 1, 0)
    traj = hamiltonian_flow_4d(H, z0, (0, 10), n_steps=100)
    
    x, y, labels = project(traj, plane='xy')
    
    assert len(x) == 100
    assert len(y) == 100
    assert labels == ('x₁', 'x₂')


def test_project_momentum():
    """Test trajectory projection onto momentum space."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 1, 0)
    traj = hamiltonian_flow_4d(H, z0, (0, 10), n_steps=100)
    
    px, py, labels = project(traj, plane='pp')
    
    assert len(px) == 100
    assert len(py) == 100
    assert labels == ('p₁', 'p₂')


def test_project_mixed():
    """Test mixed projections."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 1, 0)
    traj = hamiltonian_flow_4d(H, z0, (0, 10), n_steps=100)
    
    # x-p projection
    x, p, labels = project(traj, plane='xp')
    assert labels == ('x₁', 'p₁')


def test_energy_conservation_4d():
    """Test energy conservation in 4D flow."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    
    z0 = (1, 0, 1, 0)
    traj = hamiltonian_flow_4d(H, z0, (0, 50), 
                               integrator='symplectic', n_steps=1000)
    
    # Check energy conservation
    energy_initial = traj['energy'][0]
    energy_final = traj['energy'][-1]
    relative_error = abs(energy_final - energy_initial) / abs(energy_initial)
    
    assert relative_error < 2e-2


def test_invalid_integrator_4d():
    """Test error handling for invalid integrator."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2) / 2
    
    with pytest.raises(ValueError):
        hamiltonian_flow_4d(H, (1, 0, 0, 0), (0, 5), integrator='invalid')


def test_invalid_projection():
    """Test error handling for invalid projection."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2) / 2
    
    traj = hamiltonian_flow_4d(H, (1, 0, 0, 0), (0, 5), n_steps=10)
    
    with pytest.raises(ValueError):
        project(traj, plane='invalid')


def test_poincare_section_variable_validation():
    """Test Poincaré section with invalid variable."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2) / 2
    
    # Invalid variable name
    section_def = {'variable': 'invalid', 'value': 0, 'direction': 'positive'}
    z0 = (1, 0, 0, 0)
    
    # Should handle gracefully or raise error
    try:
        ps = poincare_section(H, section_def, z0, tmax=5, n_returns=5)
        # If it doesn't raise error, check it found no crossings
        assert len(ps['t_crossings']) == 0
    except (KeyError, ValueError):
        # Expected behavior
        pass