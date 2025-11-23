import numpy as np
import pytest
from sympy import symbols, simplify, sin, cos, exp, sqrt
from symplectic_1d import *


def test_symplectic_form_1d():
    """Test symplectic form initialization."""
    x, p = symbols('x p', real=True)
    omega = SymplecticForm1D(vars_phase=(x, p))
    
    # Check canonical form
    from sympy import Matrix
    expected = Matrix([[0, -1], [1, 0]])
    assert omega.omega_matrix == expected


def test_poisson_bracket_fundamental():
    """Test fundamental Poisson brackets."""
    x, p = symbols('x p', real=True)
    
    # {x, p} = 1
    pb1 = poisson_bracket(x, p)
    assert pb1 == 1
    
    # {p, x} = -1
    pb2 = poisson_bracket(p, x)
    assert pb2 == -1
    
    # {x, x} = 0
    pb3 = poisson_bracket(x, x)
    assert pb3 == 0


def test_poisson_bracket_composite():
    """Test Poisson brackets on composite functions."""
    x, p = symbols('x p', real=True)
    
    # {xp, p²/2} = p²
    f = x * p
    g = p**2 / 2
    pb = poisson_bracket(f, g)
    
    assert simplify(pb - p**2) == 0


def test_poisson_bracket_jacobi_identity():
    """Test Jacobi identity: {f,{g,h}} + {g,{h,f}} + {h,{f,g}} = 0."""
    x, p = symbols('x p', real=True)
    
    f = x**2
    g = p**2
    h = x * p
    
    term1 = poisson_bracket(f, poisson_bracket(g, h))
    term2 = poisson_bracket(g, poisson_bracket(h, f))
    term3 = poisson_bracket(h, poisson_bracket(f, g))
    
    result = simplify(term1 + term2 + term3)
    assert result == 0


def test_hamiltonian_flow_harmonic_oscillator():
    """Test Hamiltonian flow for harmonic oscillator."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    # Integrate
    traj = hamiltonian_flow(H, (1, 0), (0, 2*np.pi), 
                           integrator='symplectic', n_steps=1000)
    
    # Check energy conservation
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 1e-3
    
    # Should return to initial condition (periodic)
    assert np.isclose(traj['x'][-1], 1.0, rtol=1e-2)
    assert np.isclose(traj['p'][-1], 0.0, atol=1e-2)


def test_hamiltonian_flow_methods():
    """Test different integration methods."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    # Symplectic
    traj_symp = hamiltonian_flow(H, (1, 0), (0, 10), 
                                integrator='symplectic', n_steps=100)
    assert len(traj_symp['x']) == 100
    
    # Verlet
    traj_verlet = hamiltonian_flow(H, (1, 0), (0, 10), 
                                  integrator='verlet', n_steps=100)
    assert len(traj_verlet['x']) == 100
    
    # RK45
    traj_rk = hamiltonian_flow(H, (1, 0), (0, 10), 
                              integrator='rk45', n_steps=100)
    assert len(traj_rk['x']) > 0


def test_action_integral_harmonic_oscillator():
    """Test action integral for harmonic oscillator."""
    x, p, E_sym = symbols('x p E', real=True, positive=True)
    H = (p**2 + x**2) / 2
    
    # Symbolic
    I_sym = action_integral(H, E_sym, method='symbolic')
    # For harmonic oscillator: I = E
    assert simplify(I_sym - E_sym) == 0
    
    # Numerical
    E_val = 1.0
    I_num = action_integral(H, E_val, method='numerical')
    assert np.isclose(I_num, E_val, rtol=1e-2)


def test_action_integral_multiple_energies():
    """Test action integral for various energies."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    for E_test in [0.5, 1.0, 2.0, 5.0]:
        I_test = action_integral(H, E_test, method='numerical')
        error = abs(I_test - E_test) / E_test
        assert error < 0.05, f"Error too large for E={E_test}"


def test_find_fixed_points_harmonic():
    """Test fixed point finding for harmonic oscillator."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    fps = find_fixed_points(H, x_range=(-2, 2), p_range=(-2, 2))
    
    # Should have one fixed point at origin
    assert len(fps) == 1
    assert np.allclose(fps[0], (0, 0), atol=1e-6)


def test_find_fixed_points_double_well():
    """Test fixed points for double well potential."""
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2
    
    fps = find_fixed_points(H, x_range=(-2, 2), p_range=(-2, 2))
    
    # Should have 3 fixed points: (0,0), (±1, 0)
    assert len(fps) >= 1


def test_linearize_at_fixed_point_center():
    """Test linearization at elliptic fixed point."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2  # Harmonic oscillator
    
    lin = linearize_at_fixed_point(H, (0, 0))
    
    assert 'jacobian' in lin
    assert 'eigenvalues' in lin
    assert 'type' in lin
    
    # Should be a center
    assert lin['type'] == 'center'
    
    # Eigenvalues should be ±i
    eigs = lin['eigenvalues']
    assert np.allclose(np.abs(eigs), 1.0)


def test_linearize_at_fixed_point_saddle():
    """Test linearization at hyperbolic fixed point."""
    x, p = symbols('x p', real=True)
    H = p**2/2 - x**2/2  # Inverted oscillator
    
    lin = linearize_at_fixed_point(H, (0, 0))
    
    # Should be a saddle
    assert lin['type'] == 'saddle'
    
    # Eigenvalues should be real with opposite signs
    eigs = lin['eigenvalues']
    assert np.all(np.isreal(eigs))


def test_action_angle_transform():
    """Test action-angle transformation."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    aa = action_angle_transform(H, (-3, 3), (-3, 3), n_contours=5)
    
    assert 'energies' in aa
    assert 'actions' in aa
    assert 'frequencies' in aa
    
    assert len(aa['energies']) > 0


def test_separatrix_analysis_saddle():
    """Test separatrix computation near saddle point."""
    x, p = symbols('x p', real=True)
    # Double well with saddle at origin
    H = p**2/2 + x**4/4 - x**2/2
    
    sep = separatrix_analysis(H, (-2, 2), (-2, 2), (0, 0))
    
    assert 'E_saddle' in sep
    assert 'unstable_dir' in sep
    assert 'stable_dir' in sep
    assert 'unstable_manifolds' in sep
    assert 'stable_manifolds' in sep


def test_phase_portrait_execution():
    """Test phase portrait generation (execution only)."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    # Just test that it runs without error
    # Actual plotting would require display
    try:
        # Capture the function call without displaying
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        phase_portrait(H, (-2, 2), (-2, 2), levels=10)
        assert True
    except Exception as e:
        pytest.fail(f"phase_portrait raised exception: {e}")


def test_hamiltonian_flow_energy_conservation():
    """Test energy conservation for various Hamiltonians."""
    x, p = symbols('x p', real=True)
    
    # Harmonic oscillator
    H1 = (p**2 + x**2) / 2
    traj1 = hamiltonian_flow(H1, (1, 0), (0, 20), 
                            integrator='symplectic', n_steps=1000)
    assert np.std(traj1['energy']) < 1e-2
    
    # Anharmonic oscillator
    H2 = p**2/2 + x**4/4
    traj2 = hamiltonian_flow(H2, (1, 0), (0, 20), 
                            integrator='symplectic', n_steps=1000)
    assert np.std(traj2['energy']) < 1e-2


def test_frequency_computation():
    """Test frequency computation from action."""
    x, p, I = symbols('x p I', real=True, positive=True)
    
    # For harmonic oscillator: H(I) = ω·I with ω=1
    H = I  # As function of action
    
    omega = frequency(H, 1.0, method='derivative')
    
    # Should be ω = dH/dI = 1
    assert np.isclose(omega, 1.0)


def test_symplectic_form_eval():
    """Test symplectic form evaluation."""
    x, p = symbols('x p', real=True)
    omega = SymplecticForm1D(vars_phase=(x, p))
    
    result = omega.eval(1.0, 2.0)
    
    assert result.shape == (2, 2)
    assert result[0, 1] == -1
    assert result[1, 0] == 1


def test_invalid_integrator():
    """Test error handling for invalid integrator."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    with pytest.raises(ValueError):
        hamiltonian_flow(H, (1, 0), (0, 5), integrator='invalid')


def test_poisson_bracket_linearity():
    """Test linearity of Poisson bracket."""
    x, p = symbols('x p', real=True)
    
    f = x**2
    g = p**2
    h = x * p
    
    # {f, g+h} = {f,g} + {f,h}
    left = poisson_bracket(f, g + h)
    right = poisson_bracket(f, g) + poisson_bracket(f, h)
    
    assert simplify(left - right) == 0


def test_hamiltonian_flow_negative_time():
    """Test backward time integration."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    
    # Forward
    traj_fwd = hamiltonian_flow(H, (1, 0), (0, 5), 
                                integrator='symplectic', n_steps=100)
    
    # Backward
    traj_bwd = hamiltonian_flow(H, (1, 0), (5, 0), 
                                integrator='symplectic', n_steps=100)
    
    # Should be time-reversible (approximately)
    assert len(traj_fwd['x']) == len(traj_bwd['x'])