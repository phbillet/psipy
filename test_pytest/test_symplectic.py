# test_symplectic.py
# Combined test suite for the unified symplectic module.

import numpy as np
import pytest
from sympy import symbols, simplify, sin, cos, exp, sqrt, pi, Matrix

# Import the unified module (assumed to be named symplectic)
from symplectic import (
    SymplecticForm,
    hamiltonian_flow,
    poisson_bracket,
    find_fixed_points,
    linearize_at_fixed_point,
    action_integral,
    phase_portrait,
    separatrix_analysis,
    visualize_phase_space_structure,
    action_angle_transform,
    frequency,
    poincare_section,
    first_return_map,
    monodromy_matrix,
    lyapunov_exponents,
    project,
    visualize_poincare_section,
    _infer_variables,
    _get_ndof,
    _check_ndof,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def is_sympy_zero(expr):
    """Check if a sympy expression simplifies to zero."""
    return simplify(expr) == 0

# -----------------------------------------------------------------------------
# Tests for utility functions
# -----------------------------------------------------------------------------

def test_infer_variables_1d():
    x, p = symbols('x p', real=True)
    H = x**2 + p**2
    inferred = _infer_variables(H, expected_ndof=1)
    assert set(inferred) == {x, p}

def test_infer_variables_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = x1**2 + p1**2 + x2**2 + p2**2
    inferred = _infer_variables(H)
    # Order may vary; check that all are present
    assert set(inferred) == {x1, p1, x2, p2}

def test_infer_variables_ambiguous():
    a, b = symbols('a b', real=True)
    H = a**2 + b**2
    with pytest.raises(ValueError):
        _infer_variables(H)

def test_get_ndof():
    x, p = symbols('x p', real=True)
    assert _get_ndof([x, p]) == 1
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    assert _get_ndof([x1, p1, x2, p2]) == 2
    with pytest.raises(ValueError):
        _get_ndof([x, p, x2])  # odd number

def test_check_ndof():
    x, p = symbols('x p', real=True)
    _check_ndof([x, p], 1)  # should not raise
    with pytest.raises(ValueError):
        _check_ndof([x, p], 2)

# -----------------------------------------------------------------------------
# Tests for SymplecticForm
# -----------------------------------------------------------------------------

def test_symplectic_form_canonical():
    omega1 = SymplecticForm(n=1)
    assert omega1.omega_matrix == Matrix([[0, -1], [1, 0]])
    assert omega1.omega_inv == Matrix([[0, 1], [-1, 0]])

    omega2 = SymplecticForm(n=2)
    expected = Matrix([
        [0, -1, 0, 0],
        [1,  0, 0, 0],
        [0,  0, 0, -1],
        [0,  0, 1, 0]
    ])
    assert omega2.omega_matrix == expected

def test_symplectic_form_with_vars():
    x, p = symbols('x p', real=True)
    omega = SymplecticForm(vars_phase=[x, p])
    assert omega.n == 1
    assert omega.vars_phase == [x, p]

def test_symplectic_form_custom_matrix():
    x, p = symbols('x p', real=True)
    custom = Matrix([[0, -2], [2, 0]])
    omega = SymplecticForm(vars_phase=[x, p], omega_matrix=custom)
    assert omega.omega_matrix == custom

def test_symplectic_form_non_antisymmetric():
    x, p = symbols('x p', real=True)
    bad = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        SymplecticForm(vars_phase=[x, p], omega_matrix=bad)

def test_symplectic_form_eval():
    x, p = symbols('x p', real=True)
    omega = SymplecticForm(vars_phase=[x, p])
    arr = omega.eval((1.0, 2.0))
    assert arr.shape == (2, 2)
    assert arr[0, 1] == -1
    assert arr[1, 0] == 1

# -----------------------------------------------------------------------------
# Tests for Poisson bracket (generic)
# -----------------------------------------------------------------------------

def test_poisson_bracket_fundamental():
    x, p = symbols('x p', real=True)
    assert poisson_bracket(x, p, vars_phase=[x, p]) == 1
    assert poisson_bracket(p, x, vars_phase=[x, p]) == -1
    assert poisson_bracket(x, x, vars_phase=[x, p]) == 0

def test_poisson_bracket_composite():
    x, p = symbols('x p', real=True)
    f = x * p
    g = p**2 / 2
    pb = poisson_bracket(f, g, vars_phase=[x, p])
    assert is_sympy_zero(pb - p**2)

def test_poisson_bracket_jacobi():
    x, p = symbols('x p', real=True)
    f = x**2
    g = p**2
    h = x * p
    term1 = poisson_bracket(f, poisson_bracket(g, h, vars_phase=[x,p]),
                            vars_phase=[x,p])
    term2 = poisson_bracket(g, poisson_bracket(h, f, vars_phase=[x,p]),
                            vars_phase=[x,p])
    term3 = poisson_bracket(h, poisson_bracket(f, g, vars_phase=[x,p]),
                            vars_phase=[x,p])
    assert is_sympy_zero(term1 + term2 + term3)

def test_poisson_bracket_linearity():
    x, p = symbols('x p', real=True)
    f = x**2
    g = p**2
    h = x * p
    left = poisson_bracket(f, g + h, vars_phase=[x,p])
    right = poisson_bracket(f, g, vars_phase=[x,p]) + poisson_bracket(f, h, vars_phase=[x,p])
    assert is_sympy_zero(left - right)

# -----------------------------------------------------------------------------
# Tests for Hamiltonian flow (generic)
# -----------------------------------------------------------------------------

def test_hamiltonian_flow_harmonic_oscillator():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    traj = hamiltonian_flow(H, (1, 0), (0, 2*np.pi),
                            vars_phase=[x, p], integrator='symplectic', n_steps=1000)
    assert np.std(traj['energy']) < 2e-3
    assert np.isclose(traj['x'][-1], 1.0, rtol=1e-2)
    assert np.isclose(traj['p'][-1], 0.0, atol=1e-2)

def test_hamiltonian_flow_4d_uncoupled():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 0, 1)   # x1=1, p1=0, x2=0, p2=1
    traj = hamiltonian_flow(H, z0, (0, 2*np.pi),
                            vars_phase=[x1, p1, x2, p2],
                            integrator='symplectic', n_steps=1000)
    assert np.std(traj['energy']) < 2e-3
    # After one period, x1 should return to 1 (cosine)
    assert np.isclose(traj['x1'][-1], 1.0, rtol=1e-2)
    # After one period, x2 should return to 0 (sine)
    assert np.isclose(traj['x2'][-1], 0.0, atol=1e-2)

def test_hamiltonian_flow_methods_1d():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    for method in ['symplectic', 'verlet', 'rk45']:
        traj = hamiltonian_flow(H, (1, 0), (0, 10),
                                vars_phase=[x, p], integrator=method, n_steps=100)
        assert len(traj['x']) == 100
        assert len(traj['p']) == 100

def test_hamiltonian_flow_methods_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    for method in ['symplectic', 'verlet', 'rk45']:
        traj = hamiltonian_flow(H, (1,0,0,1), (0,10),
                                vars_phase=[x1,p1,x2,p2], integrator=method, n_steps=100)
        assert len(traj['x1']) == 100
        assert len(traj['x2']) == 100

def test_hamiltonian_flow_energy_conservation_1d():
    x, p = symbols('x p', real=True)
    # Anharmonic oscillator
    H = p**2/2 + x**4/4
    traj = hamiltonian_flow(H, (1, 0), (0, 20),
                            vars_phase=[x, p], integrator='symplectic', n_steps=1000)
    assert np.std(traj['energy']) < 2e-2

def test_hamiltonian_flow_energy_conservation_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2 + 0.1 * x1 * x2
    traj = hamiltonian_flow(H, (1,0,0,1), (0,20),
                            vars_phase=[x1,p1,x2,p2], integrator='symplectic', n_steps=1000)
    assert np.std(traj['energy']) < 2e-2

def test_hamiltonian_flow_negative_time():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    traj_fwd = hamiltonian_flow(H, (1,0), (0,5),
                                vars_phase=[x,p], integrator='symplectic', n_steps=100)
    traj_bwd = hamiltonian_flow(H, (1,0), (5,0),
                                vars_phase=[x,p], integrator='symplectic', n_steps=100)
    assert len(traj_fwd['x']) == len(traj_bwd['x'])

def test_invalid_integrator():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    with pytest.raises(ValueError):
        hamiltonian_flow(H, (1,0), (0,5), vars_phase=[x,p], integrator='invalid')

# -----------------------------------------------------------------------------
# Tests for fixed points and linearization (dimension-agnostic)
# -----------------------------------------------------------------------------

def test_find_fixed_points_harmonic():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    fps = find_fixed_points(H, vars_phase=[x, p])
    assert len(fps) == 1
    assert np.allclose(fps[0], (0, 0), atol=1e-6)

def test_find_fixed_points_double_well():
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2
    fps = find_fixed_points(H, vars_phase=[x, p])
    # Should be three: (0,0), (1,0), (-1,0)
    assert len(fps) == 3
    xs = sorted([fp[0] for fp in fps])
    assert np.allclose(xs, [-1, 0, 1], atol=1e-6)

def test_linearize_at_fixed_point_center():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    lin = linearize_at_fixed_point(H, (0,0), vars_phase=[x,p])
    assert lin['type'] in ('elliptic', 'center')
    eigs = lin['eigenvalues']
    assert np.allclose(np.abs(eigs), 1.0)

def test_linearize_at_fixed_point_saddle():
    x, p = symbols('x p', real=True)
    H = p**2/2 - x**2/2
    lin = linearize_at_fixed_point(H, (0,0), vars_phase=[x,p])
    assert lin['type'] in ('hyperbolic', 'saddle')
    eigs = lin['eigenvalues']
    # One positive, one negative
    assert (eigs[0] * eigs[1]) < 0

def test_find_fixed_points_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    # Two uncoupled oscillators: only origin is fixed
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    fps = find_fixed_points(H, vars_phase=[x1,p1,x2,p2])
    assert len(fps) == 1
    assert np.allclose(fps[0], (0,0,0,0), atol=1e-6)

def test_linearize_at_fixed_point_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    lin = linearize_at_fixed_point(H, (0,0,0,0), vars_phase=[x1,p1,x2,p2])
    # eigenvalues should be ±i (double)
    eigs = lin['eigenvalues']
    assert np.allclose(np.abs(eigs), 1.0)

# -----------------------------------------------------------------------------
# 1‑DOF specific tests (action_integral, phase_portrait, etc.)
# -----------------------------------------------------------------------------

def test_action_integral_harmonic():
    x, p, E_sym = symbols('x p E', real=True, positive=True)
    H = (p**2 + x**2) / 2
    # symbolic
    I_sym = action_integral(H, E_sym, vars_phase=[x,p], method='symbolic')
    assert is_sympy_zero(I_sym - E_sym)
    # numerical
    I_num = action_integral(H, 1.0, vars_phase=[x,p], method='numerical')
    assert np.isclose(I_num, 1.0, rtol=1e-2)

def test_action_integral_multiple_energies():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    for E in [0.5, 1.0, 2.0]:
        I = action_integral(H, E, vars_phase=[x,p], method='numerical')
        assert np.isclose(I, E, rtol=0.05)

def test_action_integral_double_well():
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2
    # For energies below barrier, action should be finite
    I = action_integral(H, 0.1, vars_phase=[x,p], method='numerical')
    assert np.isfinite(I) and I > 0

def test_phase_portrait_execution():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive
        phase_portrait(H, (-2,2), (-2,2), vars_phase=[x,p], levels=5)
        assert True
    except Exception as e:
        pytest.fail(f"phase_portrait raised: {e}")

def test_separatrix_analysis():
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2
    sep = separatrix_analysis(H, (-2,2), (-2,2), (0,0), vars_phase=[x,p])
    assert 'E_saddle' in sep
    assert 'unstable_manifolds' in sep
    assert 'stable_manifolds' in sep

def test_action_angle_transform():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    aa = action_angle_transform(H, (-3,3), (-3,3), vars_phase=[x,p], n_contours=5)
    assert 'energies' in aa and len(aa['energies']) > 0
    assert 'actions' in aa and len(aa['actions']) > 0
    assert 'frequencies' in aa

def test_frequency():
    x, p, I = symbols('x p I', real=True, positive=True)
    H = I  # H as function of action
    omega = frequency(H, 1.0, method='derivative')
    assert np.isclose(omega, 1.0)

# -----------------------------------------------------------------------------
# 2‑DOF specific tests (Poincaré section, monodromy, Lyapunov, project)
# -----------------------------------------------------------------------------

def test_poincare_section_basic():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    section_def = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0 = (1, 0, 0, 1)
    ps = poincare_section(H, section_def, z0, tmax=50,
                          vars_phase=[x1,p1,x2,p2], n_returns=20)
    assert 't_crossings' in ps and len(ps['t_crossings']) > 0
    assert 'section_points' in ps and len(ps['section_points']) > 0

def test_poincare_section_directions():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 0, 1)
    pos = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    both = {'variable': 'x2', 'value': 0, 'direction': 'both'}
    ps_pos = poincare_section(H, pos, z0, tmax=30, vars_phase=[x1,p1,x2,p2], n_returns=10)
    ps_both = poincare_section(H, both, z0, tmax=30, vars_phase=[x1,p1,x2,p2], n_returns=10)
    assert len(ps_both['t_crossings']) >= len(ps_pos['t_crossings'])

def test_poincare_section_invalid_variable():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2) / 2
    section = {'variable': 'invalid', 'value': 0}
    z0 = (1,0,0,0)
    with pytest.raises(ValueError):
        poincare_section(H, section, z0, tmax=5, vars_phase=[x1,p1,x2,p2])

def test_first_return_map():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    section = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0 = (1,0,0,1)
    ps = poincare_section(H, section, z0, tmax=50, vars_phase=[x1,p1,x2,p2], n_returns=5)
    if len(ps['section_points']) >= 2:
        rm = first_return_map(ps['section_points'], plot_variables=('x1','p1'))
        assert 'current' in rm
        assert 'next' in rm
        assert rm['current'].shape == (len(ps['section_points'])-1, 2)
    else:
        pytest.skip("Not enough section points for return map")

def test_monodromy_matrix_stable():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1,0,0,0)
    T = 2 * np.pi
    traj = hamiltonian_flow(H, z0, (0, T), vars_phase=[x1,p1,x2,p2],
                            integrator='rk45', n_steps=500)
    mono = monodromy_matrix(H, traj, vars_phase=[x1,p1,x2,p2], method='finite_difference')
    assert 'M' in mono and mono['M'].shape == (4,4)
    mult = mono['floquet_multipliers']
    assert len(mult) == 4
    # For harmonic oscillator, multipliers should be on unit circle
    assert np.allclose(np.abs(mult), 1.0, atol=1e-3)

def test_lyapunov_exponents():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 0.5, 0)
    traj = hamiltonian_flow(H, z0, (0, 50), vars_phase=[x1,p1,x2,p2],
                            integrator='symplectic', n_steps=500)
    dt = traj['t'][1] - traj['t'][0]
    exponents = lyapunov_exponents(traj, dt, vars_phase=[x1,p1,x2,p2], n_vectors=4)
    assert len(exponents) == 4

def test_project_functions():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 1, 0)
    traj = hamiltonian_flow(H, z0, (0, 10), vars_phase=[x1,p1,x2,p2], n_steps=100)

    # config
    x, y, lbl = project(traj, plane='xy', vars_phase=[x1,p1,x2,p2])
    assert len(x) == 100 and len(y) == 100
    assert lbl == ('x₁', 'x₂')

    # x-p
    x, p, lbl = project(traj, plane='xp', vars_phase=[x1,p1,x2,p2])
    assert lbl == ('x₁', 'p₁')

    # momentum
    px, py, lbl = project(traj, plane='pp', vars_phase=[x1,p1,x2,p2])
    assert lbl == ('p₁', 'p₂')

    # mixed
    x1, p2, lbl = project(traj, plane='x1p2', vars_phase=[x1,p1,x2,p2])
    assert lbl == ('x₁', 'p₂')

    # invalid
    with pytest.raises(ValueError):
        project(traj, plane='invalid', vars_phase=[x1,p1,x2,p2])

def test_visualize_poincare_section_execution():
    """Just test that it runs without error."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    section = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0_list = [(1,0,0,1), (0.5,0,0,0.5)]
    try:
        import matplotlib
        matplotlib.use('Agg')
        visualize_poincare_section(H, z0_list, section, vars_phase=[x1,p1,x2,p2],
                                   tmax=10, n_returns=5, plot_vars=('x1','p1'))
        assert True
    except Exception as e:
        pytest.fail(f"visualize_poincare_section raised: {e}")

# -----------------------------------------------------------------------------
# Tests for dimension‑checking decorators/helpers
# -----------------------------------------------------------------------------

def test_action_integral_wrong_dim():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    with pytest.raises(ValueError):
        action_integral(H, 1.0, vars_phase=[x1,p1,x2,p2])

def test_phase_portrait_wrong_dim():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    with pytest.raises(ValueError):
        phase_portrait(H, (-2,2), (-2,2), vars_phase=[x1,p1,x2,p2])

def test_poincare_section_wrong_dim():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    section = {'variable': 'x', 'value': 0}
    with pytest.raises(ValueError):
        poincare_section(H, section, (1,0), tmax=10, vars_phase=[x,p])

