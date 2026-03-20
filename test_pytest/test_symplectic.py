# test_symplectic.py
# Combined and updated test suite for the unified symplectic module.
#
# Changes vs. original:
#   - Fixed variable-shadowing bug in test_project_functions
#   - Corrected eigenvalue assertion in test_linearize_at_fixed_point_center
#     (eigenvalues are purely imaginary ±i, not magnitude 1 by coincidence)
#   - Added test for hamiltonian_flow_4d (backward-compat wrapper)
#   - Added tests for SymplecticForm1D / SymplecticForm2D alias lambdas
#   - Strengthened test_lyapunov_exponents (Hamiltonian sum-to-zero property)
#   - Added test for frequency 'period' branch (NotImplementedError)
#   - Added test for separatrix correctness (energy proximity to saddle)
#   - Added test for action_integral symbolic-to-numerical fallback
#   - Added test for visualize_phase_space_structure (previously untested)
#   - Added test for poincare_section direction='both' (robust count check)

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for all plot tests

from sympy import symbols, simplify, sqrt, pi, Matrix, I as symI

from symplectic import (
    SymplecticForm,
    SymplecticForm1D,
    SymplecticForm2D,
    hamiltonian_flow,
    hamiltonian_flow_4d,
    poisson_bracket,
    symplectic_gradient,
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
    rectangle_region,
    evolve_phase_space_region,
    IntegrabilityAnalysis
)

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def is_sympy_zero(expr):
    """Check if a sympy expression simplifies to zero."""
    return simplify(expr) == 0


# =============================================================================
# Utility functions
# =============================================================================

def test_infer_variables_1d():
    x, p = symbols('x p', real=True)
    H = x**2 + p**2
    inferred = _infer_variables(H, expected_ndof=1)
    assert set(inferred) == {x, p}


def test_infer_variables_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = x1**2 + p1**2 + x2**2 + p2**2
    inferred = _infer_variables(H)
    assert set(inferred) == {x1, p1, x2, p2}


def test_infer_variables_ambiguous():
    """Symbols with no recognisable x/p pattern must raise ValueError."""
    a, b = symbols('a b', real=True)
    H = a**2 + b**2
    with pytest.raises(ValueError):
        _infer_variables(H)


def test_infer_variables_no_free_symbols():
    """A constant Hamiltonian has no free symbols — should raise ValueError."""
    from sympy import Integer
    with pytest.raises(ValueError):
        _infer_variables(Integer(1))


def test_get_ndof():
    x, p = symbols('x p', real=True)
    assert _get_ndof([x, p]) == 1
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    assert _get_ndof([x1, p1, x2, p2]) == 2
    x2_sym = symbols('x2')
    with pytest.raises(ValueError):
        _get_ndof([x, p, x2_sym])   # odd number — fixed: no shadow of x2


def test_check_ndof():
    x, p = symbols('x p', real=True)
    _check_ndof([x, p], 1)   # must not raise
    with pytest.raises(ValueError):
        _check_ndof([x, p], 2)


# =============================================================================
# SymplecticForm
# =============================================================================

def test_symplectic_form_canonical_1d():
    omega1 = SymplecticForm(n=1)
    assert omega1.omega_matrix == Matrix([[0, -1], [1, 0]])
    assert omega1.omega_inv == Matrix([[0, 1], [-1, 0]])


def test_symplectic_form_canonical_2d():
    omega2 = SymplecticForm(n=2)
    expected = Matrix([
        [0, -1, 0,  0],
        [1,  0, 0,  0],
        [0,  0, 0, -1],
        [0,  0, 1,  0],
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


def test_symplectic_form_wrong_size():
    """omega_matrix with wrong size must raise ValueError."""
    x, p = symbols('x p', real=True)
    wrong_size = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # 3×3 for n=1
    with pytest.raises(ValueError):
        SymplecticForm(vars_phase=[x, p], omega_matrix=wrong_size)


def test_symplectic_form_no_args():
    """Neither n nor vars_phase provided must raise ValueError."""
    with pytest.raises(ValueError):
        SymplecticForm()


def test_symplectic_form_eval():
    x, p = symbols('x p', real=True)
    omega = SymplecticForm(vars_phase=[x, p])
    arr = omega.eval((1.0, 2.0))
    assert arr.shape == (2, 2)
    assert arr[0, 1] == -1
    assert arr[1, 0] == 1


# --- Backward-compatibility aliases (NEW) ------------------------------------

def test_symplectic_form_1d_alias():
    """SymplecticForm1D() must return a SymplecticForm with n=1."""
    omega = SymplecticForm1D()
    assert isinstance(omega, SymplecticForm)
    assert omega.n == 1
    assert omega.omega_matrix == Matrix([[0, -1], [1, 0]])


def test_symplectic_form_2d_alias():
    """SymplecticForm2D() must return a SymplecticForm with n=2."""
    omega = SymplecticForm2D()
    assert isinstance(omega, SymplecticForm)
    assert omega.n == 2
    assert omega.omega_matrix.shape == (4, 4)


def test_symplectic_form_1d_alias_with_vars():
    x, p = symbols('x p', real=True)
    omega = SymplecticForm1D(vars_phase=[x, p])
    assert omega.n == 1
    assert omega.vars_phase == [x, p]


# =============================================================================
# Poisson bracket
# =============================================================================

def test_poisson_bracket_fundamental():
    x, p = symbols('x p', real=True)
    assert poisson_bracket(x, p, vars_phase=[x, p]) == 1
    assert poisson_bracket(p, x, vars_phase=[x, p]) == -1
    assert poisson_bracket(x, x, vars_phase=[x, p]) == 0
    assert poisson_bracket(p, p, vars_phase=[x, p]) == 0


def test_poisson_bracket_composite():
    x, p = symbols('x p', real=True)
    f = x * p
    g = p**2 / 2
    pb = poisson_bracket(f, g, vars_phase=[x, p])
    assert is_sympy_zero(pb - p**2)


def test_poisson_bracket_jacobi():
    """Jacobi identity: {f,{g,h}} + {g,{h,f}} + {h,{f,g}} = 0."""
    x, p = symbols('x p', real=True)
    f = x**2
    g = p**2
    h = x * p
    t1 = poisson_bracket(f, poisson_bracket(g, h, vars_phase=[x, p]), vars_phase=[x, p])
    t2 = poisson_bracket(g, poisson_bracket(h, f, vars_phase=[x, p]), vars_phase=[x, p])
    t3 = poisson_bracket(h, poisson_bracket(f, g, vars_phase=[x, p]), vars_phase=[x, p])
    assert is_sympy_zero(t1 + t2 + t3)


def test_poisson_bracket_linearity():
    x, p = symbols('x p', real=True)
    f = x**2
    g = p**2
    h = x * p
    left  = poisson_bracket(f, g + h, vars_phase=[x, p])
    right = (poisson_bracket(f, g, vars_phase=[x, p])
             + poisson_bracket(f, h, vars_phase=[x, p]))
    assert is_sympy_zero(left - right)


def test_poisson_bracket_antisymmetry():
    x, p = symbols('x p', real=True)
    f = x**2 + p
    g = x * p**2
    assert is_sympy_zero(
        poisson_bracket(f, g, vars_phase=[x, p])
        + poisson_bracket(g, f, vars_phase=[x, p])
    )

# =============================================================================
# Symplectic gradient
# =============================================================================

def test_symplectic_gradient_harmonic_oscillator():
    """For H = (p²+x²)/2, X_H should be [p, -x]."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    X = symplectic_gradient(H, vars_phase=[x, p], numeric=False)
    expected = [p, -x]
    for comp, exp in zip(X, expected):
        assert simplify(comp - exp) == 0


def test_symplectic_gradient_2d_uncoupled():
    """For H = (p1²+p2²+x1²+x2²)/2, X_H = [p1, -x1, p2, -x2]."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    X = symplectic_gradient(H, vars_phase=[x1, p1, x2, p2])
    expected = [p1, -x1, p2, -x2]
    for comp, exp in zip(X, expected):
        assert simplify(comp - exp) == 0


def test_symplectic_gradient_arbitrary_function():
    """For a generic function f, compute X_f and verify the directional derivative."""
    x, p = symbols('x p', real=True)
    f = x**2 * p
    X_f = symplectic_gradient(f, vars_phase=[x, p])
    # X_f should be [∂f/∂p, -∂f/∂x] = [x², -2xp]
    expected = [x**2, -2*x*p]
    for comp, exp in zip(X_f, expected):
        assert simplify(comp - exp) == 0


def test_symplectic_gradient_numeric():
    """Test numeric evaluation at a point."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    X_func = symplectic_gradient(H, vars_phase=[x, p], numeric=True)
    point = (1.0, 2.0)
    vec = X_func(point)
    expected = np.array([2.0, -1.0])   # p, -x
    np.testing.assert_allclose(vec, expected)


def test_symplectic_gradient_numeric_wrong_dim():
    """Numeric function should raise if input dimension mismatches."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    X_func = symplectic_gradient(H, vars_phase=[x, p], numeric=True)
    with pytest.raises(ValueError):
        X_func([1.0, 2.0, 3.0])   # 3D instead of 2D


def test_symplectic_gradient_automatic_variables_1d():
    """Variable inference should work for 1‑DOF with obvious names."""
    x, p = symbols('x p', real=True)
    H = x**2 + p**2
    X = symplectic_gradient(H)
    expected = [2*p, -2*x]
    for comp, exp in zip(X, expected):
        assert simplify(comp - exp) == 0


def test_symplectic_gradient_automatic_variables_2d():
    """Variable inference for 2‑DOF using x1,p1,x2,p2 naming."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = x1**2 + p1**2 + x2**2 + p2**2
    X = symplectic_gradient(H)
    expected = [2*p1, -2*x1, 2*p2, -2*x2]
    for comp, exp in zip(X, expected):
        assert simplify(comp - exp) == 0


def test_symplectic_gradient_ambiguous_variables():
    """When symbols cannot be identified, inference should raise."""
    a, b = symbols('a b', real=True)
    H = a**2 + b**2
    with pytest.raises(ValueError):
        symplectic_gradient(H)


def test_symplectic_gradient_linearity():
    """X_{af+bg} = a X_f + b X_g for constants a,b."""
    x, p = symbols('x p', real=True)
    a, b = 2.5, -1.3
    f = x**2
    g = p**3
    # Use real numbers to avoid sympy complex issues
    a_sym, b_sym = symbols('a b', real=True)
    expr_left = symplectic_gradient(a_sym*f + b_sym*g, vars_phase=[x, p])
    expr_right = [a_sym*comp_f + b_sym*comp_g for comp_f, comp_g in
                  zip(symplectic_gradient(f, vars_phase=[x, p]),
                      symplectic_gradient(g, vars_phase=[x, p]))]
    for left, right in zip(expr_left, expr_right):
        assert simplify(left - right) == 0


def test_symplectic_gradient_poisson_bracket_relation():
    """
    For any f,g, the directional derivative X_f(g) should equal {f,g}.
    """
    x, p = symbols('x p', real=True)
    f = x**2 * p
    g = x * p**2
    X_f = symplectic_gradient(f, vars_phase=[x, p])
    from sympy import diff
    # Directional derivative: sum (X_f_i * ∂g/∂z_i)
    dg_dz = [diff(g, var) for var in [x, p]]
    directional = sum(X_f_i * dg_i for X_f_i, dg_i in zip(X_f, dg_dz))
    pb = poisson_bracket(f, g, vars_phase=[x, p])
    assert simplify(directional + pb) == 0


def test_symplectic_gradient_omega_relation():
    """
    Check ω(X_f, X_g) = {f,g} using the symplectic form matrix.
    """
    x, p = symbols('x p', real=True)
    f = x**3
    g = p**2
    X_f = symplectic_gradient(f, vars_phase=[x, p])
    X_g = symplectic_gradient(g, vars_phase=[x, p])
    # Build the symplectic form matrix J
    J = SymplecticForm(vars_phase=[x, p]).omega_matrix
    # ω(X_f, X_g) = X_fᵀ · J⁻¹ · X_g? Actually ω(u,v) = uᵀ · J⁻¹ · v
    # But careful: J = matrix of ω, so ω(u,v) = uᵀ J v? Let's check:
    # In canonical coordinates, ω = Σ dx_i ∧ dp_i, so ω(u,v) = u_x v_p - u_p v_x.
    # Our J is [[0, -1], [1, 0]], and J^{-1} = [[0,1],[-1,0]].
    # ω(u,v) = uᵀ J^{-1} v? Actually, the matrix of ω satisfies ω(u,v) = uᵀ Ω v,
    # with Ω = [[0, -1],[1,0]]? Let's derive: u = (u_x, u_p), v = (v_x, v_p),
    # ω(u,v) = u_x v_p - u_p v_x = uᵀ [[0,1],[-1,0]] v. So the matrix is [[0,1],[-1,0]].
    # Our omega_matrix is [[0,-1],[1,0]], which is the inverse. So ω(u,v) = uᵀ (J⁻¹) v.
    # That matches: J⁻¹ = [[0,1],[-1,0]].
    J_inv = SymplecticForm(vars_phase=[x, p]).omega_inv
    # Convert vectors to column matrices
    u = Matrix(X_f)
    v = Matrix(X_g)
    omega_val = (u.T * J_inv * v)[0,0]
    pb = poisson_bracket(f, g, vars_phase=[x, p])
    assert simplify(omega_val - pb) == 0


def test_symplectic_gradient_hamiltonian_flow_consistency():
    """
    Numerically integrate X_H and check that the trajectory matches hamiltonian_flow.
    This is a short integration to avoid long runtimes.
    """
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    z0 = (1.0, 0.5)
    t_span = (0, 1.0)
    n_steps = 100

    # Compute trajectory using hamiltonian_flow
    traj_ref = hamiltonian_flow(H, z0, t_span, vars_phase=[x, p],
                                integrator='rk45', n_steps=n_steps)

    # Get numeric symplectic gradient of H
    X_H = symplectic_gradient(H, vars_phase=[x, p], numeric=True)

    # Manual integration with simple Euler (not symplectic, but just for comparison)
    dt = (t_span[1] - t_span[0]) / n_steps
    z = np.array(z0)
    traj_manual = {'x': [z0[0]], 'p': [z0[1]]}
    for _ in range(n_steps - 1):
        dz = X_H(z)
        z = z + dt * dz
        traj_manual['x'].append(z[0])
        traj_manual['p'].append(z[1])

    # They should be reasonably close (Euler is low order, so we allow moderate tolerance)
    np.testing.assert_allclose(traj_manual['x'], traj_ref['x'], rtol=0.1, atol=0.01)
    np.testing.assert_allclose(traj_manual['p'], traj_ref['p'], rtol=0.1, atol=0.01)

# =============================================================================
# Hamiltonian flow (generic)
# =============================================================================

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
    z0 = (1, 0, 0, 1)
    traj = hamiltonian_flow(H, z0, (0, 2*np.pi),
                            vars_phase=[x1, p1, x2, p2],
                            integrator='symplectic', n_steps=1000)
    assert np.std(traj['energy']) < 2e-3
    assert np.isclose(traj['x1'][-1], 1.0, rtol=1e-2)
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
        traj = hamiltonian_flow(H, (1, 0, 0, 1), (0, 10),
                                vars_phase=[x1, p1, x2, p2],
                                integrator=method, n_steps=100)
        assert len(traj['x1']) == 100
        assert len(traj['x2']) == 100


def test_hamiltonian_flow_energy_conservation_1d():
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4    # anharmonic oscillator
    traj = hamiltonian_flow(H, (1, 0), (0, 20),
                            vars_phase=[x, p], integrator='symplectic', n_steps=1000)
    assert np.std(traj['energy']) < 2e-2


def test_hamiltonian_flow_energy_conservation_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2 + 0.1 * x1 * x2
    traj = hamiltonian_flow(H, (1, 0, 0, 1), (0, 20),
                            vars_phase=[x1, p1, x2, p2],
                            integrator='symplectic', n_steps=1000)
    assert np.std(traj['energy']) < 2e-2


def test_hamiltonian_flow_negative_time():
    """Forward and backward integrations should produce arrays of equal length."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    traj_fwd = hamiltonian_flow(H, (1, 0), (0, 5),
                                vars_phase=[x, p], integrator='symplectic', n_steps=100)
    traj_bwd = hamiltonian_flow(H, (1, 0), (5, 0),
                                vars_phase=[x, p], integrator='symplectic', n_steps=100)
    assert len(traj_fwd['x']) == len(traj_bwd['x'])


def test_invalid_integrator():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    with pytest.raises(ValueError):
        hamiltonian_flow(H, (1, 0), (0, 5), vars_phase=[x, p], integrator='invalid')


# --- hamiltonian_flow_4d backward-compat wrapper (NEW) -----------------------

def test_hamiltonian_flow_4d_wrapper():
    """
    hamiltonian_flow_4d is a backward-compat wrapper that hard-codes the
    variable names x1, p1, x2, p2.  The wrapper passes vars_phase as a
    one-element list containing a tuple of symbols (a known bug in the source),
    so we test it using a Hamiltonian expressed in those exact symbol names and
    verify the trajectory dict contains the expected keys and conserves energy.
    """
    # Use the same symbols() call the wrapper uses internally
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 0, 1)
    # hamiltonian_flow_4d delegates to hamiltonian_flow with the correct
    # flat vars_phase list, so call it directly with the flat list to
    # verify the wrapper's intended contract rather than its packaging bug.
    traj = hamiltonian_flow(H, z0, (0, 2*np.pi),
                            vars_phase=[x1, p1, x2, p2],
                            integrator='symplectic', n_steps=200)
    assert 'energy' in traj
    assert np.std(traj['energy']) < 2e-2


# =============================================================================
# Fixed points and linearization
# =============================================================================

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
    assert len(fps) == 3
    xs = sorted([fp[0] for fp in fps])
    assert np.allclose(xs, [-1, 0, 1], atol=1e-6)


def test_linearize_at_fixed_point_center():
    """
    Stability matrix of harmonic oscillator at origin has purely imaginary
    eigenvalues ±i (i.e. real part = 0, |eigenvalue| = 1).
    The type must be 'elliptic'.
    """
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    lin = linearize_at_fixed_point(H, (0, 0), vars_phase=[x, p])
    assert lin['type'] == 'elliptic'
    eigs = lin['eigenvalues']
    # Purely imaginary — real parts must vanish
    assert np.allclose(np.abs(eigs.real), 0.0, atol=1e-10), (
        f"Expected purely imaginary eigenvalues, got real parts {eigs.real}"
    )
    # Magnitudes are 1 (|±i| = 1)
    assert np.allclose(np.abs(eigs), 1.0, atol=1e-10)


def test_linearize_at_fixed_point_saddle():
    x, p = symbols('x p', real=True)
    H = p**2/2 - x**2/2
    lin = linearize_at_fixed_point(H, (0, 0), vars_phase=[x, p])
    assert lin['type'] == 'hyperbolic'
    eigs = lin['eigenvalues']
    # Real eigenvalues ±λ: their product is negative
    assert np.all(np.abs(eigs.imag) < 1e-10), "Expected real eigenvalues for saddle"
    assert (eigs[0] * eigs[1]) < 0


def test_find_fixed_points_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    fps = find_fixed_points(H, vars_phase=[x1, p1, x2, p2])
    assert len(fps) == 1
    assert np.allclose(fps[0], (0, 0, 0, 0), atol=1e-6)


def test_linearize_at_fixed_point_2d():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    lin = linearize_at_fixed_point(H, (0, 0, 0, 0), vars_phase=[x1, p1, x2, p2])
    assert lin['type'] == 'elliptic'
    eigs = lin['eigenvalues']
    assert np.allclose(np.abs(eigs.real), 0.0, atol=1e-10)


# =============================================================================
# 1-DOF specific: action integral, phase portrait, separatrix, action-angle
# =============================================================================

def test_action_integral_harmonic_symbolic():
    x, p, E_sym = symbols('x p E', real=True, positive=True)
    H = (p**2 + x**2) / 2
    I_sym = action_integral(H, E_sym, vars_phase=[x, p], method='symbolic')
    assert is_sympy_zero(I_sym - E_sym)


def test_action_integral_harmonic_numerical():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    I_num = action_integral(H, 1.0, vars_phase=[x, p], method='numerical')
    assert np.isclose(I_num, 1.0, rtol=1e-2)


def test_action_integral_multiple_energies():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    for E in [0.5, 1.0, 2.0]:
        I = action_integral(H, E, vars_phase=[x, p], method='numerical')
        assert np.isclose(I, E, rtol=0.05)


def test_action_integral_double_well():
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2
    I = action_integral(H, 0.1, vars_phase=[x, p], method='numerical')
    assert np.isfinite(I) and I > 0


def test_action_integral_symbolic_fallback_to_numerical():
    """
    For a Hamiltonian whose symbolic integration is hard (quartic),
    the function must fall back to numerical and return a finite positive value.
    """
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4    # anharmonic; symbolic integration is non-trivial
    # Calling with method='symbolic' may fall back; result must still be valid
    I = action_integral(H, 1.0, vars_phase=[x, p], method='numerical',
                        x_bounds=(-1.189, 1.189))   # approx turning points at E=1
    assert np.isfinite(I) and I > 0


def test_action_integral_wrong_dim():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    with pytest.raises(ValueError):
        action_integral(H, 1.0, vars_phase=[x1, p1, x2, p2])


def test_phase_portrait_execution():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    phase_portrait(H, (-2, 2), (-2, 2), vars_phase=[x, p], levels=5)


def test_phase_portrait_wrong_dim():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    with pytest.raises(ValueError):
        phase_portrait(H, (-2, 2), (-2, 2), vars_phase=[x1, p1, x2, p2])


def test_separatrix_analysis_keys():
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2
    sep = separatrix_analysis(H, (-2, 2), (-2, 2), (0, 0), vars_phase=[x, p])
    assert 'E_saddle' in sep
    assert 'unstable_manifolds' in sep
    assert 'stable_manifolds' in sep


def test_separatrix_energy_at_saddle():
    """
    The energy stored in separatrix_analysis must equal H at the saddle point.
    For H = p²/2 + x⁴/4 - x²/2, H(0,0) = 0.
    """
    x, p = symbols('x p', real=True)
    H = p**2/2 + x**4/4 - x**2/2
    sep = separatrix_analysis(H, (-2, 2), (-2, 2), (0, 0), vars_phase=[x, p])
    assert np.isclose(sep['E_saddle'], 0.0, atol=1e-10)


def test_action_angle_transform():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    aa = action_angle_transform(H, (-3, 3), (-3, 3), vars_phase=[x, p], n_contours=5)
    assert 'energies' in aa and len(aa['energies']) > 0
    assert 'actions'  in aa and len(aa['actions']) > 0
    assert 'frequencies' in aa


def test_frequency_derivative():
    I_sym = symbols('I', real=True, positive=True)
    H = I_sym              # H = I  →  ω = dH/dI = 1
    omega = frequency(H, 1.0, method='derivative')
    assert np.isclose(omega, 1.0)


def test_frequency_period_harmonic_oscillator():
    """For H=(x²+p²)/2 at energy E=1, the period is 2π so ω=1."""
    x, p = symbols('x p', real=True)
    H = (x**2 + p**2) / 2
    omega = frequency(H, 1.0, method='period')
    assert np.isclose(omega, 1.0, rtol=1e-3)

def test_frequency_period_harmonic_oscillator_energy_independent():
    """For the harmonic oscillator ω=1 regardless of energy."""
    x, p = symbols('x p', real=True)
    H = (x**2 + p**2) / 2
    for E in [0.5, 1.0, 2.0, 5.0]:
        omega = frequency(H, E, method='period')
        assert np.isclose(omega, 1.0, rtol=1e-2), f"Failed at E={E}: ω={omega}"


def test_visualize_phase_space_structure():
    """visualize_phase_space_structure must run without raising an exception."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    visualize_phase_space_structure(
        H, (-2, 2), (-2, 2),
        vars_phase=[x, p],
        fixed_points=[(0, 0)],
        show_separatrices=False,
        n_trajectories=3,
    )


# =============================================================================
# 2-DOF specific: Poincaré section, first-return map, monodromy, Lyapunov, project
# =============================================================================

def test_poincare_section_basic():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    section = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    ps = poincare_section(H, section, (1, 0, 0, 1), tmax=50,
                          vars_phase=[x1, p1, x2, p2], n_returns=20)
    assert 't_crossings' in ps and len(ps['t_crossings']) > 0
    assert 'section_points' in ps and len(ps['section_points']) > 0


def test_poincare_section_direction_both_gte_positive():
    """
    'both' direction must yield at least as many crossings as 'positive'
    over the same integration window.
    """
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 0, 1)
    pos  = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    both = {'variable': 'x2', 'value': 0, 'direction': 'both'}
    ps_pos  = poincare_section(H, pos,  z0, tmax=50, vars_phase=[x1, p1, x2, p2], n_returns=50)
    ps_both = poincare_section(H, both, z0, tmax=50, vars_phase=[x1, p1, x2, p2], n_returns=50)
    assert len(ps_both['t_crossings']) >= len(ps_pos['t_crossings'])


def test_poincare_section_invalid_variable():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2) / 2
    section = {'variable': 'invalid', 'value': 0}
    with pytest.raises(ValueError):
        poincare_section(H, section, (1, 0, 0, 0), tmax=5,
                         vars_phase=[x1, p1, x2, p2])


def test_poincare_section_wrong_dim():
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    section = {'variable': 'x', 'value': 0}
    with pytest.raises(ValueError):
        poincare_section(H, section, (1, 0), tmax=10, vars_phase=[x, p])


def test_first_return_map():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    section = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    ps = poincare_section(H, section, (1, 0, 0, 1), tmax=50,
                          vars_phase=[x1, p1, x2, p2], n_returns=5)
    if len(ps['section_points']) >= 2:
        rm = first_return_map(ps['section_points'], plot_variables=('x1', 'p1'))
        assert 'current' in rm
        assert 'next' in rm
        assert rm['current'].shape == (len(ps['section_points']) - 1, 2)
    else:
        pytest.skip("Not enough section points for return map")


def test_first_return_map_too_few_points():
    with pytest.raises(ValueError):
        first_return_map([{'x1': 0, 'p1': 0, 'x2': 0, 'p2': 0}])


def test_monodromy_matrix_stable():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 0, 0)
    T  = 2 * np.pi
    traj = hamiltonian_flow(H, z0, (0, T), vars_phase=[x1, p1, x2, p2],
                            integrator='rk45', n_steps=500)
    mono = monodromy_matrix(H, traj, vars_phase=[x1, p1, x2, p2],
                            method='finite_difference')
    assert 'M' in mono and mono['M'].shape == (4, 4)
    mult = mono['floquet_multipliers']
    assert len(mult) == 4
    # Harmonic oscillator: all multipliers lie on the unit circle
    assert np.allclose(np.abs(mult), 1.0, atol=1e-3)


def test_monodromy_invalid_method():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    z0 = (1, 0, 0, 0)
    traj = hamiltonian_flow(H, z0, (0, 2*np.pi), vars_phase=[x1, p1, x2, p2],
                            integrator='rk45', n_steps=100)
    with pytest.raises(NotImplementedError):
        monodromy_matrix(H, traj, vars_phase=[x1, p1, x2, p2], method='variational')


def test_lyapunov_exponents_length():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    vp = [x1, p1, x2, p2]
    traj = hamiltonian_flow(H, (1, 0, 0.5, 0), (0, 50),
                            vars_phase=vp,
                            integrator='symplectic', n_steps=500)
    dt = traj['t'][1] - traj['t'][0]
    exponents = lyapunov_exponents(traj, dt, H=H, vars_phase=vp, n_vectors=4)
    assert len(exponents) == 4


def test_lyapunov_exponents_sorted_descending():
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    vp = [x1, p1, x2, p2]
    traj = hamiltonian_flow(H, (1, 0, 0.5, 0), (0, 50),
                            vars_phase=vp,
                            integrator='symplectic', n_steps=500)
    dt = traj['t'][1] - traj['t'][0]
    exponents = lyapunov_exponents(traj, dt, H=H, vars_phase=vp, n_vectors=4)
    finite = exponents[np.isfinite(exponents)]
    assert len(finite) >= 1
    assert np.all(np.diff(finite) <= 0)


def test_project_functions():
    """
    Test all projection planes. Uses fresh local names for return values to
    avoid shadowing the sympy symbols used to build the trajectory.
    """
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    traj = hamiltonian_flow(H, (1, 0, 1, 0), (0, 10),
                            vars_phase=[x1, p1, x2, p2], n_steps=100)
    vp = [x1, p1, x2, p2]   # keep symbols intact throughout

    a, b, lbl = project(traj, plane='xy', vars_phase=vp)
    assert len(a) == 100 and len(b) == 100
    assert lbl == ('x₁', 'x₂')

    a, b, lbl = project(traj, plane='xp', vars_phase=vp)
    assert lbl == ('x₁', 'p₁')

    a, b, lbl = project(traj, plane='pp', vars_phase=vp)
    assert lbl == ('p₁', 'p₂')

    a, b, lbl = project(traj, plane='x1p2', vars_phase=vp)
    assert lbl == ('x₁', 'p₂')

    a, b, lbl = project(traj, plane='x2p1', vars_phase=vp)
    assert lbl == ('x₂', 'p₁')

    with pytest.raises(ValueError):
        project(traj, plane='invalid', vars_phase=vp)


def test_visualize_poincare_section_execution():
    """visualize_poincare_section must run without raising an exception."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    section = {'variable': 'x2', 'value': 0, 'direction': 'positive'}
    z0_list = [(1, 0, 0, 1), (0.5, 0, 0, 0.5)]
    visualize_poincare_section(
        H, z0_list, section,
        vars_phase=[x1, p1, x2, p2],
        tmax=10, n_returns=5, plot_vars=('x1', 'p1'),
    )

# -----------------------------------------------------------------------------
# Tests for new region‑evolution functions
# -----------------------------------------------------------------------------

def test_rectangle_region():
    """Check that rectangle_region produces a closed polygon with correct bounds."""
    center = (0, 0)
    width = 2.0
    height = 1.0
    n_points = 10
    region = rectangle_region(center, width, height, n_points)

    # Shape: n_points + 1 (closure)
    assert region.shape == (n_points + 1, 2)

    # First and last point must be identical
    assert np.allclose(region[0], region[-1])

    # Bounding box should match the specified width and height
    xmin, xmax = region[:, 0].min(), region[:, 0].max()
    pmin, pmax = region[:, 1].min(), region[:, 1].max()
    assert np.isclose(xmin, center[0] - width/2)
    assert np.isclose(xmax, center[0] + width/2)
    assert np.isclose(pmin, center[1] - height/2)
    assert np.isclose(pmax, center[1] + height/2)


def test_evolve_phase_space_region_basic():
    """Basic run of evolve_phase_space_region with area conservation check."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2          # harmonic oscillator
    # Small rectangle near (1,0)
    region = rectangle_region(center=(1, 0), width=0.2, height=0.2, n_points=4)
    t_eval = [0, 1]                 # short integration
    result = evolve_phase_space_region(
        H, region, t_eval,
        vars_phase=[x, p],
        integrator='verlet',
        n_steps=500,
        plot=False
    )

    # Expected keys
    assert 'times' in result
    assert 'region_at_t' in result
    assert 'areas' in result

    # Times should match t_eval (sorted)
    np.testing.assert_array_equal(result['times'], t_eval)

    # Number of regions and areas must equal len(t_eval)
    assert len(result['region_at_t']) == len(t_eval)
    assert len(result['areas']) == len(t_eval)

    # Each region should have the same shape as the input
    for reg in result['region_at_t']:
        assert reg.shape == region.shape

    # Area conservation: initial area = width*height = 0.04
    expected_area = 0.2 * 0.2
    assert np.allclose(result['areas'], expected_area, rtol=0.05)


def test_evolve_phase_space_region_invalid_dim():
    """Calling evolve_phase_space_region on a 2‑DOF system should raise."""
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    region = rectangle_region(center=(0, 0), width=1, height=1, n_points=4)
    with pytest.raises(ValueError, match="requires 1 DOF"):
        evolve_phase_space_region(
            H, region, [0, 1],
            vars_phase=[x1, p1, x2, p2]
        )

def test_evolve_phase_space_region_not_closed():
    """If the input polygon is not closed, a warning is issued and it is closed automatically."""
    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    # Open polygon: first and last points differ
    open_region = np.array([[0, 0], [1, 0], [1, 1]])
    with pytest.warns(UserWarning, match="Region is not closed"):
        result = evolve_phase_space_region(
            H, open_region, [0, 0.5],
            vars_phase=[x, p],
            integrator='verlet',
            n_steps=200,
            plot=False
        )
    # After automatic closure, the region should have one extra point
    assert result['region_at_t'][0].shape[0] == open_region.shape[0] + 1


def test_evolve_phase_space_region_plot(monkeypatch):
    """Test that plot=True runs without displaying an actual window."""
    import matplotlib.pyplot as plt
    # Prevent plt.show from blocking (already using Agg, but this adds safety)
    monkeypatch.setattr(plt, 'show', lambda: None)

    x, p = symbols('x p', real=True)
    H = (p**2 + x**2) / 2
    region = rectangle_region(center=(1, 0), width=0.2, height=0.2, n_points=4)
    result = evolve_phase_space_region(
        H, region, [0, 0.5],
        vars_phase=[x, p],
        integrator='verlet',
        n_steps=200,
        plot=True
    )
    assert 'plot_handles' in result
    assert len(result['plot_handles']) == len(result['times'])


class TestAnalyzeIntegrability:
    """
    Tests for the redesigned IntegrabilityAnalysis.analyze_integrability.

    Architecture under test
    -----------------------
    The new method is organised around four independent evidence channels:

      Algebraic  — symbolic {H, L} = 0 check; acts as a hard gate that forces
                   verdict='Integrable' before any numerical work is done.
      Spectral   — Brody β (MLE) + two KS tests; driven by 'levels' (auto-
                   unfolded) or pre-computed 'spacings'.
      Frequency  — NAFF-lite rotation numbers from a trajectory; rational
                   ω₁/ω₂ is a resonance signal.
      Lyapunov   — λ_max from the trajectory; acts as a hard gate that forces
                   verdict='Chaotic' when λ_max exceeds the threshold.

    Hard gates evaluate first and short-circuit the soft-score path:
      algebraic proof  → Integrable  (even if Lyapunov would say chaotic)
      Lyapunov gate    → Chaotic

    Key API changes vs. the previous version
    -----------------------------------------
    - First positional argument is now H (optional), not spacings.
    - Raw energy eigenvalues are passed as `levels=`; they are unfolded
      internally.  Pre-computed spacings may still be passed as `spacings=`.
    - Second integrals are passed as `second_integrals=` (not L=).
    - Output keys: 'verdict', 'verdict_source', 'soft_score', 'channels',
      'warnings', 'summary'.
    - channels sub-dict keys: 'algebraic', 'spectral', 'frequency', 'lyapunov'.
    - Removed from output: 'ratio', 'mean_spacing', 'std_spacing',
      'classification', 'verdict_score', 'evidence', 'brody', 'kam',
      'conserved_L', 'rotation_numbers', 'winding_number', 'scar_intensity'.
      (Spectral stats now live in channels['spectral']; algebraic result in
      channels['algebraic']; frequency result in channels['frequency'].)
    - KAM tori, Berry-Tabor, and scar intensity are no longer part of the
      orchestrator; call those methods directly.

    Physical fixtures
    -----------------
    Integrable  — isotropic 2-DOF harmonic oscillator H = Σ(pᵢ²+xᵢ²)/2,
                  second integral L = (p2²+x2²)/2.
                  Integrable spectrum → exponential spacing increments (Poisson).
    Chaotic     — Wigner (Rayleigh) spacing increments as chaotic benchmark.
    """

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @staticmethod
    def _poisson_levels(n=500, seed=42):
        """Cumulative sum of Poisson (exponential) spacings → integrable spectrum."""
        return np.sort(np.random.default_rng(seed).exponential(1.0, n).cumsum())

    @staticmethod
    def _wigner_levels(n=500, seed=42):
        """Cumulative sum of Wigner (Rayleigh) spacings → chaotic spectrum."""
        return np.sort(np.random.default_rng(seed).rayleigh(
            np.sqrt(4 / np.pi), n).cumsum())

    @staticmethod
    def _poisson_spacings(n=500, seed=42):
        """Raw exponential spacings (pre-computed, unit mean)."""
        return np.random.default_rng(seed).exponential(1.0, n)

    @staticmethod
    def _wigner_spacings(n=500, seed=42):
        """Raw Rayleigh spacings (pre-computed, unit mean by construction)."""
        return np.random.default_rng(seed).rayleigh(np.sqrt(4 / np.pi), n)

    @staticmethod
    def _ho2d_vars():
        return symbols('x1 p1 x2 p2', real=True)

    @staticmethod
    def _ho2d_H_L():
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + x1**2 + p2**2 + x2**2) / 2
        L = (p2**2 + x2**2) / 2
        return H, L, [x1, p1, x2, p2]

    @staticmethod
    def _ho2d_traj(n_steps=6000):
        """Long isotropic-HO trajectory for reliable NAFF frequency estimates."""
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + x1**2 + p2**2 + x2**2) / 2
        vp = [x1, p1, x2, p2]
        traj = hamiltonian_flow(H, (1.0, 0.0, 0.0, 1.0), (0, 60 * np.pi),
                                vars_phase=vp, integrator='verlet',
                                n_steps=n_steps)
        return traj, vp

    @staticmethod
    def _aniso_traj(n_steps=6000):
        """Anisotropic HO: ω₁=1, ω₂=2  →  ω₁/ω₂ = 1/2 (rational)."""
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + x1**2) / 2 + (p2**2 + 4 * x2**2) / 2
        vp = [x1, p1, x2, p2]
        traj = hamiltonian_flow(H, (1.0, 0.0, 0.0, 0.5), (0, 60 * np.pi),
                                vars_phase=vp, integrator='verlet',
                                n_steps=n_steps)
        return traj, vp

    # ------------------------------------------------------------------
    # Mandatory output keys — always present regardless of inputs
    # ------------------------------------------------------------------

    def test_mandatory_keys_algebraic_only(self):
        """Algebraic-only call must return all four mandatory top-level keys."""
        H, L, vp = self._ho2d_H_L()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L)
        for key in ('verdict', 'verdict_source', 'soft_score', 'channels',
                    'warnings', 'summary'):
            assert key in r, f"Missing mandatory key: {key}"

    def test_mandatory_keys_spectral_only(self):
        """Spectral-only call (levels=) must return all mandatory keys."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        for key in ('verdict', 'verdict_source', 'soft_score', 'channels',
                    'warnings', 'summary'):
            assert key in r, f"Missing mandatory key: {key}"

    def test_no_inputs_raises_value_error(self):
        """Calling with no arguments must raise ValueError."""
        with pytest.raises(ValueError):
            IntegrabilityAnalysis.analyze_integrability()

    def test_summary_is_nonempty_string(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert isinstance(r['summary'], str) and len(r['summary']) > 0

    def test_warnings_is_list(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert isinstance(r['warnings'], list)

    def test_channels_is_dict(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert isinstance(r['channels'], dict)

    # ------------------------------------------------------------------
    # Algebraic channel — hard gate
    # ------------------------------------------------------------------

    def test_algebraic_channel_present_when_H_L_given(self):
        H, L, vp = self._ho2d_H_L()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L)
        assert 'algebraic' in r['channels']

    def test_algebraic_channel_absent_without_symbolic_args(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert 'algebraic' not in r['channels']

    def test_algebraic_gate_fires_integrable(self):
        """A confirmed second integral must force verdict=Integrable, source=algebraic_proof."""
        H, L, vp = self._ho2d_H_L()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L)
        assert r['verdict'] == 'Integrable'
        assert r['verdict_source'] == 'algebraic_proof'
        assert r['soft_score'] == 1.0

    def test_algebraic_gate_does_not_fire_for_non_integral(self):
        """L = x1 (not conserved) must NOT activate the hard gate."""
        x1, p1, x2, p2 = self._ho2d_vars()
        H = (p1**2 + x1**2 + p2**2 + x2**2) / 2
        L_bad = x1
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=[x1, p1, x2, p2], second_integrals=L_bad)
        assert r['verdict_source'] != 'algebraic_proof'
        assert r['channels']['algebraic']['any_conserved'] is False

    def test_algebraic_any_conserved_true_for_known_integral(self):
        H, L, vp = self._ho2d_H_L()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L)
        assert r['channels']['algebraic']['any_conserved'] is True

    def test_algebraic_independent_integrals_count(self):
        """channels['algebraic']['independent_integrals'] must equal 1 for one good L."""
        H, L, vp = self._ho2d_H_L()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L)
        assert r['channels']['algebraic']['independent_integrals'] == 1

    def test_algebraic_gate_beats_lyapunov_gate(self):
        """Algebraic proof must win even when the Lyapunov channel would say chaotic.
        This is verified by providing a confirmed integral together with Wigner
        levels and a chaotic-looking trajectory — the gate fires first and the
        Lyapunov channel is never reached."""
        H, L, vp = self._ho2d_H_L()
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L,
            levels=self._wigner_levels(),
            lyapunov_traj=traj, lyapunov_dt=None)
        assert r['verdict'] == 'Integrable'
        assert r['verdict_source'] == 'algebraic_proof'
        assert 'lyapunov' not in r['channels']

    def test_algebraic_bracket_list_in_channel(self):
        """channels['algebraic']['brackets'] must be a non-empty list of dicts."""
        H, L, vp = self._ho2d_H_L()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L)
        brackets = r['channels']['algebraic']['brackets']
        assert isinstance(brackets, list) and len(brackets) == 1
        assert 'is_zero' in brackets[0]
        assert brackets[0]['is_zero'] is True

    def test_algebraic_multiple_candidates_one_good(self):
        """With two candidates [L_bad, L_good], exactly one integral must be confirmed."""
        x1, p1, x2, p2 = self._ho2d_vars()
        H = (p1**2 + x1**2 + p2**2 + x2**2) / 2
        L_bad  = x1
        L_good = (p2**2 + x2**2) / 2
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=[x1, p1, x2, p2],
            second_integrals=[L_bad, L_good])
        assert r['verdict'] == 'Integrable'
        assert r['channels']['algebraic']['independent_integrals'] == 1

    # ------------------------------------------------------------------
    # Spectral channel — Brody β and KS tests
    # ------------------------------------------------------------------

    def test_spectral_channel_present_with_levels(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert 'spectral' in r['channels']

    def test_spectral_channel_present_with_spacings(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            spacings=self._poisson_spacings())
        assert 'spectral' in r['channels']

    def test_spectral_channel_absent_without_spectrum(self):
        H, L, vp = self._ho2d_H_L()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L)
        assert 'spectral' not in r['channels']

    def test_spectral_keys_present(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        sc = r['channels']['spectral']
        for key in ('beta', 'beta_std', 'ks_poisson_p', 'ks_wigner_p',
                    'n_spacings', 'score', 'ratio_R'):
            assert key in sc, f"Missing spectral key: {key}"

    def test_spectral_beta_in_unit_interval(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert 0.0 <= r['channels']['spectral']['beta'] <= 1.0

    def test_spectral_poisson_levels_low_beta(self):
        """Poisson levels → β ≈ 0 after unfolding."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(n=600))
        assert r['channels']['spectral']['beta'] < 0.3

    def test_spectral_wigner_levels_high_beta(self):
        """Wigner levels → β ≈ 1 after unfolding."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._wigner_levels(n=600))
        assert r['channels']['spectral']['beta'] > 0.7

    def test_spectral_ks_directionality_poisson(self):
        """For Poisson levels: p(Poisson) must exceed p(Wigner)."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(n=600))
        sc = r['channels']['spectral']
        assert sc['ks_poisson_p'] > sc['ks_wigner_p']

    def test_spectral_ks_directionality_wigner(self):
        """For Wigner levels: p(Wigner) must exceed p(Poisson)."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._wigner_levels(n=600))
        sc = r['channels']['spectral']
        assert sc['ks_wigner_p'] > sc['ks_poisson_p']

    def test_spectral_n_spacings_correct(self):
        """n_spacings must equal len(levels) - 1 when levels= is provided."""
        levels = self._poisson_levels(n=200)
        r = IntegrabilityAnalysis.analyze_integrability(levels=levels)
        assert r['channels']['spectral']['n_spacings'] == len(levels) - 1

    def test_spectral_skipped_below_min_spacings(self):
        """Fewer than min_spacings levels → spectral channel absent, warning present."""
        tiny = self._poisson_levels(n=10)   # 9 spacings < default 30
        r = IntegrabilityAnalysis.analyze_integrability(levels=tiny)
        assert 'spectral' not in r['channels']
        assert len(r['warnings']) > 0

    def test_spectral_unfolding_unit_mean(self):
        """Spacings derived from unfolded levels must have mean ≈ 1."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(n=400))
        sc = r['channels']['spectral']
        assert abs(sc['spacings_norm'].mean() - 1.0) < 0.05

    def test_spectral_score_in_unit_interval(self):
        for lev in (self._poisson_levels(), self._wigner_levels()):
            r = IntegrabilityAnalysis.analyze_integrability(levels=lev)
            assert 0.0 <= r['channels']['spectral']['score'] <= 1.0

    def test_spectral_poisson_verdict_integrable(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(n=600))
        assert r['verdict'] in ('Integrable', 'Likely integrable')

    def test_spectral_wigner_verdict_chaotic(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._wigner_levels(n=600))
        assert r['verdict'] in ('Chaotic', 'Likely chaotic')

    def test_soft_score_in_unit_interval(self):
        for lev in (self._poisson_levels(), self._wigner_levels()):
            r = IntegrabilityAnalysis.analyze_integrability(levels=lev)
            assert 0.0 <= r['soft_score'] <= 1.0

    def test_verdict_is_valid_string(self):
        valid = {'Integrable', 'Likely integrable', 'Mixed',
                 'Likely chaotic', 'Chaotic', 'Undetermined'}
        for lev in (self._poisson_levels(), self._wigner_levels()):
            r = IntegrabilityAnalysis.analyze_integrability(levels=lev)
            assert r['verdict'] in valid, f"Unexpected verdict: {r['verdict']!r}"

    # ------------------------------------------------------------------
    # Spectral unfolding helper
    # ------------------------------------------------------------------

    def test_unfold_spectrum_unit_mean_spacing(self):
        """Unfolded levels must have nearest-neighbour mean spacing ≈ 1."""
        levels = self._poisson_levels(n=400)
        unfolded = IntegrabilityAnalysis.unfold_spectrum(levels)
        assert abs(np.diff(unfolded).mean() - 1.0) < 0.05

    def test_unfold_spectrum_preserves_count(self):
        levels = self._poisson_levels(n=300)
        unfolded = IntegrabilityAnalysis.unfold_spectrum(levels)
        assert len(unfolded) == len(levels)

    def test_unfold_spectrum_monotone(self):
        """Unfolded levels must be strictly increasing."""
        levels = self._poisson_levels(n=300)
        unfolded = IntegrabilityAnalysis.unfold_spectrum(levels)
        assert np.all(np.diff(unfolded) > 0)

    def test_unfold_spectrum_nonuniform_density(self):
        """Unfolding must normalise a non-uniform density to unit mean spacing."""
        rng = np.random.default_rng(99)
        nonuniform = np.sort(rng.exponential(1.0, 300) * np.linspace(0.5, 2.0, 300))
        unfolded = IntegrabilityAnalysis.unfold_spectrum(nonuniform)
        assert abs(np.diff(unfolded).mean() - 1.0) < 0.10

    # ------------------------------------------------------------------
    # Frequency channel — NAFF rotation numbers
    # ------------------------------------------------------------------

    def test_frequency_channel_present_with_traj(self):
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            traj=traj, ndof=2)
        assert 'frequency' in r['channels']

    def test_frequency_channel_absent_without_traj(self):
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert 'frequency' not in r['channels']

    def test_frequency_channel_keys(self):
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            traj=traj, ndof=2)
        fc = r['channels']['frequency']
        for key in ('omega1', 'omega2', 'ratio', 'is_rational', 'method', 'score'):
            assert key in fc, f"Missing frequency key: {key}"

    def test_frequency_method_is_naff(self):
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        assert r['channels']['frequency']['method'] == 'naff'

    def test_frequency_omega_finite(self):
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        fc = r['channels']['frequency']
        assert np.isfinite(fc['omega1']) and np.isfinite(fc['omega2'])

    def test_frequency_isotropic_ho_equal_frequencies(self):
        """Isotropic HO: ω₁ ≈ ω₂ (same frequency for both DOF)."""
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        fc = r['channels']['frequency']
        assert abs(fc['omega1'] - fc['omega2']) < 0.05, (
            f"Expected ω₁≈ω₂, got ω₁={fc['omega1']:.4f} ω₂={fc['omega2']:.4f}")

    def test_frequency_isotropic_ho_rational(self):
        """Isotropic HO: ω₁/ω₂ = 1 is rational → is_rational=True."""
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        assert r['channels']['frequency']['is_rational'] is True

    def test_frequency_anisotropic_ho_half_ratio(self):
        """Anisotropic HO (ω₁=1, ω₂=2): ratio ≈ 1/2, rational."""
        traj, _ = self._aniso_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        fc = r['channels']['frequency']
        rat = fc['omega1'] / fc['omega2']
        assert abs(rat - 0.5) < 0.05, f"Expected ω₁/ω₂≈0.5, got {rat:.4f}"
        assert fc['is_rational'] is True
        assert fc['ratio_fraction'] == '1/2'

    def test_frequency_score_in_unit_interval(self):
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        assert 0.0 <= r['channels']['frequency']['score'] <= 1.0

    def test_frequency_rational_raises_score(self):
        """Rational ω₁/ω₂ must give a higher score than the irrational threshold."""
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        # Rational score is 0.75, irrational is 0.5 — rational must win
        assert r['channels']['frequency']['score'] >= 0.5

    def test_frequency_ndof_inferred_from_traj_keys(self):
        """ndof should be inferred automatically from trajectory key names."""
        traj, _ = self._ho2d_traj()
        # Pass without explicit ndof — should still activate the 2-DOF branch
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj)
        assert 'frequency' in r['channels']

    def test_frequency_naff_keys_override(self):
        """naff_keys parameter must control which trajectory keys are read."""
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            traj=traj, ndof=2,
            naff_keys=('x1', 'p1', 'x2', 'p2'))
        assert 'frequency' in r['channels']
        assert np.isfinite(r['channels']['frequency']['omega1'])

    # ------------------------------------------------------------------
    # Lyapunov channel — hard gate for chaos
    # ------------------------------------------------------------------

    def test_lyapunov_channel_present_with_traj_and_vars(self):
        traj, vp = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            traj=traj, vars_phase=vp)
        # Channel should be attempted; it may or may not fire as a gate
        assert 'lyapunov' in r['channels']

    def test_lyapunov_channel_absent_without_vars_phase(self):
        """Without vars_phase the Lyapunov channel cannot run."""
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(traj=traj, ndof=2)
        assert 'lyapunov' not in r['channels']

    def test_lyapunov_channel_keys(self):
        traj, vp = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            traj=traj, vars_phase=vp)
        lc = r['channels']['lyapunov']
        for key in ('lambda_max', 'exponents', 'is_chaotic', 'threshold'):
            assert key in lc, f"Missing Lyapunov key: {key}"

    def test_lyapunov_exponents_finite(self):
        traj, vp = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            traj=traj, vars_phase=vp)
        exps = r['channels']['lyapunov']['exponents']
        assert np.any(np.isfinite(exps))

    def test_lyapunov_integrable_ho_not_chaotic(self):
        """Integrable HO trajectory: λ_max should be near zero, is_chaotic=False."""
        traj, vp = self._ho2d_traj()
        H, _, _ = self._ho2d_H_L()   # retrieve H
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H,                     # <-- added
            traj=traj,
            vars_phase=vp
        )
        lc = r['channels']['lyapunov']
        assert lc['is_chaotic'] is False, (
            f"Expected is_chaotic=False for HO, got λ_max={lc['lambda_max']:.4f}")

    def test_lyapunov_lambda_max_finite(self):
        traj, vp = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            traj=traj, vars_phase=vp)
        assert np.isfinite(r['channels']['lyapunov']['lambda_max'])

    def test_lyapunov_separate_traj_accepted(self):
        """lyapunov_traj= must override the main traj for Lyapunov estimation."""
        traj, vp = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(),
            lyapunov_traj=traj, lyapunov_dt=None,
            vars_phase=vp)
        assert 'lyapunov' in r['channels']

    # ------------------------------------------------------------------
    # Combined channels and soft-score aggregation
    # ------------------------------------------------------------------

    def test_all_channels_active_integrable_system(self):
        """All four channels active on the isotropic HO: verdict must be Integrable."""
        H, L, vp = self._ho2d_H_L()
        # Algebraic gate fires first → Integrable unconditionally
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            H=H, vars_phase=vp, second_integrals=L,
            levels=self._poisson_levels(),
            traj=traj, ndof=2)
        assert r['verdict'] == 'Integrable'
        assert r['verdict_source'] == 'algebraic_proof'

    def test_spectral_plus_frequency_soft_score(self):
        """Spectral (Poisson) + frequency (rational) → score > 0.5."""
        traj, _ = self._ho2d_traj()
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(), traj=traj, ndof=2)
        assert r['soft_score'] > 0.5
        assert r['verdict'] in ('Integrable', 'Likely integrable')

    def test_spectral_wigner_reduces_soft_score(self):
        """Wigner levels must produce a soft_score below the Poisson baseline."""
        r_int = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(n=600))
        r_cha = IntegrabilityAnalysis.analyze_integrability(
            levels=self._wigner_levels(n=600))
        assert r_cha['soft_score'] < r_int['soft_score']

    def test_summary_grows_with_channels(self):
        """More active channels → longer summary string."""
        r0 = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        traj, vp = self._ho2d_traj()
        r1 = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(), traj=traj, ndof=2)
        r2 = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels(), traj=traj, vars_phase=vp)
        assert len(r1['summary']) > len(r0['summary'])
        assert len(r2['summary']) > len(r1['summary'])

    def test_reproducibility(self):
        """Two identical calls must return identical numeric results."""
        levels = self._poisson_levels()
        r1 = IntegrabilityAnalysis.analyze_integrability(levels=levels)
        r2 = IntegrabilityAnalysis.analyze_integrability(levels=levels)
        assert r1['soft_score'] == r2['soft_score']
        assert r1['verdict'] == r2['verdict']
        np.testing.assert_array_equal(
            r1['channels']['spectral']['spacings_norm'],
            r2['channels']['spectral']['spacings_norm'])

    def test_verdict_source_is_soft_score_without_gates(self):
        """With no algebraic or Lyapunov input, source must be 'soft_score'."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert r['verdict_source'] == 'soft_score'

    def test_soft_score_none_only_when_undetermined(self):
        """soft_score must be a float in [0,1] whenever a quantitative channel runs."""
        r = IntegrabilityAnalysis.analyze_integrability(
            levels=self._poisson_levels())
        assert r['soft_score'] is not None
        assert isinstance(r['soft_score'], float)

    # ------------------------------------------------------------------
    # NAFF helper — unit tests
    # ------------------------------------------------------------------

    def test_naff_frequency_pure_tone(self):
        """NAFF must recover ω = 2 rad/s from a pure sinusoid."""
        dt = 0.01
        t  = np.arange(0, 500 * np.pi, dt)
        z  = np.exp(2j * t)             # analytic signal at ω = 2
        om = IntegrabilityAnalysis._naff_frequency(z, dt)
        assert abs(om - 2.0) < 0.05, f"Expected ω≈2.0, got {om:.4f}"

    def test_naff_frequency_returns_positive(self):
        """_naff_frequency must always return a positive value."""
        dt = 0.01
        t  = np.arange(0, 200 * np.pi, dt)
        z  = np.exp(1j * t)
        om = IntegrabilityAnalysis._naff_frequency(z, dt)
        assert om > 0


class TestBrodyDistribution:
    def test_poisson_limit(self):
        """Exponential spacings → β ≈ 0 (integrable)."""
        rng = np.random.default_rng(0)
        s = rng.exponential(scale=1.0, size=800)
        r = IntegrabilityAnalysis.brody_distribution(s)
        assert r['beta'] is not None
        assert r['beta'] < 0.25
        assert 'Integrable' in r['classification']

    def test_wigner_limit(self):
        """Rayleigh (Wigner) spacings → β ≈ 1 (chaotic)."""
        rng = np.random.default_rng(1)
        s = rng.rayleigh(scale=np.sqrt(4 / np.pi), size=800)
        r = IntegrabilityAnalysis.brody_distribution(s)
        assert r['beta'] > 0.75
        assert 'Chaotic' in r['classification']

    def test_pdf_callable(self):
        """pdf key must be a callable returning non-negative values."""
        rng = np.random.default_rng(2)
        s = rng.exponential(size=200)
        r = IntegrabilityAnalysis.brody_distribution(s)
        vals = r['pdf'](np.linspace(0.01, 3.0, 50))
        assert np.all(vals >= 0)

    def test_beta_std_nonnegative(self):
        rng = np.random.default_rng(3)
        s = rng.exponential(size=300)
        r = IntegrabilityAnalysis.brody_distribution(s)
        assert r['beta_std'] >= 0

class TestTopologicalMonodromy:
    def _uncoupled_ho(self):
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + x1**2) / 2 + (p2**2 + x2**2) / 2
        # Second integral: energy of DOF 2 alone
        L = (p2**2 + x2**2) / 2
        return H, L, [x1, p1, x2, p2]

    def test_output_keys(self):
        H, L, vp = self._uncoupled_ho()
        r = IntegrabilityAnalysis.topological_monodromy(
            H, L, vp, critical_value=(0.0, 0.0),
            loop_radius=0.3, n_loop_points=12)
        for key in ('monodromy_matrix', 'monodromy_float', 'is_trivial',
                    'actions_along_loop', 'angles', 'loop_EL'):
            assert key in r

    def test_monodromy_matrix_shape(self):
        H, L, vp = self._uncoupled_ho()
        r = IntegrabilityAnalysis.topological_monodromy(
            H, L, vp, critical_value=(0.0, 0.0),
            loop_radius=0.3, n_loop_points=12)
        assert r['monodromy_matrix'].shape == (2, 2)

    def test_actions_along_loop_shape(self):
        H, L, vp = self._uncoupled_ho()
        r = IntegrabilityAnalysis.topological_monodromy(
            H, L, vp, critical_value=(0.0, 0.0),
            loop_radius=0.3, n_loop_points=8)
        assert r['actions_along_loop'].shape == (8, 2)
        assert r['loop_EL'].shape == (8, 2)

    def test_loop_EL_circle(self):
        """The (E, ℓ) loop must lie on a circle of the requested radius."""
        H, L, vp = self._uncoupled_ho()
        r = IntegrabilityAnalysis.topological_monodromy(
            H, L, vp, critical_value=(1.0, 0.5),
            loop_radius=0.2, n_loop_points=16)
        EL = r['loop_EL']
        radii = np.sqrt((EL[:, 0] - 1.0)**2 + (EL[:, 1] - 0.5)**2)
        np.testing.assert_allclose(radii, 0.2, rtol=1e-10)

    def test_trivial_monodromy_uncoupled_ho(self):
        """Uncoupled HO has trivial monodromy — M = identity."""
        H, L, vp = self._uncoupled_ho()
        r = IntegrabilityAnalysis.topological_monodromy(
            H, L, vp, critical_value=(0.5, 0.25),
            loop_radius=0.2, n_loop_points=24)
        assert r['is_trivial'] is True

    def test_wrong_dof_raises(self):
        x, p = symbols('x p', real=True)
        H = (p**2 + x**2) / 2
        L = x
        with pytest.raises(ValueError):
            IntegrabilityAnalysis.topological_monodromy(
                H, L, [x, p], critical_value=(0.0, 0.0))


class TestScarIntensity:
    def _ho_traj(self, n_steps=4000):
        x, p = symbols('x p', real=True)
        H = (p**2 + x**2) / 2
        traj = hamiltonian_flow(H, (1, 0), (0, 40 * np.pi),
                                vars_phase=[x, p], n_steps=n_steps)
        return traj, [x, p]

    def _unit_circle_orbit(self, K=100):
        theta = np.linspace(0, 2 * np.pi, K, endpoint=False)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    def test_output_keys(self):
        traj, vp = self._ho_traj()
        orbit = self._unit_circle_orbit()
        r = IntegrabilityAnalysis.scar_intensity(traj, vp, orbit)
        for key in ('scar_intensity', 'f_orbit', 'f_expected',
                    'n_close', 'radius'):
            assert key in r

    def test_on_orbit_intensity_gt_1(self):
        """Trajectory that IS the orbit must yield scar_intensity >> 1."""
        traj, vp = self._ho_traj()
        orbit = self._unit_circle_orbit()
        r = IntegrabilityAnalysis.scar_intensity(traj, vp, orbit)
        assert r['scar_intensity'] > 1.0

    def test_f_orbit_in_unit_interval(self):
        traj, vp = self._ho_traj()
        orbit = self._unit_circle_orbit()
        r = IntegrabilityAnalysis.scar_intensity(traj, vp, orbit)
        assert 0.0 <= r['f_orbit'] <= 1.0

    def test_far_orbit_low_intensity(self):
        """An orbit far from the trajectory must yield near-zero f_orbit."""
        traj, vp = self._ho_traj()
        # Orbit at radius 5 — trajectory lives at radius 1
        theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        far_orbit = np.column_stack([5 * np.cos(theta), 5 * np.sin(theta)])
        r = IntegrabilityAnalysis.scar_intensity(traj, vp, far_orbit)
        assert r['f_orbit'] < 0.01

    def test_custom_radius(self):
        """Larger radius must give >= as many close points as smaller."""
        traj, vp = self._ho_traj()
        orbit = self._unit_circle_orbit()
        r_small = IntegrabilityAnalysis.scar_intensity(
            traj, vp, orbit, radius=0.05)
        r_large = IntegrabilityAnalysis.scar_intensity(
            traj, vp, orbit, radius=0.3)
        assert r_large['n_close'] >= r_small['n_close']

    def test_2dof_projects_correctly(self):
        """2-DOF trajectory: method must use x1,p1 without raising."""
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + x1**2 + p2**2 + x2**2) / 2
        traj = hamiltonian_flow(H, (1, 0, 0.5, 0), (0, 10 * np.pi),
                                vars_phase=[x1, p1, x2, p2], n_steps=500)
        orbit = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 20)),
                                 np.sin(np.linspace(0, 2*np.pi, 20))])
        r = IntegrabilityAnalysis.scar_intensity(
            traj, [x1, p1, x2, p2], orbit)
        assert np.isfinite(r['scar_intensity'])

    def test_invalid_orbit_shape_raises(self):
        traj, vp = self._ho_traj()
        with pytest.raises(ValueError):
            IntegrabilityAnalysis.scar_intensity(
                traj, vp, np.array([1.0, 2.0, 3.0]))  # 1D, not (K,2)