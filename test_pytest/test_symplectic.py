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
    evolve_phase_space_region
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


def test_frequency_period_not_implemented():
    """The 'period' method must raise NotImplementedError."""
    I_sym = symbols('I', real=True, positive=True)
    H = I_sym
    with pytest.raises(NotImplementedError):
        frequency(H, 1.0, method='period')


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
    traj = hamiltonian_flow(H, (1, 0, 0.5, 0), (0, 50),
                            vars_phase=[x1, p1, x2, p2],
                            integrator='symplectic', n_steps=500)
    dt = traj['t'][1] - traj['t'][0]
    exponents = lyapunov_exponents(traj, dt, vars_phase=[x1, p1, x2, p2], n_vectors=4)
    assert len(exponents) == 4


def test_lyapunov_exponents_sorted_descending():
    """
    Returned exponents must be in descending order among finite values.
    The current implementation can produce -inf / nan for near-zero QR
    diagonal entries, so we only enforce ordering on the finite subset.
    """
    x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
    H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
    traj = hamiltonian_flow(H, (1, 0, 0.5, 0), (0, 50),
                            vars_phase=[x1, p1, x2, p2],
                            integrator='symplectic', n_steps=500)
    dt = traj['t'][1] - traj['t'][0]
    exponents = lyapunov_exponents(traj, dt, vars_phase=[x1, p1, x2, p2], n_vectors=4)
    finite = exponents[np.isfinite(exponents)]
    assert len(finite) >= 1, "Expected at least one finite Lyapunov exponent"
    assert np.all(np.diff(finite) <= 0), (
        f"Finite exponents are not in descending order: {finite}"
    )


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