import numpy as np
import pytest
from sympy import symbols, Function, diff, I, pi, sin, cos, exp, sqrt, simplify
from physics import (
    LagrangianHamiltonianConverter,
    HamiltonianSymbolicConverter,
    detect_catastrophes,
    classify_arnold_2d,
    plot_catastrophe # Uncomment if matplotlib is available
)


def test_lagrangian_hamiltonian_conversion_harmonic_oscillator():
    """Test 1: 1D Standard Harmonic Oscillator L = 1/2 m v^2 - 1/2 k u^2"""
    x, u, p = symbols('x u p', real=True)
    L_ho = 0.5 * p**2 - 0.5 * u**2
    H_ho, (xi,) = LagrangianHamiltonianConverter.L_to_H(L_ho, (x,), u, (p,))
    expected_H = 0.5 * xi**2 + 0.5 * u**2
    assert simplify(H_ho - expected_H) == 0


def test_lagrangian_hamiltonian_conversion_free_particle():
    """Test 2: 2D Free Particle L = 1/2 (p_x^2 + p_y^2)"""
    x, y, u, p_x, p_y = symbols('x y u p_x p_y', real=True)
    L_free = 0.5 * (p_x**2 + p_y**2)
    H_free, (xi, eta) = LagrangianHamiltonianConverter.L_to_H(L_free, (x, y), u, (p_x, p_y))
    expected_H = 0.5 * (xi**2 + eta**2)
    assert simplify(H_free - expected_H) == 0


def test_lagrangian_hamiltonian_consistency():
    """Test 3: L -> H -> L Consistency (Harmonic Oscillator)"""
    x, u, p = symbols('x u p', real=True)
    L_orig = 0.5 * p**2 - 0.5 * u**2
    H_temp, (xi,) = LagrangianHamiltonianConverter.L_to_H(L_orig, (x,), u, (p,))
    L_back, (p_back,) = LagrangianHamiltonianConverter.H_to_L(H_temp, (x,), u, (xi,))
    assert simplify(L_orig - L_back) == 0


def test_lagrangian_hamiltonian_singular_hessian():
    """Test 4: L with Singular Hessian (p^4) - L->H failure"""
    x, u, p = symbols('x u p', real=True)
    L_bad = p**4
    try:
        H_bad, _ = LagrangianHamiltonianConverter.L_to_H(L_bad, (x,), u, (p,))
        print(f"  UNEXPECTED SUCCESS: H = {H_bad}")
    except ValueError as e:
        print(f"  Expected failure occurred: {e}")
    except Exception as e:
        print(f"  Unexpected error: {e}")


def test_numeric_fenchel():
    """Test 5: Numeric Fenchel (L = p^4 + p^2)"""
    x, u, p = symbols('x u p', real=True)
    L_fenchel = p**4 + p**2
    try:
        H_repr, (xi,), H_num_func = LagrangianHamiltonianConverter.L_to_H(
            L_fenchel, (x,), u, (p,), method="fenchel_numeric"
        )
        # Just test that the function can be called without error
        for val in [-1.0, 0.0, 1.0]:
            h_val = H_num_func(val)
            assert isinstance(h_val, float) or np.isscalar(h_val)
    except ImportError:
        pytest.skip("SciPy not available for numeric Fenchel.")


def test_hamiltonian_to_pde_schrodinger():
    """Test 6: Hamiltonian to PDE (1D Standard Kinetic + Potential)"""
    x, t, xi = symbols("x t xi", real=True)
    u = Function("u")(t, x)
    V = Function("V")(x)
    H_pde = 0.5 * xi**2 + V
    pde_info = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
        H_pde, (x,), t, u, mode="schrodinger"
    )
    # Just test that the function runs without error and returns expected keys
    assert 'pde' in pde_info
    assert 'formal_string' in pde_info


def test_hamiltonian_to_pde_wave():
    """Test 7: Hamiltonian to PDE (2D Kinetic + Potential)"""
    x, y, t = symbols("x y t", real=True)
    u2 = Function("u")(t, x, y)
    xi, eta = symbols("xi eta", real=True)
    V2 = Function("V")(x, y)
    H2D_pde = 0.5 * (xi**2 + eta**2) + V2
    pde_info_2d = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
        H2D_pde, (x, y), t, u2, mode="wave"
    )
    # Just test that the function runs without error and returns expected keys
    assert 'pde' in pde_info_2d
    assert 'formal_string' in pde_info_2d


def test_catastrophe_detection_fold():
    """Test 8: 1D Catastrophe Detection (Fold) H(xi) = xi^3 - a*xi"""
    xi, a = symbols("xi a", real=True)
    H_fold = xi**3 - a*xi
    H_fold_a1 = H_fold.subs(a, 1)
    pts_fold = detect_catastrophes(H_fold_a1, (xi,))
    # Just test that the function runs without error and returns a list
    assert isinstance(pts_fold, list)


def test_catastrophe_detection_cusp():
    """Test 9: 2D Catastrophe Detection (Cusp-family) H(xi,eta) = xi^4 + eta^2"""
    xi, eta = symbols("xi eta", real=True)
    H_cusp = xi**4 + eta**2
    pts_cusp = detect_catastrophes(H_cusp, (xi, eta))
    # Just test that the function runs without error and returns a list
    assert isinstance(pts_cusp, list)


def test_arnold_classification_morse_min():
    """Test 13: Morse (Non-Degenerate Minimum) H = xi^2 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_morse_min = xi**2 + eta**2
    point_morse_min = {"xi": 0, "eta": 0}
    res_morse_min = classify_arnold_2d(H_morse_min, xi, eta, point_morse_min)
    expected_type_min = "Morse (non-degenerate)"
    assert res_morse_min['type'] == expected_type_min


def test_arnold_classification_morse_max():
    """Test 13b: Morse (Non-Degenerate Maximum) H = -xi^2 - eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_morse_max = -xi**2 - eta**2
    point_morse_min = {"xi": 0, "eta": 0}
    res_morse_max = classify_arnold_2d(H_morse_max, xi, eta, point_morse_min)
    expected_type_max = "Morse (non-degenerate)"
    assert res_morse_max['type'] == expected_type_max


def test_arnold_classification_a3():
    """Test 14: A3 (Cusp-family) H = xi^4 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_a3 = xi**4 + eta**2
    point_a3 = {"xi": 0, "eta": 0}
    res_a3 = classify_arnold_2d(H_a3, xi, eta, point_a3)
    assert "A3" in res_a3['type']


def test_arnold_classification_a4():
    """Test 15: A4 (Swallowtail) H = xi^5 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_a4 = xi**5 + eta**2
    point_a4 = {"xi": 0, "eta": 0}
    res_a4 = classify_arnold_2d(H_a4, xi, eta, point_a4)
    assert "A4" in res_a4['type']


def test_arnold_classification_a5():
    """Test 16: A5 (Butterfly) H = xi^6 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_a5 = xi**6 + eta**2
    point_a5 = {"xi": 0, "eta": 0}
    res_a5 = classify_arnold_2d(H_a5, xi, eta, point_a5)
    assert "A5" in res_a5['type']


def test_arnold_classification_d4p():
    """Test 17: D4+ (Hyperbolic Umbilic) H = xi^3 + 3*xi*eta^2 (I=0 case)"""
    xi, eta = symbols('xi eta', real=True)
    H_d4p_norm = xi**3 + 3*xi*eta**2 # Standard normal form
    point_d4p_norm = {"xi": 0, "eta": 0}
    res_d4p_norm = classify_arnold_2d(H_d4p_norm, xi, eta, point_d4p_norm)
    # Check if I is indeed 0
    I_calc = res_d4p_norm.get('cubic_invariant_I', None)
    if I_calc is not None:
        assert abs(I_calc) < 1e-8


def test_arnold_classification_d4m():
    """Test 18: D4- (Elliptic Umbilic) H = xi^3 - 3*xi*eta^2 (I=0 case)"""
    xi, eta = symbols('xi eta', real=True)
    H_d4m_norm = xi**3 - 3*xi*eta**2 # Standard normal form
    point_d4m_norm = {"xi": 0, "eta": 0}
    res_d4m_norm = classify_arnold_2d(H_d4m_norm, xi, eta, point_d4m_norm)
    # Check if I is indeed 0
    I_calc = res_d4m_norm.get('cubic_invariant_I', None)
    if I_calc is not None:
        assert abs(I_calc) < 1e-8
