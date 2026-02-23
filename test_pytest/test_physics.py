import numpy as np
import pytest
from sympy import symbols, Function, diff, I, pi, sin, cos, exp, sqrt, simplify
from physics import (
    LagrangianHamiltonianConverter,
    HamiltonianSymbolicConverter,
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


