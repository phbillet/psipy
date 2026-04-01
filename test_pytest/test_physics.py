import numpy as np
import pytest
import sympy as sp
from sympy import symbols, Function, diff, I, pi, sin, cos, exp, sqrt, simplify, Abs, sign
from physics import (
    LagrangianHamiltonianConverter,
    HamiltonianSymbolicConverter,
)
import numpy.testing as npt


# -----------------------------------------------------------------------------
# Existing tests (preserved)
# -----------------------------------------------------------------------------
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
    """Test 4: L with Singular Hessian (linear) – should raise ValueError"""
    x, u, p = symbols('x u p', real=True)
    L_linear = p
    with pytest.raises(ValueError):
        LagrangianHamiltonianConverter.L_to_H(L_linear, (x,), u, (p,))


def test_numeric_fenchel():
    """Test 5: Numeric Fenchel (L = p^4 + p^2) – uses SciPy if available"""
    x, u, p = symbols('x u p', real=True)
    L_fenchel = p**4 + p**2
    try:
        H_repr, (xi,), H_num_func = LagrangianHamiltonianConverter.L_to_H(
            L_fenchel, (x,), u, (p,), method="fenchel_numeric"
        )
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
    assert 'pde' in pde_info_2d
    assert 'formal_string' in pde_info_2d


# -----------------------------------------------------------------------------
# New tests to increase coverage
# -----------------------------------------------------------------------------
def test_quadratic_singular_hessian():
    """Test that quadratic Lagrangians with singular Hessian raise ValueError."""
    x, u, p, a, b = symbols('x u p a b', real=True)
    L_linear = a * p + b
    with pytest.raises(ValueError):
        LagrangianHamiltonianConverter.L_to_H(L_linear, (x,), u, (p,))


def test_symbolic_fenchel_quadratic():
    """Symbolic Fenchel for L = p^2 should match Legendre."""
    x, u, p = symbols('x u p', real=True)
    L_quad = p**2
    H_fenchel, (xi,) = LagrangianHamiltonianConverter.L_to_H(
        L_quad, (x,), u, (p,), method="fenchel_symbolic"
    )
    H_legendre, _ = LagrangianHamiltonianConverter.L_to_H(
        L_quad, (x,), u, (p,), method="legendre"
    )
    assert simplify(H_fenchel - H_legendre) == 0


def test_symbolic_fenchel_p4():
    """Symbolic Fenchel for L = p^4 should produce the correct convex conjugate."""
    x, u, p = symbols('x u p', real=True)
    L_p4 = p**4
    H_fenchel, (xi,) = LagrangianHamiltonianConverter.L_to_H(
        L_p4, (x,), u, (p,), method="fenchel_symbolic"
    )
    # The conjugate is H = (3/4) * 4^{-1/3} * |xi|^{4/3}
    # Test numerically for positive xi
    H_func = sp.lambdify(xi, H_fenchel, 'numpy')
    def exact_H(xi_val):
        xi_val = np.asarray(xi_val)
        return (3/4) * (np.abs(xi_val) ** (4/3)) * (4 ** (-1/3))

    xi_vals = np.linspace(0.1, 5, 10)
    npt.assert_allclose(H_func(xi_vals), exact_H(xi_vals), rtol=1e-6)


def test_numeric_fenchel_1d_grid_mode():
    """Numeric Fenchel for 1D using grid mode (no SciPy)."""
    x, u, p = symbols('x u p', real=True)
    L = p**4
    fenchel_opts = {"mode": "grid", "n_grid": 1001}
    H_repr, (xi,), H_num = LagrangianHamiltonianConverter.L_to_H(
        L, (x,), u, (p,), method="fenchel_numeric", fenchel_opts=fenchel_opts
    )
    xi_vals = np.linspace(-5, 5, 10)
    h_vals = H_num(xi_vals)
    assert h_vals.shape == xi_vals.shape
    # Check approximate convexity and symmetry
    exact = lambda x: (3/4) * (np.abs(x)**(4/3)) * (4**(-1/3))
    npt.assert_allclose(h_vals, exact(xi_vals), rtol=0.1)


def test_numeric_fenchel_2d_quadratic():
    """2D numeric Fenchel for quadratic Lagrangian (grid mode)."""
    x, y, u, p_x, p_y = symbols('x y u p_x p_y', real=True)
    L = p_x**2 + p_y**2
    fenchel_opts = {"n_grid_per_dim": 31, "mode": "grid"}
    H_repr, (xi, eta), H_num = LagrangianHamiltonianConverter.L_to_H(
        L, (x, y), u, (p_x, p_y), method="fenchel_numeric", fenchel_opts=fenchel_opts
    )
    # Just test that the function returns a finite number for a few points.
    points = [(1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)]
    for xi_val, eta_val in points:
        h_num = H_num(np.array([xi_val, eta_val]))
        assert np.isfinite(h_num)


def test_numeric_fenchel_2d_scipy():
    """2D numeric Fenchel with SciPy (skip if not available)."""
    x, y, u, p_x, p_y = symbols('x y u p_x p_y', real=True)
    L = p_x**4 + p_y**4
    try:
        H_repr, (xi, eta), H_num = LagrangianHamiltonianConverter.L_to_H(
            L, (x, y), u, (p_x, p_y), method="fenchel_numeric", fenchel_opts={"mode": "auto"}
        )
        points = [(1.0, 1.0), (2.0, -1.0)]
        for xi_val, eta_val in points:
            h_num = H_num(np.array([xi_val, eta_val]))
            assert np.isfinite(h_num)
    except ImportError:
        pytest.skip("SciPy not available for 2D numeric fenchel.")


def test_return_symbol_only():
    """L_to_H with return_symbol_only=True substitutes u=0."""
    x, u, p = symbols('x u p', real=True)
    L = 0.5*p**2 - 0.5*u**2
    H_with_u, _ = LagrangianHamiltonianConverter.L_to_H(L, (x,), u, (p,), return_symbol_only=False)
    H_without_u, _ = LagrangianHamiltonianConverter.L_to_H(L, (x,), u, (p,), return_symbol_only=True)
    assert H_with_u.has(u)
    assert not H_without_u.has(u)
    assert simplify(H_without_u - H_with_u.subs(u, 0)) == 0


def test_decompose_hamiltonian():
    """Split Hamiltonian into polynomial and non‑local parts."""
    xi = symbols('xi', real=True)

    H1 = xi**2/2 + sqrt(1 + xi**2)
    poly, nonlocal_part = HamiltonianSymbolicConverter.decompose_hamiltonian(H1, (xi,))
    assert simplify(poly - xi**2/2) == 0
    assert simplify(nonlocal_part - sqrt(1 + xi**2)) == 0

    H2 = xi**2/2 + Abs(xi)
    poly, nonlocal_part = HamiltonianSymbolicConverter.decompose_hamiltonian(H2, (xi,))
    assert simplify(poly - xi**2/2) == 0
    assert simplify(nonlocal_part - Abs(xi)) == 0

    H3 = xi**2/2 + sign(xi)
    poly, nonlocal_part = HamiltonianSymbolicConverter.decompose_hamiltonian(H3, (xi,))
    assert simplify(poly - xi**2/2) == 0
    assert simplify(nonlocal_part - sign(xi)) == 0


def test_hamiltonian_to_pde_stationary():
    """Stationary PDE generation."""
    x, xi = symbols("x xi", real=True)
    u = Function("u")(x)
    V = Function("V")(x)
    H = 0.5*xi**2 + V
    pde_info = HamiltonianSymbolicConverter.hamiltonian_to_symbolic_pde(
        H, (x,), None, u, mode="stationary"
    )
    assert pde_info['mode'] == 'stationary'
    pde_eq = pde_info['pde']
    assert isinstance(pde_eq, sp.Eq)


def test_unsupported_dimensions():
    """L_to_H and H_to_L reject dimensions >2."""
    x1, x2, x3, u, p1, p2, p3 = symbols('x1 x2 x3 u p1 p2 p3', real=True)
    L = 0.5*(p1**2 + p2**2 + p3**2)
    with pytest.raises(ValueError, match="Only 1D and 2D dimensions are supported."):
        LagrangianHamiltonianConverter.L_to_H(L, (x1, x2, x3), u, (p1, p2, p3))

    xi1, xi2, xi3 = symbols('xi1 xi2 xi3', real=True)
    H = 0.5*(xi1**2 + xi2**2 + xi3**2)
    with pytest.raises(ValueError, match="Only 1D and 2D are supported."):
        LagrangianHamiltonianConverter.H_to_L(H, (x1, x2, x3), u, (xi1, xi2, xi3))


def test_symbolic_fenchel_non_differentiable():
    """Symbolic Fenchel refuses Lagrangians with Abs."""
    x, u, p = symbols('x u p', real=True)
    L_abs = Abs(p)
    with pytest.raises(ValueError, match="Symbolic Fenchel not possible for nonsmooth L"):
        LagrangianHamiltonianConverter.L_to_H(L_abs, (x,), u, (p,), method="fenchel_symbolic")


def test_numeric_fenchel_non_differentiable():
    """Numeric Fenchel works for non‑differentiable L (e.g., absolute value)."""
    x, u, p = symbols('x u p', real=True)
    L_abs = Abs(p)
    H_repr, (xi,), H_num = LagrangianHamiltonianConverter.L_to_H(
        L_abs, (x,), u, (p,), method="fenchel_numeric", fenchel_opts={"mode": "grid"}
    )
    # For |xi| <= 1, the conjugate is zero
    xi_vals = np.linspace(-1, 1, 10)
    h_vals = H_num(xi_vals)
    npt.assert_allclose(h_vals, 0, atol=1e-5)

    # For |xi| > 1, the conjugate should be positive (and large)
    xi_out = np.array([2.0])
    h_out = H_num(xi_out)
    assert h_out > 0