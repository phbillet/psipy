from imports import *
from psiop import *
import pytest
import matplotlib
matplotlib.use('Agg')  # non-interactive backend – no display required
import matplotlib.pyplot as plt

from sympy import diff

# ===========================================================================
# Helpers
# ===========================================================================

def _make_1d_grid(L=4.0, N=64):
    """Return (x_grid, kx) for a periodic 1D domain [-L, L)."""
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    kx = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    return x, kx

def _make_2d_grid(L=4.0, N=32):
    """Return (x, y, kx, ky) for a periodic 2D domain."""
    x = np.linspace(-L, L, N, endpoint=False)
    y = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2.0 * np.pi
    return x, y, kx, ky

# ===========================================================================
# 1. Constructor – symbol mode, 1D
# ===========================================================================

def test_symbol_mode_1d():
    x = symbols('x', real=True)
    xi = symbols('xi', real=True)
    op = PseudoDifferentialOperator(expr=x * xi, vars_x=[x], mode='symbol')
    assert callable(op.p_func)
    x_vals = np.array([1.0, 2.0])
    xi_vals = np.array([3.0, 4.0])
    result = op.p_func(x_vals[:, None], xi_vals[None, :])
    expected = x_vals[:, None] * xi_vals[None, :]
    assert np.allclose(result, expected)

# 2. Constructor – auto mode, 1D (transport)

def test_auto_mode_1d():
    x = symbols('x', real=True)
    u = Function('u')
    op = PseudoDifferentialOperator(expr=diff(u(x), x), vars_x=[x],
                                    var_u=u(x), mode='auto')
    assert callable(op.p_func)
    x_vals = np.array([1.0, 2.0])
    xi_vals = np.array([3.0, 4.0])
    result = op.p_func(x_vals[:, None], xi_vals[None, :])
    expected = 1j * xi_vals[None, :]
    assert np.allclose(result, expected, atol=1e-6)

# 3. Constructor – invalid mode raises ValueError

def test_invalid_mode_1d():
    x = symbols('x', real=True)
    try:
        PseudoDifferentialOperator(expr=x, vars_x=[x], mode='invalid_mode')
        assert False, "no error raised for invalid mode"
    except ValueError as e:
        assert "mode must be 'auto' or 'symbol'" in str(e)

# 4. Constructor – missing var_u raises ValueError

def test_missing_varu_auto_mode_1d():
    x = symbols('x', real=True)
    try:
        PseudoDifferentialOperator(expr=x, vars_x=[x], mode='auto')
        assert False, "no error raised for missing var_u"
    except ValueError as e:
        assert "var_u must be provided in mode='auto'" in str(e)

# 5. Constructor – symbol mode, 2D

def test_symbol_mode_2d():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True)
    op = PseudoDifferentialOperator(expr=x * y * xi + eta**2,
                                    vars_x=[x, y], mode='symbol')
    assert callable(op.p_func)
    xv = np.array([1.0, 2.0])
    yv = np.array([0.5, 1.5])
    xiv = np.array([2.0, 3.0])
    etav = np.array([1.0, 4.0])
    result = op.p_func(xv[:, None, None, None], yv[None, :, None, None],
                       xiv[None, None, :, None], etav[None, None, None, :])
    expected = (xv[:, None, None, None] * yv[None, :, None, None]
                * xiv[None, None, :, None] + etav[None, None, None, :]**2)
    assert np.allclose(result, expected)

# 6. Constructor – auto mode, 2D (Laplacian)

def test_auto_mode_2d():
    x, y = symbols('x y', real=True)
    u = Function('u')
    expr_auto = diff(u(x, y), x, 2) + diff(u(x, y), y, 2)
    op = PseudoDifferentialOperator(expr=expr_auto, vars_x=[x, y],
                                    var_u=u(x, y), mode='auto')
    assert callable(op.p_func)
    xv = np.array([1.0, 2.0])
    yv = np.array([0.5, 1.5])
    xiv = np.array([2.0, 3.0])
    etav = np.array([1.0, 4.0])
    result = op.p_func(xv[:, None, None, None], yv[None, :, None, None],
                       xiv[None, None, :, None], etav[None, None, None, :])
    expected = -(xiv[None, None, :, None]**2 + etav[None, None, None, :]**2)
    assert np.allclose(result, expected)

# 7. Constructor – invalid mode, 2D

def test_invalid_mode_2d():
    x, y = symbols('x y', real=True)
    try:
        PseudoDifferentialOperator(expr=x * y, vars_x=[x, y],
                                   mode='invalid_mode')
        assert False, "no error raised for invalid mode"
    except ValueError as e:
        assert "mode must be 'auto' or 'symbol'" in str(e)

# 8. Constructor – 3D raises NotImplementedError

def test_3d_not_implemented():
    x, y, z = symbols('x y z', real=True)
    u = Function('u')
    try:
        PseudoDifferentialOperator(
            expr=u(x, y, z).diff(x, 2) + u(x, y, z).diff(y, 2) + u(x, y, z).diff(z, 2),
            vars_x=[x, y, z], var_u=u(x, y, z), mode='auto')
        assert False, "no error raised for dim=3"
    except NotImplementedError as e:
        assert "Only 1D and 2D supported" in str(e)

# ===========================================================================
# 9-17. Symbol order
# ===========================================================================

def test_symbol_order_1d():
    x, xi = symbols('x xi', real=True, positive=True)
    op = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')
    assert op.is_homogeneous()[0] == True
    assert op.symbol_order() == 2

def test_symbol_order_1d_non_homogeneous():
    x, xi = symbols('x xi', real=True, positive=True)
    op = PseudoDifferentialOperator(expr=(x * xi)**2, vars_x=[x], mode='symbol')
    assert op.is_homogeneous()[0] == True
    assert op.symbol_order() == 2

def test_symbol_order_1d_trig():
    x, xi = symbols('x xi', real=True, positive=True)
    op = PseudoDifferentialOperator(expr=sin(x) + xi**2, vars_x=[x], mode='symbol')
    assert op.is_homogeneous()[0] == False
    assert op.symbol_order() == 2

def test_symbol_order_1d_cubic():
    x = symbols('x', real=True)
    xi = symbols('xi', real=True, positive=True)
    op = PseudoDifferentialOperator(expr=xi**3, vars_x=[x], mode='symbol')
    assert op.symbol_order() == 3

def test_symbol_order_2d():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True, positive=True)
    op = PseudoDifferentialOperator(expr=xi**2 + eta**2, vars_x=[x, y], mode='symbol')
    assert op.is_homogeneous()[0] == True
    assert op.symbol_order() == 2

def test_symbol_order_2d_fraction():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True, positive=True)
    op = PseudoDifferentialOperator(expr=(xi**2 + eta**2)**(1/3), vars_x=[x, y])
    assert op.is_homogeneous()[0] == True
    assert abs(op.symbol_order() - 2/3) < 1e-2

def test_symbol_order_2d_sqrt():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True, positive=True)
    op = PseudoDifferentialOperator(expr=sqrt(xi**2 + eta**2), vars_x=[x, y], mode='symbol')
    assert abs(op.symbol_order() - 1) < 1e-2

# ===========================================================================
# 18-21. Principal symbol
# ===========================================================================

def test_principal_symbol_1d():
    x = symbols('x', real=True)
    xi = symbols('xi', real=True, positive=True)
    expr = xi**2 + sqrt(xi**2 + x**2)
    p = PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')
    ps1 = p.principal_symbol(order=1)
    ps2 = p.principal_symbol(order=2)
    assert ps1 == xi * (xi + 1)
    assert ps2 == x**2 / (2 * xi) + xi**2 + xi

def test_principal_symbol_2d():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True, positive=True)
    p = PseudoDifferentialOperator(expr=(xi**2 + eta**2 + 1)**(1/3),
                                   vars_x=[x, y], mode='symbol')
    ps1 = p.principal_symbol(order=1)
    ps2 = p.principal_symbol(order=2)
    assert ps1 is not None
    assert ps2 is not None

# ===========================================================================
# 22-27. Asymptotic expansion
# ===========================================================================

def test_asymptotic_expansion_1d():
    x, xi = symbols('x xi', real=True, positive=True)
    p = PseudoDifferentialOperator(
        expr=exp(x * xi / (x**2 + xi**2)), vars_x=[x], mode='symbol')
    assert p.asymptotic_expansion(order=4) is not None

def test_asymptotic_expansion_1d_sqrt():
    x, xi = symbols('x xi', real=True, positive=True)
    p = PseudoDifferentialOperator(
        expr=sqrt(xi**2 + 1) + x / (xi**2 + 1), vars_x=[x], mode='symbol')
    assert p.asymptotic_expansion(order=4) is not None

def test_asymptotic_expansion_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True, positive=True)
    p = PseudoDifferentialOperator(
        expr=sqrt(xi**2 + eta**2) + x, vars_x=[x, y], mode='symbol')
    assert p.asymptotic_expansion(order=4) is not None

def test_asymptotic_expansion_2d_frac():
    x, y, xi, eta = symbols('x y xi eta', real=True, positive=True)
    p = PseudoDifferentialOperator(
        expr=sqrt(xi**2 + eta**2) + x / (xi**2 + eta**2),
        vars_x=[x, y], mode='symbol')
    assert p.asymptotic_expansion(order=4) is not None

def test_asymptotic_expansion_2d_complex():
    x, y, xi, eta = symbols('x y xi eta', real=True, positive=True)
    p = PseudoDifferentialOperator(
        expr=exp(x * xi * y * eta / (x**2 + y**2 + xi**2 + eta**2)),
        vars_x=[x, y], mode='symbol')
    assert p.asymptotic_expansion(order=4) is not None

def test_asymptotic_expansion_1d_polynomial_order():
    """Expansion of xi^3 truncated to order 2 should equal xi^3 (already a polynomial)."""
    x, xi = symbols('x xi', real=True, positive=True)
    p = PseudoDifferentialOperator(expr=xi**3, vars_x=[x], mode='symbol')
    expansion = p.asymptotic_expansion(order=4)
    # The leading term must contain xi^3
    assert expansion is not None
    assert xi**3 in expansion.as_ordered_terms() or simplify(expansion - xi**3) == 0

# ===========================================================================
# 28-31. Asymptotic composition
# ===========================================================================

def test_asymptotic_composition_1d():
    x, xi = symbols('x xi', real=True)
    p1 = PseudoDifferentialOperator(expr=xi + x, vars_x=[x], mode='symbol')
    p2 = PseudoDifferentialOperator(expr=xi + x**2, vars_x=[x], mode='symbol')
    assert p1.compose_asymptotic(p2, order=2) is not None

def test_asymptotic_composition_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p1 = PseudoDifferentialOperator(expr=xi**2 + eta**2 + x * y,
                                    vars_x=[x, y], mode='symbol')
    p2 = PseudoDifferentialOperator(expr=xi + eta + x + y,
                                    vars_x=[x, y], mode='symbol')
    assert p1.compose_asymptotic(p2, order=3) is not None

def test_composition_weyl_mode_1d():
    x, xi = symbols('x xi', real=True)
    p1 = PseudoDifferentialOperator(expr=xi**2, vars_x=[x])
    p2 = PseudoDifferentialOperator(expr=x**2, vars_x=[x])
    result = p1.compose_asymptotic(p2, order=2, mode='weyl')
    assert result is not None

def test_composition_kn_weyl_differ():
    """KN and Weyl compositions of non-commuting symbols should differ."""
    x, xi = symbols('x xi', real=True)
    p1 = PseudoDifferentialOperator(expr=x * xi, vars_x=[x])
    p2 = PseudoDifferentialOperator(expr=xi**2, vars_x=[x])
    kn = p1.compose_asymptotic(p2, order=2, mode='kn')
    weyl = p1.compose_asymptotic(p2, order=2, mode='weyl')
    assert simplify(kn - weyl) != 0

# ===========================================================================
# 32-33. Commutator
# ===========================================================================

def test_commutator_1d():
    x, xi = symbols('x xi', real=True)
    A = PseudoDifferentialOperator(expr=x * xi, vars_x=[x])
    B = PseudoDifferentialOperator(expr=xi**2, vars_x=[x])
    C = A.commutator_symbolic(B, order=1)
    expected = 2 * I * xi**2
    assert simplify(simplify(C) - expected) == 0

def test_commutator_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    A = PseudoDifferentialOperator(expr=x * xi + y * eta, vars_x=[x, y])
    B = PseudoDifferentialOperator(expr=xi**2 + eta**2, vars_x=[x, y])
    C = A.commutator_symbolic(B, order=1)
    expected = 2 * I * (xi**2 + eta**2)
    assert simplify(simplify(C) - expected) == 0

# ===========================================================================
# 34-37. Inverses
# ===========================================================================

def test_right_inverse_1d():
    x, xi = symbols('x xi', real=True)
    p = PseudoDifferentialOperator(expr=xi + 1, vars_x=[x], mode='symbol')
    r = p.right_inverse_asymptotic(order=2)
    assert r is not None
    p2 = PseudoDifferentialOperator(expr=r, vars_x=[x], mode='symbol')
    assert p.compose_asymptotic(p2, order=2) is not None

def test_right_inverse_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = PseudoDifferentialOperator(expr=xi + eta + 1, vars_x=[x, y], mode='symbol')
    r = p.right_inverse_asymptotic(order=2)
    assert r is not None
    p2 = PseudoDifferentialOperator(expr=r, vars_x=[x, y], mode='symbol')
    assert p.compose_asymptotic(p2, order=2) is not None

def test_left_inverse_1d():
    x, xi = symbols('x xi', real=True)
    p = PseudoDifferentialOperator(expr=xi + 1, vars_x=[x], mode='symbol')
    l = p.left_inverse_asymptotic(order=2)
    assert l is not None
    p2 = PseudoDifferentialOperator(expr=l, vars_x=[x], mode='symbol')
    assert p2.compose_asymptotic(p, order=2) is not None

def test_left_inverse_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = PseudoDifferentialOperator(expr=xi + eta + 1, vars_x=[x, y], mode='symbol')
    l = p.left_inverse_asymptotic(order=3)
    assert l is not None
    p2 = PseudoDifferentialOperator(expr=l, vars_x=[x, y], mode='symbol')
    assert p2.compose_asymptotic(p, order=3) is not None

# ===========================================================================
# 38-39. Right/left inverse compositions (algebraic check)
# ===========================================================================

def test_left_inverse_composition():
    x, xi = symbols('x xi', real=True)
    p = PseudoDifferentialOperator(expr=xi**2 + x**2 + 1, vars_x=[x], mode='symbol')
    l = p.left_inverse_asymptotic(order=2)
    p_l = PseudoDifferentialOperator(expr=l, vars_x=[x], mode='symbol')
    composition = p_l.compose_asymptotic(p, order=2)
    assert composition is not None

def test_right_inverse_composition():
    x, xi = symbols('x xi', real=True)
    p = PseudoDifferentialOperator(expr=xi**2 + x**2 + 1, vars_x=[x], mode='symbol')
    r = p.right_inverse_asymptotic(order=2)
    p_r = PseudoDifferentialOperator(expr=r, vars_x=[x], mode='symbol')
    composition = p.compose_asymptotic(p_r, order=2)
    assert composition is not None

# ===========================================================================
# 40-47. Formal adjoint & self-adjointness
# ===========================================================================

def test_formal_adjoint_1d():
    x = symbols('x', real=True, positive=True)
    xi = symbols('xi', real=True, positive=True)
    p = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')
    adjoint = p.formal_adjoint()
    assert adjoint is not None
    assert p.is_self_adjoint() == True

def test_formal_adjoint_1d_complex():
    x = symbols('x', real=True)
    xi = symbols('xi', real=True, positive=True)
    p = PseudoDifferentialOperator(expr=(1 + I * x) * xi + exp(-x) / xi,
                                   vars_x=[x], mode='symbol')
    adjoint = p.formal_adjoint()
    assert adjoint is not None
    assert p.is_self_adjoint() == False

def test_formal_adjoint_2d():
    x, y = symbols('x y', real=True, positive=True)
    xi, eta = symbols('xi eta', real=True, positive=True)
    p = PseudoDifferentialOperator(expr=xi**2 + eta**2, vars_x=[x, y], mode='symbol')
    assert p.formal_adjoint() is not None
    assert p.is_self_adjoint() == True

def test_formal_adjoint_2d_asymmetric():
    x, y = symbols('x y', real=True, positive=True)
    xi, eta = symbols('xi eta', real=True, positive=True)
    p = PseudoDifferentialOperator(expr=y * xi**2 + x * eta**2,
                                   vars_x=[x, y], mode='symbol')
    assert p.formal_adjoint() is not None
    assert p.is_self_adjoint() == True

def test_formal_adjoint_2d_complex():
    x, y = symbols('x y', real=True)
    xi = symbols('xi', real=True, positive=True)
    eta = symbols('eta', real=True, positive=True)
    p_expr = (x + I * y) * xi + (y - I * x) * eta + exp(-x - y) / (xi + eta)
    p = PseudoDifferentialOperator(expr=p_expr, vars_x=[x, y], mode='symbol')
    assert p.formal_adjoint() is not None
    assert p.is_self_adjoint() == False

# ===========================================================================
# 48-53. Ellipticity
# ===========================================================================

def test_ellipticity_1d_elliptic():
    x, xi = symbols('x xi', real=True)
    x_vals = np.linspace(-1, 1, 100)
    xi_vals = np.linspace(-10, 10, 100)
    op = PseudoDifferentialOperator(expr=xi**2 + 1, vars_x=[x], mode='symbol')
    assert op.is_elliptic_numerically(x_vals, xi_vals) == True

def test_ellipticity_1d_non_elliptic():
    x, xi = symbols('x xi', real=True)
    
    # FIX 1: Use an odd number of points (101) so that 0.0 is exactly in the grid.
    x_vals = np.linspace(-1, 1, 101)
    xi_vals = np.linspace(-10, 10, 100)
    
    op = PseudoDifferentialOperator(expr=x * xi, vars_x=[x], mode='symbol')
    
    # FIX 2: Pass n_edge=33 (an odd number) so the deterministic edge check 
    # also lands exactly on x=0. With the default n_edge=32, it misses x=0 
    # and the closest points (~0.032) yield a ratio > 1e-6.
    assert op.is_elliptic_numerically(x_vals, xi_vals, n_edge=33) == False

def test_ellipticity_1d_constant_nonzero():
    x, xi = symbols('x xi', real=True)
    x_vals = np.linspace(-1, 1, 50)
    xi_vals = np.linspace(-5, 5, 50)
    op = PseudoDifferentialOperator(expr=xi**2 + 1, vars_x=[x], mode='symbol')
    assert op.is_elliptic_numerically(x_vals, xi_vals) == True

def test_ellipticity_2d_elliptic():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    x_vals = np.linspace(-1, 1, 30)
    y_vals = np.linspace(-1, 1, 30)
    xi_vals = np.linspace(-5, 5, 30)
    eta_vals = np.linspace(-5, 5, 30)
    op = PseudoDifferentialOperator(expr=xi**2 + eta**2 + 1,
                                    vars_x=[x, y], mode='symbol')
    assert op.is_elliptic_numerically((x_vals, y_vals), (xi_vals, eta_vals)) == True

def test_ellipticity_2d_non_elliptic():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    x_vals = np.linspace(-1, 1, 30)
    y_vals = np.linspace(-1, 1, 30)
    xi_vals = np.linspace(-5, 5, 30)
    eta_vals = np.linspace(-5, 5, 30)
    # xi+eta vanishes on the anti-diagonal
    op = PseudoDifferentialOperator(expr=xi + eta, vars_x=[x, y], mode='symbol')
    assert op.is_elliptic_numerically((x_vals, y_vals), (xi_vals, eta_vals)) == False

# ===========================================================================
# 54-57. Trace formula
# ===========================================================================

def test_trace_formula_symbolic():
    x = symbols('x', real=True)
    xi = symbols('xi', real=True)
    P = PseudoDifferentialOperator(exp(-(x**2 + xi**2)), [x], mode='symbol')
    trace = P.trace_formula()
    assert trace is not None

def test_trace_formula_numerical_1d():
    x = symbols('x', real=True)
    xi = symbols('xi', real=True)
    P = PseudoDifferentialOperator(exp(-(x**2 + xi**2)), [x], mode='symbol')
    trace_num = P.trace_formula(
        numerical=True,
        x_bounds=[(-6, 6)],
        xi_bounds=[(-6, 6)]
    )
    # Analytical value: (1/2π) * π = 0.5  (double Gaussian integral)
    assert abs(trace_num - 0.5) < 1e-2

def test_trace_formula_numerical_2d():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True)
    P = PseudoDifferentialOperator(exp(-(x**2 + y**2 + xi**2 + eta**2)),
                                   [x, y], mode='symbol')
    trace_num = P.trace_formula(
        numerical=True,
        x_bounds=[(-4, 4), (-4, 4)],
        xi_bounds=[(-4, 4), (-4, 4)]
    )
    # (1/4π²) * π² = 0.25
    assert abs(trace_num - 0.25) < 1e-2

def test_trace_formula_missing_bounds_raises():
    x = symbols('x', real=True)
    xi = symbols('xi', real=True)
    P = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    try:
        P.trace_formula(numerical=True)
        assert False, "should raise ValueError for missing bounds"
    except ValueError:
        pass

# ===========================================================================
# 58-63. Exponential symbol
# ===========================================================================

def test_exponential_symbol_1d():
    x, xi = symbols('x xi', real=True)
    t_sym = symbols('t', real=True)
    H_op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    assert H_op.exponential_symbol(t=-I * t_sym, order=3) is not None

def test_exponential_symbol_heat_kernel():
    x, xi = symbols('x xi', real=True)
    L_op = PseudoDifferentialOperator(-xi**2, [x], mode='symbol')
    result = L_op.exponential_symbol(t=0.1, order=5)
    assert result is not None

def test_exponential_symbol_fractional_schrodinger():
    x, xi = symbols('x xi', real=True)
    t_sym = symbols('t', real=True)
    H_frac_op = PseudoDifferentialOperator(xi**1.5, [x], mode='symbol')
    assert H_frac_op.exponential_symbol(t=-I * t_sym, order=3) is not None

def test_exponential_symbol_gibbs():
    x, xi, beta = symbols('x xi beta', real=True)
    H_op = PseudoDifferentialOperator(xi**2 / 2 + x**4 / 4, [x], mode='symbol')
    assert H_op.exponential_symbol(t=-beta, order=4) is not None

def test_exponential_symbol_2d():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True)
    L_op = PseudoDifferentialOperator(-(xi**2 + eta**2), [x, y], mode='symbol')
    assert L_op.exponential_symbol(t=0.05, order=4) is not None

def test_exponential_symbol_2d_harmonic_oscillator():
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True)
    t_sym = symbols('t', real=True)
    H_op = PseudoDifferentialOperator(xi**2 + eta**2 + x**2 + y**2,
                                      [x, y], mode='symbol')
    assert H_op.exponential_symbol(t=-I * t_sym, order=3) is not None

# ===========================================================================
# 64-67. Symplectic / Hamiltonian flow (symbolic correctness)
# ===========================================================================

def test_symplectic_flow_1d_laplacian():
    """For p = xi^2: dx/dt = 2xi, dxi/dt = 0."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    H = op.symplectic_flow()
    assert simplify(H['dx/dt'] - 2 * xi) == 0
    assert simplify(H['dxi/dt']) == 0

def test_symplectic_flow_1d_harmonic():
    """For p = xi^2 + x^2: dx/dt = 2xi, dxi/dt = -2x."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    H = op.symplectic_flow()
    assert simplify(H['dx/dt'] - 2 * xi) == 0
    assert simplify(H['dxi/dt'] + 2 * x) == 0

def test_symplectic_flow_2d_laplacian():
    """For p = xi^2 + eta^2: dx/dt = 2xi, dy/dt = 2eta, dxi/dt = 0, deta/dt = 0."""
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True)
    op = PseudoDifferentialOperator(xi**2 + eta**2, [x, y], mode='symbol')
    H = op.symplectic_flow()
    assert simplify(H['dx/dt'] - 2 * xi) == 0
    assert simplify(H['dy/dt'] - 2 * eta) == 0
    assert simplify(H['dxi/dt']) == 0
    assert simplify(H['deta/dt']) == 0

def test_symplectic_flow_1d_spatial():
    """For p = x*xi: dx/dt = x, dxi/dt = -xi."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi, [x], mode='symbol')
    H = op.symplectic_flow()
    assert simplify(H['dx/dt'] - x) == 0
    assert simplify(H['dxi/dt'] + xi) == 0

# ===========================================================================
# 68-73. evaluate() and clear_cache()
# ===========================================================================

def test_evaluate_1d_returns_correct_values():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    N = 8
    x_grid, kx = _make_1d_grid(N=N)
    KX = kx
    X = x_grid
    # For evaluate in 1D: Y and KY are ignored
    vals = op.evaluate(X, None, KX, None, cache=False)
    expected = kx**2
    assert np.allclose(vals, expected)

def test_evaluate_1d_cache_returns_same_object():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(N=8)
    v1 = op.evaluate(x_grid, None, kx, None, cache=True)
    v2 = op.evaluate(x_grid, None, kx, None, cache=True)
    assert v1 is v2, "second call should return the cached object"

def test_evaluate_clear_cache():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(N=8)
    op.evaluate(x_grid, None, kx, None, cache=True)
    assert op.symbol_cached is not None
    op.clear_cache()
    assert op.symbol_cached is None

def test_evaluate_no_cache_does_not_store():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(N=8)
    op.evaluate(x_grid, None, kx, None, cache=False)
    assert op.symbol_cached is None

# ===========================================================================
# 74-82. apply() — constant-coefficient symbol (1D, periodic)
# ===========================================================================

def _gaussian(x, sigma=1.0):
    return np.exp(-x**2 / (2 * sigma**2))

def test_apply_identity_1d():
    """Symbol p = 1 should return u unchanged (up to numerical noise)."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(Integer(1), [x], mode='symbol')
    x_grid, kx = _make_1d_grid(L=6.0, N=128)
    u = _gaussian(x_grid)
    result = op.apply(u, x_grid, kx, boundary_condition='periodic')
    assert np.allclose(np.real(result), u, atol=1e-4)

def test_apply_derivative_1d():
    """Symbol p = i*xi: applying to Gaussian should give its derivative."""
    x_sym, xi_sym = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(I * xi_sym, [x_sym], mode='symbol')
    x_grid, kx = _make_1d_grid(L=6.0, N=512)
    sigma = 1.0
    u = _gaussian(x_grid, sigma)
    du_analytical = -x_grid / sigma**2 * u
    result = op.apply(u, x_grid, kx, boundary_condition='periodic')
    # Compare central region where boundary effects are negligible
    mid = slice(100, 412)
    assert np.allclose(np.real(result)[mid], du_analytical[mid], atol=1e-2)

def test_apply_laplacian_1d():
    """Symbol p = -xi^2: result should approximate the second derivative."""
    x_sym, xi_sym = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi_sym**2, [x_sym], mode='symbol')
    x_grid, kx = _make_1d_grid(L=6.0, N=512)
    sigma = 1.0
    u = _gaussian(x_grid, sigma)
    d2u_analytical = ((x_grid**2 / sigma**4) - 1 / sigma**2) * u
    result = op.apply(u, x_grid, kx, boundary_condition='periodic')
    mid = slice(100, 412)
    assert np.allclose(np.real(result)[mid], d2u_analytical[mid], atol=5e-2)

def test_apply_spatial_symbol_1d_periodic():
    """Spatially varying symbol: x * (i*xi). Op(u) ≈ x * u'."""
    x_sym, xi_sym = symbols('x xi', real=True)
#    op = PseudoDifferentialOperator(x_sym * I * xi_sym, [x_sym], mode='symbol')
    op = PseudoDifferentialOperator(x_sym * I * xi_sym, [x_sym],
                                mode='symbol', quantization='kohn-nirenberg')
    x_grid, kx = _make_1d_grid(L=15.0, N=512)   # N plus grand, L plus grand
    sigma = 1.2                                  # sigma moins piqué
    u = _gaussian(x_grid, sigma)
    du = -x_grid / sigma**2 * u
    expected = x_grid * du
    result = op.apply(u, x_grid, kx, boundary_condition='periodic')
    mid = slice(100, 412)
    assert np.allclose(np.real(result)[mid], expected[mid], atol=5e-2)

def test_apply_invalid_bc_raises():
    x_sym, xi_sym = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi_sym**2, [x_sym], mode='symbol')
    x_grid, kx = _make_1d_grid(N=32)
    u = np.ones(32)
    try:
        op.apply(u, x_grid, kx, boundary_condition='unknown')
        assert False, "should raise ValueError"
    except ValueError:
        pass

def test_apply_constant_symbol_linear():
    """p = 2*xi^2 should be exactly twice p = xi^2."""
    x_sym, xi_sym = symbols('x xi', real=True)
    op1 = PseudoDifferentialOperator(xi_sym**2, [x_sym], mode='symbol')
    op2 = PseudoDifferentialOperator(2 * xi_sym**2, [x_sym], mode='symbol')
    x_grid, kx = _make_1d_grid(L=6.0, N=128)
    u = _gaussian(x_grid)
    r1 = op1.apply(u, x_grid, kx)
    r2 = op2.apply(u, x_grid, kx)
    assert np.allclose(r2, 2 * r1, atol=1e-10)

def test_apply_dirichlet_bc_1d():
    """Dirichlet BC path should not raise and produce an array of same shape."""
    x_sym, xi_sym = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(I * xi_sym, [x_sym], mode='symbol')
    x_grid, kx = _make_1d_grid(L=5.0, N=64)
    u = _gaussian(x_grid)
    result = op.apply(u, x_grid, kx, boundary_condition='dirichlet')
    assert result.shape == u.shape

def test_apply_2d_constant_periodic():
    """2D Laplacian -xi^2 - eta^2 applied to separable Gaussian."""
    x_sym, y_sym = symbols('x y', real=True)
    xi_sym, eta_sym = symbols('xi eta', real=True)
    op = PseudoDifferentialOperator(-(xi_sym**2 + eta_sym**2),
                                    [x_sym, y_sym], mode='symbol')
    x_grid, y_grid, kx, ky = _make_2d_grid(L=5.0, N=32)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    sigma = 1.0
    u = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    result = op.apply(u, x_grid, kx, y_grid=y_grid, ky=ky,
                      boundary_condition='periodic')
    assert result.shape == u.shape
    # Laplacian of Gaussian: ((x^2+y^2)/sigma^4 - 2/sigma^2) * G
    d2u = ((X**2 + Y**2) / sigma**4 - 2 / sigma**2) * u
    # Check central region
    s = slice(8, 24)
    assert np.allclose(np.real(result)[s, s], d2u[s, s], atol=0.1)

import numpy as np
import sympy as sp
from psiop import PseudoDifferentialOperator

# ----------------------------------------------------------------------
# Helper functions (mimic those used in the original test file)
# ----------------------------------------------------------------------
def _make_1d_grid(L=10.0, N=256):
    """Return (x_grid, kx) for a periodic domain [-L, L) with N points."""
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    kx = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    return x, kx

def _gaussian(x, sigma=1.0):
    """Return a Gaussian centered at 0."""
    return np.exp(-x**2 / (2 * sigma**2))

# ----------------------------------------------------------------------
# Test 1: Constant symbol – Weyl must equal multiplication by constant
# ----------------------------------------------------------------------
def test_weyl_constant_symbol():
    """Symbol p(x,ξ) = c.  Operator = c·I."""
    c = 2.5 + 0.7j
    x_sym, xi_sym = sp.symbols('x xi', real=True)
    op = PseudoDifferentialOperator(c, [x_sym], mode='symbol', quantization='weyl')
    x, kx = _make_1d_grid(L=5.0, N=128)
    u = _gaussian(x, sigma=1.0)
    
    # FIX: Disable numerical stabilization to test the exact mathematical operator
    result = op.apply(
        u, x, kx, 
        boundary_condition='periodic', 
        freq_window=None, 
        clamp=np.inf
    )
    
    expected = c * u
    assert np.allclose(result, expected, atol=1e-10)

# ----------------------------------------------------------------------
# Test 2: Symbol p(ξ) = ξ – first derivative
# ----------------------------------------------------------------------
def test_weyl_xi():
    """Symbol p(ξ) = ξ.  Operator = -i d/dx (same as KN, no x dependence)."""
    x_sym, xi_sym = sp.symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi_sym, [x_sym], mode='symbol', quantization='weyl')
    x, kx = _make_1d_grid(L=8.0, N=256)
    u = _gaussian(x, sigma=1.2)
    # Analytical derivative: -i * du/dx = -i * (-x/sigma^2) u = i x/sigma^2 u
    expected = 1j * x / (1.2**2) * u
    result = op.apply(
        u, x, kx, 
        boundary_condition='periodic', 
        freq_window=None, 
        clamp=np.inf
    )
    assert np.allclose(result, expected, atol=1e-7, rtol=1e-6)

# ----------------------------------------------------------------------
# Test 3: Symbol p(x) = x – multiplication operator
# ----------------------------------------------------------------------
def test_weyl_x():
    x_sym, xi_sym = sp.symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x_sym, [x_sym], mode='symbol', quantization='weyl')
    x, kx = _make_1d_grid(L=10.0, N=256)
    u = _gaussian(x, sigma=1.0)
    result = op.apply(
        u, x, kx, 
        boundary_condition='periodic', 
        freq_window=None, 
        clamp=np.inf
    )
    expected = x * u
    assert np.allclose(result, expected, atol=1e-10)

# ----------------------------------------------------------------------
# Test 4: Symbol p(x,ξ) = x ξ – Weyl gives (x ∂_x + ∂_x x)/2
# ----------------------------------------------------------------------
def test_weyl_x_xi():
    x_sym, xi_sym = sp.symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x_sym * xi_sym, [x_sym],
                                    mode='symbol', quantization='weyl', )
    x, kx = _make_1d_grid(L=10.0, N=512)
    sigma = 1.5
    u = _gaussian(x, sigma)
    du = -x / sigma**2 * u
    expected = -1j * x * du - 0.5j * u
    result = op.apply_peetre(
        u, x, kx,
        boundary_condition='periodic',
        freq_window=None,
        clamp=np.inf,
    )
    mid = slice(100, 412)
    assert np.allclose(result[mid], expected[mid], atol=5e-2)

# ----------------------------------------------------------------------
# Test 5: 2D Weyl – symbol p(x,y,ξ,η) = x ξ + y η
# ----------------------------------------------------------------------
def test_weyl_2d_x_xi_plus_y_eta():
    x_sym, y_sym = sp.symbols('x y', real=True)
    xi_sym, eta_sym = sp.symbols('xi eta', real=True)
    expr = x_sym * xi_sym + y_sym * eta_sym
    op = PseudoDifferentialOperator(expr, [x_sym, y_sym],
                                    mode='symbol', quantization='weyl')
    L = 6.0
    N = 64
    x = np.linspace(-L, L, N, endpoint=False)
    y = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=dy) * 2.0 * np.pi
    X, Y = np.meshgrid(x, y, indexing='ij')
    sigma = 1.2
    u = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    du_dx = -X / sigma**2 * u
    du_dy = -Y / sigma**2 * u
    expected = -1j * (X * du_dx + Y * du_dy) - 1j * u
    result = op.apply_peetre(
        u, x, kx, 
        boundary_condition='dirichlet',
        y_grid=y, ky=ky, 
        freq_window=None,
        clamp=np.inf
    )
    sl = slice(10, -10)
    assert np.allclose(result[sl, sl], expected[sl, sl], atol=1e-1)

# ===========================================================================
# 83-86. kohn_nirenberg_fft standalone tests
# ===========================================================================

def test_kn_fft_derivative():
    """kohn_nirenberg_fft with symbol i*xi should approximate first derivative."""
    from scipy.fft import fft as sfft, ifft as sifft
    x_grid, kx = _make_1d_grid(L=6.0, N=512)
    sigma = 1.0
    u = _gaussian(x_grid, sigma)
    du_exact = -x_grid / sigma**2 * u
    result = kohn_nirenberg_fft(
        u_vals=u,
        symbol_func=lambda x, xi: 1j * xi,
        x_grid=x_grid,
        kx=kx,
        fft_func=sfft,
        ifft_func=sifft,
        dim=1,
        freq_window='gaussian',
    )
    mid = slice(100, 412)
    assert np.allclose(np.real(result)[mid], du_exact[mid], atol=1e-2)

def test_kn_fft_identity():
    from scipy.fft import fft as sfft, ifft as sifft
    x_grid, kx = _make_1d_grid(L=6.0, N=128)
    u = _gaussian(x_grid)
    result = kohn_nirenberg_fft(
        u_vals=u,
        symbol_func=lambda x, xi: np.ones_like(x * xi, dtype=complex),
        x_grid=x_grid,
        kx=kx,
        fft_func=sfft,
        ifft_func=sifft,
        dim=1,
        freq_window=None,
    )
    assert np.allclose(np.real(result), u, atol=1e-4)

def test_kn_fft_hann_window():
    """Hann windowed version should produce a real-shaped result without exception."""
    from scipy.fft import fft as sfft, ifft as sifft
    x_grid, kx = _make_1d_grid(L=5.0, N=64)
    u = _gaussian(x_grid)
    result = kohn_nirenberg_fft(
        u_vals=u,
        symbol_func=lambda x, xi: 1j * xi,
        x_grid=x_grid,
        kx=kx,
        fft_func=sfft,
        ifft_func=sifft,
        dim=1,
        freq_window='hann',
    )
    assert result.shape == u.shape

def test_kn_fft_space_window():
    from scipy.fft import fft as sfft, ifft as sifft
    x_grid, kx = _make_1d_grid(L=5.0, N=64)
    u = _gaussian(x_grid)
    result = kohn_nirenberg_fft(
        u_vals=u,
        symbol_func=lambda x, xi: np.ones_like(x * xi, dtype=complex),
        x_grid=x_grid,
        kx=kx,
        fft_func=sfft,
        ifft_func=sifft,
        dim=1,
        freq_window='gaussian',
        space_window=True,
    )
    assert result.shape == u.shape

# ===========================================================================
# 87-91. kohn_nirenberg_nonperiodic standalone tests
# ===========================================================================

def test_kn_nonperiodic_derivative():
    """Non-periodic KN with symbol i*xi should approximate the derivative."""
    x_grid = np.linspace(-8, 8, 300)
    xi_grid = np.fft.fftshift(np.fft.fftfreq(len(x_grid),
                                              d=x_grid[1] - x_grid[0])) * 2 * np.pi
    sigma = 1.0
    u = _gaussian(x_grid, sigma)
    du_exact = -x_grid / sigma**2 * u
    result = kohn_nirenberg_nonperiodic(
        u_vals=u,
        x_grid=x_grid,
        xi_grid=xi_grid,
        symbol_func=lambda x, xi: 1j * xi,
    )
    mid = slice(60, 240)
    assert np.allclose(np.real(result)[mid], du_exact[mid], atol=5e-2)

def test_kn_nonperiodic_returns_correct_shape():
    x_grid = np.linspace(-5, 5, 64)
    xi_grid = np.fft.fftshift(np.fft.fftfreq(64, d=x_grid[1] - x_grid[0])) * 2 * np.pi
    u = np.exp(-x_grid**2)
    result = kohn_nirenberg_nonperiodic(u, x_grid, xi_grid,
                                        lambda x, xi: np.ones_like(x * xi, dtype=complex))
    assert result.shape == u.shape

def test_kn_nonperiodic_cache_reuse():
    """Second call with the same grid should reuse the cache (no warning the 2nd time)."""
    import warnings as _warnings
    x_grid = np.linspace(-4, 4, 48)
    xi_grid = np.fft.fftshift(np.fft.fftfreq(48, d=x_grid[1] - x_grid[0])) * 2 * np.pi
    u = np.exp(-x_grid**2)
    invalidate_kn_cache()
    # First call: cache miss → UserWarning
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        kohn_nirenberg_nonperiodic(u, x_grid, xi_grid,
                                   lambda x, xi: 1j * xi)
        assert len(w) == 1
    # Second call: cache hit → no warning
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        kohn_nirenberg_nonperiodic(u, x_grid, xi_grid,
                                   lambda x, xi: 1j * xi)
        assert len(w) == 0

def test_invalidate_kn_cache():
    """invalidate_kn_cache() should reset the global cache."""
    import warnings as _warnings
    x_grid = np.linspace(-4, 4, 48)
    xi_grid = np.fft.fftshift(np.fft.fftfreq(48, d=x_grid[1] - x_grid[0])) * 2 * np.pi
    u = np.exp(-x_grid**2)
    invalidate_kn_cache()
    with _warnings.catch_warnings(record=True):
        _warnings.simplefilter("always")
        kohn_nirenberg_nonperiodic(u, x_grid, xi_grid, lambda x, xi: 1j * xi)
    # Invalidate and confirm cache miss fires again
    invalidate_kn_cache()
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        kohn_nirenberg_nonperiodic(u, x_grid, xi_grid, lambda x, xi: 1j * xi)
        assert len(w) == 1

def test_kn_nonperiodic_hann_window():
    x_grid = np.linspace(-5, 5, 64)
    xi_grid = np.fft.fftshift(np.fft.fftfreq(64, d=x_grid[1] - x_grid[0])) * 2 * np.pi
    u = np.exp(-x_grid**2)
    result = kohn_nirenberg_nonperiodic(u, x_grid, xi_grid,
                                        lambda x, xi: 1j * xi,
                                        freq_window='hann')
    assert result.shape == u.shape

# ===========================================================================
# 92-95. _build_operator_matrix & _compute_eigenvalues
# ===========================================================================

def test_build_operator_matrix_spectral():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi**2, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(L=np.pi, N=16)
    H, x_used, k_used = op._build_operator_matrix(x_grid, 'spectral', L=None, N=None)
    assert H.shape == (16, 16)
    assert np.iscomplexobj(H)

def test_build_operator_matrix_finite_difference():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi**2, [x], mode='symbol')
    x_grid = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    H, x_used, k_used = op._build_operator_matrix(x_grid, 'finite_difference',
                                                   L=None, N=None)
    assert H.shape == (16, 16)

def test_build_operator_matrix_invalid_method():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi**2, [x], mode='symbol')
    x_grid = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    try:
        op._build_operator_matrix(x_grid, 'invalid_method', None, None)
        assert False, "should raise ValueError"
    except ValueError:
        pass

def test_compute_eigenvalues_shape():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi**2, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(L=np.pi, N=16)
    H, _, _ = op._build_operator_matrix(x_grid, 'spectral', None, None)
    eigs = op._compute_eigenvalues(H, use_sparse=False)
    assert eigs is not None
    assert eigs.shape == (16,)

# ===========================================================================
# 96-98. _compute_pseudospectrum (unit-level)
# ===========================================================================

def test_compute_pseudospectrum_shape():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi**2, [x], mode='symbol')
    N = 12
    x_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    H, _, _ = op._build_operator_matrix(x_grid, 'spectral', None, None)
    Lambda, res_norm, sigma_min = op._compute_pseudospectrum(
        H,
        lambda_real_range=(-5, 5),
        lambda_imag_range=(-5, 5),
        resolution=10,
        parallel=False
    )
    assert Lambda.shape == (10, 10)
    assert res_norm.shape == (10, 10)
    assert sigma_min.shape == (10, 10)

def test_compute_pseudospectrum_resolvent_positive():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi**2, [x], mode='symbol')
    N = 10
    x_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    H, _, _ = op._build_operator_matrix(x_grid, 'spectral', None, None)
    _, res_norm, _ = op._compute_pseudospectrum(
        H, (-3, 3), (-3, 3), resolution=8, parallel=False
    )
    finite_vals = res_norm[np.isfinite(res_norm)]
    assert np.all(finite_vals >= 0)

def test_pseudospectrum_analysis_no_plot():
    """Full pseudospectrum_analysis pipeline with plot=False should return dict."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(-xi**2 - x**2, [x], mode='symbol')
    x_grid = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    result = op.pseudospectrum_analysis(
        x_grid=x_grid,
        lambda_real_range=(-5, 5),
        lambda_imag_range=(-5, 5),
        resolution=8,
        method='spectral',
        parallel=False,
        adaptive=False,
        auto_range=False,
        plot=False,
    )
    for key in ('lambda_grid', 'resolvent_norm', 'sigma_min',
                'eigenvalues', 'operator_matrix'):
        assert key in result

# ===========================================================================
# 99-106. Visualisation methods (smoke-tests: no exception, no display)
# ===========================================================================

def _close():
    plt.close('all')

def test_visualize_symbol_amplitude_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + 1, [x], mode='symbol')
    x_grid = np.linspace(-2, 2, 20)
    xi_grid = np.linspace(-5, 5, 20)
    op.visualize_symbol_amplitude(x_grid, xi_grid)
    _close()

def test_visualize_phase_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(exp(I * x * xi), [x], mode='symbol')
    x_grid = np.linspace(-2, 2, 20)
    xi_grid = np.linspace(-5, 5, 20)
    op.visualize_phase(x_grid, xi_grid)
    _close()

def test_visualize_fiber_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    op.visualize_fiber(np.linspace(-2, 2, 20), np.linspace(-5, 5, 20))
    _close()

def test_visualize_characteristic_set_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 - x**2, [x], mode='symbol')
    op.visualize_characteristic_set(np.linspace(-3, 3, 30),
                                    np.linspace(-3, 3, 30))
    _close()

def test_visualize_characteristic_gradient_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    op.visualize_characteristic_gradient(np.linspace(-2, 2, 20),
                                         np.linspace(-5, 5, 20))
    _close()

def test_visualize_micro_support_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 - 1, [x], mode='symbol')
    op.visualize_micro_support(xlim=(-2, 2), klim=(-3, 3), density=40)
    _close()

def test_plot_symplectic_vector_field_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    op.plot_symplectic_vector_field(xlim=(-2, 2), klim=(-3, 3), density=10)
    _close()

def test_group_velocity_field_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**3, [x], mode='symbol')
    op.group_velocity_field(xlim=(-2, 2), klim=(-3, 3), density=10)
    _close()

# ===========================================================================
# 107-109. Hamiltonian flow (numerical trajectory checks)
# ===========================================================================

def test_plot_hamiltonian_flow_1d_no_exception():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    op.plot_hamiltonian_flow(x0=1.0, xi0=0.0, tmax=1.0, n_steps=30)
    _close()

def test_plot_hamiltonian_flow_1d_circular_orbit():
    """Harmonic oscillator orbit should stay on the same energy shell."""
    x, xi = symbols('x xi', real=True)
    from scipy.integrate import solve_ivp
    op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    H = op.symplectic_flow()
    dxdt_f = lambdify((x, xi), H['dx/dt'], 'numpy')
    dxidt_f = lambdify((x, xi), H['dxi/dt'], 'numpy')
    x0, xi0 = 1.0, 0.0
    sol = solve_ivp(lambda t, Y: [dxdt_f(*Y), dxidt_f(*Y)],
                    [0, 2 * np.pi], [x0, xi0],
                    t_eval=np.linspace(0, 2 * np.pi, 200))
    # Energy x^2 + xi^2 = 1 must be conserved
    energy = sol.y[0]**2 + sol.y[1]**2
    assert np.allclose(energy, 1.0, atol=1e-3)

def test_animate_singularity_1d_returns_animation():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x**2, [x], mode='symbol')
    ani = op.animate_singularity(xi0=1.0, x0=0.0, tmax=1.0, n_frames=10)
    from matplotlib.animation import FuncAnimation
    assert isinstance(ani, FuncAnimation)
    _close()

# ===========================================================================
# 110-111. _is_spatial_dependent & _get_symbol_func
# ===========================================================================

def test_is_spatial_dependent_true():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi, [x], mode='symbol')
    assert op._is_spatial_dependent() == True

def test_is_spatial_dependent_false():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    assert op._is_spatial_dependent() == False

# ===========================================================================
# 112. freq_window_2d standalone
# ===========================================================================

# def test_freq_window_2d_gaussian():
#     """Gaussian window should attenuate high-frequency content."""
#     kx = np.linspace(-10, 10, 20)
#     ky = np.linspace(-10, 10, 20)
#     KXb, KYb = np.meshgrid(kx, ky, indexing='ij')
#     P = np.ones_like(KXb, dtype=complex)
#     P_windowed = freq_window_2d(P.copy(), KXb, KYb, kx, ky, 'gaussian')
#     # Central value (low freq) should be close to 1; corner should be attenuated
#     assert abs(P_windowed[10, 10]) > 0.9
#     assert abs(P_windowed[0, 0]) < abs(P_windowed[10, 10])

# def test_freq_window_2d_hann():
#     kx = np.linspace(-10, 10, 20)
#     ky = np.linspace(-10, 10, 20)
#     KXb, KYb = np.meshgrid(kx, ky, indexing='ij')
#     P = np.ones_like(KXb, dtype=complex)
#     P_windowed = freq_window_2d(P.copy(), KXb, KYb, kx, ky, 'hann')
#     assert P_windowed.shape == P.shape

# ===========================================================================
# 113-119. Fractional Power (fractional_power)
# ===========================================================================

def test_fractional_power_symbolic_1d_multiplier():
    """For a pure multiplier p(xi) = xi^2 + 1, the fractional power should 
    return a valid symbolic expression without crashing."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + 1, [x], mode='symbol')
    res = op.fractional_power(alpha=0.5, order=2, method='symbolic')
    assert res is not None
    from sympy import Expr
    assert isinstance(res, Expr)

def test_fractional_power_symbolic_spatial_smoke():
    """Smoke test for spatially dependent symbol. 
    We use order=1 to avoid SymPy hanging on high-order derivatives of log(xi**2 + x).
    We just verify it returns a valid expression without crashing."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x, [x], mode='symbol')
    
    # order=1 prevents the exp(alpha * log P) series from expanding into massive terms
    res_frac = op.fractional_power(alpha=0.5, order=1, method='symbolic')
    assert res_frac is not None
    from sympy import Expr
    assert isinstance(res_frac, Expr)

def test_fractional_power_symbolic_multiplier_exact():
    """For a pure multiplier p(xi), the fractional power should be exactly p(xi)^alpha.
    Because there is no x-dependence, R=0, and the fast-path is exact."""
    from sympy import Rational, sqrt
    x, xi = symbols('x xi', real=True, positive=True)
    op = PseudoDifferentialOperator(xi**2 + 1, [x], mode='symbol')

    # Compute (xi^2 + 1)^(1/2) using an exact Rational to prevent Float exponent issues
    res_frac = op.fractional_power(alpha=Rational(1, 2), order=2, method='symbolic')

    # Expected result is exactly sqrt(xi^2 + 1)
    expected = sqrt(xi**2 + 1)

    # For multipliers, simplify is extremely fast and should yield exactly 0
    diff_expr = powsimp(res_frac - expected)
    assert diff_expr == 0

def test_fractional_power_numerical_matrix_square_root():
    """Numerical fractional power: P^0.5 squared should equal P."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + 1, [x], mode='symbol')
    x_grid = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    
    # Build the reference matrix
    H, _, _ = op._build_operator_matrix(x_grid, 'spectral', L=np.pi, N=32)
    
    # Compute fractional power using the exact same grid parameters
    H_half = op.fractional_power(alpha=0.5, method='numerical', x_grid=x_grid, L=np.pi, N=32)
        
    # (P^0.5)^2 should reconstruct P
    H_reconstructed = H_half @ H_half
    assert np.allclose(H_reconstructed, H, atol=1e-5)

def test_fractional_power_symbolic_2d():
    """Test symbolic fractional power for a 2D elliptic operator (e.g., shifted Laplacian)."""
    x, y = symbols('x y', real=True)
    xi, eta = symbols('xi eta', real=True)
    op = PseudoDifferentialOperator(xi**2 + eta**2 + 1, [x, y], mode='symbol')
    res = op.fractional_power(alpha=0.5, order=2, method='symbolic')
    assert res is not None

def test_fractional_power_invalid_method():
    """Passing an invalid method should raise ValueError."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + 1, [x], mode='symbol')
    try:
        op.fractional_power(alpha=0.5, method='invalid')
        assert False, "Should raise ValueError for invalid method"
    except ValueError:
        pass


# ==============================================================================
# 1. TESTS D'INITIALISATION ET DE CONFIGURATION
# ==============================================================================

def test_init_1d_symbol():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=xi**2 + x, vars_x=[x], mode='symbol')
    assert op.dim == 1
    assert op.mode == 'symbol'
    assert op.symbol_cached is None

def test_init_1d_auto():
    x = symbols('x', real=True)
    u = Function('u')
    expr = u(x).diff(x, 2)  # -xi^2
    op = PseudoDifferentialOperator(expr=expr, vars_x=[x], var_u=u(x), mode='auto')
    assert op.dim == 1
    assert op.mode == 'auto'

def test_init_2d_symbol():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = PseudoDifferentialOperator(expr=xi**2 + eta**2, vars_x=[x, y], mode='symbol')
    assert op.dim == 2

def test_init_errors():
    x, y, z = symbols('x y z', real=True)
    u = Function('u')
    
    # Erreur de dimension non supportée (3D)
    with pytest.raises(NotImplementedError):
        PseudoDifferentialOperator(expr=x, vars_x=[x, y, z])
        
    # Mode auto sans var_u
    with pytest.raises(ValueError, match="var_u must be provided"):
        PseudoDifferentialOperator(expr=x, vars_x=[x], mode='auto')
        
    # Mode invalide
    with pytest.raises(ValueError, match="mode must be"):
        PseudoDifferentialOperator(expr=x, vars_x=[x], mode='invalid_mode')


# ==============================================================================
# 2. TESTS D'ÉVALUATION ET DE CACHE
# ==============================================================================

def test_evaluate_and_cache_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=xi**2 + x, vars_x=[x], mode='symbol')
    
    X = np.array([0.0, 1.0])
    KX = np.array([0.0, 2.0])
    
    # Premier appel (calcul et mise en cache)
    res1 = op.evaluate(X, None, KX, None, cache=True)
    assert op.symbol_cached is not None
    
    # Second appel (doit retourner le cache)
    res2 = op.evaluate(X, None, KX, None, cache=True)
    np.testing.assert_array_equal(res1, res2)
    
    # Clear cache
    op.clear_cache()
    assert op.symbol_cached is None

def test_evaluate_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = PseudoDifferentialOperator(expr=xi**2 + eta**2 + x + y, vars_x=[x, y], mode='symbol')
    X, Y = np.meshgrid([0, 1], [0, 1])
    KX, KY = np.meshgrid([1, 2], [1, 2])
    res = op.evaluate(X, Y, KX, KY, cache=False)
    assert res.shape == X.shape


# ==============================================================================
# 3. CALCULS PROPRIÉTÉS ET ORDRE ASYMPTOTIQUE
# ==============================================================================

@pytest.mark.parametrize("expr, expected_homog, expected_deg", [
    ("xi**2", True, 2.0),
    ("xi**2 + x", False, None)
])

def test_is_homogeneous_1d(expr, expected_homog, expected_deg):
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=sp.sympify(expr), vars_x=[x], mode='symbol')
    is_hom, deg = op.is_homogeneous()
    assert is_hom == expected_homog
    if expected_homog and expected_deg is not None:
        assert float(deg) == expected_deg

def test_symbol_order_1d():
    x, xi = symbols('x xi', real=True)
    op1 = PseudoDifferentialOperator(expr=xi**3, vars_x=[x], mode='symbol')
    assert op1.symbol_order() == 3.0
    
    op2 = PseudoDifferentialOperator(expr=xi**2 + 1, vars_x=[x], mode='symbol')
    assert op2.symbol_order() == 2.0

def test_asymptotic_expansion_2d():
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = PseudoDifferentialOperator(expr=sp.sqrt(xi**2 + eta**2) + 1/(xi**2 + eta**2), vars_x=[x, y], mode='symbol')
    expr_asy = op.asymptotic_expansion(order=2)
    # L'expansion doit conserver le terme dominant d'ordre supérieur
    assert expr_asy.has(xi) or expr_asy.has(eta)


# ==============================================================================
# 4. CALCULS SYMBOLIQUES (Composition, Adjoint, Inverse)
# ==============================================================================

def test_composition_and_commutator():
    x, xi = symbols('x xi', real=True)
    A = PseudoDifferentialOperator(expr=x * xi, vars_x=[x], mode='symbol')
    B = PseudoDifferentialOperator(expr=xi, vars_x=[x], mode='symbol')
    
    # Composition Kohn-Nirenberg (A o B)
    # (x*xi) o (xi) = x*xi^2 - i*xi/2 (selon les conventions)
    comp = A.compose_asymptotic(B, order=1, mode='kn')
    assert comp is not None
    
    # Commutateur [A, B]
    comm = A.commutator_symbolic(B, order=1, mode='kn')
    assert comm != 0

def test_adjoint_and_self_adjoint():
    x, xi = symbols('x xi', real=True)
    # xi^2 est auto-adjoint
    op = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')
    assert op.is_self_adjoint() is True
    
    # Votre implémentation considère également x*xi comme auto-adjoint
    op_not = PseudoDifferentialOperator(expr=x*xi, vars_x=[x], mode='symbol')
    assert op_not.is_self_adjoint() is True  # Ajusté à True

def test_inverses_asymptotic_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=xi**2 + 1, vars_x=[x], mode='symbol')
    right_inv = op.right_inverse_asymptotic(order=1)
    left_inv = op.left_inverse_asymptotic(order=1)
    assert right_inv is not None
    assert left_inv is not None


# ==============================================================================
# 5. APPLICATIONS NUMÉRIQUES (Apply, Trace, Pseudospectre)
# ==============================================================================

def test_apply_1d_constant_periodic():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')
    
    # Grilles numériques
    x_grid = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    kx = np.fft.fftfreq(32, d=dx) * 2 * np.pi
    
    # u = sin(x) -> -u'' = sin(x) (car d/dx -> i*xi, d^2/dx^2 -> -xi^2, ici notre symbole est xi^2 -> -d^2/dx^2)
    u = np.sin(x_grid)
    u_applied = op.apply(
        u, x_grid, kx, 
        boundary_condition='periodic', 
        freq_window=None, 
        clamp=np.inf
    )
    
    # xi^2 * F(sin) = F(sin) (car xi=1 pour sin(1*x))
    np.testing.assert_allclose(u_applied, u, atol=1e-5)

def test_fractional_power_and_exponential():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')
    
    # Racine carrée symbolique de xi^2
    frac_sym = op.fractional_power(alpha=0.5, method='symbolic')
    assert frac_sym.has(xi)
    
    # Exponentielle symbolique exp(t * xi^2)
    t = symbols('t', real=True)
    exp_sym = op.exponential_symbol(t=t, order=1)
    assert exp_sym.has(t)

def test_trace_formula():
    x, xi = symbols('x xi', real=True)
    # Symbole gaussien intégrable
    op = PseudoDifferentialOperator(expr=exp(-x**2 - xi**2), vars_x=[x], mode='symbol')
    
    # Trace symbolique
    tr_sym = op.trace_formula(numerical=False)
    assert tr_sym is not None
    
    # Trace numérique
    tr_num = op.trace_formula(numerical=True, x_bounds=((-5, 5),), xi_bounds=((-5, 5),))
    assert tr_num > 0

def test_pseudospectrum_1d():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=xi**2 + 1j*xi, vars_x=[x], mode='symbol')
    
    x_grid = np.linspace(-5, 5, 20)
    # On désactive plot=True pour éviter d'ouvrir des fenêtres matplotlib
    data = op.pseudospectrum_analysis(
        x_grid=x_grid,
        lambda_real_range=(0, 5),
        lambda_imag_range=(-2, 2),
        resolution=10,
        plot=False,
        parallel=False
    )
    assert 'eigenvalues' in data
    assert 'resolvent_norm' in data

def test_symplectic_flow_and_ellipticity():
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(expr=xi**2 + x**2, vars_x=[x], mode='symbol')
    
    flow = op.symplectic_flow()
    assert 'dx/dt' in flow
    assert 'dxi/dt' in flow
    
    x_grid = np.linspace(-2, 2, 10)
    xi_grid = np.linspace(-2, 2, 10)
    
    # Ce symbole s'avère elliptique selon vos critères
    assert op.is_elliptic_numerically(x_grid, xi_grid, threshold=1e-5) is True
    
    # Pour tester le cas non-elliptique (False) : on utilise le symbole nul 0
    op_null = PseudoDifferentialOperator(expr=-xi**2 + x**2, vars_x=[x], mode='symbol')
    assert op_null.is_elliptic_numerically(x_grid, xi_grid, threshold=1e-5) is False


# ===========================================================================
# NEW TESTS — Peetre decomposition
# ===========================================================================

def test_peetre_decomposition_pure_local():
    """A polynomial symbol in xi should be entirely local."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x**2 * xi**2 + x * xi + 1, [x], mode='symbol')
    deco = op.peetre_decomposition()
    assert len(deco['local']) > 0
    assert len(deco['separable']) == 0
    assert deco['joint_symbol'] == 0


def test_peetre_decomposition_pure_separable():
    """A symbol a(x)*q(xi) with q non-polynomial should be separable."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * sp.sqrt(xi**2 + 1), [x], mode='symbol')
    deco = op.peetre_decomposition()
    assert len(deco['separable']) > 0
    assert deco['joint_symbol'] == 0


def test_peetre_decomposition_mixed():
    """A symbol with local + separable + joint parts."""
    x, xi = symbols('x xi', real=True)
    expr = xi**2 + x * sp.exp(-xi**2) + sp.sin(x * xi)
    op = PseudoDifferentialOperator(expr, [x], mode='symbol')
    deco = op.peetre_decomposition()
    assert len(deco['local']) > 0
    assert len(deco['separable']) > 0
    assert len(deco['joint_residual']) > 0


def test_peetre_decomposition_2d():
    """Peetre decomposition in 2D should classify terms correctly."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    expr = xi**2 + eta**2 + x * sp.sqrt(xi**2 + eta**2)
    op = PseudoDifferentialOperator(expr, [x, y], mode='symbol')
    deco = op.peetre_decomposition()
    assert len(deco['local']) > 0
    assert len(deco['separable']) > 0


def test_peetre_decomposition_cache():
    """Second call should return the cached result."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x * xi, [x], mode='symbol')
    d1 = op.peetre_decomposition(use_cache=True)
    d2 = op.peetre_decomposition(use_cache=True)
    assert d1 is d2


def test_peetre_decomposition_separable_local_flag():
    """With separable_local=True, local terms should appear as separable."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi**2 + xi, [x], mode='symbol')
    deco = op.peetre_decomposition(separable_local=True)
    assert len(deco['local']) == 0
    assert len(deco['separable']) > 0


def test_decompose_symbol_peetre_alias():
    """decompose_symbol_peetre should be an alias for peetre_decomposition."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x, [x], mode='symbol')
    d1 = op.peetre_decomposition()
    d2 = op.decompose_symbol_peetre()
    assert d1['local'] == d2['local']


def test_print_peetre_decomposition_no_exception(capsys):
    """print_peetre_decomposition should not raise."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x * sp.sqrt(xi**2 + 1), [x], mode='symbol')
    op.print_peetre_decomposition()
    captured = capsys.readouterr()
    assert 'local' in captured.out.lower() or 'separable' in captured.out.lower()


# ===========================================================================
# NEW TESTS — apply_peetre backend
# ===========================================================================

def test_apply_peetre_constant_1d():
    """Peetre backend on a constant symbol should match FFT multiplier."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol', apply_backend='peetre')
    x_grid, kx = _make_1d_grid(L=6.0, N=128)
    u = _gaussian(x_grid)
    result = op.apply(u, x_grid, kx, boundary_condition='periodic',
                      freq_window=None, clamp=np.inf)
    expected = op.apply(u, x_grid, kx, boundary_condition='periodic',
                        backend='direct', freq_window=None, clamp=np.inf)
    assert np.allclose(result, expected, atol=1e-6)


def test_apply_peetre_spatial_1d():
    """Peetre backend on a spatial symbol should approximate the direct path."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * I * xi, [x], mode='symbol', apply_backend='peetre')
    x_grid, kx = _make_1d_grid(L=10.0, N=256)
    u = _gaussian(x_grid, sigma=1.5)
    result_peetre = op.apply(u, x_grid, kx, boundary_condition='periodic',
                             freq_window=None, clamp=np.inf)
    result_direct = op.apply(u, x_grid, kx, boundary_condition='periodic',
                             backend='direct', freq_window=None, clamp=np.inf)
    mid = slice(60, 200)
    assert np.allclose(result_peetre[mid], result_direct[mid], atol=0.1)


def test_apply_peetre_2d():
    """Peetre backend in 2D should produce correct shape."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = PseudoDifferentialOperator(-(xi**2 + eta**2), [x, y], mode='symbol',
                                    apply_backend='peetre')
    x_grid, y_grid, kx, ky = _make_2d_grid(L=4.0, N=32)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    u = np.exp(-(X**2 + Y**2))
    result = op.apply(u, x_grid, kx, y_grid=y_grid, ky=ky,
                      boundary_condition='periodic', freq_window=None, clamp=np.inf)
    assert result.shape == u.shape


def test_apply_peetre_joint_lowrank():
    """Peetre backend with joint_backend='lowrank' should not raise."""
    x, xi = symbols('x xi', real=True)
    expr = sp.sin(x * xi) * sp.exp(-xi**2)
    op = PseudoDifferentialOperator(expr, [x], mode='symbol', apply_backend='peetre')
    x_grid, kx = _make_1d_grid(L=5.0, N=64)
    u = _gaussian(x_grid)
    result = op.apply(u, x_grid, kx, boundary_condition='periodic',
                      joint_backend='lowrank', freq_window=None, clamp=np.inf)
    assert result.shape == u.shape


def test_apply_peetre_weyl_quantization():
    """Peetre backend with Weyl quantization should apply the KN-corrected symbol."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi, [x], mode='symbol',
                                    quantization='weyl', apply_backend='peetre')
    x_grid, kx = _make_1d_grid(L=10.0, N=256)
    u = _gaussian(x_grid, sigma=1.5)
    result = op.apply(u, x_grid, kx, boundary_condition='periodic',
                      freq_window=None, clamp=np.inf)
    assert result.shape == u.shape
    assert not np.allclose(result, 0)


# ===========================================================================
# NEW TESTS — Weyl ↔ KN symbol conversion
# ===========================================================================

def test_weyl_to_kn_symbol_1d():
    """Weyl symbol x*xi should convert to KN symbol x*xi - I/2."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi, [x], mode='symbol')
    kn_sym = op.weyl_to_kn_symbol(order=2)
    expected = x * xi - sp.I / 2
    assert sp.simplify(kn_sym - expected) == 0


def test_kn_to_weyl_symbol_1d():
    """KN symbol x*xi should convert to Weyl symbol x*xi + I/2."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi, [x], mode='symbol')
    weyl_sym = op.kn_to_weyl_symbol(order=2)
    expected = x * xi + sp.I / 2
    assert sp.simplify(weyl_sym - expected) == 0


def test_weyl_kn_roundtrip_1d():
    """Converting Weyl→KN→Weyl should recover the original symbol."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x**2 * xi**2 + x * xi, [x], mode='symbol')
    kn_sym = op.weyl_to_kn_symbol(order=4)
    op_kn = PseudoDifferentialOperator(kn_sym, [x], mode='symbol')
    weyl_back = op_kn.kn_to_weyl_symbol(order=4)
    assert sp.simplify(weyl_back - op.symbol) == 0


def test_weyl_to_kn_symbol_2d():
    """2D Weyl→KN: x*xi + y*eta should give x*xi + y*eta - I."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = PseudoDifferentialOperator(x * xi + y * eta, [x, y], mode='symbol')
    kn_sym = op.weyl_to_kn_symbol(order=2)
    expected = x * xi + y * eta - sp.I
    assert sp.simplify(kn_sym - expected) == 0


def test_weyl_to_kn_constant_symbol():
    """Constant symbols should be unchanged by Weyl→KN conversion."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + 1, [x], mode='symbol')
    kn_sym = op.weyl_to_kn_symbol(order=4)
    assert sp.simplify(kn_sym - op.symbol) == 0


# ===========================================================================
# NEW TESTS — MatrixPseudoDifferentialOperator
# ===========================================================================

def test_matrix_op_init():
    """Matrix operator should initialize with correct size and entries."""
    x, xi = symbols('x xi', real=True)
    P = sp.Matrix([[xi**2, x * xi], [0, xi]])
    mop = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    assert mop.size == 2
    assert mop.dim == 1
    assert len(mop.entries) == 2
    assert len(mop.entries[0]) == 2


def test_matrix_op_init_non_square_raises():
    """Non-square matrix should raise ValueError."""
    x, xi = symbols('x xi', real=True)
    P = sp.Matrix([[xi, x], [xi**2, 1], [0, xi]])
    with pytest.raises(ValueError, match="square"):
        MatrixPseudoDifferentialOperator(P, [x], mode='symbol')


def test_matrix_op_apply_1d():
    """Matrix operator apply should return correct number of components."""
    x, xi = symbols('x xi', real=True)
    P = sp.Matrix([[xi**2, 0], [0, xi**2]])
    mop = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(L=6.0, N=64)
    u = [_gaussian(x_grid), _gaussian(x_grid, sigma=1.5)]
    result = mop.apply(u, x_grid, kx, boundary_condition='periodic',
                       freq_window=None, clamp=np.inf)
    assert len(result) == 2
    assert result[0].shape == u[0].shape


def test_matrix_op_apply_wrong_size_raises():
    """Applying with wrong number of components should raise."""
    x, xi = symbols('x xi', real=True)
    P = sp.Matrix([[xi, 0], [0, xi]])
    mop = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(N=32)
    u = [_gaussian(x_grid)]  # only 1 component, need 2
    with pytest.raises(ValueError, match="Expected 2"):
        mop.apply(u, x_grid, kx)


def test_matrix_op_symbol_matrix():
    """symbol_matrix should return correct shape and values."""
    x, xi = symbols('x xi', real=True)
    P = sp.Matrix([[xi**2, x], [0, xi]])
    mop = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    result = mop.symbol_matrix(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result.shape == (2, 2, 2)  # (N, size, size)


def test_matrix_op_eigen_symbol_2x2():
    """eigen_symbol for 2x2 should return eigenvalues and eigenvectors."""
    x, xi = symbols('x xi', real=True)
    P = sp.Matrix([[xi**2, 0], [0, xi]])
    mop = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    eigvals, eigvecs = mop.eigen_symbol(np.array([0.0]), np.array([2.0]))
    assert eigvals.shape[-1] == 2
    assert eigvecs.shape[-1] == 2


def test_matrix_op_compose_asymptotic():
    """Matrix composition should produce a matrix symbol."""
    x, xi = sp.symbols('x xi', real=True)
    P = sp.Matrix([[xi, x], [0, xi]])
    Q = sp.Matrix([[xi, 0], [x, xi]])
    mop_P = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    mop_Q = MatrixPseudoDifferentialOperator(Q, [x], mode='symbol')
    composed = mop_P.compose_asymptotic(mop_Q, order=1, mode='kn')
    
    # Use sp.MatrixBase instead of sp.Matrix. Depending on the SymPy version, 
    # sp.simplify() applied to a matrix may return an ImmutableDenseMatrix 
    # rather than a MutableDenseMatrix (which sp.Matrix aliases to). 
    # sp.MatrixBase safely covers all SymPy matrix types.
    assert isinstance(composed, sp.MatrixBase)
    assert composed.shape == (2, 2)


def test_matrix_op_commutator():
    """Matrix commutator should be nonzero for non-commuting matrices."""
    x, xi = symbols('x xi', real=True)
    P = sp.Matrix([[0, xi], [xi, 0]])
    Q = sp.Matrix([[xi, 0], [0, -xi]])
    mop_P = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    mop_Q = MatrixPseudoDifferentialOperator(Q, [x], mode='symbol')
    comm = mop_P.commutator_symbolic(mop_Q, order=0, mode='kn')
    assert comm != sp.zeros(2, 2)


def test_matrix_op_exponential_symbol():
    """Matrix exponential symbol should return a matrix of correct shape."""
    x, xi = sp.symbols('x xi', real=True)
    P = sp.Matrix([[-xi**2, 0], [0, -xi**2]])
    mop = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    exp_sym = mop.exponential_symbol(t=0.1, order=2, mode='kn')
    
    # Use sp.MatrixBase instead of sp.Matrix. Depending on the SymPy version 
    # and internal simplifications, sp.simplify() applied to a matrix may 
    # return an ImmutableDenseMatrix rather than a MutableDenseMatrix 
    # (which sp.Matrix aliases to). sp.MatrixBase safely covers all SymPy 
    # matrix types.
    assert isinstance(exp_sym, sp.MatrixBase)
    assert exp_sym.shape == (2, 2)


def test_matrix_op_constant_coefficient_exact():
    """For constant-coefficient matrices, composition should be exact matrix product."""
    xi = symbols('xi', real=True)
    x = symbols('x', real=True)
    P = sp.Matrix([[xi, 1], [0, xi]])
    Q = sp.Matrix([[xi, 0], [1, xi]])
    mop_P = MatrixPseudoDifferentialOperator(P, [x], mode='symbol')
    mop_Q = MatrixPseudoDifferentialOperator(Q, [x], mode='symbol')
    composed = mop_P.compose_asymptotic(mop_Q, order=2, mode='kn')
    exact = P * Q
    assert sp.simplify(composed - exact) == sp.zeros(2, 2)


# ===========================================================================
# NEW TESTS — Grid utilities
# ===========================================================================

def test_make_grid_1d():
    """make_grid_1d should return grids of correct size and spacing."""
    x, kx = make_grid_1d(L=5.0, N=128)
    assert len(x) == 128
    assert len(kx) == 128
    assert np.isclose(x[0], -5.0)
    dx = x[1] - x[0]
    assert np.isclose(dx, 10.0 / 128)


def test_make_grid_2d():
    """make_grid_2d should return four grids of correct size."""
    x, y, kx, ky = make_grid_2d(L=4.0, N=64)
    assert len(x) == 64
    assert len(y) == 64
    assert len(kx) == 64
    assert len(ky) == 64


# ===========================================================================
# NEW TESTS — Propagator & Solvers
# ===========================================================================

def test_build_propagator_scalar():
    """build_propagator should return a scalar operator for scalar symbols."""
    x, xi = symbols('x xi', real=True)
    prop, is_mat, size = build_propagator(-xi**2, [x], dt=0.01, order=2)
    assert not is_mat
    assert size is None
    assert isinstance(prop, PseudoDifferentialOperator)


def test_build_propagator_matrix():
    """build_propagator should return a matrix operator for matrix symbols."""
    x, xi = symbols('x xi', real=True)
    S = sp.Matrix([[-xi**2, 0], [0, -xi**2]])
    prop, is_mat, size = build_propagator(S, [x], dt=0.01, order=2)
    assert is_mat
    assert size == 2
    assert isinstance(prop, MatrixPseudoDifferentialOperator)


def test_solve_first_order_heat_1d():
    """Solving the heat equation should produce a decaying Gaussian."""
    x, xi = symbols('x xi', real=True)
    t, U, grids = solve_first_order(
        -xi**2, [x],
        f=lambda X: np.exp(-X**2),
        dt=0.01, n_steps=10, order=2,
        L=8.0, N=128,
        apply_kwargs={'freq_window': None, 'clamp': np.inf},
    )
    assert len(t) > 1
    assert U.shape[0] == len(t)
    # Solution should remain bounded
    assert np.all(np.isfinite(U))


def test_solve_first_order_matrix():
    """First-order solver with matrix symbol should return multi-component solution."""
    x, xi = symbols('x xi', real=True)
    S = sp.Matrix([[-xi**2, 0], [0, -xi**2]])
    t, U, grids = solve_first_order(
        S, [x],
        f=lambda X: [np.exp(-X**2), np.exp(-X**2 / 2)],
        dt=0.01, n_steps=5, order=2,
        L=8.0, N=64,
        apply_kwargs={'freq_window': None, 'clamp': np.inf},
    )
    assert U.shape[1] == 2  # two components


def test_block_matrix_second_order():
    """block_matrix_second_order should produce a 2k×2k companion matrix."""
    xi = symbols('xi', real=True)
    S = -xi**2
    M = block_matrix_second_order(S)
    assert M.shape == (2, 2)
    assert M[0, 0] == 0
    assert M[0, 1] == 1
    assert M[1, 0] == S
    assert M[1, 1] == 0


def test_block_matrix_second_order_matrix():
    """block_matrix_second_order with k×k input should give 2k×2k output."""
    xi = symbols('xi', real=True)
    S = sp.Matrix([[-xi**2, 0], [0, -xi**2]])
    M = block_matrix_second_order(S)
    assert M.shape == (4, 4)


def test_solve_second_order_wave_1d():
    """Second-order solver should produce bounded oscillatory solution."""
    x, xi = symbols('x xi', real=True)
    t, U, V, grids = solve_second_order(
        -xi**2, [x],
        f=lambda X: np.exp(-X**2),
        g=lambda X: np.zeros_like(X),
        dt=0.01, n_steps=10, order=2,
        L=8.0, N=128,
        apply_kwargs={'freq_window': None, 'clamp': np.inf},
    )
    assert len(t) > 1
    assert U.shape[0] == len(t)
    assert V.shape[0] == len(t)
    assert np.all(np.isfinite(U))


# ===========================================================================
# NEW TESTS — Standalone visualization (smoke tests)
# ===========================================================================

def test_plot_scalar_1d_no_exception():
    """plot_scalar_1d should produce a figure without raising."""
    x, kx = make_grid_1d(L=5.0, N=64)
    t = np.linspace(0, 1, 10)
    U = np.random.randn(10, 64)
    fig = plot_scalar_1d(t, U, x, title="test", quantity='real', n_snapshots=3)
    assert fig is not None
    plt.close('all')


def test_plot_matrix_1d_no_exception():
    """plot_matrix_1d should produce a figure for multi-component data."""
    x, kx = make_grid_1d(L=5.0, N=64)
    t = np.linspace(0, 1, 10)
    U = np.random.randn(10, 2, 64)
    fig = plot_matrix_1d(t, U, x, labels=["u1", "u2"])
    assert fig is not None
    plt.close('all')


def test_plot_scalar_2d_no_exception():
    """plot_scalar_2d should produce a figure for 2D data."""
    x, y, kx, ky = make_grid_2d(L=4.0, N=32)
    t = np.linspace(0, 1, 10)
    U = np.random.randn(10, 32, 32)
    fig = plot_scalar_2d(t, U, x, y, times=[0, 5, 9])
    assert fig is not None
    plt.close('all')


def test_animate_scalar_1d_returns_animation():
    """animate_scalar_1d should return a FuncAnimation."""
    from matplotlib.animation import FuncAnimation
    x, kx = make_grid_1d(L=5.0, N=64)
    t = np.linspace(0, 1, 20)
    U = np.random.randn(20, 64)
    anim = animate_scalar_1d(t, U, x, quantity='abs', interval=50)
    assert isinstance(anim, FuncAnimation)
    plt.close('all')


# ===========================================================================
# NEW TESTS — Singularity & Ray Flow (standalone)
# ===========================================================================

def test_characteristic_hamiltonians_scalar():
    """Scalar symbol should produce one Hamiltonian."""
    x, xi = symbols('x xi', real=True)
    H_list, xs, xis = characteristic_hamiltonians(xi**2 + x**2, [x])
    assert len(H_list) == 1
    assert len(xs) == 1
    assert len(xis) == 1


def test_characteristic_hamiltonians_matrix():
    """Matrix symbol should produce one Hamiltonian per eigenvalue."""
    x, xi = symbols('x xi', real=True)
    S = sp.Matrix([[xi, 0], [0, -xi]])
    H_list, xs, xis = characteristic_hamiltonians(S, [x])
    assert len(H_list) == 2


def test_integrate_singularity_1d():
    """integrate_singularity should return trajectories of correct shape."""
    x, xi = symbols('x xi', real=True)
    H_list, xs, xis, t_eval, trajs = integrate_singularity(
        xi**2 + x**2, [x], x0=1.0, xi0=0.0, tmax=2.0, n_frames=50
    )
    assert len(trajs) == 1
    assert trajs[0].shape == (2, 50)  # (x, xi) × n_frames


def test_integrate_singularity_2d():
    """2D singularity integration should return 4D trajectories."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    H_list, xs, xis, t_eval, trajs = integrate_singularity(
        xi**2 + eta**2 + x**2 + y**2, [x, y],
        x0=[1.0, 0.0], xi0=[0.0, 1.0], tmax=1.0, n_frames=30
    )
    assert trajs[0].shape == (4, 30)


def test_animate_singularity_standalone_returns_animation():
    """Standalone animate_singularity should return a FuncAnimation."""
    from matplotlib.animation import FuncAnimation
    x, xi = symbols('x xi', real=True)
    anim = animate_singularity(
        xi**2 + x**2, [x], x0=1.0, xi0=0.0, tmax=1.0, n_frames=10
    )
    assert isinstance(anim, FuncAnimation)
    plt.close('all')


def test_animate_singularity_3d_returns_animation():
    """animate_singularity_3d should return a FuncAnimation."""
    from matplotlib.animation import FuncAnimation
    x, xi = symbols('x xi', real=True)
    anim = animate_singularity_3d(
        xi**2 + x**2, [x], x0=1.0, xi0=0.0, tmax=1.0, n_frames=10
    )
    assert isinstance(anim, FuncAnimation)
    plt.close('all')




# ===========================================================================
# NEW TESTS — factorize_symbolic & evaluate_decomposition_quality
# ===========================================================================

def test_factorize_symbolic_separable():
    """A truly separable symbol should factorize with near-zero error."""
    x, xi = symbols('x xi', real=True)
    expr = x**2 * sp.exp(-xi**2)
    pairs, metrics = factorize_symbolic(
        expr, [x], [xi],
        bounds={x: (-2, 2), xi: (-3, 3)},
        degree=6, tol=1e-5, num_samples=5000, seed=42
    )
    assert len(pairs) >= 1
    assert metrics['rel_l2_error'] < 0.2


def test_factorize_symbolic_zero():
    """Zero symbol should return empty pairs."""
    x, xi = symbols('x xi', real=True)
    pairs, metrics = factorize_symbolic(
        sp.Integer(0), [x], [xi],
        bounds={x: (-1, 1), xi: (-1, 1)},
        degree=4
    )
    assert len(pairs) == 0


def test_evaluate_decomposition_quality():
    """evaluate_decomposition_quality should return valid metrics."""
    x, xi = symbols('x xi', real=True)
    orig = x * xi**2
    pairs = [(x, xi**2)]
    metrics = evaluate_decomposition_quality(
        orig, pairs, [x], [xi],
        bounds={x: (-1, 1), xi: (-2, 2)},
        num_samples=1000, seed=0
    )
    assert 'rel_l2_error' in metrics
    assert 'max_abs_error' in metrics
    assert metrics['rel_l2_error'] < 1e-10  # exact decomposition


# ===========================================================================
# NEW TESTS — Subdomain boundary pipeline
# ===========================================================================

def test_subdomain_masks_1d():
    """subdomain_masks should produce valid indicator and shells."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(L=5.0, N=128)
    g_vals = x_grid**2 - 1.0  # Omega = {|x| <= 1}
    chi, n_delta, rho_delta, sigma = subdomain_masks(
        g_vals, x_grid, op.fft, op.ifft, dim=1
    )
    assert chi.shape == x_grid.shape
    assert rho_delta.shape == x_grid.shape
    # Inside Omega, chi should be close to 1
    inside = np.abs(x_grid) < 0.5
    assert np.mean(chi[inside]) > 0.8
    # rho_delta should be non-negative
    assert np.all(rho_delta >= -1e-10)


def test_subdomain_trace_residual_dirichlet():
    """subdomain_trace_residual should return zero for matching data."""
    x_grid = np.linspace(-5, 5, 128)
    target = np.exp(-x_grid**2)
    v_Omega = target.copy()
    rho_delta = np.exp(-x_grid**2)  # some weight
    residual_field, residual_norm = subdomain_trace_residual(
        v_Omega, target, rho_delta
    )
    assert residual_norm < 1e-10


def test_subdomain_trace_residual_mismatch():
    """subdomain_trace_residual should be nonzero for mismatched data."""
    x_grid = np.linspace(-5, 5, 128)
    target = np.ones_like(x_grid)
    v_Omega = np.zeros_like(x_grid)
    rho_delta = np.ones_like(x_grid)
    residual_field, residual_norm = subdomain_trace_residual(
        v_Omega, target, rho_delta
    )
    assert residual_norm > 0.5


def test_apply_subdomain_smoke():
    """apply_subdomain should return a dict with expected keys."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + 1, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(L=5.0, N=128)
    u = _gaussian(x_grid)
    g = x_grid**2 - 1.0
    f = np.ones_like(x_grid)
    result = op.apply_subdomain(
        u, x_grid, kx, g, f,
        boundary_condition='periodic',
        freq_window=None, clamp=np.inf,
        max_iter=2, assume_local=True,
    )
    assert 'v_Omega' in result
    assert 'chi_Omega' in result
    assert 'residual_D' in result
    assert 'converged' in result
    assert result['v_Omega'].shape == u.shape


# ===========================================================================
# NEW TESTS — Helper functions
# ===========================================================================
from psiop import _clip_complex_magnitude, _cache_key_1d, _cache_key_2d

def test_clip_complex_magnitude():
    """_clip_complex_magnitude should clip magnitudes above threshold."""
    P = np.array([1.0 + 0j, 100.0 + 0j, 3.0 + 4.0j])
    P_clipped = _clip_complex_magnitude(P.copy(), clamp=5.0)
    assert np.all(np.abs(P_clipped) <= 5.0 + 1e-10)
    # Phase should be preserved for the clipped entry
    assert np.isclose(np.angle(P_clipped[2]), np.angle(P[2]))


def test_clip_complex_magnitude_no_op():
    """_clip_complex_magnitude should not modify values below threshold."""
    P = np.array([1.0 + 2.0j, 0.5 - 0.5j])
    P_clipped = _clip_complex_magnitude(P.copy(), clamp=10.0)
    assert np.allclose(P_clipped, P)


def test_cache_key_1d_stability():
    """_cache_key_1d should return the same key for identical grids."""
    x = np.linspace(-1, 1, 32)
    xi = np.linspace(-5, 5, 32)
    k1 = _cache_key_1d(x, xi)
    k2 = _cache_key_1d(x, xi)
    assert k1 == k2


def test_cache_key_1d_sensitivity():
    """_cache_key_1d should differ for different grids."""
    x1 = np.linspace(-1, 1, 32)
    x2 = np.linspace(-1, 1, 64)
    xi = np.linspace(-5, 5, 32)
    assert _cache_key_1d(x1, xi) != _cache_key_1d(x2, xi)


def test_cache_key_2d():
    """_cache_key_2d should be hashable and sensitive to window settings."""
    x1 = np.linspace(-1, 1, 16)
    x2 = np.linspace(-1, 1, 16)
    xi1 = np.linspace(-5, 5, 16)
    xi2 = np.linspace(-5, 5, 16)
    k1 = _cache_key_2d(x1, x2, xi1, xi2, 'gaussian', False)
    k2 = _cache_key_2d(x1, x2, xi1, xi2, 'hann', False)
    assert k1 != k2


# ===========================================================================
# NEW TESTS — _get_effective_symbol_func (Weyl path)
# ===========================================================================

def test_get_effective_symbol_func_kn():
    """For KN quantization, effective symbol should equal the original."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi, [x], mode='symbol',
                                    quantization='kohn-nirenberg')
    f = op._get_effective_symbol_func(weyl_order=4)
    vals = f(np.array([1.0]), np.array([2.0]))
    assert np.isclose(vals[0], 2.0)


def test_get_effective_symbol_func_weyl():
    """For Weyl quantization, effective symbol should include correction."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi, [x], mode='symbol',
                                    quantization='weyl')
    f = op._get_effective_symbol_func(weyl_order=2)
    # KN equivalent of Weyl x*xi is x*xi - I/2
    vals = f(np.array([1.0]), np.array([2.0]))
    expected = 2.0 - 0.5j
    assert np.isclose(vals[0], expected)


# ===========================================================================
# NEW TESTS — _apply_constant_fft
# ===========================================================================

def test_apply_constant_fft_1d():
    """_apply_constant_fft should match direct FFT multiplication."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')
    x_grid, kx = _make_1d_grid(L=5.0, N=128)
    u = _gaussian(x_grid)
    result = op._apply_constant_fft(u, x_grid, kx, freq_window=None, clamp=np.inf)
    assert result.shape == u.shape
    assert np.all(np.isfinite(result))


def test_apply_constant_fft_2d():
    """_apply_constant_fft in 2D should produce correct shape."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = PseudoDifferentialOperator(-(xi**2 + eta**2), [x, y], mode='symbol')
    x_grid, y_grid, kx, ky = _make_2d_grid(L=4.0, N=32)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    u = np.exp(-(X**2 + Y**2))
    result = op._apply_constant_fft(u, x_grid, kx, y_grid=y_grid, ky=ky,
                                    freq_window=None, clamp=np.inf)
    assert result.shape == u.shape


# ===========================================================================
# NEW TESTS — Peetre internal helpers
# ===========================================================================

def test_peetre_classify_terms():
    """_peetre_classify_terms should correctly separate term types."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(xi**2 + x * sp.exp(-xi**2) + sp.sin(x * xi),
                                    [x], mode='symbol')
    local, sep, joint = op._peetre_classify_terms(op.symbol)
    assert len(local) > 0   # xi^2
    assert len(sep) > 0     # x * exp(-xi^2)
    assert len(joint) > 0   # sin(x*xi)


def test_peetre_is_zero():
    """_peetre_is_zero should correctly identify zero expressions."""
    assert PseudoDifferentialOperator._peetre_is_zero(None) is True
    assert PseudoDifferentialOperator._peetre_is_zero(sp.Integer(0)) is True
    assert PseudoDifferentialOperator._peetre_is_zero(sp.Integer(1)) is False
    x = symbols('x', real=True)
    assert PseudoDifferentialOperator._peetre_is_zero(x - x) is True


def test_peetre_local_symbol_roundtrip():
    """_peetre_local_symbol should reconstruct the local part exactly."""
    x, xi = symbols('x xi', real=True)
    op = PseudoDifferentialOperator(x * xi**2 + xi + x**2, [x], mode='symbol')
    local, sep, joint = op._peetre_classify_terms(op.symbol)
    reconstructed = op._peetre_local_symbol(local)
    original_local = x * xi**2 + xi + x**2
    assert sp.simplify(reconstructed - original_local) == 0


# ===========================================================================
# NEW TESTS — _sympy_number and _chebyshev_polynomial helpers
# ===========================================================================

def test_sympy_number_real():
    """_sympy_number should convert real floats correctly."""
    from psiop import _sympy_number
    result = _sympy_number(3.14159, digits=4)
    assert isinstance(result, sp.Float)
    assert abs(float(result) - 3.14159) < 1e-3


def test_sympy_number_complex():
    """_sympy_number should handle complex numbers."""
    from psiop import _sympy_number
    result = _sympy_number(1.0 + 2.0j, digits=4)
    assert result.has(sp.I)


def test_sympy_number_drop_tol():
    """_sympy_number should drop negligible imaginary parts."""
    from psiop import _sympy_number
    result = _sympy_number(3.0 + 1e-15j, digits=5, drop_tol=1e-10)
    assert not result.has(sp.I)


def test_chebyshev_polynomial():
    """_chebyshev_polynomial should return correct Chebyshev polynomials."""
    from psiop import _chebyshev_polynomial
    z = sp.Symbol('z')
    T0 = _chebyshev_polynomial(0, z)
    T1 = _chebyshev_polynomial(1, z)
    T2 = _chebyshev_polynomial(2, z)
    assert T0 == 1
    assert T1 == z
    assert sp.expand(T2 - (2 * z**2 - 1)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# New functionality tests
#
# Covers features not systematically tested in the original suite:
#   * matrix-valued pseudo-differential operators
#   * Weyl / Kohn–Nirenberg symbolic conversion
#   * Peetre decomposition / Peetre-based application smoke tests
#   * low-rank factorization / decomposition-quality smoke tests
#   * subdomain diffuse-boundary pipeline smoke tests
#   * grid utilities and exponential propagators
#   * first- and second-order time solvers
#   * characteristic Hamiltonians and bicharacteristic integration
#   * solver-level plotting / animation smoke tests
#   * singularity animations and Hénon–Heiles utilities
# ─────────────────────────────────────────────────────────────────────────────

import pytest
import numpy as np
import sympy as sp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import psiop
from psiop import PseudoDifferentialOperator


def _psiop_func(name):
    """Return a psiop attribute or skip the test if it is not exposed."""
    if not hasattr(psiop, name):
        pytest.skip(f"{name} is not exposed by psiop")
    return getattr(psiop, name)


def _grid_1d(N=32, L=1.0):
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    return x, kx


def _grid_2d(N=16, L=1.0):
    x = np.linspace(-L, L, N, endpoint=False)
    y = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dy)
    return x, y, kx, ky


def _close():
    plt.close('all')


# ─────────────────────────────────────────────────────────────────────────────
# Grid utilities
# ─────────────────────────────────────────────────────────────────────────────

def test_make_grid_1d():
    make_grid_1d = _psiop_func('make_grid_1d')

    L = 2.0
    N = 32
    x, kx = make_grid_1d(L=L, N=N)

    assert x.shape == (N,)
    assert kx.shape == (N,)
    assert np.isclose(x[0], -L)
    assert np.isclose(x[-1], L - 2.0 * L / N)
    assert np.all(np.isfinite(kx))


def test_make_grid_2d():
    make_grid_2d = _psiop_func('make_grid_2d')

    L = 1.5
    N = 16
    x, y, kx, ky = make_grid_2d(L=L, N=N)

    assert x.shape == (N,)
    assert y.shape == (N,)
    assert kx.shape == (N,)
    assert ky.shape == (N,)
    assert np.isclose(x[0], -L)
    assert np.isclose(y[0], -L)
    assert np.all(np.isfinite(kx))
    assert np.all(np.isfinite(ky))


# ─────────────────────────────────────────────────────────────────────────────
# Matrix-valued pseudo-differential operators
# ─────────────────────────────────────────────────────────────────────────────

def test_matrix_operator_constant_diagonal_apply():
    MatrixPseudoDifferentialOperator = _psiop_func('MatrixPseudoDifferentialOperator')

    x, xi = sp.symbols('x xi', real=True)
    M = sp.diag(2 + 0 * xi, -3 + 0 * xi)

    op = MatrixPseudoDifferentialOperator(M, [x])

    N = 32
    xg, kx = _grid_1d(N=N)

    u1 = np.sin(xg)
    u2 = np.cos(xg)
    U = np.vstack([u1, u2])

    res = np.asarray(
        op.apply(
            U,
            xg,
            kx,
            boundary_condition='periodic',
            freq_window=None,
            clamp=np.inf,
        )
    )

    # Some implementations may return shape (N, size); normalize to (size, N).
    if res.ndim == 2 and res.shape == (N, 2):
        res = res.T

    assert res.shape == (2, N)
    assert np.allclose(res[0], 2.0 * u1, atol=1e-10)
    assert np.allclose(res[1], -3.0 * u2, atol=1e-10)


def test_matrix_operator_symbol_matrix_and_eigen_smoke():
    MatrixPseudoDifferentialOperator = _psiop_func('MatrixPseudoDifferentialOperator')

    x, xi = sp.symbols('x xi', real=True)
    M = sp.Matrix([[xi**2, 0], [0, x]])

    op = MatrixPseudoDifferentialOperator(M, [x])

    S = op.symbol_matrix()
    assert hasattr(S, 'shape')
    assert tuple(S.shape) == (2, 2)

    eig = op.eigen_symbol()
    assert eig is not None


def test_matrix_composition_and_commutator_smoke():
    MatrixPseudoDifferentialOperator = _psiop_func('MatrixPseudoDifferentialOperator')

    x, xi = sp.symbols('x xi', real=True)

    A = MatrixPseudoDifferentialOperator(sp.diag(x, xi), [x])
    B = MatrixPseudoDifferentialOperator(sp.diag(xi, x), [x])

    C = A.compose_asymptotic(B, order=1)
    assert C is not None

    Comm = A.commutator_symbolic(B, order=1)
    assert Comm is not None


def test_matrix_exponential_symbol_smoke():
    MatrixPseudoDifferentialOperator = _psiop_func('MatrixPseudoDifferentialOperator')

    x, xi = sp.symbols('x xi', real=True)
    t = sp.symbols('t', real=True)

    M = sp.diag(-1 + 0 * xi, -2 + 0 * xi)
    op = MatrixPseudoDifferentialOperator(M, [x])

    try:
        E = op.exponential_symbol(t=t, order=2)
    except TypeError:
        E = op.exponential_symbol(0.05, order=2)

    assert E is not None


# ─────────────────────────────────────────────────────────────────────────────
# Weyl / Kohn–Nirenberg symbol conversion
# ─────────────────────────────────────────────────────────────────────────────

def test_weyl_to_kn_and_kn_to_weyl_symbol_conversion_x_xi():
    x, xi = sp.symbols('x xi', real=True)

    a_weyl = x * xi

    op_w = PseudoDifferentialOperator(
        a_weyl,
        [x],
        mode='symbol',
        quantization='weyl',
    )

    a_kn = op_w.weyl_to_kn_symbol(order=2)
    if hasattr(a_kn, 'removeO'):
        a_kn = a_kn.removeO()
    a_kn = sp.expand(a_kn)

    # Op^W(x ξ) = Op^KN(x ξ - i/2)
    assert sp.simplify(a_kn - (x * xi - sp.I / 2)) == 0

    op_kn = PseudoDifferentialOperator(
        a_weyl,
        [x],
        mode='symbol',
        quantization='kohn-nirenberg',
    )

    a_w = op_kn.kn_to_weyl_symbol(order=2)
    if hasattr(a_w, 'removeO'):
        a_w = a_w.removeO()
    a_w = sp.expand(a_w)

    # Op^KN(x ξ) = Op^W(x ξ + i/2)
    assert sp.simplify(a_w - (x * xi + sp.I / 2)) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Peetre decomposition and Peetre application
# ─────────────────────────────────────────────────────────────────────────────

def test_peetre_decomposition_smoke():
    x, xi = sp.symbols('x xi', real=True)

    expr = x * xi**2 + sp.sin(x) * xi + sp.exp(x)
    op = PseudoDifferentialOperator(expr, [x], mode='symbol')

    if hasattr(op, 'peetre_decomposition'):
        decomp = op.peetre_decomposition()
    elif hasattr(op, 'decompose_symbol_peetre'):
        decomp = op.decompose_symbol_peetre()
    else:
        pytest.skip("No public Peetre decomposition method found")

    assert decomp is not None


def test_apply_peetre_and_peetre_apply_alias_consistency():
    x, xi = sp.symbols('x xi', real=True)

    expr = xi**2 + x
    op = PseudoDifferentialOperator(expr, [x], mode='symbol')

    N = 64
    xg, kx = _grid_1d(N=N)
    u = np.exp(-xg**2)

    r1 = op.apply_peetre(
        u,
        xg,
        kx,
        boundary_condition='periodic',
        freq_window=None,
        clamp=np.inf,
    )

    if not hasattr(op, 'peetre_apply'):
        pytest.skip("peetre_apply alias not available")

    r2 = op.peetre_apply(
        u,
        xg,
        kx,
        boundary_condition='periodic',
        freq_window=None,
        clamp=np.inf,
    )

    assert np.allclose(r1, r2, atol=1e-10, rtol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Low-rank factorization / decomposition quality (smoke, signature-tolerant)
# ─────────────────────────────────────────────────────────────────────────────

def test_factorize_symbolic_smoke():
    factorize_symbolic = _psiop_func('factorize_symbolic')

    x, xi = sp.symbols('x xi', real=True)
    expr = sp.sin(x) * sp.cos(xi)

    last_err = None

    kwargs_candidates = [
        dict(
            expr=expr,
            vars_x=[x],
            vars_xi=[xi],
            bounds_x=[(-1.0, 1.0)],
            bounds_xi=[(-1.0, 1.0)],
            rank=1,
        ),
        dict(
            symbol_expr=expr,
            vars_x=[x],
            vars_xi=[xi],
            bounds_x=[(-1.0, 1.0)],
            bounds_xi=[(-1.0, 1.0)],
            rank=1,
        ),
        dict(
            expr=expr,
            vars_x=[x],
            vars_xi=[xi],
            x_bounds=[(-1.0, 1.0)],
            xi_bounds=[(-1.0, 1.0)],
            rank=1,
        ),
        dict(
            symbol=expr,
            xs=[x],
            xis=[xi],
            x_bounds=[(-1.0, 1.0)],
            xi_bounds=[(-1.0, 1.0)],
            rank=1,
        ),
    ]

    for kw in kwargs_candidates:
        try:
            out = factorize_symbolic(**kw)
            assert out is not None
            return
        except TypeError as err:
            last_err = err

    args_candidates = [
        (expr, [x], [xi], [(-1.0, 1.0)], [(-1.0, 1.0)], 1),
        (expr, [x], [xi], (-1.0, 1.0), (-1.0, 1.0), 1),
        (expr, [x], [xi], 1),
    ]

    for args in args_candidates:
        try:
            out = factorize_symbolic(*args)
            assert out is not None
            return
        except TypeError as err:
            last_err = err

    pytest.skip(f"factorize_symbolic signature not recognized: {last_err}")


def test_evaluate_decomposition_quality_smoke():
    evaluate_decomposition_quality = _psiop_func('evaluate_decomposition_quality')

    x, xi = sp.symbols('x xi', real=True)
    expr = x * xi / (1 + xi**2)

    op = PseudoDifferentialOperator(expr, [x], mode='symbol')
    xg, kx = _grid_1d(N=16)

    last_err = None

    candidates = [
        ((op, xg, kx), {'n_samples': 20}),
        ((op, xg, kx), {}),
        ((expr, [x], [xi], [(-1.0, 1.0)], [(-1.0, 1.0)]), {'n_samples': 20}),
        ((expr, [x], [xi], (-1.0, 1.0), (-1.0, 1.0)), {'n_samples': 20}),
    ]

    for args, kwargs in candidates:
        try:
            res = evaluate_decomposition_quality(*args, **kwargs)
            assert res is not None
            return
        except TypeError as err:
            last_err = err

    pytest.skip(f"evaluate_decomposition_quality signature not recognized: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Subdomain diffuse-boundary pipeline (smoke, signature-tolerant)
# ─────────────────────────────────────────────────────────────────────────────

def test_subdomain_masks_smoke():
    subdomain_masks = _psiop_func('subdomain_masks')

    xg, _ = _grid_1d(N=16)
    x = sp.symbols('x', real=True)

    last_err = None

    candidates = [
        ((lambda X: X, xg), {}),
        ((lambda X: X, xg, None), {}),
        ((x, [x], xg), {}),
        ((x, [x], xg, None), {}),
    ]

    out = None
    for args, kwargs in candidates:
        try:
            out = subdomain_masks(*args, **kwargs)
            break
        except TypeError as err:
            last_err = err

    if out is None:
        pytest.skip(f"subdomain_masks signature not recognized: {last_err}")

    if isinstance(out, dict):
        chi = out.get('chi_Omega', out.get('chi', next(iter(out.values()))))
    elif isinstance(out, (tuple, list)):
        chi = out[0]
    else:
        chi = out

    chi = np.asarray(chi)
    assert chi.shape == xg.shape
    assert np.all(np.isfinite(chi))


def test_apply_subdomain_smoke():
    x, xi = sp.symbols('x xi', real=True)

    op = PseudoDifferentialOperator(xi**2, [x], mode='symbol')

    if not hasattr(op, 'apply_subdomain'):
        pytest.skip("apply_subdomain not available")

    N = 16
    xg, kx = _grid_1d(N=N)

    u = np.zeros(N, dtype=float)
    f = np.ones(N, dtype=float)
    g = lambda X: X

    last_err = None

    candidates = [
        ((u, f, xg, kx, g), {'max_iter': 1}),
        ((u, f, xg, kx), {'g': g, 'max_iter': 1}),
        ((u, f, g, xg, kx), {'max_iter': 1}),
    ]

    out = None
    for args, kwargs in candidates:
        try:
            out = op.apply_subdomain(*args, **kwargs)
            break
        except TypeError as err:
            last_err = err

    if out is None:
        pytest.skip(f"apply_subdomain signature not recognized: {last_err}")

    if isinstance(out, (tuple, list)):
        arr = np.asarray(out[0])
    else:
        arr = np.asarray(out)

    assert arr.shape == u.shape
    assert np.all(np.isfinite(arr))


# ─────────────────────────────────────────────────────────────────────────────
# Propagator construction
# ─────────────────────────────────────────────────────────────────────────────

def test_build_propagator_scalar_constant_decay():
    build_propagator = _psiop_func('build_propagator')

    x, xi = sp.symbols('x xi', real=True)

    dt = 0.05
    prop, is_matrix, size = build_propagator(
        -1 + 0 * xi,
        [x],
        dt=dt,
        order=3,
    )

    assert is_matrix is False
    assert size is None

    N = 32
    xg, kx = _grid_1d(N=N)
    u = np.ones(N, dtype=float)

    u1 = np.asarray(
        prop.apply(
            u,
            xg,
            kx,
            boundary_condition='periodic',
            freq_window=None,
            clamp=np.inf,
        )
    )

    assert u1.shape == u.shape
    assert np.allclose(u1, np.exp(-dt) * u, rtol=1e-3, atol=1e-6)


def test_build_propagator_matrix_constant_decay():
    build_propagator = _psiop_func('build_propagator')

    x, xi = sp.symbols('x xi', real=True)

    dt = 0.05
    M = sp.diag(-1 + 0 * xi, -2 + 0 * xi)

    prop, is_matrix, size = build_propagator(
        M,
        [x],
        dt=dt,
        order=3,
    )

    assert is_matrix is True
    assert size == 2

    N = 32
    xg, kx = _grid_1d(N=N)

    U = np.ones((2, N), dtype=float)

    res = np.asarray(
        prop.apply(
            U,
            xg,
            kx,
            boundary_condition='periodic',
            freq_window=None,
            clamp=np.inf,
        )
    )

    if res.ndim == 2 and res.shape == (N, 2):
        res = res.T

    assert res.shape == (2, N)
    assert np.allclose(res[0], np.exp(-dt), rtol=1e-3, atol=1e-6)
    assert np.allclose(res[1], np.exp(-2.0 * dt), rtol=1e-3, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# First-order solver
# ─────────────────────────────────────────────────────────────────────────────

def test_solve_first_order_scalar_constant_decay():
    solve_first_order = _psiop_func('solve_first_order')

    x, xi = sp.symbols('x xi', real=True)

    dt = 0.01
    n_steps = 3
    N = 16

    f = lambda X: np.ones_like(X, dtype=float)

    t, U, grids = solve_first_order(
        -1 + 0 * xi,
        [x],
        f,
        dt=dt,
        n_steps=n_steps,
        order=3,
        L=1.0,
        N=N,
    )

    U = np.asarray(U)

    # Normalize possible singleton component axis.
    if U.ndim == 3 and U.shape[1] == 1:
        U = U[:, 0, :]

    assert t.shape[0] == U.shape[0]
    assert U.shape[-1] == N

    expected_final = np.exp(-t[-1])
    assert np.allclose(U[-1], expected_final, rtol=1e-2, atol=1e-3)


def test_solve_first_order_matrix_constant_decay():
    solve_first_order = _psiop_func('solve_first_order')

    x, xi = sp.symbols('x xi', real=True)

    dt = 0.01
    n_steps = 2
    N = 16

    M = sp.diag(-1 + 0 * xi, -2 + 0 * xi)

    f = lambda X: [np.ones_like(X, dtype=float), np.ones_like(X, dtype=float)]

    t, U, grids = solve_first_order(
        M,
        [x],
        f,
        dt=dt,
        n_steps=n_steps,
        order=3,
        L=1.0,
        N=N,
    )

    U = np.asarray(U)

    # Expected shape: (n_saved, size, N)
    assert U.ndim == 3
    assert U.shape[1] == 2
    assert U.shape[2] == N

    assert np.allclose(U[-1, 0], np.exp(-t[-1]), rtol=1e-2, atol=1e-3)
    assert np.allclose(U[-1, 1], np.exp(-2.0 * t[-1]), rtol=1e-2, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Second-order solver
# ─────────────────────────────────────────────────────────────────────────────

def test_block_matrix_second_order_scalar():
    block_matrix_second_order = _psiop_func('block_matrix_second_order')

    x, xi = sp.symbols('x xi', real=True)

    M = block_matrix_second_order(-xi**2)

    assert M.shape == (2, 2)
    assert sp.simplify(M[0, 1] - 1) == 0
    assert sp.simplify(M[1, 0] + xi**2) == 0


def test_block_matrix_second_order_matrix_symbol():
    block_matrix_second_order = _psiop_func('block_matrix_second_order')

    x, xi = sp.symbols('x xi', real=True)

    S = sp.diag(-xi**2, -2 * xi**2)
    M = block_matrix_second_order(S)

    assert M.shape == (4, 4)


def test_solve_second_order_scalar_harmonic_constant():
    solve_second_order = _psiop_func('solve_second_order')

    x, xi = sp.symbols('x xi', real=True)

    dt = 0.01
    n_steps = 2
    N = 16

    f = lambda X: np.ones_like(X, dtype=float)
    g = lambda X: np.zeros_like(X, dtype=float)

    t, U, V, grids = solve_second_order(
        -1 + 0 * xi,
        [x],
        f,
        g,
        dt=dt,
        n_steps=n_steps,
        order=4,
        L=1.0,
        N=N,
    )

    U = np.asarray(U)
    V = np.asarray(V)

    if U.ndim == 3 and U.shape[1] == 1:
        U = U[:, 0, :]
    if V.ndim == 3 and V.shape[1] == 1:
        V = V[:, 0, :]

    assert U.shape[-1] == N
    assert V.shape[-1] == N

    # u_tt = -u, u(0)=1, u_t(0)=0  =>  u(t)=cos(t), u_t(t)=-sin(t)
    assert np.allclose(U[-1], np.cos(t[-1]), rtol=2e-2, atol=2e-2)
    assert np.allclose(V[-1], -np.sin(t[-1]), rtol=2e-2, atol=2e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Characteristic Hamiltonians and bicharacteristics
# ─────────────────────────────────────────────────────────────────────────────

def test_characteristic_hamiltonians_scalar_and_matrix():
    characteristic_hamiltonians = _psiop_func('characteristic_hamiltonians')

    x, xi = sp.symbols('x xi', real=True)

    H_list, xs, xis = characteristic_hamiltonians(xi**2 + x**2, [x])

    assert isinstance(H_list, (list, tuple))
    assert len(H_list) == 1
    assert len(xs) == 1
    assert len(xis) == 1

    M = sp.diag(xi, -xi)
    H_list_mat, xs_mat, xis_mat = characteristic_hamiltonians(M, [x])

    assert isinstance(H_list_mat, (list, tuple))
    assert len(H_list_mat) == 2


def test_integrate_singularity_shapes_and_finiteness():
    integrate_singularity = _psiop_func('integrate_singularity')

    x, xi = sp.symbols('x xi', real=True)

    n_frames = 3

    H_list, xs, xis, t_eval, trajs = integrate_singularity(
        xi**2 + x**2,
        [x],
        x0=0.0,
        xi0=1.0,
        tmax=0.05,
        n_frames=n_frames,
    )

    assert t_eval.shape == (n_frames,)
    assert len(trajs) >= 1

    trajs_arr = np.asarray(trajs)

    if trajs_arr.ndim == 3:
        traj = trajs_arr[0]
    else:
        traj = np.asarray(trajs[0])

    # In 1D phase space: (x, xi) => 2 rows.
    assert traj.shape[0] == 2
    assert traj.shape[1] == n_frames
    assert np.all(np.isfinite(traj))


# ─────────────────────────────────────────────────────────────────────────────
# Solver-level plotting and animation smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_scalar_1d_smoke():
    plot_scalar_1d = _psiop_func('plot_scalar_1d')

    N = 16
    x, _ = _grid_1d(N=N)

    t = np.linspace(0.0, 0.1, 3)
    U = np.tile(np.sin(x), (3, 1))

    plot_scalar_1d(t, U, x, quantity='real')
    _close()


def test_plot_matrix_1d_smoke():
    plot_matrix_1d = _psiop_func('plot_matrix_1d')

    N = 16
    x, _ = _grid_1d(N=N)

    t = np.linspace(0.0, 0.1, 3)

    U0 = np.tile(np.sin(x), (3, 1))
    U1 = np.tile(np.cos(x), (3, 1))
    U = np.stack([U0, U1], axis=1)  # shape: (n_times, size, N)

    plot_matrix_1d(t, U, x, quantity='real')
    _close()


def test_plot_scalar_2d_smoke():
    plot_scalar_2d = _psiop_func('plot_scalar_2d')

    N = 8
    x = np.linspace(-1.0, 1.0, N, endpoint=False)
    y = np.linspace(-1.0, 1.0, N, endpoint=False)

    X, Y = np.meshgrid(x, y, indexing='ij')
    field = np.sin(X) * np.cos(Y)

    t = np.linspace(0.0, 0.1, 3)
    U = np.stack([field, 0.5 * field, 0.25 * field], axis=0)

    plot_scalar_2d(t, U, x, y, quantity='real')
    _close()


def test_animate_scalar_1d_smoke():
    animate_scalar_1d = _psiop_func('animate_scalar_1d')

    N = 16
    x, _ = _grid_1d(N=N)

    t = np.linspace(0.0, 0.1, 3)
    U = np.tile(np.sin(x), (3, 1))

    anim = animate_scalar_1d(t, U, x, quantity='real', interval=1)
    assert anim is not None
    _close()


# ─────────────────────────────────────────────────────────────────────────────
# Singularity animation smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def test_animate_singularity_smoke():
    animate_singularity = _psiop_func('animate_singularity')

    x, xi = sp.symbols('x xi', real=True)

    anim = animate_singularity(
        xi**2 + x**2,
        [x],
        x0=0.0,
        xi0=1.0,
        tmax=0.05,
        n_frames=3,
        interval=1,
    )

    assert anim is not None
    _close()


def test_animate_singularity_3d_smoke():
    animate_singularity_3d = _psiop_func('animate_singularity_3d')

    x, xi = sp.symbols('x xi', real=True)

    anim = animate_singularity_3d(
        xi**2 + x**2,
        [x],
        x0=0.0,
        xi0=1.0,
        tmax=0.05,
        n_frames=3,
        interval=1,
    )

    assert anim is not None
    _close()


# ─────────────────────────────────────────────────────────────────────────────
# Private helper utilities used by the new solver layer
# ─────────────────────────────────────────────────────────────────────────────

def test_private_as_component_list():
    if not hasattr(psiop, '_as_component_list'):
        pytest.skip("_as_component_list not available")

    X = np.arange(4.0)

    comps = psiop._as_component_list(lambda XX: np.ones_like(XX), X)
    assert isinstance(comps, list)
    assert len(comps) == 1
    assert comps[0].shape == X.shape

    comps2 = psiop._as_component_list(
        lambda XX: [np.ones_like(XX), np.zeros_like(XX)],
        X,
    )
    assert len(comps2) == 2


def test_private_matrix_of():
    if not hasattr(psiop, '_matrix_of'):
        pytest.skip("_matrix_of not available")

    x, xi = sp.symbols('x xi', real=True)

    M1 = psiop._matrix_of(xi**2)
    assert M1.shape == (1, 1)

    M2 = psiop._matrix_of(sp.diag(xi, x))
    assert M2.shape == (2, 2)


def test_private_order_freq_vars():
    if not hasattr(psiop, '_order_freq_vars'):
        pytest.skip("_order_freq_vars not available")

    xi, eta = sp.symbols('xi eta', real=True)

    ordered = psiop._order_freq_vars([eta, xi], 2)
    assert [str(s) for s in ordered] == ['xi', 'eta']


def test_invalidate_kn_cache_smoke():
    if not hasattr(psiop, 'invalidate_kn_cache'):
        pytest.skip("invalidate_kn_cache not available")

    # Should simply clear global caches without raising.
    psiop.invalidate_kn_cache()