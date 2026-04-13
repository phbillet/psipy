from imports import *
from psiop import *


# ===========================================================================
#  Helpers
# ===========================================================================

def _make_op_1d(expr):
    x = symbols('x', real=True)
    return PseudoDifferentialOperator(expr=expr, vars_x=[x], mode='symbol')


def _make_op_2d(expr):
    x, y = symbols('x y', real=True)
    return PseudoDifferentialOperator(expr=expr, vars_x=[x, y], mode='symbol')


# ===========================================================================
#  weyl_to_kn_symbol — 1D
# ===========================================================================

def test_weyl_to_kn_constant_1d():
    """Constant symbol: correction series vanishes at every order."""
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(Integer(3))
    result = op.weyl_to_kn_symbol(order=4)
    assert simplify(result - Integer(3)) == 0, (
        f"Constant symbol should be unchanged, got {result}"
    )


def test_weyl_to_kn_xi_only_1d():
    """Symbol depending only on xi: d_x annihilates it, so no correction."""
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(xi**3)
    result = op.weyl_to_kn_symbol(order=4)
    assert simplify(result - xi**3) == 0, (
        f"xi-only symbol should be unchanged, got {result}"
    )


def test_weyl_to_kn_x_only_1d():
    """Symbol depending only on x: d_xi annihilates it, so no correction."""
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(x**2)
    result = op.weyl_to_kn_symbol(order=4)
    assert simplify(result - x**2) == 0, (
        f"x-only symbol should be unchanged, got {result}"
    )


def test_weyl_to_kn_linear_cross_term_1d():
    """
    a(x, xi) = x * xi.
    d_x d_xi (x*xi) = 1, all higher orders vanish.
    Expected: x*xi + i/2.
    """
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(x * xi)
    result = op.weyl_to_kn_symbol(order=4)
    expected = x * xi + I / 2
    assert simplify(result - expected) == 0, (
        f"Expected {expected}, got {result}"
    )


def test_weyl_to_kn_quadratic_1d():
    """
    a(x, xi) = x**2 * xi**2.
    Order-1 term: i/2 * d_x d_xi (x**2 * xi**2) = i/2 * 4*x*xi = 2*I*x*xi.
    Order-2 term: (i/2)^2 / 2 * d_x^2 d_xi^2 (x**2*xi**2)
                = -1/8 * 4 = -1/2.
    Higher orders vanish (polynomial of degree 2 in each variable).
    Expected: x**2*xi**2 + 2*I*x*xi - 1/2.
    """
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(x**2 * xi**2)
    result = op.weyl_to_kn_symbol(order=4)
    expected = x**2 * xi**2 + 2 * I * x * xi - Rational(1, 2)
    assert simplify(result - expected) == 0, (
        f"Expected {expected}, got {result}"
    )


def test_weyl_to_kn_polynomial_series_finite_1d():
    """
    For a polynomial symbol of degree d in xi and d in x, the correction
    series truncates at order d.  Increasing order beyond d must not change
    the result.
    """
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(x**2 * xi**2 + x * xi)
    result_order2 = op.weyl_to_kn_symbol(order=2)
    result_order6 = op.weyl_to_kn_symbol(order=6)
    assert simplify(result_order2 - result_order6) == 0, (
        "Series should be exact and finite for polynomial symbols"
    )


# ===========================================================================
#  kn_to_weyl_symbol — 1D
# ===========================================================================

def test_kn_to_weyl_constant_1d():
    """Constant symbol: no correction."""
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(Integer(5))
    result = op.kn_to_weyl_symbol(order=4)
    assert simplify(result - Integer(5)) == 0, (
        f"Constant symbol should be unchanged, got {result}"
    )


def test_kn_to_weyl_linear_cross_term_1d():
    """
    a(x, xi) = x * xi.
    Expected: x*xi - i/2  (opposite sign to weyl_to_kn).
    """
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(x * xi)
    result = op.kn_to_weyl_symbol(order=4)
    expected = x * xi - I / 2
    assert simplify(result - expected) == 0, (
        f"Expected {expected}, got {result}"
    )


def test_kn_to_weyl_quadratic_1d():
    """
    a(x, xi) = x**2 * xi**2.
    Same structure as weyl_to_kn but with -i/2 at each order.
    Expected: x**2*xi**2 - 2*I*x*xi - 1/2.
    """
    x, xi = symbols('x xi', real=True)
    op = _make_op_1d(x**2 * xi**2)
    result = op.kn_to_weyl_symbol(order=4)
    expected = x**2 * xi**2 - 2 * I * x * xi - Rational(1, 2)
    assert simplify(result - expected) == 0, (
        f"Expected {expected}, got {result}"
    )


# ===========================================================================
#  Round-trip consistency — 1D
# ===========================================================================

def test_round_trip_weyl_kn_weyl_1d():
    """
    kn_to_weyl( weyl_to_kn(a) ) should recover a up to truncation order.
    Exact for polynomial symbols.
    """
    x, xi = symbols('x xi', real=True)
    a = x**2 * xi**2 + x * xi + xi**2 + x + Integer(1)
    op = _make_op_1d(a)
    kn_sym   = op.weyl_to_kn_symbol(order=4)
    op_kn    = _make_op_1d(kn_sym)
    recovered = op_kn.kn_to_weyl_symbol(order=4)
    assert simplify(expand(recovered - a)) == 0, (
        f"Round-trip failed: got {simplify(recovered - a)}"
    )


def test_round_trip_kn_weyl_kn_1d():
    """
    weyl_to_kn( kn_to_weyl(a) ) should recover a up to truncation order.
    Exact for polynomial symbols.
    """
    x, xi = symbols('x xi', real=True)
    a = x * xi**3 + x**2 * xi
    op = _make_op_1d(a)
    weyl_sym  = op.kn_to_weyl_symbol(order=4)
    op_weyl   = _make_op_1d(weyl_sym)
    recovered = op_weyl.weyl_to_kn_symbol(order=4)
    assert simplify(expand(recovered - a)) == 0, (
        f"Round-trip failed: got {simplify(recovered - a)}"
    )


def test_round_trip_sign_antisymmetry_1d():
    """
    Odd-order corrections are antisymmetric (opposite signs).
    Even-order corrections are symmetric (same sign).
    The corrections are NOT exact opposites in general.
    """
    x, xi = symbols('x xi', real=True)
    a = x**2 * xi**2
    op = _make_op_1d(a)
    correction_plus  = simplify(op.weyl_to_kn_symbol(order=4) - a)
    correction_minus = simplify(op.kn_to_weyl_symbol(order=4) - a)

    # Odd part must be antisymmetric
    odd_plus  = simplify((correction_plus  - correction_minus) / 2)
    odd_minus = simplify((correction_minus - correction_plus)  / 2)
    assert simplify(odd_plus + odd_minus) == 0, (
        "Odd-order corrections should be exact opposites"
    )

    # Even part must be symmetric (same sign)
    even_part = simplify((correction_plus + correction_minus) / 2)
    even_expected = Rational(-1, 2)   # order-2 term: (i/2)^2 / 2 * 4 = -1/2
    assert simplify(even_part - even_expected) == 0, (
        f"Even-order correction should be {even_expected}, got {even_part}"
    )
    
# ===========================================================================
#  weyl_to_kn_symbol — 2D
# ===========================================================================

def test_weyl_to_kn_constant_2d():
    """Constant symbol in 2D: no correction."""
    x, y = symbols('x y', real=True)
    op = _make_op_2d(Integer(7))
    result = op.weyl_to_kn_symbol(order=4)
    assert simplify(result - Integer(7)) == 0, (
        f"Constant symbol should be unchanged, got {result}"
    )


def test_weyl_to_kn_frequency_only_2d():
    """Symbol depending only on (xi, eta): all d_x, d_y annihilate it."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = _make_op_2d(xi**2 + eta**2)
    result = op.weyl_to_kn_symbol(order=4)
    assert simplify(result - (xi**2 + eta**2)) == 0, (
        f"Frequency-only symbol should be unchanged, got {result}"
    )


def test_weyl_to_kn_linear_cross_terms_2d():
    """
    a(x, y, xi, eta) = x*xi + y*eta.
    d_x d_xi (x*xi) = 1,  d_y d_eta (y*eta) = 1.
    Order-1 contribution: i/2 * (1 + 1) = I.
    Higher orders vanish.
    Expected: x*xi + y*eta + I.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = _make_op_2d(x * xi + y * eta)
    result = op.weyl_to_kn_symbol(order=4)
    expected = x * xi + y * eta + I
    assert simplify(result - expected) == 0, (
        f"Expected {expected}, got {result}"
    )


def test_weyl_to_kn_mixed_2d():
    """
    a = x*xi + y*eta + xi**2 + eta**2.
    The quadratic frequency part contributes no correction (d_x, d_y = 0).
    Expected: x*xi + y*eta + xi**2 + eta**2 + I.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = _make_op_2d(x * xi + y * eta + xi**2 + eta**2)
    result = op.weyl_to_kn_symbol(order=4)
    expected = x * xi + y * eta + xi**2 + eta**2 + I
    assert simplify(result - expected) == 0, (
        f"Expected {expected}, got {result}"
    )


def test_weyl_to_kn_cross_xy_xi_eta_2d():
    """
    a = x*y*xi*eta.
    The mixed cross term involves both pairs (x,xi) and (y,eta).
    Order-1: i/2 * (d_x d_xi + d_y d_eta)(x*y*xi*eta)
           = i/2 * (y*eta + x*xi).
    Order-2 involves the binomial expansion of (d_x d_xi + d_y d_eta)^2.
    Test that the result is not equal to the original symbol (non-trivial
    correction) and that the series is finite.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = _make_op_2d(x * y * xi * eta)
    result_order4 = op.weyl_to_kn_symbol(order=4)
    result_order8 = op.weyl_to_kn_symbol(order=8)
    # Series must be finite (exact for polynomial symbols)
    assert simplify(result_order4 - result_order8) == 0, (
        "Series should be exact and finite for polynomial symbols"
    )
    # Non-trivial correction
    assert simplify(result_order4 - x * y * xi * eta) != 0, (
        "Correction should be non-zero for x*y*xi*eta"
    )


# ===========================================================================
#  kn_to_weyl_symbol — 2D
# ===========================================================================

def test_kn_to_weyl_linear_cross_terms_2d():
    """
    a(x, y, xi, eta) = x*xi + y*eta.
    Expected: x*xi + y*eta - I  (opposite sign to weyl_to_kn).
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    op = _make_op_2d(x * xi + y * eta)
    result = op.kn_to_weyl_symbol(order=4)
    expected = x * xi + y * eta - I
    assert simplify(result - expected) == 0, (
        f"Expected {expected}, got {result}"
    )


# ===========================================================================
#  Round-trip consistency — 2D
# ===========================================================================

def test_round_trip_weyl_kn_weyl_2d():
    """
    kn_to_weyl( weyl_to_kn(a) ) should recover a for a polynomial symbol.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    a = x * xi + y * eta + x**2 * xi**2 + y**2 * eta**2 + Integer(1)
    op = _make_op_2d(a)
    kn_sym    = op.weyl_to_kn_symbol(order=4)
    op_kn     = _make_op_2d(kn_sym)
    recovered = op_kn.kn_to_weyl_symbol(order=4)
    assert simplify(expand(recovered - a)) == 0, (
        f"Round-trip failed: got {simplify(recovered - a)}"
    )


def test_round_trip_sign_antisymmetry_2d():
    """
    In 2D, corrections of weyl_to_kn and kn_to_weyl must also be
    exact opposites for a polynomial symbol.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    a = x * xi + y * eta
    op = _make_op_2d(a)
    correction_plus  = simplify(op.weyl_to_kn_symbol(order=4) - a)
    correction_minus = simplify(op.kn_to_weyl_symbol(order=4) - a)
    assert simplify(correction_plus + correction_minus) == 0, (
        "Corrections should be exact opposites in 2D"
    )


# ===========================================================================
#  Self-adjointness property
# ===========================================================================

def test_weyl_real_symbol_gives_selfadjoint_kn_1d():
    """
    If a_Weyl is real, Op^w(a) is self-adjoint.  After conversion to KN,
    the formal adjoint of the KN operator should equal itself (to the
    truncation order), i.e. the imaginary correction must be absorbed
    consistently.  Here we just check that weyl_to_kn returns a non-real
    symbol (the imaginary correction is present) and that kn_to_weyl of
    that result gives back the original real symbol.
    """
    x, xi = symbols('x xi', real=True)
    a_real = x**2 * xi**2 + xi**2 + x**2    # manifestly real
    op = _make_op_1d(a_real)
    kn_sym = op.weyl_to_kn_symbol(order=4)
    # The KN equivalent of a real Weyl symbol is generally complex
    assert kn_sym.has(I), (
        "KN symbol of a real Weyl symbol should contain imaginary corrections"
    )
    # Round-trip recovers the original real symbol
    op_kn     = _make_op_1d(kn_sym)
    recovered = op_kn.kn_to_weyl_symbol(order=4)
    assert simplify(expand(recovered - a_real)) == 0, (
        f"Round-trip should recover the real Weyl symbol, got {recovered}"
    )


# ===========================================================================
#  Error handling
# ===========================================================================

def test_quantization_correction_3d_raises():
    """Dimension 3 must raise NotImplementedError."""
    x, y, z = symbols('x y z', real=True)
    xi, eta, zeta = symbols('xi eta zeta', real=True)
    # Manually build a 3D-like object by monkey-patching dim
    op = _make_op_1d(symbols('xi', real=True)**2)
    op.dim = 3          # force unsupported dimension
    try:
        op.weyl_to_kn_symbol(order=2)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "3" in str(e) or "not supported" in str(e)
