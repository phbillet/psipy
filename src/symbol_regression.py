"""
symbol_regression.py
====================
Converts a numerical symbol estimate a(x_i, xi_j) on a phase-space grid
into a SymPy expression, and wraps the result in a PseudoDifferentialOperator.

Two regression strategies are provided:

SVD + polynomial fit  ('svd')
    Tests whether the symbol is rank-1 (separable: a(x,xi) = f(x)*g(xi)).
    If so, fits f and g independently as polynomials using BIC model
    selection.  Fast, robust, no external dependencies.  Exact for
    symbols of the form  p(x) * q(xi)  with polynomial p, q.

PySR symbolic regression  ('pysr')
    Fits a general symbolic expression to the flattened phase-space data
    using an evolutionary search over mathematical expressions.  Returns
    a SymPy expression directly.  Requires:  pip install pysr

Method 'auto'
    Tests separability first.  Uses SVD if the symbol is rank-1
    (SVD ratio S[1]/S[0] < svd_tol), otherwise falls back to PySR.
    If PySR is not installed, always uses SVD with a warning.

Main entry points
-----------------
symbol_grid_to_sympy   numeric grid  ->  SymPy expression
identify_operator      data pairs    ->  PseudoDifferentialOperator
validate_identification compare identified symbol to ground truth

Dependencies: numpy, sympy  (pysr optional)
"""

import warnings
import numpy as np
from sympy import (
    symbols, simplify, expand, Rational, I,
    lambdify, Symbol, nsimplify, sympify,
)


# ===========================================================================
#  Separability test
# ===========================================================================

def _test_separability(symbol_matrix, tol=5e-2):
    """
    Test whether the symbol is approximately rank-1.

    A rank-1 matrix corresponds to a separable symbol:
        a(x, xi) ~ f(x) * g(xi)

    Parameters
    ----------
    symbol_matrix : (Nx, Nxi) complex array
    tol           : float   threshold on S[1]/S[0]

    Returns
    -------
    is_separable : bool
    ratio        : float   S[1] / S[0]  (0 = perfectly separable)
    f_vec        : (Nx,)  or None   spatial factor  (unit L2 norm * S[0])
    g_vec        : (Nxi,) or None   frequency factor
    """
    U, S, Vt = np.linalg.svd(symbol_matrix, full_matrices=False)
    ratio    = float(S[1] / (S[0] + 1e-14))

    if ratio < tol:
        return True,  ratio, U[:, 0] * S[0], Vt[0, :]
    return     False, ratio, None,            None


# ===========================================================================
#  SVD + polynomial fallback  (no external dependencies)
# ===========================================================================

def _best_poly_fit(grid, values, max_deg=8):
    """
    Fit a real polynomial to (grid, values) with BIC degree selection.

    Parameters
    ----------
    grid    : (N,) real array
    values  : (N,) real or complex array  (imaginary part is ignored here;
              caller is responsible for separating real/imag before calling)
    max_deg : int

    Returns
    -------
    sympy_expr : SymPy polynomial in Symbol('t')
    best_deg   : int
    """
    t_sym    = Symbol('t')
    n        = len(grid)
    y        = np.real(values)

    best_bic  = np.inf
    best_expr = sympify(0)
    best_deg  = 0

    for deg in range(0, max_deg + 1):
        coeffs  = np.polyfit(grid, y, deg)
        residuals = y - np.polyval(coeffs, grid)
        rss     = float(np.sum(residuals ** 2))
        bic     = n * np.log(rss / n + 1e-14) + (deg + 1) * np.log(n)

        if bic < best_bic:
            best_bic  = bic
            best_deg  = deg
            # Build SymPy expression: sum c_k * t^k
            best_expr = sum(
                nsimplify(float(c), rational=False, tolerance=1e-3)
                * t_sym ** p
                for p, c in zip(range(deg, -1, -1), coeffs)
            )

    return simplify(best_expr), best_deg


def _svd_fallback(x_grid, xi_grid, symbol_matrix):
    """
    Recover a SymPy expression from a numerical symbol via SVD.

    Works best for separable symbols a(x, xi) = f(x) * g(xi).
    For non-separable symbols, uses the leading rank-1 approximation
    and emits a warning about the approximation quality.

    Returns
    -------
    sympy_expr : SymPy expression in symbols x, xi
    """
    x_sym  = Symbol('x')
    xi_sym = Symbol('xi')
    t_sym  = Symbol('t')

    is_sep, ratio, f_vec, g_vec = _test_separability(
        symbol_matrix, tol=1.0   # always extract rank-1 part
    )

    if not is_sep:
        warnings.warn(
            f"Symbol is not rank-1 (S[1]/S[0] = {ratio:.3f}). "
            "SVD fallback uses only the leading rank-1 approximation. "
            "Consider using method='pysr' for a more accurate fit.",
            UserWarning, stacklevel=3,
        )

    # Decide whether to fit real part, imag part, or both
    re_norm = np.max(np.abs(np.real(f_vec * g_vec[0])))
    # Fit real and imaginary parts of f separately; g is typically real
    f_re_expr, _ = _best_poly_fit(x_grid,  np.real(f_vec))
    f_im_expr, _ = _best_poly_fit(x_grid,  np.imag(f_vec))
    g_re_expr, _ = _best_poly_fit(xi_grid, np.real(g_vec))
    g_im_expr, _ = _best_poly_fit(xi_grid, np.imag(g_vec))

    # Rename dummy variable t -> x or xi
    f_expr = (f_re_expr + I * f_im_expr).subs(t_sym, x_sym)
    g_expr = (g_re_expr + I * g_im_expr).subs(t_sym, xi_sym)

    return simplify(expand(f_expr * g_expr))


# ===========================================================================
#  PySR wrapper
# ===========================================================================

def _fit_pysr(x_grid, xi_grid, symbol_matrix, pysr_options):
    """
    Fit a SymPy expression to a numerical symbol using PySR.

    Real and imaginary parts are fitted separately and then combined.
    Parts whose amplitude is negligible (< 1% of the dominant part) are
    skipped to avoid fitting noise.

    Parameters
    ----------
    x_grid        : (Nx,)
    xi_grid       : (Nxi,)
    symbol_matrix : (Nx, Nxi) complex
    pysr_options  : dict   overrides for PySRRegressor defaults

    Returns
    -------
    sympy_expr : SymPy expression in symbols (x, xi)
    models     : dict {'real': model, 'imag': model}  (only fitted models)
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError(
            "PySR is not installed.  Run:  pip install pysr\n"
            "Or use method='svd' for the polynomial fallback."
        )

    X_mg, XI_mg = np.meshgrid(x_grid, xi_grid, indexing='ij')
    features    = np.column_stack([X_mg.ravel(), XI_mg.ravel()])  # (N, 2)

    # Defaults – deterministic forces parallelism='serial'
    defaults = dict(
        binary_operators = ["+", "*", "^"],
        unary_operators  = ["sin", "cos", "exp"],
        constraints      = {'^': (1, 1)},  
        maxsize          = 15,
        niterations      = 50,
        verbosity        = 0,
        random_state     = 420,
        deterministic    = False,
#        parallelism      = "serial",
        procs            = 6,
    )
    defaults.update(pysr_options or {})

    re_norm = float(np.max(np.abs(symbol_matrix.real)))
    im_norm = float(np.max(np.abs(symbol_matrix.imag)))
    dominant = max(re_norm, im_norm)
    tol      = 1e-2 * dominant

    expr_re = sympify(0)
    expr_im = sympify(0)
    models  = {}

    # Symbols we want in the final expression
    x_sym  = Symbol('x')
    xi_sym = Symbol('xi')
    # PySR uses x0, x1, ... – we will replace them
    pysr_vars = [Symbol(f'x{i}') for i in range(2)]

    if re_norm > tol:
        m = PySRRegressor(**defaults)
        m.fit(features, symbol_matrix.real.ravel(), variable_names=["x", "xi"])
        expr_re_raw = m.sympy()
        # Rename variables: x0 -> x, x1 -> xi
        expr_re = expr_re_raw.subs({pysr_vars[0]: x_sym, pysr_vars[1]: xi_sym})
        models['real'] = m

    if im_norm > tol:
        m = PySRRegressor(**defaults)
        m.fit(features, symbol_matrix.imag.ravel(), variable_names=["x", "xi"])
        expr_im_raw = m.sympy()
        expr_im = expr_im_raw.subs({pysr_vars[0]: x_sym, pysr_vars[1]: xi_sym})
        models['imag'] = m

    return simplify(expr_re + I * expr_im), models

# ===========================================================================
#  Main symbol regression entry point
# ===========================================================================

def symbol_grid_to_sympy(
    x_grid,
    xi_grid,
    symbol_matrix,
    method       = 'auto',
    svd_tol      = 5e-2,
    pysr_options = None,
):
    """
    Fit a SymPy expression to a numerical symbol a(x_i, xi_j).

    Parameters
    ----------
    x_grid        : (Nx,)
    xi_grid       : (Nxi,)
    symbol_matrix : (Nx, Nxi) complex
    method        : {'auto', 'svd', 'pysr'}
        'svd'  : SVD + polynomial fit (fast, separable symbols only)
        'pysr' : symbolic regression via PySR (general, requires pysr)
        'auto' : SVD if separable, else PySR; SVD fallback if pysr missing
    svd_tol       : float   S[1]/S[0] threshold for separability in 'auto'
    pysr_options  : dict    PySR hyperparameter overrides

    Returns
    -------
    sympy_expr : SymPy expression in variables (x, xi)
    meta       : dict
        'method_used' : 'svd' or 'pysr'
        'separable'   : bool
        'svd_ratio'   : float   S[1]/S[0]
        'pysr_models' : dict (only when PySR was used)
    """
    is_sep, ratio, _, _ = _test_separability(symbol_matrix, tol=svd_tol)
    meta = {'separable': is_sep, 'svd_ratio': ratio}

    use_pysr = (method == 'pysr') or (method == 'auto' and not is_sep)

    if use_pysr:
        try:
            expr, pysr_models   = _fit_pysr(
                x_grid, xi_grid, symbol_matrix, pysr_options
            )
            meta['method_used']  = 'pysr'
            meta['pysr_models']  = pysr_models
            return expr, meta
        except ImportError as e:
            warnings.warn(
                f"{e}\nFalling back to SVD method.", UserWarning
            )

    # SVD path (either requested or fallback)
    expr              = _svd_fallback(x_grid, xi_grid, symbol_matrix)
    meta['method_used'] = 'svd'
    return expr, meta


# ===========================================================================
#  End-to-end pipeline
# ===========================================================================

def identify_operator(
    U_in,
    U_out,
    x_grid,
    window_type  = 'gaussian',
    window_width = None,
    epsilon      = 1e-6,
    xi_max       = None,
    method       = 'auto',
    pysr_options = None,
    quantization = 'kohn-nirenberg',
    weyl_order   = 4,
):
    """
    Full pipeline: (U_in, U_out) training pairs -> PseudoDifferentialOperator.

    Steps
    -----
    1. Estimate symbol numerically via STFT      [symbol_estimator]
    2. Fit SymPy expression via SVD or PySR      [this module]
    3. Wrap in PseudoDifferentialOperator        [psiop]

    Parameters
    ----------
    U_in, U_out   : (N_samples, Nx)  input/output function pairs
    x_grid        : (Nx,)            uniform periodic spatial grid
    window_type   : str              Gabor window (default 'gaussian')
    window_width  : float or None    window width in physical units
    epsilon       : float            STFT regularisation
    xi_max        : float or None    discard |xi| > xi_max
    method        : {'auto','svd','pysr'}  regression method
    pysr_options  : dict             PySR overrides
    quantization  : {'kohn-nirenberg', 'weyl'}
        The identified symbol is always a KN symbol (STFT estimates KN).
        If 'weyl', it is converted via kn_to_weyl_symbol() before wrapping.
    weyl_order    : int   truncation order for the Weyl conversion

    Returns
    -------
    op             : PseudoDifferentialOperator
    sympy_expr     : SymPy expression of the identified symbol
                     (KN convention unless quantization='weyl')
    xi_grid        : (Nxi,) frequency grid used by the estimator
    symbol_matrix  : (Nx, Nxi) numerical symbol estimate
    meta           : dict  diagnostics from symbol_grid_to_sympy

    Examples
    --------
    >>> from pde_data_generator import generate_advection_1d
    >>> x, xi, U_in, U_out, true_sym = generate_advection_1d(N_samples=300)
    >>> op, expr, xi_g, sym_mat, meta = identify_operator(
    ...     U_in, U_out, x, method='svd')
    >>> print(expr)
    >>> print(op.symbol_order())
    """
    from symbol_estimator import estimate_symbol_stft
    from psiop import PseudoDifferentialOperator

    x_sym, xi_sym = symbols('x xi', real=True)

    # Step 1 — numerical symbol estimation
    xi_grid, symbol_matrix = estimate_symbol_stft(
        U_in, U_out, x_grid,
        window_type  = window_type,
        window_width = window_width,
        epsilon      = epsilon,
        xi_max       = xi_max,
    )

    # Step 2 — symbolic regression
    sympy_expr, meta = symbol_grid_to_sympy(
        x_grid, xi_grid, symbol_matrix,
        method       = method,
        pysr_options = pysr_options,
    )

    # Step 3 — wrap as KN operator
    op = PseudoDifferentialOperator(
        expr         = sympy_expr,
        vars_x       = [x_sym],
        mode         = 'symbol',
        quantization = 'kohn-nirenberg',
    )

    # Optional conversion to Weyl
    if quantization == 'weyl':
        weyl_expr    = op.kn_to_weyl_symbol(order=weyl_order)
        op           = PseudoDifferentialOperator(
            expr         = weyl_expr,
            vars_x       = [x_sym],
            mode         = 'symbol',
            quantization = 'weyl',
        )
        meta['weyl_symbol'] = weyl_expr
        sympy_expr           = weyl_expr

    meta['identified_symbol'] = sympy_expr
    return op, sympy_expr, xi_grid, symbol_matrix, meta


# ===========================================================================
#  Validation helper
# ===========================================================================

def validate_identification(
    op_identified,
    true_symbol_func,
    x_grid,
    xi_grid,
    rel_tol = 0.15,
):
    """
    Compare the identified symbol against ground truth on the phase-space grid.

    Parameters
    ----------
    op_identified    : PseudoDifferentialOperator
    true_symbol_func : callable  a_true(x, xi)
    x_grid           : (Nx,)
    xi_grid          : (Nxi,)
    rel_tol          : float   acceptance threshold on relative L2 error

    Returns
    -------
    metrics : dict
        'rel_l2_error'   : float   relative L2 error
        'max_abs_error'  : float   maximum pointwise absolute error
        'passed'         : bool    rel_l2_error < rel_tol
    """
    x_sym, xi_sym = symbols('x xi', real=True)
    sym_func      = lambdify((x_sym, xi_sym),
                             op_identified.symbol, 'numpy')

    X_mg, XI_mg = np.meshgrid(x_grid, xi_grid, indexing='ij')

    a_id   = np.asarray(sym_func(X_mg, XI_mg), dtype=complex)
    a_true = np.asarray(true_symbol_func(X_mg, XI_mg), dtype=complex)

    err       = a_id - a_true
    rel_error = float(np.linalg.norm(err) /
                      (np.linalg.norm(a_true) + 1e-14))
    max_error = float(np.max(np.abs(err)))

    return {
        'rel_l2_error' : rel_error,
        'max_abs_error': max_error,
        'passed'       : rel_error < rel_tol,
    }