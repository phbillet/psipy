from typing import Dict, Any, List, Optional
import sympy as sp


def extract_single_nufft_term(
    term: sp.Expr, x_sym: sp.Symbol, xi_sym: sp.Symbol
) -> Optional[Dict[str, Any]]:
    """
    Attempts to factor a single (already exp-rewritten, expanded) additive
    term into:
        c(x) * g(ξ) * exp(i * λ(x) * μ(ξ))
    Returns None (never a wrong plan) on any ambiguity or coupling.
    """
    term_simp = sp.powsimp(term, combine="exp", deep=True)

    factors = sp.Mul.make_args(term_simp)
    exp_args = []
    amplitude_factors = []

    for factor in factors:
        if factor.is_Pow and factor.base == sp.E:
            exp_args.append(factor.exp)
        elif isinstance(factor, sp.exp):
            exp_args.append(factor.args[0])
        else:
            amplitude_factors.append(factor)

    if not exp_args:
        return None  # no oscillatory content in this term at all

    total_exponent = sp.expand(sum(exp_args))

    # --- Split total_exponent into real envelope + I * phase, exactly ---
    # (sum() over Add args, not .coeff(I), so nothing is silently dropped)
    exp_terms = sp.Add.make_args(total_exponent)
    imag_terms = []
    real_terms = []
    for t in exp_terms:
        c = t.coeff(sp.I)
        if sp.expand(t - sp.I * c) == 0:
            # t really is I*c with no leftover real residue
            imag_terms.append(c)
        else:
            real_terms.append(t)

    if not imag_terms:
        return None  # purely real exponent -> not oscillatory, not our tier

    phase_expr = sp.expand(sum(imag_terms))
    real_envelope = sp.expand(sum(real_terms))  # 0 if none

    # --- Phase must decouple as λ(x) * μ(ξ) exactly ---
    # phase_expr may be an unexpanded-product's Add form (e.g. x*xi - xi,
    # which IS (x-1)*xi but as_independent only splits Mul, not Add) --
    # factor first so genuinely-separable phases aren't missed.
    phase_factored = sp.factor(phase_expr)
    lambda_x, mu_xi = phase_factored.as_independent(xi_sym, as_Add=False)
    if lambda_x.has(xi_sym) or mu_xi.has(x_sym) or not phase_expr.has(xi_sym):
        return None  # coupled phase, or no x-xi coupling at all -> not this tier

    # --- Fold any real envelope (e.g. a chirp's Gaussian decay) into the
    #     amplitude via exp(), then require it separates too ---
    amplitude_expr = sp.Mul(*amplitude_factors)
    if real_envelope != 0:
        amplitude_expr = amplitude_expr * sp.exp(real_envelope)

    amplitude_factored = sp.factor(amplitude_expr) if amplitude_expr.is_Add else amplitude_expr
    c_x, g_xi = amplitude_factored.as_independent(xi_sym, as_Add=False)
    if c_x.has(xi_sym) or g_xi.has(x_sym):
        return None  # amplitude (incl. any folded envelope) is non-separable

    def safe_lambdify(sym, expr):
        f = sp.lambdify(sym, expr, modules="numpy")
        if not expr.has(sym):
            # constant expression: sympy/numpy won't broadcast this against
            # an array input on its own -- wrap it so callers always get
            # an array shaped like the input.
            import numpy as np
            const_val = complex(expr) if expr.is_number else None
            def wrapped(arr, _f=f, _c=const_val):
                arr = np.asarray(arr)
                if _c is not None:
                    return np.full_like(arr, _c, dtype=complex)
                return np.broadcast_to(_f(arr), arr.shape)
            return wrapped
        return f

    return {
        "c_x_expr": c_x,
        "g_xi_expr": g_xi,
        "lambda_x_expr": lambda_x,
        "mu_xi_expr": mu_xi,
        "c_x": safe_lambdify(x_sym, c_x),
        "g_xi": safe_lambdify(xi_sym, g_xi),
        "lambda_x": safe_lambdify(x_sym, lambda_x),
        "mu_xi": safe_lambdify(xi_sym, mu_xi),
    }


def try_nufft_decomposition(
    joint_expr: sp.Expr, x_sym: sp.Symbol, xi_sym: sp.Symbol
) -> Optional[List[Dict[str, Any]]]:
    """
    Parses joint symbol p(x, xi) to verify if all terms fit NUFFT phase patterns:
        p(x, xi) = sum_k c_k(x) * g_k(xi) * exp(i * lambda_k(x) * mu_k(xi))
    Conservative: any term that doesn't cleanly fit -> None (caller falls
    back to Chebyshev-SVD / direct quadrature), never a silently wrong plan.
    """
    # Rewrite trig/hyperbolic content as exponentials BEFORE splitting into
    # additive terms, since e.g. sin(x*xi) only becomes exp-form (and only
    # becomes two additive terms) after this rewrite.
    expr = joint_expr.rewrite(sp.exp)
    expr = sp.expand(expr)
    terms = sp.Add.make_args(expr)

    nufft_plans = []
    for term in terms:
        plan = extract_single_nufft_term(term, x_sym, xi_sym)
        if plan is None:
            return None
        nufft_plans.append(plan)

    return nufft_plans

"""
apply_nufft: Tier A of the joint-residual fallback.

Kohn-Nirenberg quantization of a joint-residual symbol p(x,xi) that has been
successfully classified by try_nufft_decomposition as a sum of terms

    p(x,xi) = sum_k c_k(x) * g_k(xi) * exp(i * lambda_k(x) * mu_k(xi))

The full KN action also carries the reconstruction kernel e^{i x xi}:

    (Op(p)u)(x) = (1/2pi) * integral  p(x,xi) * uhat(xi) * e^{i x xi} dxi

so the *total* oscillatory content per term is

    Phi_k(x,xi) = x*xi + lambda_k(x)*mu_k(xi)

which is a SUM of two rank-1 bilinear pairings sharing the xi variable, not
a single one -- so it does not reduce to a single 1D type-3 NUFFT in
general (only when lambda_k is affine in x). The general fix: embed both
pairings into R^2 and use a single 2D type-3 NUFFT:

    source points:  X_k = (xi_k, mu_k(xi_k))   with strength w_k
    target points:  Y_j = (x_j,  lambda_k(x_j))

    f(x_j) = sum_k w_k * exp(i * X_k . Y_j)
           = sum_k w_k * exp(i*(x_j*xi_k + lambda_k(x_j)*mu_k(xi_k)))
           = Phi_k-weighted sum exactly matching the KN integral for this term.

This handles linear AND nonlinear lambda_k, mu_k uniformly, with one NUFFT
call per term. Verified against brute-force direct KN quadrature to machine
precision (see validation script).
"""
from typing import List, Dict, Any
import numpy as np

try:
    import finufft
    _HAVE_FINUFFT = True
except ImportError:
    _HAVE_FINUFFT = False


def _direct_2d_type3_fallback(src_x, src_y, weights, tgt_x, tgt_y, isign=1):
    """
    O(M*N) brute-force stand-in for finufft.nufft2d3, used only if finufft
    isn't installed. Fine for validation / small grids; NOT for production
    scale (defeats the whole point of this tier) -- install finufft.
    """
    phase = isign * (tgt_x[:, None] * src_x[None, :] + tgt_y[:, None] * src_y[None, :])
    return (weights[None, :] * np.exp(1j * phase)).sum(axis=1)


def apply_nufft(
    u: np.ndarray,
    nufft_plan: List[Dict[str, Any]],
    x_grid: np.ndarray,
    kx: np.ndarray,
    dx: float,
    dxi: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply Op(p_joint) to u, where p_joint was classified by
    try_nufft_decomposition into `nufft_plan` (a list of term-dicts with
    c_x, g_xi, lambda_x, mu_xi callables -- see nufft_decomp.py).

    Parameters
    ----------
    u      : (N,) complex array, values of u on x_grid
    nufft_plan : output of try_nufft_decomposition (must not be None)
    x_grid : (N,) physical grid (uniform or Chebyshev/nonuniform -- NUFFT
             handles both, that's the point)
    kx     : (N,) frequency grid matching u's FFT convention
    dx, dxi: grid spacings / quadrature weights used to approximate the
             continuous Fourier transform and its inverse by Riemann sums
    eps    : NUFFT requested accuracy (can be pushed to ~1e-14)

    Returns
    -------
    (N,) complex array: (Op(p_joint) u)(x_grid)
    """
    u = np.asarray(u, dtype=complex)
    x_grid = np.asarray(x_grid, dtype=float)
    kx = np.asarray(kx, dtype=float)

    # continuous FT approx: uhat(xi_k) ~= sum_j u(x_j) e^{-i x_j xi_k} dx
    #
    # np.fft.fft implicitly assumes samples sit at x_j = j*dx (grid starting
    # at 0). If x_grid actually starts at x0 != 0 (e.g. a domain centered on
    # 0, x_grid = linspace(-L, L, N, endpoint=False)), the true continuous
    # FT differs from the naive fft(u)*dx by an uncompensated phase
    # e^{-i*x0*xi_k} -- without this correction, results are internally
    # self-consistent (this function vs a matching brute-force loop) but
    # both compute something other than the true KN operator.
    x0 = x_grid[0]
    uhat = np.fft.fft(u) * dx * np.exp(-1j * x0 * kx)
    # NOTE: this assumes kx is in the *unshifted* FFT frequency order that
    # matches np.fft.fft's output (i.e. kx = 2*pi*np.fft.fftfreq(N, dx)).
    # If the caller's kx is fftshift-ordered, uhat must be fftshift'd too --
    # keep these two consistent at the call site.

    result = np.zeros_like(x_grid, dtype=complex)

    for term in nufft_plan:
        c_x = term["c_x"](x_grid)      # (N,)
        g_xi = term["g_xi"](kx)        # (N,)
        lam_x = term["lambda_x"](x_grid)   # (N,)
        mu_xi = term["mu_xi"](kx)          # (N,)

        # quadrature weights for the xi-integral, folded into the NUFFT
        # source strengths (1/2pi from the KN inversion formula, dxi from
        # the Riemann-sum discretization of the continuous integral)
        weights = (g_xi * uhat * dxi / (2 * np.pi)).astype(complex)

        src_x, src_y = kx, mu_xi          # source points (xi_k, mu_k(xi_k))
        tgt_x, tgt_y = x_grid, lam_x      # target points (x_j, lambda_k(x_j))

        if _HAVE_FINUFFT:
            f = finufft.nufft2d3(
                src_x, src_y, weights, tgt_x, tgt_y, isign=1, eps=eps
            )
        else:
            f = _direct_2d_type3_fallback(src_x, src_y, weights, tgt_x, tgt_y, isign=1)

        result += c_x * f

    return result

from typing import Dict, Any, List, Optional
import sympy as sp
import numpy as np


def _split_real_imag_exponent(total_exponent):
    """Shared with the 1D version: split an exponent into (I*phase, real_envelope)
    without ever silently dropping a real residual."""
    exp_terms = sp.Add.make_args(sp.expand(total_exponent))
    imag_terms, real_terms = [], []
    for t in exp_terms:
        c = t.coeff(sp.I)
        if sp.expand(t - sp.I * c) == 0:
            imag_terms.append(c)
        else:
            real_terms.append(t)
    phase_expr = sp.expand(sum(imag_terms)) if imag_terms else sp.Integer(0)
    real_envelope = sp.expand(sum(real_terms)) if real_terms else sp.Integer(0)
    return phase_expr, real_envelope, bool(imag_terms)


def extract_single_nufft_term_2d(
    term: sp.Expr,
    x_sym: sp.Symbol, y_sym: sp.Symbol,
    xi_sym: sp.Symbol, eta_sym: sp.Symbol,
) -> Optional[Dict[str, Any]]:
    """
    Factor a single (exp-rewritten, expanded) additive term of a 2D joint
    symbol into:
        c(x,y) * g(xi,eta) * exp(i * Lambda(x,y) * M(xi,eta))

    The phase must reduce to a SINGLE product Lambda(x,y)*M(xi,eta) --
    a sum of two independent single-axis pieces needs a 4D NUFFT embedding,
    which finufft (max 3D type-3) doesn't support, so that case returns
    None (conservative: falls back to Tier B/C, never a wrong plan).
    """
    term_simp = sp.powsimp(term, combine="exp", deep=True)
    factors = sp.Mul.make_args(term_simp)

    exp_args, amplitude_factors = [], []
    for factor in factors:
        if factor.is_Pow and factor.base == sp.E:
            exp_args.append(factor.exp)
        elif isinstance(factor, sp.exp):
            exp_args.append(factor.args[0])
        else:
            amplitude_factors.append(factor)

    if not exp_args:
        return None

    total_exponent = sp.expand(sum(exp_args))
    phase_expr, real_envelope, has_osc = _split_real_imag_exponent(total_exponent)
    if not has_osc:
        return None

    freq_syms = (xi_sym, eta_sym)
    phys_syms = (x_sym, y_sym)

    # phase must factor as ONE product: (fn of x,y only) * (fn of xi,eta only)
    phase_factored = sp.factor(phase_expr)
    Lambda_xy, M_xieta = phase_factored.as_independent(*freq_syms, as_Add=False)

    coupled_to_freq = Lambda_xy.has(xi_sym) or Lambda_xy.has(eta_sym)
    coupled_to_phys = M_xieta.has(x_sym) or M_xieta.has(y_sym)
    no_real_coupling = not phase_expr.has(xi_sym) and not phase_expr.has(eta_sym)

    if coupled_to_freq or coupled_to_phys or no_real_coupling:
        return None
    # extra safety: verify the factored form actually reproduces phase_expr
    # (sp.factor can fail to find a factorization and just return the sum
    # unchanged, which as_independent would then wrongly treat as "already
    # independent" in degenerate cases -- re-expand and compare)
    if sp.expand(Lambda_xy * M_xieta - phase_expr) != 0:
        return None

    amplitude_expr = sp.Mul(*amplitude_factors)
    if real_envelope != 0:
        amplitude_expr = amplitude_expr * sp.exp(real_envelope)

    amplitude_factored = sp.factor(amplitude_expr) if amplitude_expr.is_Add else amplitude_expr
    c_xy, g_xieta = amplitude_factored.as_independent(*freq_syms, as_Add=False)
    if c_xy.has(xi_sym) or c_xy.has(eta_sym) or g_xieta.has(x_sym) or g_xieta.has(y_sym):
        return None
    if sp.expand(c_xy * g_xieta - amplitude_expr) != 0:
        return None

    return {
        "c_xy_expr": c_xy, "g_xieta_expr": g_xieta,
        "Lambda_expr": Lambda_xy, "M_expr": M_xieta,
        "c_xy": sp.lambdify((x_sym, y_sym), c_xy, "numpy"),
        "g_xieta": sp.lambdify((xi_sym, eta_sym), g_xieta, "numpy"),
        "Lambda": sp.lambdify((x_sym, y_sym), Lambda_xy, "numpy"),
        "M": sp.lambdify((xi_sym, eta_sym), M_xieta, "numpy"),
    }


def try_nufft_decomposition_2d(
    joint_expr: sp.Expr,
    x_sym: sp.Symbol, y_sym: sp.Symbol,
    xi_sym: sp.Symbol, eta_sym: sp.Symbol,
) -> Optional[List[Dict[str, Any]]]:
    """
    2D analogue of try_nufft_decomposition. Every additive term must reduce
    to c(x,y)*g(xi,eta)*exp(i*Lambda(x,y)*M(xi,eta)) (embeds in 3D, matching
    finufft's cap); any term needing 4D (independent coupling on both axes
    simultaneously) rejects the whole symbol -> caller falls back.
    """
    expr = joint_expr.rewrite(sp.exp)
    expr = sp.expand(expr)
    terms = sp.Add.make_args(expr)

    plans = []
    for term in terms:
        plan = extract_single_nufft_term_2d(term, x_sym, y_sym, xi_sym, eta_sym)
        if plan is None:
            return None
        plans.append(plan)
    return plans

"""
apply_nufft_2d: n=2 spatial case, single-joint-term tier (d=3 embed).

For a term c(x,y)*g(xi,eta)*exp(i*Lambda(x,y)*M(xi,eta)) from
try_nufft_decomposition_2d, the full KN action

    (Op(p)u)(x,y) = (1/2pi)^2 * integral p(x,y,xi,eta) * uhat(xi,eta)
                                          * e^{i(x*xi+y*eta)} dxi deta

has total phase  x*xi + y*eta + Lambda(x,y)*M(xi,eta)
                = (x, y, Lambda(x,y)) . (xi, eta, M(xi,eta))   [R^3 dot product]

-> one finufft.nufft3d3 call per term.
"""
from typing import List, Dict, Any
import numpy as np

try:
    import finufft
    _HAVE_FINUFFT = True
except ImportError:
    _HAVE_FINUFFT = False


def _direct_3d_type3_fallback(sx, sy, sz, weights, tx, ty, tz, isign=1):
    """O(M*N) brute-force stand-in for finufft.nufft3d3 (validation only)."""
    phase = isign * (
        tx[:, None] * sx[None, :]
        + ty[:, None] * sy[None, :]
        + tz[:, None] * sz[None, :]
    )
    return (weights[None, :] * np.exp(1j * phase)).sum(axis=1)


def apply_nufft_2d(
    u: np.ndarray,          # (Nx, Ny) complex, u on (x_grid, y_grid) meshgrid
    nufft_plan: List[Dict[str, Any]],
    x_grid: np.ndarray, y_grid: np.ndarray,   # 1D physical grids
    kx: np.ndarray, ky: np.ndarray,           # 1D frequency grids
    dx: float, dy: float, dxi: float, deta: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Apply Op(p_joint) to a 2D field u, where p_joint was classified by
    try_nufft_decomposition_2d into `nufft_plan`.

    Returns (Nx, Ny) complex array: (Op(p_joint) u)(x_grid, y_grid).
    """
    u = np.asarray(u, dtype=complex)
    Nx, Ny = len(x_grid), len(y_grid)
    assert u.shape == (Nx, Ny)

    # continuous 2D FT approx: uhat(xi_k,eta_l) ~= sum_{ij} u(x_i,y_j)
    #                            e^{-i(x_i*xi_k + y_j*eta_l)} dx dy
    #
    # Same grid-origin correction as apply_nufft (1D) -- see comment there.
    # Needed independently on each axis whenever x_grid/y_grid don't start
    # at 0 (e.g. a domain centered on 0).
    x0, y0 = x_grid[0], y_grid[0]
    XI0, ETA0 = np.meshgrid(kx, ky, indexing="ij")
    uhat = np.fft.fft2(u) * dx * dy * np.exp(-1j * (x0 * XI0 + y0 * ETA0))
    # shape (Nx, Ny), axes <-> (kx, ky)

    XI, ETA = np.meshgrid(kx, ky, indexing="ij")       # (Nx, Ny) freq grid
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")  # (Nx, Ny) phys grid

    result = np.zeros((Nx, Ny), dtype=complex)

    for term in nufft_plan:
        c_xy = term["c_xy"](X, Y)              # (Nx, Ny)
        g_xieta = term["g_xieta"](XI, ETA)      # (Nx, Ny)
        Lambda_xy = term["Lambda"](X, Y)        # (Nx, Ny)
        M_xieta = term["M"](XI, ETA)            # (Nx, Ny)

        # flatten to point clouds for the (nonuniform) NUFFT call
        src_xi = XI.ravel()
        src_eta = ETA.ravel()
        src_M = np.broadcast_to(M_xieta, (Nx, Ny)).ravel()
        weights = (np.broadcast_to(g_xieta, (Nx, Ny)) * uhat
                   * dxi * deta / (2 * np.pi) ** 2).ravel().astype(complex)

        tgt_x = X.ravel()
        tgt_y = Y.ravel()
        tgt_L = np.broadcast_to(Lambda_xy, (Nx, Ny)).ravel()

        if _HAVE_FINUFFT:
            f = finufft.nufft3d3(
                src_xi, src_eta, src_M, weights,
                tgt_x, tgt_y, tgt_L, isign=1, eps=eps,
            )
        else:
            f = _direct_3d_type3_fallback(
                src_xi, src_eta, src_M, weights, tgt_x, tgt_y, tgt_L, isign=1
            )

        result += c_xy * f.reshape(Nx, Ny)

    return result


if __name__ == "__main__":
    import sympy as sp

    x, y, xi, eta = sp.symbols("x y xi eta", real=True)

    # deliberately small grid: brute-force reference is O(Nx*Ny*Nxi*Neta)
    Nx, Ny = 10, 10
    Lx, Ly = 5.0, 4.0
    x_grid = np.linspace(-Lx, Lx, Nx, endpoint=False)
    y_grid = np.linspace(-Ly, Ly, Ny, endpoint=False)
    dx, dy = x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    dxi, deta = kx[1] - kx[0], ky[1] - ky[0]

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    u = (np.exp(-(X**2 + Y**2) / 6) * (1 + 0.3 * np.sin(1.3 * X) * np.cos(0.9 * Y))).astype(complex)

    def brute_force_kn_2d(p_lamb, u):
        x0, y0 = x_grid[0], y_grid[0]
        XI0, ETA0 = np.meshgrid(kx, ky, indexing="ij")
        uhat = np.fft.fft2(u) * dx * dy * np.exp(-1j * (x0 * XI0 + y0 * ETA0))
        out = np.zeros((Nx, Ny), dtype=complex)
        for i, xv in enumerate(x_grid):
            for j, yv in enumerate(y_grid):
                integrand = p_lamb(xv, yv, XI0, ETA0) * uhat * np.exp(1j * (xv * XI0 + yv * ETA0))
                out[i, j] = integrand.sum() * dxi * deta / (2 * np.pi) ** 2
        return out

    # ground-truth sanity checks (catch grid-origin/convention bugs that a
    # self-vs-self comparison below can't) -- p=1 must reconstruct u exactly
    print("--- ground-truth checks ---")
    recon = brute_force_kn_2d(lambda xv, yv, XI, ETA: 1.0, u)
    print(f"{'p=1 reconstructs u':30s}  max abs err = {np.max(np.abs(recon-u)):.3e}")
    recon_y = brute_force_kn_2d(lambda xv, yv, XI, ETA: yv, u)
    print(f"{'p=y gives y*u':30s}  max abs err = {np.max(np.abs(recon_y - u*Y)):.3e}")
    print()

    tests = [
        ("exp(I*x*xi)  [y trivial]", sp.exp(sp.I * x * xi)),
        ("sin((x+y)*xi)  [genuinely joint Lambda(x,y)=x+y]", sp.sin((x + y) * xi)),
    ]

    for name, expr in tests:
        plan = try_nufft_decomposition_2d(expr, x, y, xi, eta)
        assert plan is not None, name

        approx = apply_nufft_2d(u, plan, x_grid, y_grid, kx, ky, dx, dy, dxi, deta)

        p_lamb = sp.lambdify((x, y, xi, eta), expr, "numpy")
        direct = brute_force_kn_2d(p_lamb, u)

        err = np.max(np.abs(approx - direct))
        scale = np.max(np.abs(direct)) + 1e-300
        print(f"{name:45s}  max abs err = {err:.3e}  (scale {scale:.3e})")



    x, xi = sp.symbols("x xi", real=True)

    N = 64
    Lx = 8.0
    x_grid = np.linspace(-Lx, Lx, N, endpoint=False)
    dx = x_grid[1] - x_grid[0]
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    dxi = kx[1] - kx[0]

    rng = np.random.default_rng(0)
    u = (np.exp(-x_grid**2 / 3) * (1 + 0.4 * np.sin(1.7 * x_grid))).astype(complex)

    for name, expr in [
        ("sin(x*xi)", sp.sin(x * xi)),
        ("exp(I*x*xi)", sp.exp(sp.I * x * xi)),
        ("x*exp(I*x*xi)*cos(xi)", x * sp.exp(sp.I * x * xi) * sp.cos(xi)),
    ]:
        plan = try_nufft_decomposition(expr, x, xi)
        assert plan is not None, name

        approx = apply_nufft(u, plan, x_grid, kx, dx, dxi)

        # brute-force ground truth (must use the SAME x0-corrected uhat, or
        # this "validation" would just be comparing against a stale
        # convention again -- see the grid-origin bug writeup)
        p_lamb = sp.lambdify((x, xi), expr, "numpy")
        x0 = x_grid[0]
        uhat = np.fft.fft(u) * dx * np.exp(-1j * x0 * kx)
        direct = np.zeros(N, dtype=complex)
        for j, xj in enumerate(x_grid):
            direct[j] = np.sum(p_lamb(xj, kx) * uhat * np.exp(1j * xj * kx)) * dxi / (2 * np.pi)

        err = np.max(np.abs(approx - direct))
        scale = np.max(np.abs(direct)) + 1e-300
        print(f"{name:30s}  max abs err = {err:.3e}   (relative to scale {scale:.3e})")

    x, y, xi, eta = sp.symbols("x y xi eta", real=True)

    tests = [
        ("exp(I*x*xi)  [only x-axis coupled, y trivial -> should pass, dim 3]",
         sp.exp(sp.I * x * xi)),
        ("sin(x*xi)*cos(y*eta)  [PRODUCT of two axis oscillations -> after "
         "expansion, cross terms are single products -> should pass]",
         sp.sin(x * xi) * sp.cos(y * eta)),
        ("sin((x+y)*xi)  [Lambda(x,y)=x+y genuinely joint -> should pass, dim 3]",
         sp.sin((x + y) * xi)),
        ("exp(I*x*xi) + exp(I*y*eta)  [SUM of independent axis couplings -> needs 4D -> must reject]",
         sp.exp(sp.I * x * xi) + sp.exp(sp.I * y * eta)),
        ("exp(I*(x*xi + y*eta))  [pure reconstruction-shaped term, phase already just the base pairing "
         "summed as ONE exp -> check behavior]",
         sp.exp(sp.I * (x * xi + y * eta))),
    ]

    for name, expr in tests:
        print(f"\n{name}")
        r = try_nufft_decomposition_2d(expr, x, y, xi, eta)
        if r is None:
            print("  -> None (fallback)")
        else:
            for p in r:
                print(f"  -> c={p['c_xy_expr']}, g={p['g_xieta_expr']}, "
                      f"Lambda={p['Lambda_expr']}, M={p['M_expr']}")
                
    x, xi = sp.symbols('x xi', real=True)

    tests = [
        ("exp(I*x*xi)", sp.exp(sp.I*x*xi)),
        ("sin(x*xi)", sp.sin(x*xi)),
        ("cos(x*xi)", sp.cos(x*xi)),
        ("exp(I*x*xi - x**2)  [chirp+envelope, must fold envelope]", sp.exp(sp.I*x*xi - x**2)),
        ("x**2 * exp(I*x*xi) * sin(xi)  [separable amplitude, should pass]",
         x**2 * sp.exp(sp.I*x*xi) * sp.sin(xi)),
        ("exp(I*x**2*xi**2)  [genuinely coupled, must reject]", sp.exp(sp.I*x**2*xi**2)),
        ("exp(-x**2 - xi**2)  [pure real, not oscillatory, must reject]", sp.exp(-x**2 - xi**2)),
        ("sin(x*xi) + cos(x**2*xi)  [mixed: one clean term, one coupled -> must reject whole sum]",
         sp.sin(x*xi) + sp.cos(x**2*xi)),
    ]

    for name, expr in tests:
        print(f"\n{name}")
        result = try_nufft_decomposition(expr, x, xi)
        if result is None:
            print("  -> None (fallback to Chebyshev/direct)")
        else:
            for p in result:
                print(f"  -> c(x)={p['c_x_expr']}, g(xi)={p['g_xi_expr']}, "
                      f"lambda(x)={p['lambda_x_expr']}, mu(xi)={p['mu_xi_expr']}")