# Copyright 2026 Philippe Billet assisted by LLMs in free mode: chatGPT, Qwen, Deepseek, Gemini, Claude, le chat Mistral.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
hamiltonian_catalog.py — Hamiltonian catalog for analysis, visualization and more...
=========================================================================================

Extended catalog of 1D and 2D Hamiltonians for pseudodifferential
and semiclassical analysis, visualization, and symbolic exploration.

Each entry:
    key : {
        "expr"        : sympy.Expr,
        "dim"         : 1 or 2,
        "category"    : str,
        "description" : str
    }

Usage:
    from hamiltonian_catalog_extended import get_hamiltonian
    H, vars, meta = get_hamiltonian("henon_heiles")
"""

import sympy as sp
import numpy as np
from collections import Counter
import itertools
import os
import json


from imports import * 
from hamiltonian_db import *

# ---------------------------------------------------------------------
# SYMBOLS (shared)
# ---------------------------------------------------------------------
# Real variables (real=True)
x, y, xi, eta = sp.symbols("x y xi eta", real=True)
mu1, mu2, sigma1, sigma2, zeta, z = sp.symbols("mu1 mu2 sigma1 sigma2 zeta z", real=True)
T = sp.symbols("T", real=True)

# Real and positive variables(real=True, positive=True)
m, k, alpha, beta, gamma, delta, omega, B, g, eps, A, V0, lambda_param, theta = sp.symbols(
    "m k alpha beta gamma delta omega B g eps A V0 lambda theta", real=True, positive=True
)
R, Delta, mu, golden_ratio, A_param, B_param, C_param, f_param, g_param, L_z = sp.symbols(
    "R Delta mu golden_ratio A_param B_param C_param f_param g_param L_z", real=True, positive=True
)

# =====================================================================
# Utility functions
# =====================================================================
DYNAMIC_VARS = {x, y, xi, eta}

def get_parameters(expr: sp.Expr, dim: int) -> List[sp.Symbol]:
    """
    Extract free parameters from a Hamiltonian expression,
    i.e. symbols that are not dynamical variables.
    """
    vars_set = {x, xi} if dim == 1 else {x, y, xi, eta}
    return sorted(expr.free_symbols - vars_set, key=lambda s: s.name)

import re  # add at the top of the file if not already present


class _InitialConditionError(ValueError):
    """Raised when an initial condition leads to a singular or non-finite Hamiltonian flow."""
    pass


def _check_initial_condition_1d(H_num: sp.Expr, x0: float, xi0: float) -> None:
    """
    Check that H and its first-order derivatives are finite at (x0, xi0).
    Raises _InitialConditionError with a descriptive message if not.
    """
    f_H  = sp.lambdify((x, xi), H_num,  'numpy')
    f_Hx = sp.lambdify((x, xi), sp.diff(H_num, x),  'numpy')
    f_Hxi = sp.lambdify((x, xi), sp.diff(H_num, xi), 'numpy')
    for label, func in [("H", f_H), ("∂H/∂x", f_Hx), ("∂H/∂ξ", f_Hxi)]:
        try:
            val = float(func(x0, xi0))
            if not np.isfinite(val):
                raise _InitialConditionError(
                    f"{label}({x0}, {xi0}) = {val} (non-finite). "
                    f"Please choose a different starting point."
                )
        except (ZeroDivisionError, ValueError, FloatingPointError) as e:
            raise _InitialConditionError(
                f"{label} is singular at (x0={x0}, xi0={xi0}): {e}"
            )


def _check_initial_condition_2d(
    H_num: sp.Expr, 
    x0: float, 
    y0: float, 
    xi0: float, 
    eta0: float
) -> None:
    """
    Check that H and its first-order derivatives are finite at (x0, y0, xi0, eta0).
    Raises _InitialConditionError with a descriptive message if not.
    """
    f_H    = sp.lambdify((x, y, xi, eta), H_num, 'numpy')
    f_Hx   = sp.lambdify((x, y, xi, eta), sp.diff(H_num, x),   'numpy')
    f_Hy   = sp.lambdify((x, y, xi, eta), sp.diff(H_num, y),   'numpy')
    f_Hxi  = sp.lambdify((x, y, xi, eta), sp.diff(H_num, xi),  'numpy')
    f_Heta = sp.lambdify((x, y, xi, eta), sp.diff(H_num, eta), 'numpy')
    for label, func in [
        ("H", f_H), ("∂H/∂x", f_Hx), ("∂H/∂y", f_Hy),
        ("∂H/∂ξ", f_Hxi), ("∂H/∂η", f_Heta)
    ]:
        try:
            val = float(func(x0, y0, xi0, eta0))
            if not np.isfinite(val):
                raise _InitialConditionError(
                    f"{label}({x0}, {y0}, {xi0}, {eta0}) = {val} (non-finite). "
                    f"Please choose a different starting point."
                )
        except (ZeroDivisionError, ValueError, FloatingPointError) as e:
            raise _InitialConditionError(
                f"{label} is singular at (x0={x0}, y0={y0}, xi0={xi0}, eta0={eta0}): {e}"
            )


def visualize_hamiltonian(name: str) -> Any:
    """
    Interactive visualization of a Hamiltonian from the catalog.

    Prompts the user to provide:
      - numerical values for all free parameters (m, k, eps, ...)
      - variable ranges (x_range, xi_range, and y_range/eta_range in 2D)
      - initial conditions for geodesics
      - E_range, hbar, resolution

    Then calls geometry.visualize_symbol (1D) or geometry.visualize_symbol_2d (2D).

    Parameters
    ----------
    name : str
        Key of the Hamiltonian in the catalog.

    Returns
    -------
    Result of visualize_symbol or visualize_symbol_2d.

    Example
    -------
    >>> visualize_hamiltonian('harmonic_oscillator')
    >>> visualize_hamiltonian('henon_heiles')
    """
    from geometry import visualize_symbol, visualize_symbol_2d
    import matplotlib.pyplot as plt

    # ── 1. Retrieve the Hamiltonian ──────────────────────────────────────────
    if name not in CATALOG:
        raise KeyError(f"Unknown Hamiltonian: '{name}'. "
                       f"Use list_hamiltonians() to see available entries.")
    info   = CATALOG[name]
    H_expr = info["expr"]
    dim    = info["dim"]

    print(f"\n{'='*60}")
    print(f"  Hamiltonian : {name}  ({dim}D — {info['category']})")
    print(f"  {info['description']}")
    print(f"  H = {H_expr}")
    print(f"{'='*60}\n")

    # ── 2. Assign numerical values to free parameters ────────────────────────
    params       = get_parameters(H_expr, dim)
    param_values = {}

    if params:
        print("Free parameters detected:", [str(p) for p in params])
        print("Enter a numerical value for each parameter.\n")
        for p in params:
            while True:
                try:
                    val = float(input(f"  {p} = "))
                    param_values[p] = val
                    break
                except ValueError:
                    print("  ✗ Invalid value, please try again.")
        H_num = H_expr.subs(param_values)
    else:
        print("  (No free parameters — H depends only on dynamical variables)\n")
        H_num = H_expr

    print(f"\n  Substituted H = {H_num}\n")

    # ── Helper: ask for a (min, max) range ───────────────────────────────────
    def ask_range(label, default=(-3.0, 3.0)):
        default_str = f"[{default[0]}, {default[1]}]"
        raw = input(f"  {label} (min max, default {default_str}) : ").strip()
        if not raw:
            return default
        parts = raw.split()
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
        raise ValueError(f"Expected format: 'min max', got: '{raw}'")

    def ask_float(label, default):
        raw = input(f"  {label} (default {default}) : ").strip()
        return float(raw) if raw else default

    def ask_int(label, default):
        raw = input(f"  {label} (default {default}) : ").strip()
        return int(raw) if raw else default

    # ── 3. Variable ranges ───────────────────────────────────────────────────
    print("─── Variable ranges ────────────────────────────────────────────")
    x_range  = ask_range("x_range ")
    xi_range = ask_range("xi_range")
    if dim == 2:
        y_range   = ask_range("y_range  ")
        eta_range = ask_range("eta_range")

    # ── 4. Spectral parameters ───────────────────────────────────────────────
    print("\n─── Spectral parameters ────────────────────────────────────────")
    use_erange = input("  Define E_range? (y/n, default n) : ").strip().lower()
    if use_erange == 'y':
        E_range = ask_range("E_range", default=(0.5, 4.0))
    else:
        E_range = None

    hbar       = ask_float("hbar      ", default=1.0)
    resolution = ask_int  ("resolution", default=80 if dim == 1 else 40)

    # ── 5. Geodesic initial conditions ───────────────────────────────────────
    print("\n─── Geodesic initial conditions ────────────────────────────────")
    if dim == 1:
        print("  Format : x0 xi0 t_max [color]")
        print("  Example: 0.0 1.5 10 royalblue")
    else:
        print("  Format : x0 y0 xi0 eta0 t_max [color]")
        print("  Example: 1.0 0.0 0.0 1.5 6.28 royalblue")

    # Warn the user if H has known singularities (sqrt, log, 1/x ...)
    H_str = str(H_num)
    singularity_warnings = []
    if 'sqrt' in H_str:
        singularity_warnings.append("sqrt(·) — avoid initial conditions where the argument is zero or negative")
    if 'log' in H_str:
        singularity_warnings.append("log(·) — avoid initial conditions where the argument is zero or negative")
    if re.search(r'1/x|1/y|\bx\b\s*\*\*\s*-|\by\b\s*\*\*\s*-', H_str):
        singularity_warnings.append("1/x or 1/y — avoid x=0 or y=0")
    if singularity_warnings:
        print("\n  ⚠ Singularity warning for this Hamiltonian:")
        for w in singularity_warnings:
            print(f"    · {w}")
        print()

    COLORS = ['royalblue', 'crimson', 'seagreen', 'darkorange',
              'purple', 'gold', 'teal', 'hotpink']
    geodesics_params = []
    geo_idx = 0

    while True:
        raw = input(f"  Geodesic {geo_idx+1} (empty line to finish) : ").strip()
        if not raw:
            if geo_idx == 0:
                print("  ✗ At least one geodesic is required.")
                continue
            break
        parts = raw.split()
        try:
            if dim == 1:
                if len(parts) < 3:
                    raise ValueError
                x0, xi0, t_max = float(parts[0]), float(parts[1]), float(parts[2])
                color = parts[3] if len(parts) >= 4 else COLORS[geo_idx % len(COLORS)]
                # Validate: check H_num and its x-derivative are finite at (x0, xi0)
                _check_initial_condition_1d(H_num, x0, xi0)
                geodesics_params.append((x0, xi0, t_max, color))
            else:
                if len(parts) < 5:
                    raise ValueError
                x0, y0 = float(parts[0]), float(parts[1])
                xi0, eta0, t_max = float(parts[2]), float(parts[3]), float(parts[4])
                color = parts[5] if len(parts) >= 6 else COLORS[geo_idx % len(COLORS)]
                # Validate: check H_num and its derivatives are finite at (x0, y0, xi0, eta0)
                _check_initial_condition_2d(H_num, x0, y0, xi0, eta0)
                geodesics_params.append((x0, y0, xi0, eta0, t_max, color))
            geo_idx += 1
        except _InitialConditionError as e:
            print(f"  ✗ Invalid initial condition: {e}")
        except (ValueError, IndexError):
            expected = "x0 xi0 t_max [color]" if dim == 1 else "x0 y0 xi0 eta0 t_max [color]"
            print(f"  ✗ Invalid format. Expected: {expected}")

    # ── 6. Summary before launching ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Launching visualization of '{name}'...")
    if param_values:
        print(f"  Parameters : { {str(k): v for k, v in param_values.items()} }")
    print(f"  x_range={x_range}, xi_range={xi_range}", end="")
    if dim == 2:
        print(f", y_range={y_range}, eta_range={eta_range}", end="")
    print(f"\n  hbar={hbar}, resolution={resolution}, E_range={E_range}")
    print(f"  Geodesics : {len(geodesics_params)}")
    print(f"{'─'*60}\n")

    # ── 7. Call geometry ─────────────────────────────────────────────────────
    if dim == 1:
        result = visualize_symbol(
            symbol           = H_num,
            x_range          = x_range,
            xi_range         = xi_range,
            geodesics_params = geodesics_params,
            E_range          = E_range,
            hbar             = hbar,
            resolution       = resolution,
            x_sym            = x,
            xi_sym           = xi,
        )
    else:
        result = visualize_symbol_2d(
            symbol           = H_num,
            x_range          = x_range,
            y_range          = y_range,
            xi_range         = xi_range,
            eta_range        = eta_range,
            geodesics_params = geodesics_params,
            E_range          = E_range,
            hbar             = hbar,
            resolution       = resolution,
            x_sym            = x,
            y_sym            = y,
            xi_sym           = xi,
            eta_sym          = eta,
        )

    plt.show()
    return result
    
def get_operator(
    name: str, 
    param_values: Optional[Dict[Union[str, sp.Symbol], Any]] = None, 
    quantization: str = 'weyl'
) -> Any:
    """
    Instantiate a PseudoDifferentialOperator from a catalog Hamiltonian.

    Parameters
    ----------
    name : str
        Catalog key.
    param_values : dict, optional
        Free parameter substitutions, e.g. {'m': 1.0, 'k': 2.0}.
        If None, the symbol is used as-is (symbolic parameters kept).
    quantization : {'weyl', 'kn'}
        Quantization scheme passed to PseudoDifferentialOperator.

    Returns
    -------
    PseudoDifferentialOperator
    """
    from psiop import PseudoDifferentialOperator
    H, vars, params, info = get_hamiltonian(name)
    H_num = H.subs(_resolve_param_values(param_values)) if param_values else H
    vars_x = [x] if info['dim'] == 1 else [x, y]
    return PseudoDifferentialOperator(expr=H_num, vars_x=vars_x,
                                      mode='symbol', quantization=quantization)

def _resolve_param_values(
    param_values: Optional[Dict[Union[str, sp.Symbol], Any]]
) -> Dict[sp.Symbol, Any]:
    """
    Convert string keys in param_values to sympy Symbols,
    looking them up in the global SYMBOLS table first to preserve
    assumptions (real=True, positive=True, etc.).
    """
    if not param_values:
        return {}
    # Build a lookup from name → sympy Symbol for all known symbols
    all_symbols = {s.name: s for s in [
        x, y, xi, eta, m, k, alpha, beta, gamma, delta, omega,
        B, g, eps, A, V0, lambda_param, theta, R, Delta, mu,
        golden_ratio, A_param, B_param, C_param, f_param, g_param,
        L_z, mu1, mu2, sigma1, sigma2, zeta, z, T
    ]}
    resolved = {}
    for key, val in param_values.items():
        if isinstance(key, str):
            sym = all_symbols.get(key, sp.Symbol(key))
        else:
            sym = key  # already a sympy Symbol
        resolved[sym] = val
    return resolved

def analyze_hamiltonian(
    name: str, 
    param_values: Optional[Dict[Union[str, sp.Symbol], Any]] = None,
    x_grid: Optional[np.ndarray] = None, 
    xi_grid: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Symbolic and numerical ΨDO analysis of a catalog Hamiltonian.

    Computes: operator order, ellipticity, self-adjointness,
    principal symbol, formal adjoint, symplectic flow equations.

    Returns
    -------
    dict with keys: 'order', 'is_elliptic', 'is_self_adjoint',
                    'principal_symbol', 'formal_adjoint', 'symplectic_flow'
    """
    import numpy as np
    H, vars, params, info = get_hamiltonian(name)
    dim = info['dim']

    # Substitute parameters BEFORE instantiating the operator
    print("param_values = ", param_values)
    if param_values:
        H_num = H.subs(_resolve_param_values(param_values))
    else:
        H_num = H

    print("H_num = ", H_num)

    # Warn if free parameters remain — is_elliptic_numerically will fail
    remaining = get_parameters(H_num, dim)
    if remaining:
        raise ValueError(
            f"Cannot run analyze_hamiltonian('{name}'): symbolic parameters remain "
            f"after substitution: {[str(p) for p in remaining]}.\n"
            f"Please provide numerical values via param_values={{{', '.join(repr(str(p))+': ?' for p in remaining)}}}."
        )

    # Build operator from the fully numerical symbol
    from psiop import PseudoDifferentialOperator
    vars_x = [x] if dim == 1 else [x, y]
    op = PseudoDifferentialOperator(expr=H_num, vars_x=vars_x,
                                    mode='symbol', quantization='weyl')

    if x_grid is None:
        x_grid  = np.linspace(-3, 3, 50)
    if xi_grid is None:
        xi_grid = np.linspace(-3, 3, 50)

    result = {
        'order':            op.symbol_order(),
        'principal_symbol': op.principal_symbol(),
        'formal_adjoint':   op.formal_adjoint(),
        'symplectic_flow':  op.symplectic_flow(),
        'is_self_adjoint':  op.is_self_adjoint(),
        'is_elliptic':      op.is_elliptic_numerically(x_grid, xi_grid),
    }
    return result

def trace_hamiltonian(
    name: str, 
    param_values: Optional[Dict[Union[str, sp.Symbol], Any]] = None,
    numerical: bool = False, 
    x_bounds: Optional[Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]] = None, 
    xi_bounds: Optional[Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]] = None
) -> Any:
    """
    Compute the semiclassical trace of a catalog Hamiltonian.
    Wraps PseudoDifferentialOperator.trace_formula().

    Parameters
    ----------
    x_bounds : tuple (min, max), optional
        Spatial integration bounds, e.g. (-3, 3).
    xi_bounds : tuple (min, max), optional
        Frequency integration bounds, e.g. (-3, 3).
    """
    op = get_operator(name, param_values)

    # psiop.trace_formula expects x_bounds as ((min, max),) in 1D — wrap accordingly
    if x_bounds is not None and op.dim == 1:
        x_bounds_wrapped  = (x_bounds,)
        xi_bounds_wrapped = (xi_bounds,)
    else:
        x_bounds_wrapped  = x_bounds
        xi_bounds_wrapped = xi_bounds

    return op.trace_formula(
        numerical=numerical,
        x_bounds=x_bounds_wrapped,
        xi_bounds=xi_bounds_wrapped,
    )

def interactive_hamiltonian(name: str, param_values: dict = None, **kwargs):
    """
    Launch the interactive ipywidgets dashboard for a catalog Hamiltonian.

    Wraps PseudoDifferentialOperator.interactive_symbol_analysis().
    Requires a Jupyter environment.
    """
    from psiop import PseudoDifferentialOperator
    op = get_operator(name, param_values)
    PseudoDifferentialOperator.interactive_symbol_analysis(op, **kwargs)
    
def get_hamiltonian(
    name: str
) -> Tuple[sp.Expr, Tuple[sp.Symbol, ...], Tuple[sp.Symbol, ...], Dict[str, Any]]:
    """
    Return Hamiltonian expression, variables, and metadata.

    Parameters
    ----------
    name : str
        Key identifier for the Hamiltonian.

    Returns
    -------
    H : sympy.Expr
        The Hamiltonian expression.
    vars : tuple
        Variables (x, xi) for 1D or (x, y, xi, eta) for 2D.
    params : tuple
        Parameters in the Hamiltonian
    info : dict
        Metadata including dimension, category, and description.

    Example
    -------
    >>> H, vars, params, meta = get_hamiltonian("henon_heiles")
    >>> print(meta["description"])
    Hénon–Heiles: benchmark for mixed regular/chaotic motion.
    """
    if name not in CATALOG:
        available = list(CATALOG.keys())[:10]
        raise KeyError(
            f"Unknown Hamiltonian '{name}'.\n"
            f"Available (first 10): {available}\n"
            f"Use list_hamiltonians() to see all {len(CATALOG)} entries."
        )
    info = CATALOG[name]
    H = info["expr"]
    dim = info["dim"]
    vars = (x, xi) if dim == 1 else (x, y, xi, eta)
    params = tuple(get_parameters(H, dim))  
    return H, vars, params, info           

def list_categories() -> Dict[str, int]:
    """
    List all categories and their counts.

    Returns
    -------
    dict
        Dictionary mapping category names to counts.
    """
    c = Counter([v["category"] for v in CATALOG.values()])
    return dict(c)

def list_hamiltonians(
    category: Optional[str] = None, 
    dim: Optional[int] = None
) -> List[str]:
    """
    List Hamiltonian names, optionally filtered by category or dimension.

    Parameters
    ----------
    category : str, optional
        Filter by category (e.g., 'chaotic', 'integrable').
    dim : int, optional
        Filter by dimension (1 or 2).

    Returns
    -------
    list
        List of Hamiltonian names matching the criteria.

    Example
    -------
    >>> list_hamiltonians(category='chaotic')
    ['henon_heiles', 'quartic_coupled', ...]
    >>> list_hamiltonians(dim=1)
    ['free_particle', 'harmonic_oscillator', ...]
    """
    result = []
    for name, info in CATALOG.items():
        if category and info["category"] != category:
            continue
        if dim and info["dim"] != dim:
            continue
        result.append(name)
    return sorted(result)

def search_hamiltonians(keyword: str) -> List[str]:
    """
    Search for Hamiltonians by keyword in name or description.

    Parameters
    ----------
    keyword : str
        Search term (case-insensitive).

    Returns
    -------
    list
        List of matching Hamiltonian names.

    Example
    -------
    >>> search_hamiltonians('pendulum')
    ['double_pendulum_reduced', 'driven_pendulum', 'spherical_pendulum', ...]
    """
    keyword = keyword.lower()
    result = []
    for name, info in CATALOG.items():
        if keyword in name.lower() or keyword in info["description"].lower():
            result.append(name)
    return sorted(result)

def print_hamiltonian_info(name: str) -> None:
    """
    Print detailed information about a specific Hamiltonian.

    Parameters
    ----------
    name : str
        Hamiltonian identifier.
    """
    H, vars, params, info = get_hamiltonian(name)
    print(f"\n{'='*70}")
    print(f"Hamiltonian: {name}")
    print(f"{'='*70}")
    print(f"Category:    {info['category']}")
    print(f"Dimension:   {info['dim']}D")
    print(f"Variables:   {vars}")
    print(f"Parameters:  {params if params else '(none)'}")   # ← nouveau
    print(f"\nDescription:\n  {info['description']}")
    print(f"\nExpression:\n  H = {H}")
    print(f"{'='*70}\n")

def get_catalog_summary() -> str:
    """
    Return a formatted summary of the entire catalog.

    Returns
    -------
    str
        Multi-line summary with statistics.

    Note: use `print(get_catalog_summary())`
    """
    total = len(CATALOG)
    categories = list_categories()
    dim_1 = len([v for v in CATALOG.values() if v["dim"] == 1])
    dim_2 = len([v for v in CATALOG.values() if v["dim"] == 2])

    summary = [
        "=" * 70,
        "HAMILTONIAN CATALOG SUMMARY",
        "=" * 70,
        f"Total Hamiltonians: {total}",
        f"  - 1D systems: {dim_1}",
        f"  - 2D systems: {dim_2}",
        "",
        "Categories:",
    ]

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        summary.append(f"  {cat:20s} : {count:3d}")

    summary.append("=" * 70)
    return "\n".join(summary)

def get_hamiltonians_by_keywords(*keywords: str) -> List[str]:
    """
    Multi-keyword search with AND operator.
    Example
    -------
    >>> get_hamiltonians_by_keywords('quantum', 'oscillator')
    """
    results = []
    for name, info in CATALOG.items():
        text = (name + ' ' + info['description']).lower()
        if all(kw.lower() in text for kw in keywords):
            results.append(name)
    return sorted(results)

def get_tree() -> Dict[str, Dict[str, int]]:
    """
    Returns a hierarchical tree of categories reflecting the full scope
    of the extended Hamiltonian catalog (including physical, biological,
    social, cognitive, and speculative systems).

    Returns
    -------
    dict
        Tree structured by super-categories mapping to subcategories
        that actually appear in the catalog, with counts of Hamiltonians per subcategory.
    """
    # First, collect all actual categories used and count Hamiltonians per category
    from collections import Counter
    category_counts = Counter(v["category"] for v in CATALOG.values())
    all_categories = set(category_counts.keys())

    tree = {
        # ────────────────────────────────
        # PHYSICAL SCIENCES
        # ────────────────────────────────
        "Classical & Celestial Mechanics": [
            "integrable", "chaotic", "nonlinear", "classical", "integrable_advanced",
            "astrophysics", "geophysics", "climate"
        ],
        "Quantum & Atomic Physics": [
            "quantum", "atomic", "molecular", "nuclear", "ultracold", "mesoscopic",
            "quantum_topological_extended", "semiclassical"
        ],
        "Field Theory & High-Energy Physics": [
            "qft", "particle_physics", "string_theory", "quantum_gravity",
            "supersymmetry", "bsm", "dark_sector", "neutrino", "exotic_matter",
            "field_theory"
        ],
        "Condensed Matter & Materials": [
            "lattice", "spin_systems", "spin_glass", "defects", "metamaterials",
            "topological", "quantum_info", "quantum_info_advanced", "polymers"
        ],
        "Electromagnetism & Optics": [
            "magnetic", "optical", "plasma", "cavity_qed", "nonlinear_optics",
            "acoustics", "rotating"
        ],
        "Relativity & Gravitation": [
            "relativistic", "black_holes", "cosmology", "geometric", "geometric_advanced"
        ],
        "Statistical & Non-Equilibrium Physics": [
            "statistical", "stochastic", "stochastic_advanced", "dissipative",
            "non_equilibrium", "reaction_diffusion", "turbulence"
        ],
        "Fluids, Soft Matter & Active Systems": [
            "fluid", "granular", "active_matter", "elasticity"
        ],
        "Solitons & Nonlinear Waves": [
            "continuum_solitons", "multi_scale_chaos"
        ],

        # ────────────────────────────────
        # APPLIED & INTERDISCIPLINARY
        # ────────────────────────────────
        "Biophysics & Life Sciences": [
            "biophysics", "neuroscience", "epidemiology", "public_health", "ecology"
        ],
        "Engineering & Technology": [
            "accelerator", "control_theory", "optics", "acoustics"
        ],
        "Earth & Environmental Systems": [
            "geophysics", "climate", "agriculture", "urban"
        ],

        # ────────────────────────────────
        # INFORMATION, COGNITION & SOCIETY
        # ────────────────────────────────
        "Information & Computation": [
            "quantum_info", "quantum_info_advanced", "symbolic", "optimization",
            "inference", "network_dynamics", "modern_extensions"
        ],
        "Cognitive & Psychological Dynamics": [
            "cognitive", "wellness", "dream", "neuroscience", "education"
        ],
        "Social & Cultural Systems": [
            "econophysics", "game_dynamics", "linguistics", "religion", "folklore",
            "digital_culture", "legal", "sports", "urban", "quant_finance"
        ],

        # ────────────────────────────────
        # CREATIVE & AESTHETIC DOMAINS
        # ────────────────────────────────
        "Aesthetic & Design Domains": [
            "art_music", "typography", "architecture", "perfumery", "fashion", 
            "cuisine", "culinary", "generative", "dance", "poetics"
        ],
        "Cultural & Symbolic Practices": [
            "ceremony", "mythopoetics", "folklore"
        ],

        # ────────────────────────────────
        # WELLNESS & LIFESTYLE
        # ────────────────────────────────
        "Health & Wellness": [
            "wellness", "dream", "gardening"
        ],

        # ────────────────────────────────
        # PURE & ADVANCED MATHEMATICS
        # ────────────────────────────────
        "Mathematical Structures": [
            "pure_math", "mathematical", "twistor", "tft", "symmetry_reduced",
            "exotic"
        ],

        # ────────────────────────────────
        # SPECULATIVE & FRONTIER DOMAINS
        # ────────────────────────────────
        "Metaphysical & Speculative": [
            "metaphysical", "exotic"
        ],
    }

    # Replace each subcategory with its actual Hamiltonian count, if present
    result = {}
    for super_cat, subcats in tree.items():
        filtered = {
            cat: category_counts[cat]
            for cat in subcats
            if cat in all_categories and category_counts[cat] > 0
        }
        if filtered:
            result[super_cat] = filtered

    return result

def export_latex_table(
    category: Optional[str] = None, 
    filename: str = 'hamiltonians.tex'
) -> None:
    """
    Exports a LaTeX table of Hamiltonians.
    
    Parameters
    ----------
    category : str, optional
        Category to export (all if None).
    filename : str
        Output file name.
    """
    import sympy as sp

    # List of Hamiltonians to export
    hamiltonians = list_hamiltonians(category=category) if category else list(CATALOG.keys())

    # LaTeX table header
    lines = [
        r"\begin{longtable}{|l|c|p{8cm}|}",
        r"\hline",
        r"\textbf{Name} & \textbf{Dim} & \textbf{Hamiltonian} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"\textbf{Name} & \textbf{Dim} & \textbf{Hamiltonian} \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
    ]

    # Add rows for each Hamiltonian
    for name in hamiltonians:
        info = CATALOG[name]
        H_latex = sp.latex(info['expr'])
        dim = info['dim']
        name_latex = name.replace('_', r'\_')
        lines.append(f"{name_latex} & {dim}D & ${H_latex}$" + r" \\" + "\n")
        lines.append(r"\hline")

    # End of the table
    lines.append(r"\end{longtable}")

    # Write to file
    try:
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Exported {len(hamiltonians)} Hamiltonians to {filename}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def get_dimensional_analysis(name: str) -> Dict[str, Any]:
    """
    Basic dimensional analysis of a Hamiltonian.
    
    Parameters
    ----------
    name : str
        Name of the Hamiltonian.
    Returns
    -------
    dict
        Information about structural properties.
    """
    H, vars, params, info = get_hamiltonian(name)
    terms = H.as_ordered_terms()
    analysis = {
        'name': name,
        'dimension': info['dim'],
        'num_terms': len(terms),
        'polynomial_degree': 0,
        'has_trigonometric': False,
        'has_exponential': False,
        'has_logarithm': False,
        'has_sqrt': False,
        'has_abs': False,
        'has_rational': False,
        'complexity_score': 0
    }
    H_str = str(H)
    analysis['has_trigonometric'] = any(f in H_str for f in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc'])
    analysis['has_exponential'] = 'exp' in H_str or '**' in H_str  # crude but effective
    analysis['has_logarithm'] = 'log' in H_str
    analysis['has_sqrt'] = 'sqrt' in H_str
    analysis['has_abs'] = 'Abs' in H_str or 'abs(' in H_str.lower()
    analysis['has_rational'] = any(op in H_str for op in ['/ ', '/(', '/x', '/y'])
    
    # Safely estimate polynomial degree
    for var in vars:
        for term in terms:
            if term.has(var):
                try:
                    deg = sp.degree(term, var)
                    if deg is not None and deg >= 0:
                        analysis['polynomial_degree'] = max(analysis['polynomial_degree'], int(deg))
                except (sp.PolynomialError, ValueError, TypeError, AttributeError):
                    # Non-polynomial term (e.g., log, sqrt, exp) — skip degree calculation
                    continue

    complexity = len(H_str)
    complexity += 10 * analysis['num_terms']
    complexity += 20 * int(analysis['has_trigonometric'])
    complexity += 20 * int(analysis['has_exponential'])
    complexity += 15 * int(analysis['has_logarithm'])
    complexity += 10 * int(analysis['has_sqrt'])
    complexity += 10 * int(analysis['has_abs'])
    analysis['complexity_score'] = complexity
    return analysis

def find_similar_hamiltonians(
    name: str, 
    top_n: int = 5
) -> List[Tuple[str, int]]:
    """
    Finds similar Hamiltonians by structural analysis.
    
    Parameters
    ----------
    name : str
        Name of the reference Hamiltonian.
    top_n : int
        Number of results to return.
    Returns
    -------
    list
        List of tuples (name, similarity score).
    """
    ref_analysis = get_dimensional_analysis(name)
    ref_info = CATALOG[name]
    similarities = []
    for other_name in CATALOG:
        if other_name == name:
            continue
        other_analysis = get_dimensional_analysis(other_name)
        other_info = CATALOG[other_name]
        score = 0
        if ref_info['dim'] == other_info['dim']:
            score += 30
        if ref_info['category'] == other_info['category']:
            score += 40
        if ref_analysis['has_trigonometric'] == other_analysis['has_trigonometric']:
            score += 10
        if ref_analysis['has_exponential'] == other_analysis['has_exponential']:
            score += 10
        if ref_analysis['has_logarithm'] == other_analysis['has_logarithm']:
            score += 10
        term_diff = abs(ref_analysis['num_terms'] - other_analysis['num_terms'])
        score += max(0, 10 - term_diff)
        similarities.append((other_name, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def validate_hamiltonians() -> Dict[str, List[Any]]:
    """
    Validates all Hamiltonians to detect common errors.
    
    Returns
    -------
    dict
        Validation report with warnings and errors.
    """
    report = {
        'valid': [],
        'warnings': [],
        'errors': [],
        'suspicious': []
    }
    for name, info in CATALOG.items():
        H = info['expr']
        H_str = str(H)
        issues = []
        if 'Derivative' in H_str and 'Derivative(x, x)' in H_str:
            issues.append("Contains Derivative(x,x) which equals 1")
        if not any(var in H_str for var in ['x', 'y', 'xi', 'eta']):
            issues.append("No dynamical variables found")
        if ('1/x' in H_str or '1/y' in H_str) and 'eps' not in H_str:
            issues.append("Division by coordinate without regularization")
        if 'sqrt' in H_str and '-' in H_str:
            issues.append("Potential sqrt of negative quantity")
        if 'log' in H_str and not 'Abs' in H_str and 'eps' not in H_str:
            issues.append("Logarithm without absolute value or regularization")
        if len(H_str) > 500:
            issues.append(f"Very complex expression (length: {len(H_str)})")
        if issues:
            report['warnings'].append({
                'name': name,
                'issues': issues,
                'expression': H_str[:100] + ' ...' if len(H_str) > 100 else H_str
            })
        else:
            report['valid'].append(name)
    return report

def batch_export_hamiltonians(
    output_dir: str = 'hamiltonians_export', 
    formats: List[str] = ['json', 'yaml', 'csv']
) -> None:
    """
    Exports the entire catalog in multiple formats.
    
    Parameters
    ----------
    output_dir : str
        Output directory.
    formats : list
        Desired formats: 'json', 'yaml', 'csv', 'markdown'.
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'json' in formats:
        catalog_json = {}
        for name, info in CATALOG.items():
            catalog_json[name] = {
                'expression': str(info['expr']),
                'dimension': info['dim'],
                'category': info['category'],
                'description': info['description']
            }
        with open(f'{output_dir}/catalog.json', 'w') as f:
            json.dump(catalog_json, f, indent=2)
        print(f"✓ Exported to {output_dir}/catalog.json")
    if 'yaml' in formats:
        try:
            import yaml
            with open(f'{output_dir}/catalog.yaml', 'w') as f:
                yaml.dump(catalog_json, f, default_flow_style=False)
            print(f"✓ Exported to {output_dir}/catalog.yaml")
        except ImportError:
            print("✗ YAML export requires PyYAML package")
    if 'csv' in formats:
        with open(f'{output_dir}/catalog.csv', 'w') as f:
            f.write("Name,Dimension,Category,Description\n")
            for name, info in CATALOG.items():
                desc = info['description'].replace(',', ';')
                f.write(f'{name},{info["dim"]},{info["category"]},"{desc}"\n')
        print(f"✓ Exported to {output_dir}/catalog.csv")
    if 'markdown' in formats:
        with open(f'{output_dir}/catalog.md', 'w') as f:
            f.write("# Hamiltonian Catalog\n\n")
            f.write(f"**Total Systems**: {len(CATALOG)}\n\n")
            for category in sorted(set(info['category'] for info in CATALOG.values())):
                f.write(f"\n## {category.replace('_', ' ').title()}\n\n")
                hamiltonians = [name for name, info in CATALOG.items() if info['category'] == category]
                for name in sorted(hamiltonians):
                    info = CATALOG[name]
                    f.write(f"### {name}\n")
                    f.write(f"- **Dimension**: {info['dim']}D\n")
                    f.write(f"- **Description**: {info['description']}\n")
                    f.write(f"- **Expression**: `{info['expr']}`\n\n")
        print(f"✓ Exported to {output_dir}/catalog.md")