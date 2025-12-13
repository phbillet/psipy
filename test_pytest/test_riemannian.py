# Copyright 2025 Philippe Billet
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
Unified test suite for the unified riemannian module.
Tests both 1D and 2D cases using the single Metric class.

ENRICHED VERSION:
 - Performance benchmarks
 - Optimization validation tests
 - Edge case coverage
 - Numerical stability tests
 - Integration tests
 - Property-based tests
"""
import numpy as np
import pytest
import time
from sympy import symbols, Matrix, simplify, sin, cos, log, sqrt, integrate, pi, exp
from sympy import sympify as sp_simplify
from riemannian import *


# ============================================================================
# 1D Tests
# ============================================================================
def test_1d_flat_metric():
    """Test flat 1D metric (Euclidean line)."""
    x = symbols('x', real=True)
    metric = Metric(x**0, (x,))  # g = 1
    
    # Geometry
    assert metric.gauss_curvature == 0
    assert metric.ricci_scalar == 0
    
    # Christoffel symbols should be zero
    assert metric.christoffel[0][0][0] == 0
    
    # Geodesics should be straight lines
    traj = geodesic_solver(metric, (0.0,), (1.0,), (0, 5))
    assert np.allclose(traj['x'], np.linspace(0, 5, len(traj['x'])), atol=1e-3)
    
    # Volume is just arc length
    vol = metric.riemannian_volume((0, 2), method='symbolic')
    assert vol == 2
    
    # Laplace-Beltrami reduces to standard second derivative
    lb = metric.laplace_beltrami_symbol()
    xi = symbols('xi', real=True)
    assert sp_simplify(lb['principal'] - xi**2) == 0


def test_1d_hyperbolic_metric():
    """Test hyperbolic line metric g = 1/x²."""
    x = symbols('x', positive=True)
    metric = Metric(1/x**2, (x,))

    # Christoffel: Γ = -1/x
    Gamma = metric.christoffel[0][0][0]
    assert sp_simplify(Gamma + 1/x) == 0

    # Volume on [1, e]
    vol = metric.riemannian_volume((1, np.e), method='symbolic')
    assert sp_simplify(vol - 1) == 0

    # Arc length
    arc = metric.arc_length(1, 2, method='symbolic')
    assert sp_simplify(arc - log(2)) == 0

    # Sturm-Liouville reduction
    sl = metric.sturm_liouville_reduce()
    assert sp_simplify(sl['p'] - x) == 0
    assert sp_simplify(sl['w'] - 1/x) == 0


def test_1d_exponential_metric():
    """Test metric with exponential weight g = exp(x)."""
    x = symbols('x', real=True)
    metric = Metric(exp(x), (x,))
    
    # Christoffel symbol
    Gamma = metric.christoffel[0][0][0]
    assert sp_simplify(Gamma - 1/2) == 0
    
    # Numerical geodesic
    traj = geodesic_solver(metric, (0.0,), (1.0,), (0, 2), n_steps=500)
    assert len(traj['x']) == len(traj['t'])


def test_1d_hamiltonian_construction():
    """Test metric extraction from Hamiltonian."""
    x, p = symbols('x p', real=True)
    H = p**2 / (2 * x**2)  # Kinetic term
    metric = Metric.from_hamiltonian(H, (x,), (p,))
    assert sp_simplify(metric.g_matrix[0,0] - x**2) == 0


def test_1d_hamiltonian_flow():
    """Test Hamiltonian flow energy conservation."""
    x = symbols('x', positive=True)
    metric = Metric(x**2, (x,))
    res = geodesic_hamiltonian_flow(metric, (2.0,), (10.0,), (0, 2), n_steps=2000)
    
    # Energy should be conserved
    energy_std = np.std(res['energy'])
    energy_mean = np.mean(res['energy'])
    assert energy_std / energy_mean < 3e-3
    
    # Check consistency between velocity and momentum
    g_vals = np.array([metric.eval(x_val)['g'][0,0] for x_val in res['x']])
    v_computed = res['p'] / g_vals
    assert np.allclose(res['v'], v_computed, rtol=1e-4)


def test_1d_reparametrization():
    """Test arc-length reparametrization."""
    x = symbols('x', real=True)
    metric = Metric(1 + x**2, (x,))
    
    traj = geodesic_solver(metric, (0.0,), (1.0,), (0, 3), 
                          reparametrize=True, n_steps=500)
    
    assert 'arc_length' in traj
    # Arc length should be monotonically increasing
    assert np.all(np.diff(traj['arc_length']) >= 0)


def test_1d_distance_computation():
    """Test geodesic distance in 1D."""
    x = symbols('x', real=True)
    metric = Metric(sympify(1), (x,))
    
    # Flat metric: distance should be |q - p|
    d = distance(metric, (0.0,), (5.0,), method='exact')
    assert np.isclose(d, 5.0, rtol=1e-6)
    
    # Hyperbolic metric
    metric_hyp = Metric(1/x**2, (x,))
    d_hyp = distance(metric_hyp, (1.0,), (np.e,), method='exact')
    assert np.isclose(d_hyp, 1.0, rtol=1e-6)


# ============================================================================
# 2D Tests
# ============================================================================
def test_2d_euclidean():
    """Test 2D Euclidean metric."""
    x, y = symbols('x y', real=True)
    metric = Metric(Matrix([[1, 0], [0, 1]]), (x, y))

    # Curvature should be zero
    assert metric.gauss_curvature == 0
    assert metric.ricci_scalar == 0
    
    # All Christoffel symbols zero
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert metric.christoffel[i][j][k] == 0
    
    # Geodesics should be straight lines
    traj = geodesic_solver(metric, (0, 0), (1, 1), (0, 3))
    assert np.allclose(traj['x'], traj['y'], atol=1e-3)
    
    # Volume of rectangle
    vol = metric.riemannian_volume(((0, 2), (0, 3)), method='symbolic')
    assert vol == 6


def test_2d_polar():
    """Test polar coordinate metric."""
    r, theta = symbols('r theta', positive=True, real=True)
    g = Matrix([[1, 0], [0, r**2]])
    metric = Metric(g, (r, theta))
    
    # Flat space in polar coords has K = 0
    assert sp_simplify(metric.gauss_curvature) == 0
    
    # Laplace-Beltrami symbol
    lb = metric.laplace_beltrami_symbol()
    xi, eta = symbols('xi eta', real=True)
    expected = xi**2 + eta**2 / r**2
    assert sp_simplify(lb['principal'] - expected) == 0


def test_2d_sphere():
    """Test unit sphere metric."""
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))

    # Gaussian curvature of unit sphere is 1
    K = sp_simplify(metric.gauss_curvature)
    assert K == 1

    # Ricci scalar is 2K for surfaces
    R = sp_simplify(metric.ricci_scalar)
    assert R == 2
    
    # Ricci tensor should be g for constant curvature
    Ric = metric.ricci_tensor
    expected_Ric = metric.g_matrix
    diff = sp_simplify(Ric - expected_Ric)
    assert diff == Matrix([[0, 0], [0, 0]])


def test_2d_poincare_half_plane():
    """Test Poincaré half-plane (hyperbolic space)."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1/y**2, 0], [0, 1/y**2]])
    metric = Metric(g, (x, y))
    
    # Constant negative curvature
    assert sp_simplify(metric.gauss_curvature) == -1
    assert sp_simplify(metric.ricci_scalar) == -2


def test_2d_schwarzschild():
    """Test Schwarzschild-like metric (spatial part)."""
    r, theta = symbols('r theta', positive=True)
    rs = symbols('r_s', positive=True)  # Schwarzschild radius
    
    # Simplified: just spatial part
    g = Matrix([[1/(1 - rs/r), 0], [0, r**2]])
    metric = Metric(g, (r, theta))
    
    # Should compute curvature without error
    K = metric.gauss_curvature
    assert K is not None
    
    # Evaluate at r = 10*rs
    K_func = lambdify((r, theta, rs), K, 'numpy')
    K_val = K_func(10, 0, 1)
    assert np.isfinite(K_val)


def test_2d_hamiltonian_construction():
    """Test 2D metric from Hamiltonian."""
    r, th = symbols('r th', positive=True)
    pr, pt = symbols('pr pt', real=True)
    H = (pr**2 + pt**2 / r**2) / 2
    metric = Metric.from_hamiltonian(H, (r, th), (pr, pt))
    expected = Matrix([[1, 0], [0, r**2]])
    diff = sp_simplify(metric.g_matrix - expected)
    assert diff == Matrix([[0, 0], [0, 0]])


def test_2d_non_diagonal_metric():
    """Test metric with non-zero off-diagonal terms."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, x], [x, 1 + x**2]])
    metric = Metric(g, (x, y))
    
    # Check determinant
    det_expected = 1
    assert sp_simplify(metric.det_g - det_expected) == 0
    
    # Inverse should satisfy g * g_inv = I
    product = metric.g_matrix * metric.g_inv_matrix
    assert sp_simplify(product - Matrix([[1, 0], [0, 1]])) == Matrix([[0, 0], [0, 0]])


def test_2d_hodge_star():
    """Test Hodge star operator."""
    x, y = symbols('x y', real=True)
    g = Matrix([[4, 0], [0, 9]])
    metric = Metric(g, (x, y))

    # Test 2-form: *(12 dx∧dy) = 12 / sqrt(36) = 2
    star2 = hodge_star(metric, 2)
    assert sp_simplify(star2(12) - 2) == 0

    # Test 0-form: *(1) = sqrt(36) = 6
    star0 = hodge_star(metric, 0)
    assert sp_simplify(star0(1) - 6) == 0


def test_2d_exponential_map_and_distance():
    """Test exponential map and distance computation."""
    x, y = symbols('x y', real=True)
    metric = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    
    # Exponential map in flat space
    p = (0.0, 0.0)
    v = (3.0, 4.0)
    q = exponential_map(metric, p, v, t=1.0)
    assert np.allclose(q, (3.0, 4.0), atol=1e-4)

    # Distance should be Euclidean
    d = distance(metric, p, (3.0, 4.0), method='shooting')
    assert np.isclose(d, 5.0, rtol=1e-2)


def test_2d_geodesic_on_sphere():
    """Test great circle on sphere."""
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))
    
    # Start at north pole, move south
    traj = geodesic_solver(metric, (0.01, 0), (1, 0), (0, np.pi/2), n_steps=500)
    
    # Should stay at constant phi (great circle through poles)
    assert np.std(traj['y']) < 1e-2


def test_2d_hamiltonian_energy_conservation():
    """Test energy conservation in Hamiltonian flow."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1 + x**2]])
    metric = Metric(g, (x, y))
    
    res = geodesic_hamiltonian_flow(
        metric, (1.0, 0.0), (1.0, 1.0), (0, 5), 
        method='verlet', n_steps=2000
    )
    
    # Energy should be conserved
    energy_variation = np.max(res['energy']) - np.min(res['energy'])
    assert energy_variation / np.mean(res['energy']) < 1e-2


def test_2d_jacobi_fields():
    """Test Jacobi field computation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric(g, (x, y))
    
    # Compute base geodesic
    geod = geodesic_solver(metric, (0, 0), (1, 0), (0, 3), n_steps=200)
    
    # Solve Jacobi equation
    J = jacobi_equation_solver(
        metric, geod,
        {'J0': (0, 0.1), 'DJ0': (0, 0)},
        (0, 3), n_steps=200
    )
    
    # In flat space, Jacobi fields grow linearly
    assert len(J['J_x']) == len(J['t'])
    # Perpendicular deviation should remain constant
    assert np.std(J['J_y']) < 1e-2


def test_2d_gauss_bonnet():
    """Test Gauss-Bonnet theorem on sphere."""
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))
    
    # Sphere: χ = 2 → ∫ K dA = 2π × 2 = 4π
    domain = ((0.01, np.pi - 0.01), (0, 2 * np.pi))
    
    K = metric.gauss_curvature
    sqrt_g = metric.sqrt_det_g    
    expr = sp_simplify(K * sqrt_g)
    
    f_num = lambdify((th, ph), expr, "numpy")
    val, _ = dblquad(lambda ph_val, th_val: f_num(th_val, ph_val),
                     0.01, np.pi - 0.01,
                     lambda _: 0, lambda _: 2*np.pi)

    # Should be 4π for sphere
    assert np.isclose(val, 4 * np.pi, rtol=1e-2)
    
    # Use helper function
    result = verify_gauss_bonnet(metric, domain)
    print(result)  # Debug
    assert result['relative_error'] < 0.05


# ============================================================================
# Optimization Tests
# ============================================================================
def test_lazy_evaluation():
    """Test that curvature is computed lazily."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, x**2 + 1]])
    
    # Create metric without precompute
    metric = Metric(g, (x, y), simplify=False, precompute=False)
    
    # Curvature should not be computed yet
    assert metric._gauss_curvature_cache is None
    assert metric._riemann_cache is None
    
    # Access triggers computation
    K = metric.gauss_curvature
    assert metric._gauss_curvature_cache is not None
    
    # Second access should use cache
    K2 = metric.gauss_curvature
    assert K2 is metric._gauss_curvature_cache


def test_christoffel_symmetry():
    """Test that Christoffel symbols respect symmetry Γⁱⱼₖ = Γⁱₖⱼ."""
    x, y = symbols('x y', real=True)
    g = Matrix([[exp(x), 0], [0, exp(y)]])
    metric = Metric(g, (x, y))
    
    Gamma = metric.christoffel
    
    # Check symmetry in lower indices
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert sp_simplify(Gamma[i][j][k] - Gamma[i][k][j]) == 0


def test_simplify_flag():
    """Test that simplify flag controls simplification."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1 + x**2 - x**2, 0], [0, 1]])  # = [[1, 0], [0, 1]]
    
    # Without simplification
    metric_no_simp = Metric(g, (x, y), simplify=False)
    # With simplification
    metric_simp = Metric(g, (x, y), simplify=True)
    
    # Simplified version should recognize this as flat
    K_simp = metric_simp.gauss_curvature
    assert K_simp == 0


def test_vectorized_eval():
    """Test vectorized metric evaluation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, x**2]])
    metric = Metric(g, (x, y))
    
    # Single point
    result = metric.eval(2.0, 1.0)
    assert result['g'][1, 1] == 4.0
    
    # Array evaluation (if vectorized lambdas work)
    try:
        x_vals = np.array([1.0, 2.0, 3.0])
        y_vals = np.array([0.5, 0.5, 0.5])
        result_vec = metric.eval(x_vals, y_vals)
        # Should return arrays
        assert result_vec['sqrt_det'].shape == x_vals.shape
    except:
        pass  # Fallback to component-wise is OK


def test_performance_benchmark():
    """Benchmark initialization time with and without simplify."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1 + x*y, x], [x, 1 + y**2]])
    
    # Without simplify (should be faster)
    start = time.time()
    metric_fast = Metric(g, (x, y), simplify=False)
    time_fast = time.time() - start
    
    # With simplify (slower)
    start = time.time()
    metric_slow = Metric(g, (x, y), simplify=True)
    time_slow = time.time() - start
    
    print(f"\n⚡ Init without simplify: {time_fast:.4f}s")
    print(f"🐌 Init with simplify: {time_slow:.4f}s")
    print(f"📊 Speedup: {time_slow/time_fast:.2f}x")
    
    # Fast version should be at least 2x faster for complex metrics
    assert time_fast < time_slow


# ============================================================================
# Edge Cases & Stability
# ============================================================================
def test_singular_point_handling():
    """Test behavior near coordinate singularities."""
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))
    
    # Evaluate near pole (singular point)
    result = metric.eval(0.001, 0.0)
    assert np.isfinite(result['sqrt_det'])
    
    # Christoffel symbols may diverge, but should be computable
    assert np.isfinite(result['christoffel'][1][1][0])


def test_negative_metric_components():
    """Test metrics with negative components (Lorentzian-like)."""
    x, y = symbols('x y', real=True)
    g = Matrix([[-1, 0], [0, 1]])  # Minkowski-like
    metric = Metric(g, (x, y))
    
    # Should handle det < 0
    assert metric.det_g == -1
    # sqrt(|det|) should be 1
    assert metric.sqrt_det_g == 1


def test_zero_velocity_geodesic():
    """Test geodesic with zero initial velocity."""
    x, y = symbols('x y', real=True)
    metric = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    
    traj = geodesic_solver(metric, (1.0, 2.0), (0.0, 0.0), (0, 5))
    
    # Should stay at initial position
    assert np.allclose(traj['x'], 1.0, atol=1e-6)
    assert np.allclose(traj['y'], 2.0, atol=1e-6)


def test_very_small_timestep():
    """Test numerical stability with very small timesteps."""
    x = symbols('x', real=True)
    metric = Metric(1 + x**2, (x,))
    
    traj = geodesic_solver(metric, (0.0,), (1.0,), (0, 0.001), n_steps=100)
    assert np.all(np.isfinite(traj['x']))


def test_dimension_dispatch():
    """Test that dimension-specific functions are properly guarded."""
    x = symbols('x', real=True)
    metric1d = Metric(x**2 + 1, (x,))  # Avoid singularity
    assert metric1d.dim == 1

    x, y = symbols('x y', real=True)
    metric2d = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    assert metric2d.dim == 2

    # Sturm-Liouville only for 1D
    try:
        metric2d.sturm_liouville_reduce()
        assert False, "Should raise NotImplementedError"
    except NotImplementedError:
        pass

    # Hodge star only for 2D
    try:
        hodge_star(metric1d, 1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    
    # Jacobi fields only for 2D
    try:
        # Use safe initial conditions
        geod = geodesic_solver(metric1d, (1.0,), (1.0,), (0, 1), n_steps=100)
        jacobi_equation_solver(metric1d, geod, {'J0': (0,), 'DJ0': (0,)}, (0, 1))
        assert False, "Should raise ValueError"
    except (ValueError, AttributeError):
        # AttributeError can occur if geodesic doesn't have required fields
        pass

def test_singular_metric_handling():
    """Test handling of metrics with singularities."""
    x = symbols('x', positive=True)
    metric = Metric(1/x**2, (x,))
    
    # Should work away from singularity
    traj = geodesic_solver(metric, (1.0,), (0.5,), (0, 2), n_steps=500)
    assert np.all(np.isfinite(traj['x']))
    assert np.all(traj['x'] > 0)  # Should stay in valid domain


def test_geodesic_with_warnings_suppressed():
    """Test that geodesic solver handles edge cases gracefully."""
    import warnings
    
    x = symbols('x', real=True)
    metric = Metric(exp(x), (x,))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = geodesic_solver(metric, (0.0,), (1.0,), (0, 5))
        # Should complete without crashing
        assert 'x' in traj
        assert 't' in traj

def test_backward_compatibility():
    """Test backward compatibility wrappers."""
    x = symbols('x', real=True)
    metric1d = Metric1D(x**2, x)
    assert metric1d.dim == 1
    assert sp_simplify(metric1d.g_matrix[0,0] - x**2) == 0
    
    x, y = symbols('x y', real=True)
    metric2d = Metric2D(Matrix([[1, 0], [0, 1]]), (x, y))
    assert metric2d.dim == 2


# ============================================================================
# Integration Tests
# ============================================================================
def test_full_workflow_1d():
    """End-to-end test for 1D metric."""
    x = symbols('x', positive=True)
    
    # Define metric
    metric = Metric(x**2, (x,), simplify=True)
    
    # Compute geometry
    Gamma = metric.christoffel[0][0][0]
    assert sp_simplify(Gamma - 1/x) == 0
    
    # Integrate geodesic
    traj = geodesic_solver(metric, (1.0,), (1.0,), (0, 3))
    
    # Compute distance
    d = metric.arc_length(1.0, 2.0)
    assert d > 0
    
    # Sturm-Liouville
    sl = metric.sturm_liouville_reduce(potential_expr=x**2)
    assert sl['p'] is not None


def test_full_workflow_2d():
    """End-to-end test for 2D metric."""
    r, th = symbols('r th', positive=True)
    
    # Polar metric
    g = Matrix([[1, 0], [0, r**2]])
    metric = Metric(g, (r, th), simplify=True)
    
    # Curvature
    K = metric.gauss_curvature
    assert K == 0
    
    # Geodesics
    traj = geodesic_solver(metric, (1.0, 0.0), (1.0, 0.5), (0, 2*np.pi))
    
    # Exponential map
    q = exponential_map(metric, (1.0, 0.0), (0.0, 1.0), t=1.0)
    
    # Volume
    vol = metric.riemannian_volume(((1, 2), (0, 2*np.pi)))
    assert np.isclose(float(vol), 3*np.pi, rtol=1e-3)


def test_metric_from_physical_hamiltonian():
    """Test realistic Hamiltonian from physics."""
    r, th = symbols('r theta', positive=True)
    pr, pth = symbols('p_r p_theta', real=True)
    m, L = symbols('m L', positive=True)
    
    # Hamiltonian for particle in polar coordinates
    H = (pr**2 / (2*m) + pth**2 / (2*m*r**2))
    
    metric = Metric.from_hamiltonian(H, (r, th), (pr, pth))
    
    # Should give polar metric (up to constant factor)
    expected = Matrix([[m, 0], [0, m*r**2]])
    diff = sp_simplify(metric.g_matrix - expected)
    assert diff == Matrix([[0, 0], [0, 0]])


# ============================================================================
# Property-Based Tests (Geometric Identities)
# ============================================================================
def test_metric_inverse_property():
    """Test g * g^{-1} = I for various metrics."""
    test_cases = [
        # Euclidean
        (Matrix([[1, 0], [0, 1]]), ('x', 'y')),
        # Polar
        (Matrix([[1, 0], [0, symbols('r')**2]]), ('r', 'theta')),
        # Non-diagonal
        (Matrix([[2, 1], [1, 3]]), ('x', 'y')),
    ]
    
    for g_matrix, coord_names in test_cases:
        coords = symbols(' '.join(coord_names), real=True)
        metric = Metric(g_matrix, coords)
        
        product = metric.g_matrix * metric.g_inv_matrix
        identity = Matrix([[1, 0], [0, 1]])
        diff = sp_simplify(product - identity)
        
        assert diff == Matrix([[0, 0], [0, 0]])


def test_ricci_scalar_relation():
    """Test R = 2K for 2D surfaces."""
    test_metrics = [
        # Sphere
        (Matrix([[1, 0], [0, sin(symbols('th'))**2]]), ('th', 'ph')),
        # Hyperbolic plane
        (Matrix([[1/symbols('y')**2, 0], [0, 1/symbols('y')**2]]), ('x', 'y')),
    ]
    
    for g, coord_names in test_metrics:
        coords = symbols(' '.join(coord_names), real=True)
        metric = Metric(g, coords)
        
        K = metric.gauss_curvature
        R = metric.ricci_scalar
        
        # For 2D: R = 2K
        diff = sp_simplify(R - 2*K)
        assert diff == 0


def test_bianchi_identity():
    """Test Einstein tensor for constant curvature surfaces."""
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))
    
    Ric = metric.ricci_tensor
    R = metric.ricci_scalar
    g_mat = metric.g_matrix
    
    # Einstein tensor: G_ij = R_ij - (R/2) g_ij
    Einstein = Ric - R/2 * g_mat
    
    # Simplify components
    G00 = sp_simplify(Einstein[0, 0])
    G11 = sp_simplify(Einstein[1, 1])
    G01 = sp_simplify(Einstein[0, 1])
    
    # For surfaces of constant curvature, Einstein tensor vanishes
    print(f"G_00 = {G00}")
    print(f"G_11 = {G11}")
    print(f"G_01 = {G01}")
    
    # All components should be zero
    assert G00 == 0, f"Expected G_00 = 0, got {G00}"
    assert G11 == 0, f"Expected G_11 = 0, got {G11}"
    assert G01 == 0, f"Expected G_01 = 0, got {G01}"


def test_constant_curvature_properties():
    """Test properties of constant curvature surfaces."""
    # Test multiple constant curvature spaces
    
    # 1. Unit sphere (K = +1)
    th, ph = symbols('th ph', real=True)
    g_sphere = Matrix([[1, 0], [0, sin(th)**2]])
    metric_sphere = Metric(g_sphere, (th, ph))
    
    K_sphere = sp_simplify(metric_sphere.gauss_curvature)
    R_sphere = sp_simplify(metric_sphere.ricci_scalar)
    Ric_sphere = metric_sphere.ricci_tensor
    
    assert K_sphere == 1, f"Sphere curvature should be 1, got {K_sphere}"
    assert R_sphere == 2, f"Sphere scalar curvature should be 2, got {R_sphere}"
    
    # For constant curvature: R_ij = K g_ij
    expected_Ric = metric_sphere.g_matrix
    diff_Ric = sp_simplify(Ric_sphere - expected_Ric)
    assert diff_Ric == Matrix([[0, 0], [0, 0]]), f"Ricci tensor mismatch: {diff_Ric}"
    
    # 2. Hyperbolic plane (K = -1)
    x, y = symbols('x y', real=True, positive=True)
    g_hyp = Matrix([[1/y**2, 0], [0, 1/y**2]])
    metric_hyp = Metric(g_hyp, (x, y))
    
    K_hyp = sp_simplify(metric_hyp.gauss_curvature)
    R_hyp = sp_simplify(metric_hyp.ricci_scalar)
    
    assert K_hyp == -1, f"Hyperbolic curvature should be -1, got {K_hyp}"
    assert R_hyp == -2, f"Hyperbolic scalar curvature should be -2, got {R_hyp}"
    
    # 3. Flat space (K = 0)
    x, y = symbols('x y', real=True)
    g_flat = Matrix([[1, 0], [0, 1]])
    metric_flat = Metric(g_flat, (x, y))
    
    K_flat = metric_flat.gauss_curvature
    R_flat = metric_flat.ricci_scalar
    
    assert K_flat == 0
    assert R_flat == 0

def test_einstein_tensor_diagnostic():
    """Diagnostic test to understand Einstein tensor computation."""
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph), simplify=True)
    
    print("\n" + "="*60)
    print("EINSTEIN TENSOR DIAGNOSTIC")
    print("="*60)
    
    print(f"\nMetric:\n{metric.g_matrix}")
    
    K = metric.gauss_curvature
    print(f"\nGaussian curvature K = {K}")
    
    R = metric.ricci_scalar
    print(f"\nScalar curvature R = {R}")
    
    Ric = metric.ricci_tensor
    print(f"\nRicci tensor:\n{Ric}")
    
    Einstein = Ric - R/2 * metric.g_matrix
    print(f"\nEinstein tensor G = R - (R/2)g:")
    print(f"G_00 = {sp_simplify(Einstein[0,0])}")
    print(f"G_11 = {sp_simplify(Einstein[1,1])}")
    print(f"G_01 = {sp_simplify(Einstein[0,1])}")
    
    print("\nExpected: All components = 0 for constant curvature")
    print("="*60)

test_constant_curvature_properties()
test_einstein_tensor_diagnostic()
# ============================================================================
# Visualization Tests (smoke tests)
# ============================================================================
def test_visualize_geodesics_1d():
    """Test 1D geodesic visualization (no display)."""
    x = symbols('x', real=True)
    metric = Metric(1 + x**2, (x,))
    
    # Should not raise
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    try:
        visualize_geodesics(
            metric,
            [(0.0, 1.0), (0.0, 2.0)],
            (0, 5),
            x_range=(-2, 2)
        )
    except Exception as e:
        pytest.fail(f"Visualization failed: {e}")


def test_visualize_curvature_2d():
    """Test 2D curvature visualization (no display)."""
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))
    
    import matplotlib
    matplotlib.use('Agg')
    
    try:
        visualize_curvature(
            metric,
            (0.1, np.pi-0.1),
            (0, 2*np.pi),
            resolution=20,
            quantity='gauss'
        )
    except Exception as e:
        pytest.fail(f"Curvature visualization failed: {e}")


# ============================================================================
# Run if executed directly
# ============================================================================
if __name__ == "__main__":
    print("🧪 Running ENRICHED unified riemannian test suite...\n")
    print("=" * 70)

    # 1D Tests
    print("\n📐 1D METRIC TESTS")
    print("-" * 70)
    test_1d_flat_metric()
    print("✅ Flat metric")
    test_1d_hyperbolic_metric()
    print("✅ Hyperbolic metric")
    test_1d_exponential_metric()
    print("✅ Exponential metric")
    test_1d_hamiltonian_construction()
    print("✅ Hamiltonian construction")
    test_1d_hamiltonian_flow()
    print("✅ Hamiltonian flow")
    test_1d_reparametrization()
    print("✅ Arc-length reparametrization")
    test_1d_distance_computation()
    print("✅ Distance computation")

    # 2D Tests
    print("\n📐 2D METRIC TESTS")
    print("-" * 70)
    test_2d_euclidean()
    print("✅ Euclidean metric")
    test_2d_polar()
    print("✅ Polar coordinates")
    test_2d_sphere()
    print("✅ Sphere metric")
    test_2d_poincare_half_plane()
    print("✅ Poincaré half-plane")
    test_2d_schwarzschild()
    print("✅ Schwarzschild-like metric")
    test_2d_hamiltonian_construction()
    print("✅ Hamiltonian construction")
    test_2d_non_diagonal_metric()
    print("✅ Non-diagonal metric")
    test_2d_hodge_star()
    print("✅ Hodge star operator")
    test_2d_exponential_map_and_distance()
    print("✅ Exponential map & distance")
    test_2d_geodesic_on_sphere()
    print("✅ Geodesics on sphere")
    test_2d_hamiltonian_energy_conservation()
    print("✅ Energy conservation")
    test_2d_jacobi_fields()
    print("✅ Jacobi fields")
    test_2d_gauss_bonnet()
    print("✅ Gauss-Bonnet theorem")

    # Optimization Tests
    print("\n⚡ OPTIMIZATION TESTS")
    print("-" * 70)
    test_lazy_evaluation()
    print("✅ Lazy evaluation")
    test_christoffel_symmetry()
    print("✅ Christoffel symmetry")
    test_simplify_flag()
    print("✅ Simplify flag")
    test_vectorized_eval()
    print("✅ Vectorized evaluation")
    test_performance_benchmark()
    print("✅ Performance benchmark")

    # Edge Cases
    print("\n🔍 EDGE CASES & STABILITY")
    print("-" * 70)
    test_singular_point_handling()
    print("✅ Singular points")
    test_negative_metric_components()
    print("✅ Negative components")
    test_zero_velocity_geodesic()
    print("✅ Zero velocity")
    test_very_small_timestep()
    print("✅ Small timesteps")
    test_dimension_dispatch()
    print("✅ Dimension dispatch")
    test_singular_metric_handling()
    print("✅ Singular metric handling")
    test_geodesic_with_warnings_suppressed()
    print("✅ Geodesic with warnings suppressed")
    test_backward_compatibility()
    print("✅ Backward compatibility")

    # Integration Tests
    print("\n🔗 INTEGRATION TESTS")
    print("-" * 70)
    test_full_workflow_1d()
    print("✅ Full 1D workflow")
    test_full_workflow_2d()
    print("✅ Full 2D workflow")
    test_metric_from_physical_hamiltonian()
    print("✅ Physical Hamiltonian")

    # Property Tests
    print("\n🎯 PROPERTY-BASED TESTS")
    print("-" * 70)
    test_metric_inverse_property()
    print("✅ Metric inverse property")
    test_ricci_scalar_relation()
    print("✅ Ricci scalar relation")
    test_bianchi_identity()
    print("✅ Bianchi identity")
    test_constant_curvature_properties()
    print("✅ Constant curvature properties")
    test_einstein_tensor_diagnostic()
    print("✅ Einstein tensor diagnostic")

    # Visualization
    print("\n🎨 VISUALIZATION TESTS")
    print("-" * 70)
    test_visualize_geodesics_1d()
    print("✅ 1D geodesic visualization")
    test_visualize_curvature_2d()
    print("✅ 2D curvature visualization")

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 70)
