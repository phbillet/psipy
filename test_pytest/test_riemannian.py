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
"""
import numpy as np
import pytest
from sympy import symbols, Matrix, simplify, sin, cos, log, sqrt, integrate, pi
from riemannian import *


# ============================================================================
# 1D Tests
# ============================================================================
def test_1d_flat_metric():
    x = symbols('x', real=True)
    metric = Metric(x**0, (x,))  # g = 1
    
    # Geometry
    assert metric.gauss_curvature() == 0
    assert metric.ricci_scalar() == 0
    
    # Geodesics
    traj = geodesic_solver(metric, (0.0,), (1.0,), (0, 5))
    assert np.allclose(traj['x'], np.linspace(0, 5, len(traj['x'])), atol=1e-3)
    
    # Volume
    vol = metric.riemannian_volume((0, 2), method='symbolic')
    assert vol == 2
    
    # Laplace-Beltrami
    lb = metric.laplace_beltrami_symbol()
    xi = symbols('xi', real=True)
    assert simplify(lb['principal'] - xi**2) == 0


def test_1d_hyperbolic_metric():
    x = symbols('x', positive=True)
    metric = Metric(1/x**2, (x,))  # Hyperbolic line

    # Christoffel: Γ = -1/x
    Gamma = metric.christoffel[0][0][0]
    assert simplify(Gamma + 1/x) == 0

    # Volume on [1, e]
    vol = metric.riemannian_volume((1, np.e), method='symbolic')
    assert simplify(vol - 1) == 0

    # Sturm-Liouville reduction
    sl = metric.sturm_liouville_reduce()
    assert simplify(sl['p'] - x) == 0
    assert simplify(sl['w'] - 1/x) == 0


def test_1d_hamiltonian_construction():
    x, p = symbols('x p', real=True)
    H = p**2 / (2 * x**2)  # Kinetic term
    metric = Metric.from_hamiltonian(H, (x,), (p,))
    assert simplify(metric.g_matrix[0,0] - x**2) == 0


def test_1d_hamiltonian_flow():
    x = symbols('x', positive=True)
    metric = Metric(x**2, (x,))
    res = geodesic_hamiltonian_flow(metric, (2.0,), (10.0,), (0, 2))
    # Energy should be conserved
    assert np.std(res['energy']) / np.mean(res['energy']) < 3e-3


# ============================================================================
# 2D Tests
# ============================================================================
def test_2d_euclidean():
    x, y = symbols('x y', real=True)
    metric = Metric(Matrix([[1, 0], [0, 1]]), (x, y))

    assert metric.gauss_curvature() == 0
    traj = geodesic_solver(metric,(0, 0), (1, 1), (0, 3))
    assert np.allclose(traj['x'], traj['y'], atol=1e-3)


def test_2d_polar():
    r, theta = symbols('r theta', positive=True, real=True)
    g = Matrix([[1, 0], [0, r**2]])
    metric = Metric(g, (r, theta))
    assert simplify(metric.gauss_curvature()) == 0
    
    lb = metric.laplace_beltrami_symbol()
    
    xi, eta = symbols('xi eta', real=True)
    expected = xi**2 + eta**2 / r**2

    assert simplify(lb['principal'] - expected) == 0


def test_2d_sphere():
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))

    K = simplify(metric.gauss_curvature())
    assert K == 1  # Unit sphere

    R = simplify(metric.ricci_scalar())
    assert R == 2


def test_2d_poincare_half_plane():
    x, y = symbols('x y', real=True)
    g = Matrix([[1/y**2, 0], [0, 1/y**2]])
    metric = Metric(g, (x, y))
    assert simplify(metric.gauss_curvature()) == -1


def test_2d_hamiltonian_construction():
    r, th = symbols('r th', positive=True)
    pr, pt = symbols('pr pt', real=True)
    H = (pr**2 + pt**2 / r**2) / 2
    metric = Metric.from_hamiltonian(H, (r, th), (pr, pt))
    expected = Matrix([[1, 0], [0, r**2]])
    diff = simplify(metric.g_matrix - expected)
    assert diff == Matrix([[0, 0], [0, 0]])


def test_2d_hodge_star():
    x, y = symbols('x y', real=True)
    g = Matrix([[4, 0], [0, 9]])
    metric = Metric(g, (x, y))

    # Test 2-form: *(12 dx∧dy) = 12 / sqrt(36) = 2
    star2 = hodge_star(metric, 2)
    assert simplify(star2(12) - 2) == 0

    # Test 0-form: *(1) = sqrt(36) dx∧dy = 6 dx∧dy → represented as 6
    star0 = hodge_star(metric, 0)
    assert simplify(star0(1) - 6) == 0


def test_2d_exponential_map_and_distance():
    x, y = symbols('x y', real=True)
    metric = Metric(Matrix([[1, 0], [0, 1]]), (x, y))
    p = (0.0, 0.0)
    v = (3.0, 4.0)
    q = exponential_map(metric, p, v, t=1.0)
    assert np.allclose(q, (3.0, 4.0), atol=1e-4)

    d = distance(metric, p, (3.0, 4.0), method='shooting')
    assert np.isclose(d, 5.0, rtol=1e-3)


def test_2d_gauss_bonnet():
    th, ph = symbols('th ph', real=True)
    g = Matrix([[1, 0], [0, sin(th)**2]])
    metric = Metric(g, (th, ph))
    # Sphere: χ = 2 → ∫ K dA = 4π
    domain = ((0.01, np.pi - 0.01), (0, 2 * np.pi))
    integral = metric.riemannian_volume(domain, method='numerical')  # dA = sin(th) dth dph → 4π
    # But Gauss-Bonnet: ∫ K dA = ∫ 1 * sin(th) dth dph = 4π
    K = metric.gauss_curvature()
    sqrt_g = metric.sqrt_det_g
    from scipy.integrate import dblquad
    integrand = lambda ph, th: float((K * sqrt_g).subs({th: th, ph: ph}))
    val, _ = dblquad(integrand, 0.01, np.pi - 0.01, 0, 2*np.pi)
    assert np.isclose(val, 4 * np.pi, rtol=1e-2)


# ============================================================================
# Mixed / Edge Cases
# ============================================================================
def test_dimension_dispatch():
    x = symbols('x', real=True)
    metric1d = Metric(x**2, (x,))
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


def test_backward_compatibility():
    x = symbols('x', real=True)
    m1 = Metric1D(x**2, x)
    m2 = Metric(x**2, (x,))
    assert simplify(m1.g_expr - m2.g_matrix[0,0]) == 0

    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    m3 = Metric2D(g, (x, y))
    m4 = Metric(g, (x, y))
    assert (m3.g_matrix - m4.g_matrix) == Matrix([[0, 0], [0, 0]])


# ============================================================================
# Run if executed directly
# ============================================================================
if __name__ == "__main__":
    print("🧪 Running unified riemannian test suite...\n")

    # 1D
    test_1d_flat_metric()
    test_1d_hyperbolic_metric()
    test_1d_hamiltonian_construction()
    test_1d_hamiltonian_flow()

    # 2D
    test_2d_euclidean()
    test_2d_polar()
    test_2d_sphere()
    test_2d_poincare_half_plane()
    test_2d_hamiltonian_construction()
    test_2d_hodge_star()
    test_2d_exponential_map_and_distance()
    test_2d_gauss_bonnet()

    # Mixed
    test_dimension_dispatch()
    test_backward_compatibility()

    print("✅ All tests passed!")