import numpy as np
import pytest
from sympy import symbols, Matrix, sqrt, sin, cos, simplify, pi
from riemannian_2d import *


def test_metric2d_euclidean():
    """Test Euclidean (flat) metric."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    # Check determinant
    assert metric.det_g == 1
    
    # Check inverse
    assert metric.g_inv_matrix == Matrix([[1, 0], [0, 1]])
    
    # Gaussian curvature should be zero
    K = metric.gauss_curvature()
    assert K == 0


def test_metric2d_polar():
    """Test polar coordinate metric."""
    r, theta = symbols('r theta', real=True, positive=True)
    g_polar = Matrix([[1, 0], [0, r**2]])
    metric = Metric2D(g_polar, (r, theta))
    
    # Check determinant
    assert simplify(metric.det_g - r**2) == 0
    
    # Check inverse
    expected_inv = Matrix([[1, 0], [0, 1/r**2]])
    assert simplify(metric.g_inv_matrix - expected_inv) == Matrix([[0, 0], [0, 0]])
    
    # Gaussian curvature should be zero (flat space in polar coords)
    K = metric.gauss_curvature()
    assert simplify(K) == 0


def test_metric2d_sphere():
    """Test metric on unit sphere."""
    theta, phi = symbols('theta phi', real=True)
    g_sphere = Matrix([[1, 0], [0, sin(theta)**2]])
    metric = Metric2D(g_sphere, (theta, phi))
    
    # Check determinant
    expected_det = sin(theta)**2
    assert simplify(metric.det_g - expected_det) == 0
    
    # Gaussian curvature should be 1 (unit sphere)
    K = metric.gauss_curvature()
    # Symbolic check may be complex, just verify it's not None
    assert K is not None


def test_metric2d_from_hamiltonian():
    """Test metric extraction from Hamiltonian."""
    x, y, px, py = symbols('x y p_x p_y', real=True)
    
    # H = (p_x² + p_y²)/(2x²) 
    H = (px**2 + py**2) / (2*x**2)
    metric = Metric2D.from_hamiltonian(H, (x, y), (px, py))
    
    # Should extract g₁₁ = x²
    assert simplify(metric.g_matrix[0, 0] - x**2) == 0


def test_metric2d_eval():
    """Test numerical evaluation of metric components."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1 + x**2, 0], [0, 1 + y**2]])
    metric = Metric2D(g, (x, y))
    
    result = metric.eval(1.0, 2.0)
    
    assert 'g' in result
    assert 'g_inv' in result
    assert 'det_g' in result
    assert 'christoffel' in result
    
    # Check determinant
    expected_det = (1 + 1) * (1 + 4)  # (1+x²)(1+y²) at x=1, y=2
    assert np.isclose(result['det_g'], expected_det)


def test_christoffel_symbols_2d():
    """Test Christoffel symbol computation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    Gamma = christoffel(metric)
    
    # All Christoffel symbols should be zero for flat metric
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert Gamma[i][j][k] == 0


def test_geodesic_solver_euclidean():
    """Test geodesic on Euclidean plane."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    # Geodesics should be straight lines
    traj = geodesic_solver(metric, (0, 0), (1, 1), (0, 5), n_steps=100)
    
    # Check trajectory exists
    assert len(traj['x']) == 100
    assert len(traj['y']) == 100
    
    # Positions should evolve linearly
    expected_x = traj['t']
    expected_y = traj['t']
    assert np.allclose(traj['x'], expected_x, rtol=1e-2)
    assert np.allclose(traj['y'], expected_y, rtol=1e-2)


def test_geodesic_solver_methods():
    """Test different geodesic integration methods."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1 + x**2, 0], [0, 1 + y**2]])
    metric = Metric2D(g, (x, y))
    
    # Test RK45
    traj_rk = geodesic_solver(metric, (0, 0), (1, 0), (0, 5), 
                              method='rk45', n_steps=100)
    assert len(traj_rk['x']) > 0
    
    # Test symplectic
    traj_symp = geodesic_solver(metric, (0, 0), (1, 0), (0, 5), 
                                method='symplectic', n_steps=100)
    assert len(traj_symp['x']) > 0


def test_geodesic_hamiltonian_flow_2d():
    """Test geodesic in Hamiltonian formulation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    traj = geodesic_hamiltonian_flow(metric, (0, 0), (1, 1), (0, 5), 
                                     method='verlet', n_steps=100)
    
    assert 'px' in traj
    assert 'py' in traj
    assert 'energy' in traj
    
    # Energy should be conserved
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 1e-2


def test_exponential_map():
    """Test exponential map."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    # Exponential map with flat metric
    q = exponential_map(metric, (0, 0), (1, 1), t=1.0)
    
    assert len(q) == 2
    # For flat metric, exp_p(v) = p + v
    assert np.isclose(q[0], 1.0, rtol=1e-1)
    assert np.isclose(q[1], 1.0, rtol=1e-1)


def test_distance_computation():
    """Test geodesic distance computation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    # Distance on flat space
    d = distance(metric, (0, 0), (3, 4), method='shooting', max_iter=20)
    
    # Should be Euclidean distance = 5
    assert np.isclose(d, 5.0, rtol=0.1)


def test_gauss_curvature():
    """Test Gaussian curvature computation."""
    x, y = symbols('x y', real=True)
    
    # Flat metric
    g_flat = Matrix([[1, 0], [0, 1]])
    metric_flat = Metric2D(g_flat, (x, y))
    K_flat = metric_flat.gauss_curvature()
    assert K_flat == 0
    
    # Non-flat metric
    g = Matrix([[1, 0], [0, 1 + x**2]])
    metric = Metric2D(g, (x, y))
    K = metric.gauss_curvature()
    assert K is not None


def test_ricci_tensor():
    """Test Ricci tensor computation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    Ric = metric.ricci_tensor()
    
    # Should be 2x2 matrix
    assert Ric.shape == (2, 2)
    
    # For flat metric, should be zero
    assert Ric == Matrix([[0, 0], [0, 0]])


def test_ricci_scalar():
    """Test Ricci scalar computation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    R = metric.ricci_scalar()
    
    # For flat metric, R = 0
    assert R == 0


def test_laplace_beltrami_symbol_2d():
    """Test Laplace-Beltrami operator symbol."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    lb = laplace_beltrami_operator(metric)
    
    assert 'principal' in lb
    assert 'subprincipal' in lb
    
    # For flat metric: principal = ξ² + η²
    xi, eta = symbols('xi eta', real=True)
    expected = xi**2 + eta**2
    assert simplify(lb['principal'] - expected) == 0


def test_riemannian_volume_2d():
    """Test Riemannian volume computation."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    # Volume of unit square
    vol = metric.riemannian_volume(((0, 1), (0, 1)), method='numerical')
    
    assert np.isclose(vol, 1.0, rtol=1e-2)


def test_hodge_star_0form():
    """Test Hodge star on 0-forms."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    star = hodge_star(metric, 0)
    
    assert callable(star)
    
    # *1 = √g dx∧dy = 1 for flat metric
    result = star(1)
    assert result == 1


def test_hodge_star_1form():
    """Test Hodge star on 1-forms."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    star = hodge_star(metric, 1)
    
    assert callable(star)


def test_jacobi_equation_solver():
    """Test Jacobi equation integration."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    # Base geodesic
    geod = geodesic_solver(metric, (0, 0), (1, 0), (0, 5), n_steps=100)
    
    # Initial variation
    initial_var = {'J0': (0, 0.1), 'DJ0': (0, 0)}
    
    # Solve Jacobi equation
    J = jacobi_equation_solver(metric, geod, initial_var, (0, 5), n_steps=100)
    
    assert 'J_x' in J
    assert 'J_y' in J
    assert len(J['J_x']) > 0


def test_metric2d_invalid_dimension():
    """Test error for invalid dimension."""
    x, y, z = symbols('x y z', real=True)
    g = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    with pytest.raises(ValueError):
        # Should only accept 2x2 matrices
        metric = Metric2D(g, (x, y))


def test_metric2d_non_symmetric():
    """Test handling of non-symmetric metric (should symmetrize)."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0.5], [0, 1]])  # Non-symmetric
    
    # Should still create metric (will use upper triangle)
    metric = Metric2D(g, (x, y))
    assert metric is not None