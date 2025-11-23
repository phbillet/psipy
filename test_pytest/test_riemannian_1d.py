import numpy as np
import pytest
from sympy import symbols, sqrt, exp, sin, cos, simplify, lambdify
from riemannian_1d import *


def test_metric1d_flat():
    """Test flat metric (Euclidean)."""
    x = symbols('x', real=True)
    metric = Metric1D(1, x)
    
    # Christoffel should be zero
    assert metric.christoffel_expr == 0
    
    # Inverse metric should be 1
    assert metric.g_inv_expr == 1
    
    # Volume element should be 1
    assert metric.sqrt_det_expr == 1


def test_metric1d_power():
    """Test power law metric g = x^2."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(x**2, x)
    
    # Check inverse metric
    expected_inv = 1 / x**2
    assert simplify(metric.g_inv_expr - expected_inv) == 0
    
    # Check Christoffel symbol
    # Γ¹₁₁ = ½(log x²)' = ½ · 2/x = 1/x
    expected_gamma = 1 / x
    assert simplify(metric.christoffel_expr - expected_gamma) == 0


def test_metric1d_hyperbolic():
    """Test hyperbolic metric g = 1/x²."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(1/x**2, x)
    
    # Christoffel symbol: Γ = ½(log(1/x²))' = -1/x
    expected_gamma = -1/x
    assert simplify(metric.christoffel_expr - expected_gamma) == 0


def test_metric1d_from_hamiltonian():
    """Test metric extraction from Hamiltonian."""
    x, p = symbols('x p', real=True, positive=True)
    
    # H = p²/(2m(x)) with m(x) = x²
    H = p**2 / (2*x**2)
    metric = Metric1D.from_hamiltonian(H, x, p)
    
    # Should extract g = x²
    assert simplify(metric.g_expr - x**2) == 0


def test_metric1d_eval():
    """Test numerical evaluation of metric components."""
    x = symbols('x', real=True)
    metric = Metric1D(1 + x**2, x)
    
    x_vals = np.array([0.0, 1.0, 2.0])
    result = metric.eval(x_vals)
    
    assert 'g' in result
    assert 'g_inv' in result
    assert 'christoffel' in result
    
    # Check values
    expected_g = 1 + x_vals**2
    assert np.allclose(result['g'], expected_g)


def test_metric1d_volume():
    """Test Riemannian volume computation."""
    x = symbols('x', real=True)
    metric = Metric1D(1, x)
    
    # Volume of [0, 1] with flat metric should be 1
    vol_symbolic = metric.riemannian_volume(0, 1, method='symbolic')
    assert vol_symbolic == 1
    
    vol_numeric = metric.riemannian_volume(0, 1, method='numerical')
    assert np.isclose(vol_numeric, 1.0)


def test_metric1d_arc_length():
    """Test arc length computation."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(1/x**2, x)
    
    # Arc length for hyperbolic metric from 1 to 2
    # ∫₁² dx/x = log(2)
    arc_length = metric.arc_length(1, 2, method='numerical')
    expected = np.log(2)
    assert np.isclose(arc_length, expected, rtol=1e-3)


def test_christoffel_function():
    """Test standalone christoffel function."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(x**2, x)
    
    gamma = christoffel(metric)
    assert callable(gamma)
    
    # Test evaluation
    gamma_val = gamma(2.0)
    expected = 0.5  # Γ = 1/x at x=2
    assert np.isclose(gamma_val, expected)


def test_geodesic_integrator_flat():
    """Test geodesic integration on flat space."""
    x = symbols('x', real=True)
    metric = Metric1D(1, x)
    
    # Geodesics should be straight lines
    traj = geodesic_integrator(metric, 0.0, 1.0, (0, 5), n_steps=100)
    
    # Position should evolve as x(t) = x₀ + v₀·t
    expected_x = traj['t']
    assert np.allclose(traj['x'], expected_x, rtol=1e-2)
    
    # Velocity should be constant
    assert np.allclose(traj['v'], 1.0, rtol=1e-2)


def test_geodesic_integrator_methods():
    """Test different integration methods."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(1 + x**2, x)
    
    # Test RK4
    traj_rk4 = geodesic_integrator(metric, 1.0, 1.0, (0, 5), 
                                   method='rk4', n_steps=100)
    assert len(traj_rk4['x']) == 100
    
    # Test symplectic
    traj_symp = geodesic_integrator(metric, 1.0, 1.0, (0, 5), 
                                    method='symplectic', n_steps=100)
    assert len(traj_symp['x']) == 100
    assert 'p' in traj_symp  # Symplectic method returns momentum


def test_geodesic_hamiltonian_flow():
    """Test geodesic flow in Hamiltonian formulation."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(x**2, x)
    
    traj = geodesic_hamiltonian_flow(metric, 1.0, 1.0, (0, 5), 
                                     method='verlet', n_steps=100)
    
    assert 'x' in traj
    assert 'p' in traj
    assert 'energy' in traj
    
    # Energy should be approximately conserved
    energy_drift = np.std(traj['energy'])
    assert energy_drift < 1e-2


def test_laplace_beltrami_symbol():
    """Test Laplace-Beltrami operator symbol."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(x**2, x)
    
    lb = laplace_beltrami(metric)
    
    assert 'principal' in lb
    assert 'subprincipal' in lb
    assert 'full' in lb
    
    # Principal symbol should be g⁻¹ξ² = ξ²/x²
    xi = symbols('xi', real=True)
    expected_principal = xi**2 / x**2
    assert simplify(lb['principal'] - expected_principal) == 0


def test_sturm_liouville_reduce():
    """Test reduction to Sturm-Liouville form."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(x**2, x)
    
    sl = sturm_liouville_reduce(metric)
    
    assert 'p' in sl
    assert 'q' in sl
    assert 'w' in sl
    
    # p(x) = √g · g⁻¹ = x · 1/x² = 1/x
    expected_p = 1 / x
    assert simplify(sl['p'] - expected_p) == 0
    
    # w(x) = √g = x
    assert simplify(sl['w'] - x) == 0


def test_sturm_liouville_with_potential():
    """Test Sturm-Liouville reduction with potential."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(1, x)
    V = x**2  # Harmonic potential
    
    sl = sturm_liouville_reduce(metric, potential_expr=V)
    
    # With flat metric and potential V
    # q(x) = V(x) · √g = x²
    assert simplify(sl['q'] - x**2) == 0


def test_gauss_curvature_1d():
    """Test Gaussian curvature (should be zero for 1D)."""
    x = symbols('x', real=True)
    metric = Metric1D(1 + x**2, x)
    
    K = metric.gauss_curvature()
    assert K == 0  # 1D manifolds have zero intrinsic curvature


def test_ricci_scalar_1d():
    """Test Ricci scalar (should be zero for 1D)."""
    x = symbols('x', real=True)
    metric = Metric1D(x**2, x)
    
    R = metric.ricci_scalar()
    assert R == 0


def test_metric1d_callable_functions():
    """Test that all lambdified functions are callable."""
    x = symbols('x', real=True)
    metric = Metric1D(1 + x**2, x)
    
    x_test = 1.5
    
    assert callable(metric.g_func)
    assert callable(metric.g_inv_func)
    assert callable(metric.sqrt_det_func)
    assert callable(metric.christoffel_func)
    
    # Test evaluation
    g_val = metric.g_func(x_test)
    assert np.isfinite(g_val)


def test_geodesic_energy_conservation():
    """Test energy conservation in geodesic flow."""
    x = symbols('x', real=True, positive=True)
    metric = Metric1D(exp(x), x)
    
    traj = geodesic_hamiltonian_flow(metric, 1.0, 0.5, (0, 10), 
                                     method='verlet', n_steps=1000)
    
    # Check energy conservation
    energy_initial = traj['energy'][0]
    energy_final = traj['energy'][-1]
    relative_error = abs(energy_final - energy_initial) / abs(energy_initial)
    
    assert relative_error < 1e-2


def test_metric1d_invalid_method():
    """Test error handling for invalid methods."""
    x = symbols('x', real=True)
    metric = Metric1D(1, x)
    
    with pytest.raises(ValueError):
        geodesic_integrator(metric, 0, 1, (0, 5), method='invalid')
    
    with pytest.raises(ValueError):
        metric.riemannian_volume(0, 1, method='invalid')