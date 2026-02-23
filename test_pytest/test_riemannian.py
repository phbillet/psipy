import numpy as np
import pytest
from sympy import symbols, Matrix, sqrt, exp, sin, cos, simplify, lambdify, pi
from riemannian import (
    Metric,
    christoffel,
    geodesic_solver,
    geodesic_hamiltonian_flow,
    laplace_beltrami,
    sturm_liouville_reduce,
    exponential_map,
    distance,
    jacobi_equation_solver,
    hodge_star,
    de_rham_laplacian,
    verify_gauss_bonnet,
)


# ============================================================================
# Helpers
# ============================================================================

def flat_1d():
    x = symbols('x', real=True)
    return Metric(1, (x,)), x


def power_1d():
    x = symbols('x', real=True, positive=True)
    return Metric(x**2, (x,)), x


def flat_2d():
    x, y = symbols('x y', real=True)
    return Metric(Matrix([[1, 0], [0, 1]]), (x, y)), (x, y)


def polar_2d():
    r, theta = symbols('r theta', real=True, positive=True)
    return Metric(Matrix([[1, 0], [0, r**2]]), (r, theta)), (r, theta)


def sphere_2d():
    theta, phi = symbols('theta phi', real=True)
    return Metric(Matrix([[1, 0], [0, sin(theta)**2]]), (theta, phi)), (theta, phi)


# ============================================================================
# Metric — construction
# ============================================================================

class TestMetricConstruction:

    def test_dim_1d(self):
        m, x = flat_1d()
        assert m.dim == 1

    def test_dim_2d(self):
        m, _ = flat_2d()
        assert m.dim == 2

    def test_1d_flat_components(self):
        m, _ = flat_1d()
        assert m.g_expr == 1
        assert m.g_inv_expr == 1
        assert m.sqrt_det_expr == 1
        assert m.christoffel_sym == 0

    def test_1d_power_inverse(self):
        m, x = power_1d()
        assert simplify(m.g_inv_expr - 1/x**2) == 0

    def test_1d_power_christoffel(self):
        m, x = power_1d()
        assert simplify(m.christoffel_sym - 1/x) == 0

    def test_1d_hyperbolic_christoffel(self):
        x = symbols('x', real=True, positive=True)
        m = Metric(1/x**2, (x,))
        assert simplify(m.christoffel_sym - (-1/x)) == 0

    def test_2d_euclidean_det(self):
        m, _ = flat_2d()
        assert m.det_g == 1

    def test_2d_euclidean_inv(self):
        m, _ = flat_2d()
        assert m.g_inv_matrix == Matrix([[1, 0], [0, 1]])

    def test_2d_polar_det(self):
        m, (r, theta) = polar_2d()
        assert simplify(m.det_g - r**2) == 0

    def test_2d_polar_inv(self):
        m, (r, theta) = polar_2d()
        assert simplify(m.g_inv_matrix - Matrix([[1, 0], [0, 1/r**2]])) == Matrix([[0, 0], [0, 0]])

    def test_2d_invalid_matrix_shape(self):
        x, y, z = symbols('x y z', real=True)
        with pytest.raises(ValueError):
            Metric(Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (x, y))

    def test_wrong_number_of_coords(self):
        x, y = symbols('x y', real=True)
        with pytest.raises(ValueError):
            Metric(Matrix([[1, 0], [0, 1]]), (x, y, y))  # 3 coords for 2D matrix

    def test_from_hamiltonian_1d(self):
        x, p = symbols('x p', real=True, positive=True)
        H = p**2 / (2*x**2)
        m = Metric.from_hamiltonian(H, (x,), (p,))
        assert m.dim == 1
        assert simplify(m.g_expr - x**2) == 0

    def test_from_hamiltonian_2d(self):
        x, y, px, py = symbols('x y p_x p_y', real=True)
        H = (px**2 + py**2) / (2*x**2)
        m = Metric.from_hamiltonian(H, (x, y), (px, py))
        assert m.dim == 2
        assert simplify(m.g_matrix[0, 0] - x**2) == 0


# ============================================================================
# Metric — evaluation
# ============================================================================

class TestMetricEval:

    def test_1d_eval_keys(self):
        x = symbols('x', real=True)
        m = Metric(1 + x**2, (x,))
        result = m.eval(1.0)
        assert {'g', 'g_inv', 'sqrt_det', 'christoffel'} <= result.keys()

    def test_1d_eval_values(self):
        x = symbols('x', real=True)
        m = Metric(1 + x**2, (x,))
        x_vals = np.array([0.0, 1.0, 2.0])
        result = m.eval(x_vals)
        assert np.allclose(result['g'], 1 + x_vals**2)

    def test_2d_eval_keys(self):
        m, _ = flat_2d()
        result = m.eval(1.0, 2.0)
        assert {'g', 'g_inv', 'det_g', 'sqrt_det', 'christoffel'} <= result.keys()

    def test_2d_eval_det(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1 + x**2, 0], [0, 1 + y**2]]), (x, y))
        result = m.eval(1.0, 2.0)
        assert np.isclose(result['det_g'], 2 * 5)  # (1+1)*(1+4)

    def test_2d_callable_funcs(self):
        x = symbols('x', real=True)
        m = Metric(1 + x**2, (x,))
        for fn in (m.g_func, m.g_inv_func, m.sqrt_det_func, m.christoffel_func):
            assert callable(fn)
        assert np.isfinite(m.g_func(1.5))


# ============================================================================
# Curvature
# ============================================================================

class TestCurvature:

    def test_gauss_curvature_1d_is_zero(self):
        x = symbols('x', real=True)
        m = Metric(1 + x**2, (x,))
        assert m.gauss_curvature() == 0

    def test_ricci_scalar_1d_is_zero(self):
        m, x = power_1d()
        assert m.ricci_scalar() == 0

    def test_riemann_tensor_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            m.riemann_tensor()

    def test_gauss_curvature_flat_2d(self):
        m, _ = flat_2d()
        assert m.gauss_curvature() == 0

    def test_gauss_curvature_polar_is_zero(self):
        m, _ = polar_2d()
        assert simplify(m.gauss_curvature()) == 0

    def test_gauss_curvature_poincare(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1/y**2, 0], [0, 1/y**2]]), (x, y))
        assert simplify(m.gauss_curvature()) == -1

    def test_ricci_tensor_flat_2d(self):
        m, _ = flat_2d()
        Ric = m.ricci_tensor()
        assert Ric.shape == (2, 2)
        assert Ric == Matrix([[0, 0], [0, 0]])

    def test_ricci_scalar_flat_2d(self):
        m, _ = flat_2d()
        assert m.ricci_scalar() == 0

    def test_sphere_curvature_not_none(self):
        m, _ = sphere_2d()
        K = m.gauss_curvature()
        assert K is not None


# ============================================================================
# Laplace-Beltrami
# ============================================================================

class TestLaplaceBeltrami:

    def test_1d_principal_symbol(self):
        m, x = power_1d()
        xi = symbols('xi', real=True)
        lb = m.laplace_beltrami_symbol()
        assert simplify(lb['principal'] - xi**2/x**2) == 0

    def test_1d_subprincipal_symbol(self):
        m, x = power_1d()
        xi = symbols('xi', real=True)
        lb = m.laplace_beltrami_symbol()
        assert simplify(lb['subprincipal'] - xi/x**3) == 0

    def test_1d_lb_keys(self):
        m, _ = flat_1d()
        lb = laplace_beltrami(m)
        assert {'principal', 'subprincipal', 'full'} <= lb.keys()

    def test_2d_principal_flat(self):
        m, _ = flat_2d()
        xi, eta = symbols('xi eta', real=True)
        lb = m.laplace_beltrami_symbol()
        assert simplify(lb['principal'] - (xi**2 + eta**2)) == 0

    def test_2d_polar_principal(self):
        m, (r, theta) = polar_2d()
        xi, eta = symbols('xi eta', real=True)
        lb = m.laplace_beltrami_symbol()
        assert simplify(lb['principal'] - (xi**2 + eta**2/r**2)) == 0

    def test_2d_lb_operator_alias(self):
        from riemannian import laplace_beltrami
        m, _ = flat_2d()
        lb = laplace_beltrami(m)
        assert 'principal' in lb


# ============================================================================
# Riemannian volume and arc length
# ============================================================================

class TestVolume:

    def test_1d_flat_volume_symbolic(self):
        m, _ = flat_1d()
        assert m.riemannian_volume((0, 1), method='symbolic') == 1

    def test_1d_flat_volume_numerical(self):
        m, _ = flat_1d()
        assert np.isclose(m.riemannian_volume((0, 1), method='numerical'), 1.0)

    def test_1d_hyperbolic_volume(self):
        x = symbols('x', real=True, positive=True)
        m = Metric(1/x**2, (x,))
        vol_sym = m.riemannian_volume((1, np.e), method='symbolic')
        vol_num = m.riemannian_volume((1, np.e), method='numerical')
        assert abs(float(vol_sym) - 1.0) < 1e-9
        assert abs(vol_num - 1.0) < 1e-5

    def test_1d_invalid_method(self):
        m, _ = flat_1d()
        with pytest.raises(ValueError):
            m.riemannian_volume((0, 1), method='monte_carlo')

    def test_1d_arc_length(self):
        x = symbols('x', real=True, positive=True)
        m = Metric(1/x**2, (x,))
        arc = m.arc_length(1, 2, method='numerical')
        assert np.isclose(arc, np.log(2), rtol=1e-3)

    def test_1d_arc_length_2d_raises(self):
        m, _ = flat_2d()
        with pytest.raises(NotImplementedError):
            m.arc_length(0, 1)

    def test_2d_flat_volume_numerical(self):
        m, _ = flat_2d()
        vol = m.riemannian_volume(((0, 1), (0, 1)), method='numerical')
        assert np.isclose(vol, 1.0, rtol=1e-2)

    def test_2d_diagonal_volume_symbolic(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[4, 0], [0, 9]]), (x, y))
        vol = m.riemannian_volume(((0, 1), (0, 1)), method='symbolic')
        assert vol == 6

    def test_2d_diagonal_volume_numerical(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[4, 0], [0, 9]]), (x, y))
        vol = m.riemannian_volume(((0, 1), (0, 1)), method='numerical')
        assert np.isclose(vol, 6.0)

    def test_2d_invalid_method(self):
        m, _ = flat_2d()
        with pytest.raises(ValueError):
            m.riemannian_volume(((0, 1), (0, 1)), method='monte_carlo')


# ============================================================================
# Christoffel (stand-alone function)
# ============================================================================

class TestChristoffel:

    def test_1d_callable(self):
        m, x = power_1d()
        gamma = christoffel(m)
        assert callable(gamma)
        assert np.isclose(gamma(2.0), 0.5)  # 1/x at x=2

    def test_2d_flat_all_zero(self):
        m, _ = flat_2d()
        Gamma = christoffel(m)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    assert np.isclose(Gamma[i][j][k](0.0, 0.0), 0.0)


# ============================================================================
# Geodesic solver
# ============================================================================

class TestGeodesicSolver:

    # --- 1D ---

    def test_1d_flat_straight_line(self):
        m, _ = flat_1d()
        traj = geodesic_solver(m, 0.0, 1.0, (0, 5), n_steps=100)
        assert np.allclose(traj['x'], traj['t'], rtol=1e-2)
        assert np.allclose(traj['v'], 1.0, rtol=1e-2)

    def test_1d_rk4_length(self):
        m, _ = power_1d()
        traj = geodesic_solver(m, 1.0, 1.0, (0, 5), method='rk4', n_steps=100)
        assert len(traj['x']) == 100

    def test_1d_symplectic_returns_momentum(self):
        m, _ = power_1d()
        traj = geodesic_solver(m, 1.0, 1.0, (0, 5), method='symplectic', n_steps=100)
        assert 'p' in traj

    def test_1d_adaptive_length(self):
        m, _ = flat_1d()
        traj = geodesic_solver(m, 0.0, 1.0, (0, 5), method='adaptive', n_steps=100)
        assert len(traj['x']) == 100

    def test_1d_invalid_method(self):
        m, _ = flat_1d()
        with pytest.raises(ValueError):
            geodesic_solver(m, 0, 1, (0, 5), method='invalid')

    def test_1d_flat_exact(self):
        m, _ = flat_1d()
        traj = geodesic_solver(m, 0.0, 2.0, (0, 2.0), n_steps=100)
        assert abs(traj['x'][-1] - 4.0) < 1e-5

    # --- 2D ---

    def test_2d_flat_straight_line(self):
        m, _ = flat_2d()
        traj = geodesic_solver(m, (0, 0), (1, 1), (0, 5), n_steps=100)
        assert np.allclose(traj['x'], traj['t'], rtol=1e-2)
        assert np.allclose(traj['y'], traj['t'], rtol=1e-2)

    def test_2d_rk45_output_shape(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1 + x**2, 0], [0, 1 + y**2]]), (x, y))
        traj = geodesic_solver(m, (0, 0), (1, 0), (0, 5), method='rk45', n_steps=100)
        assert len(traj['x']) == 100

    def test_2d_symplectic_output_shape(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1 + x**2, 0], [0, 1 + y**2]]), (x, y))
        traj = geodesic_solver(m, (0, 0), (1, 0), (0, 5), method='symplectic', n_steps=100)
        assert len(traj['x']) > 0

    def test_2d_poincare_vertical_geodesic(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1/y**2, 0], [0, 1/y**2]]), (x, y))
        traj = geodesic_solver(m, (0, 1), (0, 1), (0, 1), method='rk45')
        assert np.allclose(traj['x'], 0, atol=1e-4)
        assert traj['y'][-1] > 1

    def test_2d_reparametrize(self):
        m, _ = flat_2d()
        traj = geodesic_solver(m, (0, 0), (1, 1), (0, 10), method='rk45',
                               reparametrize=True)
        assert 'arc_length' in traj
        assert np.isclose(traj['arc_length'][-1], np.sqrt(2) * 10, rtol=1e-3)

    def test_2d_invalid_method(self):
        m, _ = flat_2d()
        with pytest.raises(ValueError):
            geodesic_solver(m, (0, 0), (1, 0), (0, 5), method='magic')


# ============================================================================
# Hamiltonian flow
# ============================================================================

class TestHamiltonianFlow:

    def test_1d_energy_keys(self):
        m, _ = power_1d()
        res = geodesic_hamiltonian_flow(m, 1.0, 1.0, (0, 5), method='verlet', n_steps=100)
        assert {'x', 'p', 'energy'} <= res.keys()

    def test_1d_energy_conservation_verlet(self):
        x = symbols('x', real=True, positive=True)
        m = Metric(exp(x), (x,))
        res = geodesic_hamiltonian_flow(m, 1.0, 0.5, (0, 10), method='verlet', n_steps=1000)
        E0 = res['energy'][0]
        assert abs(res['energy'][-1] - E0) / abs(E0) < 1e-2

    def test_1d_symplectic_euler_bounded(self):
        m, _ = power_1d()
        x0, p0 = 2.0, 10.0
        E0 = p0**2 / (2 * x0**2)
        res = geodesic_hamiltonian_flow(m, x0, p0, (0, 10.0),
                                        method='symplectic_euler', n_steps=2000)
        assert np.std(res['energy']) / E0 < 5e-2

    def test_1d_invalid_method(self):
        m, _ = flat_1d()
        with pytest.raises(ValueError):
            geodesic_hamiltonian_flow(m, 0, 1, (0, 1), method='bad')

    def test_2d_energy_keys(self):
        m, _ = flat_2d()
        traj = geodesic_hamiltonian_flow(m, (0, 0), (1, 1), (0, 5),
                                          method='verlet', n_steps=100)
        assert {'px', 'py', 'energy'} <= traj.keys()

    def test_2d_energy_conservation(self):
        m, _ = flat_2d()
        traj = geodesic_hamiltonian_flow(m, (0, 0), (1, 1), (0, 5),
                                          method='verlet', n_steps=100)
        assert np.std(traj['energy']) < 1e-2

    def test_2d_symplectic_energy_flat(self):
        m, _ = flat_2d()
        traj = geodesic_solver(m, (0, 0), (1, 1), (0, 10), method='symplectic', n_steps=100)
        assert np.std(traj['energy']) < 1e-10


# ============================================================================
# Sturm-Liouville (1D only)
# ============================================================================

class TestSturmLiouville:

    def test_coefficients(self):
        m, x = power_1d()
        sl = sturm_liouville_reduce(m)
        assert {'p', 'q', 'w', 'p_func', 'q_func', 'w_func'} <= sl.keys()
        assert simplify(sl['p'] - 1/x) == 0
        assert simplify(sl['w'] - x) == 0

    def test_with_potential(self):
        x = symbols('x', real=True, positive=True)
        m = Metric(1, (x,))
        sl = sturm_liouville_reduce(m, potential_expr=x**2)
        assert simplify(sl['q'] - x**2) == 0

    def test_2d_raises(self):
        m, _ = flat_2d()
        with pytest.raises(NotImplementedError):
            sturm_liouville_reduce(m)


# ============================================================================
# Exponential map and distance (2D only)
# ============================================================================

class TestExponentialMapDistance:

    def test_exp_map_flat(self):
        m, _ = flat_2d()
        q = exponential_map(m, (0, 0), (1, 1), t=1.0)
        assert len(q) == 2
        assert np.isclose(q[0], 1.0, rtol=1e-1)
        assert np.isclose(q[1], 1.0, rtol=1e-1)

    def test_exp_map_exact_flat(self):
        m, _ = flat_2d()
        q = exponential_map(m, (0, 0), (3, 4), t=1.0)
        assert np.allclose(q, (3, 4), atol=1e-4)

    def test_exp_map_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            exponential_map(m, 0, 1)

    def test_distance_shooting(self):
        m, _ = flat_2d()
        d = distance(m, (0, 0), (3, 4), method='shooting', max_iter=20)
        assert np.isclose(d, 5.0, rtol=0.1)

    def test_distance_optimize(self):
        m, _ = flat_2d()
        d = distance(m, (0, 0), (3, 4), method='optimize')
        assert np.isclose(d, 5.0, rtol=5e-2)

    def test_distance_invalid_method(self):
        m, _ = flat_2d()
        with pytest.raises(ValueError):
            distance(m, (0, 0), (1, 1), method='bad')

    def test_distance_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            distance(m, 0, 1)


# ============================================================================
# Hodge star and de Rham Laplacian (2D only)
# ============================================================================

class TestHodge:

    def test_0form_flat(self):
        m, _ = flat_2d()
        star = hodge_star(m, 0)
        assert callable(star)
        assert star(1) == 1  # √g = 1

    def test_2form_flat(self):
        m, _ = flat_2d()
        star = hodge_star(m, 2)
        assert callable(star)
        from sympy import simplify as sp_simplify
        assert sp_simplify(star(1)) == 1

    def test_2form_diagonal(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[4, 0], [0, 9]]), (x, y))
        star2 = hodge_star(m, 2)
        from sympy import simplify as sp_simplify
        assert sp_simplify(star2(12)) == 2  # 12 / 6

    def test_1form_callable(self):
        m, _ = flat_2d()
        star = hodge_star(m, 1)
        assert callable(star)

    def test_invalid_degree(self):
        m, _ = flat_2d()
        with pytest.raises(ValueError):
            hodge_star(m, 3)

    def test_hodge_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            hodge_star(m, 0)

    def test_de_rham_0form(self):
        m, _ = flat_2d()
        lb = de_rham_laplacian(m, 0)
        assert 'principal' in lb

    def test_de_rham_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            de_rham_laplacian(m, 0)


# ============================================================================
# Jacobi equation (2D only)
# ============================================================================

class TestJacobi:

    def test_jacobi_flat(self):
        m, _ = flat_2d()
        geod = geodesic_solver(m, (0, 0), (1, 0), (0, 5), n_steps=100)
        J = jacobi_equation_solver(m, geod, {'J0': (0, 0.1), 'DJ0': (0, 0)},
                                   (0, 5), n_steps=100)
        assert {'J_x', 'J_y', 'DJ_x', 'DJ_y'} <= J.keys()
        assert len(J['J_x']) > 0

    def test_jacobi_sphere_bounded(self):
        m, _ = sphere_2d()
        geod = geodesic_solver(m, (np.pi/2, 0), (0, 1), (0, 2), n_steps=200)
        J = jacobi_equation_solver(m, geod, {'J0': (0, 0), 'DJ0': (0.1, 0)}, (0, 2))
        assert np.max(np.abs(J['J_x'])) < 1.0

    def test_jacobi_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            jacobi_equation_solver(m, {}, {}, (0, 1))


# ============================================================================
# Gauss-Bonnet (2D only)
# ============================================================================

class TestGaussBonnet:

    def test_returns_dict(self):
        m, _ = flat_2d()
        result = verify_gauss_bonnet(m, ((0, 1), (0, 1)))
        assert {'integral', 'expected', 'integration_error', 'relative_error'} <= result.keys()

    def test_flat_integral_zero(self):
        m, _ = flat_2d()
        result = verify_gauss_bonnet(m, ((0, 1), (0, 1)))
        assert np.isclose(result['integral'], 0.0, atol=1e-6)

    def test_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            verify_gauss_bonnet(m, ((0, 1), (0, 1)))