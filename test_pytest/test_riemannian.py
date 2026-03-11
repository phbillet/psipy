import numpy as np
import pytest
from unittest.mock import patch
from sympy import (
    symbols, Matrix, sqrt, exp, sin, cos, simplify, lambdify, pi,
    zeros, Rational
)
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
    visualize_geodesics,
    visualize_curvature,
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


def poincare_2d():
    x, y = symbols('x y', real=True)
    return Metric(Matrix([[1/y**2, 0], [0, 1/y**2]]), (x, y)), (x, y)


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
        assert simplify(
            m.g_inv_matrix - Matrix([[1, 0], [0, 1/r**2]])
        ) == Matrix([[0, 0], [0, 0]])

    def test_2d_invalid_matrix_shape(self):
        x, y, z = symbols('x y z', real=True)
        with pytest.raises(ValueError):
            Metric(Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), (x, y))

    def test_wrong_number_of_coords(self):
        x, y = symbols('x y', real=True)
        with pytest.raises(ValueError):
            Metric(Matrix([[1, 0], [0, 1]]), (x, y, y))

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

    def test_from_hamiltonian_2d_off_diagonal(self):
        """Off-diagonal kinetic term exercises the g_inv_12 code path."""
        x, y, px, py = symbols('x y px py', real=True)
        # H = ½(px² + 2*px*py + 2*py²)  → g_inv = [[1,1],[1,2]] → g = [[2,-1],[-1,1]]
        H = (px**2 + 2*px*py + 2*py**2) / 2
        m = Metric.from_hamiltonian(H, (x, y), (px, py))
        g_inv_expected = Matrix([[1, 1], [1, 2]])
        g_expected = g_inv_expected.inv()
        assert simplify(m.g_matrix - g_expected) == zeros(2, 2)

    def test_from_hamiltonian_3d_raises(self):
        x, y, z, px, py, pz = symbols('x y z px py pz', real=True)
        H = (px**2 + py**2 + pz**2) / 2
        with pytest.raises(ValueError):
            Metric.from_hamiltonian(H, (x, y, z), (px, py, pz))


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

    def test_1d_eval_array_broadcasting(self):
        """eval should accept a numpy array and return array-shaped results."""
        x = symbols('x', real=True, positive=True)
        m = Metric(x**2, (x,))
        x_vals = np.linspace(1.0, 3.0, 10)
        result = m.eval(x_vals)
        assert np.allclose(result['g'], x_vals**2)
        assert np.allclose(result['g_inv'], 1.0 / x_vals**2)

    def test_2d_eval_keys(self):
        m, _ = flat_2d()
        result = m.eval(1.0, 2.0)
        assert {'g', 'g_inv', 'det_g', 'sqrt_det', 'christoffel'} <= result.keys()

    def test_2d_eval_det(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1 + x**2, 0], [0, 1 + y**2]]), (x, y))
        result = m.eval(1.0, 2.0)
        assert np.isclose(result['det_g'], 2 * 5)  # (1+1)*(1+4)

    def test_2d_eval_array_broadcasting(self):
        """2D eval should accept numpy arrays (meshgrid use-case)."""
        m, _ = flat_2d()
        xs = np.linspace(0.0, 1.0, 5)
        ys = np.linspace(0.0, 1.0, 5)
        X, Y = np.meshgrid(xs, ys)
        result = m.eval(X, Y)
        assert result['g'].shape == (2, 2, 5, 5)
        assert np.allclose(result['det_g'], 1.0)

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

    def test_ricci_tensor_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            m.ricci_tensor()

    def test_gauss_curvature_flat_2d(self):
        m, _ = flat_2d()
        assert m.gauss_curvature() == 0

    def test_gauss_curvature_polar_is_zero(self):
        m, _ = polar_2d()
        assert simplify(m.gauss_curvature()) == 0

    def test_gauss_curvature_poincare(self):
        m, _ = poincare_2d()
        assert simplify(m.gauss_curvature()) == -1

    def test_gauss_curvature_sphere_is_one(self):
        """Sphere K must equal +1, not merely be non-None."""
        m, _ = sphere_2d()
        K = simplify(m.gauss_curvature())
        assert simplify(K - 1) == 0, f"Expected K=1, got {K}"

    def test_ricci_tensor_flat_2d(self):
        m, _ = flat_2d()
        Ric = m.ricci_tensor()
        assert Ric.shape == (2, 2)
        assert Ric == Matrix([[0, 0], [0, 0]])

    def test_ricci_scalar_flat_2d(self):
        m, _ = flat_2d()
        assert m.ricci_scalar() == 0

    def test_ricci_scalar_sphere_equals_2(self):
        """For unit sphere, R = 2K = 2."""
        m, _ = sphere_2d()
        R = simplify(m.ricci_scalar())
        assert simplify(R - 2) == 0, f"Expected R=2 for unit sphere, got {R}"

    def test_riemann_tensor_antisymmetry(self):
        """R[i][j][k][l] = -R[i][j][l][k] (antisymmetry in last two indices)."""
        m, _ = sphere_2d()
        R = m.riemann_tensor()
        for i in range(2):
            for j in range(2):
                assert simplify(R[i][j][0][1] + R[i][j][1][0]) == 0


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
        m, _ = flat_2d()
        lb = laplace_beltrami(m)
        assert 'principal' in lb

    def test_1d_flat_subprincipal_zero(self):
        """For the flat 1D metric g=1, the subprincipal symbol must vanish."""
        m, _ = flat_1d()
        xi = symbols('xi', real=True)
        lb = m.laplace_beltrami_symbol()
        assert simplify(lb['subprincipal']) == 0


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

    def test_2d_polar_nonzero_component(self):
        """Γ¹₀₀ for polar metric g=diag(1,r²) should equal -r (i.e. Γ^r_{θθ}=-r)."""
        m, (r, theta) = polar_2d()
        Gamma = christoffel(m)
        # Γ^0_{11} = -r  (in (r,θ) coords, upper index 0 = r direction)
        assert np.isclose(Gamma[0][1][1](2.0, 0.5), -2.0)


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
        m, _ = poincare_2d()
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

    def test_2d_geodesic_speed_constant_flat(self):
        """On the flat metric, |v|² must remain constant along the geodesic."""
        m, _ = flat_2d()
        traj = geodesic_solver(m, (0, 0), (1, 2), (0, 5), method='rk45', n_steps=200)
        speed_sq = traj['vx']**2 + traj['vy']**2
        assert np.allclose(speed_sq, speed_sq[0], rtol=1e-3)


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
        """Energy drift for symplectic Euler must stay within 5 % relative std.
        NOTE: the function signature expects a velocity v0, not a momentum."""
        m, _ = power_1d()
        x0 = 2.0
        # Choose a true velocity; internally the code converts p = g(x0)*v0
        v0 = 1.0
        E0 = float(m.g_func(x0) * v0**2 / 2)   # ½ g v² = KE
        res = geodesic_hamiltonian_flow(m, x0, v0, (0, 10.0),
                                        method='symplectic_euler', n_steps=2000)
        assert np.std(res['energy']) / abs(E0) < 5e-2

    def test_1d_velocity_not_momentum_api(self):
        """Verify that geodesic_hamiltonian_flow treats its second argument as
        velocity (not raw momentum): for g=x², passing v0 at x0 should give
        H(x0) = ½ g(x0) v0² = ½ x0² v0²."""
        m, _ = power_1d()
        x0, v0 = 3.0, 2.0
        res = geodesic_hamiltonian_flow(m, x0, v0, (0, 0.01), method='verlet', n_steps=10)
        expected_E = 0.5 * float(m.g_func(x0)) * v0**2
        assert np.isclose(res['energy'][0], expected_E, rtol=1e-6), (
            f"Initial energy {res['energy'][0]:.6f} ≠ expected {expected_E:.6f}. "
            "geodesic_hamiltonian_flow must accept velocity, not momentum."
        )

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
        assert np.std(traj['energy']) < 1e-3

    def test_2d_velocity_not_momentum_api(self):
        """Same API check for 2D: initial energy = ½ gij vi vj."""
        m, _ = flat_2d()
        p0, v0 = (0.0, 0.0), (3.0, 4.0)
        traj = geodesic_hamiltonian_flow(m, p0, v0, (0, 0.01), method='verlet', n_steps=5)
        expected_E = 0.5 * (v0[0]**2 + v0[1]**2)  # flat metric
        assert np.isclose(traj['energy'][0], expected_E, rtol=1e-6)


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

    def test_callables_are_callable(self):
        m, x = power_1d()
        sl = sturm_liouville_reduce(m)
        for key in ('p_func', 'q_func', 'w_func'):
            assert callable(sl[key])
        assert np.isfinite(sl['p_func'](2.0))

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

    def test_distance_symmetry_flat(self):
        """d(p,q) == d(q,p) on the flat plane."""
        m, _ = flat_2d()
        d1 = distance(m, (1, 2), (4, 6), method='shooting')
        d2 = distance(m, (4, 6), (1, 2), method='shooting')
        assert np.isclose(d1, d2, rtol=1e-2)

    def test_distance_origin_is_zero(self):
        m, _ = flat_2d()
        d = distance(m, (2, 3), (2, 3), method='shooting')
        assert np.isclose(d, 0.0, atol=1e-3)


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
        assert simplify(star(1)) == 1

    def test_2form_diagonal(self):
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[4, 0], [0, 9]]), (x, y))
        star2 = hodge_star(m, 2)
        assert simplify(star2(12)) == 2  # 12 / 6

    def test_1form_callable(self):
        m, _ = flat_2d()
        star = hodge_star(m, 1)
        assert callable(star)

    def test_1form_flat_rotation(self):
        """On the flat metric (g_inv=I, sqrt_g=1) the 1-form Hodge star is:
        beta_x = alpha_y,  beta_y = alpha_x  (from the implementation formula
        beta_x = g^00 alpha_y - g^01 alpha_x,  beta_y = -g^01 alpha_y + g^11 alpha_x).
        """
        x_sym, y_sym = symbols('x y', real=True)
        m = Metric(Matrix([[1, 0], [0, 1]]), (x_sym, y_sym))
        star = hodge_star(m, 1)
        # α = dx  → (alpha_x=1, alpha_y=0): beta_x=0, beta_y=1
        bx, by = star(1, 0)
        assert simplify(bx) == 0
        assert simplify(by) == 1
        # α = dy  → (alpha_x=0, alpha_y=1): beta_x=1, beta_y=0
        bx2, by2 = star(0, 1)
        assert simplify(bx2) == 1
        assert simplify(by2) == 0

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

    def test_de_rham_1form(self):
        """de Rham Laplacian on 1-forms should return a dict with 'principal'."""
        m, _ = flat_2d()
        lb = de_rham_laplacian(m, 1)
        assert 'principal' in lb

    def test_de_rham_1form_principal_flat(self):
        """Principal symbol of de Rham Δ on 1-forms = ξ² + η² on flat metric."""
        m, _ = flat_2d()
        xi, eta = symbols('xi eta', real=True)
        lb = de_rham_laplacian(m, 1)
        assert simplify(lb['principal'] - (xi**2 + eta**2)) == 0

    def test_de_rham_invalid_degree(self):
        m, _ = flat_2d()
        with pytest.raises(NotImplementedError):
            de_rham_laplacian(m, 2)

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

    def test_jacobi_flat_linear_growth(self):
        """On a flat manifold, transverse Jacobi fields grow linearly."""
        m, _ = flat_2d()
        geod = geodesic_solver(m, (0, 0), (1, 0), (0, 5), n_steps=200)
        J = jacobi_equation_solver(m, geod, {'J0': (0, 0), 'DJ0': (0, 1)},
                                   (0, 5), n_steps=200)
        # J_y(t) should be approximately t (linear in t)
        assert np.allclose(J['J_y'], J['t'], rtol=1e-1)

    def test_jacobi_sphere_bounded(self):
        m, _ = sphere_2d()
        geod = geodesic_solver(m, (np.pi/2, 0), (0, 1), (0, 2), n_steps=200)
        J = jacobi_equation_solver(m, geod, {'J0': (0, 0), 'DJ0': (0.1, 0)}, (0, 2))
        assert np.max(np.abs(J['J_x'])) < 1.0

    def test_jacobi_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            jacobi_equation_solver(m, {}, {}, (0, 1))

    def test_jacobi_output_length(self):
        m, _ = flat_2d()
        geod = geodesic_solver(m, (0, 0), (1, 0), (0, 3), n_steps=50)
        J = jacobi_equation_solver(m, geod, {'J0': (0.1, 0), 'DJ0': (0, 0)},
                                   (0, 3), n_steps=50)
        assert len(J['t']) == 50


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

    def test_poincare_integral_negative(self):
        """K = -1 on the Poincaré half-plane: ∫∫ K √g dA must be negative."""
        m, _ = poincare_2d()
        result = verify_gauss_bonnet(m, ((0, 1), (1, 2)))
        assert result['integral'] < 0

    def test_poincare_integral_value(self):
        """For Poincaré metric on [0,L]×[y1,y2], ∫∫ K dA = -L(1/y1 - 1/y2)."""
        m, _ = poincare_2d()
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 1.0, 2.0
        result = verify_gauss_bonnet(m, ((x_min, x_max), (y_min, y_max)))
        # K = -1, √g = 1/y², so ∫ K√g dA = -∫₀¹ dx ∫₁² 1/y² dy = -(1)(1/2) = -0.5
        expected = -(x_max - x_min) * (1/y_min - 1/y_max)
        assert np.isclose(result['integral'], expected, rtol=1e-3)

    def test_1d_raises(self):
        m, _ = flat_1d()
        with pytest.raises(NotImplementedError):
            verify_gauss_bonnet(m, ((0, 1), (0, 1)))


# ============================================================================
# Visualisation — smoke tests (matplotlib calls are mocked out)
# ============================================================================

class TestVisualisation:

    @patch('matplotlib.pyplot.show')
    def test_visualize_geodesics_1d_runs(self, mock_show):
        """visualize_geodesics must complete without exception for 1D.
        Uses a non-constant metric (x²) so lambdify returns an array, not a scalar."""
        m, _ = power_1d()
        visualize_geodesics(
            m,
            initial_conditions=[(1.0, 0.5), (1.0, 1.0)],
            tspan=(0, 1),
            n_steps=50,
        )
        assert mock_show.called

    @patch('matplotlib.pyplot.show')
    def test_visualize_geodesics_2d_runs(self, mock_show):
        """visualize_geodesics must complete without exception for 2D."""
        m, _ = flat_2d()
        visualize_geodesics(
            m,
            initial_conditions=[((0, 0), (1, 0)), ((0, 0), (0, 1))],
            tspan=(0, 3),
            n_steps=50,
        )
        assert mock_show.called

    @patch('matplotlib.pyplot.show')
    def test_visualize_curvature_1d_runs(self, mock_show):
        """visualize_curvature must complete without exception for 1D."""
        m, _ = power_1d()
        visualize_curvature(m, x_range=(0.5, 3.0), quantity='metric')
        assert mock_show.called

    @patch('matplotlib.pyplot.show')
    def test_visualize_curvature_2d_runs(self, mock_show):
        """visualize_curvature must complete without exception for 2D (gauss).
        Uses g = diag(1+x², 1+y²); the fix in _visualize_curvature_2d ensures
        a constant curvature result is broadcast to the meshgrid shape."""
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1 + x**2, 0], [0, 1 + y**2]]), (x, y))
        visualize_curvature(
            m,
            x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
            quantity='gauss', resolution=10,
        )
        assert mock_show.called

    @patch('matplotlib.pyplot.show')
    def test_visualize_curvature_2d_ricci_runs(self, mock_show):
        """visualize_curvature must complete without exception for 2D (ricci_scalar)."""
        x, y = symbols('x y', real=True)
        m = Metric(Matrix([[1 + x**2, 0], [0, 1 + y**2]]), (x, y))
        visualize_curvature(
            m,
            x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
            quantity='ricci_scalar', resolution=10,
        )
        assert mock_show.called

    def test_visualize_curvature_2d_no_range_raises(self):
        """Without x_range/y_range, 2D curvature visualisation must raise."""
        m, _ = flat_2d()
        with pytest.raises(ValueError):
            visualize_curvature(m, quantity='gauss')