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
Unified Riemannian geometry toolkit for 1D and 2D manifolds.
Supports:
 - Metric tensor g_ij(x) (1×1 or 2×2)
 - Christoffel symbols Γⁱⱼₖ
 - Curvature (Gauss, Ricci, scalar; 0 in 1D)
 - Geodesics (Lagrangian & Hamiltonian)
 - Laplace-Beltrami operator (with Sturm-Liouville reduction in 1D)
 - Hodge star, de Rham Laplacian (2D only)
 - Jacobi fields, Gauss-Bonnet (2D only)
 - Exponential map, geodesic distance
 - Volume, arc length, visualization
"""
from imports import *

class Metric:
    """
    Unified Riemannian metric for 1D or 2D manifolds.

    Parameters
    ----------
    g_input : sympy expression (1D) or 2×2 sympy Matrix (2D)
        Metric tensor component(s).
    coords : tuple of sympy symbols
        Coordinate variables (length 1 or 2).

    Examples
    --------
    >>> # 1D
    >>> x = symbols('x')
    >>> metric = Metric(x**2, (x,))
    >>> # 2D
    >>> x, y = symbols('x y')
    >>> g = Matrix([[1, 0], [0, x**2]])
    >>> metric = Metric(g, (x, y))
    >>> # From Hamiltonian
    >>> p = symbols('p')
    >>> metric = Metric.from_hamiltonian(p**2/(2*x**2), (x,), (p,))
    """
    def __init__(self, g_input, coords):
        self.dim = len(coords)
        self.coords = coords

        if self.dim == 1:
            x, = coords
            self.g_matrix = Matrix([[simplify(g_input)]])
        elif self.dim == 2:
            if not isinstance(g_input, Matrix) or g_input.shape != (2, 2):
                raise ValueError("2D metric must be a 2×2 sympy Matrix")
            self.g_matrix = simplify(g_input)
        else:
            raise ValueError("Only 1D and 2D manifolds are supported")

        self.det_g = simplify(self.g_matrix.det())
        self.sqrt_det_g = simplify(sqrt(abs(self.det_g)))
        self.g_inv_matrix = simplify(self.g_matrix.inv())
        self.christoffel = self._compute_christoffel()
        self._lambdify_all()

    def _compute_christoffel(self):
        n = self.dim
        g = self.g_matrix
        g_inv = self.g_inv_matrix
        Gamma = {}
        for i in range(n):
            Gamma[i] = {}
            for j in range(n):
                Gamma[i][j] = {}
                for k in range(n):
                    expr = 0
                    for ell in range(n):
                        term1 = diff(g[k, ell], self.coords[j])
                        term2 = diff(g[j, ell], self.coords[k])
                        term3 = diff(g[j, k], self.coords[ell])
                        expr += g_inv[i, ell] * (term1 + term2 - term3) / 2
                    Gamma[i][j][k] = simplify(expr)
        return Gamma

    def _lambdify_all(self):
        n = self.dim
        self.g_func = {(i,j): lambdify(self.coords, self.g_matrix[i,j], 'numpy')
                       for i in range(n) for j in range(n)}
        self.g_inv_func = {(i,j): lambdify(self.coords, self.g_inv_matrix[i,j], 'numpy')
                           for i in range(n) for j in range(n)}
        self.sqrt_det_g_func = lambdify(self.coords, self.sqrt_det_g, 'numpy')
        self.christoffel_func = {}
        for i in range(n):
            self.christoffel_func[i] = {}
            for j in range(n):
                self.christoffel_func[i][j] = {}
                for k in range(n):
                    self.christoffel_func[i][j][k] = lambdify(
                        self.coords, self.christoffel[i][j][k], 'numpy'
                    )

    @classmethod
    def from_hamiltonian(cls, H_expr, coords, momenta):
        """
        Extract metric from Hamiltonian kinetic term.
        
        For H = ½ g^ij pᵢ pⱼ + V, extract inverse metric from Hessian:
            g^ij = ∂²H/∂pᵢ∂pⱼ
        
        Parameters
        ----------
        H_expr : sympy expression
            Hamiltonian H(x, y, pₓ, pᵧ).
        vars_xy : tuple
            Position variables (x, y).
        vars_p : tuple
            Momentum variables (pₓ, pᵧ).
        
        Returns
        -------
        Metric2D
            Metric with gᵢⱼ = (g^ij)⁻¹.
        
        Examples
        --------
        >>> x, p = symbols('x p', real=True)
        >>> H = p**2/(2*x**2) + x**2/2
        >>> metric = Metric.from_hamiltonian(H, x, p)
        >>> print(metric.g_expr)
        x**2
        >>> x, y, px, py = symbols('x y p_x p_y', real=True)
        >>> H = (px**2 + py**2)/(2*x**2)
        >>> metric = Metric.from_hamiltonian(H, (x,y), (px,py))
        """
        if len(coords) != len(momenta):
            raise ValueError("Coordinate and momentum lists must match in length")
        n = len(coords)
        # Build inverse metric: g^ij = ∂²H / ∂p_i ∂p_j
        g_inv = Matrix([[diff(H_expr, momenta[i], momenta[j])
                         for j in range(n)] for i in range(n)])
        # Invert and simplify element-wise to preserve Matrix type
        g = g_inv.inv()
        g = Matrix([[simplify(g[i, j]) for j in range(n)] for i in range(n)])
        return cls(g, coords)

    def eval(self, *coord_vals):
        """
        Evaluate metric components at given points.
        
        Parameters
        ----------
        x_vals, y_vals : float or ndarray
            Coordinate values.
        
        Returns
        -------
        dict
            Dictionary containing metric tensors and geometric quantities.
        """
        n = self.dim
        result = {
            'g': np.array([[self.g_func[(i,j)](*coord_vals) for j in range(n)] for i in range(n)]),
            'g_inv': np.array([[self.g_inv_func[(i,j)](*coord_vals) for j in range(n)] for i in range(n)]),
            'sqrt_det': self.sqrt_det_g_func(*coord_vals),
        }
        if n == 1:
            result['christoffel'] = self.christoffel_func[0][0][0](*coord_vals)
        else:
            result['christoffel'] = {
                i: {j: {k: self.christoffel_func[i][j][k](*coord_vals)
                        for k in range(n)} for j in range(n)} for i in range(n)
            }
        return result

    # ========================
    # Curvature
    # ========================
    def gauss_curvature(self):
        """
        Compute Gaussian curvature K.
        
        For a 2D Riemannian manifold, the Gaussian curvature is:
            K = R₁₂₁₂ / |g|
        
        where R₁₂₁₂ is a component of the Riemann curvature tensor.
        
        Returns
        -------
        sympy expression
            Gaussian curvature K(x, y).
        
        Notes
        -----
        By Gauss-Bonnet theorem: ∫∫_M K dA = 2π χ(M)
        where χ is the Euler characteristic.
        
        Examples
        --------
        >>> x, y = symbols('x y', real=True)
        >>> g = Matrix([[1, 0], [0, 1]])
        >>> metric = Metric(g, (x, y))
        >>> print(metric.gauss_curvature())
        0
        """
        if self.dim == 1:
            return sympify(0)
        R = self.riemann_tensor()
        g = self.g_matrix
        R_xyxy = g[0,0] * R[0][1][0][1] + g[0,1] * R[1][1][0][1]
        return simplify(R_xyxy / self.det_g)

    def riemann_tensor(self):
        """
        Compute Riemann curvature tensor Rⁱⱼₖₗ.
        
        Returns
        -------
        dict
            Nested dict with all non-zero components.
        
        Notes
        -----
        In 2D, only one independent component exists (up to symmetries).
        """
        if self.dim == 1:
            return {}
        n = 2
        Gamma = self.christoffel
        R = {}
        for i in range(n):
            R[i] = {}
            for j in range(n):
                R[i][j] = {}
                for k in range(n):
                    R[i][j][k] = {}
                    for ell in range(n):
                        expr = diff(Gamma[i][j][ell], self.coords[k])
                        expr -= diff(Gamma[i][j][k], self.coords[ell])
                        for m in range(n):
                            expr += Gamma[i][m][k] * Gamma[m][j][ell]
                            expr -= Gamma[i][m][ell] * Gamma[m][j][k]
                        R[i][j][k][ell] = simplify(expr)
        return R

    def ricci_tensor(self):
        """
        Compute Ricci curvature tensor Rᵢⱼ.
        
        Rᵢⱼ = Rᵏᵢₖⱼ (contraction of Riemann tensor)
        
        Returns
        -------
        sympy Matrix
            2×2 Ricci tensor.
        """
        if self.dim == 1:
            return Matrix([[0]])
        R_full = self.riemann_tensor()
        n = 2
        Ric = zeros(n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Ric[i,j] += R_full[k][i][k][j]
        return simplify(Ric)

    def ricci_scalar(self):
        """
        Compute scalar curvature R.
        
        R = g^ij Rᵢⱼ
        
        For 2D surfaces: R = 2K (twice the Gaussian curvature).
        
        Returns
        -------
        sympy expression
            Scalar curvature R(x, y).
        """
        if self.dim == 1:
            return sympify(0)
        Ric = self.ricci_tensor()
        g_inv = self.g_inv_matrix
        R = sum(g_inv[i,j] * Ric[i,j] for i in range(2) for j in range(2))
        return simplify(R)

    # ========================
    # Laplace-Beltrami
    # ========================
    def laplace_beltrami_symbol(self):
        """
        Compute symbol of Laplace-Beltrami operator.
        
        Principal symbol: g^ij ξᵢ ξⱼ
        Subprincipal: transport terms from √|g| factor
        
        Returns
        -------
        dict
            Symbol components: 'principal', 'subprincipal', 'full'.
        
        Examples
        --------
        >>> x, y = symbols('x y', real=True)
        >>> g = Matrix([[1, 0], [0, 1]])
        >>> metric = Metric(g, (x, y))
        >>> symbol = metric.laplace_beltrami_symbol()
        >>> print(symbol['principal'])
        xi**2 + eta**2
        """
        n = self.dim
        if n == 1:
            x, = self.coords
            xi = symbols('xi', real=True)
            principal = self.g_inv_matrix[0,0] * xi**2
            log_sqrt_g = log(self.sqrt_det_g)
            transport = simplify(diff(log_sqrt_g, x) * self.g_inv_matrix[0,0])
            subprincipal = transport * xi
        elif n == 2:
            x, y = self.coords
            xi, eta = symbols('xi eta', real=True)
            g_inv = self.g_inv_matrix
            principal = (g_inv[0,0] * xi**2 +
                         2 * g_inv[0,1] * xi * eta +
                         g_inv[1,1] * eta**2)
            sqrt_g = self.sqrt_det_g
            coeff_x = diff(sqrt_g * g_inv[0,0], x) + diff(sqrt_g * g_inv[0,1], y)
            coeff_y = diff(sqrt_g * g_inv[1,0], x) + diff(sqrt_g * g_inv[1,1], y)
            subprincipal = simplify((coeff_x * xi + coeff_y * eta) / sqrt_g)
        else:
            raise NotImplementedError("Laplace-Beltrami symbol only implemented for 1D and 2D")
        
        return {
            'principal': simplify(principal),
            'subprincipal': simplify(subprincipal),
            'full': simplify(principal + I * subprincipal)
        }

    # ========================
    # Volume & length
    # ========================
    def riemannian_volume(self, domain, method='symbolic'):
        """
        Compute Riemannian volume of a domain.
        
        Vol(Ω) = ∫∫_Ω √|g| dx dy
        
        Parameters
        ----------
        domain : tuple
            For rectangular: ((x_min, x_max), (y_min, y_max)).
            For custom: callable that defines integration region.
        method : {'numerical', 'symbolic'}
            Integration method.
        
        Returns
        -------
        float or sympy expression
            Volume of the domain.
        """
        if self.dim == 1:
            a, b = domain
            if method == 'symbolic':
                return integrate(self.sqrt_det_g, (self.coords[0], a, b))
            else:
                f = lambdify(self.coords[0], self.sqrt_det_g, 'numpy')
                return quad(f, a, b)[0]
        else:
            (x0, x1), (y0, y1) = domain
            if method == 'symbolic':
                x, y = self.coords
                return integrate(self.sqrt_det_g, (x, x0, x1), (y, y0, y1))
            else:
                f = lambdify(self.coords, self.sqrt_det_g, 'numpy')
                return dblquad(lambda Y, X: f(X, Y), x0, x1, y0, y1)[0]

    def arc_length(self, a, b, method='numerical'):
        """
        Compute arc length between two points.
        
        L = ∫ₐᵇ √g₁₁(x) dx
        
        Parameters
        ----------
        x_min, x_max : float
            Endpoints.
        method : {'symbolic', 'numerical'}
            Computation method.
        
        Returns
        -------
        float
            Arc length.
        """
        if self.dim != 1:
            raise ValueError("Arc length defined only for 1D")
        return self.riemannian_volume((a, b), method=method)

    # ========================
    # Sturm-Liouville (1D)
    # ========================
    def sturm_liouville_reduce(self, potential_expr=None):
        """
        Reduce Laplace-Beltrami to Sturm-Liouville form.
        
        The eigenvalue problem:
            -Δg u + V(x) u = λ u
        
        can be written in Sturm-Liouville form:
            -(p(x) u')' + q(x) u = λ w(x) u
        
        Parameters
        ----------
        metric : Metric1D
            Riemannian metric.
        potential_expr : sympy expression, optional
            Potential function V(x).
        
        Returns
        -------
        dict
            Dictionary with 'p', 'q', 'w' (coefficient functions).
        
        Notes
        -----
        For the Laplace-Beltrami:
            p(x) = √g g¹¹
            q(x) = V(x) √g
            w(x) = √g
        
        Examples
        --------
        >>> x = symbols('x', real=True)
        >>> metric = Metric(x**2, x)
        >>> sl = sturm_liouville_reduce(metric)
        >>> print(sl['p'])
        1/x
        """
        if self.dim != 1:
            raise NotImplementedError("Sturm-Liouville form only for 1D")
        x, = self.coords
        sqrt_g = self.sqrt_det_g
        g_inv = self.g_inv_matrix[0,0]
        p_expr = simplify(sqrt_g * g_inv)
        w_expr = sqrt_g
        q_expr = simplify(potential_expr * sqrt_g) if potential_expr else sympify(0)
        return {
            'p': p_expr, 'q': q_expr, 'w': w_expr,
            'p_func': lambdify(x, p_expr, 'numpy'),
            'q_func': lambdify(x, q_expr, 'numpy'),
            'w_func': lambdify(x, w_expr, 'numpy')
        }

# ============================================================================
# Unified functions (dispatched by dimension)
# ============================================================================
def geodesic_solver(metric, p0, v0, tspan, method='rk45', n_steps=1000, reparametrize=False):
    """
    Integrate geodesic equations on 2D manifold.
    
    Geodesic equation:
        ẍⁱ + Γⁱⱼₖ ẋʲ ẋᵏ = 0
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    p0 : tuple
        Initial position (x₀, y₀).
    v0 : tuple
        Initial velocity (vₓ₀, vᵧ₀).
    tspan : tuple
        Time interval (t_start, t_end).
    method : str
        Integration method: 'rk45', 'rk4', 'symplectic', 'verlet'.
    n_steps : int
        Number of steps.
    reparametrize : bool
        If True, reparametrize by arc length.
    
    Returns
    -------
    dict
        Trajectory with 't', 'x', 'y', 'vx', 'vy' arrays.
    
    Examples
    --------
    >>> x, y = symbols('x y', real=True)
    >>> g = Matrix([[1, 0], [0, 1]])
    >>> metric = Metric2D(g, (x, y))
    >>> traj = geodesic_solver(metric, (0, 0), (1, 1), (0, 10))
    >>> plt.plot(traj['x'], traj['y'])
    """
    from scipy.integrate import solve_ivp
    if metric.dim == 1:
        x0, = p0
        v0x, = v0
        Gamma = metric.christoffel_func[0][0][0]
        def ode(t, y):
            x, vx = y
            return [vx, -Gamma(x) * vx**2]
        sol = solve_ivp(ode, tspan, [x0, v0x], t_eval=np.linspace(*tspan, n_steps))
        result = {'t': sol.t, 'x': sol.y[0], 'v': sol.y[1]}
        if reparametrize:
            ds = metric.sqrt_det_g_func(result['x']) * np.abs(result['v'])
            result['arc_length'] = cumulative_trapezoid(ds, result['t'], initial=0)
        return result
    else:
        Gamma = metric.christoffel_func
        def ode(t, y):
            x, yv, vx, vy = y
            ax = -(Gamma[0][0][0](x, yv) * vx**2 +
                   2 * Gamma[0][0][1](x, yv) * vx * vy +
                   Gamma[0][1][1](x, yv) * vy**2)
            ay = -(Gamma[1][0][0](x, yv) * vx**2 +
                   2 * Gamma[1][0][1](x, yv) * vx * vy +
                   Gamma[1][1][1](x, yv) * vy**2)
            return [vx, vy, ax, ay]
        sol = solve_ivp(ode, tspan, [*p0, *v0], t_eval=np.linspace(*tspan, n_steps))
        result = {'t': sol.t, 'x': sol.y[0], 'y': sol.y[1], 'vx': sol.y[2], 'vy': sol.y[3]}
        if reparametrize:
            ds = np.sqrt(
                metric.g_func[(0,0)](result['x'], result['y']) * result['vx']**2 +
                2 * metric.g_func[(0,1)](result['x'], result['y']) * result['vx'] * result['vy'] +
                metric.g_func[(1,1)](result['x'], result['y']) * result['vy']**2
            )
            result['arc_length'] = cumulative_trapezoid(ds, result['t'], initial=0)
        return result

def geodesic_hamiltonian_flow(metric, p0, v0_or_p0, tspan, method='verlet', n_steps=1000, momentum_input=False):
    """
    Integrate geodesic flow in Hamiltonian formulation.
    
    Hamiltonian: H = ½ g^ij pᵢ pⱼ
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    p0 : tuple
        Initial position (x₀, y₀).
    v0 : tuple
        Initial velocity (vₓ₀, vᵧ₀).
    tspan : tuple
        Time interval.
    method : str
        Symplectic integrator.
    n_steps : int
        Number of steps.
    
    Returns
    -------
    dict
        Phase space trajectory with energy.
    """
    dt = (tspan[1] - tspan[0]) / n_steps
    t_vals = np.linspace(tspan[0], tspan[1], n_steps)
    if metric.dim == 1:
        x_vals = np.zeros(n_steps)
        p_vals = np.zeros(n_steps)
        x_vals[0] = p0[0]
        if momentum_input:
            p_vals[0] = v0_or_p0[0]
        else:
            g_eval = metric.eval(p0[0])['g'][0,0]
            p_vals[0] = g_eval * v0_or_p0[0]
        x_sym, = metric.coords
        g_inv = metric.g_inv_matrix[0,0]
        dg_inv_dx = lambdify(x_sym, diff(g_inv, x_sym), 'numpy')
        for i in range(n_steps - 1):
            x, p = x_vals[i], p_vals[i]
            Fx = -0.5 * dg_inv_dx(x) * p**2
            if method == 'verlet':
                p_half = p + 0.5 * dt * Fx
                x_new = x + dt * metric.g_inv_func[(0,0)](x) * p_half
                Fx_new = -0.5 * dg_inv_dx(x_new) * p_half**2
                p_new = p_half + 0.5 * dt * Fx_new
                x_vals[i+1], p_vals[i+1] = x_new, p_new
        v_vals = np.array([metric.g_inv_func[(0,0)](x) * p for x, p in zip(x_vals, p_vals)])
        energy = 0.5 * np.array([metric.g_inv_func[(0,0)](x) * p**2 for x, p in zip(x_vals, p_vals)])
        return {'t': t_vals, 'x': x_vals, 'v': v_vals, 'p': p_vals, 'energy': energy}
    else:
        # Reuse logic from riemannian_2d.py with minor refactoring
        x_vals, y_vals = np.zeros(n_steps), np.zeros(n_steps)
        px_vals, py_vals = np.zeros(n_steps), np.zeros(n_steps)
        x_vals[0], y_vals[0] = p0
        if momentum_input:
            px_vals[0], py_vals[0] = v0_or_p0
        else:
            g_eval = metric.eval(*p0)['g']
            px_vals[0] = g_eval[0][0] * v0_or_p0[0] + g_eval[0][1] * v0_or_p0[1]
            py_vals[0] = g_eval[1][0] * v0_or_p0[0] + g_eval[1][1] * v0_or_p0[1]
        x_sym, y_sym = metric.coords
        g_inv = metric.g_inv_matrix
        dg_inv_dx = {(i,j): lambdify((x_sym, y_sym), diff(g_inv[i,j], x_sym), 'numpy')
                     for i in range(2) for j in range(2)}
        dg_inv_dy = {(i,j): lambdify((x_sym, y_sym), diff(g_inv[i,j], y_sym), 'numpy')
                     for i in range(2) for j in range(2)}
        for i in range(n_steps - 1):
            x, y = x_vals[i], y_vals[i]
            px, py = px_vals[i], py_vals[i]
            g_inv_vals = metric.eval(x, y)['g_inv']
            Fx = -0.5 * (dg_inv_dx[(0,0)](x, y) * px**2 +
                         2 * dg_inv_dx[(0,1)](x, y) * px * py +
                         dg_inv_dx[(1,1)](x, y) * py**2)
            Fy = -0.5 * (dg_inv_dy[(0,0)](x, y) * px**2 +
                         2 * dg_inv_dy[(0,1)](x, y) * px * py +
                         dg_inv_dy[(1,1)](x, y) * py**2)
            if method == 'verlet':
                px_half = px + 0.5 * dt * Fx
                py_half = py + 0.5 * dt * Fy
                vx = g_inv_vals[0][0] * px_half + g_inv_vals[0][1] * py_half
                vy = g_inv_vals[1][0] * px_half + g_inv_vals[1][1] * py_half
                x_new = x + dt * vx
                y_new = y + dt * vy
                Fx_new = -0.5 * (dg_inv_dx[(0,0)](x_new, y_new) * px_half**2 +
                                 2 * dg_inv_dx[(0,1)](x_new, y_new) * px_half * py_half +
                                 dg_inv_dx[(1,1)](x_new, y_new) * py_half**2)
                Fy_new = -0.5 * (dg_inv_dy[(0,0)](x_new, y_new) * px_half**2 +
                                 2 * dg_inv_dy[(0,1)](x_new, y_new) * px_half * py_half +
                                 dg_inv_dy[(1,1)](x_new, y_new) * py_half**2)
                px_new = px_half + 0.5 * dt * Fx_new
                py_new = py_half + 0.5 * dt * Fy_new
                x_vals[i+1], y_vals[i+1] = x_new, y_new
                px_vals[i+1], py_vals[i+1] = px_new, py_new
        energy = np.zeros(n_steps)
        vx_vals, vy_vals = np.zeros(n_steps), np.zeros(n_steps)
        for i in range(n_steps):
            g_inv_vals = metric.eval(x_vals[i], y_vals[i])['g_inv']
            energy[i] = 0.5 * (g_inv_vals[0][0] * px_vals[i]**2 +
                               2 * g_inv_vals[0][1] * px_vals[i] * py_vals[i] +
                               g_inv_vals[1][1] * py_vals[i]**2)
            vx_vals[i] = g_inv_vals[0][0] * px_vals[i] + g_inv_vals[0][1] * py_vals[i]
            vy_vals[i] = g_inv_vals[1][0] * px_vals[i] + g_inv_vals[1][1] * py_vals[i]
        return {'t': t_vals, 'x': x_vals, 'y': y_vals,
                'vx': vx_vals, 'vy': vy_vals,
                'px': px_vals, 'py': py_vals, 'energy': energy}

def exponential_map(metric, p, v, t=1.0):
    """
    Compute exponential map exp_p(tv).
    
    The exponential map sends a tangent vector v at point p to the
    point reached by following the geodesic with initial velocity v
    for parameter time t.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    p : tuple
        Base point (x₀, y₀).
    v : tuple
Initial tangent vector (vₓ, vᵧ).
    t : float
        Parameter value (geodesic "time").
    method : str
        Integration method.
    
    Returns
    -------
    tuple
        End point (x(t), y(t)).
    
    Examples
    --------
    >>> x, y = symbols('x y', real=True)
    >>> g = Matrix([[1, 0], [0, 1]])
    >>> metric = Metric(g, (x, y))
    >>> q = exponential_map(metric, (0, 0), (1, 1), t=1.0)
    >>> print(q)  # Should be (1, 1) for flat metric
    """
    traj = geodesic_solver(metric, p, v, (0, t), n_steps=100)
    if metric.dim == 1:
        return traj['x'][-1]
    else:
        return (traj['x'][-1], traj['y'][-1])

def distance(metric, p, q, method='shooting', max_iter=50, tol=1e-6):
    """
    Compute geodesic distance between two points.
    
    Uses shooting method: find initial velocity v such that
    exp_p(v) = q, then distance = ||v||_g.
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric.
    p, q : tuple
        Points (x₁, y₁) and (x₂, y₂).
    method : {'shooting', 'optimize'}
        Distance computation method.
    max_iter : int
        Maximum iterations for shooting.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    float
        Geodesic distance d(p, q).
    
    Notes
    -----
    - 'shooting': iteratively adjust initial velocity.
    - 'optimize': minimize functional over curves.
    """
    if metric.dim == 1:
        # In 1D, distance = ∫_p^q √g dx
        if method == 'exact':
            x = metric.coords[0]
            integrand = metric.sqrt_det_g
            return float(integrate(integrand, (x, p[0], q[0])))
        else:
            from scipy.integrate import quad
            f = metric.sqrt_det_g_func
            return quad(f, p[0], q[0])[0]
    else:
        if method == 'shooting':
            # Initial guess: straight line velocity
            v_guess = np.array([q[0] - p[0], q[1] - p[1]])
            
            for iteration in range(max_iter):
                # Shoot geodesic
                q_reached = exponential_map(metric, p, tuple(v_guess), t=1.0)
                
                # Compute error
                error = np.array([q_reached[0] - q[0], q_reached[1] - q[1]])
                error_norm = np.linalg.norm(error)
                
                if error_norm < tol:
                    break
                
                # Update velocity (simple gradient descent)
                v_guess -= 0.5 * error
            
            # Compute distance as norm of velocity
            g_eval = metric.eval(p[0], p[1])
            dist_sq = (g_eval['g'][0,0] * v_guess[0]**2 +
                       2 * g_eval['g'][0,1] * v_guess[0] * v_guess[1] +
                       g_eval['g'][1,1] * v_guess[1]**2)
            
            return np.sqrt(dist_sq)
        
        elif method == 'optimize':
            from scipy.optimize import minimize
            
            def energy_functional(v):
                """Energy of geodesic with initial velocity v."""
                q_reached = exponential_map(metric, p, tuple(v), t=1.0)
                endpoint_error = ((q_reached[0] - q[0])**2 + 
                                (q_reached[1] - q[1])**2)
                
                # Energy = ½ ||v||²_g + penalty for not reaching q
                g_eval = metric.eval(p[0], p[1])
                energy = 0.5 * (g_eval['g'][0,0] * v[0]**2 +
                               2 * g_eval['g'][0,1] * v[0] * v[1] +
                               g_eval['g'][1,1] * v[1]**2)
                
                return energy + 1e6 * endpoint_error
            
            v_init = np.array([q[0] - p[0], q[1] - p[1]])
            result = minimize(energy_functional, v_init, method='BFGS')
            
            return np.sqrt(2 * result.fun)
        
        else:
            raise ValueError("method must be 'shooting' or 'optimize'")

# ============================================================================
# 2D-only functions (with guard)
# ============================================================================
def hodge_star(metric, form_degree):
    """
    Compute Hodge star operator on differential forms.
    
    For a k-form ω, *ω is an (n-k)-form where n = 2.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    form_degree : int
        Degree of the form (0, 1, or 2).
    
    Returns
    -------
    callable or Matrix
        Hodge star operator.
    
    Notes
    -----
    - On 0-forms: *f = f √|g| dx∧dy
    - On 1-forms: *(f dx + g dy) = (g √|g|) dx - (f √|g|) dy
    - On 2-forms: *(f dx∧dy) = f/√|g|
    
    Examples
    --------
    >>> x, y = symbols('x y', real=True)
    >>> metric = Metric2D(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> star1 = hodge_star(metric, 1)
    """
    if metric.dim != 2:
        raise ValueError("Hodge star requires 2D manifold")
    sqrt_g = metric.sqrt_det_g
    g_inv = metric.g_inv_matrix
    x, y = metric.coords
    if form_degree == 0:
        return lambda f: f * sqrt_g
    elif form_degree == 1:
        def star(alpha_x, alpha_y):
            beta_x = (g_inv[0,0] * alpha_y - g_inv[0,1] * alpha_x) * sqrt_g
            beta_y = (-g_inv[0,1] * alpha_y + g_inv[1,1] * alpha_x) * sqrt_g
            return (beta_x, beta_y)
        return star
    elif form_degree == 2:
        return lambda f: f / sqrt_g
    else:
        raise ValueError("form_degree must be 0, 1, or 2")

def jacobi_equation_solver(metric, geodesic, initial_variation, tspan, 
                           n_steps=1000):
    """
    Solve Jacobi equation for geodesic deviation.
    
    The Jacobi equation describes how nearby geodesics separate:
        D²J/dt² + R(J, γ̇)γ̇ = 0
    
    where J is the Jacobi field (deviation vector) and R is the
    Riemann curvature tensor.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    geodesic : dict
        Base geodesic trajectory.
    initial_variation : dict
        Initial conditions: 'J0' (position), 'DJ0' (velocity).
    tspan : tuple
        Time interval.
    n_steps : int
        Number of steps.
    
    Returns
    -------
    dict
        Jacobi field: 'J_x', 'J_y', 'DJ_x', 'DJ_y'.
    
    Notes
    -----
    Used to study stability of geodesics and focal points.
    
    Examples
    --------
    >>> # Compute geodesic deviation on sphere
    >>> metric = Metric2D(...)  # Sphere metric
    >>> geod = geodesic_solver(metric, (0, 0), (1, 0), (0, np.pi))
    >>> J = jacobi_equation_solver(metric, geod, 
    ...                            {'J0': (0, 0.1), 'DJ0': (0, 0)},
    ...                            (0, np.pi))
    """
    from scipy.integrate import solve_ivp
    
    # Compute Riemann tensor symbolically
    R = metric.riemann_tensor()
    
    # Lambdify Riemann tensor components
    x_sym, y_sym = metric.vars_xy
    R_func = {}
    for i in range(2):
        R_func[i] = {}
        for j in range(2):
            R_func[i][j] = {}
            for k in range(2):
                R_func[i][j][k] = {}
                for ell in range(2):
                    R_func[i][j][k][ell] = lambdify(
                        (x_sym, y_sym), R[i][j][k][ell], 'numpy'
                    )
    
    # Interpolate geodesic
    from scipy.interpolate import interp1d
    t_geod = geodesic['t']
    x_interp = interp1d(t_geod, geodesic['x'], kind='cubic')
    y_interp = interp1d(t_geod, geodesic['y'], kind='cubic')
    vx_interp = interp1d(t_geod, geodesic['vx'], kind='cubic')
    vy_interp = interp1d(t_geod, geodesic['vy'], kind='cubic')
    
    Gamma = metric.christoffel_func
    
    def jacobi_ode(t, state):
        """
        Jacobi equation as first-order system.
        State = [J_x, J_y, DJ_x, DJ_y]
        """
        J_x, J_y, DJ_x, DJ_y = state
        
        # Evaluate along base geodesic
        x = x_interp(t)
        y = y_interp(t)
        vx = vx_interp(t)
        vy = vy_interp(t)
        
        # Curvature term: R(J, v)v
        # (R(J, v)v)^i = R^i_jkl J^j v^k v^l
        curv_x = 0
        curv_y = 0
        
        J = [J_x, J_y]
        v = [vx, vy]
        
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    curv_x += R_func[0][j][k][ell](x, y) * J[j] * v[k] * v[ell]
                    curv_y += R_func[1][j][k][ell](x, y) * J[j] * v[k] * v[ell]
        
        # Covariant derivative: D(DJ)/dt = d(DJ)/dt - Γ(DJ, v)
        DDJ_x = -(Gamma[0][0][0](x, y) * DJ_x * vx +
                  Gamma[0][0][1](x, y) * (DJ_x * vy + DJ_y * vx) +
                  Gamma[0][1][1](x, y) * DJ_y * vy +
                  curv_x)
        
        DDJ_y = -(Gamma[1][0][0](x, y) * DJ_x * vx +
                  Gamma[1][0][1](x, y) * (DJ_x * vy + DJ_y * vx) +
                  Gamma[1][1][1](x, y) * DJ_y * vy +
                  curv_y)
        
        return [DJ_x, DJ_y, DDJ_x, DDJ_y]
    
    # Solve Jacobi equation
    J0 = initial_variation['J0']
    DJ0 = initial_variation['DJ0']
    
    sol = solve_ivp(
        jacobi_ode,
        tspan,
        [J0[0], J0[1], DJ0[0], DJ0[1]],
        t_eval=np.linspace(tspan[0], tspan[1], n_steps)
    )
    
    return {
        't': sol.t,
        'J_x': sol.y[0],
        'J_y': sol.y[1],
        'DJ_x': sol.y[2],
        'DJ_y': sol.y[3]
    }

def verify_gauss_bonnet(metric, domain):
    """
    Numerically verify Gauss-Bonnet theorem.
    
    Theorem: ∫∫_M K dA = 2π χ(M)
    
    where K is Gaussian curvature, dA is area element, and χ is
    Euler characteristic.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    domain : tuple
        Integration domain ((x_min, x_max), (y_min, y_max)).
    resolution : int
        Grid resolution.
    
    Returns
    -------
    dict
        Results: 'integral', 'expected' (2π χ), 'error'.
    
    Examples
    --------
    >>> # Sphere: χ = 2, so integral should be 4π
    >>> x, y = symbols('x y', real=True)
    >>> g_sphere = Matrix([[1, 0], [0, sin(x)**2]])
    >>> metric = Metric(g_sphere, (x, y))
    >>> result = verify_gauss_bonnet(metric, ((0.01, np.pi-0.01), (0, 2*np.pi)))
    >>> print(f"Integral: {result['integral']:.4f}, Expected: {result['expected']:.4f}")
    """
    if metric.dim != 2:
        raise ValueError("Gauss-Bonnet requires 2D manifold")
    from scipy.integrate import dblquad
    K = metric.gauss_curvature()
    integrand = K * metric.sqrt_det_g
    f = lambdify(metric.coords, integrand, 'numpy')
    (x0, x1), (y0, y1) = domain
    integral, _ = dblquad(lambda Y, X: f(X, Y), x0, x1, y0, y1)
    return {'integral': integral, 'expected': 2 * np.pi, 'relative_error': abs(integral - 2*np.pi)/(2*np.pi)}

# ============================================================================
# Visualization
# ============================================================================
def visualize_geodesics(metric, initial_conditions, tspan, **kwargs):
    """
    Visualize geodesics on 2D manifold with optional curvature background.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    initial_conditions : list of tuples
        List of ((x₀, y₀), (vₓ₀, vᵧ₀)).
    tspan : tuple
        Time interval.
    x_range, y_range : tuple, optional
        Plot ranges (inferred if None).
    plot_curvature : bool
        Show Gaussian curvature as background.
    n_steps : int
        Integration steps per geodesic.
    
    Examples
    --------
    >>> x = symbols('x', real=True)
    >>> metric = Metric(1 + x**2, x)
    >>> ics = [(0, 1), (0, 2), (1, 1)]
    >>> visualize_geodesics(metric, ics, (0, 10))
    >>> x, y = symbols('x y', real=True)
    >>> g = Matrix([[1, 0], [0, 1 + x**2]])
    >>> metric = Metric(g, (x, y))
    >>> ics = [((0, 0), (1, 0)), ((0, 0), (0.7, 0.7))]
    >>> visualize_geodesics(metric, ics, (0, 5))
    """
    if metric.dim == 1:
        # Reuse 1D logic
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        x_plot = np.linspace(*kwargs.get('x_range', (0, 5)), 200)
        ax1.plot(x_plot, metric.g_func[(0,0)](x_plot), 'k-', label='g₁₁(x)')
        for x0, v0 in initial_conditions:
            traj = geodesic_solver(metric, (x0,), (v0,), tspan)
            ax2.plot(traj['t'], traj['x'], label=f'x₀={x0}, v₀={v0}')
        ax1.legend(); ax2.legend(); ax2.grid(True)
        plt.show()
    else:
        # Reuse 2D logic (with curvature background if possible)
        fig, ax = plt.subplots()
        trajectories = [geodesic_solver(metric, p0, v0, tspan) for p0, v0 in initial_conditions]
        for traj in trajectories:
            ax.plot(traj['x'], traj['y'])
        ax.set_aspect('equal')
        plt.show()

def visualize_curvature(metric, x_range, y_range, resolution=50, 
                       quantity='gauss', cmap='RdBu_r'):
    """
    Visualize curvature on 2D manifold.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    x_range, y_range : tuple
        Domain ranges.
    resolution : int
        Grid resolution.
    quantity : str
        Curvature to plot: 'gauss', 'ricci_scalar', 'ricci_tensor'.
    cmap : str
        Colormap name.
    
    Examples
    --------
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', real=True, positive=True)
    >>> g_sphere = Matrix([[1, 0], [0, sin(x)**2]])
    >>> metric = Metric2D(g_sphere, (x, y))
    >>> visualize_curvature(metric, (0.1, np.pi-0.1), (0, 2*np.pi))
    """
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    if quantity == 'gauss':
        K_expr = metric.gauss_curvature()
        K_func = lambdify(metric.vars_xy, K_expr, 'numpy')
        Z = K_func(X, Y)
        title = 'Gaussian Curvature K(x,y)'
    
    elif quantity == 'ricci_scalar':
        R_expr = metric.ricci_scalar()
        R_func = lambdify(metric.vars_xy, R_expr, 'numpy')
        Z = R_func(X, Y)
        title = 'Ricci Scalar R(x,y)'
    
    else:
        raise ValueError("quantity must be 'gauss' or 'ricci_scalar'")
    
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
    plt.colorbar(label=title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ============================================================================
# Backward compatibility (optional)
# ============================================================================
class Metric1D(Metric):
    def __init__(self, g_expr, x):
        super().__init__(g_expr, (x,))

class Metric2D(Metric):
    def __init__(self, g_matrix, coords):
        super().__init__(g_matrix, coords)
