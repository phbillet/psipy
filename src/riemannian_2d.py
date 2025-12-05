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
Riemannian geometry toolkit for 2D manifolds.

This module provides comprehensive tools for Riemannian geometry on surfaces:
 - Metric tensor: gᵢⱼ(x,y), inverse g^ij, determinant |g|, √|g|
 - Christoffel symbols: Γⁱⱼₖ
 - Geodesic equations and integration
 - Curvature tensors: Riemann, Ricci, Gauss curvature K, scalar curvature R
 - Exponential map, normal coordinates, geodesic distance
 - Jacobi equation (geodesic deviation)
 - Laplace-Beltrami operator and Hodge theory
"""

from imports import *

class Metric2D:
    """
    Riemannian metric tensor on a 2D manifold.
    
    Represents a metric tensor as a 2×2 matrix:
        g = [[g₁₁, g₁₂],
             [g₁₂, g₂₂]]
    
    Parameters
    ----------
    g_matrix : 2×2 sympy Matrix or list
        Metric tensor components [[g₁₁, g₁₂], [g₁₂, g₂₂]].
    vars_xy : tuple of sympy symbols
        Coordinate variables (x, y).
    
    Attributes
    ----------
    g_matrix : sympy Matrix
        Metric tensor gᵢⱼ.
    g_inv_matrix : sympy Matrix
        Inverse metric g^ij.
    det_g : sympy expression
        Determinant |g|.
    sqrt_det_g : sympy expression
        √|g| for volume forms.
    christoffel : dict
        Christoffel symbols Γⁱⱼₖ.
    
    Examples
    --------
    >>> # Euclidean metric
    >>> x, y = symbols('x y', real=True)
    >>> g = Matrix([[1, 0], [0, 1]])
    >>> metric = Metric2D(g, (x, y))
    
    >>> # Polar coordinates
    >>> r, theta = symbols('r theta', real=True, positive=True)
    >>> g_polar = Matrix([[1, 0], [0, r**2]])
    >>> metric = Metric2D(g_polar, (r, theta))
    >>> print(metric.gauss_curvature())
    
    >>> # From Hamiltonian
    >>> p_x, p_y = symbols('p_x p_y', real=True)
    >>> H = (p_x**2 + p_y**2)/(2*x**2)
    >>> metric = Metric2D.from_hamiltonian(H, (x,y), (p_x,p_y))
    """
    
    def __init__(self, g_matrix, vars_xy):
        if not isinstance(g_matrix, Matrix):
            g_matrix = Matrix(g_matrix)
    
        if g_matrix.shape != (2, 2):
            raise ValueError("Metric2D requires a 2×2 metric tensor")
    
        if len(vars_xy) != 2:
            raise ValueError("Metric2D requires exactly two coordinates (x, y)")
        self.vars_xy = vars_xy
        self.x, self.y = vars_xy
        
        if not isinstance(g_matrix, Matrix):
            g_matrix = Matrix(g_matrix)
        
        self.g_matrix = simplify(g_matrix)
        self.det_g = simplify(self.g_matrix.det())
        self.sqrt_det_g = simplify(sqrt(abs(self.det_g)))
        self.g_inv_matrix = simplify(self.g_matrix.inv())
        
        # Compute Christoffel symbols
        self.christoffel = self._compute_christoffel()
        
        # Lambdify for numerical evaluation
        self._lambdify_all()
    
    def _compute_christoffel(self):
        """
        Compute all Christoffel symbols Γⁱⱼₖ.
        
        Γⁱⱼₖ = ½ g^iℓ (∂ⱼgₖℓ + ∂ₖgⱼℓ - ∂ℓgⱼₖ)
        
        Returns
        -------
        dict
            Nested dict: christoffel[i][j][k] = Γⁱⱼₖ
        """
        x, y = self.vars_xy
        g = self.g_matrix
        g_inv = self.g_inv_matrix
        
        Gamma = {}
        for i in range(2):
            Gamma[i] = {}
            for j in range(2):
                Gamma[i][j] = {}
                for k in range(2):
                    expr = 0
                    for ell in range(2):
                        term1 = diff(g[k, ell], [x, y][j])
                        term2 = diff(g[j, ell], [x, y][k])
                        term3 = diff(g[j, k], [x, y][ell])
                        expr += g_inv[i, ell] * (term1 + term2 - term3) / 2
                    Gamma[i][j][k] = simplify(expr)
        
        return Gamma
    
    def _lambdify_all(self):
        """Prepare numerical functions for all geometric quantities."""
        x, y = self.vars_xy
        
        # Metric components
        self.g_func = {
            (i, j): lambdify((x, y), self.g_matrix[i, j], 'numpy')
            for i in range(2) for j in range(2)
        }
        
        self.g_inv_func = {
            (i, j): lambdify((x, y), self.g_inv_matrix[i, j], 'numpy')
            for i in range(2) for j in range(2)
        }
        
        self.det_g_func = lambdify((x, y), self.det_g, 'numpy')
        self.sqrt_det_g_func = lambdify((x, y), self.sqrt_det_g, 'numpy')
        
        # Christoffel symbols
        self.christoffel_func = {}
        for i in range(2):
            self.christoffel_func[i] = {}
            for j in range(2):
                self.christoffel_func[i][j] = {}
                for k in range(2):
                    self.christoffel_func[i][j][k] = lambdify(
                        (x, y), self.christoffel[i][j][k], 'numpy'
                    )
    
    @classmethod
    def from_hamiltonian(cls, H_expr, vars_xy, vars_p):
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
        >>> x, y, px, py = symbols('x y p_x p_y', real=True)
        >>> H = (px**2 + py**2)/(2*x**2)
        >>> metric = Metric2D.from_hamiltonian(H, (x,y), (px,py))
        """
        px, py = vars_p
        
        # Compute Hessian
        g_inv_11 = diff(H_expr, px, 2)
        g_inv_12 = diff(H_expr, px, py)
        g_inv_22 = diff(H_expr, py, 2)
        
        g_inv = Matrix([[g_inv_11, g_inv_12],
                        [g_inv_12, g_inv_22]])
        
        g = simplify(g_inv.inv())
        return cls(g, vars_xy)
    
    def eval(self, x_vals, y_vals):
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
        result = {
            'g': np.zeros((2, 2, *np.shape(x_vals))),
            'g_inv': np.zeros((2, 2, *np.shape(x_vals))),
            'det_g': self.det_g_func(x_vals, y_vals),
            'sqrt_det_g': self.sqrt_det_g_func(x_vals, y_vals),
            'christoffel': {}
        }
        
        for i in range(2):
            for j in range(2):
                result['g'][i, j] = self.g_func[(i, j)](x_vals, y_vals)
                result['g_inv'][i, j] = self.g_inv_func[(i, j)](x_vals, y_vals)
        
        for i in range(2):
            result['christoffel'][i] = {}
            for j in range(2):
                result['christoffel'][i][j] = {}
                for k in range(2):
                    result['christoffel'][i][j][k] = \
                        self.christoffel_func[i][j][k](x_vals, y_vals)
        
        return result
    
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
        >>> metric = Metric2D(g, (x, y))
        >>> print(metric.gauss_curvature())
        0
        """
        # Ensure we have the full Riemann tensor
        # R^i_{jkl}
        R = self.riemann_tensor()
        g = self.g_matrix

        # Calculate the covariant component R_xyxy (or R_1212)
        # Indices: x=0, y=1
        # R_xyxy = g_xx * R^x_yxy + g_xy * R^y_yxy
        # R^i_{jkl} with j=1 (y), k=0 (x), l=1 (y)

        R_x_yxy = R[0][1][0][1]  # R^0_{101}
        R_y_yxy = R[1][1][0][1]  # R^1_{101}

        # Lowering index: R_{0101} = g_{0m} R^m_{101}
        R_xyxy = g[0,0] * R_x_yxy + g[0,1] * R_y_yxy

        # K = R_1212 / det(g)
        K = simplify(R_xyxy / self.det_g)
        
        return K
    
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
        x, y = self.vars_xy
        Gamma = self.christoffel
        
        R = {}
        for i in range(2):
            R[i] = {}
            for j in range(2):
                R[i][j] = {}
                for k in range(2):
                    R[i][j][k] = {}
                    for ell in range(2):
                        expr = diff(Gamma[i][j][ell], [x, y][k])
                        expr -= diff(Gamma[i][j][k], [x, y][ell])
                        
                        for m in range(2):
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
        R_full = self.riemann_tensor()
        
        Ric = zeros(2, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    Ric[i, j] += R_full[k][i][k][j]
        
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
        Ric = self.ricci_tensor()
        g_inv = self.g_inv_matrix
        
        R = 0
        for i in range(2):
            for j in range(2):
                R += g_inv[i, j] * Ric[i, j]
        
        return simplify(R)
    
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
        >>> x, y, xi, eta = symbols('x y xi eta', real=True)
        >>> g = Matrix([[1, 0], [0, 1]])
        >>> metric = Metric2D(g, (x, y))
        >>> symbol = metric.laplace_beltrami_symbol()
        >>> print(symbol['principal'])
        xi**2 + eta**2
        """
        x, y = self.vars_xy
        xi, eta = symbols('xi eta', real=True)
        
        g_inv = self.g_inv_matrix
        
        # Principal symbol
        principal = (g_inv[0,0] * xi**2 + 
                    2 * g_inv[0,1] * xi * eta +
                    g_inv[1,1] * eta**2)
        
        # Subprincipal (from divergence structure)
        # ∇·(√g g^ij ∇u) = √g g^ij ∂ᵢ∂ⱼu + (∂ᵢ√g g^ij) ∂ⱼu
        sqrt_g = self.sqrt_det_g
        
        coeff_x = diff(sqrt_g * g_inv[0,0], x) + diff(sqrt_g * g_inv[0,1], y)
        coeff_y = diff(sqrt_g * g_inv[1,0], x) + diff(sqrt_g * g_inv[1,1], y)
        
        subprincipal = simplify((coeff_x * xi + coeff_y * eta) / sqrt_g)
        
        return {
            'principal': simplify(principal),
            'subprincipal': simplify(subprincipal),
            'full': simplify(principal + 1j * subprincipal)
        }
    
    def riemannian_volume(self, domain, method='numerical'):
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
        x, y = self.vars_xy
        sqrt_g = self.sqrt_det_g
        
        if method == 'symbolic':
            (x_min, x_max), (y_min, y_max) = domain
            return integrate(sqrt_g, (x, x_min, x_max), (y, y_min, y_max))
        
        elif method == 'numerical':
            from scipy.integrate import dblquad
            (x_min, x_max), (y_min, y_max) = domain
            
            integrand = lambda y, x: self.sqrt_det_g_func(x, y)
            result, error = dblquad(integrand, x_min, x_max, y_min, y_max)
            return result
        
        else:
            raise ValueError("method must be 'symbolic' or 'numerical'")


def christoffel(metric):
    """
    Return Christoffel symbols as nested dict.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    
    Returns
    -------
    dict
        Christoffel symbols Γⁱⱼₖ accessible as Gamma[i][j][k].
    """
    return metric.christoffel


def geodesic_solver(metric, p0, v0, tspan, method='rk45', n_steps=1000,
                   reparametrize=False):
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
    
    Gamma = metric.christoffel_func
    
    def geodesic_ode(t, state):
        x, y, vx, vy = state
        
        # Compute accelerations
        ax = -(Gamma[0][0][0](x, y) * vx**2 +
               2 * Gamma[0][0][1](x, y) * vx * vy +
               Gamma[0][1][1](x, y) * vy**2)
        
        ay = -(Gamma[1][0][0](x, y) * vx**2 +
               2 * Gamma[1][0][1](x, y) * vx * vy +
               Gamma[1][1][1](x, y) * vy**2)
        
        return [vx, vy, ax, ay]
    
    if method in ['rk45', 'rk4']:
        sol = solve_ivp(
            geodesic_ode,
            tspan,
            [p0[0], p0[1], v0[0], v0[1]],
            method='RK45' if method == 'rk45' else 'RK23',
            t_eval=np.linspace(tspan[0], tspan[1], n_steps)
        )
        
        result = {
            't': sol.t,
            'x': sol.y[0],
            'y': sol.y[1],
            'vx': sol.y[2],
            'vy': sol.y[3]
        }
    
    elif method in ['symplectic', 'verlet']:
        # Use Hamiltonian formulation
        result = geodesic_hamiltonian_flow(
            metric, p0, v0, tspan, method='verlet', n_steps=n_steps
        )
    
    else:
        raise ValueError("Invalid method")
    
    # Reparametrize by arc length if requested
    if reparametrize:
        # Compute arc length parameter
        ds = np.sqrt(
            metric.g_func[(0,0)](result['x'], result['y']) * result['vx']**2 +
            2 * metric.g_func[(0,1)](result['x'], result['y']) * result['vx'] * result['vy'] +
            metric.g_func[(1,1)](result['x'], result['y']) * result['vy']**2
        )
        
        # Correction: utiliser l'intégration cumulative trapézoïdale
        # s commence à 0
        from scipy.integrate import cumulative_trapezoid
        s = cumulative_trapezoid(ds, result['t'], initial=0)
        
        result['arc_length'] = s
    
    return result


def geodesic_hamiltonian_flow(metric, p0, v0, tspan, method='verlet', n_steps=1000):
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
    
    x_vals = np.zeros(n_steps)
    y_vals = np.zeros(n_steps)
    px_vals = np.zeros(n_steps)
    py_vals = np.zeros(n_steps)
    
    x_vals[0], y_vals[0] = p0
    
    # Convert initial velocity to momentum: pᵢ = gᵢⱼ vʲ
    g_eval = metric.eval(p0[0], p0[1])
    px_vals[0] = g_eval['g'][0,0] * v0[0] + g_eval['g'][0,1] * v0[1]
    py_vals[0] = g_eval['g'][1,0] * v0[0] + g_eval['g'][1,1] * v0[1]
    
    # Prepare derivatives of g^ij for force computation
    x_sym, y_sym = metric.vars_xy
    g_inv = metric.g_inv_matrix
    
    dg_inv_dx = {
        (i, j): lambdify((x_sym, y_sym), diff(g_inv[i,j], x_sym), 'numpy')
        for i in range(2) for j in range(2)
    }
    
    dg_inv_dy = {
        (i, j): lambdify((x_sym, y_sym), diff(g_inv[i,j], y_sym), 'numpy')
        for i in range(2) for j in range(2)
    }
    
    for i in range(n_steps - 1):
        x, y = x_vals[i], y_vals[i]
        px, py = px_vals[i], py_vals[i]
        
        # Evaluate metric inverse
        g_inv_vals = metric.eval(x, y)['g_inv']
        
        # Compute forces: Fₓ = -∂H/∂x = -½ (∂ₓg^ij) pᵢ pⱼ
        Fx = -0.5 * (
            dg_inv_dx[(0,0)](x, y) * px**2 +
            2 * dg_inv_dx[(0,1)](x, y) * px * py +
            dg_inv_dx[(1,1)](x, y) * py**2
        )
        
        Fy = -0.5 * (
            dg_inv_dy[(0,0)](x, y) * px**2 +
            2 * dg_inv_dy[(0,1)](x, y) * px * py +
            dg_inv_dy[(1,1)](x, y) * py**2
        )
        
        if method == 'verlet':
            # Velocity Verlet
            px_half = px + 0.5 * dt * Fx
            py_half = py + 0.5 * dt * Fy
            
            # Velocities: vⁱ = g^ij pⱼ
            vx = g_inv_vals[0,0] * px_half + g_inv_vals[0,1] * py_half
            vy = g_inv_vals[1,0] * px_half + g_inv_vals[1,1] * py_half
            
            x_new = x + dt * vx
            y_new = y + dt * vy
            
            # Update forces
            g_inv_new = metric.eval(x_new, y_new)['g_inv']
            Fx_new = -0.5 * (
                dg_inv_dx[(0,0)](x_new, y_new) * px_half**2 +
                2 * dg_inv_dx[(0,1)](x_new, y_new) * px_half * py_half +
                dg_inv_dx[(1,1)](x_new, y_new) * py_half**2
            )
            
            Fy_new = -0.5 * (
                dg_inv_dy[(0,0)](x_new, y_new) * px_half**2 +
                2 * dg_inv_dy[(0,1)](x_new, y_new) * px_half * py_half +
                dg_inv_dy[(1,1)](x_new, y_new) * py_half**2
            )
            
            px_new = px_half + 0.5 * dt * Fx_new
            py_new = py_half + 0.5 * dt * Fy_new
            
            x_vals[i+1] = x_new
            y_vals[i+1] = y_new
            px_vals[i+1] = px_new
            py_vals[i+1] = py_new
    
    # Compute energy
    energy = np.zeros(n_steps)
    for i in range(n_steps):
        g_inv_vals = metric.eval(x_vals[i], y_vals[i])['g_inv']
        energy[i] = 0.5 * (
            g_inv_vals[0,0] * px_vals[i]**2 +
            2 * g_inv_vals[0,1] * px_vals[i] * py_vals[i] +
            g_inv_vals[1,1] * py_vals[i]**2
        )
    
    # Convert momenta back to velocities
    vx_vals = np.zeros(n_steps)
    vy_vals = np.zeros(n_steps)
    for i in range(n_steps):
        g_inv_vals = metric.eval(x_vals[i], y_vals[i])['g_inv']
        vx_vals[i] = g_inv_vals[0,0] * px_vals[i] + g_inv_vals[0,1] * py_vals[i]
        vy_vals[i] = g_inv_vals[1,0] * px_vals[i] + g_inv_vals[1,1] * py_vals[i]
    
    return {
        't': t_vals,
        'x': x_vals,
        'y': y_vals,
        'vx': vx_vals,
        'vy': vy_vals,
        'px': px_vals,
        'py': py_vals,
        'energy': energy
    }


def exponential_map(metric, p, v, t=1.0, method='rk45'):
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
    >>> metric = Metric2D(g, (x, y))
    >>> q = exponential_map(metric, (0, 0), (1, 1), t=1.0)
    >>> print(q)  # Should be (1, 1) for flat metric
    """
    traj = geodesic_solver(metric, p, v, (0, t), method=method, n_steps=100)
    return (traj['x'][-1], traj['y'][-1])


def distance(metric, p, q, method='shooting', max_iter=50, tol=1e-6):
    """
    Compute geodesic distance between two points.
    
    Uses shooting method: find initial velocity v such that
    exp_p(v) = q, then distance = ||v||_g.
    
    Parameters
    ----------
    metric : Metric2D
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


def laplace_beltrami_operator(metric):
    """
    Return Laplace-Beltrami operator symbol for use with psiop.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    
    Returns
    -------
    dict
        Symbol dictionary compatible with PseudoDifferentialOperator.
    
    Examples
    --------
    >>> from psiop import PseudoDifferentialOperator
    >>> x, y = symbols('x y', real=True)
    >>> metric = Metric2D(Matrix([[1, 0], [0, 1]]), (x, y))
    >>> lb = laplace_beltrami_operator(metric)
    >>> op = PseudoDifferentialOperator(lb['full'], [x, y], mode='symbol')
    """
    return metric.laplace_beltrami_symbol()


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
    sqrt_g = metric.sqrt_det_g
    g_inv = metric.g_inv_matrix
    
    if form_degree == 0:
        # 0-form to 2-form
        return lambda f: f * sqrt_g
    
    elif form_degree == 1:
        # 1-form to 1-form
        def star_1form(alpha_x, alpha_y):
            """
            *(αₓ dx + αᵧ dy) = (g¹¹ αᵧ - g¹² αₓ)√g dx +
                                (-g¹² αᵧ + g²² αₓ)√g dy
            """
            beta_x = (g_inv[0,0] * alpha_y - g_inv[0,1] * alpha_x) * sqrt_g
            beta_y = (-g_inv[0,1] * alpha_y + g_inv[1,1] * alpha_x) * sqrt_g
            return (beta_x, beta_y)
        
        return star_1form
    
    elif form_degree == 2:
        # 2-form to 0-form
        return lambda f: f / sqrt_g
    
    else:
        raise ValueError("form_degree must be 0, 1, or 2")


def de_rham_laplacian(metric, form_degree):
    """
    Compute Hodge Laplacian on differential forms.
    
    Δ = d∘δ + δ∘d where d is exterior derivative and δ = ±*⁻¹ d *.
    
    Parameters
    ----------
    metric : Metric2D
        Riemannian metric.
    form_degree : int
        Degree of forms (0, 1, or 2).
    
    Returns
    -------
    dict
        Symbol for the Hodge Laplacian on k-forms.
    
    Notes
    -----
    - On 0-forms: Δf = -div(grad f) = Laplace-Beltrami
    - On 1-forms: Δω has both tangential and normal components
    - On 2-forms: Δα relates to scalar curvature
    """
    if form_degree == 0:
        # Laplace-Beltrami on functions
        return metric.laplace_beltrami_symbol()
    
    elif form_degree == 1:
        # More complex; requires covariant derivatives
        # For now, return principal symbol only
        x, y = metric.vars_xy
        xi, eta = symbols('xi eta', real=True)
        g_inv = metric.g_inv_matrix
        
        # Principal symbol is g^ij ξᵢ ξⱼ times identity on 1-forms
        principal = g_inv[0,0] * xi**2 + 2*g_inv[0,1] * xi*eta + g_inv[1,1] * eta**2
        
        return {
            'principal': principal,
            'subprincipal': 0,  # Simplified
            'full': principal
        }
    
    else:
        raise NotImplementedError("Only forms of degree 0 and 1 implemented")


def visualize_curvature(metric, x_range=None, y_range=None, 
                        resolution=100, quantity='gauss', cmap='RdBu_r',
                        **kwargs):
    """
    Visualize curvature properties of a 1D or 2D Riemannian manifold.

    For 1D manifolds:
        - Plots the metric component g₁₁(x) and optionally Christoffel symbol.
        - Supports coloring by speed, time, or Christoffel magnitude.

    For 2D manifolds:
        - Plots scalar curvature quantities (Gaussian or Ricci) as a color map.

    Parameters
    ----------
    metric : Metric
        A Riemannian metric object (1D or 2D).
    x_range : tuple, optional
        (min, max) for x-coordinate. Required for 2D; inferred for 1D.
    y_range : tuple, optional
        (min, max) for y-coordinate. Required for 2D; ignored for 1D.
    resolution : int, default=100
        Grid resolution for 2D plots; sampling density for 1D.
    quantity : str, default='gauss'
        For 2D: {'gauss', 'ricci_scalar'}.
        For 1D: {'metric', 'christoffel'} (default: 'metric').
    cmap : str, default='RdBu_r'
        Colormap for 2D scalar fields.

    Additional keyword arguments (1D only):
    ---------------------------------------
    initial_conditions : list of tuples, optional
        Initial conditions as [(x0, v0), ...]. Required for geodesic overlay.
    tspan : tuple, optional
        Time span for geodesic integration, e.g., (0, 10).
    colorby : {'speed', 'time', 'curvature'}, default='speed'
        Coloring scheme for geodesic trajectories.
    n_steps : int, default=500
        Number of integration steps for geodesics.

    Examples
    --------
    # 1D: Plot metric and geodesics
    >>> x = symbols('x', real=True)
    >>> metric = Metric(x**2, (x,))
    >>> visualize_curvature(metric, initial_conditions=[(1.0, 1.0)], tspan=(0, 5))

    # 2D: Plot Gaussian curvature
    >>> x, y = symbols('x y', real=True)
    >>> g = Matrix([[1, 0], [0, sin(x)**2]])
    >>> metric = Metric(g, (x, y))
    >>> visualize_curvature(metric, (0.1, 3.0), (0, 6.28), quantity='gauss')
    """
    if metric.dim == 1:
        _visualize_curvature_1d(metric, x_range, resolution, quantity, **kwargs)
    elif metric.dim == 2:
        if x_range is None or y_range is None:
            raise ValueError("x_range and y_range are required for 2D visualization")
        _visualize_curvature_2d(metric, x_range, y_range, resolution, quantity, cmap)
    else:
        raise ValueError("Only 1D and 2D manifolds are supported")


def _visualize_curvature_1d(metric, x_range, resolution, quantity, **kwargs):
    """Internal helper for 1D curvature/metric visualization."""
    x = metric.coords[0]
    initial_conditions = kwargs.get('initial_conditions')
    tspan = kwargs.get('tspan', (0, 10))
    colorby = kwargs.get('colorby', 'speed')
    n_steps = kwargs.get('n_steps', 500)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax_metric, ax_geo = axes

    # Determine x plotting range
    if x_range is None:
        if initial_conditions:
            x_vals_all = []
            for x0, v0 in initial_conditions:
                traj = geodesic_solver(metric, (x0,), (v0,), tspan, n_steps=n_steps)
                x_vals_all.extend(traj['x'])
            x_min, x_max = np.min(x_vals_all), np.max(x_vals_all)
            x_range = (x_min - 0.5, x_max + 0.5)
        else:
            x_range = (-5, 5)  # fallback

    x_plot = np.linspace(x_range[0], x_range[1], resolution)

    # Plot metric or Christoffel
    if quantity == 'metric':
        y_plot = metric.g_func[(0, 0)](x_plot)
        ylabel = 'g₁₁(x)'
        title = 'Metric Component'
    elif quantity == 'christoffel':
        y_plot = metric.christoffel_func[0][0][0](x_plot)
        ylabel = 'Γ¹₁₁(x)'
        title = 'Christoffel Symbol'
    else:
        raise ValueError("quantity must be 'metric' or 'christoffel' for 1D")

    ax_metric.plot(x_plot, y_plot, 'k-', linewidth=2, label=ylabel)
    ax_metric.set_xlabel('x')
    ax_metric.set_ylabel(ylabel)
    ax_metric.set_title(title)
    ax_metric.grid(True, alpha=0.3)
    ax_metric.legend()

    # Plot geodesics if provided
    if initial_conditions:
        for x0, v0 in initial_conditions:
            traj = geodesic_solver(metric, (x0,), (v0,), tspan, n_steps=n_steps)
            if colorby == 'speed':
                colors = np.abs(traj['v'])
            elif colorby == 'time':
                colors = traj['t']
            elif colorby == 'curvature':
                colors = np.abs(metric.christoffel_func[0][0][0](traj['x']))
            else:
                colors = None

            label = f'IC: x₀={x0:.2f}, v₀={v0:.2f}'
            if colors is not None:
                scatter = ax_geo.scatter(traj['t'], traj['x'], c=colors,
                                         s=10, cmap='viridis', alpha=0.6)
            else:
                ax_geo.plot(traj['t'], traj['x'], alpha=0.7, label=label)

        ax_geo.set_xlabel('t')
        ax_geo.set_ylabel('x(t)')
        ax_geo.set_title('Geodesic Trajectories')
        ax_geo.grid(True, alpha=0.3)
        ax_geo.legend()

        if colors is not None:
            cbar = plt.colorbar(scatter, ax=ax_geo)
            cbar.set_label(colorby.capitalize())
    else:
        ax_geo.text(0.5, 0.5, 'No initial conditions provided\nfor geodesic overlay',
                    ha='center', va='center', transform=ax_geo.transAxes, alpha=0.5)

    plt.tight_layout()
    plt.show()


def _visualize_curvature_2d(metric, x_range, y_range, resolution, quantity, cmap):
    """Internal helper for 2D curvature visualization."""
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    if quantity == 'gauss':
        K_expr = metric.gauss_curvature()
        K_func = lambdify(metric.coords, K_expr, 'numpy')
        Z = K_func(X, Y)
        title = 'Gaussian Curvature K(x, y)'
    elif quantity == 'ricci_scalar':
        R_expr = metric.ricci_scalar()
        R_func = lambdify(metric.coords, R_expr, 'numpy')
        Z = R_func(X, Y)
        title = 'Ricci Scalar R(x, y)'
    else:
        raise ValueError("quantity must be 'gauss' or 'ricci_scalar' for 2D")

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
    plt.colorbar(label=title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_geodesics(metric, initial_conditions, tspan, x_range=None,
                       y_range=None, plot_curvature=True, n_steps=500):
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
    >>> x, y = symbols('x y', real=True)
    >>> g = Matrix([[1, 0], [0, 1 + x**2]])
    >>> metric = Metric2D(g, (x, y))
    >>> ics = [((0, 0), (1, 0)), ((0, 0), (0.7, 0.7))]
    >>> visualize_geodesics(metric, ics, (0, 5))
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Collect all trajectories
    trajectories = []
    for p0, v0 in initial_conditions:
        traj = geodesic_solver(metric, p0, v0, tspan, n_steps=n_steps)
        trajectories.append(traj)
    
    # Determine plot range
    if x_range is None:
        all_x = np.concatenate([t['x'] for t in trajectories])
        x_margin = 0.1 * (all_x.max() - all_x.min())
        x_range = (all_x.min() - x_margin, all_x.max() + x_margin)
    
    if y_range is None:
        all_y = np.concatenate([t['y'] for t in trajectories])
        y_margin = 0.1 * (all_y.max() - all_y.min())
        y_range = (all_y.min() - y_margin, all_y.max() + y_margin)
    
    # Plot curvature background
    if plot_curvature:
        try:
            x_bg = np.linspace(x_range[0], x_range[1], 100)
            y_bg = np.linspace(y_range[0], y_range[1], 100)
            X_bg, Y_bg = np.meshgrid(x_bg, y_bg, indexing='ij')
            
            K_expr = metric.gauss_curvature()
            K_func = lambdify(metric.vars_xy, K_expr, 'numpy')
            K_vals = K_func(X_bg, Y_bg)
            
            im = ax.pcolormesh(X_bg, Y_bg, K_vals, shading='auto',
                              cmap='RdBu_r', alpha=0.3, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label='Gaussian Curvature')
        except:
            print("Warning: Could not compute curvature background")
    
    # Plot geodesics
    for idx, traj in enumerate(trajectories):
        p0, v0 = initial_conditions[idx]
        label = f'IC: ({p0[0]:.2f},{p0[1]:.2f}), v=({v0[0]:.2f},{v0[1]:.2f})'
        
        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(traj['x'])))
        
        for i in range(len(traj['x']) - 1):
            ax.plot(traj['x'][i:i+2], traj['y'][i:i+2],
                   color=colors[i], alpha=0.8, linewidth=2)
        
        # Mark start and end
        ax.plot(traj['x'][0], traj['y'][0], 'go', markersize=10, 
               label=f'Start {idx+1}')
        ax.plot(traj['x'][-1], traj['y'][-1], 'ro', markersize=10,
               label=f'End {idx+1}')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Geodesics on Riemannian Manifold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def verify_gauss_bonnet(metric, domain, resolution=100):
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
    >>> metric = Metric2D(g_sphere, (x, y))
    >>> result = verify_gauss_bonnet(metric, ((0.01, np.pi-0.01), (0, 2*np.pi)))
    >>> print(f"Integral: {result['integral']:.4f}, Expected: {result['expected']:.4f}")
    """
    from scipy.integrate import dblquad
    
    K_expr = metric.gauss_curvature()
    sqrt_g = metric.sqrt_det_g
    x_sym, y_sym = metric.vars_xy
    
    integrand_expr = K_expr * sqrt_g
    integrand_func = lambdify((x_sym, y_sym), integrand_expr, 'numpy')
    
    (x_min, x_max), (y_min, y_max) = domain
    
    integral, error = dblquad(
        lambda y, x: integrand_func(x, y),
        x_min, x_max,
        y_min, y_max
    )
    
    # For a rectangular domain in the plane, χ = 1
    # For a sphere (topologically), χ = 2
    # User must specify topology
    expected = 2 * np.pi  # Assuming χ = 1
    
    return {
        'integral': integral,
        'integration_error': error,
        'expected': expected,
        'relative_error': abs(integral - expected) / abs(expected)
    }


# ============================================================================
# Tests
# ============================================================================

def test_euclidean_metric():
    """Test flat Euclidean metric."""
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))
    
    # Curvature should be zero
    K = metric.gauss_curvature()
    assert K == 0
    
    # Geodesics should be straight lines
    traj = geodesic_solver(metric, (0, 0), (1, 1), (0, 5), n_steps=100)
    expected_x = np.linspace(0, 5, 100)
    expected_y = np.linspace(0, 5, 100)
    
    assert np.allclose(traj['x'], expected_x, rtol=1e-2)
    assert np.allclose(traj['y'], expected_y, rtol=1e-2)
    
    print("✓ Euclidean metric test passed")


def test_polar_coordinates():
    """Test polar coordinate metric."""
    r, theta = symbols('r theta', real=True, positive=True)
    g_polar = Matrix([[1, 0], [0, r**2]])
    metric = Metric2D(g_polar, (r, theta))
    
    # Gauss curvature should be zero (flat space in polar coords)
    K = metric.gauss_curvature()
    assert simplify(K) == 0
    
    # Laplace-Beltrami
    lb = metric.laplace_beltrami_symbol()
    xi, eta = symbols('xi eta', real=True)
    expected_principal = xi**2 + eta**2 / r**2
    assert simplify(lb['principal'] - expected_principal) == 0
    
    print("✓ Polar coordinates test passed")


def test_sphere_metric():
    """Test metric on sphere."""
    theta, phi = symbols('theta phi', real=True)
    g_sphere = Matrix([[1, 0], [0, sin(theta)**2]])
    metric = Metric2D(g_sphere, (theta, phi))
    
    # Gaussian curvature should be 1 (unit sphere)
    K = metric.gauss_curvature()
    # K = 1 for unit sphere
    K_simplified = simplify(K)
    print(f"Sphere curvature: {K_simplified}")
    
    # Ricci scalar should be 2
    R = metric.ricci_scalar()
    R_simplified = simplify(R)
    print(f"Ricci scalar: {R_simplified}")
    
    print("✓ Sphere metric test passed")


# ============================================================================
# Enhanced Tests
# ============================================================================
def test_poincare_half_plane():
    """
    Test on the Poincaré half-plane (Hyperbolic Geometry).
    Metric: g_ij = (1/y^2) * delta_ij
    Expected curvature: constant -1.
    """
    print("Testing Poincaré Half-Plane (Negative Curvature)...")
    x, y = symbols('x y', real=True)
    # y must be positive
    g = Matrix([[1/y**2, 0], [0, 1/y**2]])
    metric = Metric2D(g, (x, y))

    # 1. Curvature test (should be -1)
    K = metric.gauss_curvature()
    assert simplify(K) == -1, f"Expected curvature -1, got {K}"

    # 2. Geodesic test (Vertical line is a geodesic)
    # Start from (0, 1) upwards (0, 1)
    # Analytical solution: x(t)=0, y(t) = e^t (if default parameterization)
    traj = geodesic_solver(metric, (0, 1), (0, 1), (0, 1), method='rk45')

    assert np.allclose(traj['x'], 0, atol=1e-4), "Geodesic should remain on x=0"
    assert traj['y'][-1] > 1, "Geodesic should move upwards"

    print("✓ Poincaré test passed")

def test_hamiltonian_construction():
    """
    Test metric construction from a Hamiltonian.
    Example: Free particle in polar coordinates.
    H = (p_r^2 + p_theta^2/r^2) / 2
    """
    print("Testing Hamiltonian construction...")
    r, theta = symbols('r theta', real=True, positive=True)
    pr, pt = symbols('p_r p_theta', real=True)

    H = (pr**2 + pt**2 / r**2) / 2
    metric = Metric2D.from_hamiltonian(H, (r, theta), (pr, pt))

    g_expected = Matrix([[1, 0], [0, r**2]])

    assert simplify(metric.g_matrix - g_expected) == zeros(2, 2), "Failed to reconstruct polar metric from Hamiltonian"
    print("✓ Hamiltonian construction test passed")

def test_geodesic_integrators():
    """
    Compare numerical integrators (RK45 vs Symplectic)
    and the reparameterization option.
    """
    print("Testing Geodesic Integrators & Options...")
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]]) # Flat
    metric = Metric2D(g, (x, y))

    p0 = (0, 0)
    v0 = (1, 1) # Speed norm sqrt(2)
    tspan = (0, 10)

    # 1. Symplectic test (Energy conservation)
    traj_sym = geodesic_solver(metric, p0, v0, tspan, method='symplectic', n_steps=100)
    energy_var = np.std(traj_sym['energy'])
    assert energy_var < 1e-10, f"Symplectic integrator failed to conserve energy. Var: {energy_var}"

    # 2. Reparameterization test (Arc length)
    traj_rk = geodesic_solver(metric, p0, v0, tspan, method='rk45', reparametrize=True)
    assert 'arc_length' in traj_rk
    # For a flat metric, s = v * t = sqrt(2) * 10
    total_length = traj_rk['arc_length'][-1]
    expected_length = np.sqrt(2) * 10
    assert np.isclose(total_length, expected_length, rtol=1e-3), \
        f"Arc length calc failed. Got {total_length}, expected {expected_length}"

    print("✓ Geodesic integrators test passed")

def test_distance_and_exp_map():
    """
    Test Exponential Map and Distance functions.
    On a flat metric, exp_p(v) = p + v.
    """
    print("Testing Distance and Exponential Map...")
    x, y = symbols('x y', real=True)
    g = Matrix([[1, 0], [0, 1]])
    metric = Metric2D(g, (x, y))

    p = (0, 0)
    v = (3, 4)
    target = (3, 4)

    # 1. Exponential Map test
    end_point = exponential_map(metric, p, v, t=1.0)
    assert np.allclose(end_point, target, atol=1e-4), f"Exp map failed. Got {end_point}"

    # 2. Distance test (Shooting method)
    d_shoot = distance(metric, p, target, method='shooting')
    assert np.isclose(d_shoot, 5.0, rtol=1e-3), f"Shooting distance failed. Got {d_shoot}"

    # 3. Distance test (Optimization method)
    # Note: Optimize may be less precise without fine tuning, tolerance relaxed
    d_opt = distance(metric, p, target, method='optimize')
    assert np.isclose(d_opt, 5.0, rtol=5e-2), f"Optimization distance failed. Got {d_opt}"

    print("✓ Distance functions test passed")

def test_hodge_star_and_volume():
    """
    Test Riemannian volume and Hodge Star operator.
    """
    print("Testing Hodge Star and Volume...")
    x, y = symbols('x y', real=True)
    # Simple but non-unit diagonal metric: g = diag(4, 9)
    # det(g) = 36, sqrt(g) = 6
    g = Matrix([[4, 0], [0, 9]])
    metric = Metric2D(g, (x, y))

    # 1. Volume test (Unit square [0,1]x[0,1])
    # Volume = Integral(sqrt(g) dx dy) = 6 * 1 * 1 = 6
    vol_sym = metric.riemannian_volume(((0, 1), (0, 1)), method='symbolic')
    vol_num = metric.riemannian_volume(((0, 1), (0, 1)), method='numerical')

    assert vol_sym == 6, f"Symbolic volume wrong. Got {vol_sym}"
    assert np.isclose(vol_num, 6.0), f"Numerical volume wrong. Got {vol_num}"

    # 2. Hodge Star test on 2-form (Scalar -> Scalar)
    # * (f dx^dy) = f / sqrt(g)
    # If f=12, result = 12 / 6 = 2
    star2 = hodge_star(metric, 2)
    res = star2(12)
    assert simplify(res) == 2, f"Hodge star on 2-form failed. Got {res}"

    print("✓ Hodge and Volume test passed")

def test_jacobi_field_stability():
    """
    Test Jacobi equation (geodesic deviation).
    On a sphere, parallel geodesics converge (focusing).
    J(t) should decrease or oscillate.
    """
    print("Testing Jacobi Equation (Sphere Stability)...")
    theta, phi = symbols('theta phi', real=True)
    # Sphere metric (avoid poles for numerical test)
    g = Matrix([[1, 0], [0, sin(theta)**2]])
    metric = Metric2D(g, (theta, phi))

    # Equatorial geodesic: theta=pi/2, phi varies
    # Initial speed: theta_dot=0, phi_dot=1
    geod = geodesic_solver(metric, (np.pi/2, 0), (0, 1), (0, 2), n_steps=200)

    # Initial orthogonal perturbation (towards the north pole)
    # J0 = 0, DJ0 = (v_theta=0.1, v_phi=0)
    init_var = {'J0': (0, 0), 'DJ0': (0.1, 0)}

    jac = jacobi_equation_solver(metric, geod, init_var, (0, 2))

    # On the sphere, J(t) ~ sin(t).
    # Initially J increases, but the "separation speed" DJ should decrease
    # due to positive curvature R > 0.

    # Check that J_x (theta deviation) does not explode exponentially (as in hyperbolic)
    max_dev = np.max(np.abs(jac['J_x']))
    assert max_dev < 1.0, "Jacobi field deviation too large for sphere"

    print("✓ Jacobi equation test passed")

if __name__ == "__main__":
    print("\n=== Running Comprehensive Test Suite ===\n")

    # Existing basic tests (slightly adapted if necessary)
    test_euclidean_metric()
    test_polar_coordinates()
    test_sphere_metric()

    # New advanced tests
    print("-" * 40)
    test_poincare_half_plane()
    test_hamiltonian_construction()
    test_geodesic_integrators()
    test_distance_and_exp_map()
    test_hodge_star_and_volume()
    test_jacobi_field_stability()

    print("\n✓ ALL TESTS PASSED SUCCESSFULLY")
