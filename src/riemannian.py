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

OPTIMIZED VERSION:
 - Lazy evaluation with property caching
 - Symmetry exploitation in Christoffel symbols
 - Vectorized lambdas for matrices
 - Optional simplification (default: False)
 - Pre-allocated numpy arrays
 - Cached derivative computations
"""
from imports import *
from sympy import simplify as sp_simplify
from functools import lru_cache

class Metric:
    """
    Unified Riemannian metric for 1D or 2D manifolds.

    Parameters
    ----------
    g_input : sympy expression (1D) or 2×2 sympy Matrix (2D)
        Metric tensor component(s).
    coords : tuple of sympy symbols
        Coordinate variables (length 1 or 2).
    simplify : bool, optional (default: False)
        Apply symbolic simplification (can be slow).
    precompute : bool, optional (default: True)
        Pre-compute lambdified functions at initialization.

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
    def __init__(self, g_input, coords, simplify=False, precompute=True):
        self.dim = len(coords)
        self.coords = coords
        self._simplify = simplify

        # Build metric matrix
        if self.dim == 1:
            x, = coords
            g_expr = sp_simplify(g_input) if simplify else g_input
            self.g_matrix = Matrix([[g_expr]])
        elif self.dim == 2:
            if not isinstance(g_input, Matrix) or g_input.shape != (2, 2):
                raise ValueError("2D metric must be a 2×2 sympy Matrix")
            self.g_matrix = sp_simplify(g_input) if simplify else g_input
        else:
            raise ValueError("Only 1D and 2D manifolds are supported")

        # Basic geometric quantities
        self.det_g = self.g_matrix.det()
        if simplify:
            self.det_g = sp_simplify(self.det_g)
        self.sqrt_det_g = sqrt(abs(self.det_g))
        if simplify:
            self.sqrt_det_g = sp_simplify(self.sqrt_det_g)
        
        self.g_inv_matrix = self.g_matrix.inv()
        if simplify:
            self.g_inv_matrix = Matrix([[sp_simplify(self.g_inv_matrix[i, j]) 
                                         for j in range(self.dim)] 
                                        for i in range(self.dim)])

        # Lazy-evaluated caches
        self._christoffel_cache = None
        self._christoffel_func_cache = None
        self._riemann_cache = None
        self._ricci_tensor_cache = None
        self._ricci_scalar_cache = None
        self._gauss_curvature_cache = None
        self._derivatives_cache = {}

        # Pre-compute numerical functions
        if precompute:
            self._lambdify_all()

    def _lambdify_all(self):
        """Vectorized lambdification of metric components."""
        n = self.dim
        
        # Vectorized matrix lambdas
        try:
            self.g_func_matrix = lambdify(
                self.coords, self.g_matrix,
                modules=['numpy', {'ImmutableDenseMatrix': np.array}]
            )
        except:
            # Fallback to component-wise
            self.g_func_matrix = None
            
        try:
            self.g_inv_func_matrix = lambdify(
                self.coords, self.g_inv_matrix,
                modules=['numpy', {'ImmutableDenseMatrix': np.array}]
            )
        except:
            self.g_inv_func_matrix = None
        
        # Component-wise fallback
        self.g_func = {(i,j): lambdify(self.coords, self.g_matrix[i,j], 'numpy')
                       for i in range(n) for j in range(n)}
        self.g_inv_func = {(i,j): lambdify(self.coords, self.g_inv_matrix[i,j], 'numpy')
                           for i in range(n) for j in range(n)}
        
        # Scalar functions
        self.sqrt_det_g_func = lambdify(self.coords, self.sqrt_det_g, 'numpy')

    @property
    def christoffel(self):
        """Lazy-computed Christoffel symbols (exploiting symmetry)."""
        if self._christoffel_cache is None:
            self._christoffel_cache = self._compute_christoffel()
        return self._christoffel_cache

    def _compute_christoffel(self):
        """
        Compute Christoffel symbols exploiting symmetry Γⁱⱼₖ = Γⁱₖⱼ.
        Reduces computation by ~50%.
        """
        n = self.dim
        g = self.g_matrix
        g_inv = self.g_inv_matrix
        
        # Initialize nested dictionary structure first
        Gamma = {}
        for i in range(n):
            Gamma[i] = {}
            for j in range(n):
                Gamma[i][j] = {}
        
        # Compute exploiting symmetry in lower indices
        for i in range(n):
            for j in range(n):
                for k in range(j, n):  # Only k >= j (exploit symmetry)
                    expr = 0
                    for ell in range(n):
                        term1 = diff(g[k, ell], self.coords[j])
                        term2 = diff(g[j, ell], self.coords[k])
                        term3 = diff(g[j, k], self.coords[ell])
                        expr += g_inv[i, ell] * (term1 + term2 - term3) / 2
                    
                    if self._simplify:
                        expr = sp_simplify(expr)
                    
                    # Set both symmetric entries
                    Gamma[i][j][k] = expr
                    if k != j:
                        Gamma[i][k][j] = expr
        
        return Gamma

    @property
    def christoffel_func(self):
        """Lazy-computed numerical Christoffel symbols."""
        if self._christoffel_func_cache is None:
            self._christoffel_func_cache = self._lambdify_christoffel()
        return self._christoffel_func_cache

    def _lambdify_christoffel(self):
        """Lambdify Christoffel symbols."""
        n = self.dim
        Gamma_sym = self.christoffel
        Gamma_func = {}
        
        for i in range(n):
            Gamma_func[i] = {}
            for j in range(n):
                Gamma_func[i][j] = {}
                for k in range(n):
                    Gamma_func[i][j][k] = lambdify(
                        self.coords, Gamma_sym[i][j][k], 'numpy'
                    )
        
        return Gamma_func

    def _get_derivative(self, expr, coord):
        """Cached derivative computation."""
        key = (id(expr), coord)
        if key not in self._derivatives_cache:
            self._derivatives_cache[key] = diff(expr, coord)
        return self._derivatives_cache[key]

    @classmethod
    def from_hamiltonian(cls, H_expr, coords, momenta, simplify=False):
        """
        Extract metric from Hamiltonian kinetic term.
        
        For H = ½ g^ij pᵢ pⱼ + V, extract inverse metric from Hessian:
            g^ij = ∂²H/∂pᵢ∂pⱼ
        
        Parameters
        ----------
        H_expr : sympy expression
            Hamiltonian H(x, y, pₓ, pᵧ).
        coords : tuple
            Position variables (x, y).
        momenta : tuple
            Momentum variables (pₓ, pᵧ).
        simplify : bool
            Apply simplification.
        
        Returns
        -------
        Metric
            Metric with gᵢⱼ = (g^ij)⁻¹.
        
        Examples
        --------
        >>> x, p = symbols('x p', real=True)
        >>> H = p**2/(2*x**2) + x**2/2
        >>> metric = Metric.from_hamiltonian(H, (x,), (p,))
        >>> print(metric.g_matrix)
        Matrix([[x**2]])
        """
        if len(coords) != len(momenta):
            raise ValueError("Coordinate and momentum lists must match in length")
        n = len(coords)
        
        # Build inverse metric: g^ij = ∂²H / ∂p_i ∂p_j
        g_inv = Matrix([[diff(H_expr, momenta[i], momenta[j])
                         for j in range(n)] for i in range(n)])
        
        # Invert
        g = g_inv.inv()
        if simplify:
            g = Matrix([[sp_simplify(g[i, j]) for j in range(n)] for i in range(n)])
        
        return cls(g, coords, simplify=simplify)

    def eval(self, *coord_vals):
        """
        Evaluate metric components at given points (vectorized).
        
        Parameters
        ----------
        *coord_vals : float or ndarray
            Coordinate values.
        
        Returns
        -------
        dict
            Dictionary containing metric tensors and geometric quantities.
        """
        n = self.dim
        
        # Try vectorized evaluation first
        if hasattr(self, 'g_func_matrix') and self.g_func_matrix is not None:
            try:
                g_val = self.g_func_matrix(*coord_vals)
                g_inv_val = self.g_inv_func_matrix(*coord_vals)
                
                result = {
                    'g': g_val,
                    'g_inv': g_inv_val,
                    'sqrt_det': self.sqrt_det_g_func(*coord_vals),
                }
            except:
                # Fallback to component-wise
                result = {
                    'g': np.array([[self.g_func[(i,j)](*coord_vals) 
                                   for j in range(n)] for i in range(n)]),
                    'g_inv': np.array([[self.g_inv_func[(i,j)](*coord_vals) 
                                       for j in range(n)] for i in range(n)]),
                    'sqrt_det': self.sqrt_det_g_func(*coord_vals),
                }
        else:
            result = {
                'g': np.array([[self.g_func[(i,j)](*coord_vals) 
                               for j in range(n)] for i in range(n)]),
                'g_inv': np.array([[self.g_inv_func[(i,j)](*coord_vals) 
                                   for j in range(n)] for i in range(n)]),
                'sqrt_det': self.sqrt_det_g_func(*coord_vals),
            }
        
        # Christoffel symbols (on demand)
        if n == 1:
            result['christoffel'] = self.christoffel_func[0][0][0](*coord_vals)
        else:
            result['christoffel'] = {
                i: {j: {k: self.christoffel_func[i][j][k](*coord_vals)
                        for k in range(n)} for j in range(n)} for i in range(n)
            }
        
        return result

    # ========================
    # Curvature (lazy-evaluated)
    # ========================
    @property
    def gauss_curvature(self):
        """
        Compute Gaussian curvature K (lazy-evaluated).
        
        Returns
        -------
        sympy expression
            Gaussian curvature K(x, y).
        """
        if self._gauss_curvature_cache is not None:
            return self._gauss_curvature_cache
        
        if self.dim == 1:
            self._gauss_curvature_cache = sympify(0)
        else:
            R = self.riemann_tensor
            g = self.g_matrix
            R_xyxy = g[0,0] * R[0][1][0][1] + g[0,1] * R[1][1][0][1]
            self._gauss_curvature_cache = sp_simplify(R_xyxy / self.det_g) if self._simplify else R_xyxy / self.det_g
        
        return self._gauss_curvature_cache

    @property
    def riemann_tensor(self):
        """
        Compute Riemann curvature tensor Rⁱⱼₖₗ (lazy-evaluated).
        
        Returns
        -------
        dict
            Nested dict with all non-zero components.
        """
        if self._riemann_cache is not None:
            return self._riemann_cache
        
        if self.dim == 1:
            self._riemann_cache = {}
            return self._riemann_cache
        
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
                        R[i][j][k][ell] = sp_simplify(expr) if self._simplify else expr
        
        self._riemann_cache = R
        return R

    @property
    def ricci_tensor(self):
        """
        Compute Ricci curvature tensor Rᵢⱼ (lazy-evaluated).
        
        Returns
        -------
        sympy Matrix
            2×2 Ricci tensor.
        """
        if self._ricci_tensor_cache is not None:
            return self._ricci_tensor_cache
        
        if self.dim == 1:
            self._ricci_tensor_cache = Matrix([[0]])
            return self._ricci_tensor_cache
        
        R_full = self.riemann_tensor
        n = 2
        Ric = zeros(n)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Ric[i,j] += R_full[k][i][k][j]
        
        self._ricci_tensor_cache = sp_simplify(Ric) if self._simplify else Ric
        return self._ricci_tensor_cache

    @property
    def ricci_scalar(self):
        """
        Compute scalar curvature R (lazy-evaluated).
        
        Returns
        -------
        sympy expression
            Scalar curvature R(x, y).
        """
        if self._ricci_scalar_cache is not None:
            return self._ricci_scalar_cache
        
        if self.dim == 1:
            self._ricci_scalar_cache = sympify(0)
            return self._ricci_scalar_cache
        
        Ric = self.ricci_tensor
        g_inv = self.g_inv_matrix
        R = sum(g_inv[i,j] * Ric[i,j] for i in range(2) for j in range(2))
        
        self._ricci_scalar_cache = sp_simplify(R) if self._simplify else R
        return self._ricci_scalar_cache

    # ========================
    # Laplace-Beltrami
    # ========================
    def laplace_beltrami_symbol(self):
        """
        Compute symbol of Laplace-Beltrami operator.
        
        Returns
        -------
        dict
            Symbol components: 'principal', 'subprincipal', 'full'.
        """
        n = self.dim
        if n == 1:
            x, = self.coords
            xi = symbols('xi', real=True)
            principal = self.g_inv_matrix[0,0] * xi**2
            log_sqrt_g = log(self.sqrt_det_g)
            transport = diff(log_sqrt_g, x) * self.g_inv_matrix[0,0]
            if self._simplify:
                transport = sp_simplify(transport)
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
            subprincipal = (coeff_x * xi + coeff_y * eta) / sqrt_g
            if self._simplify:
                subprincipal = sp_simplify(subprincipal)
        else:
            raise NotImplementedError("Laplace-Beltrami symbol only implemented for 1D and 2D")
        
        full = principal + I * subprincipal
        if self._simplify:
            principal = sp_simplify(principal)
            full = sp_simplify(full)
        
        return {
            'principal': principal,
            'subprincipal': subprincipal,
            'full': full
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
                f = self.sqrt_det_g_func
                return quad(f, a, b)[0]
        else:
            (x0, x1), (y0, y1) = domain
            if method == 'symbolic':
                x, y = self.coords
                return integrate(self.sqrt_det_g, (x, x0, x1), (y, y0, y1))
            else:
                f = self.sqrt_det_g_func
                return dblquad(lambda Y, X: f(X, Y), x0, x1, y0, y1)[0]

    def arc_length(self, a, b, method='numerical'):
        """
        Compute arc length between two points.
        
        L = ∫ₐᵇ √g₁₁(x) dx
        
        Parameters
        ----------
        a, b : float
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
        
        Returns
        -------
        dict
            Dictionary with 'p', 'q', 'w' (coefficient functions).
        """
        if self.dim != 1:
            raise NotImplementedError("Sturm-Liouville form only for 1D")
        
        x, = self.coords
        sqrt_g = self.sqrt_det_g
        g_inv = self.g_inv_matrix[0,0]
        
        p_expr = sqrt_g * g_inv
        if self._simplify:
            p_expr = sp_simplify(p_expr)
        
        w_expr = sqrt_g
        q_expr = potential_expr * sqrt_g if potential_expr else sympify(0)
        if self._simplify and potential_expr:
            q_expr = sp_simplify(q_expr)
        
        return {
            'p': p_expr, 'q': q_expr, 'w': w_expr,
            'p_func': lambdify(x, p_expr, 'numpy'),
            'q_func': lambdify(x, q_expr, 'numpy'),
            'w_func': lambdify(x, w_expr, 'numpy')
        }

# ============================================================================
# Unified functions (dispatched by dimension) - OPTIMIZED
# ============================================================================
def geodesic_solver(metric, p0, v0, tspan, method='rk45', n_steps=1000, reparametrize=False):
    """
    Integrate geodesic equations (optimized with pre-allocation).
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric.
    p0 : tuple
        Initial position.
    v0 : tuple
        Initial velocity.
    tspan : tuple
        Time interval (t_start, t_end).
    method : str
        Integration method.
    n_steps : int
        Number of steps.
    reparametrize : bool
        If True, reparametrize by arc length.
    
    Returns
    -------
    dict
        Trajectory with 't', 'x', 'y', 'vx', 'vy' arrays.
    """
    from scipy.integrate import solve_ivp
    
    if metric.dim == 1:
        x0, = p0
        v0x, = v0
        Gamma = metric.christoffel_func[0][0][0]
        
        def ode(t, y):
            x, vx = y
            # Avoid evaluating at singular points
            if not np.isfinite(x):
                return [0, 0]
            gamma_val = Gamma(x)
            if not np.isfinite(gamma_val):
                return [vx, 0]
            return [vx, -gamma_val * vx**2]
        
        sol = solve_ivp(ode, tspan, [x0, v0x], 
                       t_eval=np.linspace(*tspan, n_steps),
                       method='RK45')
        
        # Check if integration was successful
        if not sol.success:
            import warnings
            warnings.warn(f"Geodesic integration failed: {sol.message}")
        
        # Handle empty solution
        if len(sol.y) < 2:
            t_vals = np.linspace(*tspan, n_steps)
            return {'t': t_vals, 'x': np.full_like(t_vals, x0), 'v': np.full_like(t_vals, v0x)}
        
        result = {'t': sol.t, 'x': sol.y[0], 'v': sol.y[1]}
        
        if reparametrize:
            ds = metric.sqrt_det_g_func(result['x']) * np.abs(result['v'])
            result['arc_length'] = cumulative_trapezoid(ds, result['t'], initial=0)
        
        return result
    else:
        Gamma = metric.christoffel_func
        
        def ode(t, y):
            x, yv, vx, vy = y
            
            # Check for finite values
            if not (np.isfinite(x) and np.isfinite(yv)):
                return [0, 0, 0, 0]
            
            try:
                ax = -(Gamma[0][0][0](x, yv) * vx**2 +
                       2 * Gamma[0][0][1](x, yv) * vx * vy +
                       Gamma[0][1][1](x, yv) * vy**2)
                ay = -(Gamma[1][0][0](x, yv) * vx**2 +
                       2 * Gamma[1][0][1](x, yv) * vx * vy +
                       Gamma[1][1][1](x, yv) * vy**2)
                
                # Check for NaN/Inf
                if not (np.isfinite(ax) and np.isfinite(ay)):
                    return [vx, vy, 0, 0]
                
                return [vx, vy, ax, ay]
            except:
                return [vx, vy, 0, 0]
        
        sol = solve_ivp(ode, tspan, [*p0, *v0], 
                       t_eval=np.linspace(*tspan, n_steps),
                       method='RK45')
        
        if not sol.success:
            import warnings
            warnings.warn(f"Geodesic integration failed: {sol.message}")
        
        result = {'t': sol.t, 'x': sol.y[0], 'y': sol.y[1], 'vx': sol.y[2], 'vy': sol.y[3]}
        
        if reparametrize:
            ds = np.sqrt(
                metric.g_func[(0,0)](result['x'], result['y']) * result['vx']**2 +
                2 * metric.g_func[(0,1)](result['x'], result['y']) * result['vx'] * result['vy'] +
                metric.g_func[(1,1)](result['x'], result['y']) * result['vy']**2
            )
            result['arc_length'] = cumulative_trapezoid(ds, result['t'], initial=0)
        
        return result

def geodesic_hamiltonian_flow(metric, p0, v0_or_p0, tspan, method='verlet', 
                              n_steps=1000, momentum_input=False):
    """
    Integrate geodesic flow in Hamiltonian formulation (optimized).
    
    Pre-computes derivatives and uses pre-allocated arrays.
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric.
    p0 : tuple
        Initial position.
    v0_or_p0 : tuple
        Initial velocity or momentum.
    tspan : tuple
        Time interval.
    method : str
        Symplectic integrator.
    n_steps : int
        Number of steps.
    momentum_input : bool
        If True, v0_or_p0 is momentum, else velocity.
    
    Returns
    -------
    dict
        Phase space trajectory with energy.
    """
    dt = (tspan[1] - tspan[0]) / n_steps
    t_vals = np.linspace(tspan[0], tspan[1], n_steps)
    
    if metric.dim == 1:
        # Pre-allocate arrays
        x_vals = np.empty(n_steps)
        p_vals = np.empty(n_steps)
        
        x_vals[0] = p0[0]
        if momentum_input:
            p_vals[0] = v0_or_p0[0]
        else:
            g_eval = metric.eval(p0[0])['g'][0,0]
            p_vals[0] = g_eval * v0_or_p0[0]
        
        # Pre-compute derivative function
        x_sym, = metric.coords
        g_inv = metric.g_inv_matrix[0,0]
        dg_inv_dx = lambdify(x_sym, diff(g_inv, x_sym), 'numpy')
        g_inv_func = metric.g_inv_func[(0,0)]
        
        for i in range(n_steps - 1):
            x, p = x_vals[i], p_vals[i]
            Fx = -0.5 * dg_inv_dx(x) * p**2
            
            if method == 'verlet':
                p_half = p + 0.5 * dt * Fx
                x_new = x + dt * g_inv_func(x) * p_half
                Fx_new = -0.5 * dg_inv_dx(x_new) * p_half**2
                p_new = p_half + 0.5 * dt * Fx_new
                x_vals[i+1], p_vals[i+1] = x_new, p_new
        
        # Vectorized post-processing
        v_vals = g_inv_func(x_vals) * p_vals
        energy = 0.5 * g_inv_func(x_vals) * p_vals**2
        
        return {'t': t_vals, 'x': x_vals, 'v': v_vals, 'p': p_vals, 'energy': energy}
    
    else:  # 2D case
        # Pre-allocate arrays
        x_vals = np.empty(n_steps)
        y_vals = np.empty(n_steps)
        px_vals = np.empty(n_steps)
        py_vals = np.empty(n_steps)
        
        x_vals[0], y_vals[0] = p0
        
        if momentum_input:
            px_vals[0], py_vals[0] = v0_or_p0
        else:
            g_eval = metric.eval(*p0)['g']
            px_vals[0] = g_eval[0][0] * v0_or_p0[0] + g_eval[0][1] * v0_or_p0[1]
            py_vals[0] = g_eval[1][0] * v0_or_p0[0] + g_eval[1][1] * v0_or_p0[1]
        
        # Pre-compute derivative functions
        x_sym, y_sym = metric.coords
        g_inv = metric.g_inv_matrix
        
        dg_inv_dx = {(i,j): lambdify((x_sym, y_sym), diff(g_inv[i,j], x_sym), 'numpy')
                     for i in range(2) for j in range(2)}
        dg_inv_dy = {(i,j): lambdify((x_sym, y_sym), diff(g_inv[i,j], y_sym), 'numpy')
                     for i in range(2) for j in range(2)}

        for i in range(n_steps - 1):
            x, y = x_vals[i], y_vals[i]
            px, py = px_vals[i], py_vals[i]
            
            # Forces from Hamiltonian
            Fx = -0.5 * (dg_inv_dx[(0,0)](x, y) * px**2 +
                         2 * dg_inv_dx[(0,1)](x, y) * px * py +
                         dg_inv_dx[(1,1)](x, y) * py**2)
            Fy = -0.5 * (dg_inv_dy[(0,0)](x, y) * px**2 +
                         2 * dg_inv_dy[(0,1)](x, y) * px * py +
                         dg_inv_dy[(1,1)](x, y) * py**2)
            
            if method == 'verlet':
                # Half-step momentum
                px_half = px + 0.5 * dt * Fx
                py_half = py + 0.5 * dt * Fy
                
                # Full-step position
                g_inv_vals = metric.eval(x, y)['g_inv']
                vx = g_inv_vals[0][0] * px_half + g_inv_vals[0][1] * py_half
                vy = g_inv_vals[1][0] * px_half + g_inv_vals[1][1] * py_half
                
                x_new = x + dt * vx
                y_new = y + dt * vy
                
                # Half-step momentum (second half)
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
        
        # Vectorized post-processing
        energy = np.empty(n_steps)
        vx_vals = np.empty(n_steps)
        vy_vals = np.empty(n_steps)
        
        for i in range(n_steps):
            g_inv_vals = metric.eval(x_vals[i], y_vals[i])['g_inv']
            energy[i] = 0.5 * (g_inv_vals[0][0] * px_vals[i]**2 +
                               2 * g_inv_vals[0][1] * px_vals[i] * py_vals[i] +
                               g_inv_vals[1][1] * py_vals[i]**2)
            vx_vals[i] = g_inv_vals[0][0] * px_vals[i] + g_inv_vals[0][1] * py_vals[i]
            vy_vals[i] = g_inv_vals[1][0] * px_vals[i] + g_inv_vals[1][1] * py_vals[i]
        
        return {
            't': t_vals, 'x': x_vals, 'y': y_vals,
            'vx': vx_vals, 'vy': vy_vals,
            'px': px_vals, 'py': py_vals, 'energy': energy
        }

def exponential_map(metric, p, v, t=1.0):
    """
    Compute exponential map exp_p(tv).
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric.
    p : tuple
        Base point.
    v : tuple
        Initial tangent vector.
    t : float
        Parameter value.
    
    Returns
    -------
    tuple or float
        End point.
    """
    traj = geodesic_solver(metric, p, v, (0, t), n_steps=100)
    if metric.dim == 1:
        return traj['x'][-1]
    else:
        return (traj['x'][-1], traj['y'][-1])

def distance(metric, p, q, method='shooting', max_iter=50, tol=1e-6):
    """
    Compute geodesic distance between two points.
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric.
    p, q : tuple
        Points.
    method : {'shooting', 'optimize', 'exact'}
        Distance computation method.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    
    Returns
    -------
    float
        Geodesic distance d(p, q).
    """
    if metric.dim == 1:
        if method == 'exact':
            x = metric.coords[0]
            integrand = metric.sqrt_det_g
            return float(abs(integrate(integrand, (x, p[0], q[0]))))
        else:
            from scipy.integrate import quad
            f = metric.sqrt_det_g_func
            return abs(quad(f, p[0], q[0])[0])
    else:
        if method == 'shooting':
            # Initial guess: straight line velocity
            v_guess = np.array([q[0] - p[0], q[1] - p[1]])
            
            for iteration in range(max_iter):
                q_reached = exponential_map(metric, p, tuple(v_guess), t=1.0)
                error = np.array([q_reached[0] - q[0], q_reached[1] - q[1]])
                error_norm = np.linalg.norm(error)
                
                if error_norm < tol:
                    break
                
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
                q_reached = exponential_map(metric, p, tuple(v), t=1.0)
                endpoint_error = ((q_reached[0] - q[0])**2 + 
                                (q_reached[1] - q[1])**2)
                
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
# 2D-only functions
# ============================================================================
def hodge_star(metric, form_degree):
    """
    Compute Hodge star operator on differential forms.
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric (must be 2D).
    form_degree : int
        Degree of the form (0, 1, or 2).
    
    Returns
    -------
    callable or Matrix
        Hodge star operator.
    """
    if metric.dim != 2:
        raise ValueError("Hodge star requires 2D manifold")
    
    sqrt_g = metric.sqrt_det_g
    g_inv = metric.g_inv_matrix
    
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
    Solve Jacobi equation for geodesic deviation (optimized).
    
    Uses CubicSpline for faster interpolation.
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric (must be 2D).
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
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline
    
    if metric.dim != 2:
        raise ValueError("Jacobi equation solver requires 2D manifold")
    
    # Compute Riemann tensor symbolically
    R = metric.riemann_tensor
    
    # Lambdify Riemann tensor components
    x_sym, y_sym = metric.coords
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
    
    # Interpolate geodesic with CubicSpline (faster than interp1d)
    t_geod = geodesic['t']
    x_interp = CubicSpline(t_geod, geodesic['x'])
    y_interp = CubicSpline(t_geod, geodesic['y'])
    vx_interp = CubicSpline(t_geod, geodesic['vx'])
    vy_interp = CubicSpline(t_geod, geodesic['vy'])
    
    Gamma = metric.christoffel_func
    
    def jacobi_ode(t, state):
        J_x, J_y, DJ_x, DJ_y = state
        
        x = float(x_interp(t))
        y = float(y_interp(t))
        vx = float(vx_interp(t))
        vy = float(vy_interp(t))
        
        # Curvature term: R(J, v)v
        curv_x = 0
        curv_y = 0
        
        J = [J_x, J_y]
        v = [vx, vy]
        
        for j in range(2):
            for k in range(2):
                for ell in range(2):
                    curv_x += R_func[0][j][k][ell](x, y) * J[j] * v[k] * v[ell]
                    curv_y += R_func[1][j][k][ell](x, y) * J[j] * v[k] * v[ell]
        
        # Covariant derivative
        DDJ_x = -(Gamma[0][0][0](x, y) * DJ_x * vx +
                  Gamma[0][0][1](x, y) * (DJ_x * vy + DJ_y * vx) +
                  Gamma[0][1][1](x, y) * DJ_y * vy +
                  curv_x)
        
        DDJ_y = -(Gamma[1][0][0](x, y) * DJ_x * vx +
                  Gamma[1][0][1](x, y) * (DJ_x * vy + DJ_y * vx) +
                  Gamma[1][1][1](x, y) * DJ_y * vy +
                  curv_y)
        
        return [DJ_x, DJ_y, DDJ_x, DDJ_y]
    
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

def verify_gauss_bonnet(metric, domain, euler_char=None):
    """
    Numerically verify Gauss-Bonnet theorem.
    
    Theorem: ∫∫_M K dA = 2π χ(M)
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric (must be 2D).
    domain : tuple
        Integration domain ((x_min, x_max), (y_min, y_max)).
    euler_char : int, optional
        Euler characteristic. If None, assumes sphere (χ=2).
    
    Returns
    -------
    dict
        Results: 'integral', 'expected', 'relative_error'.
    
    Examples
    --------
    >>> # Sphere: χ = 2
    >>> result = verify_gauss_bonnet(metric_sphere, domain)
    >>> # Torus: χ = 0
    >>> result = verify_gauss_bonnet(metric_torus, domain, euler_char=0)
    """
    if metric.dim != 2:
        raise ValueError("Gauss-Bonnet requires 2D manifold")
    
    from scipy.integrate import dblquad
    
    K = metric.gauss_curvature
    sqrt_g = metric.sqrt_det_g    
    integrand = K * sqrt_g
    f = lambdify(metric.coords, integrand, 'numpy')
    
    (x0, x1), (y0, y1) = domain
    integral, _ = dblquad(lambda Y, X: f(X, Y), x0, x1, y0, y1)
    
    # Default: sphere has χ = 2
    if euler_char is None:
        euler_char = 2
    
    expected = 2 * np.pi * euler_char
    
    return {
        'integral': integral,
        'expected': expected,
        'euler_characteristic': euler_char,
        'relative_error': abs(integral - expected) / abs(expected) if expected != 0 else float('inf')
    }

# ============================================================================
# Visualization (optimized)
# ============================================================================
def visualize_geodesics(metric, initial_conditions, tspan, 
                       x_range=None, y_range=None, 
                       plot_curvature=False, n_steps=1000):
    """
    Visualize geodesics on manifold (optimized).
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric.
    initial_conditions : list
        List of initial conditions.
    tspan : tuple
        Time interval.
    x_range, y_range : tuple, optional
        Plot ranges.
    plot_curvature : bool
        Show curvature as background.
    n_steps : int
        Integration steps.
    """
    if metric.dim == 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        x_plot = np.linspace(*x_range if x_range else (0, 5), 200)
        ax1.plot(x_plot, metric.g_func[(0,0)](x_plot), 'k-', label='g₁₁(x)')
        ax1.set_ylabel('Metric component')
        ax1.legend()
        ax1.grid(True)
        
        for x0, v0 in initial_conditions:
            traj = geodesic_solver(metric, (x0,), (v0,), tspan, n_steps=n_steps)
            ax2.plot(traj['t'], traj['x'], label=f'x₀={x0}, v₀={v0}')
        
        ax2.set_xlabel('t')
        ax2.set_ylabel('x(t)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.show()
    
    else:  # 2D case
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Compute all trajectories
        trajectories = [geodesic_solver(metric, p0, v0, tspan, n_steps=n_steps)
                       for p0, v0 in initial_conditions]
        
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
                
                K_expr = metric.gauss_curvature
                K_func = lambdify(metric.coords, K_expr, 'numpy')
                K_vals = K_func(X_bg, Y_bg)
                
                im = ax.pcolormesh(X_bg, Y_bg, K_vals, shading='auto',
                                  cmap='RdBu_r', alpha=0.3, vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax, label='Gaussian Curvature')
            except Exception as e:
                print(f"Warning: Could not compute curvature background: {e}")
        
        # Plot geodesics
        for idx, traj in enumerate(trajectories):
            p0, v0 = initial_conditions[idx]
            
            # Color gradient by time
            colors = plt.cm.viridis(np.linspace(0, 1, len(traj['x'])))
            
            # Plot trajectory segments
            for i in range(len(traj['x']) - 1):
                ax.plot(traj['x'][i:i+2], traj['y'][i:i+2],
                       color=colors[i], alpha=0.8, linewidth=2)
            
            # Mark start and end
            ax.plot(traj['x'][0], traj['y'][0], 'go', markersize=10)
            ax.plot(traj['x'][-1], traj['y'][-1], 'ro', markersize=10)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Geodesics on Riemannian Manifold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        plt.tight_layout()
        plt.show()

def visualize_curvature(metric, x_range, y_range, resolution=50, 
                       quantity='gauss', cmap='RdBu_r'):
    """
    Visualize curvature on 2D manifold (optimized).
    
    Parameters
    ----------
    metric : Metric
        Riemannian metric (must be 2D).
    x_range, y_range : tuple
        Domain ranges.
    resolution : int
        Grid resolution.
    quantity : str
        Curvature to plot: 'gauss' or 'ricci_scalar'.
    cmap : str
        Colormap name.
    """
    if metric.dim != 2:
        raise ValueError("Curvature visualization requires 2D manifold")
    
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    if quantity == 'gauss':
        K_expr = metric.gauss_curvature
        K_func = lambdify(metric.coords, K_expr, 'numpy')
        Z = K_func(X, Y)
        title = 'Gaussian Curvature K(x,y)'
    elif quantity == 'ricci_scalar':
        R_expr = metric.ricci_scalar
        R_func = lambdify(metric.coords, R_expr, 'numpy')
        Z = R_func(X, Y)
        title = 'Ricci Scalar R(x,y)'
    else:
        raise ValueError("quantity must be 'gauss' or 'ricci_scalar'")
    
    # Handle constant curvature case
    Z = np.array(Z)
    if Z.ndim == 0:
        Z = np.full_like(X, float(Z))
    
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
# Backward compatibility
# ============================================================================
class Metric1D(Metric):
    """Backward compatibility wrapper for 1D metrics."""
    def __init__(self, g_expr, x, simplify=False):
        super().__init__(g_expr, (x,), simplify=simplify)

class Metric2D(Metric):
    """Backward compatibility wrapper for 2D metrics."""
    def __init__(self, g_matrix, coords, simplify=False):
        super().__init__(g_matrix, coords, simplify=simplify)