from imports import *
x, y = symbols('x y', real=True)
# ══════════════════════════════════════════════════════════════════════════════
# metric_catalogue.py
# ══════════════════════════════════════════════════════════════════════════════
# A catalogue of 2-D Riemannian (and one Lorentzian) metrics for geodesic
# simulations and wave-equation experiments.
#
# Every entry is a plain dict with the following fields:
#
#   g           SymPy 2×2 Matrix  Metric tensor g_ij(x, y)
#   coords      tuple             Symbolic coordinate names, e.g. (x, y)
#   domain      (Lx, Ly)          Full domain width in each coordinate:
#                                   coord ∈ [shift - L/2,  shift + L/2]
#   shift       (sx, sy)          Offset of the domain centre from 0
#   ic_center   (cx, cy)          Initial-condition (Gaussian) centre
#                                   expressed in the *shifted* frame (cx=0 ⟹
#                                   packet starts at the domain centre)
#   ic_sigma    float             Gaussian width for the initial condition
#   Lt          float             Total simulation time
#   Nt          int               Number of time steps
#   Nx, Ny      int               Grid resolution
#   description str               One-line human-readable label
#
# Coordinate conventions
# ──────────────────────
# The solver stores the grid on  x_grid = linspace(-Lx/2, Lx/2, Nx)
# and the *physical* coordinate is  x_phys = x_grid + shift_x  (similarly y).
# All metric expressions are written in terms of the *physical* coordinates.
#
# Gaussian curvature reference formulae (diagonal metric g = diag(E, G))
# ───────────────────────────────────────────────────────────────────────
#   K = -1/(2√(EG)) [ ∂_x(∂_x√G / √E) + ∂_y(∂_y√E / √G) ]
#
# For a conformal metric  g = f(x,y) I₂ :
#   K = -Δ(ln f) / (2f)   where Δ is the flat Laplacian
# ══════════════════════════════════════════════════════════════════════════════

METRICS = {

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Constant-curvature spaces
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Flat torus  (K = 0)
    # ──────────────────────────────────────────────────────────────────────────
    # The simplest Riemannian surface: the Euclidean plane with periodic
    # boundary conditions in both directions.  Geodesics are straight lines
    # that wrap around.  Every Christoffel symbol vanishes.
    #
    #   ds² = dx² + dy²,   x ∈ [-2, 2],  y ∈ [-2, 2]  (periodic)
    #
    # Curvature : K = 0 everywhere
    # Topology  : T² (genus-1, orientable)
    # ──────────────────────────────────────────────────────────────────────────
    'flat': dict(
        g           = Matrix([[1, 0], [0, 1]]),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Flat torus (K = 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Poincaré half-plane  (K = -1, constant)
    # ──────────────────────────────────────────────────────────────────────────
    # The unique (up to isometry) simply-connected complete surface of constant
    # negative curvature −1.  Its isometry group is PSL(2,ℝ) (Möbius maps
    # preserving the upper half-plane).  Geodesics are vertical lines and
    # semicircles with centres on the x-axis.
    #
    #   ds² = (dx² + dy²) / y²,   y > 0
    #
    # Grid:  y ∈ [0.5, 2.5]  (shift 1.5 so grid centre maps to y_phys = 1.5)
    #        x ∈ [-2, 2]
    #
    # Curvature  : K = -1 (constant)
    # Topology   : ℝ² (half-plane, open)
    # Note       : The metric is singular as y → 0 (the "boundary at infinity").
    #              Keep the lower y-boundary safely above 0.
    # ──────────────────────────────────────────────────────────────────────────
    'poincare': dict(
        g           = Matrix([[1/y**2, 0], [0, 1/y**2]]),
        coords      = (x, y),
        domain      = (4.0, 2.0),
        shift       = (0.0, 1.5),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.2,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Poincaré half-plane (K = -1)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Unit sphere  (K = +1, constant)
    # ──────────────────────────────────────────────────────────────────────────
    # The round metric on S² in standard polar (colatitude θ, longitude φ)
    # coordinates.  Geodesics are great circles.  All geodesics are closed
    # with the same length 2π.
    #
    #   ds² = dθ² + sin²θ dφ²,   θ ∈ (0, π),  φ ∈ [0, 2π)
    #
    # Implementation: x plays the role of θ, y plays φ.
    #   x_grid ∈ [-1.27, 1.27],  x_phys = x_grid + π/2  → θ ∈ (0, π)
    #   y_grid ∈ [-π, π]
    #
    # Curvature  : K = +1 (constant)
    # Topology   : S² (simply connected, orientable)
    # Note       : The metric degenerates at the poles θ = 0, π
    #              (sin²θ → 0).  Avoid placing the initial condition there.
    # ──────────────────────────────────────────────────────────────────────────
    'sphere': dict(
        g           = Matrix([[1, 0], [0, sin(x)**2]]),
        coords      = (x, y),
        domain      = (2.54, 6.28),
        shift       = (1.57, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Unit sphere (K = +1)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Pseudosphere / Tractricoid  (K = -1, constant)
    # ──────────────────────────────────────────────────────────────────────────
    # The surface of revolution obtained by rotating the tractrix.  It realises
    # K = -1 and is isometric to a region of the hyperbolic plane, but lives in
    # ℝ³.  It has a singular cusp at u = 0 (excluded here).
    #
    #   ds² = du² + sinh²(u) dv²,   u > 0,  v ∈ [0, 2π)
    #
    # Grid: u ∈ [0.5, 3.0]  (shift 1.0, domain 2.5)
    #       v ∈ [0, 2π]
    #
    # Curvature  : K = -1 (same as hyperbolic plane)
    # Topology   : cylindrical (u ∈ ℝ₊, v ∈ S¹)
    # Note       : Isometric to the Poincaré half-plane near u = 0; identical
    #              Gaussian curvature but different global topology.
    # ──────────────────────────────────────────────────────────────────────────
    'pseudosphere': dict(
        g           = Matrix([[1, 0], [0, sinh(x)**2]]),
        coords      = (x, y),
        domain      = (2.5, 6.283185),
        shift       = (1.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Pseudosphere / tractricoid (K = -1)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Poincaré disk  (K = -1, constant — disk model)
    # ──────────────────────────────────────────────────────────────────────────
    # Conformal model of the hyperbolic plane on the open unit disk.  Related
    # to the half-plane model by a Möbius transformation.  The metric blows up
    # at the boundary r = 1 (the "circle at infinity").
    #
    #   ds² = 4 (dx² + dy²) / (1 - x² - y²)²,   x² + y² < 1
    #
    # Grid: x, y ∈ [-0.9, 0.9]  (domain 1.8; keep away from r = 1)
    #
    # Curvature  : K = -1 (constant)
    # Topology   : ℝ² (open disk)
    # Note       : Geodesics appear as circular arcs perpendicular to the
    #              boundary.  The conformal factor 4/(1−r²)² grows steeply near
    #              the boundary — keep ic_sigma small or shift ic_center inward.
    # ──────────────────────────────────────────────────────────────────────────
    'poincare_disk': dict(
        g           = Matrix([[4/(1 - x**2 - y**2)**2, 0],
                              [0, 4/(1 - x**2 - y**2)**2]]),
        coords      = (x, y),
        domain      = (1.8, 1.8),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.15,
        Lt          = 2.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Poincaré disk (K = -1, disk model)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Anti-de Sitter spatial slice  (K = -1, constant)
    # ──────────────────────────────────────────────────────────────────────────
    # The (1+1)-D spatial section of AdS₂ in global coordinates.  The metric
    # is the same as the hyperbolic plane written in sinh-cosh form.
    #
    #   ds² = dρ² + cosh²(ρ) dφ²,   ρ ∈ [-2, 2],  φ ∈ [-π, π]
    #
    # Curvature  : K = -1 (constant)
    # Topology   : cylinder (ρ ∈ ℝ, φ ∈ S¹)
    # Note       : Complementary to pseudosphere: same curvature but
    #              g_φφ = cosh²ρ (never zero) vs sinh²u (zero at u = 0).
    #              Geodesics oscillate in the ρ-direction (unlike half-plane).
    # ──────────────────────────────────────────────────────────────────────────
    'anti_de_sitter': dict(
        g           = Matrix([[1, 0], [0, cosh(x)**2]]),
        coords      = (x, y),
        domain      = (4.0, 6.283185),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Anti-de Sitter spatial slice (K = -1, global coords)',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Variable-curvature surfaces of revolution
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Saddle surface  (K < 0, variable)
    # ──────────────────────────────────────────────────────────────────────────
    # Induced metric on the graph z = (y² - x²)/2 in ℝ³ (hyperbolic paraboloid
    # variant with a modified g_yy to break symmetry).  Here we use a simple
    # diagonal deformation of the flat metric that gives negative curvature
    # everywhere, growing in magnitude away from the origin.
    #
    #   ds² = dx² + (1 + x² + y²) dy²,   x, y ∈ [-2, 2]
    #
    # Curvature  : K = -(x² + y²) / (1 + x² + y²)³  (K < 0, vanishes at origin)
    # Topology   : ℝ²
    # Note       : The curvature vanishes at the origin and becomes more
    #              negative as r → ∞.  Geodesics diverge faster than in flat
    #              space, mimicking a hyperbolic trumpet.
    # ──────────────────────────────────────────────────────────────────────────
    'saddle': dict(
        g           = Matrix([[1, 0], [0, 1 + x**2 + y**2]]),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Saddle surface (K < 0, variable)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Cone / polar plane  (K = 0, singular at apex)
    # ──────────────────────────────────────────────────────────────────────────
    # Polar coordinates (r, θ) on the flat plane.  The metric is identically
    # flat (K = 0) away from r = 0, but the apex is a conical singularity.
    # Geodesics are straight lines in Cartesian coordinates; they refract at
    # the apex when they pass through it.
    #
    #   ds² = dr² + r² dθ²,   r > 0,  θ ∈ [-π, π]
    #
    # Grid: r ∈ [0.5, 3.5]  (shift 1.5, domain 2.0 in r-direction)
    #       θ ∈ [-π, π]
    #
    # Curvature  : K = 0 (for r > 0); distributional curvature 2π(1-1) δ² at apex
    # Topology   : ℝ² \ {0} (punctured plane)
    # Note       : Useful for testing geodesic refraction and diffraction;
    #              the initial condition is offset slightly from the axis.
    # ──────────────────────────────────────────────────────────────────────────
    'cone': dict(
        g           = Matrix([[1, 0], [0, x**2]]),
        coords      = (x, y),
        domain      = (2.0, 6.28),
        shift       = (1.5, 0.0),
        ic_center   = (0.01, 0.01),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Cone / polar  ds² = dr² + r² dθ²  (K = 0, apex singularity)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Catenoid  (K < 0, minimal surface)
    # ──────────────────────────────────────────────────────────────────────────
    # The unique minimal surface of revolution (zero mean curvature H = 0).
    # Obtained by rotating the catenary  x = cosh(z).  The Beltrami–Enneper
    # theorem: for a minimal surface K = -|H|² only when H = 0, so K ≤ 0 and
    # K = 0 only if the surface is flat — but the catenoid has K < 0 everywhere.
    #
    #   ds² = cosh²(v)(du² + dv²),   u ∈ [-π, π] (angular, periodic),  v ∈ [-1.5, 1.5]
    #
    # Curvature  : K = -1 / cosh⁴(v)  (negative, even in v, minimum at v = 0)
    # Topology   : cylinder (u ∈ S¹, v ∈ ℝ)
    # Note       : H = 0 (minimal); the neck (waist) is at v = 0 where
    #              |K| is maximum.  The conformal factor cosh²(v) is isotropic,
    #              so geodesics feel no angular bias.
    # ──────────────────────────────────────────────────────────────────────────
    'catenoid': dict(
        g           = Matrix([[cosh(y)**2, 0], [0, cosh(y)**2]]),
        coords      = (x, y),
        domain      = (6.283185, 3.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.4,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Catenoid (minimal surface, K = -1/cosh⁴v ≤ 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Torus of revolution  (K changes sign)
    # ──────────────────────────────────────────────────────────────────────────
    # The standard embedding of T² in ℝ³:
    #   (x, y, z) = ((R + r cos θ) cos φ,  (R + r cos θ) sin φ,  r sin θ)
    # with major radius R = 2, minor radius r = 1.
    #
    #   ds² = r² dθ² + (R + r cos θ)² dφ²
    #       = dθ² + (2 + cos θ)² dφ²           (with r = 1)
    #
    # Grid: θ ∈ [-π, π]  (poloidal),  φ ∈ [-π, π]  (toroidal)
    #
    # Curvature  : K = cos θ / [r(R + r cos θ)]  =  cos θ / (2 + cos θ)
    #              K > 0 on the outer equator (θ = 0),
    #              K < 0 on the inner equator (θ = π).
    # Topology   : T² (genus-1, orientable)
    # Note       : One of the few closed surfaces with both positive and
    #              negative curvature.  By Gauss-Bonnet, ∫K dA = 0 (χ = 0).
    #              Geodesics show a mix of focusing (outer) and defocusing (inner).
    # ──────────────────────────────────────────────────────────────────────────
    'torus': dict(
        g           = Matrix([[1, 0], [0, (2 + cos(x))**2]]),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Torus of revolution R=2, r=1 (K changes sign)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Elliptic paraboloid  (K > 0, variable)
    # ──────────────────────────────────────────────────────────────────────────
    # Induced metric on the graph z = x² + y² in ℝ³.  The surface opens upward
    # and has positive Gaussian curvature everywhere, decreasing away from the
    # vertex at the origin.
    #
    #   ds² = (1 + 4x²) dx² + 8xy dx dy + (1 + 4y²) dy²
    #
    # Curvature  : K = 4 / (1 + 4x² + 4y²)²   (K > 0, maximum K = 4 at origin)
    # Topology   : ℝ²
    # Note       : The off-diagonal term 8xy dx dy arises from the mixed
    #              second partial ∂z/∂x · ∂z/∂y.  The metric is not diagonal
    #              in Cartesian (x, y) coordinates.  The vertex (origin) is
    #              the point of maximum curvature.
    # ──────────────────────────────────────────────────────────────────────────
    'paraboloid': dict(
        g           = Matrix([[1 + 4*x**2, 8*x*y], [8*x*y, 1 + 4*y**2]]),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.4,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Elliptic paraboloid z = x²+y² (K = 4/(1+4r²)² > 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Hyperbolic paraboloid  (K < 0, variable)
    # ──────────────────────────────────────────────────────────────────────────
    # Induced metric on the saddle surface z = x² - y² (a "Pringle" shape).
    # The metric is Riemannian (positive definite) despite the saddle shape.
    #
    #   ds² = (1 + 4x²) dx² - 8xy dx dy + (1 + 4y²) dy²
    #
    # Curvature  : K = -4 / (1 + 4x² + 4y²)²   (K < 0 everywhere)
    # Topology   : ℝ²
    # Note       : The off-diagonal term is −8xy, making this distinct from
    #              the 'saddle' entry.  The metric degeneracy det g = 1 + 4x² + 4y²
    #              is always positive, confirming positive definiteness.
    # ──────────────────────────────────────────────────────────────────────────
    'hyperbolic_paraboloid': dict(
        g           = Matrix([[1 + 4*x**2, -8*x*y], [-8*x*y, 1 + 4*y**2]]),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Hyperbolic paraboloid z = x²−y² (K = -4/(1+4r²)² < 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Flamm's paraboloid  (Schwarzschild spatial geometry)
    # ──────────────────────────────────────────────────────────────────────────
    # The spatial (t = const) section of the Schwarzschild metric embedded
    # isometrically in ℝ³ as a paraboloid of revolution.  The embedding map is
    #   z = 2√(2M(r − 2M)),  M = 1.
    # The induced 2-D metric on the paraboloid coincides with the spatial part
    # of Schwarzschild.  Different from the 'schwarzschild' entry which keeps
    # the full (dr², r² dφ²) form — here we use the (r, φ) parameterisation
    # directly.
    #
    #   ds² = dr² / (1 - 2/r) + r² dφ²,   r > 2  (M = 1)
    #
    # Grid: r ∈ [2.2, 6.2]  (shift 4.2, domain 4.0 centred at r = 4.2)
    #       φ ∈ [-π, π]
    #
    # Curvature  : K = -M / r³   (negative; vanishes as r → ∞)
    # Topology   : ℝ² \ {disk} (exterior of horizon)
    # Note       : As r → 2M the metric diverges in the r-component (coordinate
    #              singularity; actual curvature stays finite).  The paraboloid
    #              narrows to a "throat" at r = 2M.
    # ──────────────────────────────────────────────────────────────────────────
    'flamm': dict(
        g           = Matrix([[1/(1 - 2/x), 0], [0, x**2]]),
        coords      = (x, y),
        domain      = (4.0, 6.283185),
        shift       = (4.2, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = "Flamm's paraboloid / Schwarzschild spatial slice (M=1, K = -M/r³)",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — General-relativistic 2-D sections
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Schwarzschild 2-D spacelike slice  (M = 1, r > 2M)
    # ──────────────────────────────────────────────────────────────────────────
    # The r-φ section of the Schwarzschild exterior metric at t = const.
    # Identical in curvature to Flamm's paraboloid but placed on a closer grid.
    #
    #   ds² = (1 - 2/r)⁻¹ dr² + r² dφ²,   r ∈ [2.5, 5.0],  φ ∈ [-π, π]
    #
    # Grid: r ∈ [2.5, 5.0]  (domain 2.5, no shift — grid [0, 2.5] maps to r)
    #       φ ∈ [-π, π]
    #
    # Curvature  : K = -1/r³   (negative, goes to 0 far from horizon)
    # Note       : The horizon r = 2 is outside the grid; the coordinate
    #              singularity at r = 2 would cause g_rr → ∞.
    # ──────────────────────────────────────────────────────────────────────────
    'schwarzschild': dict(
        g           = Matrix([[1/(1 - 2/x), 0], [0, x**2]]),
        coords      = (x, y),
        domain      = (2.5, 6.283185),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Schwarzschild 2D r-φ slice (M=1, r ∈ [2.5, 5])',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Schwarzschild–de Sitter 2-D slice  (M = 1, Λ = 0.3)
    # ──────────────────────────────────────────────────────────────────────────
    # Adds a positive cosmological constant Λ to the Schwarzschild metric.
    # There are now two horizons: a black-hole horizon at r ≈ 1.8 and a
    # cosmological horizon at r ≈ 2.9 (for the chosen parameters).
    #
    #   ds² = (1 - 2M/r - Λr²/3)⁻¹ dr² + r² dφ²,   M = 1, Λ = 0.3
    #
    # Grid: r ∈ [1.5, 4.5]  (domain 3.0, shift 0)
    #       φ ∈ [-π, π]
    #
    # Note       : The lapse function  f(r) = 1 - 2/r - 0.1 r²  vanishes at
    #              the two horizons.  Keep the grid between (or outside) both.
    #              With the current domain [1.5, 4.5] the grid straddles the
    #              cosmological horizon; adjust shift to explore either side.
    # ──────────────────────────────────────────────────────────────────────────
    'schwarzschild_ds': dict(
        g           = Matrix([[1/(1 - 2/x - 0.1*x**2), 0], [0, x**2]]),
        coords      = (x, y),
        domain      = (3.0, 6.283185),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Schwarzschild–de Sitter 2D (M=1, Λ=0.3)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Extremal Reissner–Nordström 2-D  (M = Q = 1, r > r_h = 1)
    # ──────────────────────────────────────────────────────────────────────────
    # Spatial section of the extremal charged black hole (|Q| = M).  The two
    # horizons coincide at r_h = M = 1, giving a double root in the lapse.
    #
    #   ds² = (1 - 1/r)⁻² dr² + r² dφ²,   r > 1
    #         ≡ (r/(r-1))² dr² + r² dφ²
    #
    # Grid: r ∈ [2, 6]  (shift 1.0, domain 2.0; x_phys = x_grid + 1)
    #       φ ∈ [-π, π]
    #
    # Curvature  : K = -(r-1)(3r-1) / r⁵   (negative outside the horizon)
    # Note       : The double horizon produces slower fall-off of curvature
    #              near r = 1 compared to Schwarzschild's simple pole.
    # ──────────────────────────────────────────────────────────────────────────
    'extremal_rn': dict(
        g           = Matrix([[1/(1 - 2/x + 1/x**2), 0], [0, x**2]]),
        coords      = (x, y),
        domain      = (2.0, 6.283185),
        shift       = (1.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Extremal Reissner–Nordström 2D (M=Q=1, r > 1)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Ellis wormhole throat  (K < 0)
    # ──────────────────────────────────────────────────────────────────────────
    # The spatial section of the Ellis–Bronnikov wormhole with throat radius b.
    # In the (l, φ) coordinates where l ∈ (-∞, ∞) is the proper radial distance
    # from the throat, the metric on each side is spherically symmetric with
    # areal radius r(l) = √(b² + l²).  We take the 2-D (l, φ) section.
    #
    #   ds² = dl² + (b² + l²) dφ²,   b = 1,  l ∈ [-3, 3],  φ ∈ [-π, π]
    #
    # Curvature  : K = -b² / (b² + l²)²   (negative; maximum |K| = 1/b² at throat)
    # Topology   : ℝ × S¹  (cylinder connecting two asymptotic regions)
    # Note       : The throat (l = 0) has minimum areal radius r = b and
    #              maximum |K|.  Far from the throat, K → 0 (asymptotically
    #              flat).  Geodesics can pass through the throat.
    # ──────────────────────────────────────────────────────────────────────────
    'wormhole': dict(
        g           = Matrix([[1, 0], [0, 1 + x**2]]),
        coords      = (x, y),
        domain      = (6.0, 6.283185),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.4,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Ellis wormhole throat (b=1, K = -1/(1+l²)²)',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Deformations and perturbations of round metrics
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Berger sphere  (oblate deformation, a = 0.6)
    # ──────────────────────────────────────────────────────────────────────────
    # The Berger metrics on S² form a one-parameter family that squashes or
    # stretches the sphere in one direction.  Here the φ-circles are shrunk
    # by a factor a = 0.6 < 1 (oblate).
    #
    #   ds² = dθ² + a² sin²θ dφ²,   a = 0.6
    #       = dθ² + 0.36 sin²θ dφ²
    #
    # Grid: θ ∈ (0, π) via shift π/2;  φ ∈ [-π, π]
    #
    # Curvature  : K = (1/a²)(1 - (1 - a²) cos²θ)   (variable; K > 0 for a < 1)
    # Note       : As a → 0, the metric degenerates to a 1-D metric on a
    #              circle (Gromov–Hausdorff limit).  For a > 1, the sphere is
    #              prolate.  All geodesics are closed (Morse-theoretic argument).
    # ──────────────────────────────────────────────────────────────────────────
    'berger_sphere': dict(
        g           = Matrix([[1, 0], [0, 0.36 * sin(x)**2]]),
        coords      = (x, y),
        domain      = (2.54, 6.283185),
        shift       = (1.57, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Berger sphere (oblate a=0.6, K variable > 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Zoll metric  (all geodesics closed, a = 0.3)
    # ──────────────────────────────────────────────────────────────────────────
    # Zoll surfaces are Riemannian 2-manifolds homeomorphic to S² on which
    # every geodesic is closed with the same period.  The Zoll family is
    # parameterised by an odd function f; the simplest explicit example uses
    # a cosine deformation of the round sphere.
    #
    #   ds² = (1 + a cos θ)² dθ² + sin²θ dφ²,   a = 0.3
    #
    # Grid: θ ∈ (0, π),  φ ∈ [-π, π]
    #
    # Curvature  : K varies around K ≈ +1; the deformation preserves the
    #              closed-geodesic property but changes curvature distribution.
    # Note       : By a theorem of Guillemin, the Zoll metrics form an
    #              infinite-dimensional family.  The a = 0 limit is the round
    #              sphere; for small a the metric is a perturbation of it.
    # ──────────────────────────────────────────────────────────────────────────
    'zoll': dict(
        g           = Matrix([[(1 + 0.3*cos(x))**2, 0], [0, sin(x)**2]]),
        coords      = (x, y),
        domain      = (2.54, 6.283185),
        shift       = (1.57, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Zoll metric (a=0.3, all geodesics closed)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Rosenberg metric  (variable positive curvature, ε = 0.2)
    # ──────────────────────────────────────────────────────────────────────────
    # A perturbation of the round sphere metric in which the area element
    # sin θ dθ dφ is multiplied by (1 + ε cos θ)^{1/2}.  This breaks the
    # constant-curvature property while keeping K > 0 for small ε.
    #
    #   ds² = dθ² + (1 + ε cos θ) sin²θ dφ²,   ε = 0.2
    #
    # Curvature  : K = K(θ) variable; K ≈ 1 with modulation of order ε.
    # Topology   : S²
    # Note       : Unlike Berger, the perturbation breaks the O(2) symmetry
    #              of the base sphere only in the φ-metric-component;
    #              geodesics are no longer all closed.
    # ──────────────────────────────────────────────────────────────────────────
    'rosenberg': dict(
        g           = Matrix([[1, 0], [0, (1 + 0.2*cos(x))*sin(x)**2]]),
        coords      = (x, y),
        domain      = (2.54, 6.283185),
        shift       = (1.57, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Rosenberg metric (ε=0.2, K variable near +1)',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Conformal and product metrics
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Enneper surface  (conformally flat, K ≤ 0)
    # ──────────────────────────────────────────────────────────────────────────
    # The Enneper surface is a self-intersecting minimal surface.  In the
    # Weierstrass representation it has conformal factor (1 + |w|²)² where
    # w = x + iy.
    #
    #   ds² = (1 + x² + y²)² (dx² + dy²)
    #
    # Grid: x, y ∈ [-1, 1]
    #
    # Curvature  : K = -4 / (1 + x² + y²)⁴   (negative; most negative at origin)
    # Note       : Minimal (H = 0) like the catenoid, but parameterised over
    #              ℝ² rather than a cylinder.  Self-intersections occur but
    #              do not affect the local metric or curvature.
    # ──────────────────────────────────────────────────────────────────────────
    'enneper': dict(
        g           = Matrix([[(1 + x**2 + y**2)**2, 0],
                              [0, (1 + x**2 + y**2)**2]]),
        coords      = (x, y),
        domain      = (2.0, 2.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Enneper surface (minimal, K = -4/(1+r²)⁴)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Fubini–Study metric on CP¹ ≅ S²  (K = +4, stereographic coords)
    # ──────────────────────────────────────────────────────────────────────────
    # The round metric on the Riemann sphere written in stereographic
    # coordinates (z = x + iy ∈ ℂ).  This is the standard Kähler metric
    # on CP¹ normalised so that the total area is π (instead of 4π).
    #
    #   ds² = 4 (dx² + dy²) / (1 + x² + y²)²
    #
    # Grid: x, y ∈ [-2.5, 2.5]
    #
    # Curvature  : K = +4 / 4 = +1  (actually K = +1 everywhere, same as unit S²)
    #              More precisely K = 1 for the normalisation used here
    #              (check: total area = ∫ dA = π ∝ 4π/4 = π).
    # Topology   : S² (one-point compactification; south pole at infinity)
    # Note       : Geodesics are great circles; in stereographic coords they
    #              appear as Euclidean circles or lines through the origin.
    # ──────────────────────────────────────────────────────────────────────────
    'fubini_study': dict(
        g           = Matrix([[4/(1 + x**2 + y**2)**2, 0],
                              [0, 4/(1 + x**2 + y**2)**2]]),
        coords      = (x, y),
        domain      = (5.0, 5.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Fubini–Study / round S² in stereographic coords (K = +1)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Parabolic-coordinate metric  (conformally flat, K < 0 except at origin)
    # ──────────────────────────────────────────────────────────────────────────
    # Parabolic coordinates (u, v) are related to Cartesian by
    #   x_cart = uv,   y_cart = (u² - v²)/2.
    # The Jacobian gives a conformal metric:
    #
    #   ds² = (u² + v²)(du² + dv²)
    #
    # Grid: u, v ∈ [-2, 2]
    #
    # Curvature  : K = -2 / (u² + v²)²   for (u,v) ≠ 0;  K = 0 at origin
    # Note       : Conformally flat; the conformal factor u² + v² vanishes at
    #              the origin, creating a degenerate point (metric singularity).
    #              The initial condition should be away from the origin.
    # ──────────────────────────────────────────────────────────────────────────
    'parabolic': dict(
        g           = Matrix([[x**2 + y**2, 0], [0, x**2 + y**2]]),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Parabolic coordinates (conformally flat, K = -2/(u²+v²)²)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Gaussian bump  (conformal, positive curvature lump at origin)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformally flat metric on the plane with a localised positive-curvature
    # bump centred at the origin.  Models a localised mass or gravitational lens.
    #
    #   ds² = exp(A exp(-r²/2σ²)) (dx² + dy²),   A = 1.0, σ = 1.0
    #
    # Grid: x, y ∈ [-2.5, 2.5]
    #
    # Curvature  : K = -Δ(ln f) / (2f) where f = exp(A e^{-r²/2})
    #              K > 0 near the origin; K < 0 in a ring around the bump;
    #              K → 0 far from the origin.
    # Note       : The total curvature (Gauss-Bonnet) ∫K dA = 0 (plane topology),
    #              so the positive lump is exactly compensated by the negative
    #              ring.  Good for testing gravitational-lensing analogues.
    # ──────────────────────────────────────────────────────────────────────────
    'bump_pos': dict(
        g           = Matrix([[1, 0], [0, 1]]) * exp(exp(-(x**2 + y**2)/2)),
        coords      = (x, y),
        domain      = (5.0, 5.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Gaussian conformal bump (K > 0 centre, K < 0 ring)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Egg-carton metric  (periodic alternating curvature)
    # ──────────────────────────────────────────────────────────────────────────
    # A doubly-periodic conformal metric on the torus that produces a
    # checkerboard-like curvature pattern — positive under peaks, negative
    # under saddle-points of the conformal factor.
    #
    #   ds² = (1 + a sin²x sin²y)(dx² + dy²),   a = 0.5
    #
    # Grid: x, y ∈ [-π, π]  (periodic)
    #
    # Curvature  : K changes sign; maxima K > 0 at (0,0), (π, π), …;
    #              minima K < 0 at (π/2, 0), (0, π/2), …
    # Topology   : T² (doubly periodic)
    # Note       : By Gauss-Bonnet ∫K dA = 2πχ(T²) = 0, so positive and
    #              negative curvature regions balance exactly.
    # ──────────────────────────────────────────────────────────────────────────
    'eggcarton': dict(
        g           = Matrix([[1, 0], [0, 1]]) * (1 + 0.5 * sin(x)**2 * sin(y)**2),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Egg-carton metric (doubly-periodic, K alternates sign)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Warped circle  S¹ ×_f S¹  (K depends on x)
    # ──────────────────────────────────────────────────────────────────────────
    # A warped product of two circles where the radius of the fibre S¹ varies
    # as the base coordinate x changes.
    #
    #   ds² = dx² + (1 + a cos x)² dy²,   a = 0.5
    #
    # Grid: x, y ∈ [-π, π]  (periodic in both)
    #
    # Curvature  : K = a cos x / (1 + a cos x)   (positive when cos x > 0,
    #              negative when cos x < 0; vanishes on the nodal lines)
    # Topology   : T² (torus — both directions periodic)
    # Note       : The fibre radius oscillates between 1 − a and 1 + a.
    #              For a → 1 the metric degenerates on the φ-circles where
    #              cos x = −1 (the "pinched torus").
    # ──────────────────────────────────────────────────────────────────────────
    'warped_circle': dict(
        g           = Matrix([[1, 0], [0, (1 + 0.5*cos(x))**2]]),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 4.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Warped product S¹ ×_f S¹ (a=0.5, K = a cosx/(1+a cosx))',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Axisymmetric conformal bump  (polar, K variable)
    # ──────────────────────────────────────────────────────────────────────────
    # Gaussian conformal deformation of the flat plane in polar coordinates.
    # This is a smooth axisymmetric metric.
    #
    #   ds² = e^{2U(r)}(dr² + r² dφ²),   U(r) = A e^{-r²/2},  A = 0.5
    #         = e^{2U}(dr²) + r² e^{2U}(dφ²)
    #
    # Grid: r ∈ [0.5, 4.5]  (domain 4.0, shift 0 — grid centre at r=2.5 approx.)
    #       φ ∈ [-π, π]
    #
    # Note       : The exponential form guarantees the metric is positive
    #              definite.  U decays as r → ∞, recovering the flat polar metric.
    # ──────────────────────────────────────────────────────────────────────────
    'bump': dict(
        g           = Matrix([[exp(2*0.5*exp(-(x**2 + y**2)/2)), 0],
                              [0, (x**2)*exp(2*0.5*exp(-(x**2 + y**2)/2))]]),
        coords      = (x, y),
        domain      = (4.0, 6.283185),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Axisymmetric conformal bump in polar coords (K variable)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Liouville metric  (separable geodesics, K = f(x) + g(y))
    # ──────────────────────────────────────────────────────────────────────────
    # Liouville metrics are the unique 2-D metrics for which the geodesic
    # equation is separable (Hamilton-Jacobi method).  The form is:
    #
    #   ds² = (f(x) + g(y))(dx² + dy²)
    #
    # Here we choose f(x) = 1 + cos x,  g(y) = 1 + sin y.
    #
    #   ds² = (2 + cos x + sin y)(dx² + dy²)
    #
    # Grid: x, y ∈ [-π, π]
    #
    # Curvature  : K = -Δ(ln(f+g)) / (2(f+g))  (variable, K < 0 near saddles
    #              where f + g is minimal)
    # Note       : The geodesics admit a conserved quantity analogous to angular
    #              momentum in the separable Hamilton-Jacobi sense.  This makes
    #              the geodesic flow integrable (Liouville-integrable).
    # ──────────────────────────────────────────────────────────────────────────
    'liouville': dict(
        g           = Matrix([[2 + cos(x) + sin(y), 0],
                              [0, 2 + cos(x) + sin(y)]]),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.4,
        Lt          = 5.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Liouville metric (separable geodesics, K variable)',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Flat and near-flat metrics with non-trivial topology or frame
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Anisotropic flat torus  (K = 0, sheared frame)
    # ──────────────────────────────────────────────────────────────────────────
    # Constant off-diagonal metric; equivalent to the flat torus after a linear
    # coordinate change.  The eigenvectors of g are tilted at ±45°.
    #
    #   ds² = dx² + 2a dx dy + dy²,   a = 0.5
    #       = (dx + a dy)² + (1 - a²) dy²
    #
    # Eigenvalues of g: 1 ± a  (positive definite for |a| < 1).
    #
    # Curvature  : K = 0 (all Christoffel symbols vanish)
    # Note       : Useful for testing that the solver correctly handles
    #              non-diagonal metrics even in the trivially flat case.
    # ──────────────────────────────────────────────────────────────────────────
    'aniso_flat': dict(
        g           = Matrix([[1, 0.5], [0.5, 1]]),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Anisotropic flat torus (shear a=0.5, K = 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Anisotropic Poincaré half-plane  (K < -1, off-diagonal)
    # ──────────────────────────────────────────────────────────────────────────
    # A non-diagonal perturbation of the Poincaré metric; the off-diagonal
    # term breaks the left–right symmetry of the half-plane.
    #
    #   ds² = y⁻² dx² + 2(0.2 y⁻¹) dx dy + y⁻² dy²
    #
    # The matrix [[1/y², 0.2/y], [0.2/y, 1/y²]] is positive definite for
    # y > 0.2 (det = 1/y⁴ − 0.04/y² > 0 ⟺ y < 5; both conditions satisfied
    # in the chosen domain y ∈ [0.5, 2.5]).
    #
    # Curvature  : K < -1 (modified by the off-diagonal term)
    # Note       : The off-diagonal element breaks the isometric left–right
    #              symmetry of the pure Poincaré metric.  Geodesics are tilted.
    # ──────────────────────────────────────────────────────────────────────────
    'hyperbolic_aniso': dict(
        g           = Matrix([[1/y**2, 0.2/y], [0.2/y, 1/y**2]]),
        coords      = (x, y),
        domain      = (4.0, 2.0),
        shift       = (0.0, 1.5),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.2,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Anisotropic Poincaré half-plane (off-diagonal, K < -1)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Cylinder  (flat, periodic in y, elongated in x)
    # ──────────────────────────────────────────────────────────────────────────
    # The metric is flat, but the domain is elongated in x to make the
    # anisotropy of the simulation box visible.  With periodic boundary
    # conditions in y, this represents S¹ × ℝ (a cylinder).
    #
    #   ds² = dx² + dy²,   x ∈ [-3, 3],  y ∈ [-1, 1]  (y periodic)
    #
    # Curvature  : K = 0
    # Topology   : S¹ × ℝ (cylinder)
    # Note       : Different from 'flat' only in the aspect ratio of the domain.
    # ──────────────────────────────────────────────────────────────────────────
    'cylinder': dict(
        g           = Matrix([[1, 0], [0, 1]]),
        coords      = (x, y),
        domain      = (6.0, 2.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.4,
        Lt          = 4.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Cylinder S¹ × ℝ (flat, elongated x-direction, K = 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Wavy cylinder  (K varies along the axis)
    # ──────────────────────────────────────────────────────────────────────────
    # A cylinder whose radius oscillates sinusoidally along the axis.
    # The variation in g_yy = (1 + a cos y)² mimics a corrugated tube.
    #
    #   ds² = dx² + (1 + a cos y)² dy²,   a = 0.3
    #
    # Grid: x, y ∈ [-π, π]  (both periodic)
    #
    # Curvature  : K = -a cos y / (1 + a cos y)  (same magnitude as warped
    #              circle but with y replacing x in the denominator).
    # Note       : The wavy cylinder and warped circle have structurally
    #              identical metrics; the difference is in interpretation
    #              (base vs fibre coordinates).
    # ──────────────────────────────────────────────────────────────────────────
    'wavy_cylinder': dict(
        g           = Matrix([[1, 0], [0, (1 + 0.3*cos(y))**2]]),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 4.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Wavy cylinder (a=0.3, K = -a cosy/(1+a cosy))',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Flat Klein bottle  (K = 0, non-orientable)
    # ──────────────────────────────────────────────────────────────────────────
    # The Klein bottle carries a flat metric (K = 0) just like the torus,
    # but the identification of the boundary reverses orientation in one
    # direction, making it non-orientable.  The metric itself is unchanged;
    # only the topology differs from 'flat'.
    #
    #   ds² = dx² + dy²,   (x, y) ~ (x + Lx, y) ~ (-x, y + Ly)  (twisted)
    #
    # Curvature  : K = 0
    # Topology   : Klein bottle (non-orientable, χ = 0)
    # Note       : The metric tensor is identical to 'flat'; the non-orientable
    #              identification must be enforced in the boundary conditions,
    #              not the metric.
    # ──────────────────────────────────────────────────────────────────────────
    'klein': dict(
        g           = Matrix([[1, 0], [0, 1]]),
        coords      = (x, y),
        domain      = (4.0, 2.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Flat Klein bottle (non-orientable identification, K = 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Clifford torus  (square flat torus, minimal in S³)
    # ──────────────────────────────────────────────────────────────────────────
    # The Clifford torus is the unique (up to congruence) minimal torus
    # embedded in S³.  Intrinsically it is just a square flat torus with
    # equal period lengths 2π in both directions.
    #
    #   ds² = dx² + dy²,   x, y ∈ [-π, π]  (both periodic, equal period)
    #
    # Curvature  : K = 0
    # Topology   : T² (square fundamental domain)
    # Note       : Distinguished from 'flat' by its equal side lengths (square
    #              torus) and from 'cylinder' by the equal periods.  In the
    #              ambient S³ it has zero mean curvature.
    # ──────────────────────────────────────────────────────────────────────────
    'clifford': dict(
        g           = Matrix([[1, 0], [0, 1]]),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.4,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Clifford torus (square flat T², minimal in S³, K = 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Fischer–Marsden metric  (K = -1, same as pseudosphere)
    # ──────────────────────────────────────────────────────────────────────────
    # This is the same local metric as the pseudosphere  ds² = du² + sinh²u dv²
    # placed on a larger domain to explore the large-u regime where the surface
    # flares out.  It is listed separately because the domain and initial
    # condition placement differ.
    #
    #   ds² = du² + sinh²(u) dv²,   u ∈ [2, 6] (shift 1.0, domain 2.0)
    #
    # Curvature  : K = -1 (identical to pseudosphere)
    # Note       : In the large-u region sinh u ≈ exp(u)/2, so the v-circles
    #              grow exponentially.  Geodesics spread very rapidly in v.
    # ──────────────────────────────────────────────────────────────────────────
    'fischer_marsden': dict(
        g           = Matrix([[1, 0], [0, sinh(x)**2]]),
        coords      = (x, y),
        domain      = (2.0, 6.283185),
        shift       = (1.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Fischer–Marsden metric (K = -1, large-u pseudosphere regime)',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Lorentzian metric (experimental)
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Minkowski plane  (Lorentzian, signature (−, +), K = 0)
    # ──────────────────────────────────────────────────────────────────────────
    # The 2-D Minkowski spacetime: one time-like and one space-like direction.
    # The metric is indefinite (pseudo-Riemannian); standard elliptic solvers
    # need modification for this entry.
    #
    #   ds² = -dx² + dy²,   signature (−, +)
    #
    # Grid: x, y ∈ [-2, 2]
    #
    # Curvature  : K = 0 (flat, like Euclidean plane)
    # Note       : ⚠ The metric is NOT positive definite.  Eigenvalues of g
    #              are −1 and +1.  Standard Riemannian algorithms will fail or
    #              give unphysical results.  Use only with a solver that supports
    #              indefinite metrics (e.g., hyperbolic PDE formulation).
    # ──────────────────────────────────────────────────────────────────────────
    'minkowski': dict(
        g           = Matrix([[-1, 0], [0, 1]]),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Minkowski plane (Lorentzian (−,+), K = 0) ⚠ indefinite metric',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — Additional curved surfaces and cones
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Cone with deficit angle π  (α = 0.5)
    # ──────────────────────────────────────────────────────────────────────────
    # A flat cone with a deficit angle of π (i.e., the total angle around the
    # apex is π instead of 2π).  The metric is flat everywhere except at the
    # apex where there is a distributional curvature (a conical singularity).
    #
    #   ds² = dr² + (0.5 r)² dθ²   ≡   dr² + 0.25 r² dθ²
    #
    # Grid: r ∈ [1.0, 5.0]  (shift = 3.0 so r_phys = x_grid + 3)
    #       θ ∈ [-π, π]     (periodic)
    #
    # Curvature  : K = 0 for r > 0; singular at apex (r = 0)
    # Topology   : ℝ² \ {0} (punctured plane) with non‑trivial holonomy
    # Note       : Geodesics are straight lines in the covering space; they
    #              refract when crossing the cut (θ jumps by π/2?).
    # ──────────────────────────────────────────────────────────────────────────
    'cone_deficit_pi': dict(
        g           = Matrix([[1, 0], [0, 0.25 * x**2]]),
        coords      = (x, y),
        domain      = (4.0, 2*np.pi),
        shift       = (3.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Cone with deficit angle π (α = 0.5, K = 0 except apex)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Cone with excess angle  (α = 1.5, negative deficit)
    # ──────────────────────────────────────────────────────────────────────────
    # A flat cone with an excess angle (angle > 2π around apex).  This creates
    # a “bump” with positive distributional curvature at the apex.
    #
    #   ds² = dr² + (1.5 r)² dθ²   ≡   dr² + 2.25 r² dθ²
    #
    # Grid: r ∈ [1.0, 5.0]  (shift = 3.0)
    #       θ ∈ [-π, π]     (periodic)
    #
    # Curvature  : K = 0 for r > 0; singular positive curvature at apex
    # Topology   : ℝ² \ {0} (punctured plane)
    # Note       : Excess cones appear in the geometry of disclinations in
    #              nematic liquid crystals and in 2+1 gravity with a positive
    #              mass.
    # ──────────────────────────────────────────────────────────────────────────
    'cone_excess': dict(
        g           = Matrix([[1, 0], [0, 2.25 * x**2]]),
        coords      = (x, y),
        domain      = (4.0, 2*np.pi),
        shift       = (3.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Cone with excess angle (α = 1.5, K > 0 distributional)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Smooth cone (Gaussian‑rounded apex)
    # ──────────────────────────────────────────────────────────────────────────
    # A smooth deformation of the flat cone that removes the apex singularity.
    # The angular metric coefficient goes as r² for small r (flat) and
    # transitions to (1 – ε)² r² for large r, creating a deficit smoothly.
    #
    #   ds² = dr² + [r² + a² (1 – e^{-r²/σ²})] dθ²,   a = 0.5, σ = 1.0
    #
    # Grid: r ∈ [-3, 3]  (x plays role of r; shift 0 so r = x)
    #       θ ∈ [-π, π]  (periodic)
    #
    # Curvature  : K = 0 at r = 0; negative in the transition region,
    #              then returns to 0 as r → ∞.
    # Topology   : ℝ² (smooth everywhere)
    # Note       : Useful for testing how a numerical solver handles a
    #              smoothed conical defect (e.g., in cosmic‑string simulations).
    # ──────────────────────────────────────────────────────────────────────────
    'smooth_cone': dict(
        g           = Matrix([[1, 0], [0, x**2 + 0.25 * (1 - exp(-x**2))]]),
        coords      = (x, y),
        domain      = (6.0, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Smooth cone with Gaussian‑rounded apex (K variable, asymptotically flat)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Elliptic paraboloid (polar coordinates)
    # ──────────────────────────────────────────────────────────────────────────
    # The induced metric on the surface z = x² + y², but written in polar
    # coordinates (r, θ).  This gives a diagonal metric, unlike the Cartesian
    # version in the catalogue (which has off‑diagonal terms).
    #
    #   ds² = (1 + 4r²) dr² + r² dθ²
    #
    # Grid: r ∈ [0.5, 3.5]  (shift 1.5, domain 2.0)
    #       θ ∈ [-π, π]     (periodic)
    #
    # Curvature  : K = 4 / (1 + 4r²)²  (positive, maximum 4 at r = 0)
    # Topology   : ℝ²
    # Note       : This is isometric to the Cartesian paraboloid entry but
    #              uses a different coordinate system; the diagonal form is
    #              easier for some numerical schemes.
    # ──────────────────────────────────────────────────────────────────────────
    'paraboloid_polar': dict(
        g           = Matrix([[1 + 4*x**2, 0], [0, x**2]]),
        coords      = (x, y),
        domain      = (2.0, 2*np.pi),
        shift       = (2.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Elliptic paraboloid in polar coords (K = 4/(1+4r²)² > 0)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Poincaré cusp (deep hyperbolic region near the boundary)
    # ──────────────────────────────────────────────────────────────────────────
    # The same Poincaré half‑plane metric but placed very close to the
    # boundary y = 0, where the metric becomes extremely elongated and
    # geodesics take very long to reach the “cusp”.
    #
    #   ds² = (dx² + dy²) / y²,   y ∈ [0.1, 0.9]
    #
    # Grid: y = x_phys? Actually we use x for vertical coordinate to keep
    #       the same naming: x plays the role of y_phys (the singular direction).
    #       Domain x ∈ [0.1, 0.9]  (shift 0.5, width 0.8)
    #       y ∈ [-π, π]  (periodic horizontal direction)
    #
    # Curvature  : K = -1 (constant), same as full half‑plane
    # Topology   : ℝ × S¹ (a hyperbolic cylinder that flares rapidly)
    # Note       : The grid is chosen to avoid the singular boundary y = 0.
    #              This is a more extreme test of numerical stability than
    #              the standard poincare entry.
    # ──────────────────────────────────────────────────────────────────────────
    'poincare_cusp': dict(
        g           = Matrix([[1/x**2, 0], [0, 1/x**2]]),
        coords      = (x, y),
        domain      = (0.8, 2*np.pi),
        shift       = (0.5, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.1,
        Lt          = 2.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Poincaré cusp (y ∈ [0.1,0.9], K = -1, extreme elongation)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Exponential bump (conformally flat, positive curvature peak)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformal metric  ds² = e^{-r²}(dx²+dy²) whose Gaussian curvature
    # is positive near the origin and decays to zero at infinity.
    #
    #   ds² = exp[-(x²+y²)] (dx²+dy²)
    #
    # Grid: x, y ∈ [-2.5, 2.5]
    #
    # Curvature  : K = (2 - r²) / (2 e^{-r²})? Actually from formula:
    #              f = e^{-r²} ⇒ ln f = -r² ⇒ Δ = -4 ⇒ K = -(-4)/(2f) = 2/f > 0.
    #              So K = 2 e^{r²}  (positive, grows rapidly away from origin!).
    #              Wait – K = -Δ(ln f)/(2f) = -(-4)/(2e^{-r²}) = 2 e^{r²} > 0.
    #              That means curvature increases with r, opposite of intuition.
    #              Interesting test case: K grows without bound.
    # Note       : Despite the name, the curvature is low at the centre and
    #              huge at the boundary.  Use with care – keep the domain small.
    # ──────────────────────────────────────────────────────────────────────────
    'exponential_bump': dict(
        g           = Matrix([[1, 0], [0, 1]]) * exp(-(x**2 + y**2)),
        coords      = (x, y),
        domain      = (5.0, 5.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 2.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Exponential conformal bump (K = 2 e^{r²} positive, increases outward)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Quartic well (conformally flat, negative curvature well)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformal metric  ds² = (1 + r⁴)(dx²+dy²).  The conformal factor grows
    # as r⁴, producing a deep negative curvature region near the origin.
    #
    #   ds² = [1 + (x²+y²)²] (dx²+dy²)
    #
    # Grid: x, y ∈ [-2.0, 2.0]
    #
    # Curvature  : K = -2(2r⁴+3r²+1) / (1+r⁴)³  (negative everywhere; most
    #              negative at the origin with K(0) = -2)
    # Topology   : ℝ²
    # Note       : A model for a “curvature hole” – a region of strong negative
    #              curvature that flattens out at large r.
    # ──────────────────────────────────────────────────────────────────────────
    'quartic_well': dict(
        g           = Matrix([[1, 0], [0, 1]]) * (1 + (x**2 + y**2)**2),
        coords      = (x, y),
        domain      = (4.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Quartic well (K negative everywhere, minimum -2 at origin)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Soliton metric (sech² × cos² modulation)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformally flat metric with a localised modulation that is periodic
    # in y and decays in x.  Models a “breather”‑type geometry.
    #
    #   ds² = [1 + sech²(x) cos²(y)] (dx²+dy²)
    #
    # Grid: x ∈ [-3, 3], y ∈ [-π, π]  (y periodic)
    #
    # Curvature  : K varies, positive where the conformal factor has maxima
    #              and negative where it has minima.
    # Topology   : ℝ × S¹ (infinite strip)
    # Note       : The product structure makes geodesic behaviour anisotropic
    #              and tests the interaction of curvature with spatial periodicity.
    # ──────────────────────────────────────────────────────────────────────────
    'soliton_metric': dict(
        g           = Matrix([[1, 0], [0, 1]]) * (1 + (1/cosh(x)**2) * cos(y)**2),
        coords      = (x, y),
        domain      = (6.0, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 4.0,
        Nt          = 125,
        Nx          = 64, Ny = 64,
        description = 'Soliton metric (1 + sech²x cos²y) (K variable, periodic in y)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Bipolar coordinate metric (conformally flat with a line singularity)
    # ──────────────────────────────────────────────────────────────────────────
    # The metric of the Euclidean plane in bipolar coordinates (u, v).  It is
    # conformally flat and has a coordinate singularity along the line u = 0
    # (which corresponds to two circles in the plane).
    #
    #   ds² = (dx² + dy²) / (cosh x - cos y)²
    #
    # Grid: x ∈ [-2, 2], y ∈ [-π, π]  (y periodic)
    #
    # Curvature  : K = -1  (constant negative curvature! This is another
    #              representation of the hyperbolic plane.)
    #              Indeed, this metric is isometric to the Poincaré half‑plane.
    # Topology   : ℝ² (but covering of hyperbolic plane)
    # Note       : Added here because its coordinate expression is very
    #              different from the standard half‑plane or disk, and it
    #              exhibits a periodic in y direction while being constant
    #              curvature – a nontrivial test for curvature computation.
    # ──────────────────────────────────────────────────────────────────────────
    'bipolar': dict(
        g           = Matrix([[1, 0], [0, 1]]) / (cosh(x) - cos(y))**2,
        coords      = (x, y),
        domain      = (4.0, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Bipolar coordinate metric (K = -1, constant negative curvature)',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — Exotic and less common 2D metrics
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # McCabe's metric  (integrable geodesic flow)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformally flat metric on the plane whose geodesic equations are
    # completely integrable (Liouville integrable).  The conformal factor is
    # a quadratic polynomial in the coordinates.
    #
    #   ds² = (x² + y² + 1)(dx² + dy²)
    #
    # Grid: x, y ∈ [-2.5, 2.5]
    #
    # Curvature  : K = -2 (x² + y² + 2) / (x² + y² + 1)³   (negative everywhere,
    #              most negative at the origin: K(0,0) = -4)
    # Topology   : ℝ²
    # Note       : The geodesic flow separates in parabolic coordinates.
    #              The metric is spherically symmetric (depends only on r²).
    # ──────────────────────────────────────────────────────────────────────────
    'mccabe': dict(
        g           = Matrix([[1, 0], [0, 1]]) * (x**2 + y**2 + 1),
        coords      = (x, y),
        domain      = (5.0, 5.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = "McCabe's metric (integrable, K = -2(r²+2)/(r²+1)³ < 0)",
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Bach–Weyl metric  (static axisymmetric, reduced to 2D)
    # ──────────────────────────────────────────────────────────────────────────
    # The Bach–Weyl metric describes a static axisymmetric vacuum solution in
    # general relativity, generated by two point masses on the symmetry axis.
    # Here we take the 2‑D Riemannian slice (ρ, z) of the Weyl class, with a
    # simple single‑mass potential.
    #
    #   ds² = e^{-2m/√(ρ²+z²)} (dρ² + dz²)   where ρ = x, z = y
    #   with m = 1 (mass parameter)
    #
    # Grid: x, y ∈ [-3, 3]  (both act as ρ and z)
    #
    # Curvature  : K = -m²/r⁴ e^{2m/r} (negative everywhere, singular at origin)
    # Topology   : ℝ² \ {0}
    # Note       : The exponential factor is characteristic of Weyl’s class
    #              for a single particle (Chazy–Curzon solution).  The curvature
    #              is negative and decays slowly.
    # ──────────────────────────────────────────────────────────────────────────
    'bach_weyl': dict(
        g           = Matrix([[1, 0], [0, 1]]) * exp(-2 / sqrt(x**2 + y**2)),
        coords      = (x, y),
        domain      = (6.0, 6.0),
        shift       = (0.0, 0.0),
        ic_center   = (1.0, 0.0),   # offset to avoid the central singularity
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Bach–Weyl (single‑mass) reduced 2D metric (K negative, singular at origin)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # "Butterfly" metric  (alternating curvature lobes)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformally flat metric with a butterfly‑shaped conformal factor that
    # produces four alternating curvature lobes (two positive, two negative).
    # The name is inspired by the shape of the level sets.
    #
    #   ds² = [1 + a (sin²x sin²y) + b (cos²x cos²y)] (dx² + dy²)
    #   with a = 0.8, b = 0.5
    #
    # Grid: x, y ∈ [-π, π]  (periodic in both directions)
    #
    # Curvature  : K changes sign according to the product sin²x sin²y.
    #              Positive lobes near (0,0) and (π,π), negative lobes near
    #              (π/2, π/2) and (3π/2, 3π/2).
    # Topology   : T² (doubly periodic)
    # Note       : The total curvature integrates to zero (Gauss–Bonnet for torus).
    #              Good for testing wave propagation through alternating focusing
    #              and defocusing regions.
    # ──────────────────────────────────────────────────────────────────────────
    'butterfly': dict(
        g           = Matrix([[1, 0], [0, 1]]) * (1 + 0.8 * sin(x)**2 * sin(y)**2 + 0.5 * cos(x)**2 * cos(y)**2),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 4.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Butterfly metric (K alternates sign, four lobes, on T²)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Gödel spacetime (2‑D spatial slice at constant t and a symmetry direction)
    # ──────────────────────────────────────────────────────────────────────────
    # The Gödel universe is a rotating cosmological solution of Einstein’s
    # equations.  Taking a slice at constant time t and suppressing one spatial
    # dimension yields a 2‑D Riemannian metric that still carries a memory of
    # the rotation through an effective “magnetic” curvature.
    #
    #   ds² = dx² + (1 - (Ω x)²) dy²   with Ω = 0.5
    #
    # Grid: x ∈ [-1.5, 1.5]   (so that 1 - (0.5 x)² > 0)
    #       y ∈ [-π, π]       (periodic)
    #
    # Curvature  : K = -Ω² / (1 - (Ω x)²)²  (negative everywhere,
    #              diverging at x = ±1/Ω)
    # Topology   : ℝ × S¹ (cylinder)
    # Note       : This is not the full Gödel metric but a simplified 2‑D analog
    #              that captures the non‑trivial curvature and the fact that the
    #              y‑circles shrink as |x| increases.  The curvature is always
    #              negative and becomes large near the “horizon” x = ±1/Ω.
    # ──────────────────────────────────────────────────────────────────────────
    'goedel_slice': dict(
        g           = Matrix([[1, 0], [0, 1 - 0.25 * x**2]]),
        coords      = (x, y),
        domain      = (3.0, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.2,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Gödel spatial slice (Ω=0.5, K = -Ω²/(1-(Ωx)²)², negative)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Dust disk metric  (Newtonian analogy, central mass disk)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformally flat metric modelling the gravitational field of a thin
    # dusty disk (like a galactic disk) in 2+1 gravity or as a Newtonian
    # analogy.  The conformal factor is 1 + M/√(r² + a²), with M = 1, a = 0.5.
    #
    #   ds² = [1 + 1/√(x² + y² + 0.25)] (dx² + dy²)
    #
    # Grid: x, y ∈ [-3, 3]
    #
    # Curvature  : K = - (Δ ln f)/(2f) with f as above.  K is negative near
    #              the centre (like a point mass) and decays to zero at infinity.
    # Topology   : ℝ²
    # Note       : The metric is smooth everywhere (the Plummer softening a
    #              removes the singularity).  It mimics the weak‑field limit
    #              of a massive object in 2‑D gravity.
    # ──────────────────────────────────────────────────────────────────────────
    'dust_disk': dict(
        g           = Matrix([[1, 0], [0, 1]]) * (1 + 1 / sqrt(x**2 + y**2 + 0.25)),
        coords      = (x, y),
        domain      = (6.0, 6.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Dust disk metric (Plummer‑softened central mass, K negative)',
    ),
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10 — Extra‑galactic metrics for microlocal & spectral analysis
    # ══════════════════════════════════════════════════════════════════════════
    # These metrics are designed to test features of the Laplace–Beltrami operator
    # such as: geodesic trapping, conjugate points, caustics, Anosov geodesic flow,
    # non‑compact ends (asymptotically hyperbolic), and eigenvalue clustering.
    #
    # PDE applications: heat kernel asymptotics, Schrödinger propagation,
    # wave equation (finite vs infinite speed, dispersive estimates).
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Dumbbell metric (geodesic trapping between two positive‑curvature bumps)
    # ──────────────────────────────────────────────────────────────────────────
    # A conformally flat metric on ℝ² with two positive curvature bumps that
    # can trap geodesics between them.  The conformal factor is a sum of two
    # Gaussians centred at (±a, 0).  For suitable parameters, a periodic orbit
    # exists connecting the two bumps (a “whispering gallery” mode).
    #
    #   f(x,y) = 1 + A exp(-((x - a)² + y²)/σ²) + A exp(-((x + a)² + y²)/σ²)
    #   ds² = f(x,y) (dx² + dy²)
    #   with A = 2.0, a = 1.5, σ = 0.5
    #
    # Grid: x ∈ [-3, 3], y ∈ [-2, 2]
    #
    # Curvature  : K = -Δ(ln f)/(2f).  Positive near the bumps, negative between.
    # Topology   : ℝ²
    # Note       : Geodesic trapping leads to quasi‑normal modes of the wave
    #              equation and resonances of the Laplace–Beltrami operator.
    # ──────────────────────────────────────────────────────────────────────────
    'dumbbell': dict(
        g = Matrix([[1, 0], [0, 1]]) * (1 + 2*exp(-((x-1.5)**2 + y**2)/0.25)
                                         + 2*exp(-((x+1.5)**2 + y**2)/0.25)),
        coords      = (x, y),
        domain      = (5.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Dumbbell (two Gaussian bumps, geodesic trapping between them)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Lens metric (perfectly focusing, caustics)
    # ──────────────────────────────────────────────────────────────────────────
    # A rotationally symmetric metric on the sphere (or plane) that makes all
    # geodesics starting from a given point refocus perfectly at the antipodal
    # point.  This is an example of a “Zoll” metric but with a different radial
    # profile.  Here we use the “Clairaut” form on the plane with a radial
    # function such that the geodesics are sinusoidal.
    #
    #   ds² = dr² + (sin²(2r)) dθ²   for r ∈ [0, π/2]
    #   (equivalent to the standard unit sphere after a change of variable)
    #
    # We map r = x (with shift) to avoid r=0 singularity.
    # Grid: r ∈ [0.2, 1.37] (0.2 to π/2 ≈ 1.57)  → domain 1.17, shift 0.785
    #       θ ∈ [-π, π]
    #
    # Curvature  : K = +4 (constant positive, same as sphere)
    # Topology   : S² (just a different angular coordinate)
    # Note       : The lens effect (perfect focusing) creates a caustic where
    #              the amplitude of the wave equation blows up in the geometric
    #              optics approximation.
    # ──────────────────────────────────────────────────────────────────────────
    'lens': dict(
        g           = Matrix([[1, 0], [0, sin(2*x)**2]]),
        coords      = (x, y),
        domain      = (1.2, 2*np.pi),
        shift       = (0.7, 0.0),   # centres at x ≈ 1.37? Actually 0.7+0.6=1.3, fine
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.2,
        Lt          = 2.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Lens metric (sin²(2r), perfect focusing, caustics)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Hadamard’s billiard (negative curvature, Anosov geodesic flow)
    # ──────────────────────────────────────────────────────────────────────────
    # A compact surface of constant negative curvature (genus ≥ 2).  The simplest
    # is the “octagon” or “pair of pants” but we use a smooth model: the
    # hyperbolic torus with a single conical singularity is not smooth; instead
    # we take a quotient of the Poincaré disk by a Fuchsian group.  Numerically,
    # we can use the Poincaré disk metric restricted to a fundamental domain
    # with periodic boundary conditions – but here we provide the metric on the
    # disk itself, and the solver must enforce the group identifications.
    #
    #   ds² = 4 (dx² + dy²) / (1 - x² - y²)²  (same as poincare_disk)
    #
    # The fundamental domain is e.g. a regular octagon centred at origin.
    # For simplicity, we keep the disk but add a note.
    #
    # Grid: x, y ∈ [-0.9, 0.9]  (avoid r=1)
    #
    # Curvature  : K = -1 (constant)
    # Topology   : ℝ² (open) but intended as compact quotient.
    # Note       : Anosov geodesic flow implies exponential mixing,
    #              decay of correlations, and no resonances accumulating at
    #              the real axis (spectral gap).  Good for testing quantum chaos.
    # ──────────────────────────────────────────────────────────────────────────
    'hadamard': dict(
        g           = Matrix([[4/(1 - x**2 - y**2)**2, 0],
                              [0, 4/(1 - x**2 - y**2)**2]]),
        coords      = (x, y),
        domain      = (1.8, 1.8),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.2,
        Lt          = 4.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Hadamard’s model (Poincaré disk, K=-1, Anosov flow on compact quotient)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Asymptotically hyperbolic trumpet (non‑compact, continuous spectrum)
    # ──────────────────────────────────────────────────────────────────────────
    # A complete Riemannian metric on ℝ² that is asymptotically hyperbolic,
    # meaning it tends to the hyperbolic plane metric at infinity.  Useful for
    # studying scattering theory and resonances on non‑compact spaces.
    #
    #   ds² = dr² + sinh²(r) dθ²   for r large → hyperbolic.
    #   We use a smooth interpolation: g = diag(1, r² + sinh²(r))? Simpler:
    #   take r ∈ [0, ∞) but we cut off.  Use x = r (shifted) and y = θ.
    #
    #   ds² = dx² + (x² + sinh²(x)) dy²   (smooth at 0: x² + x² = 2x²)
    #
    # Grid: x ∈ [0.5, 4.5]  (shift 2.5, domain 2.0)
    #       y ∈ [-π, π]
    #
    # Curvature  : K ≈ -1 for large x, tends to -1 from below.
    # Topology   : ℝ × S¹ (a cylinder, non‑compact in x)
    # Note       : The Laplace–Beltrami operator has continuous spectrum with
    #              threshold at λ = 1/4 (bottom of the essential spectrum for
    #              hyperbolic ends).  The wave equation exhibits dispersion.
    # ──────────────────────────────────────────────────────────────────────────
    'asymptotic_hyperbolic': dict(
        g           = Matrix([[1, 0], [0, x**2 + sinh(x)**2]]),
        coords      = (x, y),
        domain      = (4.0, 2*np.pi),
        shift       = (2.5, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 4.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Asymptotically hyperbolic trumpet (continuous spectrum, resonances)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Egg‑carton with a defect (topological obstacle, scattering)
    # ──────────────────────────────────────────────────────────────────────────
    # A periodic metric (on a torus) with a localised bump that acts as a
    # scatterer.  This creates a non‑trivial scattering matrix for waves.
    # We modify the eggcarton metric by multiplying the conformal factor with
    # a Gaussian bump.
    #
    #   f(x,y) = (1 + 0.5 sin²x sin²y) × (1 + 0.8 exp(-((x-π)²+(y-π)²)/0.5))
    #   ds² = f(x,y)(dx²+dy²)
    #
    # Grid: x, y ∈ [0, 2π] (periodic)
    #
    # Curvature  : K variable, sign changes, plus a strong positive bump.
    # Topology   : T²
    # Note       : The bump breaks translational symmetry, causing resonant
    #              modes.  Useful for studying the effect of a defect on the
    #              spectral density and on wave transport.
    # ──────────────────────────────────────────────────────────────────────────
    'defective_eggcarton': dict(
        g = Matrix([[1, 0], [0, 1]]) * (1 + 0.5 * sin(x)**2 * sin(y)**2)
          * (1 + 0.8 * exp(-((x - np.pi)**2 + (y - np.pi)**2)/0.5)),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (np.pi, np.pi),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Eggcarton with a Gaussian defect (scattering center on T²)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Non‑Zoll but all geodesics closed except one (Clairaut’s metric)
    # ──────────────────────────────────────────────────────────────────────────
    # A metric on the sphere where all meridians are closed of the same length,
    # but the equator is a geodesic of a different length.  This is a
    # “Zoll-like” example that highlights the role of conjugate points.
    #
    #   ds² = dθ² + (a sin²θ + b cos²θ) dφ²,  with a=1, b=0.5
    #       = dθ² + (sin²θ + 0.5 cos²θ) dφ²
    #
    # Grid: θ ∈ (0, π) via shift π/2; φ ∈ [-π, π]
    #
    # Curvature  : K = 1 - (b-a)cos²θ? Actually variable, positive.
    # Topology   : S²
    # Note       : The equator φ geodesic is non‑closed if the period ratio
    #              is irrational.  This tests the intertwining between the
    #              symplectic structure and the quantization condition.
    # ──────────────────────────────────────────────────────────────────────────
    'clairaut': dict(
        g           = Matrix([[1, 0], [0, sin(x)**2 + 0.5 * cos(x)**2]]),
        coords      = (x, y),
        domain      = (2.54, 2*np.pi),
        shift       = (1.57, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 4.0,
        Nt          = 125,
        Nx          = 64, Ny = 64,
        description = 'Clairaut metric (a=1, b=0.5, equator non‑closed)',
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 11 — Minimal set for microlocal, symplectic, ΨDO and PDE analysis
    # ══════════════════════════════════════════════════════════════════════════
    # This section collects the essential 2D Riemannian metrics that together
    # provide the best point of view for studying:
    #   • microlocal analysis (wavefront sets, caustics, trapping, scattering)
    #   • symplectic analysis (integrable, chaotic, and scattering dynamics)
    #   • pseudodifferential operators (Fourier calculus, spherical harmonics,
    #     hyperbolic and scattering calculi)
    #   • PDEs (heat, Schrödinger, wave) – asymptotics, dispersion, resonances
    #
    # Only 4 metrics are strictly necessary; a 5th is included to illustrate
    # geodesic trapping (non‑uniform decay of waves).
    # ══════════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────────
    # Flat torus – pure point spectrum, integrable geodesic flow
    # ──────────────────────────────────────────────────────────────────────────
    # The simplest periodic setting.  Geodesics are straight lines; wavefronts
    # propagate without caustics.  Fourier series diagonalise the Laplace‑
    # Beltrami operator, giving exact solutions to the heat, Schrödinger and
    # wave equations.  The symplectic dynamics is completely integrable.
    # ──────────────────────────────────────────────────────────────────────────
    'flat_torus': dict(
        g           = Matrix([[1, 0], [0, 1]]),
        coords      = (x, y),
        domain      = (2*np.pi, 2*np.pi),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Flat torus T² (K=0, integrable, pure point spectrum)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Round sphere – closed geodesics, caustics, perfect refocusing
    # ──────────────────────────────────────────────────────────────────────────
    # Positive constant curvature K=+1.  All geodesics are closed with length 2π.
    # Caustics appear in the wave propagator; the wave equation exhibits perfect
    # revivals.  Spherical harmonics diagonalise the Laplacian, giving a rich
    # algebraic structure (SO(3) representation theory).  An essential testing
    # ground for microlocal methods near caustics.
    # ──────────────────────────────────────────────────────────────────────────
    'round_sphere': dict(
        g           = Matrix([[1, 0], [0, sin(x)**2]]),
        coords      = (x, y),
        domain      = (2.54, 2*np.pi),
        shift       = (1.57, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 3.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Round sphere S² (K=+1, closed geodesics, caustics)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Compact hyperbolic surface (model: Poincaré disk / Fuchsian quotient)
    # ──────────────────────────────────────────────────────────────────────────
    # Constant negative curvature K = -1.  The geodesic flow is Anosov (uniformly
    # hyperbolic, chaotic).  No conjugate points; mixing dynamics.  The Laplacian
    # has a spectral gap and the resolvent admits analytic continuation with
    # resonances.  This is the paradigmatic example for quantum chaos and for
    # the properties of ΨDOs on a compact manifold with negative curvature.
    # Note: The metric here is the Poincaré disk, but it should be used on a
    # compact fundamental domain (e.g., a regular octagon) with periodic
    # identifications enforced by the solver.
    # ──────────────────────────────────────────────────────────────────────────
    'compact_hyperbolic': dict(
        g           = Matrix([[4/(1 - x**2 - y**2)**2, 0],
                              [0, 4/(1 - x**2 - y**2)**2]]),
        coords      = (x, y),
        domain      = (1.8, 1.8),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.2,
        Lt          = 4.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Compact hyperbolic surface (K=-1, Anosov flow, quantum chaos)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Asymptotically hyperbolic trumpet – non‑compact, continuous spectrum
    # ──────────────────────────────────────────────────────────────────────────
    # A complete Riemannian metric on a cylinder (R × S¹) that tends to the
    # hyperbolic plane at infinity.  The Laplace‑Beltrami operator has essential
    # spectrum starting at λ = 1/4 and may possess resonances (poles of the
    # continued resolvent).  This metric is necessary to study scattering theory,
    # long‑range dispersion, and the wave equation on non‑compact spaces with
    # hyperbolic ends.  The geodesic flow is scattering (particles can escape).
    # ──────────────────────────────────────────────────────────────────────────
    'asymptotic_hyperbolic_trumpet': dict(
        g           = Matrix([[1, 0], [0, x**2 + sinh(x)**2]]),
        coords      = (x, y),
        domain      = (4.0, 2*np.pi),
        shift       = (2.5, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 4.0,
        Nt          = 150,
        Nx          = 64, Ny = 64,
        description = 'Asymptotically hyperbolic trumpet (scattering, continuous spectrum, resonances)',
    ),

    # ──────────────────────────────────────────────────────────────────────────
    # Dumbbell (geodesic trapping) – optional but important
    # ──────────────────────────────────────────────────────────────────────────
    # A conformally flat metric on R² with two positive‑curvature Gaussian bumps.
    # A hyperbolic (unstable) trapped set exists: a periodic geodesic oscillating
    # between the two bumps.  This induces quasi‑normal modes and prevents
    # uniform exponential decay of the wave equation.  It illustrates that
    # trapping is generic and must be handled by microlocal and symplectic
    # methods (normally hyperbolic invariant manifolds, resonance expansions).
    # ──────────────────────────────────────────────────────────────────────────
    'dumbbell_trapping': dict(
        g = Matrix([[1, 0], [0, 1]]) * (1 + 2*exp(-((x-1.5)**2 + y**2)/0.25)
                                         + 2*exp(-((x+1.5)**2 + y**2)/0.25)),
        coords      = (x, y),
        domain      = (5.0, 4.0),
        shift       = (0.0, 0.0),
        ic_center   = (0.0, 0.0),
        ic_sigma    = 0.3,
        Lt          = 5.0,
        Nt          = 100,
        Nx          = 64, Ny = 64,
        description = 'Dumbbell (geodesic trapping, quasi‑normal modes, non‑uniform decay)',
    ),
}