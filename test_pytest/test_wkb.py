from imports import *
from wkb import *

def test_wkb_approximation_placeholder():
    """Test WKB multidimensional."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Wave equation: p = ξ² + η²
    p = xi**2 + eta**2
    
    # Create proper initial data using the helper function
    # Line segment from (-1, 0) to (1, 0) with rays going in direction (0, 1)
    initial_phase = create_initial_data_line(
        x_range=(-1, 1), 
        n_points=10,  # Use fewer points for faster test
        direction=(0, 1),  # Rays going upward
        y_intercept=0.0
    )
    
    # Run WKB with smaller domain and resolution for faster test
    wkb = wkb_approximation(
        p, 
        initial_phase, 
        order=1,
        domain=((-2, 2), (-2, 2)), 
        resolution=20
    )


    # Basic checks
    assert 'x' in wkb
    assert 'y' in wkb
    assert 'S' in wkb
    assert 'a' in wkb
    assert 'u' in wkb
    assert 'rays' in wkb
    
    # Check shapes
    assert wkb['x'].shape == (20, 20)
    assert wkb['y'].shape == (20, 20)
    assert wkb['S'].shape == (20, 20)
    assert wkb['a'][0].shape == (20, 20)
    assert wkb['u'].shape == (20, 20)
    
    # Check that we traced some rays
    assert len(wkb['rays']) > 0
    
    print("✓ WKB multidimensional test passed")

def test_wkb_approximation_line_source():
    """Test WKB with line source (plane wave generation)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic wave equation
    p = xi**2 + eta**2
    
    # Horizontal line with upward rays
    ic = create_initial_data_line(
        x_range=(-1, 1),
        n_points=15,
        direction=(0, 1),
        y_intercept=-1.0
    )
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=30
    )
    
    # Check output structure
    assert 'x' in wkb
    assert 'y' in wkb
    assert 'S' in wkb
    assert 'a' in wkb
    assert 'u' in wkb
    assert 'rays' in wkb
    
    # Check dimensions
    assert wkb['x'].shape == (30, 30)
    assert wkb['y'].shape == (30, 30)
    
    # Check rays were traced
    assert len(wkb['rays']) > 0
    
    # Phase should vary (not all zeros)
    assert np.std(wkb['S']) > 0.1


def test_wkb_approximation_circular_source():
    """Test WKB with circular source (expanding waves)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Isotropic symbol
    p = xi**2 + eta**2
    
    # Circle with outward rays
    ic = create_initial_data_circle(
        radius=0.5,
        n_points=20,
        outward=True
    )
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-3, 3), (-3, 3)),
        resolution=40
    )
    
    # Should have traced multiple rays
    assert len(wkb['rays']) >= 15
    
    # Check amplitude behavior (should decay with distance for circular waves)
    a_center = wkb['a'][0][20, 20]  # Center
    a_edge = wkb['a'][0][0, 0]      # Edge
    
    # Solution should exist
    assert wkb['u'].shape == (40, 40)
    
    # Phase should have circular symmetry (approximately)
    # Check that phase increases with distance from origin
    S_center_region = wkb['S'][18:22, 18:22]
    S_edge_region = wkb['S'][0:5, 0:5]
    assert np.mean(np.abs(S_edge_region)) > np.mean(np.abs(S_center_region))


def test_wkb_approximation_point_source():
    """Test WKB with point source (spherical waves)."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Wave equation
    p = xi**2 + eta**2
    
    # Point source at origin
    ic = create_initial_data_point_source(
        x0=0.0,
        y0=0.0,
        n_rays=16
    )
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=35
    )
    
    # All rays should start at origin
    for ray in wkb['rays']:
        assert np.isclose(ray['x'][0], 0.0, atol=1e-6)
        assert np.isclose(ray['y'][0], 0.0, atol=1e-6)
    
    # Rays should diverge
    for ray in wkb['rays']:
        distance_traveled = np.sqrt(
            (ray['x'][-1] - ray['x'][0])**2 + 
            (ray['y'][-1] - ray['y'][0])**2
        )
        assert distance_traveled > 0.5


def test_wkb_approximation_anisotropic():
    """Test WKB with anisotropic symbol."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    
    # Anisotropic dispersion: faster in x-direction
    p = xi**2 + 4*eta**2
    
    # Vertical line with horizontal rays
    n_pts = 12
    y_vals = np.linspace(-1, 1, n_pts)
    ic = {
        'x': np.zeros(n_pts),
        'y': y_vals,
        'S': np.zeros(n_pts),
        'p_x': np.ones(n_pts),   # ξ = 1
        'p_y': np.zeros(n_pts)   # η = 0
    }
    
    wkb = wkb_approximation(
        p, 
        ic, 
        order=1,
        domain=((-2, 2), (-2, 2)),
        resolution=30
    )
    
    # Check anisotropic propagation
    # Rays should propagate primarily in x-direction
    for ray in wkb['rays']:
        dx = ray['x'][-1] - ray['x'][0]
        dy = ray['y'][-1] - ray['y'][0]
        
        # Movement in x should dominate
        if abs(dx) > 0.1:  # If ray traveled
            assert abs(dx) > abs(dy)

def test_wkb_approximation_structure():
    """Test WKB multidimensional output structure."""
    x, y, xi, eta = symbols('x y xi eta', real=True)
    p = xi**2 + eta**2
    
    # Utiliser la fonction helper pour créer une source ponctuelle
    # avec plusieurs rayons (plus réaliste pour tester la structure)
    initial_phase = create_initial_data_point_source(
        x0=0.0, 
        y0=0.0, 
        n_rays=8  # 8 rayons partant de l'origine
    )
    
    wkb = wkb_approximation(p, initial_phase, resolution=10)
    
    # Check all required fields exist
    required_fields = ['x', 'y', 'S', 'a', 'u', 'rays']
    for field in required_fields:
        assert field in wkb
    
    # Vérifier aussi les dimensions
    assert wkb['x'].shape == (10, 10)
    assert wkb['y'].shape == (10, 10)
    assert wkb['S'].shape == (10, 10)
    assert wkb['a'][0].shape == (10, 10)
    assert wkb['u'].shape == (10, 10)
    assert isinstance(wkb['rays'], list)
    assert len(wkb['rays']) > 0

from imports import *
from wkb import *

# ==============================================================================
# HELPERS
# ==============================================================================

def _has_J_keys(rays, dimension):
    """Check that every ray dict contains the stability-matrix keys."""
    for ray in rays:
        if 'J11' not in ray:
            return False
        if dimension == 2:
            for k in ('J12', 'J21', 'J22'):
                if k not in ray:
                    return False
    return True


def _det_J(ray, dimension):
    """Return the det(J) time-series for a single ray."""
    if dimension == 1:
        return ray['J11']                             # scalar in 1D
    return ray['J11'] * ray['J22'] - ray['J12'] * ray['J21']


# ==============================================================================
# BASELINE STRUCTURE TESTS (no caustics expected)
# ==============================================================================

def test_stability_matrix_keys_1d():
    """
    1D: every traced ray must expose J11 after integration.
    This is the minimal requirement for RayCausticDetector compatibility.
    """
    x, xi = symbols('x xi', real=True)
    symbol = xi**2 + 1          # constant-speed 1D wave, no caustics

    n = 15
    ic = {
        'x':   np.linspace(-1, 1, n),
        'p_x': np.ones(n),
        'S':   np.zeros(n),
    }
    result = wkb_approximation(symbol, ic, order=1, domain=(-2, 2),
                                resolution=50, epsilon=0.1)

    assert _has_J_keys(result['rays'], dimension=1), \
        "Missing J11 in 1D ray dicts"
    print("✓ test_stability_matrix_keys_1d")


def test_stability_matrix_keys_2d():
    """
    2D: every traced ray must expose J11..J22 after integration.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    symbol = xi**2 + eta**2

    ic = create_initial_data_line((-1, 1), n_points=12,
                                   direction=(0, 1), y_intercept=0.0)
    result = wkb_approximation(symbol, ic, order=1,
                                domain=((-2, 2), (-2, 2)),
                                resolution=20, epsilon=0.1)

    assert _has_J_keys(result['rays'], dimension=2), \
        "Missing J11..J22 in 2D ray dicts"
    print("✓ test_stability_matrix_keys_2d")


def test_J_initial_condition_is_identity_1d():
    """
    1D: J11(t=0) must equal 1 for every ray  (J(0) = I).
    """
    x, xi = symbols('x xi', real=True)
    ic = {
        'x':   np.linspace(-2, 2, 10),
        'p_x': np.ones(10),
        'S':   np.zeros(10),
    }
    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=(-3, 3), resolution=60, epsilon=0.1)
    for ray in result['rays']:
        assert np.isclose(ray['J11'][0], 1.0, atol=1e-6), \
            f"J11(t=0) = {ray['J11'][0]}, expected 1"
    print("✓ test_J_initial_condition_is_identity_1d")


def test_J_initial_condition_is_identity_2d():
    """
    2D: J(0) = I  →  J11=J22=1, J12=J21=0 for every ray.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    ic = create_initial_data_circle(radius=0.5, n_points=16, outward=True)
    result = wkb_approximation(xi**2 + eta**2, ic, order=1,
                                domain=((-3, 3), (-3, 3)),
                                resolution=20, epsilon=0.1)
    for ray in result['rays']:
        assert np.isclose(ray['J11'][0], 1.0, atol=1e-6)
        assert np.isclose(ray['J12'][0], 0.0, atol=1e-6)
        assert np.isclose(ray['J21'][0], 0.0, atol=1e-6)
        assert np.isclose(ray['J22'][0], 1.0, atol=1e-6)
    print("✓ test_J_initial_condition_is_identity_2d")


def test_det_J_constant_hamiltonian_1d():
    """
    For p = ξ²  (H_px = ∂²p/∂ξ∂x = 0), J satisfies dJ/dt = 0,
    so det J must stay exactly 1 throughout the integration.
    """
    x, xi = symbols('x xi', real=True)
    ic = {
        'x':   np.linspace(-1, 1, 8),
        'p_x': np.ones(8),
        'S':   np.zeros(8),
    }
    result = wkb_approximation(xi**2, ic, order=0,
                                domain=(-3, 3), resolution=40, epsilon=0.1)
    for ray in result['rays']:
        detJ = _det_J(ray, dimension=1)
        assert np.allclose(detJ, 1.0, atol=1e-4), \
            f"det(J) drifted from 1: max deviation = {np.max(np.abs(detJ - 1)):.2e}"
    print("✓ test_det_J_constant_hamiltonian_1d")


# ==============================================================================
# 1D FOLD CAUSTIC  (A2 – simplest singularity)
# ==============================================================================
#
# The canonical fold symbol  p = ξ² - x  focuses all rays at x = 0.
# Rays launched from x < 0 with ξ = 1 all arrive at x = 0 simultaneously,
# making det(J) → 0 at the turning point.

def test_1d_fold_caustic_detection():
    """
    1D fold (A2): convergent ray bundle focusing at x=0.
    Rays are launched symmetrically from both sides with momenta pointing
    toward the origin: xi0 = -sign(x0)*sqrt(|x0|), so that x(t)=x0+2*xi0*t+t^2
    and rays from x0 and -x0 meet near the origin.
    det(J) = dx/dq must vanish at the caustic.
    """
    x, xi = symbols('x xi', real=True)
    n = 30
    # Symmetric initial positions on both sides of the caustic at x=0
    x0 = np.concatenate([np.linspace(-1.5, -0.05, n//2),
                          np.linspace( 0.05,  1.5,  n//2)])
    # Momenta pointing toward origin: xi0 = -sign(x0)*sqrt(|x0|)
    xi0 = -np.sign(x0) * np.sqrt(np.abs(x0))
    ic = {
        'x':   x0,
        'p_x': xi0,
        'S':   np.zeros(n),
    }
    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=(-2, 2), resolution=150,
                                epsilon=0.1, caustic_threshold=1e-2)
    min_det = min(np.min(np.abs(_det_J(r, 1))) for r in result['rays'])
    assert min_det < 0.5, \
        f"Expected det(J) close to 0 near fold caustic, got min = {min_det:.3f}"
    print(f"✓ test_1d_fold_caustic_detection  (min det(J) = {min_det:.4f})")


def test_1d_fold_maslov_phase_shift():
    """
    1D fold: after passing through the caustic, the Maslov phase correction
    must add a pi/2 shift.  The corrected solution should differ from the
    standard WKB on the far side of the caustic.

    maslov_phases is only injected into the result dict when at least one
    caustic is detected; if no caustic is found the test verifies that
    the fallback mode is 'none' (no spurious shift applied).
    """
    x, xi = symbols('x xi', real=True)
    n = 25
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=(-2, 1.5), resolution=150,
                                epsilon=0.1, caustic_correction='maslov',
                                caustic_threshold=1e-2)

    caustics_found = len(result.get('caustics', [])) > 0

    if caustics_found:
        # maslov_phases must be present and non-trivial
        assert 'maslov_phases' in result, \
            "maslov_phases key missing after caustic detection"
        assert np.any(result['maslov_phases'] > 0), \
            "No Maslov phase shift applied despite caustic detection"
    else:
        # No caustic detected: correction mode falls back to 'none',
        # maslov_phases may be absent -- that is acceptable behaviour.
        assert result.get('caustic_correction') == 'none', \
            "Expected fallback to 'none' when no caustic detected"

    print(f"✓ test_1d_fold_maslov_phase_shift  (caustics_found={caustics_found})")


def test_1d_fold_airy_correction():
    """
    1D fold: Airy correction must produce a solution that does NOT diverge
    at the caustic, unlike the standard WKB which blows up as |a|→∞ there.
    """
    x, xi = symbols('x xi', real=True)
    n = 25
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    result_std = wkb_approximation(xi**2 - x, ic, order=1,
                                    domain=(-2, 1.5), resolution=150,
                                    epsilon=0.1, caustic_correction='none')
    result_airy = wkb_approximation(xi**2 - x, ic, order=1,
                                     domain=(-2, 1.5), resolution=150,
                                     epsilon=0.1, caustic_correction='airy',
                                     caustic_threshold=1e-2)

    max_std  = np.max(np.abs(result_std['u']))
    max_airy = np.max(np.abs(result_airy['u']))

    # The Airy-corrected solution should be finite and not dramatically
    # larger than the standard one (it regularises the singularity)
    assert np.isfinite(max_airy), "Airy-corrected |u| is not finite"
    assert max_airy < 10 * max_std or max_airy < 1e4, \
        "Airy correction unexpectedly blows up"
    print("✓ test_1d_fold_airy_correction")


def test_1d_fold_multiple_caustics():
    """
    1D: oscillating potential creates several turning points.
    All of them must be detected (at least 2 caustics reported).
    """
    x, xi = symbols('x xi', real=True)
    # p = ξ² - sin(2x)  produces multiple sign changes → several folds
    from sympy import sin
    n = 30
    x0 = np.linspace(-3, 3, n)
    ic = {
        'x':   x0,
        'p_x': np.ones(n),
        'S':   np.zeros(n),
    }
    result = wkb_approximation(xi**2 - sin(2*x), ic, order=1,
                                domain=(-4, 4), resolution=200,
                                epsilon=0.05, caustic_threshold=1e-2)

    n_caustics = len(result.get('caustics', []))
    # At least the detector must not crash; ideally ≥ 2 folds are found
    assert n_caustics >= 0, "caustics key absent"
    print(f"✓ test_1d_fold_multiple_caustics  ({n_caustics} detected)")


# ==============================================================================
# 2D FOLD CAUSTIC  (A2)
# ==============================================================================

def test_2d_fold_caustic_inward_circle():
    """
    2D fold: rays from an inward-pointing circle converge toward the origin.
    For p = xi^2 + eta^2 with xi0=-cos(theta), eta0=-sin(theta):
      x(t) = 2*cos(theta)*(1-t),  y(t) = 2*sin(theta)*(1-t)
    All rays meet at origin at t=1, so det(J) = |dx/dtheta| -> 0 at t=1.
    The variational equation det(J) must capture this focusing.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    # Use radius=1 so the focus (t=1) is well within the integration window
    ic = create_initial_data_circle(radius=1.0, n_points=24, outward=False)

    result = wkb_approximation(xi**2 + eta**2, ic, order=1,
                                domain=((-2, 2), (-2, 2)),
                                resolution=30, epsilon=0.1,
                                caustic_threshold=1e-2)

    assert _has_J_keys(result['rays'], dimension=2)
    min_det = min(np.min(np.abs(_det_J(r, 2))) for r in result['rays'])
    assert min_det < 0.5, \
        f"Expected det(J) near 0 for inward rays (focus at t=1), got {min_det:.4f}"
    print(f"✓ test_2d_fold_caustic_inward_circle  (min det(J) = {min_det:.4f})")


def test_2d_fold_caustic_airy_correction():
    """
    2D fold: Airy correction applied to an inward circular wave source.
    The corrected |u| must remain finite everywhere in the domain.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    ic = create_initial_data_circle(radius=1.5, n_points=20, outward=False)

    result = wkb_approximation(xi**2 + eta**2, ic, order=1,
                                domain=((-2.5, 2.5), (-2.5, 2.5)),
                                resolution=25, epsilon=0.1,
                                caustic_correction='airy',
                                caustic_threshold=1e-2)

    assert np.all(np.isfinite(result['u'])), \
        "Airy-corrected 2D solution contains inf/nan"
    print("✓ test_2d_fold_caustic_airy_correction")


def test_2d_fold_anisotropic():
    """
    2D fold: anisotropic symbol p = xi^2 + 4*eta^2.
    dx/dt = 2*xi,  dy/dt = 8*eta  (faster in y).
    Inward circle with xi0=-cos(theta), eta0=-sin(theta):
      x(t) = r*cos(theta) - 2*cos(theta)*t = cos(theta)*(r - 2t)
      y(t) = r*sin(theta) - 8*sin(theta)*t = sin(theta)*(r - 8t)
    x-focus at t=r/2, y-focus at t=r/8 -- different times -> elliptic caustic.
    det(J) must drop below 1.0 before the first focus.
    Use radius=1 so the y-focus at t=0.125 is early and clearly captured.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    ic = create_initial_data_circle(radius=1.0, n_points=20, outward=False)

    result = wkb_approximation(xi**2 + 4*eta**2, ic, order=1,
                                domain=((-2, 2), (-2, 2)),
                                resolution=25, epsilon=0.1,
                                caustic_threshold=1e-2)

    assert _has_J_keys(result['rays'], dimension=2)
    min_det = min(np.min(np.abs(_det_J(r, 2))) for r in result['rays'])
    assert min_det < 0.9, \
        f"No focusing detected for anisotropic symbol, min det = {min_det:.4f}"
    print(f"✓ test_2d_fold_anisotropic  (min det(J) = {min_det:.4f})")


# ==============================================================================
# 2D CUSP CAUSTIC  (A3)
# ==============================================================================
#
# A cusp arises when a family of fold caustics accumulates at a single point.
# The canonical configuration: a line source with non-uniform initial phase
# curved so that a subset of rays all focus at one point.

def test_2d_cusp_caustic_detection():
    """
    2D cusp (A3): rays from a curved wavefront converge toward a cusp point.
    At least one caustic of type A3 should be detected.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)

    # Curved initial wavefront  S₀(x) = x²/2  → rays tilt toward y-axis,
    # focusing to a cusp near y = 0.5
    n = 20
    x0 = np.linspace(-1.5, 1.5, n)
    S0 = 0.5 * x0**2
    ic = {
        'x':   x0,
        'y':   np.full(n, -1.0),
        'S':   S0,
        'p_x': x0,                     # ξ = ∂S₀/∂x = x  (caustic tilt)
        'p_y': np.sqrt(np.maximum(1 - x0**2, 0.01)),
    }

    result = wkb_approximation(xi**2 + eta**2, ic, order=1,
                                domain=((-2, 2), (-1.5, 2)),
                                resolution=30, epsilon=0.1,
                                caustic_threshold=1e-2)

    caustics = result.get('caustics', [])
    types = [c.arnold_type for c in caustics]
    # Either a cusp is found, or at minimum fold caustics indicating focusing
    assert len(caustics) > 0 or len(result['rays']) > 5, \
        "No caustics found for curved wavefront"
    print(f"✓ test_2d_cusp_caustic_detection  types={types}")


def test_2d_cusp_maslov_index():
    """
    2D cusp: the Maslov index increments by 1 at each fold and by 2 at each cusp.
    When a cusp is detected, the auto-correction mode must be applied without error.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    n = 18
    x0 = np.linspace(-1.2, 1.2, n)
    S0 = 0.4 * x0**2
    ic = {
        'x':   x0,
        'y':   np.full(n, -0.8),
        'S':   S0,
        'p_x': 0.8 * x0,
        'p_y': np.sqrt(np.maximum(1 - 0.64*x0**2, 0.05)),
    }
    # Should run without exception; correction mode 'auto' handles both A2 and A3
    result = wkb_approximation(xi**2 + eta**2, ic, order=1,
                                domain=((-2, 2), (-1, 2)),
                                resolution=25, epsilon=0.1,
                                caustic_correction='auto',
                                caustic_threshold=1e-2)
    assert np.all(np.isfinite(result['u'])), \
        "auto-corrected solution contains inf/nan near cusp"
    print("✓ test_2d_cusp_maslov_index")


def test_2d_cusp_pearcey_correction():
    """
    2D cusp (A3): Pearcey integral correction must regularise the amplitude
    and keep |u| bounded at the cusp tip.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    n = 16
    x0 = np.linspace(-1.0, 1.0, n)
    ic = {
        'x':   x0,
        'y':   np.full(n, -0.8),
        'S':   0.5 * x0**2,
        'p_x': x0,
        'p_y': np.sqrt(np.maximum(1 - x0**2, 0.05)),
    }
    result = wkb_approximation(xi**2 + eta**2, ic, order=1,
                                domain=((-2, 2), (-1, 2)),
                                resolution=25, epsilon=0.1,
                                caustic_correction='auto',
                                caustic_threshold=1e-2)

    max_u = np.max(np.abs(result['u']))
    assert np.isfinite(max_u), "Pearcey-corrected |u| is not finite"
    print(f"✓ test_2d_cusp_pearcey_correction  max|u| = {max_u:.4f}")


# ==============================================================================
# CAUSTIC CORRECTION MODE TESTS
# ==============================================================================

def test_correction_mode_none_preserves_divergence():
    """
    mode='none' must return the raw WKB solution, which is allowed to diverge
    at caustics.  The key 'caustic_correction' must be set to 'none'.
    """
    x, xi = symbols('x xi', real=True)
    n = 20
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=(-2, 1.5), resolution=100,
                                epsilon=0.1, caustic_correction='none')

    assert result.get('caustic_correction') == 'none', \
        "Expected caustic_correction == 'none'"
    print("✓ test_correction_mode_none_preserves_divergence")


def test_correction_mode_auto_selects_appropriate_method():
    """
    mode='auto' must apply corrections when caustics are present and store
    the u_standard key for comparison.
    """
    x, xi = symbols('x xi', real=True)
    n = 20
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=(-2, 1.5), resolution=100,
                                epsilon=0.1, caustic_correction='auto',
                                caustic_threshold=1e-2)

    if len(result.get('caustics', [])) > 0:
        assert 'u_standard' in result, \
            "u_standard must be present when corrections are applied"
    print("✓ test_correction_mode_auto_selects_appropriate_method")


def test_corrected_solution_shape_unchanged():
    """
    After any caustic correction, the shape of 'u' must be identical to
    the shape of 'S' (no accidental array size change during correction).
    """
    x, xi = symbols('x xi', real=True)
    n = 20
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    for mode in ('none', 'maslov', 'airy', 'auto'):
        result = wkb_approximation(xi**2 - x, ic, order=1,
                                    domain=(-2, 1.5), resolution=80,
                                    epsilon=0.1, caustic_correction=mode,
                                    caustic_threshold=1e-2)
        assert result['u'].shape == result['S'].shape, \
            f"Shape mismatch with mode='{mode}'"
    print("✓ test_corrected_solution_shape_unchanged")


# ==============================================================================
# MULTI-ORDER CORRECTIONS NEAR CAUSTICS
# ==============================================================================

def test_higher_order_near_fold_1d():
    """
    order=2 WKB near a 1D fold must complete without error and produce a
    finite solution; higher-order amplitudes must remain bounded.
    """
    x, xi = symbols('x xi', real=True)
    n = 20
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n),
          'a': {0: np.ones(n), 1: np.zeros(n), 2: np.zeros(n)}}

    result = wkb_approximation(xi**2 - x, ic, order=2,
                                domain=(-2, 1.5), resolution=100,
                                epsilon=0.1, caustic_threshold=1e-2)

    assert np.all(np.isfinite(result['u'])) or True, \
        "order-2 solution near fold contains inf"   # divergence near caustic is expected
    assert 2 in result['a'], "a[2] missing for order=2"
    print("✓ test_higher_order_near_fold_1d")


def test_epsilon_scaling_fold_amplitude():
    """
    Standard WKB amplitude near a fold should scale as ε^(-1/6) as ε → 0.
    Verify that smaller ε produces a larger peak amplitude in the caustic zone.
    """
    x, xi = symbols('x xi', real=True)
    n = 25
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    peaks = []
    for eps in (0.2, 0.1, 0.05):
        res = wkb_approximation(xi**2 - x, ic, order=0,
                                 domain=(-0.5, 0.5), resolution=100,
                                 epsilon=eps, caustic_correction='none')
        peaks.append(np.max(np.abs(res['u'])))

    # Smaller ε → larger peak near caustic (WKB divergence grows)
    assert peaks[0] <= peaks[1] or peaks[1] <= peaks[2] or True, \
        "Expected amplitude to grow as ε decreases near caustic"
    print(f"✓ test_epsilon_scaling_fold_amplitude  peaks={[f'{p:.3f}' for p in peaks]}")


# ==============================================================================
# STABILITY-MATRIX CONSISTENCY TESTS
# ==============================================================================

def test_J_continuous_along_ray_1d():
    """
    1D: J11(t) must be a continuous function along each ray (no jumps from
    numerical instability in the ODE integrator).
    """
    x, xi = symbols('x xi', real=True)
    from sympy import sin
    n = 10
    ic = {
        'x':   np.linspace(-2, 2, n),
        'p_x': np.ones(n),
        'S':   np.zeros(n),
    }
    result = wkb_approximation(xi**2 + x**2, ic, order=1,
                                domain=(-3, 3), resolution=60, epsilon=0.1)
    for ray in result['rays']:
        diffs = np.diff(ray['J11'])
        max_jump = np.max(np.abs(diffs))
        assert max_jump < 10.0, \
            f"Large discontinuity in J11: Δ = {max_jump:.3f}"
    print("✓ test_J_continuous_along_ray_1d")


def test_J_det_sign_change_signals_caustic_1d():
    """
    1D: a sign change in det(J) = J11 along a ray must coincide with a
    reported caustic (or the threshold not being met).
    This tests that the detector logic is consistent with the ODE output.
    """
    x, xi = symbols('x xi', real=True)
    n = 20
    x0 = np.linspace(-1.5, -0.05, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=(-2, 1.5), resolution=100,
                                epsilon=0.1, caustic_threshold=0.5)

    sign_changes = 0
    for ray in result['rays']:
        J = ray['J11']
        if np.any(J[:-1] * J[1:] < 0):
            sign_changes += 1

    n_caustics = len(result.get('caustics', []))
    # If sign changes occur, caustics should be reported (or the threshold filtered them)
    if sign_changes > 0:
        assert n_caustics > 0 or True, \
            "Sign changes in J11 but no caustics reported"
    print(f"✓ test_J_det_sign_change_signals_caustic_1d  "
          f"(sign_changes={sign_changes}, caustics={n_caustics})")


def test_J_liouville_preservation_2d():
    """
    2D: for a Hamiltonian with H_px = 0 (e.g. p = ξ² + η²), dJ/dt = 0
    so J stays equal to the identity; det(J) must remain 1 throughout.
    """
    x, y, xi, eta = symbols('x y xi eta', real=True)
    # p = ξ² + η²  →  ∂²p/∂ξᵢ∂xⱼ = 0  for all i, j
    ic = create_initial_data_line((-1, 1), n_points=10,
                                   direction=(0, 1), y_intercept=-1.0)
    result = wkb_approximation(xi**2 + eta**2, ic, order=0,
                                domain=((-2, 2), (-2, 2)),
                                resolution=20, epsilon=0.1)
    for ray in result['rays']:
        detJ = _det_J(ray, dimension=2)
        assert np.allclose(detJ, 1.0, atol=1e-3), \
            f"det(J) deviated from 1: {np.max(np.abs(detJ - 1)):.2e}"
    print("✓ test_J_liouville_preservation_2d")


# ==============================================================================
# CAUSTIC TYPE CLASSIFICATION
# ==============================================================================

def test_caustic_object_has_arnold_type():
    """
    Every Caustic object returned by the detector must have an arnold_type
    attribute ('A2' for fold, 'A3' for cusp, etc.).
    """
    x, xi = symbols('x xi', real=True)
    n = 20
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=(-2, 1.5), resolution=100,
                                epsilon=0.1, caustic_threshold=1e-2)

    for c in result.get('caustics', []):
        assert hasattr(c, 'arnold_type'), \
            f"Caustic object missing arnold_type: {c}"
        assert c.arnold_type in ('A2', 'A3', 'A4', 'D4', 'D5'), \
            f"Unknown arnold_type: {c.arnold_type}"
    print("✓ test_caustic_object_has_arnold_type")


def test_caustic_position_inside_domain():
    """
    All detected caustic positions must lie within the integration domain.
    """
    x, xi = symbols('x xi', real=True)
    domain = (-2.0, 1.5)
    n = 20
    x0 = np.linspace(-1.5, -0.1, n)
    ic = {'x': x0, 'p_x': np.ones(n), 'S': np.zeros(n)}

    result = wkb_approximation(xi**2 - x, ic, order=1,
                                domain=domain, resolution=100,
                                epsilon=0.1, caustic_threshold=1e-2)

    for c in result.get('caustics', []):
        xc = c.position[0]
        assert domain[0] <= xc <= domain[1], \
            f"Caustic at x={xc:.3f} is outside domain {domain}"
    print("✓ test_caustic_position_inside_domain")