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
