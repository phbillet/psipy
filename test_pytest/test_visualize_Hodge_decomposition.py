import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sympy import symbols, Matrix
from riemannian import Metric, hodge_decomposition, visualize_hodge_decomposition

# Test constants
DOMAIN_FLAT = ((0, 1), (0, 1))
RES_SMALL = 20

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def m_flat():
    x, y = symbols('x y', real=True)
    return Metric(Matrix([[1, 0], [0, 1]]), (x, y))

@pytest.fixture
def coords_2d():
    return symbols('x y', real=True)


# -----------------------------------------------------------------------------
# Test class
# -----------------------------------------------------------------------------

class TestVisualizeHodgeDecomposition:
    """
    Tests for the unified visualize_hodge_decomposition function.
    The Agg backend is used to avoid opening display windows.
    """

    @pytest.fixture(autouse=True)
    def use_agg(self):
        yield
        plt.close('all')

    # -------------------------------------------------------------------------
    # 1‑form tests
    # -------------------------------------------------------------------------

    def test_1form_auto_detect(self, m_flat, coords_2d):
        """1‑form decomposition with automatic form degree detection."""
        x, y = coords_2d
        dec = hodge_decomposition(m_flat, (-y, x), DOMAIN_FLAT, RES_SMALL)
        # Should run without raising; domain not needed because grid is present
        visualize_hodge_decomposition(dec)

    def test_1form_explicit_form_degree(self, m_flat, coords_2d):
        """Explicitly pass form_degree=1."""
        x, y = coords_2d
        dec = hodge_decomposition(m_flat, (-y, x), DOMAIN_FLAT, RES_SMALL)
        visualize_hodge_decomposition(dec, form_degree=1)

    # -------------------------------------------------------------------------
    # 2‑form tests
    # -------------------------------------------------------------------------

    def test_2form_auto_detect(self, m_flat):
        """2‑form decomposition with automatic detection (contains grid)."""
        dec = hodge_decomposition(m_flat, 1, DOMAIN_FLAT, RES_SMALL, form_degree=2)
        visualize_hodge_decomposition(dec)   # domain not needed

    def test_2form_explicit_form_degree(self, m_flat):
        """Explicitly pass form_degree=2."""
        dec = hodge_decomposition(m_flat, 1, DOMAIN_FLAT, RES_SMALL, form_degree=2)
        visualize_hodge_decomposition(dec, form_degree=2)

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

    def test_invalid_dict_raises(self):
        """Dictionary without expected keys raises."""
        invalid = {'something': np.zeros((10, 10))}
        with pytest.raises(ValueError, match="Cannot infer form degree"):
            visualize_hodge_decomposition(invalid)

    def test_invalid_form_degree_with_dict(self, m_flat, coords_2d):
        """
        Explicit form_degree that does not match dictionary keys raises KeyError.
        """
        x, y = coords_2d
        dec_1 = hodge_decomposition(m_flat, (-y, x), DOMAIN_FLAT, RES_SMALL)
        with pytest.raises(KeyError):
            visualize_hodge_decomposition(dec_1, form_degree=2)

    def test_valid_dict_without_grid_needs_domain(self, m_flat, coords_2d):
        """
        If a decomposition dictionary lacks a grid, domain must be provided.
        This test constructs a valid decomposition (1‑form) and removes the
        grid key to simulate that situation.
        """
        x, y = coords_2d
        dec = hodge_decomposition(m_flat, (-y, x), DOMAIN_FLAT, RES_SMALL)
        # Remove the grid key to mimic a decomposition without grid
        dec_no_grid = {k: v for k, v in dec.items() if k != 'grid'}

        # Should raise ValueError because domain is missing
        with pytest.raises(ValueError, match="No grid found in decomposition and `domain` not provided"):
            visualize_hodge_decomposition(dec_no_grid)

        # Should succeed when domain is provided
        visualize_hodge_decomposition(dec_no_grid, domain=DOMAIN_FLAT, resolution=RES_SMALL)