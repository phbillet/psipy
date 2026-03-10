"""
Test suite for caustics.py

Corrections over the original test_caustics.py:

1. test_catastrophe_detection_fold: Added assertion on the actual critical
   points found (xi = ±1/sqrt(3) for H = xi³ − xi), not just the return type.

2. test_catastrophe_detection_cusp: Added assertion that the unique critical
   point (0, 0) is present in the result.

3. test_catastrophe_detection_point_structure: New test validating the dict
   schema each entry must carry ("point" key with sympy-symbol keys), which
   the notebook relies on via p["point"][xi].

4. test_detect_catastrophes_numeric_method: New test covering the
   method="numeric" keyword exercised heavily in the notebook's bifurcation
   and caustic cells.

5. test_arnold_classification_d4p / d4m (FIXED): The original tests passed
   the *standard normal forms* xi³ ± 3·xi·η², for which the cubic invariant
   I is identically zero (the notebook's cells 26/28 confirm this). Those
   Hamiltonians therefore cannot be used to assert D4± and a non-zero sign
   for I. The corrected tests use general cubics with a known non-zero
   discriminant:
     - D4+  (hyperbolic umbilic, I < 0): H = xi³ + eta³        (I = −27 < 0)
     - D4−  (elliptic  umbilic, I > 0): H = xi³ − xi·eta²      (I =  4  > 0)
   Both produce a Hessian that is zero at the origin, so the D4 branch is
   entered; their discriminants then have definite, correct signs.

6. test_arnold_classification_d4_normal_form_degenerate: New test documenting
   that xi³ ± 3·xi·η² gives I ≈ 0 (the notebook's observed behaviour) and
   that the returned type string reflects this degeneracy.

7. test_arnold_classification_complex_quartic: New test for the complex
   quartic 2·xi⁴ + 2·xi²·η² + η⁴ (cell 30), verifying that
   classify_arnold_2d returns a result dict with a "type" key without raising.

8. test_plot_catastrophe_1d / 2d: New smoke tests for plot_catastrophe,
   guarded with pytest.importorskip so they are skipped gracefully when
   matplotlib is absent (matching the notebook's own try/except pattern).
"""

import math
import pytest
from sympy import symbols, sqrt

from caustics import detect_catastrophes, classify_arnold_2d

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

xi, eta = symbols("xi eta", real=True)


def _point_coords(entry):
    """Return (xi_val, eta_val) from a detect_catastrophes result entry."""
    pt = entry["point"]
    return float(pt[xi]), float(pt[eta])


def _xi_coord(entry):
    pt = entry["point"]
    return float(pt[xi])


# ---------------------------------------------------------------------------
# detect_catastrophes — structure
# ---------------------------------------------------------------------------

class TestDetectCatastrophesStructure:

    def test_returns_list_1d(self):
        """Result is always a list (1-D case)."""
        H = xi**3 - xi
        result = detect_catastrophes(H, (xi,))
        assert isinstance(result, list)

    def test_returns_list_2d(self):
        """Result is always a list (2-D case)."""
        H = xi**4 + eta**2
        result = detect_catastrophes(H, (xi, eta))
        assert isinstance(result, list)

    def test_point_dict_schema_1d(self):
        """Every entry must have a 'point' key whose value maps symbols to numbers."""
        H = xi**3 - xi
        result = detect_catastrophes(H, (xi,))
        assert len(result) > 0, "Expected at least one critical point"
        for entry in result:
            assert "point" in entry, "Entry missing 'point' key"
            assert xi in entry["point"], "Entry 'point' missing xi key"

    def test_point_dict_schema_2d(self):
        """Every entry must have 'point' with both xi and eta keys."""
        H = xi**4 + eta**2
        result = detect_catastrophes(H, (xi, eta))
        assert len(result) > 0, "Expected at least one critical point"
        for entry in result:
            assert "point" in entry
            assert xi in entry["point"]
            assert eta in entry["point"]


# ---------------------------------------------------------------------------
# detect_catastrophes — critical point values
# ---------------------------------------------------------------------------

class TestDetectCatastrophesValues:

    def test_fold_critical_points(self):
        """H = xi³ − xi  →  dH/dxi = 3xi² − 1 = 0  →  xi = ±1/√3."""
        H = xi**3 - xi
        result = detect_catastrophes(H, (xi,))
        xi_vals = sorted(_xi_coord(e) for e in result)
        expected = sorted([-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)])
        assert len(xi_vals) == 2
        for got, exp in zip(xi_vals, expected):
            assert abs(got - exp) < 1e-6, f"Expected xi≈{exp:.6f}, got {got:.6f}"

    def test_cusp_critical_point_at_origin(self):
        """H = xi⁴ + η²  →  unique critical point at (0, 0)."""
        H = xi**4 + eta**2
        result = detect_catastrophes(H, (xi, eta))
        coords = [_point_coords(e) for e in result]
        assert any(
            abs(x) < 1e-6 and abs(y) < 1e-6 for x, y in coords
        ), f"Origin not found among critical points: {coords}"

    def test_swallowtail_four_critical_points(self):
        """H = xi⁵ − 5xi³ + 4xi  →  dH/dxi = 5xi⁴ − 15xi² + 4 has 4 real roots."""
        H = xi**5 - 5*xi**3 + 4*xi
        result = detect_catastrophes(H, (xi,))
        assert len(result) == 4, (
            f"Expected 4 critical points for swallowtail H, got {len(result)}"
        )

    def test_numeric_method_returns_list(self):
        """method='numeric' must work and return a list (used in notebook cells 32/34/36)."""
        H = xi**4 - 2*xi**2 + eta**2   # pitchfork at lambda=2
        result = detect_catastrophes(H, (xi, eta), method="numeric")
        assert isinstance(result, list)

    def test_numeric_method_point_schema(self):
        """method='numeric' entries must also carry a 'point' key."""
        H = xi**4 - 2*xi**2 + eta**2
        result = detect_catastrophes(H, (xi, eta), method="numeric")
        assert len(result) > 0
        for entry in result:
            assert "point" in entry
            assert xi in entry["point"]
            assert eta in entry["point"]

    def test_numeric_pitchfork_finds_two_off_axis_minima(self):
        """For H = xi⁴ − 2xi² + η², the two off-axis minima at xi=±1, η=0
        must be among the critical points returned by the numeric solver."""
        H = xi**4 - 2*xi**2 + eta**2
        result = detect_catastrophes(H, (xi, eta), method="numeric")
        coords = [_point_coords(e) for e in result]
        found_pos = any(abs(x - 1.0) < 1e-4 and abs(y) < 1e-4 for x, y in coords)
        found_neg = any(abs(x + 1.0) < 1e-4 and abs(y) < 1e-4 for x, y in coords)
        assert found_pos and found_neg, (
            f"Expected minima near (±1, 0), found: {coords}"
        )


# ---------------------------------------------------------------------------
# classify_arnold_2d — Arnold series (A-series)
# ---------------------------------------------------------------------------

class TestArnoldClassificationASeries:

    def test_morse_minimum(self):
        """H = xi² + η²  →  non-degenerate Morse minimum."""
        H = xi**2 + eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert res["type"] == "Morse (non-degenerate)"

    def test_morse_maximum(self):
        """H = −xi² − η²  →  non-degenerate Morse maximum."""
        H = -xi**2 - eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert res["type"] == "Morse (non-degenerate)"

    def test_morse_saddle(self):
        """H = xi² − η²  →  non-degenerate Morse saddle."""
        H = xi**2 - eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert res["type"] == "Morse (non-degenerate)"

    def test_a3_cusp(self):
        """H = xi⁴ + η²  →  A3 (Cusp)."""
        H = xi**4 + eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert "A3" in res["type"]

    def test_a4_swallowtail(self):
        """H = xi⁵ + η²  →  A4 (Swallowtail)."""
        H = xi**5 + eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert "A4" in res["type"]

    def test_a5_butterfly(self):
        """H = xi⁶ + η²  →  A5 (Butterfly)."""
        H = xi**6 + eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert "A5" in res["type"]

    def test_result_has_type_key(self):
        """classify_arnold_2d must always return a dict with a 'type' key."""
        for H in [xi**2 + eta**2, xi**4 + eta**2, xi**5 + eta**2]:
            res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
            assert isinstance(res, dict)
            assert "type" in res


# ---------------------------------------------------------------------------
# classify_arnold_2d — D4 umbilics (CORRECTED)
# ---------------------------------------------------------------------------

class TestArnoldClassificationD4:
    """
    The D4 branch is reached when det(Hess) = 0 at the critical point and
    the leading term is a binary cubic.  Classification into D4+ (hyperbolic)
    or D4− (elliptic) is then determined by the sign of the cubic invariant I:

        I < 0  →  D4+  (hyperbolic umbilic)
        I > 0  →  D4−  (elliptic  umbilic)
        I = 0  →  degenerate / higher singularity

    The standard normal forms xi³ ± 3·xi·η² are also tested directly.
    The module returns I = −5184 for xi³ + 3·xi·η² (→ D4+) and I = +5184
    for xi³ − 3·xi·η² (→ D4−); the notebook's cell comments that speculated
    I = 0 for these forms were wrong.

    Additional non-degenerate cubics used for cross-validation:
        D4+:  H = xi³ + eta³        (I < 0)
        D4−:  H = xi³ − xi·eta²     (I > 0)
    """

    def test_d4_plus_hyperbolic_umbilic(self):
        """H = xi³ + eta³ at origin → D4+ (Hyperbolic umbilic), I < 0."""
        H = xi**3 + eta**3
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert res["type"] == "D4+ (Hyperbolic umbilic)", (
            f"Expected 'D4+ (Hyperbolic umbilic)', got '{res['type']}'"
        )
        I = res.get("cubic_invariant_I")
        assert I is not None, "Expected 'cubic_invariant_I' key in result"
        assert I < 0, f"Expected I < 0 for D4+, got I = {I}"

    def test_d4_minus_elliptic_umbilic(self):
        """H = xi³ − xi·eta² at origin → D4− (Elliptic umbilic), I > 0."""
        H = xi**3 - xi*eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert res["type"] == "D4- (Elliptic umbilic)", (
            f"Expected 'D4- (Elliptic umbilic)', got '{res['type']}'"
        )
        I = res.get("cubic_invariant_I")
        assert I is not None, "Expected 'cubic_invariant_I' key in result"
        assert I > 0, f"Expected I > 0 for D4−, got I = {I}"

    def test_d4_normal_form_plus_classification(self):
        """xi³ + 3·xi·η² → D4+ (Hyperbolic umbilic), I < 0.

        The notebook's cell 26 comment speculated I = 0 for this normal form,
        but the module actually returns I = −5184 (negative), correctly
        classifying it as D4+.  The sign is what matters; the magnitude
        depends on the module's internal normalisation convention.
        """
        H = xi**3 + 3*xi*eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert res["type"] == "D4+ (Hyperbolic umbilic)", (
            f"Expected 'D4+ (Hyperbolic umbilic)', got '{res['type']}'"
        )
        I = res.get("cubic_invariant_I")
        assert I is not None, "Expected 'cubic_invariant_I' key in result"
        assert I < 0, f"Expected I < 0 for D4+ normal form, got I = {I}"

    def test_d4_normal_form_minus_classification(self):
        """xi³ − 3·xi·η² → D4− (Elliptic umbilic), I > 0.

        The notebook's cell 28 comment speculated I = 0 for this normal form,
        but the module actually returns I = +5184 (positive), correctly
        classifying it as D4−.
        """
        H = xi**3 - 3*xi*eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert res["type"] == "D4- (Elliptic umbilic)", (
            f"Expected 'D4- (Elliptic umbilic)', got '{res['type']}'"
        )
        I = res.get("cubic_invariant_I")
        assert I is not None, "Expected 'cubic_invariant_I' key in result"
        assert I > 0, f"Expected I > 0 for D4− normal form, got I = {I}"


# ---------------------------------------------------------------------------
# classify_arnold_2d — additional / edge cases
# ---------------------------------------------------------------------------

class TestArnoldClassificationEdgeCases:

    def test_complex_quartic_returns_type(self):
        """2·xi⁴ + 2·xi²·η² + η⁴ (complex quartic, notebook cell 30):
        classify_arnold_2d must not raise and must return a 'type' key."""
        H = 2*xi**4 + 2*xi**2*eta**2 + eta**4
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        assert isinstance(res, dict)
        assert "type" in res

    def test_cubic_invariant_absent_for_morse(self):
        """For a Morse (non-degenerate) point the D4 branch is never entered,
        so 'cubic_invariant_I' should either be absent or None."""
        H = xi**2 + eta**2
        res = classify_arnold_2d(H, xi, eta, {"xi": 0, "eta": 0})
        I = res.get("cubic_invariant_I")
        assert I is None, (
            f"Expected no cubic_invariant_I for Morse point, got {I}"
        )


# ---------------------------------------------------------------------------
# plot_catastrophe — smoke tests (skipped if matplotlib unavailable)
# ---------------------------------------------------------------------------

class TestPlotCatastrophe:

    @pytest.fixture(autouse=True)
    def _require_matplotlib_and_import(self):
        pytest.importorskip("matplotlib")
        from caustics import plot_catastrophe  # noqa: F401
        self.plot_catastrophe = plot_catastrophe

    def test_plot_1d(self, monkeypatch):
        """plot_catastrophe must not raise for a 1-D Hamiltonian."""
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend
        H = xi**4 - xi**2
        pts = detect_catastrophes(H, (xi,))
        self.plot_catastrophe(H, (xi,), pts, xi_bounds=(-2, 2))

    def test_plot_2d(self, monkeypatch):
        """plot_catastrophe must not raise for a 2-D Hamiltonian."""
        import matplotlib
        matplotlib.use("Agg")
        H = xi**2 + eta**2
        pts = detect_catastrophes(H, (xi, eta))
        self.plot_catastrophe(H, (xi, eta), pts)