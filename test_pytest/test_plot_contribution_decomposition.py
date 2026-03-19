# test_plot_contribution_decomposition.py
"""
Test suite for plot_contribution_decomposition() — Section 9.

Structure
---------
PCD-01  Empty critical_points list raises ValueError
PCD-02  Returns (fig, axes) of correct types
PCD-03  1D single Morse — default lambda_values, all panels present
PCD-04  1D single Morse — axes list length matches panel flags
PCD-05  show_correction=False suppresses Panel 2
PCD-06  show_coherent_sum=False suppresses Panel 3
PCD-07  Both flags False — single panel returned
PCD-08  Custom lambda_values are reflected on the x-axis data
PCD-09  Custom figsize is forwarded to the figure
PCD-10  Leading-term curve decays at the expected λ^(-1/2) rate (1D Morse)
PCD-11  Leading-term curve decays at the expected λ^(-1)   rate (2D Morse SP)
PCD-12  Leading-term curve decays at the expected λ^(-1)   rate (2D Morse Laplace)
PCD-13  Correction curve is smaller than leading for large λ (1D Morse)
PCD-14  Airy singularity — correction curve is absent / near zero (1D)
PCD-15  Two critical points — Panel 1 has exactly two data lines + one ref line
PCD-16  Coherent sum ≤ incoherent bound at every λ (two Morse points)
PCD-17  Coherent sum matches single-point total when n_pts == 1
PCD-18  2D Laplace two minima — both contributions present in Panel 1
PCD-19  Saddle-point (2D) — runs without error, single panel when sum suppressed
PCD-20  Figure title contains method name and dimension string

Run with:
    pytest test_plot_contribution_decomposition.py -v
"""

import warnings
import numpy as np
import sympy as sp
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from asymptotic import (
    Analyzer, AsymptoticEvaluator, SaddlePointEvaluator,
    IntegralMethod, plot_contribution_decomposition,
)

# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def _suppress_show(monkeypatch):
    """Suppress plt.show() in every test."""
    monkeypatch.setattr(plt, 'show', lambda: None)


@pytest.fixture()
def close_figs():
    """Close all figures after each test to avoid resource warnings."""
    yield
    plt.close('all')


# --- helpers ------------------------------------------------------------------

def _sp_analyzer(phi, amp, vars_):
    return Analyzer(phi, amp, vars_, method=IntegralMethod.STATIONARY_PHASE)


def _lap_analyzer(phi, amp, vars_):
    return Analyzer(phi, amp, vars_, method=IntegralMethod.LAPLACE)


def _morse_1d():
    """1D Morse stationary-phase: φ = x²/2, single critical point at x=0."""
    x = sp.Symbol('x')
    an = _sp_analyzer(x**2 / 2, sp.Integer(1), [x])
    pts = an.find_critical_points([np.zeros(1)])
    cps = [an.analyze_point(p) for p in pts]
    return cps, an


def _morse_2d_sp():
    """2D Morse stationary-phase: φ = x²/2 + y²/2, critical point at origin."""
    x, y = sp.symbols('x y')
    an = _sp_analyzer(x**2 / 2 + y**2 / 2, sp.Integer(1), [x, y])
    pts = an.find_critical_points([np.zeros(2)])
    cps = [an.analyze_point(p) for p in pts]
    return cps, an


def _morse_2d_lap():
    """2D Morse Laplace: ψ = x²/2 + y²/2, minimum at origin."""
    x, y = sp.symbols('x y')
    an = _lap_analyzer(x**2 / 2 + y**2 / 2, sp.Integer(1), [x, y])
    pts = an.find_critical_points([np.zeros(2)])
    cps = [an.analyze_point(p) for p in pts]
    return cps, an


def _double_morse_1d():
    """1D double-well: φ = x⁴/4 - x²/2, two Morse points at x = ±1."""
    x = sp.Symbol('x')
    an = _sp_analyzer(x**4 / 4 - x**2 / 2, sp.Integer(1), [x])
    pts = an.find_critical_points([np.array([1.0]), np.array([-1.0])])
    cps = [an.analyze_point(p) for p in pts]
    return cps, an


def _airy_1d():
    """1D Airy: φ = x³/3, degenerate critical point at x=0."""
    x = sp.Symbol('x')
    an = _sp_analyzer(x**3 / 3, sp.Integer(1), [x])
    pts = an.find_critical_points([np.zeros(1)])
    cps = [an.analyze_point(p) for p in pts]
    return cps, an


def _double_morse_2d_lap():
    """2D Laplace double well: ψ = (x²-1)² + y²/2, minima at (±1, 0)."""
    x, y = sp.symbols('x y')
    phi = (x**2 - 1)**2 + y**2 / 2
    an = _lap_analyzer(phi, sp.Integer(1), [x, y])
    pts = an.find_critical_points([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])
    cps = [an.analyze_point(p) for p in pts]
    return cps, an


def _saddle_2d():
    """2D saddle-point: φ = (1/2 + i/4)(x² + y²)."""
    x, y = sp.symbols('x y')
    phi = (sp.Rational(1, 2) + sp.I * sp.Rational(1, 4)) * (x**2 + y**2)
    an = Analyzer(phi, sp.Integer(1), [x, y],
                  method=IntegralMethod.SADDLE_POINT)
    se = SaddlePointEvaluator()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        saddles = se.find_saddle_points(an, [np.zeros(2)])
    cps = [an.analyze_point(s) for s in saddles]
    return cps, an


def _log_slope(lams, vals):
    """Empirical slope of log|vals| vs log(lams) via linear regression."""
    mask = np.isfinite(vals) & (vals > 0)
    if mask.sum() < 2:
        return float('nan')
    return np.polyfit(np.log(lams[mask]), np.log(vals[mask]), 1)[0]


def _panel1_data_lines(fig):
    """Return the plotted Line2D objects from the first Axes (Panel 1)."""
    return fig.axes[0].get_lines()


# ============================================================================
# Section 9 — plot_contribution_decomposition tests (20 cases)
# ============================================================================

class TestPlotContributionDecomposition:

    # ------------------------------------------------------------------
    # PCD-01  Empty list raises ValueError
    # ------------------------------------------------------------------
    def test_pcd01_empty_critical_points_raises(self, close_figs):
        """Passing an empty list must raise ValueError immediately."""
        x = sp.Symbol('x')
        an = _sp_analyzer(x**2 / 2, sp.Integer(1), [x])
        with pytest.raises(ValueError, match="at least one"):
            plot_contribution_decomposition([], an)

    # ------------------------------------------------------------------
    # PCD-02  Return types are (Figure, list)
    # ------------------------------------------------------------------
    def test_pcd02_return_types(self, close_figs):
        """Function must return (matplotlib.Figure, list of Axes)."""
        import matplotlib.figure
        cps, an = _morse_1d()
        fig, axes = plot_contribution_decomposition(cps, an)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(axes, list)
        assert all(hasattr(ax, 'plot') for ax in axes)

    # ------------------------------------------------------------------
    # PCD-03  Default call: all three panels present
    # ------------------------------------------------------------------
    def test_pcd03_default_three_panels(self, close_figs):
        """With show_correction=True, show_coherent_sum=True (defaults)
        the figure must have exactly 3 Axes."""
        cps, an = _morse_1d()
        fig, axes = plot_contribution_decomposition(cps, an)
        assert len(fig.axes) == 3
        assert len(axes) == 3

    # ------------------------------------------------------------------
    # PCD-04  axes list length matches panel flags
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("correction,coherent,expected", [
        (True,  True,  3),
        (True,  False, 2),
        (False, True,  2),
        (False, False, 1),
    ])
    def test_pcd04_panel_count(self, correction, coherent, expected, close_figs):
        """Number of axes must equal the number of active panels."""
        cps, an = _morse_1d()
        fig, axes = plot_contribution_decomposition(
            cps, an,
            show_correction=correction,
            show_coherent_sum=coherent,
        )
        assert len(axes) == expected
        assert len(fig.axes) == expected

    # ------------------------------------------------------------------
    # PCD-05  show_correction=False suppresses Panel 2
    # ------------------------------------------------------------------
    def test_pcd05_no_correction_panel(self, close_figs):
        """show_correction=False → 2 panels; Panel 2 ylabel absent."""
        cps, an = _morse_1d()
        fig, axes = plot_contribution_decomposition(
            cps, an, show_correction=False, show_coherent_sum=True
        )
        assert len(axes) == 2
        ylabels = [ax.get_ylabel() for ax in axes]
        assert not any("I_1" in lbl or "correction" in lbl.lower()
                       for lbl in ylabels)

    # ------------------------------------------------------------------
    # PCD-06  show_coherent_sum=False suppresses Panel 3
    # ------------------------------------------------------------------
    def test_pcd06_no_coherent_sum_panel(self, close_figs):
        """show_coherent_sum=False → 2 panels; no 'Coherent' title."""
        cps, an = _morse_1d()
        fig, axes = plot_contribution_decomposition(
            cps, an, show_correction=True, show_coherent_sum=False
        )
        assert len(axes) == 2
        titles = [ax.get_title() for ax in axes]
        assert not any("coherent" in t.lower() or "total" in t.lower()
                       for t in titles)

    # ------------------------------------------------------------------
    # PCD-07  Both flags False → single panel
    # ------------------------------------------------------------------
    def test_pcd07_single_panel(self, close_figs):
        """show_correction=False, show_coherent_sum=False → exactly 1 panel."""
        cps, an = _morse_1d()
        fig, axes = plot_contribution_decomposition(
            cps, an, show_correction=False, show_coherent_sum=False
        )
        assert len(axes) == 1

    # ------------------------------------------------------------------
    # PCD-08  Custom lambda_values are used
    # ------------------------------------------------------------------
    def test_pcd08_custom_lambda_values(self, close_figs):
        """The x-data of the first plotted line must match lambda_values."""
        cps, an = _morse_1d()
        lams = np.logspace(1, 3, 25)
        fig, axes = plot_contribution_decomposition(
            cps, an, lambda_values=lams,
            show_correction=False, show_coherent_sum=False
        )
        # First data line in Panel 1 (skip the reference dashed line)
        data_lines = [ln for ln in axes[0].get_lines() if ln.get_linestyle() != ':']
        assert len(data_lines) >= 1
        np.testing.assert_allclose(data_lines[0].get_xdata(), lams, rtol=1e-10)

    # ------------------------------------------------------------------
    # PCD-09  Custom figsize is forwarded
    # ------------------------------------------------------------------
    def test_pcd09_custom_figsize(self, close_figs):
        """figsize=(14, 6) must be reflected in the figure dimensions."""
        cps, an = _morse_1d()
        fig, _ = plot_contribution_decomposition(
            cps, an,
            show_correction=False, show_coherent_sum=False,
            figsize=(14, 6),
        )
        w, h = fig.get_size_inches()
        assert abs(w - 14) < 0.5
        assert abs(h - 6) < 0.5

    # ------------------------------------------------------------------
    # PCD-10  Leading-term slope ≈ -1/2 for 1D Morse (SP)
    # ------------------------------------------------------------------
    def test_pcd10_slope_1d_morse_sp(self, close_figs):
        """Panel 1 leading curve must decay as λ^(-1/2) for a 1D Morse point."""
        cps, an = _morse_1d()
        lams = np.logspace(1, 4, 60)
        evaluator = AsymptoticEvaluator()
        leads = np.array([abs(evaluator.evaluate(cps[0], l).leading_term)
                          for l in lams])
        slope = _log_slope(lams, leads)
        assert abs(slope - (-0.5)) < 0.05, f"Expected slope ≈ -0.5, got {slope:.3f}"

    # ------------------------------------------------------------------
    # PCD-11  Leading-term slope ≈ -1 for 2D Morse (SP)
    # ------------------------------------------------------------------
    def test_pcd11_slope_2d_morse_sp(self, close_figs):
        """Leading curve must decay as λ^(-1) for a 2D Morse SP point."""
        cps, an = _morse_2d_sp()
        lams = np.logspace(1, 4, 60)
        evaluator = AsymptoticEvaluator()
        leads = np.array([abs(evaluator.evaluate(cps[0], l).leading_term)
                          for l in lams])
        slope = _log_slope(lams, leads)
        assert abs(slope - (-1.0)) < 0.05, f"Expected slope ≈ -1.0, got {slope:.3f}"

    # ------------------------------------------------------------------
    # PCD-12  Leading-term slope ≈ -1 for 2D Morse (Laplace)
    # ------------------------------------------------------------------
    def test_pcd12_slope_2d_morse_laplace(self, close_figs):
        """Leading curve must decay as λ^(-1) for a 2D Laplace Morse point."""
        cps, an = _morse_2d_lap()
        lams = np.logspace(1, 4, 60)
        evaluator = AsymptoticEvaluator()
        leads = np.array([abs(evaluator.evaluate(cps[0], l).leading_term)
                          for l in lams])
        slope = _log_slope(lams, leads)
        assert abs(slope - (-1.0)) < 0.05, f"Expected slope ≈ -1.0, got {slope:.3f}"

    # ------------------------------------------------------------------
    # PCD-13  Correction < leading for large λ (1D Morse SP)
    # ------------------------------------------------------------------
    def test_pcd13_correction_smaller_than_leading(self, close_figs):
        """For λ ≥ 100 the correction term must be strictly smaller than the
        leading term for a 1D Morse stationary-phase point."""
        cps, an = _morse_1d()
        evaluator = AsymptoticEvaluator()
        for lam in [100.0, 500.0, 1000.0]:
            res = evaluator.evaluate(cps[0], lam)
            assert abs(res.correction_term) < abs(res.leading_term), (
                f"Correction ≥ leading at λ={lam}: "
                f"|corr|={abs(res.correction_term):.3e}, "
                f"|lead|={abs(res.leading_term):.3e}"
            )

    # ------------------------------------------------------------------
    # PCD-14  Airy singularity — correction is zero / absent
    # ------------------------------------------------------------------
    def test_pcd14_airy_correction_absent(self, close_figs):
        """For an Airy (degenerate) critical point the correction term
        must be zero and Panel 2 must contain no plotted data."""
        cps, an = _airy_1d()
        if not cps:
            pytest.skip("Airy critical point not found")
        lams = np.logspace(1, 3, 30)
        fig, axes = plot_contribution_decomposition(
            cps, an, lambda_values=lams,
            show_correction=True, show_coherent_sum=False
        )
        # Verify correction is numerically zero
        evaluator = AsymptoticEvaluator()
        for lam in [10.0, 100.0]:
            res = evaluator.evaluate(cps[0], lam)
            assert abs(res.correction_term) < 1e-12, (
                f"Expected zero correction for Airy, got {res.correction_term}"
            )
        # Panel 2 must have no lines with actual y-data (or only empty lines)
        corr_panel = axes[1]
        data_lines = [ln for ln in corr_panel.get_lines()
                      if len(ln.get_ydata()) > 0 and
                      np.any(np.isfinite(ln.get_ydata()))]
        assert len(data_lines) == 0, (
            "Panel 2 should have no data lines for an Airy singularity"
        )

    # ------------------------------------------------------------------
    # PCD-15  Two critical points — Panel 1 line count
    # ------------------------------------------------------------------
    def test_pcd15_two_points_line_count(self, close_figs):
        """With 2 critical points, Panel 1 must have exactly 2 data lines
        plus 1 reference (dotted) line = 3 total."""
        cps, an = _double_morse_1d()
        if len(cps) < 2:
            pytest.skip("Double-well did not find two critical points")
        lams = np.logspace(1, 3, 30)
        fig, axes = plot_contribution_decomposition(
            cps, an, lambda_values=lams,
            show_correction=False, show_coherent_sum=False
        )
        lines = axes[0].get_lines()
        # Data lines: linestyle != ':' (reference is dotted)
        data_lines = [ln for ln in lines if ln.get_linestyle() != ':']
        ref_lines  = [ln for ln in lines if ln.get_linestyle() == ':']
        assert len(data_lines) == 2, (
            f"Expected 2 data lines, got {len(data_lines)}"
        )
        assert len(ref_lines) == 1, (
            f"Expected 1 reference line, got {len(ref_lines)}"
        )

    # ------------------------------------------------------------------
    # PCD-16  Coherent ≤ incoherent at every λ (two Morse points)
    # ------------------------------------------------------------------
    def test_pcd16_coherent_le_incoherent(self, close_figs):
        """The coherent sum |Σ contributions| must be ≤ the incoherent bound
        Σ|contributions| at every λ (triangle inequality)."""
        cps, an = _double_morse_1d()
        if len(cps) < 2:
            pytest.skip("Double-well did not find two critical points")
        lams = np.logspace(1, 3, 40)
        fig, axes = plot_contribution_decomposition(
            cps, an, lambda_values=lams,
            show_correction=True, show_coherent_sum=True
        )
        # Panel 3 carries two lines: coherent (solid) and incoherent (dashed)
        sum_panel = axes[2]
        panel_lines = sum_panel.get_lines()
        assert len(panel_lines) >= 2
        coherent   = panel_lines[0].get_ydata()
        incoherent = panel_lines[1].get_ydata()
        assert np.all(coherent <= incoherent + 1e-10), (
            "Coherent sum exceeds incoherent bound — triangle inequality violated"
        )

    # ------------------------------------------------------------------
    # PCD-17  Single point: coherent sum equals the point's total
    # ------------------------------------------------------------------
    def test_pcd17_single_point_coherent_equals_total(self, close_figs):
        """With one critical point the coherent sum curve must equal
        |total_value| at every λ."""
        cps, an = _morse_1d()
        lams = np.logspace(1, 3, 30)
        evaluator = AsymptoticEvaluator()
        expected = np.array([abs(evaluator.evaluate(cps[0], l).total_value)
                             for l in lams])
        fig, axes = plot_contribution_decomposition(
            cps, an, lambda_values=lams,
            show_correction=True, show_coherent_sum=True
        )
        sum_panel = axes[2]
        coherent_line = sum_panel.get_lines()[0]
        np.testing.assert_allclose(
            coherent_line.get_ydata(), expected, rtol=1e-6,
            err_msg="Single-point coherent sum does not match total_value"
        )

    # ------------------------------------------------------------------
    # PCD-18  2D Laplace double well — two contributions in Panel 1
    # ------------------------------------------------------------------
    def test_pcd18_2d_laplace_two_minima(self, close_figs):
        """2D Laplace with two minima at (±1, 0): Panel 1 must show
        two distinct data curves, both with positive y-values."""
        cps, an = _double_morse_2d_lap()
        if len(cps) < 2:
            pytest.skip("2D Laplace double-well did not find two minima")
        lams = np.logspace(1, 3, 30)
        fig, axes = plot_contribution_decomposition(
            cps, an, lambda_values=lams,
            show_correction=True, show_coherent_sum=True
        )
        data_lines = [ln for ln in axes[0].get_lines()
                      if ln.get_linestyle() != ':']
        assert len(data_lines) == 2
        for ln in data_lines:
            assert np.all(np.isfinite(ln.get_ydata()))
            assert np.all(ln.get_ydata() > 0)

    # ------------------------------------------------------------------
    # PCD-19  Saddle-point 2D — runs without error
    # ------------------------------------------------------------------
    def test_pcd19_saddle_point_2d_no_error(self, close_figs):
        """plot_contribution_decomposition must complete without exception
        for a 2D saddle-point problem."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cps, an = _saddle_2d()
        if not cps:
            pytest.skip("No saddle point found")
        lams = np.logspace(1, 3, 20)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fig, axes = plot_contribution_decomposition(
                cps, an, lambda_values=lams,
                show_correction=True, show_coherent_sum=False
            )
        assert fig is not None
        assert len(axes) >= 1

    # ------------------------------------------------------------------
    # PCD-20  Figure title contains method name and dimension string
    # ------------------------------------------------------------------
    def test_pcd20_figure_title_content(self, close_figs):
        """The suptitle must mention the method name and the dimension."""
        cps, an = _morse_2d_sp()
        fig, _ = plot_contribution_decomposition(cps, an)
        title = fig.texts[0].get_text() if fig.texts else ""
        assert "stationary" in title.lower() or "phase" in title.lower(), (
            f"Method name missing from title: {title!r}"
        )
        assert "2d" in title.lower() or "2D" in title, (
            f"Dimension string missing from title: {title!r}"
        )

