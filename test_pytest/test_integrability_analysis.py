# test_integrability_analysis.py
# Test suite for IntegrabilityAnalysis (symplectic.py).
#
# Coverage:
#   weyl_law                — scalar correctness, monotonicity, hbar scaling
#   analyze_integrability   — Poisson / Wigner / intermediate classification,
#                             edge cases (constant spacings, single spacing)
#   berry_tabor_formula     — dict-style orbits, attribute-style orbits,
#                             dual-protocol adapter, symmetry, edge cases
#   detect_kam_tori         — clustering by action, single orbit, empty input,
#                             attribute- and dict-style orbit objects
#   winding_number          — CCW / CW circular orbit, straight trajectory
#   rotation_numbers        — isotropic oscillator, anisotropic ratio,
#                             key-name customisation

import numpy as np
import pytest
from types import SimpleNamespace

from sympy import symbols

from symplectic import (
    IntegrabilityAnalysis,
    hamiltonian_flow,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_orbit_ns(energy, period, action, stability=-0.1):
    """Return a SimpleNamespace mimicking a geometry.py PeriodicOrbit object."""
    return SimpleNamespace(
        energy=energy, period=period, action=action, stability=stability
    )


def _make_orbit_dict(energy, period, action, stability=-0.1):
    """Return a plain dict mimicking a symplectic.py trajectory-derived orbit."""
    return dict(energy=energy, period=period, action=action, stability=stability)


def _circular_traj(radius=1.0, n=1000, ccw=True):
    """
    Synthetic 2-DOF trajectory dict for a circular orbit.

    The winding-number convention: arctan2(y, x) increases for CCW motion,
    so a genuine CCW circle requires x = cos(t), y = sin(t) (y leads by 90°).
    Using two cosines with a phase offset does *not* guarantee CCW in the
    (x, y) plane and must be avoided here.

    Keys follow rotation_numbers defaults: 'x1','p1' for DOF-1,
    'x2','p2' for DOF-2.
    """
    t = np.linspace(0, 2 * np.pi * 5, n)   # 5 full turns
    sign = 1 if ccw else -1
    return {
        't':  t,
        'x1': radius * np.cos(sign * t),
        'p1': radius * np.sin(sign * t),   # (x1, p1) traces a CCW circle when sign=+1
        'x2': radius * np.cos(sign * t),
        'p2': radius * np.sin(sign * t),
    }


# =============================================================================
# weyl_law
# =============================================================================

class TestWeylLaw:

    def test_positive_for_positive_energy(self):
        """N(E) must be strictly positive for E > 0."""
        assert IntegrabilityAnalysis.weyl_law(2.0, ndof=1) > 0
        assert IntegrabilityAnalysis.weyl_law(2.0, ndof=2) > 0

    def test_zero_energy_gives_zero(self):
        """N(0) = 0 for any ndof."""
        assert IntegrabilityAnalysis.weyl_law(0.0, ndof=1) == 0.0
        assert IntegrabilityAnalysis.weyl_law(0.0, ndof=2) == 0.0

    def test_monotone_in_energy(self):
        """N(E) must be strictly increasing in E."""
        energies = np.linspace(0.5, 5.0, 10)
        values = [IntegrabilityAnalysis.weyl_law(E, ndof=1) for E in energies]
        assert np.all(np.diff(values) > 0)

    def test_1d_formula(self):
        """For ndof=1, N(E) = E / (2πℏ)."""
        E, hbar = 3.0, 1.0
        expected = E / (2 * np.pi * hbar)
        result = IntegrabilityAnalysis.weyl_law(E, ndof=1, hbar=hbar)
        assert np.isclose(result, expected)

    def test_2d_formula(self):
        """For ndof=2, N(E) = E² / (2πℏ)²."""
        E, hbar = 2.0, 1.0
        expected = (E / (2 * np.pi * hbar)) ** 2
        result = IntegrabilityAnalysis.weyl_law(E, ndof=2, hbar=hbar)
        assert np.isclose(result, expected)

    def test_hbar_scaling(self):
        """Halving ℏ should multiply N(E) by 2^ndof."""
        E, ndof = 4.0, 2
        n1 = IntegrabilityAnalysis.weyl_law(E, ndof=ndof, hbar=1.0)
        n2 = IntegrabilityAnalysis.weyl_law(E, ndof=ndof, hbar=0.5)
        assert np.isclose(n2 / n1, 2 ** ndof)

    def test_ndof_scaling(self):
        """N(E) with ndof=2 grows faster than with ndof=1 for E > 2π."""
        E = 10.0
        n1 = IntegrabilityAnalysis.weyl_law(E, ndof=1)
        n2 = IntegrabilityAnalysis.weyl_law(E, ndof=2)
        assert n2 > n1


# =============================================================================
# analyze_integrability
# =============================================================================

class TestAnalyzeIntegrability:
    """
    Tests for IntegrabilityAnalysis.analyze_integrability.

    The method accepts spacings via the keyword argument ``spacings=`` and
    returns a dict with top-level keys:
        verdict, verdict_source, soft_score, channels, warnings, summary

    Spectral statistics (ratio_R, mean/std of raw spacings, Brody β, KS
    p-values) live under ``result['channels']['spectral']`` when at least
    ``min_spacings`` (default 30) spacings are supplied.  For smaller inputs
    the spectral channel is skipped and the verdict is 'Undetermined'.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(spacings, **kwargs):
        """Call analyze_integrability with spacings as a keyword argument."""
        return IntegrabilityAnalysis.analyze_integrability(
            spacings=spacings, **kwargs
        )

    @staticmethod
    def _spectral(info):
        """Return the spectral sub-dict, or None if the channel was skipped."""
        return info.get('channels', {}).get('spectral')

    # ------------------------------------------------------------------
    # Classification / verdict tests
    # ------------------------------------------------------------------

    def test_poisson_spacings_classified_integrable(self):
        """Exponential (Poisson) spacings → verdict in the integrable family."""
        rng = np.random.default_rng(0)
        spacings = rng.exponential(scale=1.0, size=1000)
        info = self._run(spacings)
        assert info['verdict'] in ('Integrable', 'Likely integrable'), (
            f"Expected integrable verdict, got '{info['verdict']}'"
        )

    def test_wigner_spacings_classified_chaotic(self):
        """
        Wigner-distributed spacings (Rayleigh ≈ GOE marginal) → verdict in
        the chaotic family.
        """
        rng = np.random.default_rng(1)
        spacings = rng.rayleigh(scale=0.8, size=2000)
        info = self._run(spacings)
        assert info['verdict'] in ('Chaotic', 'Likely chaotic'), (
            f"Expected chaotic verdict, got '{info['verdict']}'"
        )

    def test_intermediate_ratio_R_in_range(self):
        """
        A 50/50 mix of Poisson and equal spacings should produce a ratio_R
        between the pure Poisson value (≈ 2) and the pure GOE value (≈ 4/3).
        We test that ratio_R lands in the intermediate band [1.4, 1.7].
        """
        rng = np.random.default_rng(2)
        poisson  = rng.exponential(1.0, size=500)
        uniform  = np.ones(500) * np.mean(poisson)
        mixed    = np.concatenate([poisson[:250], uniform[:250]])
        info     = self._run(mixed)
        sp       = self._spectral(info)
        assert sp is not None, "Spectral channel should be active for 500 spacings"
        ratio_R  = sp['ratio_R']
        assert 1.4 <= ratio_R <= 1.7, (
            f"Expected intermediate ratio_R, got {ratio_R:.3f} "
            f"(verdict: {info['verdict']})"
        )

    # ------------------------------------------------------------------
    # Return-structure tests
    # ------------------------------------------------------------------

    def test_returns_required_top_level_keys(self):
        """Output dict must contain the six documented top-level keys."""
        rng = np.random.default_rng(3)
        spacings = rng.exponential(1.0, size=100)
        info = self._run(spacings)
        for key in ('verdict', 'verdict_source', 'soft_score',
                    'channels', 'warnings', 'summary'):
            assert key in info, f"Missing top-level key: {key}"

    def test_spectral_channel_keys(self):
        """When enough spacings are supplied the spectral channel must expose
        its documented sub-keys."""
        rng = np.random.default_rng(3)
        spacings = rng.exponential(1.0, size=200)
        info = self._run(spacings)
        sp = self._spectral(info)
        assert sp is not None, "Spectral channel absent for 200 spacings"
        for key in ('beta', 'ks_poisson_p', 'ks_wigner_p',
                    'ratio_R', 'n_spacings', 'spacings_norm'):
            assert key in sp, f"Missing spectral key: {key}"

    def test_mean_and_std_of_raw_spacings_correct(self):
        """
        The spectral channel stores the *normalised* spacings in
        ``spacings_norm``.  The mean and std of the *original* spacings
        can be recovered as:
            mean = spacings_arr.mean()
            std  = spacings_arr.std()
        We verify that spacings_norm has unit mean (by construction) and
        that std(spacings_norm) * mean_original ≈ std_original.
        """
        rng = np.random.default_rng(4)
        spacings = rng.exponential(1.5, size=200)
        info = self._run(spacings)
        sp   = self._spectral(info)
        assert sp is not None
        s_norm = sp['spacings_norm']
        mean_orig = np.mean(spacings)
        std_orig  = np.std(spacings)
        assert np.isclose(s_norm.mean(), 1.0, atol=1e-10), (
            "spacings_norm should have unit mean"
        )
        assert np.isclose(s_norm.std() * mean_orig, std_orig, rtol=1e-6), (
            "std of normalised spacings × mean should equal original std"
        )

    def test_ratio_R_is_float(self):
        """ratio_R in the spectral channel must be a plain Python float."""
        rng = np.random.default_rng(5)
        spacings = rng.exponential(1.0, size=200)
        info = self._run(spacings)
        sp   = self._spectral(info)
        assert sp is not None
        assert isinstance(sp['ratio_R'], float)

    def test_soft_score_is_float_in_unit_interval(self):
        """soft_score must be a float in [0, 1] when a quantitative channel
        is active."""
        rng = np.random.default_rng(5)
        spacings = rng.exponential(1.0, size=200)
        info = self._run(spacings)
        assert isinstance(info['soft_score'], float)
        assert 0.0 <= info['soft_score'] <= 1.0

    # ------------------------------------------------------------------
    # Input-format and edge-case tests
    # ------------------------------------------------------------------

    def test_accepts_list_input(self):
        """analyze_integrability should accept a plain Python list of spacings."""
        spacings = [0.5, 1.0, 1.5, 2.0, 0.8, 1.2] * 10   # 60 items ≥ min_spacings
        info = self._run(spacings)
        assert 'verdict' in info

    def test_too_few_spacings_skips_spectral_channel(self):
        """
        Fewer than min_spacings (default 30) spacings → spectral channel
        absent and a warning is recorded.
        """
        info = self._run([2.5])
        assert self._spectral(info) is None, (
            "Spectral channel should be absent for a single spacing"
        )
        assert any('spacings' in w.lower() or 'few' in w.lower()
                   for w in info['warnings']), (
            "Expected a warning about insufficient spacings"
        )

    def test_constant_spacings_ratio_R_is_one(self):
        """All-equal spacings: s_norm ≡ 1 everywhere, so ratio_R = 1."""
        spacings = np.ones(100) * 3.7
        info = self._run(spacings)
        sp   = self._spectral(info)
        assert sp is not None
        assert np.isclose(sp['ratio_R'], 1.0), (
            f"Expected ratio_R=1 for constant spacings, got {sp['ratio_R']}"
        )

    def test_verdict_is_string(self):
        """verdict must be a non-empty string."""
        rng = np.random.default_rng(6)
        spacings = rng.exponential(1.0, size=100)
        info = self._run(spacings)
        assert isinstance(info['verdict'], str) and info['verdict']

    def test_n_spacings_matches_input_length(self):
        """spectral['n_spacings'] must equal the number of spacings passed in."""
        rng = np.random.default_rng(7)
        spacings = rng.exponential(1.0, size=150)
        info = self._run(spacings)
        sp   = self._spectral(info)
        assert sp is not None
        assert sp['n_spacings'] == len(spacings)


# =============================================================================
# berry_tabor_formula
# =============================================================================

class TestBerryTaborFormula:

    def test_positive_for_nearby_orbit(self):
        """Density must be strictly positive near an orbit's energy."""
        orbits = [_make_orbit_dict(energy=1.0, period=2 * np.pi, action=1.0)]
        rho = IntegrabilityAnalysis.berry_tabor_formula(orbits, energy=1.0)
        assert rho > 0.0

    def test_dict_and_namespace_give_same_result(self):
        """Dict-style and attribute-style orbits must produce identical output."""
        e, T, a = 2.0, np.pi, 1.5
        orbit_d = _make_orbit_dict(energy=e, period=T, action=a)
        orbit_n = _make_orbit_ns(energy=e, period=T, action=a)
        rho_d = IntegrabilityAnalysis.berry_tabor_formula([orbit_d], energy=2.0)
        rho_n = IntegrabilityAnalysis.berry_tabor_formula([orbit_n], energy=2.0)
        assert np.isclose(rho_d, rho_n)

    def test_wider_window_gives_higher_density_far_away(self):
        """
        For an orbit at E=1, evaluating at E=3 with a wide window should give
        a higher density than with a narrow window.
        """
        orbits = [_make_orbit_dict(energy=1.0, period=2 * np.pi, action=1.0)]
        rho_narrow = IntegrabilityAnalysis.berry_tabor_formula(
            orbits, energy=3.0, window=0.1
        )
        rho_wide = IntegrabilityAnalysis.berry_tabor_formula(
            orbits, energy=3.0, window=2.0
        )
        assert rho_wide > rho_narrow

    def test_additive_in_orbits(self):
        """
        Splitting a list of orbits and summing should equal the combined result,
        because the formula is a simple sum over orbits.
        """
        o1 = _make_orbit_dict(energy=1.0, period=1.0, action=1.0)
        o2 = _make_orbit_dict(energy=2.0, period=2.0, action=2.0)
        combined  = IntegrabilityAnalysis.berry_tabor_formula([o1, o2], energy=1.5)
        separate1 = IntegrabilityAnalysis.berry_tabor_formula([o1], energy=1.5)
        separate2 = IntegrabilityAnalysis.berry_tabor_formula([o2], energy=1.5)
        assert np.isclose(combined, separate1 + separate2)

    def test_longer_period_gives_higher_density(self):
        """
        Two orbits at the same energy; the one with the longer period should
        contribute more to the density at that energy.
        """
        short = _make_orbit_dict(energy=1.0, period=1.0, action=1.0)
        long  = _make_orbit_dict(energy=1.0, period=10.0, action=1.0)
        rho_s = IntegrabilityAnalysis.berry_tabor_formula([short], energy=1.0)
        rho_l = IntegrabilityAnalysis.berry_tabor_formula([long],  energy=1.0)
        assert rho_l > rho_s

    def test_symmetry_around_orbit_energy(self):
        """
        ρ(E₀ + δ) should equal ρ(E₀ − δ) for a single orbit centred at E₀.
        """
        orbit = _make_orbit_dict(energy=2.0, period=np.pi, action=1.0)
        delta = 0.3
        rho_plus  = IntegrabilityAnalysis.berry_tabor_formula(
            [orbit], energy=2.0 + delta, window=1.0
        )
        rho_minus = IntegrabilityAnalysis.berry_tabor_formula(
            [orbit], energy=2.0 - delta, window=1.0
        )
        assert np.isclose(rho_plus, rho_minus, rtol=1e-10)

    def test_returns_float(self):
        """Return type must be a plain Python float."""
        orbits = [_make_orbit_dict(energy=1.0, period=1.0, action=1.0)]
        result = IntegrabilityAnalysis.berry_tabor_formula(orbits, energy=1.0)
        assert isinstance(result, float)

    def test_empty_orbit_list_gives_zero(self):
        """No orbits → density is zero."""
        rho = IntegrabilityAnalysis.berry_tabor_formula([], energy=1.0)
        assert rho == 0.0

    def test_real_trajectory_orbit_dict(self):
        """
        Orbit dict produced by hamiltonian_flow + manual fields (period, action)
        should be accepted without errors.
        """
        x, p = symbols('x p', real=True)
        H = (p**2 + x**2) / 2
        traj = hamiltonian_flow(H, (1, 0), (0, 2 * np.pi),
                                vars_phase=[x, p], n_steps=500)
        orbit = dict(
            energy=float(np.mean(traj['energy'])),
            period=2 * np.pi,
            action=1.0,
            stability=-0.01,
        )
        rho = IntegrabilityAnalysis.berry_tabor_formula([orbit], energy=0.5)
        assert rho >= 0.0


# =============================================================================
# detect_kam_tori
# =============================================================================

class TestDetectKamTori:

    def test_empty_input(self):
        """Empty orbit list must return n_tori=0 and empty tori list."""
        result = IntegrabilityAnalysis.detect_kam_tori([])
        assert result['n_tori'] == 0
        assert result['tori'] == []

    def test_single_orbit_single_torus(self):
        """A single orbit must form exactly one torus."""
        orbit = _make_orbit_ns(energy=1.0, period=6.28, action=1.0)
        result = IntegrabilityAnalysis.detect_kam_tori([orbit])
        assert result['n_tori'] == 1
        assert len(result['tori']) == 1

    def test_two_well_separated_actions_two_tori(self):
        """Two action clusters far apart must yield two distinct tori."""
        orbits = [
            _make_orbit_ns(action=1.00, energy=1.0, period=6.28),
            _make_orbit_ns(action=1.02, energy=1.0, period=6.28),
            _make_orbit_ns(action=5.00, energy=5.0, period=6.28),
            _make_orbit_ns(action=5.03, energy=5.0, period=6.28),
        ]
        result = IntegrabilityAnalysis.detect_kam_tori(orbits, tolerance=0.5)
        assert result['n_tori'] == 2

    def test_tight_cluster_is_single_torus(self):
        """Three orbits with nearly identical action → one torus."""
        orbits = [
            _make_orbit_dict(action=2.00, energy=2.0, period=6.28),
            _make_orbit_dict(action=2.01, energy=2.0, period=6.28),
            _make_orbit_dict(action=2.02, energy=2.0, period=6.28),
        ]
        result = IntegrabilityAnalysis.detect_kam_tori(orbits, tolerance=0.5)
        assert result['n_tori'] == 1

    def test_torus_dict_has_required_keys(self):
        """Every torus entry must contain all documented keys."""
        orbits = [_make_orbit_ns(action=1.0, energy=1.0, period=6.28)]
        torus = IntegrabilityAnalysis.detect_kam_tori(orbits)['tori'][0]
        for key in ('id', 'n_orbits', 'action', 'energy', 'period', 'stable'):
            assert key in torus, f"Missing key: {key}"

    def test_torus_action_is_mean_of_members(self):
        """Torus action should be the mean action of its member orbits."""
        orbits = [
            _make_orbit_ns(action=1.0, energy=1.0, period=6.28),
            _make_orbit_ns(action=1.2, energy=1.2, period=6.30),
        ]
        result = IntegrabilityAnalysis.detect_kam_tori(orbits, tolerance=1.0)
        assert result['n_tori'] == 1
        torus = result['tori'][0]
        assert np.isclose(torus['action'], 1.1)

    def test_stable_flag_positive_stability(self):
        """An orbit with positive stability exponent → torus.stable = False."""
        orbits = [_make_orbit_ns(action=1.0, energy=1.0, period=6.28, stability=0.5)]
        result = IntegrabilityAnalysis.detect_kam_tori(orbits)
        assert result['tori'][0]['stable'] is False

    def test_stable_flag_negative_stability(self):
        """An orbit with negative stability exponent → torus.stable = True."""
        orbits = [_make_orbit_ns(action=1.0, energy=1.0, period=6.28, stability=-0.5)]
        result = IntegrabilityAnalysis.detect_kam_tori(orbits)
        assert result['tori'][0]['stable'] is True

    def test_n_tori_equals_len_tori(self):
        """n_tori must always equal len(tori)."""
        orbits = [
            _make_orbit_dict(action=0.5, energy=0.5, period=6.28),
            _make_orbit_dict(action=2.5, energy=2.5, period=6.28),
            _make_orbit_dict(action=5.0, energy=5.0, period=6.28),
        ]
        result = IntegrabilityAnalysis.detect_kam_tori(orbits, tolerance=0.1)
        assert result['n_tori'] == len(result['tori'])

    def test_dict_and_namespace_give_same_clustering(self):
        """Dict-style and attribute-style orbits must produce identical tori."""
        actions = [1.0, 1.05, 4.0, 4.05]
        orbits_ns   = [_make_orbit_ns(action=a, energy=a, period=6.28) for a in actions]
        orbits_dict = [_make_orbit_dict(action=a, energy=a, period=6.28) for a in actions]
        res_ns   = IntegrabilityAnalysis.detect_kam_tori(orbits_ns,   tolerance=0.5)
        res_dict = IntegrabilityAnalysis.detect_kam_tori(orbits_dict, tolerance=0.5)
        assert res_ns['n_tori'] == res_dict['n_tori']

    def test_tolerance_controls_merging(self):
        """
        A large tolerance merges all orbits into one torus; a small one keeps
        them separate.
        """
        orbits = [
            _make_orbit_ns(action=1.0, energy=1.0, period=6.28),
            _make_orbit_ns(action=3.0, energy=3.0, period=6.28),
        ]
        merged    = IntegrabilityAnalysis.detect_kam_tori(orbits, tolerance=10.0)
        separated = IntegrabilityAnalysis.detect_kam_tori(orbits, tolerance=0.1)
        assert merged['n_tori'] == 1
        assert separated['n_tori'] == 2


# =============================================================================
# winding_number
# =============================================================================

class TestWindingNumber:

    def test_ccw_circular_orbit_winding_positive(self):
        """A counter-clockwise orbit should have a positive winding number.

        The natural CCW-rotating pair in the trajectory dict is (x1, p1):
        x1 = cos(t), p1 = sin(t) traces the unit circle counter-clockwise,
        so arctan2(p1, x1) increases and the winding number is positive.
        Using (x1, x2) instead would give arctan2(cos(t+π/4), cos(t)),
        which is a Lissajous figure with retrograde winding — hence the
        explicit choice of x_key='x1', y_key='p1' here.
        """
        traj = _circular_traj(ccw=True)
        w = IntegrabilityAnalysis.winding_number(traj, x_key='x1', y_key='p1')
        assert w > 0

    def test_cw_circular_orbit_winding_negative(self):
        """A clockwise orbit should have a negative winding number."""
        traj = _circular_traj(ccw=False)
        w = IntegrabilityAnalysis.winding_number(traj, x_key='x1', y_key='p1')
        assert w < 0

    def test_winding_number_approximately_integer_multiple(self):
        """For 5 complete CCW turns the winding number should be close to 5."""
        traj = _circular_traj(n=5000, ccw=True)   # 5 turns
        w = IntegrabilityAnalysis.winding_number(traj, x_key='x1', y_key='p1')
        assert np.isclose(w, 5.0, atol=0.05), f"Expected ~5, got {w:.3f}"

    def test_winding_number_returns_float(self):
        traj = _circular_traj()
        w = IntegrabilityAnalysis.winding_number(traj, x_key='x1', y_key='p1')
        assert isinstance(w, float)

    def test_winding_number_default_keys(self):
        """
        Default key names are 'x' and 'y'; a trajectory with those keys should
        work without specifying x_key/y_key explicitly.
        """
        t = np.linspace(0, 2 * np.pi, 500)
        traj = {'t': t, 'x': np.cos(t), 'y': np.sin(t)}
        w = IntegrabilityAnalysis.winding_number(traj)
        assert np.isclose(w, 1.0, atol=0.05)

    def test_winding_number_physical_hamiltonian(self):
        """
        Isotropic 2-DOF oscillator: the (x1, x2) orbit is a Lissajous figure
        with winding number 0 for equal frequencies starting in phase.
        """
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
        traj = hamiltonian_flow(H, (1, 0, 0, 1), (0, 10 * np.pi),
                                vars_phase=[x1, p1, x2, p2], n_steps=5000)
        w = IntegrabilityAnalysis.winding_number(traj, x_key='x1', y_key='x2')
        # Both at frequency 1, 90° out of phase → winding number ≈ ±5
        assert abs(w) > 4.0


# =============================================================================
# rotation_numbers
# =============================================================================

class TestRotationNumbers:

    def test_isotropic_oscillator_equal_rotation_numbers(self):
        """
        For H = (p1²+p2²+x1²+x2²)/2, both rotation numbers should be equal.
        """
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + p2**2 + x1**2 + x2**2) / 2
        traj = hamiltonian_flow(H, (1, 0, 0, 1), (0, 20 * np.pi),
                                vars_phase=[x1, p1, x2, p2], n_steps=10000)
        om1, om2 = IntegrabilityAnalysis.rotation_numbers(traj)
        assert np.isclose(om1, om2, rtol=0.05), (
            f"Expected equal rotation numbers, got {om1:.4f} and {om2:.4f}"
        )

    def test_anisotropic_oscillator_frequency_ratio(self):
        """
        For H = (p1²+x1²)/2 + (p2²+4x2²)/2, the angular frequencies are
        ω₁=1 and ω₂=2, so rotation_number_2 / rotation_number_1 ≈ 2.
        """
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + x1**2) / 2 + (p2**2 + 4 * x2**2) / 2
        traj = hamiltonian_flow(H, (1, 0, 0.5, 0), (0, 20 * np.pi),
                                vars_phase=[x1, p1, x2, p2], n_steps=10000)
        om1, om2 = IntegrabilityAnalysis.rotation_numbers(traj)
        ratio = om2 / om1 if abs(om1) > 1e-8 else np.nan
        assert np.isclose(ratio, 2.0, rtol=0.05), (
            f"Expected frequency ratio 2, got {ratio:.4f}"
        )

    def test_rotation_numbers_returns_two_floats(self):
        """Return value must be a 2-tuple of floats."""
        traj = _circular_traj()
        result = IntegrabilityAnalysis.rotation_numbers(traj)
        assert len(result) == 2
        om1, om2 = result
        assert isinstance(om1, float)
        assert isinstance(om2, float)

    def test_custom_key_names(self):
        """
        Trajectories using non-default key names (e.g. 'q1','dp1') should work
        when the caller specifies the correct key arguments.
        """
        t = np.linspace(0, 10 * np.pi, 2000)
        traj = {
            't':  t,
            'q1': np.cos(t), 'dp1': np.sin(t),
            'q2': np.cos(t), 'dp2': np.sin(t),
        }
        om1, om2 = IntegrabilityAnalysis.rotation_numbers(
            traj, x_key='q1', p_key='dp1', y_key='q2', q_key='dp2'
        )
        assert np.isclose(om1, om2, rtol=0.05)

    def test_rotation_numbers_sign_consistent_with_direction(self):
        """
        For a pure CW rotation, both rotation numbers should be negative;
        for CCW, both should be positive.
        """
        traj_ccw = _circular_traj(ccw=True)
        traj_cw  = _circular_traj(ccw=False)
        om_ccw1, om_ccw2 = IntegrabilityAnalysis.rotation_numbers(traj_ccw)
        om_cw1,  om_cw2  = IntegrabilityAnalysis.rotation_numbers(traj_cw)
        assert om_ccw1 > 0 and om_ccw2 > 0
        assert om_cw1  < 0 and om_cw2  < 0

    def test_rotation_numbers_scale_with_integration_time(self):
        """
        Rotation numbers (per unit time) should be stable regardless of how
        long we integrate: doubling integration time should give the same ratio.
        """
        x1, p1, x2, p2 = symbols('x1 p1 x2 p2', real=True)
        H = (p1**2 + p2**2 + x1**2 + x2**2) / 2

        traj_short = hamiltonian_flow(H, (1, 0, 0, 1), (0, 10 * np.pi),
                                      vars_phase=[x1, p1, x2, p2], n_steps=3000)
        traj_long  = hamiltonian_flow(H, (1, 0, 0, 1), (0, 20 * np.pi),
                                      vars_phase=[x1, p1, x2, p2], n_steps=6000)

        om1_s, _ = IntegrabilityAnalysis.rotation_numbers(traj_short)
        om1_l, _ = IntegrabilityAnalysis.rotation_numbers(traj_long)
        assert np.isclose(om1_s, om1_l, rtol=0.05), (
            f"Short: {om1_s:.4f}, Long: {om1_l:.4f}"
        )