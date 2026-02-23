import numpy as np
import pytest
from sympy import symbols, Function, diff, I, pi, sin, cos, exp, sqrt, simplify
from caustics import (
    detect_catastrophes,
    classify_arnold_2d,
    plot_catastrophe # Uncomment if matplotlib is available
)

def test_catastrophe_detection_fold():
    """Test 8: 1D Catastrophe Detection (Fold) H(xi) = xi^3 - a*xi"""
    xi, a = symbols("xi a", real=True)
    H_fold = xi**3 - a*xi
    H_fold_a1 = H_fold.subs(a, 1)
    pts_fold = detect_catastrophes(H_fold_a1, (xi,))
    # Just test that the function runs without error and returns a list
    assert isinstance(pts_fold, list)


def test_catastrophe_detection_cusp():
    """Test 9: 2D Catastrophe Detection (Cusp-family) H(xi,eta) = xi^4 + eta^2"""
    xi, eta = symbols("xi eta", real=True)
    H_cusp = xi**4 + eta**2
    point_morse_min = {"xi": 0, "eta": 0}
    pts_cusp = detect_catastrophes(H_cusp, (xi, eta))
    # Just test that the function runs without error and returns a list
    assert isinstance(pts_cusp, list)


def test_arnold_classification_morse_min():
    """Test 13: Morse (Non-Degenerate Minimum) H = xi^2 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_morse_min = xi**2 + eta**2
    point_morse_min = {"xi": 0, "eta": 0}
    res_morse_min = classify_arnold_2d(H_morse_min, xi, eta, point_morse_min)
    expected_type_min = "Morse (non-degenerate)"
    assert res_morse_min['type'] == expected_type_min


def test_arnold_classification_morse_max():
    """Test 13b: Morse (Non-Degenerate Maximum) H = -xi^2 - eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_morse_max = -xi**2 - eta**2
    point_morse_min = {"xi": 0, "eta": 0}
    res_morse_max = classify_arnold_2d(H_morse_max, xi, eta, point_morse_min)
    expected_type_max = "Morse (non-degenerate)"
    assert res_morse_max['type'] == expected_type_max


def test_arnold_classification_a3():
    """Test 14: A3 (Cusp-family) H = xi^4 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_a3 = xi**4 + eta**2
    point_a3 = {"xi": 0, "eta": 0}
    res_a3 = classify_arnold_2d(H_a3, xi, eta, point_a3)
    assert "A3" in res_a3['type']


def test_arnold_classification_a4():
    """Test 15: A4 (Swallowtail) H = xi^5 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_a4 = xi**5 + eta**2
    point_a4 = {"xi": 0, "eta": 0}
    res_a4 = classify_arnold_2d(H_a4, xi, eta, point_a4)
    assert "A4" in res_a4['type']


def test_arnold_classification_a5():
    """Test 16: A5 (Butterfly) H = xi^6 + eta^2"""
    xi, eta = symbols('xi eta', real=True)
    H_a5 = xi**6 + eta**2
    point_a5 = {"xi": 0, "eta": 0}
    res_a5 = classify_arnold_2d(H_a5, xi, eta, point_a5)
    assert "A5" in res_a5['type']


def test_arnold_classification_d4p():
    xi, eta = symbols('xi eta', real=True)
    H_d4p = xi**3 + 3*xi*eta**2          # hyperbolic umbilic (D4+)
    point = {"xi": 0, "eta": 0}
    res = classify_arnold_2d(H_d4p, xi, eta, point)
    assert res["type"] == "D4+ (Hyperbolic umbilic)"
    I = res.get("cubic_invariant_I")
    assert I is not None and I < 0       # discriminant negative

def test_arnold_classification_d4m():
    xi, eta = symbols('xi eta', real=True)
    H_d4m = xi**3 - 3*xi*eta**2          # elliptic umbilic (D4-)
    point = {"xi": 0, "eta": 0}
    res = classify_arnold_2d(H_d4m, xi, eta, point)
    assert res["type"] == "D4- (Elliptic umbilic)"
    I = res.get("cubic_invariant_I")
    assert I is not None and I > 0       # discriminant positive
