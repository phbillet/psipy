from __future__ import annotations

# ==============================================================================
# 1. Standard Library
# ==============================================================================
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import lru_cache, partial
import itertools
import multiprocessing
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

# ==============================================================================
# 2. Third-Party Core & Utilities
# ==============================================================================
from PIL import Image
import librosa
import librosa.display
import soundfile as sf

# ==============================================================================
# 3. NumPy
# ==============================================================================
import numpy as np
from numpy.linalg import svd

# Enable complex square root handling globally
np.sqrt = np.lib.scimath.sqrt

# ==============================================================================
# 4. SciPy
# ==============================================================================
from scipy.cluster.hierarchy import fcluster, linkage

# Integration & ODEs
from scipy.integrate import (
    cumulative_trapezoid,
    dblquad,
    nquad,
    odeint,
    quad,
    solve_ivp,
)
from scipy.integrate import trapezoid as scipy_trapezoid

# Interpolation & Spatial Data
from scipy.interpolate import RegularGridInterpolator, griddata, interp1d
from scipy.spatial.distance import cdist

# Linear Algebra & Optimization
from scipy.linalg import expm, fractional_matrix_power, svdvals
from scipy.optimize import bisect, fsolve, minimize, minimize_scalar

# Image Processing & Filtering
from scipy.ndimage import gaussian_filter1d, maximum_filter

# Signal & Windowing
from scipy.signal import find_peaks
from scipy.signal.windows import hann

# Fourier Transforms
from scipy.fft import fft, fft2, fftfreq, fftshift, ifft, ifft2, ifftshift

# Sparse Matrices & Solvers
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    diags,
    lil_matrix,
)
from scipy.sparse import bmat as sparse_bmat
from scipy.sparse import eye as sparse_eye
from scipy.sparse.linalg import eigs, spsolve, svds

# Special Functions & Statistics
from scipy.special import (
    airy,
    eval_genlaguerre,
    eval_hermite,
    gamma,
    jv,
    kv,
    pbdv,
    sph_harm_y,
)
from scipy.special import legendre as scipy_legendre
from scipy.stats import kstest, linregress, norm, wasserstein_distance

# ==============================================================================
# 5. SymPy (Symbolic Math)
# ==============================================================================
from sympy import (
    E,
    I,
    Abs,
    Add,
    Basic,
    DiracDelta,
    Expr,
    Float,
    Function,
    Heaviside,
    Integer,
    Lambda,
    Matrix,
    MatrixBase,
    Max,
    Mul,
    N,
    Number,
    Piecewise,
    Poly,
    Pow,
    Rational,
    S,
    Symbol,
    # Special Polynomials
    assoc_laguerre,
    assoc_legendre,
    binomial,
    cancel,
    chebyshevu,
    # Calculus & Special Functions
    diff,
    Derivative,
    exp,
    expand,
    expand_trig,
    factor,
    factorial,
    fourier_transform,
    gegenbauer,
    hermite,
    integrate,
    inverse_fourier_transform,
    laguerre,
    lambdify,
    latex,
    legendre,
    ln,
    log,
    nsimplify,
    oo,
    pi,
    powdenest,
    powsimp,
    pprint,
    preorder_traversal,
    radsimp,
    ratsimp,
    series,
    simplify,
    solve,
    srepr,
    symbols,
    sympify,
    together,
    trigsimp,
    zeros,
)
from sympy.core.function import AppliedUndef
from sympy.core.numbers import One, Zero
from sympy.matrices import eye

# Complex and Structural Functions
from sympy import (
    arg,
    conjugate,
    degree,
    Eq,
    im,
    re,
    sign,
)

# Trigonometric & Hyperbolic Functions
from sympy import (
    acos,
    acosh,
    acot,
    acoth,
    acsc,
    acsch,
    asec,
    asech,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    cot,
    coth,
    csc,
    csch,
    sec,
    sech,
    sin,
    sinc,
    sinh,
    sqrt,
    tan,
    tanh,
)

# ==============================================================================
# 6. Matplotlib & Visualization
# ==============================================================================
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.gridspec import GridSpec

# Matplotlib Modules & Utilities
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Patch
import matplotlib.pyplot as plt

# Colormaps & Formatting
from matplotlib import cm, rc
from matplotlib.collections import LineCollection
import matplotlib.tri as tri

from mpl_toolkits.mplot3d import Axes3D

# Configure Matplotlib settings
plt.rcParams["text.usetex"] = False

# ==============================================================================
# 7. IPython & Interactive Widgets
# ==============================================================================
from IPython.display import HTML, Video, clear_output, display
from ipywidgets import (
    Dropdown,
    FloatSlider,
    HBox,
    VBox,
    interact,
    interactive_output,
)

# ==============================================================================
# 8. Global Configuration Constants
# ==============================================================================
FFT_WORKERS = max(1, os.cpu_count() or 1)
NUM_COLS = 150

# Custom / Project Utilities
from misc import *  # Note: PEP 8 discourages wildcard imports, but retained per project design