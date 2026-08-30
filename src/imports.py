import numpy as np
np.sqrt = np.lib.scimath.sqrt
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft, fftfreq, fftshift, ifftshift
from scipy.signal.windows import hann
from scipy.integrate import solve_ivp, dblquad, nquad, cumulative_trapezoid, quad
from scipy.ndimage import maximum_filter
from scipy.sparse import diags
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import svds
from scipy.integrate import trapezoid as scipy_trapezoid
from scipy.interpolate import griddata, interp1d
from scipy.linalg import expm
from sympy import (
    symbols, Function, 
    solve, pprint, Mul,
    lambdify, Eq, simplify, trigsimp, N, powsimp,
    radsimp, ratsimp, cancel, nsimplify, 
    Lambda, Piecewise, Basic, degree, Pow, preorder_traversal, Heaviside, 
    powdenest, expand, Matrix, expand_trig, 
    sqrt, I,  pi, series, oo, 
    Add, Mul, Poly, 
    re, im, arg, Abs, conjugate, 
    sin, cos, tan, cot, sec, csc, sinc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
    exp, ln, log, factorial, 
    gegenbauer, chebyshevu, legendre, assoc_legendre, hermite, laguerre, assoc_laguerre,
    diff, Derivative, integrate, 
    fourier_transform, inverse_fourier_transform,zeros,
    Integer, Rational, 
    latex, together, eye, sympify, 
)
from sympy.core.numbers import Zero, One
from sympy.core.function import AppliedUndef
from scipy.special import legendre, eval_hermite, airy, eval_genlaguerre, jv, kv, sph_harm_y, gamma
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.stats import wasserstein_distance
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib import rc
from functools import partial
from PIL import Image
import librosa, librosa.display
import soundfile as sf
from misc import * 
from IPython.display import display, clear_output, HTML, Video
from ipywidgets import interact, FloatSlider, Dropdown, VBox, HBox, interactive_output
import itertools
from mpl_toolkits.mplot3d import Axes3D
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Tuple, Union, Optional, Dict
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

plt.rcParams['text.usetex'] = False
FFT_WORKERS = max(1, os.cpu_count())
NUM_COLS = 150
