import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft, fftfreq, fftshift, ifftshift
from scipy.signal.windows import hann
from scipy.integrate import solve_ivp
from scipy.ndimage import maximum_filter
from scipy.sparse import diags
from scipy.sparse.linalg import svds 
from scipy.integrate import trapezoid as scipy_trapezoid
from sympy import (
    symbols, Function, 
    solve, pprint, Mul,
    lambdify, expand, Eq, simplify, trigsimp, N,
    radsimp, ratsimp, cancel,
    Lambda, Piecewise, Basic, degree, Pow, preorder_traversal, Heaviside, 
    powdenest, expand, Matrix,
    sqrt, I,  pi, series, oo, 
    re, im, arg, Abs, conjugate, 
    sin, cos, tan, cot, sec, csc, sinc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
    exp, ln, log, factorial, 
    gegenbauer, chebyshevu, legendre, assoc_legendre, hermite, laguerre, assoc_laguerre,
    diff, Derivative, integrate, 
    fourier_transform, inverse_fourier_transform,
)
from sympy.core.function import AppliedUndef
from scipy.special import legendre, eval_hermite, airy, eval_genlaguerre, jv, kv, sph_harm, gamma
from scipy.spatial.distance import cdist
from scipy.stats import norm
from scipy.stats import wasserstein_distance
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation
from matplotlib import rc
from functools import partial
from PIL import Image
import librosa, librosa.display
import soundfile as sf
from misc import * 
from IPython.display import display, clear_output, HTML, Video
from ipywidgets import interact, FloatSlider, Dropdown, VBox, HBox, interactive_output
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
import os

plt.rcParams['text.usetex'] = False
FFT_WORKERS = max(1, os.cpu_count() // 2)
NUM_COLS = 150
