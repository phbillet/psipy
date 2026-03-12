# Copyright 2026 Philippe Billet assisted by LLMs in free mode: chatGPT, Qwen, Deepseek, Gemini, Claude, le chat Mistral.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
:math:`\psi\pi` — A Python package for semiclassical analysis, microlocal methods, and geometric PDEs
============================================================================================

psipy provides a unified symbolic‑numerical framework for studying linear partial
differential equations using techniques from semiclassical analysis, microlocal
analysis, and differential geometry.  It is designed for both pedagogical
exploration and research, with a strong emphasis on asymptotic methods,
pseudodifferential operators, and geometric structures.

The package is organised into several modules, each focusing on a specific
aspect of the theory.  All modules automatically detect the spatial dimension
(1D or 2D) from the input data, allowing a consistent interface across problems.

Modules
-------

asymptotic
    Asymptotic evaluation of oscillatory and Laplace‑type integrals
    I(λ) = ∫ a(x) exp(iλ φ(x)) dx as λ → ∞.  Implements stationary phase,
    Laplace's method, and saddle‑point (steepest descent) with automatic
    method selection based on the phase φ.

caustics
    Catastrophe classification (Arnold's A₂, A₃, A₄, D₄±), detection of ray
    caustics via the stability matrix, and uniform asymptotic corrections
    using Airy (fold), Pearcey (cusp), and swallowtail integrals.

fio_bridge
    Fourier Integral Operators: links the pseudodifferential calculus
    (psiop) with asymptotic evaluation (asymptotic).  Builds FIOs for
    operator actions, propagators, and compositions, delegating all
    critical‑point work to the asymptotic module.

geometry
    Geometric visualisation and semiclassical analysis of Hamiltonian
    symbols H(x,ξ).  Computes Hamiltonian flows, caustics, periodic orbits,
    Maslov indices, Gutzwiller trace formula, and produces richly annotated
    multi‑panel figures.

microlocal
    High‑level microlocal analysis toolkit.  Handles characteristic varieties,
    bicharacteristic flows, wavefront sets, WKB approximations, caustic
    detection, and Bohr–Sommerfeld quantisation in a dimension‑agnostic way.

physics
    Lagrangian and Hamiltonian mechanics utilities.  Performs symbolic and
    numerical Legendre (and Legendre–Fenchel) transforms, decomposes
    Hamiltonians into local (polynomial) and non‑local parts, and generates
    formal PDEs (Schrödinger, wave, stationary).

propagator
    Van Vleck–Pauli–Morette semiclassical propagator.  Assembles the full
    semiclassical wavefunction

        ψ(x, t) = Σ_k  exp(i S_k/ℏ − i μ_k π/2) / √|det J_k|

    from a fan of classical rays traced on any 1D or 2D Riemannian metric
    defined by a ``riemannian.Metric`` object (or derived from a Hamiltonian
    expression H = ½ gⁱʲ pᵢ pⱼ via ``Metric.from_hamiltonian``).

psiop
    Comprehensive symbolic‑numerical environment for pseudodifferential
    operators.  Constructs symbols, computes asymptotic expansions,
    composes operators (Kohn–Nirenberg / Weyl), builds formal inverses,
    adjoints, exponentials, and visualises pseudospectra.

riemannian
    Unified Riemannian geometry in 1D and 2D.  Defines a metric, computes
    Christoffel symbols, geodesics, curvature tensors, Laplace–Beltrami
    operator, and (in 2D) exponential maps, Jacobi fields, and de Rham
    cohomology quantities.

solver
    Spectral PDE solver for 1D/2D equations with periodic or Dirichlet
    conditions.  Supports linear/nonlinear problems, pseudo‑differential
    operators, and time‑dependent/stationary equations.  Uses Fourier
    spectral methods and exponential time‑stepping (ETD‑RK4).

symplectic
    Symplectic geometry toolkit for Hamiltonian mechanics with arbitrary
    degrees of freedom.  Provides symplectic integrators (Euler, Verlet),
    Poisson brackets, fixed‑point analysis, action‑angle variables (1‑DOF),
    and Poincaré sections / monodromy matrices (2‑DOF).

wkb
    Multidimensional WKB (Wentzel–Kramers–Brillouin) approximation with
    caustic corrections.  Traces rays (bicharacteristics), solves transport
    equations to arbitrary order, detects caustics via the stability
    matrix, and applies uniform Airy / Pearcey corrections.

All modules are designed to work together seamlessly while remaining
independently usable.  The package relies on SymPy for symbolic manipulation,
NumPy/SciPy for numerical computation, and Matplotlib for visualisation.

For detailed mathematical background and references, consult the individual
module docstrings.
"""
from importlib.metadata import version

# Imports publics
from .psiop import *
from .solver import *
from .physics import *
from .geometry import *
from .hamiltonian_catalog import *
from .riemannian import *
from .symplectic import *
from .microlocal import *
from .asymptotic import *
from .fio_bridge import *
from .wkb import *
from .propagator import *

# Version du package
__version__ = version("psipy")

# Liste des noms exposés par `from psipy import *`
__all__ = [
    "PseudoDifferentialOperator",
    "PDESolver",
    "LagrangianHamiltonianConverter",
    "HamiltonianSymbolicConverter",
    "SymbolGeometry",
    "SymbolVisualizer",
    "SpectralAnalysis",
    "SymbolGeometry2D",
    "SymbolVisualizer2D",
    "Utilities2D",
    # Riemannian 1D
    'Metric1D',
    'geodesic_integrator',
    'laplace_beltrami',

    # Riemannian 2D
    'Metric2D',
    'geodesic_solver',
    'exponential_map',

    # Symplectic 1D
    'SymplecticForm1D',
    'hamiltonian_flow',
    'poisson_bracket',

    # Symplectic 2D
    'SymplecticForm2D',
    'hamiltonian_flow_4d',
    'poincare_section',

    # Microlocal 1D
    'characteristic_variety',
    'bicharacteristic_flow',
    'wkb_ansatz',
    'bohr_sommerfeld_quantization',

    # Microlocal 2D
    'characteristic_variety_2d',
    'bichar_flow_2d',
    'compute_maslov_index',

    # Propagator — Van Vleck–Pauli–Morette semiclassical wavefunction
    'RayData',
    'WKBResult',
    'compute_wavefunction',
    'van_vleck_sum',
    'plot_wavefunction',
    'plot_ray_fan',
    'plot_interference_detail',
]