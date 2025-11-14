# Hamiltonian Catalog — A Symbolic Library for Dynamical Systems

## Overview

**`HamiltonianCatalog`** is a comprehensive, symbolic library of **1D and 2D Hamiltonian systems**. It provides a massive, curated, and categorized collection of dynamical systems defined using **SymPy**.

This package bridges the gap between theoretical models and numerical applications. It is designed to be a foundational tool for researchers, educators, and developers working in:

  * Semiclassical and microlocal analysis
  * Quantum dynamics and chaos theory
  * Symbolic mathematics and code generation
  * Data-driven discovery of physical systems

The catalog's scope is exceptionally broad, spanning classical mechanics, quantum physics, biophysics, econophysics, and even speculative, cognitive, and aesthetic models.

-----

## Key Features

  * 🔹 **Vast Symbolic Database**: A collection of **over 600** Hamiltonians, each with a symbolic expression, dimension, category, and description.
  * 🔹 **Purely Symbolic**: All Hamiltonians are provided as `sympy.Expr` objects, allowing for effortless symbolic manipulation, differentiation, substitution, and analysis.
  * 🔹 **Rich Categorization**: Systems are classified into **over 90** distinct sub-categories (e.g., `chaotic`, `integrable`, `quantum`, `biophysics`, `econophysics`, `metaphysical`).
  * 🔹 **Hierarchical Structure**: A top-level `get_tree()` function organizes all categories into intuitive super-domains (e.g., "Physical Sciences," "Information & Cognition," "Creative Domains").
  * 🔹 **Powerful Search Utilities**: Includes functions to search, filter, and list Hamiltonians by name, keyword, category, or dimension.
  * 🔹 **Structural & Similarity Analysis**:
      * `get_dimensional_analysis()`: Analyzes expression properties (e.g., polynomial degree, presence of `sin`, `exp`, `log`).
      * `find_similar_hamiltonians()`: Finds structurally and categorically similar systems.
  * 🔹 **Multi-Format Export**: Built-in utilities to export the catalog (or subsets) to **LaTeX**, **JSON**, **YAML**, **CSV**, and **Markdown**.

-----

## Scope and Capabilities

  * **Dimensions**: 1D systems `(x, xi)` and 2D systems `(x, y, xi, eta)`.
  * **Symbolic Format**: Standardized `sympy.symbols` for coordinates, momenta, and common physical parameters (`m`, `k`, `alpha`, `omega`, etc.).
  * **Broad Thematic Domains**: The catalog includes, but is not limited to:
      * Classical & Celestial Mechanics
      * Quantum & Atomic Physics
      * Field Theory & High-Energy Physics
      * Condensed Matter & Materials
      * Relativity & Gravitation
      * Statistical & Non-Equilibrium Physics
      * Biophysics & Life Sciences
      * Information, Cognition & Society
      * Economic & Social Dynamics
      * Creative, Aesthetic & Speculative Systems

-----

## Example Usage

### Fetch a Specific Hamiltonian

```python
from hamiltonian_catalog import get_hamiltonian, print_hamiltonian_info

# Get the symbolic expression and metadata
H, vars, meta = get_hamiltonian("henon_heiles")

# H -> (xi**2 + eta**2)/2 + (x**2 + y**2)/2 + alpha*(x**2*y - y**3/3)
# vars -> (x, y, xi, eta)
# meta -> {'dim': 2, 'category': 'chaotic', ...}

# Pretty-print the full entry
print_hamiltonian_info("henon_heiles")
```

```
======================================================================
Hamiltonian: henon_heiles
======================================================================
Category:    chaotic
Dimension:   2D
Variables:   (x, y, xi, eta)

Description:
  Hénon–Heiles: benchmark for mixed regular/chaotic motion.

Expression:
  H = alpha*(x**2*y - y**3/3) + xi**2/2 + eta**2/2 + x**2/2 + y**2/2
======================================================================
```

### Search and Filter the Catalog

```python
from hamiltonian_catalog import search_hamiltonians, list_hamiltonians

# Find all systems related to "pendulum"
results = search_hamiltonians("pendulum")
# ['coupled_pendula', 'double_pendulum_reduced', 'driven_pendulum',
#  'spherical_pendulum', 'wilberforce_spring']

# List all 1D systems in the 'atomic' category
results_1d = list_hamiltonians(category='atomic', dim=1)
# ['hulthen', 'morse', 'poschl_teller', 'rosen_morse']
```

-----

## Core Utilities

| Function | Description |
| --- | --- |
| `get_hamiltonian(name)` | Fetches the symbolic `Expr`, variables, and metadata for a single system. |
| `list_hamiltonians(category, dim)` | Lists all Hamiltonian names, with optional filters for category and dimension. |
| `search_hamiltonians(keyword)` | Searches Hamiltonian names and descriptions for a case-insensitive keyword. |
| `list_categories()` | Returns a `dict` of all categories and the count of systems in each. |
| `get_tree()` | Returns a hierarchical `dict` mapping high-level domains to their sub-categories. |
| `get_dimensional_analysis(name)` | Reports structural properties (e.g., `has_trigonometric`, `polynomial_degree`). |
| `find_similar_hamiltonians(name)` | Finds structurally and categorically similar systems. |
| `export_latex_table(category, file)` | Exports a specific category (or all) to a LaTeX `longtable` file. |
| `batch_export_hamiltonians(dir)` | Exports the entire catalog to JSON, YAML, CSV, and Markdown in a directory. |

---

## 📜 License

Licensed under the **Apache License 2.0** — see [LICENSE](./LICENSE) for details.
Copyright © 2025 Philippe Billet

---
[Back to README](../README.md)