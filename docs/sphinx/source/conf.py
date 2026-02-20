# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'psipy'
copyright = '2026, Philippe Billet'
author = 'Philippe Billet'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # génère la doc depuis les docstrings
    'sphinx.ext.napoleon',       # supporte Google/NumPy style docstrings
    'sphinx.ext.viewcode',       # ajoute des liens vers le source
    'sphinx.ext.autosummary',    # génère des tableaux de résumé
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,       # inclut même les membres sans docstring
    'show-inheritance': True,
}

html_theme = 'sphinx_rtd_theme'  # thème Read the Docs, propre et lisible

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'  # thème Read the Docs, propre et lisible
html_static_path = ['_static']
