# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
sys.path.insert(0, os.path.abspath('../../../'))

import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser

project = 'snngrow'
copyright = '2024, Utarn Technology Co., Ltd.'
author = 'Lei Yunlin, Gao Lanyu, Yang Xu'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for LaTeX output -------------------------------------------------
latex_engine = 'xelatex'
latex_elements = {'preamble':r'\usepackage{physics}'}

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']
