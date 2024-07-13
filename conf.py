# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0,os.path.abspath('.'))
sys.path.append('optimizacionpaquete_meliyork_1')

project = 'Paquete de optimizacion'

copyright = '2024, Melissa York'
author = 'Melissa York'
release = '12/07/24'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'es'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme =  "nature"
html_static_path = ['_static']

import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    html_output = os.path.join(os.environ['READTHEDOCS_OUTPUT'], 'html')
else:
    html_output = '_build/html'
