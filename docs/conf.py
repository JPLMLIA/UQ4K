"""
#!/usr/bin/env python
#
# uq4k documentation build configuration file, created by
# sphinx-quickstart on Fri Jun  9 13:47:02 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If EXTENSIONS (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory is
# relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#

"""
import os
import sys

import uq4k

sys.path.insert(0, os.path.abspath('..'))


# -- General configuration ---------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# EXTENSIONS coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
EXTENSIONS = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']

# Add any paths that contain templates here, relative to this directory.
TEMPLATES_PATH = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# SOURCE_SUFFIX = ['.rst', '.md']
SOURCE_SUFFIX = '.rst'

# The master toctree document.
MASTER_DOC = 'index'

# General information about the PROJECT.
PROJECT = 'uq4k'
COPYRIGHT = "2021, Mahdy Shirdel"
AUTHOR = "Mahdy Shirdel"

# The version info for the PROJECT you're documenting, acts as replacement
# for |version| and |RELEASE|, also used in various other places throughout
# the built documents.
#
# The short X.Y version.
VERSION = uq4k.__version__
# The full version, including alpha/beta/rc tags.
RELEASE = uq4k.__version__

# The LANGUAGE for content autogenerated by Sphinx. Refer to documentation
# for a list of supported LANGUAGEs.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "LANGUAGE" from the command line for these cases.
LANGUAGE = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to HTML_STATIC_PATH and html_extra_path
EXCLUDE_PATTERNS = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
PYGMENTS_STYLE = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
TODO_INCLUDE_TODOS = False


# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
HTML_THEME = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# HTML_THEME_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
HTML_STATIC_PATH = ['_static']


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
HTMLHELP_BASENAME = 'uq4kdoc'


# -- Options for LaTeX output ------------------------------------------

LATEX_ELEMENTS = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, AUTHOR, documentclass
# [howto, manual, or own class]).
LATEX_DOCUMENTS = [
    (MASTER_DOC, 'uq4k.tex', 'uq4k Documentation', 'Mahdy Shirdel', 'manual'),
]


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, AUTHORs, manual section).
MAN_PAGES = [(MASTER_DOC, 'uq4k', 'uq4k Documentation', [AUTHOR], 1)]


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, AUTHOR,
#  dir menu entry, description, category)
TEXINFO_DOCUMENTS = [
    (
        MASTER_DOC,
        'uq4k',
        'uq4k Documentation',
        AUTHOR,
        'uq4k',
        'One line description of PROJECT.',
        'Miscellaneous',
    ),
]
