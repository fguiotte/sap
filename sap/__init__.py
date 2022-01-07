#!/usr/bin/env python
# file __init__.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 12 nov. 2019
"""
Simple Attribute Profiles
=========================

This package provides:

1. Easy to use tree structures.
2. Straight forward attribute profiles computation.
3. Simple pattern spectra computation and display.

This package heavily relies on the outstanding library `higra
<https://higra.readthedocs.io>`_ which provide efficient tree structure
in C++. For more specific needs you definitely have to check this out.

Documentation
-------------

Documentation is available on docstrings and on the web page `python-sap
documentation <https://python-sap.rtfd.io>`_.

Once sap imported with:

>>> import sap

Use `help()` on the module, submodules, classes and functions to print
directly the docstrings.

>>> help(sap.trees)

Submodules access
-----------------

For simplicity, the submodules classes and functions are directly
available at the root of the module. In doing so:

>>> from sap import trees
>>> trees.MaxTree

Is equivalent to:

>>> import sap
>>> sap.MaxTree

Organisation of the submodules
------------------------------

- :mod:`sap.trees`: Submodule to build trees of images. This submodule
  also include functions and methods to easily process attribute filters
  on the trees.
- :mod:`sap.profiles`: Submodule to compute Attribute Profiles (and
  other profiles) of images, to concatenate them and to display them.
- :mod:`sap.spectra`: Submodule to compute Pattern Spectra of trees and
  to display them. Pattern Spectra can be useful to set thresholds of
  attribute filters and Attribute Profiles.

"""

from .trees import *
from .profiles import *
from .spectra import *
