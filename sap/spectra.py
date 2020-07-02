#!/usr/bin/env python
# file spectra.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 01 juil. 2020
"""
Spectra
========

This submodule contains pattern spectra related functions.

Example
-------

>>> import sap
>>> import numpy as np
>>> from matplotlib import pyplot as plt

>>> image = np.arange(100 * 100).reshape(100, 100)

Create the pattern spectrum (PS) of `image` with attributes area and
compactness with the max-tree.

>>> max_tree = sap.MaxTree(image)
>>> ps = sap.spectrum2d(tree, 'area', 'compactness', x_log=True)
>>> sap.show_spectrum(*ps)

.. image:: ../../img/ps.png
"""

def get_bins(attribute, count=10, space='lin', outliers=0):
    return get_space(*np.quantile(attribute, (outliers, 1-outliers)), count, space)

def get_space(vmin, vmax, thresholds=10, space='lin'):
    if space == 'lin':
        return _get_space_lin(vmin, vmax, thresholds)
    if space == 'geo':
        vmin = 0.1 * np.sign(vmax) if vmin == 0 else vmin
        #vmax = 0.1 * np.sign(vmin) if vmax == 0 else vmax
        return _get_space_geo(vmin, vmax, thresholds)

def _get_space_lin(vmin, vmax, thresholds):
    return np.linspace(vmin, vmax, thresholds)

def _get_space_geo(vmin, vmax, thresholds):
    return np.geomspace(vmin, vmax, thresholds)

def spectrum2d(tree, x_attribute, y_attribute, x_count=100, y_count=100, x_log=False, y_log=False, weighted=True):
    x = tree.get_attribute(x_attribute)
    y = tree.get_attribute(y_attribute)
    
    bins = (get_bins(x, x_count, 'geo' if x_log else 'lin'),
            get_bins(y, y_count, 'geo' if y_log else 'lin'))
    
    weights = tree.get_attribute('area') / tree._image.size if weighted else None
    
    s, xedges, yedges = np.histogram2d(x, y, bins=bins, density=None, weights=weights)
    
    return s, xedges, yedges, x_log, y_log


def show_spectrum(s, xedges, yedges, x_log, y_log, log_scale=True):
    pc = plt.pcolormesh(xedges, yedges, s.T, norm=mpl.colors.LogNorm() if log_scale else None)
    
    plt.gca().set_xlim(xedges[0], xedges[-1])
    plt.gca().set_ylim(yedges[0], yedges[-1])
    
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

