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

>>> tree = sap.MaxTree(image)
>>> ps = sap.spectrum2d(tree, 'area', 'compactness', x_log=True)
>>> sap.show_spectrum(*ps)

.. image:: ../../img/ps.png
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

def get_bins(x, count=10, space='lin', outliers=0.):
    """
    Return the bin edges over the values of x.

    Parameters
    ----------
    x : ndarray
        The values to be binned.
    count : int, optional
        Bin count to be returned. Default is 10.
    space : str, optional
        Spacing rule used to compute the bin. Can be 'lin' for linear
        spacing (default) or 'geo' for logarithmic spacing.
    outliers : float, optional
        Extremum quantiles to be considered as outliers to remove from
        the values before computing the bins.

    Returns
    -------
    bin_edges : ndarray, shape(count + 1,)
        The edges defining the bins.

    See Also
    --------
    get_space : Return spaced numbers with min and max values.

    """
    return get_space(*np.quantile(x, (outliers, 1 - outliers)), count + 1, space)

def get_space(vmin, vmax, thresholds=10, space='lin'):
    """
    Return spaced numbers over the range defined by vmin and vmax.

    Parameters
    ----------
    vmin : scalar
        The min value of the range.
    vmax : scalar
        The max value of the range.
    thresholds : int, optional
        The count of samples to be returned. Default is 10.
    space : str, optional
        Spacing rule used to compute the samples. Can be 'lin' for
        linear spacing (default) or 'geo' for logarithmic spacing.

    Returns
    -------
    samples : ndarray, shape(thresholds,)
        Spaced numbers over the range defined by vmin and vmax.

    See Also
    --------
    get_bins : Return the bin edges of a distribution.

    Notes
    -----
    When using 'geo' spacing, the range cannot include 0. The function
    will offset ``vmin`` to 0.1 if ``vmin`` is 0, as a workaround.

    """
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

def spectrum2d(tree, x_attribute, y_attribute, x_count=100, y_count=100, 
               x_log=False, y_log=False, weighted=True, normalized=True,
               node_mask=None):
    """
    Compute 2D pattern spectrum.

    Parameters
    ----------
    tree : sap.trees.Tree
        The tree used for creating the pattern spectrum.
    x_attribute : str
        The name of the attribute to be used on the x-axis.
    y_attribute : str
        The name of the attribute to be used on the y-axis.
    x_count : int, optional
        The number of bins along the x-axis. Default is 100.
    y_count : int, optional
        The number of bins along the y-axis. Default is 100.
    x_log : bool, optional
        If ``True``, the x-axis will be set to be a log scale. Default
        is ``False``.
    y_log : bool, optional
        If ``True``, the y-axis will be set to be a log scale. Default
        is ``False``.
    weighted : bool, optional
        If ``True``, the pattern spectrum is weighted. Each node of the
        tree will be weighted according to its size. This is the normal
        behaviour of pattern spectrum. If ``False`` the spectrum is not
        weighted, the output is a 2D histogram counting the number of
        nodes. Default is ``True``.
    normalized : bool, optional
        If ``True``, the weights of the spectrum are normalized with the
        size of the image. If ``False`` or ``weighted`` is ``False``,
        the spectrum is not normalized. Default is ``True``.
    node_mask : ndarray, optional
        Boolean mask array of the node to be considered in the spectrum.

    Returns
    -------
    s : ndarray, shape(x_count, y_count)
        The pattern spectrum.
    xedges : ndarray, shape(x_count + 1,)
        The bin edges along the x-axis.
    yedges : ndarray, shape(y_count + 1,)
        The bin edges along the y-axis.
    x_log : bool
        The parameter x_log indicating if the x-axis is a log scale.
    y_log : bool
        The parameter y_log indicating if the y-axis is a log scale.

    See Also
    --------
    sap.trees.available_attributes : List the name of available attributes.

    """
    x = tree.get_attribute(x_attribute)
    y = tree.get_attribute(y_attribute)

    bins = (get_bins(x, x_count, 'geo' if x_log else 'lin'),
            get_bins(y, y_count, 'geo' if y_log else 'lin'))

    weights = _compute_node_weights(tree) if weighted else None

    weights = weights / tree._image.size if normalized and weighted else weights
    #weights = weights / weights.max() if normalized and weighted else weights
    #weights = weights / weights.sum() if normalized and weighted else weights

    node_mask = np.ones_like(x, dtype=bool) if node_mask is None else node_mask
    weights = weights[node_mask] if weighted else None

    s, xedges, yedges = np.histogram2d(x[node_mask], y[node_mask], 
                                       bins=bins, density=None, 
                                       weights=weights)

    return s, xedges, yedges, x_log, y_log


def _compute_node_weights(tree):
    """
    Compute the node weights for weighted spectra
    """
    dh = tree._alt - tree._alt[tree._tree.parents()]
    area = tree.get_attribute('area')

    return area * dh


def show_spectrum(s, xedges, yedges, x_log, y_log, log_scale=True):
    """
    Display a pattern spectrum with matplotlib.

    Parameters
    ----------
    s : ndarray
        The pattern spectrum.
    xedges : ndarray
        The bin edges along the x-axis.
    yedges : ndarray
        The bin edges along the y-axis.
    x_log : bool
        Parameter indicating if the x-axis is a log scale.
    y_log : bool
        Parameter indicating if the y-axis is a log scale.
    log_scale : bool
        If ``True``, the colormap use a log scale. Default is ``True``.

    See Also
    --------
    spectrum2d : Compute a 2D pattern spectrum.

    Examples
    --------

    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> import sap

    Create a toy image, compute the "area" and "compactness" pattern
    spectrum with the tree of shape of the image.

    >>> image = np.arange(100 * 100).reshape(100, 100)
    >>> tree = sap.TosTree(image)
    >>> ps = sap.spectrum2d(tree, 'area', 'compactness', x_log=True)

    Setup a matplotlib figure with 2 subplots.

    >>> plt.figure(figsize=(10,4))
    >>> plt.subplot(1, 2, 1)

    Draw the spectrum with a linear color map.

    >>> sap.show_spectrum(*ps, log_scale=False)

    Decorate the subplot.

    >>> plt.colorbar()
    >>> plt.xlabel('area')
    >>> plt.ylabel('compactness')
    >>> plt.title('Linear colormap')

    >>> plt.subplot(1, 2, 2)

    Draw the spectrum with a log color map (default).

    >>> sap.show_spectrum(*ps)

    Decorate the subplot.

    >>> plt.colorbar()
    >>> plt.xlabel('area')
    >>> plt.title('Log colormap')

    Display the figure.

    >>> plt.show()

    .. image:: ../../img/ps_log.png

    """
    # XXX: pcolormesh is borked in matplotlib 3.5
    # https://github.com/matplotlib/matplotlib/issues/21917
    #pc = plt.pcolormesh(xedges, yedges, s.T, norm=mpl.colors.LogNorm() if log_scale else None)
    pc = plt.pcolor(xedges, yedges, s.T, norm=mpl.colors.LogNorm() if log_scale else None)

    plt.gca().set_xlim(xedges[0], xedges[-1])
    plt.gca().set_ylim(yedges[0], yedges[-1])

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

