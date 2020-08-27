#!/usr/bin/env python
# file utils.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 18 mars 2020
"""
Utils
=====

Various utilities unrelated to trees or profiles.

"""

import numpy as np

def ndarray_hash(x, l=8, c=1000):
    """
    Compute a hash from a numpy array.

    Parameters
    ----------
    x : ndarray
        The array to hash.
    l : int, optional
        The length of the hash. Must be an even number.
    c : int, optional
        A variable to affect the sampling of the hash. It has to be the
        same along the matching process. Refer to notes.

    Returns
    -------
    hash : str
        The hash of array x.

    Notes
    -----
    Python hash is slow and will offset the random generator in each
    kernel. The hash of the same data will not match in different
    kernels.

    The idea is to sparsely sample the data to speed up the hash
    computation. By fixing the number of samples the hash computation
    will take a fixed amount of time, no matter the size of the data.

    This hash function output a hash of :math:`x` in hexadecimal. The
    length of the hash is :math:`l`. The hashes are consistent when
    tuning the length :math:`l`: shorter hashes are contained in the
    longer ones for the same data :math:`x`. The samples count taken in
    :math:`x` is :math:`\\frac{l \\times c}{2}`.

    """
    rs = np.random.RandomState(42)
    x = np.require(x, requirements='C')
    bt = np.frombuffer(x, np.uint8)
    ss = rs.choice(bt, int(l / 2) * c).reshape(-1, c).sum(1, np.uint8)
    return ''.join(['{:02x}'.format(x) for x in ss])

def local_patch(arr, patch_size=7):
    """
    Create local patches around each value of the array

    Parameters
    ----------
    arr : ndarray
        The input data.
    patch_size : int
        The size :math:`w` of the patches. For a 2D nadarray the
        returned patch size will be :math:`w \\times w`.

    Returns
    -------
    patches : ndarray
        The local patches. The shape of the returned array is
        ``arr.shape + (patch_size,) * arr.ndim``.

    Notes
    -----
    This implementation is memory efficient. The returned patches are a
    view of original array and are not writeable.

    This function works regardless of the dimension of ``arr`` with
    hypercubes shaped patches, according to the dimension of ``arr``.

    See Also
    --------
    local_patch_f : use a function over the local patches.

    """
    a = np.pad(arr, int(patch_size / 2), 'reflect')
    shape = tuple(np.array(a.shape) - patch_size + 1) + (patch_size,) * a.ndim
    strides = a.strides * 2
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

def local_patch_f(arr, patch_size=7, f=np.mean):
    """local_patch_f(arr, patch_size=7, f=np.mean)
    Describe local patches around each value of the array

    Parameters
    ----------
    arr : ndarray
        The input data.
    patch_size : int
        The size :math:`w` of the patches.
    f : function
        The function to run over the local patches. For now it is
        necessary to use a function with ``axis`` parameter such as
        ``np.mean``, ``np.std``, etc... See more functions on `Numpy
        documentation
        <https://docs.scipy.org/doc/numpy/reference/routines.statistics.html>`_.

    Returns
    -------
    patches : ndarray
        The description of the local patches. The shape of the returned
        array is ``arr.shape``.

    Notes
    -----
    Refer to :func:`local_patch` for full documentation.

    See Also
    --------
    local_patch : create the local patches.

    """
    n = local_patch(arr, patch_size)
    return f(n, axis=tuple(~(np.arange(arr.ndim))))
