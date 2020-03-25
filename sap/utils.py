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

    This hash function output a hash of $x$ in hexadecimal. The length
    of the hash is :math:`l`. The hashes are consistent when tuning the
    length :math:`l`: shorter hashes are contained in the longer ones
    for the same data :math:`x`. The samples count taken in :math:`x` is
    :math:`\\frac{l \\times c}{2}`.

    """
    rs = np.random.RandomState(42)
    bt = np.frombuffer(x, np.uint8)
    ss = rs.choice(bt, int(l / 2) * c).reshape(-1, c).sum(1, np.uint8)
    return ''.join(['{:02x}'.format(x) for x in ss])
