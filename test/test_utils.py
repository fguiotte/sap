#!/usr/bin/env python
# file test_utils.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 18 mars 2020

from sap import utils
import numpy as np

def test_hash():
    h = utils.ndarray_hash(np.arange(1337))

    assert h == '83a894f8', 'Wrong hash returned'

def test_local_patch():
    x = np.arange(25).reshape(5,5)

    patchs = utils.local_patch(x, 3)

    assert patchs.shape == (5, 5, 3, 3), 'Patches have a wrong shape'

    assert (patchs[2, 2] == np.array([[ 6,  7,  8],
                                      [11, 12, 13],
                                      [16, 17, 18]])).all(),\
            'Wrong patch returned'

def test_local_patch():
    x = np.arange(25).reshape(5,5)

    patchs = utils.local_patch(x, 3)

    assert patchs.shape == (5, 5, 3, 3), 'Patches have a wrong shape'

    assert (patchs[2, 2] == np.array([[ 6,  7,  8],
                                      [11, 12, 13],
                                      [16, 17, 18]])).all(),\
            'Wrong patch returned'

def test_local_patch_f():
    x = np.arange(25).reshape(5,5)

    mean_x = utils.local_patch_f(x, 3, np.mean)

    assert mean_x.shape == x.shape, 'Result have a wrong shape'

    assert mean_x[2,2] == np.array([[ 6,  7,  8],
                                    [11, 12, 13],
                                    [16, 17, 18]]).mean(), \
            'Wrong value returned'
