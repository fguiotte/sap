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
