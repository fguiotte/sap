#!/usr/bin/env python
# file test_spectra.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 02 juil. 2020
"""Abstract

doc.
"""

import sap
import pytest
import numpy as np
from matplotlib import pyplot as plt

@pytest.fixture
def image():
    return np.arange(100 * 100).reshape(100, 100)

@pytest.fixture
def maxt(image):
    return sap.MaxTree(image)

@pytest.mark.parametrize('vmin, vmax, params, exptd_count', [
    (1, 10, {}, 10),
    (1, 10, {'space': 'geo'}, 10),
    (0, 10, {'space': 'geo'}, 10),
    (0, 10, {'thresholds': 5}, 5),
])
def test_get_space(vmin, vmax, params, exptd_count):
    samples = sap.get_space(vmin, vmax, **params)

    assert samples.size == exptd_count

@pytest.mark.parametrize('params, exptd_count', [
    ({}, 11),
    ({'count': 5}, 6),
    ({'space': 'geo'}, 11),
    ({'outliers': .1}, 11)
])
def test_get_bins(maxt, params, exptd_count):
    area = maxt.get_attribute('area')
    bins = sap.get_bins(area, **params)

    assert bins.size == exptd_count

@pytest.mark.parametrize('params', [
    {},
    {'weighted': False},
    {'normalized': False},
])
def test_spectrum2d(maxt, params):
    ps = sap.spectrum2d(maxt, 'area', 'compactness', **params)

@pytest.mark.parametrize('params', [
    {},
    {'x_log': True, 'y_log': True},
])
def test_show_spectrum(maxt, params):
    ps = sap.spectrum2d(maxt, 'area', 'compactness', **params)
    sap.show_spectrum(*ps)
    plt.close()
