#!/usr/bin/env python
# file test_profiles.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 10 déc. 2019

import sap
import pytest
import numpy as np

@pytest.fixture
def image():
    return np.arange(100 * 100).reshape(100, 100)

@pytest.mark.parametrize('adjacency, attribute, exptd_stacks, exptd_profiles', 
        [(4, {'area': [10, 100]}, 1, (5,)), 
         (8, {'area': [10, 100]}, 1, (5,)),
         (4, {'compactness': [.1, .5], 'volume': [100, 5000, 1000]}, 2, (5, 7))
        ])
def test_attribute_profiles(image, attribute, adjacency,
        exptd_stacks, exptd_profiles):

    aps = sap.attribute_profiles(image, attribute, adjacency)

    assert len(aps.data) == len(aps.description) == exptd_stacks, \
    'Expected description and data missmatch'

    for ap, ep in zip(aps.data, exptd_profiles):
        assert ap.shape[0] == ep, 'Expected profiles count missmatch'

