#!/usr/bin/env python
# file test_profiles.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 10 déc. 2019

import sap
import pytest
import numpy as np
from matplotlib import pyplot as plt
import tempfile

@pytest.fixture
def image():
    return np.arange(100 * 100).reshape(100, 100)

@pytest.fixture
def profiles(image):
    return sap.attribute_profiles(image, {'area': [10, 100],
                                          'compactness': [.1, .5]},
                                        image_name='image')

@pytest.mark.parametrize('adjacency, attribute, exptd_stacks, exptd_profiles', 
        [(4, {'area': [10, 100]}, 1, (5,)), 
         (8, {'area': [10, 100]}, 1, (5,)),
         (4, {'compactness': [.1, .5], 'volume': [100, 5000, 1000]}, 2, (5, 7))
        ])
def test_attribute_profiles(image, attribute, adjacency,
        exptd_stacks, exptd_profiles):

    aps = sap.attribute_profiles(image, attribute, adjacency)

    assert len(aps) == exptd_stacks, \
    'Expected stacks missmatch'

    for ap, ep in zip(aps, exptd_profiles):
        assert ap.data.shape[0] == ep, 'Expected profiles count missmatch'

def test_attribute_profiles_assert(image):
    with pytest.raises(AttributeError) as e:
        sap.Profiles([image], [{}, {}, {}])

def test_profiles_iter(profiles):
    n = len(profiles)
    i = 0
    for profile in profiles:
        i += 1
    assert n == i, 'Wrong number of profiles expected in iter'


@pytest.mark.parametrize('params', [{}, {'height': 4}, {'fname': 'test.png'}])
def test_show_profiles(profiles, params, tmpdir):
    if 'fname' in params:
        params['fname'] = tmpdir.join(params['fname'])

    sap.show_profiles(profiles[0], **params)
    plt.close()

@pytest.mark.parametrize('params', [{}, {'height': 4}, {'fname': 'test.png'},
    {'image': 'image'}, {'attribute': 'area'}])
def test_show_all_profiles(profiles, params, tmpdir):
    if 'fname' in params:
        params['fname'] = tmpdir.join(params['fname'])

    sap.show_all_profiles(profiles[0], **params)
    plt.close()

def test_differential(profiles):
    sap.differential(profiles)

def test_profiles_m_diff(profiles):
    profiles.diff()
