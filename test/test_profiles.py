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
@pytest.fixture
def profiles_b(image):
    return sap.attribute_profiles(image, {'area': [1000, 10000],
                                          'height': [.1, .5]},
                                        image_name='image b')

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

@pytest.mark.parametrize('adjacency, attribute, exptd_stacks, exptd_profiles',
        [(4, {'area': [10, 100]}, 1, (3,)),
         (8, {'area': [10, 100]}, 1, (3,)),
         (4, {'compactness': [.1, .5], 'volume': [100, 5000, 1000]}, 2, (3, 4))
        ])
def test_self_dual_attribute_profiles(image, attribute, adjacency,
        exptd_stacks, exptd_profiles):

    sdaps = sap.self_dual_attribute_profiles(image, attribute, adjacency)

    assert len(sdaps) == exptd_stacks, \
    'Expected stacks missmatch'

    for ap, ep in zip(sdaps, exptd_profiles):
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

def test_concatenate(profiles, profiles_b):
    np = sap.concatenate((profiles, profiles_b))
    assert  len(np) == len(profiles) + len(profiles_b), 'Length mismatch \
    with concatenated profiles'
    assert (profiles[0].data == np[0].data).all(), 'Data do not correspond'

    single_profile = profiles[0]
    nps = sap.concatenate((single_profile, profiles_b))
    assert len(nps) == 1 + len(profiles_b), 'Concatenate with profiles of \
    length 1 error'

    sap.concatenate((single_profile, profiles, profiles_b)), 'Concatenate more\
            than 2 profiles'

def test_profiles_add(profiles, profiles_b):
    np = profiles + profiles_b

    assert len(np) == len(profiles) + len(profiles_b)
    assert (profiles[0].data == np[0].data).all(), 'Data do not correspond'

def test_profiles_iadd(profiles, profiles_b):
    np = profiles
    np += profiles_b

    assert len(np) == len(profiles) + len(profiles_b)
    assert (profiles_b[-1].data == np[-1].data).all(), 'Data do not correspond'

def test_vectorize(profiles):
    vectors = sap.vectorize(profiles)

    assert len(vectors) == sum([len(x.data) for x in profiles]), 'Length of\
    vectors mismatch'

def test_profiles_vectorize(profiles):
    vectors = profiles.vectorize()

    assert len(vectors) == sum([len(x.data) for x in profiles]), 'Length of\
    vectors mismatch'

def test_strip_profiles(profiles):
    np = sap.strip_profiles(lambda x: x['operation'] != 'thinning', profiles)

    for nap, ap in zip(np, profiles):
        assert len(nap.data) == (len(ap.data) - 1) / 2, \
            'thinning profiles should be half of all profiles minus one.'

def test_strip_profiles_copy(profiles):
    np = sap.strip_profiles_copy(profiles)

    for p in np:
        assert not 'copy' in [x['operation'] for x in
            p.description['profiles']], 'There is a original image in\
            filtered profiles'

def test_profiles_strip(profiles):
    np = profiles.strip(lambda x: x['operation'] != 'thickening')

    for nap, ap in zip(np, profiles):
        assert len(nap.data) == (len(ap.data) - 1) / 2, \
            'thinning profiles should be half of all profiles minus one.'

def test_profiles_strip_copy(profiles):
    np = profiles.strip_copy()

    for p in np:
        assert not 'copy' in [x['operation'] for x in
            p.description['profiles']], 'There is a original image in\
            filtered profiles'

