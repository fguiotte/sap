#!/usr/bin/env python
# file test_Tree.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 13 nov. 2019

import pytest
import numpy as np
import sap

@pytest.fixture
def image():
    return np.arange(100 * 100).reshape(100, 100)

@pytest.fixture
def max_tree(image):
    return sap.MaxTree(image)

@pytest.fixture
def min_tree(image):
    return sap.MinTree(image)

@pytest.fixture
def tos_tree(image):
    return sap.TosTree(image)

@pytest.fixture
def aplha_tree(image):
    return sap.AlphaTree(image)

def test_Tree_constructor():
    with pytest.raises(TypeError):
        sap.Tree(None, None)

def test_Tree_adjacency(image):
    with pytest.raises(NotImplementedError):
        sap.MaxTree(image, 42)

@pytest.mark.parametrize('adjacency', [4, 8])
def test_MaxTree_constructor(image, adjacency):
    t = sap.MaxTree(image, adjacency)

@pytest.mark.parametrize('adjacency', [4, 8])
def test_MinTree_constructor(image, adjacency):
    t = sap.MinTree(image, adjacency)

def test_TosTree_constructor(image):
    t = sap.TosTree(image)

def test_AlphaTree_constructor(image):
    t = sap.AlphaTree(image)

def test_OmegaTree_constructor(image):
    t = sap.OmegaTree(image)

@pytest.mark.parametrize('watershed_attribute, markers',
        [('area', None),
         ('dynamics', None),
         ('volume', None),
         ('parents', None),
         ('area', np.ones_like(image))
        ])
def test_WatershedTree_constructor(image, watershed_attribute, markers):
    markers = np.ones_like(image)
    t = sap.WatershedTree(image, markers, watershed_attribute=watershed_attribute)

#def test_WatershedTree_construct(image):
#    graph = sap.hg.get_4_adjacency_graph(image.shape)
#
#    get_area = lambda tree, _: sap.hg.attribute_area(tree)
#    direct_function = sap.hg.watershed_hierarchy_by_area(graph, weight) 
#    lambda_function = sap.hg.watershed_hierarchy_by_attribute(graph, weight, get_area)
#
#    assert direct_function == lambda_function

def test_AlphaTree_exception(image):
    with pytest.raises(AttributeError):
        sap.AlphaTree(image, weight_function='L42')

    with pytest.raises(NotImplementedError):
        sap.AlphaTree(image, weight_function=np.array)

def test_available_attributes(max_tree):
    att_dict = max_tree.available_attributes()
    assert len(att_dict) > 0, 'Returned list is empty'

@pytest.mark.parametrize('attribute', [
'area',
'child_number',
'compactness',
'contour_length',
#'contour_strength',
'depth',
'dynamics',
'edge_length',
#'extinction_value',
'extrema',
'frontier_length',
#'frontier_strength',
'gaussian_region_weights_model',
'height',
'lca_map',
'mean_vertex_weights',
#'piecewise_constant_Mumford_Shah_energy',
'regular_altitudes',
'sibling',
'vertex_area',
'vertex_list',
'volume',])
def test_MaxTree_get_attribute(max_tree, attribute):
    attribute = max_tree.get_attribute(attribute)
    assert attribute is not None, 'get_attribute returned nothing'
    assert len(attribute) != 0, 'get_attribute returned an empty list'

def test_num_nodes(max_tree):
    assert max_tree.num_nodes() == 20000, 'Max tree did not returned correct number of nodes.'

def test_reconstruct(max_tree, image):
    deleted_nodes = np.zeros(max_tree.num_nodes(), dtype=bool)
    filtered_image = max_tree.reconstruct(deleted_nodes)
    assert (filtered_image == image).all(), 'Did not returned the same input image'

    deleted_nodes = np.ones(max_tree.num_nodes(), dtype=bool)
    filtered_image = max_tree.reconstruct(deleted_nodes)
    assert (filtered_image == 0).all(), 'Did not returned the filtered image'

    filtered_image = max_tree.reconstruct()
    assert (filtered_image == image).all(), 'Default input not working'

    filtered_image = max_tree.reconstruct(False)
    assert (filtered_image == image).all(), 'Boolean input not working'

@pytest.mark.parametrize('feature', ['altitude', 'area', 'compactness'])
def test_reconstruct_feature(max_tree, image, feature):
    filtered_image = max_tree.reconstruct(feature=feature)
    assert filtered_image is not None, 'Reconstruct returned nothing'

@pytest.mark.parametrize('filtering', ['direct', 'min', 'max', 'subtractive'])
def test_reconstruct_filtering(max_tree, image, filtering):
    filtered_image = max_tree.reconstruct(filtering=filtering)
    assert filtered_image is not None, 'Reconstruct returned nothing'

    assert (filtered_image == image).all(), 'Reconstruct should return the image'

@pytest.mark.parametrize('filtering', ['min', 'max', 'subtractive'])
def test_reconstruct_filtering_increasing(max_tree, image, filtering):
    threshold = 100
    area = max_tree.get_attribute('area')

    fdirect = max_tree.reconstruct(area < threshold, filtering='direct')
    frule = max_tree.reconstruct(area < threshold, filtering=filtering)

    assert (fdirect == frule).all(), \
            'Filtering rule {} did not return same result than rule direct'.format(filtering)

def test_str(max_tree):
    assert str(max_tree) == "MaxTree{'adjacency': 4,\n 'image.dtype': dtype('int64'),\n 'image.shape': (100, 100),\n 'image_hash': '0155fbbf',\n 'image_name': None,\n 'num_nodes': 20000}", \
    '__str__ of Tree did not returned expected output'

    mt = sap.MaxTree(None, None)
    assert str(mt) == 'MaxTree{}'

def test_io(max_tree, tmpdir):
    save_file = tmpdir + '/tree.npz'
    sap.save(save_file, max_tree)

    mt = sap.load(save_file)

    assert str(mt) == str(max_tree), 'Loaded Tree is different than saved Tree'

