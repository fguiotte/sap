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


@pytest.mark.parametrize('adjacency', [4, 8])
def test_MaxTree_constructor(image, adjacency):
    t = sap.MaxTree(image, adjacency)

@pytest.mark.parametrize('adjacency', [4, 8])
def test_MinTree_constructor(image, adjacency):
    t = sap.MinTree(image, adjacency)

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
    deleted_nodes = np.zeros(max_tree.num_nodes(), dtype=np.bool)
    filtered_image = max_tree.reconstruct(deleted_nodes)
    assert (filtered_image == image).all(), 'Did not returned the same input image'

    deleted_nodes = np.ones(max_tree.num_nodes(), dtype=np.bool)
    filtered_image = max_tree.reconstruct(deleted_nodes)
    assert (filtered_image == 0).all(), 'Did not returned the filtered image'

    filtered_image = max_tree.reconstruct()
    assert (filtered_image == image).all(), 'Default input not working'

    filtered_image = max_tree.reconstruct(False)
    assert (filtered_image == image).all(), 'Boolean input not working'
