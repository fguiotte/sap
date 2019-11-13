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

@pytest.mark.parametrize('adjacency', [4, 8])
def test_MaxTree_constructor(image, adjacency):
    t = sap.MaxTree(image, adjacency)

@pytest.mark.parametrize('adjacency', [4, 8])
def test_MinTree_constructor(image, adjacency):
    t = sap.MinTree(image, adjacency)
