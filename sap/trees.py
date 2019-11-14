#!/usr/bin/env python
# file Tree.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 13 nov. 2019
"""
Trees
=====

This submodule contain the component tree helpers.

Example
-------

Basic creation of a tree and area filtering of an image with the
MaxTree:

>>>  t = sap.MaxTree(image)
>>>  area = t.get_attribute('area')
>>>  filtered_image = t.reconstruct(area < 100)

"""

import higra as hg

class Tree:
    """Tree

    Abstract class for trees representations of images. You should not
    instantiate directly Tree, use MaxTree or MinTree instead.

    Parameters
    ----------
    image: ndarray
        The image to be represented with the tree structure.
    adjacency: int
        The pixel connectivity to use during the tree creation.
        Determines the number of pixels to be taken into account in the
        neighborhood of each pixel. The allowed adjacency are 4 or 8.
        Default is 4.
        
    """
    def __init__(self, image, adjacency=4):
        if self.__class__ == Tree:
            raise TypeError('Do not instantiate directly abstract class Tree.')

        self._adjacency = adjacency
        self._image = image
        
    def _get_adjacency_graph(self):
        if self._adjacency == 4:
            return hg.get_4_adjacency_graph(self._image.shape)
        elif self._adjacency == 8:
            return hg.get_8_adjacency_graph(self._image.shape)
        else:
            raise NotImplementedError('adjacency of {} is not '
                    'implemented.'.format(self._adjacency))
    
    def get_attribute(self, attribute):
        """Get attribute values of the tree nodes 

        Details of the function.

        Parameters
        ------
        attribute: str
            Name of the attribute. Can be 'area' or 'comptactness' for
            exemple.

        Returns
        -------
        attribute: ndarray
            The values of attribute for each nodes.

        """
        compute = getattr(hg, 'attribute_' + attribute)
        args = {}
        if 'altitudes' in inspect.signature(compute).parameters:
            args['altitudes'] = self._alt
        return compute(self._tree, **args)
    
    def reconstruct(self, deleted_nodes=None):
        return hg.reconstruct_leaf_data(self._tree, self._alt, deleted_nodes)


class MaxTree(Tree):
    def __init__(self, image, adjacency=4):
        super().__init__(image, adjacency)
        graph = self._get_adjacency_graph()
        self._tree, self._alt = hg.component_tree_max_tree(graph, image)


class MinTree(Tree):
    def __init__(self, image, adjacency=4):
        super().__init__(image, adjacency)
        graph = self._get_adjacency_graph()
        self._tree, self._alt = hg.component_tree_min_tree(graph, image)


