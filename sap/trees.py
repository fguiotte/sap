#!/usr/bin/env python
# file Tree.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 13 nov. 2019
"""
Trees
=====

This submodule contains the component tree classes.

Example
-------

Simple creation of the max-tree of an image, compute the area attributes
of the nodes and reconstruct a filtered image removing nodes with area
less than 100 pixels:

>>>  t = sap.MaxTree(image)
>>>  area = t.get_attribute('area')
>>>  filtered_image = t.reconstruct(area < 100)

"""

import higra as hg
import inspect

class Tree:
    """Tree

    Abstract class for trees representations of images. You should not
    instantiate directly Tree, use MaxTree or MinTree instead.

    Parameters
    ----------
    image: ndarray
        The image to be represented by the tree structure.
    adjacency: int
        The pixel connectivity to use during the tree creation. It
        determines the number of pixels to be taken into account in the
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

    def get_attribute(self, attribute_name):
        """Get attribute values of the tree nodes

        The attributes depends of Higra's

        Parameters
        ------
        attribute_name: str
            Name of the attribute. Can be 'area' or 'comptactness' for
            exemple.

        Returns
        -------
        attribute_values: ndarray
            The values of attribute for each nodes.

        See also
        --------
        TODO: list available attributes

        """
        compute = getattr(hg, 'attribute_' + attribute_name)
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
