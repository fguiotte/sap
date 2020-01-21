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
import numpy as np
import inspect

def available_attributes():
    """
    Return a dictionary of available attributes and parameters.

    Returns
    -------
    dict_of_attributes : dict
        The names of available attributes and parameters required.
        The names are keys (str) and the parameters are values (list
        of str) of the dictionary.

    See Also
    --------
    get_attribute : Return the attribute values of the tree nodes.

    Notes
    -----
    The list of available attributes is generated dynamically. It is
    dependent of higra's installed version. For more details, please
    refer to `higra documentation
    <https://higra.readthedocs.io/en/stable/python/tree_attributes.html>`_
    according to the appropriate higra's version.

    Example
    -------
    >>> sap.available_attributes()
    {'area': ['vertex_area=None', 'leaf_graph=None'],
     'compactness': ['area=None', 'contour_length=None', ...],
     ...
     'volume': ['altitudes', 'area=None']}

    """
    params_remove = ['tree']
    dict_of_attributes = {}
    for x in inspect.getmembers(hg):
        if x[0].startswith('attribute_'):
            attribute_name = x[0].replace('attribute_', '')
            attribute_param = \
                [str(x) for x in inspect.signature(x[1]).parameters.values()]
            if attribute_param[0] != 'tree': continue
            attribute_param = \
                list(filter(lambda x: x not in params_remove, attribute_param))
            dict_of_attributes[attribute_name] = attribute_param
    return dict_of_attributes

class Tree:
    """
    Abstract class for tree representations of images.

    Notes
    ----- 
    You should not instantiate class `Tree` directly, use `MaxTree` or
    `MinTree` instead.


    """
    def __init__(self, image, adjacency):
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

    def available_attributes(self=None):
        """
        Return a dictionary of available attributes and parameters.

        Returns
        -------
        dict_of_attributes : dict
            The names of available attributes and parameters required.
            The names are keys (str) and the parameters are values (list
            of str) of the dictionary.

        See Also
        --------
        get_attribute : Return the attribute values of the tree nodes.

        Notes
        -----
        The list of available attributes is generated dynamically. It is
        dependent of higra's installed version. For more details, please
        refer to `higra documentation
        <https://higra.readthedocs.io/en/stable/python/tree_attributes.html>`_
        according to the appropriate higra's version.

        Example
        -------
        >>> sap.Tree.available_attributes()
        {'area': ['vertex_area=None', 'leaf_graph=None'],
         'compactness': ['area=None', 'contour_length=None', ...],
         ...
         'volume': ['altitudes', 'area=None']}

        """
        return available_attributes()

    def get_attribute(self, attribute_name, **kwargs):
        """
        Get attribute values of the tree nodes.

        Parameters
        ------
        attribute_name : str
            Name of the attribute (e.g. 'area', 'compactness', ...)

        Returns
        -------
        attribute_values: ndarray
            The values of attribute for each nodes.

        See Also
        --------
        available_attributes : Return the list of available attributes.

        Notes
        -----
        Some attributes require additional parameters. Please refer to
        `available_attributes`. If not stated, some additional
        parameters are automatically deducted. These deducted parameters
        are 'altitudes' and 'vertex_weights'.

        The available attributes depends of higra's installed version.
        For further details Please refer to `higra documentation
        <https://higra.readthedocs.io/en/stable/python/tree_attributes.html>`_
        according to the appropriate higra's version.

        Examples
        --------
        >>> image = np.arange(20 * 50).reshape(20, 50)
        >>> t = sap.MaxTree(image)
        >>> t.get_attribute('area')
        array([   1.,    1.,    1., ...,  998.,  999., 1000.])

        """
        compute = getattr(hg, 'attribute_' + attribute_name)

        if 'altitudes' in inspect.signature(compute).parameters:
            kwargs['altitudes'] = kwargs.get('altitudes', self._alt)

        if 'vertex_weights' in inspect.signature(compute).parameters:
            kwargs['vertex_weights'] = kwargs.get('vertex_weights', self._image)

        return compute(self._tree, **kwargs)

    def reconstruct(self, deleted_nodes=None):
        """
        Return the reconstructed image according to deleted nodes.

        Parameters
        ----------
        deleted_nodes : ndarray or boolean
            Boolean array of node to delete with `len(deleted_nodes) ==
            tree.num_nodes()`.
        
        Returns
        -------
        filtered_image : ndarray
            The reconstructed image.

        """
        if isinstance(deleted_nodes, bool):
            deleted_nodes = np.array((deleted_nodes,) * self.num_nodes())

        return hg.reconstruct_leaf_data(self._tree, self._alt, deleted_nodes)
    
    def num_nodes(self):
        """
        Return the node count of the tree.

        Returns
        -------
        nodes_count : int
            The node count of the tree.

        """
        return self._tree.num_vertices()

class MaxTree(Tree):
    """
    Max tree class, the local maxima values of the image are in leafs.

    Parameters
    ----------
    image : ndarray
        The image to be represented by the tree structure.
    adjacency : int
        The pixel connectivity to use during the tree creation. It
        determines the number of pixels to be taken into account in the
        neighborhood of each pixel. The allowed adjacency are 4 or 8.
        Default is 4.

    Notes
    -----
    Inherits all methods of `Tree` class.

    """
    def __init__(self, image, adjacency=4):
        super().__init__(image, adjacency)
        graph = self._get_adjacency_graph()
        self._tree, self._alt = hg.component_tree_max_tree(graph, image)


class MinTree(Tree):
    """
    Min tree class, the local minima values of the image are in leafs.

    Parameters
    ----------
    image : ndarray
        The image to be represented by the tree structure.
    adjacency : int
        The pixel connectivity to use during the tree creation. It
        determines the number of pixels to be taken into account in the
        neighborhood of each pixel. The allowed adjacency are 4 or 8.
        Default is 4.

    Notes
    -----
    Inherits all methods of `Tree` class.

    """
    def __init__(self, image, adjacency=4):
        super().__init__(image, adjacency)
        graph = self._get_adjacency_graph()
        self._tree, self._alt = hg.component_tree_min_tree(graph, image)
