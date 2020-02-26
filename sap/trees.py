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
import tempfile
from pathlib import Path

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

def save(file, tree):
    """Save a tree to a NumPy archive file.

    Parameters
    ----------
    file : str or pathlib.Path
        File to which the tree is saved.
    tree: Tree
        Tree to be saved.

    Examples
    --------

    >>> mt = sap.MaxTree(np.random.random((100,100)))
    >>> sap.save('tree.npz', mt)

    """
    tree_file = Path(tempfile.mkstemp()[1])
    graph_file = Path(tempfile.mkstemp()[1])
    # TODO: Remove _alt once higra fixed
    hg.save_tree(str(tree_file), tree._tree, {'_alt': tree._alt})
    hg.save_graph_pink(str(graph_file), tree._graph)

    with tree_file.open('rb') as f:
        tree_bytes = f.read()

    with graph_file.open('rb') as f:
        graph_bytes = f.read()

    tree_file.unlink()
    graph_file.unlink()

    data = tree.__dict__.copy()
    data['_tree'] = tree_bytes
    data['_graph'] = graph_bytes
    data['__class__'] = tree.__class__

    np.savez_compressed(file, **data)

def load(file):
    """Load a tree from a Higra tree file.

    Parameters
    ----------
    file : str or pathlib.Path
        File to which the tree is loaded.

    Examples
    --------

    >>> mt = sap.MaxTree(np.arange(10000).reshape(100,100))
    >>> sap.save('tree.npz', mt)

    >>> sap.load('tree.npz')
    MaxTree{num_nodes: 20000, image.shape: (100, 100), image.dtype: int64}

    """
    data = np.load(str(file), allow_pickle=True)

    payload = {}
    for f in data.files:
        payload[f] = data[f].item() if data[f].size == 1 else data[f]

    tree_file = Path(tempfile.mkstemp()[1])
    graph_file = Path(tempfile.mkstemp()[1])

    with tree_file.open('wb') as f:
        f.write(payload['_tree'])

    with graph_file.open('wb') as f:
        f.write(payload['_graph'])

    tree_cls = payload.pop('__class__')

    tree = tree_cls(None)
    tree.__dict__.update(payload)

    tree._tree = hg.read_tree(str(tree_file))[0]
    tree._graph = hg.read_graph_pink(str(graph_file))[0]

    tree_file.unlink()
    graph_file.unlink()

    return tree


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

        if image is not None:
            self._graph = self._get_adjacency_graph()
            self._construct()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__ + \
           '{{num_nodes: {}, image.shape: {}, image.dtype: {}}}'.format(
           self.num_nodes(), self._image.shape, self._image.dtype)

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

    def _construct(self):
        self._tree, self._alt = hg.component_tree_max_tree(self._graph, self._image)


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

    def _construct(self):
        self._tree, self._alt = hg.component_tree_min_tree(self._graph, self._image)

class TosTree(Tree):
    """
    Tree of shapes, the local maxima values of the image are in leafs.

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
    Inherits all the methods of `Tree` class.

    Todo
    ----
    - take into account adjacency

    """
    def __init__(self, image, adjacency=4):
        super().__init__(image, adjacency)

    def _construct(self):
        self._tree, self._alt = hg.component_tree_tree_of_shapes_image2d(self._image)

