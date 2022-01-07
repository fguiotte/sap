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

import inspect
from pathlib import Path
import tempfile
from pprint import pformat
import functools

import numpy as np
import higra as hg

from .utils import *

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
    You should not instantiate class :class:`Tree` directly, use `MaxTree` or
    `MinTree` instead.

    """
    def __init__(self, image, adjacency, image_name=None, operation_name='non def'):
        if self.__class__ == Tree:
            raise TypeError('Do not instantiate directly abstract class Tree.')

        self._image_name = image_name
        self._image_hash = ndarray_hash(image) if image is not None else None
        self._adjacency = adjacency
        self._image = image
        self.operation_name = operation_name

        if image is not None:
            self._graph = self._get_adjacency_graph()
            self._construct()

    def __str__(self):
        return str(self.__repr__())

    def get_params(self):
        return {'image_name': self._image_name,
                'image_hash': self._image_hash,
                'adjacency': self._adjacency}

    def __repr__(self):
        if hasattr(self, '_tree'):
            rep = self.get_params()
            rep.update({'num_nodes': self.num_nodes(),
                    'image.shape': self._image.shape,
                    'image.dtype': self._image.dtype})
        else:
            rep = {}
        return self.__class__.__name__ + pformat(rep)

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
        Tree.get_attribute : Return the attribute values of the tree nodes.

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

    def _get_higra_attribute_func(self, attribute_name):
        try:
            attribute_func = getattr(hg, 'attribute_' + attribute_name)
        except AttributeError:
            raise ValueError(f'Wrong attribute or out feature: \'{attribute_name}\'')

        return attribute_func

    def _set_params_on_higra_attribute_func(self, attribute_func, **kwargs):
        kwargs = {} if kwargs is None else kwargs

        if 'altitudes' in inspect.signature(attribute_func).parameters:
            kwargs['altitudes'] = kwargs.get('altitudes', self._alt)

        if 'vertex_weights' in inspect.signature(attribute_func).parameters:
            kwargs['vertex_weights'] = kwargs.get('vertex_weights', self._image)
        return functools.partial(attribute_func, **kwargs)

    def _get_higra_attribute_func_with_default(self, attribute_name, **kwargs):
        attribute_func = self._get_higra_attribute_func(attribute_name)
        attribute_func = self._set_params_on_higra_attribute_func(attribute_func, **kwargs)

        return attribute_func


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
        compute = self._get_higra_attribute_func_with_default(attribute_name, **kwargs)

        return compute(self._tree)

    def reconstruct(self, deleted_nodes=None, feature='altitude',
                    filtering='direct'):
        """
        Return the reconstructed image according to deleted nodes.

        Parameters
        ----------
        deleted_nodes : ndarray or boolean, optional
            Boolean array of nodes to delete. The length of the array should be
            of same of node count.
        feature : str, optional
            The feature to be reconstructed. Can be any attribute of the
            tree (see :func:`available_attributes`). The default is
            `'altitude'`, the grey level of the node.
        filtering : str, optional
            The filtering rule to use. It can be 'direct', 'min', 'max' or
            'subtractive'. Default is 'direct'.

        Returns
        -------
        filtered_image : ndarray
            The reconstructed image.

        Examples
        --------
        >>> image = np.arange(5 * 5).reshape(5, 5)
        >>> mt = sap.MaxTree(image)

        >>> mt.reconstruct()
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])

        >>> area = mt.get_attribute('area')

        >>> mt.reconstruct(area > 10)
        array([[ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])


        """
        if isinstance(deleted_nodes, bool):
            deleted_nodes = np.array((deleted_nodes,) * self.num_nodes())
        elif deleted_nodes is None:
            deleted_nodes = np.zeros(self.num_nodes(), dtype=bool)

        feature_value = self._alt if feature == 'altitude' else \
                        self.get_attribute(feature) if isinstance(feature, str) \
                        else feature

        rules = {'direct': self._filtering_direct,
                 'min': self._filtering_min,
                 'max': self._filtering_max,
                 'subtractive': self._filtering_subtractive}

        feature_value, deleted_nodes = rules[filtering](feature_value, deleted_nodes)

        return hg.reconstruct_leaf_data(self._tree, feature_value, deleted_nodes)

    def _filtering_direct(self, feature_value, direct):
        deleted_nodes = direct.astype(bool)
        return feature_value, deleted_nodes

    def _filtering_min(self, feature_value, direct):
        deleted_nodes = hg.propagate_sequential(self._tree, direct,
                ~direct).astype(bool)
        return feature_value, deleted_nodes

    def _filtering_max(self, feature_value, direct):
        deleted_nodes = hg.accumulate_and_min_sequential(self._tree, direct,
                    np.ones(self._tree.num_leaves()),
                    hg.Accumulators.min).astype(bool)
        return feature_value, deleted_nodes

    def _filtering_subtractive(self, feature_value, direct):
        deleted_nodes = direct.astype(bool)
        delta = feature_value - feature_value[self._tree.parents()]
        delta[direct] = 0
        delta[self._tree.root()] = feature_value[self._tree.root()]
        feature_value = hg.propagate_sequential_and_accumulate(self._tree, delta,
                                                             hg.Accumulators.sum)
        return feature_value, deleted_nodes

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
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display.

    Notes
    -----
    Inherits all methods of :class:`Tree` class.

    """
    def __init__(self, image, adjacency=4, image_name=None):
        super().__init__(image, adjacency, image_name, 'thickening')

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
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display.

    Notes
    -----
    Inherits all methods of :class:`Tree` class.

    """
    def __init__(self, image, adjacency=4, image_name=None):
        super().__init__(image, adjacency, image_name, 'thinning')

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
        **Not implemented yet**, this parameter is set for compatibility
        with other tree constructors.
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display.

    Notes
    -----
    Inherits all the methods of :class:`Tree` class.

    Todo
    ----
    - take into account adjacency

    """
    def __init__(self, image, adjacency=4, image_name=None):
        super().__init__(image, adjacency, image_name, 'sd filtering')

    def _construct(self):
        self._tree, self._alt = hg.component_tree_tree_of_shapes_image2d(self._image)

class AlphaTree(Tree):
    """
    Alpha tree, partition the image depending of the weight between pixels.

    Parameters
    ----------
    image : ndarray
        The image to be represented by the tree structure.
    adjacency : int
        The pixel connectivity to use during the tree creation. It
        determines the number of pixels to be taken into account in the
        neighborhood of each pixel. The allowed adjacency are 4 or 8.
        Default is 4.
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display.
    weight_function : str or higra.WeightFunction
        The weight function to use during the construction of the tree.
        Can be 'L0', 'L1', 'L2', 'L2_squared', 'L_infinity', 'max',
        'min', 'mean' or a `higra.WeightFunction`. The default is
        'L1'.

    Notes
    -----
    Inherits all the methods of :class:`Tree` class.

    """
    def __init__(self, image, adjacency=4, image_name=None, weight_function='L1'):
        if isinstance(weight_function, str):
            try:
                self._weight_function = getattr(hg.WeightFunction, weight_function)
            except AttributeError:
                raise AttributeError('Wrong value \'{}\' for attribute' \
                ' weight_function'.format(weight_function))
        elif isinstance(weight_function, hg.higram.WeightFunction):
            self._weight_function = weight_function
        else:
            raise NotImplementedError('Unknow type \'{}\' for parameter' \
                    ' weight_function'.format(type(weight_function)))

        super().__init__(image, adjacency, image_name, 'alpha filtering')

    def _construct(self):
        weight = hg.weight_graph(self._graph, self._image, self._weight_function)
        self._tree, alt = hg.quasi_flat_zone_hierarchy(self._graph, weight)
        self._alt, self._variance = hg.attribute_gaussian_region_weights_model(self._tree, self._image)


class OmegaTree(Tree):
    """
    Partition the image depending of the constrained weight between pixels.

    Parameters
    ----------
    image : ndarray
        The image to be represented by the tree structure.
    adjacency : int
        The pixel connectivity to use during the tree creation. It
        determines the number of pixels to be taken into account in the
        neighborhood of each pixel. The allowed adjacency are 4 or 8.
        Default is 4.
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display.

    Notes
    -----
    Inherits all the methods of :class:`Tree` class.

    """
    def __init__(self, image, adjacency=4, image_name=None):
        super().__init__(image, adjacency, image_name, '(ω) filtering')

    def _construct(self):
        edge_weights = hg.weight_graph(self._graph, self._image, getattr(hg.WeightFunction, 'L1'))
        vertex_weights = hg.linearize_vertex_weights(self._image, self._graph)

        tree, alt = hg.quasi_flat_zone_hierarchy(self._graph, edge_weights)

        min_value = hg.accumulate_sequential(tree, vertex_weights, hg.Accumulators.min)
        max_value = hg.accumulate_sequential(tree, vertex_weights, hg.Accumulators.max)
        value_range = max_value - min_value

        range_parents = value_range[tree.parents()]
        violated_constraints = value_range >= range_parents
        self._tree, node_map = hg.simplify_tree(tree, violated_constraints)

        self._alt, self._variance = hg.attribute_gaussian_region_weights_model(self._tree, self._image)

class WatershedTree(Tree):
    """
    Construct a hierarchical watershed from the gradient of the input image.

    Parameters
    ----------
    image : 2D ndarray
        The image from which the gradient (an edge-weighted graph) is constructed.
    markers : 2D ndarray of same dimension as 'image'  
        Prior-knowledge to be combined to the image gradient before the
        construction of the hierarchical watershed. See notes.
    adjacency : int
        The pixel connectivity used to compute edge-weighted graph which
        represents the image gradient. The allowed adjacency are 4 or 8.
        Default is 4.
    image_name : str, optional
        The name of the image.
    weight_function : str
        The function used to compute dissimilarity between neighbour
        pixels. Default is 'L1' (absolute different between pixel
        values).
    watershed_attribute : str
        The criteria used to guide the construction of the hierarchical
        watershed. The allowed criteria are : 'area', 'volume',
        'dynamics' and 'parents'.

    Notes
    -----
    Inherits all the methods of :class:`Tree` class.

    The :attr:`markers` parameter is prior-knowledge to be combined to
    the image gradient before the construction of the hierarchical
    watershed. The method is described in :

        Maia, Deise Santana, Minh-Tan Pham, and Sébastien Lefèvre.
        `Watershed-based attribute profiles for pixel classification of remote sensing data <https://hal.archives-ouvertes.fr/hal-03199313>`_.
        International Conference on Discrete Geometry and Mathematical Morphology.
        Springer, Cham, 2021.

    We expect the markers to be a gray-scale image in which dark and
    homogeneous regions have the highest probability of belonging to the
    same catchment basins.  If :attr:`markers` is ``None``, it is
    replaced by an ndarray of ones, the result will be equivalent of not
    using markers at all.

    """
    def __init__(self, image, markers=None, adjacency=4, image_name=None, weight_function='L1', watershed_attribute='area'):
        self._watershed_attribute = watershed_attribute

        if isinstance(weight_function, str):
            try:
                self._weight_function = getattr(hg.WeightFunction, weight_function)
            except AttributeError:
                raise AttributeError('Wrong value \'{}\' for attribute'
                ' weight_function'.format(weight_function))

        elif isinstance(weight_function, hg.higram.WeightFunction):
            self._weight_function = weight_function
        else:
            raise NotImplementedError(
                    'Unknow type \'{}\' for parameter'
                    ' weight_function'.format(type(weight_function)))

        self._markers = np.ones_like(image) if markers is None else markers
        super().__init__(image, adjacency, image_name, 'watershed filtering')

    def _construct(self):
        markers_gradient = hg.weight_graph(self._graph, self._markers, hg.WeightFunction.max)
        weight = hg.weight_graph(self._graph, self._image, self._weight_function)
        weight *= markers_gradient

        ws_hierachies = {
                'area': hg.watershed_hierarchy_by_area,
                'dynamics': hg.watershed_hierarchy_by_dynamics,
                'volume': hg.watershed_hierarchy_by_volume,
                'parents': hg.watershed_hierarchy_by_number_of_parents,
            }

        self._tree, alt = ws_hierachies[self._watershed_attribute](
                self._graph, weight)

        # Node represented by the average gray level inside a node
        self._alt, self._variance = hg.attribute_gaussian_region_weights_model(self._tree, self._image)
