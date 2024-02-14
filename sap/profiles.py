#!/usr/bin/env python
# file profiles.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 13 déc. 2019
"""
Profiles
========

This submodule contains the attribute profiles related classes.

Example
-------

>>> import sap
>>> import numpy as np

>>> image = np.arange(5*5).reshape(5, 5)

Create the attribute profiles (AP) of `image` based on area attribute
and three thresholds.

>>> aps = sap.attribute_profiles(image, {'area': [10, 100, 1000]})
>>> aps.vectorize()
[[...]]

Create the extended AP of `image` based on area compactness and volume
attributes.

>>> attributes = {'compactness': [.1, .5, .7], 'volume': [10, 100]}
>>> eaps = sap.attribute_profiles(image, attributes)
>>> eaps.vectorize()
[[...]]

Concatenation of profiles to create complex extended profiles.

>>> profiles = sap.attribute_profiles(image, {'area': [10, 100]}) \\
...            + sap.feature_profiles(image, {'compactness': [.3, .7]}) \\
...            + sap.self_dual_attribute_profiles(image, {'height': [5, 15]})
Profiles[{'attribute': 'area',
  'filtering rule': 'direct',
  'image': -7518820387991786804,
  'name': 'attribute profiles',
  'out feature': 'altitude',
  'profiles': [{'operation': 'thinning', 'threshold': 100},
               {'operation': 'thinning', 'threshold': 10},
               {'operation': 'copy feature altitude'},
               {'operation': 'thickening', 'threshold': 10},
               {'operation': 'thickening', 'threshold': 100}]},
 {'attribute': 'compactness',
  'filtering rule': 'direct',
  'image': -7518820387991786804,
  'name': 'feature profiles',
  'out feature': 'compactness',
  'profiles': [{'operation': 'thinning', 'threshold': 0.7},
               {'operation': 'thinning', 'threshold': 0.3},
               {'operation': 'copy feature compactness'},
               {'operation': 'thickening', 'threshold': 0.3},
               {'operation': 'thickening', 'threshold': 0.7}]},
 {'attribute': 'height',
  'filtering rule': 'direct',
  'image': -7518820387991786804,
  'name': 'self dual attribute profiles',
  'out feature': 'altitude',
  'profiles': [{'operation': 'copy feature altitude'},
               {'operation': 'sd filtering', 'threshold': 5},
               {'operation': 'sd filtering', 'threshold': 15}]}]


"""

import numpy as np
from tqdm.auto import tqdm
from pprint import pformat
from matplotlib import pyplot as plt
from pathlib import Path
from . import trees
from .utils import *

class Profiles:
    """
    Base class for profiles.

    Parameters
    ----------
    data : list of ndarray
        List of ndarray representing profiles grouped by image or attribute
        filtering.
    description : list of dict
        List of dictionary containing the metadata of the profiles.

    """
    def __init__(self, data, description):
        if len(data) != len(description):
            raise AttributeError('Data and description missmatch.')

        self.data = data if len(data) > 1 else data[0]
        self.description = description if len(data) > 1 else description[0]
        self.len = len(data)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__ + pformat(self.description)

    def __iter__(self):
        if self.len == 1:
            yield self
            return

        for data, description in zip(self.data, self.description):
            yield Profiles([data], [description])

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        if self.len == 1 and key == 0:
            return self

        return Profiles([self.data[key]], [self.description[key]])

    def __add__(self, other):
        return concatenate((self, other))

    def diff(self):
        """Compute the differential of profiles.

        Refer to :func:`differential` for full documentation.

        Returns
        -------
        differential : Profiles
            The processed differential profiles.

        """
        return differential(self)

    def lf(self, local_feature=(np.mean, np.std), patch_size=7):
        """lf(self, local_feature=(np.mean, np.std), patch_size=7)
        Compute the local features of profiles

        Refer to :func:`local_features` for full documentation.

        local_feature : function or tuple of functions
            The function(s) to describe the local patches.
        patch_size : int
            The size of the patches.

        Returns
        -------
        local_features : Profiles
            The local features of ``profiles``.

        """
        return local_features(self, local_feature, patch_size)

    def vectorize(self):
        """Return the vectors of the profiles.

        Refer to :func:`vectorize` for full documentation.

        Returns
        -------
        vectors : numpy.ndarray
            The vectors of the profiles.

        See Also
        --------
        vectorize : equivalent function.

        """
        return vectorize(self)

    def strip(self, condition):
        """strip(lambda x: x['operation'] != 'thinning')

        Remove profiles according to condition. Iteration is done on
        profiles description.

        Refer to :func:`strip_profiles` for full documentation.

        Parameters
        ----------
        condition : function
            The function (or lambda function) to use on profiles description
            to filter the profiles.

        Returns
        -------
        new_profiles : Profiles
            Filtered profiles.

        See Also
        --------
        strip_profiles : equivalent function
        """
        return strip_profiles(condition, self)

    def strip_copy(self):
        """Remove all the copied images in profiles.

        Refer to :func:`strip_profiles_copy` for full documentation.

        Parameters
        ----------
        profiles : Profiles
            The profiles to strip on the copied images.

        Returns
        -------
        new_profiles : Profiles
            Copy of profiles without copied image.

        See Also
        --------
        strip_profiles_copy : equivalent function

        """
        return strip_profiles_copy(self)

def create_profiles(tree, attribute, out_feature='altitude',
        filtering_rule='direct', profiles_name='unknow'):
    """
    Compute the profiles of an images. Generic function.

    Parameters
    ----------
    image : ndarray
        The image to be profiled.
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, iterable of thresholds).
    tree_type : sap.trees.Tree, serie of sap.trees.Tree
        Tree or pair of tree for non dual filtering (e.g. min-tree and
        max-tree for attribute profiles).
    adjacency : int, optional
        Adjacency used for the tree construction. Default is 4.
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display. If not set, the name is replaced by the hash of the
        image.
    out_feature : str or list, optional
        Out feature of the profiles. Can be 'altitude' (default), 'same'
        or a list of feature. If 'same' then out feature of the profiles
        match the filtering attribute. Refer to :func:`feature_profiles`
        and :func:`self_dual_feature_profiles` for more details.
    filtering_rule : str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.
    profiles_name : str, optional
        Name of the profiles (e.g. `'attribute profiles'`).

    Todo
    ----
    out_feature takes a list of features.

    Example
    -------
    >>> image = np.arange(5*5).reshape(5, 5)

    >>> sap.create_profiles(image, {'area': [5, 10]},
    ...                     (sap.MinTree, sap.MaxTree))
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'image': -7518820387991786804,
     'name': 'unknow',
     'out feature': 'altitude',
     'profiles': [{'operation': 'thinning', 'threshold': 10},
                  {'operation': 'thinning', 'threshold': 5},
                  {'operation': 'copy feature altitude'},
                  {'operation': 'thickening', 'threshold': 5},
                  {'operation': 'thickening', 'threshold': 10}]}

    """
    data = []
    description = []

    # Create Trees
    try:
        if isinstance(tree, trees.Tree):
            # Dual tree
            ndual = False
            thinning_tree = None
            thickening_tree = tree
        else:
            # Non dual trees
            ndual = True
            thinning_tree = tree[0]
            thickening_tree = tree[1]
    except:
        raise TypeError('Parameter tree_type must be a tuple or a single' \
                ' instance of Tree, not {}'.format(tree))

    out_features = (out_feature, ) if isinstance(out_feature, str) else out_feature

    iter_count = (sum(len(x) for x in attribute.values()) * (1 + ndual) + \
            len(attribute)) * len(out_features)
    ttq = tqdm(desc='Total', total=iter_count)
    for att, thresholds in attribute.items():
        tq = tqdm(total=(len(thresholds) * (1 + ndual) + 1) * len(out_features), desc=att)

        for out_feature in out_features:
            profiles = []; profiles_description = []
            of = att if out_feature == 'same' else out_feature

            if ndual:
                # thinning
                prof, desc = _compute_profiles(thinning_tree, att,
                            thresholds[::-1], (ttq, tq), of, filtering_rule)
                profiles += prof
                profiles_description += desc

            # Origin
            tq.update(); ttq.update()
            profiles += [thickening_tree.reconstruct(feature=of)]
            profiles_description += [{'operation': 'copy feature {}'.format(of)}]

            # thickening
            prof, desc = _compute_profiles(thickening_tree, att, thresholds,
                                           (ttq, tq), of, filtering_rule)
            profiles += prof
            profiles_description += desc


            data += [np.stack(profiles)]
            description += [{'tree': thickening_tree.get_params(),
                             'name': profiles_name,
                             'attribute': att,
                             'profiles': profiles_description,
                             'filtering rule': filtering_rule,
                             'out feature': of}]
        tq.close()
    ttq.close()

    return Profiles(data, description)

def _compute_profiles(tree, attribute, thresholds, tqs,
        feature='altitude', rule='direct'):
    data = []
    desc = []

    for t in thresholds:
        for tq in tqs: tq.update()
        desc += [{'operation': tree.operation_name, 'threshold': t}]
        deleted_nodes = tree.get_attribute(attribute) < t
        data += [tree.reconstruct(deleted_nodes, feature, rule)]

    return data, desc

def attribute_profiles(image, attribute, adjacency=4, image_name=None,
        filtering_rule='direct'):
    """
    Compute the attribute profiles of an image.

    Parameters
    ----------
    image : ndarray
        The image
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, number).
    adjacency : int
        Adjacency used for the tree construction. Default is 4.
    image_name : str
        The name of the image (optional). Useful to track filtering
        process and display. If not set, the name is replaced by the
        hash of the image.
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.


    Examples
    --------

    >>> image = np.arange(5*5).reshape(5,5)

    >>> sap.attribute_profiles(image, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'image': -7518820387991786804,
     'name': 'attribute profiles',
     'out feature': 'altitude',
     'profiles': [{'operation': 'thinning', 'threshold': 100},
                  {'operation': 'thinning', 'threshold': 10},
                  {'operation': 'copy feature altitude'},
                  {'operation': 'thickening', 'threshold': 10},
                  {'operation': 'thickening', 'threshold': 100}]}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.

    """
    maxt = trees.MaxTree(image, adjacency, image_name)
    mint = trees.MinTree(image, adjacency, image_name)

    return create_profiles((mint, maxt), attribute, 'altitude',
            filtering_rule, 'attribute profiles')

def self_dual_attribute_profiles(image, attribute, adjacency=4,
        image_name=None, filtering_rule='direct'):
    """
    Compute the self dual attribute profiles of an image.

    Parameters
    ----------
    image : ndarray
        The image
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, number).
    adjacency : int
        Adjacency used for the tree construction. Default is 4.
    image_name : str
        The name of the image (optional). Useful to track filtering
        process and display. If not set, the name is replaced by the
        hash of the image.
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Examples
    --------

    >>> image = np.arange(5*5).reshape(5,5)

    >>> sap.self_dual_attribute_profiles(image, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'image': -7518820387991786804,
     'name': 'self dual attribute profiles',
     'out feature': 'altitude',
     'profiles': [{'operation': 'copy feature altitude'},
                  {'operation': 'sd filtering', 'threshold': 10},
                  {'operation': 'sd filtering', 'threshold': 100}]}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.
    attribute_profiles : other profiles.

    """
    tost = trees.TosTree(image, adjacency, image_name)
    return create_profiles(tost, attribute, 'altitude',
                filtering_rule, 'self dual attribute profiles')

def self_dual_feature_profiles(image, attribute, adjacency=4, image_name=None,
        out_feature='same', filtering_rule='direct'):
    """
    Compute the self dual features profiles of an image.

    Parameters
    ----------
    image : ndarray
        The image
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, number).
    adjacency : int
        Adjacency used for the tree construction. Default is 4.
    image_name : str
        The name of the image (optional). Useful to track filtering
        process and display. If not set, the name is replaced by the
        hash of the image.
    out_feature : str or list, optional
        Out feature of the profiles. Can be 'altitude' (default), 'same'
        or a list of feature. If 'same' then out feature of the profiles
        match the filtering attribute.
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Examples
    --------

    >>> image = np.arange(5*5).reshape(5,5)

    >>> sap.self_dual_feature_profiles(image, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'image': -7518820387991786804,
     'name': 'self dual feature profiles',
     'out feature': 'area',
     'profiles': [{'operation': 'copy feature area'},
                  {'operation': 'sd filtering', 'threshold': 10},
                  {'operation': 'sd filtering', 'threshold': 100}]}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.
    attribute_profiles : other profiles.

    """
    tost = trees.TosTree(image, adjacency, image_name)
    return create_profiles(tost, attribute, out_feature, filtering_rule,
                           'self dual feature profiles')

def feature_profiles(image, attribute, adjacency=4, image_name=None,
        out_feature='same', filtering_rule='direct'):
    """
    Compute the feature profiles of an image.

    Parameters
    ----------
    image : ndarray
        The image
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, number).
    adjacency : int
        Adjacency used for the tree construction. Default is 4.
    image_name : str
        The name of the image (optional). Useful to track filtering
        process and display. If not set, the name is replaced by the
        hash of the image.
    out_feature : str or list, optional
        Out feature of the profiles. Can be 'altitude' (default), 'same'
        or a list of feature. If 'same' then out feature of the profiles
        match the filtering attribute.
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Examples
    --------

    >>> image = np.arange(5*5).reshape(5,5)

    >>> sap.feature_profiles(image, {'area': [5, 10]})
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'image': -7518820387991786804,
     'name': 'feature profiles',
     'out feature': 'area',
     'profiles': [{'operation': 'thinning', 'threshold': 10},
                  {'operation': 'thinning', 'threshold': 5},
                  {'operation': 'copy feature area'},
                  {'operation': 'thickening', 'threshold': 5},
                  {'operation': 'thickening', 'threshold': 10}]}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.
    attribute_profiles : other profiles.

    """
    maxt = trees.MaxTree(image, adjacency, image_name)
    mint = trees.MinTree(image, adjacency, image_name)

    return create_profiles((mint, maxt), attribute,
               out_feature, filtering_rule, 'feature profiles')

def alpha_profiles(image, attribute, adjacency=4,
                   image_name=None, filtering_rule='direct'):
    """
    Compute the alpha profiles of an image.

    Parameters
    ----------
    image : ndarray
        The image
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, number).
    adjacency : int
        Adjacency used for the tree construction. Default is 4.
    image_name : str
        The name of the image (optional). Useful to track filtering
        process and display. If not set, the name is replaced by the
        hash of the image.
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Examples
    --------

    >>> image = np.arange(5 * 5).reshape(5, 5)
    >>> sap.alpha_profiles(image, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'name': 'alpha profiles',
     'out feature': 'altitude',
     'profiles': [{'operation': 'copy feature altitude'},
                  {'operation': 'alpha filtering', 'threshold': 10},
                  {'operation': 'alpha filtering', 'threshold': 100}],
     'tree': {'adjacency': 4, 'image_hash': '44f17c0f', 'image_name': None}}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.

    """
    atree = trees.AlphaTree(image, adjacency, image_name)
    return create_profiles(atree, attribute, 'altitude', filtering_rule, 'alpha profiles')

def omega_profiles(image, attribute, adjacency=4,
                   image_name=None, filtering_rule='direct'):
    """
    Compute the omega profiles of an image.

    Parameters
    ----------
    image : ndarray
        The image
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, number).
    adjacency : int
        Adjacency used for the tree construction. Default is 4.
    image_name : str
        The name of the image (optional). Useful to track filtering
        process and display. If not set, the name is replaced by the
        hash of the image.
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Examples
    --------

    >>> image = np.arange(5 * 5).reshape(5, 5)
    >>> sap.omega_profiles(image, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'name': 'omega profiles',
     'out feature': 'altitude',
     'profiles': [{'operation': 'copy feature altitude'},
                  {'operation': '(ω) filtering', 'threshold': 10},
                  {'operation': '(ω) filtering', 'threshold': 100}],
     'tree': {'adjacency': 4, 'image_hash': '44f17c0f', 'image_name': None}}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.

    """
    otree = trees.OmegaTree(image, adjacency, image_name)
    return create_profiles(otree, attribute, 'altitude', filtering_rule, 'omega profiles')

def watershed_profiles(image, attribute, markers=None, adjacency=4, image_name=None, filtering_rule='direct', weight_function = 'L1', watershed_attribute='area'):
    """
    Compute the watershed profiles of an image.

    Parameters
    ----------
    image : ndarray
        The image
    attribute : dict
        Dictionary of attribute (as key, str) with according thresholds
        (as values, number).
    markers : 2D ndarray of same dimension as 'image'  
        Prior-knowledge to be combined to the image gradient before the
        construction of the hierarchical watershed.  If ``None``, an
        ndarray of ones is used, the result will be equivalent of not
        using markers at all.
    adjacency : int
        Adjacency used for the tree construction. Default is 4.
    image_name : str
        The name of the image (optional). Useful to track filtering
        process and display. If not set, the name is replaced by the
        hash of the image.
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.
    weight_function : str
        The function used to compute dissimilarity between neighbour
        pixels. Default is 'L1' (absolute different between pixel
        values).
    watershed_attribute : str
        The criteria used to guide the contruction of the hierarchical
        watershed. The allowed criteria are : 'area', 'volume',
        'dynamics' and 'parents'.

    Examples
    --------

    >>> image = np.arange(5 * 5).reshape(5, 5)
    >>> markers = np.ones((5,5))
    >>> sap.watershed_profiles(image, markers, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'filtering rule': 'direct',
     'name': 'watershed profiles',
     'out feature': 'altitude',
     'profiles': [{'operation': 'copy feature altitude'},
                  {'operation': 'watershed filtering', 'threshold': 10},
                  {'operation': 'watershed filtering', 'threshold': 100}],
     'tree': {'adjacency': 4, 'image_hash': '44f17c0f', 'image_name': None}}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.

    """
    atree = trees.WatershedTree(image, markers, adjacency, image_name, weight_function, watershed_attribute)
    return create_profiles(atree, attribute, 'altitude', filtering_rule, 'watershed profiles')

def _show_profiles(profiles, height=None, fname=None, **kwargs):
    assert len(profiles) == 1, 'Show profile only for one attribute at a time.'

    # Set vmin and vmax if not set
    if not 'vmin' in kwargs:
        kwargs['vmin'] = profiles.data.min()
    if not 'vmax' in kwargs:
        kwargs['vmax'] = profiles.data.max()

    if height is not None:
        plt.figure(figsize=_figsize(profiles, height))

    suptitle = '{} - {}'.format(profiles.description['tree']['image_name'], profiles.description['attribute'])

    for i, (im, profile) in enumerate(zip(profiles.data, profiles.description['profiles'])):
        plt.subplot(1, len(profiles.data), i+1)
        plt.imshow(im, **kwargs)
        plt.title(_title(profile))

    plt.tight_layout()
    plt.suptitle(suptitle)

    if fname:
        fname = Path(fname)
        fn = fname.parent / Path(fname.stem + '_{}'.format(profiles.description['attribute']) + fname.suffix)
        plt.savefig(fn)

def show_all_profiles(profiles, attribute=None, image=None, height=None, fname=None, **kwargs):
    """Display profiles with matplotlib.

    Parameters
    ----------
    profiles : sap.profiles.Profiles
        The profiles to display.
    attribute : sring, optional
        Name of attribute to display. By default display all the
        attributes contained in profiles.
    image : string, optional
        Name of the image to display. By default display the profiles of
        all images.
    height : scalar, optional, default: None
        Height of the figure in inches. Automatically adjust the size of
        the figure to display correctly the profiles and the title with
        matplot.
    fname : str or PathLike, optional
        If set, the file path to save the figure. The attribute name is
        automatically inserted in the file name.

    See Also
    --------
    show_profiles : Display a profiles stack.

    Notes
    -----
    This is a utility function to call recursively `show_profiles`.
    Attribute and image filters are available to filter the profiles to
    display.

    """

    # Filter profiles according to attribute if attribute is set
    if attribute:
        profiles = filter(lambda x: x.description['attribute'] == attribute, profiles)
    # Same for image
    if image:
        profiles = filter(lambda x: x.description['tree']['image_name'] == image, profiles)

    for p in profiles:
        show_profiles(p, height, fname, **kwargs)

def strip_profiles_copy(profiles):
    """Remove all the copied images in profiles.

    Copy are the original images where profiles are computed on.

    Parameters
    ----------
    profiles : Profiles
        The profiles to strip on the copied images.

    Returns
    -------
    new_profiles : Profiles
        Copy of profiles without copied image.

    See Also
    --------
    sap.strip_profiles : Filter profiles according to condition.

    """
    return strip_profiles(lambda x: x['operation'] == 'copy', profiles)

def strip_profiles(condition, profiles):
    """strip_profiles(lambda x: x['operation'] != 'thinning', profiles)

    Remove profiles according to condition. Iteration is done on
    profiles description (see Notes).

    Parameters
    ----------
    condition : function
        The function (or lambda function) to use on profiles description
        to filter the profiles.
    profiles : Profiles
        The profiles to filter.

    Returns
    -------
    new_profiles : Profiles
        Filtered profiles.

    Notes
    -----

    The condition is tested on the description of each profiles.
    Considering this stack:

    >>> aps
    Profiles{'attribute': 'area',
     'image': -8884649894275650052,
     'profiles': [{'operation': 'thinning', 'threshold': 1000},
                  {'operation': 'thinning', 'threshold': 100},
                  {'operation': 'thinning', 'threshold': 10},
                  {'operation': 'copy'},
                  {'operation': 'thickening', 'threshold': 10},
                  {'operation': 'thickening', 'threshold': 100},
                  {'operation': 'thickening', 'threshold': 1000}]}

    The condition function is tested on each item of the list
    ``'profiles'``.

    See Also
    --------
    Profiles.strip : Remove profiles based on condition.

    Examples
    --------

    Strip profiles depending on thresholds level:

    >>> image = np.random.random((100, 100))
    >>> aps = sap.attribute_profiles(image, {'area': [10, 100, 1000]})

    >>> sap.strip_profiles(lambda x: 'threshold' in x and x['threshold'] > 20, aps)
    Profiles{'attribute': 'area',
     'image': 2376333419322655105,
     'profiles': [{'operation': 'thinning', 'threshold': 10},
                  {'operation': 'copy'},
                  {'operation': 'thickening', 'threshold': 10}]}

    Strip profiles depending on operation:

    >>> sap.strip_profiles(lambda x: x['operation'] == 'thinning', aps)
    Profiles{'attribute': 'area',
     'image': 2376333419322655105,
     'profiles': [{'operation': 'copy'},
                  {'operation': 'thickening', 'threshold': 10},
                  {'operation': 'thickening', 'threshold': 100},
                  {'operation': 'thickening', 'threshold': 1000}]}

    """
    new_profiles = []
    for ap in profiles:
        # Process the profile filter
        prof_filter = [not condition(x) for x in ap.description['profiles']]

        # Create filtered description
        new_desc = ap.description.copy()
        new_desc['profiles'] = [p for p, f in zip(ap.description['profiles'], prof_filter) if f]

        # Filter the new data
        new_data = ap.data[prof_filter]

        new_profiles += [Profiles([new_data], [new_desc])]

    return concatenate(new_profiles)

def differential(profiles):
    """Compute the differential of profiles.

    Parameters
    ----------
    profiles : Profiles
        Attribute profiles or other profiles to process the
        differential on.

    Returns
    -------
    differential : Profiles
        The processed differential profiles.

    """
    new_data = []
    new_desc = []

    for p in profiles:
        new_data += [p.data[:-1] - p.data[1:]]
        new_desc += [p.description.copy()]
        d = new_desc[-1]
        d['profiles'] = [{'operation': 'differential',
                          'profiles': [x, y]} for x, y in zip(d['profiles'][:-1],
                                                              d['profiles'][1:])]
    return Profiles(new_data, new_desc)

def local_features(profiles, local_feature=(np.mean, np.std), patch_size=7):
    """local_features(profiles, local_feature=(np.mean, np.std), patch_size=7)
    Compute the local features of profiles

    Parameters
    ----------
    profiles : Profiles
        Input Profiles.
    local_feature : function or tuple of functions
        The function(s) to describe the local patches.
    patch_size : int
        The size of the patches.

    Returns
    -------
    local_feature : Profiles
        The local features of ``profiles``.

    """
    try:
        iter(local_feature)
    except TypeError:
        local_feature = (local_feature,)

    new_data = []
    new_desc = []

    for p in profiles:
        for f in local_feature:
            nd = [local_patch_f(d, patch_size, f) for d in p.data]
            new_data += [np.array(nd)]
            new_desc += [p.description.copy()]
            d = new_desc[-1]
            d['profiles'] = [{'operation': 'local feature',
                              'function': f.__name__,
                              'patch size': patch_size,
                              'profile': x} for x in d['profiles']]
    return Profiles(new_data, new_desc)

def show_profiles(profiles, height=None, fname=None, **kwargs):
    """Display a profiles stack with matplotlib.

    Parameters
    ----------
    profiles : Profiles
        The profiles to display. Can be only of length 1.
    height : scalar, optional, default: None
        Height of the figure in inches. Automatically adjust the size of
        the figure to display correctly the profiles and the title with
        matplot.
    fname : str or PathLike, optional
        If set, the file path to save the figure. The attribute name is
        automatically inserted in the file name.

    See Also
    --------
    show_profiles_all : Display several profiles at once.

    """
    _show_profiles(profiles, height, fname, **kwargs)

def _figsize(profiles, height):
    """Compute size of fig given height."""
    shape = profiles.data.shape[1:]
    count = profiles.data.shape[0]
    hw_ratio = shape[1] / shape[0]
    width = height * hw_ratio * count
    return (width, 1.1 * height)

def _title(profile):
    """Process a title of a fig."""
    if profile['operation'] == 'differential':
        p1, p2 = profile['profiles']
        return 'differential ({}, {})'.format(_title(p1), _title(p2))
    elif profile['operation'] == 'local feature':
        p = profile['profile']
        return 'local feature {} ({})'.format(profile['function'], _title(p))
    else:
        return ' '.join([str(x) for x in profile.values()])

def concatenate(sequence):
    """concatenate((profiles_1, profiles_2, ...))

    Concatenate a sequence of profiles.

    Parameters
    ----------
    sequence : sequence of Profiles
        The sequence of profiles to concatenate.

    Returns
    -------
    profiles : Profiles
        The concatenated profiles.

    Notes
    -----
    This function can be replaced by operator ``+``.


    Examples
    --------

    Create two APs:

    >>> aps_a = sap.attribute_profiles(image, {'area': [10, 100]})
    >>> aps_b = sap.attribute_profiles(image, {'compactness': [.1, .5]})

    And concatenate:

    >>> aps = sap.concatenate((aps_a, aps_b))

    That is equivalent to:

    >>> aps = aps_a + aps_b

    >>> len(aps) == len(aps_a) + len(aps_b)
    True

    """

    return Profiles([x.data for y in sequence for x in y],
                    [x.description for y in sequence for x in y])

def vectorize(profiles):
    """Return the classification vectors of the profiles.

    Parameters
    ----------
    profiles : Profiles
        Profiles on which process the vectors.

    Returns
    -------
    vectors : numpy.ndarray
        The vectors of the profiles.

    See Also
    --------
    Profiles.vectorize : get the vectors of profiles.

    Example
    -------

    >>> image = np.random.random((100, 100))
    >>> aps = sap.attribute_profiles(image, {'area': [10, 100]})

    >>> vectors = sap.vectorize(aps)
    >>> vectors.shape
    (5, 100, 100)

    """

    return np.concatenate([p.data for p in profiles])

