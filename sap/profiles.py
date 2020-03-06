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

"""

import numpy as np
from tqdm.auto import tqdm
from pprint import pformat
from matplotlib import pyplot as plt
from pathlib import Path
from . import trees

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
        return 'Profiles' + pformat(self.description)

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

        Returns
        -------
        differential : Profiles
            The processed differential profiles.

        """
        return differential(self)

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

def create_profiles(image, attribute, tree_type, operation=None,
        adjacency=4, image_name=None, out_feature='altitude',
        filtering_rule='direct'):
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
    operation : str or iterable of str
        Name or names of the filtering processed by tree_type. Must
        match tree_type count.
    adjacency : int, optional
        Adjacency used for the tree construction. Default is 4.
    image_name : str, optional
        The name of the image Useful to track filtering process and
        display. If not set, the name is replaced by the hash of the
        image.
    out_feature: str, optional
        Out feature of the profiles. Can be 'altitude' (default) or
        'same' so that out feature of the profiles match the filtering
        attribute (cf.  :func:`feature_profiles` and
        :func:`self_dual_feature_profiles`).
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Todo
    ----
    out_feature takes a list of features.

    Example
    -------
    >>> image = np.arange(5*5).reshape(5, 5)

    >>> sap.create_profiles(image, {'area': [5, 10]},
    ...    (sap.MinTree, sap.MaxTree), ('thinning', 'thickening'))
    Profiles{'attribute': 'area',
     'image': -7204331716152014795,
     'profiles': [{'operation': 'thinning', 'threshold': 10},
                  {'operation': 'thinning', 'threshold': 5},
                  {'operation': 'copy'},
                  {'operation': 'thickening', 'threshold': 5},
                  {'operation': 'thickening', 'threshold': 10}]}

    """
    data = []
    description = []

    # Create Trees
    try:
        if isinstance(tree_type, type):
            # Dual tree
            ndual = False
            thinning_tree = None
            thickening_tree = tree_type(image, adjacency)
        else:
            # Non dual trees
            ndual = True
            thinning_tree = tree_type[0](image, adjacency)
            thickening_tree = tree_type[1](image, adjacency)
    except:
        raise TypeError('Parameter tree_type must be a tuple or a single type '\
        'of Tree, not {}'.format(tree_type))

    # Get operation names
    try:
        if not ndual:
            thickening_operation = operation
        else:
            thinning_operation = operation[0]
            thickening_operation = operation[1]
    except:
        raise TypeError('Parameter oparation must match tree_type count, '\
                'a single string or an iterable, not {}'.format(operation))

    # Check out_feature
    if not out_feature in ('same', 'altitude'):
        raise ValueError('Unknow value "{}" for parameter '\
                'out_feature'.format(out_feature))

    iter_count = sum(len(x) for x in attribute.values()) * (1 + ndual) + len(attribute)
    ttq = tqdm(desc='Total', total=iter_count)
    for att, thresholds in attribute.items():
        profiles = []; profiles_description = []
        tq = tqdm(total=len(thresholds) * (1 + ndual) + 1, desc=att)

        of = att if out_feature == 'same' else out_feature

        if ndual:
            # thinning
            prof, desc = _compute_profiles(thinning_tree, att,
                    thresholds[::-1], thinning_operation, (ttq, tq), of,
                    filtering_rule)
            profiles += prof
            profiles_description += desc

        # Origin
        tq.update(); ttq.update()
        profiles += [thickening_tree.reconstruct(feature=of)]
        profiles_description += [{'operation': 'copy feature {}'.format(of)}]

        # thickening
        prof, desc = _compute_profiles(thickening_tree, att, thresholds,
                thickening_operation, (ttq, tq), of, filtering_rule)
        profiles += prof
        profiles_description += desc

        tq.close()

        data += [np.stack(profiles)]
        description += [{'attribute': att,
                         'profiles': profiles_description,
                         'image': image_name if image_name else
                         hash(image.data.tobytes())}]
    ttq.close()

    return Profiles(data, description)

def _compute_profiles(tree, attribute, thresholds, operation, tqs, 
        feature='altitude', rule='direct'):
    data = []
    desc = []

    for t in thresholds:
        for tq in tqs: tq.update()
        desc += [{'operation': operation, 'threshold': t}]
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

    >>> image = np.random.random((100, 100))
    >>> sap.attribute_profiles(image, {'area': [10, 100]})
    Profiles[{'attribute': 'area',
    'image': 6508374204896978831,
    'profiles': [{'operation': 'thinning', 'threshold': 100},
                 {'operation': 'thinning', 'threshold': 10},
                 {'operation': 'copy'},
                 {'operation': 'thickening', 'threshold': 10},
                 {'operation': 'thickening', 'threshold': 100}]}]

    See Also
    --------
    sap.trees.available_attributes : List available attributes.

    """
    return create_profiles(image, attribute, (trees.MinTree, trees.MaxTree),
            ('thinning', 'thickening'), adjacency, image_name, 'altitude',
            filtering_rule)

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

    >>> image = np.random.random((100, 100))
    >>> sap.self_dual_attribute_profiles(image, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'image': 2760575455804575354,
     'profiles': [{'operation': 'copy'},
                  {'operation': 'sdap filtering', 'threshold': 10},
                  {'operation': 'sdap filtering', 'threshold': 100}]}
    See Also
    --------
    sap.trees.available_attributes : List available attributes.
    attribute_profiles : other profiles.

    """
    return create_profiles(image, attribute, trees.TosTree, 'sdap filtering',
                           adjacency, image_name, 'altitude', filtering_rule)

def self_dual_feature_profiles(image, attribute, adjacency=4, image_name=None,
        filtering_rule='direct'):
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
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Examples
    --------

    >>> image = np.random.random((100, 100))
    >>> sap.self_dual_feature_profiles(image, {'area': [10, 100]})
    Profiles{'attribute': 'area',
     'image': 2760575455804575354,
     'profiles': [{'operation': 'copy'},
                  {'operation': 'sdfp filtering', 'threshold': 10},
                  {'operation': 'sdfp filtering', 'threshold': 100}]}
    See Also
    --------
    sap.trees.available_attributes : List available attributes.
    attribute_profiles : other profiles.

    """
    return create_profiles(image, attribute, trees.TosTree, 'sdfp filtering',
                           adjacency, image_name, 'same', filtering_rule)

def feature_profiles(image, attribute, adjacency=4, image_name=None,
        filtering_rule='direct'):
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
    filtering_rule: str, optional
        The filtering rule to use. It can be 'direct', 'min', 'max' or
        'subtractive'. Default is 'direct'.

    Examples
    --------

    >>> image = np.arange(5*5).reshape(5,5)

    >>> sap.feature_profiles(image, {'area': [5, 10]})
    Profiles{'attribute': 'area',
     'image': 3055489024601913429,
     'profiles': [{'operation': 'feature profile thinning', 'threshold': 10},
                  {'operation': 'feature profile thinning', 'threshold': 5},
                  {'operation': 'copy'},
                  {'operation': 'feature profile thickening', 'threshold': 5},
                  {'operation': 'feature profile thickening', 'threshold': 10}]}

    See Also
    --------
    sap.trees.available_attributes : List available attributes.
    attribute_profiles : other profiles.

    """
    return create_profiles(image, attribute, (trees.MinTree, trees.MaxTree),
            ('feature profile thinning', 'feature profile thickening'),
            adjacency, image_name, 'same', filtering_rule)

def _show_profiles(profiles, height=None, fname=None, **kwargs):
    assert len(profiles) == 1, 'Show profile only for one attribute at a time.'

    # Set vmin and vmax if not set
    if not 'vmin' in kwargs:
        kwargs['vmin'] = profiles.data.min()
    if not 'vmax' in kwargs:
        kwargs['vmax'] = profiles.data.max()

    if height is not None:
        plt.figure(figsize=_figsize(profiles, height))

    suptitle = '{} - {}'.format(profiles.description['image'], profiles.description['attribute'])

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
        profiles = filter(lambda x: x.description['image'] == image, profiles)

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
        Attribute profiles or other profiles to process the differential
        on.

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

    Examples
    --------

    >>> aps_a = sap.attribute_profiles(image, {'area': [10, 100]})
    >>> aps_b = sap.attribute_profiles(image, {'compactness': [.1, .5]})

    >>> aps = sap.concatenate((aps_a, aps_b))

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

    return np.concatenate(profiles.data)

