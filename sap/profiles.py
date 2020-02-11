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

Todo
----

Names:
    - vectorize or concatenate ?

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
    
    def diff(self):
        """Compute the differential of profiles.

        Returns
        -------
        differential : Profiles
            The processed differential profiles.
        
        """
        return differential(self)


def attribute_profiles(image, attribute, adjacency=4, image_name=None):
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

    Examples
    --------

    >>> image = np.random.random((100, 100))
    >>> sap.attribute_profiles(image, {'area': [10, 100]})
    Profiles[{'attribute': 'area',
    'image': 6508374204896978831,
    'profiles': [{'operation': 'open', 'threshold': 100},
                 {'operation': 'open', 'threshold': 10},
                 {'operation': 'copy'},
                 {'operation': 'close', 'threshold': 10},
                 {'operation': 'close', 'threshold': 100}]}]

    See Also
    --------
    sap.trees.available_attributes : List available attributes.

    """
    data = []
    description = []

    max_tree = trees.MaxTree(image, adjacency)
    min_tree = trees.MinTree(image, adjacency)

    iter_count = sum(len(x) for x in attribute.values()) * 2 + len(attribute)
    ttq = tqdm(desc='Total', total=iter_count)
    for att, thresholds in attribute.items():
        profiles = []; profiles_description = []
        tq = tqdm(total=len(thresholds) * 2 + 1, desc=att)

        # Open
        prof, desc = _compute_profiles(min_tree, att, thresholds[::-1], 'open', (ttq, tq))
        profiles += prof
        profiles_description += desc

        # Origin
        tq.update(); ttq.update()
        profiles += [image]
        profiles_description += [{'operation': 'copy'}]

        # Close
        prof, desc = _compute_profiles(max_tree, att, thresholds, 'close', (ttq, tq))
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

def _compute_profiles(tree, attribute, thresholds, operation, tqs):
    data = []
    desc = []

    for t in thresholds:
        for tq in tqs: tq.update()
        desc += [{'operation': operation, 'threshold': t}]
        deleted_nodes = tree.get_attribute(attribute) < t
        data += [tree.reconstruct(deleted_nodes)]

    return data, desc

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

