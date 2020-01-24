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
            return self

        for data, description in zip(self.data, self.description):
            yield Profiles([data], [description])

    def __len__(self):
        return self.len


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

def show_profiles(profiles, attribute=None, subplot=True, height=4, fname=None, **kwargs):
    """Display profiles with matplotlib.
    
    Parameters
    ----------
    profiles : sap.Profiles
        The profiles to display.
    attribute : optional
        Name of attribute to display. If None then display all the
        attributes containde in profiles.
    subplot : bool, optional, default: True
        When true use only one figure to display all the profiles.     
    height : scalar, optional, default: 4
        Height in inches.
    fname : str or PathLike, optional
        If set, the file path to save the figure.
    
    Notes
    -----
    
    
    """
    # Filter profiles according to attribute if attribute is set
    if attribute:
        profiles = filter(lambda x: x.description['attribute'] == attribute, profiles)
        
    # Keep a copy of kwargs for each profile to display
    kwargs_save = kwargs.copy()
    
    for p in profiles:
        kwargs = kwargs_save.copy()
        # Set vmin and vmax if not set
        if not 'vmin' in kwargs:
            kwargs['vmin'] = p.data.min()
        if not 'vmax' in kwargs:
            kwargs['vmax'] = p.data.max()

        suptitle = '{} - {}'.format(p.description['image'], p.description['attribute'])
        print(suptitle)
        if subplot:
            plt.figure(figsize=_figsize(p, height, subplot))
        for i, (im, profile) in enumerate(zip(p.data, p.description['profiles'])):
            if subplot:
                plt.subplot(1, len(p.data), i+1)
            else:
                plt.figure(figsize=_figsize(p, height, subplot))
            plt.imshow(im, **kwargs)
            plt.title(_title(profile))
            if not subplot:
                if fname:
                    fname = Path(fname)
                    fn = fname.parent / Path(fname.stem + '_{}_{}'.format(p.description['attribute'], 
                         _title(profile).replace(' ', '_')) + fname.suffix)
                    plt.savefig(fn)
                plt.show()
        if subplot:
            plt.tight_layout()
            plt.suptitle(suptitle)
            if fname:
                fname = Path(fname)
                fn = fname.parent / Path(fname.stem + '_{}'.format(p.description['attribute']) + fname.suffix)
                plt.savefig(fn)
            plt.show()

def _figsize(profiles, height, subplot):
    """Compute size of fig given height and subplot."""
    shape = profiles.data.shape[1:]
    count = profiles.data.shape[0]
    hw_ratio = shape[1] / shape[0]
    width = height * hw_ratio if not subplot else height * hw_ratio * count
    return (width, (1 + .1 * subplot) * height)
    
def _title(profile):
    """Process a title of a fig."""
    if profile['operation'] == 'differential':
        p1, p2 = profile['profiles']
        return 'differential ({}, {})'.format(title(p1), title(p2))
    else:
        return ' '.join([str(x) for x in profile.values()])

