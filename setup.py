#!/usr/bin/env python
# file setup.py
# author Florent Guiotte <florent.guiotte@irisa.fr>
# version 0.0
# date 12 nov. 2019

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='sap',
    version='0.0.11',
    author='Florent Guiotte',
    author_email='florent.guiotte@irisa.fr',
    description='Simple Attribute Profiles',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.inria.fr/fguiotte/sap',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'higra',
        'tqdm',
        'matplotlib',
        'pathlib'
    ],
)
