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
    version='1.0.0',
    author='Florent Guiotte',
    author_email='florent@guiotte.fr',
    description='Simple Attribute Profiles',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fguiotte/sap',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'higra',
        'tqdm',
        'matplotlib',
    ],
)
