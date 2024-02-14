# Simple Attribute Profiles (SAP)


[![build status](https://github.com/fguiotte/sap/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/fguiotte/sap/actions/workflows/main.yml)
[![codecov](https://img.shields.io/codecov/c/github/fguiotte/sap?logo=codecov&token=D8VCLKNAYW)](https://codecov.io/gh/fguiotte/sap)
[![Documentation Status](https://img.shields.io/readthedocs/python-sap/master)](https://python-sap.readthedocs.io/en/master)
[![Pypi version](https://img.shields.io/pypi/v/sap.svg)](https://pypi.org/project/sap/)
[![Pypi python](https://img.shields.io/pypi/pyversions/sap)](https://pypi.org/project/sap/)


SAP is a Python package to easily compute morphological attribute
profiles (AP) of images.

## Installation

```shell
pip install sap
```

## Documentation

Documentation is available on <https://python-sap.rtfd.io>.

### Notebooks

**TODO:**

### Code examples

```python
import sap
import numpy as np
import matplotlib.pyplot as plt

image = np.random.random((512, 512))

t = sap.MaxTree(image)
area = t.get_attribute('area')
filtered_image = t.reconstruct(area < 100)

plt.imshow(filtered_image)
plt.show()
```

## Develop status

[![build status](https://github.com/fguiotte/sap/actions/workflows/main.yml/badge.svg?branch=develop)](https://github.com/fguiotte/sap/actions/workflows/main.yml)
[![codecov](https://img.shields.io/codecov/c/github/fguiotte/sap/develop?logo=codecov&token=D8VCLKNAYW)](https://codecov.io/gh/fguiotte/sap)
[![Documentation Status](https://img.shields.io/readthedocs/python-sap/develop)](https://python-sap.readthedocs.io/en/develop)
