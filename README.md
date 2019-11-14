# Simple Attribute Profiles (SAP)


[![build status](https://gitlab.inria.fr/fguiotte/sap/badges/master/pipeline.svg)](https://gitlab.inria.fr/fguiotte/sap/commits/master)
[![coverage report](https://gitlab.inria.fr/fguiotte/sap/badges/master/coverage.svg)](https://gitlab.inria.fr/fguiotte/sap/commits/master)
[![Documentation Status](https://readthedocs.org/projects/python-sap/badge/?version=master)](https://python-sap.readthedocs.io/en/latest/?badge=master)


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

