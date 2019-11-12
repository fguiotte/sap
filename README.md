# Simple Attribute Profiles (SAP)


[![pipeline status](https://gitlab.inria.fr/fguiotte/sap/badges/master/pipeline.svg)](https://gitlab.inria.fr/fguiotte/sap/commits/master)
[![coverage report](https://gitlab.inria.fr/fguiotte/sap/badges/master/coverage.svg)](https://gitlab.inria.fr/fguiotte/sap/commits/master)


SAP is a Python package to easily compute morphological attribute
profiles (AP) of images.

## Installation

```shell
pip install sap
```

## Documentation

Documentation is available at <https://python-sap.readthedocs.org/>.

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


