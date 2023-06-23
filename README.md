# IterSHAP: Iterative feature selection using SHAP values
Author: Frank van Mourik, University of Twente

## Installation
Install via [pip](https://pypi.org/project/itershap/) using: ```pip install itershap``` (requires Python version >=3.10,<3.11).

## Usage
```py
from itershap import IterSHAP

X, y = get_data() # Replace with data location

fs = IterSHAP() # Create a IterSHAP feature selection object
fs.fit(X, y) # Execute IterSHAP on input data
X_transformed = fs.transform(X) # Only keep the via IterSHAP selected features
```

## Benefits
* Performs well on small high-dimensional datasets
* Guarantees to return a feature subset
* Model-agnostic (limited by [shap](https://github.com/slundberg/shap) supported models)
* Validated on synthesised data
* Benchmarked on [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

## License
Available under the MIT license, which can be found [here](LICENSE.txt)
