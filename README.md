# S-DATE Library

The S-DATE (Synthetic Data Augmentation TEchnique) library provides functionality for generating synthetic data through a unique approach that leverages dimensionality reduction and distance-based sample averaging.

## Features

- Generate synthetic data using the S-DATE methodology.
- Flexible distance metrics: Euclidean, Manhattan, and Cosine.
- Support for dimensionality reduction using PCA or SVD.
- Customizable number of closest samples for synthetic data generation.
- Visualization tools for comparing original and synthetic data.

## Installation

To install the S-DATE library, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/s_data.git
    cd s_data
    ```

2. Install the library in editable mode:
    ```bash
    pip install -e .
    ```

3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Here's a quick example of how to use the S-DATE library:

```python
import pandas as pd
from s_data import generate_synthetic_data

# Example DataFrame with your features
data = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': [4, 5, 6],
    'feature3': [7, 8, 9]
})

# Generate synthetic data
synthetic_data = generate_synthetic_data(
    features=data,
    num_samples=4000,
    dim_reduction_method='PCA',
    distance_metric='euclidean',
    normalization=True,
    visualize=True,
    closest_samples=10
)
