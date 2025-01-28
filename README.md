![Read the Docs](https://img.shields.io/readthedocs/sweepexp)
![tests](https://github.com/Gordi42/sweepexp/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/github/Gordi42/sweepexp/graph/badge.svg?token=SHVDIIOL8Y)](https://codecov.io/github/Gordi42/sweepexp)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub License](https://img.shields.io/github/license/Gordi42/sweepexp)

<p align="center">
<img src="docs/source/_static/sweepexp_logo.png?raw=true">
</p>

A python package for running parallel experiments across parameter grids with MPI.

## Features

- **Flexible Experimentation with Parameter Grids:** SweepExp simplifies running experiments over a grid of parameter combinations. Results are stored in an xarray dataset for easy access and analysis.
- **Parallelization:** Support for parallelization using multiprocessing, or MPI for high-performance computing.
- **User-Friendly API:**  Define the function to be tested, set up parameter sweeps, and specify return types effortlessly.

## Installation

SweepExp can be installed via pip:

```bash
pip install sweepexp
```

## Usage
The followin example shows how to setup a simple experiment that is run on a grid of parameters. Where each parameter combination is run in parallel on separate processes. 

```python

from sweepexp import SweepExpParallel

# Define a function to be tested
def my_custom_experiment(x: float, y: float) -> dict:
    """Add and multiply two numbers."""
    return {"addition": x + y, "multiplication": x * y}

sweep = SweepExpParallel(
    func = my_custom_experiment,
    parameters = { "x": [1, 2], "y": [3, 4, 5] },
    return_values = { "addition": float, "multiplication": float },
)

sweep.run()

print(sweep.data)
```

with the output:
```
<xarray.Dataset> Size: 160B
Dimensions:         (x: 2, y: 3)
Coordinates:
    * x               (x) int64 16B 1 2
    * y               (y) int64 24B 3 4 5
Data variables:
    status          (x, y) <U1 24B 'C' 'C' 'C' 'C' 'C' 'C'
    addition        (x, y) float64 48B 4.0 5.0 6.0 5.0 6.0 7.0
    multiplication  (x, y) float64 48B 3.0 4.0 5.0 6.0 8.0 10.0
```

For more information on how to use the package, please refer to the [documentation](https://sweepexp.readthedocs.io/)


## Author
- [Silvano Gordian Rosenau](silvano.rosenau@uni-hamburg.de)

## License
SweepExp is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
