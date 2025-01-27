![Read the Docs](https://img.shields.io/readthedocs/sweepexp)
![tests](https://github.com/Gordi42/sweepexp/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/github/Gordi42/sweepexp/graph/badge.svg?token=SHVDIIOL8Y)](https://codecov.io/github/Gordi42/sweepexp)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![GitHub License](https://img.shields.io/github/license/Gordi42/sweepexp)

# SweepExp

A python package for running parallel experiments across parameter grids with MPI.

## Features

- **Flexible Experimentation with Parameter Grids:** SweepExp simplifies running experiments over a grid of parameter combinations. Results are efficiently stored in zarr format, ready for analysis with xarray.
- **Parallelization:** Support for parallelization using multiprocessing, or MPI for high-performance computing.
- **User-Friendly API:**  Define the function to be tested, set up parameter sweeps, and specify return types effortlessly.

## Installation

SweepExp can be installed via pip (once published on PyPI):

```bash
pip install sweepexp
```

Or clone the repository and install it locally:

```bash
git clone https://github.com/Gordi42/sweepexp
cd sweepexp
pip install -e .
```

## Usage
The followin example shows how to setup a simple experiment that is run on a grid of parameters. Where each parameter combination is run in parallel on separate processes. The results are saved to a zarr file, which can be easily loaded with xarray.

```python

from sweepexp import SweepExpParallel

# Define a function to be tested
def my_function(param1: str, param2: float) -> dict:
    # Do something with the parameters
    result1 = param1.upper()
    result2 = param2 ** 2
    return {"result1": result1, "result2": result2}

sweep = SweepExpParallel(
    func = my_function,
    parameters = {
        "param1": ["a", "b", "c"],
        "param2": [1.0, 2.0, 3.0]
    },
    return_values = {
        "result1": str,
        "result2": float
    },
    save_path = "data/results.zarr",
)

results = sweep.run()

print(results)
```
For more information on how to use the package, please refer to the [documentation](https://sweepexp.readthedocs.io/)


## Author
- [Silvano Gordian Rosenau](silvano.rosenau@uni-hamburg.de)

## License
SweepExp is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
