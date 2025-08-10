"""Main entry point for the sweepexp package."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import sweepexp as se

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from pathlib import Path


def sweepexp(
        func: Callable,
        parameters: dict[str, list],
        mode: Literal["sequential", "parallel", "mpi"] = "sequential",
        save_path: Path | str | None = None,
        **kwargs: dict,
        ) -> se.SweepExp | se.SweepExpMPI | se.SweepExpParallel:
    """
    Create a new instance of the SweepExp class.

    Parameters
    ----------
    func : Callable
        The experiment function to run. The function should take the parameters
        as keyword arguments and return a dictionary with the return values.
    parameters : dict[str, list]
        The parameters to sweep over. The keys are the parameter names and the
        values are lists of the parameter values.
    mode : "sequential" | "parallel" | "mpi", default="sequential"
        The mode to run the experiments in.
    save_path : Path | str | None (optional)
        The path to save the results to. Supported file formats are: '.zarr',
        '.nc', '.cdf', '.pkl'. The '.zarr' and '.nc' formats only support
        numeric and boolean data. Only the '.pkl' format supports saving data
        of any type.
    **kwargs : dict
        Additional settings:
        - `timeit`: bool, measure the duration of each experiment.
        - `auto_save`: bool, automatically save the results after each experiment.
        - `enable_priorities`: bool, run experiments with higher priority first.
        - `pass_uuid`: bool, pass a unique identifier to the experiment function.

    Returns
    -------
    SweepExp | SweepExpMPI | SweepExpParallel
        An instance of the appropriate SweepExp class based on the mode.

    Examples
    --------
    .. code-block:: python

        from sweepexp import sweepexp

        # Create a simple experiment function
        def my_experiment(x: int, y: float) -> dict:
            return {"sum": x + y, "product": x * y}

        # Initialize the sweepexp object
        sweep = sweepexp(
            func=my_experiment,
            parameters={"x": [1, 2, 3], "y": [4, 5, 6]},
        )

        # Run the sweep
        sweep.run()

    """
    # update the kwargs with the provided parameters
    kwargs.update({
        "func": func,
        "parameters": parameters,
        "save_path": save_path,
    })
    if mode == "mpi":
        try:
            return se.SweepExpMPI(**kwargs)
        except ImportError:
            msg = "Failed to import 'mpi4py'. "
            msg += "Fallback to 'parallel' mode."
            se.log.warning(msg)
            return se.SweepExpParallel(**kwargs)
    if mode == "parallel":
        return se.SweepExpParallel(**kwargs)
    if mode == "sequential":
        return se.SweepExp(**kwargs)

    msg = f"Unknown mode '{mode}'. "
    msg += "Supported modes are: 'sequential', 'parallel', 'mpi'."
    raise ValueError(msg)
