"""Running the experiments in parallel using mpi."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from mpi4py import MPI

from sweepexp import SweepExp, log

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

MAIN_RANK = 0
MY_RANK = MPI.COMM_WORLD.Get_rank()
IS_MAIN_RANK = MY_RANK == MAIN_RANK
WAIT_TIME = 0.05  # 50 ms

class SweepExpMPI(SweepExp):

    """
    Run a parameter sweep in parallel using MPI.

    Parameters
    ----------
    func : Callable
        The experiment function to run. The function should take the parameters
        as keyword arguments and return a dictionary with the return values.
    parameters : dict[str, list]
        The parameters to sweep over. The keys are the parameter names and the
        values are lists of the parameter values.
    return_values : dict[str, type]
        The return values of the experiment function. The keys are the return
        value names and the values are the types of the return values.
    save_path : Path | str | None
        The path to save the results to. Supported file formats are: '.zarr',
        '.nc', '.cdf', '.pkl'. The '.zarr' and '.nc' formats only support
        numeric and boolean data. Only the '.pkl' format supports saving data
        of any type.

    Description
    -----------
    The SweepExpMPI class can be used to run a custom experiment function with
    different parameter combinations. The results of the experiments are saved
    as an xarray dataset. The dataset can be saved to a file and loaded later
    to continue the experiments. All parameter combinations are run
    in parallel using mpi4py.

    SweepExp supports the following additional features:
    - Custom arguments: Add custom arguments to the experiment function.
    - UUID: Pass a unique identifier to the experiment function.
    - Auto save: Automatically save the results after each experiment.
    - Timeit: Measure the duration of each experiment.
    - Priorities: Run experiments with higher priority first.

    Examples
    --------

    .. code-block:: python
        :caption: my_experiment.py

        from sweepexp import SweepExpParallel

        # Create a simple experiment function
        def my_experiment(x: int, y: float) -> dict:
            return {"sum": x + y, "product": x * y}

        # Initialize the SweepExp object
        sweep = SweepExpParallel(
            func=my_experiment,
            parameters={"x": [1, 2, 3], "y": [4, 5, 6]},
            return_values={"sum": int},
        )

        # Run the sweep
        sweep.run()

    To run the experiment on 4 CPUs using MPI, use the following command:

    .. code-block:: bash

        mpiexec -n 4 python3 my_experiment.py

    Or, alternatively any other MPI launcher, e.g., `mpirun`, `srun`, etc.

    """

    # Override save and load methods to only save and load on the main rank
    def save(self, mode: Literal["x", "w"] = "x") -> None:  # noqa: D102
        if IS_MAIN_RANK:
            super().save(mode=mode)

    def run(self, status: str | list[str] | None = "N") -> xr.Dataset:  # noqa: D102
        # Check that at least two ranks are available
        min_size = 2
        if MPI.COMM_WORLD.Get_size() < min_size:
            msg = "At least two ranks are required to run the sweep."
            raise ValueError(msg)
        if not IS_MAIN_RANK:
            self._handle_jobs()
        else:
            self._manage_jobs(status)

        return self.data

    # ----------------------------------------------------------------
    #  Methods for the main rank
    # ----------------------------------------------------------------
    def _manage_jobs(self, status: str | list[str] | None) -> None:
        # Create a list of all experiments that need to be run
        indices = self._get_indices(status)
        number_of_experiments = len(indices[0])
        log.info(f"Found {number_of_experiments} experiments to run.")
        # Sort the experiments based on the priorities
        indices = self._sort_indices(indices)

        # Create a job for each experiment
        jobs = list(zip(*indices, strict=True))

        free_workers = list(range(1, MPI.COMM_WORLD.Get_size()))

        active_jobs = []

        while jobs or active_jobs:

            # Start new jobs
            while jobs and free_workers:
                index = jobs.pop(0)
                kwargs = self._get_kwargs(index)
                worker = free_workers.pop(0)
                MPI.COMM_WORLD.send(kwargs, dest=worker)
                active_jobs.append((worker, index))

            # Check if any of the active jobs have finished
            for worker, index in active_jobs:
                # Check if the process is still alive
                if not MPI.COMM_WORLD.Iprobe(source=worker):
                    continue
                result = MPI.COMM_WORLD.recv(source=worker)
                self._handle_finished_job(index, result)
                free_workers.append(worker)
                active_jobs.remove((worker, index))
                # log the number of remaining jobs
                log.debug(f"Number of remaining jobs: {len(jobs) + len(active_jobs)}")

            # Sleep for a short time to prevent busy waiting
            if ( (jobs and not free_workers) or
                 (not jobs and active_jobs) ):
                time.sleep(WAIT_TIME)

        # Send a signal to the workers to stop
        for worker in free_workers:
            MPI.COMM_WORLD.send(None, dest=worker)


    def _handle_finished_job(self, index: tuple[int, ...], result: tuple) -> None:
        # unpack the result
        return_values, status, duration = result

        # Set the status and return values of the experiment
        self._set_status_at(index, status)
        self._set_return_values_at(index, return_values)

        # Set the duration of the experiment
        if self.timeit:
            self._set_duration_at(index, duration)
            log.debug(f"Experiment took {duration:.2f} seconds.")

        # Save the results (if enabled)
        if self.auto_save:
            self.save(mode="w")

    # ----------------------------------------------------------------
    #  Methods for workers
    # ----------------------------------------------------------------

    def _handle_jobs(self) -> None:
        """Handle the jobs."""
        while True:
            # Receive the index of the parameter combination to be run
            kwargs = MPI.COMM_WORLD.recv(source=MAIN_RANK)
            if kwargs is None:
                break
            # Run the experiment and send the results back to the main rank
            MPI.COMM_WORLD.send(self._run_experiment(kwargs), dest=MAIN_RANK)

    def _run_experiment(self, kwargs: dict[str: any]) -> tuple:
        """Run a single experiment."""
        log.debug(f"Rank {MY_RANK} - Start experiment with kwargs: {kwargs}")
        if self.timeit:
            start_time = time.time()

        try:
            return_values = self.func(**kwargs)
            status = "C"
        except Exception as error:  # noqa: BLE001
            log.error(f"Error in experiment {kwargs}: {error}")
            return_values = {}
            status = "F"

        # Calculate the duration of the experiment
        duration = time.time() - start_time if self.timeit else 0

        log.debug(f"Rank {MY_RANK} - Finished experiment with kwargs: {kwargs}")

        return return_values, status, duration
