"""Running the experiments in parallel using multiprocessing."""
from __future__ import annotations

import multiprocessing as mp
import time
from typing import TYPE_CHECKING

from sweepexp import SweepExp, log

if TYPE_CHECKING:  # pragma: no cover
    import xarray as xr

WAIT_TIME = 0.05  # 50 ms


class SweepExpParallel(SweepExp):

    """
    Run a parameter sweep in parallel using multiprocessing.

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
    The SweepExpParallel class can be used to run a custom experiment function with
    different parameter combinations. The results of the experiments are saved
    as an xarray dataset. The dataset can be saved to a file and loaded later
    to continue the experiments. All parameter combinations are run
    in parallel using multiprocessing. The number of workers can be specified
    using the 'max_workers' parameter in the 'run' method.

    SweepExp supports the following additional features:
    - Custom arguments: Add custom arguments to the experiment function.
    - UUID: Pass a unique identifier to the experiment function.
    - Auto save: Automatically save the results after each experiment.
    - Timeit: Measure the duration of each experiment.
    - Priorities: Run experiments with higher priority first.

    Examples
    --------
    .. code-block:: python

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

    """

    def run(self,
            status: str | list[str] | None = "N",
            max_workers: int | None = None,
            ) -> xr.Dataset:
        """
        Run all experiments with the status 'N' (not started).

        Parameters
        ----------
        status : str | list[str] | None
            The status of the experiments to run. If None, all experiments with
            status 'N' (not started) are run.
        max_workers : int | None
            The maximum number of workers to run in parallel. If None, the
            number of workers is set to the number of CPUs available.

        Returns
        -------
        xr.Dataset
            The results of the experiments as an xarray dataset

        Examples
        --------

        .. code-block:: python

            from sweepexp import SweepExp
            sweep = SweepExp(...)  # Initialize the sweep

            # Run all experiments with status 'N'
            sweep.run()

            # Run all experiments with status 'C'
            sweep.run("C")

            # Run all experiments with status 'S' and 'F'
            sweep.run(["S", "F"])

            # Run all experiments with status 'N' using 4 workers
            sweep.run(max_workers=4)

        """
        # Set the max_workers to the number of CPUs if not specified
        max_workers = max_workers or mp.cpu_count()

        # Create a list of all experiments that need to be run
        indices = self._get_indices(status)
        number_of_experiments = len(indices[0])
        log.info(f"Found {number_of_experiments} experiments to run.")
        # Set the experiment function
        self._set_experiment_function()

        # Sort the experiments based on the priorities
        indices = self._sort_indices(indices)

        # Create a job for each experiment
        jobs = [self._create_job(index)
                     for index in zip(*indices, strict=True)]

        self._run_jobs(jobs, max_workers)
        return self.data

    def _run_jobs(self, jobs: list[dict[str, any]], max_workers: int) -> None:
        """Run the list of processes in parallel."""
        # Create a list to store the active processes
        active_jobs = []

        # Run the experiments
        while jobs or active_jobs:
            # Start new processes
            while jobs and len(active_jobs) < max_workers:
                # Remove the first process, start it and add it to the active processes
                job = jobs.pop(0)
                job["process"].start()
                active_jobs.append(job)

            # Check if any of the active processes have finished
            for job in active_jobs:
                # Check if the process is still alive
                if job["process"].is_alive():
                    continue
                self._handle_finished_job(job)
                active_jobs.remove(job)
                log.debug(f"Number of remaining jobs: {len(jobs) + len(active_jobs)}")

            # Sleep if there are processes left but we can't start new ones
            if ( (jobs and len(active_jobs) >= max_workers) or
                 (not jobs and active_jobs) ):
                time.sleep(WAIT_TIME)

    def _handle_finished_job(self, job: dict[str, any]) -> None:
        """Handle the return values of a finished job."""
        # Get the index of the experiment
        index = job["index"]
        # Get the return values from the queue
        if self.timeit:
            return_values, duration = job["queue"].get()
        else:
            return_values = job["queue"].get()
        # Check if the return values are an exception
        if isinstance(return_values, Exception):
            log.error(f"Error in experiment {self._get_kwargs(index)}: {return_values}")
            return_values = {}
            status = "F"
        else:
            status = "C"

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

    def _create_job(self, index: tuple[int, ...]) -> dict[str, any]:
        # Get the kwargs for the experiment
        kwargs = self._get_kwargs(index)
        # Create a queue for the experiment
        queue = mp.Queue()
        # Create a process for the experiment
        process = mp.Process(
            target=self._exp_func,
            args=(kwargs, queue),
        )
        return {
            "process": process,
            "queue": queue,
            "index": index,
        }

    def _set_experiment_function(self) -> None:
        """Wrap the experiment function to be run in a separate process."""
        def wrapper(kwargs: dict[str, any],  # pragma: no cover
                    queue: mp.Queue) -> None:
            log.debug(f"Start experiment with kwargs: {kwargs}")
            # Save the start time if timeit is enabled
            if self.timeit:
                start_time = time.time()

            # Try to run the experiment function
            try:
                return_values = self.func(**kwargs)
            except Exception as e:  # noqa: BLE001
                return_values = e

            # Save the end time if timeit is enabled
            if self.timeit:
                end_time = time.time()
                return_values = (return_values, end_time - start_time)

            # Put the return values in the queue
            queue.put(return_values)
            log.debug(f"Finished experiment with kwargs: {kwargs}")
        self._exp_func = wrapper
