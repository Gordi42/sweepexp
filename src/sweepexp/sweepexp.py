"""Running parameter sweeps sequentially."""
from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import dill
import numpy as np
import xarray as xr

from sweepexp import log

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


RESERVED_ARGUMENTS = {"uuid", "status", "duration", "priority"}
POSSIBLE_STATUSES = {"N", "C", "F", "S"}
SUPPORTED_EXTENSIONS = {".zarr", ".nc", ".cdf", ".pkl"}

class SweepExp:

    """
    Run a parameter sweep sequentially.

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
    The SweepExp class can be used to run a custom experiment function with
    different parameter combinations. The results of the experiments are saved
    as an xarray dataset. The dataset can be saved to a file and loaded later
    to continue the experiments. All parameter combinations are run
    sequentially. For parallel execution, consider using the 'SweepExpParallel'
    or 'SweepExpMPI' classes.

    SweepExp supports the following additional features:
    - Custom arguments: Add custom arguments to the experiment function.
    - UUID: Pass a unique identifier to the experiment function.
    - Auto save: Automatically save the results after each experiment.
    - Timeit: Measure the duration of each experiment.
    - Priorities: Run experiments with higher priority first.

    Examples
    --------
    .. code-block:: python

        from sweepexp import SweepExp

        # Create a simple experiment function
        def my_experiment(x: int, y: float) -> dict:
            return {"sum": x + y, "product": x * y}

        # Initialize the SweepExp object
        sweep = SweepExp(
            func=my_experiment,
            parameters={"x": [1, 2, 3], "y": [4, 5, 6]},
            return_values={"sum": int},
        )

        # Run the sweep
        sweep.run()

    """

    def __init__(self,
                 func: Callable,
                 parameters: dict[str, list],
                 return_values: dict[str, type],
                 save_path: Path | str | None = None) -> None:

        # Check that none of the parameters are reserved
        reserved_parameters = set(parameters) & RESERVED_ARGUMENTS
        if reserved_parameters:
            msg = "The following parameter names are reserved: "
            msg += f"{reserved_parameters}."
            msg += "Please choose different names."
            raise ValueError(msg)
        reserved_return_values = set(return_values) & RESERVED_ARGUMENTS
        if reserved_return_values:
            msg = "The following return value names are reserved: "
            msg += f"{reserved_return_values}."
            msg += "Please choose different names."
            raise ValueError(msg)

        # Set the parameters
        self._func = func
        self._parameters = self._convert_parameters(parameters)
        self._return_values = self._convert_return_types(return_values)
        self._save_path = None if save_path is None else Path(save_path)

        self._custom_arguments = set()
        self.pass_uuid = False
        self.auto_save = False
        self.timeit = False
        self.enable_priorities = False

        # create the xarray dataset (or load it from a file)
        path_exists = self.save_path is not None and self.save_path.exists()
        self._data = self._load_data_from_file() if path_exists else self._create_data()

    def add_custom_argument(self, name: str, default_value: any) -> None:
        """
        Add custom arguments to the experiment function.

        Parameters
        ----------
        name : str
            The name of the argument.
        default_value : any
            The default value of the argument.

        Examples
        --------

        .. code-block:: python

            from sweepexp import SweepExp

            # Create a function that takes a custom argument
            def my_experiment(param1: int, custom: float) -> dict:
                return {"product": param1 * custom}

            sweep = SweepExp(
                func=my_experiment,
                parameters={"param1": [1, 2, 3]},
                return_values={"product": int},
            )

            # Add a custom argument
            sweep.add_custom_argument("custom", 2.0)
            # Set the custom argument to 3.0 for the second experiment
            sweep.data["custom"].data[1] = 3.0
            # Run the sweep
            sweep.run()

        """
        # check that the name is not reserved
        if name in RESERVED_ARGUMENTS:
            msg = f"Argument '{name}' is a reserved argument."
            msg += "Please choose a different name."
            raise ValueError(msg)
        # check that the name is not already in the parameters
        if name in self.parameters:
            msg = f"Argument '{name}' is already a parameter."
            raise ValueError(msg)
        if name in self.custom_arguments:
            msg = f"Argument '{name}' is already a custom argument."
            raise ValueError(msg)
        # add the argument to the custom arguments
        self.custom_arguments.add(name)
        # add the argument to the data
        self.data[name] = xr.DataArray(
            data=np.full(self.shape, default_value),
            dims=self.parameters.keys(),
        )

    # ================================================================
    #  Running experiments
    # ================================================================
    def run(self, status: str | list[str] | None = "N") -> xr.Dataset:
        """
        Run all experiments with the status 'N' (not started).

        Parameters
        ----------
        status : str | list[str] | None
            The status of the experiments to run. If None, all experiments with
            status 'N' (not started) are run.

        Returns
        -------
        xr.Dataset
            The xarray dataset with the results of the experiments.

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

        """
        # Create a list of all experiments that need to be run
        indices = self._get_indices(status)
        number_of_experiments = len(indices[0])
        log.info(f"Found {number_of_experiments} experiments to run.")
        # Sort the experiments based on the priorities
        indices = self._sort_indices(indices)
        # Run all experiments
        for index in zip(*indices, strict=False):
            # log how many experiments are left
            log.debug(f"{number_of_experiments} experiments left.")
            number_of_experiments -= 1
            self._run_single(index)

        return self.data

    def _get_indices(self, status: str | list[str] | None) -> np.ndarray:
        """Get the indices of the experiments that match the given status."""
        status = status or list(POSSIBLE_STATUSES)
        if isinstance(status, str):
            status = [status]

        # Get the indices of the experiments with the given status
        return np.argwhere(np.isin(self.status.data, status)).T

    def _sort_indices(self, indices: np.ndarray) -> np.ndarray:
        """Sort the indices based on the priorities."""
        if not self.enable_priorities:
            return indices
        # Get the priorities of the experiments
        priorities = self.priority.data[tuple(indices)]
        # Sort the indices based on the priorities
        return indices.T[np.argsort(-priorities)].T

    def _get_kwargs(self, index: tuple[int]) -> dict[str, any]:
        """Get the keyword arguments for the experiment at the given index."""
        # Get the values of the parameters at the given index
        kwargs = {name: self.data[name].data[ind]
                  for name, ind in zip(self.parameters, index, strict=True)}
        # Get the custom arguments
        kwargs.update({name: self.data[name].data[index]
                       for name in self.custom_arguments})
        return kwargs

    def _run_single(self, index: tuple[int]) -> None:
        """Run the experiment at the given index."""
        kwargs = self._get_kwargs(index)
        log.debug(f"Running experiment with kwargs: {kwargs}")
        if self.timeit:
            start_time = time.time()
        try:
            return_values = self.func(**kwargs)
            status = "C"
        except Exception as error:  # noqa: BLE001
            log.error(f"Error in experiment {self._get_kwargs(index)}: {error}")
            return_values = {}
            status = "F"

        self._set_status_at(index, status)
        self._set_return_values_at(index, return_values)

        if self.timeit:
            duration = time.time() - start_time
            self._set_duration_at(index, duration)
            log.debug(f"Experiment took {duration:.2f} seconds.")

        if self.auto_save:
            self.save(mode="w")

    def _set_return_values_at(self,
                              index: tuple[int],
                              return_values: dict[str, any]) -> None:
        """Set the return values of the experiment at the given index."""
        for name, value in return_values.items():
            self.data[name].data[index] = value

    def _set_duration_at(self, index: tuple[int], duration: float) -> None:
        """Set the duration of the experiment at the given index."""
        self.duration.data[index] = duration

    # ================================================================
    #  Data handling
    # ================================================================
    def _create_data(self) -> xr.DataArray:
        """Create the xarray dataset."""
        # Add metadata variables
        variables = [
            {"name": "status", "type": str, "value": "N"},
        ]
        # Add the return values
        for name, dtype in self.return_values.items():
            variables.append({"name": name, "type": dtype, "value": np.nan})
        # Create the xarray dataset and return it
        data = xr.Dataset(
            data_vars={var["name"]: (
                    self.parameters.keys(),
                    np.full(self.shape, var["value"], dtype=var["type"]))
                for var in variables},
            coords=self.parameters,
        )
        # Add a bit of info to the status variable
        data["status"].attrs = {
            "long_name": "Experiment status.",
            "description": "The status of the experiment.",
            "values": "N: not started, C: completed, F: failed, S: skip",
        }
        return data

    def _load_data_from_file(self) -> xr.DataArray:
        """Load the xarray dataset from a file."""
        log.info(f"Foud data at {self.save_path}. Loading data.")
        data = self.load(self.save_path)

        # We need to do a bunch of checks to make sure the data is correct

        # Compare the parameters
        if set(data.coords) != set(self.parameters.keys()):
            msg = "Parameter mismatch: "
            msg += f"Expected: {set(self.parameters.keys())}, "
            msg += f"Got: {set(data.coords)}."
            raise ValueError(msg)

        # Check if the parameter values are the same
        for name, values in self.parameters.items():
            obt_values = data.coords[name].values
            # check if values are of a numeric type
            if (np.issubdtype(values.dtype, np.number) and
                    not np.issubdtype(values.dtype, complex)):
                # we skip complex numbers, because they are stored weirdly in
                # netcdf files
                parameter_mismatch = not np.allclose(values, obt_values)
            # check if values are of a boolean type
            elif np.issubdtype(values.dtype, bool):
                parameter_mismatch = not np.all(values == obt_values)
            # else the values are of type object and all entries are strings
            elif all(isinstance(val, str) for val in values):
                parameter_mismatch = not all(values == obt_values)
            # Otherwise, we can only check if the lengths are the same
            else:
                parameter_mismatch = len(values) != len(obt_values)
            if parameter_mismatch:
                msg = f"Parameter mismatch for {name}: "
                msg += f"Expected: {values}, "
                msg += f"Got: {obt_values}."
                raise ValueError(msg)

        # Compare the return values
        if not set(self.return_values).issubset(set(data.data_vars)):
            msg = "Return value mismatch: "
            msg += f"Expected: {set(self.return_values)}, "
            msg += f"Got: {set(data.data_vars)}."
            raise ValueError(msg)

        # Check if the return types are the same
        return data

    def save(self, mode: Literal["x", "w"] = "x") -> None:
        """
        Save the xarray dataset to the save path.

        Parameters
        ----------
        mode : Literal["x", "w"] (default="x")
            What to do if there is already data at the save path.
            If "x", raise a FileExistsError. If "w", overwrite the data.

        """
        if self.save_path is None:
            msg = "The save path is not set. Set the save path before saving."
            raise ValueError(msg)

        if mode == "w":
            self._remove_existing_data()
        elif mode == "x" and self.save_path.exists():
                msg = "There is already data at the save path. "
                msg += "Use the 'mode=overwrite' argument to overwrite the data."
                raise FileExistsError(msg)

        # if the extension is zarr, save the data to zarr:
        if self.save_path.suffix == ".zarr":
            with warnings.catch_warnings():
                [warnings.filterwarnings("ignore", message=msg) for msg in [
                    ".* not recognized .* Zarr hierarchy.",
                    ".* Zarr format 3 specification.*"]]
                self.data.to_zarr(self.save_path)
                return
        # if the extension is nc, save the data to netcdf:
        if self.save_path.suffix in [".nc", ".cdf"]:
            self.data.to_netcdf(self.save_path, auto_complex=True)
            return
        # if the extension is .pkl save the data to a pickle file:
        if self.save_path.suffix == ".pkl":
            with Path.open(self.save_path, "wb") as file:
                dill.dump(self.data, file)
            return
        msg = "The file extension is not supported."
        msg += f" Supported extensions are: {SUPPORTED_EXTENSIONS}."
        raise ValueError(msg)

    @staticmethod
    def load(save_path: Path | str) -> xr.DataArray:
        """
        Load the xarray dataset from a file.

        Parameters
        ----------
        save_path : Path | str
            The path to the file to load the data from.
            Must have one of the following extensions: '.zarr', '.nc', '.cdf', '.pkl'.

        Returns
        -------
        data : xr.DataArray
            The xarray dataset with the data.

        .. note::
            Not the SweepExp object is returned, but only the data.

        Examples
        --------
        .. code-block:: python

            # Create a SweepExp object
            sweep = SweepExp(
                func=lambda x: {"y": x},
                parameters={"x": [1, 2, 3]},
                return_values={"y": int},
                save_path="my_data.zarr"
            )
            # Run the sweep and save it
            sweep.run()
            sweep.save()
            # Load the data from the file
            df = SweepExp.load("my_data.zarr")

        """
        save_path = Path(save_path)
        # if the extension is zarr, load the data from zarr:
        if save_path.suffix == ".zarr":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message=".* Zarr format 3 specification.*")
                return xr.open_zarr(save_path)
        # if the extension is nc, load the data from netcdf:
        if save_path.suffix in [".nc", ".cdf"]:
            return xr.open_dataset(save_path)
        # if the extension is .pkl load the data from a pickle file:
        if save_path.suffix == ".pkl":
            with Path.open(save_path, "rb") as file:
                return dill.load(file)  # noqa: S301
        msg = "The file extension is not supported."
        msg += f" Supported extensions are: {SUPPORTED_EXTENSIONS}."
        raise ValueError(msg)

    def _remove_existing_data(self) -> None:
        """Remove the existing data at the save path."""
        if self.save_path is None:
            return
        if not self.save_path.exists():
            return
        if self.save_path.suffix == ".zarr":
            # zarr files are directories, so we need to remove the directory
            shutil.rmtree(self.save_path)
            return
        self.save_path.unlink()

    # ================================================================
    #  Status handling
    # ================================================================

    def reset_status(self, states: str | list[str] | None) -> None:
        """
        Reset the status of experiments to 'N' (not started).

        Parameters
        ----------
        states : list[str] | None
            The states to reset. If None, all states with status 'C' (completed)
            and 'F' (failed) are reset

        Examples
        --------

        .. code-block:: python

            from sweepexp import SweepExp
            sweep = SweepExp(...)  # Initialize the sweep

            # Reset all experiments with status 'C' and 'F' to 'N'
            sweep.reset_status()
            # Reset all experiments with status 'C' to 'N'
            sweep.reset_status("C")
            # Reset all experiments with status 'S' and 'F' to 'N'
            sweep.reset_status(["S", "F"])

        """
        states = states or ["C", "F"]
        if isinstance(states, str):
            states = [states]

        # Check if the states are valid
        valid_states = ["N", "C", "F", "S"]
        if not set(states).issubset(valid_states):
            msg = "Invalid states: "
            msg += f"Got: {states}. "
            msg += f"But expected: {valid_states}."
            raise ValueError(msg)

        # Reset the status of all experiments with the given states
        for state in states:
            self._set_status(state, "N")

    def _set_status(self, old_status: str, new_status: str) -> None:
        """Set the status of all experiments with the old status to the new status."""
        # Get the indices of the experiments with the old status
        indices = np.where(self.status == old_status)
        # Set the status of the experiments to the new status
        self._set_status_at(indices, new_status)

    def _set_status_at(self, index: tuple[int], status: str) -> None:
        """Set the status of the experiment at the given index."""
        self.status.data[index] = status

    # ================================================================
    #  Conversion functions
    # ================================================================
    @staticmethod
    def _convert_parameters(parameters: dict[str, list]) -> dict[str, np.ndarray]:
        """Convert the parameters to a dictionary of numpy arrays."""
        for name, values in parameters.items():
            # if the values are already a numpy array, just use them
            if isinstance(values, np.ndarray):
                parameters[name] = values
            # check if all values are numeric or boolean
            elif (all(np.issubdtype(type(val), np.number) for val in values) or
                  all(isinstance(val, bool) for val in values)):
                parameters[name] = np.array(values)
            # else the dtype is object
            else:
                parameters[name] = np.array(values, dtype=object)
        return parameters

    @staticmethod
    def _convert_return_types(return_values: dict[str, type]) -> dict[str, np.dtype]:
        """Convert the return types to numpy dtypes."""
        for return_name, return_type in return_values.items():
            if return_type is str:
                return_values[return_name] = np.dtype(object)
            else:
                return_values[return_name] = np.dtype(return_type)
        return return_values

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def func(self) -> Callable:
        """The experiment function to run."""
        return self._func

    @property
    def parameters(self) -> dict[str, list]:
        """The parameters to sweep over."""
        return self._parameters

    @property
    def return_values(self) -> dict[str, type]:
        """The return values of the experiment function."""
        return self._return_values

    @property
    def custom_arguments(self) -> set[str]:
        """Custom arguments of the experiment function."""
        return self._custom_arguments

    @property
    def save_path(self) -> Path | None:
        """
        Path to save the results to.

        Supported file formats are: '.zarr', '.nc', '.cdf', '.pkl'.
        The '.zarr' and '.nc' formats only support numeric and boolean data.
        Only the '.pkl' format supports saving data of any type.
        """
        return self._save_path

    @property
    def pass_uuid(self) -> bool:
        """
        Whether to pass the uuid to the experiment function.

        Description
        -----------
        When the pass_uuid property is set to True, a uuid (unique identifier)
        will be assigned to each experiment. This uuid is passed to the experiment
        function as an argument. Make sure to add a uuid argument to the function.

        Examples
        --------

        .. code-block:: python

            from sweepexp import SweepExp

            # Create a function that takes the uuis an an argument and write
            # something to a file with the uuid in the name
            def my_experiment(x: int, uuid: str) -> dict:
                with open(f"output_{uuid}.txt", "w") as file:
                    file.write(f"Experiment with x={x} and uuid={uuid}.")
                return {}

            sweep = SweepExp(
                func=my_experiment,
                parameters={"x": [1, 2, 3]},
                return_values={},
            )

            # Enable the uuid
            sweep.pass_uuid = True
            # Run the sweep
            sweep.run()

        """
        return self._pass_uuid

    @pass_uuid.setter
    def pass_uuid(self, pass_uuid: bool) -> None:
        self._pass_uuid = pass_uuid
        if not pass_uuid:
            # remove the uuid from the custom arguments if it is there
            self._custom_arguments.discard("uuid")
            return
        # Add the uuid to the custom arguments
        self._custom_arguments.add("uuid")
        # Check if the uuid is already in the data
        if "uuid" in self.data.data_vars:
            return
        # If not, add the uuid to the data
        self.data["uuid"] = xr.DataArray(
            data=np.array([str(uuid4())
                           for _ in range(np.prod(self.shape))],
                           ).reshape(self.shape),
            dims=self.parameters.keys(),
            attrs={"units": "",
                   "long_name": "UUID of the experiment.",
                   "description": "A unique identifier for each experiment."},
        )

    @property
    def auto_save(self) -> bool:
        """Whether to automatically save the results after each finished experiment."""
        return self._auto_save

    @auto_save.setter
    def auto_save(self, auto_save: bool) -> None:
        self._auto_save = auto_save

    @property
    def timeit(self) -> bool:
        """
        Whether to measure the duration of each experiment.

        Description
        -----------
        When the timeit property is set to True, a new variable 'duration' is
        added to the data. This variable stores the duration of each experiment
        in seconds.
        """
        return self._timeit

    @timeit.setter
    def timeit(self, timeit: bool) -> None:
        self._timeit = timeit
        if not timeit:
            return
        # Check if the duration is already in the data
        if "duration" in self.data.data_vars:
            return
        # If not, add the duration to the data
        self.data["duration"] = xr.DataArray(
            data=np.full(self.shape, np.nan, dtype=float),
            dims=self.parameters.keys(),
            attrs={"units": "seconds",
                   "long_name": "Duration of the experiment.",
                   "description": "The time it took to run the experiment."},
        )

    @property
    def enable_priorities(self) -> bool:
        """Whether to enable priorities for the experiments."""
        return self._enable_priorities

    @enable_priorities.setter
    def enable_priorities(self, enable_priorities: bool) -> None:
        self._enable_priorities = enable_priorities
        if not enable_priorities:
            return
        # Check if the priority is already in the data
        if "priority" in self.data.data_vars:
            return
        # If not, add the priority to the data
        self.data["priority"] = xr.DataArray(
            data=np.full(self.shape, 0, dtype=int),
            dims=self.parameters.keys(),
            attrs={"units": "",
                   "long_name": "Priority of each experiment.",
                   "description": "Experiments with higher priority are run first."},
        )

    @property
    def shape(self) -> tuple[int]:
        """The shape of the parameter grid."""
        return tuple(len(values) for values in self.parameters.values())

    # ----------------------------------------------------------------
    #  Xarray properties
    # ----------------------------------------------------------------

    @property
    def data(self) -> xr.Dataset:
        """The data of the experiment."""
        return self._data

    @property
    def uuid(self) -> xr.DataArray:
        """
        The uuid of each parameter combination.

        Description
        -----------
        The uuid is a string that uniquely identifies each experiment. By default,
        uuids are disbabled to save memory. They can be enabled by setting the
        'pass_uuid' property to True:

        .. code-block:: python

            sweep = SweepExp(...)
            sweep.pass_uuid = True

        """
        # check if uuid is enabled
        if not self.pass_uuid:
            msg = "UUID is disabled. "
            msg += "Set 'pass_uuid' to True before accessing the uuid."
            raise AttributeError(msg)

        return self.data["uuid"]

    @property
    def status(self) -> xr.DataArray:
        """
        The status of each parameter combination.

        Possible values are:
        - 'N': not started
        - 'C': completed
        - 'F': failed
        - 'S': skip
        """
        return self.data["status"]

    @property
    def duration(self) -> xr.DataArray:
        """
        The duration of each experiment.

        Description
        -----------
        The duration is the time it took to run the experiment with the given
        parameter combination. The duration is only available if the 'timeit'
        property is set to True. Make sure to enable it with:

        .. code-block:: python

            sweep = SweepExp(...)
            sweep.timeit = True

        """
        # check if timeit is enabled
        if not self.timeit:
            msg = "Timeit is disabled. "
            msg += "Set 'timeit' to True before accessing the duration."
            raise AttributeError(msg)
        return self.data["duration"]

    @property
    def priority(self) -> xr.DataArray:
        """
        The priority of each experiment.

        Description
        -----------
        The priority is an integer that determines the order in which the
        experiments are run. Experiments with higher priority are run first.
        Make sure to enable the priorities before accessing them with:

        .. code-block:: python

            sweep = SweepExp(...)
            sweep.enable_priorities = True

        """
        # check if priorities are enabled
        if not self.enable_priorities:
            msg = "Priorities are disabled. "
            msg += "Set 'enable_priorities' to True before accessing the priority."
            raise AttributeError(msg)
        return self.data["priority"]
