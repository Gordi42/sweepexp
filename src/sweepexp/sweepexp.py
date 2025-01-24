"""Running parameter sweeps sequentially."""
from __future__ import annotations

import warnings
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import dill
import numpy as np
import xarray as xr

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


class SweepExp:

    """Base class for the sweepexp experiment."""

    def __init__(self,
                 func: Callable,
                 parameters: dict[str, list],
                 return_values: dict[str, type],
                 save_path: Path | str | None = None) -> None:
        # Set the parameters
        self._func = func
        self._parameters = self._convert_parameters(parameters)
        self._return_values = self._convert_return_types(return_values)
        self._save_path = None if save_path is None else Path(save_path)

        self.pass_uuid = False
        self.auto_save = False

        # create the xarray dataset (or load it from a file)
        path_exists = self.save_path is not None and self.save_path.exists()
        self._data = self._load_data_from_file() if path_exists else self._create_data()

    @abstractmethod
    def run(self, **kwargs: any) -> None:  # pragma: no cover
        """Run the experiment."""
        raise NotImplementedError

    # ================================================================
    #  Data handling
    # ================================================================
    def _create_data(self) -> xr.DataArray:
        """Create the xarray dataset."""
        # Add metadata variables
        variables = [
            {"name": "uuid", "type": "str", "value": ""},
            {"name": "duration", "type": float, "value": np.nan},
            {"name": "status", "type": object, "value": "not started"},
        ]
        # Add the return values
        for name, dtype in self.return_values.items():
            variables.append({"name": name, "type": dtype, "value": np.nan})
        # Create the xarray dataset
        data = xr.Dataset(
            data_vars={var["name"]: (
                    self.parameters.keys(),
                    np.full(self.shape, var["value"], dtype=var["type"]))
                for var in variables},
            coords=self.parameters,
        )
        # Create the uuids
        uuids = np.array([str(uuid4())
                          for _ in range(np.prod(self.shape))],
                        ).reshape(self.shape)

        # set the uuids in the xarray dataset
        data["uuid"].data = uuids
        return data

    def _load_data_from_file(self) -> None:
        """Load the xarray dataset from a file."""
        # TODO(Silvano): Add content
        raise NotImplementedError

    def save(self) -> None:
        """Save the xarray dataset to the save path."""
        if self.save_path is None:
            msg = "The save path is not set. Set the save path before saving."
            raise ValueError(msg)

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
        msg += " Supported extensions are: '.zarr', '.nc', '.cdf', '.pkl'."
        raise ValueError(msg)

    @staticmethod
    def load(save_path: Path | str) -> xr.DataArray:
        """Load the xarray dataset from a file."""
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
        msg += " Supported extensions are: '.zarr', '.nc', '.cdf', '.pkl'."
        raise ValueError(msg)


    def reset_status(self) -> None:
        """Reset the status of all experiments to 'not started'."""
        raise NotImplementedError

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
        """Whether to pass the uuid to the experiment function."""
        return self._pass_uuid

    @pass_uuid.setter
    def pass_uuid(self, pass_uuid: bool) -> None:
        self._pass_uuid = pass_uuid

    @property
    def auto_save(self) -> bool:
        """Whether to automatically save the results after each finished experiment."""
        return self._auto_save

    @auto_save.setter
    def auto_save(self, auto_save: bool) -> None:
        self._auto_save = auto_save

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
        """The uuid of each parameter combination."""
        return self.data["uuid"]

    @property
    def status(self) -> xr.DataArray:
        """The status of each parameter combination."""
        return self.data["status"]

    @property
    def duration(self) -> xr.DataArray:
        """The duration of each experiment."""
        return self.data["duration"]

