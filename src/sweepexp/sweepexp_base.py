"""Base class for the sweepexp experiment."""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Callable

import xarray as xr

class SweepExpBase:

    """Base class for the sweepexp experiment."""

    def __init__(self,
                 func: Callable,
                 parameters: dict[str, list],
                 return_values: dict[str, type],
                 save_path: Path | str | None = None) -> None:
        # Set the parameters
        self.func = func
        self.parameters = parameters
        self.return_values = return_values
        self.save_path = save_path

        self.pass_uuid = False

        # create the xarray dataset (or load it from a file)
        path_exists = self.save_path is not None and self.save_path.exists()
        self.data = self._load_data_from_file() if path_exists else self._create_data()

    @abstractmethod
    def run(self, **kwargs: any) -> None:
        """Run the experiment."""
        raise NotImplementedError

    # ================================================================
    #  Data handling
    # ================================================================
    def _create_data(self) -> None:
        """Create the xarray dataset."""
        # TODO(Silvano): Add content
        raise NotImplementedError

    def _load_data_from_file(self) -> None:
        """Load the xarray dataset from a file."""
        # TODO(Silvano): Add content
        raise NotImplementedError

    def reset_status(self) -> None:
        """Reset the status of all experiments to 'not started'."""
        raise NotImplementedError

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def func(self) -> Callable:
        """The experiment function to run."""
        return self._func

    @func.setter
    def func(self, func: Callable) -> None:
        self._func = func

    @property
    def parameters(self) -> dict[str, list]:
        """The parameters to sweep over."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict[str, list]) -> None:
        # TODO(Silvano): Add parameter setter (this must update the other properties)
        raise NotImplementedError

    @property
    def return_values(self) -> dict[str, type]:
        """The return values of the experiment function."""
        return self._return_values

    @return_values.setter
    def return_values(self, _: any) -> None:
        msg = "The return values can only be set by the constructor."
        raise AttributeError(msg)

    @property
    def save_path(self) -> Path | None:
        """Path to save the results to."""
        return self._save_path

    @save_path.setter
    def save_path(self, save_path: str) -> None:
        if save_path is None:
            self._save_path = None
        self._save_path = Path(save_path)

    @property
    def pass_uuid(self) -> bool:
        """Whether to pass the uuid to the experiment function."""
        return self._pass_uuid

    @pass_uuid.setter
    def pass_uuid(self, pass_uuid: bool) -> None:
        self._pass_uuid = pass_uuid

    # ----------------------------------------------------------------
    #  Xarray properties
    # ----------------------------------------------------------------

    @property
    def data(self) -> xr.Dataset:
        """The data of the experiment."""
        return self._data

    @data.setter
    def data(self, data: xr.Dataset) -> None:
        self._data = data

    @property
    def skip_mask(self) -> xr.DataArray:
        """Mask to skip the experiments (Values with True will skip)."""
        return self.data["skip_mask"]

    @property
    def uuid(self) -> xr.DataArray:
        """The uuid of each parameter combination."""
        return self.data["uuid"]

