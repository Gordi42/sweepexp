"""Test the SweepExpBase class."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sweepexp import SweepExpBase


# ================================================================
#  Helpers
# ================================================================
class MyObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: MyObject) -> bool:
        return self.value == other.value

# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture(params=[
    pytest.param({
        "a": [1, 2],  # int
        "b": [1.0],  # float
        "c": [1.0 + 1j],  # complex
        "d": ["a"],  # str
        "e": [True],  # bool
        "f": np.linspace(0, 1, 2),  # np.ndarray
    }, id="different types"),
    pytest.param({
        "g": [MyObject(1)],  # object
        "h": [1, "a", True],  # mixed
    }, id="with objects", marks=pytest.mark.objects),
    pytest.param({
        "a": [1, 2, 3, 4],
    }, id="single parameter"),
    pytest.param({
        "a b": [1, 2],  # with space
        "c_d": [1.0],  # with underscore
        "e-f": [1.0 + 1j],  # with dash
    }, id="different names"),
])
def parameters(request):
    return request.param

@pytest.fixture(params=[
    pytest.param([{"name": "int", "type": int, "value": 1},
                  {"name": "float", "type": float, "value": 1.0},
                    {"name": "complex", "type": complex, "value": 1.0 + 1j},
                    {"name": "str", "type": str, "value": "a"},
                    {"name": "bool", "type": bool, "value": True},
                    {"name": "np", "type": np.ndarray, "value": np.linspace(0, 1, 10)},
                    {"name": "object", "type": object, "value": MyObject(1)},
                  ], id="different types"),
    pytest.param([{"name": "object", "type": object, "value": MyObject(1)},
                  ], id="with objects", marks=pytest.mark.objects),
    pytest.param([{"name": "int", "type": int, "value": 1}],
                 id="single return value"),
    pytest.param([{"name": "with space", "type": int, "value": 1},
                    {"name": "with_underscore", "type": int, "value": 1},
                    {"name": "with-dash", "type": int, "value": 1},
                 ], id="different names"),
    pytest.param([], id="no return values"),
])
def return_values(request):
    return request.param

@pytest.fixture
def exp_func(return_values):
    def func(**_kwargs: dict) -> dict:
        return {var["name"]: var["value"] for var in return_values}
    return func

@pytest.fixture
def return_dict(return_values):
    return {var["name"]: var["type"] for var in return_values}

@pytest.fixture(params=[".pkl", ".zarr", ".nc"])
def save_path(temp_dir, request):
    return temp_dir / f"test{request.param}"

# ================================================================
#  Tests
# ================================================================

# ----------------------------------------------------------------
#  Test initialization
# ----------------------------------------------------------------
def test_init_no_file(parameters, return_dict, exp_func):
    """Test the initialization of the SweepExpBase class without a file."""
    # Create the experiment
    exp = SweepExpBase(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
    )
    assert isinstance(exp, SweepExpBase)

def test_init_with_nonexistent_file(parameters, return_dict, exp_func, save_path):
    """Test the initialization of the SweepExpBase class with a nonexistent file."""
    # Create the experiment
    exp = SweepExpBase(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    )
    assert isinstance(exp, SweepExpBase)

def test_init_with_valid_existing_file(): ...

def test_init_with_invalid_existing_file(): ...

# ----------------------------------------------------------------
#  Test properties
# ----------------------------------------------------------------

def test_properties_get(parameters, return_dict, exp_func):
    """Test the properties of the SweepExpBase class."""
    # Create the experiment
    exp = SweepExpBase(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
    )

    # Check the public properties
    assert exp.func == exp_func
    assert exp.parameters == parameters
    assert exp.return_values == return_dict
    assert exp.save_path is None
    assert exp.pass_uuid is False
    assert exp.auto_save is False
    assert len(exp.shape) == len(parameters)

    # Check if the xarray dataarrays can be accessed
    assert isinstance(exp.data, xr.Dataset)
    assert isinstance(exp.uuid, xr.DataArray)
    assert isinstance(exp.status, xr.DataArray)
    assert isinstance(exp.duration, xr.DataArray)

    # Check the content of the xarray dataarrays
    # All uuids should be unique
    assert len(exp.uuid.values.flatten()) == len(set(exp.uuid.values.flatten()))
    # All status values should be "not started"
    assert all(exp.status.values.flatten() == "not started")
    # All durations should be np.nan
    assert np.isnan(exp.duration.values).all()

def test_properties_set(parameters, return_dict, exp_func):
    """Test setting the properties of the SweepExpBase class."""
    # Create the experiment
    exp = SweepExpBase(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
    )


    # Test setting properties that are allowed

    # pass_uuid
    assert not exp.pass_uuid
    exp.pass_uuid = True
    assert exp.pass_uuid

    # auto_save
    assert not exp.auto_save
    exp.auto_save = True
    assert exp.auto_save

    # test setting values in the xarray dataarrays
    loc = (slice(None),) * len(parameters)
    # uuid
    uuid = "test"
    assert not (exp.uuid.values == uuid).any()
    exp.uuid.loc[loc] = uuid
    assert (exp.uuid.values == uuid).any()
    # status
    status = "skip"
    assert not (exp.status.values == status).any()
    exp.status.loc[loc] = status
    assert (exp.status.values == status).any()
    # duration
    duration = 1.0
    assert not (exp.duration.values == duration).any()
    exp.duration.loc[loc] = duration
    assert (exp.duration.values == duration).any()

    # Test readonly properties (should raise an AttributeError)
    readonly_properties = ["func", "parameters", "return_values", "save_path",
                           "data", "uuid", "status", "duration"]
    for prop in readonly_properties:
        with pytest.raises(AttributeError):
            setattr(exp, prop, None)


# ----------------------------------------------------------------
#  Test data saving and loading
# ----------------------------------------------------------------

def test_save(parameters, return_dict, exp_func, save_path, request):
    """Test saving the data."""
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExpBase(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    )
    # Check that the file does not exist
    assert not save_path.exists()
    # Save the data
    exp.save()
    # Check that the file exists
    assert save_path.exists()

def test_load(parameters, return_dict, exp_func, save_path, request):
    """Test loading the data."""
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExpBase(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    )
    # Save the data
    exp.save()
    # try to load the dataset
    ds = SweepExpBase.load(save_path)
    # Check that all variables exist
    for var in exp.data.variables:
        assert var in ds.variables


@pytest.mark.parametrize("invalid_file", ["test", "test.txt", "test.csv", "test.json"])
def test_invalid_file_format(invalid_file):
    """Test loading a file with an invalid format."""
    msg = "The file extension is not supported."
    # loading
    with pytest.raises(ValueError, match=msg):
        SweepExpBase.load(invalid_file)
    # saving
    exp = SweepExpBase(
        func=lambda: None,
        parameters={"a": [1]},
        return_values={},
        save_path=invalid_file,
    )
    with pytest.raises(ValueError, match=msg):
        exp.save()

# ----------------------------------------------------------------
#  Test conversion functions
# ----------------------------------------------------------------

@pytest.mark.parametrize(*("para_in, dtype", [
    pytest.param([1, 2], np.dtype("int64"), id="int"),
    pytest.param([1, 2.0], np.dtype("float64"), id="float"),
    pytest.param([1, 2.0 + 1j], np.dtype("complex128"), id="complex"),
    pytest.param(["a", "boo"], np.dtype(object), id="str"),
    pytest.param([True, False], np.dtype(bool), id="bool"),
    pytest.param(np.linspace(0, 1, 10), np.dtype("float64"), id="np.ndarray"),
    pytest.param([MyObject(1)], np.dtype(object), id="object"),
]))
def test_convert_parameters(para_in, dtype):
    """Test the _convert_parameters function."""
    converted = SweepExpBase._convert_parameters({"a": para_in})["a"]
    assert converted.dtype is dtype

@pytest.mark.parametrize(*("type_in, type_out", [
    pytest.param(int, np.dtype("int64"), id="int"),
    pytest.param(float, np.dtype("float64"), id="float"),
    pytest.param(complex, np.dtype("complex128"), id="complex"),
    pytest.param(str, np.dtype(object), id="str"),
    pytest.param(bool, np.dtype(bool), id="bool"),
    pytest.param(np.ndarray, np.dtype(object), id="np.ndarray"),
    pytest.param(object, np.dtype(object), id="object"),
]))
def test_convert_return_types(type_in, type_out):
    """Test the _convert_return_types function."""
    converted = SweepExpBase._convert_return_types({"a": type_in})["a"]
    assert converted is type_out
