"""Test the SweepExp class."""
from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from sweepexp import SweepExp


# ================================================================
#  Helpers
# ================================================================
class MyObject:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: MyObject) -> bool:
        if not isinstance(other, MyObject):
            return False
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

@pytest.fixture(params=["kwarg", "property"])
def method(request):
    return request.param

# ================================================================
#  Tests
# ================================================================

# ----------------------------------------------------------------
#  Test initialization
# ----------------------------------------------------------------
def test_init_no_file(parameters, exp_func):
    """Test the initialization of the SweepExp class without a file."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
    )
    assert isinstance(exp, SweepExp)

def test_init_with_nonexistent_file(parameters, exp_func, save_path):
    """Test the initialization of the SweepExp class with a nonexistent file."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        save_path=save_path,
    )
    assert isinstance(exp, SweepExp)

def test_init_with_valid_existing_file(
        parameters, exp_func, save_path, request):
    """Test the initialization of the SweepExp class with a valid existing file."""
    # Skip the test if objects are present (since they cannot be saved)
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")

    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        save_path=save_path,
        timeit=True,
    )
    # Modify some properties
    loc = (slice(None),) * len(parameters)
    exp.status.loc[loc] = "S"
    exp.save()

    # Create a new experiment with the same file
    sweep = SweepExp(
        func=exp_func,
        parameters=parameters,
        save_path=save_path,
    )

    # Check that the experiment was loaded correctly
    assert isinstance(sweep, SweepExp)
    # Check that the changes are present
    assert (sweep.status.values == "S").any()
    assert sweep.timeit
    assert not sweep.auto_save

@pytest.mark.parametrize(*("para, msg", [
    pytest.param({"extra": [1]},
                 "Parameter mismatch", id="extra parameter"),
    pytest.param({"int": [1, 3]},
                 "Parameter mismatch", id="different parameter values (int)"),
    pytest.param({"bool": [False]},
                    "Parameter mismatch", id="different parameter values (bool)"),
    pytest.param({"float": [1.01/3 + 1e-4]},
                    "Parameter mismatch", id="different parameter values (float)"),
    pytest.param({"str": ["b"]},
                  "Parameter mismatch", id="different parameter values (str)"),
    pytest.param({"np": np.linspace(0, 1.1, 2)},
                  "Parameter mismatch", id="different parameter values (np)"),
]))
def test_init_with_invalid_existing_file(para, msg, save_path):
    """Test the initialization of the SweepExp class with an invalid existing file."""
    parameters = {"int": [1, 2], "bool": [True], "float": [1.01/3], "str": ["a"],
                  "np": np.linspace(0, 1, 2)}
    # Create the experiment
    SweepExp(
        func=lambda: None,  # dummy function (does not matter here)
        parameters=parameters,
        save_path=save_path,
    ).save()

    parameters.update(para)

    with pytest.raises(ValueError, match=msg):
        SweepExp(
            func=lambda: None,  # dummy function (does not matter here)
            parameters=parameters,
            save_path=save_path,
        )

# ----------------------------------------------------------------
#  Test properties
# ----------------------------------------------------------------

def test_properties_get(parameters, exp_func):
    """Test the properties of the SweepExp class."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
    )

    # Check the public properties
    assert exp.func == exp_func
    assert exp.parameters == parameters
    assert exp.save_path is None
    assert exp.pass_uuid is False
    assert exp.auto_save is False
    assert len(exp.shape) == len(parameters)

    # Check if the xarray dataarrays can be accessed
    assert isinstance(exp.data, xr.Dataset)
    assert isinstance(exp.status, xr.DataArray)

    # Check the content of the xarray dataarrays
    # All status values should be "not started"
    assert all(exp.status.values.flatten() == "N")

def test_properties_set(parameters, exp_func):
    """Test setting the properties of the SweepExp class."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
    )

    # Test setting properties that are allowed

    # auto_save
    assert not exp.auto_save
    exp.auto_save = True
    assert exp.auto_save

    # test setting values in the xarray dataarrays
    loc = (slice(None),) * len(parameters)
    status = "S"
    assert not (exp.status.values == status).any()
    exp.status.loc[loc] = status
    assert (exp.status.values == status).any()

    # Test readonly properties (should raise an AttributeError)
    readonly_properties = ["func", "parameters", "save_path", "data", "status"]
    for prop in readonly_properties:
        with pytest.raises(AttributeError):
            setattr(exp, prop, None)

def test_uuid(parameters, exp_func):
    """Test the uuid property."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
    )
    # UUID disabled:
    # Check that uuid is not in the data variables
    assert "uuid" not in exp.data.data_vars
    # Check that the uuid property can not be accessed
    msg = "UUID is disabled."
    with pytest.raises(AttributeError, match=msg):
        _ = exp.uuid
    # Check that uuid is not in the custom arguments
    assert "uuid" not in exp.custom_arguments

    # Enable the uuid property
    exp.pass_uuid = True
    # Check that the uuid is now in the custom arguments
    assert "uuid" in exp.custom_arguments
    # Check that the uuid is now in the data variables
    assert "uuid" in exp.data.data_vars
    # Check that the uuid property can be accessed
    assert isinstance(exp.uuid, xr.DataArray)
    # Check that the uuid is unique
    assert len(exp.uuid.values.flatten()) == len(set(exp.uuid.values.flatten()))

    # Disable the uuid property
    old_uuid = exp.uuid
    exp.pass_uuid = False
    # Check that the uuid is not in the custom arguments
    assert "uuid" not in exp.custom_arguments
    # Check that we can not access the uuid property anymore
    with pytest.raises(AttributeError, match=msg):
        _ = exp.uuid

    # Enable the uuid property again and check that the uuid is the same
    exp.pass_uuid = True
    assert exp.uuid.equals(old_uuid)
    assert "uuid" in exp.custom_arguments

def test_duration(parameters, exp_func):
    """Test the duration property."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
    )
    # Timeit disabled:
    # Check that duration is not in the data variables
    assert "duration" not in exp.data.data_vars
    # Check that the duration property can not be accessed
    msg = "Timeit is disabled."
    with pytest.raises(AttributeError, match=msg):
        _ = exp.duration

    # Enable the duration property
    exp.timeit = True
    # Check that the duration is now in the data variables
    assert "duration" in exp.data.data_vars
    # Check that the duration property can be accessed
    assert isinstance(exp.duration, xr.DataArray)
    # Check that all values are nan
    assert np.isnan(exp.duration.values).all()
    # Check that the duration has attributes
    for attr in ["long_name", "description"]:
        assert attr in exp.duration.attrs

    # Set the duration to a value
    loc = (slice(None),) * len(parameters)
    exp.duration.loc[loc] = 1
    duration = exp.duration

    # Disable the duration property
    exp.timeit = False
    # Check that we can not access the duration property anymore
    with pytest.raises(AttributeError, match=msg):
        _ = exp.duration

    # Enable the duration property again and check that the duration is the same
    exp.timeit = True
    assert exp.duration.equals(duration)

def test_priority_property(parameters, exp_func):
    """Test the priority property."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
    )
    # Priority disabled:
    # Check that priority is not in the data variables
    assert "priority" not in exp.data.data_vars
    # Check that the priority property can not be accessed
    msg = "Priorities are disabled."
    with pytest.raises(AttributeError, match=msg):
        _ = exp.priority

    # Enable the priority property
    exp.enable_priorities = True
    # Check that the priority is now in the data variables
    assert "priority" in exp.data.data_vars
    # Check that the priority property can be accessed
    assert isinstance(exp.priority, xr.DataArray)
    # Check that all values are 0
    assert (exp.priority.values == 0).all()
    # Check that the priority has attributes
    for attr in ["units", "long_name", "description"]:
        assert attr in exp.priority.attrs

    # Set the priority to a value
    loc = (slice(None),) * len(parameters)
    exp.priority.loc[loc] = 1
    priority = exp.priority

    # Disable the priority property
    exp.enable_priorities = False
    # Check that we can not access the priority property anymore
    with pytest.raises(AttributeError, match=msg):
        _ = exp.priority

    # Enable the priority property again and check that the priority is the same
    exp.enable_priorities = True
    assert exp.priority.equals(priority)

# ----------------------------------------------------------------
#  Custom arguments
# ----------------------------------------------------------------

@pytest.mark.parametrize(*("name, value", [
    pytest.param("test", 1, id="int"),
    pytest.param("test", 1.0, id="float"),
    pytest.param("test", 1.0 + 1j, id="complex"),
    pytest.param("test", "a", id="str"),
    pytest.param("test", True, id="bool"),
    pytest.param("test", MyObject(1), id="object"),
    pytest.param("test", None, id="None"),
]))
def test_valid_custom_arguments(name, value):
    """Test the custom_arguments property and the adding function."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3], "y": ["a", "b", "c"]},
    )
    # Check that the custom arguments are empty
    assert exp.custom_arguments == set()
    # Enable uuid and check that it is in the custom arguments
    exp.pass_uuid = True
    assert "uuid" in exp.custom_arguments
    # Disable uuid and check that it is not in the custom arguments
    exp.pass_uuid = False
    assert "uuid" not in exp.custom_arguments
    # Add a custom argument
    exp.add_custom_argument(name, value)
    # Check that the custom argument is in the custom arguments
    assert name in exp.custom_arguments
    # Check that a dataarray with the custom argument is in the data
    assert name in exp.data.data_vars
    # Check that the values are correct
    assert (exp.data[name].values == value).all()

@pytest.mark.parametrize("name", [
    "uuid", "duration", "priority", "status", "x", "existing"])
def test_invalid_custom_arguments(name):
    """Test the add_custom_argument function with invalid arguments."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3], "y": ["a", "b", "c"]},
    )
    exp.add_custom_argument("existing", 1)
    with pytest.raises(ValueError, match="is taken."):
        exp.add_custom_argument(name, 1)

# ----------------------------------------------------------------
#  Test data saving and loading
# ----------------------------------------------------------------

@pytest.mark.parametrize("mode", ["x", "w"])
def test_save(parameters, exp_func, save_path, request, mode):
    """Test saving the data."""
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        save_path=save_path,
    )
    # Check that the file does not exist
    assert not save_path.exists()
    # Save the data
    exp.save(mode)
    # Check that the file exists
    assert save_path.exists()

def test_load(parameters, exp_func, save_path, request):
    """Test loading the data."""
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        save_path=save_path,
    )
    # Save the data
    exp.save()
    # try to load the dataset
    ds = SweepExp.load(save_path)
    # Check that all variables exist
    for var in exp.data.variables:
        assert var in ds.variables

@pytest.mark.parametrize("invalid_file", ["test", "test.txt", "test.csv", "test.json"])
def test_invalid_file_format(invalid_file):
    """Test loading a file with an invalid format."""
    msg = "The file extension is not supported."

    # loading
    with pytest.raises(ValueError, match=msg):
        SweepExp.load(invalid_file)

    # saving
    exp = SweepExp(
        func=lambda: None,
        parameters={"a": [1]},
        save_path=invalid_file,
    )
    with pytest.raises(ValueError, match=msg):
        exp.save()

    # saving when no save path is set
    exp = SweepExp(
        func=lambda: None,
        parameters={"a": [1]},
    )
    msg = "The save path is not set. Set the save path before saving."
    with pytest.raises(ValueError, match=msg):
        exp.save()

def test_save_existing_data(parameters, exp_func, save_path, request):
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        save_path=save_path,
    )
    assert not save_path.exists()
    exp.save()
    # Check that the file exists
    assert save_path.exists()
    # Save the data with the default argument should raise an error
    msg = "There is already data at the save path."
    with pytest.raises(FileExistsError, match=msg):
        exp.save()
    # With mode="w" the file should be overwritten
    exp.save(mode="w")

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
    converted = SweepExp._convert_parameters({"a": para_in})["a"]
    assert converted.dtype is dtype

# ----------------------------------------------------------------
#  Test status updates
# ----------------------------------------------------------------

@pytest.mark.parametrize(*("states, expected_status", [
    pytest.param(None,
                 np.array([["N", "N", "S"],
                           ["N", "N", "S"],
                           ["S", "N", "N"]]),
                 id="default"),
    pytest.param("S",
                 np.array([["F", "N", "N"],
                           ["F", "N", "N"],
                           ["N", "C", "N"]]),
                 id="skip"),
    pytest.param(["F", "S"],
                 np.array([["N", "N", "N"],
                           ["N", "N", "N"],
                           ["N", "C", "N"]]),
                 id="finish and skip"),
]))
def test_reset_status(states, expected_status):
    """Test the reset_status function."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3], "y": ["a", "b", "c"]},
    )
    exp.status.values = np.array([["F", "N", "S"],
                                  ["F", "N", "S"],
                                  ["S", "C", "N"]])
    # Reset the status
    exp.reset_status(states)
    # Check that the status is as expected
    assert (exp.status.values == expected_status).all()

@pytest.mark.parametrize("states", ["X", "f", "s", "c", "n"])
def test_reset_status_invalid(states):
    """Test the reset_status function with invalid states."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3], "y": ["a", "b", "c"]},
    )
    # Reset the status with invalid states
    with pytest.raises(ValueError, match="Invalid states"):
        exp.reset_status(states)

# ----------------------------------------------------------------
#  Test the run helper functions
# ----------------------------------------------------------------

@pytest.mark.parametrize(*("status, expepcted_indices", [
    pytest.param("N", np.array([[0, 0, 0], [0, 1, 0]]), id="N"),
    pytest.param("S", np.array([[0, 1, 1], [0, 2, 0]]), id="S"),
    pytest.param("F", np.array([[0, 2, 1]]), id="F"),
    pytest.param("C", np.array([[0, 0, 1]]), id="C"),
    pytest.param(None, np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0],
                                 [0, 1, 1], [0, 2, 0], [0, 2, 1]]), id="all"),
    pytest.param(["F", "C"], np.array([[0, 0, 1], [0, 2, 1]]), id="F and C"),
]))
def test_get_indices(status, expepcted_indices):
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1.0], "y": ["a", "b", "c"], "z": [1, 2]},
    )
    # set the status
    exp.status.values = np.array([[["N", "C"], ["N", "S"], ["S", "F"]]])
    # get the indices
    indices = exp._get_indices(status)
    # check that the indices are as expected
    assert np.all(expepcted_indices == indices.T)

@pytest.mark.parametrize(*("with_priorities, expected_indices, first_kw", [
    pytest.param(True,
                 np.array([[0, 2, 0], [0, 0, 1], [0, 1, 0]]),
                 {"x": 1.0, "y": "c", "z": 1},
                 id="with priorities"),
    pytest.param(False,
                 np.array([[0, 0, 1], [0, 1, 0], [0, 2, 0]]),
                 {"x": 1.0, "y": "a", "z": 2},
                 id="without priorities"),
]))
def test_sort_indices(with_priorities, expected_indices, first_kw):
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1.0], "y": ["a", "b", "c"], "z": [1, 2]},
    )
    # set the priority
    exp.enable_priorities = True
    exp.priority.values = np.array([[[4, 2], [1, 5], [6, 3]]])
    exp.enable_priorities = with_priorities
    # get the indices
    indices = np.array([[0, 0, 1], [0, 1, 0], [0, 2, 0]]).T
    # sort the indices
    indices = exp._sort_indices(indices)
    # check that the indices are as expected
    assert np.all(expected_indices == indices.T)
    # get the first index and check that the correct kwargs are returned
    first_index = next(zip(*indices, strict=True))
    assert exp._get_kwargs(first_index) == first_kw

@pytest.mark.parametrize("ret_values", [
    pytest.param({"a": 1}, id="int"),
    pytest.param({"b": 1.0}, id="float"),
    pytest.param({"c": 1.0 + 1j}, id="complex"),
    pytest.param({"d": "a"}, id="str"),
    pytest.param({"e": False}, id="bool"),
    pytest.param({"f": np.linspace(0, 1, 10)}, id="np.ndarray"),
    pytest.param({"g": MyObject(1)}, id="object"),
    pytest.param({"a": 1, "b": 1.0}, id="int and float"),
])
def test_set_return_values(ret_values):
    """Test the _set_return_values function."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1.0], "y": ["a", "b", "c"], "z": [1, 2]},
    )
    exp._set_return_values_at((0, 1, 0), ret_values)
    # Check that the return values are as expected
    for key, value in ret_values.items():
        # check that the key is in the data variables
        assert key in exp.data.data_vars
        # check that the value is correct
        assert np.all(exp.data[key].values[0, 1, 0] == value)
        # check that the other values are nan
        assert np.all(exp.data[key].values[0, 0, 0] != value)

@pytest.mark.parametrize("arg", [
    pytest.param(1, id="int"),
    pytest.param(1.0, id="float"),
    pytest.param(1.0 + 1j, id="complex"),
    pytest.param("a", id="str"),
    pytest.param(True, id="bool"),
    pytest.param(MyObject(1), id="object"),
    pytest.param(None, id="None"),
])
def test_get_value_from_index(arg):
    exp = SweepExp(func=lambda: None, parameters={"x": [arg]})
    value = exp._get_value_from_index("x", (0, ))
    assert value == arg
    assert type(value) is type(arg)

@pytest.mark.parametrize(*("params, index, expected_kwargs", [
    pytest.param({"a": [1, 2, 3, 4]},
                 (0, ),
                 {"a": 1},
                 id="single parameter"),
    pytest.param({"a": [1, 2], "b": [1.0], "c": [1.0 + 1j],
                  "d": ["a"], "e": [True], "f": np.linspace(0, 1, 2)},
                 (1, 0, 0, 0, 0, 1),
                 {"a": 2, "b": 1.0, "c": 1.0 + 1j,
                  "d": "a", "e": True, "f": 1.0},
                 id="all types"),
    pytest.param({"g": [MyObject(1)], "h": [1, "a", True]},
                 (0, 1),
                 {"g": MyObject(1), "h": "a"},
                 id="objects"),
]))
def test_get_kwargs(params, index, expected_kwargs):
    """Test the _get_kwargs function."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters=params,
    )
    # Get the kwargs
    kwargs = exp._get_kwargs(index)
    assert isinstance(kwargs, dict)
    # Check that the kwargs are as expected
    assert kwargs == expected_kwargs

def test_get_kwargs_with_custom_arguments():
    """Test the _get_kwargs function with custom arguments."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"a": [1, 2, 3, 4]},
    )
    exp.add_custom_argument("test", 1)
    # Get the kwargs
    kwargs = exp._get_kwargs((0, ))
    assert isinstance(kwargs, dict)
    # Check that the kwargs are as expected
    assert kwargs == {"a": 1, "test": 1}
    # test with uuid
    exp.pass_uuid = True
    kwargs = exp._get_kwargs((2, ))
    assert isinstance(kwargs, dict)
    # Check that the kwargs are as expected
    assert kwargs == {"a": 3, "test": 1, "uuid": exp.uuid.values.flatten()[2]}

@pytest.mark.parametrize("dim_name", ["x", "result", "status"])
def test_add_new_xarray_dimension_from_returned_dataarray_invalid_name(dim_name):
    sweep = SweepExp( func=lambda x: 2*x, parameters={"x": [1, 2, 3]})
    sweep.run()

    success = sweep._add_new_xarray_dimension_from_returned_dataarray(
        dim_name=dim_name, coordinates=[1, 1.2, 1.7])
    assert not success, "Adding a new dimension with a taken name should fail"

def test_add_new_xarray_dimension_from_returned_dataarray_existing_dim():
    sweep = SweepExp( func=lambda x: 2*x, parameters={"x": [1, 2, 3]})
    sweep.run()

    success = sweep._add_new_xarray_dimension_from_returned_dataarray(
        dim_name="y", coordinates=[1, 1.2, 1.7])

    assert success, "Adding a new dimension with a new name should succeed"
    assert "y" in sweep.data.dims, "The new dimension should be added to the data"

    success = sweep._add_new_xarray_dimension_from_returned_dataarray(
        dim_name="y", coordinates=[1, 1.2, 1.7])
    assert success, "Should succeed as the cooridinates are the same"

    success = sweep._add_new_xarray_dimension_from_returned_dataarray(
        dim_name="y", coordinates=[1, 1.2, 1.8])
    assert not success, "Should fail as the coordinates are different"

    success = sweep._add_new_xarray_dimension_from_returned_dataarray(
        dim_name="y", coordinates=[1, 1.2, 1.7, 1.8])
    assert not success, "Should fail as the coordinates are different in length"

def test_add_xarray_dataarray():
    sweep = SweepExp( func=lambda x: 2*x, parameters={"x": [1, 2, 3]})
    sweep.run()

    u = xr.DataArray([0, 1, 2], dims=["nd"])
    sweep._add_xarray_dataarray("test", u)
    # check if the dataarray is added
    assert "nd" in sweep.data.dims
    assert "test" in sweep.data.data_vars
    # we set the data to 0, and try to add the dataarray again
    # it should not change the data
    assert not (sweep.data["test"].values == 0).all()
    sweep.data["test"].data[:] = 0
    sweep._add_xarray_dataarray("test", u)
    # check if the dataarray is still the same
    assert (sweep.data["test"].values == 0).all()

    # now we try to add a dataarray with a different shape but the same name
    u = xr.DataArray([0, 1, 2, 3], dims=["nd"])
    assert "test" not in sweep._invalid_names
    sweep._add_xarray_dataarray("test", u)
    assert "test" in sweep._invalid_names




@pytest.mark.parametrize(*("return_values, keys", [
    (1, ["result"]),
    ({"a": 1}, ["a"]),
    ({"a": 1, "b": 2}, ["a", "b"]),
    ((1, 3, 1), ["result_1", "result_2", "result_3"]),
]))
def test_process_return_values(return_values, keys):
    """Test the _process_return_values function."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3]},
    )
    processed = exp._process_return_values(return_values)
    assert isinstance(processed, dict)
    # Check that the key is in the processed dict
    for key in keys:
        assert key in processed

@pytest.mark.parametrize("return_value", [
    pytest.param([3, 2], id="list"),
    pytest.param((3, 2), id="tuple"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(xr.DataArray([3.1, 2]), id="DataArray"),
    pytest.param(xr.Dataset(), id="Dataset"),
])
def test_add_unsupported_return_type(return_value, caplog):
    """Test the _process_return_values function with an unsupported return type."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3]},
    )
    with caplog.at_level(logging.ERROR):
        _processed = exp._add_new_return_value("test", return_value)

    # Check that the error message is logged
    assert any("Unsupported return value type for" in msg for msg in caplog.messages)
    assert "test" in exp._invalid_names
    # Check that the return value is added
    assert "test" in exp.data.data_vars
    assert np.isnan(exp.data["test"].values).all()
    assert exp.data["test"].dtype == np.dtype("float64")

@pytest.mark.parametrize(*("value, dtype", [
    pytest.param(1, np.dtype("int64"), id="int"),
    pytest.param(1.0, np.dtype("float64"), id="float"),
    pytest.param(1.0 + 1j, np.dtype("complex128"), id="complex"),
    pytest.param("a", np.dtype(object), id="str"),
    pytest.param(True, np.dtype(bool), id="bool"),
    pytest.param(np.linspace(0, 1, 10), np.dtype(object), id="np.ndarray"),
    pytest.param(MyObject(1), np.dtype(object), id="object"),
]))
def test_add_new_return_value(value, dtype):
    """Test the _add_new_return_value function."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3]},
    )
    # Add a new return value
    exp._add_new_return_value("test", value)
    # Check that the return value is added
    assert "test" in exp.data.data_vars
    assert exp.data["test"].dtype == dtype

@pytest.mark.parametrize("used_name", ["duration", "priority", "status", "uuid", "x"])
def test_rename_return_value(used_name, caplog):
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3]},
    )
    # Add a new return value
    with caplog.at_level(logging.WARNING):
        exp._add_new_return_value(used_name, 1)
    # New name
    new_name = f"{used_name}_renamed"
    # Check that the return value is added
    assert new_name in exp.data.data_vars
    assert any("is already taken" in msg for msg in caplog.messages)

test_values = {"int": 1, "float": 1.0, "complex": 1.0 + 1j,
               "str": "a", "bool": True, "np": np.linspace(0, 1, 10),
               "object": MyObject(1)}
test_dtypes = {"int": np.dtype("int64"), "float": np.dtype("float64"),
               "complex": np.dtype("complex128"), "str": np.dtype(object),
               "bool": np.dtype(bool), "np": np.dtype(object),
               "object": np.dtype(object)}
cast_map = {
    "int": {"int": np.dtype("int64"), "float": np.dtype("float64"),
            "complex": np.dtype("complex128"), "str": np.dtype(object),
            "bool": np.dtype("int64"), "np": np.dtype(object),
            "object": np.dtype(object)},
    "float": {"int": np.dtype("float64"), "float": np.dtype("float64"),
              "complex": np.dtype("complex128"), "str": np.dtype(object),
              "bool": np.dtype("float64"), "np": np.dtype(object),
              "object": np.dtype(object)},
    "complex": {"int": np.dtype("complex128"), "float": np.dtype("complex128"),
                "complex": np.dtype("complex128"), "str": np.dtype(object),
                "bool": np.dtype("complex128"), "np": np.dtype(object),
                "object": np.dtype(object)},
    "str": {"int": np.dtype(object), "float": np.dtype(object),
            "complex": np.dtype(object), "str": np.dtype(object),
            "bool": np.dtype(object), "np": np.dtype(object),
            "object": np.dtype(object)},
    "bool": {"int": np.dtype("int64"), "float": np.dtype("float64"),
             "complex": np.dtype("complex128"), "str": np.dtype(object),
             "bool": np.dtype(bool), "np": np.dtype(object),
             "object": np.dtype(object)},
    "np": {"int": np.dtype(object), "float": np.dtype(object),
           "complex": np.dtype(object), "str": np.dtype(object),
           "bool": np.dtype(object), "np": np.dtype(object),
           "object": np.dtype(object)},
    "object": {"int": np.dtype(object), "float": np.dtype(object),
               "complex": np.dtype(object), "str": np.dtype(object),
               "bool": np.dtype(object), "np": np.dtype(object),
               "object": np.dtype(object)},
}
@pytest.mark.parametrize("or_type", test_dtypes.keys())
@pytest.mark.parametrize("value_type", test_dtypes.keys())
def test_upgrade_return_value_type(or_type, value_type):
    """Test the _upgrade_return_value_type function."""
    original_dtype = test_dtypes[or_type]
    oritinal_value = test_values[or_type]
    value = test_values[value_type]
    expected_dtype = cast_map[or_type][value_type]
    # Create the experiment
    exp = SweepExp(
        func=None,
        parameters={"x": [1, 2, 3]},
    )
    # Set the return value with the original dtype
    exp._set_return_value_at((0, ), "a", oritinal_value)
    assert exp.data["a"].dtype == original_dtype
    # set a new value with a different dtype
    exp._set_return_value_at((1, ), "a", value)
    # Check that the dtype is as expected
    assert exp.data["a"].dtype == expected_dtype

def test_run_single():
    """Test the _run_single function."""
    # Define a simple function
    def simple_func(x: int, y: MyObject) -> dict:
        return {"addition": x + y.value, "product": MyObject(x * y.value)}

    # Create the experiment
    exp = SweepExp(
        func=simple_func,
        parameters={"x": [1, 2, 3], "y": [MyObject(1), MyObject(2)]},
    )
    # Run the experiment
    exp._run_single((2, 0))
    # Check that the status is as expected
    assert exp.status.values[2, 0] == "C"
    # Check that the return values are as expected
    assert exp.data["addition"].values[2, 0] == 4  # noqa: PLR2004
    assert exp.data["product"].values[2, 0] == MyObject(3)

def test_run_api(caplog):
    exp = SweepExp(func=lambda x: {"a": 2*x}, parameters={"x": [1, 2]})
    with caplog.at_level(logging.WARNING):
        exp.run(status="N", max_workers=2)
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that the return values are as expected
    assert (exp.data["a"].values == [2, 4]).all()
    # Check that a warning was logged for the max_workers
    assert "max_workers" in caplog.text

@pytest.mark.parametrize("arg", [
    pytest.param(1, id="int"),
    pytest.param(1.0, id="float"),
    pytest.param(1.0 + 1j, id="complex"),
    pytest.param("a", id="str"),
    pytest.param(True, id="bool"),
    pytest.param(MyObject(1), id="object"),
    pytest.param(None, id="None"),
])
def test_argument_type(arg, caplog):
    """Test the _argument_type function."""
    def my_func(x: any) -> dict:
        assert x == arg
        assert type(x) is type(arg)

    # Run the sweep
    with caplog.at_level("DEBUG"):
        data = SweepExp(func=my_func, parameters={"x": [arg]}).run()

    # Fail the test if any ERROR log was recorded
    assert not any(record.levelname == "ERROR" for record in caplog.records), (
        "Errors were logged: "
        f"{[r.message for r in caplog.records if r.levelname == 'ERROR']}"
    )

    assert (data.status == "C").all()
    kwargs = {"x": arg}
    assert f"Running experiment with kwargs: {kwargs}" in caplog.text


# ----------------------------------------------------------------
#  Test the run function
# ----------------------------------------------------------------

def test_standard_run():
    # Define a simple function
    def simple_func(x: int, y: MyObject) -> dict:
        return {"addition": x + y.value, "product": MyObject(x * y.value)}

    # Create the experiment
    exp = SweepExp(
        func=simple_func,
        parameters={"x": [1, 2, 3], "y": [MyObject(1), MyObject(2)]},
    )
    # Check that the status is not started
    assert (exp.status.values == "N").all()
    # Run the experiment
    exp.run()
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that the return values are as expected
    assert (exp.data["addition"].values == [[2, 3], [3, 4], [4, 5]]).all()
    assert (exp.data["product"].values == [[MyObject(1), MyObject(2)],
                                           [MyObject(2), MyObject(4)],
                                           [MyObject(3), MyObject(6)]]).all()

def test_complex_run():
    """Test the run function with a more complex setup."""
    # Define a function that returns a dictionary with multiple values
    def complex_func(x: int, y: MyObject) -> dict:
        changed_variable = 1 if x == 1 else "a"  # variable that changes type

        return {
            "normal_dtype": x + y.value,
            "object": MyObject(x * y.value),
            "changed_variable": changed_variable,
            "unsupported": [1, 3, 4],  # list are not supported
            "duration": x * 3,  # this variable will be renamed
            "none_value": None,
            "x": True,  # this variable will be renamed
        }

    # Create the experiment
    exp = SweepExp(
        func=complex_func,
        parameters={"x": [1, 2, 3], "y": [MyObject(1), MyObject(2)]},
    )
    # Run the experiment
    exp.run()
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that all variables are in the return values and data
    for key in ["normal_dtype", "object", "changed_variable",
                "unsupported", "duration_renamed", "none_value", "x_renamed"]:
        assert key in exp.data.data_vars
    # Check that the data types are as expected
    assert exp.data["normal_dtype"].dtype == np.dtype("int64")
    assert exp.data["object"].dtype == np.dtype(object)
    assert exp.data["changed_variable"].dtype == np.dtype(object)
    assert exp.data["unsupported"].dtype == np.dtype("float64")
    assert exp.data["duration_renamed"].dtype == np.dtype("int64")
    assert exp.data["none_value"].dtype == np.dtype(object)
    assert exp.data["x_renamed"].dtype == np.dtype("bool")

def test_run_with_uuid(temp_dir, method):
    # Create a function that takes the uuis an an argument and write
    # something to a file with the uuid in the name
    def my_experiment(x: int, uuid: str) -> dict:
        with open(f"{temp_dir}/output_{uuid}.txt", "w") as file:  # noqa: PTH123
            file.write(f"Experiment with x={x} and uuid={uuid}.")
        return {}

    if method == "kwarg":
        # Create the experiment with uuid enabled via kwargs
        sweep = SweepExp(
            func=my_experiment,
            parameters={"x": [1, 2, 3]},
            pass_uuid=True,  # Enable uuid via kwargs
        )
    else:
        sweep = SweepExp(
            func=my_experiment,
            parameters={"x": [1, 2, 3]},
        )
        sweep.pass_uuid = True

    # Run the sweep
    sweep.run()
    # Check that the three files were created
    for i in range(3):
        uuid = sweep.uuid.values.flatten()[i]
        assert (temp_dir / f"output_{uuid}.txt").exists()
        with open(f"{temp_dir}/output_{uuid}.txt") as file:  # noqa: PTH123
            assert file.read() == f"Experiment with x={i+1} and uuid={uuid}."

def test_run_with_timeit(method):
    # define a function that takes some time
    def slow_func(wait_time: float) -> dict:
        time.sleep(wait_time)
        return {}
    # Create the experiment
    if method == "kwarg":
        exp = SweepExp(
            func=slow_func,
            parameters={"wait_time": [0.3, 0.6, 0.9]},
            timeit=True,  # Enable timeit via kwargs
        )
    else:
        exp = SweepExp(
            func=slow_func,
            parameters={"wait_time": [0.3, 0.6, 0.9]},
        )
        exp.timeit = True
    # Run the experiment
    exp.run()
    # Check that the duration is not nan
    assert not np.isnan(exp.duration.values).all()
    # Check that the duration is as expected
    tolerance = 0.1
    assert np.allclose(exp.duration.values, [0.3, 0.6, 0.9], atol=tolerance)

def test_run_with_failures():
    def fail_func(should_fail: bool) -> dict:  # noqa: FBT001
        if should_fail:
            raise ValueError
        return {}
    # Create the experiment
    exp = SweepExp(
        func=fail_func,
        parameters={"should_fail": [False, True]},
    )
    # Run the experiment
    exp.run()
    # Check that the status is as expected
    assert (exp.status.values == [["C", "F"]]).all()

def test_run_with_custom_arguments():
    def custom_func(para1: int, custom: float) -> dict:
        return {"res": para1 + custom}

    # Create the experiment
    exp = SweepExp(
        func=custom_func,
        parameters={"para1": [1, 2, 3]},
    )

    # Add a custom argument
    exp.add_custom_argument("custom", 1.0)
    # Set the custom argument
    exp.data["custom"].data = np.array([1.0, 2.0, 3.0])
    # Run the experiment
    exp.run()
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that the return values are as expected
    assert (exp.data["res"].values == [2.0, 4.0, 6.0]).all()

def test_run_with_auto_save(save_path, method):
    if method == "kwarg":
        # Create the experiment with auto_save enabled via kwargs
        exp = SweepExp(
            func=lambda x: {"res": 2 * x},
            parameters={"x": [1, 2, 3]},
            save_path=save_path,
            auto_save=True,
        )
    else:
        exp = SweepExp(
            func=lambda x: {"res": 2 * x},
            parameters={"x": [1, 2, 3]},
            save_path=save_path,
        )
        exp.auto_save = True

    # modify the save method to check if it is called
    exp.save = MagicMock(wraps=exp.save)
    exp.run()
    # check that the save method was called
    assert exp.save.called
    # check that the method was called three times
    assert exp.save.call_count == len(exp.data["res"].values.flatten())

def test_run_continue(save_path):
    exp = SweepExp(
        func=lambda x: {"res": 2 * x, "x": 10 * x},
        parameters={"x": [1, 2, 3]},
        save_path=save_path,
        timeit=True,
        auto_save=True,
    )

    exp.status.loc[{"x": 2}] = "S"

    # Run the experiment
    exp.run()

    assert exp._name_mapping == {"x": "x_renamed"}

    # reload the experiment with a modified function
    exp2 = SweepExp(
        func=lambda x: {"res": 20 * x, "x": 1 * x},
        parameters={"x": [1, 2, 3]},
        save_path=save_path,
    )

    # check that timeit and auto_save are still enabled
    assert exp2.timeit
    assert exp2.auto_save

    exp2.run("S")
    assert exp2._name_mapping == {"x": "x_renamed"}

    # Check that the status is as expected
    assert (exp2.status.values == "C").all()
    assert (exp2.data["res"].values == [2, 40, 6]).all()
    assert (exp2.data["x_renamed"].values == [10, 2, 30]).all()
