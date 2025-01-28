"""Test the SweepExp class."""
from __future__ import annotations

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

# ================================================================
#  Tests
# ================================================================

# ----------------------------------------------------------------
#  Test initialization
# ----------------------------------------------------------------
def test_init_no_file(parameters, return_dict, exp_func):
    """Test the initialization of the SweepExp class without a file."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
    )
    assert isinstance(exp, SweepExp)

def test_init_with_nonexistent_file(parameters, return_dict, exp_func, save_path):
    """Test the initialization of the SweepExp class with a nonexistent file."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    )
    assert isinstance(exp, SweepExp)

def test_init_with_valid_existing_file(
        parameters, return_dict, exp_func, save_path, request):
    """Test the initialization of the SweepExp class with a valid existing file."""
    # Skip the test if objects are present (since they cannot be saved)
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")

    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    )
    # Modify some properties
    loc = (slice(None),) * len(parameters)
    exp.status.loc[loc] = "S"
    # get the first name of the return dict
    if return_dict.keys():
        name = next(iter(return_dict.keys()))
        exp.data[name].loc[loc] = 1
    # Save the data
    exp.save()

    # Create a new experiment with the same file
    sweep = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    )

    # Check that the experiment was loaded correctly
    assert isinstance(sweep, SweepExp)
    # Check that the changes are present
    assert (sweep.status.values == "S").any()
    if return_dict.keys():
        assert (sweep.data[name].values == 1).any()

@pytest.mark.parametrize(*("para, ret, msg", [
    pytest.param({"extra": [1]}, {},
                 "Parameter mismatch", id="extra parameter"),
    pytest.param({"int": [1, 3]}, {},
                 "Parameter mismatch", id="different parameter values (int)"),
    pytest.param({"bool": [False]}, {},
                    "Parameter mismatch", id="different parameter values (bool)"),
    pytest.param({"float": [1.01/3 + 1e-4]}, {},
                    "Parameter mismatch", id="different parameter values (float)"),
    pytest.param({"str": ["b"]}, {},
                  "Parameter mismatch", id="different parameter values (str)"),
    pytest.param({"np": np.linspace(0, 1.1, 2)}, {},
                  "Parameter mismatch", id="different parameter values (np)"),
    pytest.param({}, {"extra": int},
                 "Return value mismatch", id="extra return value"),
]))
def test_init_with_invalid_existing_file(para, ret, msg, save_path):
    """Test the initialization of the SweepExp class with an invalid existing file."""
    parameters = {"int": [1, 2], "bool": [True], "float": [1.01/3], "str": ["a"],
                  "np": np.linspace(0, 1, 2)}
    return_dict = {"r_int": int, "r_bool": bool, "r_float": float, "r_str": str}
    # Create the experiment
    SweepExp(
        func=lambda: None,  # dummy function (does not matter here)
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    ).save()

    parameters.update(para)
    return_dict.update(ret)

    with pytest.raises(ValueError, match=msg):
        SweepExp(
            func=lambda: None,  # dummy function (does not matter here)
            parameters=parameters,
            return_values=return_dict,
            save_path=save_path,
        )

@pytest.mark.parametrize(*("parameters, return_dict, msg", [
    pytest.param({"status": [1]}, {}, "parameter",
                 id="status in parameters"),
    pytest.param({}, {"status": int}, "return value",
                 id="status in return values"),
]))
def test_init_reserved_names(parameters, return_dict, msg):
    """Test the initialization of the SweepExp class with reserved names."""
    # Create the experiment
    with pytest.raises(ValueError, match=msg):
        SweepExp(
            func=lambda: None,  # dummy function (does not matter here)
            parameters=parameters,
            return_values=return_dict,
        )

# ----------------------------------------------------------------
#  Test properties
# ----------------------------------------------------------------

def test_properties_get(parameters, return_dict, exp_func):
    """Test the properties of the SweepExp class."""
    # Create the experiment
    exp = SweepExp(
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
    assert isinstance(exp.status, xr.DataArray)

    # Check the content of the xarray dataarrays
    # All status values should be "not started"
    assert all(exp.status.values.flatten() == "N")

def test_properties_set(parameters, return_dict, exp_func):
    """Test setting the properties of the SweepExp class."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
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
    readonly_properties = ["func", "parameters", "return_values", "save_path",
                           "data", "status"]
    for prop in readonly_properties:
        with pytest.raises(AttributeError):
            setattr(exp, prop, None)

def test_uuid(parameters, return_dict, exp_func):
    """Test the uuid property."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
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

def test_duration(parameters, return_dict, exp_func):
    """Test the duration property."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
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
    for attr in ["units", "long_name", "description"]:
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

def test_priority_property(parameters, return_dict, exp_func):
    """Test the priority property."""
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
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
        return_values={},
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

@pytest.mark.parametrize(*("name, msg", [
    pytest.param("uuid", "reserved"),
    pytest.param("duration", "reserved"),
    pytest.param("priority", "reserved"),
    pytest.param("status", "reserved"),
    pytest.param("x", "parameter"),
    pytest.param("existing", "already a custom"),
]))
def test_invalid_custom_arguments(name, msg):
    """Test the add_custom_argument function with invalid arguments."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1, 2, 3], "y": ["a", "b", "c"]},
        return_values={},
    )
    exp.add_custom_argument("existing", 1)
    with pytest.raises(ValueError, match=msg):
        exp.add_custom_argument(name, 1)

# ----------------------------------------------------------------
#  Test data saving and loading
# ----------------------------------------------------------------

@pytest.mark.parametrize("mode", ["x", "w"])
def test_save(parameters, return_dict, exp_func, save_path, request, mode):  # noqa: PLR0913
    """Test saving the data."""
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
        save_path=save_path,
    )
    # Check that the file does not exist
    assert not save_path.exists()
    # Save the data
    exp.save(mode)
    # Check that the file exists
    assert save_path.exists()

def test_load(parameters, return_dict, exp_func, save_path, request):
    """Test loading the data."""
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
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
        return_values={},
        save_path=invalid_file,
    )
    with pytest.raises(ValueError, match=msg):
        exp.save()

    # saving when no save path is set
    exp = SweepExp(
        func=lambda: None,
        parameters={"a": [1]},
        return_values={},
    )
    msg = "The save path is not set. Set the save path before saving."
    with pytest.raises(ValueError, match=msg):
        exp.save()

def test_save_existing_data(parameters, return_dict, exp_func, save_path, request):
    skip = request.node.get_closest_marker("objects")
    if skip is not None and save_path.suffix in [".zarr", ".nc"]:
        pytest.skip("Skipping test with objects")
    # Create the experiment
    exp = SweepExp(
        func=exp_func,
        parameters=parameters,
        return_values=return_dict,
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
    assert save_path.exists()

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
    converted = SweepExp._convert_return_types({"a": type_in})["a"]
    assert converted is type_out

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
        return_values={},
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
        return_values={},
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
        return_values={},
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
        return_values={},
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

@pytest.mark.parametrize(*("ret_dict, ret_values", [
    pytest.param({"a": int}, {"a": 1}, id="int"),
    pytest.param({"b": float}, {"b": 1.0}, id="float"),
    pytest.param({"c": complex}, {"c": 1.0 + 1j}, id="complex"),
    pytest.param({"d": str}, {"d": "a"}, id="str"),
    pytest.param({"e": bool}, {"e": False}, id="bool"),
    pytest.param({"f": np.ndarray}, {"f": np.linspace(0, 1, 10)}, id="np.ndarray"),
    pytest.param({"g": object}, {"g": MyObject(1)}, id="object"),
    pytest.param({"a": int, "b": float}, {"a": 1, "b": 1.0}, id="int and float"),
]))
def test_set_return_values(ret_dict, ret_values):
    """Test the _set_return_values function."""
    # Create the experiment
    exp = SweepExp(
        func=lambda: None,
        parameters={"x": [1.0], "y": ["a", "b", "c"], "z": [1, 2]},
        return_values=ret_dict,
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
        return_values={},
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
        return_values={},
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

def test_run_single():
    """Test the _run_single function."""
    # Define a simple function
    def simple_func(x: int, y: MyObject) -> dict:
        return {"addition": x + y.value, "product": MyObject(x * y.value)}

    # Create the experiment
    exp = SweepExp(
        func=simple_func,
        parameters={"x": [1, 2, 3], "y": [MyObject(1), MyObject(2)]},
        return_values={"addition": int, "product": object},
    )
    # Run the experiment
    exp._run_single((2, 0))
    # Check that the status is as expected
    assert exp.status.values[2, 0] == "C"
    # Check that the return values are as expected
    assert exp.data["addition"].values[2, 0] == 4  # noqa: PLR2004
    assert exp.data["product"].values[2, 0] == MyObject(3)

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
        return_values={"addition": float, "product": object},
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

def test_run_with_uuid(temp_dir):
    # Create a function that takes the uuis an an argument and write
    # something to a file with the uuid in the name
    def my_experiment(x: int, uuid: str) -> dict:
        with open(f"{temp_dir}/output_{uuid}.txt", "w") as file:  # noqa: PTH123
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
    # Check that the three files were created
    for i in range(3):
        uuid = sweep.uuid.values.flatten()[i]
        assert (temp_dir / f"output_{uuid}.txt").exists()
        with open(f"{temp_dir}/output_{uuid}.txt") as file:  # noqa: PTH123
            assert file.read() == f"Experiment with x={i+1} and uuid={uuid}."

def test_run_with_timeit():
    # define a function that takes some time
    def slow_func(wait_time: float) -> dict:
        time.sleep(wait_time)
        return {}
    # Create the experiment
    exp = SweepExp(
        func=slow_func,
        parameters={"wait_time": [0.3, 0.6, 0.9]},
        return_values={},
    )
    # Enable the timeit property
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
        return_values={},
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
        return_values={"res": float},
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

def test_run_with_auto_save(save_path):

    exp = SweepExp(
        func=lambda x: {"res": 2 * x},
        parameters={"x": [1, 2, 3]},
        return_values={"res": int},
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
