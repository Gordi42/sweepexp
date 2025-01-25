"""Test the SweepExp class."""
from __future__ import annotations

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

def test_save(parameters, return_dict, exp_func, save_path, request):
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
    exp.save()
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
#  Test the run function
# ----------------------------------------------------------------

def test_standard_run(): ...

def test_run_with_uuid(): ...

def test_run_with_timeit(): ...

def test_run_with_priorities(): ...

def test_run_with_failures(): ...

def test_run_with_custom_arguments(): ...
