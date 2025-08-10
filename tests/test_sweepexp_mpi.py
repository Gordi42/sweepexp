"""
Test the SweepExpParallel class.

Description
-----------
This test should be run with mpi. For example with srun:

.. code-block:: bash

    srun -n 2 pytest tests/test_sweepexp_mpi.py

Or with mpirun:

.. code-block:: bash

    mpirun -n 2 pytest tests/test_sweepexp_mpi.py

"""
from __future__ import annotations

import pytest

# Try to import mpi4py and SweetExpMPI
try:
    from mpi4py import MPI
except ImportError:
    # Skip the test if mpi4py is not available
    pytest.skip("mpi4py is not available", allow_module_level=True)

import time
from unittest.mock import MagicMock

import numpy as np
import xarray as xr

from sweepexp import SweepExpMPI


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

@pytest.fixture
def tmp_dir(temp_dir):
    # Make sure that the temporary directory is the same on all ranks
    return MPI.COMM_WORLD.bcast(temp_dir, root=0)

@pytest.fixture(params=[".pkl", ".zarr", ".nc"])
def save_path(tmp_dir, request):
    return tmp_dir / f"test{request.param}"

# ================================================================
#  Tests
# ================================================================

@pytest.mark.mpi(min_size=2)
def test_mpi_world_size():
    """Test the number of ranks."""
    size = MPI.COMM_WORLD.Get_size()
    min_size = 2
    assert size >= min_size

@pytest.mark.parametrize("arg", [
    pytest.param(1, id="int"),
    pytest.param(1.0, id="float"),
    pytest.param(1.0 + 1j, id="complex"),
    pytest.param("a", id="str"),
    pytest.param(True, id="bool"),
    pytest.param(MyObject(1), id="object"),
    pytest.param(None, id="None"),
])
@pytest.mark.mpi(min_size=2)
def test_argument_type(arg, caplog):
    """Test the _argument_type function."""
    def my_func(x: any) -> dict:
        assert x == arg
        assert type(x) is type(arg)

    # Run the sweep
    with caplog.at_level("DEBUG"):
        data = SweepExpMPI(func=my_func, parameters={"x": [arg]}).run()

    # Fail the test if any ERROR log was recorded
    assert not any(record.levelname == "ERROR" for record in caplog.records), (
        "Errors were logged: "
        f"{[r.message for r in caplog.records if r.levelname == 'ERROR']}"
    )

    if MPI.COMM_WORLD.Get_rank() == 0:
        assert (data.status == "C").all()
    elif isinstance(arg, MyObject):
        pass  # The Address of the object is different on each rank
    elif MPI.COMM_WORLD.Get_rank() == 1:
        kwargs = {"x": arg}
        assert f"Rank 1 - Starting: {kwargs}" in caplog.text

@pytest.mark.mpi(min_size=2)
def test_run_api(caplog):
    exp = SweepExpMPI(func=lambda x: {"a": 2*x}, parameters={"x": [1, 2]})
    with caplog.at_level("WARNING"):
        exp.run(status="N", max_workers=2)

    # Validate only on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that the return values are as expected
    assert (exp.data["a"].values == [2, 4]).all()
    # Check that a warning was logged for the max_workers
    assert "max_workers" in caplog.text

@pytest.mark.mpi(min_size=2)
def test_standard_run():
    # Define a simple function
    def simple_func(x: int, y: MyObject) -> dict:
        return {"addition": x + y.value, "product": MyObject(x * y.value)}

    # Create the experiment
    exp = SweepExpMPI(
        func=simple_func,
        parameters={"x": [1, 2, 3], "y": [MyObject(1), MyObject(2)]},
    )
    # Check that the status is not started
    assert (exp.status.values == "N").all()
    # Run the experiment
    exp.run()
    # Validate only on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that the return values are as expected
    assert (exp.data["addition"].values == [[2, 3], [3, 4], [4, 5]]).all()
    assert (exp.data["product"].values == [[MyObject(1), MyObject(2)],
                                           [MyObject(2), MyObject(4)],
                                           [MyObject(3), MyObject(6)]]).all()

@pytest.mark.mpi(min_size=2)
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
            "data_arr": xr.DataArray([y.value, -x], coords={"d2": [0.0, 1.1]}),
        }

    # Create the experiment
    exp = SweepExpMPI(
        func=complex_func,
        parameters={"x": [1, 2, 3], "y": [MyObject(1), MyObject(2)]},
    )
    # Run the experiment
    exp.run()
    # Validate only on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that all variables are in the return values and data
    for key in ["normal_dtype", "object", "changed_variable", "unsupported",
                "duration_renamed", "none_value", "x_renamed", "data_arr"]:
        assert key in exp.data.data_vars
    # Check that the data types are as expected
    assert exp.data["normal_dtype"].dtype == np.dtype("int64")
    assert exp.data["object"].dtype == np.dtype(object)
    assert exp.data["changed_variable"].dtype == np.dtype(object)
    assert exp.data["unsupported"].dtype == np.dtype("float64")
    assert exp.data["duration_renamed"].dtype == np.dtype("int64")
    assert exp.data["none_value"].dtype == np.dtype(object)
    assert exp.data["x_renamed"].dtype == np.dtype("bool")
    # check that the DataArray is added correctly
    assert set(exp.data["data_arr"].dims) == {"x", "y", "d2"}
    assert exp.data["data_arr"].dtype == np.dtype("int64")
    assert np.allclose(exp.data["data_arr"].sel({"x": 1, "d2": 0.0}).values, [1, 2])

@pytest.mark.mpi(min_size=2)
def test_run_with_xarray_dataarray(caplog):
    # Define a function that returns a DataArray
    def my_experiment(x: int, y: float) -> xr.DataArray:
        return {
            # valid return values
            "result": xr.DataArray([x, x + 1, y], dims=["d1"]),
            "x": xr.DataArray([y, y + 1, x], dims=["d1"], attrs={"units": "m"}),
            "status": xr.DataArray([y, -x], coords={"d2": [0.0, 1.1]}),
            "object": xr.DataArray([MyObject(x), MyObject(y)],
                                   coords={"d2": [0.0, 1.1]}),
            "valid4": xr.DataArray(
                [[x, x + 1], [y, y + 1], [x + y, x - y]],
                dims=["d1", "d2"],
                coords={"d2": [0.0, 1.1]}),
            # invalid return values
            "invalid1": xr.DataArray([x, y], dims=["x"]),  # wrong dimensions
            "invalid2": xr.DataArray([x, y], dims=["d1"]),  # dimension mismatch
        }

    # Create the experiment
    exp = SweepExpMPI(
        func=my_experiment,
        parameters={"x": [1, 2, 3], "y": [1.0, 2.0]},
    )
    with caplog.at_level("WARNING"):
        exp.run()

    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    # Check that the status is as expected
    assert (exp.status.values == "C").all()

    # Check if the coordinates and dimensions of the dataset are as expected
    assert set(exp.data.dims) == {"x", "y", "d1", "d2"}
    assert set(exp.data.coords) == {"x", "y", "d1", "d2"}
    assert np.allclose(exp.data["d1"].values, [0, 1, 2])
    assert np.allclose(exp.data["d2"].values, [0.0, 1.1])

    # Check if the coordinates of the DataArrays are as expected
    assert set(exp.data["result"].dims) == {"x", "y", "d1"}
    assert set(exp.data["x_renamed"].dims) == {"x", "y", "d1"}
    assert set(exp.data["status_renamed"].dims) == {"x", "y", "d2"}
    assert set(exp.data["valid4"].dims) == {"x", "y", "d1", "d2"}

    # Check that all DataArrays have the correct values
    for key in ["result", "x_renamed", "status_renamed", "valid4"]:
        assert not exp.data[key].isnull().any(), f"{key} should not contain NaN values"
    # Check the values of the DataArrays
    assert np.allclose(exp.data["result"].sel({"x": 1, "y": 2.0}).values, [1, 2, 2])
    assert np.allclose(exp.data["x_renamed"].sel({"x": 2, "y": 1.0}).values, [1, 2, 2])
    assert np.allclose(exp.data["status_renamed"].sel({"x": 3, "y": 1.0}).values, [1, -3])
    assert np.allclose(exp.data["valid4"].sel({"x": 2, "y": 1.0}).values,
                          [[2, 3], [1, 2], [3, 1]])

@pytest.mark.mpi(min_size=2)
def test_run_with_uuid(tmp_dir):
    # Create a function that takes the uuis an an argument and write
    # something to a file with the uuid in the name
    def my_experiment(x: int, uuid: str) -> dict:
        with open(f"{tmp_dir}/output_{uuid}.txt", "w") as file:  # noqa: PTH123
            file.write(f"Experiment with x={x} and uuid={uuid}.")
        return {}

    sweep = SweepExpMPI(
        func=my_experiment,
        parameters={"x": [1, 2, 3]},
    )

    # Enable the uuid
    sweep.pass_uuid = True
    # Run the sweep
    sweep.run()
    # Validate only on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # Check that the three files were created
    for i in range(3):
        uuid = sweep.uuid.values.flatten()[i]
        assert (tmp_dir / f"output_{uuid}.txt").exists()
        with open(f"{tmp_dir}/output_{uuid}.txt") as file:  # noqa: PTH123
            assert file.read() == f"Experiment with x={i+1} and uuid={uuid}."

@pytest.mark.mpi(min_size=2)
def test_run_with_timeit():
    # define a function that takes some time
    def slow_func(wait_time: float) -> dict:
        time.sleep(wait_time)
        return {}
    # Create the experiment
    exp = SweepExpMPI(
        func=slow_func,
        parameters={"wait_time": [0.3, 0.6, 0.9]},
    )
    # Enable the timeit property
    exp.timeit = True
    # Run the experiment
    exp.run()
    # Validate only on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # Check that the duration is not nan
    assert not np.isnan(exp.duration.values).all()
    # Check that the duration is as expected
    tolerance = 0.1
    assert np.allclose(exp.duration.values, [0.3, 0.6, 0.9], atol=tolerance)

@pytest.mark.mpi(min_size=3)
def test_run_speed():
    """Test if the parallel run is faster than the serial run."""
    # Define a simple function that takes some time
    wait_time = 1.0
    def slow_func(_uselss: int) -> dict:
        time.sleep(wait_time)
        return {}

    # Create the experiment
    exp = SweepExpMPI(
        func=slow_func,
        parameters={"_uselss": [0, 1]},
    )
    # Run the experiment in parallel
    start = time.time()
    exp.run()

    parallel_duration = time.time() - start
    tolerance = 0.3
    # Check that the parallel run took roughly the same time as the wait time
    assert np.isclose(parallel_duration, wait_time, atol=tolerance)

@pytest.mark.mpi(min_size=2)
def test_run_with_failures():
    def fail_func(should_fail: bool) -> dict:  # noqa: FBT001
        if should_fail:
            raise ValueError
        return {}
    # Create the experiment
    exp = SweepExpMPI(
        func=fail_func,
        parameters={"should_fail": [False, True]},
    )
    # Run the experiment
    exp.run()
    # Only validate on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # Check that the status is as expected
    assert (exp.status.values == [["C", "F"]]).all()

@pytest.mark.mpi(min_size=2)
def test_run_with_custom_arguments():
    def custom_func(para1: int, custom: float) -> dict:
        return {"res": para1 + custom}

    # Create the experiment
    exp = SweepExpMPI(
        func=custom_func,
        parameters={"para1": [1, 2, 3]},
    )

    # Add a custom argument
    exp.add_custom_argument("custom", 1.0)
    # Set the custom argument
    exp.data["custom"].data = np.array([1.0, 2.0, 3.0])
    # Run the experiment
    exp.run()
    # Validate only on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return
    # Check that the status is as expected
    assert (exp.status.values == "C").all()
    # Check that the return values are as expected
    assert (exp.data["res"].values == [2.0, 4.0, 6.0]).all()

@pytest.mark.mpi(min_size=2)
def test_run_with_auto_save(save_path):
    def complex_func(x: int) -> dict:
        return {
            "res": 2*x,
            "unsupported": [1, 3, 4],  # list are not supported
            "duration": x * 3.1,  # this variable will be renamed
            "none_value": None,
            "x": True,  # this variable will be renamed
            "data_arr": xr.DataArray([x, -x], coords={"d2": [0.0, 1.1]}),
        }

    exp = SweepExpMPI(
        func=complex_func,
        parameters={"x": [1, 2, 3]},
        save_path=save_path,
        auto_save=True,
    )

    # modify the save method to check if it is called
    exp.save = MagicMock(wraps=exp.save)
    exp.run()

    # Validate only on the main rank
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    exp2 = SweepExpMPI(
        func=complex_func,
        parameters={"x": [1, 2, 3]},
        save_path=save_path)

    # check that the save method was called
    assert exp.save.called
    # check that the method was called three times
    assert exp.save.call_count == len(exp.data["res"].values.flatten())
    for ex in [exp, exp2]:
        # Check that the status is as expected
        assert (ex.status.values == "C").all()
        # Check that all variables are in the data
        for key in ["res", "unsupported", "x_renamed", "duration_renamed",
                    "none_value", "data_arr"]:
            assert key in ex.data.data_vars
        # Check that the data is as expected
        assert np.allclose(ex.data["res"].values, [2, 4, 6])
        assert ex.data["unsupported"].isnull().all().item()
        assert ex.data["x_renamed"].dtype == np.dtype("bool")
        assert np.allclose(ex.data["duration_renamed"].values, [3.1, 6.2, 9.3])
        assert ex.data["none_value"].isnull().all().item()
        assert set(ex.data["data_arr"].dims) == {"x", "d2"}
        assert np.allclose(ex.data["data_arr"].values, [[1, -1], [2, -2], [3, -3]])

@pytest.mark.mpi(min_size=2)
def test_run_with_existing_file(save_path):
    # Create the experiment, run it and save it
    exp = SweepExpMPI(
        func=lambda x: {"res": 2 * x},
        parameters={"x": [1, 2, 3]},
        save_path=save_path,
    )
    # Run the experiment
    exp.run()
    exp.save()
    # Wait for all ranks to finish
    MPI.COMM_WORLD.barrier()

    # Check that the file was created
    assert save_path.exists()

    # Create a new experiment with the same save path and check if the data is loaded
    exp2 = SweepExpMPI(
        func=lambda x: {"res": 2 * x},
        parameters={"x": [1, 2, 3]},
        save_path=save_path,
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert (exp2.data["res"].values == [2, 4, 6]).all()

@pytest.mark.mpi(min_size=2)
def test_run_continue(save_path):
    def complex_func1(x: int) -> dict:
        return {
            "res": 2*x,
            "x": True,  # this variable will be renamed
            "unsupported": [1, 3, 4],  # list are not supported
            "duration": x * 3.1,  # this variable will be renamed
            "none_value": None,
            "data_arr": xr.DataArray([x, -x], coords={"d2": [0.0, 1.1]}),
        }

    exp = SweepExpMPI(
        func=complex_func1,
        parameters={"x": [1, 2, 3]},
        save_path=save_path,
        timeit=True,
        auto_save=True,
    )

    exp.status.loc[{"x": 2}] = "S"

    # Run the experiment
    exp.run()

    # reload the experiment with a modified function
    def complex_func2(x: int) -> dict:
        return {
            "res": 20*x,
            "x": False,  # this variable will be renamed
            "unsupported": [1, 3, 4],  # list are not supported
            "duration": x * 2.9,  # this variable will be renamed
            "none_value": None,
            "data_arr": xr.DataArray([-x, x], coords={"d2": [0.0, 1.1]}),
        }
    exp2 = SweepExpMPI(
        func=complex_func2,
        parameters={"x": [1, 2, 3]},
        save_path=save_path,
    )
    exp2.run("S")

    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    assert "x" in exp._name_mapping
    assert "unsupported" in exp._invalid_names

    # check that timeit and auto_save are still enabled
    assert exp2.timeit
    assert exp2.auto_save

    assert "x" in exp2._name_mapping

    # Check that the data is as expected
    assert (exp2.status.values == "C").all()
    assert (exp2.data["res"].values == [2, 40, 6]).all()
    assert (exp2.data["x_renamed"].values == [True, False, True]).all()
    assert (exp2.data["duration_renamed"].values == [3.1, 5.8, 9.3]).all()
    assert exp2.data["none_value"].isnull().all().item()
    assert set(exp2.data["data_arr"].dims) == {"x", "d2"}
    assert np.allclose(exp2.data["data_arr"].values, [[1, -1], [-2, 2], [3, -3]])