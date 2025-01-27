"""Test the SweepExpParallel class."""

from __future__ import annotations

import multiprocessing as mp
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from sweepexp import SweepExpParallel


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

@pytest.fixture(params=[".pkl", ".zarr", ".nc"])
def save_path(temp_dir, request):
    return temp_dir / f"test{request.param}"

# ================================================================
#  Tests
# ================================================================

def test_standard_run():
    # Define a simple function
    def simple_func(x: int, y: MyObject) -> dict:
        return {"addition": x + y.value, "product": MyObject(x * y.value)}

    # Create the experiment
    exp = SweepExpParallel(
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

    sweep = SweepExpParallel(
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
    exp = SweepExpParallel(
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

def test_run_speed():
    """Test if the parallel run is faster than the serial run."""
    number_of_cpus = mp.cpu_count()
    min_number_of_cpus = 2
    if number_of_cpus < min_number_of_cpus:
        pytest.skip("This test requires at least 2 CPUs.")

    # Define a simple function that takes some time
    wait_time = 0.5
    def slow_func(_uselss: int) -> dict:
        time.sleep(wait_time)
        return {}

    # Create the experiment
    exp = SweepExpParallel(
        func=slow_func,
        parameters={"_uselss": [0, 1]},
        return_values={},
    )
    # Run the experiment in parallel
    start = time.time()
    exp.run()

    parallel_duration = time.time() - start
    tolerance = 0.3
    # Check that the parallel run took roughly the same time as the wait time
    assert np.isclose(parallel_duration, wait_time, atol=tolerance)

def test_run_with_failures():
    def fail_func(should_fail: bool) -> dict:  # noqa: FBT001
        if should_fail:
            raise ValueError
        return {}
    # Create the experiment
    exp = SweepExpParallel(
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
    exp = SweepExpParallel(
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
    exp = SweepExpParallel(
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
