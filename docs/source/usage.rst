Usage
=====

Let's say we have a function that depends on multiple parameters:

.. code-block:: python

   def my_function(param1: str, param2: float) -> dict:
      # Do something with the parameters
      result1 = param1.upper()
      result2 = param2 ** 2
      return {"result1": result1, "result2": result2}

and we want to test the function with different values of the parameters.
We can use the `SweepExpParallel` class to create a grid of parameters and run the
function with all the combinations:

.. code-block:: python

   from sweepexp import SweepExpParallel

   sweep = SweepExpParallel(
      func = my_function,
      parameters = {
         "param1": ["a", "b", "c"],
         "param2": [1.0, 2.0, 3.0]
      },
      return_values = {
         "result1": str,
         "result2": float
      },
      save_path = "data/results.zarr",
   )

   results = sweep.run()

   print(results)

TODO: Make a dropdown with the output

The individual experiments for each parameter combination are run in parallel
on separate processes. Alternatively, you can use the `SweepExp` class to run the
experiments sequentially, or the `SweepExpMPI` class to run the experiments in parallel
using MPI.

The results are saved to a `zarr` file, which can be easily loaded with xarray:

.. code-block:: python

   import xarray as xr

   ds = xr.open_zarr("data/results.zarr")
   print(ds)

TODO: Make a dropdown with the output

Let's walk though the example above:

1. We define a function `my_function` that takes two parameters `param1` and
   `param2` and returns a dictionary with two values `result1` and `result2`.
2. We create a `SweepExpParallel` object with the following parameters:

   - `func`: The function to run.
   - `parameters`: A dictionary with the parameters to sweep. The keys are the parameter names and the values are lists with the values to test.
   - `return_values`: A dictionary with the return values of the function. The keys are the return value names and the values are the types of the return values.
   - `save_path`: The path to the output file.
3. We run the experiments with the `run` method.

Mask Parameter Combinations
---------------------------
In some cases, we may want to exclude some parameter combinations from being
tested. This can be done by modifying the `skip_mask` attribute of the `SweepExp`
object. The `skip_mask` attribute is a xarray DataArray of boolean values that
indicates which parameter combinations to skip. Entries with `True` will be
skipped.

For example if we want to skip the parameter combination `param1='a'` and
`param2=2.0` in the example above, we can do the following:

.. code-block:: python

   sweep.skip_mask.loc[{"param1": "a", "param2": 2.0}] = True

Experiment ID
-------------
Each experiment is assigned a UUID that can be passed to the function as
an argument. This can be useful when writing the result of an individual experiment
to a file. The UUID can be accessed with the `uuid` attribute of the `SweepExp`
object.

The following example shows how to modify the above example to include the UUID:

.. code-block::

   from sweepexp import SweepExpParallel

   # Define the function that takes the UUID as an argument
   def my_function(param1: str, param2: float, uuid: str) -> dict:
      # Do something with the parameters
      result1 = param1.upper()
      result2 = param2 ** 2

      # Write the result to a file
      with open(f"data/{uuid}.txt", "w") as f:
         f.write(f"result1: {result1}\n")
         f.write(f"result2: {result2}\n")

      return {"result1": result1, "result2": result2}

   # Create the SweepExp object
   sweep = SweepExpParallel(
      func = my_function,
      parameters = {
         "param1": ["a", "b", "c"],
         "param2": [1.0, 2.0, 3.0]
      },
      return_values = {
         "result1": str,
         "result2": float
      },
      save_path = "data/results.zarr",
   )

   # Mark the experiment ID to be passed to the function
   sweep.pass_uuid = True

   # Run the experiments
   results = sweep.run()

Experiment Status
-----------------
The status of the experiments is stored in the `status` attribute of the `SweepExp`.
The `status` attribute is a xarray DataArray of strings that indicates the status
of each experiment. The status can be one of the following values:

- `not_started`: The experiment has not been started.
- `completed`: The experiment has been completed.
- `failed`: The experiment has failed.

By default, the `run` method will only run experiments that have not been started.
To rerun all experiments, regardless of their status, you can reset the status
with the `reset_status` method:

.. code-block:: python

   sweep.reset_status()

.. note::
    When passing a file path to the `save_path` parameter, where the file
    exists, the `SweepExp` object will attempt to load the existing file. Since
    this will also load the status of the experiments, the `run` method will
    only run experiments that have not been started.
