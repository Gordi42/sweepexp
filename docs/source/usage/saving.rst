Save Results to a File
=======================

Data can be saved in different file formats. To do this, specify the file path
in the constructor. The file extension determines the file format. Supported
file formats include:
- `.nc` or `.cdf` for NetCDF files
- `.zarr` for Zarr files
- `.pkl` for Pickle files
Pickle files offer the advantage of storing generic Python objects but are not
platform-independent. The following example demonstrates saving a dataset to a
NetCDF file:

.. code-block:: python

    from sweepexp import SweepExp

    # Create a SweepExp object around a function that calculates the ratio of two numbers
    sweep = SweepExp(
        func = lambda a, b: { "rat": a / b },
        parameters = { "a": [1, 4], "b": [1, 2] },
        return_values = { "rat": float },
        save_path = "results.cdf",
    )

    # Run the experiments
    sweep.run()

    # Save the results to a file
    sweep.save()

When executing this code a second time, a file named `results.cdf` already exists.
In this case, the data is automatically loaded when the SweepExp object is
initialized. Thus, calling `sweep.run()` during the second execution does not 
perform any experiments, as their status is already marked as "C" (completed).

However, calling `sweep.save()` again will raise an error because the file
already exists. To overwrite the file, use the parameter `mode="w"`:

.. code-block:: python

    sweep.save(mode="w")

Autosave
--------
To prevent data loss, enable the auto_save option by setting it to True.
When enabled, the data is automatically saved after each completed experiment.
This ensures that, in the event of an unexpected program crash, already
completed experiments do not need to be repeated.

.. code-block:: python

    from sweepexp import SweepExp

    sweep = SweepExp(
        func = lambda a, b: { "rat": a / b },
        parameters = { "a": [1, 4], "b": [1, 2] },
        return_values = { "rat": float },
        save_path = "results.cdf",
    )
    sweep.auto_save = True
    sweep.run()
