Multidimensional Output
=======================
In some cases, your experiment function may return multidimensional data
This can for example be a NumPy array, or an `xarray.DataArray` / `xarray.Dataset`.

While it is technically possible to return a NumPy array, this approach is not recommended.
When `sweepexp` stores results, it creates an `xarray.Dataset` with one entry for each parameter combination.  
If the function returns a NumPy array, it is stored as an object inside the Dataset.  
As a consequence, the output cannot be saved as NetCDF.

A better solution is to return an `xarray.DataArray` or `xarray.Dataset`.  
In this case:
- The output dimensions are automatically integrated into the result Dataset.
- The data can be saved and processed as NetCDF.
- Output attributes are preserved.

.. note::

    The output dimensions must not overlap with the sweep parameter dimensions.  

Returning an `xarray.DataArray`
-------------------------------

.. code-block:: python

    from sweepexp import sweepexp
    import xarray as xr


    def my_custom_experiment(x: float) -> float:
        return {"xr_data": xr.DataArray([x, x + 1, x - 1], coords={"y": [1, 2, 3]})}

    sweep = sweepexp(
        func = my_custom_experiment,
        parameters = { "x": [1, 2] },
    )

    print(sweep.run())

.. code-block::

    <xarray.Dataset> Size: 96B
    Dimensions:  (x: 2, y: 3)
    Coordinates:
    * x        (x) int64 16B 1 2
    * y        (y) int64 24B 1 2 3
    Data variables:
        status   (x) <U1 8B 'C' 'C'
        xr_data  (x, y) int64 48B 1 2 0 2 3 1

In this example:

- The function returns an `xarray.DataArray` named `"xr_data"` and dimension `y` with length 3.
- After the sweep, `xr_data` has dimensions `(x, y)`,  

Returning an `xarray.Dataset`
-----------------------------
Similarly, you can return an `xarray.Dataset`. For which each `DataArray` inside
the Dataset becomes a separate variable in the results.

Notes:

- Dataset-level metadata is **not** preserved.
- Attributes of the contained DataArrays **are** preserved.
