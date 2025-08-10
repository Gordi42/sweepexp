Return Types
============

Overview
--------
When defining your experiment function, it can return results in several formats.  
However, only certain return types are allowed for compatibility with the sweepexp framework.  
You can return:

- A single value  
- A tuple of values  
- A dictionary of values (recommended)  

These containers must only include supported data types (see below).

Return Value Formats
--------------------

.. tab-set::

    .. tab-item:: Dictionary (recommended)

        Returning a dictionary allows you to name your outputs explicitly, which makes the results easier to interpret and access.

        .. code-block:: python

            from sweepexp import sweepexp

            def my_custom_experiment(x: float, y: float) -> dict:
                """Add and multiply two numbers."""
                return {"addition": x + y, "multiplication": x * y}

            sweep = sweepexp(
                func=my_custom_experiment,
                parameters={"x": [1, 2], "y": [3, 4, 5]},
            )

            print(sweep.run().data_vars)

        .. code-block::

            Data variables:
                status          (x, y) <U1 24B 'C' 'C' 'C' 'C' 'C' 'C'
                addition        (x, y) int64 48B 4 5 6 5 6 7
                multiplication  (x, y) int64 48B 3 4 5 6 8 10

    .. tab-item:: Tuple

        Returning a tuple packs multiple values without explicit names.  
        Results are accessible via generic keys like "result_1", "result_2", etc.

        .. code-block:: python

            from sweepexp import sweepexp

            def my_custom_experiment(x: float, y: float) -> tuple:
                """Add and multiply two numbers."""
                return x + y, x * y

            sweep = sweepexp(
                func=my_custom_experiment,
                parameters={"x": [1, 2], "y": [3, 4, 5]},
            )

            print(sweep.run().data_vars)

        .. code-block::

            Data variables:
                status    (x, y) <U1 24B 'C' 'C' 'C' 'C' 'C' 'C'
                result_1  (x, y) int64 48B 4 5 6 5 6 7
                result_2  (x, y) int64 48B 3 4 5 6 8 10

    .. tab-item:: Single Value

        When your function returns a single value, it is stored under the key "result".

        .. code-block:: python

            from sweepexp import sweepexp

            def my_custom_experiment(x: float, y: float) -> float:
                """Add two numbers."""
                return x + y

            sweep = sweepexp(
                func=my_custom_experiment,
                parameters={"x": [1, 2], "y": [3, 4, 5]},
            )

            print(sweep.run().data_vars)

        .. code-block::

            Data variables:
                status   (x, y) <U1 24B 'C' 'C' 'C' 'C' 'C' 'C'
                result   (x, y) int64 48B 4 5 6 5 6 7

Allowed Return Types
--------------------

Only the following data types are supported for return values (or their elements if returned as tuples or dictionaries):

+---------------------+--------------------------+---------------------------+
| Type                | dtype                    | Fill Value                |
+=====================+==========================+===========================+
| `int`               | `np.int64`               | `-9223372036854775808`    |
+---------------------+--------------------------+---------------------------+
| `float`             | `np.float64`             | `np.nan`                  |
+---------------------+--------------------------+---------------------------+
| `complex`           | `np.complex128`          | `np.nan`                  |
+---------------------+--------------------------+---------------------------+
| `str`               | `object`                 | `np.nan`                  |
+---------------------+--------------------------+---------------------------+
| `bool`              | `np.bool_`               | `False`                   |
+---------------------+--------------------------+---------------------------+
| `xarray.DataArray`  | dtype matches DataArray  | `np.nan`                  |
+---------------------+--------------------------+---------------------------+
| `xarray.Dataset`    | dtype matches DataArrays | `np.nan`                  |
+---------------------+--------------------------+---------------------------+
| `np.ndarray` (not_recommended) | `object`      | `np.nan`                  |
+---------------------+--------------------------+---------------------------+
| `object` (not recommended) | `object`          | `np.nan`                  |
+---------------------+--------------------------+---------------------------+
| `list`              | Not supported            |                           |
+---------------------+--------------------------+---------------------------+
| `tuple`             | Not supported            |                           |
+---------------------+--------------------------+---------------------------+
| `dict`              | Not supported            |                           |
+---------------------+--------------------------+---------------------------+

Note that while the experiment function itself may return a dictionary or tuple to package multiple values, these containers may only include the supported types listed above.

For detailed information on handling `xarray.DataArray` and `xarray.Dataset`, refer to the :ref:`multidimensional output <multidimensional_output>` section.





xarray DataArrays sind der empfohlene Rückgabetyp für multidimensionale Daten.
Die Dimensionen und Koordinaten des DataArrays werden automatisch im SweepExp Dataset hinzugefügt. Attribute und Metadaten des DataArrays werden ebenfalls übernommen.



`xarray.Dataset`
~~~~~~~~~~~~~~~~
Für komplexere Datenstrukturen ist es möglich, ein `xarray.Dataset` zurückzugeben.
Dabei werden die einzelnen DataArrays als Variablen im Ergebnis gespeichert.
Beachte, dass Metadaten des Datasets nicht übernommen werden, die Attribute der
enthaltenen DataArrays jedoch schon.

NumPy Arrays sind zwar erlaubt als Rückgabetypen, aber sie sind nicht empfohlen.
Die Ergebnisse werden in einem DataArray vom Typ `object` gespeichert. Ergebnisse
können dann nicht mehr als netCDF oder `zarr` gespeichert werden. Um dieses Problem zu umgehen, solltest du stattdessen `xarray.DataArray` verwenden.