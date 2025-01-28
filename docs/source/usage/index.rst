Usage
=====

.. toctree::
    :maxdepth: 1
    :hidden:

    status
    saving
    custom_input_arguments
    timeit
    mpi
    priorities
    logging

The sweepexp library provides a straightforward way to run experiments with
multiple parameter combinations. This tutorial demonstrate its ease of use,
focusing on the basic functionality. I.e. showcase how to define an experiment,
run it sequentially, or switch to parallel execution with minimal changes.

Defining Your Experiment
------------------------
The first step is to define the custom function that you want to evaluate for
different parameter combinations. Here's an example function:

.. code-block::

    def my_custom_experiment(x: float, y: float) -> dict:
        """Add and multiply two numbers."""
        return {"addition": x + y, "multiplication": x * y}

This simple function adds and multiplies two input numbers, but it could
represent any custom computation, including more complex or expensive tasks,
such as training a machine learning model or running a simulation. The input and
output values can be customized to fit your specific use case. Important is that
the function returns a dictionary with the results.

Setting Up the Sweep
--------------------
Next, we create a `SweepExp` object that will run the experiment for all
parameter combinations. We define the parameters to sweep, the return values of
the function, and the function itself. Here's how to set up the sweep for the
example function:

.. code-block:: python

    from sweepexp import SweepExp

    sweep = SweepExp(
        func = my_custom_experiment,
        parameters = { "x": [1, 2], "y": [3, 4, 5] },
        return_values = { "addition": float, "multiplication": float },
    )

The `parameters` dictionary contains the parameter names as keys and lists of
values to test as values. The `return_values` dictionary contains the return
value names as keys and the types of the return values as values. The `func`
parameter is the function to run. In this case, we want to test the "x" parameter
with two values (1 and 2) and the "y" parameter with three values (3, 4, and 5).
This makes a total of 6 experiments to run. Let's now run the experiments and
print the results:

.. code-block:: python

    sweep.run()

    print(sweep.data)

The `run` method executes the experiments and stores the results in the `data`
attribute of the `SweepExp` object. The `data` attribute is an xarray Dataset
that contains the results of the experiments. The output will look like this:

.. code-block::

    <xarray.Dataset> Size: 160B
    Dimensions:         (x: 2, y: 3)
    Coordinates:
      * x               (x) int64 16B 1 2
      * y               (y) int64 24B 3 4 5
    Data variables:
        status          (x, y) <U1 24B 'C' 'C' 'C' 'C' 'C' 'C'
        addition        (x, y) float64 48B 4.0 5.0 6.0 5.0 6.0 7.0
        multiplication  (x, y) float64 48B 3.0 4.0 5.0 6.0 8.0 10.0

Parallel Execution using multiprocessing or MPI
-----------------------------------------------
Since the experiments are independent, they can be run in parallel to speed up
the process. The sweepexp library provides two classes for parallel execution:
`SweepExpParallel` for multiprocessing and `SweepExpMPI` for MPI. The setup is
similar to the sequential execution, but the experiments are executed in parallel.

The following example demonstrates how to run the same experiment as above in the
three different modes: sequentially, in parallel with multiprocessing, and in
parallel with MPI.

.. tab-set::

    .. tab-item:: Sequentially

        .. code-block:: python

            from sweepexp import SweepExp


            def my_custom_experiment(x: float, y: float) -> dict:
                """Add and multiply two numbers."""
                return {"addition": x + y, "multiplication": x * y}

            sweep = SweepExp(
                func = my_custom_experiment,
                parameters = { "x": [1, 2], "y": [3, 4, 5] },
                return_values = { "addition": float, "multiplication": float },
            )

            sweep.run()

    .. tab-item:: Parallel (multiprocessing)

        .. code-block:: python

            from sweepexp import SweepExpParallel


            def my_custom_experiment(x: float, y: float) -> dict:
                """Add and multiply two numbers."""
                return {"addition": x + y, "multiplication": x * y}

            sweep = SweepExpParallel(
                func = my_custom_experiment,
                parameters = { "x": [1, 2], "y": [3, 4, 5] },
                return_values = { "addition": float, "multiplication": float },
            )

            sweep.run()
      
    .. tab-item:: Parallel (MPI)

        .. code-block:: python

            from sweepexp import SweepExpMPI


            def my_custom_experiment(x: float, y: float) -> dict:
                """Add and multiply two numbers."""
                return {"addition": x + y, "multiplication": x * y}

            sweep = SweepExpMPI(
                func = my_custom_experiment,
                parameters = { "x": [1, 2], "y": [3, 4, 5] },
                return_values = { "addition": float, "multiplication": float },
            )

            sweep.run()

        .. code-block:: bash

            mpiexec -l -n 4 python my_script.py

For more advanced features, such as measuring the duration of experiments, marking
experiments to be skipped, or optimizing the execution order, check out the other
tutorials:

Advanced Topics
---------------

.. grid:: 1 2 2 1
    :margin: 4 4 0 0
    :gutter: 2

    .. grid-item-card::  Status
        :link: status
        :link-type: doc

        Track statuses and control which experiments to run.

    .. grid-item-card::  Save Results to a File
        :link: saving
        :link-type: doc

        Learn how to save, load and autosave results.

    .. grid-item-card::  Adding Custom Input Arguments
        :link: package_api
        :link-type: doc

        Add custom arguments to the function and modify them for each parameter combination.

    .. grid-item-card::  Measure the Duration of Experiments
        :link: timeit
        :link-type: doc

        Learn how to automatically measure the execution time of experiments.

    .. grid-item-card::  Run Experiments in Parallel with MPI
        :link: mpi
        :link-type: doc

        Learn how to run experiments in parallel using MPI.
   
    .. grid-item-card::  Optimize with Priorities
        :link: priorities
        :link-type: doc

        Learn how to optimize the execution time by controlling the order of experiments.

    .. grid-item-card::  Logging
        :link: logging
        :link-type: doc

        Learn how to control the detail level of logging messages.
