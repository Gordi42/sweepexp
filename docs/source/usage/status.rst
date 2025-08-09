Status
======

Why Use the Status Mechanism?
-----------------------------
When running experiments with many parameter combinations, things don't always
go as planned. Some experiments might fail due to invalid inputs or runtime errors.
Others might involve combinations of parameters that you know are unnecessary
and want to skip entirely. Restarting all experiments from scratch every time
you encounter an issue wastes time and resources.

The status mechanism in sweepexp solves this problem by providing a way to
monitor and control the progress of your experiments. Each experiment is assigned
a status, which allows to track what has been completed, what has failed,
and what has been skipped.

What Are the Possible Status Values?
------------------------------------

You can keep track of the status of the experiments by accessing the `status`
attribute of the `SweepExp` object. The `status` attribute is an xarray DataArray
of strings that indicates the status of each experiment. The status can be one
of the following values:

- `N` (Not Started): The experiment has not been started.
- `C` (Completed): The experiment has been completed.
- `F` (Failed): The experiment has failed.
- `S` (Skip): The experiment was marked as skipped.

The status is useful for two main reasons. First, it allows you to check which
experiments have been successfully completed and which have not. Second, it can
be used to control which experiments should be executed. When creating a new
`SweepExp` object, the status of all experiments is automatically set to `N`, which
means that they have not been started. When the `run` method is called, only the
experiments with the status `N` are executed. This allows you to skip specific
parameter combinations by setting the status of these experiments to `S`.

How to Access and Modify the Status
-----------------------------------
Let's look at an example where we define a function that calculates the ratio
of two numbers (a/b). If the second parameter b is `0`, an error should be raised,
causing the function to fail.

.. code-block:: python
    
    from sweepexp import sweepexp

    def ratio(a: int, b: int) -> float:
        if b == 0:
            raise ValueError("Division by zero")
        return { "rat": a / b }

    sweep = sweepexp(
        func = ratio,
        parameters = { "a": [1, 4], "b": [0, 2] },
    )

We set the status of the experiment with the parameters `a=1` and `b=0` to `S`
(skip) and run the experiments.

.. code-block:: python

    # Set the status of the experiment with a=1 and b=0 to skip
    sweep.status.loc[{"a": 1, "b": 0}] = "S"

    # Run the experiments
    sweep.run()

    # Print the results
    print(sweep.data.rat.values)

.. code-block::

    ERROR - Error in experiment {'a': 4, 'b': 0}: Division by zero
    [[nan 0.5]
     [nan 2. ]]

Note that the rows of the results matrix correspond to the different values of
the parameter `a`, and the columns correspond to the different values of the
parameter `b`. As we can see, the experiment with the parameters `a=1` and `b=0` has no value.
Also, the experiment with the parameters `a=4` and `b=0` failed due to a division
by zero error. Let's now look at the status of the experiments.

.. code-block:: python

    print(sweep.status.values)

.. code-block::

    [['S' 'C']
     ['F' 'C']]

In the status matrix, we see that only the two experiments with the parameters
`b=2` have been successfully completed. The other two experiments were either
skipped or failed.

Running Experiments with a Specific Status
------------------------------------------

Now, we want to run the experiment with the status 'S'. To do this, we can either
manually change the status of this experiment back to `N` or call the `run` method
with the argument `status='S'`. This will execute all experiments with the status
'S'.

.. code-block:: python

    # Run the experiments with status 'S'
    sweep.run(status="S")

    # Print the results
    print(sweep.data.rat.values)

.. code-block::

    ERROR - Error in experiment {'a': 1, 'b': 0}: Division by zero
    [[nan 0.5]
     [nan 2. ]]

As we can see, the experiment with the parameters `a=1` and `b=0` was executed
and failed due to a division by zero error.

It is also possible to pass a list of status values to the `run` method. This
would make the method execute all experiments that have one of the status values
in the list. For example, to run all experiments that are either 'N' or 'C':

.. code-block:: python

    # Run the experiments with status 'N' or 'C'
    sweep.run(status=["N", "C"])
