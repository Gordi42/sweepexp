Measure the Duration of Experiments
===================================
The runtime of each experiment can be measured by setting the `timeit` attribute
to `True`. This will store the duration of each experiment in the `duration`
attribute of the `SweepExp` object. The duration is measured in seconds.

.. code-block::

    import time
    from sweepexp import SweepExp

    # Define a function that sleeps for a given number of seconds
    def sleep(wait_time: float) -> dict:
        time.sleep(wait_time)
        return {}

    # Create a SweepExp object around the function
    sweep = SweepExp(
        func = sleep,
        parameters = { "wait_time": [0.2, 0.5, 0.8] },
        return_values = {},
    )

    # Enable the timeit option
    sweep.timeit = True

    # Run the experiments
    sweep.run()

    # Print the duration of each experiment
    print(sweep.duration.values)

.. code-block::

    [0.20015073 0.50012445 0.80014301]
