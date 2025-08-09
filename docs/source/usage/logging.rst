Logging
=======
When running parameter sweep experiments, having detailed output can be
beneficial for monitoring progress, debugging, or gaining insights into the
execution process. By configuring the logging level, you can control the amount
of detail included in the output.

SweepExp uses Python's logging module for output. By default, the logging level
is set to only show warnings and errors. You can adjust this level to display
detailed debug information by setting the logging level to "DEBUG":

.. code-block:: python

    from sweepexp import sweepexp, log

    log.setLevel("DEBUG")

    def my_custom_experiment(x: float, y: float) -> dict:
        """Add and multiply two numbers."""
        return {"addition": x + y, "multiplication": x * y}

    sweep = sweepexp(
        func = my_custom_experiment,
        parameters = { "x": [1, 2], "y": [3, 4, 5] },
    )

    sweep.run()

with output:

.. code-block::

    INFO - Found 6 experiments to run.
    DEBUG - 6 experiments left.
    DEBUG - Running experiment with kwargs: {'x': 1, 'y': 3}
    DEBUG - 5 experiments left.
    DEBUG - Running experiment with kwargs: {'x': 1, 'y': 4}
    DEBUG - 4 experiments left.
    DEBUG - Running experiment with kwargs: {'x': 1, 'y': 5}
    DEBUG - 3 experiments left.
    DEBUG - Running experiment with kwargs: {'x': 2, 'y': 3}
    DEBUG - 2 experiments left.
    DEBUG - Running experiment with kwargs: {'x': 2, 'y': 4}
    DEBUG - 1 experiments left.
    DEBUG - Running experiment with kwargs: {'x': 2, 'y': 5}
