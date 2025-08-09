Task-Prioritization for Faster Execution
========================================
Efficiently scheduling experiments is crucial for minimizing execution time when
running experiments in parallel. By controlling the order in which experiments
are executed, you can ensure that worker processes remain busy for as long as
possible, avoiding idle time.

Why Prioritize Experiments?
---------------------------
Consider the following example with three experiments:

- Experiment A takes 1 second
- Experiment B takes 2 seconds
- Experiment C takes 3 seconds

Case 1: Non-Optimal Order
~~~~~~~~~~~~~~~~~~~~~~~~~
If we run the experiments in the order A → B → C, the total execution time is 4
seconds. Here's why:

- **Worker 1:** Executes A (1s) and C (3s), totaling 4 seconds.
- **Worker 2:** Executes B (2s), leaving it idle for the remaining 2 seconds.

This scenario is visualized in Figure 1.

.. figure:: /_static/priorities_nonoptimal.svg
   :width: 60%
   :align: center
   :alt: Non-optimal execution order

   Figure 1: Non-optimal execution order

Case 2: Optimal Order
~~~~~~~~~~~~~~~~~~~~~
If we run the experiments in the order C → B → A, the total execution time is
only 3 seconds. Here's why:

- **Worker 1:** Executes C (3s).
- **Worker 2:** Executes B (2s) and A (1s), totaling 3 seconds.

This is the optimal way to schedule the experiments, as shown in Figure 2.

.. figure:: /_static/priorities_optimal.svg
   :width: 60%
   :align: center
   :alt: Optimal execution order

   Figure 2: Optimal execution order

Rule of Thumb: Execute Longest Experiments First
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scheduling the longest experiments first helps to keep all workers busy for as
long as possible, minimizing the total execution time.

Controlling Execution Order with Priorities
-------------------------------------------
You can control the order of execution using priorities. To enable this feature:

1. Set the `enable_priorities` flag to True.
2. Assign a priority to each experiment using the `priority` attribute.
    - Higher priority experiments (e.g., priority 3) will be executed before lower priority ones (e.g., priority 1).
    - By default, all experiments have priority 0.
    - Negative priorities are also allowed.

Example: Comparing Execution with and without Priorities
--------------------------------------------------------
Let's revisit the example above and see the difference when priorities are used.

.. tab-set::

    .. tab-item:: Without Priorities

        .. code-block:: python

            import time
            from sweepexp import SweepExpParallel

            def my_slow_function(wait_time: float) -> dict:
                time.sleep(wait_time)
                return {}

            sweep = SweepExpParallel(
                func = my_slow_function,
                parameters = { "wait_time": [1, 2, 3] },
            )
            # We want to measure the total duration of the experiments
            start_time = time.time()

            # Run the experiments in parallel with MPI
            sweep.run(max_workers=2)

            # Print the total duration
            print(f"Total duration: {time.time() - start_time:.2f} seconds")

        This script will output:

        .. code-block:: none

            Total duration: 4.12 seconds

    .. tab-item:: With Priorities

        .. code-block:: python

            import time
            from sweepexp import SweepExpParallel

            def my_slow_function(wait_time: float) -> dict:
                time.sleep(wait_time)
                return {}

            sweep = SweepExpParallel(
                func = my_slow_function,
                parameters = { "wait_time": [1, 2, 3] },
                enable_priorities=True,
            )
            # Set priorities (higher number -> first executed)
            sweep.priority.data = [1, 2, 3]

            # We want to measure the total duration of the experiments
            start_time = time.time()

            # Run the experiments in parallel with MPI
            sweep.run(max_workers=2)

            # Print the total duration
            print(f"Total duration: {time.time() - start_time:.2f} seconds")

        This script will output:

        .. code-block:: none

            Total duration: 3.12 seconds

As you can see, by setting the priorities, we can control the order in which
experiments are executed, and optimize the total execution time.
