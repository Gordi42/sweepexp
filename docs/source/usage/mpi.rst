Running Experiments parallel with MPI
=====================================
Parallelizing experiments can significantly reduce the execution time,
especially for computationally expensive functions. With SweepExp, you can easily
parallelize your experiments using MPI (Message Passing Interface). To achieve
this, you'll need the mpi4py module and an MPI implementation such as OpenMPI or
MPICH.

Using SweepExpMPI
-----------------
The `SweepExpMPI` class is designed to distribute and execute experiments in
parallel across multiple processes. This class coordinates tasks between a main
process (rank 0) and worker processes (other ranks). The main process assigns
tasks and collects results, while the worker processes execute the experiments.

To execute experiments in parallel with MPI, the `mpi4py` module, as well as an
MPI implementation like `OpenMPI` or `MPICH` must be installed.

After installation, experiments can be executed in parallel with MPI by using
the `SweepExpMPI` class instead of `SweepExp` or `SweepExpParallel`. The following
example shows how an experiment can be executed in parallel with MPI:

Here's an example:

.. code-block:: python
    :caption: mpi_example.py

    import time
    from sweepexp import SweepExpMPI

    def my_slow_function(param: float) -> dict:
        time.sleep(2)
        return {"result": param ** 2}

    sweep = SweepExpMPI(
        func = my_slow_function,
        parameters = { "param": [1, 2, 3] },
    )
    # We want to measure the total duration of the experiments
    start_time = time.time()
    
    # Run the experiments in parallel with MPI
    sweep.run()

    # Calculate the total duration
    total_duration = time.time() - start_time
    print(f"Total duration: {total_duration:.2f} seconds")

    print(sweep.data.result.values)

To execute the script in parallel, use the `mpiexec` (or `mpirun`, `srun`, etc.)
command followed by the number of processes and the script name. For example, to
use 4 processes:

.. code-block:: bash

    mpiexec -l -n 4 python mpi_example.py

When you run the script, you'll see output similar to this:

.. code-block::

    [0] Total duration: 2.11 seconds
    [0] [1. 4. 9.]
    [2] Total duration: 2.18 seconds
    [2] [nan nan nan]
    [3] Total duration: 2.21 seconds
    [3] [nan nan nan]
    [1] Total duration: 2.21 seconds
    [1] [nan nan nan]

Explanation of the Output
^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Main Process Results:** The main process ([0]) displays the total execution time and the results of all experiments. Only the main process has access to the aggregated results.

2. **Worker Processes:** Worker processes ([1], [2], [3]) handle the execution of experiments and return results to the main process. Since the results are only collected and stored by the main process, the worker processes display nan values for the results.

3. **Execution Time:** The total duration (approximately 2 seconds) corresponds to the time required to execute a single instance of the slow function, as all three experiments are executed in parallel.

Dynamic Task Assignment
-----------------------
If there are more experiments than available worker processes, tasks are
dynamically assigned. Once a worker process finishes a task, it is assigned the
next available experiment. Hence, the number of processes does not need to match
to the number of experiments.