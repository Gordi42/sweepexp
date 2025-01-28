Running Experiments parallel with MPI
=====================================

German
------

Um Experimente parallel mit MPI auszuführen muss das Modul `mpi4py`, sowie
eine MPI-Implementierung wie `OpenMPI` oder `MPICH` installiert sein.

Nach der Installation können Experimente parallel mit MPI ausgeführt werden,
indem die Klasse `SweepExpMPI` statt `SweepExp` oder `SweepExpParallel` verwendet wird.
Das folgende Beispiel zeigt, wie ein Experiment parallel mit MPI ausgeführt werden kann:

.. code-block:: python

    import time
    from sweepexp import SweepExpMPI

    def my_slow_function(param: int) -> dict:
        time.sleep(2)
        return {"result": param ** 2}

    sweep = SweepExpMPI(
        func = my_slow_function,
        parameters = { "param": [1, 2] },
        return_values = { "result": int },
    )
    # We want to measure the total duration of the experiments
    start_time = time.time()
    
    # Run the experiments in parallel with MPI
    sweep.run()

    # Calculate the total duration
    total_duration = time.time() - start_time
    print(f"Total duration: {total_duration:.2f} seconds")

    print(sweep.data.result.values)


