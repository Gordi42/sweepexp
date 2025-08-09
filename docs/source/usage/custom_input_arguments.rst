Adding Custom Input Arguments
=============================
When running parameter sweep experiments, some settings or configurations may
not logically belong to the parameter grid but still need to vary between
experiments. For example, you might want to specify unique filenames for saving
outputs, paths for logging, or experiment-specific metadata. Adding these as
custom arguments allows you to manage such variations without unnecessarily
inflating the parameter grid.

How to Use Custom Arguments
---------------------------
With sweepexp, you can add custom arguments that apply to the experiment's
function alongside the parameters in the grid. These custom arguments are added
with add_custom_argument() and have a default value (shared across all
experiments). After adding the custom argument, you can modify its values in the
data attribute to assign unique values for each parameter combination.

Here's how you can use this feature:

1: **Add the Custom Argument**: Use `add_custom_argument()` to introduce the new argument and define a default value.

2: **Customize the Argument for Each Experiment** Modify the argument values in the data attribute to assign unique values for each grid point.

3: **Run the Experiment**: The custom argument will automatically be passed to the function for each parameter combination.

Example
-------
In the following example, a unique filename is passed to a function that writes
two parameters to a file. The filename is set as a custom argument and modified
for each experiment:

.. code-block:: python

    from sweepexp import sweepexp

    # Create a function that writes two parameters to a file
    def write_to_file(param1: int, param2: int, filename: str):
        with open(filename, "w") as f:
            f.write(f"param1: {param1}, param2: {param2}")
        return {}

    # Create a SweepExp object around the function
    sweep = sweepexp(
        func = write_to_file,
        parameters = { "param1": [1, 4], "param2": [1, 2] },
    )

    # Create an additional argument for the filename
    sweep.add_custom_argument("filename", default_value="output.txt")
    # Change the filenames for the experiments
    sweep.data["filename"].values = [["output1.txt", "output2.txt"],
                                     ["output3.txt", "output4.txt"]]
    # Run the experiments
    sweep.run()

Unique UUIDs for Filenames
--------------------------
You can also use a built-in feature in sweepexp to automatically generate a
unique UUID for each experiment, which can be useful for creating unique
filenames without manually specifying them. This is particularly helpful when
running a large number of experiments where manually assigning filenames might
not be feasible.

Here's the same example as above, but using the UUID feature to generate unique
filenames:

.. code-block::

    from sweepexp import sweepexp

    # Create a function that writes two parameters to a file
    def write_to_file(param1: int, param2: int, uuid: str):
        filename = f"output_{uuid}.txt"
        print(f"Writing to: {filename}")
        with open(filename, "w") as f:
            f.write(f"param1: {param1}, param2: {param2}")
        return {}

    # Create a SweepExp object around the function
    sweep = sweepexp(
        func = write_to_file,
        parameters = { "param1": [1, 4], "param2": [1, 2] },
        pass_uuid=True,  # Enable UUID passing
    )

    sweep.run()

.. code-block::
    Writing to: output_bcab54c1-3123-4be9-8e36-cdb1b7796124.txt
    Writing to: output_3a89354b-9e6c-41b2-bd80-13a600e875d4.txt
    Writing to: output_93a781e1-edd6-4b25-946c-c9c5ce0f9965.txt
    Writing to: output_280c1299-5813-46d0-b225-8b26f3bddbf2.txt

The UUIDs are generated automatically for each experiment, and the filenames are
created accordingly. You can access the UUID of each experiment in the data
attribute under the "uuid" key.

.. tip::
    The UUID feature is not limited to filenames. You can use it in any 
    possible case where you need a unique identifier for each experiment.
