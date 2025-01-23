# Building the Documentation
The documentation for this package can be automatically built and deployed, 
for example, to platforms like ReadTheDocs upon updates to a specific branch. 
However, if you wish to build the documentation locally to preview changes 
before pushing, follow the steps below.


## Setting Up a Build Environment
To set up a dedicated environment for building the documentation, 
navigate to the root of the repository and execute the following commands:

```bash
conda create -y --name docs-env python=3.11
conda activate docs-env
python3 -m pip install --upgrade --no-cache-dir setuptools sphinx readthedocs-sphinx-ext
python3 -m pip install --exists-action=w --no-cache-dir -r docs/requirements.txt
```


## Building the Documentation
Once you are in the root directory of the package repository, you can build the 
documentation using `make`. Ensure that the `make` command is available on your 
system and that the `docs-env` environment is activated. 
Then, run the following commands:

```bash
cd docs  
make html  
```

The built documentation will be located in the `docs/build/html` directory. To view it, open the `index.html` file in your web browser.


## Quick Build Option
Generating the API documentation and gallery can be time-consuming. To 
accelerate the build process, you can use the `QUICKBUILD` option:

```bash
make html QUICKBUILD=true  
```

This will skip the generation of the API documentation and gallery, building 
only the remaining sections of the documentation.

