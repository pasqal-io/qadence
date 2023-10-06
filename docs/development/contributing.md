If you want to contribute to Qadence, feel free to branch out from `main` and send a merge request to the Qadence repository.
This will be reviewed, commented and eventually integrated in the codebase.

## Install from source

Before installing `qadence` from source, make sure you have Python >=3.9. For development, the preferred method to
install `qadence` is to use [hatch](https://hatch.pypa.io/latest/). Clone this repository and run:

```bash
python -m pip install hatch

# to enter into a shell with all dependencies
python -m hatch -v shell

# to run a script into the shell
python -m hatch -v run my_script_with_qadence.py
```

If you after some time you have issues with your development environment, you can rebuild it by running:

```bash
python -m hatch env prune
python -m hatch -v shell
```

You also have the following (non recommended) installation methods:

* install with `pip` in development mode by simply running `pip install -e .`. Notice that in this way
  you will install all the dependencies, including extras.
* install it with `conda` by simply using `pip` within a clean Conda environment.

## Before developing

Before starting to develop code, please keep in mind the following:

1. Use `pre-commit` hooks to make sure that the code is properly linted before pushing a new commit. To do so, execute the following commands in the virtual environment where you installed Qadence:

```bash
python -m pip install pre-commit  # this will be already available if you installed the package with Poetry
pre-commit install  # this will install the pre-commit hook
pre-commit run --all-files
```

2. Make sure that the unit tests and type checks are passing since the merge request will not be accepted if the automatic CI/CD pipeline do not pass. To do so, execute the following commands in the virtual environment where you installed Qadence:

```bash
# if you used Hatch for installing these dependencies will be already available
python -m pip install pytest pytest-cov mypy

# run the full test suite without some longer running tests
# remove the `-m` option to run the full test suite
python -m hatch -v run test -m "not slow" # with Hatch outside the shell
pytest -m "not slow"  # with pytest directly
```

## Build documentation

For building the documentation locally, we recommend to use `hatch` as follows:

```bash
python -m hatch -v run docs:build
python -m hatch -v run docs:serve
```

Notice that this will build the documentation in strict mode, thus it will fail if even just one warning is detected.
