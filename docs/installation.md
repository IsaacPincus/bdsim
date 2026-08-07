# Installation

## Prerequisites

```bash
sudo apt update && sudo apt install -y build-essential cmake
```

## Python environment

Create a virtual environment *above* the project directory, then install the
Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy nanobind h5py
```

## Build the extension and install the package

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBDSIM_PYTHON=ON \
      -Dnanobind_DIR=$(python -m nanobind --cmake_dir)
cmake --build build -j
cp build/_bdsim*.so python/bdsim/

pip install -e .          # from the directory containing pyproject.toml
```

The editable install makes `import bdsim` work from anywhere --- both when running
scripts and for editors and type checkers. The compiled `_bdsim*.so` is built by
CMake and copied into the package; rebuilding it does **not** require reinstalling.

## Check it works

```bash
ctest --test-dir build --output-on-failure     # 14 tests
python examples/demo.py
```

## Optional

| Purpose | Package |
|---|---|
| Reading the Fortran reference output | `netCDF4` |
| Plots in the validation scripts | `matplotlib` |
| Kolmogorov--Smirnov tests in `compare_fortran.py` | `scipy` |
| Building this documentation | `sphinx myst-parser sphinx-rtd-theme` |

## Building the documentation

```bash
pip install sphinx myst-parser sphinx-rtd-theme
make -C docs html
```

These are documentation dependencies only; they are deliberately not runtime
dependencies of `bdsim`, so a fresh virtual environment will not have them and
`make html` will stop with a message telling you what to install.

The build invokes `python -m sphinx` rather than the `sphinx-build` script. That
follows whichever interpreter is active, so it works when a `pip install --user`
has put the scripts somewhere off `PATH`, and it cannot accidentally pick up a
system-wide `sphinx-build` belonging to a different Python.

The compiled extension is mocked during the build, so the documentation builds
even with nothing compiled.
