# How to run

Tested on Ubuntu 24.04 with GCC 13, CMake 3.28 and Python 3.12–3.13.

## Requirements

```bash
sudo apt install build-essential cmake libeigen3-dev python3-pip python3-venv
```

Eigen 3.4 is fetched automatically if it is not installed. Catch2 v3 (tests only) is always fetched with CMake's
`FetchContent`, so the first configure needs network access.

## C++ library, tests and examples

```bash
make test                  # = cmake -S . -B build/cmake && cmake --build build/cmake && ctest --test-dir build/cmake
./build/cmake/cpp/examples/01_hierarchy
```

CMake options: `MVG_BUILD_TESTS` (ON), `MVG_BUILD_EXAMPLES` (ON), `MVG_BUILD_PYTHON` (OFF; `pip` turns it on).
Every example is also registered as a smoke test, so `ctest` runs them.

## Python module

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[notebooks]"     # builds the C++ core with scikit-build-core + pybind11
python -c "import mvg; print(mvg.line_through([1, 2, 1], [3, 4, 1]))"
```

After changing C++ code, run `pip install -e .` again to rebuild the extension module.

## Notebooks

Each chapter has two copies of the same notebook:

* `2_notebooks/exercises/NN_<chapter>.ipynb` — the reader's copy. Cells marked ✏️ contain
  `raise NotImplementedError`; replace that line with your code and run the `assert` cell below it.
* `2_notebooks/solutions/NN_<chapter>.ipynb` — solved and executed.

```bash
jupyter lab 2_notebooks/exercises         # needs: pip install jupyterlab
```

**Colab.** Open a notebook from GitHub in Colab (File → Open notebook → GitHub). The first cell installs the
module with `%pip install git+https://github.com/Rudip1/learn-computer-vision`; building the C++ core takes about a
minute.

### Editing notebooks

The notebooks are generated; do not edit the `.ipynb` files by hand. The source of chapter `NN` is
`tools/notebooks/NN_<chapter>.md`: Markdown text, ` ```python ` fences for code cells, `+++` to split Markdown
cells, and `# BEGIN SOLUTION` / `# END SOLUTION` around the lines that the exercise copy leaves to the reader.

```bash
make notebooks             # regenerate both copies from the sources, then execute every solution notebook
make notebooks-check       # what CI runs: sources and exercises agree, no outputs in exercises, solutions execute
```

## Figures

All figures in `1_theory/figures/` are produced by `tools/make_figures.py` from the module itself:

```bash
make figures
```

## Code style

C++17, `snake_case` functions and variables, `PascalCase` types; `make format` runs `clang-format` (Google base,
4-space indent, 110 columns).
