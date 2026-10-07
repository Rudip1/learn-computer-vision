# Convenience targets. Everything here is plain CMake / pip / Python underneath; see HOW_TO_RUN.md.

BUILD_DIR ?= build/cmake
PYTHON    ?= python3
JOBS      ?= $(shell nproc 2>/dev/null || echo 4)

.PHONY: all build test install notebooks notebooks-build notebooks-check figures format clean help

help:
	@echo "make build      configure and build the C++ library, tests and examples ($(BUILD_DIR))"
	@echo "make test       build, then run the Catch2 tests with ctest"
	@echo "make install    pip install -e . (builds the Python module)"
	@echo "make notebooks  regenerate notebooks from tools/notebooks/ and execute every solution notebook"
	@echo "make figures    regenerate 1_theory/figures/"
	@echo "make all        build + test + install + notebooks"

all: test install notebooks

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) -j $(JOBS)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure -j $(JOBS)

install:
	$(PYTHON) -m pip install -e .

notebooks-build:
	$(PYTHON) tools/build_notebooks.py

notebooks: notebooks-build
	$(PYTHON) tools/execute_notebooks.py

notebooks-check:
	$(PYTHON) tools/build_notebooks.py --check
	$(PYTHON) tools/strip_notebooks.py --check
	$(PYTHON) tools/execute_notebooks.py --no-save

figures:
	$(PYTHON) tools/make_figures.py

# Google base style, 4-space indent, 110 columns (passed inline so that no extra config file is needed).
CLANG_STYLE = {BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 110, AccessModifierOffset: -2}

format:
	clang-format -i -style="$(CLANG_STYLE)" cpp/include/mvg/*.hpp cpp/src/*.cpp cpp/tests/*.cpp cpp/tests/*.hpp cpp/examples/*.cpp \
		python/mvg/_bindings/*.cpp python/mvg/_bindings/*.hpp

clean:
	rm -rf build
