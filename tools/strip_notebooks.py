#!/usr/bin/env python3
"""Remove outputs and execution counts from notebooks (default: every exercise notebook).

    python tools/strip_notebooks.py                 # strip 2_notebooks/exercises/*.ipynb in place
    python tools/strip_notebooks.py --check         # fail if any exercise notebook has outputs
    python tools/strip_notebooks.py path/to/nb.ipynb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def has_outputs(nb) -> bool:
    return any(c.cell_type == "code" and (c.outputs or c.execution_count) for c in nb.cells)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="*", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    paths = args.notebooks or sorted((ROOT / "2_notebooks" / "exercises").glob("*.ipynb"))
    dirty = []
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        if not has_outputs(nb):
            continue
        dirty.append(path)
        if not args.check:
            for c in nb.cells:
                if c.cell_type == "code":
                    c.outputs = []
                    c.execution_count = None
            nbformat.write(nb, path)
    for path in dirty:
        print(("has outputs: " if args.check else "stripped: ") + str(path))
    return 1 if (args.check and dirty) else 0


if __name__ == "__main__":
    sys.exit(main())
