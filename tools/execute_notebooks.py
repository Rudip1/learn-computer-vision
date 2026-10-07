#!/usr/bin/env python3
"""Execute notebooks top to bottom and store their outputs in place.

    python tools/execute_notebooks.py                       # every solution notebook
    python tools/execute_notebooks.py 2_notebooks/solutions/01_projective_geometry.ipynb
    python tools/execute_notebooks.py --no-save             # execute only (CI): fail on the first error

A notebook fails if any cell raises. The working directory of each run is the notebook's own directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebooks", nargs="*", type=Path)
    ap.add_argument("--no-save", action="store_true", help="do not write outputs back")
    ap.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    args = ap.parse_args()

    paths = args.notebooks or sorted((ROOT / "2_notebooks" / "solutions").glob("*.ipynb"))
    failed = []
    for path in paths:
        path = path.resolve()
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(path.parent)}})
        t0 = time.time()
        try:
            client.execute()
        except CellExecutionError as err:
            print(f"FAILED {path.relative_to(ROOT)}\n{err}")
            failed.append(path)
            continue
        print(f"ok     {path.relative_to(ROOT)}  ({time.time() - t0:.1f} s)")
        if not args.no_save:
            for cell in nb.cells:  # execution metadata only adds noise to diffs
                cell.get("metadata", {}).pop("execution", None)
            nbformat.write(nb, path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
