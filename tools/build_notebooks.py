#!/usr/bin/env python3
"""Build the exercise and solution notebooks from their Markdown sources.

Each chapter notebook has one source, ``tools/notebooks/NN_<chapter>.md``:

* text outside code fences becomes Markdown cells; a line containing only ``+++`` splits Markdown cells;
* every fenced block opened with three backticks and ``python`` becomes one code cell;
* inside a code cell, the lines between ``# BEGIN SOLUTION`` and ``# END SOLUTION`` are kept in the solution
  notebook and replaced by a placeholder in the exercise notebook.

Outputs:  2_notebooks/solutions/NN_<chapter>.ipynb  and  2_notebooks/exercises/NN_<chapter>.ipynb  (no outputs).

    python tools/build_notebooks.py            # build all
    python tools/build_notebooks.py 01 03      # build chapters whose file name starts with 01 or 03
    python tools/build_notebooks.py --check    # fail if a committed exercise notebook differs from its source
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tools" / "notebooks"
OUT = ROOT / "2_notebooks"

FENCE_OPEN = re.compile(r"^```python\s*$")
FENCE_CLOSE = re.compile(r"^```\s*$")
PLACEHOLDER = ["# ✏️ YOUR CODE HERE", 'raise NotImplementedError("exercise: replace this line with your code")']


def parse(text: str) -> list[tuple[str, str]]:
    """Split a source file into (kind, text) cells, kind in {"markdown", "code"}."""
    cells: list[tuple[str, str]] = []
    md: list[str] = []
    code: list[str] | None = None

    def flush_md():
        body = "\n".join(md).strip("\n")
        if body.strip():
            cells.append(("markdown", body))
        md.clear()

    for line in text.splitlines():
        if code is None:
            if FENCE_OPEN.match(line):
                flush_md()
                code = []
            elif line.strip() == "+++":
                flush_md()
            else:
                md.append(line)
        else:
            if FENCE_CLOSE.match(line):
                cells.append(("code", "\n".join(code).strip("\n")))
                code = None
            else:
                code.append(line)
    if code is not None:
        raise ValueError("unterminated code fence")
    flush_md()
    return cells


def strip_solution(code: str) -> str:
    out: list[str] = []
    inside = False
    for line in code.splitlines():
        stripped = line.strip()
        if stripped == "# BEGIN SOLUTION":
            if inside:
                raise ValueError("nested BEGIN SOLUTION")
            inside = True
            indent = line[: len(line) - len(line.lstrip())]
            out.extend(indent + p for p in PLACEHOLDER)
        elif stripped == "# END SOLUTION":
            if not inside:
                raise ValueError("END SOLUTION without BEGIN")
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise ValueError("unterminated BEGIN SOLUTION")
    return "\n".join(out)


def keep_solution(code: str) -> str:
    return "\n".join(l for l in code.splitlines() if l.strip() not in ("# BEGIN SOLUTION", "# END SOLUTION"))


def make_notebook(cells: list[tuple[str, str]], solution: bool) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, text in cells:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            body = keep_solution(text) if solution else strip_solution(text)
            nb.cells.append(nbformat.v4.new_code_cell(body))
    for i, cell in enumerate(nb.cells):  # stable ids keep diffs small
        cell["id"] = f"cell-{i:03d}"
    return nb


def sources(prefixes: list[str]) -> list[Path]:
    files = sorted(SRC.glob("[0-9][0-9]_*.md"))
    if prefixes:
        files = [f for f in files if any(f.name.startswith(p) for p in prefixes)]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chapters", nargs="*", help="file-name prefixes, e.g. 01 02")
    ap.add_argument("--check", action="store_true", help="only verify the exercise notebooks are up to date")
    args = ap.parse_args()

    status = 0
    for src in sources(args.chapters):
        cells = parse(src.read_text(encoding="utf-8"))
        name = src.stem + ".ipynb"
        exercise = make_notebook(cells, solution=False)
        ex_path = OUT / "exercises" / name
        if args.check:
            current = ex_path.read_text(encoding="utf-8") if ex_path.exists() else ""
            if current != nbformat.writes(exercise) + "\n":
                print(f"out of date: {ex_path.relative_to(ROOT)} (run tools/build_notebooks.py)")
                status = 1
            continue
        for sub, nb in (("exercises", exercise), ("solutions", make_notebook(cells, solution=True))):
            path = OUT / sub / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(nbformat.writes(nb) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
