"""Walk the repository and ``py_compile`` every .py file.

This is a quick smoke test: it catches syntax errors without needing to
actually install all the heavy runtime dependencies (torch, pykeen,
spacy, ...). Run with:

    python scripts/check_syntax.py
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path


ROOTS = ("src", "scripts", "app")


def main() -> int:
    bad: list[tuple[str, str]] = []
    for root in ROOTS:
        for f in Path(root).rglob("*.py"):
            try:
                py_compile.compile(str(f), doraise=True)
            except py_compile.PyCompileError as e:
                bad.append((str(f), str(e)))
    if bad:
        for f, err in bad:
            print(f"FAIL {f}\n    {err}")
        return 1
    print(f"OK -- all .py files compile cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
