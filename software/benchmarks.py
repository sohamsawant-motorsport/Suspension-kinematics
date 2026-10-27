# -*- coding: utf-8 -*-
"""Optional reference data, kept out of this repository.

Benchmark values are read at run time rather than written into the code.
Where the code wants a reference car to compare against, it asks here. Nothing
in this repository depends on an answer.

If a file called `benchmarks_private.py` sits next to this one, its values are
used. If it does not, every benchmark overlay is simply hidden and every check
that compares against one is skipped -- the tool, the solver and the rest of
the validation suite run exactly the same either way.

`benchmarks_private.example.py` documents the shape. Copy it, rename it, and
fill it with figures for whatever reference car you are entitled to use.
"""

import os
import sys

# GT3_PRIVATE_DIR points at a folder OUTSIDE this repository, so reference
# data never sits inside the tree and publishing it cannot leak anything even
# if .gitignore is edited by mistake.
_dir = os.environ.get("GT3_PRIVATE_DIR")
if _dir and os.path.isdir(_dir) and _dir not in sys.path:
    sys.path.insert(0, _dir)

try:
    import benchmarks_private as _p
    AVAILABLE = True
except ImportError:
    _p = None
    AVAILABLE = False


def get(name, default=None):
    """One reference value, or `default` when no private file is installed."""
    if _p is None:
        return default
    return getattr(_p, name, default)


def require(name):
    """Same, but says clearly what is missing instead of failing obscurely."""
    v = get(name)
    if v is None:
        raise RuntimeError(
            f"{name!r} needs reference data. Copy "
            "software/benchmarks_private.example.py to "
            "software/benchmarks_private.py and fill it in.")
    return v
