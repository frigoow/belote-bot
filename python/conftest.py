"""Make the compiled belote_cpp module importable from the CMake build tree.

Build it with BELOTE_PYTHON=ON (see README); the tests look in build/ and in
this directory (for a copied .so).
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for path in (ROOT / "build", ROOT / "python"):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
