"""Problem 2: local missing-modality robustness."""

# --- numpy 1.26 ↔ 2.x unpickle compat shim ---
# Attachment 3 / 4 pickles were saved under numpy 2.x and reference private
# module paths (numpy._core.numeric, numpy._core.multiarray, ...) that do not
# exist in numpy 1.26.  We alias those names to their numpy 1.26 equivalents
# the moment any submodule of problem2 is imported, so that pickle.load() works
# transparently without modifying data.py / scripts/run.py source files
# (their sha256 is recorded inside the saved checkpoint and must stay stable).
import sys as _sys
import types as _types
import numpy as _np  # noqa: E402
import numpy.core.numeric as _nc  # noqa: E402
import numpy.core.multiarray as _ma  # noqa: E402
import numpy.core.umath as _um  # noqa: E402
import numpy.core.shape_base as _sb  # noqa: E402
import numpy.core.fromnumeric as _fn  # noqa: E402

_nc_pkg = _types.ModuleType("numpy._core")
_sys.modules.setdefault("numpy._core", _nc_pkg)
for _name, _mod in (("numeric", _nc), ("multiarray", _ma), ("umath", _um),
                    ("shape_base", _sb), ("fromnumeric", _fn)):
    _sys.modules.setdefault(f"numpy._core.{_name}", _mod)
    setattr(_nc_pkg, _name, _mod)
