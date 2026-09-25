"""Problem 3: interpretable emotion prediction with traceable evidence."""

# --- numpy 1.26 ↔ 2.x unpickle compat shim ---
# See problem2/__init__.py for the rationale; the same trick is needed here
# because Attachment 4 pickles also come from numpy 2.x.
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
