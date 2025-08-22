"""Package initialization for aklab_imaging.

This file dynamically imports known submodules, exposes their public
symbols into the package namespace, and builds a combined ``__all__``.
This avoids referencing module objects (e.g. ``spectrometer``) that
aren't defined when using ``from .module import *``.
"""

import importlib
from typing import List

__all__: List[str] = []

# List of submodules to pull public names from
_submodules = [
    "spectrometer",
    "imaging_tools",
    "FLI",
    "data_view",
    "textcolor",
    "thr640",
]

for _name in _submodules:
    try:
        _mod = importlib.import_module(f".{_name}", __package__)
    except Exception:
        # If a submodule fails to import, skip it silently — package can
        # still be partially usable.
        continue

    # Determine public names for this module
    if hasattr(_mod, "__all__"):
        _names = list(getattr(_mod, "__all__"))
    else:
        _names = [n for n in dir(_mod) if not n.startswith("_")]

    # Re-export each public name in the package namespace
    for _n in _names:
        globals()[_n] = getattr(_mod, _n)

    __all__.extend(_names)

# Remove duplicates while preserving order
_seen = set()
__all__ = [x for x in __all__ if not (x in _seen or _seen.add(x))]

# Package version (optional)
try:
    from ._version import __version__  # type: ignore
except Exception:
    __version__ = "0.0.0"
