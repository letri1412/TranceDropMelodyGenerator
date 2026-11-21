"""Bootstrap patch for this workspace (loaded via PYTHONPATH).

Problem
-------
On this machine (CPython 3.15.x, Windows) ``os.mkdir(path, 0o700)`` creates a
directory guarded by a restrictive security descriptor that even the *current*
process cannot write into afterwards.  ``tempfile.mkdtemp`` always asks for
0o700, so every ``tempfile`` based workflow (pip, venv/ensurepip, many
libraries) fails with ``PermissionError``.

Fix
---
On Windows the POSIX ``mode`` argument of ``mkdir`` has no portable meaning,
so we simply drop it.  Everything else is unchanged.
"""

import os
import sys

if sys.platform == "win32" and not getattr(os, "_dsh_mkdir_patch", False):
    _real_mkdir = os.mkdir

    def _mkdir(path, mode=0o777, *, dir_fd=None):
        # Ignore POSIX mode bits on Windows; keep dir_fd semantics if given.
        if dir_fd is None:
            return _real_mkdir(path)
        return _real_mkdir(path, dir_fd=dir_fd)

    os.mkdir = _mkdir
    os._dsh_mkdir_patch = True
