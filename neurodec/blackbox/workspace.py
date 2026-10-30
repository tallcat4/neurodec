# SPDX-License-Identifier: GPL-3.0-or-later
"""Short-lived CLI files with the enclosing directory's Windows access rules."""
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def audio_workspace():
    if os.name != 'nt':
        with tempfile.TemporaryDirectory() as path:
            yield Path(path)
        return
    # Windows mkdir(mode=0o700), used by TemporaryDirectory, installs a new ACL
    # that can exclude the restricted token running the CLI. Inherit the user's
    # temp directory ACL instead; do not change permissions on existing paths.
    root = Path(tempfile.gettempdir()).resolve()
    path = root / ('neurodec-' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != root:
            raise RuntimeError('Temporary path escaped its parent')
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
