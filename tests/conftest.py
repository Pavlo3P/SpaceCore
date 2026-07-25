from __future__ import annotations
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _restore_ambient_check_level():
    """Restore the ambient validation level after each test.

    ``check_level`` moved off :class:`Context` onto the bound object; a test may
    set the process-wide ambient default via ``sc.set_check_level(...)`` to build
    objects at a given level. This fixture snapshots and restores it so that
    setting cannot leak between tests (avoids the Erratic Test smell).
    """
    import spacecore as sc

    previous = sc.get_check_level()
    yield
    sc.set_check_level(previous)
