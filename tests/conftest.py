"""Configuration and fixtures for pytest."""
import tempfile
from pathlib import Path

import pytest


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)
