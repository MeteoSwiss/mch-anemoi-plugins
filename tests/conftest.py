"""Test configuration."""

from pathlib import Path
import os
import sys

import pytest


eccodes_definitions = Path(sys.prefix) / "share" / "eccodes-cosmo-resources" / "definitions"
os.environ["ECCODES_DEFINITION_PATH"] = str(eccodes_definitions)

@pytest.fixture
def data_dir() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "data"
