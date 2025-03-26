"""Test configuration."""

from pathlib import Path

import pytest


@pytest.fixture
def data_dir() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "data"
