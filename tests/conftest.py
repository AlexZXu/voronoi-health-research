import tempfile
from pathlib import Path

import pytest

from app import create_app
from app.config import Config


class TestConfig(Config):
    TESTING = True
    INSTANCE_DIR = Path(tempfile.gettempdir()) / "voronoi-health-tests"
    STORE_PATH = INSTANCE_DIR / "state.json"
    UPLOAD_FOLDER = INSTANCE_DIR / "uploads"


@pytest.fixture()
def app():
    if TestConfig.STORE_PATH.exists():
        TestConfig.STORE_PATH.unlink()
    return create_app(TestConfig)


@pytest.fixture()
def client(app):
    return app.test_client()
