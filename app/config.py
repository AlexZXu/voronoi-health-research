from pathlib import Path


class Config:
    SECRET_KEY = "dev-voronoi-secret"
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data_sets"
    INSTANCE_DIR = BASE_DIR / "instance"
    STORE_PATH = INSTANCE_DIR / "state.json"
    UPLOAD_FOLDER = INSTANCE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    DEFAULT_DISTANCE_MODE = "planar"
