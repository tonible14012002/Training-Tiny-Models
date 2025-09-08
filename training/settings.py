import pathlib


BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
TRANSFORMER_CACHE_DIR = BASE_DIR / ".cache/models"
TRANSFORMER_DATASETS_DIR = BASE_DIR / ".cache/datasets"