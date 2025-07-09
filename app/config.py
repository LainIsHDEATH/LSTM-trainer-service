import os
from pathlib import Path

BASE_DIR = Path(os.getenv("MODELS_DIR", "../models"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "storagedb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
STORE_API = os.getenv("STORE_API", "http://localhost:8082/api")
