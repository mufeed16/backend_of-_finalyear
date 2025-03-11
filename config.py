import os
from datetime import timedelta
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
VECTOR_STORE = BASE_DIR / "vectorstore" / "db_faiss"

# Flask Configuration
class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/chatbot_db")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.urandom(24).hex())
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    UPLOADS_DEFAULT_DEST = str(UPLOAD_FOLDER)
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    ADMIN_KEY = os.getenv("ADMIN_KEY", "adminkey")

def init_app(app):
    """Initialize Flask application configuration"""
    app.config.from_object(Config)
    
    # Ensure required directories exist
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    VECTOR_STORE.mkdir(parents=True, exist_ok=True)

    return app

# Data paths
DATA_DIR = str(UPLOAD_FOLDER)
