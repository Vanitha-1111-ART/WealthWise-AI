import os
from pathlib import __name__
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load env variables from backend/.env if it exists
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

class Settings(BaseSettings):
    # App Settings
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    PROJECT_NAME: str = "WealthWise AI - Digital Wealth Management Platform"

    # Database
    # Using async pg connection string by default
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql+asyncpg://postgres:postgres@localhost:5432/wealthwise"
    )

    # Security
    JWT_SECRET_KEY: str = os.getenv(
        "JWT_SECRET_KEY", 
        "9a15f9b457e51c890de2efcb67bca9c472390fcdb49237c1de72f0b7c3d2890a"
    )
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

    # Gemini API Key
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    class Config:
        case_sensitive = True
        extra = "allow"  # Allow extra env variables

settings = Settings()
