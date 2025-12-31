from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "WNG Activity Engine"
    ENVIRONMENT: Literal["local", "dev", "prod"] = "local"
    API_V1_PREFIX: str = "/api/v1"

    # Database
    # Use sqlite+aiosqlite for local development by default if no env var is provided
    DATABASE_URL: str = "sqlite+aiosqlite:///./dev.db"

    # Security
    SECRET_KEY: str = "change_this_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)


settings = Settings()
