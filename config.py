from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/causal"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "causal-artifacts"

    MAX_UPLOAD_MB: int = 200
    DEFAULT_N_BOOTSTRAP: int = 200
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"


settings = Settings()
