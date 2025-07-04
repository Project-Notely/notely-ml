from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Notely ML API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    class Config:
        env_file = ".env.local"


settings = Settings()
