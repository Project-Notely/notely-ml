import os
import sys

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application settings
    PROJECT_NAME: str = "Notely ML API"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    GEMINI_API_KEY: str

    MONGODB_URL: str | None = None
    DATABASE_NAME: str | None = "notely"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_gemini_api_key(cls, v):
        if not v or not v.strip():
            raise ValueError("GEMINI_API_KEY is required and cannot be empty")
        if not v.startswith("AI"):
            raise ValueError(
                "GEMINI_API_KEY must start with 'AI' (invalid API key format)"
            )
        return v.strip()

    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v):
        if not isinstance(v, int) or v < 1 or v > 65535:
            raise ValueError("PORT must be a valid port number (1-65535)")
        return v

    @field_validator("MONGODB_URL")
    @classmethod
    def validate_mongodb_url(cls, v):
        if v is not None and v.strip():
            if not v.startswith(("mongodb://", "mongodb+srv://")):
                raise ValueError(
                    "MONGODB_URL must start with 'mongodb://' or 'mongodb+srv://'"
                )
            return v.strip()
        return v

    model_config = SettingsConfigDict(env_file=".env.local", case_sensitive=True)


class _SettingsProxy:
    """Proxy class that provides lazy loading of settings with clean attribute access."""

    def __init__(self):
        self._settings: Settings | None = None

    def _load_settings(self) -> Settings:
        """Load and validate settings with comprehensive error handling."""
        if self._settings is not None:
            return self._settings

        try:
            # Check if .env.local exists
            env_file = ".env.local"
            if not os.path.exists(env_file):
                print(f"❌ Configuration Error: {env_file} file not found")
                print(
                    f"   Create a {env_file} file with required environment variables"
                )
                sys.exit(1)

            # Load and validate settings
            self._settings = Settings()

            # Success message (only show once)
            print("✅ Configuration loaded and validated successfully")
            print(
                f"   Project: {self._settings.PROJECT_NAME} v{self._settings.VERSION}"
            )
            print(
                f"   Environment: {'Development' if self._settings.DEBUG else 'Production'}"
            )

            return self._settings

        except ValidationError as e:
            print("❌ Configuration Validation Failed:")
            print("   Please fix the following issues in your .env.local file:\n")

            for error in e.errors():
                field = error.get("loc", ["unknown"])[0]
                message = error.get("msg", "Unknown validation error")

                if "missing" in message.lower():
                    print(f"   • {field}: Missing required environment variable")
                else:
                    print(f"   • {field}: {message}")

            print("\n   Example .env.local file:")
            print("   GEMINI_API_KEY=AIza...")
            print("   DEBUG=true")
            print("   PORT=8000")

            sys.exit(1)

        except Exception as e:
            print(f"❌ Unexpected configuration error: {e}")
            sys.exit(1)

    def __getattr__(self, name: str):
        """Lazy load settings on first attribute access."""
        settings_obj = self._load_settings()
        return getattr(settings_obj, name)

    def init(self) -> Settings:
        """Explicitly initialize configuration for fail-fast behavior."""
        return self._load_settings()


# Export the settings object - use like: settings.GEMINI_API_KEY
settings = _SettingsProxy()
