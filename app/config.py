import os
from functools import lru_cache


class Settings:
    """Application configuration values derived from environment variables."""

    def __init__(self) -> None:
        self.google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.hospital_api_base_url: str = os.getenv(
            "HOSPITAL_API_BASE_URL", "http://34.93.7.250:8001/api"
        )
        self.hospital_client_id: str | None = os.getenv(
            "HOSPITAL_CLIENT_ID", "444d9283-88fa-4320-a084-1adb97085d41"
        )
        self.http_timeout_seconds: float = float(
            os.getenv("HTTP_TIMEOUT_SECONDS", "10")
        )

    def require_google_api_key(self) -> str:
        """Return the configured Google API key, raising if it is missing."""
        if not self.google_api_key:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY environment variable. "
                "Set it before starting the application."
            )
        return self.google_api_key

    def require_hospital_client_id(self) -> str:
        """Return the configured hospital client identifier, raising if absent."""
        if not self.hospital_client_id:
            raise RuntimeError(
                "Missing HOSPITAL_CLIENT_ID environment variable required for "
                "appointment bookings."
            )
        return self.hospital_client_id


@lru_cache
def get_settings() -> Settings:
    """Provide a cached Settings instance."""
    return Settings()


settings = get_settings()
