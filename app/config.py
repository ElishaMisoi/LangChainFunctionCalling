from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings, loaded from environment or `.env` file.

    Set values in a `.env` file at the project root (the repo contains
    `requirements.txt` with `python-dotenv`) or export them in your shell.
    """
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_api_version: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    open_meteo_geocode_url: Optional[str] = "https://geocoding-api.open-meteo.com/v1/search"
    open_meteo_forecast_url: Optional[str] = "https://api.open-meteo.com/v1/forecast"
    newsdata_base: Optional[str] = "https://newsdata.io/api/1"
    newsdata_api_key: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance. BaseSettings will read from the
    environment and the `.env` file specified in `model_config`.
    """
    # type: ignore[arg-type] - pydantic BaseSettings reads env at runtime
    return Settings()

class _LazySettingsProxy:
    def __getattr__(self, name):
        return getattr(get_settings(), name)

    def __repr__(self):
        return repr(get_settings())

# Backwards-compatible module-level symbol. Accessing attributes will create
# the real Settings instance on first use and cache it.
settings = _LazySettingsProxy()