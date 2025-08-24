from __future__ import annotations
import requests
from functools import lru_cache
from urllib.parse import quote_plus
from typing import Optional, Dict, Any
from ..config import get_settings
from pydantic import BaseModel

def _geocode_url() -> str:
    return get_settings().open_meteo_geocode_url or "https://geocoding-api.open-meteo.com/v1/search"

def _forecast_url() -> str:
    return get_settings().open_meteo_forecast_url or "https://api.open-meteo.com/v1/forecast"

class WeatherResponse(BaseModel):
    location: str
    temperature_c: float | None
    windspeed_kmh: float | None
    winddirection_deg: float | None
    condition_code: int | None
    condition_label: str | None
    observed_at: str | None
    provider: str

# --- helpers ---
@lru_cache(maxsize=256)
def geocode_location(location: str) -> Dict[str, Any]:
    """Geocode a free-form location string using Open-Meteo Geocoding API.

    Returns a dict with keys name, country, latitude, longitude.
    Raises ValueError when location cannot be resolved.
    """
    # Build a simple URL-based query like: ?name={location}&count=1
    geo_url = f"{_geocode_url()}?name={quote_plus(location)}&count=1"
    resp = requests.get(geo_url, timeout=10)
    resp.raise_for_status()
    geo = resp.json()
    results = geo.get("results") or []
    if not results:
        raise ValueError(f"Could not find coordinates for '{location}'")
    r0 = results[0]
    return {
        "name": r0["name"],
        "country": r0.get("country"),
        "latitude": r0["latitude"],
        "longitude": r0["longitude"],
    }

def _weather_code_label(code: Optional[int]) -> Optional[str]:
    if code is None:
        return None
    # Open-Meteo WMO weather interpretation codes (condensed)
    if code == 0: return "Clear sky"
    if code in (1, 2, 3): return "Mainly clear / Partly cloudy / Overcast"
    if code in (45, 48): return "Fog"
    if code in (51, 53, 55): return "Drizzle"
    if code in (56, 57): return "Freezing drizzle"
    if code in (61, 63, 65): return "Rain"
    if code in (66, 67): return "Freezing rain"
    if code in (71, 73, 75): return "Snow"
    if code == 77: return "Snow grains"
    if code in (80, 81, 82): return "Rain showers"
    if code in (85, 86): return "Snow showers"
    if code == 95: return "Thunderstorm"
    if code in (96, 99): return "Thunderstorm with hail"
    return f"Code {code}"

# --- public API ---
def get_current_weather(location: str) -> WeatherResponse:
    """
    Fetch current weather for a free-form `location` string using Open-Meteo
    geocoding followed by the forecast API.

    Returns a `WeatherResponse` pydantic model suitable for API responses or
    for use as a tool return value.
    """
    loc = geocode_location(location)
    params = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "current_weather": True,
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
        "timezone": "auto",
    }

    resp = requests.get(_forecast_url(), params=params, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    current_weather = payload.get("current_weather")

    if not current_weather:
        raise RuntimeError("Open-Meteo did not return current_weather")
    code = current_weather.get("weathercode")

    result = {
        "location": ", ".join([v for v in [loc["name"], loc.get("country")] if v]),
        "temperature_c": current_weather.get("temperature"),
        "windspeed_kmh": current_weather.get("windspeed"),
        "winddirection_deg": current_weather.get("winddirection"),
        "condition_code": code,
        "condition_label": _weather_code_label(code),
        "observed_at": current_weather.get("time"),
        "provider": "open-meteo",
    }
    
    return WeatherResponse(**result)