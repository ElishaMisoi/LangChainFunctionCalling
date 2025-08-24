from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from ..config import get_settings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _newsdata_base() -> str:
    return get_settings().newsdata_base or "https://newsdata.io/api/1"

def _newsdata_api_key() -> str:
    key = get_settings().newsdata_api_key
    if not key:
        raise RuntimeError("newsdata_api_key is not set in settings")
    return key

# Session with sensible retries/backoff
_SESSION = requests.Session()
_RETRY = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))

def _make_request(endpoint: str, params: dict) -> dict:
    params = params.copy()
    params["apikey"] = _newsdata_api_key()
    url = f"{_newsdata_base()}/{endpoint}"
    resp = _SESSION.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()

def _normalize_article(a: dict) -> dict:
    return {
        "title": a.get("title"),
        "description": a.get("description"),
        "link": a.get("link") or a.get("url"),
        "source": (a.get("source") or {}).get("name") if isinstance(a.get("source"), dict) else a.get("source"),
        "pubDate": a.get("pubDate") or a.get("pubDateISO") or a.get("published_at"),
        "language": a.get("language"),
    }

def get_top_headlines(country: Optional[str] = None, category: Optional[str] = None,
                      language: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch recent/top headlines. Uses the 'latest' endpoint (or 'search' fallback if needed).
    """
    params = {}
    if country:
        params["country"] = country
    if category:
        params["category"] = category
    if language:
        params["language"] = language
    data = _make_request("latest", params)
    articles = data.get("results") or data.get("articles") or []
    normalized = [_normalize_article(a) for a in articles]
    return normalized[:limit]

def search_news(q: str, language: Optional[str] = None,
                from_date: Optional[str] = None, to_date: Optional[str] = None,
                limit: int = 5) -> List[Dict[str, Any]]:
    """
    Keyword search across news. Use ISO dates for from_date/to_date if desired (YYYY-MM-DD).
    """
    params = {"q": q}
    if language:
        params["language"] = language
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    data = _make_request("news", params)
    articles = data.get("results") or data.get("articles") or []
    normalized = [_normalize_article(a) for a in articles]
    return normalized[:limit]

def get_sources() -> List[Dict[str, Any]]:
    """Return available sources (id, name, category)."""
    data = _make_request("sources", {})
    return data.get("sources", [])