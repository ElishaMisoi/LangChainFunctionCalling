from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from .chain import chat_chain
from langchain_core.messages import AIMessage
from .config import get_settings
from typing import Optional
from .services.weather_service import get_current_weather, WeatherResponse
from .services.news_service import get_top_headlines, search_news

app = FastAPI(
    title="LangChain Function Calling Bot",
    version="0.1.0",
    description=(
        "A small example FastAPI application demonstrating a conversational"
        " chain backed by LangChain, Function Calling and Azure OpenAI."
    ),
    openapi_tags=[
        {"name": "health", "description": "Health and readiness endpoints."},
        {"name": "chat", "description": "Chat endpoints for conversational LLM interactions."},
    {"name": "weather", "description": "Weather lookup endpoints (Open-Meteo)."},
    {"name": "news", "description": "News lookup endpoints (NewsData)."},
    ],
)

class ChatRequest(BaseModel):
    input: str = Field(..., description="User message", example="What can you do for me?")  # type: ignore[arg-type]
    session_id: str = Field(default="default", description="Conversation/session id", example="user-123")  # type: ignore[arg-type]

class ChatResponse(BaseModel):
    """Response wrapper containing the assistant's textual reply."""
    output: str = Field(..., description="Assistant reply text", example="I can provide information, answer questions, help with problem-solving, generate ideas, and assist with tasks like writing, planning, or learning. Let me know what you need!.")  # type: ignore[arg-type]

@app.get("/healthz", tags=["health"], summary="Health check", response_description="Service health status")
def healthz():
    """Simple liveness/readiness endpoint.

    Returns a JSON object with the internal service status. Useful for load
    balancers and runtime checks.
    """
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect the root path to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")

@app.get("/swagger", include_in_schema=False)
def swagger_redirect():
    """Backward-compatible route that redirects to Swagger UI at /docs."""
    return RedirectResponse(url="/docs")
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Send a chat message",
    response_description="Assistant reply",
)
def chat(req: ChatRequest):
    try:
        result = chat_chain.invoke(
            {"input": req.input},
            config={"configurable": {"session_id": req.session_id}}
        )
        output = None
        if isinstance(result, dict):
            if "output" in result:
                output = result["output"]
            elif "history" in result:
                history = result["history"]
                for msg in reversed(history):
                    if isinstance(msg, AIMessage):
                        output = msg.content
                        break
                    # Fallback: dict with type 'ai'
                    if isinstance(msg, dict) and msg.get("type") == "ai":
                        output = msg.get("content")
                        break
                if output is None:
                    output = str(result)
            else:
                output = str(result)
        else:
            output = str(result)
        # Ensure output is always a string
        if not isinstance(output, str):
            output = str(output)
        return ChatResponse(output=output)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex)) from ex

@app.get(
    "/weather/current",
    response_model=WeatherResponse,
    tags=["weather"],
    summary="Get current weather for a location",
    response_description="Current weather data for the requested location",
)
def weather_current(
    location: str = Query(..., description="Free-form location name, e.g., 'Nairobi' or 'Amboseli National Park'"),
):
    """Return the current weather for a location.

    Query parameters:
    - location: free-form location string (e.g., 'Nairobi' or 'Amboseli National Park')

    Responses
    - 200: A `WeatherResponse` object with temperature, wind, and condition.
    - 400: Bad request when the location cannot be resolved or the provider
      returns an error.
    """
    try:
        return get_current_weather(location)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    
@app.get(
    "/news/top",
    tags=["news"],
    summary="Get top news headlines",
    response_description="A list of top news headlines based on the provided filters",
)
def news_top(
    country: Optional[str] = Query(None, description="Country code, e.g., 'us'"),
    category: Optional[str] = Query(None, description="News category, e.g., 'technology'"),
    language: Optional[str] = Query(None, description="Language code, e.g., 'en'"),
    limit: int = Query(5, ge=1, le=50, description="Number of articles to return (1-50)"),
):
    """Fetch top news headlines.

    Query parameters:
    - country: Optional country code to filter news (e.g., 'us').
    - category: Optional category to filter news (e.g., 'technology').
    - language: Optional language code to filter news (e.g., 'en').
    - limit: Number of articles to return (default: 5, max: 50).

    Responses:
    - 200: A list of top news articles.
    - 500: Internal server error if the news service fails.
    """
    try:
        return {"articles": get_top_headlines(country, category, language, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/news/search",
    tags=["news"],
    summary="Search news articles",
    response_description="A list of news articles matching the search query",
)
def news_search(
    q: str = Query(..., description="Search query string, e.g., 'climate change'"),
    language: Optional[str] = Query(None, description="Language code, e.g., 'en'"),
    from_date: Optional[str] = Query(None, description="Start date for the search, e.g., '2025-01-01'"),
    to_date: Optional[str] = Query(None, description="End date for the search, e.g., '2025-01-31'"),
    limit: int = Query(5, ge=1, le=50, description="Number of articles to return (1-50)"),
):
    """Search for news articles.

    Query parameters:
    - q: Required search query string (e.g., 'climate change').
    - language: Optional language code to filter news (e.g., 'en').
    - from_date: Optional start date for the search (e.g., '2025-01-01').
    - to_date: Optional end date for the search (e.g., '2025-01-31').
    - limit: Number of articles to return (default: 5, max: 50).

    Responses:
    - 200: A list of news articles matching the search query.
    - 500: Internal server error if the news service fails.
    """
    try:
        return {"articles": search_news(q, language, from_date, to_date, limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def validate_settings():
    """Validate required configuration at application startup and fail fast
    with a clear error message if any required setting is missing."""
    settings = get_settings()
    required = [
        "azure_openai_endpoint",
        "azure_openai_api_key",
        "azure_openai_api_version",
        "azure_openai_deployment",
    ]
    missing = [k for k in required if not getattr(settings, k)]
    if missing:
        raise RuntimeError(
            f"Missing required configuration values: {', '.join(missing)}. "
            "Please populate .env (see .env.example) or set environment variables."
        )