from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from .chain import chat_chain
from .config import get_settings

app = FastAPI(
    title="LangChain Function Calling Bot",
    version="0.1.0",
    description=(
        "A small example FastAPI application demonstrating a conversational"
        " chain backed by LangChain and Azure OpenAI."
    ),
    openapi_tags=[
        {"name": "health", "description": "Health and readiness endpoints."},
        {"name": "chat", "description": "Chat endpoints for conversational LLM interactions."},
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
        return ChatResponse(output=result)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex)) from ex

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