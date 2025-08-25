from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.chains.openai_functions import create_openai_fn_chain
from typing import Dict
from functools import lru_cache
import threading
from .llm import get_llm
from .services.weather_service import get_current_weather
from .services.news_service import search_news
from langchain.tools import tool


# System prompt keeps answers crisp but helpful.
SYSTEM = (
    "You are a concise, accurate AI assistant. "
    "You can call functions to get weather or news. "
    "Default to short, actionable answers. "
    "If uncertain, say so and suggest next steps."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# --- Register tools for function calling ---
@tool
def weather_tool(location: str):
    """Get the current weather for a location."""
    return get_current_weather(location).dict()

from typing import Optional

@tool
def news_tool(
    q: str,
    language: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 5
):
    """Search news articles by query and optional filters."""
    return search_news(q, language, from_date, to_date, limit)

TOOLS = [weather_tool, news_tool]

# Very simple in-memory session store (swap for Redis/DB in prod)
_session_store: Dict[str, InMemoryChatMessageHistory] = {}
_session_store_lock = threading.Lock()

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    with _session_store_lock:
        if session_id not in _session_store:
            _session_store[session_id] = InMemoryChatMessageHistory()
        return _session_store[session_id]

# --- Chain with function calling ---
@lru_cache()
def get_core_chain():
    """Build and cache the core runnable chain with function calling for Azure."""
    from langchain.agents import initialize_agent, AgentType
    llm = get_llm()
    agent = initialize_agent(
        TOOLS,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
    )
    return agent

@lru_cache()
def make_chat_chain():
    core_chain = get_core_chain()
    return RunnableWithMessageHistory(
        core_chain,
        lambda session_id: get_session_history(session_id),
        input_messages_key="input",
        history_messages_key="history",
    )

class _LazyChatChain:
    def __getattr__(self, name):
        return getattr(make_chat_chain(), name)

# Backwards-compatible module-level symbol.
chat_chain = _LazyChatChain()