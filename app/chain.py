from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import Dict
from functools import lru_cache
from .llm import get_llm

# System prompt keeps answers crisp but helpful.
SYSTEM = (
    "You are a concise, accurate AI assistant. "
    "Default to short, actionable answers. "
    "If uncertain, say so and suggest next steps."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Very simple in-memory session store (swap for Redis/DB in prod)
_session_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]

@lru_cache()
def get_core_chain():
    """Build and cache the core runnable chain (prompt | llm | output parser).

    This is lazy so import-time doesn't attempt to create the LLM client
    before environment is set up.
    """
    llm = get_llm()
    return prompt | llm | StrOutputParser()

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