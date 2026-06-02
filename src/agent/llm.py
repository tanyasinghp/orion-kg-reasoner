from typing import Any, override
from collections.abc import Mapping

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_ollama import ChatOllama

from config import settings


def get_llm() -> BaseChatModel:
    """Get the Large Language Model to be used by the agent."""
    llm = ChatOllama(
        model=settings.llm.ollama.model_name,
        temperature=settings.llm.ollama.temperature,
        num_ctx=settings.llm.ollama.context_window_size,
        num_predict=-2,
    )

    if settings.llm.enable_caching:
        set_llm_cache(InMemoryCache())

    return llm
