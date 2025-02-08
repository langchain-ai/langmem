"""Knowledge extraction and semantic memory management.

This module provides utilities for extracting and managing semantic knowledge from conversations:

1. Functional Transformations:
    - `create_memory_manager(model, schemas=None) -> ((messages, existing=None) -> list[tuple[str, Memory]])`:
        Extract structured information from conversations
    - `create_thread_extractor(model, schema=None) -> ((messages) -> Summary)`:
        Generate structured summaries from conversations

2. Stateful Operations:
    Components that persist and manage memories in LangGraph's BaseStore:
    - `create_memory_store_manager(model, store=None) -> ((messages) -> None)`:
        Apply enrichment with integrated storage
    - `create_manage_memory_tool(store=None) -> Tool[dict, str]`:
        Tool for creating/updating stored memories
    - `create_search_memory_tool(store=None) -> Tool[dict, list[Memory]]`:
        Tool for searching stored memories

"""

from langmem.knowledge.extraction import (
    MemoryPhase,
    create_memory_manager,
    create_memory_searcher,
    create_memory_store_manager,
    create_thread_extractor,
)
from langmem.knowledge.tools import create_manage_memory_tool, create_search_memory_tool

__all__ = [
    "create_manage_memory_tool",
    "create_search_memory_tool",
    "create_memory_manager",
    "create_memory_searcher",
    "create_memory_store_manager",
    "create_thread_extractor",
    "MemoryPhase",
]
