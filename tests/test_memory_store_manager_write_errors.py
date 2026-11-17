"""Regression test for MemoryStoreManager.invoke() silently swallowing store write/delete errors.

The synchronous `invoke()` path submitted `store.put()`/`store.delete()` through
an executor without inspecting the returned futures, so a failed persistence
returned success. `ainvoke()` already propagates these errors via
`asyncio.gather`. This test asserts `invoke()` raises the store's error instead.
"""

from unittest.mock import Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from langmem.knowledge import extraction
from langmem.knowledge.extraction import ExtractedMemory, Memory, MemoryStoreManager


class UnavailableStore(InMemoryStore):
    """InMemoryStore whose writes always fail."""

    def put(self, namespace, key, value, **kwargs):
        raise OSError("store unavailable")


def test_invoke_propagates_store_write_errors():
    store = UnavailableStore()
    model = Mock(spec=BaseChatModel)
    output = [ExtractedMemory("preference", Memory(content="prefers tea"))]

    # Replace only model extraction with a fixed output; exercise the real
    # store manager, thread pool and failing store.
    with patch.object(
        extraction,
        "create_memory_manager",
        return_value=RunnableLambda(lambda _: output),
    ):
        manager = MemoryStoreManager(
            model,
            store=store,
            query_model=None,
            namespace=("memories",),
        )
        with pytest.raises(OSError, match="store unavailable"):
            manager.invoke(
                {"messages": [{"type": "human", "content": "I prefer tea."}]}
            )
