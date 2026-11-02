"""Regression test for MemoryStoreManager.invoke() double-search bug.

`MemoryStoreManager.invoke()` (the synchronous path) used to run the store
search twice for every dilated window when no `query_model` was configured:
once inline and once through an executor, discarding the first result. This
test asserts the store is searched exactly once per window, matching the
`ainvoke()` path.
"""

from unittest.mock import Mock

from langchain_core.language_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from langmem.knowledge.extraction import MemoryStoreManager


class CountingStore(InMemoryStore):
    """InMemoryStore that counts `search` invocations."""

    def __init__(self):
        super().__init__()
        self.search_count = 0

    def search(self, *args, **kwargs):
        self.search_count += 1
        return super().search(*args, **kwargs)


def test_invoke_searches_once_per_window():
    store = CountingStore()
    model = Mock(spec=BaseChatModel)

    manager = MemoryStoreManager(
        model,
        query_limit=5,  # query_limit // 4 == 1 -> a single dilated window
        store=store,
        query_model=None,
        namespace=("memories",),
    )
    # Avoid a real LLM call in the enrichment phase.
    manager.memory_manager = Mock()
    manager.memory_manager.invoke.return_value = []

    manager.invoke({"messages": [{"type": "human", "content": "hello"}]})

    # A single dilated window must produce exactly one store search.
    assert store.search_count == 1, f"expected 1 store search, got {store.search_count}"
