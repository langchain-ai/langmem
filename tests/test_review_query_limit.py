from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.store.memory import InMemoryStore

from langmem.knowledge.extraction import MemoryStoreManager


@pytest.mark.parametrize("query_limit", [5, 15])
def test_invoke_respects_query_limit(query_limit):
    store = InMemoryStore()
    for i in range(20):
        store.put(
            ("memories",),
            str(i),
            {"kind": "Memory", "content": {"content": f"memory {i}"}},
        )
    manager = MemoryStoreManager(
        Mock(spec=BaseChatModel),
        query_limit=query_limit,
        store=store,
        query_model=None,
        namespace=("memories",),
    )
    manager.memory_manager = Mock()
    manager.memory_manager.invoke.return_value = []

    manager.invoke({"messages": [{"type": "human", "content": "hello"}]})

    existing = manager.memory_manager.invoke.call_args.args[0]["existing"]
    assert len(existing) == query_limit
