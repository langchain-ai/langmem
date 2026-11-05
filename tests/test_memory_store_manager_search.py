import asyncio
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from langmem import create_memory_store_manager
from langmem.knowledge import extraction


class RecordingStore(InMemoryStore):
    def __init__(self):
        super().__init__(
            index={"dims": 2, "embed": lambda texts: [[1.0, 0.0] for _ in texts]}
        )
        self.search_count = 0

    def search(self, *args, **kwargs):
        self.search_count += 1
        return super().search(*args, **kwargs)

    async def asearch(self, *args, **kwargs):
        self.search_count += 1
        return await super().asearch(*args, **kwargs)


@pytest.mark.parametrize("query_limit", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_small_query_limits_retrieve_existing_memories(query_limit, asynchronous):
    store = RecordingStore()
    store.put(
        ("memories",),
        "known",
        {"kind": "Memory", "content": {"content": "prefers tea"}},
    )
    manager_inputs = []
    with patch.object(
        extraction,
        "create_memory_manager",
        return_value=RunnableLambda(lambda value: manager_inputs.append(value) or []),
    ):
        manager = create_memory_store_manager(
            FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
            store=store,
            namespace=("memories",),
            query_limit=query_limit,
        )

    request = {"messages": [HumanMessage(content="I prefer coffee")], "max_steps": 1}
    if asynchronous:
        asyncio.run(manager.ainvoke(request))
    else:
        manager.invoke(request)

    if query_limit == 0:
        assert store.search_count == 0
        assert manager_inputs[0]["existing"] == []
    else:
        assert store.search_count > 0
        assert len(manager_inputs[0]["existing"]) == 1
        assert manager_inputs[0]["existing"][0][2] == {"content": "prefers tea"}
