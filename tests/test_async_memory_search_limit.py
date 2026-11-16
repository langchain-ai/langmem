import asyncio
from unittest.mock import patch

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
        self.search_limits = []

    async def asearch(self, *args, **kwargs):
        self.search_limits.append(kwargs.get("limit"))
        return await super().asearch(*args, **kwargs)


def test_async_search_honors_query_limit_above_store_default():
    store = RecordingStore()
    for index in range(15):
        store.put(
            ("memories",),
            f"memory-{index}",
            {"kind": "Memory", "content": {"content": f"memory {index}"}},
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
            query_limit=20,
        )

    asyncio.run(
        manager.ainvoke(
            {
                "messages": [HumanMessage(content="Tell me about my memories")],
                "max_steps": 1,
            }
        )
    )

    assert store.search_limits == [20]
    assert len(manager_inputs[0]["existing"]) == 15
