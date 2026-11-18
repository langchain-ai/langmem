from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from langmem.knowledge import extraction


class RecordingStore(InMemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ):
        self.search_calls.append(
            {
                "namespace_prefix": namespace_prefix,
                "query": query,
                "limit": limit,
            }
        )
        return super().search(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )


def test_invoke_without_query_model_searches_each_window_once(monkeypatch) -> None:
    monkeypatch.setattr(
        extraction,
        "create_memory_manager",
        lambda *args, **kwargs: RunnableLambda(lambda _: []),
    )
    store = RecordingStore()
    manager = extraction.MemoryStoreManager(
        FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        store=store,
        query_limit=5,
        namespace=("memories",),
    )

    manager.invoke(
        {
            "messages": [HumanMessage(content="I prefer dark mode.")],
            "max_steps": 1,
        }
    )

    assert len(store.search_calls) == 1
    assert store.search_calls[0]["limit"] == 5
