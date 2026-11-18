from typing import Any

import pytest
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.config import get_store
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from langmem import ReflectionExecutor
from langmem.utils import NamespaceTemplate


class MemoryReflector(Runnable):
    namespace = NamespaceTemplate(("memories",))

    def invoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict:
        if input:
            get_store().put(self.namespace(), input["key"], input)
        return input


def test_runtime_store_reaches_background_reflector():
    store = InMemoryStore()
    memory = {"key": "preference", "text": "Prefers concise answers"}

    with ReflectionExecutor(MemoryReflector()) as executor:

        @entrypoint(store=store)
        def remember(payload):
            return executor.submit(payload, thread_id=None).result(timeout=3)

        assert remember.invoke(memory) == memory
        assert store.get(("memories",), "preference").value == memory


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["search", "asearch"])
async def test_search_uses_inferred_store(asynchronous):
    store = InMemoryStore()
    memory = {"text": "Prefers concise answers"}
    store.put(("memories",), "preference", memory)

    with ReflectionExecutor(MemoryReflector()) as executor:

        @entrypoint(store=store)
        def bind_store(payload):
            return executor.submit(payload, thread_id=None).result(timeout=3)

        # A no-op reflector isolates retrieval from background writes.
        bind_store.invoke({})
        results = await executor.asearch() if asynchronous else executor.search()

        assert len(results) == 1
        assert results[0]["key"] == "preference"
        assert results[0]["value"] == memory
        assert tuple(results[0]["namespace"]) == ("memories",)


def test_inferred_store_is_reused_without_runtime_context():
    store = InMemoryStore()
    memory = {"key": "later", "text": "Remember this after the graph returns"}

    with ReflectionExecutor(MemoryReflector()) as executor:

        @entrypoint(store=store)
        def bind_store(payload):
            return executor.submit(payload, thread_id=None).result(timeout=3)

        bind_store.invoke({})
        future = executor.submit(memory, {"configurable": {}}, thread_id=None)

        assert future.result(timeout=3) == memory
        assert store.get(("memories",), "later").value == memory


def test_explicit_store_takes_precedence_over_runtime_store():
    explicit_store = InMemoryStore()
    runtime_store = InMemoryStore()
    memory = {"key": "preference", "text": "Use the explicitly supplied store"}

    with ReflectionExecutor(MemoryReflector(), store=explicit_store) as executor:

        @entrypoint(store=runtime_store)
        def remember(payload):
            return executor.submit(payload, thread_id=None).result(timeout=3)

        assert remember.invoke(memory) == memory
        assert explicit_store.get(("memories",), "preference").value == memory
        assert runtime_store.get(("memories",), "preference") is None
        assert executor.search()[0]["value"] == memory


@pytest.mark.parametrize("in_graph", [False, True], ids=["no-runtime", "no-store"])
def test_missing_store_rejects_submission(in_graph):
    with ReflectionExecutor(MemoryReflector()) as executor:

        @entrypoint()
        def remember(payload):
            return executor.submit(payload, thread_id=None).result(timeout=3)

        with pytest.raises(ValueError, match="could not resolve store"):
            if in_graph:
                remember.invoke({})
            else:
                executor.submit({}, {"configurable": {}}, thread_id=None)


def test_missing_store_does_not_prevent_later_valid_submission():
    store = InMemoryStore()
    memory = {"key": "recovered", "text": "Store became available"}

    with ReflectionExecutor(MemoryReflector()) as executor:
        with pytest.raises(ValueError, match="could not resolve store"):
            executor.submit({}, {"configurable": {}}, thread_id=None)

        @entrypoint(store=store)
        def remember(payload):
            return executor.submit(payload, thread_id=None).result(timeout=3)

        assert remember.invoke(memory) == memory
        assert store.get(("memories",), "recovered").value == memory
