import uuid

import pytest
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool

CONFIG = {"configurable": {"user": "alice"}}
NAMESPACE = ("memories", "alice")


def make_tool():
    store = InMemoryStore()
    tool = create_manage_memory_tool(namespace=("memories", "{user}"), store=store)
    return tool, store


def created_id(result: str) -> str:
    # The tool replies "created memory <id>".
    return result.split()[-1]


def test_update_existing_memory_overwrites_it():
    tool, store = make_tool()
    memory_id = created_id(tool.invoke({"content": "likes tea"}, config=CONFIG))

    tool.invoke(
        {"action": "update", "id": memory_id, "content": "likes coffee"}, config=CONFIG
    )

    assert store.get(NAMESPACE, memory_id).value == {"content": "likes coffee"}


def test_update_of_deleted_memory_does_not_recreate_it():
    tool, store = make_tool()
    memory_id = created_id(tool.invoke({"content": "home"}, config=CONFIG))
    tool.invoke({"action": "delete", "id": memory_id}, config=CONFIG)

    with pytest.raises(ValueError, match="does not exist"):
        tool.invoke(
            {"action": "update", "id": memory_id, "content": "new home"},
            config=CONFIG,
        )

    assert store.get(NAMESPACE, memory_id) is None


def test_update_of_unknown_id_does_not_create_a_memory():
    tool, store = make_tool()
    unknown = str(uuid.uuid4())

    with pytest.raises(ValueError, match="does not exist"):
        tool.invoke({"action": "update", "id": unknown, "content": "x"}, config=CONFIG)

    assert store.search(NAMESPACE) == []


@pytest.mark.anyio
async def test_async_update_of_deleted_memory_does_not_recreate_it():
    tool, store = make_tool()
    memory_id = created_id(await tool.ainvoke({"content": "tea"}, config=CONFIG))
    await tool.ainvoke({"action": "delete", "id": memory_id}, config=CONFIG)

    with pytest.raises(ValueError, match="does not exist"):
        await tool.ainvoke(
            {"action": "update", "id": memory_id, "content": "likes coffee"},
            config=CONFIG,
        )

    assert await store.aget(NAMESPACE, memory_id) is None
