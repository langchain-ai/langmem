"""Tests for update semantics in ``create_manage_memory_tool``.

Regression coverage for langmem#217: ``action="update"`` used to be a plain
``store.put``, so it silently re-created a memory that had been deleted and
created one for an id that never existed.
"""

import uuid

import pytest
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool

NAMESPACE = ("memories", "{user}")
ALICE = ("memories", "alice")
BOB = ("memories", "bob")
ALICE_CONFIG = {"configurable": {"user": "alice"}}
BOB_CONFIG = {"configurable": {"user": "bob"}}


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def tool(store):
    return create_manage_memory_tool(namespace=NAMESPACE, store=store)


def _create(tool, content: str, config=ALICE_CONFIG) -> str:
    return tool.invoke({"action": "create", "content": content}, config=config).split()[
        -1
    ]


def test_update_of_existing_memory_still_works(tool, store):
    memory_id = _create(tool, "Alice lives at 1 Example St")

    assert (
        tool.invoke(
            {
                "action": "update",
                "id": memory_id,
                "content": "Alice lives at 2 Example St",
            },
            config=ALICE_CONFIG,
        )
        == f"updated memory {memory_id}"
    )

    item = store.get(ALICE, memory_id)
    assert item is not None
    assert item.value == {"content": "Alice lives at 2 Example St"}


def test_update_of_deleted_memory_raises(tool, store):
    memory_id = _create(tool, "Alice lives at 1 Example St")
    assert tool.invoke({"action": "delete", "id": memory_id}, config=ALICE_CONFIG) == (
        f"Deleted memory {memory_id}"
    )
    assert store.get(ALICE, memory_id) is None

    with pytest.raises(ValueError, match=memory_id):
        tool.invoke(
            {
                "action": "update",
                "id": memory_id,
                "content": "Alice lives at 2 Example St",
            },
            config=ALICE_CONFIG,
        )

    assert store.get(ALICE, memory_id) is None


def test_update_of_never_existing_id_raises(tool, store):
    missing_id = str(uuid.uuid4())

    with pytest.raises(ValueError, match=missing_id):
        tool.invoke(
            {"action": "update", "id": missing_id, "content": "ghost"},
            config=ALICE_CONFIG,
        )

    assert store.get(ALICE, missing_id) is None


def test_update_is_scoped_to_the_resolved_namespace(tool, store):
    memory_id = _create(tool, "Alice likes tea")

    with pytest.raises(ValueError, match=memory_id):
        tool.invoke(
            {"action": "update", "id": memory_id, "content": "Bob likes tea"},
            config=BOB_CONFIG,
        )

    assert store.get(BOB, memory_id) is None
    assert store.get(ALICE, memory_id).value == {"content": "Alice likes tea"}


def test_error_message_points_the_model_at_search(tool):
    with pytest.raises(ValueError, match="Search for the memory again"):
        tool.invoke(
            {"action": "update", "id": str(uuid.uuid4()), "content": "x"},
            config=ALICE_CONFIG,
        )


def test_delete_of_missing_memory_is_still_idempotent(tool, store):
    missing_id = str(uuid.uuid4())
    assert tool.invoke({"action": "delete", "id": missing_id}, config=ALICE_CONFIG) == (
        f"Deleted memory {missing_id}"
    )
    assert store.get(ALICE, missing_id) is None


def test_create_with_id_is_rejected(tool):
    with pytest.raises(ValueError, match="cannot provide a MEMORY ID"):
        tool.invoke(
            {"action": "create", "id": str(uuid.uuid4()), "content": "x"},
            config=ALICE_CONFIG,
        )


@pytest.mark.anyio
async def test_async_update_of_existing_memory_still_works(tool, store):
    memory_id = await _acreate(tool, "Alice lives at 1 Example St")

    assert (
        await tool.ainvoke(
            {
                "action": "update",
                "id": memory_id,
                "content": "Alice lives at 2 Example St",
            },
            config=ALICE_CONFIG,
        )
        == f"updated memory {memory_id}"
    )

    item = store.get(ALICE, memory_id)
    assert item.value == {"content": "Alice lives at 2 Example St"}


@pytest.mark.anyio
async def test_async_update_of_deleted_memory_raises(tool, store):
    memory_id = await _acreate(tool, "Alice lives at 1 Example St")
    await tool.ainvoke({"action": "delete", "id": memory_id}, config=ALICE_CONFIG)

    with pytest.raises(ValueError, match=memory_id):
        await tool.ainvoke(
            {
                "action": "update",
                "id": memory_id,
                "content": "Alice lives at 2 Example St",
            },
            config=ALICE_CONFIG,
        )

    assert store.get(ALICE, memory_id) is None


@pytest.mark.anyio
async def test_async_update_of_never_existing_id_raises(tool, store):
    missing_id = str(uuid.uuid4())

    with pytest.raises(ValueError, match=missing_id):
        await tool.ainvoke(
            {"action": "update", "id": missing_id, "content": "ghost"},
            config=ALICE_CONFIG,
        )

    assert store.get(ALICE, missing_id) is None


async def _acreate(tool, content: str, config=ALICE_CONFIG) -> str:
    result = await tool.ainvoke({"action": "create", "content": content}, config=config)
    return result.split()[-1]
