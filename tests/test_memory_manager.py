import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

import langmem.knowledge.extraction as extraction


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


class Preference(BaseModel):
    category: str
    value: str


class RemoveDoc(BaseModel):
    json_doc_id: str


class CapturingExtractor:
    def __init__(self):
        self.payloads = []

    def _response(self, payload):
        self.payloads.append(payload)
        tool_calls = (
            [
                {
                    "name": "Triple",
                    "args": {},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ]
            if len(self.payloads) == 1
            else []
        )
        return {
            "responses": [],
            "response_metadata": [],
            "messages": [AIMessage(content="", tool_calls=tool_calls)],
        }

    def invoke(self, payload, config=None):
        return self._response(payload)

    async def ainvoke(self, payload, config=None):
        return self._response(payload)


class SequencedExtractor:
    def __init__(self, steps):
        self.payloads = []
        self.steps = steps

    def _response(self, payload):
        step_index = len(self.payloads)
        self.payloads.append(payload)
        responses, response_metadata = self.steps[step_index]
        tool_calls = [
            {
                "name": response.__repr_name__(),
                "args": {},
                "id": f"call-{index}",
                "type": "tool_call",
            }
            for index, response in enumerate(responses)
        ]
        return {
            "responses": responses,
            "response_metadata": response_metadata,
            "messages": [AIMessage(content="", tool_calls=tool_calls)],
        }

    def invoke(self, payload, config=None):
        return self._response(payload)

    async def ainvoke(self, payload, config=None):
        return self._response(payload)


def _setup_manager(monkeypatch, *, extractor=None, schemas=None):
    extractor = extractor or CapturingExtractor()
    monkeypatch.setattr(extraction, "init_chat_model", lambda model: object())
    monkeypatch.setattr(
        extraction, "create_extractor", lambda model, **kwargs: extractor
    )
    manager = extraction.MemoryManager(
        "fake-model",
        schemas=schemas or [Triple],
    )
    return manager, extractor


def _input():
    return {
        "messages": [HumanMessage(content="Alice still likes Python.")],
        "existing": [
            (
                "memory-1",
                "Triple",
                {
                    "subject": "Alice",
                    "predicate": "likes",
                    "object": "Python",
                },
            )
        ],
        "max_steps": 2,
    }


def _assert_schema_preserved(extractor):
    assert extractor.payloads[1]["existing"] == _input()["existing"]


def _assert_public_response_unchanged(result):
    assert result == [
        extraction.ExtractedMemory(
            id="memory-1",
            content={
                "subject": "Alice",
                "predicate": "likes",
                "object": "Python",
            },
        )
    ]


def test_invoke_preserves_schema_for_unchanged_dict_memory(monkeypatch):
    manager, extractor = _setup_manager(monkeypatch)

    result = manager.invoke(_input())

    _assert_schema_preserved(extractor)
    _assert_public_response_unchanged(result)


@pytest.mark.anyio
async def test_ainvoke_preserves_schema_for_unchanged_dict_memory(monkeypatch):
    manager, extractor = _setup_manager(monkeypatch)

    result = await manager.ainvoke(_input())

    _assert_schema_preserved(extractor)
    _assert_public_response_unchanged(result)


def test_updated_memory_does_not_restore_stale_content_or_other_schema(
    monkeypatch,
):
    updated = Triple(subject="Alice", predicate="likes", object="Rust")
    extractor = SequencedExtractor(
        [
            ([updated], [{"json_doc_id": "memory-1"}]),
            ([], []),
        ]
    )
    manager, _ = _setup_manager(
        monkeypatch,
        extractor=extractor,
        schemas=[Triple, Preference],
    )
    preference = {"category": "editor", "value": "vim"}
    input_ = {
        "messages": [HumanMessage(content="Alice now likes Rust.")],
        "existing": [
            (
                "memory-1",
                "Triple",
                {
                    "subject": "Alice",
                    "predicate": "likes",
                    "object": "Python",
                },
            ),
            ("memory-2", "Preference", preference),
        ],
        "max_steps": 2,
    }

    result = manager.invoke(input_)

    assert extractor.payloads[1]["existing"] == [
        extraction.ExtractedMemory(id="memory-1", content=updated),
        ("memory-2", "Preference", preference),
    ]
    assert result == [
        extraction.ExtractedMemory(id="memory-1", content=updated),
        extraction.ExtractedMemory(id="memory-2", content=preference),
    ]


def test_deleted_external_memory_is_not_restored_in_next_step(monkeypatch):
    removal = RemoveDoc(json_doc_id="memory-1")
    extractor = SequencedExtractor(
        [
            ([removal], [{}]),
            ([], []),
        ]
    )
    manager, _ = _setup_manager(monkeypatch, extractor=extractor)

    result = manager.invoke(_input())

    assert extractor.payloads[1]["existing"] == []
    assert result == [
        extraction.ExtractedMemory(id="memory-1", content=removal),
    ]
