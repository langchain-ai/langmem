import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

import langmem.knowledge.extraction as extraction

pytestmark = pytest.mark.anyio


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str


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


def _setup_manager(monkeypatch):
    extractor = CapturingExtractor()
    monkeypatch.setattr(extraction, "init_chat_model", lambda model: object())
    monkeypatch.setattr(
        extraction, "create_extractor", lambda model, **kwargs: extractor
    )
    manager = extraction.MemoryManager("fake-model", schemas=[Triple])
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


async def test_ainvoke_preserves_schema_for_unchanged_dict_memory(monkeypatch):
    manager, extractor = _setup_manager(monkeypatch)

    result = await manager.ainvoke(_input())

    _assert_schema_preserved(extractor)
    _assert_public_response_unchanged(result)
