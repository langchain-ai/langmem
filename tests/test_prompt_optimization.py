from types import SimpleNamespace
from typing import Any

import pytest

from langmem.prompts import optimization


class _StubOptimizer:
    def invoke(self, *args: Any, **kwargs: Any) -> str:
        return "unchanged"

    async def ainvoke(self, *args: Any, **kwargs: Any) -> str:
        return "unchanged"


class _StubClassifier:
    def invoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"responses": [SimpleNamespace(which=[])]}

    async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"responses": [SimpleNamespace(which=[])]}


def _multi_prompt_input() -> dict[str, Any]:
    return {
        "trajectories": "No prompt needs updating.",
        "prompts": [
            {"name": "first", "prompt": "First prompt"},
            {"name": "second", "prompt": "Second prompt"},
        ],
    }


def _create_optimizer(monkeypatch: pytest.MonkeyPatch, tool_choices: list[str]):
    monkeypatch.setattr(
        optimization,
        "create_prompt_optimizer",
        lambda *args, **kwargs: _StubOptimizer(),
    )

    def create_extractor(*args: Any, tool_choice: str, **kwargs: Any):
        tool_choices.append(tool_choice)
        return _StubClassifier()

    monkeypatch.setattr(optimization, "create_extractor", create_extractor)
    return optimization.MultiPromptOptimizer("openai:test")


def test_multi_prompt_optimizer_uses_portable_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_choices: list[str] = []
    optimizer = _create_optimizer(monkeypatch, tool_choices)

    optimizer.invoke(_multi_prompt_input())

    assert tool_choices == ["any"]


@pytest.mark.anyio
async def test_multi_prompt_optimizer_uses_portable_tool_choice_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_choices: list[str] = []
    optimizer = _create_optimizer(monkeypatch, tool_choices)

    await optimizer.ainvoke(_multi_prompt_input())

    assert tool_choices == ["any"]
