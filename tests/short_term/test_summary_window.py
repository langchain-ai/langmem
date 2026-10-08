import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langmem.short_term.summarization import (
    SummarizationNode,
    asummarize_messages,
    summarize_messages,
)
from tests.short_term.utils import FakeChatModel


def summarize(messages, model, asynchronous, **kwargs):
    options = {
        "running_summary": None,
        "model": model,
        "max_tokens": 8,
        "max_tokens_before_summary": 6,
        "summary_window": 2,
        "max_summary_tokens": 1,
        "token_counter": len,
        **kwargs,
    }
    if asynchronous:
        return asyncio.run(asummarize_messages(messages, **options))
    return summarize_messages(messages, **options)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_summary_window_waits_for_trigger_then_summarizes_oldest_prefix(asynchronous):
    model = FakeChatModel(responses=[AIMessage(content="summary")])
    messages = [HumanMessage(content=str(i), id=str(i)) for i in range(6)]

    before_trigger = summarize(messages[:5], model, asynchronous)
    assert before_trigger.running_summary is None
    assert before_trigger.messages == messages[:5]
    assert len(model.invoke_calls) == 0

    at_trigger = summarize(messages, model, asynchronous)
    assert at_trigger.running_summary is not None
    assert at_trigger.running_summary.summarized_message_ids == {"0", "1"}
    assert at_trigger.messages[1:] == messages[2:]
    assert len(model.invoke_calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_summary_window_expands_to_fit_remaining_context(asynchronous):
    model = FakeChatModel(responses=[AIMessage(content="summary")])
    messages = [HumanMessage(content=str(i), id=str(i)) for i in range(8)]

    result = summarize(messages, model, asynchronous, max_tokens=6)

    assert result.running_summary is not None
    assert result.running_summary.summarized_message_ids == {"0", "1", "2"}
    assert result.messages[1:] == messages[3:]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_summary_window_reuses_running_summary_without_duplicate_messages(asynchronous):
    model = FakeChatModel(responses=[AIMessage(content="summary")])
    messages = [HumanMessage(content=str(i), id=str(i)) for i in range(6)]
    first = summarize(messages, model, asynchronous)

    second = summarize(
        messages + [HumanMessage(content="6", id="6")],
        model,
        asynchronous,
        running_summary=first.running_summary,
    )

    assert second.running_summary is not None
    assert second.running_summary.summarized_message_ids == {"0", "1", "2", "3"}
    assert second.messages[1:] == messages[4:] + [HumanMessage(content="6", id="6")]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_summary_window_rejects_non_positive_values(asynchronous):
    model = FakeChatModel(responses=[])
    messages = [HumanMessage(content="hello", id="0")]

    with pytest.raises(ValueError, match="summary_window"):
        summarize(messages, model, asynchronous, summary_window=0)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_summarization_node_forwards_summary_window(asynchronous):
    model = FakeChatModel(responses=[AIMessage(content="summary")])
    messages = [HumanMessage(content=str(i), id=str(i)) for i in range(6)]
    node = SummarizationNode(
        model=model,
        max_tokens=8,
        max_tokens_before_summary=6,
        summary_window=2,
        max_summary_tokens=1,
        token_counter=len,
    )

    result = (
        asyncio.run(node.ainvoke({"messages": messages}))
        if asynchronous
        else node.invoke({"messages": messages})
    )

    assert result["context"]["running_summary"].summarized_message_ids == {"0", "1"}
    assert result["summarized_messages"][1:] == messages[2:]
