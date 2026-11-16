"""Messages dropped by pre-summary trimming are observable (issue #207).

`summarize_messages` removes the messages it summarizes from the returned context.
When the messages to summarize are bigger than `max_tokens`, the oldest of them are
trimmed before the summarization model call, so those messages leave the context
without ever reaching the model. They were not reported anywhere before: the only
signal, `RunningSummary.summarized_message_ids`, listed them too, because it is
built from the pre-trim window.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langmem.short_term.summarization import (
    RunningSummary,
    asummarize_messages,
    summarize_messages,
)

pytestmark = pytest.mark.anyio

# Character-counting counter, as in the issue's reproducer: deterministic and
# independent of any provider tokenizer.
char_counter = lambda items: sum(len(message.content) for message in items)  # noqa: E731


class RecordingModel:
    """Returns a fixed summary and records the messages it was given."""

    def __init__(self):
        self.received: list = []

    def invoke(self, messages):
        self.received = list(messages)
        return AIMessage(content="summary")

    async def ainvoke(self, messages):
        self.received = list(messages)
        return AIMessage(content="summary")


def test_dropped_before_summary_ids_expose_trimmed_messages():
    # max_tokens 8 with a two-message window of 14 characters: both messages are in
    # the summarize window, but only the newer one fits the summarization model.
    messages = [
        HumanMessage(id="m1", content="EARLY_A"),
        HumanMessage(id="m2", content="LATE002"),
    ]
    model = RecordingModel()

    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        max_tokens=8,
        max_summary_tokens=1,
        token_counter=char_counter,
    )

    sent = {message.id for message in model.received if message.id}
    summary = result.running_summary

    assert sent == {"m2"}, "the older message should not reach the summarization model"
    # Unchanged behavior: everything removed from the context is still reported here.
    assert summary.summarized_message_ids == {"m1", "m2"}
    assert summary.last_summarized_message_id == "m2"
    # New: the message that left the context without being summarized is visible.
    assert summary.dropped_before_summary_message_ids == {"m1"}
    assert summary.dropped_before_summary_message_ids == (
        summary.summarized_message_ids - sent
    )


async def test_dropped_before_summary_ids_expose_trimmed_messages_async():
    messages = [
        HumanMessage(id="m1", content="EARLY_A"),
        HumanMessage(id="m2", content="LATE002"),
    ]
    model = RecordingModel()

    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        max_tokens=8,
        max_summary_tokens=1,
        token_counter=char_counter,
    )

    sent = {message.id for message in model.received if message.id}
    summary = result.running_summary

    assert sent == {"m2"}
    assert summary.summarized_message_ids == {"m1", "m2"}
    assert summary.dropped_before_summary_message_ids == {"m1"}


def test_nothing_dropped_when_the_window_fits():
    # The window is exactly the summarization budget, so nothing is trimmed and the
    # summarization model sees every message.
    messages = [HumanMessage(id=f"m{i}", content=f"aaaa{i}") for i in range(1, 5)]
    model = RecordingModel()

    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        max_tokens=20,
        max_summary_tokens=5,
        token_counter=char_counter,
    )

    assert {message.id for message in model.received if message.id} == {
        "m1",
        "m2",
        "m3",
        "m4",
    }
    assert result.running_summary.dropped_before_summary_message_ids == set()


def test_dropped_ids_carry_forward_across_calls():
    summary = RunningSummary(
        summary="s",
        summarized_message_ids={"m1"},
        last_summarized_message_id="m1",
        dropped_before_summary_message_ids={"m0"},
    )
    messages = [HumanMessage(id=f"n{i}", content="bbbbbbbb") for i in range(1, 5)]
    model = RecordingModel()

    result = summarize_messages(
        messages,
        running_summary=summary,
        model=model,
        max_tokens=30,
        max_summary_tokens=1,
        token_counter=char_counter,
    )

    updated = result.running_summary
    # An earlier drop is remembered on a later call.
    assert "m0" in updated.dropped_before_summary_message_ids
    assert {"n1", "n2", "n3", "n4"} <= updated.summarized_message_ids


def test_running_summary_without_the_new_field_still_builds():
    # Older code and callers construct RunningSummary with three fields.
    summary = RunningSummary(
        summary="s", summarized_message_ids={"m1"}, last_summarized_message_id="m1"
    )
    assert summary.dropped_before_summary_message_ids == set()


def test_summarization_node_reports_dropped_messages():
    from langmem.short_term.summarization import SummarizationNode

    node = SummarizationNode(
        model=RecordingModel(),
        max_tokens=8,
        max_summary_tokens=1,
        token_counter=char_counter,
    )
    messages = [
        HumanMessage(id="m1", content="EARLY_A"),
        HumanMessage(id="m2", content="LATE002"),
    ]

    result = node.invoke({"messages": messages, "context": {}})

    summary = result["context"]["running_summary"]
    assert summary.dropped_before_summary_message_ids == {"m1"}
