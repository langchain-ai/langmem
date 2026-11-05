"""Messages partially trimmed before summarization are observable (issue #207).

Pre-summary trimming can keep only the tail of a long message: the message ID
survives, so it is not reported in `dropped_before_summary_message_ids`, yet the
summarization model saw less than its full content.
`partially_trimmed_before_summary_message_ids` records those IDs so callers can
distinguish full retention, partial trim and complete drop.
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


def _three_case_messages():
    # The summarize window becomes [m0, m1, m2] (m3 stays in the context), and
    # the window does not fit the summarization budget: m0 is dropped whole,
    # m1 survives with only "TAIL", m2 is retained in full.
    return [
        HumanMessage(id="m0", content="AAAA"),
        HumanMessage(id="m1", content="EARLY\nTAIL"),
        HumanMessage(id="m2", content="zz"),
        HumanMessage(id="m3", content="LAST123"),
    ]


def test_partial_trim_three_cases():
    messages = _three_case_messages()
    model = RecordingModel()

    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        max_tokens=9,
        max_summary_tokens=1,
        token_counter=char_counter,
    )

    sent = {message.id: message.content for message in model.received if message.id}
    summary = result.running_summary

    # The model saw m1's tail only, plus m2 in full; m0 never reached it.
    assert sent == {"m1": "TAIL", "m2": "zz"}
    # Complete drop, partial trim and full retention are reported separately.
    assert summary.dropped_before_summary_message_ids == {"m0"}
    assert summary.partially_trimmed_before_summary_message_ids == {"m1"}
    assert "m2" not in summary.partially_trimmed_before_summary_message_ids
    # Unchanged behavior: everything removed from the context is still reported.
    assert summary.summarized_message_ids == {"m0", "m1", "m2"}
    assert summary.last_summarized_message_id == "m2"


async def test_partial_trim_three_cases_async():
    messages = _three_case_messages()
    model = RecordingModel()

    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        max_tokens=9,
        max_summary_tokens=1,
        token_counter=char_counter,
    )

    sent = {message.id: message.content for message in model.received if message.id}
    summary = result.running_summary

    assert sent == {"m1": "TAIL", "m2": "zz"}
    assert summary.dropped_before_summary_message_ids == {"m0"}
    assert summary.partially_trimmed_before_summary_message_ids == {"m1"}


def test_no_partial_trim_when_the_window_fits():
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
    assert result.running_summary.partially_trimmed_before_summary_message_ids == set()


def test_partial_trim_ids_carry_forward_across_calls():
    # A running summary saved before this field existed restores with an empty
    # set; tracking then starts with the next summarization.
    summary = RunningSummary(
        summary="s",
        summarized_message_ids={"x0"},
        last_summarized_message_id="x0",
        dropped_before_summary_message_ids=set(),
    )
    assert summary.partially_trimmed_before_summary_message_ids == set()

    messages = _three_case_messages()
    model = RecordingModel()

    result = summarize_messages(
        messages,
        running_summary=summary,
        model=model,
        max_tokens=9,
        max_summary_tokens=1,
        token_counter=char_counter,
    )

    updated = result.running_summary
    assert updated.partially_trimmed_before_summary_message_ids == {"m1"}
    assert updated.dropped_before_summary_message_ids == {"m0"}
    assert {"m0", "m1", "m2"} <= updated.summarized_message_ids
