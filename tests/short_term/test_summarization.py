from typing import List

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages.utils import count_tokens_approximately

from langmem.short_term.summarization import summarize_messages


class FakeChatModel(FakeMessagesListChatModel):
    """Mock chat model for testing the summarizer."""

    invoke_calls: list[list[BaseMessage]] = []

    def __init__(self, responses: list[BaseMessage]):
        """Initialize with predefined responses."""
        super().__init__(
            responses=responses or [AIMessage(content="This is a mock summary.")]
        )

    def invoke(self, input: List[BaseMessage]) -> AIMessage:
        """Mock invoke method that returns predefined responses."""
        self.invoke_calls.append(input)
        return super().invoke(input)

    def bind(self, **kwargs):
        """Mock bind method that returns self."""
        return self


def test_empty_input():
    model = FakeChatModel(responses=[])

    # Test with empty message list
    result = summarize_messages(
        [],
        running_summary=None,
        model=model,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == []
    assert len(model.invoke_calls) == 0

    # Test with only system message
    system_msg = SystemMessage(content="You are a helpful assistant.", id="sys")
    result = summarize_messages(
        [system_msg],
        running_summary=None,
        model=model,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == [system_msg]


def test_no_summarization_needed():
    model = FakeChatModel(responses=[])

    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
    ]

    # Tokens are under the limit, so no summarization should occur
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == messages
    assert len(model.invoke_calls) == 0  # Model should not have been called


def test_summarize_too_many_tokens():
    model = FakeChatModel(responses=[])
    # base case: no system message:
    # - max tokens to summarize = 5
    # - max remaining tokens = 4
    # - max total tokens = 9
    with pytest.raises(ValueError):
        summarize_messages(
            [AIMessage(content=f"Message {i}", id=f"{i}") for i in range(10)],
            running_summary=None,
            model=model,
            token_counter=len,
            max_tokens=5,
            max_summary_tokens=1,
        )

    # system message:
    # - max tokens to summarize = 5
    # - max remaining tokens = 3
    # - max total tokens = 8
    with pytest.raises(ValueError):
        summarize_messages(
            # will raise if > 8 (excluding system message)
            [SystemMessage(content="You are a helpful assistant.", id="system")]
            + [AIMessage(content=f"Message {i}", id=f"{i}") for i in range(9)],
            running_summary=None,
            model=model,
            token_counter=len,
            max_tokens=5,
            max_summary_tokens=1,
        )


def test_summarize_first_time():
    model = FakeChatModel(
        responses=[AIMessage(content="This is a summary of the conversation.")]
    )

    # Create enough messages to trigger summarization
    messages = [
        # these messages will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_summary_tokens = 1
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.invoke_calls) == 1

    # Check that the result has the expected structure:
    # - First message should be a summary
    # - Last 3 messages should be the last 3 original messages
    assert len(result.messages) == 4
    assert result.messages[0].type == "system"
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[-3:]

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value is not None
    assert summary_value.summary == "This is a summary of the conversation."
    assert summary_value.summarized_message_ids == set(
        msg.id for msg in messages[:6]
    )  # All messages except the latest

    # Test subsequent invocation (no new summary needed)
    result = summarize_messages(
        messages,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=max_summary_tokens,
    )
    assert len(result.messages) == 4
    assert result.messages[0].type == "system"
    assert (
        result.messages[0].content
        == "Summary of the conversation so far: This is a summary of the conversation."
    )
    assert result.messages[1:] == messages[-3:]


def test_with_system_message():
    """Test summarization with a system message present."""
    model = FakeChatModel(
        responses=[AIMessage(content="Summary with system message present.")]
    )

    # Create messages with a system message
    messages = [
        # this will not be summarized, but will be added post-summarization
        SystemMessage(content="You are a helpful assistant.", id="0"),
        # these messages will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_tokens = 6
    # we're using len() as a token counter, so a summary is simply 1 "token"
    max_summary_tokens = 1
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.invoke_calls) == 1
    assert model.invoke_calls[0] == messages[1:7] + [
        HumanMessage(content="Create a summary of the conversation above:")
    ]

    # Check that the result has the expected structure:
    # - System message should be preserved
    # - Second message should be a summary of messages 2-5
    # - Last 3 messages should be the last 3 original messages
    assert len(result.messages) == 5
    assert result.messages[0].type == "system"
    assert result.messages[1].type == "system"  # Summary message
    assert "summary" in result.messages[1].content.lower()
    assert result.messages[2:] == messages[-3:]


def test_approximate_token_counter():
    model = FakeChatModel(responses=[AIMessage(content="Summary with empty messages.")])

    # Create messages with some empty content
    messages = [
        # these will be summarized
        HumanMessage(content="", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        # these will be added to the result post-summarization
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=count_tokens_approximately,
        max_tokens=60,
        max_summary_tokens=10,
    )

    # Check that summarization still works with empty messages
    assert len(result.messages) == 2
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[-1:]


def test_large_number_of_messages():
    """Test summarization with a large number of messages."""
    model = FakeChatModel(responses=[AIMessage(content="Summary of many messages.")])

    # Create a large number of messages
    messages = []
    for i in range(20):  # 20 pairs of messages = 40 messages total
        messages.append(HumanMessage(content=f"Human message {i}", id=f"h{i}"))
        messages.append(AIMessage(content=f"AI response {i}", id=f"a{i}"))

    # Add one final message
    messages.append(HumanMessage(content="Final message", id=f"h{len(messages)}"))

    # Call the summarizer
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=22,
        max_summary_tokens=0,
    )

    # Check that summarization works with many messages
    assert (
        len(result.messages) == 20
    )  # summary (for the first 22 messages) + 19 remaining original messages
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[22:]  # last 19 original messages

    # Check that the model was called with a subset of messages
    # The implementation might limit how many messages are sent to the model
    assert len(model.invoke_calls) == 1


def test_subsequent_summarization_with_new_messages():
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including new messages."),
        ]
    )

    # First batch of messages
    messages1 = [
        # these will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        # this will be propagated to the next summarization
        HumanMessage(content="Latest message 1", id="7"),
    ]

    # First summarization
    max_tokens = 6
    max_summary_tokens = 1
    result = summarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Verify the first summarization result
    assert "summary" in result.messages[0].content.lower()
    assert len(result.messages) == 2
    assert result.messages[-1] == messages1[-1]
    assert len(model.invoke_calls) == 1

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6  # first 6 messages

    # Add more messages to trigger another summarization
    new_messages = [
        # these will be summarized (including accounting for the previous summary!)
        AIMessage(content="Response to latest 1", id="8"),
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        # these will be kept in the final result
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]

    messages2 = messages1.copy()
    messages2.extend(new_messages)

    # Second summarization
    result2 = summarize_messages(
        messages2,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Get the messages sent to the model in the second call
    second_call_messages = model.invoke_calls[1]

    # Check that the previous summary is included in the prompt
    prompt_message = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_message.content
    assert "Extend this summary" in prompt_message.content

    # Check that only the new messages are sent to the model, not already summarized ones
    assert len(second_call_messages) == 5  # 4 messages + prompt
    assert [msg.content for msg in second_call_messages[:-1]] == [
        "Latest message 1",
        "Response to latest 1",
        "Message 4",
        "Response 4",
    ]

    # Verify the structure of the final result
    assert "summary" in result2.messages[0].content.lower()
    assert len(result2.messages) == 6  # Summary + last 4 messages
    assert result2.messages[-5:] == messages2[-5:]

    # Check the updated summary
    updated_summary_value = result2.running_summary
    assert updated_summary_value.summary == "Updated summary including new messages."
    # Verify all messages except the last 5 were summarized
    assert len(updated_summary_value.summarized_message_ids) == len(messages2) - 5


def test_last_ai_with_tool_calls_not_summarized():
    model = FakeChatModel(responses=[AIMessage(content="Summary without tool calls.")])

    messages = [
        # these will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(
            content="",
            id="2",
            tool_calls=[{"name": "tool_1", "id": "1", "args": {"arg1": "value1"}}],
        ),
        ToolMessage(content="Call other tool", tool_call_id="1", id="3"),
        # these will be kept in the final result
        AIMessage(
            content="",
            id="4",
            tool_calls=[{"name": "tool_2", "id": "2", "args": {"arg1": "value1"}}],
        ),
        HumanMessage(content="Message 2", id="5"),
    ]

    # Call the summarizer
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=4,
        max_summary_tokens=1,
    )

    # Check that the AI message with tool calls wasn't summarized
    assert len(result.messages) == 3
    assert result.messages[0].type == "system"  # Summary
    assert result.messages[-2:] == messages[-2:]  # The AI message with tool calls
    assert result.running_summary.summarized_message_ids == set(
        msg.id for msg in messages[:-2]
    )


def test_last_human_message_not_summarized():
    model = FakeChatModel(
        responses=[AIMessage(content="Summary without last human message.")]
    )

    # Create messages where the last one is from human
    messages = [
        AIMessage(content="Response 1", id="1"),
        HumanMessage(content="Message 2", id="2"),
        AIMessage(content="Response 2", id="3"),
        # these will be kept in the final result
        HumanMessage(content="Message 3", id="4"),
        AIMessage(content="Response 3", id="5"),
    ]

    # Call the summarizer
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=4,  # This would normally include all messages
        max_summary_tokens=1,
    )

    # Check that the last human message wasn't summarized
    assert len(result.messages) == 3
    assert result.messages[0].type == "system"  # Summary
    assert result.messages[-2:] == messages[-2:]  # The last human message
    assert result.running_summary.summarized_message_ids == set(
        msg.id for msg in messages[:-2]
    )


def test_missing_message_ids():
    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response"),  # Missing ID
    ]
    with pytest.raises(ValueError, match="Messages are required to have ID field"):
        summarize_messages(
            messages,
            running_summary=None,
            model=FakeChatModel(responses=[]),
            max_tokens=10,
            max_summary_tokens=1,
        )


def test_duplicate_message_ids():
    model = FakeChatModel(responses=[AIMessage(content="Summary")])

    # First summarization
    messages1 = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
    ]

    result = summarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=2,
        max_summary_tokens=1,
    )

    # Second summarization with a duplicate ID
    messages2 = [
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="1"),  # Duplicate ID
    ]

    with pytest.raises(ValueError, match="has already been summarized"):
        summarize_messages(
            messages1 + messages2,
            running_summary=result.running_summary,
            model=model,
            token_counter=len,
            max_tokens=5,
            max_summary_tokens=1,
        )
