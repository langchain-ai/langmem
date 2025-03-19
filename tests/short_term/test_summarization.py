from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from langmem.short_term.summarization import summarize_messages


class FakeChatModel(FakeMessagesListChatModel):
    """Mock chat model for testing the summarizer."""
    invoke_calls: list[list[BaseMessage]] = []

    def __init__(self, responses: list[BaseMessage]):
        """Initialize with predefined responses."""
        super().__init__(responses=responses or [AIMessage(content="This is a mock summary.")])

    def invoke(self, input: List[BaseMessage]) -> AIMessage:
        """Mock invoke method that returns predefined responses."""
        self.invoke_calls.append(input)
        return super().invoke(input)

    def bind(self, **kwargs):
        """Mock bind method that returns self."""
        return self


def test_summarize_first_time():
    """Test summarization when it happens for the first time."""
    model = FakeChatModel(responses=[AIMessage(content="This is a summary of the conversation.")])

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
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=0,
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

    # Check that summary was stored in the store
    summary_value = result.running_summary
    assert summary_value is not None
    assert summary_value.summary == "This is a summary of the conversation."
    assert (
        summary_value.summarized_message_ids == set(msg.id for msg in messages[:6])
    )  # All messages except the latest

    # Test subsequent invocation
    result = summarize_messages(
        messages,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=0,
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
    model = FakeChatModel(responses=[AIMessage(content="Summary with system message present.")])

    # Create messages with a system message
    messages = [
        # this is counted towards the max_tokens
        SystemMessage(content="You are a helpful assistant.", id="0"),
        # these messages will be summarized
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        # these messages will be added to the result post-summarization
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=0,
    )

    # Check that model was called
    assert len(model.invoke_calls) == 1
    assert model.invoke_calls[0] == messages[1:5] + [
        HumanMessage(content="Create a summary of the conversation above:")
    ]

    # Check that the result has the expected structure:
    # - System message should be preserved
    # - Second message should be a summary of messages 2-5
    # - Last 5 messages should be the last 5 original messages
    assert len(result.messages) == 7
    assert result.messages[0].type == "system"
    assert result.messages[1].type == "system"  # Summary message
    assert "summary" in result.messages[1].content.lower()
    assert result.messages[2:] == messages[-5:]


def test_subsequent_summarization():
    """Test that subsequent summarizations build on previous summaries."""
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including new messages."),
        ]
    )

    # First batch of messages
    messages1 = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Latest message 1", id="7"),
    ]

    # First summarization
    result = summarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=0,
    )

    # Add more messages to trigger another summarization
    # We need to add at least max_messages (4) new messages
    messages2 = messages1 + [
        AIMessage(content="Response to latest 1", id="8"),
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]

    summary_value = result.running_summary
    assert summary_value.summary == "First summary of the conversation."

    # Second summarization
    result2 = summarize_messages(
        messages2,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=0,
    )

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Check that the second call includes the previous summary
    second_call_messages = model.invoke_calls[1]
    assert any(
        "First summary" in str(msg.content)
        if hasattr(msg, "content")
        else "First summary" in str(msg)
        for msg in second_call_messages
    )

    # Check that the final result has the updated summary
    assert "summary" in result2.messages[0].content.lower()
    assert result2.messages[-3:] == messages2[-3:]

    # Check that the updated summary was stored
    summary_value = result2.running_summary
    assert summary_value.summary == "Updated summary including new messages."


def test_with_empty_messages():
    """Test summarization with empty message content."""
    model = FakeChatModel(responses=[AIMessage(content="Summary with empty messages.")])

    def count_non_empty_messages(messages: list[BaseMessage]) -> int:
        return sum(1 for msg in messages if msg.content)

    # Create messages with some empty content
    messages = [
        HumanMessage(content="", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    result = summarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=count_non_empty_messages,
        max_tokens=6,
        max_summary_tokens=0,
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


def test_only_summarize_new_messages():
    """Test that only new messages are summarized, not all messages."""
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including only new messages."),
        ]
    )


    # First batch of messages
    messages1 = [
        # first 18 tokens will be summarized
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
    result = summarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=0,
    )

    # Verify the first summarization happened
    assert len(model.invoke_calls) == 1

    # Check that the summary was stored correctly
    summary_value = result.running_summary
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6  # first 6 messages

    # Add more messages to trigger another summarization
    # We need to add at least max_messages (4) new messages
    # The result1 already contains a summary message + last 3 messages from messages1
    messages2 = messages1.copy()

    # Add enough new messages to trigger summarization (at least max_messages)
    new_messages = [
        # these will be summarized
        AIMessage(content="Response to latest 1", id="8"),
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        # these will be propagated to the next summarization
        HumanMessage(content="Latest message 2", id="13"),
    ]

    # Add all but the last message to messages_to_summarize
    messages2.extend(new_messages)

    # Second summarization
    result2 = summarize_messages(
        messages2,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=0,
    )

    # Check that model was called twice
    assert len(model.invoke_calls) == 2

    # Get the messages sent to the model in the second call
    second_call_messages = model.invoke_calls[1]

    # The last message in second_call_messages should be the prompt with the previous summary
    prompt_message = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_message.content
    assert "Extend this summary" in prompt_message.content

    # Check that the messages sent to the model are only the new ones
    assert [msg.content for msg in second_call_messages[:-1]] == [
        "Latest message 1",
        "Response to latest 1",
        "Message 4",
        "Response 4",
        "Message 5",
        "Response 5",
    ]

    # Check that the updated summary was stored
    updated_summary_value = result2.running_summary
    assert (
        updated_summary_value.summary
        == "Updated summary including only new messages."
    )
