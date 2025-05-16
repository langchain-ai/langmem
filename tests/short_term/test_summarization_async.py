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

from langmem.short_term.summarization import asummarize_messages, SummarizationNode

pytestmark = pytest.mark.anyio


class FakeChatModel(FakeMessagesListChatModel):
    """Mock chat model for testing the summarizer."""

    ainvoke_calls: list[list[BaseMessage]] = []

    def __init__(self, responses: list[BaseMessage]):
        """Initialize with predefined responses."""
        super().__init__(
            responses=responses or [AIMessage(content="This is a mock summary.")]
        )

    def ainvoke(self, input: List[BaseMessage]) -> AIMessage:
        """Mock ainvoke method that returns predefined responses."""
        self.ainvoke_calls.append(input)
        return super().ainvoke(input)

    def bind(self, **kwargs):
        """Mock bind method that returns self."""
        return self


async def test_async_empty_input():
    model = FakeChatModel(responses=[])

    # Test with empty message list
    result = await asummarize_messages(
        [],
        running_summary=None,
        model=model,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == []
    assert len(model.ainvoke_calls) == 0

    # Test with only system message
    system_msg = SystemMessage(content="You are a helpful assistant.", id="sys")
    result = await asummarize_messages(
        [system_msg],
        running_summary=None,
        model=model,
        max_tokens=10,
        max_summary_tokens=1,
    )

    # Check that no summarization occurred
    assert result.running_summary is None
    assert result.messages == [system_msg]


async def test_async_no_summarization_needed():
    model = FakeChatModel(responses=[])

    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
    ]

    # Tokens are under the limit, so no summarization should occur
    result = await asummarize_messages(
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
    assert len(model.ainvoke_calls) == 0  # Model should not have been called


async def test_async_summarize_first_time():
    model = FakeChatModel(
        responses=[AIMessage(content="This is a summary of the conversation.")]
    )

    # Create enough messages to trigger summarization
    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.ainvoke_calls) == 1

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
    )

    # Test subsequent invocation (no new summary needed)
    result2 = await asummarize_messages(
        messages,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=max_summary_tokens,
    )
    assert len(result2.messages) == 4
    assert result2.messages[0].type == "system"
    assert (
        result2.messages[0].content
        == "Summary of the conversation so far: This is a summary of the conversation."
    )
    assert result2.messages[1:] == messages[-3:]


async def test_async_max_tokens_before_summary():
    model = FakeChatModel(
        responses=[AIMessage(content="This is a summary of the conversation.")]
    )

    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    # Call the summarizer
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_tokens_before_summary=8,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.ainvoke_calls) == 1

    # Check structure: summary + last original message
    assert len(result.messages) == 2
    assert result.messages[0].type == "system"
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[-1:]

    # Check the summary value
    summary_value = result.running_summary
    assert summary_value is not None
    assert summary_value.summary == "This is a summary of the conversation."
    assert summary_value.summarized_message_ids == set(
        msg.id for msg in messages[:8]
    )

    # Test subsequent invocation
    result2 = await asummarize_messages(
        messages,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=6,
        max_tokens_before_summary=8,
        max_summary_tokens=max_summary_tokens,
    )
    assert len(result2.messages) == 2
    assert result2.messages[0].type == "system"
    assert (
        result2.messages[0].content
        == "Summary of the conversation so far: This is a summary of the conversation."
    )
    assert result2.messages[1:] == messages[-1:]


async def test_async_with_system_message():
    """Test summarization with a system message present."""
    model = FakeChatModel(
        responses=[AIMessage(content="Summary with system message present.")]
    )

    messages = [
        SystemMessage(content="You are a helpful assistant.", id="0"),
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    max_tokens = 6
    max_summary_tokens = 1
    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Check that model was called
    assert len(model.ainvoke_calls) == 1
    assert model.ainvoke_calls[0] == messages[1:7] + [
        HumanMessage(content="Create a summary of the conversation above:")
    ]

    # Structure: system preserved + summary + last 3
    assert len(result.messages) == 5
    assert result.messages[0].type == "system"
    assert result.messages[1].type == "system"
    assert "summary" in result.messages[1].content.lower()
    assert result.messages[2:] == messages[-3:]


async def test_async_approximate_token_counter():
    model = FakeChatModel(responses=[AIMessage(content="Summary with empty messages.")])

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

    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=count_tokens_approximately,
        max_tokens=50,
        max_summary_tokens=10,
    )

    assert len(result.messages) == 2
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[-1:]


async def test_async_large_number_of_messages():
    """Test summarization with a large number of messages asynchronously."""
    model = FakeChatModel(responses=[AIMessage(content="Summary of many messages.")])

    messages = []
    for i in range(20):
        messages.append(HumanMessage(content=f"Human message {i}", id=f"h{i}"))
        messages.append(AIMessage(content=f"AI response {i}", id=f"a{i}"))
    messages.append(HumanMessage(content="Final message", id=f"h{len(messages)}"))

    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=22,
        max_summary_tokens=0,
    )

    assert len(result.messages) == 20
    assert "summary" in result.messages[0].content.lower()
    assert result.messages[1:] == messages[22:]
    assert len(model.ainvoke_calls) == 1


async def test_async_subsequent_summarization_with_new_messages():
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

    max_tokens = 6
    max_summary_tokens = 1
    result1 = await asummarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    # Verify first summary
    assert "summary" in result1.messages[0].content.lower()
    assert len(result1.messages) == 2
    assert result1.messages[-1] == messages1[-1]
    assert len(model.ainvoke_calls) == 1

    summary_value = result1.running_summary
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6

    # Second batch
    new_messages = [
        AIMessage(content="Response to latest 1", id="8"),
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]
    messages2 = messages1 + new_messages

    result2 = await asummarize_messages(
        messages2,
        running_summary=summary_value,
        model=model,
        token_counter=len,
        max_tokens=max_tokens,
        max_summary_tokens=max_summary_tokens,
    )

    assert len(model.ainvoke_calls) == 2
    second_call = model.ainvoke_calls[1]
    prompt_msg = second_call[-1]
    assert "First summary of the conversation" in prompt_msg.content
    assert "Extend this summary" in prompt_msg.content
    assert len(second_call) == 5
    assert [m.content for m in second_call[:-1]] == [
        "Message 4",
        "Response 4",
        "Message 5",
        "Response 5",
    ]
    assert "summary" in result2.messages[0].content.lower()
    assert len(result2.messages) == 4
    assert result2.messages[-3:] == messages2[-3:]
    assert result2.running_summary.summary == "Updated summary including new messages."
    assert len(result2.running_summary.summarized_message_ids) == len(messages2) - 3


async def test_async_last_ai_with_tool_calls():
    model = FakeChatModel(responses=[AIMessage(content="Summary without tool calls.")])

    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(
            content="",
            id="2",
            tool_calls=[
                {"name": "tool_1", "id": "1", "args": {"arg1": "value1"}},
                {"name": "tool_2", "id": "2", "args": {"arg1": "value1"}},
            ],
        ),
        ToolMessage(content="Call tool 1", tool_call_id="1", name="tool_1", id="3"),
        ToolMessage(content="Call tool 2", tool_call_id="2", name="tool_2", id="4"),
        AIMessage(content="Response 1", id="5"),
        HumanMessage(content="Message 2", id="6"),
    ]

    result = await asummarize_messages(
        messages,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens_before_summary=2,
        max_tokens=6,
        max_summary_tokens=1,
    )

    # Check that the AI message with tool calls was summarized together with the tool messages
    assert len(result.messages) == 3
    assert result.messages[0].type == "system"
    assert result.messages[-2:] == messages[-2:]
    assert result.running_summary.summarized_message_ids == set(
        msg.id for msg in messages[:-2]
    )


async def test_async_missing_message_ids():
    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response"),  # Missing ID
    ]
    with pytest.raises(ValueError, match="Messages are required to have ID field"):
        await asummarize_messages(
            messages,
            running_summary=None,
            model=FakeChatModel(responses=[]),
            max_tokens=10,
            max_summary_tokens=1,
        )


async def test_async_duplicate_message_ids():
    model = FakeChatModel(responses=[AIMessage(content="Summary")])

    messages1 = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
    ]

    result = await asummarize_messages(
        messages1,
        running_summary=None,
        model=model,
        token_counter=len,
        max_tokens=2,
        max_summary_tokens=1,
    )

    messages2 = [
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="1"),  # Duplicate ID
    ]

    with pytest.raises(ValueError, match="has already been summarized"):
        await asummarize_messages(
            messages1 + messages2,
            running_summary=result.running_summary,
            model=model,
            token_counter=len,
            max_tokens=5,
            max_summary_tokens=1,
        )


async def test_async_summarization_node():
    model = FakeChatModel(
        responses=[AIMessage(content="This is a summary of the conversation.")]
    )

    messages = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Message 4", id="7"),
        AIMessage(content="Response 4", id="8"),
        HumanMessage(content="Latest message", id="9"),
    ]

    summarization_node = SummarizationNode(
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
    )
    # use the async entrypoint
    result = await summarization_node.ainvoke({"messages": messages})

    # Check that model was called
    assert len(model.ainvoke_calls) == 1

    assert len(result["summarized_messages"]) == 4
    assert result["summarized_messages"][0].type == "system"
    assert "summary" in result["summarized_messages"][0].content.lower()
    assert result["summarized_messages"][1:] == messages[-3:]

    summary_value = result["context"]["running_summary"]
    assert summary_value.summary == "This is a summary of the conversation."

    # Test subsequent invocation
    result2 = await summarization_node.ainvoke(
        {"messages": messages, "context": {"running_summary": summary_value}}
    )
    assert len(result2["summarized_messages"]) == 4
    assert result2["summarized_messages"][0].type == "system"
    assert (
        result2["summarized_messages"][0].content
        == "Summary of the conversation so far: This is a summary of the conversation."
    )
    assert result2["summarized_messages"][1:] == messages[-3:]


async def test_async_summarization_node_same_key():
    model = FakeChatModel(
        responses=[
            AIMessage(content="First summary of the conversation."),
            AIMessage(content="Updated summary including new messages."),
        ]
    )

    messages1 = [
        HumanMessage(content="Message 1", id="1"),
        AIMessage(content="Response 1", id="2"),
        HumanMessage(content="Message 2", id="3"),
        AIMessage(content="Response 2", id="4"),
        HumanMessage(content="Message 3", id="5"),
        AIMessage(content="Response 3", id="6"),
        HumanMessage(content="Latest message 1", id="7"),
    ]

    summarization_node = SummarizationNode(
        model=model,
        token_counter=len,
        max_tokens=6,
        max_summary_tokens=1,
        input_messages_key="messages",
        output_messages_key="messages",
    )
    # first async invoke
    result = await summarization_node.ainvoke({"messages": messages1})

    assert result["messages"][0].type == "remove"
    assert "summary" in result["messages"][1].content.lower()
    assert len(result["messages"]) == 3

    summary_value = result["context"]["running_summary"]
    assert summary_value.summary == "First summary of the conversation."
    assert len(summary_value.summarized_message_ids) == 6

    new_messages = [
        AIMessage(content="Response to latest 1", id="8"),
        HumanMessage(content="Message 4", id="9"),
        AIMessage(content="Response 4", id="10"),
        HumanMessage(content="Message 5", id="11"),
        AIMessage(content="Response 5", id="12"),
        HumanMessage(content="Message 6", id="13"),
        AIMessage(content="Response 6", id="14"),
        HumanMessage(content="Latest message 2", id="15"),
    ]

    messages2 = result["messages"][1:].copy()
    messages2.extend(new_messages)

    result2 = await summarization_node.ainvoke(
        {"messages": messages2, "context": {"running_summary": summary_value}}
    )

    assert len(model.ainvoke_calls) == 2
    second_call_messages = model.ainvoke_calls[1]
    prompt_msg = second_call_messages[-1]
    assert "First summary of the conversation" in prompt_msg.content
    assert "Extend this summary" in prompt_msg.content

    assert result2["messages"][0].type == "remove"
    assert "summary" in result2["messages"][1].content.lower()
    assert len(result2["messages"]) == 5
    assert result2["messages"][-3:] == messages2[-3:]

    updated_summary_value = result2["context"]["running_summary"]
    assert updated_summary_value.summary == "Updated summary including new messages."
    assert len(updated_summary_value.summarized_message_ids) == 12
