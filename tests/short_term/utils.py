from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel


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
    
    async def ainvoke(self, input: List[BaseMessage]) -> AIMessage:
        """Mock invoke method that returns predefined responses."""
        self.invoke_calls.append(input)
        return await super().ainvoke(input)

    def bind(self, **kwargs):
        """Mock bind method that returns self."""
        return self