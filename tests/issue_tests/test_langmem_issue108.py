"""
Tests for langmem issue #108 reproduction.

This test suite reproduces the issue where langmem configuration 
behaves differently when using config['configurable'] vs context parameter.
"""

import pytest
from dataclasses import dataclass
from dotenv import load_dotenv

from langmem import create_manage_memory_tool
from langmem.errors import ConfigurationError
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent

# Load environment variables, specifically OPENAI_API_KEY
load_dotenv()


@dataclass
class ContextSchema:
    user_id: str


@pytest.fixture
def memory_store():
    """Create an in-memory store for testing."""
    return InMemoryStore(
        index={"embed": "openai:text-embedding-3-small"}
    )


@pytest.fixture
def manage_memory_tool(memory_store):
    """Create memory management tool."""
    return create_manage_memory_tool(
        namespace=("{user_id}",),
        store=memory_store
    )


@pytest.fixture
def react_agent(manage_memory_tool, memory_store):
    """Create ReAct agent with memory tool."""
    return create_react_agent(
        "openai:gpt-4o-mini",
        tools=[manage_memory_tool],
        prompt="You are a helpful assistant.",
        context_schema=ContextSchema,
        store=memory_store
    )


@pytest.fixture
def test_message():
    """Test message for agent invocation."""
    return {
        "messages": [
            {"role": "user", "content": "I live in San Francisco, California."}
        ]
    }


def test_agent_invoke_with_configurable_config(react_agent, test_message):
    """Test agent invocation using config['configurable'] parameter - should succeed."""
    try:
        result = react_agent.invoke(
            test_message,
            config={
                "configurable": {
                    "user_id": "123"
                }
            }
        )
        assert result is not None, "Expected successful result from agent invocation"
        assert "messages" in result, "Expected 'messages' key in result"
    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError should not be raised with config['configurable']: {e}")


def test_agent_invoke_with_context_parameter(react_agent, test_message):
    """Test agent invocation using context parameter - should succeed."""
    try:
        result = react_agent.invoke(
            test_message,
            context={
                "user_id": "123"
            }
        )
        assert result is not None, "Expected successful result from agent invocation"
        assert "messages" in result, "Expected 'messages' key in result"
    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError should not be raised with context parameter: {e}")
