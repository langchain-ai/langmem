---
title: How to Use Memory Tools in Custom Agents
---

# How to Use Memory Tools in Custom Agents

LangMem's memory tools let your custom agents store and search memories, enabling persistent knowledge across conversations.

## Installation

Install LangMem:

```bash
pip install -U langmem
```

## Example using Anthropic API

Here's how to add memory tools to your custom agent implementation:

```python
import anthropic
from typing import List, Dict, Any
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore


def execute_tool(tools_by_name: Dict[str, Any], tool_call: Dict[str, Any]) -> str:
    """Execute a tool call and return the result"""
    tool_name = tool_call.name

    if tool_name not in tools_by_name:
        return f"Error: Tool {tool_name} not found"

    tool = tools_by_name[tool_name]
    try:
        result = tool.invoke(tool_call.input)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_agent(tools: List[Any], user_input: str, max_steps: int = 5) -> str:
    """Run a simple agent loop that can use tools"""
    # Setup
    client = anthropic.Anthropic()
    tools_by_name = {tool.name: tool for tool in tools}

    # Convert tools to Anthropic's format
    anthropic_tools = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.tool_call_schema.model_json_schema(),
        }
        for tool in tools
    ]

    messages = [{"role": "user", "content": user_input}]

    # REACT loop
    for step in range(max_steps):
        # Get next action from Claude
        tools = anthropic_tools if step < max_steps - 1 else []
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            temperature=0.7,
            tools=tools,
            messages=messages,
        )
        tool_calls = [
            content for content in response.content if content.type == "tool_use"
        ]
        if not tool_calls:
            # No more tools to call, return the final response
            return "".join(
                [block.text for block in response.content if block.type == "text"]
            )
        messages.append({"role": "assistant", "content": response.content})
        for tool_call in tool_calls:
            tool_result = execute_tool(tools_by_name, tool_call)

            # Add the tool call and result to the conversation
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": tool_result,
                        }
                    ],
                },
            )

    return "Reached maximum number of steps"


# Set up memory store and tools
store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )  # (1)!
memory_tools = [
    create_manage_memory_tool(namespace="memories", store=store),
    create_search_memory_tool(namespace="memories", store=store),
]

# Run the agent
result = run_agent(
    tools=memory_tools,
    user_input="Remember that I like cherry pie. Then remember that I dislike rocky road.",
)
print(result)
# I've created both memories. I'll remember that you like cherry pie and dislike rocky road ice cream...
print(store.search(("memories",)))
# [
#     Item(
#         namespace=["memories"],
#         key="79d6d323-c6ec-408a-ae75-bda1fcbebd6f",
#         value={"content": "User likes cherry pie"},
#         created_at="2025-02-07T23:26:00.975678+00:00",
#         updated_at="2025-02-07T23:26:00.975682+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="72705ea8-babf-4ddd-bf0f-7426dd0e4f35",
#         value={"content": "User dislikes rocky road"},
#         created_at="2025-02-07T23:26:02.995210+00:00",
#         updated_at="2025-02-07T23:26:02.995215+00:00",
#         score=None,
#     ),
# ]
```

1. For production use, replace `InMemoryStore` with a persistent store like [`PostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore).

## Example using OpenAI API

Here's the same agent implementation using OpenAI:

```python
from typing import Any, Dict, List

from langgraph.store.memory import InMemoryStore
from openai import OpenAI

from langmem import create_manage_memory_tool, create_search_memory_tool


def execute_tool(tools_by_name: Dict[str, Any], tool_call: Dict[str, Any]) -> str:
    """Execute a tool call and return the result"""
    tool_name = tool_call["function"]["name"]

    if tool_name not in tools_by_name:
        return f"Error: Tool {tool_name} not found"

    tool = tools_by_name[tool_name]
    try:
        result = tool.invoke(tool_call["function"]["arguments"])
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_agent(tools: List[Any], user_input: str, max_steps: int = 5) -> str:
    """Run a simple agent loop that can use tools"""
    # Setup
    client = OpenAI()
    tools_by_name = {tool.name: tool for tool in tools}

    # Convert tools to OpenAI's format
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.tool_call_schema.model_json_schema(),
            },
        }
        for tool in tools
    ]

    messages = [{"role": "user", "content": user_input}]

    # REACT loop
    for step in range(max_steps):
        # Get next action
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=openai_tools if step < max_steps - 1 else [],
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            # No more tools to call, return the final response
            return message.content

        messages.append(
            {"role": "assistant", "content": message.content, "tool_calls": tool_calls}
        )

        for tool_call in tool_calls:
            tool_result = execute_tool(tools_by_name, tool_call.model_dump())
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    return "Reached maximum number of steps"


# Set up memory store and tools
store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )
memory_tools = [
    create_manage_memory_tool(namespace="memories", store=store),
    create_search_memory_tool(namespace="memories", store=store),
]

# Run the agent
result = run_agent(
    tools=memory_tools,
    user_input="Remember that I like cherry pie. Then remember that I dislike rocky road.",
)
print(result)
# I've remembered that you like cherry pie and that you dislike rocky road...
print(store.search(("memories",)))

# [
#     Item(
#         namespace=["memories"],
#         key="6d3d82d9-724c-47af-aa2f-1d1e917f8bc2",
#         value={"content": '{"action":"create","content":"likes cherry pie"}'},
#         created_at="2025-02-07T23:49:46.056925+00:00",
#         updated_at="2025-02-07T23:49:46.056928+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="cf40797f-b00a-41eb-ab96-e6aeadb468e3",
#         value={"content": '{"action":"create","content":"dislikes rocky road"}'},
#         created_at="2025-02-07T23:49:47.353574+00:00",
#         updated_at="2025-02-07T23:49:47.353579+00:00",
#         score=None,
#     ),
# ]

```

## How It Works

The examples above show how to integrate LangMem's memory tools into a custom agent that:

1. Uses Anthropic or OpenAI to call an LLM
2. Implements a basic REACT loop for tool usage
3. Handles tool execution and message history management
4. Provides memory persistence through LangGraph's store system

The memory tools provide two key capabilities:

- [`create_manage_memory_tool`](../reference/tools.md#langmem.create-manage-memory-tool): Lets the agent create, update, and delete memories
- [`create_search_memory_tool`](../reference/tools.md#langmem.create-search-memory-tool): Lets the agent search through previously stored memories

The agent can use these tools to maintain context and remember important information across conversations.
