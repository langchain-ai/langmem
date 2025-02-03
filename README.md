# LangMem Memory Utilities

LLM apps work best if they can remember important preferences, skills, and knowledge. LangMem provides utilities commonly used to build memory systems that help your agents:

1. **Remember user preferences** - Store and recall user settings, preferences, and important facts
1. **Learn from interactions** - Extract and save key information from conversations
1. **Use context intelligently** - Use relevant memories when needed using semantic search

## Installation

```bash
pip install -U langmem
```

## Usage

- [Agent actively manages memories](#build-an-agent-that-actively-manages-memories)
- [Passive memory reflection](#passive-memory-reflection)
- [Optimize a prompt with the hosted service](#optimize-a-prompt-with-the-hosted-service)
- [Manage semantic memories with the hosted service](#how-to-connect-to-the-hosted-service)

## What's inside

LangMem is organized around three key patterns. The main entrypoints are provided below, but if you'd prefer to dive in headfirst, check out the [quick examples](#quick-examples) below.

### 1. Functional transformations

Pure functions that transform conversations and existing memories into new memory states. Langmem exposes factory methods to configure these functions.

- [`create_memory_enricher(model,/, schemas = None) -> ((messages, existing) -> (id, memory))`](https://langchain-ai.github.io/langmem/reference/#langmem.create_memory_enricher): Extract structured information from conversations
- [`create_prompt_optimizer(model,/,kind = "gradient", config = None) -> (([(messages, feedback), ...], prompt) -> (prompt))`](https://langchain-ai.github.io/langmem/reference/#langmem.create_prompt_optimizer): Learn from trajectories to improve system prompts
- [`create_multi_prompt_optimizer(model,/,kind = "gradient", config = None) -> ((list[(messages, feedback), ...], list[prompt]) -> (list[prompt]))`](https://langchain-ai.github.io/langmem/reference/#langmem.create_multi_prompt_optimizer): Optimize multiple prompts from a compound AI system.

### 2. Stateful Operations

Components that persist and manage memories in a BaseStore:

- [`create_memory_store_enricher(model,/, schemas = None, store = None) -> ((messages) -> None)`](https://langchain-ai.github.io/langmem/reference/#langmem.create_memory_store_enricher): Apply enrichment with LangGraph's integrated BaseStore
- [`create_manage_memory_tool(store = None) -> Tool[dict, str]`](https://langchain-ai.github.io/langmem/reference/#langmem.create_manage_memory_tool): Tool for creating/updating stored memories
- [`create_search_memory_tool(store = None) -> Tool[dict, list[Memory]]`](https://langchain-ai.github.io/langmem/reference/#langmem.create_search_memory_tool): Tool for searching stored memories

Example of stateful memory management:

```python
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore

# Set up storage
store = InMemoryStore()

# Create tools that operate on stored memories
tools = [
    create_manage_memory_tool(),  # Persist memories
    create_search_memory_tool(),  # Search stored memories
]

model = init_chat_model("model_name").bind_tools(tools)
```

### 3. Deployable Graphs

LangGraph-compatible components for production deployment:

- [`langmem.graphs.semantic`]: Extract and store memories from conversations

  ```python
  Input = {
      "messages": list[tuple[list[Message], dict]],  # Messages or (messages, feedback) pairs
      "schemas": None | list[dict] | dict,  # Optional schema definitions to extract
      "namespace": tuple[str, ...] | None,  # Optional namespace for memories
  }

  Output = {
      "updated_memories": list[dict],
  }
  ```

- `langmem.graphs.prompts`: Optimize system prompts from conversation feedback

  ```python
  Prompt = {
      "name": str,
      "prompt": str,
      "when_to_update": str,
      "update_instructions": str,
  }
  Input = {
      "prompts": list[Prompt] | str,  # Prompts to optimize
      "threads": list[
          tuple[list[Message], dict]
      ],  # List of (conversation, feedback) pairs
  }
  Output = {
      "optimized_prompts": list[Prompt],
  }


  ```

These graphs can be deployed directly on the LangGraph platform. See the [hosted service example](#example-using-the-hosted-memory-service) below for usage.

## Quick Examples

Most operations expect an API key for your favorite LLM provider. For example:

```bash
export ANTHROPIC_API_KEY="sk-..."
```

### Build an agent that actively manages memories

Here's how to create an agent that can manage its own memories "consciously", or in the hot path:

```python
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain.chat_models import init_chat_model

# Set up memory storage
store = InMemoryStore()

# Create memory management tools
tools = [
    create_manage_memory_tool(),  # For creating/updating memories
    create_search_memory_tool(),  # For searching past memories
]

model = init_chat_model("anthropic:claude-3-5-sonnet-latest").bind_tools(tools)

# Set up memory storage
store = InMemoryStore()

# Create memory management tools
tools = [
    create_manage_memory_tool(),  # For creating/updating memories
    create_search_memory_tool(),  # For searching past memories
]

model = init_chat_model("anthropic:claude-3-5-sonnet-latest").bind_tools(tools)

# These tools are stateful and let the agent "consciously"
# 1. Create/update/delete memories
# 2. Search for relevant memories
# It saves these to the configured "BaseStore" (in our case an ephemeral "InMemoryStore")
tools_by_name = {t.name: t for t in tools}

system_prompt = (
    """You are a helpful assistant. Save memories whenever you learn something new."""
)


@task
async def execute_tool(tool_call: dict):
    result = await tools[tool_call["name"]].ainvoke(tool_call["args"])
    return {
        "role": "tool",
        "content": result,
        "tool_call_id": tool_call["id"],
    }


@entrypoint(checkpointer=MemorySaver(), store=store)
async def assistant(
    messages: list[dict], *, previous: list[dict] | None = None
) -> entrypoint.final[list[dict], list[dict]]:
    messages = (previous or []) + messages
    response = None
    while True:
        if response:
            if response.tool_calls:
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_messages.append(await execute_tool(tool_call))
                messages.extend(tool_messages)
            else:
                break
        response = await model.ainvoke(
            [{"role": "system", "content": system_prompt}] + messages
        )
        messages.append(response)

    return entrypoint.final(value=response, save=messages)


async def main():
    config = {"configurable": {"thread_id": "user123"}}
    messages = [{"role": "user", "content": "I really like shoo-fly pie."}]
    async for step in assistant.astream(messages, config, stream_mode="messages"):
        print(step)


asyncio.run(main())
```

### Build an agent that learns from passive reflection

(Coming soon)

### Optimize a prompt with the hosted service

LangMem provides deployment-ready graphs that can be used via the LangGraph platform. We have a managed graph deployed at [`https://langmem-v0-544fccf4898a5e3c87bdca29b5f9ab21.us.langgraph.app`](https://langmem-v0-544fccf4898a5e3c87bdca29b5f9ab21.us.langgraph.app/docs). You can also test this out locally from the langgraph CLI using `langgraph dev`.

The first service it exposes is one to help your agent learn instructions and core memories to store in your prompts based on conversation history and feedback:

```python
import os
from langgraph_sdk import get_client

# Required: LangSmith API key (US region)
os.environ["LANGSMITH_API_KEY"] = "<your key>"

# Connect to the hosted service
url = "https://langmem-v0-544fccf4898a5e3c87bdca29b5f9ab21.us.langgraph.app"
client = get_client(url=url)

# Use the prompt optimization graph
results = await client.runs.wait(
    None,
    "optimize_prompts",
    input={
        "threads": [[conversation, feedback]],
        "prompts": [
            {
                "name": "assistant_prompt",
                "prompt": "You are a helpful assistant.",
                "when_to_update": "When user requests more detail",
                "update_instructions": "Add instructions about providing context",
            }
        ],
    },
    config={"configurable": {"model": "claude-3-5-sonnet-latest"}},
)
```


#### Store semantic memories with the hosted service

The hosted service also provides a graph that can be used to extract and store semantic knowledge from conversations:

```python
# Example conversation
conversation = [
    {"role": "user", "content": "I prefer dark mode and minimalist interfaces"},
    {"role": "assistant", "content": "I'll remember your UI preferences."},
]

# Extract memories with optional schema
results = await client.runs.wait(
    None,
    "extract_memories",
    input={
        "messages": conversation,
        "schemas": [
            {  # Optional: define memory structure
                "title": "UserPreference",
                "type": "object",
                "properties": {
                    "preference": {"type": "string"},
                    "category": {"type": "string"},
                },
                "description": "User preferences",
            }
        ],
    },
    config={"configurable": {"model": "claude-3-5-sonnet-latest"}},
)

# Search memories
memories = await client.store.search_items((), query="UI preferences")
```


## Conceptual guide

Below is a high-level conceptual overview of a way to think about memory management.

### 1. Formation Pattern

How memories are created:

| Pattern                   | Description                                        | Best For                                                                           | Tools                                                          |
| ------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Conscious (Agent Tools)   | Agent actively decides to save during conversation | - Direct user feedback<br>- Explicit rules and preferences<br>- Teaching the agent | - `create_manage_memory_tool`<br>- `create_search_memory_tool` |
| Subconscious (Background) | Separate LLM analyzes conversations/trajectories   | - Pattern discovery<br>- Learning from experience<br>- Complex relationships       | - `create_memory_enricher`<br>- `create_memory_store_enricher` |

Example of conscious memory formation:

```python
from langmem import create_manage_memory_tool, create_search_memory_tool

tools = {
    "manage_memory": create_manage_memory_tool(),
    "search_memory": create_search_memory_tool(),
}
model = init_chat_model("model_name").bind_tools(tools.values())
```

Example of subconscious memory formation:

```python
from langmem import create_memory_store_enricher

enricher = create_memory_store_enricher(
    "model_name", schemas=[UserProfile], enable_inserts=True
)
memories = await enricher.manage_memories(conversation)
```

### 2. Storage Pattern

How memories are structured:

| Pattern             | Description                         | Best For                                                                 | Implementation                                    |
| ------------------- | ----------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- |
| Constrained Profile | Single schema, continuously updated | - User preferences<br>- System settings<br>- Current state               | Use `create_memory_enricher` with defined schemas |
| Event Stream        | Expansive list of discrete memories | - Conversation history<br>- Learning experiences<br>- Evolving knowledge | Use `create_manage_memory_tool` with kind="multi" |

Example of constrained profile:

```python
from pydantic import BaseModel
from langmem import create_memory_enricher


class UserProfile(BaseModel):
    preferences: dict[str, str]
    settings: dict[str, Any]


enricher = create_memory_enricher("model_name", schemas=[UserProfile], kind="single")
```

Example saving semantic facts

```python
from langmem import create_manage_memory_tool

memory_tool = create_manage_memory_tool(
    kind="multi", namespace_prefix=("user", "experiences")
)
```

### 3. Retrieval Pattern

How memories are accessed:

| Pattern                   | Description                                    | When to Use                                                              | Implementation                                              |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| Always-On (System Prompt) | Critical context included in every interaction | - Core rules<br>- User preferences<br>- Session state                    | Use `create_prompt_optimizer` with memory integration       |
| Associative (Search)      | Contextually searched when needed              | - Historical conversations<br>- Specific knowledge<br>- Past experiences | Use `create_search_memory_tool` or `create_memory_searcher` |