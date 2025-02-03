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

- \[`langmem.graphs.semantic`\]: Extract and store memories from conversations

  ```python
  Input = {
      "messages": list[
          tuple[list[Message], dict]
      ],  # Messages or (messages, feedback) pairs
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

LangMem provides utilities for managing different types of long-term memory in AI systems. This guide explores the key concepts and patterns for implementing memory effectively.

## Memory Types

Like human memory systems, AI agents can utilize different types of memory for different purposes:

### Semantic Memory

Semantic memory stores facts and knowledge that can be used to ground agent responses. In LangMem, semantic memories can be managed in two ways:

1. **Profile Pattern**

   - A single, continuously updated JSON document containing well-scoped information
   - Best for: User preferences, system settings, and current state

   ```python
   from pydantic import BaseModel


   class UserPreferences(BaseModel):
       preferences: dict[str, str]
       settings: dict[str, str]


   enricher = create_memory_enricher(
       "claude-3-5-sonnet-latest",
       schemas=[UserPreferences],
       instructions="Extract user preferences and settings",
       enable_inserts=True,
       enable_deletes=False,
   )
   ```

1. **Collection Pattern**

   - A set of discrete memory documents that grow over time

   ```python
   memory_tool = create_manage_memory_tool(
       instructions="Save important user preferences and context",
       namespace_prefix=("user", "experiences"),
       kind="multi",
   )
   ```

### Episodic Memory

Episodic memory helps agents recall past events and experiences:

```python
from langmem.graphs.semantic import MemoryGraph

memory_graph = MemoryGraph(
    store=store, extractor="claude-3-5-sonnet-latest", indexing={"embed": embed}
)

# Store conversation history
await memory_graph.arun(
    {
        "messages": [
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "I'll remember that preference"},
        ],
        "namespace": ("user123", "conversations"),
        "schemas": None,  # Optional schema for structured extraction
    }
)
```

### Procedural Memory

Procedural memory helps agents remember how to perform tasks through system prompts and instructions:

```python
from langmem.graphs.prompts import PromptGraph

prompt = {
    "name": "main",
    "prompt": "You are a helpful assistant. Current rules:\n{instructions}",
    "when_to_update": "When user feedback indicates unclear instructions",
    "update_instructions": "Improve clarity while maintaining core functionality",
}

prompt_graph = PromptGraph()
optimized = await prompt_graph.arun(
    {
        "prompts": [prompt],
        "threads": [(conversation, "Make instructions more specific")],
    }
)
```

## Writing Memories

LangMem supports two primary approaches to memory formation:

### Conscious memory formation

Agents can actively manage memories "in the hot path" by calling tools to save, update, and delete memories.

```python
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool

memory_tool = create_manage_memory_tool(
    instructions="Save important user preferences and context",
    namespace_prefix=("user", "preferences"),
    kind="multi",
)
agent = create_react_agent("anthropic:claude-3-5-sonnet-latest", tool=[memory_tool])

agent.invoke("Did you know I hold the world record for most stubbed toes in one day?")
```

### Background memory formation

Memory formation happens asynchronously through reflection:

```python
enricher = create_memory_store_enricher(
    "claude-3-5-sonnet-latest",
    schemas=[UserProfile],
    enable_background=True,
    query_model="claude-3-5-haiku-latest",  # Faster model for searches
    query_limit=5,
)

async for conversation in message_stream:
    await enricher.amanage_memories(conversation)
```

## Long-term storage

LangMem's lowest-level primitives are purely **functional** - they take trajectories and current memory state (prompts or similar memories) as input and return updated memory state. These primitives form the foundation for higher-level utilities that integrate with LangGraph for persistent storage.

For storage, LangMem uses LangGraph's `BaseStore` interface, which provides a hierarchical document store with semantic search capabilities. Memories are organized using:

1. **Namespaces**: Logical groupings similar to directories (e.g., `(user_id, app_context)`)
1. **Keys**: Unique identifiers within namespaces (like filenames)
1. **Storage**: JSON documents with metadata and vector embeddings

```python
def embed(texts: list[str]) -> list[list[float]]:
    # Replace with actual embedding function
    return [[1.0, 2.0] * len(texts)]


store = InMemoryStore(index={"embed": embed, "dims": 2})
namespace = ("user123", "preferences")
store.put(
    namespace,
    "ui_settings",
    {"rules": ["User prefers dark mode", "User likes minimalist interfaces"]},
)
```

While you can work with `BaseStore` directly, LangMem provides higher-level primitives (memory tools, stateful utilities, graphs) that manage memories on behalf of your agent, handling the storage operations automatically.

## Implementation Patterns

<<<<<<< Updated upstream
| Pattern                   | Description                                    | When to Use                                                              | Implementation                                              |
| ------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| Always-On (System Prompt) | Critical context included in every interaction | - Core rules<br>- User preferences<br>- Session state                    | Use `create_prompt_optimizer` with memory integration       |
| Associative (Search)      | Contextually searched when needed              | - Historical conversations<br>- Specific knowledge<br>- Past experiences | Use `create_search_memory_tool` or `create_memory_searcher` |
=======
### Profile-Based Pattern

```python
# 1. Define schema
class UserPreferences(BaseModel):
    dietary_restrictions: list[str]
    communication_style: str
    ui_preferences: dict[str, str]


# 2. Configure enricher with storage
store = InMemoryStore(index={"embed": embed, "dims": 384})
enricher = create_memory_store_enricher(
    "claude-3-5-sonnet-latest",
    schemas=[UserPreferences],
    store=store,
    instructions="Extract and maintain user preferences",
    namespace_prefix=("user", "profile"),
)

# 3. Process conversations
await enricher.amanage_memories(
    [{"role": "user", "content": "I'm vegetarian and prefer formal communication"}]
)
```

### Collection-Based Pattern

```python
# 1. Set up storage and tools
store = InMemoryStore(index={"embed": embed, "dims": 384})
memory_tool = create_manage_memory_tool(
    store=store, kind="multi", namespace_prefix=("user", "experiences")
)

# 2. Save discrete memories
await memory_tool.ainvoke(
    {
        "content": "User prefers dark mode",
        "context": "UI preferences discussion",
        "timestamp": "2025-02-03",
    }
)

# 3. Search memories
search_tool = create_search_memory_tool(
    store=store, namespace_prefix=("user", "experiences")
)
results = await search_tool.ainvoke({"query": "What are the user's UI preferences?"})
```

This architecture allows you to make deliberate choices about:

1. **What** to remember (memory types)
1. **When** to remember (update timing)
1. **How** to remember (storage patterns)
1. **When** to retrieve (access patterns)
>>>>>>> Stashed changes
