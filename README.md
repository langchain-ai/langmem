# LangMem

LangMem exports utilities for extracting, updating, generalizing, storing, and retrieving information from interactions in LLM applications. It offers:

1. Pure functions to extract and enrich memories from trajectories & interactions
2. Configurable memory managers and agent tools that integrate with LangGraph's storage layer
3. Deployable entrypoints that let you build and deploy memory systems in LangGraph Platform

This lets you build your own memory layer faster, using the level of abstraction and durability that suits your needs. Compared to raw LLM extraction, LangMem handles memory management configuration, letting you define how to form new memories and evolve or prune old ones through validated types. Compared to high-level frameworks, it exposes the lower-level primitives used to build exactly the memory system you want without being forced into adopting a specific database or storage layer.

## Installation

```bash
pip install -U langmem
```

Configure your environment with an API key for your favorite LLM provider:

```bash
export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
```

Here's how to create an agent with memory in just a few lines:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

store = InMemoryStore()

# Create an agent with memory tools
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[
        create_manage_memory_tool(namespace=("memories",)),  # Store memories
        create_search_memory_tool(namespace=("memories",)),  # Search memories
    ],
    store=store,
)
```

Then use the agent:
```python
agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Remember that I prefer dark mode."}
        ]
    }
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my preferences?"}]}
)
print(response["messages"][-1].content)
# Output: "You've told me that you prefer dark mode."
```

The agent can now store important information from conversations, search its memory when relevant, and persist knowledge across conversations.

For more control over memory management, check out:

- [Memory Tools](guides/memory_tools.md) - Configure how memories are handled
- [Functional Primitives](concepts/conceptual_guide.md#functional-core) - Build custom memory systems
- [Storage Options](guides/memory_tools.md#storage) - Add persistent storage

## Next Steps

For more examples and detailed documentation:

- [Quickstart Guide](quickstart.md) - Get up and running
- [Core Concepts](concepts/conceptual_guide.md#memory-in-llm-applications) - Learn key ideas
- [API Reference](reference/index.md) - Full function documentation
- [Integration Guides](guides/memory_tools.md) - Common patterns and best practices

## Requirements

- Python 3.11+
- Access to a supported LLM provider (Anthropic, OpenAI, etc.)
- Optional: [LangGraph](https://github.com/langchain-ai/langgraph) [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) implementation for persistent storage (for the stateful primitives); if you're deploying on LangGraph Platform, this is included without any additional configuration.
