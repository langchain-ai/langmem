# LangMem

LangMem helps agents learn and adapt from their interactions over time.

It provides tooling to extract important information from conversations, optimize agent behavior through prompt refinement, and maintain long-term memory.

It offers both functional primitives you can use with any storage system and native integration with LangGraph's storage layer.

This lets your agents continuously improve, personalize their responses, and maintain consistent behavior across sessions.

## Key features

- 🧩 **Core memory API** that works with any storage system
- 🧠 **Memory management tools** that agents can use to record and search information during active conversations "in the hot path"
- ⚙️ **Background memory manager** that automatically extracts, consolidates, and updates agent knowledge
- ⚡ **Native integration with LangGraph's Long-term Memory Store**, available by default in all LangGraph Platform deployments

## Installation

```bash
pip install -U langmem
```

Configure your environment with an API key for your favorite LLM provider:

```bash
export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
```

## Creating an Agent

Here's how to create an agent that actively manages its own long-term memory in just a few lines:

```python
# Import core components (1)
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up storage (2)
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
) 

# Create an agent with memory capabilities (3)
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[
        # Memory tools use LangGraph's BaseStore for persistence (4)
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)
```

1. The memory tools work in any LangGraph app. Here we use [`create_react_agent`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.create_react_agent) to run an LLM with tools, but you can add these tools to your existing agents or build [custom memory systems](concepts/conceptual_guide.md#functional-core) without agents.

2. [`InMemoryStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.memory.InMemoryStore) keeps memories in process memory—they'll be lost on restart. For production, use the [AsyncPostgresStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore) or a similar DB-backed store to persist memories across server restarts.

3. The memory tools ([`create_manage_memory_tool`](reference/tools.md#langmem.create_manage_memory_tool) and [`create_search_memory_tool`](reference/tools.md#langmem.create_search_memory_tool)) let you control what gets stored. The agent extracts key information from conversations, maintains memory consistency, and knows when to search past interactions. See [Memory Tools](guides/memory_tools.md) for configuration options.

Then use the agent:

```python
# Store a new memory (1)
agent.invoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer dark mode."}]}
)

# Retrieve the stored memory (2)
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my lighting preferences?"}]}
)
print(response["messages"][-1].content)
# Output: "You've told me that you prefer dark mode."
```

1. The agent gets to decide what and when to store the memory. No special commands needed—just chat normally and the agent uses [`create_manage_memory_tool`](reference/tools.md#langmem.create_manage_memory_tool) to store relevant details.

2. The agent maintains context between chats. When you ask about previous interactions, the LLM can invoke [`create_search_memory_tool`](reference/tools.md#langmem.create_search_memory_tool) to search for memories with similar content. See [Memory Tools](guides/memory_tools.md) to customize memory storage and retrieval, and see the [hot path quickstart](https://langchain-ai.github.io/langmem/hot_path_quickstart) for a more complete example on how to include memories without the agent having to expliictly search.

The agent can now store important information from conversations, search its memory when relevant, and persist knowledge across conversations.

## Next Steps

For more examples and detailed documentation:

- [Hot Path Quickstart](https://langchain-ai.github.io/langmem/hot_path_quickstart) - Learn how to let your LangGraph agent manage its own memory "in the hot path"
- [Background Quickstart](https://langchain-ai.github.io/langmem/background_quickstart) - Learn how to use a memory manager "in the background"
- [Core Concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide) - Learn key ideas
- [API Reference](https://langchain-ai.github.io/langmem/reference) - Full function documentation
- Build RSI 🙂 