---
title: Hot Path Quickstart Guide
description: Get started with LangMem
---

# Hot Path Quickstart Guide

Memories can be created in two ways:

1. 👉 **In the hot path (this guide):** the agent consciously saves notes using tools.
2. In the background: memories are "subconsciously" extracted automatically from conversations (see [Background Quickstart](background_quickstart.md)).

![Hot Path Quickstart Diagram](concepts/img/hot_path_vs_background.png)

In this guide, we will create a LangGraph agent that actively manages its own long-term memory through LangMem's `manage_memory` tool.

## Prerequisites

First, install LangMem:

```bash
pip install -U langmem
```

Configure your environment with an API key for your favorite LLM provider:

```bash
export ANTHROPIC_API_KEY="sk-..."  # Or another supported LLM provider
```

## Agent

Here's a complete example showing how to create an agent with memory that persists across conversations:

```python hl_lines="16-20 42-46"
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store 
from langmem import (
    # Lets agent create, update, and delete memories (1)
    create_manage_memory_tool,
)


def prompt(state):
    """Prepare the messages for the LLM."""
    # Get store from configured contextvar; (5)
    store = get_store() # Same as that provided to `create_react_agent`
    memories = store.search(
        # Search within the same namespace as the one
        # we've configured for the agent
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


store = InMemoryStore(
    index={ # Store extracted memories (4)
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
) 
checkpointer = MemorySaver() # Checkpoint graph state (2)

agent = create_react_agent( 
    "anthropic:claude-3-5-sonnet-latest",
    prompt=prompt,
    tools=[ # Add memory tools (3)
        # The agent can call "manage_memory" to
        # create, update, and delete memories by ID
        # Namespaces add scope to memories. To
        # scope memories per-user, do ("memories", "{user_id}"): (6)
        create_manage_memory_tool(namespace=("memories",)),
    ],
    # Our memories will be stored in this provided BaseStore instance
    store=store,
    # And the graph "state" will be checkpointed after each node
    # completes executing for tracking the chat history and durable execution
    checkpointer=checkpointer, 
)
```

1. The tools [`create_manage_memory_tool`](reference/tools.md#langmem.create_manage_memory_tool) and [`create_search_memory_tool`](reference/tools.md#langmem.create_search_memory_tool) allow agents to manually store and retrieve information from their memory. The `namespace` parameter scopes the memories, ensuring that data is kept separate based on however you configure it.

    Here, we save all memories to the ("memories",) namespace, meaning no matter which user interacts with the agent, all memories would be shared in the same directory. We could also configure it to organize memories in other ways. For instance::

    | Organization Pattern | Namespace Example | Use Case |
    |---------------------|------------------|-----------|
    | By user | `("memories", "{user_id}")` | Separate memories per user |
    | By assistant | `("memories", "{assistant_id}")` | An assistant may have memories that span multiple users |
    | By user & organization | `("memories", "{organization_id}", "{user_id}")` | Let you search across an organization while scoping memories per user |
    | Further subdivisions | `("memories", "{user_id}", "manual_memories")` | Organize different types of user data |

    Each entry in a namespace is like a directory on a computer. If you provide a bracketed namespace variable (like "{user_id}"), LangMem will replace it with the value from the `configurable` field in the `RunnableConfig` at runtime.

2. The [`MemorySaver`](https://langchain-ai.github.io/langgraph/reference/checkpoints/) checkpointer maintains conversation history within each "thread". 

    You can think of threads like conversations, akin to an email thread. This "short-term" memory tracks the state of the agent/graph , ensuring that conversations remain independent. For production deployments, use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore). `InMemoryStore` works fine for development but doesn't persist data between restarts.

3. These tools (and any of the other stateful components) will also work in any node in [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph), [`@entrypoint`](https://langchain-ai.github.io/langgraph/reference/func/#langgraph.func.entrypoint), and any other `langgraph` graph. We're using `create_react_agent` here because it's easy to use and concise to write. Check out its [api ref](https://langchain-ai.github.io/langgraph/reference/prebuilt/?h=create+react#langgraph.prebuilt.chat_agent_executor.create_react_agent) for more information on what the agent is.

4. The [`InMemoryStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.PostgresStore.asearch) provides ephemeral storage suitable for development. In production, replace this with a DB-backed [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/stores/#basestore) implementation for persistence. When deploying on the LangGraph platform, a postgres-backed store is automatically provided. This store enables saving and retrieving information from any namespace, letting you scope memories by user, agent, organization, or other arbitrary categories.

    Note that the `Store` is different from the checkpointer / "MemorySaver". The store lets you store any information according to preferred hierarchy. The checkpointer tracks state (including the conversation history) within each "thread" for durable execution.

    They can address overlapping concerns, but the store is more flexible and well-suited for long-term, cross-thread memory.

5. `get_store()` gets whichever store you've compiled into the graph. This is easier than having to pass it through each function explicitly.

    LangGraph manages some objects (such as the `config`, `store`, etc.) using [contextvars](https://docs.python.org/3/library/contextvars.html); this lets you fetch the store or other configured information from the context without having to add them to all your function signatures. This is especially convenient when fetching contextual information with `tools` or within the `prompt` function here.

6. To see how to dynamically configure namespaces, see [how to dynamically configure namespaces](guides/dynamically_configure_namespaces.md).

## Using the agent

You can interact with the graph by `invoke`'ing it.
If the agent decides to save a memory, it will call the `manage_memory` tool.

```python
config = {"configurable": {"thread_id": "thread-a"}}

# Use the agent. The agent hasn't saved any memories,
# so it doesn't know about us
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Know which display mode I prefer?"}
        ]
    },
    config=config,
)
print(response["messages"][-1].content)
# Output: "I don't seem to have any stored memories about your display mode preferences..."

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "dark. Remember that."}
        ]
    },
    # We will continue the conversation (thread-a) by using the config with
    # the same thread_id
    config=config,
)

# New thread = new conversation!
# highlight-next-line
new_config = {"configurable": {"thread_id": "thread-b"}}
# The agent will only be able to recall
# whatever it explicitly saved using the manage_memories tool
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remember me? What are my preferences?"}]},
    # highlight-next-line
    config=new_config,
)
print(response["messages"][-1].content)
# Output: "Based on my memory search, I can see that you've previously indicated a preference for dark display mode..."
```

This example demonstrates memory persistence across conversations and thread isolation between users. The agent stores the user's dark mode preference in one thread and can access it from another thread by searching for it.


## Next Steps

In this quick start, you configured an agent to manage its memory "in the hot path" using tools. Check out the following guides for other features:

- [Reflection Quickstart](background_quickstart.md) – Learn how to manage memories "in the background" using `create_memory_store_manager`.