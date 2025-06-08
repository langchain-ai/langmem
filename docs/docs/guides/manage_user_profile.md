---
title: How to Manage User Profiles
---

# How to Manage User Profiles

User profiles help your LLM maintain consistent, up-to-date information about users across conversations. Unlike [semantic memory collections](./extract_semantic_memories.md) which track evolving knowledge, profiles focus on maintaining a concise, structured representation of the user (or the agent itself).

- Personal context (name, language, timezone)
- Communication preferences (formality, detail level, expertise)
- Interaction highlights (last conversation, common topics, key relationships)

This guide shows how to automatically extract and maintain user profiles from conversations.

## Basic Usage

```python
from langmem import create_memory_manager
from pydantic import BaseModel
from typing import Optional


# Define profile structure
class UserProfile(BaseModel):
    """Represents the full representation of a user."""
    name: Optional[str] = None
    language: Optional[str] = None
    timezone: Optional[str] = None


# Configure extraction
manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[UserProfile], # (optional) customize schema (1)
    instructions="Extract user profile information",
    enable_inserts=False,  # Profiles update in-place (2)
)

# First conversation
conversation1 = [{"role": "user", "content": "I'm Alice from California"}]
memories = manager.invoke({"messages": conversation1})
print(memories[0])
# ExtractedMemory(id='profile-1', content=UserProfile(
#    name='Alice',
#    language=None,
#    timezone='America/Los_Angeles'
# ))

# Second conversation updates existing profile
conversation2 = [{"role": "user", "content": "I speak Spanish too!"}]
update = manager.invoke({"messages": conversation2, "existing": memories})
print(update[0])
# ExtractedMemory(id='profile-1', content=UserProfile(
#    name='Alice',
#    language='Spanish',  # Updated
#    timezone='America/Los_Angeles'
# ))
```

1. You can use Pydantic models or json schemas to define your profile. This ensures type safety for stored data and serves to instruct the model on what type of information is important for your application.

2. Unlike semantic memory extraction, we set `enable_inserts=False`, meaning it will only ever manage a single instance of the memory.

    For more about profiles, see [Semantic Memory](../concepts/conceptual_guide.md#semantic-memory-facts-and-knowledge).

## With LangGraph's Long-term Memory Store

To maintain profiles across conversations, use `create_memory_store_manager`:

```python
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_config
from langmem import create_memory_store_manager

# Set up store and models (1)
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

# Create profile manager (2)
manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("users", "{user_id}", "profile"),  # Isolate profiles by user
    schemas=[UserProfile],
    enable_inserts=False,  # Update existing profile only
)

@entrypoint(store=store)
def chat(messages: list):
    # Get user's profile for personalization
    configurable = get_config()["configurable"]
    results = store.search(
        ("users", configurable["user_id"], "profile")
    )
    profile = None
    if results:
        profile = f"""<User Profile>:

{results[0].value}
</User Profile>
"""
    
    # Use profile in system message
    response = my_llm.invoke([
        {
            "role": "system",
            "content": f"""You are a helpful assistant.{profile}"""
        },
        *messages
    ])

    # Update profile with any new information
    manager.invoke({"messages": messages})
    return response

# Example usage
await chat.ainvoke(
    [{"role": "user", "content": "I'm Alice from California"}],
    config={"configurable": {"user_id": "user-123"}}
)

await chat.ainvoke(
    [{"role": "user", "content": "I just passed the N1 exam!"}],
    config={"configurable": {"user_id": "user-123"}}
)

print(store.search(("users", "user-123", "profile")))
```

1. For production, use [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore) instead of `InMemoryStore`

2. The namespace pattern lets you organize profiles by:
    ```python
    # Individual users
    ("users", "user-123", "profile")
    
    # Teams/departments
    ("users", "team-sales", "profile")
    
    # Roles
    ("users", "admin-1", "profile")
    ```

See [Storage System](../concepts/conceptual_guide.md#storage-system) for more about store configuration.
