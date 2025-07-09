---
title: How to Extract Semantic Memories
---

# How to Extract Semantic Memories

Need to extract multiple related facts from conversations? Here's how to use LangMem's collection pattern for semantic memories. For single-document patterns like user profiles, see [Manage User Profile](./manage_user_profile.md).

## Without storage

Extract semantic memories:

```python
from langmem import create_memory_manager # (1)!
from pydantic import BaseModel

class Triple(BaseModel): # (2)!
    """Store all new facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

# Configure extraction
manager = create_memory_manager(  
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Triple], 
    instructions="Extract user preferences and any other useful information",
    enable_inserts=True,
    enable_deletes=True,
)
```

1. LangMem has two similar objects for extracting and enriching memory collections:

    - `create_memory_manager`: (this examples) You control storage and updates
    - `create_memory_store_manager`: Handles the memory search, upserts, and deletes directly in whichever BaseStore is configured for the graph context

    The latter uses the former. Both of these work by prompting an LLM to use parallel tool calling to extract new memories, update old ones, and (if configured) delete old ones.

2. Here our custom "`Triple`" memory schema shapes memory extraction. Without context, memories can be ambiguous when retrieved later:
    ```python
    {"content": "User said yes"}  # No context; unhelpful
    ```
    Adding context helps the LLM apply memories correctly:
    ```python
    {
        "subject": "user",
        "predicate": "response",
        "object": "yes",
        "context": "When asked about attending team meeting"
    }
    {
        "subject": "user",
        "predicate": "response",
        "object": "no",
        "context": "When asked if they were batman"
    }
    ```
    It's often a good idea to either schematize memories to encourage certain fields to be stored consistently, or at least to include instructions so the LLM
    saves memories that are sufficiently informative in isolation.

After the first short interaction, the system has extracted some semantic triples:

```python
# First conversation - extract triples
conversation1 = [
    {"role": "user", "content": "Alice manages the ML team and mentors Bob, who is also on the team."}
]
memories = manager.invoke({"messages": conversation1})
print("After first conversation:")
for m in memories:
    print(m)
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
```

The second conversation updates some existing memories. Since we have enabled "deletes", the manager will return `RemoveDoc` objects to indicate that the memory should be removed, and a new memory will be created in its place. Since this uses the core "functional" API (aka, it doesn't read or write to a database), you can control what "removal" means, be that a soft or hard delete, or simply a down-weighting of the memory.

```python
# Second conversation - update and add triples
conversation2 = [
    {"role": "user", "content": "Bob now leads the ML team and the NLP project."}
]
update = manager.invoke({"messages": conversation2, "existing": memories})
print("After second conversation:")
for m in update:
    print(m)
# ExtractedMemory(id='65fd9b68-77a7-4ea7-ae55-66e1dd603046', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='7f8be100-5687-4410-b82a-fa1cc8d304c0', content=Triple(subject='Bob', predicate='leads', object='ML_team', context=None))
# ExtractedMemory(id='f4c09154-2557-4e68-8145-8ccd8afd6798', content=Triple(subject='Bob', predicate='leads', object='NLP_project', context=None))
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
existing = [m for m in update if isinstance(m.content, Triple)]
```

The third conversation overwrites even more memories.
```python
# Delete triples about an entity
conversation3 = [
    {"role": "user", "content": "Alice left the company."}
]
final = manager.invoke({"messages": conversation3, "existing": existing})
print("After third conversation:")
for m in final:
    print(m)
# ExtractedMemory(id='7ca76217-66a4-4041-ba3d-46a03ea58c1b', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='35b443c7-49e2-4007-8624-f1d6bcb6dc69', content=RemoveDoc(json_doc_id='0214f151-b0c5-40c4-b621-db36b845956c'))
# ExtractedMemory(id='65fd9b68-77a7-4ea7-ae55-66e1dd603046', content=RemoveDoc(json_doc_id='f1bf258c-281b-4fda-b949-0c1930344d59'))
# ExtractedMemory(id='7f8be100-5687-4410-b82a-fa1cc8d304c0', content=Triple(subject='Bob', predicate='leads', object='ML_team', context=None))
# ExtractedMemory(id='f4c09154-2557-4e68-8145-8ccd8afd6798', content=Triple(subject='Bob', predicate='leads', object='NLP_project', context=None))
# ExtractedMemory(id='f1bf258c-281b-4fda-b949-0c1930344d59', content=Triple(subject='Alice', predicate='manages', object='ML_team', context=None))
# ExtractedMemory(id='0214f151-b0c5-40c4-b621-db36b845956c', content=Triple(subject='Alice', predicate='mentors', object='Bob', context=None))
# ExtractedMemory(id='258dbf2d-e4ac-47ac-8ffe-35c70a3fe7fc', content=Triple(subject='Bob', predicate='is_member_of', object='ML_team', context=None))
```

For more about semantic memories, see [Memory Types](../concepts/conceptual_guide.md#memory-types).

## With storage

The same extraction can be managed automatically by LangGraph's BaseStore:

```python
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

# Set up store and models
store = InMemoryStore(  # (1)!
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("chat", "{user_id}", "triples"),  # (2)!
    schemas=[Triple],
    instructions="Extract all user information and events as triples.",
    enable_inserts=True,
    enable_deletes=True,
)
my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
```

1. For production deployments, use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore). `InMemoryStore` works fine for development but doesn't persist data between restarts.

2. Namespace patterns control memory isolation:
   ```python
   # User-specific memories
   ("chat", "user_123", "triples")
   
   # Team-shared knowledge
   ("chat", "team_x", "triples")
   
   # Global memories
   ("chat", "global", "triples")
   ```
   See [Storage System](../concepts/conceptual_guide.md#storage-system) for namespace design patterns

You can also extract multiple memory types at once:

```text
schemas=[Triple, Preference, Relationship]
```

Each type can have its own extraction rules and storage patterns
Namespaces let you organize memories by user, team, or domain:

```python
# User-specific memories
("chat", "user_123", "triples")

# Team-shared knowledge
("chat", "team_x", "triples")

# Domain-specific extraction
("chat", "user_123", "preferences")
```

The `{user_id}` placeholder is replaced at runtime:

```python {skip}
# Extract memories for User A
manager.invokse(
    {"messages": [{"role": "user", "content": "I prefer dark mode"}]},
    config={"configurable": {"user_id": "user-a"}}  # (1)!
)
```

1. Uses `("chat", "user-a", "triples")`


```python
# Define app with store context
@entrypoint(store=store) # (1)!
def app(messages: list):
    response = my_llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            *messages
        ]
    )

    # Extract and store triples (Uses store from @entrypoint context)
    manager.invoke({"messages": messages}) 
    return response
```

1. `@entrypoint` provides the BaseStore context:

    - Handles cross-thread coordination
    - Manages connection pooling
    See [BaseStore guide](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) for production setup

Then running the app:
```python
# First conversation
app.invoke(
    [
        {
            "role": "user",
            "content": "Alice manages the ML team and mentors Bob, who is also on the team.",
        },
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Second conversation
app.invoke(
    [
        {"role": "user", "content": "Bob now leads the ML team and the NLP project."},
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Third conversation
app.invoke(
    [
        {"role": "user", "content": "Alice left the company."},
    ],
    config={"configurable": {"user_id": "user123"}},
)

# Check stored triples
for item in store.search(("chat", "user123")):
    print(item.namespace, item.value)

# Output:
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'is_member_of', 'object': 'ML_team', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'leads', 'object': 'ML_team', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Bob', 'predicate': 'leads', 'object': 'NLP_project', 'context': None}}
# ('chat', 'user123', 'triples') {'kind': 'Triple', 'content': {'subject': 'Alice', 'predicate': 'employment_status', 'object': 'left_company', 'context': None}}

```


See [Storage System](../concepts/conceptual_guide.md#storage-system) for namespace patterns. This approach is also compatible with the [ReflectionExecutor](../reference/utils.md#langmem.ReflectionExecutor) to defer & deduplicate memory processing.

## Using a Memory Manager Agent

The technique above tries to manage memory, including insertions, deletions, and deletions, in a single LLM call. If there is a lot of new information, this may be complicated for the LLM to multi-task. Alternatively, you could create an agent, similar to that in the [quick start](../hot_path_quickstart.md), which you prompt to manage memory over multiple LLM calls. You can still serparate this agent from your user-facing agent, but it can give your LLM extra time to process new information and complex updates.

```python
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up store and checkpointer
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


def prompt(state):
    """Prepare messages with context from existing memories."""
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a memory manager. Extract and manage all important knowledge, rules, and events using the provided tools.
    


Existing memories:
<memories>
{memories}
</memories>

Use the manage_memory tool to update and contextualize existing memories, create new ones, or delete old ones that are no longer valid.
You can also expand your search of existing memories to augment using the search tool."""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


# Create the memory extraction agent
manager = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    prompt=prompt,
    tools=[
        # Agent can create/update/delete memories
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
)


# Run extraction in background
@entrypoint(store=store)
def app(messages: list):
    response = my_llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            *messages,
        ]
    )

    # Extract and store triples (Uses store from @entrypoint context)
    manager.invoke({"messages": messages})
    return response


app.invoke(
    [
        {
            "role": "user",
            "content": "Alice manages the ML team and mentors Bob, who is also on the team.",
        }
    ]
)

print(store.search(("memories",)))

# [
#     Item(
#         namespace=["memories"],
#         key="5ca8dacc-7d46-40bb-9b3d-f4c2dc5c4b30",
#         value={"content": "Alice is the manager of the ML (Machine Learning) team"},
#         created_at="2025-02-11T00:28:01.688490+00:00",
#         updated_at="2025-02-11T00:28:01.688499+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="586783fa-e501-4835-8651-028c2735f0d0",
#         value={"content": "Bob works on the ML team"},
#         created_at="2025-02-11T00:28:04.408826+00:00",
#         updated_at="2025-02-11T00:28:04.408841+00:00",
#         score=None,
#     ),
#     Item(
#         namespace=["memories"],
#         key="19f75f64-8787-4150-a439-22068b00118a",
#         value={"content": "Alice mentors Bob on the ML team"},
#         created_at="2025-02-11T00:28:06.951838+00:00",
#         updated_at="2025-02-11T00:28:06.951847+00:00",
#         score=None,
#     ),
# ]
```

This approach is also compatible with the [ReflectionExecutor](../reference/utils.md#langmem.ReflectionExecutor) to defer & deduplicate memory processing.

## When to Use Semantic Memories

Semantic memories help agents learn from conversations. They extract and store meaningful information that might be useful in future interactions. For example, when discussing a project, the agent might remember technical requirements, team structure, or key decisions - anything that could provide helpful context later.

The goal is to build understanding over time, just like humans do through repeated interactions. Not everything needs to be remembered - focus on information that helps the agent be more helpful in future conversations. Semantic memory works best when the agent is able to save important memories and the dense relationships between them so that it can later recall not just "what" but "why" and "how".