# LangMem Memory Utilities

LLM apps work best if they can remember important preferences, skills, and knowledge. LangMem provides utilities commonly used to build memory systems that help your agents:

1. **Remember user preferences** - Store and recall user settings, preferences, and important facts
1. **Learn from interactions** - Extract and save key information from conversations
1. **Use context intelligently** - Use relevant memories when needed using semantic search

LangMem offers a variety of utilities you can use in any framework, as well as higher-level primitives that integrate with any LangGraph application's persistent [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore).

## Installation

```bash
pip install -U langmem
```

## Examples

- [Agent actively manages memories](#build-an-agent-that-actively-manages-memories)
- [Passive memory reflection](#passive-memory-reflection)

## What's inside

LangMem's main utilities enrich or optimize a memory "state" given new input (conversations/agent trajectories with optional feedback). These utilities are organized on three levels of abstraction. If you're keen to dive deeper, you can also dive headfirst into the [quick examples](#quick-examples).

### 1. Functional primitives

Langmem's core primitives are **functional**, meaning they have **no side effects** and do **not** depend on any persistance layer.

Each function accepts two arguments: "trajectories" with optional annotations, and the current "state" of the memory. The functions return the updated "state" of the memory. Current states include: single prompts, multi-prompt "systems" (aka multi-agent systems, workflows, etc.), and lists of memory objects (dictionaries that follow custom schemas).

#### create_memory_enricher

Extract and update memories from a conversation as a list of memory objects, which could follow any schema and by default are unstructured `{"content": str}` dicts. This is especially useful for managing semantic memories (facts, relationships, etc.), episodic events, and profiles (user preferences, core information, etc.).

<details>
<summary>Example Usage</summary>

```python
from langmem import create_memory_enricher

enricher = create_memory_enricher(
    "anthropic:claude-3-5-sonnet-latest",
    # schemas=[Memory],  # You can provide 1 or more custom memory schemas
    instructions="Extract the information in the form subject:object:predicts.",
    enable_inserts=True, # "Inserts" mean that new memories can be created. Otherwise, you will only get updates to existing memories
    enable_deletes=False, # "Deletes" here will return a special "RemoveDoc" object indicating a memory should be removed
)

conversation = [
    {"role": "user", "content": "I prefer dark mode in all my apps"},
    {"role": "assistant", "content": "I'll remember that preference"},
]

memories = enricher.invoke({"messages": conversation})
print(memories[0].content)  # First memory
```

Output:

```text
content='user:appearance_preference:prefers dark mode in applications'
```

</details>

For more information, check out the [reference docs](https://langchain-ai.github.io/langmem/reference/#langmem.create_memory_enricher).

#### create_prompt_optimizer

Learn from trajectories and user conversations to improve the instructions of a system prompt based on environmental or user feedback. This is useful for learning concise forms of core rules, preferences, and other important information that you want to be made always available in the prompt.

To create the optimizer, configure the model, kind/strategy of optimization, and (optionally) a strategy-specific configuration dictionary.

<details>
<summary>Example Usage</summary>

```python
from langmem import create_prompt_optimizer

optimizer = create_prompt_optimizer(
    # The model will be used for the optmization step.
    "anthropic:claude-3-5-sonnet-latest",
    kind="metaprompt",  # Could also be "gradient" or "prompt_memory"
    config={  # The config depends on the kind of optimizer you're instantiating
        "max_reflection_steps": 3,
        "min_reflection_steps": 0,
    },
)
```

The optimizer is a [Runnable](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#runnable) object (aka, it has synchronous `invoke` and asynchronous `ainvoke` methods).

```python
prompt = "You are a helpful assistant."
trajectory = [
    {"role": "user", "content": "How do i print in python?"},
    {
        "role": "assistant",
        "content": "Oh that is an excellent question! It seems you are learning programming languages...",
    },
    {"role": "user", "content": "I just want the code"},
]
feedback = None  # If you receive explicit user feedback or have other signals you want to include, you may provide here.
optimized = optimizer.invoke(
    {"trajectories": [(trajectory, feedback)], "prompt": prompt}
)
print(optimized)
```

Output:

```text
You are a helpful assistant. When responding to programming questions:
1. Always start with a practical code example that directly answers the question
2. Keep explanations clear and concise
3. Format code blocks using appropriate markdown
4. Only add supplementary information after providing the direct answer

For non-programming questions, prioritize direct and practical answers that address the core question being asked. Be concise while remaining helpful and accurate.
```

</details>

For more information, check out the [reference docs](https://langchain-ai.github.io/langmem/reference/#langmem.create_prompt_optimizer).

#### create_multi_prompt_optimizer

Optimize multiple prompts simultaneously based on conversation trajectories and feedback. This is useful when you have a multi-agent system or workflow where several components need to learn and adapt their behavior together.

<details>
<summary>Example Usage</summary>

````python
from langmem import create_multi_prompt_optimizer

multi_optimizer = create_multi_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="metaprompt",  # Strategy for updating prompts
    config={"max_reflection_steps": 3},  # Optional config for the strategy
)

code_trajectory = [
    {"role": "user", "content": "How do I read a CSV file in Python?"},
    {"role": "assistant", "content": "Here's how to read a CSV:\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```"},
    {"role": "user", "content": "I got a FileNotFoundError"},
]
code_feedback = {"user_score": 0, "user_comment": "gave up too soon"}

docs_trajectory = [
    {"role": "user", "content": "What's new in pandas 2.0?"},
    {"role": "assistant", "content": "According to the official changelog, key changes in pandas 2.0 include:\n1. Arrow-backed data types\n2. Copy-on-Write\n3. PyArrow string dtype as default"},
]
docs_feedback = None # Not all trajectories need explicit feedback

prompts = [
    {
        "name": "code_executor",
        "prompt": "You write and debug Python code.",
        "when_to_update": "When code examples need improvement in reliability or user experience.",
        "update_instructions": "Focus on error handling, input validation, and clear error messages.",
    },
    {
        "name": "documentation_researcher",
        "prompt": "You research and explain technical documentation.",
        "when_to_update": "When explanations of technical concepts or updates could be clearer.",
        "update_instructions": "Maintain accuracy while improving clarity and structure.",
    },
]

# Optimize multiple prompts together
optimized = multi_optimizer.invoke(
    {
        "trajectories": [(code_trajectory, code_feedback), (docs_trajectory, None)],
        "prompts": prompts,
    }
)
for opt in optimized:
    print(opt["name"])
    print(opt["prompt"])
    print("-" * 80)

````

Output:

```text
code_executor
You write and debug Python code with a focus on production-ready solutions. Follow these guidelines:

1. Always include appropriate error handling (try/except blocks) in your code examples
2. Validate inputs and file paths before processing
3. Provide clear error messages and recovery steps
4. When a user reports an error, explain:
   - What caused the error
   - How to fix it
   - How to prevent it in the future
5. Include comments explaining key error handling and validation steps

For file operations, database connections, and other critical operations, always demonstrate proper error handling and resource cleanup.

Your solutions should be complete and ready for production use, not just basic examples.
------------------------------------------------------------------------------
documentation_researcher
You research and explain technical documentation.
```

</details>

For more information on this function, check out the [reference docs](https://langchain-ai.github.io/langmem/reference/#langmem.create_multi_prompt_optimizer).

### 2. Stateful primitives

The stateful primitives _extend_ the functional primitives above to integrate with LangGraph's persistent [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore).

#### create_memory_store_enricher

Apply enrichment with LangGraph's integrated BaseStore. This allows you to automatically extract and store memories from conversations using a persistent store.

<details>
<summary>Example Usage</summary>

_Note: we are running this inline; in production, you would want to run reflection as a separate run, e.g., using `client.runs.create(...)`_

```python
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from langmem import create_memory_store_enricher

store = InMemoryStore()
enricher = create_memory_store_enricher(
    "anthropic:claude-3-5-sonnet-latest", namespace=("memories",)
)
my_llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


@entrypoint(store=store)
def app(messages: list):
    items = store.search(("memories",), query=str(messages[-1]["content"]))
    memories = "\n".join(str(item.value) for item in items)
    response = my_llm.invoke(
        [
            {
                "role": "system",
                "content": f"You are a helpful assistant.\n\nMemories:\n{memories}",
            }
        ]
        + messages
    )
    enricher.invoke({"messages": messages})

    return response


app.invoke(
    [
        {"role": "user", "content": "I prefer dark mode in all my apps"},
    ]
)
print(store.search(("memories",)))
```

Output:

```text
[
  Item(
    namespace=['memories'],
    key='63a013eb-3e6a-433c-8733-8b98ed8a9933',
    value={
      'kind': 'Memory',
      'content': {
        'content': 'The user has a strong preference for dark mode interfaces in applications.'
      }
    },
    created_at='2025-02-04T00:43:53.418582+00:00',
    updated_at='2025-02-04T00:43:53.418584+00:00',
    score=None
  )
]
```

</details>

For more information, check out the [reference docs](https://langchain-ai.github.io/langmem/reference/#langmem.create_memory_store_enricher).

#### create_manage_memory_tool

A tool for creating and updating stored memories. This is particularly useful when you want to give an agent the ability to explicitly manage its memory store.

<details>
<summary>Example Usage</summary>

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool

store = InMemoryStore()

# Create an agent with the manage memory tool
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[
        create_manage_memory_tool(
            instructions="Note everything the user tells you. Even if it's banal.",
            namespace=("memories",),
        )
    ],
    store=store,
)

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Remember that I am allergic to javascript"}
        ]
    }
)

# Verify the memory was stored
print(store.search(("memories",)))
```

Output:

```text
[
  Item(
    namespace=['memories'],
    key='01bc5be3-8534-4faf-8e27-b02b3d125079',
    value={'content': 'User is allergic to JavaScript'},
    created_at='2025-02-04T00:50:58.617656+00:00',
    updated_at='2025-02-04T00:50:58.617658+00:00',
    score=None
  )
]
```

</details>

For more information, check out the [reference docs](https://langchain-ai.github.io/langmem/reference/#langmem.create_manage_memory_tool).

#### create_search_memory_tool

A tool for searching stored memories. Best used alongside the manage memory tool to give agents the ability to both store and retrieve memories.

<details>
<summary>Example Usage</summary>

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from langmem import create_search_memory_tool

store = InMemoryStore()

# Create agent with both memory tools
namespace=("memories", "{langgraph_user_id}")
tools = [create_search_memory_tool(namespace=namespace)]
store.put(
    namespace, key="amemory", value={"content": "User is allergic to JavaScript"}
)

agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest", tools=tools, store=store
)
final_state = agent.invoke(
    {"messages": [{"role": "user", "content": "What allergies do I have?"}]}
)
print(final_state["messages"][-1].content)
```

```text
Based on the search results, I found that you are allergic to JavaScript. This appears to be the only allergy I have information about in my memory.
```

</details>

For more information, check out the [reference docs](https://langchain-ai.github.io/langmem/reference/#langmem.create_search_memory_tool).

### 3. Prebuilt entrypoints

These are LangGraph instances that can be directly deployed on the LangGraph platform to perform memory extraction, enrichment, and querying or to perform prompt optimization.

#### Semantic memory

This graph exposes a **stateful memory enrichment** entrypoint similar to that of the `create_memory_store_enricher` utility. This interface allows you to automatically extract and store memories from conversations using a persistent store. By default, all information is namespaced by the authenticated `langsmith_user_id`.


<details>
<summary>Example Usage</summary>

Assuming you've launched a server using `langgraph dev`:

```python
from langgraph_sdk import get_client

# Connect to local development server
client = get_client(url="http://localhost:2024")

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

_Note: The graph's API looks like the following:_

```python
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage


class InputState(TypedDict):
    messages: list[AnyMessage]
    schemas: None | list[dict] | dict  # Default is memories with content: str
    namespace: tuple[str, ...] | None = None


class OutputState(TypedDict):
    updated_memories: list[dict]
```


</details>

#### Prompt optimization

This graph exposes a **stateless** prompt optimization entrypoint that proposes updated prompts based on the provided trajectoreis.


<details>
<summary>Example Usage</summary>

Assuming you've launched a server using `langgraph dev`:

```python
from langgraph_sdk import get_client

# Connect to local development server
client = get_client(url="http://localhost:2024")

# Example conversation with feedback
conversation = [
    {"role": "user", "content": "How do I read a CSV?"},
    {"role": "assistant", "content": "Use pandas: df = pd.read_csv('file.csv')"},
    {"role": "user", "content": "I got an error"},
]
feedback = {"score": 3, "comment": "Should explain imports and error handling"}

# Use the prompt optimization graph
results = await client.runs.wait(
    None,
    "optimize_prompts",
    input={
        "threads": [[conversation, feedback]],
        "prompts": [
            {
                "name": "code_assistant",
                "prompt": "You help write Python code.",
                "when_to_update": "When explanations could be clearer",
                "update_instructions": "Improve clarity while maintaining brevity",
            }
        ],
    },
    config={"configurable": {"model": "claude-3-5-sonnet-latest"}},
)

print(results["updated_prompts"][0]["prompt"])
# You help write Python code. Always:
# 1. Include necessary imports
# 2. Add basic error handling
# 3. Explain common pitfalls
```

_Note: The graph's API looks like the following:_

```python
class Prompt(TypedDict):
    name: str
    prompt: str
    update_instructions: str | None
    when_to_update: str | None


# The input to the graph
class InputState(TypedDict):
    prompts: list[Prompt] | str
    threads: list[tuple[list[AnyMessage], dict[str, str]]]


class OutputState(TypedDict):
    updated_prompts: list[Prompt]

```
</details>

## Quick Examples

After installing `langmem`, please ensure you've configured your envirnoment with an API key for your favorite LLM provider. For example:

```bash
export ANTHROPIC_API_KEY="sk-..."
```

### Build an agent that actively manages memories

Here's how to create an agent that can manage its own memories "consciously", or in the hot path:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool, create_search_memory_tool

store = InMemoryStore()

tools = [
    create_manage_memory_tool(namespace=("memories",)),  # Persist memories
    create_search_memory_tool(namespace=("memories",)),  # Search stored memories
]

agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest", tools=tools, store=store
)
agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Remember, I am allergic to javascript."}
        ]
    }
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's stored in your memory?"}]}
)
```

### Build an agent that learns from passive reflection

(Coming soon)

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
   from langmem import create_memory_enricher


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
       namespace=("user", "experiences"),
       kind="multi",
   )
   ```

### Episodic Memory

Episodic memory helps agents recall past events and experiences:

TODO: Add exampel implementation.

### Procedural Memory

Procedural memory helps agents remember how to perform tasks through system prompts and instructions:

## Writing Memories

LangMem supports two primary approaches to memory formation:

### Conscious memory formation

Agents can actively manage memories "in the hot path" by calling tools to save, update, and delete memories. Our [create_manage_memory_tool](https://langchain-ai.github.io/langmem/reference/#langmem.create_manage_memory_tool) works for this purpose.

```python
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool
from langgraph.store.memory import InMemoryStore

memory_tool = create_manage_memory_tool(
    instructions="Save important user preferences and context",
    namespace=("user", "preferences"),
)
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest", tools=[memory_tool], store=InMemoryStore()
)

agent.invoke(
    {
        "messages": "Did you know I hold the world record for most stubbed toes in one day?",
    }
)
```

### Subconscious memory formation

Memories and other data enrichment can alternatively run as a background process, which happens asynchronously through reflection. This is facilitated by the  the `create_memory_enricher` utility and its stateful variant `create_memory_store_enricher`.

## Storage patterns

LangMem's lowest-level primitives are purely **functional** - they take trajectories and current memory state (prompts or similar memories) as input and return updated memory state. These primitives form the foundation for higher-level utilities that integrate with LangGraph for persistent storage. You can use them to persist learned information in different ways:

### Long-term storage

As mentioned above, several of LangMem's memory utilities make use of LangGraph's [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) interface, which provides a namespaced document store with semantic search capabilities. Memories are organized using:

1. **Namespaces**: Logical groupings similar to directories (e.g., `("memories", "{user_id}", app_context)`)
2. **Keys**: Unique identifiers within namespaces (like filenames)
3. **Storage**: JSON documents, which can be indexed for semantic search


### Prompt

Prompts can also be a memory format that can be used for long-term storage. These can be saved in places like LangSmith's [prompt hub](https://docs.smith.langchain.com/prompt_engineering/how_to_guides#prompt-hub), checked in to your filesystem, or also saved in the [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore). 


### Custom

By using the [functional primitives](#1-functional-primitives) directly, you can choose to save memories in any arbitrary location, be that a vector store, a filesystem, or a regular database.


While you can always work with [`BaseStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) directly, LangMem provides higher-level primitives (memory tools, stateful utilities, graphs) that manage memories on behalf of your agent, handling the storage operations automatically.

This architecture allows you to make deliberate choices about:

1. **What** to remember ([memory types](#memory-types))
2. **When** to remember ([writing memories](#writing-memories))
3. **Where** to store information and how to "remember" it ([storage patterns](#storage-patterns))
