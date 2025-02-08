---
title: Delayed Background Memory Processing
description: Process memories during conversation quiet periods
---

# Delayed Background Memory Processing

When conversations are active, an agent may receive many messages in quick succession. Instead of processing each message immediately for long-term memory management, you can wait for conversation activity to settle. This guide shows how to use [`ReflectionExecutor`](../reference/utils.md#langmem.ReflectionExecutor) to debounce memory processing.

## Problem

Processing memories on every message has drawbacks:
- Redundant work when messages arrive in quick succession
- Incomplete context when processing mid-conversation
- Unnecessary token consumption

[`ReflectionExecutor`](../reference/utils.md#langmem.ReflectionExecutor) defers memory processing and cancels redundant work:

```python hl_lines="11 21-24"
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import ReflectionExecutor, create_memory_store_manager

# Create memory manager to extract memories from conversations (1)
memory_manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories",),
)
# Wrap memory_manager to handle deferred background processing (2)
executor = ReflectionExecutor(memory_manager)

@entrypoint(store=InMemoryStore())
def chat(message: str):
    response = llm.invoke(message)
    # Format conversation for memory processing
    # Must follow OpenAI's message format
    to_process = {"messages": [{"role": "user", "content": message}] + [response]}
    
    # Wait 30 minutes before processing
    # If new messages arrive before then:
    # 1. Cancel pending processing task
    # 2. Reschedule with new messages included
    delay = 0.5 # In practice would choose longer (30-60 min)
    # depending on app context.
    executor.submit(to_process, after_seconds=delay)
    return response.content
```

1. The [`create_memory_store_manager`](../reference/memory.md#langmem.create_memory_store_manager) creates a Runnable that extracts memories from conversations. It processes messages in OpenAI's format:
   ```python
   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
   ```

2. The [`ReflectionExecutor`](../reference/utils.md#langmem.ReflectionExecutor) handles background processing of memories. For each conversation thread:
   
    - Maintains a queue of pending memory tasks
    - Cancels old tasks when new messages arrive
    - Only processes after the specified delay

    This debouncing ensures you process complete conversation context instead of fragments.

    !!! warning "Serverless Deployments"
        Local threads terminate between serverless function invocations. Use the LangGraph Platform's remote executor instead.

        ```python
        ReflectionExecutor(
            "my_memory_manager", 
            ("memories",), 
            url="http://localhost:2024",
        )
        ```
