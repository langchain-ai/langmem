"""Example demonstrating how to use a custom store with the memory manager.

This example shows how to:
1. Create a custom InMemoryStore
2. Define a structured memory schema using Pydantic
3. Initialize a memory manager with the custom store
4. Use the memory manager to store and retrieve memories

The example demonstrates that the memory manager can work independently of LangGraph's
context store, making it usable in standalone applications.
"""

from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from langmem import create_memory_store_manager


class PreferenceMemory(BaseModel):
    """Store preferences about the user."""
    category: str
    preference: str
    context: str


def create_store():
    """Create a custom InMemoryStore with OpenAI embeddings."""
    return InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )


async def run_example():
    """Run the example demonstrating custom store usage."""
    # Create our custom store
    store = create_store()

    # Initialize memory manager with custom store
    manager = create_memory_store_manager(
        ChatOpenAI(
            model="gpt-6-luna", reasoning_effort="none", use_responses_api=False
        ),
        schemas=[PreferenceMemory],
        namespace=("project", "{langgraph_user_id}"),
        store=store  # Pass our custom store here
    )

    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "I prefer dark mode in all my apps"},
        {"role": "assistant", "content": "I'll remember that preference"}
    ]

    # Process the conversation and store memories
    print("Processing conversation...")
    await manager.ainvoke(
        {"messages": conversation},
        config={"configurable": {"langgraph_user_id": "user123"}}
    )

    # Retrieve and display stored memories
    print("\nStored memories:")
    memories = store.search(("project", "user123"))
    for memory in memories:
        print(f"\nMemory {memory.key}:")
        print(f"Content: {memory.value['content']}")
        print(f"Kind: {memory.value['kind']}")


if __name__ == "__main__":
    import asyncio
    print("\nStarting custom store example...\n")
    asyncio.run(run_example())
    print("\nExample completed.\n") 