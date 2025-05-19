from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from langmem import create_memory_store_manager

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories", "{langgraph_user_id}"),
)


class PreferenceMemory(BaseModel):
    """Store preferences about the user."""

    category: str
    preference: str
    context: str


store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)
manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[PreferenceMemory],
    namespace=("project", "team_1"),
)


@entrypoint(store=store)
async def graph(message: str):
    # Hard code the response :)
    response = {"role": "assistant", "content": "I'll remember that preference"}
    await manager.ainvoke(
        {"messages": [{"role": "user", "content": message}, response]}
    )
    return response
