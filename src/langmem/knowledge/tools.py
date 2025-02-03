import typing
import uuid

from langchain_core.tools import StructuredTool
from langgraph.utils.config import get_store

from langmem import utils

## LangGraph Tools


def create_manage_memory_tool(
    instructions: str = """Proactively call this tool when you:
1. Identify a new USER preference.
2. Receive an explicit USER request to remember something or otherwise alter your behavior.
3. Are working and want to record important context.
4. Identify that an existing MEMORY is incorrect or outdated.""",
    namespace: tuple[str, ...] = (
        "memories",
        "{langgraph_user_id}",
    ),
    kind: typing.Literal["single", "multi"] = "multi",
):
    """Create a tool for managing persistent memories in conversations.

    This function creates a tool that allows AI assistants to create, update, and delete
    persistent memories that carry over between conversations. The tool helps maintain
    context and user preferences across sessions.

    Args:
        instructions: Custom instructions for when to use the memory tool.
            Defaults to a predefined set of guidelines for proactive memory management.
        namespace: Storage namespace
            structure for organizing memories.
        kind: Whether to support single or multiple
            memories per conversation.

    Tip:
        This tool connects with the LangGraph [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) configured in your graph or entrypoint.
        It will not work if you do not provide a store.

    !!! example "Examples"
        ```python
        from langgraph.func import entrypoint
        from langgraph.store.memory import InMemoryStore

        memory_tool = create_manage_memory_tool(
            # All memories saved to this tool will live within this namespace
            # The brackets will be populated at runtime by the configurable values
            namespace=("project_memories", "{langgraph_user_id}"),
        )

        store = InMemoryStore()


        @entrypoint(store=store)
        async def workflow(state: dict, *, previous=None):
            # Other work....
            result = await memory_tool.ainvoke(state)
            print(result)
            return entrypoint.final(value=result, save={})


        config = {
            "configurable": {
                "langgraph_user_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
        # Create a new memory
        await workflow.ainvoke(
            {"content": "Team prefers to use Python for backend development"},
            config=config,
        )
        # Output: 'created memory 123e4567-e89b-12d3-a456-426614174000'

        # Update an existing memory
        result = await workflow.ainvoke(
            {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "Team uses Python for backend and TypeScript for frontend",
                "action": "update",
            },
            config=config,
        )
        print(result)
        # Output: 'updated memory 123e4567-e89b-12d3-a456-426614174000'
        ```

    Returns:
        memory_tool (Tool): A decorated async function that can be used as a tool for memory management.
            The tool supports creating, updating, and deleting memories with proper validation.
    """
    namespacer = (
        utils.NamespaceTemplate(namespace)
        if isinstance(namespace, tuple)
        else namespace
    )

    async def amanage_memory(
        content: typing.Optional[str] = None,
        action: typing.Literal["create", "update", "delete"] = "create",
        *,
        id: typing.Optional[uuid.UUID] = None,
    ):
        store = get_store()

        if action == "create" and id is not None:
            raise ValueError(
                "You cannot provide a MEMORY ID when creating a MEMORY. Please try again, omitting the id argument."
            )

        if action in ("delete", "update") and not id:
            raise ValueError(
                "You must provide a MEMORY ID when deleting or updating a MEMORY."
            )
        namespace = namespacer()
        if action == "delete":
            await store.adelete(namespace, key=str(id))
            return f"Deleted memory {id}"

        id = id or uuid.uuid4()
        await store.aput(
            namespace,
            key=str(id),
            value={"content": content},
        )
        return f"{action}d memory {id}"

    def manage_memory(
        content: typing.Optional[str] = None,
        action: typing.Literal["create", "update", "delete"] = "create",
        *,
        id: typing.Optional[uuid.UUID] = None,
    ):
        store = get_store()

        if action == "create" and id is not None:
            raise ValueError(
                "You cannot provide a MEMORY ID when creating a MEMORY. Please try again, omitting the id argument."
            )

        if action in ("delete", "update") and not id:
            raise ValueError(
                "You must provide a MEMORY ID when deleting or updating a MEMORY."
            )
        namespace = namespacer()
        if action == "delete":
            store.delete(namespace, key=str(id))
            return f"Deleted memory {id}"

        id = id or uuid.uuid4()
        store.put(
            namespace,
            key=str(id),
            value={"content": content},
        )
        return f"{action}d memory {id}"

    description = """Create, update, or delete persistent MEMORIES that will be carried over to future conversations.
        {instructions}""".format(instructions=instructions)

    return StructuredTool.from_function(
        manage_memory, amanage_memory, name="manage_memory", description=description
    )


_MEMORY_SEARCH_INSTRUCTIONS = ""


def create_search_memory_tool(
    instructions: str = _MEMORY_SEARCH_INSTRUCTIONS,
    namespace: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
):
    namespacer = utils.NamespaceTemplate(namespace)

    async def asearch_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        store = get_store()
        namespace = namespacer()
        memories = await store.asearch(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )
        return [m.dict() for m in memories], memories

    def search_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        store = get_store()
        namespace = namespacer()
        memories = store.search(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )
        return [m.dict() for m in memories], memories

    description = """Search your long-term memories for information relevant to your current context. {instructions}""".format(
        instructions=instructions
    )

    return StructuredTool.from_function(
        search_memory,
        asearch_memory,
        name="search_memory",
        description=description,
        response_format="content_and_artifact",
    )
