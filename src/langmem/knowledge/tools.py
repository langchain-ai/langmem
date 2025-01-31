import typing
import uuid

from langchain_core.tools import tool
from langgraph.utils.config import get_store

from langmem import utils

## LangGraph Tools


def create_manage_memory_tool(
    instructions: str = "",
    namespace_prefix: tuple[str, ...] | utils.NamespaceTemplate = (
        "memories",
        "{user_id}",
    ),
    kind: typing.Literal["single", "multi"] = "multi",
):
    """Create a tool for managing persistent memories in conversations.

    This function creates a tool that allows AI assistants to create, update, and delete
    persistent memories that carry over between conversations. The tool helps maintain
    context and user preferences across sessions.

    Tip:
        This tool connects with the LangGraph BaseStore configured in your graph or entrypoint.
        It will not work if you do not provide a store.

    !!! example "Examples"
        ```python
        from langgraph.func import entrypoint
        from langgraph.store.memory import InMemoryStore

        memory_tool = create_manage_memory_tool(
            # All memories saved to this tool will live within this namespace
            # The brackets will be populated at runtime by the configurable values
            namespace_prefix=("project_memories", "{langgraph_user_id}"),
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

    Args:
        instructions (str, optional): Custom instructions for when to use the memory tool.
            Defaults to a predefined set of guidelines for proactive memory management.
        namespace_prefix (Union[tuple[str, ...], NamespaceTemplate], optional): Storage namespace
            structure for organizing memories. Defaults to ("memories", "{user_id}").
        kind (Literal["single", "multi"], optional): Whether to support single or multiple
            memories per conversation. Defaults to "multi".

    Returns:
        memory_tool (Tool): A decorated async function that can be used as a tool for memory management.
            The tool supports creating, updating, and deleting memories with proper validation.
    """
    if not instructions:
        instructions = """Proactively call this tool when you:
1. Identify a new USER preference.
2. Receive an explicit USER request to remember something or otherwise alter your behavior.
3. Are working and want to record important context.
4. Identify that an existing MEMORY is incorrect or outdated."""
    namespacer = (
        utils.NamespaceTemplate(namespace_prefix)
        if isinstance(namespace_prefix, tuple)
        else namespace_prefix
    )

    @tool
    async def manage_memory(
        content: typing.Optional[str] = None,
        action: typing.Literal["create", "update", "delete"] = "create",
        *,
        id: typing.Optional[uuid.UUID] = None,
    ):
        """Create, update, or delete persistent MEMORIES that will be carried over to future conversations.
        {instructions}"""
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

    manage_memory.__doc__.format(instructions=instructions)

    return manage_memory


_MEMORY_SEARCH_INSTRUCTIONS = ""


def create_search_memory_tool(
    instructions: str = _MEMORY_SEARCH_INSTRUCTIONS,
    namespace_prefix: tuple[str, ...] = ("memories", "{user_id}"),
):
    namespacer = utils.NamespaceTemplate(namespace_prefix)

    @tool(response_format="content_and_artifact")
    async def search_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        """Search your long-term memories for information relevant to your current context. {instructions}"""
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

    search_memory.__doc__.format(instructions=instructions)  # type: ignore

    return search_memory
