import logging
import typing
import uuid

from langchain_core.tools import StructuredTool
from langgraph.store.base import BaseStore
from langgraph.utils.config import get_store

from langmem import errors, utils

logger = logging.getLogger(__name__)

# LangGraph Tools


def create_manage_memory_tool(
    namespace: tuple[str, ...] | str,
    *,
    instructions: str = "Proactively call this tool when you:\n\n"
    "1. Identify a new USER preference.\n"
    "2. Receive an explicit USER request to remember something or otherwise alter your behavior.\n"
    "3. Are working and want to record important context.\n"
    "4. Identify that an existing MEMORY is incorrect or outdated.\n",
    schema: typing.Type = str,
    actions_permitted: typing.Optional[
        tuple[typing.Literal["create", "update", "delete"], ...]
    ] = ("create", "update", "delete"),
    store: typing.Optional[BaseStore] = None,
    name: str = "manage_memory",
):
    """Create a tool for managing persistent memories in conversations.

    This function creates a tool that allows AI assistants to create, update, and delete
    persistent memories that carry over between conversations. The tool helps maintain
    context and user preferences across sessions.


    Args:
        instructions: Custom instructions for when to use the memory tool.
            Defaults to a predefined set of guidelines for proactive memory management.
        namespace: The namespace structure for organizing memories in LangGraph's BaseStore.
            Uses runtime configuration with placeholders like `{langgraph_user_id}`.
        store: The BaseStore to use for searching. If not provided, the tool will use the configured BaseStore in your graph or entrypoint.
            Only set if you intend on using these tools outside the LangGraph context.

    Returns:
        memory_tool (Tool): A decorated async function that can be used as a tool for memory management.
            The tool supports creating, updating, and deleting memories with proper validation.

    The resulting tool has a signature that looks like the following:
        ```python
        from typing import Literal


        def manage_memory(
            content: str | None = None,  # Content for new/updated memory
            id: str | None = None,  # ID of existing memory to update/delete
            action: Literal["create", "update", "delete"] = "create",
        ) -> str: ...
        ```
        _Note: the tool supports both sync and async usage._

    !!! note "Namespace Configuration"
        The namespace is configured at runtime through the `config` parameter:
        ```python
        # Example: Per-user memory storage
        config = {"configurable": {"langgraph_user_id": "user-123"}}
        # Results in namespace: ("memories", "user-123")

        # Example: Team-wide memory storage
        config = {"configurable": {"langgraph_user_id": "team-x"}}
        # Results in namespace: ("memories", "team-x")
        ```

    Tip:
        This tool connects with the LangGraph [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) configured in your graph or entrypoint.
        It will not work if you do not provide a store.

    !!! example "Examples"
        ```python
        from langmem import create_manage_memory_tool
        from langgraph.func import entrypoint
        from langgraph.store.memory import InMemoryStore

        memory_tool = create_manage_memory_tool(
            # All memories saved to this tool will live within this namespace
            # The brackets will be populated at runtime by the configurable values
            namespace=("project_memories", "{langgraph_user_id}"),
        )

        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )


        @entrypoint(store=store)
        async def workflow(state: dict, *, previous=None):
            # Other work....
            result = await memory_tool.ainvoke(state)
            print(result)
            return entrypoint.final(value=result, save={})


        config = {
            "configurable": {
                # This value will be formatted into the namespace you configured above ("project_memories", "{langgraph_user_id}")
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

        You can use in LangGraph's prebuilt `create_react_agent`:

        ```python
        from langgraph.prebuilt import create_react_agent
        from langgraph.config import get_config, get_store

        def prompt(state):
            config = get_config()
            memories = get_store().search(
                # Search within the same namespace as the one
                # we've configured for the agent
                ("memories", config["configurable"]["langgraph_user_id"]),
            )
            system_prompt = f"\"\"You are a helpful assistant.
        <memories>
        {memories}
        </memories>
        \"\"\"
            system_message = {"role": "system", "content": system_prompt}
            return [system_message, *state["messages"]]

        agent = create_react_agent(
            "anthropic:claude-3-5-sonnet-latest",
            tools=[
                create_manage_memory_tool(namespace=("memories", "{langgraph_user_id}")),
            ],
            store=store,
        )

        agent.invoke(
            {"messages": [{"role": "user", "content": "We've decided we like golang more than python for backend work"}]},
            config=config,
        )
        ```


        If you want to customize the expected schema for memories, you can do so by providing a `schema` argument.
        ```python
        from pydantic import BaseModel


        class UserProfile(BaseModel):
            name: str
            age: int | None = None
            recent_memories: list[str] = []
            preferences: dict | None = None


        memory_tool = create_manage_memory_tool(
            # All memories saved to this tool will live within this namespace
            # The brackets will be populated at runtime by the configurable values
            namespace=("memories", "{langgraph_user_id}", "user_profile"),
            schema=UserProfile,
            actions_permitted=["create", "update"],
            instructions="Update the existing user profile (or create a new one if it doesn't exist) based on the shared information.",
        )
        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )
        agent = create_react_agent(
            "anthropic:claude-3-5-sonnet-latest",
            prompt=prompt,
            tools=[
                memory_tool,
            ],
            store=store,
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "I'm 60 years old and have been programming for 5 days.",
                    }
                ]
            },
            config=config,
        )
        result["messages"][-1].pretty_print()
        # I've created a memory with your age of 60 and noted that you started programming 5 days ago...

        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Just had by 61'st birthday today!!"}
                ]
            },
            config=config,
        )
        result["messages"][-1].pretty_print()
        # Happy 61st birthday! 🎂 I've updated your profile to reflect your new age. Is there anything else I can help you with?
        print(
            store.search(
                ("memories", "123e4567-e89b-12d3-a456-426614174000", "user_profile")
            )
        )
        # [Item(
        #     namespace=['memories', '123e4567-e89b-12d3-a456-426614174000', 'user_profile'],
        #     key='1528553b-0900-4363-8dc2-c6b72844096e',
        #     value={
        # highlight-next-line
        #         'content': UserProfile(
        #             name='User',
        #             age=61,
        #             recent_memories=['Started programming 5 days ago'],
        #             preferences={'programming_experience': '5 days'}
        #         )
        #     },
        #     created_at='2025-02-07T01:12:14.383762+00:00',
        #     updated_at='2025-02-07T01:12:14.383763+00:00',
        #     score=None
        # )]
        ```

        If you want to limit the actions that can be taken by the tool, you can do so by providing a `actions_permitted` argument.

    """
    namespacer = utils.NamespaceTemplate(namespace)
    if not actions_permitted:
        raise ValueError("actions_permitted cannot be empty")
    action_type = typing.Literal[actions_permitted]

    default_action = "create" if "create" in actions_permitted else actions_permitted[0]
    initial_store = store

    async def amanage_memory(
        content: typing.Optional[schema] = None,  # type: ignore
        action: action_type = default_action,  # type: ignore
        *,
        id: typing.Optional[uuid.UUID] = None,
    ):
        store = _get_store(initial_store)
        if action not in actions_permitted:
            raise ValueError(
                f"Invalid action {action}. Must be one of {actions_permitted}."
            )

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
            value={"content": _ensure_json_serializable(content)},
        )
        return f"{action}d memory {id}"

    def manage_memory(
        content: typing.Optional[schema] = None,  # type: ignore
        action: action_type = default_action,  # type: ignore
        *,
        id: typing.Optional[uuid.UUID] = None,
    ):
        store = _get_store(initial_store)
        if action not in actions_permitted:
            raise ValueError(
                f"Invalid action {action}. Must be one of {actions_permitted}."
            )

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
            value={"content": _ensure_json_serializable(content)},
        )
        return f"{action}d memory {id}"

    if len(actions_permitted) == 1:
        verbs = f"{actions_permitted[0]} a memory"
    elif len(actions_permitted) == 2:
        verbs = (
            f"{actions_permitted[0].capitalize()} or {actions_permitted[1]} a memory"
        )
    else:
        prefix_names = ", ".join(
            (actions_permitted[0].capitalize(), *actions_permitted[1:-1])
        )
        verbs = f"{prefix_names}, or {actions_permitted[-1]} a memory"
    description = f"""{verbs} to persist across conversations.
Include the MEMORY ID when updating or deleting a MEMORY. Omit when creating a new MEMORY - it will be created for you.
{instructions}"""

    return StructuredTool.from_function(
        manage_memory, amanage_memory, name=name, description=description
    )


_MEMORY_SEARCH_INSTRUCTIONS = ""


def create_search_memory_tool(
    namespace: tuple[str, ...] | str,
    *,
    instructions: str = _MEMORY_SEARCH_INSTRUCTIONS,
    store: BaseStore | None = None,
    response_format: typing.Literal["content", "content_and_artifact"] = "content",
    name: str = "search_memory",
):
    """Create a tool for searching memories stored in a LangGraph BaseStore.

    This function creates a tool that allows AI assistants to search through previously stored
    memories using semantic or exact matching. The tool returns both the memory contents and
    the raw memory objects for advanced usage.

    Args:
        instructions: Custom instructions for when to use the search tool.
            Defaults to a predefined set of guidelines.
        namespace: The namespace structure for organizing memories in LangGraph's BaseStore.
            Uses runtime configuration with placeholders like `{langgraph_user_id}`.
            See [Memory Namespaces](../concepts/conceptual_guide.md#memory-namespaces).
        store: The BaseStore to use for searching. If not provided, the tool will use the configured BaseStore in your graph or entrypoint.
            Only set if you intend on using these tools outside the LangGraph context.

    Returns:
        search_tool (Tool): A decorated function that can be used as a tool for memory search.
            The tool returns both serialized memories and raw memory objects.

    The resulting tool has a signature that looks like the following:
        ```python
        def search_memory(
            query: str,  # Search query to match against memories
            limit: int = 10,  # Maximum number of results to return
            offset: int = 0,  # Number of results to skip
            filter: dict | None = None,  # Additional filter criteria
        ) -> tuple[list[dict], list]: ...  # Returns (serialized memories, raw memories)
        ```
    _Note: the tool supports both sync and async usage._


    Tip:
        This tool connects with the LangGraph [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) configured in your graph or entrypoint.
        It will not work if you do not provide a store.

    !!! example "Examples"
        ```python
        from langmem import create_search_memory_tool
        from langgraph.func import entrypoint
        from langgraph.store.memory import InMemoryStore

        search_tool = create_search_memory_tool(
            namespace=("project_memories", "{langgraph_user_id}"),
        )

        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )


        @entrypoint(store=store)
        async def workflow(state: dict, *, previous=None):
            # Search for memories about Python
            memories, _ = await search_tool.ainvoke(
                {"query": "Python preferences", "limit": 5}
            )
            print(memories)
            return entrypoint.final(value=memories, save={})
        ```
    """
    namespacer = utils.NamespaceTemplate(namespace)
    initial_store = store

    async def asearch_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        store = _get_store(initial_store)
        namespace = namespacer()
        memories = await store.asearch(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )
        if response_format == "content_and_artifact":
            return utils.dumps([m.dict() for m in memories]), memories
        return utils.dumps([m.dict() for m in memories])

    def search_memory(
        query: str,
        *,
        limit: int = 10,
        offset: int = 0,
        filter: typing.Optional[dict] = None,
    ):
        store = _get_store(initial_store)
        namespace = namespacer()
        memories = store.search(
            namespace,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
        )
        if response_format == "content_and_artifact":
            return utils.dumps([m.dict() for m in memories]), memories
        return utils.dumps([m.dict() for m in memories])

    description = """Search your long-term memories for information relevant to your current context. {instructions}""".format(
        instructions=instructions
    )

    return StructuredTool.from_function(
        search_memory,
        asearch_memory,
        name=name,
        description=description,
        response_format=response_format,
    )


def _get_store(initial_store: BaseStore | None = None) -> BaseStore:
    try:
        if initial_store is not None:
            store = initial_store
        else:
            store = get_store()
        return store
    except RuntimeError as e:
        raise errors.ConfigurationError("Could not get store") from e


def _ensure_json_serializable(content: typing.Any) -> typing.Any:
    # Right now just support primitives and pydantic models
    if isinstance(content, (str, int, float, bool, dict, list)):
        return content
    if hasattr(content, "model_dump"):
        try:
            return content.model_dump(mode="json")
        except Exception as e:
            logger.error(e)
            return str(content)
    return content
