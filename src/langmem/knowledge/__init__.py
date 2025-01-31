"""Knowledge extraction and semantic memory management.

This module provides utilities for extracting and managing semantic knowledge from conversations:

1. Functional Transformations:
    - `create_memory_enricher(model, schemas=None) -> ((messages, existing=None) -> list[tuple[str, Memory]])`:
        Extract structured information from conversations
    - `create_thread_extractor(model, schema=None) -> ((messages) -> Summary)`:
        Generate structured summaries from conversations

2. Stateful Operations:
    Components that persist and manage memories in LangGraph's BaseStore:
    - `create_memory_store_enricher(model, store=None) -> ((messages) -> None)`:
        Apply enrichment with integrated storage
    - `create_manage_memory_tool(store=None) -> Tool[dict, str]`:
        Tool for creating/updating stored memories
    - `create_search_memory_tool(store=None) -> Tool[dict, list[Memory]]`:
        Tool for searching stored memories

"""

import asyncio
import typing
import uuid

import langsmith as ls
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.store.base import SearchItem
from langgraph.utils.config import get_store
from pydantic import BaseModel, Field
from trustcall import create_extractor
from typing_extensions import TypedDict

from langmem import utils
from langmem.knowledge.tools import create_manage_memory_tool, create_search_memory_tool

## LangGraph Tools


def create_thread_extractor(
    model: str,
    schema: typing.Union[None, BaseModel, type] = None,
    instructions: str = "You are tasked with summarizing the following conversation.",
):
    """Creates a conversation thread summarizer using schema-based extraction.

    This function creates an asynchronous callable that takes conversation messages and produces
    a structured summary based on the provided schema. If no schema is provided, it uses a default
    schema with title and summary fields.

    Args:
        model (str): The chat model to use for summarization (name or instance)
        schema (Optional[Union[BaseModel, type]], optional): Pydantic model for structured output.
            Defaults to a simple summary schema with title and summary fields.
        instructions (str, optional): System prompt template for the summarization task.
            Defaults to a basic summarization instruction.

    Returns:
        extractor (Callable[[list], typing.Awaitable[typing.Any]]): Async callable that takes a list of messages and returns a structured summary

    !!! example "Examples"
        ```python
        from langmem import create_thread_extractor

        summarizer = create_thread_extractor("gpt-4")

        messages = [
            {"role": "user", "content": "Hi, I'm having trouble with my account"},
            {
                "role": "assistant",
                "content": "I'd be happy to help. What seems to be the issue?",
            },
            {"role": "user", "content": "I can't reset my password"},
        ]

        summary = await summarizer(messages)
        print(summary.title)
        # Output: "Password Reset Assistance"
        print(summary.summary)
        # Output: "User reported issues with password reset process..."
        ```

    """

    class SummarizeThread(BaseModel):
        """Summarize the thread."""

        title: str
        summary: str

    schema_ = schema or SummarizeThread
    extractor = create_extractor(model, tools=[schema_], tool_choice="any")

    async def summarize_conversation(messages: list[AnyMessage]):
        id_ = str(uuid.uuid4())
        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": f"Summarize the conversation below:\n\n"
                f"<conversation_{id_}>\n{utils.get_conversation}\n</conversation_{id_}>",
            },
        ]
        response = await extractor.ainvoke(messages)
        result = response["responses"][0]
        if isinstance(result, schema_):
            return result
        return result.model_dump(mode="json")

    return summarize_conversation


_MEMORY_INSTRUCTIONS = """You are tasked with extracting or upserting memories for all entities, concepts, etc.

Extract all important facts or entities. If an existing MEMORY is incorrect or outdated, update it based on the new information.
"""


@typing.overload
def create_memory_enricher(
    model: str | BaseChatModel,
    /,
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_deletes: bool = False,
    schemas: None = None,
) -> typing.Callable[
    [list[AnyMessage], typing.Optional[list[str]]], typing.Awaitable[tuple[str, str]]
]: ...


@typing.overload
def create_memory_enricher(
    model: str | BaseChatModel,
    /,
    schemas: list,
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_deletes: bool = False,
) -> typing.Callable[
    [
        list[AnyMessage],
        typing.Optional[
            typing.Union[
                list[tuple[str, BaseModel]],
                list[tuple[str, str, dict]],
            ]
        ],
    ],
    typing.Awaitable[tuple[str, BaseModel]],
]: ...


def create_memory_enricher(  # type: ignore
    model: str | BaseChatModel,
    /,
    schemas: list | None = None,
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_deletes: bool = False,
):
    """Create a memory enricher that processes conversation messages and generates structured memory entries.

    This function creates an async callable that analyzes conversation messages and existing memories
    to generate or update structured memory entries. It can identify implicit preferences,
    important context, and key information from conversations, organizing them into
    well-structured memories that can be used to improve future interactions.

    The enricher supports both unstructured string-based memories and structured memories
    defined by Pydantic models, all automatically persisted to the configured storage.

    !!! example "Examples"
        Basic unstructured memory enrichment:
        ```python
        from langmem import create_memory_enricher

        enricher = create_memory_enricher("anthropic:claude-3-5-sonnet-latest")

        conversation = [
            {"role": "user", "content": "I prefer dark mode in all my apps"},
            {"role": "assistant", "content": "I'll remember that preference"},
        ]

        # Extract memories from conversation
        memories = await enricher(conversation)
        print(memories[0][1])  # First memory's content
        # Output: "User prefers dark mode for all applications"
        ```

        Structured memory enrichment with Pydantic models:
        ```python
        from pydantic import BaseModel
        from langmem import create_memory_enricher

        class PreferenceMemory(BaseModel):
            \"\"\"Store the user's preference\"\"\"
            category: str
            preference: str
            context: str

        enricher = create_memory_enricher(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[PreferenceMemory]
        )

        # Same conversation, but with structured output
        conversation = [
            {"role": "user", "content": "I prefer dark mode in all my apps"},
            {"role": "assistant", "content": "I'll remember that preference"}
        ]
        memories = await enricher(conversation)
        print(memories[0][1])
        # Output:
        # PreferenceMemory(
        #     category="ui",
        #     preference="dark_mode",
        #     context="User explicitly stated preference for dark mode in all applications"
        # )
        ```

        Working with existing memories:
        ```python
        conversation = [
            {
                "role": "user",
                "content": "Actually I changed my mind, dark mode hurts my eyes",
            },
            {"role": "assistant", "content": "I'll update your preference"},
        ]

        # The enricher will upsert; working with the existing memory instead of always creating a new one
        updated_memories = await enricher(conversation, memories)
        ```

    !!! warning
        When using structured memories with Pydantic models, ensure all models are properly
        defined before creating the enricher. Models cannot be modified after the enricher
        is created.

    !!! tip
        For better memory organization:
        1. Use specific, well-defined Pydantic models for different types of memories
        2. Keep memory content concise and focused
        3. Include relevant context in structured memories
        4. Use enable_deletes=True if you want to automatically remove outdated memories

    Args:
        model (Union[str, BaseChatModel]): The language model to use for memory enrichment.
            Can be a model name string or a BaseChatModel instance.
        schemas (Optional[list]): List of Pydantic models defining the structure of memory
            entries. Each model should define the fields and validation rules for a type
            of memory. If None, uses unstructured string-based memories. Defaults to None.
        instructions (str, optional): Custom instructions for memory generation and
            organization. These guide how the model extracts and structures information
            from conversations. Defaults to predefined memory instructions.
        enable_inserts (bool, optional): Whether to allow creating new memory entries.
            When False, the enricher will only update existing memories. Defaults to True.
        enable_deletes (bool, optional): Whether to allow deleting existing memories
            that are outdated or contradicted by new information. Defaults to False.

    Returns:
        enricher (Callable[[list], typing.Awaitable[typing.Any]]): An async function that processes conversations and returns memory entries. The function signature depends on whether schemas are provided:

            - With schemas: (messages: list[Message], existing: Optional[list]) -> list[tuple[str, BaseModel]]
            - Without schemas: (messages: list[Message], existing: Optional[list[str]]) -> list[tuple[str, str]]
    """

    model = model if isinstance(model, BaseChatModel) else init_chat_model(model)
    str_type = False
    if schemas is None:

        class Memory(BaseModel):
            """Call this tool once for each new memory you want to record. Use multi-tool calling to record multiple new memories."""

            content: str = Field(
                description="The memory as a well-written, standalone episode/fact/note/preference/etc."
                " Refer to the user's instructions for more information the prefered memory organization."
            )

        schemas = [Memory]
        str_type = True

    @ls.traceable
    async def enrich_memories(
        messages: list[AnyMessage],
        existing: typing.Optional[
            typing.Union[
                list[str],
                list[tuple[str, BaseModel]],
                list[tuple[str, str, dict]],
            ]
        ] = None,
    ):
        id_ = str(uuid.uuid4())
        session = (
            f"\n\n<session_{id_}>\n{utils.get_conversation(messages)}\n</session_{id_}>"
            if messages
            else ""
        )
        coerced = [
            {"role": "system", "content": "You are a memory subroutine for an AI.\n\n"},
            {
                "role": "user",
                "content": f"{instructions}\n\nEnrich, prune, and organize memories based on any new information."
                " If an existing memory is incorrect or outdated, update it based on the new information. "
                "All operations must be done in single parallel call."
                f"{session}",
            },
        ]
        if str_type and existing and all(isinstance(ex, str) for ex in existing):
            existing = [
                (str(uuid.uuid4()), "Memory", Memory(content=ex)) for ex in existing
            ]
        existing = [
            (
                tuple(e)
                if isinstance(e, (tuple, list)) and len(e) == 3
                else (
                    e[0],
                    e[1].__repr_name__() if isinstance(e[1], BaseModel) else "__any__",
                    e[1],
                )
            )
            for e in (existing or [])
        ]
        extractor = create_extractor(
            model,
            tools=schemas,
            tool_choice="any",
            enable_inserts=enable_inserts,
            enable_deletes=enable_deletes,
            # For now, don't fail on existing schema mismatches
            existing_schema_policy=False,
        )
        response = await extractor.ainvoke({"messages": coerced, "existing": existing})
        results = [
            (rmeta.get("json_doc_id", str(uuid.uuid4())), r)
            for r, rmeta in zip(response["responses"], response["response_metadata"])
        ]
        if existing:
            for id_, _, mem in existing:
                if not any(id_ == id for id, _ in results):
                    results.append((id_, mem))
        return results

    return enrich_memories


def create_memory_searcher(
    model: str | BaseChatModel,
    prompt: str = "You are a memory search assistant.",
    *,
    namespace_prefix: tuple[str, ...] = ("memories", "{user_id}"),
):
    """Creates a memory search pipeline with automatic query generation.

    This function builds a pipeline that combines query generation, memory search,
    and result ranking into a single component. It uses the provided model to
    generate effective search queries based on conversation context.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for search query generation.
            Can be a model name string or a BaseChatModel instance.
        prompt (str, optional): System prompt template for search assistant.
            Defaults to a basic search prompt.
        namespace_prefix (tuple[str, ...], optional): Storage namespace structure for organizing memories.
            Defaults to ("memories", "{user_id}").

    Returns:
        searcher (Callable[[list], typing.Awaitable[typing.Any]]): A pipeline that takes conversation messages and returns sorted memory artifacts,
            ranked by relevance score.

    !!! example "Examples"
        ```python
        from langmem import create_memory_searcher
        from langgraph.store.memory import InMemoryStore
        from langgraph.func import entrypoint

        store = InMemoryStore()
        user_id = "abcd1234"
        store.put(
            ("memories", user_id), key="preferences", value={"content": "I like sushi"}
        )
        searcher = create_memory_searcher(
            "openai:gpt-4o-mini", namespace_prefix=("memories", "{user_id}")
        )


        @entrypoint(store=store)
        async def search_memories(messages: list):
            results = await searcher.ainvoke({"messages": messages})
            print(results[0].value["content"])
            # Output: "I like sushi"


        await search_memories.ainvoke(
            [{"role": "user", "content": "What do I like to eat?"}],
            config={"configurable": {"user_id": user_id}},
        )
        ```

    """
    template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("placeholder", "{messages}"),
            ("user", "Search for memories relevant to the above context."),
        ]
    )
    model = model if isinstance(model, BaseChatModel) else init_chat_model(model)
    search_tool = create_search_memory_tool(namespace_prefix=namespace_prefix)
    query_gen = model.bind_tools(
        [search_tool],
        tool_choice="search_memory",
    )

    def return_sorted(tool_messages: list):
        artifacts = {
            (*item.namespace, item.key): item
            for msg in tool_messages
            for item in (msg.artifact or [])
        }
        return [
            v
            for v in sorted(
                artifacts.values(),
                key=lambda item: item.score if item.score is not None else 0,
                reverse=True,
            )
        ]

    return (
        template
        | utils.merge_message_runs
        | query_gen
        | (lambda msg: [msg])
        | ToolNode([search_tool])
        | return_sorted
    ).with_config({"run_name": "search_memory_pipeline"})


class MemoryPhase(TypedDict, total=False):
    instructions: str
    include_messages: bool
    enable_inserts: bool
    enable_deletes: bool


def create_memory_store_enricher(
    model: str | BaseChatModel,
    /,
    *,
    schemas: list | None = None,
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_deletes: bool = True,
    query_model: str | BaseChatModel | None = None,
    query_limit: int = 5,
    namespace_prefix: tuple[str, ...] = ("memories", "{user_id}"),
    phases: list[MemoryPhase] | None = None,
):
    """End-to-end memory management system with automatic storage integration.

    This function creates a comprehensive memory management system that combines:
    1. Automatic memory search based on conversation context
    2. Memory extraction and enrichment
    3. Persistent storage operations with versioning

    The system automatically searches for relevant memories, extracts new information,
    updates existing memories, and maintains a versioned history of all changes.

    !!! example "Examples"
        Basic memory storage and retrieval:
        ```python
        from langmem import create_memory_store_enricher
        from langgraph.store.memory import InMemoryStore
        from langgraph.func import entrypoint

        store = InMemoryStore()
        enricher = create_memory_store_enricher("anthropic:claude-3-5-sonnet-latest")


        @entrypoint(store=store)
        async def manage_preferences(messages: list):
            # First conversation - storing a preference
            await enricher({"messages": messages})


        # Store a new preference
        await manage_preferences.ainvoke(
            [
                {"role": "user", "content": "I prefer dark mode in all my apps"},
                {"role": "assistant", "content": "I'll remember that preference"},
            ],
            config={"configurable": {"user_id": "user123"}},
        )

        # Later conversation - automatically retrieves and uses the stored preference
        await manage_preferences.ainvoke(
            [
                {"role": "user", "content": "What theme do I prefer?"},
                {
                    "role": "assistant",
                    "content": "You prefer dark mode for all applications",
                },
            ],
            config={"configurable": {"user_id": "user123"}},
        )
        ```

        Structured memory management with custom schemas:
        ```python
        from pydantic import BaseModel
        from langmem import create_memory_store_enricher
        from langgraph.store.memory import InMemoryStore
        from langgraph.func import entrypoint


        class PreferenceMemory(BaseModel):
            \"\"\"Store user preferences.\"\"\"
            category: str
            preference: str
            context: str


        store = InMemoryStore()
        enricher = create_memory_store_enricher(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[PreferenceMemory],
            namespace_prefix=("project", "team_1", "{user_id}"),
        )

        @entrypoint(store=store)
        async def manage_preferences(messages: list):
            await enricher({"messages": messages})

        # Store structured memory
        await manage_preferences.ainvoke(
            [
                {"role": "user", "content": "I prefer dark mode in all my apps"},
                {"role": "assistant", "content": "I'll remember that preference"},
            ],
            config={"configurable": {"user_id": "user123"}},
        )

        # Memory is automatically stored and can be retrieved in future conversations
        # The system will also automatically update it if preferences change
        ```

        Using a separate model for search queries:
        ```python
        from langmem import create_memory_store_enricher
        from langgraph.store.memory import InMemoryStore
        from langgraph.func import entrypoint

        store = InMemoryStore()
        enricher = create_memory_store_enricher(
            "anthropic:claude-3-5-sonnet-latest",  # Main model for memory processing
            query_model="anthropic:claude-3-5-haiku-latest",  # Faster model for search
            query_limit=10,  # Retrieve more relevant memories
        )


        @entrypoint(store=store)
        async def manage_memories(messages: list):
            # The system will use the faster model to search for relevant memories
            # and the more capable model to process and update them
            await enricher({"messages": messages})


        await manage_memories.ainvoke(
            [
                {"role": "user", "content": "What are my preferences?"},
                {
                    "role": "assistant",
                    "content": "Let me check your stored preferences...",
                },
            ],
            config={"configurable": {"user_id": "user123"}},
        )
        ```
    !!! warning
        Memory operations are performed automatically and may modify existing memories.
        If you need to prevent automatic updates, set enable_inserts=False and
        enable_deletes=False.

    !!! tip
        For optimal performance:
        1. Use a smaller, faster model for query_model to improve search speed
        2. Adjust query_limit based on your needs - higher values provide more
           context but may slow down processing
        3. Structure your namespace_prefix to organize memories logically,
           e.g., ("project", "team", "{user_id}")
        4. Consider using enable_deletes=False if you want to maintain
           a history of all memory changes

    Args:
        model (Union[str, BaseChatModel]): The primary language model to use for memory
            enrichment. Can be a model name string or a BaseChatModel instance.
        schemas (Optional[list]): List of Pydantic models defining the structure of memory
            entries. Each model should define the fields and validation rules for a type
            of memory. If None, uses unstructured string-based memories. Defaults to None.
        instructions (str, optional): Custom instructions for memory generation and
            organization. These guide how the model extracts and structures information
            from conversations. Defaults to predefined memory instructions.
        enable_inserts (bool, optional): Whether to allow creating new memory entries.
            When False, the enricher will only update existing memories. Defaults to True.
        enable_deletes (bool, optional): Whether to allow deleting existing memories
            that are outdated or contradicted by new information. Defaults to True.
        query_model (Optional[Union[str, BaseChatModel]], optional): Optional separate
            model for memory search queries. Using a smaller, faster model here can
            improve performance. If None, uses the primary model. Defaults to None.
        query_limit (int, optional): Maximum number of relevant memories to retrieve
            for each conversation. Higher limits provide more context but may slow
            down processing. Defaults to 5.
        namespace_prefix (tuple[str, ...], optional): Storage namespace structure for
            organizing memories. Supports templated values like "{user_id}" which are
            populated from the runtime context. Defaults to ("memories", "{user_id}").

    Returns:
        enricher (Callable): An async function that processes conversations and automatically manages memories in the configured storage. The function works by:

            - Memory search and retrieval
            - Memory creation and updates
            - Storage operations and versioning
    """
    model = model if isinstance(model, BaseChatModel) else init_chat_model(model)
    query_model = (
        model
        if query_model is None
        else (
            query_model
            if isinstance(query_model, BaseChatModel)
            else init_chat_model(query_model)
        )
    )

    first_pass_enricher = create_memory_enricher(
        model,
        schemas=schemas,
        instructions=instructions,
        enable_inserts=enable_inserts,
        enable_deletes=enable_deletes,
    )

    def build_phase_enricher(phase: MemoryPhase):
        return create_memory_enricher(
            model,
            schemas=schemas,
            instructions=phase.get(
                "instructions",
                "You are a memory manager. Deduplicate, consolidate, and enrich these memories.",
            ),
            enable_inserts=phase.get("enable_inserts", True),
            enable_deletes=phase.get("enable_deletes", True),
        )

    def apply_enricher_output(
        enricher_output: list[tuple[str, BaseModel] | tuple[str, dict]],
        store_based: list[tuple[str, str, dict]],
        store_map: dict[str, SearchItem],
        ephemeral: list[tuple[str, str, dict]],
    ):
        store_dict = {
            st_id: (st_id, kind, content) for (st_id, kind, content) in store_based
        }
        ephemeral_dict = {
            st_id: (st_id, kind, content) for (st_id, kind, content) in ephemeral
        }

        removed_ids = []

        for stable_id, model_data in enricher_output:
            if isinstance(model_data, BaseModel):
                if (
                    hasattr(model_data, "__repr_name__")
                    and model_data.__repr_name__() == "RemoveDoc"
                ):
                    removal_id = model_data.json_doc_id
                    if removal_id and removal_id in store_map:
                        print(
                            f"Popping permanent memory {removal_id}",
                            removal_id in store_dict,
                            flush=True,
                        )
                        removed_ids.append(removal_id)
                    else:
                        print(
                            f"Popping ephemeral memory {removal_id}",
                            removal_id in ephemeral,
                            flush=True,
                        )
                    store_dict.pop(removal_id, None)
                    ephemeral_dict.pop(removal_id, None)
                    continue

                new_content = model_data.model_dump(
                    mode="json"
                )  # Could maybe just keep
                new_kind = model_data.__repr_name__()
            else:
                new_kind = store_dict.get(stable_id, (None, "", None))[1]
                new_content = model_data

            if not new_kind:
                new_kind = "Memory"

            if stable_id in store_dict:
                st_id, _, _ = store_dict[stable_id]
                store_dict[stable_id] = (st_id, new_kind, new_content)
            elif stable_id in ephemeral_dict:
                st_id, _, _ = ephemeral_dict[stable_id]
                ephemeral_dict[stable_id] = (st_id, new_kind, new_content)
            else:
                ephemeral_dict[stable_id] = (stable_id, new_kind, new_content)
        return list(store_dict.values()), list(ephemeral_dict.values()), removed_ids

    search_tool = create_search_memory_tool(namespace_prefix=namespace_prefix)
    query_gen = query_model.bind_tools([search_tool], tool_choice="auto")
    namespacer = utils.NamespaceTemplate(namespace_prefix)

    async def manage_memories(messages: list[AnyMessage]):
        store = get_store()
        namespace = namespacer()
        convo = utils.get_conversation(messages)

        # Ask the model which store-based memories might be relevant
        search_req = await query_gen.ainvoke(
            f"Use parallel tool calling to search for distinct memories or aspects that would be relevant to this conversation::\n\n<convo>\n{convo}\n</convo>."
        )
        all_search_results = await asyncio.gather(
            *(
                store.asearch(namespace, **(tc["args"] | {"limit": query_limit}))
                for tc in search_req.tool_calls
            )
        )

        search_results = {}
        for results in all_search_results:
            for it in results:
                search_results[(it.namespace, it.key)] = it

        sorted_results = sorted(
            search_results.values(),
            key=lambda it: it.score if it.score is not None else float("-inf"),
            reverse=True,
        )[:query_limit]

        store_map: dict[str, SearchItem] = {}
        for item in sorted_results:
            stable_id = uuid.uuid5(
                uuid.NAMESPACE_DNS, str((*item.namespace, item.key))
            ).hex
            store_map[stable_id] = item

        store_based = []  # Original items that are found in the store
        for st_id, artifact in store_map.items():
            val = artifact.value
            store_based.append((st_id, val["kind"], val["content"]))

        ephemeral: list[tuple[str, str, dict]] = []
        removed_store_ids: set[str] = set()

        first_pass_result = await first_pass_enricher(messages, existing=store_based)
        store_based, ephemeral, newly_removed = apply_enricher_output(
            first_pass_result, store_based, store_map, ephemeral
        )
        for rid in newly_removed:
            removed_store_ids.add(rid)

        if phases:
            for phase in phases:
                enricher = build_phase_enricher(phase)
                phase_messages = (
                    messages if phase.get("include_messages", False) else []
                )
                phase_result = await enricher(
                    phase_messages, existing=store_based + ephemeral
                )

                store_based, ephemeral, newly_removed = apply_enricher_output(
                    phase_result, store_based, store_map, ephemeral
                )
                for rid in newly_removed:
                    removed_store_ids.add(rid)

        final_mem = store_based + ephemeral

        final_puts = []
        for st_id, kind, content in final_mem:
            if st_id in removed_store_ids:
                continue
            if st_id in store_map:
                old_art = store_map[st_id]
                changed = (
                    old_art.value["kind"] != kind or old_art.value["content"] != content
                )
                if changed:
                    # Updates
                    final_puts.append(
                        {
                            "namespace": old_art.namespace,
                            "key": old_art.key,
                            "value": {
                                "kind": kind,
                                "content": content,
                            },
                        }
                    )
            else:
                # New inserts
                final_puts.append(
                    {
                        "namespace": namespace,
                        "key": st_id,
                        "value": {
                            "kind": kind,
                            "content": content,
                        },
                    }
                )

        final_deletes = []
        for st_id in removed_store_ids:
            if st_id in store_map:
                art = store_map[st_id]
                final_deletes.append((art.namespace, art.key))

        await asyncio.gather(
            *(store.aput(**put) for put in final_puts),
            *(store.adelete(ns, key) for (ns, key) in final_deletes),
        )

        return final_puts

    return manage_memories


__all__ = [
    "create_manage_memory_tool",
    "create_memory_enricher",
    "create_memory_searcher",
    "create_memory_store_enricher",
    "create_thread_extractor",
]
