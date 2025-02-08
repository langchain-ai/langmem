import asyncio
import typing
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import get_executor_for_config
from langgraph.prebuilt import ToolNode
from langgraph.store.base import SearchItem
from langgraph.utils.config import get_store
from pydantic import BaseModel, Field
from trustcall import create_extractor
from typing_extensions import TypedDict

from langmem import utils
from langmem.knowledge.tools import create_search_memory_tool

## LangGraph Tools


class MessagesState(TypedDict):
    messages: list[AnyMessage]


class MemoryState(MessagesState):
    existing: typing.NotRequired[list[BaseModel]]


class SummarizeThread(BaseModel):
    title: str
    summary: str


class ExtractedMemory(typing.NamedTuple):
    id: str
    content: BaseModel


S = typing.TypeVar("S", bound=BaseModel)


class Memory(BaseModel):
    """Call this tool once for each new memory you want to record. Use multi-tool calling to record multiple new memories."""

    content: str = Field(
        description="The memory as a well-written, standalone episode/fact/note/preference/etc."
        " Refer to the user's instructions for more information the prefered memory organization."
    )


@typing.overload
def create_thread_extractor(
    model: str,
    /,
    schema: None = None,
    instructions: str = "You are tasked with summarizing the following conversation.",
) -> Runnable[MessagesState, SummarizeThread]: ...


@typing.overload
def create_thread_extractor(
    model: str,
    /,
    schema: S,
    instructions: str = "You are tasked with summarizing the following conversation.",
) -> Runnable[MessagesState, S]: ...


def create_thread_extractor(
    model: str,
    /,
    schema: typing.Union[None, BaseModel, type] = None,
    instructions: str = "You are tasked with summarizing the following conversation.",
) -> Runnable[MessagesState, BaseModel]:
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

    ???+ example "Examples"
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

        summary = await summarizer.ainvoke({"messages": messages})
        print(summary.title)
        # Output: "Password Reset Assistance"
        print(summary.summary)
        # Output: "User reported issues with password reset process..."
        ```

    """
    if schema is None:
        schema = SummarizeThread

    extractor = create_extractor(model, tools=[schema], tool_choice="any")

    template = ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            (
                "user",
                "Call the provided tool based on the conversation below:\n\n<conversation>{conversation}</conversation>",
            ),
        ]
    )

    def merge_messages(input: dict) -> dict:
        conversation = utils.get_conversation(input["messages"])

        return {"conversation": conversation} | {
            k: v for k, v in input.items() if k != "messages"
        }

    return (
        merge_messages | template | extractor | (lambda out: out["responses"][0])
    ).with_config({"run_name": "thread_extractor"})  # type: ignore


_MEMORY_INSTRUCTIONS = """You are tasked with extracting or upserting memories for all entities, concepts, etc.

Extract all important facts or entities. If an existing MEMORY is incorrect or outdated, update it based on the new information.
"""


class MemoryManager(Runnable[MemoryState, list[ExtractedMemory]]):
    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        schemas: typing.Sequence[typing.Union[BaseModel, type]] = (Memory,),
        instructions: str = (
            "You are tasked with extracting or upserting memories for all entities, concepts, etc.\n\n"
            "Extract all important facts or entities. If an existing MEMORY is incorrect or outdated, "
            "update it based on the new information."
        ),
        enable_inserts: bool = True,
        enable_deletes: bool = False,
    ):
        self.model = (
            model if isinstance(model, BaseChatModel) else init_chat_model(model)
        )
        self.schemas = schemas or (Memory,)
        self.instructions = instructions
        self.enable_inserts = enable_inserts
        self.enable_deletes = enable_deletes

    def _prepare_messages(self, messages: list[AnyMessage]) -> list[dict]:
        id_ = str(uuid.uuid4())
        session = (
            f"\n\n<session_{id_}>\n{utils.get_conversation(messages)}\n</session_{id_}>"
        )
        return [
            {"role": "system", "content": "You are a memory subroutine for an AI.\n\n"},
            {
                "role": "user",
                "content": (
                    f"{self.instructions}\n\nEnrich, prune, and organize memories based on any new information. "
                    f"If an existing memory is incorrect or outdated, update it based on the new information. "
                    f"All operations must be done in single parallel call.{session}"
                ),
            },
        ]

    def _prepare_existing(
        self,
        existing: typing.Optional[
            typing.Union[
                list[str], list[tuple[str, BaseModel]], list[tuple[str, str, dict]]
            ]
        ],
    ) -> list[tuple[str, str, typing.Any]]:
        if existing is None:
            return []
        if all(isinstance(ex, str) for ex in existing):
            MemoryModel = self.schemas[0]
            return [
                (str(uuid.uuid4()), "Memory", MemoryModel(content=ex))
                for ex in existing
            ]
        result = []
        for e in existing:
            if isinstance(e, (tuple, list)) and len(e) == 3:
                result.append(tuple(e))
            else:
                # Assume a two-element tuple: (id, value)
                id_, value = e[0], e[1]
                kind = (
                    value.__repr_name__() if isinstance(value, BaseModel) else "__any__"
                )
                result.append((id_, kind, value))
        return result

    async def ainvoke(
        self,
        input: MemoryState,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[ExtractedMemory]:
        messages = input["messages"]
        existing = input.get("existing")
        prepared_messages = self._prepare_messages(messages)
        prepared_existing = self._prepare_existing(existing)
        extractor = create_extractor(
            self.model,
            tools=self.schemas,
            tool_choice="any",
            enable_inserts=self.enable_inserts,
            enable_deletes=self.enable_deletes,
            existing_schema_policy=False,
        )
        payload = {"messages": prepared_messages, "existing": prepared_existing}
        response = await extractor.ainvoke(payload)
        results = [
            (rmeta.get("json_doc_id", str(uuid.uuid4())), r)
            for r, rmeta in zip(response["responses"], response["response_metadata"])
        ]
        # Merge in any existing memories not updated.
        for mem_tuple in prepared_existing:
            mem_id, _, mem = mem_tuple
            if not any(mem_id == rid for rid, _ in results):
                results.append((mem_id, mem))
        return [ExtractedMemory(id=rid, content=content) for rid, content in results]

    def invoke(
        self,
        input: MemoryState,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[ExtractedMemory]:
        messages = input["messages"]
        existing = input.get("existing")
        prepared_messages = self._prepare_messages(messages)
        prepared_existing = self._prepare_existing(existing)
        extractor = create_extractor(
            self.model,
            tools=list(self.schemas),
            tool_choice="any",
            enable_inserts=self.enable_inserts,
            enable_deletes=self.enable_deletes,
            existing_schema_policy=False,
        )
        payload = {"messages": prepared_messages, "existing": prepared_existing}
        response = extractor.invoke(payload)
        results = [
            (rmeta.get("json_doc_id", str(uuid.uuid4())), r)
            for r, rmeta in zip(response["responses"], response["response_metadata"])
        ]
        for mem_tuple in prepared_existing:
            mem_id, _, mem = mem_tuple
            if not any(mem_id == rid for rid, _ in results):
                results.append((mem_id, mem))
        return [ExtractedMemory(id=rid, content=content) for rid, content in results]

    async def __call__(
        self,
        messages: typing.Sequence[AnyMessage],
        existing: typing.Optional[typing.Sequence[ExtractedMemory]] = None,
    ) -> list[ExtractedMemory]:
        input: MemoryState = {"messages": messages}
        if existing is not None:
            input["existing"] = existing
        return await self.ainvoke(input)


def create_memory_manager(
    model: str | BaseChatModel,
    /,
    schemas: typing.Sequence[typing.Union[BaseModel, type]] = (Memory,),
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_deletes: bool = False,
) -> Runnable[MemoryState, list[ExtractedMemory]]:
    """Create a memory manager that processes conversation messages and generates structured memory entries.

    This function creates an async callable that analyzes conversation messages and existing memories
    to generate or update structured memory entries. It can identify implicit preferences,
    important context, and key information from conversations, organizing them into
    well-structured memories that can be used to improve future interactions.

    The manager supports both unstructured string-based memories and structured memories
    defined by Pydantic models, all automatically persisted to the configured storage.

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
            When False, the manager will only update existing memories. Defaults to True.
        enable_deletes (bool, optional): Whether to allow deleting existing memories
            that are outdated or contradicted by new information. Defaults to False.

    Returns:
        manager: An runnable that processes conversations and returns `ExtractedMemory`'s. The function signature depends on whether schemas are provided

    ???+ example "Examples"
        Basic unstructured memory enrichment:
        ```python
        from langmem import create_memory_manager

        manager = create_memory_manager("anthropic:claude-3-5-sonnet-latest")

        conversation = [
            {"role": "user", "content": "I prefer dark mode in all my apps"},
            {"role": "assistant", "content": "I'll remember that preference"},
        ]

        # Extract memories from conversation
        memories = await manager(conversation)
        print(memories[0][1])  # First memory's content
        # Output: "User prefers dark mode for all applications"
        ```

        Structured memory enrichment with Pydantic models:
        ```python
        from pydantic import BaseModel
        from langmem import create_memory_manager

        class PreferenceMemory(BaseModel):
            \"\"\"Store the user's preference\"\"\"
            category: str
            preference: str
            context: str

        manager = create_memory_manager(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[PreferenceMemory]
        )

        # Same conversation, but with structured output
        conversation = [
            {"role": "user", "content": "I prefer dark mode in all my apps"},
            {"role": "assistant", "content": "I'll remember that preference"}
        ]
        memories = await manager(conversation)
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

        # The manager will upsert; working with the existing memory instead of always creating a new one
        updated_memories = await manager(conversation, memories)
        ```
    """

    return MemoryManager(
        model,
        schemas=schemas,
        instructions=instructions,
        enable_inserts=enable_inserts,
        enable_deletes=enable_deletes,
    )


def create_memory_searcher(
    model: str | BaseChatModel,
    prompt: str = "Search for distinct memories relevant to different aspects of the provided context.",
    *,
    namespace: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
) -> Runnable[MessagesState, typing.Awaitable[list[SearchItem]]]:
    """Creates a memory search pipeline with automatic query generation.

    This function builds a pipeline that combines query generation, memory search,
    and result ranking into a single component. It uses the provided model to
    generate effective search queries based on conversation context.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for search query generation.
            Can be a model name string or a BaseChatModel instance.
        prompt (str, optional): System prompt template for search assistant.
            Defaults to a basic search prompt.
        namespace: The namespace structure for organizing memories in LangGraph's BaseStore.
            Uses runtime configuration with placeholders like `{langgraph_user_id}`.
            See [Memory Namespaces](../concepts/conceptual_guide.md#memory-namespaces).
            Defaults to ("memories", "{langgraph_user_id}").

    ???+ note "Namespace Configuration"
        If the namespae has template variables "{variable_name}", they will be configured at
        runtime through the `config` parameter:
        ```python
        # Example: Search user's memories
        config = {"configurable": {"langgraph_user_id": "user-123"}}
        # Searches in namespace: ("memories", "user-123")

        # Example: Search team knowledge
        config = {"configurable": {"langgraph_user_id": "team-x"}}
        # Searches in namespace: ("memories", "team-x")
        ```

    Returns:
        searcher (Callable[[list], typing.Awaitable[typing.Any]]): A pipeline that takes conversation messages and returns sorted memory artifacts,
            ranked by relevance score.

    ???+ example "Examples"
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
            "openai:gpt-4o-mini", namespace=("memories", "{langgraph_user_id}")
        )


        @entrypoint(store=store)
        async def search_memories(messages: list):
            results = await searcher.ainvoke({"messages": messages})
            print(results[0].value["content"])
            # Output: "I like sushi"


        await search_memories.ainvoke(
            [{"role": "user", "content": "What do I like to eat?"}],
            config={"configurable": {"langgraph_user_id": user_id}},
        )
        ```

    """
    template = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("placeholder", "{messages}"),
            ("user", "\n\nSearch for memories relevant to the above context."),
        ]
    )

    # Initialize model and search tool
    model_instance = (
        model if isinstance(model, BaseChatModel) else init_chat_model(model)
    )
    search_tool = create_search_memory_tool(
        namespace=namespace, response_format="content_and_artifact"
    )
    query_gen = model_instance.bind_tools([search_tool], tool_choice="search_memory")

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

    return (  # type: ignore
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


class MemoryStoreManagerInput(TypedDict):
    """Input schema for MemoryStoreManager."""

    messages: list[AnyMessage]


class MemoryStoreManager(Runnable[MemoryStoreManagerInput, list[dict]]):
    def __init__(
        self,
        model: str | BaseChatModel,
        /,
        *,
        schemas: list | None = None,
        instructions: str = (
            "You are tasked with extracting or upserting memories for all entities, concepts, etc.\n\n"
            "Extract all important facts or entities. If an existing MEMORY is incorrect or outdated, update it based on the new information."
        ),
        enable_inserts: bool = True,
        enable_deletes: bool = True,
        query_model: str | BaseChatModel | None = None,
        query_limit: int = 20,
        namespace: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
        phases: list[MemoryPhase] | None = None,
    ):
        self.model = (
            model if isinstance(model, BaseChatModel) else init_chat_model(model)
        )
        self.query_model = (
            None
            if query_model is None
            else (
                query_model
                if isinstance(query_model, BaseChatModel)
                else init_chat_model(query_model)
            )
        )
        self.schemas = schemas
        self.instructions = instructions
        self.enable_inserts = enable_inserts
        self.enable_deletes = enable_deletes
        self.query_limit = query_limit
        self.phases = phases or []
        self.namespace = utils.NamespaceTemplate(namespace)

        self.memory_manager = create_memory_manager(
            self.model,
            schemas=schemas,
            instructions=instructions,
            enable_inserts=enable_inserts,
            enable_deletes=enable_deletes,
        )
        self.search_tool = create_search_memory_tool(
            namespace=namespace,
            instructions="Queries should be formatted as hypothetical memories that would be relevant to the current conversation.",
        )
        self.query_gen = None
        if self.query_model is not None:
            self.query_gen = self.query_model.bind_tools(
                [self.search_tool], tool_choice="any"
            )

    @staticmethod
    def _stable_id(item: SearchItem) -> str:
        return uuid.uuid5(uuid.NAMESPACE_DNS, str((*item.namespace, item.key))).hex

    @staticmethod
    def _apply_manager_output(
        manager_output: list[ExtractedMemory],
        store_based: list[tuple[str, str, dict]],
        store_map: dict[str, SearchItem],
        ephemeral: list[tuple[str, str, dict]],
    ) -> tuple[list[tuple[str, str, dict]], list[tuple[str, str, dict]], list[str]]:
        store_dict = {sid: (sid, kind, content) for (sid, kind, content) in store_based}
        ephemeral_dict = {
            sid: (sid, kind, content) for (sid, kind, content) in ephemeral
        }
        removed_ids = []
        for extracted in manager_output:
            stable_id = extracted.id
            model_data = extracted.content
            if isinstance(model_data, BaseModel):
                if (
                    hasattr(model_data, "__repr_name__")
                    and model_data.__repr_name__() == "RemoveDoc"
                ):
                    removal_id = getattr(model_data, "json_doc_id", None)
                    if removal_id and removal_id in store_map:
                        removed_ids.append(removal_id)
                    store_dict.pop(removal_id, None)
                    ephemeral_dict.pop(removal_id, None)
                    continue
                new_content = model_data.model_dump(mode="json")
                new_kind = model_data.__repr_name__()
            else:
                new_kind = store_dict.get(stable_id, (stable_id, "Memory", {}))[1]
                new_content = model_data
            if stable_id in store_dict:
                store_dict[stable_id] = (stable_id, new_kind, new_content)
            else:
                ephemeral_dict[stable_id] = (stable_id, new_kind, new_content)
        return list(store_dict.values()), list(ephemeral_dict.values()), removed_ids

    def _build_phase_manager(
        self, phase: MemoryPhase
    ) -> Runnable[MessagesState, list[ExtractedMemory]]:
        return create_memory_manager(
            self.model,
            schemas=self.schemas,
            instructions=phase.get(
                "instructions",
                "You are a memory manager. Deduplicate, consolidate, and enrich these memories.",
            ),
            enable_inserts=phase.get("enable_inserts", True),
            enable_deletes=phase.get("enable_deletes", True),
        )

    @staticmethod
    def _sort_results(
        search_results_lists: list[list[SearchItem]], query_limit: int
    ) -> dict[str, SearchItem]:
        search_results = {}
        for results in search_results_lists:
            for item in results:
                search_results[(tuple(item.namespace), item.key)] = item
        sorted_results = sorted(
            search_results.values(),
            key=lambda it: it.score if it.score is not None else float("-inf"),
            reverse=True,
        )[:query_limit]
        return {MemoryStoreManager._stable_id(item): item for item in sorted_results}

    async def ainvoke(
        self,
        input: MemoryStoreManagerInput,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[dict]:
        store = get_store()
        namespace = self.namespace(config)

        if self.query_gen:
            convo = utils.get_conversation(input["messages"])
            query_text = (
                f"Use parallel tool calling to search for distinct memories relevant to this conversation.:\n\n"
                f"<convo>\n{convo}\n</convo>."
            )
            query_req = await self.query_gen.ainvoke(query_text)
            search_results_lists = await asyncio.gather(
                *[
                    store.asearch(
                        namespace, **({**tc["args"], "limit": self.query_limit})
                    )
                    for tc in query_req.tool_calls
                ]
            )
        else:
            # Search over "query_limit" timespans starting from the most recent
            queries = utils.get_dialated_windows(
                input["messages"], self.query_limit // 4
            )
            search_results_lists = await asyncio.gather(
                *[store.asearch(namespace, query=query) for query in queries]
            )

        store_map = self._sort_results(search_results_lists, self.query_limit)

        store_based = [
            (sid, item.value["kind"], item.value["content"])
            for sid, item in store_map.items()
        ]
        ephemeral: list[tuple[str, str, dict]] = []
        removed_ids: set[str] = set()

        # --- Enrich memories using the composed MemoryManager (async) ---
        enriched = await self.memory_manager.ainvoke(
            {"messages": input["messages"], "existing": store_based}
        )
        store_based, ephemeral, removed = self._apply_manager_output(
            enriched, store_based, store_map, ephemeral
        )
        removed_ids.update(removed)

        # Process additional phases.
        for phase in self.phases:
            phase_manager = self._build_phase_manager(phase)
            phase_messages = (
                input["messages"] if phase.get("include_messages", False) else []
            )
            phase_input = {
                "messages": phase_messages,
                "existing": store_based + ephemeral,
            }
            phase_enriched = await phase_manager.ainvoke(phase_input)
            store_based, ephemeral, removed = self._apply_manager_output(
                phase_enriched, store_based, store_map, ephemeral
            )
            removed_ids.update(removed)

        final_mem = store_based + ephemeral
        final_puts = []
        for sid, kind, content in final_mem:
            if sid in removed_ids:
                continue
            if sid in store_map:
                old_art = store_map[sid]
                if old_art.value["kind"] != kind or old_art.value["content"] != content:
                    final_puts.append(
                        {
                            "namespace": old_art.namespace,
                            "key": old_art.key,
                            "value": {"kind": kind, "content": content},
                        }
                    )
            else:
                final_puts.append(
                    {
                        "namespace": namespace,
                        "key": sid,
                        "value": {"kind": kind, "content": content},
                    }
                )

        final_deletes = []
        for sid in removed_ids:
            if sid in store_map:
                art = store_map[sid]
                final_deletes.append((art.namespace, art.key))

        await asyncio.gather(
            *(store.aput(**put) for put in final_puts),
            *(store.adelete(ns, key) for (ns, key) in final_deletes),
        )

        return final_puts

    def invoke(
        self,
        input: MemoryStoreManagerInput,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[dict]:
        store = get_store()
        namespace = self.namespace(config)
        convo = utils.get_conversation(input["messages"])

        with get_executor_for_config(config) as executor:
            if self.query_gen:
                convo = utils.get_conversation(input["messages"])
                query_text = (
                    f"Use parallel tool calling to search for distinct memories relevant to this conversation.:\n\n"
                    f"<convo>\n{convo}\n</convo>."
                )
                query_req = self.query_gen.invoke(query_text)
                search_results_futs = [
                    executor.submit(
                        store.search,
                        namespace,
                        **({**tc["args"], "limit": self.query_limit}),
                    )
                    for tc in query_req.tool_calls
                ]
            else:
                # Search over "query_limit" timespans starting from the most recent
                queries = utils.get_dialated_windows(
                    input["messages"], self.query_limit // 4
                )
                search_results_lists = [
                    store.search(namespace, query=query) for query in queries
                ]
                search_results_futs = [
                    executor.submit(
                        store.search,
                        namespace,
                        query=query,
                        limit=self.query_limit,
                    )
                    for query in queries
                ]

        search_results_lists = [fut.result() for fut in search_results_futs]
        store_map = self._sort_results(search_results_lists, self.query_limit)
        store_based = [
            (sid, item.value["kind"], item.value["content"])
            for sid, item in store_map.items()
        ]
        ephemeral: list[tuple[str, str, dict]] = []
        removed_ids: set[str] = set()

        enriched = self.memory_manager.invoke(
            {"messages": input["messages"], "existing": store_based}
        )
        store_based, ephemeral, removed = self._apply_manager_output(
            enriched, store_based, store_map, ephemeral
        )
        removed_ids.update(removed)

        for phase in self.phases:
            phase_manager = self._build_phase_manager(phase)
            phase_messages = (
                input["messages"] if phase.get("include_messages", False) else []
            )
            phase_input = {
                "messages": phase_messages,
                "existing": store_based + ephemeral,
            }
            phase_enriched = phase_manager.invoke(phase_input)
            store_based, ephemeral, removed = self._apply_manager_output(
                phase_enriched, store_based, store_map, ephemeral
            )
            removed_ids.update(removed)

        final_mem = store_based + ephemeral
        final_puts = []
        for sid, kind, content in final_mem:
            if sid in removed_ids:
                continue
            if sid in store_map:
                old_art = store_map[sid]
                if old_art.value["kind"] != kind or old_art.value["content"] != content:
                    final_puts.append(
                        {
                            "namespace": old_art.namespace,
                            "key": old_art.key,
                            "value": {"kind": kind, "content": content},
                        }
                    )
            else:
                final_puts.append(
                    {
                        "namespace": namespace,
                        "key": sid,
                        "value": {"kind": kind, "content": content},
                    }
                )

        final_deletes = []
        for sid in removed_ids:
            if sid in store_map:
                art = store_map[sid]
                final_deletes.append((art.namespace, art.key))

        with get_executor_for_config(config) as executor:
            for put in final_puts:
                executor.submit(store.put, **put)
            for ns, key in final_deletes:
                executor.submit(store.delete, ns, key)

        return final_puts

    async def __call__(self, messages: typing.Sequence[AnyMessage]) -> list[dict]:
        return await self.ainvoke({"messages": messages})


def create_memory_store_manager(
    model: str | BaseChatModel,
    /,
    *,
    schemas: list | None = None,
    instructions: str = (
        "You are tasked with extracting or upserting memories for all entities, concepts, etc.\n\n"
        "Extract all important facts or entities. If an existing MEMORY is incorrect or outdated, update it based on the new information."
    ),
    enable_inserts: bool = True,
    enable_deletes: bool = True,
    query_model: str | BaseChatModel | None = None,
    query_limit: int = 5,
    namespace: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
    phases: list[MemoryPhase] | None = None,
) -> MemoryStoreManager:
    """Enriches memories stored in the configured BaseStore.

    The system automatically searches for relevant memories, extracts new information,
    updates existing memories, and maintains a versioned history of all changes.

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
            When False, the manager will only update existing memories. Defaults to True.
        enable_deletes (bool, optional): Whether to allow deleting existing memories
            that are outdated or contradicted by new information. Defaults to True.
        query_model (Optional[Union[str, BaseChatModel]], optional): Optional separate
            model for memory search queries. Using a smaller, faster model here can
            improve performance. If None, uses the primary model. Defaults to None.
        query_limit (int, optional): Maximum number of relevant memories to retrieve
            for each conversation. Higher limits provide more context but may slow
            down processing. Defaults to 5.
        namespace (tuple[str, ...], optional): Storage namespace structure for
            organizing memories. Supports templated values like "{langgraph_user_id}" which are
            populated from the runtime context. Defaults to `("memories", "{langgraph_user_id}")`.

    Returns:
        manager: An runnable that processes conversations and automatically manages memories in the LangGraph BaseStore.

    The basic data flow works as follows:

    ```mermaid
    sequenceDiagram
    participant Client
    participant Manager
    participant Store
    participant LLM

    Client->>Manager: conversation history
    Manager->>Store: find similar memories
    Store-->>Manager: memories
    Manager->>LLM: analyze & extract
    LLM-->>Manager: memory updates
    Manager->>Store: apply changes
    Manager-->>Client: updated memories
    ```

    ???+ example "Examples"
        Run memory extraction "inline" within your LangGraph app.
        By default, each "memory" is a simple string:
        ```python
        import os

        from anthropic import AsyncAnthropic
        from langchain_core.runnables import RunnableConfig
        from langgraph.func import entrypoint
        from langgraph.store.memory import InMemoryStore

        from langmem import create_memory_store_manager

        store = InMemoryStore()
        manager = create_memory_store_manager("anthropic:claude-3-5-sonnet-latest", namespace=("memories", "{langgraph_user_id}"))
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


        @entrypoint(store=store)
        async def my_agent(message: str, config: RunnableConfig):
            memories = await store.asearch(
                ("memories", config["configurable"]["langgraph_user_id"]),
                query=message,
            )
            llm_response = await client.messages.create(
                model="claude-3-5-sonnet-latest",
                system="You are a helpful assistant.\\n\\n## Memories from the user:"
                f"\\n<memories>\\n{memories}\\n</memories>",
                max_tokens=2048,
                messages=[{"role": "user", "content": message}],
            )
            response = {"role": "assistant", "content": llm_response.content[0].text}

            await manager.ainvoke(
                {"messages": [{"role": "user", "content": message}, response]},
            )
            return response["content"]


        response_1 = await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config={"configurable": {"langgraph_user_id": "user123"}},
        )
        print("response_1:", response_1)
        # Later conversation - automatically retrieves and uses the stored preference
        response_2 = await my_agent.ainvoke(
            "What theme do I prefer?",
            config={"configurable": {"langgraph_user_id": "user123"}},
        )
        print("response_2:", response_2)
        # You can list over memories in the user's namespace manually:
        print(store.search(("memories", "user123")))
        ```

        You can customize what each memory can look like by defining **schemas**:
        ```python
        from langgraph.func import entrypoint
        from langgraph.store.memory import InMemoryStore
        from pydantic import BaseModel

        from langmem import create_memory_store_manager

        store = InMemoryStore()
        manager = create_memory_store_manager(
            "anthropic:claude-3-5-sonnet-latest",
            namespace=("memories", "{langgraph_user_id}"),
        )

        class PreferenceMemory(BaseModel):
            \"\"\"Store preferences about the user.\"\"\"
            category: str
            preference: str
            context: str


        store = InMemoryStore()
        manager = create_memory_store_manager(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[PreferenceMemory],
            namespace=("project", "team_1", "{langgraph_user_id}"),
        )


        @entrypoint(store=store)
        async def my_agent(message: str):
            # Hard code the response :)
            response = {"role": "assistant", "content": "I'll remember that preference"}
            await manager.ainvoke(
                {"messages": [{"role": "user", "content": message}, response]}
            )
            return response


        # Store structured memory
        await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config={"configurable": {"langgraph_user_id": "user123"}},
        )

        # See the extracted memories yourself
        print(store.search(("memories", "user123")))

        # Memory is automatically stored and can be retrieved in future conversations
        # The system will also automatically update it if preferences change
        ```

    By default, relevant memories are recalled by directly embedding the new messages. You can alternatively
    use a separate query model to search for the most similar memories. Here's how it works:

    ```mermaid
        sequenceDiagram
            participant Client
            participant Manager
            participant QueryLLM
            participant Store
            participant MainLLM

            Client->>Manager: messages
            Manager->>QueryLLM: generate search query
            QueryLLM-->>Manager: optimized query
            Manager->>Store: find memories
            Store-->>Manager: memories
            Manager->>MainLLM: analyze & extract
            MainLLM-->>Manager: memory updates
            Manager->>Store: apply changes
            Manager-->>Client: result
    ```

    ???+ example "Using an LLM to search for memories"
        ```python
        from langmem import create_memory_store_manager
        from langgraph.store.memory import InMemoryStore
        from langgraph.func import entrypoint

        store = InMemoryStore()
        manager = create_memory_store_manager(
            "anthropic:claude-3-5-sonnet-latest",  # Main model for memory processing
            query_model="anthropic:claude-3-5-haiku-latest",  # Faster model for search
            query_limit=10,  # Retrieve more relevant memories
            namespace=("memories", "{langgraph_user_id}"),
        )


        @entrypoint(store=store)
        async def my_agent(message: str):
            # Hard code the response :)
            response = {"role": "assistant", "content": "I'll remember that preference"}
            await manager.ainvoke(
                {"messages": [{"role": "user", "content": message}, response]}
            )
            return response


        await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config={"configurable": {"langgraph_user_id": "user123"}},
        )

        # See the extracted memories yourself
        print(store.search(("memories", "user123")))
        ```

    In the examples above, we were calling the manager in the main thread. In a real application, you'll
    likely want to background the execution of the manager, either by executing it in a background thread or on a separate server.
    To do so, you can use the `ReflectionExecutor` class:

    ```mermaid
    sequenceDiagram
        participant Agent
        participant Background
        participant Store

        Agent->>Agent: process message
        Agent-->>User: response
        Agent->>Background: schedule enrichment<br/>(after_seconds=0)
        Note over Background,Store: Memory processing happens<br/>in background thread
    ```

    ???+ example "Running reflections in the background"
        Background enrichment using @entrypoint:
        ```python
        from langmem import create_memory_store_manager, ReflectionExecutor
        from langgraph.prebuilt import create_react_agent
        from langgraph.store.memory import InMemoryStore
        from langgraph.func import entrypoint

        store = InMemoryStore()
        manager = create_memory_store_manager(
            "anthropic:claude-3-5-sonnet-latest", namespace=("memories", "{user_id}")
        )
        reflection = ReflectionExecutor(manager, store=store)
        agent = create_react_agent(
            "anthropic:claude-3-5-sonnet-latest", tools=[], store=store
        )


        @entrypoint(store=store)
        async def chat(messages: list):
            response = await agent.ainvoke({"messages": messages})

            fut = reflection.submit(
                {
                    "messages": response["messages"],
                },
                # We'll schedule this immediately.
                # Adding a delay lets you **debounce** and deduplicate reflection work
                # whenever the user is actively engaging with the agent.
                after_seconds=0,
            )

            return fut


        fut = await chat.ainvoke(
            [{"role": "user", "content": "I prefer dark mode in my apps"}],
            config={"configurable": {"user_id": "user-123"}},
        )
        # Inspect the result
        fut.result()  # Wait for the reflection to complete; This is only for demoing the search inline
        print(store.search(("memories", "user-123")))
        ```
    """
    return MemoryStoreManager(
        model,
        schemas=schemas,
        instructions=instructions,
        enable_inserts=enable_inserts,
        enable_deletes=enable_deletes,
        query_model=query_model,
        query_limit=query_limit,
        namespace=namespace,
        phases=phases,
    )


__all__ = [
    "create_memory_manager",
    "create_memory_searcher",
    "create_memory_store_manager",
    "create_thread_extractor",
]
