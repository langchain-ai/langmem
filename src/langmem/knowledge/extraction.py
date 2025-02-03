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


class MemoryEnricher(Runnable[MemoryState, list[ExtractedMemory]]):
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


def create_memory_enricher(
    model: str | BaseChatModel,
    /,
    schemas: typing.Sequence[typing.Union[BaseModel, type]] = (Memory,),
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_deletes: bool = False,
) -> Runnable[MemoryState, list[ExtractedMemory]]:
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

    return MemoryEnricher(
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
    namespace_prefix: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
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
        namespace_prefix (tuple[str, ...], optional): Storage namespace structure for organizing memories.
            Defaults to ("memories", "{langgraph_user_id}").

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
            "openai:gpt-4o-mini", namespace_prefix=("memories", "{langgraph_user_id}")
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
    search_tool = create_search_memory_tool(namespace_prefix=namespace_prefix)
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


class MemoryStoreEnricherInput(BaseModel):
    """Input schema for MemoryStoreEnricher."""

    messages: list[AnyMessage]


class MemoryStoreEnricher(Runnable[MemoryStoreEnricherInput, list[dict]]):
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
        query_limit: int = 5,
        namespace_prefix: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
        phases: list[MemoryPhase] | None = None,
    ):
        self.model = (
            model if isinstance(model, BaseChatModel) else init_chat_model(model)
        )
        self.query_model = (
            self.model
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
        self.namespacer = utils.NamespaceTemplate(namespace_prefix)

        self.memory_enricher = create_memory_enricher(
            self.model,
            schemas=schemas,
            instructions=instructions,
            enable_inserts=enable_inserts,
            enable_deletes=enable_deletes,
        )
        self.search_tool = create_search_memory_tool(namespace_prefix=namespace_prefix)
        self.query_gen = self.query_model.bind_tools(
            [self.search_tool], tool_choice="any"
        )

    @staticmethod
    def _stable_id(item: SearchItem) -> str:
        return uuid.uuid5(uuid.NAMESPACE_DNS, str((*item.namespace, item.key))).hex

    @staticmethod
    def _apply_enricher_output(
        enricher_output: list[ExtractedMemory],
        store_based: list[tuple[str, str, dict]],
        store_map: dict[str, SearchItem],
        ephemeral: list[tuple[str, str, dict]],
    ) -> tuple[list[tuple[str, str, dict]], list[tuple[str, str, dict]], list[str]]:
        store_dict = {sid: (sid, kind, content) for (sid, kind, content) in store_based}
        ephemeral_dict = {
            sid: (sid, kind, content) for (sid, kind, content) in ephemeral
        }
        removed_ids = []
        for extracted in enricher_output:
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

    def _build_phase_enricher(
        self, phase: MemoryPhase
    ) -> Runnable[MessagesState, list[ExtractedMemory]]:
        return create_memory_enricher(
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
        return {MemoryStoreEnricher._stable_id(item): item for item in sorted_results}

    async def ainvoke(
        self,
        input: MemoryStoreEnricherInput,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[dict]:
        store = get_store()
        namespace = self.namespacer(config)
        convo = utils.get_conversation(input["messages"])

        query_text = (
            f"Use parallel tool calling to search for distinct memories relevant to this conversation:\n\n"
            f"<convo>\n{convo}\n</convo>."
        )
        query_req = await self.query_gen.ainvoke(query_text)
        search_results_lists = await asyncio.gather(
            *[
                store.asearch(namespace, **({**tc["args"], "limit": self.query_limit}))
                for tc in query_req.tool_calls
            ]
        )
        store_map = self._sort_results(search_results_lists, self.query_limit)

        store_based = [
            (sid, item.value["kind"], item.value["content"])
            for sid, item in store_map.items()
        ]
        ephemeral: list[tuple[str, str, dict]] = []
        removed_ids: set[str] = set()

        # --- Enrich memories using the composed MemoryEnricher (async) ---
        enriched = await self.memory_enricher.ainvoke(
            {"messages": input["messages"], "existing": store_based}
        )
        store_based, ephemeral, removed = self._apply_enricher_output(
            enriched, store_based, store_map, ephemeral
        )
        removed_ids.update(removed)

        # Process additional phases.
        for phase in self.phases:
            phase_enricher = self._build_phase_enricher(phase)
            phase_messages = (
                input["messages"] if phase.get("include_messages", False) else []
            )
            phase_input = {
                "messages": phase_messages,
                "existing": store_based + ephemeral,
            }
            phase_enriched = await phase_enricher.ainvoke(phase_input)
            store_based, ephemeral, removed = self._apply_enricher_output(
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
        input: MemoryStoreEnricherInput,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[dict]:
        store = get_store()
        namespace = self.namespacer(config)
        convo = utils.get_conversation(input["messages"])

        query_text = (
            f"Use parallel tool calling to search for distinct memories relevant to this conversation:\n\n"
            f"<convo>\n{convo}\n</convo>."
        )
        query_req = self.query_gen.invoke(query_text)
        with get_executor_for_config(config) as executor:
            search_results_futs = [
                executor.submit(
                    store.search,
                    namespace,
                    **({**tc["args"], "limit": self.query_limit}),
                )
                for tc in query_req.tool_calls
            ]
        search_results_lists = [fut.result() for fut in search_results_futs]
        store_map = self._sort_results(search_results_lists, self.query_limit)
        store_based = [
            (sid, item.value["kind"], item.value["content"])
            for sid, item in store_map.items()
        ]
        ephemeral: list[tuple[str, str, dict]] = []
        removed_ids: set[str] = set()

        enriched = self.memory_enricher.invoke(
            {"messages": input["messages"], "existing": store_based}
        )
        store_based, ephemeral, removed = self._apply_enricher_output(
            enriched, store_based, store_map, ephemeral
        )
        removed_ids.update(removed)

        for phase in self.phases:
            phase_enricher = self._build_phase_enricher(phase)
            phase_messages = (
                input["messages"] if phase.get("include_messages", False) else []
            )
            phase_input = {
                "messages": phase_messages,
                "existing": store_based + ephemeral,
            }
            phase_enriched = phase_enricher.invoke(phase_input)
            store_based, ephemeral, removed = self._apply_enricher_output(
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


def create_memory_store_enricher(
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
    namespace_prefix: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
    phases: list[MemoryPhase] | None = None,
) -> MemoryStoreEnricher:
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
            config={"configurable": {"langgraph_user_id": "user123"}},
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
            config={"configurable": {"langgraph_user_id": "user123"}},
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
            namespace_prefix=("project", "team_1", "{langgraph_user_id}"),
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
            config={"configurable": {"langgraph_user_id": "user123"}},
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
            config={"configurable": {"langgraph_user_id": "user123"}},
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
           e.g., ("project", "team", "{langgraph_user_id}")
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
            organizing memories. Supports templated values like "{langgraph_user_id}" which are
            populated from the runtime context. Defaults to ("memories", "{langgraph_user_id}").

    Returns:
        enricher (Callable): An async function that processes conversations and automatically manages memories in the configured storage. The function works by:

            - Memory search and retrieval
            - Memory creation and updates
            - Storage operations and versioning
    """
    return MemoryStoreEnricher(
        model,
        schemas=schemas,
        instructions=instructions,
        enable_inserts=enable_inserts,
        enable_deletes=enable_deletes,
        query_model=query_model,
        query_limit=query_limit,
        namespace_prefix=namespace_prefix,
        phases=phases,
    )


__all__ = [
    "create_memory_enricher",
    "create_memory_searcher",
    "create_memory_store_enricher",
    "create_thread_extractor",
]
