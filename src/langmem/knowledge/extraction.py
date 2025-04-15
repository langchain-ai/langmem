import asyncio
import datetime
import typing
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.config import get_executor_for_config
from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    NotProvided,
)
from langgraph.store.base import (
    Item as BaseItem,
)
from langgraph.store.base import (
    SearchItem as BaseSearchItem,
)
from langgraph.utils.config import ensure_config, get_store
from pydantic import BaseModel, Field
from trustcall import create_extractor
from typing_extensions import TypedDict

from langmem import utils
from langmem.knowledge.tools import create_search_memory_tool

## LangGraph Tools


class Item(BaseItem):
    value: BaseModel | dict[str, typing.Any]

    def dict(self) -> dict:
        result = super().dict()
        if isinstance(self.value, BaseModel):
            result["value"] = self.value.model_dump(mode="json")
        return result


class SearchItem(BaseSearchItem):
    value: BaseModel | dict[str, typing.Any]

    def dict(self) -> dict:
        result = super().dict()
        if isinstance(self.value, BaseModel):
            result["value"] = self.value.model_dump(mode="json")
        return result


class MessagesState(TypedDict):
    messages: list[AnyMessage]


class MemoryState(MessagesState):
    existing: typing.NotRequired[list[tuple[str, BaseModel]]]
    max_steps: int  # Default of 1


class SummarizeThread(BaseModel):
    title: str
    summary: str


class ExtractedMemory(typing.NamedTuple):
    id: str
    content: BaseModel


S = typing.TypeVar("S", bound=type)


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


_MEMORY_INSTRUCTIONS = """You are a long-term memory manager maintaining a core store of semantic, procedural, and episodic memory. These memories power a life-long learning agent's core predictive model.

What should the agent learn from this interaction about the user, itself, or how it should act? Reflect on the input trajectory and current memories (if any).

1. **Extract & Contextualize**  
   - Identify essential facts, relationships, preferences, reasoning procedures, and context
   - Caveat uncertain or suppositional information with confidence levels (p(x)) and reasoning
   - Quote supporting information when necessary

2. **Compare & Update**  
   - Attend to novel information that deviates from existing memories and expectations.
   - Consolidate and compress redundant memories to maintain information-density; strengthen based on reliability and recency; maximize SNR by avoiding idle words.
   - Remove incorrect or redundant memories while maintaining internal consistency

3. **Synthesize & Reason**  
   - What can you conclude about the user, agent ("I"), or environment using deduction, induction, and abduction?
   - What patterns, relationships, and principles emerge about optimal responses?
   - What generalizations can you make?
   - Qualify conclusions with probabilistic confidence and justification

As the agent, record memory content exactly as you'd want to recall it when predicting how to act or respond. 
Prioritize retention of surprising (pattern deviation) and persistent (frequently reinforced) information, ensuring nothing worth remembering is forgotten and nothing false is remembered. Prefer dense, complete memories over overlapping ones."""


class Done(BaseModel):
    """Only call this tool once you are done forming & consolidating memories.
    Before that, continue to refine existing memories by patching and removing them
    or create new ones."""

    pass


class MemoryManager(Runnable[MemoryState, list[ExtractedMemory]]):
    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        schemas: typing.Sequence[typing.Union[BaseModel, type]] = (Memory,),
        instructions: str = _MEMORY_INSTRUCTIONS,
        enable_inserts: bool = True,
        enable_updates: bool = True,
        enable_deletes: bool = False,
    ):
        self.model = (
            model if isinstance(model, BaseChatModel) else init_chat_model(model)
        )
        self.schemas = schemas or (Memory,)
        self.instructions = instructions
        self.enable_inserts = enable_inserts
        self.enable_updates = enable_updates
        self.enable_deletes = enable_deletes

    async def ainvoke(
        self,
        input: MemoryState,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[ExtractedMemory]:
        max_steps = input.get("max_steps")
        if max_steps is None:
            max_steps = 1
        messages = input["messages"]
        existing = input.get("existing")
        prepared_messages = self._prepare_messages(messages, max_steps)
        prepared_existing = self._prepare_existing(existing)
        # Track external memory IDs (those passed in from outside)
        external_ids = {mem_id for mem_id, _, _ in prepared_existing}

        extractor = create_extractor(
            self.model,
            tools=list(self.schemas),
            enable_inserts=self.enable_inserts,
            enable_updates=self.enable_updates,
            enable_deletes=self.enable_deletes,
            existing_schema_policy=False,
        )
        # initial payload uses the full prepared_existing list
        payload = {"messages": prepared_messages, "existing": prepared_existing}
        # Use a dict to record the latest update for each memory id.
        results: dict[str, BaseModel] = {}

        for i in range(max_steps):
            if i == 1:
                extractor = create_extractor(
                    self.model,
                    tools=list(self.schemas) + [Done],
                    enable_inserts=self.enable_inserts,
                    enable_updates=self.enable_updates,
                    enable_deletes=self.enable_deletes,
                    existing_schema_policy=False,
                )
            response = await extractor.ainvoke(payload)
            is_done = False
            step_results = {}
            for r, rmeta in zip(response["responses"], response["response_metadata"]):
                if hasattr(r, "__repr_name__") and r.__repr_name__() == "Done":
                    is_done = True
                    continue
                mem_id = (
                    r.json_doc_id
                    if hasattr(r, "__repr_name__") and r.__repr_name__() == "RemoveDoc"
                    else rmeta.get("json_doc_id", str(uuid.uuid4()))
                )
                step_results[mem_id] = r
            results.update(step_results)

            for mem_id, _, mem in prepared_existing:
                if mem_id not in results:
                    results[mem_id] = mem

            ai_msg = response["messages"][-1]
            if is_done or not ai_msg.tool_calls:
                break
            if i < max_steps - 1:
                actions = [
                    (
                        "updated"
                        if rmeta.get("json_doc_id")
                        else (
                            "deleted"
                            if hasattr(r, "__repr_name__")
                            and r.__repr_name__() == "RemoveDoc"
                            else "inserted"
                        )
                    )
                    for r, rmeta in zip(
                        response["responses"], response["response_metadata"]
                    )
                ]
                prepared_messages = (
                    prepared_messages
                    + [response["messages"][-1]]
                    + [
                        {
                            "role": "tool",
                            "content": f"Memory {rid} {action}.",
                            "tool_call_id": tc["id"],
                        }
                        for tc, ((rid, _), action) in zip(
                            ai_msg.tool_calls, zip(list(step_results.items()), actions)
                        )
                    ]
                )
                # For the next iteration payload, drop all removal objects.
                payload = {
                    "messages": prepared_messages,
                    "existing": self._filter_response(
                        list(results.items()), external_ids, exclude_removals=True
                    ),
                }

        # For the final response, include removals only if they refer to an external memory.
        return self._filter_response(
            list(results.items()), external_ids, exclude_removals=False
        )

    def invoke(
        self,
        input: MemoryState,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[ExtractedMemory]:
        max_steps = input.get("max_steps")
        if max_steps is None:
            max_steps = 1
        messages = input["messages"]
        existing = input.get("existing")
        prepared_messages = self._prepare_messages(messages, max_steps)
        prepared_existing = self._prepare_existing(existing)
        # Track external memory IDs (those passed in from outside)
        external_ids = {mem_id for mem_id, _, _ in prepared_existing}

        extractor = create_extractor(
            self.model,
            tools=list(self.schemas),
            enable_inserts=self.enable_inserts,
            enable_updates=self.enable_updates,
            enable_deletes=self.enable_deletes,
            existing_schema_policy=False,
        )
        payload = {"messages": prepared_messages, "existing": prepared_existing}
        # Use a dict to record the latest update for each memory id.
        results: dict[str, BaseModel] = {}

        for i in range(max_steps):
            if i == 1:
                extractor = create_extractor(
                    self.model,
                    tools=list(self.schemas) + [Done],
                    enable_inserts=self.enable_inserts,
                    enable_updates=self.enable_updates,
                    enable_deletes=self.enable_deletes,
                    existing_schema_policy=False,
                )
            response = extractor.invoke(payload)
            is_done = False
            step_results: dict[str, BaseModel] = {}
            for r, rmeta in zip(response["responses"], response["response_metadata"]):
                if hasattr(r, "__repr_name__") and r.__repr_name__() == "Done":
                    is_done = True
                    continue
                mem_id = (
                    r.json_doc_id
                    if (
                        hasattr(r, "__repr_name__") and r.__repr_name__() == "RemoveDoc"
                    )
                    else rmeta.get("json_doc_id", str(uuid.uuid4()))
                )
                step_results[mem_id] = r
            results.update(step_results)

            # Ensure any memory from the initial payload that hasn't been updated is retained.
            for mem_id, _, mem in prepared_existing:
                if mem_id not in results:
                    results[mem_id] = mem

            ai_msg = response["messages"][-1]
            if is_done or not ai_msg.tool_calls:
                break
            if i < max_steps - 1:
                actions = [
                    (
                        "updated"
                        if rmeta.get("json_doc_id")
                        else (
                            "deleted"
                            if (
                                hasattr(r, "__repr_name__")
                                and r.__repr_name__() == "RemoveDoc"
                            )
                            else "inserted"
                        )
                    )
                    for r, rmeta in zip(
                        response["responses"], response["response_metadata"]
                    )
                ]
                prepared_messages = (
                    prepared_messages
                    + [response["messages"][-1]]
                    + [
                        {
                            "role": "tool",
                            "content": f"Memory {rid} {action}.",
                            "tool_call_id": tc["id"],
                        }
                        for tc, ((rid, _), action) in zip(
                            ai_msg.tool_calls, zip(list(step_results.items()), actions)
                        )
                    ]
                )
                payload = {
                    "messages": prepared_messages,
                    "existing": self._filter_response(
                        list(results.items()), external_ids, exclude_removals=True
                    ),
                }

        return self._filter_response(
            list(results.items()), external_ids, exclude_removals=False
        )

    async def __call__(
        self,
        messages: typing.Sequence[AnyMessage],
        existing: typing.Optional[typing.Sequence[ExtractedMemory]] = None,
    ) -> list[ExtractedMemory]:
        input: MemoryState = {"messages": messages}
        if existing is not None:
            input["existing"] = existing
        return await self.ainvoke(input)

    def _prepare_messages(
        self, messages: list[AnyMessage], max_steps: int = 1
    ) -> list[dict]:
        id_ = str(uuid.uuid4())
        session = (
            f"\n\n<session_{id_}>\n{utils.get_conversation(messages)}\n</session_{id_}>"
        )
        if max_steps > 1:
            session = f"{session}\n\nYou have a maximum of {max_steps - 1} attempts"
            " to form and consolidate memories from this session."
        return [
            {"role": "system", "content": "You are a memory subroutine for an AI."},
            {
                "role": "user",
                "content": (
                    f"{self.instructions}\n\nEnrich, prune, and organize memories based on any new information. "
                    f"If an existing memory is incorrect or outdated, update it based on the new information. "
                    f"All operations must be done in single parallel multi-tool call."
                    f" Avoid duplicate extractions. {session}"
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

    @staticmethod
    def _filter_response(
        memories: list[ExtractedMemory],
        external_ids: set[str],
        exclude_removals: bool = False,
    ) -> list[ExtractedMemory]:
        """
        When exclude_removals is True (for the next iteration payload),
        drop any memory whose content is a RemoveDoc.
        When False (final response), drop removal objects only for internal memories.
        """
        results = []
        for rid, value in memories:
            is_removal = (
                hasattr(value, "__repr_name__") and value.__repr_name__() == "RemoveDoc"
            )
            if exclude_removals:
                if is_removal:
                    continue
            else:
                # Final response: if this is a removal *and* its id is not external, skip it.
                if is_removal and (rid not in external_ids):
                    continue
            results.append(ExtractedMemory(id=rid, content=value))
        return results


def create_memory_manager(
    model: str | BaseChatModel,
    /,
    *,
    schemas: typing.Sequence[S] = (Memory,),
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_updates: bool = True,
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
        enable_updates (bool, optional): Whether to allow updating existing memories
            that are outdated or contradicted by new information. Defaults to True.
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
        updated_memories = await manager.ainvoke(
            {"messages": conversation, "existing": memories}
        )
        ```

        Insertion-only memories:
        ```python
        manager = create_memory_manager(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[PreferenceMemory],
            enable_updates=False,
            enable_deletes=False,
        )

        conversation = [
            {
                "role": "user",
                "content": "Actually I changed my mind, dark mode is the best mode",
            },
            {"role": "assistant", "content": "I'll update your preference"},
        ]

        # The manager will only create new memories
        updated_memories = await manager.ainvoke(
            {"messages": conversation, "existing": memories}
        )
        print(updated_memories)
        ```

        Providing multiple max steps for extraction and synthesis:
        ```python
        manager = create_memory_manager(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[PreferenceMemory],
        )

        conversation = [
            {"role": "user", "content": "I prefer dark mode in all my apps"},
            {"role": "assistant", "content": "I'll remember that preference"},
        ]

        # Set max steps for extraction and synthesis
        max_steps = 3
        memories = await manager.ainvoke(
            {"messages": conversation, "max_steps": max_steps}
        )
        print(memories)
        ```
    """

    return MemoryManager(
        model,
        schemas=schemas,
        instructions=instructions,
        enable_inserts=enable_inserts,
        enable_updates=enable_updates,
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

        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )
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

    def search(msg: AIMessage) -> list[SearchItem]:
        return search_tool.batch(
            [tc for tc in msg.tool_calls if tc["name"] == "search_memory"]
        )

    async def search_async(msg: AIMessage) -> list[SearchItem]:
        return await search_tool.abatch(
            [tc for tc in msg.tool_calls if tc["name"] == "search_memory"]
        )

    searcher = RunnableLambda(search, search_async)

    return (  # type: ignore
        template | utils.merge_message_runs | query_gen | searcher | return_sorted
    ).with_config({"run_name": "search_memory_pipeline"})


class MemoryPhase(TypedDict, total=False):
    instructions: str
    include_messages: bool
    enable_inserts: bool
    enable_deletes: bool


class MemoryStoreManagerInput(TypedDict):
    """Input schema for MemoryStoreManager."""

    messages: list[AnyMessage]
    max_steps: int  # Default of 1


class MemoryStoreManager(Runnable[MemoryStoreManagerInput, list[dict]]):
    def __init__(
        self,
        model: str | BaseChatModel,
        /,
        *,
        schemas: list[S] | None = None,
        default: str | dict | S | None = None,
        default_factory: (
            typing.Callable[[RunnableConfig], str | dict | S] | None
        ) = None,
        instructions: str = _MEMORY_INSTRUCTIONS,
        enable_inserts: bool = True,
        enable_deletes: bool = True,
        query_model: str | BaseChatModel | None = None,
        query_limit: int = 5,
        namespace: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
        store: BaseStore | None = None,
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
        self.schemas = schemas if schemas is not None else (Memory,)
        self.schema_name_map = {schema.__name__: schema for schema in self.schemas}
        self.default_factory = default_factory
        if default is not None:
            if self.default_factory is not None:
                raise ValueError("Cannot specify both default and default_factory")
            coerced = self._coerce_default(default, self.schemas)
            self.default_factory = lambda _: coerced
        self.instructions = instructions
        self.enable_inserts = enable_inserts
        self.enable_deletes = enable_deletes
        self.query_limit = query_limit
        self.phases = phases or []
        self.namespace = utils.NamespaceTemplate(namespace)
        self._store = store

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

    @property
    def store(self) -> BaseStore:
        """Get the store to use for memory storage.

        Returns the store provided during initialization if available,
        otherwise falls back to the store configured in the LangGraph context.
        """
        if self._store is not None:
            return self._store
        try:
            self._store = get_store()
        except RuntimeError as e:
            raise ValueError(
                "Memory Manager's store not configured in LangGraph context. "
                "First use in the graph before calling, or initialize with an instance of the store."
            ) from e
        return self._store

    @staticmethod
    def _coerce_default(
        default: str | dict | S, schemas: tuple[S, ...]
    ) -> str | dict | S:
        if isinstance(default, str) and schemas == (Memory,):
            return Memory(content=default)
        elif isinstance(default, BaseModel):
            return default
        else:
            for schema in schemas:
                try:
                    return schema(**default)
                except Exception:
                    pass
            else:
                raise ValueError(
                    "The provided default did not match any of the expected schemas."
                    f"\nDefault:\n{default}\n\nSchemas:\n{schemas}"
                )
        raise ValueError(f"Invalid default type: {type(default)}")

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
        store = self.store
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
        if not store_map and self.default_factory is not None:
            config = ensure_config(config)
            default = self.default_factory(config)
            coerced = self._coerce_default(default, self.schemas)
            dumped = {
                "kind": coerced.__repr_name__(),
                "content": coerced.model_dump(mode="json"),
            }
            await store.aput(
                namespace,
                key="default",
                value=dumped,
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            store_map = self._sort_results(
                [
                    [
                        SearchItem(
                            namespace, "default", dumped, created_at=now, updated_at=now
                        )
                    ]
                ],
                self.query_limit,
            )

        store_based = [
            (sid, item.value["kind"], item.value["content"])
            for sid, item in store_map.items()
        ]
        ephemeral: list[tuple[str, str, dict]] = []
        removed_ids: set[str] = set()

        # --- Enrich memories using the composed MemoryManager (async) ---
        enriched = await self.memory_manager.ainvoke(
            {
                "messages": input["messages"],
                "existing": store_based,
                "max_steps": input.get("max_steps"),
            }
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
        store = self.store
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
        if not store_map and self.default_factory is not None:
            config = ensure_config(config)
            default = self.default_factory(config)
            coerced = self._coerce_default(default, self.schemas)
            dumped = {
                "kind": coerced.__repr_name__(),
                "content": coerced.model_dump(mode="json"),
            }
            store.put(
                namespace,
                key="default",
                value=dumped,
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            store_map = self._sort_results(
                [
                    [
                        SearchItem(
                            namespace, "default", dumped, created_at=now, updated_at=now
                        )
                    ]
                ],
                self.query_limit,
            )
        store_based = [
            (sid, item.value["kind"], item.value["content"])
            for sid, item in store_map.items()
        ]
        ephemeral: list[tuple[str, str, dict]] = []
        removed_ids: set[str] = set()

        enriched = self.memory_manager.invoke(
            {
                "messages": input["messages"],
                "existing": store_based,
                "max_steps": input.get("max_steps"),
            }
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

    def _coerce_value(
        self, value: dict[str, typing.Any]
    ) -> BaseModel | dict[str, typing.Any]:
        if "kind" not in value or "content" not in value:
            return value
        kind = value["kind"]
        content = value["content"]
        if kind not in self.schema_name_map:
            return value
        schema = self.schema_name_map[kind]
        return schema.model_validate(content)

    def _coerce_item(self, item: BaseItem | None) -> Item | None:
        if item is None:
            return None
        return Item(
            namespace=item.namespace,
            key=item.key,
            value=self._coerce_value(item.value),
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    def _coerce_search_item(self, item: SearchItem) -> SearchItem:
        return SearchItem(
            namespace=item.namespace,
            key=item.key,
            value=self._coerce_value(item.value),
            created_at=item.created_at,
            updated_at=item.updated_at,
            score=item.score,
        )

    def get(
        self,
        key: str,
        *,
        refresh_ttl: typing.Optional[bool] = None,
        config: typing.Optional[RunnableConfig] = None,
    ) -> typing.Optional[Item]:
        """Retrieve a single item in the store.

        Args:
            key: Unique identifier within the namespace.
            refresh_ttl: Whether to refresh TTLs for the returned item.
                If None (default), uses the store's default refresh_ttl setting.
                If no TTL is specified, this argument is ignored.

        Returns:
            The retrieved item or None if not found.

        ???+ example "Examples"
            Retrieve an item:
            ```python
            item = manager.get("report")
            ```
        """
        namespace = self.get_namespace(config)
        result = self._coerce_item(
            self.store.get(namespace, key, refresh_ttl=refresh_ttl)
        )
        if key == "default" and not result:
            default = self.default_factory(ensure_config(config))
            coerced = self._coerce_default(default, self.schemas)
            now = datetime.datetime.now(datetime.timezone.utc)
            return Item(namespace, "default", coerced, created_at=now, updated_at=now)
        return result

    def search(
        self,
        *,
        query: typing.Optional[str] = None,
        filter: typing.Optional[dict[str, typing.Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: typing.Optional[bool] = None,
        config: typing.Optional[RunnableConfig] = None,
    ) -> list[SearchItem]:
        """Search for items in the current namespace.

        Args:
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.
            refresh_ttl: Whether to refresh TTLs for the returned items.
                If no TTL is specified, this argument is ignored.

        Returns:
            List of items matching the search criteria.

        ???+ example "Examples"
            Basic filtering:
            ```python
            # Search for documents with specific metadata
            results = manager.search(
                query="app preferences",
                filter={"type": "article", "status": "published"},
            )
            ```

            Natural language search (requires vector store implementation):
            ```python
            # Search for semantically similar documents
            results = manager.search(
                query="machine learning applications in healthcare",
                filter={"type": "research_paper"},
                limit=5,
            )
            ```
        """
        namespace = self.get_namespace(config)
        results = [
            self._coerce_search_item(it)
            for it in self.store.search(
                namespace,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )
        ]
        if self.default_factory and not results:
            default = self.default_factory(ensure_config(config))
            coerced = self._coerce_default(default, self.schemas)
            now = datetime.datetime.now(datetime.timezone.utc)
            return [
                SearchItem(
                    namespace, "default", coerced, created_at=now, updated_at=now
                )
            ]
        return results

    def put(
        self,
        key: str,
        value: dict[str, typing.Any],
        index: typing.Optional[typing.Union[typing.Literal[False], list[str]]] = None,
        *,
        ttl: typing.Union[typing.Optional[float], "NotProvided"] = NOT_PROVIDED,
        config: typing.Optional[RunnableConfig] = None,
    ) -> None:
        """Store or update an item in the store.

        Args:
            key: Unique identifier within the namespace.
            value: Dictionary containing the item's data. Must contain string keys
                and JSON-serializable values.
            index: Controls how the item's fields are indexed for search:

                - None (default): Use `fields` you configured when creating the store (if any)
                    If you do not initialize the store with indexing capabilities,
                    the `index` parameter will be ignored
                - False: Disable indexing for this item
                - list[str]: List of field paths to index
            ttl: Time to live in minutes. Support for this argument depends on your store adapter.
                If specified, the item will expire after this many minutes from when it was last accessed.
                None means no expiration. Expired runs will be deleted opportunistically.
                By default, the expiration timer refreshes on both read operations (get/search)
                and write operations (put/update), whenever the item is included in the operation.

        ???+ example "Examples"
            Store item. Indexing depends on how you configure the store.
            ```python
            manager.put("report", {"memory": "Will likes ai"})
            ```

            Do not index item for semantic search. Still accessible through get()
            and search() operations but won't have a vector representation.
            ```python
            manager.put("report", {"memory": "Will likes ai"}, index=False)
            ```

            Index specific fields for search.
            ```python
            manager.put("report", {"memory": "Will likes ai"}, index=["memory"])
            ```
        """
        return self.store.put(
            self.get_namespace(config),
            key,
            value,
            index=index,
            ttl=ttl,
        )

    def delete(self, key: str, config: typing.Optional[RunnableConfig] = None) -> None:
        """Delete an item.

        Args:
            key: Unique identifier within the namespace.
        """
        self.store.delete(self.get_namespace(config), key)

    async def aget(
        self,
        key: str,
        *,
        refresh_ttl: typing.Optional[bool] = None,
        config: typing.Optional[RunnableConfig] = None,
    ) -> typing.Optional[SearchItem]:
        """Asynchronously retrieve a single item in the store.

        Args:
            key: Unique identifier within the namespace.
            refresh_ttl: Whether to refresh TTLs for the returned item.
                If None (default), uses the store's default refresh_ttl setting.
                If no TTL is specified, this argument is ignored.

        Returns:
            The retrieved item or None if not found.

        ???+ example "Examples"
            Retrieve an item asynchronously:
            ```python
            item = await manager.aget("report")
            ```
        """
        namespace = self.get_namespace(config)
        it = await self.store.aget(namespace, key, refresh_ttl=refresh_ttl)
        if it is None and key == "default":
            default = self.default_factory(ensure_config(config))
            coerced = self._coerce_default(default, self.schemas)
            now = datetime.datetime.now(datetime.timezone.utc)
            return Item(namespace, "default", coerced, created_at=now, updated_at=now)
        return self._coerce_item(it)

    async def asearch(
        self,
        *,
        query: typing.Optional[str] = None,
        filter: typing.Optional[dict[str, typing.Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: typing.Optional[bool] = None,
        config: typing.Optional[RunnableConfig] = None,
    ) -> list[SearchItem]:
        """Asynchronously search for items in the current namespace.

        Args:
            query: Optional query for natural language search.
            filter: Key-value pairs to filter results.
            limit: Maximum number of items to return.
            offset: Number of items to skip before returning results.
            refresh_ttl: Whether to refresh TTLs for the returned items.
                If None (default), uses the store's TTLConfig.refresh_default setting.
                If TTLConfig is not provided or no TTL is specified, this argument is ignored.

        Returns:
            List of items matching the search criteria.

        ???+ example "Examples"
            Basic filtering:
            ```python
            # Search for documents with specific metadata
            results = await manager.asearch(
                filter={"type": "article", "status": "published"}
            )
            ```

            Natural language search (requires vector store implementation):
            ```python
            # Search for semantically similar documents
            results = await manager.asearch(
                query="machine learning applications in healthcare",
                filter={"type": "research_paper"},
                limit=5,
            )
            ```
        """
        namespace = self.get_namespace(config)
        results = [
            self._coerce_search_item(it)
            for it in await self.store.asearch(
                namespace,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl,
            )
        ]
        if self.default_factory and not results:
            default = self.default_factory(ensure_config(config))
            coerced = self._coerce_default(default, self.schemas)
            now = datetime.datetime.now(datetime.timezone.utc)
            # note that we don't actually put this in the store here!
            return [
                SearchItem(
                    namespace, "default", coerced, created_at=now, updated_at=now
                )
            ]
        return results

    async def aput(
        self,
        key: str,
        value: dict[str, typing.Any],
        index: typing.Optional[typing.Union[typing.Literal[False], list[str]]] = None,
        *,
        ttl: typing.Union[typing.Optional[float], "NotProvided"] = NOT_PROVIDED,
        config: typing.Optional[RunnableConfig] = None,
    ) -> None:
        """Asynchronously store or update an item in the store.

        Args:
            namespace: Hierarchical path for the item, represented as a tuple of strings.
                Example: ("documents", "user123")
            key: Unique identifier within the namespace. Together with namespace forms
                the complete path to the item.
            value: Dictionary containing the item's data. Must contain string keys
                and JSON-serializable values.
            index: Controls how the item's fields are indexed for search:

                - None (default): Use `fields` you configured when creating the store (if any)
                    If you do not initialize the store with indexing capabilities,
                    the `index` parameter will be ignored
                - False: Disable indexing for this item
                - list[str]: List of field paths to index, supporting:
                    - Nested fields: "metadata.title"
                    - Array access: "chapters[*].content" (each indexed separately)
                    - Specific indices: "authors[0].name"
            ttl: Time to live in minutes. Support for this argument depends on your store adapter.
                If specified, the item will expire after this many minutes from when it was last accessed.
                None means no expiration. Expired runs will be deleted opportunistically.
                By default, the expiration timer refreshes on both read operations (get/search)
                and write operations (put/update), whenever the item is included in the operation.

        Note:
            Indexing support depends on your store implementation.
            If you do not initialize the store with indexing capabilities,
            the `index` parameter will be ignored.

            Similarly, TTL support depends on the specific store implementation.
            Some implementations may not support expiration of items.

        ???+ example "Examples"
            Store item. Indexing depends on how you configure the store.
            ```python
            await manager.aput("report", {"memory": "Will likes ai"})
            ```

            Do not index item for semantic search. Still accessible through get()
            and search() operations but won't have a vector representation.
            ```python
            await manager.aput("report", {"memory": "Will likes ai"}, index=False)
            ```

            Index specific fields for search (if store configured to index items):
            ```python
            await manager.aput(
                "report",
                {
                    "memory": "Will likes ai",
                    "context": [{"content": "..."}, {"content": "..."}],
                },
                index=["memory", "context[*].content"],
            )
            ```
        """
        return await self.store.aput(
            self.get_namespace(config), key, value, index, ttl=ttl
        )

    async def adelete(
        self, key: str, config: typing.Optional[RunnableConfig] = None
    ) -> None:
        """Asynchronously delete an item.

        Args:
            key: Unique identifier within the namespace.
        """
        await self.store.adelete(self.get_namespace(config), key)

    def get_namespace(
        self, config: typing.Optional[RunnableConfig] = None
    ) -> tuple[str, ...]:
        return self.namespace(ensure_config(config))


def create_memory_store_manager(
    model: str | BaseChatModel,
    /,
    *,
    schemas: list[S] | None = None,
    instructions: str = _MEMORY_INSTRUCTIONS,
    default: str | dict | S | None = None,
    default_factory: typing.Callable[[RunnableConfig], str | dict | S] | None = None,
    enable_inserts: bool = True,
    enable_deletes: bool = False,
    query_model: str | BaseChatModel | None = None,
    query_limit: int = 5,
    namespace: tuple[str, ...] = ("memories", "{langgraph_user_id}"),
    store: BaseStore | None = None,
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
        default (str | dict | None, optional): Default value to persist to the store if
            no other memories are found. Defaults to None. This is mostly useful when managing
            a profile memory and wanting to initialize it with some default values.
            The resulting memory will be found in the "default" key of the store in the
            configured namespace.
        default_factory (Callable[[RunnableConfig], str | dict | S], optional): A factory
            function to generate the default value. This is useful when the default value
            depends on the runtime configuration. Defaults to None.
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
        store (Optional[BaseStore], optional): The store to use for memory storage.
            If None, uses the store configured in the LangGraph config. Defaults to None.
            When using LangGraph Platform, the server will manage the store for you.
        phases (Optional[list]): List of MemoryPhase objects defining the phases of the memory enrichment process.

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

        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )

        manager = create_memory_store_manager("anthropic:claude-3-5-sonnet-latest", namespace=("memories", "{langgraph_user_id}"))
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


        @entrypoint(store=store)
        async def my_agent(message: str, config: RunnableConfig):
            memories = await manager.asearch(
                query=message,
                config=config,
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

        config = {"configurable": {"langgraph_user_id": "user123"}}
        response_1 = await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config=config,
        )
        print("response_1:", response_1)
        # Later conversation - automatically retrieves and uses the stored preference
        response_2 = await my_agent.ainvoke(
            "What theme do I prefer?",
            config=config,
        )
        print("response_2:", response_2)
        # You can list over memories in the user's namespace manually:
        print(manager.search(query="app preferences", config=config))
        ```

        You can customize what each memory can look like by defining **schemas**:
        ```python
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
            \"\"\"Store preferences about the user.\"\"\"
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
        config = {"configurable": {"langgraph_user_id": "user123"}}
        await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config=config,
        )

        # See the extracted memories yourself
        print(manager.search(query="app preferences", config=config))

        # Memory is automatically stored and can be retrieved in future conversations
        # The system will also automatically update it if preferences change
        ```

        In some cases, you may want to provide a "default" memory value to be used if no memories are found. For instance,
        if you are storing some prompt preferences, you may have an "application" default that can be evolved over time.
        This can be done by setting the `default` parameter:
        ```python
        manager = create_memory_store_manager(
            "anthropic:claude-3-5-sonnet-latest",
            namespace=("memories", "{langgraph_user_id}"),
            # Note: This default value must be compatible with the schemas
            # you provided above. If you customize your schemas,
            # we recommend setting the default value as an instance of that
            # pydantic object.
            default="Use a concise and professional tone in all responses. The user likes light mode.",
        )


        # ... same agent as before ...
        @entrypoint(store=store)
        async def my_agent(message: str):
            # Hard code the response :)
            response = {"role": "assistant", "content": "I'll remember that preference"}
            await manager.ainvoke(
                {"messages": [{"role": "user", "content": message}, response]}
            )
            return response


        # Store structured memory
        config = {"configurable": {"langgraph_user_id": "user124"}}
        await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config=config,
        )

        # See the extracted memories yourself
        print(manager.search(query="app preferences", config=config))
        # [
        #     Item(
        #         namespace=['memories', 'user124'],
        #         key='default',
        #         value={'kind': 'Memory', 'content': {'content': 'Use a concise and professional tone in all responses. The user prefers dark mode in all apps'}},
        #         created_at='2025-04-14T22:20:25.148884+00:00',
        #         updated_at='2025-04-14T22:20:25.148892+00:00',
        #         score=None
        #     )
        # ]
        ```

        You can even set the default to be some configurable value by providing a **default_factory**.
        ```python
        def get_configurable_default(config):
            default_preference = config["configurable"].get(
                "preference", "Use a concise and professional tone in all responses."
            )
            return default_preference


        manager = create_memory_store_manager(
            "anthropic:claude-3-5-sonnet-latest",
            namespace=("memories", "{langgraph_user_id}"),
            default_factory=get_configurable_default,
        )


        # ... same agent as before ...
        @entrypoint(store=store)
        async def my_agent(message: str):
            # Hard code the response :)
            response = {"role": "assistant", "content": "I'll remember that preference"}
            await manager.ainvoke(
                {"messages": [{"role": "user", "content": message}, response]}
            )
            return response


        # Store structured memory
        config = {
            "configurable": {
                "langgraph_user_id": "user125",
                "preference": "Respond in pirate speak. User likes light mode",
            }
        }
        await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config=config,
        )

        # See the extracted memories yourself
        print(manager.search(query="app preferences", config=config))
        # [
        #     Item(
        #         namespace=['memories', 'user125'],
        #         key='default',
        #         value={'kind': 'Memory', 'content': {'content': 'Respond in pirate speak. User prefers dark mode in all apps'}},
        #         created_at='2025-04-14T22:20:25.148884+00:00',
        #         updated_at='2025-04-14T22:20:25.148892+00:00',
        #         score=None
        #     )
        # ]
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

        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )
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

        config = {"configurable": {"langgraph_user_id": "user123"}}
        await my_agent.ainvoke(
            "I prefer dark mode in all my apps",
            config=config,
        )

        # See the extracted memories yourself
        print(manager.search(config=config))
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

        store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small",
            }
        )
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

        config = {"configurable": {"user_id": "user-123"}}
        fut = await chat.ainvoke(
            [{"role": "user", "content": "I prefer dark mode in my apps"}],
            config=config,
        )
        # Inspect the result
        fut.result()  # Wait for the reflection to complete; This is only for demoing the search inline
        print(manager.search(query="app preferences", config=config))
        ```
    """
    return MemoryStoreManager(
        model,
        schemas=schemas,
        default=default,
        default_factory=default_factory,
        instructions=instructions,
        enable_inserts=enable_inserts,
        enable_deletes=enable_deletes,
        query_model=query_model,
        query_limit=query_limit,
        namespace=namespace,
        store=store,
        phases=phases,
    )


__all__ = [
    "create_memory_manager",
    "create_memory_searcher",
    "create_memory_store_manager",
    "create_thread_extractor",
]
