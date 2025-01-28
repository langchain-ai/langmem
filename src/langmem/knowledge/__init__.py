import asyncio
import typing
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.utils.config import get_store
from pydantic import BaseModel, Field, model_validator
from trustcall import create_extractor
from langmem import utils
from langmem.knowledge.tools import create_search_memory_tool, create_manage_memory_tool
import langsmith as ls

## LangGraph Tools


def create_thread_extractor(
    model: str,
    schema: typing.Union[None, BaseModel, type] = None,
    instructions: str = "You are tasked with summarizing the following conversation.",
):
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
    enable_deletions: bool = False,
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
    enable_deletions: bool = False,
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
    enable_deletions: bool = False,
):
    model = model if isinstance(model, BaseChatModel) else init_chat_model(model)
    str_type = False
    if schemas is None:

        class Memory(BaseModel):
            """Call this tool to extract memories for things like preferences, instructions, important context, events, and anything else you want to remember about for future conversations."""

            content: str

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
        coerced = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": f"Enrich, prune, and organize memories based on the following interaction. If an existing memory is incorrect or outdated, update it based on the new information. "
                f"<session_{id_}>\n{utils.get_conversation(messages)}\n</session_{id_}>",
            },
        ]
        if str_type and existing and all(isinstance(ex, str) for ex in existing):
            existing = [(str(uuid.uuid4()), Memory(content=ex)) for ex in existing]
        existing_keys = [str(ex[0]) for ex in existing] if existing else []

        class RemoveMemory:
            """Use this tool to remove an invalid or redundant memory. To consolidate two memories, Patch the one to retain and call this tool with the keys of the two memories as arguments."""

            json_doc_id: str = Field(
                description=f"The json_doc_id of the memory to remove. Must be one of: {existing_keys}."
            )

            @model_validator(mode="after")
            def validate_keys(cls, values: typing.Any) -> typing.Any:
                if str(values["json_doc_id"]) not in existing_keys:
                    raise ValueError(
                        f"json_doc_id key {values['json_doc_id']} not in existing keyss {existing_keys}"
                    )
                return values

        schemas_ = schemas + ([RemoveMemory] if (enable_deletions and existing) else [])
        extractor = create_extractor(
            model, tools=schemas_, tool_choice="any", enable_inserts=enable_inserts
        )

        response = await extractor.ainvoke({"messages": coerced, "existing": existing})
        result = [
            (rmeta.get("json_doc_id", str(uuid.uuid4())), r)
            for r, rmeta in zip(response["responses"], response["response_metadata"])
        ]
        return result

    return enrich_memories


def create_memory_searcher(
    model: str | BaseChatModel,
    prompt: str = "You are a memory search assistant.",
    *,
    namespace_prefix: tuple[str, ...] = ("memories", "{user_id}"),
):
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


def create_memory_store_enricher(
    model: str | BaseChatModel,
    *,
    schemas: list | None = None,
    instructions: str = _MEMORY_INSTRUCTIONS,
    enable_inserts: bool = True,
    enable_deletions: bool = True,
    query_model: str | BaseChatModel | None = None,
    namespace_prefix: tuple[str, ...] = ("memories", "{user_id}"),
):
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
    if schemas:
        evolver = create_memory_enricher(
            model,
            schemas=schemas,
            instructions=instructions,
            enable_inserts=enable_inserts,
            enable_deletions=enable_deletions,
        )
    else:
        evolver = create_memory_enricher(
            model,
            instructions=instructions,
            enable_inserts=enable_inserts,
            enable_deletions=enable_deletions,
        )
    search_tool = create_search_memory_tool(namespace_prefix=namespace_prefix)
    query_gen = query_model.bind_tools(
        [search_tool],
        tool_choice="search_memory",
    )

    namespacer = utils.NamespaceTemplate(namespace_prefix)

    async def manage_memories(messages: list[AnyMessage]):
        store = get_store()
        namespace = namespacer()
        convo = utils.get_conversation(messages)
        msg = await query_gen.ainvoke(
            f"""Generate a search query to retrieve memories based on the conversation: \n\n<convo>\n{convo}\n</convo>."""
        )
        all_search_results = await asyncio.gather(
            *(store.asearch(namespace, **tc["args"]) for tc in msg.tool_calls)
        )
        memories = [
            (r.key, r.value["kind"], r.value["content"])
            for search_results in all_search_results
            for r in search_results
        ]
        new_memories = await evolver(messages, existing=memories)
        deletions, others = [], []
        for key, content in new_memories:
            if content.__repr_name__() == "RemoveMemory":
                deletions.append(key)
            else:
                others.append((key, content))

        put_kwargs = [
            {
                "namespace": namespace,
                "key": key,
                "value": {
                    "kind": content.__repr_name__(),
                    "content": content.model_dump(mode="json"),
                },
            }
            for key, content in others
        ]
        await asyncio.gather(
            *(
                *(
                    store.aput(
                        **kwargs,
                    )
                    for kwargs in put_kwargs
                ),
                *(store.adelete(namespace, key=str(key)) for key in deletions),
            )
        )

        return [kwargs for kwargs in put_kwargs]

    return manage_memories


__all__ = [
    "create_manage_memory_tool",
    "create_memory_enricher",
    "create_memory_searcher",
    "create_memory_store_enricher",
    "create_thread_extractor",
]
