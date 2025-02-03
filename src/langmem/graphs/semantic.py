import typing

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import StateGraph
from typing_extensions import TypedDict

from langmem import create_memory_store_enricher
from langmem.knowledge import MemoryPhase


class InputState(TypedDict, total=False):
    messages: typing.Required[
        list[AnyMessage] | list[tuple[list[AnyMessage], dict[str, str]]]
    ]
    schemas: None | list[dict] | dict
    namespace: tuple[str, ...] | None


class OutputState(TypedDict):
    updated_memories: list
    root_namesapace: tuple[str, ...]


class Config(TypedDict):
    model: str
    query_model: str | None
    enable_inserts: bool
    enable_deletes: bool
    instructions: str
    phases: list[MemoryPhase] | None


async def enrich(state: InputState, config: RunnableConfig):
    messages = state.get("messages", [])
    if not messages:
        return {"updated_memories": []}
    if isinstance(messages[0], list):
        messages = [m[0] for m in messages]
    namespace = state.get("namespace") or ()
    configurable = config.get("configurable", {})
    model = configurable.get("model", "anthropic:claude-3-5-sonnet-latest")
    schemas = state.get("schemas", None)

    instructions = """You are a memory subroutine for an AI. Extract all relevant knowledge you can infer from the provided context,\
 including all facts, events, concepts, etc. and their relationships. Reflect deeply to ensure all information is correct, complete, and relevant.
 If similar information has already been saved, you can patch or consolidate the existing memories to ensure it is up-to-date and complete.

Use parallel tool calling to extract all information the AI will need to adapt and remember. 
"""
    instructions = configurable.get("instructions", instructions)
    phases = config.get("phases")
    if not phases:
        phases = [
            # {
            #     "instructions": "You are memory subroutine for an AI. Reflect deeply on all the extracted memories."
            #     " What *new* generalizations can you make about the user, the AI, or the context that will teach the AI to perform?"
            #     " What insights are now more nuanced or conditional than before?"
            #     " Focus on synthesizing new logical conclusions, connections, preferences,"
            #     " facts, and any other noteworthy information based on what you can deduce from the existing memories."
            #     " Include all relevant context in your memories.",
            #     "enable_deletes": False,
            # },
            {
                "instructions": "You are memory subroutine for an AI. Consolidate and de-deuplicate all similar memories to reduce waste."
                " You want each final memory to be complete, distinct, "
                " and de-duplicated. To consolidate two or more memories, Patch one with the synthesized content and Remove the others,"
                " doing so all in a single parallel tool call operation.\n\n",
                "enable_deletes": True,
                "enable_inserts": False,
            },
        ]
    enricher = create_memory_store_enricher(
        model,
        instructions=instructions,
        query_model=configurable.get("query_model", "openai:gpt-4o-mini"),
        schemas=[schemas] if isinstance(schemas, dict) else schemas,
        enable_inserts=configurable.get("enable_inserts", True),
        enable_deletes=configurable.get("enable_deletes", True),
        namespace=("{langgraph_auth_user_id}", "semantic", *namespace),
        phases=phases,
    )

    updated_memories = await enricher(messages)
    return {
        "updated_memories": updated_memories,
        "root_namespace": ("semantic", configurable.get("langgraph_auth_user_id", "")),
    }


graph = (
    StateGraph(input=InputState, output=OutputState, config_schema=Config)
    .add_node("enrich", enrich)
    .add_edge("__start__", "enrich")
    .compile()
)
graph.name = "enrich_memories"
