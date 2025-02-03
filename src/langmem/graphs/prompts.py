from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import StateGraph
from typing_extensions import TypedDict

from langmem import Prompt, create_multi_prompt_optimizer, create_prompt_optimizer


class InputState(TypedDict):
    prompts: list[Prompt] | str
    threads: list[tuple[list[AnyMessage], dict[str, str]]]


class OutputState(TypedDict):
    updated_prompts: list[Prompt]


class Config(TypedDict):
    model: str
    kind: str


async def optimize(state: InputState, config: RunnableConfig):
    prompts = state.get("prompts")
    if not prompts:
        return
    configurable = config.get("configurable", {})
    model = configurable.get("model", "claude-3-5-sonnet-latest")
    kind = configurable.get("kind", "gradient")
    threads = [
        (messages, feedback or "") for messages, feedback in state.get("threads") or []
    ]

    if isinstance(prompts, str):
        prompts = [
            Prompt(
                name="prompt",
                prompt=prompts,
                update_instructions="",
                when_to_update=None,
            )
        ]
    if (
        isinstance(prompts, list)
        and len(prompts) == 1
        and prompts[0].get("when_to_update") is None
    ):
        optimizer = create_prompt_optimizer(
            model,
            kind,
            configurable,
        )
        result = await optimizer(threads, prompts[0], configurable)
        return {"updated_prompts": [result]}
    else:
        optimizer = create_multi_prompt_optimizer(model, kind=kind)
        result = await optimizer(threads, prompts)
        return {"updated_prompts": result}


optimize_prompts = (
    StateGraph(input=InputState, output=OutputState, config_schema=Config)
    .add_node(optimize)
    .add_edge("__start__", "optimize")
    .compile()
)
optimize_prompts.name = "optimize_prompts"
