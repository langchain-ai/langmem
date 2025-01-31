from typing import Optional

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore

from langmem.prompts.prompt import INSTRUCTION_REFLECTION_PROMPT, GeneralResponse
from langmem.prompts.utils import get_trajectory_clean


class ReflectionState(MessagesState):
    feedback: Optional[str]
    instructions: str


async def update_general(state: ReflectionState, config, store: BaseStore):
    reflection_model = init_chat_model(
        "claude-3-5-sonnet-latest", model_provider="anthropic", temperature=0
    )
    namespace = tuple(config["configurable"]["namespace"])
    key = config["prompt_key"]
    result = await store.aget(namespace, key)

    async def get_output(messages, current_prompt, feedback, instructions):
        trajectory = get_trajectory_clean(messages)
        prompt = INSTRUCTION_REFLECTION_PROMPT.format(
            current_prompt=current_prompt,
            trajectory=trajectory,
            feedback=feedback,
            instructions=instructions,
        )
        _output = await reflection_model.with_structured_output(
            GeneralResponse, method="json_schema"
        ).ainvoke(
            prompt,
            config={"configurable": config["configurable"].get("model_config", {})},
        )
        return _output

    output = await get_output(
        state["messages"],
        result.value["data"],
        state["feedback"],
        state["instructions"],
    )
    if output["update_prompt"]:
        await store.aput(namespace, key, {"data": output["new_prompt"]}, index=False)


general_reflection_graph = StateGraph(ReflectionState)
general_reflection_graph.add_node(update_general)
general_reflection_graph.add_edge(START, "update_general")
general_reflection_graph.add_edge("update_general", END)
general_reflection_graph = general_reflection_graph.compile()
