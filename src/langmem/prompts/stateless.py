from langchain.chat_models import init_chat_model
from langmem.prompts.prompt import (
    INSTRUCTION_REFLECTION_PROMPT,
    GeneralResponse,
    INSTRUCTION_REFLECTION_MULTIPLE_PROMPT,
)
from langmem.prompts.utils import get_trajectory_clean
from langsmith import traceable
from langmem.utils import get_var_healer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langmem import Prompt


class PromptMemory:
    def __init__(self, model=None):
        if model is not None:
            if isinstance(model, str):
                model = init_chat_model(model, temperature=0)
            self.model = model.with_structured_output(
                GeneralResponse, method="json_schema"
            )
        else:
            self.model = init_chat_model(
                "claude-3-5-sonnet-latest", model_provider="anthropic", temperature=0
            ).with_structured_output(GeneralResponse, method="json_schema")

    @traceable
    def reflect(
        self, messages, current_prompt: str, feedback: str = "", instructions: str = ""
    ):
        trajectory = get_trajectory_clean(messages)
        prompt = INSTRUCTION_REFLECTION_PROMPT.format(
            current_prompt=current_prompt,
            trajectory=trajectory,
            feedback=feedback,
            instructions=instructions,
        )
        _output = self.model.invoke(prompt)
        return _output.new_prompt

    @traceable
    async def areflect(
        self, messages, current_prompt: str, feedback: str = "", instructions: str = ""
    ):
        trajectory = get_trajectory_clean(messages)
        prompt = INSTRUCTION_REFLECTION_PROMPT.format(
            current_prompt=current_prompt,
            trajectory=trajectory,
            feedback=feedback,
            instructions=instructions,
        )
        _output = await self.model.ainvoke(prompt)
        return _output.new_prompt


class PromptMemoryMultiple:
    def __init__(self, model=None):
        if model is not None:
            if isinstance(model, str):
                model = init_chat_model(model, temperature=0)
            self.model = model.with_structured_output(
                GeneralResponse, method="json_schema"
            )
        else:
            self.model = init_chat_model(
                "claude-3-5-sonnet-latest", model_provider="anthropic", temperature=0
            ).with_structured_output(GeneralResponse, method="json_schema")

    @staticmethod
    def _get_data(trajectories_with_feedback):
        if isinstance(trajectories_with_feedback, str):
            return trajectories_with_feedback
        data = []
        for i, (messages, feedback) in enumerate(trajectories_with_feedback):
            trajectory = get_trajectory_clean(messages)
            data.append(
                f"<trajectory {i}>\n{trajectory}\n</trajectory {i}>\n<feedback {i}>\n{feedback}\n</feedback {i}>"
            )
        return "\n".join(data)

    @traceable
    def reflect(self, trajectories_with_feedback, prompt: "Prompt"):
        data = self._get_data(trajectories_with_feedback)
        healer = get_var_healer(prompt["prompt"])
        prompt = INSTRUCTION_REFLECTION_MULTIPLE_PROMPT.format(
            current_prompt=prompt["prompt"],
            data=data,
            instructions=prompt["update_instructions"],
        )
        _output = self.model.invoke(prompt)
        return healer(_output["new_prompt"])

    @traceable
    async def areflect(self, trajectories_with_feedback, prompt: "Prompt"):
        data = self._get_data(trajectories_with_feedback)
        healer = get_var_healer(prompt["prompt"])
        prompt = INSTRUCTION_REFLECTION_MULTIPLE_PROMPT.format(
            current_prompt=prompt["prompt"],
            data=data,
            instructions=prompt["update_instructions"],
        )
        _output = await self.model.ainvoke(prompt)
        return healer(_output["new_prompt"])
