from typing import TYPE_CHECKING

from langchain.chat_models import init_chat_model

from langmem.prompts.prompt import (
    INSTRUCTION_REFLECTION_MULTIPLE_PROMPT,
    INSTRUCTION_REFLECTION_PROMPT,
    GeneralResponse,
)
from langmem.prompts.utils import get_trajectory_clean
from langmem.utils import get_var_healer

if TYPE_CHECKING:
    pass

from typing import Any, List, Optional, Tuple, Union

import langsmith as ls
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict


# For type clarity, define a typed dict for the input:
class SingleMemoryInput(TypedDict, total=False):
    """Represents the input to PromptMemory.invoke / PromptMemory.ainvoke."""

    messages: Any
    current_prompt: str
    feedback: str
    instructions: str


class PromptMemory(Runnable[SingleMemoryInput, str]):
    """
    Class that encapsulates the old single-trajectory reflection logic.
    Provides sync (invoke) and async (ainvoke) methods.
    """

    def __init__(self, model: Optional[Union[str, Any]] = None):
        """
        If 'model' is None, default is used.
        If 'model' is a str, pass it to init_chat_model(...).
        Otherwise we assume 'model' is already an initialized chat model.
        """
        if model is None:
            # Example: default to some Claude-based model
            model = init_chat_model(
                "claude-3-5-sonnet-latest",
                model_provider="anthropic",
                temperature=0,
            )
        elif isinstance(model, str):
            model = init_chat_model(model, temperature=0)

        # Extend it to produce structured output
        self.model = model.with_structured_output(GeneralResponse, method="json_schema")

    def invoke(
        self,
        input: SingleMemoryInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous reflection method.
        """
        with ls.trace(name="prompt_memory_reflect", inputs=input):
            # Grab the fields from 'input'
            messages = input["messages"]
            current_prompt = input.get("current_prompt", "")
            feedback = input.get("feedback", "")
            instructions = input.get("instructions", "")

            # Format the reflection prompt
            trajectory = get_trajectory_clean(messages)
            prompt_str = INSTRUCTION_REFLECTION_PROMPT.format(
                current_prompt=current_prompt,
                trajectory=trajectory,
                feedback=feedback,
                instructions=instructions,
            )

            # Invoke model synchronously
            _output = self.model.invoke(prompt_str)
            new_prompt = _output.new_prompt  # from the GeneralResponse schema

            return new_prompt

    async def ainvoke(
        self,
        input: SingleMemoryInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async reflection method.
        """
        async with ls.trace(name="prompt_memory_reflect", inputs=input):
            messages = input["messages"]
            current_prompt = input.get("current_prompt", "")
            feedback = input.get("feedback", "")
            instructions = input.get("instructions", "")

            trajectory = get_trajectory_clean(messages)
            prompt_str = INSTRUCTION_REFLECTION_PROMPT.format(
                current_prompt=current_prompt,
                trajectory=trajectory,
                feedback=feedback,
                instructions=instructions,
            )

            _output = await self.model.ainvoke(prompt_str)
            return _output.new_prompt

    async def __call__(
        self,
        messages: Any,
        current_prompt: str,
        feedback: str = "",
        instructions: str = "",
    ) -> str:
        """
        Convenience method allowing `await prompt_memory(messages, prompt, ...)`.
        Simply forwards to ainvoke.
        """
        return await self.ainvoke(
            {
                "messages": messages,
                "current_prompt": current_prompt,
                "feedback": feedback,
                "instructions": instructions,
            }
        )


class MultipleMemoryInput(TypedDict, total=False):
    """
    Input to PromptMemoryMultiple.invoke / .ainvoke.
    `trajectories_with_feedback` can be either:
       - A string
       - Or a list of (messages, feedback) pairs
    `prompt` is a dict with "prompt" and "update_instructions" fields
    (mirroring your older usage).
    """

    trajectories_with_feedback: Union[str, List[Tuple[Any, str]]]
    prompt: dict


class PromptMemoryMultiple(Runnable[MultipleMemoryInput, str]):
    """
    Class-based approach for multi-trajectory reflection, with sync/async entry points.
    """

    def __init__(self, model: Optional[Union[str, Any]] = None):
        if model is None:
            model = init_chat_model(
                "claude-3-5-sonnet-latest", model_provider="anthropic", temperature=0
            )
        elif isinstance(model, str):
            model = init_chat_model(model, temperature=0)

        self.model = model.with_structured_output(GeneralResponse, method="json_schema")

    @staticmethod
    def _get_data(trajectories_with_feedback: Union[str, List[Tuple[Any, str]]]) -> str:
        """
        Re-implements your old code to produce the combined string:
          <trajectory i>...</trajectory i>
          <feedback i>...</feedback i>
        or pass through if it's already a string.
        """
        if isinstance(trajectories_with_feedback, str):
            return trajectories_with_feedback

        data_pieces = []
        for i, (messages, feedback) in enumerate(trajectories_with_feedback):
            trajectory = get_trajectory_clean(messages)
            data_pieces.append(
                f"<trajectory {i}>\n{trajectory}\n</trajectory {i}>\n"
                f"<feedback {i}>\n{feedback}\n</feedback {i}>"
            )
        return "\n".join(data_pieces)

    def invoke(
        self,
        input: MultipleMemoryInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        Sync reflection over multiple trajectories.
        """
        with ls.trace(name="prompt_memory_multiple_reflect", inputs=input):
            trajectories_with_feedback = input["trajectories_with_feedback"]
            prompt_data = input["prompt"]  # same shape as old 'Prompt'

            data_str = self._get_data(trajectories_with_feedback)
            healer = get_var_healer(prompt_data["prompt"])

            prompt_str = INSTRUCTION_REFLECTION_MULTIPLE_PROMPT.format(
                current_prompt=prompt_data["prompt"],
                data=data_str,
                instructions=prompt_data.get("update_instructions", ""),
            )

            _output = self.model.invoke(prompt_str)
            # The "new_prompt" field in the JSON schema
            new_prompt = _output["new_prompt"]
            return healer(new_prompt)

    async def ainvoke(
        self,
        input: MultipleMemoryInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async reflection over multiple trajectories.
        """
        async with ls.trace(name="prompt_memory_multiple_reflect", inputs=input):
            trajectories_with_feedback = input["trajectories_with_feedback"]
            prompt_data = input["prompt"]

            data_str = self._get_data(trajectories_with_feedback)
            healer = get_var_healer(prompt_data["prompt"])

            prompt_str = INSTRUCTION_REFLECTION_MULTIPLE_PROMPT.format(
                current_prompt=prompt_data["prompt"],
                data=data_str,
                instructions=prompt_data.get("update_instructions", ""),
            )

            _output = await self.model.ainvoke(prompt_str)
            return healer(_output["new_prompt"])

    async def __call__(
        self,
        trajectories_with_feedback: Union[str, List[Tuple[Any, str]]],
        prompt: dict,
    ) -> str:
        """
        Convenience: `await pmem_multi(trajectories_with_feedback, prompt)`.
        Defers to `ainvoke`.
        """
        return await self.ainvoke(
            {
                "trajectories_with_feedback": trajectories_with_feedback,
                "prompt": prompt,
            }
        )
