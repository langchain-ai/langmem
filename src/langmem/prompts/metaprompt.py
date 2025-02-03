import typing
from typing import Any, Optional

import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from trustcall import create_extractor
from typing_extensions import TypedDict

from langmem import utils
from langmem.prompts import types as prompt_types
from langmem.prompts.types import Prompt

DEFAULT_MAX_REFLECTION_STEPS = 5
DEFAULT_MIN_REFLECTION_STEPS = 1


class MetapromptOptimizerConfig(TypedDict, total=False):
    """Configuration for the metaprompt optimizer."""

    metaprompt: str
    max_reflection_steps: int
    min_reflection_steps: int


DEFAULT_METAPROMPT = """You are helping an AI assistant learn by optimizing its prompt.

## Background

Below is the current prompt:

<current_prompt>
{prompt}
</current_prompt>

The developer provided these instructions regarding when/how to update:

<update_instructions>
{update_instructions}
</update_instructions>

## Session Data
Analyze the session(s) (and any user feedback) below:

<trajectories>
{trajectories}
</trajectories>

## Instructions

1. Reflect on the agent's performance on the given session(s) and identify any real failure modes (e.g., style mismatch, unclear or incomplete instructions, flawed reasoning, etc.).
2. Recommend the minimal changes necessary to address any real failures. If the prompt performs perfectly, simply respond with the original prompt without making any changes.
3. Retain any f-string variables in the existing prompt exactly as they are (e.g. {{variable_name}}).

IFF changes are warranted, focus on actionable edits. Be concrete. Edits should be appropriate for the identified failure modes. For example, consider synthetic few-shot examples for style or clarifying decision boundaries, or adding or modifying explicit instructions for conditionals, rules, or logic fixes; or provide step-by-step reasoning guidelines for multi-step logic problems if the model is failing to reason appropriately."""


class MetaPromptOptimizer(Runnable[prompt_types.OptimizerInput, str]):
    def __init__(
        self,
        model: typing.Union[str, BaseChatModel],
        config: Optional[MetapromptOptimizerConfig] = None,
    ):
        self.model = model
        self.config = config or {}
        self._final_config = MetapromptOptimizerConfig(
            metaprompt=self.config.get("metaprompt", DEFAULT_METAPROMPT),
            max_reflection_steps=self.config.get(
                "max_reflection_steps", DEFAULT_MAX_REFLECTION_STEPS
            ),
            min_reflection_steps=self.config.get(
                "min_reflection_steps", DEFAULT_MIN_REFLECTION_STEPS
            ),
        )

        # Initialize chains once
        self.reflect_chain = create_extractor(
            model, tools=[self.think, self.critique], tool_choice="any"
        )

    @staticmethod
    def think(thought: str) -> str:
        """Reflection tool implementation."""
        return ""

    @staticmethod
    def critique(criticism: str) -> str:
        """Critique tool implementation."""
        return ""

    async def ainvoke(
        self,
        input: prompt_types.OptimizerInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        async with ls.trace(
            name="optimize_prompt",
            inputs={"input": input} if isinstance(input, str) else input,
            metadata={"kind": "metaprompt"},
        ) as rt:
            prompt_str, update_instructions, sessions_str = (
                self._process_sessions_and_prompt(input)
            )
            if not sessions_str:
                return prompt_str

            result_obj = await self._areflect_then_update(
                sessions_str,
                prompt_str,
                update_instructions,
            )
            result = self._process_result(result_obj, prompt_str)
            rt.add_outputs({"output": result})
            return result

    def invoke(
        self,
        input: prompt_types.OptimizerInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        with ls.trace(
            name="optimize_prompt",
            inputs={"input": input} if isinstance(input, str) else input,
            metadata={"kind": "metaprompt"},
        ) as rt:
            prompt_str, update_instructions, sessions_str = (
                self._process_sessions_and_prompt(input)
            )
            if not sessions_str:
                return prompt_str

            result_obj = self._reflect_then_update(
                sessions_str,
                prompt_str,
                update_instructions,
            )
            result = self._process_result(result_obj, prompt_str)
            rt.add_outputs({"output": result})
            return result

    async def __call__(
        self,
        trajectories: prompt_types.OptimizerInput | str,
        prompt: typing.Union[str, Prompt],
    ) -> str:
        return await self.ainvoke({"trajectories": trajectories, "prompt": prompt})

    def _process_sessions_and_prompt(
        self, input: prompt_types.OptimizerInput
    ) -> tuple[str, str, str]:
        prompt = input["prompt"]
        trajectories = input["trajectories"]

        prompt_str = prompt if isinstance(prompt, str) else prompt.get("prompt", "")
        update_instructions = (
            "" if isinstance(prompt, str) else prompt.get("update_instructions", "")
        )
        sessions_str = (
            trajectories
            if isinstance(trajectories, str)
            else utils.format_sessions(trajectories)
        )
        return prompt_str, update_instructions, sessions_str

    async def _areflect_then_update(
        self,
        sessions_str: str,
        prompt_str: str,
        update_instructions: str,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": self._final_config["metaprompt"].format(
                    prompt=prompt_str,
                    update_instructions=update_instructions,
                    trajectories=sessions_str,
                ),
            }
        ]
        any_chain, final_chain = self._get_extractors(self.model, prompt_str)

        for ix in range(self._final_config["max_reflection_steps"]):
            if ix < self._final_config["max_reflection_steps"] - 1:
                if ix < self._final_config["min_reflection_steps"] - 1:
                    chain = self.reflect_chain
                else:
                    chain = any_chain
                response = await chain.ainvoke(messages)
            else:
                response = await final_chain.ainvoke(messages)
                return response["responses"][0]

            ai_msg: AIMessage = response["messages"][-1]
            messages.append(ai_msg)
            for tc in ai_msg.tool_calls or []:
                messages.append(
                    {"role": "tool", "content": "", "tool_call_id": tc["id"]}
                )

        raise RuntimeError("Exceeded reflection steps without final output")

    def _reflect_then_update(
        self,
        sessions_str: str,
        prompt_str: str,
        update_instructions: str,
    ) -> str:
        # Sync version using invoke
        messages = [
            {
                "role": "user",
                "content": self._final_config["metaprompt"].format(
                    prompt=prompt_str,
                    update_instructions=update_instructions,
                    trajectories=sessions_str,
                ),
            }
        ]
        any_chain, final_chain = self._get_extractors(self.model, prompt_str)
        for ix in range(self._final_config["max_reflection_steps"]):
            if ix < self._final_config["max_reflection_steps"] - 1:
                if ix < self._final_config["min_reflection_steps"] - 1:
                    chain = self.reflect_chain
                else:
                    chain = any_chain
                response = chain.invoke(messages)
            else:
                response = final_chain.invoke(messages)
                return response["responses"][0]

            ai_msg: AIMessage = response["messages"][-1]
            messages.append(ai_msg)
            for tc in ai_msg.tool_calls or []:
                messages.append(
                    {"role": "tool", "content": "", "tool_call_id": tc["id"]}
                )

        raise RuntimeError("Exceeded reflection steps without final output")

    def _process_result(self, result_obj: Any, original_prompt: str) -> str:
        improved_prompt = result_obj.improved_prompt
        if not improved_prompt or improved_prompt.strip().lower().startswith(
            "no recommend"
        ):
            return original_prompt
        return improved_prompt

    def _get_extractors(self, model: str | BaseChatModel, current_prompt: str) -> tuple:
        schema_tool = utils.get_prompt_extraction_schema(current_prompt)
        any_chain = create_extractor(
            model, tools=[self.think, self.critique, schema_tool], tool_choice="any"
        )
        final_chain = create_extractor(
            model, tools=[schema_tool], tool_choice="OptimizedPromptOutput"
        )
        return any_chain, final_chain


def create_metaprompt_optimizer(
    model: str | BaseChatModel, config: MetapromptOptimizerConfig | None = None
):
    """
    Creates a single-step prompt-updater.  If reflect_and_critique=True and max_reflection_steps>1,
    it does some "think/critique" calls before the final 'optimized prompt' call.
    Otherwise it just does one direct call to produce the updated prompt.
    """
    return MetaPromptOptimizer(model, config)
