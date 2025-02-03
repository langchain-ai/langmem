from typing import Any, Optional, Union

import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from trustcall import create_extractor
from typing_extensions import TypedDict

from langmem import utils
from langmem.prompts import types as prompt_types

DEFAULT_MAX_REFLECTION_STEPS = 5
DEFAULT_MIN_REFLECTION_STEPS = 1

DEFAULT_GRADIENT_PROMPT = """You are reviewing the performance of an AI assistant in a given interaction. 

## Instructions

The current prompt that was used for the session is provided below.

<current_prompt>
{prompt}
</current_prompt>

The developer provided the following instructions around when and how to update the prompt:

<update_instructions>
{update_instructions}
</update_instructions>

## Session data

Analyze the following trajectories (and any associated user feedback) (either conversations with a user or other work that was performed by the assistant):

<trajectories>
{trajectories}
</trajectories>

## Task

Analyze the conversation, including the user’s request and the assistant’s response, and evaluate:
1. How effectively the assistant fulfilled the user’s intent.
2. Where the assistant might have deviated from user expectations or the desired outcome.
3. Specific areas (correctness, completeness, style, tone, alignment, etc.) that need improvement.

If the prompt seems to do well, then no further action is needed. We ONLY recommend updates if there is evidence of failures.
When failures occur, we want to recommend the minimal required changes to fix the problem.

Focus on actionable changes and be concrete.

1. Summarize the key successes and failures in the assistant’s response. 
2. Identify which failure mode(s) best describe the issues (examples: style mismatch, unclear or incomplete instructions, flawed logic or reasoning, hallucination, etc.).
3. Based on these failure modes, recommend the most suitable edit strategy. For example, consider::
   - Use synthetic few-shot examples for style or clarifying decision boundaries.
   - Use explicit instruction updates for conditionals, rules, or logic fixes.
   - Provide step-by-step reasoning guidelines for multi-step logic problems.
4. Provide detailed, concrete suggestions for how to update the prompt accordingly.

But remember, the final updated prompt should only be changed if there is evidence of poor performance, and our recommendations should be minimally invasive.
Do not recommend generic changes that aren't clearly linked to failure modes.

First think through the conversation and critique the current behavior.
If you believe the prompt needs to further adapt to the target context, provide precise recommendations.
Otherwise, mark `warrants_adjustment` as False and respond with 'No recommendations.'"""

DEFAULT_GRADIENT_METAPROMPT = """You are optimizing a prompt to handle its target task more effectively.

<current_prompt>
{current_prompt}
</current_prompt>

We hypothesize the current prompt underperforms for these reasons:

<hypotheses>
{hypotheses}
</hypotheses>

Based on these hypotheses, we recommend the following adjustments:

<recommendations>
{recommendations}
</recommendations>

Respond with the updated prompt. Remember to ONLY make changes that are clearly necessary. Aim to be minimally invasive:"""


class GradientOptimizerConfig(TypedDict, total=False):
    """Configuration for the gradient optimizer."""

    gradient_prompt: str
    metaprompt: str
    max_reflection_steps: int
    min_reflection_steps: int


# For uniformity, let's define the expected "input" structure:
class GradientOptimizerInput(TypedDict, total=False):
    """Input to the gradient optimizer."""

    trajectories: prompt_types.OptimizerInput | str
    prompt: str | prompt_types.Prompt


class GradientPromptOptimizer(Runnable[GradientOptimizerInput, str]):
    """
    Class-based Gradient Prompt Optimizer with both sync and async entry points (invoke/ainvoke).
    Mirrors the logic in create_gradient_prompt_optimizer, but avoids duplication
    by splitting the reflection loop and final update into dedicated sync/async methods.
    """

    def __init__(
        self,
        model: Union[str, BaseChatModel],
        config: Optional[GradientOptimizerConfig] = None,
    ):
        self.model = model
        config = config or {}
        self._config = GradientOptimizerConfig(
            gradient_prompt=config.get("gradient_prompt", DEFAULT_GRADIENT_PROMPT),
            metaprompt=config.get("metaprompt", DEFAULT_GRADIENT_METAPROMPT),
            max_reflection_steps=config.get(
                "max_reflection_steps", DEFAULT_MAX_REFLECTION_STEPS
            ),
            min_reflection_steps=config.get(
                "min_reflection_steps", DEFAULT_MIN_REFLECTION_STEPS
            ),
        )

        def think(thought: str) -> str:
            """A reflection tool, used to reason over complexities and hypothesize fixes."""
            return "Take your time thinking through problems."

        def critique(criticism: str) -> str:
            """A critique tool for diagnosing flaws in reasoning."""
            return "Reflect critically on the previous hypothesis."

        def recommend(
            warrants_adjustment: bool,
            hypotheses: Optional[str] = None,
            full_recommendations: Optional[str] = None,
        ) -> str:
            """
            Decides whether a prompt should be adjusted.
            If warrants_adjustment is True, we incorporate recommended changes.
            If not, we respond 'No recommendations.'
            """
            return ""

        self.just_think_chain = create_extractor(
            model,
            tools=[think, critique],
            tool_choice="any",
        )
        self.any_chain = create_extractor(
            model,
            tools=[think, critique, recommend],
            tool_choice="any",
        )
        self.final_chain = create_extractor(
            model,
            tools=[recommend],
            tool_choice="recommend",
        )

    async def _areact_agent(self, inputs: str) -> Any:
        """
        Async version of the reflection loop.
        Follows the logic of your old react_agent, but coded inline, returning the final "recommend" response object.
        """
        messages = [{"role": "user", "content": inputs}]
        max_steps = self._config["max_reflection_steps"]
        min_steps = self._config["min_reflection_steps"]

        for ix in range(max_steps):
            # Choose chain:
            if ix == max_steps - 1:
                chain = self.final_chain
            elif ix < min_steps:
                chain = self.just_think_chain
            else:
                chain = self.any_chain

            response = await chain.ainvoke(messages)

            # Look for a final "recommend" response in the chain output:
            final_response = next(
                (r for r in response["responses"] if r.__repr_name__() == "recommend"),
                None,
            )
            if final_response:
                return final_response

            # Otherwise keep looping:
            msg: AIMessage = response["messages"][-1]
            messages.append(msg)
            # Insert special "tool" role messages if the AI message invoked tools:
            for tc in msg.tool_calls or []:
                messages.append(
                    {"role": "tool", "content": "", "tool_call_id": tc["id"]}
                )

        raise ValueError(
            f"Failed to generate a final recommendation after {max_steps} attempts"
        )

    def _react_agent(self, inputs: str) -> Any:
        """
        Sync version of the reflection loop.
        We do the same logic but calling chain.invoke() instead of chain.ainvoke().
        """
        messages = [{"role": "user", "content": inputs}]
        max_steps = self._config["max_reflection_steps"]
        min_steps = self._config["min_reflection_steps"]

        for ix in range(max_steps):
            if ix == max_steps - 1:
                chain = self.final_chain
            elif ix < min_steps:
                chain = self.just_think_chain
            else:
                chain = self.any_chain

            response = chain.invoke(messages)

            final_response = next(
                (r for r in response["responses"] if r.__repr_name__() == "recommend"),
                None,
            )
            if final_response:
                return final_response

            msg: AIMessage = response["messages"][-1]
            messages.append(msg)
            for tc in msg.tool_calls or []:
                messages.append(
                    {"role": "tool", "content": "", "tool_call_id": tc["id"]}
                )

        raise ValueError(
            f"Failed to generate a final recommendation after {max_steps} attempts"
        )

    async def _aupdate_prompt(
        self,
        hypotheses: str,
        recommendations: str,
        current_prompt: str,
        update_instructions: str,
    ) -> str:
        """
        Async version of the final update step.
        Uses a specialized extractor with a schema tool to parse the improved prompt.
        """
        schema = utils.get_prompt_extraction_schema(current_prompt)
        extractor = create_extractor(
            self.model,
            tools=[schema],
            tool_choice="OptimizedPromptOutput",
        )
        prompt_input = self._config["metaprompt"].format(
            current_prompt=current_prompt,
            recommendations=recommendations,
            hypotheses=hypotheses,
            update_instructions=update_instructions,
        )
        result = await extractor.ainvoke(prompt_input)
        return result["responses"][0].improved_prompt

    def _update_prompt(
        self,
        hypotheses: str,
        recommendations: str,
        current_prompt: str,
        update_instructions: str,
    ) -> str:
        """Sync version of the final update step."""
        schema = utils.get_prompt_extraction_schema(current_prompt)
        extractor = create_extractor(
            self.model,
            tools=[schema],
            tool_choice="OptimizedPromptOutput",
        )
        prompt_input = self._config["metaprompt"].format(
            current_prompt=current_prompt,
            recommendations=recommendations,
            hypotheses=hypotheses,
            update_instructions=update_instructions,
        )
        result = extractor.invoke(prompt_input)
        return result["responses"][0].improved_prompt

    def _process_input(
        self, input: GradientOptimizerInput
    ) -> tuple[str, str, str, str]:
        """
        Extract prompt_str, sessions_str, feedback, update_instructions from input.
        """
        prompt_data = input["prompt"]
        sessions_data = input["trajectories"]

        if isinstance(prompt_data, str):
            prompt_str = prompt_data
            feedback = ""
            update_instructions = ""
        else:
            prompt_str = prompt_data.get("prompt", "")
            feedback = prompt_data.get("feedback", "")
            update_instructions = prompt_data.get("update_instructions", "")

        if isinstance(sessions_data, str):
            sessions_str = sessions_data
        else:
            sessions_str = utils.format_sessions(sessions_data) if sessions_data else ""

        return prompt_str, sessions_str, feedback, update_instructions

    async def ainvoke(
        self,
        input: GradientOptimizerInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        The async entry point. This is analogous to your old `optimize_prompt`.
        1) Process input
        2) Run reflection steps (async)
        3) If not warrants_adjustment -> return original prompt
           else run final update (async)
        """
        with ls.trace(
            name="gradient_optimizer", inputs=input, metadata={"kind": "gradient"}
        ):
            prompt_str, sessions_str, feedback, update_instructions = (
                self._process_input(input)
            )
            if not sessions_str:
                return prompt_str  # no trajectories => no change

            # Format the initial question to the reflection chain:
            reflection_input = self._config["gradient_prompt"].format(
                trajectories=sessions_str,
                feedback=feedback,
                prompt=prompt_str,
                update_instructions=update_instructions,
            )

            # 1) reflection steps:
            final_response = await self._areact_agent(reflection_input)
            if not final_response.warrants_adjustment:
                return prompt_str

            # 2) final update if warranted:
            improved_prompt = await self._aupdate_prompt(
                final_response.hypotheses,
                final_response.full_recommendations,
                prompt_str,
                update_instructions,
            )
            return improved_prompt

    def invoke(
        self,
        input: GradientOptimizerInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> str:
        """
        The sync entry point: same logic as ainvoke, but calls sync reflection & update.
        """
        with ls.trace(
            name="gradient_optimizer", inputs=input, metadata={"kind": "gradient"}
        ):
            prompt_str, sessions_str, feedback, update_instructions = (
                self._process_input(input)
            )
            if not sessions_str:
                return prompt_str

            reflection_input = self._config["gradient_prompt"].format(
                trajectories=sessions_str,
                feedback=feedback,
                prompt=prompt_str,
                update_instructions=update_instructions,
            )

            # reflection steps (sync):
            final_response = self._react_agent(reflection_input)
            if not final_response.warrants_adjustment:
                return prompt_str

            # final update (sync):
            improved_prompt = self._update_prompt(
                final_response.hypotheses,
                final_response.full_recommendations,
                prompt_str,
                update_instructions,
            )
            return improved_prompt

    async def __call__(
        self,
        trajectories: prompt_types.OptimizerInput | str,
        prompt: Union[str, prompt_types.Prompt],
    ) -> str:
        """
        Allow the object to be called like: await gradient_optimizer(trajectories, prompt).
        This simply defers to `ainvoke` with the required structure.
        """
        return await self.ainvoke({"trajectories": trajectories, "prompt": prompt})


def create_gradient_prompt_optimizer(
    model: Union[str, BaseChatModel], config: Optional[GradientOptimizerConfig] = None
) -> GradientPromptOptimizer:
    """
    Original factory function that just returns the new class-based optimizer.
    """
    return GradientPromptOptimizer(model, config)
