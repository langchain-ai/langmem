import asyncio
import typing

import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel, Field, model_validator
from trustcall import create_extractor

import langmem.utils as utils
from langmem.prompts import types as prompt_types
from langmem.prompts.gradient import (
    GradientOptimizerConfig,
    create_gradient_prompt_optimizer,
)
from langmem.prompts.metaprompt import (
    MetapromptOptimizerConfig,
    create_metaprompt_optimizer,
)
from langmem.prompts.stateless import PromptMemoryMultiple
from langmem.prompts.types import Prompt

KINDS = typing.Literal["gradient", "metaprompt", "prompt_memory"]


class PromptOptimizerProto(Runnable[prompt_types.OptimizerInput, str]):
    """
    Protocol for a single-prompt optimizer that can be called as:
       await optimizer(trajectories, prompt)
    or
       await optimizer.ainvoke({"trajectories": ..., "prompt": ...})
    returning an updated prompt string.
    """

    async def __call__(
        self,
        trajectories: typing.Sequence[prompt_types.AnnotatedTrajectory] | str,
        prompt: str | Prompt,
    ) -> str: ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional[GradientOptimizerConfig] = None,
) -> PromptOptimizerProto: ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional[MetapromptOptimizerConfig] = None,
) -> PromptOptimizerProto: ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
    config: None = None,
) -> PromptOptimizerProto: ...


def create_prompt_optimizer(
    model: str | BaseChatModel,
    /,
    *,
    kind: KINDS = "gradient",
    config: typing.Union[
        GradientOptimizerConfig, MetapromptOptimizerConfig, None
    ] = None,
) -> PromptOptimizerProto:
    """Create a prompt optimizer that improves prompt effectiveness.

    This function creates an optimizer that can analyze and improve prompts for better
    performance with language models. It supports multiple optimization strategies to
    iteratively enhance prompt quality and effectiveness.

    !!! example "Examples"
        Basic prompt optimization:
        ```python
        from langmem import create_prompt_optimizer

        optimizer = create_prompt_optimizer("anthropic:claude-3-5-sonnet-latest")

        # Example conversation with feedback
        conversation = [
            {"role": "user", "content": "Tell me about the solar system"},
            {"role": "assistant", "content": "The solar system consists of..."},
        ]
        feedback = {"clarity": "needs more structure"}

        # Use conversation history to improve the prompt
        trajectories = [(conversation, feedback)]
        better_prompt = await optimizer.ainvoke(
            {"trajectories": trajectories, "prompt": "You are an astronomy expert"}
        )
        print(better_prompt)
        # Output: 'Provide a comprehensive overview of the solar system...'
        ```

        Optimizing with conversation feedback:
        ```python
        from langmem import create_prompt_optimizer

        optimizer = create_prompt_optimizer(
            "anthropic:claude-3-5-sonnet-latest", kind="prompt_memory"
        )

        # Conversation with feedback about what could be improved
        conversation = [
            {"role": "user", "content": "How do I write a bash script?"},
            {"role": "assistant", "content": "Let me explain bash scripting..."},
        ]
        feedback = "Response should include a code example"

        # Use the conversation and feedback to improve the prompt
        trajectories = [(conversation, {"feedback": feedback})]
        better_prompt = await optimizer(trajectories, "You are a coding assistant")
        print(better_prompt)
        # Output: 'You are a coding assistant that always includes...'
        ```

        Meta-prompt optimization for complex tasks:
        ```python
        from langmem import create_prompt_optimizer

        optimizer = create_prompt_optimizer(
            "anthropic:claude-3-5-sonnet-latest",
            kind="metaprompt",
            config={"max_reflection_steps": 3, "min_reflection_steps": 1},
        )

        # Complex conversation that needs better structure
        conversation = [
            {"role": "user", "content": "Explain quantum computing"},
            {"role": "assistant", "content": "Quantum computing uses..."},
        ]
        feedback = "Need better organization and concrete examples"

        # Optimize with meta-learning
        trajectories = [(conversation, feedback)]
        improved_prompt = await optimizer(
            trajectories, "You are a quantum computing expert"
        )
        ```

    !!! warning
        The optimizer may take longer to run with more complex strategies:
        - gradient: Fastest but may need multiple iterations
        - prompt_memory: Medium speed, depends on conversation history
        - metaprompt: Slowest but most thorough optimization

    !!! tip
        For best results:
        1. Choose the optimization strategy based on your needs:
           - gradient: Good for iterative improvements
           - prompt_memory: Best when you have example conversations
           - metaprompt: Ideal for complex, multi-step tasks
        2. Provide specific feedback in conversation trajectories
        3. Use config options to control optimization behavior
        4. Start with simpler strategies and only use more complex
           ones if needed

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["gradient", "prompt_memory", "metaprompt"]): The optimization
            strategy to use. Each strategy offers different benefits:
            - gradient: Iteratively improves through reflection
            - prompt_memory: Uses successful past prompts
            - metaprompt: Learns optimal patterns via meta-learning
            Defaults to "gradient".
        config (Optional[OptimizerConfig]): Configuration options for the optimizer.
            The type depends on the chosen strategy:
                - GradientOptimizerConfig for kind="gradient"
                - PromptMemoryConfig for kind="prompt_memory"
                - MetapromptOptimizerConfig for kind="metaprompt"
            Defaults to None.

    Returns:
        optimizer (PromptOptimizerProto): A callable that takes conversation trajectories and/or prompts and returns optimized versions.
    """
    if kind == "gradient":
        return create_gradient_prompt_optimizer(model, config)  # type: ignore
    elif kind == "metaprompt":
        return create_metaprompt_optimizer(model, config)  # type: ignore
    elif kind == "prompt_memory":
        return PromptMemoryMultiple(model)  # type: ignore
    else:
        raise NotImplementedError(
            f"Unsupported optimizer kind: {kind}.\nExpected one of {KINDS}"
        )


class MultiPromptOptimizer(
    Runnable[prompt_types.MultiPromptOptimizerInput, list[Prompt]]
):
    def __init__(
        self,
        model: str | BaseChatModel,
        /,
        *,
        kind: typing.Literal["gradient", "prompt_memory", "metaprompt"] = "gradient",
        config: typing.Optional[dict] = None,
    ):
        self.model = model
        self.kind = kind
        self.config = config
        # Build a single-prompt optimizer used internally
        self._optimizer = create_prompt_optimizer(model, kind=kind, config=config)

    async def ainvoke(
        self,
        input: prompt_types.MultiPromptOptimizerInput,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[Prompt]:
        async with ls.trace(
            name="multi_prompt_optimizer.ainvoke",
            inputs=input,
            metadata={"kind": self.kind},
        ) as rt:
            trajectories = input["trajectories"]
            prompts = input["prompts"]

            # Get available prompt names.
            choices = [p["name"] for p in prompts]
            sessions_str = (
                trajectories
                if isinstance(trajectories, str)
                else utils.format_sessions(trajectories)
            )

            # If only one prompt and no explicit when_to_update instruction, simply update it.
            if len(prompts) == 1 and prompts[0].get("when_to_update") is None:
                updated_prompt = await self._optimizer(trajectories, prompts[0])
                rt.add_outputs({"output": [{**prompts[0], "prompt": updated_prompt}]})
                return [{**prompts[0], "prompt": updated_prompt}]

            class Classify(BaseModel):
                """After analyzing the provided trajectories, determine which prompt modules (if any) contributed to degraded performance."""

                reasoning: str = Field(
                    description="Reasoning for which prompts to update."
                )
                which: list[str] = Field(
                    description=f"List of prompt names that should be updated. Must be among {choices}"
                )

                @model_validator(mode="after")
                def validate_choices(self) -> "Classify":
                    invalid = set(self.which) - set(choices)
                    if invalid:
                        raise ValueError(
                            f"Invalid choices: {invalid}. Must be among: {choices}"
                        )
                    return self

            classifier = create_extractor(
                self.model, tools=[Classify], tool_choice="Classify"
            )
            prompt_joined_content = "".join(
                f"{p['name']}: {p['prompt']}\n" for p in prompts
            )
            classification_prompt = f"""Analyze the following trajectories and decide which prompts 
ought to be updated to improve the performance on future trajectories:

{sessions_str}

Below are the prompts being optimized:
{prompt_joined_content}

Return JSON with "which": [...], listing the names of prompts that need updates."""
            result = await classifier.ainvoke(classification_prompt)
            to_update = result["responses"][0].which  # type: ignore

            which_to_update = [p for p in prompts if p["name"] in to_update]

            # Update each chosen prompt concurrently.
            updated_results = await asyncio.gather(
                *[self._optimizer(trajectories, prompt=p) for p in which_to_update]
            )
            updated_map = {
                p["name"]: new_text
                for p, new_text in zip(which_to_update, updated_results)
            }

            # Merge updates back into the prompt list.
            final_list = []
            for p in prompts:
                if p["name"] in updated_map:
                    final_list.append({**p, "prompt": updated_map[p["name"]]})
                else:
                    final_list.append(p)
            rt.add_outputs({"output": final_list})
            return final_list

    def invoke(
        self,
        input: prompt_types.MultiPromptOptimizerInput,
        config: typing.Optional[RunnableConfig] = None,
        **kwargs: typing.Any,
    ) -> list[Prompt]:
        with ls.trace(
            name="multi_prompt_optimizer.invoke",
            inputs=input,
            metadata={"kind": self.kind},
        ) as rt:
            trajectories = input["trajectories"]
            prompts = input["prompts"]

            choices = [p["name"] for p in prompts]
            sessions_str = (
                trajectories
                if isinstance(trajectories, str)
                else utils.format_sessions(trajectories)
            )

            if len(prompts) == 1 and prompts[0].get("when_to_update") is None:
                updated_prompt = self._optimizer.invoke(
                    {"trajectories": trajectories, "prompt": prompts[0]}
                )
                result = [{**prompts[0], "prompt": updated_prompt}]
                rt.add_outputs({"output": result})
                return typing.cast(list[Prompt], result)

            class Classify(BaseModel):
                """After analyzing the provided trajectories, determine which prompt modules (if any) contributed to degraded performance."""

                reasoning: str = Field(
                    description="Reasoning for which prompts to update."
                )
                which: list[str] = Field(
                    description=f"List of prompt names that should be updated. Must be among {choices}"
                )

                @model_validator(mode="after")
                def validate_choices(self) -> "Classify":
                    invalid = set(self.which) - set(choices)
                    if invalid:
                        raise ValueError(
                            f"Invalid choices: {invalid}. Must be among: {choices}"
                        )
                    return self

            classifier = create_extractor(
                self.model, tools=[Classify], tool_choice="Classify"
            )
            prompt_joined_content = "".join(
                f"{p['name']}: {p['prompt']}\n" for p in prompts
            )
            classification_prompt = f"""Analyze the following trajectories and decide which prompts 
ought to be updated to improve the performance on future trajectories:

{sessions_str}

Below are the prompts being optimized:
{prompt_joined_content}

Return JSON with "which": [...], listing the names of prompts that need updates."""
            result = classifier.invoke(classification_prompt)
            to_update = result["responses"][0].which  # type: ignore

            which_to_update = [p for p in prompts if p["name"] in to_update]
            updated_map = {}
            for p in which_to_update:
                updated_text = self._optimizer.invoke(
                    {"trajectories": trajectories, "prompt": p}
                )
                updated_map[p["name"]] = updated_text

            final_list = []
            for p in prompts:
                if p["name"] in updated_map:
                    final_list.append({**p, "prompt": updated_map[p["name"]]})
                else:
                    final_list.append(p)
            rt.add_outputs({"output": final_list})
            return final_list

    async def __call__(
        self,
        trajectories: typing.Sequence[prompt_types.AnnotatedTrajectory] | str,
        prompts: list[Prompt],
    ) -> list[Prompt]:
        """Allow calling the object like: await optimizer(trajectories, prompts)"""
        return await self.ainvoke(
            prompt_types.MultiPromptOptimizerInput(
                trajectories=trajectories, prompts=prompts
            )
        )


def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    /,
    *,
    kind: typing.Literal["gradient", "prompt_memory", "metaprompt"] = "gradient",
    config: typing.Optional[dict] = None,
) -> Runnable[prompt_types.MultiPromptOptimizerInput, list[Prompt]]:
    """Create a multi-prompt optimizer that improves prompt effectiveness.

    This function creates an optimizer that can analyze and improve prompts for better
    performance with language models. It supports multiple optimization strategies to
    iteratively enhance prompt quality and effectiveness.

    !!! example "Examples"
        Basic prompt optimization:
        ```python
        from langmem import create_multi_prompt_optimizer

        optimizer = create_multi_prompt_optimizer("anthropic:claude-3-5-sonnet-latest")

        # Example conversation with feedback
        conversation = [
            {"role": "user", "content": "Tell me about the solar system"},
            {"role": "assistant", "content": "The solar system consists of..."},
        ]
        feedback = {"clarity": "needs more structure"}

        # Use conversation history to improve the prompts
        trajectories = [(conversation, feedback)]
        prompts = [
            {"name": "research", "prompt": "Research the given topic thoroughly"},
            {"name": "summarize", "prompt": "Summarize the research findings"},
        ]
        better_prompts = await optimizer.ainvoke(
            {"trajectories": trajectories, "prompts": prompts}
        )
        print(better_prompts)
        ```

        Optimizing with conversation feedback:
        ```python
        from langmem import create_multi_prompt_optimizer

        optimizer = create_multi_prompt_optimizer(
            "anthropic:claude-3-5-sonnet-latest", kind="prompt_memory"
        )

        # Conversation with feedback about what could be improved
        conversation = [
            {"role": "user", "content": "How do I write a bash script?"},
            {"role": "assistant", "content": "Let me explain bash scripting..."},
        ]
        feedback = "Response should include a code example"

        # Use the conversation and feedback to improve the prompts
        trajectories = [(conversation, {"feedback": feedback})]
        prompts = [
            {"name": "explain", "prompt": "Explain the concept"},
            {"name": "example", "prompt": "Provide a practical example"},
        ]
        better_prompts = await optimizer(trajectories, prompts)
        ```

        Meta-prompt optimization for complex tasks:
        ```python
        from langmem import create_multi_prompt_optimizer

        optimizer = create_multi_prompt_optimizer(
            "anthropic:claude-3-5-sonnet-latest",
            kind="metaprompt",
            config={"max_reflection_steps": 3, "min_reflection_steps": 1},
        )

        # Complex conversation that needs better structure
        conversation = [
            {"role": "user", "content": "Explain quantum computing"},
            {"role": "assistant", "content": "Quantum computing uses..."},
        ]
        feedback = "Need better organization and concrete examples"

        # Optimize with meta-learning
        trajectories = [(conversation, feedback)]
        prompts = [
            {"name": "concept", "prompt": "Explain quantum concepts"},
            {"name": "application", "prompt": "Show practical applications"},
            {"name": "example", "prompt": "Give concrete examples"},
        ]
        improved_prompts = await optimizer(trajectories, prompts)
        ```

    !!! warning
        The optimizer may take longer to run with more complex strategies:
        - gradient: Fastest but may need multiple iterations
        - prompt_memory: Medium speed, depends on conversation history
        - metaprompt: Slowest but most thorough optimization

    !!! tip
        For best results:
        1. Choose the optimization strategy based on your needs:
           - gradient: Good for iterative improvements
           - prompt_memory: Best when you have example conversations
           - metaprompt: Ideal for complex, multi-step tasks
        2. Provide specific feedback in conversation trajectories
        3. Use config options to control optimization behavior
        4. Start with simpler strategies and only use more complex
           ones if needed

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["gradient", "prompt_memory", "metaprompt"]): The optimization
            strategy to use. Each strategy offers different benefits:
            - gradient: Iteratively improves through reflection
            - prompt_memory: Uses successful past prompts
            - metaprompt: Learns optimal patterns via meta-learning
            Defaults to "gradient".
        config (Optional[OptimizerConfig]): Configuration options for the optimizer.
            The type depends on the chosen strategy:
                - GradientOptimizerConfig for kind="gradient"
                - PromptMemoryConfig for kind="prompt_memory"
                - MetapromptOptimizerConfig for kind="metaprompt"
            Defaults to None.

    Returns:
        MultiPromptOptimizer: A Runnable that takes conversation trajectories and prompts
            and returns optimized versions.
    """
    return MultiPromptOptimizer(model, kind=kind, config=config)


__all__ = ["create_prompt_optimizer", "create_multi_prompt_optimizer"]
