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

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["gradient", "prompt_memory", "metaprompt"]): The optimization
            strategy to use. Each strategy offers different benefits:

            - gradient: Separates concerns between finding areas for improvement
                and recommending updates
            - prompt_memory: Simple single-shot metaprompt
            - metaprompt: Supports reflection but each step is a single LLM call.
        config (Optional[OptimizerConfig]): Configuration options for the optimizer.
            The type depends on the chosen strategy:

                - GradientOptimizerConfig for kind="gradient"
                - PromptMemoryConfig for kind="prompt_memory"
                - MetapromptOptimizerConfig for kind="metaprompt"
            Defaults to None.

    Returns:
        optimizer (PromptOptimizerProto): A callable that takes conversation trajectories and/or prompts and returns optimized versions.

    ## Optimization Strategies

    ### 1. Gradient Optimizer
    ```mermaid
    sequenceDiagram
        participant U as User
        participant O as Optimizer
        participant R as Reflection
        participant U2 as Update

        U->>O: Prompt + Feedback
        loop For min_steps to max_steps
            O->>R: Think/Critique Current State
            R-->>O: Proposed Improvements
            O->>U2: Apply Update
            U2-->>O: Updated Prompt
        end
        O->>U: Final Optimized Prompt
    ```

    The gradient optimizer uses reflection to propose improvements:

    1. Analyzes prompt and feedback through reflection cycles
    2. Proposes specific improvements
    3. Applies single-step updates

    Configuration (GradientOptimizerConfig):

    - gradient_prompt: Custom prompt for predicting "what to improve"
    - metaprompt: Custom prompt for applying the improvements
    - max_reflection_steps: Maximum reflection iterations (default: 3)
    - min_reflection_steps: Minimum reflection iterations (default: 1)

    ### 2. Meta-Prompt Optimizer
    ```mermaid
    sequenceDiagram
        participant U as User
        participant M as MetaOptimizer
        participant A as Analysis
        participant U2 as Update

        U->>M: Prompt + Examples
        M->>A: Analyze Examples
        A-->>M: Proposed Update
        M->>U2: Apply Update
        U2-->>U: Enhanced Prompt
    ```

    Uses meta-learning to directly propose updates:

    1. Analyzes examples to understand patterns
    2. Proposes direct prompt updates
    3. Applies updates in a single step

    Configuration (MetapromptOptimizerConfig):

    - metaprompt: Custom instructions on how to update the prompt
    - max_reflection_steps: Maximum meta-learning steps (default: 3)
    - min_reflection_steps: Minimum meta-learning steps (default: 1)

    ### 3. Prompt Memory Optimizer
    ```mermaid
    sequenceDiagram
        participant U as User
        participant P as PromptMemory
        participant M as Memory

        U->>P: Prompt + History
        P->>M: Extract Patterns
        M-->>P: Success Patterns
        P->>U: Updated Prompt
    ```

    Learns from conversation history:
    1. Extracts successful patterns from past interactions
    2. Identifies improvement areas from feedback
    3. Applies learned patterns to new prompts

    No additional configuration required.

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

    !!! warning "Performance Considerations"

        Each strategy has different LLM call patterns:

        - prompt_memory: 1 LLM call total
            - Fastest as it only needs one pass
        - metaprompt: 1-5 LLM calls (configurable)
            - Each step is one LLM call
            - Default range: min 2, max 5 reflection steps
        - gradient: 2-10 LLM calls (configurable)
            - Each step requires 2 LLM calls (think + critique)
            - Default range: min 2, max 5 reflection steps

    !!! tip "Strategy Selection"
        Choose based on your needs:
        1. Prompt Memory: Simplest prompting strategy
            - Limited ability to learn from complicated patterns
        2. Metaprompt: Balance of speed and improvement
            - Moderate cost (2-5 LLM calls)
        3. Gradient: Most thorough but expensive
            - Highest cost (4-10 LLM calls)
            - Uses separation of concerns to extract feedback from more conversational context.
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

    This function creates an optimizer that can analyze and improve multiple prompts
    simultaneously using the same optimization strategy. Each prompt is optimized using
    the selected strategy (see `create_prompt_optimizer` for strategy details).

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

    ```mermaid
    sequenceDiagram
        participant U as User
        participant M as Multi-prompt Optimizer
        participant C as Credit Assigner
        participant O as Single-prompt Optimizer
        participant P as Prompts

        U->>M: Annotated Trajectories + Prompts
        activate M
        Note over M: Using pre-initialized<br/>single-prompt optimizer

        M->>C: Analyze trajectories
        activate C
        Note over C: Determine which prompts<br/>need improvement
        C-->>M: Credit assignment results
        deactivate C

        loop For each prompt needing update
            M->>O: Optimize prompt
            activate O
            O->>P: Apply optimization strategy
            Note over O,P: Gradient/Memory/Meta<br/>optimization
            P-->>O: Optimized prompt
            O-->>M: Return result
            deactivate O
        end

        M->>U: Return optimized prompts
        deactivate M
    ```

    The system optimizer:

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

        Controlling the max number of reflection steps:
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
        # Explicit feedback is optional
        feedback = None

        # Optimize with meta-learning
        trajectories = [(conversation, feedback)]
        prompts = [
            {"name": "concept", "prompt": "Explain quantum concepts"},
            {"name": "application", "prompt": "Show practical applications"},
            {"name": "example", "prompt": "Give concrete examples"},
        ]
        improved_prompts = await optimizer(trajectories, prompts)
        ```
    """
    return MultiPromptOptimizer(model, kind=kind, config=config)


__all__ = ["create_prompt_optimizer", "create_multi_prompt_optimizer"]
