import asyncio
import typing
from typing import Protocol, runtime_checkable

import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field, model_validator
from trustcall import create_extractor

import langmem.utils as utils
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


@runtime_checkable
class PromptOptimizerProto(Protocol):
    """
    Protocol for a single-prompt optimizer that can be called as:
       await optimizer(sessions, prompt)
    or
       await optimizer.ainvoke({"sessions": ..., "prompt": ...})
    returning an updated prompt string.
    """

    async def __call__(
        self,
        sessions: list[tuple[list[AnyMessage], typing.Optional[dict[str, str]]]] | str,
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
):
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
        sessions = [(conversation, feedback)]
        better_prompt = await optimizer.ainvoke(
            {"sessions": sessions, "prompt": "You are an astronomy expert"}
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
        sessions = [(conversation, {"feedback": feedback})]
        better_prompt = await optimizer(sessions, "You are a coding assistant")
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
        sessions = [(conversation, feedback)]
        improved_prompt = await optimizer(
            sessions, "You are a quantum computing expert"
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
        2. Provide specific feedback in conversation sessions
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
        optimizer (PromptOptimizerProto): A callable that takes conversation sessions and/or prompts and returns optimized versions.
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


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional[GradientOptimizerConfig] = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]: ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional[MetapromptOptimizerConfig] = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]: ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
    config: None = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]: ...


def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    /,
    *,
    kind: typing.Literal["gradient", "prompt_memory", "metaprompt"] = "gradient",
    config: typing.Optional[dict] = None,
) -> typing.Callable[
    [
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str]]
        | str,
        list[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]:
    """Create an optimizer for multiple prompts with shared context.

    This function creates an optimizer that can improve multiple related prompts
    while maintaining consistency and leveraging shared context between them.
    It's particularly useful for optimizing chains of prompts or prompt templates
    that work together in a multi-agent or multi-step system.

    The optimizer analyzes the relationships and dependencies between prompts to ensure
    they work together effectively while maintaining their distinct roles.

    Args:
        model (Union[str, BaseChatModel]): The language model to use for optimization.
            Can be a model name string or a BaseChatModel instance.
        kind (Literal["gradient", "prompt_memory", "metaprompt"]): The optimization
            strategy to use. Each strategy offers different benefits:
            - gradient: Iteratively improves while maintaining consistency
            - prompt_memory: Uses successful prompt combinations
            - metaprompt: Learns optimal patterns for multi-step tasks
            Defaults to "gradient".
        config (Optional[OptimizerConfig]): Configuration options for the optimizer.
            The type depends on the chosen strategy:
            - GradientOptimizerConfig for kind="gradient"
            - PromptMemoryConfig for kind="prompt_memory"
            - MetapromptOptimizerConfig for kind="metaprompt"
            Defaults to None.

    !!! example "Examples"
        Basic multi-prompt optimization:
        ```python
        from langmem import create_multi_prompt_optimizer

        optimizer = create_multi_prompt_optimizer("anthropic:claude-3-5-sonnet-latest")

        prompts = [
            {"name": "research", "prompt": "Research the given topic thoroughly"},
            {"name": "summarize", "prompt": "Summarize the research findings"},
        ]

        # Example conversation showing basic usage
        conversation = [
            {"role": "user", "content": "Tell me about renewable energy"},
            {"role": "assistant", "content": "Here's what I found in my research..."},
            {
                "role": "assistant",
                "content": "To summarize the key points about renewable energy...",
            },
        ]

        # Optimize both prompts together
        # Feedback is optional; without it, the optimizer seeks to infer feedback directly from the converstaion
        sessions = [(conversation, {})]
        better_prompts = await optimizer(sessions, prompts)
        print(better_prompts[0]["prompt"])
        # Output: 'Conduct comprehensive research on the topic...'
        ```

        Optimizing with conversation feedback:
        ```python
        from langmem import create_multi_prompt_optimizer

        optimizer = create_multi_prompt_optimizer(
            "anthropic:claude-3-5-sonnet-latest", kind="prompt_memory"
        )

        # Example conversation showing how prompts work together
        conversation = [
            {"role": "user", "content": "Tell me about quantum computing"},
            {"role": "assistant", "content": "Here's my research..."},
            {"role": "assistant", "content": "To summarize the key points..."},
        ]
        feedback = {
            "research": "Include more technical details",
            "summary": "Make it more accessible",
        }

        # Optimize both prompts based on the conversation
        sessions = [(conversation, feedback)]
        improved_prompts = await optimizer(sessions, prompts)
        ```

        Complex multi-agent optimization:
        ```python
        from langmem import create_multi_prompt_optimizer

        optimizer = create_multi_prompt_optimizer(
            "anthropic:claude-3-5-sonnet-latest",
            kind="metaprompt",
            config={"max_reflection_steps": 3},
        )

        # Define a chain of prompts for a complex task
        prompts = [
            {"name": "planner", "prompt": "Plan the analysis steps"},
            {"name": "researcher", "prompt": "Gather information for each step"},
            {"name": "analyzer", "prompt": "Analyze the gathered information"},
            {"name": "writer", "prompt": "Write the final report"},
        ]

        # Example conversation showing the full workflow
        conversation = [
            {"role": "user", "content": "Analyze the impact of AI on healthcare"},
            {"role": "assistant", "content": "Here's our analysis plan..."},
            {"role": "assistant", "content": "Research findings for each area..."},
            {"role": "assistant", "content": "Analysis of the implications..."},
            {"role": "assistant", "content": "Final report on AI in healthcare..."},
        ]
        feedback = {"organization": "needs better coordination between steps"}

        # Optimize the entire prompt chain
        sessions = [(conversation, feedback)]
        optimized_chain = await optimizer(sessions, prompts)
        ```

    !!! tip
        For effective multi-prompt optimization, provide useful when_to_update instructions to help
        the optimizer know how the prompts relate



    Returns:
        Callable: An async function that takes multiple prompts or messages and returns
        optimized versions.
    """
    _optimizer = create_prompt_optimizer(model, kind=kind, config=config)  # type: ignore

    @ls.traceable
    async def process_multi_prompt_sessions(
        sessions: (
            list[list[AnyMessage]]
            | list[AnyMessage]
            | list[tuple[list[AnyMessage], str]]
            | str
        ),
        prompts: list[Prompt],
    ) -> list[Prompt]:
        choices = [p["name"] for p in prompts]
        sessions_str = utils.format_sessions(sessions)
        if (
            isinstance(prompts, list)
            and len(prompts) == 1
            and prompts[0].get("when_to_update") is None
        ):
            updated_prompt = await _optimizer(sessions, prompts[0])
            return [
                {
                    **prompts[0],
                    "prompt": updated_prompt,
                }
            ]

        class Classify(BaseModel):
            """Classify which prompts merit updating for this conversation."""

            reasoning: str = Field(description="Reasoning for which prompts to update.")

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

        classifier = create_extractor(model, tools=[Classify], tool_choice="Classify")
        prompt_joined_content = "".join(
            f"{p['name']}: {p['prompt']}\n" for p in prompts
        )
        classification_prompt = f"""Analyze the following sessions and decide which prompts 
ought to be updated to improve the performance on future sessions:

{sessions_str}

Below are the prompts being optimized:
{prompt_joined_content}

Return JSON with "which": [...], listing the names of prompts that need updates."""
        result = await classifier.ainvoke(classification_prompt)
        to_update = result["responses"][0].which

        which_to_update = [p for p in prompts if p["name"] in to_update]

        # For each chosen prompt, call the single-prompt optimizer
        updated_results = await asyncio.gather(
            *[_optimizer(sessions, prompt=p) for p in which_to_update]
        )

        # Merge updated prompt text back into original prompt objects
        updated_map = {
            p["name"]: new_text for p, new_text in zip(which_to_update, updated_results)
        }

        final_list = []
        for p in prompts:
            if p["name"] in updated_map:
                final_list.append({**p, "prompt": updated_map[p["name"]]})
            else:
                final_list.append(p)

        return final_list

    return process_multi_prompt_sessions
