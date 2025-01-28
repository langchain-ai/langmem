import asyncio
import typing

import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langmem import utils
from langmem.prompts.stateless import PromptMemoryMultiple
from langmem.prompts.types import Prompt
from langmem.prompts.gradient import (
    GradientOptimizerConfig,
    create_gradient_prompt_optimizer,
)
from langmem.prompts.metaprompt import (
    MetapromptOptimizerConfig,
    create_metaprompt_optimizer,
)
from pydantic import BaseModel, Field, model_validator
from trustcall import create_extractor
from typing import Protocol, runtime_checkable

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
    kind: KINDS = "gradient",
    config: typing.Union[
        GradientOptimizerConfig, MetapromptOptimizerConfig, None
    ] = None,
) -> PromptOptimizerProto:
    """
    Returns an object that can be awaited or .ainvoke(...)'d to produce an updated prompt.

    Example usage:
        optimizer = create_prompt_optimizer(..., kind="gradient")
        new_prompt = await optimizer(sessions, prompt)
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
    kind: KINDS = "gradient",
    config: typing.Union[
        GradientOptimizerConfig, MetapromptOptimizerConfig, None
    ] = None,
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
    """
    Returns an async function that:
       1) Classifies which prompts should be updated
       2) Calls the single-prompt optimizer for each
       3) Returns the updated list of prompts
    """
    _optimizer = create_prompt_optimizer(model, kind, config)

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
        classification_prompt = f"""Analyze the following sessions and decide which prompts 
ought to be updated to improve the performance on future sessions:

{sessions_str}

Below are the prompts being optimized:
{ ''.join(f"{p['name']}: {p['prompt']}\n" for p in prompts) }

Return JSON with "which": [...], listing the names of prompts that need updates.
"""
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
