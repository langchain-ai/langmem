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

KINDS = typing.Literal["gradient", "metaprompt", "prompt_memory"]


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional["GradientOptimizerConfig"] = None,
): ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional["MetapromptOptimizerConfig"] = None,
): ...


@typing.overload
def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
): ...


def create_prompt_optimizer(
    model: str | BaseChatModel,
    kind: KINDS = "gradient",
    config: typing.Union[
        "GradientOptimizerConfig", "MetapromptOptimizerConfig", None
    ] = None,
) -> typing.Callable[
    [list[tuple[list[AnyMessage], typing.Optional[dict[str, str]]]], str | Prompt],
    typing.Awaitable[str],
]:
    if kind == "gradient":
        return create_gradient_prompt_optimizer(model, config)  # type: ignore
    elif kind == "metaprompt":
        return create_metaprompt_optimizer(model, config)  # type: ignore
    elif kind == "prompt_memory":
        return PromptMemoryMultiple(model).areflect  # type: ignore
    else:
        raise NotImplementedError(
            f"Unsupported optimizer kind: {kind}.\nExpected one of {KINDS}"
        )


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["gradient"] = "gradient",
    config: typing.Optional["GradientOptimizerConfig"] = None,
): ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["metaprompt"] = "metaprompt",
    config: typing.Optional["MetapromptOptimizerConfig"] = None,
): ...


@typing.overload
def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: typing.Literal["prompt_memory"] = "prompt_memory",
): ...


def create_multi_prompt_optimizer(
    model: str | BaseChatModel,
    kind: KINDS = "gradient",
    config: typing.Union[
        "GradientOptimizerConfig", "MetapromptOptimizerConfig", None
    ] = None,
) -> typing.Callable[
    [
        list[tuple[list[AnyMessage], typing.Optional[dict[str, str]]]],
        typing.Sequence[Prompt],
    ],
    typing.Awaitable[list[Prompt]],
]:
    _optimizer = create_prompt_optimizer(model, kind, config)
    # list[tuple[list[AnyMessage], typing.Optional[dict[str, str]]]]

    @ls.traceable
    async def process_multi_prompt_sessions(
        sessions: (
            list[list[AnyMessage]]
            | list[AnyMessage]
            | list[tuple[list[AnyMessage], str]]
            | tuple[list[AnyMessage], str]
            | str
        ),
        prompts: list[Prompt],
    ):
        choices = [p["name"] for p in prompts]
        sessions = utils.format_sessions(sessions)

        class Classify(BaseModel):
            """Classify which prompts merit updating for this conversation."""

            reasoning: str = Field(
                description="Reasoning for classifying which prompts merit updating. Cite any relevant evidence."
            )

            which: list[str] = Field(
                description=f"List of prompt names that should be updated. Must be one or more of: {choices}"
            )

            @model_validator(mode="after")
            def validate_choices(self) -> "Classify":
                invalid = set(self.which) - set(choices)
                if invalid:
                    raise ValueError(
                        f"Invalid choices: {invalid}. Must be one of: {choices}"
                    )
                return self

        classifier = create_extractor(model, tools=[Classify], tool_choice="Classify")
        prompts_str = "\n\n".join(f"{p['name']}: {p['prompt']}" for p in prompts)
        result = await classifier.ainvoke(
            f"""Analyze the following sessions and decide which prompts ought to be updated to improve the performance on future sessions:
{sessions}

Below are the prompts being optimized:
{prompts_str}
Consider any instructions on when_to_update when making a decision.
"""
        )
        to_update = result["responses"][0].which
        which_to_update = [p for p in prompts if p["name"] in to_update]
        results = await asyncio.gather(
            *(_optimizer(sessions, prompt=p) for p in which_to_update)
        )
        updated = {p["name"]: r for p, r in zip(which_to_update, results)}
        # Return the final prompts
        final = []
        for p in prompts:
            if p["name"] in updated:
                final.append({**p, "prompt": updated[p["name"]]})
            else:
                final.append(p)
        return final

    return process_multi_prompt_sessions
