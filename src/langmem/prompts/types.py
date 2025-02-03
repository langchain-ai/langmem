import typing

from langchain_core.messages import AnyMessage
from typing_extensions import Required, TypedDict


class Prompt(TypedDict, total=False):
    """A structured prompt definition for use in prompt optimization and management.

    This TypedDict defines the structure of a prompt, including its content and metadata
    that helps guide optimization. The 'total=False' flag indicates that only 'name' and
    'prompt' fields are required, while others are optional.

    Attributes:
        name: A unique identifier for the prompt within a chain or collection.
            This helps track and reference specific prompts during optimization.
        prompt: The actual prompt text that will be sent to the language model.
        update_instructions: Optional guidelines for how this prompt should be modified
            during optimization. For example: "Keep technical terms but simplify explanations"
        when_to_update: Optional rules for when this prompt should be modified in relation
            to other prompts. Useful in multi-prompt optimization to maintain dependencies.
            For example: "Update after 'extract' prompt to maintain consistent terminology"
    """

    name: Required[str]
    prompt: Required[str]
    update_instructions: str | None
    when_to_update: str | None


class AnnotatedTrajectory(typing.NamedTuple):
    messages: typing.Sequence[AnyMessage]
    feedback: dict[str, str] | str | None = None


class OptimizerInput(TypedDict):
    trajectories: typing.Sequence[AnnotatedTrajectory] | str
    prompt: str | Prompt


class MultiPromptOptimizerInput(TypedDict):
    """Input for the multi-prompt optimizer."""

    trajectories: typing.Sequence[AnnotatedTrajectory] | str
    prompts: list[Prompt]
