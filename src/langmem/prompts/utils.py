import re
import typing

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, model_validator

from langmem import utils


def _get_msg_title_repr(title: str) -> str:
    """Get a title representation for a message.

    Args:
        title: The title.
        bold: Whether to bold the title. Default is False.

    Returns:
        The title representation.
    """
    padded = " " + title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    return f"{sep}{padded}{second_sep}"


def get_trajectory_clean(messages):
    response = []
    for m in messages:
        if isinstance(m, BaseMessage):
            response.append(m.pretty_repr())
        elif isinstance(m, dict) and "role" in m and "content" in m:
            title = _get_msg_title_repr(m["role"])
            name = m.get("name")
            # TODO: handle non-string content.
            if name is not None:
                title += f"\nName: {name}"
            response.append(title + "\n\n" + m["content"])

    return "\n".join(response)


def get_prompt_extraction_schema(
    original_prompt: str,
):
    required_variables = set(re.findall(r"\{(.+?)\}", original_prompt, re.MULTILINE))
    if required_variables:
        variables_str = ", ".join(f"{{{var}}}" for var in required_variables)
        prompt_description = (
            f" The prompt section being optimized contains the following f-string variables to be templated in: {variables_str}."
            " You must retain all of these variables in your improved prompt. No other input variables are allowed."
        )
    else:
        prompt_description = (
            " The prompt section being optimized contains no input f-string variables."
            " Any brackets {{ foo }} you emit will be escaped and not used."
        )

    pipeline = utils.get_var_healer(set(required_variables), all_required=True)

    class OptimizedPromptOutput(BaseModel):
        """Schema for the optimized prompt output."""

        analysis: str = Field(
            description="First, analyze the current results and plan improvements to reconcile them."
        )
        improved_prompt: typing.Optional[str] = Field(
            description="Finally, generate the full updated prompt to address the identified issues. "
            f" {prompt_description}"
        )

        @model_validator(mode="before")
        @classmethod
        def validate_input_variables(cls, data: typing.Any) -> typing.Any:
            assert "improved_prompt" in data
            data["improved_prompt"] = pipeline(data["improved_prompt"])
            return data

    return OptimizedPromptOutput
