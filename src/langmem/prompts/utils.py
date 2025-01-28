import re
import typing

from langmem import utils
from pydantic import BaseModel, Field, model_validator


def get_trajectory_clean(messages):
    response = []
    for m in messages:
        response.append(m.pretty_repr())
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
            f" <TO_OPTIMIZE> and </TO_OPTIMIZE> tags, in f-string format. Do not include <TO_OPTIMIZE> in your response. {prompt_description}"
        )

        @model_validator(mode="before")
        @classmethod
        def validate_input_variables(cls, data: typing.Any) -> typing.Any:
            assert "improved_prompt" in data
            data["improved_prompt"] = pipeline(data["improved_prompt"])
            return data

    return OptimizedPromptOutput
