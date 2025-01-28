import re
import typing
import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.messages.utils import merge_message_runs
from langchain_core.messages import AnyMessage
from langgraph.utils.config import get_config
from pydantic import BaseModel, Field, model_validator


class NamespaceTemplate:
    __slots__ = ("template", "vars")

    def __init__(self, template: tuple[str, ...]):
        self.template = template
        self.vars = {
            ix: _get_key(ns)
            for ix, ns in enumerate(template)
            if _get_key(ns) is not None
        }

    def __call__(self, config: RunnableConfig | None = None):
        config = config or get_config()
        if self.vars:
            configurable = config["configurable"] if "configurable" in config else {}
            return tuple(
                configurable.get(self.vars[ix], ns) if ix in self.vars else ns
                for ix, ns in enumerate(self.template)
            )
        else:
            return self.template


def _get_key(ns: str):
    return ns.strip(r"{}") if isinstance(ns, str) and ns.startswith("{") else None


def get_conversation(messages: list):
    merged = merge_message_runs(messages)
    return "\n\n".join(m.pretty_repr() for m in merged)


# List[Tuple[List[AnyMessage], Dict[str, Any]]]


def format_sessions(
    sessions: (
        list[list[AnyMessage]]
        | list[AnyMessage]
        | list[tuple[list[AnyMessage], str | dict[str, typing.Any]]]
        | tuple[list[AnyMessage], str | dict[str, typing.Any]]
        | str
    ),
):
    if not sessions:
        return ""

    if isinstance(sessions, str):
        sessions = [(sessions, "")]
    elif isinstance(sessions, tuple) and isinstance(sessions[0], list):
        sessions = [sessions]
    if not isinstance(sessions, list):
        raise ValueError(
            f"Expected list of session, feedback pairs, but got: {type(sessions)} {sessions}"
        )
    collected = []
    for session_ in sessions:
        if isinstance(session_, (list, tuple)) and len(session_) > 1:
            collected.append((session_[0], session_[1]))
        else:
            collected.append((session_, ""))

    acc = []
    ids_ = [uuid.uuid4().hex for _ in sessions]
    for id_, (session, feedback) in zip(ids_, collected):
        if feedback:
            feedback = (
                f"\n\nFeedback for session {id_}:\n<FEEDBACK>\n{feedback}\n</FEEDBACK>"
            )
        acc.append(
            f"<session_{id_}>\n{get_conversation(session)}{feedback}\n</session_{id_}>"
        )
    return "\n\n".join(acc)


def get_var_healer(vars: set[str] | str, all_required: bool = False):
    if isinstance(vars, str):
        vars = set(re.findall(r"\{(.+?)\}", vars, re.MULTILINE))
    var_to_uuid = {f"{{{v}}}": uuid.uuid4().hex for v in vars}
    uuid_to_var = {v: k for k, v in var_to_uuid.items()}

    def escape(input_string: str) -> str:
        result = re.sub(r"(?<!\{)\{(?!\{)", "{{", input_string)
        result = re.sub(r"(?<!\})\}(?!\})", "}}", result)
        return result

    if not vars:
        return escape

    mask_pattern = re.compile("|".join(map(re.escape, var_to_uuid.keys())))
    unmask_pattern = re.compile("|".join(map(re.escape, var_to_uuid.values())))

    strip_to_optimize_pattern = re.compile(
        r"<TO_OPTIMIZE.*?>|</TO_OPTIMIZE>", re.MULTILINE | re.DOTALL
    )

    def assert_all_required(input_string: str) -> str:
        if not all_required:
            return input_string

        missing = [var for var in vars if f"{{{var}}}" not in input_string]
        if missing:
            raise ValueError(f"Missing required variable: {', '.join(missing)}")

        return input_string

    def mask(input_string: str) -> str:
        return mask_pattern.sub(lambda m: var_to_uuid[m.group(0)], input_string)

    def unmask(input_string: str) -> str:
        return unmask_pattern.sub(lambda m: uuid_to_var[m.group(0)], input_string)

    def pipe(input_string: str) -> str:
        return unmask(
            strip_to_optimize_pattern.sub(
                "", escape(mask(assert_all_required(input_string)))
            )
        )

    return pipe


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

    pipeline = get_var_healer(set(required_variables), all_required=True)

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
