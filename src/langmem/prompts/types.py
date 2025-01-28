from typing_extensions import TypedDict, Required


class Prompt(TypedDict, total=False):
    name: Required[str]
    prompt: Required[str]
    update_instructions: str | None
    when_to_update: str | None
