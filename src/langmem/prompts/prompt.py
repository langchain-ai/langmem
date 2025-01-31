from typing_extensions import TypedDict

INSTRUCTION_REFLECTION_PROMPT = """You are helping an AI agent improve. You can do this by changing their system prompt.

These is their current prompt:
<current_prompt>
{current_prompt}
</current_prompt>

Here was the agent's trajectory:
<trajectory>
{trajectory}
</trajectory>

Here is the user's feedback:

<feedback>
{feedback}
</feedback>

Here are instructions for updating the agent's prompt:

<instructions>
{instructions}
</instructions>


Based on this, return an updated prompt

You should return the full prompt, so if there's anything from before that you want to include, make sure to do that. Feel free to override or change anything that seems irrelevant. You do not need to update the prompt - if you don't want to, just return `update_prompt = False` and an empty string for new prompt."""


class GeneralResponse(TypedDict):
    logic: str
    update_prompt: bool
    new_prompt: str


INSTRUCTION_REFLECTION_MULTIPLE_PROMPT = """You are helping an AI agent improve. You can do this by changing their system prompt.

These is their current prompt:
<current_prompt>
{current_prompt}
</current_prompt>

Here are examples of various agent trajectories and associated feedback:
<data>
{data}
</data>

Here are instructions for updating the agent's prompt:

<instructions>
{instructions}
</instructions>


Based on this, return an updated prompt

You should return the full prompt, so if there's anything from before that you want to include, make sure to do that. Feel free to override or change anything that seems irrelevant. You do not need to update the prompt - if you don't want to, just return `update_prompt = False` and an empty string for new prompt."""
