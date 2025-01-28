import langsmith as ls
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage
from langmem import utils
from langmem.prompts import types
from trustcall import create_extractor
from typing_extensions import TypedDict

DEFAULT_MAX_REFLECTION_STEPS = 5
DEFAULT_MIN_REFLECTION_STEPS = 1

DEFAULT_GRADIENT_PROMPT = """You are reviewing the performance of an AI assistant in a given interaction. 

## Instructions

The current prompt that was used for the session is provided below.

<current_prompt>
{prompt}
</current_prompt>

The developer provided the following instructions around when and how to update the prompt:

<update_instructions>
{update_instructions}
</update_instructions>

## Session data

Analyze the following sessions (and any associated user feedback) (either conversations with a user or other work that was performed by the assistant):

<sessions>
{sessions}
</sessions>

## Feedback

The following feedback is provided for this session:

<feedback>
{feedback}
</feedback>

## Task

Analyze the conversation, including the user’s request and the assistant’s response, and evaluate:
1. How effectively the assistant fulfilled the user’s intent.
2. Where the assistant might have deviated from user expectations or the desired outcome.
3. Specific areas (correctness, completeness, style, tone, alignment, etc.) that need improvement.

If the prompt seems to do well, then no further action is needed. We ONLY recommend updates if there is evidence of failures.
When failures occur, we want to recommend the minimal required changes to fix the problem.

Focus on actionable changes and be concrete.

1. Summarize the key successes and failures in the assistant’s response. 
2. Identify which failure mode(s) best describe the issues (examples: style mismatch, unclear or incomplete instructions, flawed logic or reasoning, hallucination, etc.).
3. Based on these failure modes, recommend the most suitable edit strategy. For example, consider::
   - Use synthetic few-shot examples for style or clarifying decision boundaries.
   - Use explicit instruction updates for conditionals, rules, or logic fixes.
   - Provide step-by-step reasoning guidelines for multi-step logic problems.
4. Provide detailed, concrete suggestions for how to update the prompt accordingly.

But remember, the final updated prompt should only be changed if there is evidence of poor performance, and our recommendations should be minimally invasive.
Do not recommend generic changes that aren't clearly linked to failure modes.

First think through the conversation and critique the current behavior.
If you believe the prompt needs to further adapt to the target context, provide precise recommendations.
Otherwise, mark `warrants_adjustment` as False and respond with 'No recommendations.'"""


DEFAULT_GRADIENT_METAPROMPT = """You are optimizing a prompt to handle its target task more effectively.

<current_prompt>
{current_prompt}
</current_prompt>

We hypothesize the current prompt underperforms for these reasons:

<hypotheses>
{hypotheses}
</hypotheses>

Based on these hypotheses, we recommend the following adjustments:

<recommendations>
{recommendations}
</recommendations>

Respond with the updated prompt. Remember to ONLY make changes that are clearly necessary. Aim to be minimally invasive:"""


class GradientOptimizerConfig(TypedDict, total=False):
    """Configuration for the gradient optimizer."""

    gradient_prompt: str
    metaprompt: str
    max_reflection_steps: int
    min_reflection_steps: int


def create_gradient_prompt_optimizer(
    model: str | BaseChatModel, config: GradientOptimizerConfig | None = None
):
    config = config or GradientOptimizerConfig()
    config = GradientOptimizerConfig(
        gradient_prompt=config.get("gradient_prompt", DEFAULT_GRADIENT_PROMPT),
        metaprompt=config.get("metaprompt", DEFAULT_GRADIENT_METAPROMPT),
        max_reflection_steps=config.get(
            "max_reflection_steps", DEFAULT_MAX_REFLECTION_STEPS
        ),
        min_reflection_steps=config.get(
            "min_reflection_steps", DEFAULT_MIN_REFLECTION_STEPS
        ),
    )

    @ls.traceable
    async def react_agent(
        model: str | BaseChatModel, inputs: str, max_steps: int, min_steps: int
    ):
        messages = [
            {"role": "user", "content": inputs},
        ]
        just_think = create_extractor(
            model,
            tools=[think, critique],
            tool_choice="any",
        )
        any_chain = create_extractor(
            model,
            tools=[think, critique, recommend],
            tool_choice="any",
        )
        final_chain = create_extractor(
            model,
            tools=[recommend],
            tool_choice="recommend",
        )
        for ix in range(max_steps):
            if ix == max_steps - 1:
                chain = final_chain
            elif ix < min_steps:
                chain = just_think
            else:
                chain = any_chain
            response = await chain.ainvoke(messages)
            final_response = next(
                (r for r in response["responses"] if r.__repr_name__() == "recommend"),
                None,
            )
            if final_response:
                return final_response
            msg: AIMessage = response["messages"][-1]
            messages.append(msg)
            ids = [tc["id"] for tc in (msg.tool_calls or [])]
            for id_ in ids:
                messages.append({"role": "tool", "content": "", "tool_call_id": id_})

        raise ValueError(f"Failed to generate response after {max_steps} attempts")

    def think(thought: str):
        """First call this to reason over complicated domains, uncover hidden input/output patterns, theorize why previous hypotheses failed, and creatively conduct error analyses (e.g., deep diagnostics/recursively analyzing "why" something failed). List characteristics of the data generating process you failed to notice before. Hypothesize fixes, prioritize, critique, and repeat calling this tool until you are confident in your next solution."""
        return "Take as much time as you need! If you're stuck, take a step back and try something new."

    def critique(criticism: str):
        """Then, critique your thoughts and hypotheses. Identify flaws in your previous hypotheses and current thinking. Forecast why the hypotheses won't work. Get to the bottom of what is really driving the problem. This tool returns no new information but gives you more time to plan."""
        return "Take as much time as you need. It's important to think through different strategies."

    def recommend(
        warrants_adjustment: bool,
        hypotheses: str | None = None,
        full_recommendations: str | None = None,
    ):
        """Once you've finished thinking, decide whether the session indicates the prompt should be adjusted.
        If so, hypothesize why the prompt is inadequate and provide a clear and specific recommendation for how to improve the prompt.
        Specify the precise changes and edit strategy. Specify what things not to touch.
        If not, respond with 'No recommendations.'"""

    @ls.traceable
    async def update_prompt(
        hypotheses: str,
        recommendations: str,
        current_prompt: str,
        update_instructions: str,
    ):
        schema = utils.get_prompt_extraction_schema(current_prompt)

        extractor = create_extractor(
            model,
            tools=[schema],
            tool_choice="OptimizedPromptOutput",
        )
        result = await extractor.ainvoke(
            config["metaprompt"].format(
                current_prompt=current_prompt,
                recommendations=recommendations,
                hypotheses=hypotheses,
                update_instructions=update_instructions,
            )
        )
        return result["responses"][0].improved_prompt

    @ls.traceable(metadata={"kind": "gradient"})
    async def optimize_prompt(
        sessions: list[tuple[list[AnyMessage], dict[str, str] | str]] | str,
        prompt: str | types.Prompt,
    ):
        prompt_str = prompt if isinstance(prompt, str) else prompt.get("prompt", "")
        if not sessions:
            return prompt_str
        elif isinstance(sessions, str):
            sessions = sessions
        else:
            sessions = utils.format_sessions(sessions)

        feedback = "" if isinstance(prompt, str) else prompt.get("feedback", "")
        update_instructions = (
            "" if isinstance(prompt, str) else prompt.get("update_instructions", "")
        )

        inputs = config["gradient_prompt"].format(
            sessions=sessions,
            feedback=feedback,
            prompt=prompt_str,
            update_instructions=update_instructions,
        )
        result = await react_agent(
            model,
            inputs,
            max_steps=config["max_reflection_steps"],
            min_steps=config["min_reflection_steps"],
        )
        if result.warrants_adjustment:
            return await update_prompt(
                result.hypotheses,
                result.full_recommendations,
                prompt_str,
                update_instructions,
            )
        return prompt_str

    return optimize_prompt
