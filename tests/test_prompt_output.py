import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langmem import create_prompt_optimizer
from langmem.utils import get_prompt_extraction_schema


@pytest.mark.parametrize("prompt", ["Answer briefly.", "Answer about {topic}."])
def test_prompt_output_accepts_null(prompt):
    schema = get_prompt_extraction_schema(prompt)
    result = schema.model_validate(
        {"analysis": "No changes needed.", "improved_prompt": None}
    )
    assert result.improved_prompt is None


def test_prompt_output_still_checks_required_variables():
    schema = get_prompt_extraction_schema("Answer about {topic}.")
    with pytest.raises(ValueError, match="Missing required variable"):
        schema.model_validate(
            {"analysis": "Revise the tone.", "improved_prompt": "Be concise."}
        )


def test_metaprompt_retains_original_prompt_for_null_output():
    class LocalModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = LocalModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "OptimizedPromptOutput",
                        "args": {
                            "analysis": "No changes needed.",
                            "improved_prompt": None,
                        },
                        "id": "local-call",
                    }
                ],
            )
        ]
    )
    optimizer = create_prompt_optimizer(
        model, kind="metaprompt", config={"max_reflection_steps": 1}
    )
    original = "Answer briefly."
    assert (
        optimizer.invoke(
            {
                "prompt": original,
                "trajectories": [([HumanMessage(content="Hello")], "Already correct.")],
            }
        )
        == original
    )
