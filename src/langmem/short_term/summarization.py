from typing import Callable, Optional, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts.chat import ChatPromptTemplate, ChatPromptValue
from pydantic import BaseModel

TokenCounter = Callable[[list[BaseMessage]], int]


DEFAULT_INITIAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        ("user", "Create a summary of the conversation above:"),
    ]
)


DEFAULT_EXISTING_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
        (
            "user",
            "This is summary of the conversation to date: {existing_summary}\n\n"
            "Extend the summary by taking into account the new messages above:",
        ),
    ]
)

DEFAULT_FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        # if exists
        ("placeholder", "{system_message}"),
        ("system", "Summary of conversation earlier: {summary}"),
        ("placeholder", "{messages}"),
    ]
)


class SummaryInfo(BaseModel):
    # the summary of the conversation so far
    summary: str
    # the messages that have been most recently summarized
    summarized_messages: list[BaseMessage]
    # keep track of the total number of messages that have been summarized thus far
    total_summarized_messages: int = 0


class SummarizationResult(BaseModel):
    # the messages that will be returned to the user
    messages: list[BaseMessage]
    # SummaryInfo (empty if messages were not summarized)
    summary: SummaryInfo | None = None


def summarize_messages(
    messages: list[BaseMessage],
    *,
    model: BaseChatModel,
    max_tokens: int,
    max_summary_tokens: int = 256,
    # TODO: replaces this with approximate token counter
    token_counter: TokenCounter = len,
    existing_summary: Optional[SummaryInfo] = None,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
) -> SummarizationResult:
    """A memory handler that summarizes messages when they exceed a token limit and replaces summarized messages with a single summary message.

    Args:
        messages: The list of messages to process.
        max_tokens: Maximum number of tokens to return.
        model: The language model to use for generating summaries.
        max_summary_tokens: Maximum number of tokens to return from the summarization LLM.
        token_counter: Function to count tokens in a message. Defaults to approximate counting.
        existing_summary: Optional existing summary.
        initial_summary_prompt: Prompt template for generating the first summary.
        existing_summary_prompt: Prompt template for updating an existing summary.
        final_prompt: Prompt template that combines summary with the remaining messages before returning.
    """
    if max_summary_tokens >= max_tokens:
        raise ValueError("`max_summary_tokens` must be less than `max_tokens`.")

    summarization_model = model.bind(max_tokens=max_summary_tokens)

    # First handle system message if present
    if messages and isinstance(messages[0], SystemMessage):
        existing_system_message = messages[0]
        # remove the system message from the list of messages to summarize
        messages = messages[1:]
        # adjust the token budget to account for the system message to be added
        max_tokens -= token_counter([existing_system_message])
    else:
        existing_system_message = None

    if not messages:
        return SummarizationResult(
            summary=existing_summary,
            messages=(
                messages
                if existing_system_message is None
                else [existing_system_message] + messages
            ),
        )

    summary_value = existing_summary
    total_summarized_messages = (
        summary_value.total_summarized_messages if summary_value else 0
    )

    # Single pass through messages to count tokens and find cutoff point
    n_tokens = 0
    idx = max(0, total_summarized_messages - 1)
    # we need to output messages that fit within max_tokens.
    # assuming that the summarization LLM also needs at most max_tokens
    # that will be turned into at most max_summary_tokens, you can try
    # to process at most max_tokens * 2 - max_summary_tokens
    max_total_tokens = max_tokens * 2 - max_summary_tokens
    for i in range(total_summarized_messages, len(messages)):
        n_tokens += token_counter([messages[i]])

        # If we're still under max_tokens, update the potential cutoff point
        if n_tokens <= max_tokens:
            idx = i

        # Check if we've exceeded the absolute maximum
        if n_tokens >= max_total_tokens:
            raise ValueError(
                f"summarize_messages cannot handle more than {max_total_tokens} tokens. "
                "Please increase the `max_tokens` or decrease the input size."
            )

    # If we haven't exceeded max_tokens, return original messages
    if n_tokens <= max_tokens:
        # we don't need to summarize, but we might still need to include the existing summary
        messages_to_summarize = None
    else:
        messages_to_summarize = messages[total_summarized_messages : idx + 1]

    # If the last message is:
    # (1) an AI message with tool calls - remove it
    #   to avoid issues w/ the LLM provider (as it will lack a corresponding tool message)
    # (2) a human message - remove it,
    #   since it is a user input and it doesn't make sense to summarize it without a corresponding AI message
    while messages_to_summarize and (
        (
            isinstance(messages_to_summarize[-1], AIMessage)
            and messages_to_summarize[-1].tool_calls
        )
        or isinstance(messages_to_summarize[-1], HumanMessage)
    ):
        messages_to_summarize.pop()

    if messages_to_summarize:
        if existing_summary:
            summary_messages = cast(
                ChatPromptValue,
                existing_summary_prompt.invoke(
                    {
                        "messages": messages_to_summarize,
                        "existing_summary": summary_value.summary,
                    }
                ),
            )
        else:
            summary_messages = cast(
                ChatPromptValue,
                initial_summary_prompt.invoke({"messages": messages_to_summarize}),
            )

        summary_message_response = summarization_model.invoke(summary_messages.messages)
        total_summarized_messages += len(messages_to_summarize)
        summary_value = SummaryInfo(
            summary=summary_message_response.content,
            summarized_messages=messages_to_summarize,
            total_summarized_messages=total_summarized_messages,
        )

    if summary_value:
        updated_messages = cast(
            ChatPromptValue,
            final_prompt.invoke(
                {
                    "system_message": [existing_system_message]
                    if existing_system_message
                    else [],
                    "summary": summary_value.summary,
                    "messages": messages[total_summarized_messages:],
                }
            ),
        )
        return SummarizationResult(
            summary=summary_value,
            messages=updated_messages.messages,
        )
    else:
        # no changes are needed
        return SummarizationResult(
            summary=None,
            messages=(
                messages
                if existing_system_message is None
                else [existing_system_message] + messages
            ),
        )
