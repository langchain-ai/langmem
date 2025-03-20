from dataclasses import dataclass
from typing import Any, Callable, Iterable, cast

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts.chat import ChatPromptTemplate, ChatPromptValue
from langgraph.utils.runnable import RunnableCallable
from pydantic import BaseModel

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]


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
            "This is summary of the conversation so far: {existing_summary}\n\n"
            "Extend this summary by taking into account the new messages above:",
        ),
    ]
)

DEFAULT_FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        # if exists
        ("placeholder", "{system_message}"),
        ("system", "Summary of the conversation so far: {summary}"),
        ("placeholder", "{messages}"),
    ]
)


@dataclass
class RunningSummary:
    """Object for storing information about the previous summarization.

    Used on subsequent calls to summarize_messages to avoid summarizing the same messages.
    """

    summary: str
    """Latest summary of the messages, updated every time the summarization is performed."""

    summarized_message_ids: set[str]
    """The IDs of all of the messages that have been previously summarized."""


@dataclass
class SummarizationResult:
    """Result of message summarization."""

    messages: list[AnyMessage]
    """List of updated messages that are ready to be input to the LLM after summarization, including a message with a summary (if any)."""

    running_summary: RunningSummary | None = None
    """Information about previous summarization (the summary and the IDs of the previously summarized messages.
    Can be None if no summarization was performed (not enough messages to summarize).
    """


def summarize_messages(
    messages: list[AnyMessage],
    *,
    running_summary: RunningSummary | None,
    model: LanguageModelLike,
    max_tokens: int,
    max_summary_tokens: int = 256,
    token_counter: TokenCounter = count_tokens_approximately,
    initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
    final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
) -> SummarizationResult:
    """Summarize messages when they exceed a token limit and replace them with a summary message.

    This function processes the messages from oldest to newest: once the cumulative number of message tokens
    reaches max_tokens, all messages within max_tokens are summarized and replaced with a new summary message.
    The resulting list of messages is [summary_message] + remaining_messages.

    Args:
        messages: The list of messages to process.
        running_summary: Optional running summary object with information about the previous summarization. If provided:
            - only messages that were **not** previously summarized will be processed
            - if no new summary is generated, the running summary will be added to the returned messages
            - if a new summary needs to be generated, it is generated by incorporating the existing summary value from the running summary
        model: The language model to use for generating summaries.
        max_tokens: Maximum number of tokens to return.
            Will also be used as a threshold for triggering the summarization: once the cumulative number of message tokens
            reaches max_tokens, all messages within max_tokens will be summarized.

            !!! Note

                If the last message within max_tokens is an AI message with tool calls or a human message,
                this message will not be summarized, and instead will be added to the returned messages.
        max_summary_tokens: Maximum number of tokens to budget for the summary.

            !!! Note

                This parameter is not passed to the summary-generating LLM to limit the length of the summary.
                It is only used for correctly estimating the threshold for summarization.
                If you want to enforce it, you would need to pass `model.bind(max_tokens=max_summary_tokens)`
                as the `model` parameter to this function.
        token_counter: Function to count tokens in a message. Defaults to approximate counting.
            For more accurate counts you can use `model.get_num_tokens_from_messages`.
        initial_summary_prompt: Prompt template for generating the first summary.
        existing_summary_prompt: Prompt template for updating an existing (running) summary.
        final_prompt: Prompt template that combines summary with the remaining messages before returning.

    Returns:
        A SummarizationResult object containing the updated messages and a running summary.
            - messages: list of updated messages ready to be input to the LLM
            - running_summary: RunningSummary object
                - summary: text of the latest summary
                - summarized_message_ids: set of message IDs that were previously summarized

    Example:
        ```pycon
        >>> from langgraph.graph import StateGraph, START, MessagesState
        >>> from langgraph.checkpoint.memory import InMemorySaver
        >>> from langmem.short_term.summarization import summarize_messages, RunningSummary
        >>> from langchain_openai import ChatOpenAI

        >>> model = ChatOpenAI(model="gpt-4o")
        >>> summarization_model = model.bind(max_tokens=128)

        >>> class SummaryState(MessagesState):
        ...     summary: RunningSummary | None

        >>> def call_model(state):
        ...     summarization_result = summarize_messages(
        ...         state["messages"],
        ...         running_summary=state.get("summary"),
        ...         model=summarization_model,
        ...         max_tokens=256,
        ...         max_summary_tokens=128
        ...     )
        ...     response = model.invoke(summarization_result.messages)
        ...     state_update = {"messages": [response]}
        ...     if summarization_result.running_summary:
        ...         state_update["summary"] = summarization_result.running_summary
        ...     return state_update

        >>> checkpointer = InMemorySaver()
        >>> workflow = StateGraph(SummaryState)
        >>> workflow.add_node(call_model)
        >>> workflow.add_edge(START, "call_model")
        >>> graph = workflow.compile(checkpointer=checkpointer)

        >>> config = {"configurable": {"thread_id": "1"}}
        >>> graph.invoke({"messages": "hi, my name is bob"}, config)
        >>> graph.invoke({"messages": "write a short poem about cats"}, config)
        >>> graph.invoke({"messages": "now do the same but for dogs"}, config)
        >>> graph.invoke({"messages": "what's my name?"}, config)
        ```
    """
    if max_summary_tokens >= max_tokens:
        raise ValueError("`max_summary_tokens` must be less than `max_tokens`.")

    max_tokens_to_summarize = max_tokens
    # Adjust the remaining token budget to account for the summary to be added
    max_remaining_tokens = max_tokens - max_summary_tokens
    # First handle system message if present
    if messages and isinstance(messages[0], SystemMessage):
        existing_system_message = messages[0]
        # remove the system message from the list of messages to summarize
        messages = messages[1:]
        # adjust the remaining token budget to account for the system message to be re-added
        max_remaining_tokens -= token_counter([existing_system_message])
    else:
        existing_system_message = None

    if not messages:
        return SummarizationResult(
            running_summary=running_summary,
            messages=(
                messages
                if existing_system_message is None
                else [existing_system_message] + messages
            ),
        )

    # Get previously summarized messages, if any
    summarized_message_ids = set()
    if running_summary:
        summarized_message_ids = running_summary.summarized_message_ids
        # Adjust the summarization token budget to account for the previous summary
        max_tokens_to_summarize -= token_counter(
            [SystemMessage(content=running_summary.summary)]
        )

    total_summarized_messages = len(summarized_message_ids)

    # Go through messages to count tokens and find cutoff point
    n_tokens = 0
    idx = max(0, total_summarized_messages - 1)
    # We need to output messages that fit within max_tokens.
    # Assuming that the summarization LLM also needs at most max_tokens
    # that will be turned into at most max_summary_tokens, you can try
    # to process at most max_tokens * 2 - max_summary_tokens
    max_total_tokens = max_tokens + max_remaining_tokens
    for i in range(total_summarized_messages, len(messages)):
        message = messages[i]
        if message.id is None:
            raise ValueError("Messages are required to have ID field.")

        if message.id in summarized_message_ids:
            raise ValueError(
                f"Message with ID {message.id} has already been summarized."
            )

        n_tokens += token_counter([message])

        # If we're still under max_tokens_to_summarize, update the potential cutoff point
        if n_tokens <= max_tokens_to_summarize:
            idx = i

        # Check if we've exceeded the absolute maximum
        if n_tokens > max_total_tokens:
            raise ValueError(
                f"`summarize_messages` cannot handle more than {max_total_tokens} tokens: "
                f"resulting message history will exceed max_tokens limit ({max_tokens}). "
                "Please adjust `max_tokens` / `max_summary_tokens` or decrease the input size."
            )

    # If we haven't exceeded max_tokens, we don't need to summarize
    # Note: we don't return here since we might still need to include the existing summary
    if n_tokens <= max_tokens:
        messages_to_summarize = []
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
        if running_summary:
            summary_messages = cast(
                ChatPromptValue,
                existing_summary_prompt.invoke(
                    {
                        "messages": messages_to_summarize,
                        "existing_summary": running_summary.summary,
                    }
                ),
            )
        else:
            summary_messages = cast(
                ChatPromptValue,
                initial_summary_prompt.invoke({"messages": messages_to_summarize}),
            )

        summary_response = model.invoke(summary_messages.messages)
        summarized_message_ids = summarized_message_ids | set(
            message.id for message in messages_to_summarize
        )
        total_summarized_messages += len(messages_to_summarize)
        running_summary = RunningSummary(
            summary=summary_response.content,
            summarized_message_ids=summarized_message_ids,
        )

    if running_summary:
        updated_messages = cast(
            ChatPromptValue,
            final_prompt.invoke(
                {
                    "system_message": [existing_system_message]
                    if existing_system_message
                    else [],
                    "summary": running_summary.summary,
                    "messages": messages[total_summarized_messages:],
                }
            ),
        )
        return SummarizationResult(
            running_summary=running_summary,
            messages=updated_messages.messages,
        )
    else:
        # no changes are needed
        return SummarizationResult(
            running_summary=None,
            messages=(
                messages
                if existing_system_message is None
                else [existing_system_message] + messages
            ),
        )


class SummarizationNode(RunnableCallable):
    def __init__(
        self,
        *,
        model: LanguageModelLike,
        max_tokens: int,
        max_summary_tokens: int = 1,
        token_counter: TokenCounter = len,
        initial_summary_prompt: ChatPromptTemplate = DEFAULT_INITIAL_SUMMARY_PROMPT,
        existing_summary_prompt: ChatPromptTemplate = DEFAULT_EXISTING_SUMMARY_PROMPT,
        final_prompt: ChatPromptTemplate = DEFAULT_FINAL_SUMMARY_PROMPT,
        input_messages_key: str = "messages",
        output_messages_key: str = "summarized_messages",
        name: str = "summarization",
    ) -> None:
        """A LangGraph node that summarizes messages when they exceed a token limit and replaces them with a summary message.

        Processes the messages from oldest to newest: once the cumulative number of message tokens
        reaches max_tokens, all messages within the token limit are summarized and replaced with a new summary message.
        The resulting list of messages is [summary_message] + remaining_messages.

        Args:
            model: The language model to use for generating summaries.
            max_tokens: Maximum number of tokens to return.
                Will be used as a threshold for triggering the summarization: once the cumulative number of message tokens
                reaches max_tokens, all messages within max_tokens will be summarized.

                !!! Note

                    If the last message within max_tokens is an AI message with tool calls or a human message,
                    this message will not be summarized, and instead will be added to the returned messages.
            max_summary_tokens: Maximum number of tokens to return from the summarization LLM.
            token_counter: Function to count tokens in a message. Defaults to approximate counting.
            initial_summary_prompt: Prompt template for generating the first summary.
            existing_summary_prompt: Prompt template for updating an existing summary.
            final_prompt: Prompt template that combines summary with the remaining messages before returning.
            input_messages_key: Key in the input graph state that contains the list of messages to summarize.
            output_messages_key: Key in the state update that contains the list of updated messages.
                !!! Note

                    `output_messages_key` must be **different** from the `input_messages_key`.
                    This is done to decouple summarized messages from the main list of messages in the graph state (i.e., `input_messages_key`).
                    If you want to update / overwrite the main list of messages, you would need to use summarize_messages function directly or wrap
                    the invocation of this node in a different node.

            name: Name of the summarization node.

        Returns:
            LangGraph state update in the following format:
                ```
                {
                    output_messages_key: <list of updated messages ready to be input to the LLM after summarization, including a message with a summary (if any)>,
                    "context": {"running_summary": <RunningSummary object>}
                }
                ```

        Example:

            ```pycon
            >>> from typing import Any, TypedDict
            >>> from langchain_openai import ChatOpenAI
            >>> from langchain_core.messages import AnyMessage
            >>> from langgraph.graph import StateGraph, START, MessagesState
            >>> from langgraph.checkpoint.memory import InMemorySaver
            >>> from langmem.short_term.summarization import SummarizationNode, RunningSummary
            >>>
            >>> model = ChatOpenAI(model="gpt-4o")
            >>> summarization_model = model.bind(max_tokens=128)
            >>>
            >>> class State(MessagesState):
            ...     context: dict[str, Any]
            ...
            >>> class LLMInputState(TypedDict):
            ...     summarized_messages: list[AnyMessage]
            ...     context: dict[str, Any]
            ...
            >>> summarization_node = SummarizationNode(
            ...     model=summarization_model,
            ...     max_tokens=256,
            ...     max_summary_tokens=128,
            ... )
            >>>
            >>> def call_model(state: LLMInputState):
            ...     response = model.invoke(state["summarized_messages"])
            ...     return {"messages": [response]}
            ...
            >>> checkpointer = InMemorySaver()
            >>> workflow = StateGraph(State)
            >>> workflow.add_node(call_model)
            >>> workflow.add_node("summarize", summarization_node)
            >>> workflow.add_edge(START, "summarize")
            >>> workflow.add_edge("summarize", "call_model")
            >>> graph = workflow.compile(checkpointer=checkpointer)
            >>>
            >>> config = {"configurable": {"thread_id": "1"}}
            >>> graph.invoke({"messages": "hi, my name is bob"}, config)
            >>> graph.invoke({"messages": "write a short poem about cats"}, config)
            >>> graph.invoke({"messages": "now do the same but for dogs"}, config)
            >>> graph.invoke({"messages": "what's my name?"}, config)
            ```
        """
        super().__init__(self._func, name=name, trace=False)
        self.model = model
        self.max_tokens = max_tokens
        self.max_summary_tokens = max_summary_tokens
        self.token_counter = token_counter
        self.initial_summary_prompt = initial_summary_prompt
        self.existing_summary_prompt = existing_summary_prompt
        self.final_prompt = final_prompt
        if input_messages_key == output_messages_key:
            raise ValueError(
                "`input_messages_key` and `output_messages_key` must be different."
            )

        self.input_messages_key = input_messages_key
        self.output_messages_key = output_messages_key

    def _func(self, input: dict[str, Any] | BaseModel) -> dict[str, Any]:
        if isinstance(input, dict):
            messages = input.get(self.input_messages_key)
            context = input.get("context", {})
        elif isinstance(input, BaseModel):
            messages = getattr(input, self.input_messages_key, None)
            context = getattr(input, "context", {})
        else:
            raise ValueError(f"Invalid input type: {type(input)}")

        if messages is None:
            raise ValueError(
                f"Missing required field `{self.input_messages_key}` in the input."
            )

        summarization_result = summarize_messages(
            messages,
            running_summary=context.get("running_summary"),
            model=self.model,
            max_tokens=self.max_tokens,
            max_summary_tokens=self.max_summary_tokens,
            token_counter=self.token_counter,
            initial_summary_prompt=self.initial_summary_prompt,
            existing_summary_prompt=self.existing_summary_prompt,
            final_prompt=self.final_prompt,
        )

        state_update = {self.output_messages_key: summarization_result.messages}
        if summarization_result.running_summary:
            state_update["context"] = {
                **context,
                "running_summary": summarization_result.running_summary,
            }
        return state_update
