---
title: How to Manage Long Context with Summarization
---

# How to Manage Long Context with Summarization

In modern LLM applications, context size can grow quickly and hit provider limitations, whether you're building chatbots with many conversation turns or agentic systems with numerous tool calls.

One effective strategy for handling this is to summarize earlier messages once they reach a certain threshold. This guide demonstrates how to implement this approach in your LangGraph application using LangMem's prebuilt `summarize_messages` and `SummarizationNode`.

## Using in a Simple Chatbot

Below is an example of a simple multi-turn chatbot with summarization:

```python
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import summarize_messages, RunningSummary
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
# highlight-next-line
summarization_model = model.bind(max_tokens=128)  # (1)!

# We will keep track of our running summary in the graph state
class SummaryState(MessagesState):
    summary: RunningSummary | None

# Define the node that will be calling the LLM
def call_model(state: SummaryState) -> SummaryState:
    # highlight-next-line
    summarization_result = summarize_messages(  # (2)!
        state["messages"],
        # IMPORTANT: Pass running summary, if any
        # highlight-next-line
        running_summary=state.get("summary"),  # (3)!
        # highlight-next-line
        token_counter=model.get_num_tokens_from_messages,
        model=summarization_model, 
        max_tokens=256,  # (4)!
        max_tokens_before_summary=256,  # (5)!
        max_summary_tokens=128
    )
    response = model.invoke(summarization_result.messages)
    state_update = {"messages": [response]}
    if summarization_result.running_summary:  # (6)!
        state_update["summary"] = summarization_result.running_summary
    return state_update


checkpointer = InMemorySaver()
builder = StateGraph(SummaryState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
# highlight-next-line
graph = builder.compile(checkpointer=checkpointer)  # (7)!

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
graph.invoke({"messages": "what's my name?"}, config)
```

1. We're also setting max output tokens for the summarization model. This should match `max_summary_tokens` in `summarize_messages` for better
token budget estimates.
2. We will attempt to summarize messages before the LLM is called.
If the messages in `state["messages"]` fit into `max_tokens_before_summary` budget,
we will simply return those messages. Otherwise, we will summarize
and return `[summary_message] + remaining_messages`
3. Pass running summary, if any. This is what
allows `summarize_messages` to avoid re-summarizing the same
messages on every conversation turn.
4. This is the max token budget for the resulting list of messages after summarization.
5. This is the token threshold at which summarization will kick in. Defaults to `max_tokens`.
6. If we generated a summary, add it as a state update and overwrite
the previously generated summary, if any.
7. It's important to compile the graph with a checkpointer,
otherwise the graph won't remember previous conversation turns.

!!! Note "Using in UI"

    An important question is how to present messages to the users in the UI of your app. We recommend rendering the full, unmodified message history. You may choose to additionally render the summary and messages that are passed to the LLM. We also recommend using separate LangGraph state keys for the full message history (e.g., `"messages"`) and summarization results (e.g., `"summary"`). In `SummarizationNode`, summarization results are stored in a separate state key called `context` (see example below).

### Using `SummarizationNode`

You can also separate the summarization into a dedicated node. Let's explore how to modify the above example to use `SummarizationNode` for achieving the same results:

```python
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)


class State(MessagesState):
    # highlight-next-line
    context: dict[str, Any]  # (1)!


class LLMInputState(TypedDict):  # (2)!
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

# highlight-next-line
summarization_node = SummarizationNode(  # (3)!
    token_counter=model.get_num_tokens_from_messages,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

# IMPORTANT: we're passing a private input state here to isolate the summarization
# highlight-next-line
def call_model(state: LLMInputState):  # (4)!
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node(call_model)
# highlight-next-line
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
graph.invoke({"messages": "what's my name?"}, config)
```

1. We will keep track of our running summary in the `context` field
(expected by the `SummarizationNode`).
2. Define private state that will be used only for filtering
the inputs to `call_model` node.
3. SummarizationNode uses `summarize_messages` under the hood and
automatically handles existing summary propagation that we had
to manually do in the above example.
4. The model-calling node now is simply a single LLM invocation.

## Using in a ReAct Agent

A common use case is summarizing message history in a tool calling agent. Below example demonstrates how to implement this in a ReAct-style LangGraph agent:

```python
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

class State(MessagesState):
    context: dict[str, Any]

def search(query: str):
    """Search the web."""
    if "weather" in query.lower():
        return "The weather is sunny in New York, with a high of 104 degrees."
    elif "broadway" in query.lower():
        return "Hamilton is always on!"
    else:
        raise "Not enough information"

tools = [search]

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)

summarization_node = SummarizationNode(
    token_counter=model.get_num_tokens_from_messages,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=1024,
    max_summary_tokens=128,
)

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

def call_model(state: LLMInputState):
    response = model.bind_tools(tools).invoke(state["summarized_messages"])
    return {"messages": [response]}

# Define a router that determines whether to execute tools or exit
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"

checkpointer = InMemorySaver()
builder = StateGraph(State)
# highlight-next-line
builder.add_node("summarize_node", summarization_node)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("summarize_node")
builder.add_edge("summarize_node", "call_model")
builder.add_conditional_edges("call_model", should_continue, path_map=["tools", END])
# highlight-next-line
builder.add_edge("tools", "summarize_node")  # (1)!
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, i am bob"}, config)
graph.invoke({"messages": "what's the weather in nyc this weekend"}, config)
graph.invoke({"messages": "what's new on broadway?"}, config)
```

1. Instead of returning to LLM after executing tools, we first return to the summarization node.