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
# NOTE: we're also setting max output tokens for the summary
# this should match max_summary_tokens in `summarize_messages` for better
# token budget estimates
# highlight-next-line
summarization_model = model.bind(max_tokens=128)

# We will keep track of our running summary in the graph state
class SummaryState(MessagesState):
    summary: RunningSummary | None

# Define the node that will be calling the LLM
def call_model(state: SummaryState) -> SummaryState:
    # We will attempt to summarize messages before the LLM is called
    # If the messages in state["messages"] fit into max tokens budget,
    # we will simply return those messages. Otherwise, we will summarize
    # and return [summary_message] + remaining_messages
    # highlight-next-line
    summarization_result = summarize_messages(
        state["messages"],
        # IMPORTANT: Pass running summary, if any. This is what
        # allows summarize_messages to avoid re-summarizing the same
        # messages on every conversation turn
        # highlight-next-line
        running_summary=state.get("summary"),
        # by default this is using approximate token counting,
        # but you can also use LLM-specific one, like below
        # highlight-next-line
        token_counter=model.get_num_tokens_from_messages,
        model=summarization_model, 
        max_tokens=256,
        max_summary_tokens=128
    )
    response = model.invoke(summarization_result.messages)
    state_update = {"messages": [response]}
    # If we generated a summary, add it as a state update and overwrite
    # the previously generated summary, if any
    if summarization_result.running_summary:
        state_update["summary"] = summarization_result.running_summary
    return state_update


checkpointer = InMemorySaver()
builder = StateGraph(SummaryState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
# It's important to compile the graph with a checkpointer, otherwise
# we won't be storing previous conversation history as well as running summary
# between invocations
# highlight-next-line
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
graph.invoke({"messages": "what's my name?"}, config)
```

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


# We will keep track of our running summary in the `context` field
# (expected by the `SummarizationNode`)
class State(MessagesState):
    # highlight-next-line
    context: dict[str, Any]


# Define private state that will be used only for filtering
# the inputs to call_model node
class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

# SummarizationNode uses `summarize_messages` under the hood and
# automatically handles existing summary propagation that we had
# to manually do in the above example 
# highlight-next-line
summarization_node = SummarizationNode(
    token_counter=model.get_num_tokens_from_messages,
    model=summarization_model,
    max_tokens=256,
    max_summary_tokens=128,
)

# The model-calling node now is simply a single LLM invocation
# IMPORTANT: we're passing a private input state here to isolate the summarization
# highlight-next-line
def call_model(state: LLMInputState):
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

## Using in a ReAct Agent

A common use case is summarizing message history in a tool calling agent. Below example demonstrates how to implement this in a ReAct-style LangGraph agent:

```python
from typing import Any, TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary

class State(MessagesState):
    context: dict[str, Any]

search = TavilySearchResults(max_results=3)
tools = [search]

model = ChatOpenAI(model="gpt-4o")
summarization_model = model.bind(max_tokens=128)

summarization_node = SummarizationNode(
    token_counter=model.get_num_tokens_from_messages,
    model=summarization_model,
    max_tokens=2048,
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
# instead of returning to LLM after executing tools, we first return to the summarization node
# highlight-next-line
builder.add_edge("tools", "summarize_node")
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, i am bob"}, config)
graph.invoke({"messages": "what's the weather in nyc this weekend"}, config)
graph.invoke({"messages": "what's new on broadway?"}, config)
```