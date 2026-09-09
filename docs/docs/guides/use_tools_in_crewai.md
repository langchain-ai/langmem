---
title: How to Use Memory Tools in CrewAI
---

# How to Use Memory Tools in CrewAI

LangMem's memory tools let your CrewAI agents store and search memories, enabling persistent knowledge across conversations.

## Installation

Install LangMem and CrewAI:

```bash
pip install -U crewai langmem
```

## Basic Usage

Add memory tools to your CrewAI agents:

```python
from crewai import Agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore

# Set up memory store
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)  # (1)!

# Create memory tools
memory_tools = [
    create_manage_memory_tool(namespace="memories", store=store),
    create_search_memory_tool(namespace="memories", store=store),
]

# Create an agent with memory tools
knowledge_agent = Agent(
    role="Knowledge Manager",
    goal="Build and maintain a knowledge base",
    backstory="""You are a knowledge management expert who excels at
    organizing and storing important information for future use.""",
    tools=memory_tools,
    verbose=True,
)
```

1. For production use, replace `InMemoryStore` with a persistent store like [`PostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore).

## Complete Example

Build a crew that maintains and uses a shared knowledge base:

```python
from crewai import Agent, Crew, Task
from langgraph.store.memory import InMemoryStore

from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up shared store
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Create base tools
base_tools = [
    create_manage_memory_tool(namespace="memories", store=store),
    create_search_memory_tool(namespace="memories", store=store),
]

# Create agents
learner = Agent(
    role="Knowledge Learner",
    goal="Learn and store new information in the knowledge base",
    backstory="Your name is Hannah and you are a little too into Demon Slayers.",
    tools=base_tools,
    function_calling_llm="gpt-4o-mini",
)

teacher = Agent(
    role="Knowledge Teacher",
    goal="Use stored knowledge to answer questions",
    tools=base_tools,
    backstory="You were born on the Nile in the midst of the great pestilence..",
    function_calling_llm="gpt-4o-mini",
)

# Create tasks
learn_task = Task(
    description="Save some of your favorite Demon Slayer quotes in memory.",
    agent=learner,
    expected_output="Response:",
)


# Create and run crew
crew = Crew(agents=[learner, teacher], tasks=[learn_task])
result = crew.kickoff()

teach_task = Task(
    description="Search your memories for information about your teammates.",
    agent=teacher,
    expected_output="Response:",
)
crew = Crew(agents=[learner, teacher], tasks=[teach_task])
result = crew.kickoff()
print(store.search(("memories",)))
# Output:
# [
#     Item(
#         namespace=["memories"],
#         key="bd61f87e-d591-45b7-b950-3f9318604ea3",
#         value={
#             "content": '"The bond between Nezuko and I can never be severed. I will always protect her." - Tanjiro Kamado\n"You have to find your own path, you have to find your own way to live!" - Kanao Tsuyuri\n"It’s not the face that makes someone a monster; it’s the choices they make with their lives." - Giyu Tomioka\n"Give me strength! I want to be strong enough to face my own failures!" - Zenitsu Agatsuma\n"Never give up! Never stop fighting until your last breath!" - Giyu Tomioka'
#         },
#         created_at="2025-02-07T22:24:37.736962+00:00",
#         updated_at="2025-02-07T22:24:37.736969+00:00",
#         score=None,
#     )
# ]
```
