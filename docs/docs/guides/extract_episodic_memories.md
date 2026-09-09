---
title: How to Extract Episodic Memories
---

# How to Extract Episodic Memories

Need your agent to learn from experience? Here's how to use LangMem for experience replay—capturing not just what happened, but the complete chain of thought that led to success. While [semantic memory](./extract_semantic_memories.md) builds knowledge ("what"), episodic memory captures expertise ("how").

## Without storage

Below is an example of how to use LangMem to extract episodic memories without storage.
Feel free to adapt and modify the code to your needs.

```python
from langmem import create_memory_manager
from pydantic import BaseModel, Field


class Episode(BaseModel):  # (1)!
    """Write the episode from the perspective of the agent within it. Use the benefit of hindsight to record the memory, saving the agent's key internal thought process so it can learn over time."""

    observation: str = Field(..., description="The context and setup - what happened")
    thoughts: str = Field(
        ...,
        description="Internal reasoning process and observations of the agent in the episode that let it arrive"
        ' at the correct action and result. "I ..."',
    )
    action: str = Field(
        ...,
        description="What was done, how, and in what format. (Include whatever is salient to the success of the action). I ..",
    )
    result: str = Field(
        ...,
        description="Outcome and retrospective. What did you do well? What could you do better next time? I ...",
    )


manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Episode],  # (2)!
    instructions="Extract examples of successful explanations, capturing the full chain of reasoning. Be concise in your explanations and precise in the logic of your reasoning.",
    enable_inserts=True,
)
```

1. Unlike semantic triples that store facts, episodes capture the full context of successful interactions
2. The Episode schema becomes part of the memory manager's prompt, helping it extract complete reasoning chains that guide future responses. The manager extracts complete episodes, not just individual facts.

After a successful explanation:

```python
conversation = [
    {
        "role": "user",
        "content": "What's a binary tree? I work with family trees if that helps",
    },
    {
        "role": "assistant",
        "content": "A binary tree is like a family tree, but each parent has at most 2 children. Here's a simple example:\n   Bob\n  /  \\\nAmy  Carl\n\nJust like in family trees, we call Bob the 'parent' and Amy and Carl the 'children'.",
    },
    {
        "role": "user",
        "content": "Oh that makes sense! So in a binary search tree, would it be like organizing a family by age?",
    },
]

episodes = manager.invoke({"messages": conversation})
print(episodes[0])

# ExtractedMemory(
#     id="2e5c551f-58a7-40c2-96b3-cabdfa5ccb31",
#     content=Episode(
#         observation="In a teaching interaction, I used a family tree analogy to explain binary trees, which led to a successful understanding. The student then made an insightful connection to binary search trees and age ordering.",
#         thoughts="I noticed that connecting abstract data structures to familiar concepts like family relationships made the concept more accessible. The student's quick grasp and ability to extend the analogy to binary search trees showed the effectiveness of this approach. Using relatable examples helps bridge the gap between technical concepts and everyday understanding.",
#         action='I explained binary trees using a family tree metaphor, drawing a simple diagram with "Bob" as parent and "Amy" and "Carl" as children. This visualization provided a concrete, relatable example that built on the student\'s existing knowledge of family trees.',
#         result="The explanation was highly successful, evidenced by the student's immediate comprehension (\"Oh that makes sense!\") and their ability to make the cognitive leap to understanding binary search trees' ordering property. For future explanations, I should continue using familiar analogies while being prepared to build upon them for more complex concepts. The family tree analogy proved particularly effective for explaining hierarchical structures.",
#     ),
# )
```

## With storage

Let's extend our example to store and retrieve episodes using LangGraph's storage system:

```python
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

# Set up vector store for similarity search
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Configure memory manager with storage
manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories", "episodes"),
    schemas=[Episode],
    instructions="Extract exceptional examples of noteworthy problem-solving scenarios, including what made them effective.",
    enable_inserts=True,
)

llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")


@entrypoint(store=store)
def app(messages: list):
    # Step 1: Find similar past episodes
    similar = store.search(
        ("memories", "episodes"),
        query=messages[-1]["content"],
        limit=1,
    )

    # Step 2: Build system message with relevant experience
    system_message = "You are a helpful assistant."
    if similar:
        system_message += "\n\n### EPISODIC MEMORY:"
        for i, item in enumerate(similar, start=1):
            episode = item.value["content"]
            system_message += f"""

Episode {i}:
When: {episode["observation"]}
Thought: {episode["thoughts"]}
Did: {episode["action"]}
Result: {episode["result"]}
        """

    # Step 3: Generate response using past experience
    response = llm.invoke([{"role": "system", "content": system_message}, *messages])

    # Step 4: Store this interaction if successful
    manager.invoke({"messages": messages})
    return response


app.invoke(
    [
        {
            "role": "user",
            "content": "What's a binary tree? I work with family trees if that helps",
        },
    ],
)
print(store.search(("memories", "episodes"), query="Trees"))

# [
#     Item(
#         namespace=["memories", "episodes"],
#         key="57f6005b-00f3-4f81-b384-961cb6e6bf97",
#         value={
#             "kind": "Episode",
#             "content": {
#                 "observation": "User asked about binary trees and mentioned familiarity with family trees. This presented an opportunity to explain a technical concept using a relatable analogy.",
#                 "thoughts": "I recognized this as an excellent opportunity to bridge understanding by connecting a computer science concept (binary trees) to something the user already knows (family trees). The key was to use their existing mental model of hierarchical relationships in families to explain binary tree structures.",
#                 "action": "Used family tree analogy to explain binary trees: Each person (node) in a binary tree can have at most two children (left and right), unlike family trees where people can have multiple children. Drew parallel between parent-child relationships in both structures while highlighting the key difference of the two-child limitation in binary trees.",
#                 "result": "Successfully translated a technical computer science concept into familiar terms. This approach demonstrated effective teaching through analogical reasoning - taking advantage of existing knowledge structures to build new understanding. For future similar scenarios, this reinforces the value of finding relatable real-world analogies when explaining technical concepts. The family tree comparison was particularly effective because it maintained the core concept of hierarchical relationships while clearly highlighting the key distinguishing feature (binary limitation).",
#             },
#         },
#         created_at="2025-02-09T03:40:11.832614+00:00",
#         updated_at="2025-02-09T03:40:11.832624+00:00",
#         score=0.30178054939692683,
#     )
# ]
```

1. For production, use a persistent store like [`AsyncPostgresStore`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.postgres.AsyncPostgresStore)

Let's break down what's happening:

1. **Setup**: We create a vector store for similarity search and configure the memory manager with storage support
2. **Search**: When handling a message, we first search for similar past episodes
3. **Apply**: If we find a relevant episode, we include its learnings in the system message
4. **Learn**: After generating a response, we store this interaction as a new episode

This creates a learning loop where the agent continuously improves by drawing on past experiences.

## When to Use Episodic Memory

Episodic memory drives adaptive learning. While semantic memory builds a knowledge base of facts ("Python is a programming language"), episodic memory captures the expertise of how to apply that knowledge effectively ("explaining Python using snake analogies confused users, but comparing it to recipe steps worked well").

This experience replay helps agents:
- Adapt teaching style based on what worked
- Learn from successful problem-solving approaches
- Build a library of proven techniques
- Understand not just what to do, but why it works

For more about memory types and their roles, see [Memory Types](../concepts/conceptual_guide.md#memory-types).
