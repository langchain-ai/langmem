---
title: Core Concepts
---

# Long-term Memory in LLM Applications

Long-term memory allows agents to remember important information across conversations. LangMem provides ways to extract meaningful details from chats, store them, and use them to improve future interactions. At its core, each memory operation in LangMem follows the same pattern:

1. Accept conversation(s) and current memory state
2. Prompt an LLM to determine how to expand or consolidate the memory state
3. Respond with the updated memory state

The best memory systems are often application-specific. In designing yours, the following questions can serve as a useful guide:

1. **What** [type of content](#memory-types) should your agent learn: facts/knowledge? summary of past events? Rules and style?
2. **When** should the [memories be formed](#writing-memories) (and **who** should form the memories)
3. **Where** should memories [be stored](#storage-system)? (in the prompt? Semantic store?). This largely determines how they will be recalled.

## Types of Memory {#memory-types}

Memory in LLM applications can reflect some of the structure of human memory, with each type serving a distinct purpose in building adaptive, context-aware systems:


| Memory Type | Purpose | Agent Example | Human Example | Typical Storage Pattern |
|-------------|---------|---------------|---------------|-----------------------|
| Semantic | Facts & Knowledge | User preferences; knowledge triplets | Knowing Python is a programming language | Profile or Collection |
| Episodic | Past Experiences | Few-shot examples; Summaries of past conversations | Remembering your first day at work | Collection |
| Procedural | System Behavior | Core personality and response patterns | Knowing how to ride a bicycle | Prompt rules or Collection |

### Semantic Memory: Facts and Knowledge

[Semantic memory](https://en.wikipedia.org/wiki/Semantic_memory) stores the essential facts and other information that ground an agent's responses. Two common representations of semantic memory are collections (to record an unbounded amount of knowledge to be searched at runtime) and profiles (to record task-specific information that follows a strict schema that is easily looked up by user or agent). 

#### Collection

Collections are what most people think of when they imagine agent long-term memory. In this type, memories are stored as individual documents or records. For each new conversation, the memory system can decide to insert new memories to the store. 

Using a collection-type memory adds some complexity to the process of updating your memory state. The system must reconcile new information with previous beliefes, either  _deleting_/_invalidating_ or _updating_/_consolidating_ existing memories. If the system over-extracts, this could lead to reduced precision of memories when your agent needs to search the store. If it under-extracts, this could lead to low recall. LangMem uses a memory enrichment process that strives to balance memory creation and consolidation, while letting you, the developer, customize the instructions to further shift the strength of each.

Finally, memory relevance is more than just semantic similarity. Recall should combine similarity with "importance" of the memory, as well as the memory's "strength", which is a function of how recently/frequently it was used.

![Collection update process](img/update-list.png)

??? example "Extracting semantic memories as collections"

    ??? note "Setup"

        ```python
        from langmem import create_memory_manager
        
        # highlight-next-line
        manager = create_memory_manager(
            "anthropic:claude-3-5-sonnet-latest",
            instructions="Extract all noteworthy facts, events, and relationships. Indicate their importance.",
            # highlight-next-line
            enable_inserts=True,
        )

        # Process a conversation to extract semantic memories
        conversation = [
            {"role": "user", "content": "I work at Acme Corp in the ML team"},
            {"role": "assistant", "content": "I'll remember that. What kind of ML work do you do?"},
            {"role": "user", "content": "Mostly NLP and large language models"}
        ]
        ```

    ```python
    memories = manager.invoke({"messages": conversation})
    # Example memories:
    # [
    #     ExtractedMemory(
    #         id="27e96a9d-8e53-4031-865e-5ec50c1f7ad5",
    #         content=Memory(
    #             content="[IMPORTANT] User prefers to be called Lex (short for Alex) and appreciates"
    #             " casual, witty communication style with relevant emojis."
    #         ),
    #     ),
    #     ExtractedMemory(
    #         id="e2f6b646-cdf1-4be1-bb40-0fd91d25d00f",
    #         content=Memory(
    #             content="[BACKGROUND] Lex is proficient in Python programming and specializes in developing"
    #             " AI systems with a focus on making them sound more natural and less corporate."
    #         ),
    #     ),
    #     ExtractedMemory(
    #         id="c1e03ebb-a393-4e8d-8eb7-b928d8bed510",
    #         content=Memory(
    #             content="[HOBBY] Lex is a competitive speedcuber (someone who solves Rubik's cubes competitively),"
    #             " showing an interest in both technical and recreational puzzle-solving."
    #         ),
    #     ),
    #     ExtractedMemory(
    #         id="ee7fc6e4-0118-425f-8704-6b3145881ff7",
    #         content=Memory(
    #             content="[PERSONALITY] Based on communication style and interests, Lex appears to value authenticity,"
    #             " creativity, and technical excellence while maintaining a fun, approachable demeanor."
    #         ),
    #     ),
    # ]

    ```

#### Profiles

**Profiles** on the other hand are well-scoped for a particular task. Profiles are a single document that represents the current state, like a user's main goals with using an app, their preferred name and response stele, etc. When new information arrives, it updates the existing document rather than creating a new one. This approach is ideal when you only care about the latest state and want to avoid remembering extraneous information.

![Profile update process](img/update-profile.png)

??? example "Managing user preferences with profiles"

    ??? note "Setup"

        ```python
        from langmem import create_memory_manager
        from pydantic import BaseModel


        class UserProfile(BaseModel):
            """Save the user's preferences."""
            name: str
            preferred_name: str
            response_style_preference: str
            special_skills: list[str]
            other_preferences: list[str]


        manager = create_memory_manager(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[UserProfile],
            instructions="Extract user preferences and settings",
            enable_inserts=False,
        )

        # Extract user preferences from a conversation
        conversation = [
            {"role": "user", "content": "Hi! I'm Alex but please call me Lex. I'm a wizard at Python and love making AI systems that don't sound like boring corporate robots 🤖"},
            {"role": "assistant", "content": "Nice to meet you, Lex! Love the anti-corporate-robot stance. How would you like me to communicate with you?"},
            {"role": "user", "content": "Keep it casual and witty - and maybe throw in some relevant emojis when it feels right ✨ Also, besides AI, I do competitive speedcubing!"},
        ]
        ```

    ```python
    profile = manager.invoke({"messages": conversation})[0]
    print(profile)
    # Example profile:
    # ExtractedMemory(
    #     id="6f555d97-387e-4af6-a23f-a66b4e809b0e",
    #     content=UserProfile(
    #         name="Alex",
    #         preferred_name="Lex",
    #         response_style_preference="casual and witty with appropriate emojis",
    #         special_skills=[
    #             "Python programming",
    #             "AI development",
    #             "competitive speedcubing",
    #         ],
    #         other_preferences=[
    #             "prefers informal communication",
    #             "dislikes corporate-style interactions",
    #         ],
    #     ),
    # )
    ```

Choose between profiles and collections based on how you'll use the data: profiles excel when you need quick access to current state and when you have data requirements about what type of information you can store. They are also easy to present to a user for manual edting. Collections are useful when you want to track knowledge across many interactions without loss of information, and when you want to recall certain information contextually rather than every time.

### Episodic Memory: Past Experiences

Episodic memory preserves successful interactions as learning examples that guide future behavior. Unlike semantic memory which stores facts, episodic memory captures the full context of an interaction—the situation, the thought process that led to success, and why that approach worked. These memories help the agent learn from experience, adapting its responses based on what has worked before.

??? example "Defining and extracting episodes"
    
    ??? note "Setup"

        ```python
        from pydantic import BaseModel, Field
        from langmem import create_memory_manager

        class Episode(BaseModel):
            """An episode captures how to handle a specific situation, including the reasoning process
            and what made it successful."""
            
            observation: str = Field(
                ..., 
                description="The situation and relevant context"
            )
            thoughts: str = Field(
                ...,
                description="Key considerations and reasoning process"
            )
            action: str = Field(
                ...,
                description="What was done in response"
            )
            result: str = Field(
                ...,
                description="What happened and why it worked"
            )

        # highlight-next-line
        manager = create_memory_manager(
            "anthropic:claude-3-5-sonnet-latest",
            schemas=[Episode],
            instructions="Extract examples of successful interactions. Include the context, thought process, and why the approach worked.",
            enable_inserts=True,
        )

        # Example conversation
        conversation = [
            {"role": "user", "content": "What's a binary tree? I work with family trees if that helps"},
            {"role": "assistant", "content": "A binary tree is like a family tree, but each parent has at most 2 children. Here's a simple example:\n   Bob\n  /  \\\nAmy  Carl\n\nJust like in family trees, we call Bob the 'parent' and Amy and Carl the 'children'."},
            {"role": "user", "content": "Oh that makes sense! So in a binary search tree, would it be like organizing a family by age?"},
        ]
        ```

    ```python
    # Extract episode(s)
    episodes = manager.invoke({"messages": conversation})
    # Example episode:
    # [
    #     ExtractedMemory(
    #         id="f9194af3-a63f-4d8a-98e9-16c66e649844",
    #         content=Episode(
    #             observation="User struggled debugging a recursive "
    #                         "function for longest path in binary "
    #                         "tree, unclear on logic.",
    #             thoughts="Used explorer in treehouse village "
    #                      "metaphor to explain recursion:\n"
    #                      "- Houses = Nodes\n"
    #                      "- Bridges = Edges\n"
    #                      "- Explorer's path = Traversal",
    #             action="Reframed problem using metaphor, "
    #                    "outlined steps:\n"
    #                    "1. Check left path\n"
    #                    "2. Check right path\n"
    #                    "3. Add 1 for current position\n"
    #                    "Highlighted common bugs",
    #             result="Metaphor helped user understand logic. "
    #                    "Worked because it:\n"
    #                    "1. Made concepts tangible\n"
    #                    "2. Created mental model\n"
    #                    "3. Showed key steps\n"
    #                    "4. Pointed to likely bugs",
    #         ),
    #     )
    # ]
    ```


### Procedural Memory: System Instructions

Procedural memory encodes how an agent should behave and respond. It starts with system prompts that define core behavior, then evolves through feedback and experience. As the agent interacts with users, it refines these instructions, learning which approaches work best for different situations.

![Instructions update process](img/update-instructions.png)

??? example "Optimizing prompts based on feedback"

    ??? note "Setup"

        ```python
        from langmem import create_prompt_optimizer

        # highlight-next-line
        optimizer = create_prompt_optimizer(
            "anthropic:claude-3-5-sonnet-latest",
            kind="metaprompt",
            config={"max_reflection_steps": 3}
        )
        ```
    ```python
    prompt = "You are a helpful assistant."
    trajectory = [
        {"role": "user", "content": "Explain inheritance in Python"},
        {"role": "assistant", "content": "Here's a detailed theoretical explanation..."},
        {"role": "user", "content": "Show me a practical example instead"},
    ]
    optimized = optimizer.invoke({
        "trajectories": [(trajectory, {"user_score": 0})], 
        "prompt": prompt
    })
    print(optimized)
    # You are a helpful assistant with expertise in explaining technical concepts clearly and practically. When explaining programming concepts:

    # 1. Start with a brief, practical explanation supported by a concrete code example
    # 2. If the user requests more theoretical details, provide them after the practical example
    # 3. Always include working code examples for programming-related questions
    # 4. Pay close attention to user preferences - if they ask for a specific approach (like practical examples or theory), adapt your response accordingly
    # 5. Use simple, clear language and break down complex concepts into digestible parts

    # When users ask follow-up questions or request a different approach, immediately adjust your explanation style to match their preferences. If they ask for practical examples, provide them. If they ask for theory, explain the concepts in depth.
    ```



## Writing memories {#writing-memories}

Memories can form in two ways, each suited for different needs. Active formation happens during conversations, enabling immediate updates when critical context emerges. Background formation occurs between interactions, allowing deeper pattern analysis without impacting response time. This dual approach lets you balance responsiveness with thorough learning.

| Formation Type | Latency Impact | Update Speed | Processing Load | Use Case |
|----------------|----------------|--------------|-----------------|-----------|
| Active | Higher | Immediate | During Response | Critical Context Updates |
| Background | None | Delayed | Between/After Calls | Pattern Analysis, Summaries |

![Hot path vs background memory processing](img/hot_path_vs_background.png)

### Conscious Formation

You may want your agent to save memories "in the hot path." This active memory formation happens during the conversation, enabling immediate updates when critical context emerges. This approach is easy to implement and lets the agent itself choose how to store and update its memory. However, it adds perceptible latency to user interactions, and it adds one more obstacle to the agent's ability to satisfy the user's needs.

Check out the ["hot path" quickstart](../hot_path_quickstart.md) for an example of how to use this technique.

### Subconcious Formation

"Subconcious" memory formation refers to the technique of prompting an LLM to reflect on a conversation after it occurs (or after it has been inactive for some period), finding patterns and extracting insights without slowing down the immediate interaction or adding complexity to the agent's tool choice decisions. This approach is perfect for ensuring higher recall of exracted information.

Check out the ["background" quickstart](../background_quickstart.md) for an example of how to use this technique.

## Integration Patterns

LangMem's memory utilities are organized in two layers of integration patterns:

### 1. Core API {#functional-core}

At its heart, LangMem provides functions that transform memory state without side effects. These primitives are the building blocks for memory operations:

- [**Memory Managers**](../reference/memory.md#langmem.create_memory_manager): Extract new memories, update or remove outdated memories, and consolidate and generalize from existing memories based on new conversation information
- [**Prompt Optimizers**](../reference/prompt_optimization.md#langmem.create_prompt_optimizer): Update prompt rules and core behavior based on conversation information (with optional feedback)

These core functions do not depend on any particular database or storage system. You can use them in any application.

### 2. Stateful Integration

The next layer up depends on LangGraph's long-term memory store. These components use the core API above to transform memories that existin in the store and upsert/delete them as needed when new conversation information comesin:

- [**Store Managers**](../reference/memory.md#langmem.create_memory_store_manager): Automatically persist extracted memories
- [**Memory Management Tools**](../reference/tools.md#langmem.create_manage_memory_tool): Give agents direct access to memory operations

Use these if you're using LangGraph Platform or LangGraph OSS, since it's an easy way to add memory capabilities to your agents.


## Storage System {#storage-system}

??? note "Storage is optional"
    
    Remember that LangMem's core functionality is built around that don't require any specific storage layer. The storage features described here are part of LangMem's higher-level integration with LangGraph, useful when you want built-in persistence.


When using LangMem's stateful operators or platform services, the storage system is built on LangGraph's storage primitives, providing a flexible and powerful way to organize and access memories. The storage system is designed around two concepts:

### Memory Namespaces {#memory-namespaces}

Memories are organized into namespaces that allows for natural segmentation of data:

- **Multi-Level Namespaces**: Group memories by organization, user, application, or any other hierarchical structure
- **Contextual Keys**: Identify memories uniquely within their namespace
- **Structured Content**: Store rich, structured data with metadata for better organization

??? example "Organizing memories hierarchically"

    ```python
    # Organize memories by organization -> configurable user -> context
    namespace = ("acme_corp", "{user_id}", "code_assistant")
    ```

Namespaces can include template variables (such as `"{user_id}"`) to be populated at runtime from `configurable` fields in the `RunnableConfig`.
See [how to dynamically configure namespaces](../guides/dynamically_configure_namespaces.md) for an example, or the [NamespaceTemplate](../reference/utils.md#langmem.utils.NamespaceTemplate) reference docs for more details.

### Flexible Retrieval

If you use one of the managed APIs, LangMem will integrate directly with LangGraph's [BaseStore](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore) interface for memory storage and retrieval. The storage system supports multiple ways to retrieve memories:

- [**Direct Access**](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get): Get a specific memory by key
- [**Semantic Search**](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search): Find memories by semantic similarity
- [**Metadata Filtering**](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.search): Filter memories by their attributes

For more details on storage capabilities, see the [LangGraph Storage documentation](https://langchain-ai.github.io/langgraph/reference/store/).
