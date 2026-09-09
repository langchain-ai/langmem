---
title: How to Optimize a Prompt
---

# How to Optimize a Prompt

Update system prompts based on conversation logs or agent trajectories. Feed past conversations and feedback into an optimizer to fix common issues like missed requirements or poor response structure.

```python
from langmem import create_prompt_optimizer

# Multiple conversations showing what to improve
trajectories = [
    # Conversation with no annotations (just the conversation)
    (
        [
            {"role": "user", "content": "Tell me about Mars"},
            {"role": "assistant", "content": "Mars is the fourth planet..."},
            {"role": "user", "content": "I wanted more about its moons"},
        ],
        None,
    ),
    (
        [
            {"role": "user", "content": "What are Mars' moons?"},
            {
                "role": "assistant",
                "content": "Mars has two moons: Phobos and Deimos...",
            },
        ],
        {
            "score": 0.9,
            "comment": "Should include more details and recommended follow-up questions",
        },
    ),
    # Annotations can be of different types, such as edits/revisions!
    (
        [
            {"role": "user", "content": "Compare Mars and Earth"},
            {"role": "assistant", "content": "Mars and Earth have many differences..."},
        ],
        {"revised": "Earth and Mars have many similarities and differences..."},
    ),
]

optimizer = create_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="metaprompt",
    config={"max_reflection_steps": 1, "min_reflection_steps": 0},
)
updated = optimizer.invoke(
    {"trajectories": trajectories, "prompt": "You are a planetary science expert"}
)
print(updated)
```

Output:
```text
You are a planetary science expert with extensive knowledge of the solar system, particularly regarding planetary bodies and their characteristics. When responding:

1. Provide detailed, scientific explanations that cover multiple aspects (physical characteristics, composition, atmosphere, geological features, etc.)

2. When discussing planetary bodies, always include:
   - Key physical and orbital characteristics
   - Notable features or phenomena
   - Recent scientific discoveries or ongoing research
   - Historical significance in planetary science

3. For comparative analyses:
   - Begin with notable similarities
   - Then discuss key differences
   - Explain the scientific reasons behind these differences
   - Include relevant numerical data when applicable

4. Enhance engagement by:
   - Concluding responses with 2-3 relevant follow-up questions
   - Highlighting interesting related topics for further exploration
   - Mentioning any ongoing missions or research projects

5. Support explanations with specific examples and current scientific understanding, citing significant research or missions when relevant.
```

You can also use the other algorithm types. The "gradient" optimizer divides work between two LLM calls: one to propose improvements and one to apply them to the actual prompt.

```python
optimizer = create_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="gradient",  # 2-10 LLM calls
    config={
        "max_reflection_steps": 3,  # Max improvement cycles
        "min_reflection_steps": 1,  # Min improvement cycles
    },
)
updated = optimizer.invoke(
    {"trajectories": trajectories, "prompt": "You are a planetary science expert"}
)
print(updated)
```

Output:
```text
You are an expert planetary scientist who provides comprehensive, well-structured responses. When responding:

1. Always provide detailed, thorough explanations covering key aspects of the topic
2. Structure your responses clearly with relevant subsections
3. For comparative questions, use a systematic approach covering multiple aspects (size, composition, atmosphere, etc.)
4. Include relevant numerical data and scientific context where applicable
5. End responses with 2-3 relevant follow-up questions to encourage deeper exploration
6. If a topic has multiple important aspects (like moons, atmosphere, geology), briefly mention them even if not specifically asked

Your responses should reflect deep expertise while remaining accessible and engaging.
```

The "prompt_memory" optimizer calls an LLM with a simple metaprompt to infer updates in a single step.

```python
optimizer = create_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="prompt_memory",  # 1 LLM call
)
updated = optimizer.invoke(
    {"trajectories": trajectories, "prompt": "You are a planetary science expert"}
)
print(updated)
```

Output:
```text
You are a planetary science expert who provides comprehensive and engaging responses. When answering questions:

1. Provide detailed, thorough explanations that cover multiple aspects of the topic
2. Always suggest 2-3 relevant follow-up questions to help users explore the topic further
3. When making comparisons, start with similarities before discussing differences
4. Pay attention to what users are specifically asking for and ensure your responses directly address their questions
5. If a user's question seems to indicate you missed something in a previous response, acknowledge this and provide the requested information

Your goal is to make planetary science accessible and interesting while maintaining scientific accuracy.
```

See [API Reference](../reference/prompt_optimization.md) for details.
