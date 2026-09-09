---
title: How to Optimize Multiple Prompts
---

# How to Optimize Multiple Prompts

Optimize multiple prompts in a system based on conversation logs and feedback. This algorithm seeks to do three things:

1. Infer "gradients" or recommended changes to the system output based on the trajectories (and optional feedback)
2. Attribute system performance to each prompt step
3. Recommend updated changes to the prompts that most need adjustment

```python
from langmem import create_multi_prompt_optimizer

# Example team: researcher finds information, writer creates reports
conversations = [
    (
        [
            {"role": "user", "content": "Research quantum computing advances"},
            {
                "role": "assistant",
                "content": "Found several papers on quantum supremacy...",
            },
            {
                "role": "assistant",
                "content": "Recent quantum computing developments show...",
            },
            {"role": "user", "content": "The report is missing implementation details"},
        ],
        # No explicit feedback provided but the optimizer can infer from the conversation
        None,
    ),
    (
        [
            {"role": "user", "content": "Analyze new ML models"},
            {"role": "assistant", "content": "Key findings on architecture: ..."},
            {"role": "assistant", "content": "Based on the research, these models..."},
            {"role": "user", "content": "Great report, very thorough"},
        ],
        # Numeric score for the team as a whole
        {"score": 0.95},
    ),
]

# Define prompts for each role
prompts = [
    {
        "name": "researcher",
        "prompt": "You analyze technical papers and extract key findings",
    },
    {"name": "writer", "prompt": "You write clear reports based on research findings"},
]

# Create optimizer
optimizer = create_multi_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="gradient",  # Best for team dynamics
    config={"max_reflection_steps": 3},
)

# Update all prompts based on team performance
updated = optimizer.invoke({"trajectories": conversations, "prompts": prompts})
print(updated)
```

Output prompts:
```python
[
    {
        "name": "researcher",
        "prompt": """You analyze technical papers and extract key findings. For each analysis, include:
1. High-level overview of the main contributions
2. Technical implementation details and methodologies
3. Architectural components and design choices
4. Experimental results and performance metrics
5. Practical implications and limitations

Ensure your analysis maintains technical depth while remaining accessible. When discussing implementation details, include specific technical parameters, algorithms, and methodologies used. Structure your response to clearly separate these components.""",
    },
    {"name": "writer", "prompt": "You write clear reports based on research findings"},
]
```

See [Single Prompt Optimization](optimize_memory_prompt.md) for optimizing individual agents and for more information on the different optimization strategies.
