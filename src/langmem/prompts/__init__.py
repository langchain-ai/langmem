"""Functionality for optimizing and managing LLM prompts.

This module provides utilities for improving prompts based on conversation history:

1. Single Prompt Optimization:
    - `create_prompt_optimizer(model, kind="metaprompt"|"gradient"|"prompt_memory") -> PromptOptimizerProto`:
        Create an optimizer for improving individual prompts using different strategies:
        - metaprompt: Uses meta-prompting for structured improvements
        - gradient: Uses gradient-based techniques for iterative refinement
        - prompt_memory: Uses conversation history for context-aware updates

2. Multi-Prompt Optimization:
    - `create_multi_prompt_optimizer(model, kind="metaprompt"|"gradient"|"prompt_memory") -> ((trajectories, prompts) -> list[Prompt])`:
        Create an optimizer for improving multiple related prompts while maintaining consistency
        and leveraging shared context between them. Particularly useful for prompt chains
        and multi-agent systems.

"""

from langmem.prompts.optimization import (
    create_multi_prompt_optimizer,
    create_prompt_optimizer,
)

__all__ = [
    "create_multi_prompt_optimizer",
    "create_prompt_optimizer",
]
