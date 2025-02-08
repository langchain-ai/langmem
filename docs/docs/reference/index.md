---
title: Reference
description: API reference for LangMem
---

# API Reference

Welcome to the LangMem API reference! The documentation is organized into three main sections:

## [Memory Management](memory.md)

Core memory management utilities:

- [`create_memory_manager`](memory.md#langmem.create_memory_manager) - Stateless memory extraction and updates
- [`create_memory_store_manager`](memory.md#langmem.create_memory_store_manager) - Stateful memory management with BaseStore

## [Memory Tools](tools.md)

Agent tools for memory management:

- [`create_manage_memory_tool`](tools.md#langmem.create_manage_memory_tool) - Tool for storing and updating memories
- [`create_search_memory_tool`](tools.md#langmem.create_search_memory_tool) - Tool for searching stored memories

## [Prompt Optimization](prompt_optimization.md)

Utilities for optimizing prompts:

- [`create_prompt_optimizer`](prompt_optimization.md#langmem.create_prompt_optimizer) - Single prompt optimization
- [`create_multi_prompt_optimizer`](prompt_optimization.md#langmem.create_multi_prompt_optimizer) - Multi-prompt system optimization

## [Utilities](utils.md)

- [`NamespaceTemplate`](utils.md#langmem.utils.NamespaceTemplate) - internal namespace template utility
- [`ReflectionExecutor`](utils.md#langmem.ReflectionExecutor) - Reflection executor to schedule memory management remotely or in a background thread.