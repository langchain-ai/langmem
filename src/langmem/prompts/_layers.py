# This is not considered a part of the public API right now and may change at any time.
import asyncio
import functools
import re
import typing
from typing import Any, Literal, Optional, Union

from langchain_core.messages import AnyMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.store.base import Item, SearchItem
from langgraph.utils.config import get_store
from pydantic import BaseModel
from typing_extensions import TypedDict

from langmem import create_manage_memory_tool, create_memory_searcher
from langmem.utils import NamespaceTemplate

# Basically a declarative API for memories that could be composed in prompts.
# Formatting? (here vs. elsewhere)


class MessagesState(TypedDict, total=False):
    messages: list[AnyMessage]
    query: str | list[str]


class MemoryLayer(Runnable):
    __slots__ = (
        "name",
        "namespace",
        "kind",
        "update_instructions",
        "schemas",
        "limit",
        "_manager_tool",
        "_search_tool",
    )

    @typing.overload
    def __init__(
        self,
        name: str,
        namespace: tuple[str, ...],
        *,
        update_instructions: Optional[str] = None,
        schemas: Union[type[str], list[type[BaseModel]], list[dict]] = str,
        kind: Literal["multi"],
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        name: str,
        namespace: tuple[str, ...],
        *,
        kind: Literal["single"],
        update_instructions: Optional[str] = None,
        schemas: Union[type[str], list[type[BaseModel]], list[dict]] = str,
        limit: int = 1000,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        namespace: tuple[str, ...],
        *,
        kind: Literal["single", "multi"] = "multi",
        update_instructions: Optional[str] = None,
        schemas: Union[type[str], list[type[BaseModel]], list[dict]] = str,
        limit: int = 1000,
    ) -> None:
        """Initialize a memory layer.

        Args:
            name: Optional[str]: Human readable name for this memory layer. Sanitized version of this will also be used in the appropriate location to
                organize memory.
            namespace: Optional[tuple[str, ...]]: Scope for this memory layer (e.g. user_id, org_id)
            update_instructions: Optional[str]: System-prompt instructions for when and how to update the contents of this memory.
            schemas: Union[type[str], list[type[BaseModel]], list[dict]]: Schema for validating memories
            kind: Literal["single", "multi"]: Type of memory storage/lookup - "single" or "multi". Default is 'multi'.
            limit: int = 1000: If "multi" kind, the maximum number of memories to return
        """
        self.name = name
        self.update_instructions = update_instructions
        self._schemas = schemas
        self.kind = kind
        self.limit = limit
        if kind not in ("single", "multi"):
            raise ValueError(f"Unknown kind: {kind}")
        if kind == "single" and isinstance(schemas, (list, tuple)) and len(schemas) > 1:
            raise ValueError("Single memory layer cannot have multiple schemas")
        self.namespace = NamespaceTemplate(namespace + (_sanitize_name(name),))
        self._manager_tool = None
        self._search_tool = None
        self._search, self._asearch = create_search_utils(
            self.namespace,
            kind,  # type: ignore
            limit=limit,
        )

    def invoke(
        self,
        input: MessagesState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> list[SearchItem]:
        """Run the layer."""
        queries = _get_query(input) if self.kind == "multi" else []
        return self._search(queries)

    async def ainvoke(
        self,
        input: MessagesState,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> list[SearchItem]:
        """Run the layer."""
        queries = _get_query(input) if self.kind == "multi" else []
        return await self._asearch(queries)

    def get_manager_tool(self):
        if self._manager_tool is None:
            self._manager_tool = create_manage_memory_tool(
                namespace=self.namespace,
                instructions=self.update_instructions or "",
            )
        return self._manager_tool

    def as_tool(
        self,
        args_schema: Optional[type[BaseModel]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        arg_types: Optional[dict[str, type]] = None,
    ) -> BaseTool:
        if self._search_tool is None:
            self._search_tool = create_memory_searcher(
                namespace=self.namespace,
                schemas=self.schemas,
            )
        return self._search_tool


@typing.overload
def create_search_utils(
    namespace: NamespaceTemplate,
    kind: Literal["single"],
): ...


@typing.overload
def create_search_utils(
    namespace: NamespaceTemplate,
    kind: Literal["multi"],
    /,
    filter: dict[str, Any] | None = None,
    limit: int = 10,
    offset: int = 0,
): ...


def create_search_utils(
    namespace: NamespaceTemplate,
    kind: Literal["single", "multi"],
    *,
    filter: dict[str, Any] | None = None,
    limit: int = 10,
    offset: int = 0,
) -> tuple[
    typing.Callable[typing.Sequence[str], list[SearchItem]],
    typing.Callable[typing.Sequence[str], typing.Awaitable[list[SearchItem]]],
]:
    if kind == "single":
        return _search_single, _asearch_single
    else:
        return functools.partial(
            _search_multi,
            namespace=namespace,
            filter=filter,
            limit=limit,
            offset=offset,
        ), functools.partial(
            _asearch_multi,
            namespace=namespace,
            filter=filter,
            limit=limit,
            offset=offset,
        )


def _search_single(
    _: typing.Sequence[str], /, namespace: NamespaceTemplate, **kwargs: Any
) -> list[SearchItem]:
    store = get_store()
    namespace = namespace()
    item = store.get(namespace, key="memory")
    if item:
        return [
            SearchItem(
                namespace=item.namespace,
                key=item.key,
                value=item.value,
                created_at=item.created_at,
                updated_at=item.updated_at,
                score=None,
            )
        ]
    return []


async def _asearch_single(
    _: typing.Sequence[str], /, namespace: NamespaceTemplate, **kwargs: Any
) -> list[SearchItem]:
    store = get_store()
    namespace = namespace()
    item = await store.aget(namespace, key="memory")
    if item:
        return [
            SearchItem(
                namespace=item.namespace,
                key=item.key,
                value=item.value,
                created_at=item.created_at,
                updated_at=item.updated_at,
                score=None,
            )
        ]
    return []


def _search_multi(
    queries: typing.Sequence[str],
    *,
    namespace: NamespaceTemplate,
    filter: dict[str, Any] | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[SearchItem]:
    store = get_store()
    all_items = []
    # Note: offset wouldn't really work for multi-query
    # this is also not concurrent. Recommed async
    namespace = namespace()
    for q in queries:
        all_items.append(
            store.search(namespace, query=q, filter=filter, limit=limit, offset=offset)
        )
    return _sort_multiple(all_items, limit)


async def _asearch_multi(
    queries: typing.Sequence[str],
    /,
    *,
    namespace: NamespaceTemplate,
    filter: dict[str, Any] | None = None,
    limit: int = 10,
    offset: int = 0,
) -> list[SearchItem]:
    store = get_store()
    namespace = namespace()
    all_items = await asyncio.gather(
        *(
            store.asearch(namespace, query=q, filter=filter, limit=limit, offset=offset)
            for q in queries
        )
    )
    return _sort_multiple(all_items, limit)


def _get_query(state: MessagesState) -> list[str]:
    query = state.get("query")
    if query is None:
        messages_ = state.get("messages") or []
        query = ["\n".join(m.pretty_repr() for m in messages_[-3:])]
    if isinstance(query, str):
        query = [query]
    return query


def _sort_multiple(
    items: typing.Sequence[typing.Sequence[Union[Item, SearchItem]]], limit: int
) -> list[SearchItem]:
    """Sort and deduplicate search items by score, returning top k results.

    Args:
        items: Sequence of Items or SearchItems to sort
        limit: Maximum number of items to return

    Returns:
        List of unique SearchItems, sorted by score in descending order
    """
    unique_items: dict[tuple[str, ...], SearchItem] = {}
    for group in items:
        for item in group:
            key = (*item.namespace, item.key)
            existing = unique_items.get(key)
            if not isinstance(item, SearchItem):
                item = SearchItem(
                    namespace=item.namespace,
                    key=item.key,
                    value=item.value,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    score=-9999,
                )
            elif item.score is None:
                item.score = -9999

            if existing is None or item.score > existing.score:  # type: ignore
                unique_items[key] = item

    return sorted(
        unique_items.values(), key=lambda x: getattr(x, "score", -9999), reverse=True
    )[:limit]


_name_sanitizer = re.compile(r"[^a-zA-Z0-9]+")


def _sanitize_name(name: str) -> str:
    return _name_sanitizer.sub("-", name).strip("-")
