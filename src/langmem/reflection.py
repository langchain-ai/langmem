import queue
import threading
import time
import typing
import uuid
from concurrent.futures import Future
from contextvars import Context, copy_context
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Protocol

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.config import get_config
from langgraph.constants import CONF, CONFIG_KEY_STORE
from langgraph.store.base import BaseStore
from langgraph_sdk import get_client, get_sync_client
from langsmith.utils import ContextThreadPoolExecutor
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient
SENTINEL = object()


class MemoryItem(TypedDict):
    """Represents a single document or data entry in the graph's Store.

    Items are used to store cross-thread memories.
    """

    namespace: list[str]
    """The namespace of the item. A namespace is analogous to a document's directory."""
    key: str
    """The unique identifier of the item within its namespace.
    
    In general, keys needn't be globally unique.
    """
    value: dict[str, Any]
    """The value stored in the item. This is the document itself."""
    created_at: datetime
    """The timestamp when the item was created."""
    updated_at: datetime
    """The timestamp when the item was last updated."""
    score: Optional[float]


class Executor(Protocol):
    def submit(
        self,
        payload: dict[str, Any],
        /,
        after_seconds: int = 0,
        thread_id: Optional[typing.Union[str, uuid.UUID]] = None,
    ) -> Future: ...

    def search(
        self,
        query: Optional[str] = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryItem]: ...

    async def asearch(
        self,
        query: Optional[str] = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryItem]: ...

    def __enter__(self) -> "Executor": ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool: ...


@typing.overload
def ReflectionExecutor(
    namespace: str | tuple[str, ...],
    reflector: str,
    /,
    *,
    url: Optional[str] = None,
    client: Optional["LangGraphClient"] = None,
    sync_client: Optional["SyncLangGraphClient"] = None,
) -> "RemoteReflectionExecutor": ...


@typing.overload
def ReflectionExecutor(
    namespace: str | tuple[str, ...],
    reflector: Runnable,
    /,
    *,
    store: BaseStore,
) -> "LocalReflectionExecutor": ...


def ReflectionExecutor(
    namespace: str | tuple[str, ...],
    reflector: Runnable | str,
    /,
    *,
    url: Optional[str] = None,
    client: Optional["LangGraphClient"] = None,
    sync_client: Optional["SyncLangGraphClient"] = None,
    store: Optional["BaseStore"] = None,
) -> Executor:
    """Create a reflection executor for either local or remote execution.

    Args:
        namespace: A string to identify the reflection domain (e.g. "user:123").
        reflector: Either a callable that implements the reflection logic or a string that names a remote graph.
        url: Optional URL for remote processing.
        client: Optional LangGraph client for remote processing.
        sync_client: Optional sync LangGraph client for remote processing.
        store: Required store for local processing.

    Returns:
        Either a LocalReflectionExecutor or RemoteReflectionExecutor based on the reflector type.
    """
    if isinstance(reflector, str):
        return RemoteReflectionExecutor(
            namespace, reflector, url=url, client=client, sync_client=sync_client
        )
    else:
        if store is None:
            raise ValueError("store is required for local reflection")
        return LocalReflectionExecutor(namespace, reflector, store)


class RemoteReflectionExecutor:
    """Handles remote reflection tasks via LangGraph client."""

    def __init__(
        self,
        namespace: str | tuple[str, ...],
        reflector: str,
        *,
        url: Optional[str] = None,
        client: Optional["LangGraphClient"] = None,
        sync_client: Optional["SyncLangGraphClient"] = None,
    ):
        self.namespace = namespace if isinstance(namespace, tuple) else (namespace,)
        self._assistant_id = reflector
        self._aclient = client or get_client(url=url)
        self._client = sync_client or get_sync_client(url=url)
        self._executor = ContextThreadPoolExecutor()

    def submit(
        self,
        payload: dict[str, Any],
        /,
        after_seconds: int = 0,
        thread_id: Optional[typing.Union[str, uuid.UUID]] = SENTINEL,  # type: ignore[arg-type]
    ) -> Future:
        config = get_config()
        if thread_id is SENTINEL and CONF in config and "thread_id" in config[CONF]:
            thread_id = config["configurable"]["thread_id"]

        def task(thread_id: str | None):
            if (
                not thread_id
                and "configurable" in config
                and "thread_id" in config["configurable"]
            ):
                thread_id = config["configurable"]["thread_id"]

            self._client.runs.create(
                thread_id=thread_id,  # type: ignore
                assistant_id=self._assistant_id,
                input=payload,
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "namespace": self.namespace,
                    }
                },
                multitask_strategy="rollback",
                after_seconds=after_seconds,
                if_not_exists="create",
            )
            return None

        return self._executor.submit(task, str(thread_id) if thread_id else None)

    def search(
        self,
        query: Optional[str] = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryItem]:
        results = self._client.store.search_items(
            self.namespace, query=query, filter=filter, limit=limit, offset=offset
        )
        items = typing.cast(list[MemoryItem], results["items"])
        for it in items:
            it["namespace"] = tuple(it["namespace"])  # type: ignore
        return items

    async def asearch(
        self,
        query: Optional[str] = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryItem]:
        results = await self._aclient.store.search_items(
            self.namespace, query=query, filter=filter, limit=limit, offset=offset
        )
        items = typing.cast(list[MemoryItem], results["items"])
        for it in items:
            it["namespace"] = tuple(it["namespace"])  # type: ignore
        return items

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


class LocalReflectionExecutor:
    """Handles local reflection tasks with queuing and cancellation support."""

    def __init__(
        self, namespace: str | tuple[str, ...], reflector: Runnable, store: BaseStore
    ):
        self.namespace = namespace if isinstance(namespace, tuple) else (namespace,)
        self._reflector = reflector
        self._store = store
        self._task_queue = queue.PriorityQueue()
        self._pending_tasks: dict[str, PendingTask] = {}
        self._worker_running = True
        self._worker = threading.Thread(target=_process_queue(self), daemon=True)
        self._worker.start()

    def submit(
        self,
        payload: dict[str, Any],
        after_seconds: int = 0,
        thread_id: Optional[typing.Union[str, uuid.UUID]] = SENTINEL,  # type: ignore[arg-type]
    ) -> Future:
        config = get_config()
        if thread_id is SENTINEL:
            if CONF in config and "thread_id" in config[CONF]:
                thread_id = config["configurable"]["thread_id"]
        elif thread_id:
            thread_id = str(thread_id)
        thread_id = typing.cast(typing.Optional[str], thread_id)
        # Cancel any existing task with the same thread_id
        if thread_id in self._pending_tasks:
            existing = self._pending_tasks.get(thread_id)
            if existing:
                existing.cancel_event.set()
                existing.future.cancel()

        future = Future()
        cancel_event = threading.Event()
        task = PendingTask(
            thread_id=thread_id,
            payload=payload,
            after_seconds=after_seconds,
            submit_time=time.time(),
            future=future,
            cancel_event=cancel_event,
            context=copy_context(),
        )
        if thread_id:
            self._pending_tasks[thread_id] = task
        self._task_queue.put((time.time() + after_seconds, task))
        return future

    def search(
        self,
        query: Optional[str] = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryItem]:
        results = self._store.search(
            self.namespace, query=query, filter=filter, limit=limit, offset=offset
        )
        return typing.cast(list[MemoryItem], [it.dict() for it in results])

    async def asearch(
        self,
        query: Optional[str] = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[MemoryItem]:
        results = await self._store.asearch(
            self.namespace, query=query, filter=filter, limit=limit, offset=offset
        )
        return typing.cast(list[MemoryItem], [it.dict() for it in results])

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._worker_running = False
        if cancel_futures:
            for task in self._pending_tasks.values():
                task.cancel_event.set()
                task.future.cancel()
        if wait:
            self._worker.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False


class PendingTask(NamedTuple):
    """Represents a task pending execution in the reflection queue."""

    thread_id: str | None
    payload: dict
    after_seconds: int
    submit_time: float
    future: Future
    cancel_event: threading.Event
    context: Context


def _process_queue(self):
    while self._worker_running:
        try:
            execute_at, task = self._task_queue.get(timeout=1)

            now = time.time()
            if execute_at > now:
                time.sleep(min(execute_at - now, 1))
                if execute_at > time.time():
                    self._task_queue.put((execute_at, task))
                    continue

            if task.cancel_event.is_set():
                self._pending_tasks.pop(task.thread_id, None)
                task.future.set_result(None)
                continue

            try:
                config = get_config()
                configurable = config.setdefault(CONF, {})
                configurable[CONFIG_KEY_STORE] = self._store
                var_child_runnable_config.set(config)
                result = self._reflector.invoke(task.payload)
                task.future.set_result(result)
            except Exception as e:
                task.future.set_exception(e)
            finally:
                self._pending_tasks.pop(task.thread_id, None)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")


__all__ = ["LocalReflectionExecutor", "RemoteReflectionExecutor", "ReflectionExecutor"]
