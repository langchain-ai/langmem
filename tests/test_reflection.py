"""Local reflection task cancellation tests.

Covers langchain-ai/langmem#210: `LocalReflectionExecutor.shutdown(cancel_futures=True)`
must cancel a task that is still waiting in the queue even when it was submitted with
`thread_id=None`, while leaving a task the worker already picked up to finish.
"""

import time
from concurrent.futures import Future
from threading import Event

from langchain_core.runnables import Runnable
from langgraph.store.memory import InMemoryStore
from langsmith import tracing_context

from langmem import ReflectionExecutor
from langmem.reflection import PendingTask
from langmem.utils import NamespaceTemplate


def _make_reflector(store, blocker_id="blocker"):
    """Build a reflector that blocks on `blocker_id` until the release event is set."""
    state = {"invoked": [], "entered": Event(), "release": Event()}

    class Reflector(Runnable):
        namespace = NamespaceTemplate(("memories",))

        def invoke(self, payload, config=None, **kwargs):
            if payload["id"] == blocker_id:
                state["entered"].set()
                if not state["release"].wait(5):
                    raise TimeoutError("blocker was not released")
            state["invoked"].append(payload["id"])
            store.put(("memories",), payload["id"], payload)
            return payload

    return Reflector(), state


def _wait_for_worker_to_take_queued_tasks(executor, timeout=5.0):
    """Barrier: wait until the worker has taken every task off the queue.

    Used to make the delayed-task case deterministic: after this returns, the worker holds
    the task and is waiting for its run time, so only the requeue path can cancel it.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with executor._queued_tasks_lock:
            if not executor._queued_tasks:
                return True
        time.sleep(0.01)
    return False


def _queue_while_blocked(*, cancel_futures, blocker_thread_id, queued_thread_id):
    """Submit a blocking task, then queue a second one and shut the executor down."""
    store = InMemoryStore()
    reflector, state = _make_reflector(store)
    with tracing_context(enabled=False):
        executor = ReflectionExecutor(reflector, store=store)
        try:
            executor.submit(
                {"id": "blocker"},
                config={"configurable": {}},
                thread_id=blocker_thread_id,
            )
            assert state["entered"].wait(5), "the blocker task never started"
            queued = executor.submit(
                {"id": "queued"},
                config={"configurable": {}},
                thread_id=queued_thread_id,
            )
            assert "queued" not in state["invoked"]
            executor.shutdown(wait=False, cancel_futures=cancel_futures)
        finally:
            state["release"].set()
            executor.shutdown(wait=True)
    return queued, state, store


def test_shutdown_cancels_queued_task_without_thread_id():
    queued, state, store = _queue_while_blocked(
        cancel_futures=True, blocker_thread_id=None, queued_thread_id=None
    )

    assert queued.cancelled() is True
    assert "queued" not in state["invoked"]
    assert store.get(("memories",), "queued") is None
    # The task that had already started is allowed to finish.
    assert state["invoked"] == ["blocker"]
    assert store.get(("memories",), "blocker") is not None


def test_shutdown_cancels_queued_task_with_thread_id():
    queued, state, store = _queue_while_blocked(
        cancel_futures=True,
        blocker_thread_id="blocker-thread",
        queued_thread_id="queued-thread",
    )

    assert queued.cancelled() is True
    assert "queued" not in state["invoked"]
    assert store.get(("memories",), "queued") is None


def test_shutdown_runs_queued_task_when_not_cancelling():
    queued, state, store = _queue_while_blocked(
        cancel_futures=False, blocker_thread_id=None, queued_thread_id=None
    )

    assert queued.cancelled() is False
    assert queued.result() == {"id": "queued"}
    assert state["invoked"] == ["blocker", "queued"]
    assert store.get(("memories",), "queued") is not None


def test_shutdown_cancels_delayed_task_while_the_worker_waits_for_its_run_time():
    """A task taken off the queue but not due yet must still be cancelled."""
    store = InMemoryStore()
    reflector, state = _make_reflector(store)
    with tracing_context(enabled=False):
        executor = ReflectionExecutor(reflector, store=store)
        try:
            queued = executor.submit(
                {"id": "delayed"},
                config={"configurable": {}},
                thread_id=None,
                after_seconds=3,
            )
            assert _wait_for_worker_to_take_queued_tasks(executor), (
                "the worker never took the delayed task off the queue"
            )
            executor.shutdown(wait=False, cancel_futures=True)
            # Longer than one get/sleep/requeue round of the worker.
            time.sleep(1.5)
        finally:
            executor.shutdown(wait=True)

    assert queued.cancelled() is True
    assert "delayed" not in state["invoked"]
    assert store.get(("memories",), "delayed") is None


def test_shutdown_runs_delayed_task_when_not_cancelling():
    store = InMemoryStore()
    reflector, state = _make_reflector(store)
    with tracing_context(enabled=False):
        executor = ReflectionExecutor(reflector, store=store)
        try:
            queued = executor.submit(
                {"id": "delayed"},
                config={"configurable": {}},
                thread_id=None,
                after_seconds=2,
            )
            assert _wait_for_worker_to_take_queued_tasks(executor)
            executor.shutdown(wait=True)
        finally:
            executor.shutdown(wait=True)

    assert queued.cancelled() is False
    assert queued.result() == {"id": "delayed"}
    assert state["invoked"] == ["delayed"]
    assert store.get(("memories",), "delayed") is not None


def test_shutdown_cancels_every_queued_threadless_task():
    store = InMemoryStore()
    reflector, state = _make_reflector(store)
    with tracing_context(enabled=False):
        executor = ReflectionExecutor(reflector, store=store)
        try:
            executor.submit(
                {"id": "blocker"}, config={"configurable": {}}, thread_id=None
            )
            assert state["entered"].wait(5), "the blocker task never started"
            threadless = [
                executor.submit(
                    {"id": f"queued-{index}"},
                    config={"configurable": {}},
                    thread_id=None,
                )
                for index in range(3)
            ]
            keyed = executor.submit(
                {"id": "keyed"},
                config={"configurable": {}},
                thread_id="other-thread",
            )
            executor.shutdown(wait=False, cancel_futures=True)
        finally:
            state["release"].set()
            executor.shutdown(wait=True)

    assert all(future.cancelled() for future in threadless)
    assert keyed.cancelled() is True
    assert state["invoked"] == ["blocker"]
    for index in range(3):
        assert store.get(("memories",), f"queued-{index}") is None
    assert store.get(("memories",), "keyed") is None


def test_requeue_refuses_a_delayed_task_once_queued_work_is_cancelled():
    """The worker must not put a delayed task back on the queue after cancellation."""
    store = InMemoryStore()
    reflector, _ = _make_reflector(store)
    executor = ReflectionExecutor(reflector, store=store)
    try:
        task = PendingTask(
            thread_id=None,
            payload={"id": "delayed"},
            after_seconds=5,
            submit_time=time.time(),
            future=Future(),
            cancel_event=Event(),
            config=None,
        )
        executor._cancel_queued_tasks()

        assert executor._requeue(task) is False
        assert task.future.cancelled() is True
        assert task.cancel_event.is_set() is True
    finally:
        executor.shutdown(wait=True)
