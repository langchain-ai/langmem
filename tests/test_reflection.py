import threading

from langmem.reflection import LocalReflectionExecutor


class BlockingReflector:
    namespace = staticmethod(lambda: ("memories",))

    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.calls: list[int] = []

    def invoke(self, payload: dict[str, int]) -> int:
        task_id = payload["id"]
        self.calls.append(task_id)
        if task_id == 1:
            self.first_started.set()
            if not self.release_first.wait(timeout=2):
                raise TimeoutError("first task was not released")
        return task_id


def test_replacing_running_task_does_not_cancel_its_future() -> None:
    reflector = BlockingReflector()
    executor = LocalReflectionExecutor(reflector, store=object())
    config = {"configurable": {}}

    try:
        first = executor.submit({"id": 1}, config, thread_id="same")
        assert reflector.first_started.wait(timeout=2)
        second = executor.submit({"id": 2}, config, after_seconds=30, thread_id="same")
        reflector.release_first.set()

        assert first.result(timeout=2) == 1
        assert second.cancel()
        assert reflector.calls == [1]
    finally:
        reflector.release_first.set()
        executor.shutdown(cancel_futures=True)


def test_old_task_cleanup_preserves_latest_pending_owner() -> None:
    executor = LocalReflectionExecutor(BlockingReflector(), store=object())
    config = {"configurable": {}}

    try:
        executor.submit({"id": 1}, config, after_seconds=30, thread_id="same")
        with executor._pending_tasks_lock:
            old_task = executor._pending_tasks["same"]

        latest_future = executor.submit(
            {"id": 2}, config, after_seconds=30, thread_id="same"
        )
        executor._discard_pending_task(old_task)

        with executor._pending_tasks_lock:
            assert executor._pending_tasks["same"].future is latest_future
    finally:
        executor.shutdown(cancel_futures=True)
