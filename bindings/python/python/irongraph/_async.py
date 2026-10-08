"""Async scheduling for the existing native database owner."""

import asyncio
import threading
import uuid
import weakref

_limits = weakref.WeakKeyDictionary()
_controls = weakref.WeakKeyDictionary()
_background = set()


def _track(task):
    _background.add(task)

    def finished(completed):
        _background.discard(completed)
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(finished)
    return task


async def _cancel_registered(owner, operation_id, work):
    while not work.done():
        try:
            if await asyncio.to_thread(owner.cancel, operation_id):
                return
        except RuntimeError:
            return
        await asyncio.sleep(0.001)


async def _close_opened(work):
    try:
        owner = await work
    except BaseException:
        return
    await asyncio.to_thread(owner.close)


async def _await_work(work):
    # Cancellation affects this waiter, while the native worker retains its owner.
    # Native cancellation errors are consumed by _track after a cancelled waiter.
    waiter = asyncio.get_running_loop().create_future()

    def completed(task):
        if waiter.done():
            return
        if task.cancelled():
            waiter.cancel()
        elif task.exception() is not None:
            waiter.set_exception(task.exception())
        else:
            waiter.set_result(task.result())

    work.add_done_callback(completed)
    return await waiter


async def invoke(method, args, kwargs, operation_key=None, owner=None, bounded=True, opening=False):
    loop = asyncio.get_running_loop()
    kwargs = dict(kwargs)
    operation_id = None
    if operation_key is not None:
        options = dict(kwargs.get(operation_key) or {})
        operation_id = options.get("operation_id")
        if operation_id is None:
            operation_id = "python:" + uuid.uuid4().hex
            options["operation_id"] = operation_id
        kwargs[operation_key] = options
    limits = _limits if bounded else _controls
    limit = limits.setdefault(loop, asyncio.Semaphore(64))
    await limit.acquire()
    cancelled = threading.Event()

    def execute():
        if cancelled.is_set():
            raise asyncio.CancelledError()
        return method(*args, **kwargs)

    work = _track(asyncio.create_task(asyncio.to_thread(execute)))
    work.add_done_callback(lambda _: limit.release())
    try:
        return await _await_work(work)
    except asyncio.CancelledError:
        if operation_id is not None or opening:
            cancelled.set()
        if operation_id is not None:
            _track(asyncio.create_task(_cancel_registered(owner, operation_id, work)))
        elif opening:
            _track(asyncio.create_task(_close_opened(work)))
        raise


async def enter(owner):
    return owner


async def exit(owner):
    await invoke(owner.close, (), {}, bounded=False)
    return False
