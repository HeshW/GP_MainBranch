from __future__ import annotations

import asyncio
from functools import wraps
from inspect import isawaitable
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")


def run_async(value: Coroutine[Any, Any, T] | T) -> T:
    """Run async helpers from sync entrypoints used by tests and CLI tools."""
    if not isawaitable(value):
        return value

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError("run_async cannot be used inside an active event loop")
    return asyncio.run(value)


def sync_from_async(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_async(func(*args, **kwargs))

    return wrapper
