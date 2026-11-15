import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class StoreThread:
    """Runs every store operation on one dedicated thread: the daemon's single writer.

    Qdrant local mode and the SQLite connection are only ever touched from here,
    so requests and the worker never race on them.
    """

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rem-store")

    async def run(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self._pool, partial(fn, *args, **kwargs))

    def call(self, fn, *args, **kwargs):
        """Blocking variant for code outside the event loop (startup, shutdown)."""
        return self._pool.submit(fn, *args, **kwargs).result()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
