"""`rem daemon`: one process that owns the stores, serves the API and runs the worker.

Exit codes: 0 clean stop, 2 another instance is running, 3 configuration error.
"""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import sys

import uvicorn

from rem.config import RemConfig
from rem.daemon.app import build_app
from rem.daemon.lock import AlreadyRunning, InstanceLock
from rem.daemon.service import RemService
from rem.daemon.extraction import ExtractionHandler
from rem.daemon.worker import JobHandler, Worker
from rem.errors import RemError
from rem.llm.factory import create_llm

EXIT_OK, EXIT_ALREADY_RUNNING, EXIT_CONFIG = 0, 2, 3

log = logging.getLogger("rem.daemon")


class _Server(uvicorn.Server):
    """uvicorn re-raises SIGTERM after shutting down, which would kill us before cleanup.
    rem installs its own handlers instead, so the daemon always exits through its finally blocks."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


def run_daemon(cfg: RemConfig, handler: JobHandler | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.umask(0o077)  # socket, lock and data files are private to this user
    try:
        sock = cfg.socket_path
    except ValueError as e:
        log.error("%s", e)
        return EXIT_CONFIG

    lock = InstanceLock(cfg.lock_path)
    try:
        lock.acquire()
    except AlreadyRunning as e:
        log.error("%s", e)
        return EXIT_ALREADY_RUNNING
    try:
        # We hold the lock, so any socket file here was left by a crashed daemon.
        sock.unlink(missing_ok=True)
        sock.parent.mkdir(parents=True, exist_ok=True)
        try:
            if handler is None:
                llm = create_llm(cfg.llm)  # raises before any network call if remote is not allowed
                if llm is not None:
                    handler = ExtractionHandler(llm, cfg.extraction)
            service = RemService(cfg)
        except RemError as e:
            log.error("%s", e)
            return EXIT_CONFIG
        try:
            asyncio.run(_serve(service, cfg, handler))
        finally:
            service.close()
            sock.unlink(missing_ok=True)
        log.info("stopped")
        return EXIT_OK
    finally:
        lock.release()


async def _serve(service: RemService, cfg: RemConfig, handler: JobHandler | None) -> None:
    sock = cfg.socket_path
    recovered = await service.recover_jobs()
    if recovered:
        log.info("recovered %d interrupted job(s)", recovered)

    # Bind the socket ourselves: uvicorn's own uds= path chmods it to 0666 after binding.
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(sock))
    os.chmod(sock, 0o600)
    server = _Server(uvicorn.Config(build_app(service), log_level="warning", lifespan="off"))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: setattr(server, "should_exit", True))
    stop = asyncio.Event()
    worker = None
    if handler is None:
        log.info("extraction disabled: no [llm] configured; jobs stay pending")
    else:
        worker = asyncio.create_task(Worker(service, handler).run(stop))
    log.info("listening on %s (pid %d)", sock, os.getpid())
    try:
        await server.serve(sockets=[listener])  # returns once a signal sets should_exit
    finally:
        stop.set()
        if worker is not None:
            await worker
        listener.close()
