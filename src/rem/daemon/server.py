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
from rem.daemon.schemas import IngestBody
from rem.daemon.service import RemService
from rem.daemon.extraction import ExtractionHandler
from rem.daemon.worker import JobHandler, Worker
from rem.errors import RemError
from rem.llm.factory import create_llm
from rem.projects import resolve_project
from rem.spool import drain_spool, spool_count

EXIT_OK, EXIT_ALREADY_RUNNING, EXIT_CONFIG = 0, 2, 3
SPOOL_POLL_SECONDS = 30  # a hook that timed out spools without starting a daemon: pick its turns up while running

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
        llm = None
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
            recovered = service.store.call(service.ingest_store.recover)
            if recovered:
                log.info("recovered %d interrupted job(s)", recovered)
            _drain(service, cfg)
            asyncio.run(_serve(service, cfg, handler))
        finally:
            service.close()
            if llm is not None:
                llm.close()
            sock.unlink(missing_ok=True)
        log.info("stopped")
        return EXIT_OK
    finally:
        lock.release()


_spool_failures: dict[str, int] = {}  # per-file ingest failure counts, kept for the daemon process's lifetime


def _drain(service: RemService, cfg: RemConfig) -> None:
    """Ingest turns that hooks spooled while the daemon was down (blocking; runs the ingest on the store thread)."""

    def ingest(body: dict) -> None:
        b = IngestBody.model_validate(body)  # same validation as POST /ingest; ValueError -> spool/bad/
        cwd = b.cwd_path()
        project = resolve_project(cwd) if cwd is not None else None  # git runs here, not on the store thread
        service.store.call(service.ingest_store.ingest, b.source, b.session_id, cwd,
                           b.turns_for_ingest(), b.enqueue, project)

    drained = drain_spool(cfg.data_dir, ingest, _spool_failures)
    if drained:
        log.info("drained %d spooled ingest(s)", drained)


async def _drain_periodically(server: uvicorn.Server, service: RemService, cfg: RemConfig) -> None:
    """Drain again as soon as hooks can reach us (turns spooled while we were starting up), then
    every SPOOL_POLL_SECONDS while spool files exist (turns a hook spooled after a timeout)."""
    while not server.started:
        await asyncio.sleep(0.02)
    loop = asyncio.get_running_loop()
    while True:
        if spool_count(cfg.data_dir):
            try:
                await loop.run_in_executor(None, _drain, service, cfg)
            except Exception as e:  # keep serving; the next round retries
                log.error("spool drain failed: %s", type(e).__name__)
        await asyncio.sleep(SPOOL_POLL_SECONDS)


async def _serve(service: RemService, cfg: RemConfig, handler: JobHandler | None) -> None:
    sock = cfg.socket_path
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
    late_drain = asyncio.create_task(_drain_periodically(server, service, cfg))
    log.info("listening on %s (pid %d)", sock, os.getpid())
    try:
        await server.serve(sockets=[listener])  # returns once a signal sets should_exit
    finally:
        late_drain.cancel()
        stop.set()
        if worker is not None:
            await worker
        listener.close()
