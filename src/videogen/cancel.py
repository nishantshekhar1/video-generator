"""Stop a run while a stage is still working."""

from __future__ import annotations

import subprocess
import threading
from typing import Any


class Stopped(Exception):
    """The user stopped this run."""


_events: dict[str, threading.Event] = {}
_procs: dict[str, list[subprocess.Popen[str]]] = {}
_lock = threading.Lock()
_local = threading.local()


def bind(run_id: str) -> None:
    _local.run_id = run_id


def current_run_id() -> str | None:
    return getattr(_local, "run_id", None)


def event_for(run_id: str) -> threading.Event:
    with _lock:
        event = _events.get(run_id)
        if event is None:
            event = threading.Event()
            _events[run_id] = event
        return event


def clear_stop(run_id: str) -> None:
    event_for(run_id).clear()


def request_stop(run_id: str) -> None:
    event_for(run_id).set()
    kill_running(run_id)


def raise_if_stopped() -> None:
    run_id = current_run_id()
    if run_id and event_for(run_id).is_set():
        raise Stopped("Stopped")


def _register(run_id: str, proc: subprocess.Popen[str]) -> None:
    with _lock:
        _procs.setdefault(run_id, []).append(proc)


def _unregister(run_id: str, proc: subprocess.Popen[str]) -> None:
    with _lock:
        procs = _procs.get(run_id, [])
        if proc in procs:
            procs.remove(proc)


def kill_running(run_id: str) -> None:
    with _lock:
        procs = list(_procs.get(run_id, []))
    for proc in procs:
        if proc.poll() is None:
            proc.kill()


class CommandResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_command(cmd: list[str], **kwargs: Any) -> CommandResult:
    """Run a process and kill it when this run is stopped."""
    raise_if_stopped()
    run_id = current_run_id() or ""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)
    if run_id:
        _register(run_id, proc)
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.4)
                break
            except subprocess.TimeoutExpired:
                if run_id and event_for(run_id).is_set():
                    proc.kill()
                    proc.communicate()
                    raise Stopped("Stopped")
        if run_id and event_for(run_id).is_set():
            raise Stopped("Stopped")
        return CommandResult(proc.returncode or 0, stdout or "", stderr or "")
    finally:
        if run_id:
            _unregister(run_id, proc)
