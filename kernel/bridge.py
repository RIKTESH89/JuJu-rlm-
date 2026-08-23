"""Host side of the kernel->host bridge.

The kernel is a separate OS process, so code running inside it calls back over a
Unix domain socket. The server runs as its own asyncio task: a cell that calls
the host cannot finish until the host answers, so the host must be able to
answer while execute() is still waiting on that same cell.

Wire format is newline-delimited JSON:
    request   {"type": str, "payload": dict}
    response  {"ok": true, "result": {...}} | {"ok": false, "error": str}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

Handler = Callable[[dict], Awaitable[dict]]

SOCKET_NAME = "host.sock"


async def handle_ping(payload: dict) -> dict:
    return {"pong": True}


class HostBridge:
    """Serves host capabilities to code running inside the kernel."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.socket_path = self.session_dir / SOCKET_NAME
        self.handlers: Dict[str, Handler] = {"ping": handle_ping}
        self.server: Optional[asyncio.AbstractServer] = None
        self._server_task: Optional[asyncio.Task] = None

    def register(self, request_type: str, handler: Handler) -> None:
        self.handlers[request_type] = handler

    @property
    def serving(self) -> bool:
        return self.server is not None

    async def _socket_is_live(self) -> bool:
        """True if something is actually listening on our path right now."""
        try:
            reader, writer = await asyncio.open_unix_connection(str(self.socket_path))
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return False
        writer.close()
        return True

    async def _clear_stale_socket(self) -> None:
        """A socket file may be a live neighbour or a corpse. Only remove corpses."""
        if not self.socket_path.exists():
            return

        if await self._socket_is_live():
            raise RuntimeError(
                f"socket {self.socket_path} is owned by a live session; "
                "refusing to take it over"
            )

        # Nothing answered, so this is a leftover from a crash.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)

    async def start(self) -> None:
        """Begin listening. Serving runs as its own task."""
        if self.server is not None:
            return

        self.session_dir.mkdir(parents=True, exist_ok=True)
        await self._clear_stale_socket()

        self.server = await asyncio.start_unix_server(
            self._serve_connection, path=str(self.socket_path), start_serving=False
        )
        self._server_task = asyncio.create_task(self.server.serve_forever())

        # serve_forever() is what actually starts listening, so hand control to
        # it before returning; otherwise the first connection is refused.
        while not self.server.is_serving():
            await asyncio.sleep(0)

    async def stop(self) -> None:
        """Shut the server down and remove the socket. Safe to call twice."""
        if self.server is not None:
            self.server.close()
            with contextlib.suppress(Exception):
                await self.server.wait_closed()
            self.server = None

        if self._server_task is not None:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None

        self.remove_socket()

    def remove_socket(self) -> None:
        """Unlink the socket file. Synchronous, so a crash handler can call it."""
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.socket_path)

    async def _serve_connection(self, reader, writer) -> None:
        """One connection: read requests line by line, answer each one."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return

                response = await self._dispatch(line)
                writer.write(json.dumps(response).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()

    async def _dispatch(self, line: bytes) -> dict:
        """Turn one request line into a response dict. Never raises."""
        try:
            request = json.loads(line.decode())
        except ValueError as exc:
            return {"ok": False, "error": f"malformed request: {exc}"}

        if not isinstance(request, dict):
            return {"ok": False, "error": "request must be a JSON object"}

        request_type = request.get("type")
        if not isinstance(request_type, str):
            return {"ok": False, "error": "request is missing a string 'type'"}

        handler = self.handlers.get(request_type)
        if handler is None:
            known = ", ".join(sorted(self.handlers)) or "none"
            return {
                "ok": False,
                "error": f"unknown request type {request_type!r}; known types: {known}",
            }

        payload = request.get("payload") or {}
        if not isinstance(payload, dict):
            return {"ok": False, "error": "'payload' must be a JSON object"}

        try:
            result = await handler(payload)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        return {"ok": True, "result": result}
