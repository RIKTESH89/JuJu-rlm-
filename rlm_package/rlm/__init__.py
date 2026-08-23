"""Calls from inside the kernel back into the host agent.

The module is callable, so both of these do the same thing:

    answer = await rlm("Review auth.py for security issues", name="auth-reviewer")
    answer = await rlm.run("Review auth.py for security issues", name="auth-reviewer")

The host listens on a Unix socket whose path arrives in MYAGENT_HOST_SOCKET.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import ModuleType

SOCKET_ENV = "MYAGENT_HOST_SOCKET"


async def host_request(request_type: str, payload: dict | None = None) -> dict:
    """Ask the host to do something and return its result."""
    socket_path = os.environ.get(SOCKET_ENV)
    if not socket_path:
        raise RuntimeError(f"{SOCKET_ENV} is not set; no host to call")

    reader, writer = await asyncio.open_unix_connection(socket_path)
    try:
        request = {"type": request_type, "payload": payload or {}}
        writer.write(json.dumps(request).encode() + b"\n")
        await writer.drain()

        line = await reader.readline()
    finally:
        writer.close()

    if not line:
        raise RuntimeError("host closed the connection without responding")

    response = json.loads(line.decode())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "host reported an unknown error"))
    return response.get("result", {})


async def run(prompt: str, **options) -> str:
    """Run a child agent on `prompt` and return its final answer.

    Keyword arguments: name, model. Anything else is rejected rather than
    dropped — a silently ignored option looks like it worked.
    """
    allowed = {"name", "model"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError(
            "rlm.run() got unexpected keyword argument(s): {}; accepted: {}".format(
                ", ".join(unknown), ", ".join(sorted(allowed))
            )
        )

    result = await host_request(
        "rlm.run",
        {
            "prompt": prompt,
            "name": options.get("name"),
            "model": options.get("model"),
        },
    )
    return result["answer"]


async def list_subagents() -> list:
    """The children this agent has spawned, with status and timing."""
    result = await host_request("rlm.list_subagents")
    return result["subagents"]


class _CallableModule(ModuleType):
    """Lets `await rlm(...)` mean `await rlm.run(...)`."""

    async def __call__(self, prompt: str, **options) -> str:
        return await run(prompt, **options)


sys.modules[__name__].__class__ = _CallableModule
