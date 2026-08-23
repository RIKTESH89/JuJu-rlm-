import asyncio
import contextlib
import json
import shutil
import uuid
from pathlib import Path

from kernel.bridge import HostBridge
from kernel.manager import KernelManager


def session_dir() -> Path:
    """A short-lived session dir. Kept short: Unix socket paths cap near 104 bytes."""
    return Path.home() / ".myagent" / "sessions" / f"t{uuid.uuid4().hex[:8]}"


@contextlib.asynccontextmanager
async def bridged_kernel():
    """A running bridge plus a kernel that knows how to reach it."""
    directory = session_dir()
    bridge = HostBridge(directory)
    await bridge.start()
    manager = KernelManager(host_socket=bridge.socket_path)
    try:
        yield bridge, manager
    finally:
        await manager.shutdown()
        await bridge.stop()
        shutil.rmtree(directory, ignore_errors=True)


async def raw_request(socket_path: Path, request: dict) -> dict:
    """Speak the protocol directly, without going through the kernel."""
    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    writer.write(json.dumps(request).encode() + b"\n")
    await writer.drain()
    line = await reader.readline()
    writer.close()
    return json.loads(line.decode())


def test_ping_over_the_socket():
    async def body():
        async with bridged_kernel() as (bridge, _):
            response = await raw_request(bridge.socket_path, {"type": "ping"})
            assert response == {"ok": True, "result": {"pong": True}}

    asyncio.run(body())


def test_unknown_type_is_refused_not_ignored():
    async def body():
        async with bridged_kernel() as (bridge, _):
            response = await raw_request(bridge.socket_path, {"type": "nope"})
            assert response["ok"] is False
            assert "unknown request type" in response["error"]
            assert "nope" in response["error"]

            malformed = await raw_request(bridge.socket_path, {"payload": {}})
            assert malformed["ok"] is False
            assert "type" in malformed["error"]

    asyncio.run(body())


def test_cell_can_call_the_host():
    """The required proof: a model-executed cell pings the host and gets an answer."""

    async def body():
        async with bridged_kernel() as (_, kernel):
            result = await kernel.execute("print(await rlm.host_request('ping'))")
            assert result.error is None
            assert result.stdout.strip() == "{'pong': True}"

    asyncio.run(body())


def test_bootstrap_names_exist_without_importing():
    async def body():
        async with bridged_kernel() as (_, kernel):
            result = await kernel.execute("print(sorted({'rlm','os','json','re','Path'} & set(dir())))")
            assert result.error is None
            assert result.stdout.strip() == "['Path', 'json', 'os', 're', 'rlm']"

    asyncio.run(body())


def test_host_error_becomes_runtime_error_in_kernel():
    async def body():
        async with bridged_kernel() as (bridge, kernel):
            async def explode(payload):
                raise ValueError("handler blew up")

            bridge.register("explode", explode)
            result = await kernel.execute("await rlm.host_request('explode')")
            assert result.error is not None
            assert "RuntimeError" in result.error
            assert "handler blew up" in result.error

    asyncio.run(body())


def test_server_answers_while_execute_is_waiting():
    """The deadlock proof.

    While execute() is blocked waiting for a cell to finish, the bridge must
    still answer requests. If the server shared that task, this would hang.
    """

    async def body():
        async with bridged_kernel() as (bridge, kernel):
            await kernel.execute("warm = 1")  # pay kernel startup up front

            slow_cell = asyncio.create_task(
                kernel.execute("import time; time.sleep(6); print('cell done')")
            )
            await asyncio.sleep(1)  # let the cell get going
            assert not slow_cell.done()

            # host answers *now*, mid-cell
            response = await asyncio.wait_for(
                raw_request(bridge.socket_path, {"type": "ping"}), timeout=5
            )
            assert response["result"] == {"pong": True}
            assert not slow_cell.done(), "cell finished early; concurrency unproven"

            result = await asyncio.wait_for(slow_cell, timeout=30)
            assert result.stdout.strip() == "cell done"

    asyncio.run(body())


def test_cell_that_calls_host_does_not_deadlock():
    """A cell that blocks on the host completes — the true deadlock scenario."""

    async def body():
        async with bridged_kernel() as (bridge, kernel):
            async def slow_pong(payload):
                await asyncio.sleep(2)
                return {"pong": "eventually"}

            bridge.register("slow", slow_pong)
            result = await asyncio.wait_for(
                kernel.execute("print(await rlm.host_request('slow'))"), timeout=60
            )
            assert result.error is None
            assert result.stdout.strip() == "{'pong': 'eventually'}"

    asyncio.run(body())
