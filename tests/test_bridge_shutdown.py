import asyncio
import os
import shutil
import uuid
from pathlib import Path

import pytest

import agent
from kernel.bridge import HostBridge
from kernel.session import Session


def session_dir() -> Path:
    return Path.home() / ".myagent" / "sessions" / f"t{uuid.uuid4().hex[:8]}"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def sockets_under(directory: Path):
    return sorted(p for p in directory.rglob("*.sock"))


def fake_child_run(messages, tools, approve, model=None, raise_on_error=False, tokens=None, depth=0):
    tokens.billed_input += 1
    return "done"


def test_shutdown_removes_socket_and_finishes_server_task():
    async def body():
        directory = session_dir()
        bridge = HostBridge(directory)
        await bridge.start()

        assert bridge.socket_path.exists()
        assert bridge._server_task is not None
        assert not bridge._server_task.done()

        task = bridge._server_task
        await bridge.stop()

        assert not bridge.socket_path.exists(), "socket file left behind"
        assert task.done(), "server task still running"
        shutil.rmtree(directory, ignore_errors=True)

    asyncio.run(body())


def test_shutdown_twice_does_not_raise():
    async def body():
        directory = session_dir()
        bridge = HostBridge(directory)
        await bridge.start()

        await bridge.stop()
        await bridge.stop()  # must be a no-op, not an error
        shutil.rmtree(directory, ignore_errors=True)

    asyncio.run(body())


def test_dispose_without_a_kernel_does_not_raise():
    """The lazy case: a session where no code ever ran."""

    async def body():
        directory = session_dir()
        session = Session(directory, asyncio.get_event_loop())
        await session.start()

        assert session.kernel._km is None, "kernel should not have started"

        await session.dispose()
        await session.dispose()  # idempotent
        shutil.rmtree(directory, ignore_errors=True)

    asyncio.run(body())


def test_starting_over_a_dead_sessions_socket_succeeds():
    async def body():
        directory = session_dir()
        directory.mkdir(parents=True, exist_ok=True)
        leftover = directory / "host.sock"
        leftover.write_text("")  # a corpse: a file nothing is listening on
        assert leftover.exists()

        bridge = HostBridge(directory)
        await bridge.start()
        assert bridge.serving

        # and it really works, not just bound
        reader, writer = await asyncio.open_unix_connection(str(bridge.socket_path))
        writer.write(b'{"type": "ping"}\n')
        await writer.drain()
        assert b'"pong"' in await reader.readline()
        writer.close()

        await bridge.stop()
        shutil.rmtree(directory, ignore_errors=True)

    asyncio.run(body())


def test_starting_over_a_live_sessions_socket_raises():
    async def body():
        directory = session_dir()
        owner = HostBridge(directory)
        await owner.start()

        intruder = HostBridge(directory)
        with pytest.raises(RuntimeError, match="live session"):
            await intruder.start()

        # the owner is untouched
        assert owner.socket_path.exists()
        assert owner.serving

        await owner.stop()
        shutil.rmtree(directory, ignore_errors=True)

    asyncio.run(body())


def test_parent_shutdown_leaves_no_sockets_and_no_kernels(monkeypatch):
    async def body():
        directory = session_dir()
        parent = Session(directory, asyncio.get_event_loop(), depth=0, max_depth=1)
        await parent.start()

        monkeypatch.setattr(agent, "run", fake_child_run)
        await parent.spawn("first", name="one")
        await parent.spawn("second", name="two")
        assert len(parent.children) == 2

        pids = []
        for child in parent.children.values():
            await child.kernel.execute("1")  # force the kernel to exist
            pid = await child.kernel_pid()
            assert pid and alive(pid)
            pids.append(pid)

        assert len(sockets_under(directory)) == 3  # parent + two children

        await parent.dispose()

        assert sockets_under(directory) == [], "socket files survived shutdown"
        for pid in pids:
            for _ in range(50):
                if not alive(pid):
                    break
                await asyncio.sleep(0.1)
            assert not alive(pid), f"kernel {pid} outlived its parent"

        shutil.rmtree(directory, ignore_errors=True)

    asyncio.run(body())
