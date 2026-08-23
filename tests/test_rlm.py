import asyncio
import contextlib
import os
import shutil
import uuid
from pathlib import Path

import pytest

from kernel.session import Session


def session_dir() -> Path:
    """Short path: Unix sockets cap out near 104 bytes."""
    return Path.home() / ".myagent" / "sessions" / f"t{uuid.uuid4().hex[:8]}"


@contextlib.asynccontextmanager
async def root(max_depth=None):
    directory = session_dir()
    previous = os.environ.get("RLM_MAX_DEPTH")
    if max_depth is not None:
        os.environ["RLM_MAX_DEPTH"] = str(max_depth)
    try:
        session = Session.root(directory, asyncio.get_event_loop())
        await session.start()
        try:
            yield session
        finally:
            await session.dispose()
            shutil.rmtree(directory, ignore_errors=True)
    finally:
        if previous is None:
            os.environ.pop("RLM_MAX_DEPTH", None)
        else:
            os.environ["RLM_MAX_DEPTH"] = previous


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_child_returns_its_final_answer():
    """await rlm(...) blocks and returns the child's answer text."""

    async def body():
        async with root() as session:
            answer = await session.kernel.execute(
                "print(await rlm('say exactly HELLO and nothing else'))"
            )
            assert answer.error is None, answer.error
            assert "HELLO" in answer.stdout

    asyncio.run(body())


def test_depth_limit_stops_grandchildren():
    """Default max depth 1: a child may not spawn its own child."""

    async def body():
        async with root() as session:
            assert session.depth == 0
            assert session.max_depth == 1

            # the child is a real session at depth 1; ask *it* to spawn
            answer = await session.spawn(
                "Run this exact line and report what happens: "
                "await rlm('anything at all')"
            )
            assert isinstance(answer, str)

            child = list(session.children.values())[0]
            assert child.depth == 1

            with pytest.raises(RuntimeError, match="depth limit"):
                await child._handle_run({"prompt": "should be refused"})

    asyncio.run(body())


def test_grandchild_allowed_when_max_depth_is_two():
    async def body():
        async with root(max_depth=2) as session:
            assert session.max_depth == 2
            await session.spawn("say OK")
            child = list(session.children.values())[0]
            assert child.depth == 1
            assert child.max_depth == 2

            # at depth 1 with max 2, the child may still spawn
            answer = await child._handle_run({"prompt": "say OK"})
            assert isinstance(answer["answer"], str)
            assert child.children, "grandchild was not created"
            grandchild = list(child.children.values())[0]
            assert grandchild.depth == 2

    asyncio.run(body())


def test_unknown_kwarg_is_rejected():
    """A silently dropped option would look like it worked."""

    async def body():
        async with root() as session:
            result = await session.kernel.execute(
                "await rlm('hi', tempurature=0.5)"
            )
            assert result.error is not None
            assert "TypeError" in result.error
            assert "tempurature" in result.error

    asyncio.run(body())


def test_child_names_are_readable_and_unique():
    async def body():
        async with root() as session:
            await session.spawn("say OK", name="auth-reviewer")
            await session.spawn("say OK", name="auth-reviewer")
            names = sorted(session.children)
            assert names[0] == "auth-reviewer"
            assert names[1] == "auth-reviewer-2"

            await session.spawn("Review the login flow for bugs")
            derived = [n for n in session.children if n.startswith("review-the-login")]
            assert derived, sorted(session.children)

    asyncio.run(body())


def test_disposing_parent_kills_child_kernel():
    """No orphan kernels."""

    async def body():
        directory = session_dir()
        session = Session.root(directory, asyncio.get_event_loop())
        await session.start()

        await session.spawn("say OK")
        child = list(session.children.values())[0]

        # force the child's kernel to exist, then note its pid
        await child.kernel.execute("1")
        pid = await child.kernel_pid()
        assert pid and alive(pid)

        await session.dispose()
        for _ in range(50):
            if not alive(pid):
                break
            await asyncio.sleep(0.1)

        assert not alive(pid), f"child kernel {pid} outlived its parent"
        shutil.rmtree(directory, ignore_errors=True)

    asyncio.run(body())
