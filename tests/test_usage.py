import asyncio
import contextlib
import shutil
import uuid
from pathlib import Path

import pytest

import agent
from kernel.session import CHILD_USAGE, Session


def session_dir() -> Path:
    return Path.home() / ".myagent" / "sessions" / f"t{uuid.uuid4().hex[:8]}"


@contextlib.asynccontextmanager
async def root():
    directory = session_dir()
    session = Session(directory, asyncio.get_event_loop(), depth=0, max_depth=1)
    await session.start()
    try:
        yield session
    finally:
        await session.dispose()
        shutil.rmtree(directory, ignore_errors=True)


def fake_child_run(input_tokens: int, output_tokens: int, answer: str = "done"):
    """Stand in for a real model call, spending a known number of tokens."""

    def run(messages, tools, approve, model=None, raise_on_error=False, tokens=None, depth=0):
        tokens.context = input_tokens + output_tokens
        tokens.billed_input += input_tokens
        tokens.billed_output += output_tokens
        return answer

    return run


def test_child_is_billed_to_parent_without_touching_its_context(monkeypatch):
    """The bug this guards: child tokens must not make the parent look full."""

    async def body():
        async with root() as parent:
            parent.tokens.context = 5000
            parent.tokens.billed_input = 4000
            parent.tokens.billed_output = 1000

            billed_before = parent.billed_tokens
            context_before = parent.context_tokens
            assert context_before == 5000
            assert billed_before == 5000

            monkeypatch.setattr(agent, "run", fake_child_run(700, 300))
            answer = await parent.spawn("do a thing")
            assert answer == "done"

            assert parent.billed_tokens == billed_before + 1000, "child was not billed"
            assert parent.context_tokens == context_before, "child inflated parent context"

    asyncio.run(body())


def test_usage_record_is_persisted_and_recomputes_on_reload(monkeypatch):
    async def body():
        async with root() as parent:
            parent.current_call_id = "call-abc123"
            monkeypatch.setattr(agent, "run", fake_child_run(120, 80))
            await parent.spawn("first task")

            records = parent.usage_records()
            assert len(records) == 1
            record = records[0]
            assert record["type"] == CHILD_USAGE
            assert record["parent_message_id"] == "call-abc123"
            assert record["usage"] == {"input": 120, "output": 80, "total": 200}

            # a fresh Session over the same directory recomputes the same total
            reloaded = Session(parent.session_dir, asyncio.get_event_loop())
            assert reloaded.child_billed == 200

    asyncio.run(body())


def test_registry_lists_children_with_status_and_timing(monkeypatch):
    async def body():
        async with root() as parent:
            monkeypatch.setattr(agent, "run", fake_child_run(10, 10))
            await parent.spawn("review the auth module", name="auth-reviewer")

            listed = parent.list_subagents()
            assert len(listed) == 1
            entry = listed[0]

            for field in (
                "child_id", "name", "session_dir", "model",
                "status", "started_at", "duration_ms",
            ):
                assert field in entry, f"missing {field}"

            assert entry["name"] == "auth-reviewer"
            assert entry["status"] == "completed"
            assert entry["started_at"] > 0
            assert entry["duration_ms"] >= 0
            assert entry["session_dir"].endswith(entry["child_id"])

    asyncio.run(body())


def test_failed_child_is_recorded_as_failed(monkeypatch):
    async def body():
        async with root() as parent:
            def explode(messages, tools, approve, model=None, raise_on_error=False, tokens=None, depth=0):
                raise RuntimeError("provider exploded")

            monkeypatch.setattr(agent, "run", explode)
            answer = await parent.spawn("doomed task")

            assert answer.startswith("CHILD FAILED: ")
            assert parent.list_subagents()[0]["status"] == "failed"

    asyncio.run(body())


def test_list_subagents_visible_from_a_cell(monkeypatch):
    async def body():
        async with root() as parent:
            monkeypatch.setattr(agent, "run", fake_child_run(5, 5))
            await parent.spawn("say OK", name="scout")

            result = await parent.kernel.execute(
                "rows = await rlm.list_subagents()\n"
                "print([(r['name'], r['status']) for r in rows])"
            )
            assert result.error is None, result.error
            assert "('scout', 'completed')" in result.stdout

    asyncio.run(body())


def test_live_child_is_billed_to_the_parent():
    """The same assertion against a real child agent, not a stub."""

    async def body():
        async with root() as parent:
            context_before = parent.context_tokens
            billed_before = parent.billed_tokens

            answer = await parent.spawn("say exactly HELLO and nothing else")
            if answer.startswith("CHILD FAILED") and "RateLimit" in answer:
                pytest.skip("provider quota exhausted; cannot bill a real child")

            assert parent.billed_tokens > billed_before, "real child was not billed"
            assert parent.context_tokens == context_before

    asyncio.run(body())
