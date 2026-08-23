"""End-to-end tests: the real agent, the real kernel, a real model.

These assert behaviour, not wording. They cost real model calls and take
minutes; they are the slowest suite in the project by a wide margin.
"""

import asyncio
import contextlib
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

import agent
import tools as tools_module
from fixture_repo import (
    BIG_FILE_NAME,
    PY_FILE_COUNT,
    SECRET_RETURN,
    TARGET_FUNCTION,
    build,
)
from kernel.session import Session
from tools import IPythonTool


def running_kernels() -> set:
    """PIDs of every ipykernel process on this machine."""
    found = subprocess.run(
        ["pgrep", "-f", "ipykernel_launcher"], capture_output=True, text=True
    )
    return {int(line) for line in found.stdout.split() if line.strip()}


BASELINE_KERNELS = running_kernels()


class RecordingTool(IPythonTool):
    """The real tool, plus a log of every cell and every result."""

    def __init__(self, session, log):
        super().__init__(session)
        self._log = log

    def execute(self, args: dict) -> str:
        result = super().execute(args)
        self._log.append({"session": self.session.name, "code": args["code"], "result": result})
        return result


class Harness:
    """Drives multi-turn conversations against one live session."""

    def __init__(self, session, log):
        self.session = session
        self.log = log
        self.messages = [
            {"role": "system", "content": tools_module.build_system(
                session.depth, session.max_depth)}
        ]

    async def turn(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        mark = len(self.log)

        def drive():
            return agent.run(
                self.messages,
                {"ipython": RecordingTool(self.session, self.log)},
                lambda tool, args: True,
                tokens=self.session.tokens,
            )

        reply = await self.session.loop.run_in_executor(None, drive)
        self.cells_this_turn = self.log[mark:]
        return reply or ""

    def transcript(self) -> str:
        return "\n".join(str(m.get("content") or "") for m in self.messages)


@contextlib.asynccontextmanager
async def live_agent(tmp_path, monkeypatch, max_depth=1):
    """A real session whose kernel and cwd sit inside a fixture repo."""
    repo = build(Path(tmp_path) / "repo")
    monkeypatch.chdir(repo)

    log = []
    # Children build their own tools; make those recorded too.
    real_get_tools = tools_module.get_tools

    def recorded(read_only=False, session=None):
        if session is None:
            return real_get_tools(read_only=read_only)
        return {"ipython": RecordingTool(session, log)}

    monkeypatch.setattr(tools_module, "get_tools", recorded)

    directory = Path.home() / ".myagent" / "sessions" / f"e{uuid.uuid4().hex[:8]}"
    session = Session(directory, asyncio.get_event_loop(), depth=0, max_depth=max_depth)
    session._e2e_log = log
    await session.start()
    try:
        yield Harness(session, log), session, repo
    finally:
        await session.dispose()
        shutil.rmtree(directory, ignore_errors=True)



SCAN_CALLS = ("rglob", "iglob", "glob(", "os.walk", "listdir", "%%bash", "find ")


def test_state_persists_across_turns(tmp_path, monkeypatch):
    """Turn two must reuse the value, not rescan the filesystem."""

    async def body():
        async with live_agent(tmp_path, monkeypatch) as (harness, _, _):
            first = await harness.turn(
                "Count the .py files in this directory tree and store the number "
                "in a variable called py_count. Tell me the number."
            )
            assert str(PY_FILE_COUNT) in first, first

            second = await harness.turn(
                "What was that count? Use the value you already computed; "
                "do not scan the filesystem again."
            )
            assert str(PY_FILE_COUNT) in second, second

            rescans = [
                cell["code"] for cell in harness.cells_this_turn
                if any(call in cell["code"] for call in SCAN_CALLS)
            ]
            assert not rescans, f"turn two rescanned the filesystem: {rescans}"

    asyncio.run(body())


def test_context_economy_on_a_large_file(tmp_path, monkeypatch):
    """A 5000-line file must not end up in the transcript."""

    async def body():
        async with live_agent(tmp_path, monkeypatch) as (harness, _, repo):
            whole_file = (repo / BIG_FILE_NAME).read_text()

            reply = await harness.turn(
                f"In {BIG_FILE_NAME} there is a function called {TARGET_FUNCTION}. "
                "What value does it return? Answer with just the number."
            )
            assert str(SECRET_RETURN) in reply, reply

            transcript = harness.transcript()
            assert whole_file not in transcript, "the entire file was pasted in"

            leaked = len(re.findall(r"MARKER_\d{5}", transcript))
            assert leaked < 300, f"{leaked} filler lines leaked into the transcript"

    asyncio.run(body())


def test_delegation_creates_a_child_and_uses_its_answer(tmp_path, monkeypatch):
    async def body():
        async with live_agent(tmp_path, monkeypatch) as (harness, session, _):
            answers = []
            real_spawn = Session.spawn

            async def spy(self, prompt, name=None, model=None):
                answer = await real_spawn(self, prompt, name=name, model=model)
                answers.append(answer)
                return answer

            monkeypatch.setattr(Session, "spawn", spy)

            reply = await harness.turn(
                "Delegate this to a child agent with rlm(): ask it to report how many "
                "files in this directory end in .py. Print the child's answer verbatim, "
                "then repeat it in your reply."
            )

            children = list(session.session_dir.glob("sub-*"))
            assert children, "no child session directory was created"
            assert answers, "rlm() was never called"
            assert session.list_subagents(), "child was not registered"

            transcript = harness.transcript()
            words = [w for w in re.findall(r"[A-Za-z0-9]{4,}", answers[0])][:5]
            assert any(w in transcript for w in words), (
                f"child answer never reached the parent transcript: {answers[0][:200]}"
            )
            assert str(PY_FILE_COUNT) in reply, reply

    asyncio.run(body())


def test_depth_cap_is_reported_and_child_recovers(tmp_path, monkeypatch):
    """At the cap, rlm() fails clearly and the child does the work itself."""

    async def body():
        async with live_agent(tmp_path, monkeypatch, max_depth=1) as (_, session, _):
            answer = await session.spawn(
                "First run this exact cell: await rlm('count the .py files'). "
                "It is expected to fail. Then, without using rlm again, count the "
                "files ending in .py in the current directory yourself and report "
                "the number as a plain integer.",
                name="capped",
            )

            child = session.children["capped"]
            child_cells = [c for c in _log_of(session) if c["session"] == child.name]
            failures = [c for c in child_cells if "depth limit reached" in c["result"]]
            assert failures, "the child never saw a depth-limit error"
            assert "may not create children" in failures[0]["result"]

            assert not answer.startswith("CHILD FAILED"), answer
            assert str(PY_FILE_COUNT) in answer, (
                f"child gave up instead of finishing the work: {answer[:300]}"
            )

    asyncio.run(body())


def test_error_recovery_after_a_traceback(tmp_path, monkeypatch):
    """A raising cell is data to work with, not a dead end."""

    async def body():
        async with live_agent(tmp_path, monkeypatch) as (harness, _, _):
            reply = await harness.turn(
                "Run this cell first, exactly as written, with no try/except:\n"
                "    data = json.loads(Path('totals.json').read_text())\n"
                "Then, whatever happens, find the right file in this directory, "
                "repair its contents if they are malformed, and tell me the value "
                "of 'total'."
            )

            results = [cell["result"] for cell in harness.cells_this_turn]
            # An uncaught failure, surfaced to the model as a real traceback.
            saw_traceback = any(
                "Traceback" in r and "Error" in r for r in results
            )
            assert saw_traceback, f"no cell raised uncaught: {results}"
            assert len(harness.cells_this_turn) >= 2, "the model never retried"
            assert "1234" in reply, f"gave up instead of recovering: {reply[:300]}"

    asyncio.run(body())


def test_no_orphan_kernels_survive():
    """Runs last: nothing this suite started may still be alive."""
    leaked = running_kernels() - BASELINE_KERNELS
    assert not leaked, f"orphan ipykernel processes: {sorted(leaked)}"


def _log_of(session):
    """Every cell run under this session, children included."""
    return getattr(session, "_e2e_log", [])
