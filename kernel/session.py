"""A session: one kernel, one host bridge, and the children it spawned.

Each session owns everything an agent needs to run — its own kernel process,
its own Unix socket, its own directory — so a child is a complete agent rather
than a helper borrowing the parent's machinery.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, Set

from kernel.bridge import HostBridge
from kernel.manager import KernelManager

DEPTH_ENV = "RLM_DEPTH"
MAX_DEPTH_ENV = "RLM_MAX_DEPTH"
DEFAULT_MAX_DEPTH = 1

TASK_PREFIX = "[task from parent] "
FAILURE_PREFIX = "CHILD FAILED: "

USAGE_FILE = "usage.jsonl"
CHILD_USAGE = "child_usage"


def slugify(prompt: str, words: int = 4) -> str:
    """A short readable handle taken from the first few words of a prompt."""
    found = re.findall(r"[a-z0-9]+", prompt.lower())[:words]
    return "-".join(found) or "task"


class Session:
    """One agent's worth of state."""

    def __init__(
        self,
        session_dir: Path,
        loop: asyncio.AbstractEventLoop,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
        model: Optional[str] = None,
        name: str = "root",
        tokens=None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.loop = loop
        self.depth = depth
        self.max_depth = max_depth
        self.model = model
        self.name = name

        self.children: Dict[str, "Session"] = {}
        self.child_records: Dict[str, dict] = {}
        self._running: Set[asyncio.Future] = set()

        # Which assistant turn is currently driving this session, so child cost
        # can be attributed back to it.
        self.current_call_id: Optional[str] = None
        self._disposed = False

        import agent

        self.tokens = tokens if tokens is not None else agent.TokenState()

        self.bridge = HostBridge(self.session_dir)
        self.kernel = KernelManager(
            host_socket=self.bridge.socket_path,
            env_extra={
                DEPTH_ENV: str(depth),
                MAX_DEPTH_ENV: str(max_depth),
            },
        )

    @classmethod
    def root(cls, session_dir: Path, loop, model: Optional[str] = None) -> "Session":
        """The top-level session, taking its limits from the environment."""
        import agent

        return cls(
            session_dir,
            loop,
            depth=int(os.environ.get(DEPTH_ENV, 0)),
            max_depth=int(os.environ.get(MAX_DEPTH_ENV, DEFAULT_MAX_DEPTH)),
            model=model,
            tokens=agent.DEFAULT_TOKENS,
        )

    async def start(self) -> None:
        await self.bridge.start()
        self.bridge.register("rlm.run", self._handle_run)
        self.bridge.register("rlm.list_subagents", self._handle_list)

    # -- calling into the kernel -----------------------------------------

    def execute_code(self, code: str):
        """Run a cell from synchronous tool code, honouring Ctrl-C."""
        pending = asyncio.run_coroutine_threadsafe(self.kernel.execute(code), self.loop)
        while True:
            try:
                return pending.result()
            except KeyboardInterrupt:
                asyncio.run_coroutine_threadsafe(
                    self.kernel.interrupt(), self.loop
                ).result()

    async def kernel_pid(self) -> Optional[int]:
        """PID of this session's kernel process, if one is running."""
        manager = self.kernel._km
        if manager is None or manager.provisioner is None:
            return None
        return manager.provisioner.pid

    # -- children ---------------------------------------------------------

    async def _handle_run(self, payload: dict) -> dict:
        """Bridge handler for rlm.run."""
        # Depth is checked before any work is done, so a refused call costs
        # nothing — no directory, no kernel, no model call.
        if self.depth >= self.max_depth:
            raise RuntimeError(
                "rlm depth limit reached: at depth {} with max depth {}; "
                "this agent may not create children".format(self.depth, self.max_depth)
            )

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError("rlm.run requires a non-empty 'prompt'")

        answer = await self.spawn(
            prompt, name=payload.get("name"), model=payload.get("model")
        )
        return {"answer": answer}

    def _unique_name(self, wanted: str) -> str:
        """Names must not collide among siblings."""
        if wanted not in self.children:
            return wanted
        n = 2
        while f"{wanted}-{n}" in self.children:
            n += 1
        return f"{wanted}-{n}"

    async def _handle_list(self, payload: dict) -> dict:
        """Bridge handler for rlm.list_subagents."""
        return {"subagents": self.list_subagents()}

    def list_subagents(self) -> list:
        """This session's direct children, newest last."""
        return [dict(record) for record in self.child_records.values()]

    # -- cost ------------------------------------------------------------

    @property
    def context_tokens(self) -> int:
        """How full this agent's own context window is. Children never add here."""
        return self.tokens.context

    @property
    def billed_tokens(self) -> int:
        """Everything this session is charged for, its children included."""
        return self.tokens.billed_total + self.child_billed

    @property
    def child_billed(self) -> int:
        return sum(
            record["usage"]["total"] for record in self.usage_records()
        )

    def usage_records(self) -> list:
        """Usage records on disk, so a reloaded session recomputes the same totals."""
        path = self.session_dir / USAGE_FILE
        if not path.exists():
            return []
        records = []
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def record_child_usage(self, child_id: str, usage: dict, call_id=None) -> dict:
        """Append a child's cost to this session's billing log."""
        record = {
            "type": CHILD_USAGE,
            "parent_message_id": call_id if call_id is not None else self.current_call_id,
            "child_id": child_id,
            "usage": usage,
        }
        self.session_dir.mkdir(parents=True, exist_ok=True)
        with open(self.session_dir / USAGE_FILE, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    # -- terminal feedback ------------------------------------------------

    async def _show_progress(self, record: dict) -> None:
        """Keep the terminal alive while a child runs; it can take minutes."""
        try:
            while True:
                elapsed = time.time() - record["started_at"]
                sys.stdout.write(
                    "\r[{} | {} | {:.0f}s | {}]  ".format(
                        record["name"], record["model"], elapsed, record["status"]
                    )
                )
                sys.stdout.flush()
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    def _finish_line(self, record: dict) -> None:
        sys.stdout.write(
            "\r[{} | {} | {:.1f}s | {}]  \n".format(
                record["name"],
                record["model"],
                record["duration_ms"] / 1000,
                record["status"],
            )
        )
        sys.stdout.flush()

    async def spawn(
        self, prompt: str, name: Optional[str] = None, model: Optional[str] = None
    ) -> str:
        """Run a full child agent to completion and return its final answer."""
        import agent

        child_id = uuid.uuid4().hex[:8]
        child = Session(
            self.session_dir / f"sub-{child_id}",
            self.loop,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            model=model or self.model,
            name=self._unique_name(name or f"{slugify(prompt)}-{child_id}"),
        )
        self.children[child.name] = child
        await child.start()

        record = {
            "child_id": child_id,
            "name": child.name,
            "session_dir": str(child.session_dir),
            "model": child.model or agent.MODEL,
            "status": "running",
            "started_at": time.time(),
            "duration_ms": 0.0,
        }
        self.child_records[child.name] = record

        # Attribute the child's cost to the turn that asked for it, captured
        # now because the parent moves on to other turns later.
        call_id = self.current_call_id

        started = time.monotonic()
        progress = asyncio.ensure_future(self._show_progress(record))
        running = asyncio.ensure_future(child.run_agent(prompt))
        self._running.add(running)

        try:
            answer = await running
            record["status"] = (
                "failed" if answer.startswith(FAILURE_PREFIX) else "completed"
            )
            return answer
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        finally:
            self._running.discard(running)
            progress.cancel()
            record["duration_ms"] = (time.monotonic() - started) * 1000

            # A child's tokens are billed to the parent but must never be
            # added to the parent's context size.
            usage = {
                "input": child.tokens.billed_input,
                "output": child.tokens.billed_output,
                "total": child.tokens.billed_total,
            }
            record["usage"] = usage
            self.record_child_usage(child_id, usage, call_id)
            self._finish_line(record)

    async def run_agent(self, prompt: str) -> str:
        """Drive this session's own agent loop over one prompt."""
        import agent
        from tools import build_system, get_tools

        messages = [
            {"role": "system", "content": build_system(self.depth, self.max_depth)},
            {"role": "user", "content": TASK_PREFIX + prompt},
        ]
        tools = get_tools(session=self)

        def drive():
            return agent.run(
                messages,
                tools,
                lambda tool, args: True,
                self.model,
                raise_on_error=True,
                tokens=self.tokens,
            )

        try:
            answer = await self.loop.run_in_executor(None, drive)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return "{}{}: {}".format(FAILURE_PREFIX, type(exc).__name__, exc)

        if not answer.strip():
            return FAILURE_PREFIX + "child produced no answer"
        return answer

    # -- teardown ---------------------------------------------------------

    async def dispose(self) -> None:
        """Tear down this session and every child beneath it. Safe to call twice."""
        if self._disposed:
            return
        self._disposed = True

        for running in list(self._running):
            running.cancel()
        self._running.clear()

        for child in list(self.children.values()):
            await child.dispose()
        self.children.clear()

        await self.kernel.shutdown()
        await self.bridge.stop()
        shutil.rmtree(self.session_dir, ignore_errors=True)

    def remove_sockets(self) -> None:
        """Unlink this session's socket and every child's, without the loop.

        The last-ditch path: if the event loop is wedged at exit, this still
        leaves no socket files behind.
        """
        for child in self.children.values():
            child.remove_sockets()
        self.bridge.remove_socket()
