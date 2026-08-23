import os
import re
import subprocess
import platform
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# API keys should be provided via environment variables for security
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_KEY", "")

MODEL = os.environ.get("DEV_MODEL", "poolside/laguna-s-2.1:free")

IDENTITY = "You are juju, a terminal coding agent. Be concise. Prefer tools over guessing."

TOOL_RULES = """You have one tool: `ipython`. It runs Python in a kernel that stays alive for the whole
session. Every capability you have starts there.

STATE PERSISTS. Variables, imports, and functions you define stay available on every later
turn. Use this. Assign the result of every file read, search, or command to a named variable
so you can slice and filter it later without redoing the work.

CONTEXT LIVES IN VARIABLES, NOT IN YOUR REPLIES. Never print a large result just to look at
it. Store it, then print only the part you need:

    matches = [p for p in Path(".").rglob("*.py") if "auth" in p.read_text()]
    print(len(matches), matches[:5])

FILES: read, search, and edit with Python. Path.read_text(), Path.write_text(), the `re`
module. There is no separate file tool.

SHELL: use a `%%bash` cell. It must be the very first line of the cell — no comments, no
blank lines, no imports above it. Each `%%bash` cell is a throwaway subshell, so `cd`,
`export`, and `source` do NOT carry to the next cell. Keep dependent shell steps in one cell.
For a persistent working directory use `%cd`; for persistent environment variables use
os.environ.

DO NOT install project dependencies into the kernel. The kernel is your control room, not
the project's runtime. To run a project's tests or scripts, use the project's own environment
and its documented commands (`uv run ...`, `.venv/bin/python ...`, `npm test`). A failure
from the project's real environment is the real answer.

DELEGATION: `await rlm("task description", name="short-name")` starts a full child agent with
its own fresh context window. It returns the child's final answer as a string.

Delegate when a subtask is self-contained and would otherwise flood your context — reading a
large unfamiliar area of the codebase, an independent review, a long investigation whose
details you do not need. Give the child everything it needs in the prompt; it cannot see your
conversation. Ask it for a specific, compact answer.

Do not delegate work you can finish in one or two cells. A child costs a model call and start-up
time, and its answer is a summary, not the full detail."""

CHILD_NOTE = (
    "You are a child agent. Your task prompt is labeled [task from parent]. "
    "Answer concisely and directly — your reply is consumed by another agent, not by a human."
)


def build_system(depth: Optional[int] = None, max_depth: Optional[int] = None) -> str:
    """The system prompt. Depth-aware, so a child is told what it may not do."""
    if depth is None:
        depth = int(os.environ.get("RLM_DEPTH", 0))
    if max_depth is None:
        max_depth = int(os.environ.get("RLM_MAX_DEPTH", 1))

    parts = [
        IDENTITY,
        "",
        TOOL_RULES,
        "",
        "Environment:",
        f"os: {platform.platform()}",
        f"files: {', '.join(sorted(os.listdir('.')))}",
    ]

    if os.path.exists("JUJU.md"):
        with open("JUJU.md") as handle:
            parts += ["", "Project instructions:", handle.read().rstrip()]

    if depth < max_depth:
        availability = (
            "rlm() is available at this depth: you may delegate to child agents."
        )
    else:
        availability = (
            "rlm() is NOT available at this depth. Any call to rlm() will fail, "
            "so do this work directly yourself."
        )

    parts += [
        "",
        f"cwd: {os.getcwd()}",
        f"RLM_DEPTH: {depth}",
        f"RLM_MAX_DEPTH: {max_depth}",
        availability,
    ]

    if depth > 0:
        parts += ["", CHILD_NOTE]

    return "\n".join(parts) + "\n"


SYSTEM = build_system()


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    is_read_only: bool = True

    # Set by the agent loop before each call, so a tool can attribute work
    # back to the assistant turn that asked for it.
    current_call_id: Optional[str] = None

    @abstractmethod
    def execute(self, args: dict) -> str:
        """Execute the tool with the given arguments."""

    def to_openai(self) -> dict:
        """Convert tool to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class _FileTool(Tool):
    """Base class for file-based tools."""

    is_read_only = True

    def _read(self, path: str) -> str:
        with open(path, errors="ignore") as f:
            return f.read()


class ReadFileTool(_FileTool):
    name = "read_file"
    description = "Read a file from disk and return its contents."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        try:
            return self._read(args["path"])
        except Exception as e:
            return f"Error: {e}"


class WriteFileTool(_FileTool):
    name = "write_file"
    description = "Write content to a file on disk, overwriting it if it exists."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Content to write."},
        },
        "required": ["path", "content"],
    }
    is_read_only = False

    def execute(self, args: dict) -> str:
        try:
            with open(args["path"], "w") as f:
                f.write(args["content"])
            return f"Wrote {len(args['content'])} chars to {args['path']}"
        except Exception as e:
            return f"Error: {e}"


class EditFileTool(_FileTool):
    name = "edit_file"
    description = "Replace an exact string in a file with a new string."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old": {"type": "string", "description": "Exact string to replace."},
            "new": {"type": "string", "description": "String to replace it with."},
        },
        "required": ["path", "old", "new"],
    }
    is_read_only = False

    def execute(self, args: dict) -> str:
        try:
            content = self._read(args["path"])
            if args["old"] not in content:
                return f"Error: string not found in {args['path']}"
            with open(args["path"], "w") as f:
                f.write(content.replace(args["old"], args["new"]))
            return f"Edited {args['path']}"
        except Exception as e:
            return f"Error: {e}"


class GrepTool(Tool):
    name = "grep"
    description = "Search files under a directory for a regex. Returns path:line:text."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {"type": "string", "description": "Directory to search."},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    def execute(self, args: dict) -> str:
        hits: list[str] = []
        search_path = args.get("path", ".")
        pattern = re.compile(args["pattern"], re.IGNORECASE)
        for root, _, files in os.walk(search_path):
            for name in files:
                p = os.path.join(root, name)
                try:
                    with open(p, errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if pattern.search(line):
                                hits.append(f"{p}:{i}:{line.rstrip()}")
                except OSError:
                    continue
        return "\n".join(hits)


class BashTool(Tool):
    name = "bash"
    description = "Run a shell command and return its stdout and stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["command"],
    }
    is_read_only = False

    def execute(self, args: dict) -> str:
        try:
            result = subprocess.run(
                args["command"], shell=True, capture_output=True, text=True
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Error: {e}"


class TodoWriteTool(Tool):
    name = "todo_write"
    description = "Record the task plan as a todo list, and update it as work progresses."
    parameters = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "The full todo list, in order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The task."},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done"],
                        },
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["items"],
    }
    is_read_only = True

    def __init__(self):
        self.todos: list[dict] = []

    def execute(self, args: dict) -> str:
        self.todos = args["items"]
        marks = {"pending": " ", "in_progress": "~", "done": "x"}
        return "\n".join(
            f"[{marks[t['status']]}] {t['content']}" for t in self.todos
        )


class SpawnAgentTool(Tool):
    name = "task"
    description = "Run a sub-agent on a self-contained task and return its final answer."
    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short task label."},
            "prompt": {"type": "string", "description": "Full instructions for the sub-agent."},
        },
        "required": ["description", "prompt"],
    }
    is_read_only = False

    def __init__(self):
        self._tools: dict[str, Tool] | None = None

    def _get_tools(self) -> dict[str, Tool]:
        if self._tools is None:
            # Exclude task tool to prevent recursion
            self._tools = {n: t for n, t in TOOL_REGISTRY.items() if n != "task"}
        return self._tools

    def execute(self, args: dict) -> str:
        import agent

        messages = [
            {"role": "system", "content": agent.SYSTEM},
            {"role": "user", "content": args["prompt"]},
        ]
        tools = self._get_tools()
        return agent.run(messages, tools, lambda tool, args: True)


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a web page and return its readable text."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL of the page to fetch."},
        },
        "required": ["url"],
    }
    is_read_only = True

    def execute(self, args: dict) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v2/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_KEY}"},
                json={"url": args["url"], "formats": ["markdown"]},
                timeout=30,
            )
            return response.json()["data"]["markdown"][:4000]
        except Exception as e:
            return f"Error: {e}"


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web and return matching titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "limit": {"type": "integer", "description": "How many results to return."},
        },
        "required": ["query"],
    }
    is_read_only = True

    def execute(self, args: dict) -> str:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v2/search",
                headers={"Authorization": f"Bearer {FIRECRAWL_KEY}"},
                json={
                    "query": args["query"],
                    "limit": args.get("limit", 5),
                    "sources": ["web"],
                },
                timeout=30,
            )
            results = response.json()["data"]["web"]
            return "\n\n".join(
                f"{r['title']}\n{r['url']}\n{r['description'][:300]}" for r in results
            )
        except Exception as e:
            return f"Error: {e}"




# --------------------------------------------------------------------------
# The single tool. Everything above is kept for reference but unregistered.
# --------------------------------------------------------------------------

MAX_OUTPUT = 8000
HEAD_CHARS = 3000
TAIL_CHARS = 3000
PROTECTED_ERROR_LINES = 20


_root: Optional["object"] = None


def root_session():
    """The top-level session, created on first use so idle sessions cost nothing."""
    global _root
    if _root is None:
        import asyncio
        import atexit
        import threading
        import uuid

        from kernel.session import Session

        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()

        session_dir = Path.home() / ".myagent" / "sessions" / f"s{uuid.uuid4().hex[:8]}"

        # Built on the loop: on Python 3.9 asyncio.Lock() binds to whatever loop
        # is current when it is constructed. The bridge lives here too, so it can
        # answer a cell that is still running.
        async def build():
            session = Session.root(session_dir, loop)
            await session.start()
            return session

        _root = asyncio.run_coroutine_threadsafe(build(), loop).result()
        atexit.register(shutdown_root)
    return _root


def shutdown_root() -> None:
    """Tear the root session down from any exit path: clean, Ctrl-C, or crash."""
    global _root
    if _root is None:
        return

    import asyncio

    session, _root = _root, None
    try:
        asyncio.run_coroutine_threadsafe(session.dispose(), session.loop).result(
            timeout=15
        )
    except Exception:
        # The loop may be wedged or already gone; at least leave no sockets.
        session.remove_sockets()


def shape_output(result) -> str:
    """Render an ExecResult for the model, under a character budget."""
    text = result.stdout + result.stderr
    if result.result is not None:
        text += result.result

    if result.error:
        if text and not text.endswith("\n"):
            text += "\n"
        text += result.error

    if not text.strip():
        return "(no output)"

    if len(text) <= MAX_OUTPUT:
        return text

    # The tail of a failure is the part that explains it, so never cut into
    # the last lines of a traceback.
    tail_chars = TAIL_CHARS
    if result.error:
        protected = "\n".join(result.error.splitlines()[-PROTECTED_ERROR_LINES:])
        tail_chars = max(tail_chars, len(protected))

    head, tail = text[:HEAD_CHARS], text[-tail_chars:]
    dropped = len(text) - len(head) - len(tail)
    if dropped <= 0:
        return text
    return f"{head}\n... [{dropped} characters truncated] ...\n{tail}"


class IPythonTool(Tool):
    """The one tool. Each session gets its own, bound to its own kernel."""

    name = "ipython"
    description = (
        "Execute Python code in a persistent IPython kernel. State persists across calls. "
        "Use %%bash as the first line of the cell to run shell commands."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute."},
        },
        "required": ["code"],
    }
    is_read_only = False

    def __init__(self, session=None):
        self._session = session

    @property
    def session(self):
        return self._session or root_session()

    def execute(self, args: dict) -> str:
        session = self.session
        session.current_call_id = self.current_call_id
        return shape_output(session.execute_code(args["code"]))


# Centralized tool registry — single source of truth.
# Phase 2: every previous tool is replaced by the single `ipython` tool. The
# classes above stay defined but unregistered, so the model never sees them.
ALL_TOOLS = [IPythonTool()]

TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def get_tools(read_only: bool = False, session=None) -> dict[str, Tool]:
    """Tools for one agent. Each session gets its own instances, never shared."""
    tools = TOOL_REGISTRY if session is None else {"ipython": IPythonTool(session)}
    if read_only:
        return {n: t for n, t in tools.items() if t.is_read_only}
    return dict(tools)
