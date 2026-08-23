# Migration Done

`juju` was a tool-menu agent: nine hand-written tools, one function call each. It is now a
Recursive Language Model — a single `ipython` tool over a persistent kernel, where the agent
writes Python instead of picking from a menu, and can spawn full child agents from inside a
cell.

The companion document [`RLM_MIGRATION_NOTES.md`](RLM_MIGRATION_NOTES.md) describes the
*old* architecture and the risks predicted before the work started. This one records what
actually changed.

---

## What the old tools were

Nine registered tools, all in `tools.py`, all replaced:

| Tool | What it did | How you do it now |
|---|---|---|
| `read_file` | return a file's contents | `Path("x.py").read_text()` |
| `write_file` | overwrite a file | `Path("x.py").write_text(...)` |
| `edit_file` | exact-string replace | `text.replace(old, new)` then `write_text` |
| `grep` | regex search across a tree | `re` + `Path.rglob` |
| `bash` | run a shell command | a `%%bash` cell |
| `todo_write` | hold a checklist | an ordinary Python list that persists |
| `task` | run a sub-agent | `await rlm("...", name="...")` |
| `web_fetch` | scrape a URL to markdown | not replaced — see *Lost capabilities* |
| `web_search` | web search | not replaced — see *Lost capabilities* |

The replacement is one tool:

```json
{"name": "ipython", "parameters": {"code": {"type": "string"}}}
```

---

## What changed, phase by phase

**Phase 1 — the kernel.** `kernel/manager.py` runs a persistent IPython kernel through
`jupyter_client`. It lives in its own virtualenv at `~/.myagent/kernel-venv`, built with
`uv` when available and `python -m venv` otherwise, so the kernel never runs in the host
interpreter. A marker file records the Python version, package versions, and a hash of the
`rlm` sources; drift rebuilds the venv. Startup is lazy — a session that never runs code
never pays the ~1-2s and ~150MB. `execute()` is serialized behind an `asyncio.Lock` and has
no default timeout, so a long child agent inside a cell is not killed.

**Phase 2 — one tool.** Every tool was unregistered in favour of `ipython`. Output is shaped
before it reaches the model: stdout, then stderr, then the result repr; over 8000 characters
it keeps the first and last 3000 with a truncation marker; a traceback is always included
and its last 20 lines are never cut; empty output becomes the literal `(no output)`. Ctrl-C
during a cell interrupts the kernel instead of killing the process.

**Phase 3 — the bridge.** The kernel is a separate OS process, so `kernel/bridge.py` serves
a Unix socket at `<session_dir>/host.sock` with newline-delimited JSON. The server runs as
its own task, concurrent with the task waiting on kernel output — without that, a cell that
calls the host would deadlock, since the cell cannot finish until the host answers. The
in-kernel client is a small package, `rlm`, installed into the kernel venv and imported
during a silent bootstrap cell along with `os`, `json`, `re`, and `Path`.

**Phase 4 — recursion.** `kernel/session.py` introduces `Session`: one kernel, one bridge,
one directory, a depth, and a set of children. `await rlm("task", name="...")` blocks and
returns the child's final answer as a string. The child is a complete agent — its own
message list, its own lazily-created kernel, its own socket, the same `agent.run` loop — not
a lightweight helper. Depth is checked before any work is done: `RLM_DEPTH` against
`RLM_MAX_DEPTH`, default 1, so the root may spawn children and those children may not.
A child that fails returns `CHILD FAILED: ...` as data the parent can reason about, rather
than raising into the parent's cell.

**Phase 5 — bookkeeping.** Each session keeps a registry of its direct children (id, name,
directory, model, status, start time, duration), readable from a cell via
`await rlm.list_subagents()`. Token accounting was split in two, which required replacing
the old module-level `context_tokens` global with a per-session `agent.TokenState`:

- **context size** — drives compaction; a child's tokens never touch it
- **billed tokens** — what the session is charged for; a child's tokens always land here

Child usage is appended to `<session_dir>/usage.jsonl` as
`{"type": "child_usage", "parent_message_id": ..., "usage": {...}}`, so a reloaded session
recomputes the same totals. A status line ticks once a second while a child runs, because a
blocking `rlm()` call can take minutes and a silent terminal looks like a hang.

**Phase 6 — the prompt.** The tool section was replaced with rules that teach the actual
runtime: state persists, context lives in variables rather than replies, files are Python,
shell is `%%bash`, do not install project dependencies into the kernel, and when to delegate.
The prompt is now built by `tools.build_system(depth, max_depth)` rather than being a module
constant, because a child must be told its own depth — and told plainly that `rlm()` will
fail when it is at the cap.

**Phase 6.5 — cleanup.** Sockets are removed on every exit path: clean exit, Ctrl-C, and
unhandled exception, via `try/finally` in the REPL and an `atexit` hook. Shutdown is
idempotent. A leftover socket from a crashed session is probed before being unlinked, so a
new session can never silently steal a socket that a live session still owns.

---

## Test coverage

| Suite | Covers |
|---|---|
| `tests/test_kernel.py` | persistence, tracebacks, `%%bash`, interrupt |
| `tests/test_bridge.py` | protocol, unknown types, concurrency, deadlock |
| `tests/test_bridge_shutdown.py` | socket lifecycle, stale vs live sockets, no orphans |
| `tests/test_rlm.py` | recursion, depth cap, kwargs, child kernel cleanup |
| `tests/test_usage.py` | billed vs context separation, registry, persistence |
| `tests/test_rlm_e2e.py` | the real agent against a fixture repo |

The end-to-end suite asserts behaviour rather than wording: that turn two reuses a stored
value instead of rescanning, that a 5000-line file never lands in the transcript, that
delegation really creates a child directory whose answer reaches the parent, that a capped
child reports the depth error and then finishes the work itself, that an uncaught traceback
leads to a retry rather than a report of failure, and that no kernel processes survive.

---

## Dead code, safe to delete

All of it is in `tools.py`, still defined but unregistered since Phase 2 — **236 of that
file's 533 lines**:

| Class | Lines |
|---|---|
| `_FileTool` | 141-148 |
| `ReadFileTool` | 151-166 |
| `WriteFileTool` | 169-188 |
| `EditFileTool` | 191-214 |
| `GrepTool` | 217-244 |
| `BashTool` | 247-266 |
| `TodoWriteTool` | 269-303 |
| `SpawnAgentTool` | 306-336 |
| `WebFetchTool` | 339-361 |
| `WebSearchTool` | 364-394 |

Deleting those classes also makes these dead:

- `import re`, `import subprocess`, `import requests` in `tools.py` — used only by
  `GrepTool`, `BashTool`, and the two web tools
- `FIRECRAWL_KEY` (`tools.py:15`) and the `FIRECRAWL_KEY` entry in `.env`
- the `requests>=2.31` dependency in `pyproject.toml`

Nothing outside `tools.py` imports any of them; `TOOL_REGISTRY` contains only `ipython`.

Not dead, despite appearances: `Tool`, `Tool.to_openai`, `get_tools`, `shape_output`,
`root_session`, and `shutdown_root` are all live.

---

## Known gaps

**Plan mode is empty.** `get_tools(read_only=True)` returns `{}`, because `ipython` is not
read-only. Plan mode used to work by removing dangerous tools from the schema list, which a
single-tool design cannot do. `/plan` currently sends the model zero tools.

**Lost capabilities.** `web_fetch` and `web_search` have no replacement. The kernel can reach
the network with `urllib`, but the Firecrawl-backed scrape and search are gone. They would
return naturally as host-bridge handlers.

**Children keep their kernels.** A finished child stays in the parent's registry with its
kernel alive until the parent disposes, so a session that spawns many children accumulates
~150MB each. The registry makes this visible; nothing reaps it yet.

**Approval does not propagate.** Approving a cell that calls `rlm()` implicitly approves
everything the child does — children run with auto-approval and full `ipython` access,
including writing files into the working directory.

**The e2e suite is expensive.** Each delegation test runs two real agents. A full pass
exceeds a free-tier daily quota of 50 model calls.
