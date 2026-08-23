# RLM Migration Notes

A factual map of how the `juju` agent works as of this commit. Descriptive only — no proposed design.

Codebase is three modules plus packaging:

| File | Lines | Role |
|---|---|---|
| `juju.py` | 47 | CLI entry point and REPL |
| `agent.py` | 159 | The agent loop, compaction, model client |
| `tools.py` | 333 | Tool base class, all nine tools, the registry, the system prompt |

Entry point is declared in `pyproject.toml` as `juju = "juju:main"`.

---

## 1. The agent loop

`agent.run(messages, tools, approve)` — [`agent.py:89-159`](agent.py#L89). It is a `while True` loop that keeps calling the model until the model stops asking for tools.

| Step | Location | What happens |
|---|---|---|
| Auto-compact check | `agent.py:95-96` | `if context_tokens > CONTEXT_WINDOW - RESERVE_TOKENS: compact(messages)` — runs *before* the model call |
| Model call | `agent.py:99-105` | `client.chat.completions.create(model, messages, tools=[...], stream=True, stream_options={"include_usage": True})` |
| Stream consumption | `agent.py:114-139` | Iterates chunks; text printed live, tool calls accumulated |
| Exit condition | `agent.py:141-144` | If `finish_reason != "tool_calls"`, append `{"role": "assistant", "content": reply}` and **return the reply string** |
| Assistant + tool_calls appended | `agent.py:146-148` | `{"role": "assistant", "content": reply, "tool_calls": tool_calls}` |
| Tool dispatch | `agent.py:149-159` | For each call: look up tool, `json.loads` the arguments, print the call, gate it, execute, append result |

**Tool-call reassembly.** Because the response is streamed, one tool call arrives split across many chunks — the `id` in one, the function name in another, and `arguments` as string fragments. `agent.py:127-139` merges deltas into a slot indexed by `tc.index`, concatenating `arguments`. Only after the stream ends is `arguments` a complete JSON string, which is why `json.loads` happens at `agent.py:151` and not during streaming.

**Result append.** `agent.py:157-159` appends `{"role": "tool", "tool_call_id": call["id"], "content": result}`. The loop then repeats and the model sees the results.

**Dispatch is a bare dict lookup.** `tool = tools[call["function"]["name"]]` at `agent.py:150` has no membership check. A hallucinated tool name raises `KeyError` and terminates the process. This has been observed in practice (a model emitted a call to a nonexistent `shell` tool while in plan mode).

**Execution is strictly sequential.** `agent.py:149` is a plain `for` loop; each `tool.execute()` blocks. There are no threads, no asyncio, no executors anywhere in the codebase (verified by grep for `thread|asyncio|concurrent|await|async|Pool|Executor` — zero hits outside `.venv`).

**Approval gate.** `agent.py:153` calls the injected `approve(tool, args)` callback. The REPL passes `ask` ([`juju.py:7-9`](juju.py#L7)): `return tool.is_read_only or input("[y/n] ") == "y"`. Short-circuit `or` means read-only tools never prompt. On denial the literal string `"user denied"` is appended as the tool result (`agent.py:156`).

---

## 2. Tools

All nine live in `tools.py`. All subclass `Tool` ([`tools.py:32-53`](tools.py#L32)), an `ABC` with abstract `execute`. `Tool.to_openai()` (`tools.py:44-53`) renders the OpenAI function shape.

`_FileTool` ([`tools.py:56-63`](tools.py#L56)) is an intermediate base holding a shared `_read` helper; `ReadFileTool`, `WriteFileTool`, and `EditFileTool` inherit it (`WriteFileTool` never calls `_read`).

Every `execute` except `TodoWriteTool`'s is wrapped in `try/except Exception` returning `f"Error: {e}"`. Note `except Exception` does **not** catch `KeyboardInterrupt` (see section 8).

### `read_file`

- **Class:** `ReadFileTool` — [`tools.py:66-81`](tools.py#L66)
- **`is_read_only`:** `True`
- **Description sent to model:** "Read a file from disk and return its contents."
- **What it does:** Opens the path and returns the whole file as a string. Uses `_FileTool._read`, which passes `errors="ignore"`, so undecodable bytes are silently dropped rather than raising.

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to the file to read."
    }
  },
  "required": [
    "path"
  ]
}
```

### `write_file`

- **Class:** `WriteFileTool` — [`tools.py:84-103`](tools.py#L84)
- **`is_read_only`:** `False`  → prompts for approval
- **Description sent to model:** "Write content to a file on disk, overwriting it if it exists."
- **What it does:** Opens the path in `"w"` mode and writes `content`, truncating any existing file. Returns a confirmation string (`Wrote N chars to PATH`), not the content.

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to the file to write."
    },
    "content": {
      "type": "string",
      "description": "Content to write."
    }
  },
  "required": [
    "path",
    "content"
  ]
}
```

### `edit_file`

- **Class:** `EditFileTool` — [`tools.py:106-129`](tools.py#L106)
- **`is_read_only`:** `False`  → prompts for approval
- **Description sent to model:** "Replace an exact string in a file with a new string."
- **What it does:** Reads the file, and if `old` is not present returns `Error: string not found in PATH` **without writing**. Otherwise writes back `content.replace(old, new)` — replacing *every* occurrence, not just the first.

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to the file to edit."
    },
    "old": {
      "type": "string",
      "description": "Exact string to replace."
    },
    "new": {
      "type": "string",
      "description": "String to replace it with."
    }
  },
  "required": [
    "path",
    "old",
    "new"
  ]
}
```

### `grep`

- **Class:** `GrepTool` — [`tools.py:132-159`](tools.py#L132)
- **`is_read_only`:** `True`
- **Description sent to model:** "Search files under a directory for a regex. Returns path:line:text."
- **What it does:** Walks `path` (default `"."`) with `os.walk`, opening every file it finds and testing each line against the compiled pattern. Returns `path:line:text` per hit, newline-joined. The pattern is compiled with `re.IGNORECASE`, so case-sensitive search is not reachable. `OSError` on an unreadable file is swallowed and the file skipped. There is no ignore list, so `.git/` and `.venv/` are traversed.

```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Regex to search for."
    },
    "path": {
      "type": "string",
      "description": "Directory to search."
    }
  },
  "required": [
    "pattern"
  ]
}
```

### `bash`

- **Class:** `BashTool` — [`tools.py:162-181`](tools.py#L162)
- **`is_read_only`:** `False`  → prompts for approval
- **Description sent to model:** "Run a shell command and return its stdout and stderr."
- **What it does:** Runs the command through `subprocess.run(..., shell=True, capture_output=True, text=True)` and returns `stdout + stderr` concatenated. No timeout is set, so a hanging command hangs the agent indefinitely.

```json
{
  "type": "object",
  "properties": {
    "command": {
      "type": "string",
      "description": "Shell command to run."
    }
  },
  "required": [
    "command"
  ]
}
```

### `todo_write`

- **Class:** `TodoWriteTool` — [`tools.py:184-218`](tools.py#L184)
- **`is_read_only`:** `True`
- **Description sent to model:** "Record the task plan as a todo list, and update it as work progresses."
- **What it does:** Overwrites `self.todos` with the supplied list and returns it rendered as `[x] / [~] / [ ]` lines. The model resends the entire list every call; there is no server-side merge. State lives on the tool instance.

```json
{
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "description": "The full todo list, in order.",
      "items": {
        "type": "object",
        "properties": {
          "content": {
            "type": "string",
            "description": "The task."
          },
          "status": {
            "type": "string",
            "enum": [
              "pending",
              "in_progress",
              "done"
            ]
          }
        },
        "required": [
          "content",
          "status"
        ]
      }
    }
  },
  "required": [
    "items"
  ]
}
```

### `task`

- **Class:** `SpawnAgentTool` — [`tools.py:221-251`](tools.py#L221)
- **`is_read_only`:** `False`  → prompts for approval
- **Description sent to model:** "Run a sub-agent on a self-contained task and return its final answer."
- **What it does:** Spawns a sub-agent. See section 6.

```json
{
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "Short task label."
    },
    "prompt": {
      "type": "string",
      "description": "Full instructions for the sub-agent."
    }
  },
  "required": [
    "description",
    "prompt"
  ]
}
```

### `web_fetch`

- **Class:** `WebFetchTool` — [`tools.py:254-276`](tools.py#L254)
- **`is_read_only`:** `True`
- **Description sent to model:** "Fetch a web page and return its readable text."
- **What it does:** POSTs the URL to `https://api.firecrawl.dev/v2/scrape` with `formats: ["markdown"]`, and returns `data.markdown` truncated to the first 4000 characters. HTML stripping is done by Firecrawl, not locally.

```json
{
  "type": "object",
  "properties": {
    "url": {
      "type": "string",
      "description": "URL of the page to fetch."
    }
  },
  "required": [
    "url"
  ]
}
```

### `web_search`

- **Class:** `WebSearchTool` — [`tools.py:279-309`](tools.py#L279)
- **`is_read_only`:** `True`
- **Description sent to model:** "Search the web and return matching titles, URLs, and snippets."
- **What it does:** POSTs to `https://api.firecrawl.dev/v2/search` with `query`, `limit` (default 5), `sources: ["web"]`. Returns `data.web` rendered as title / URL / snippet (snippet capped at 300 chars), joined by blank lines.

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The search query."
    },
    "limit": {
      "type": "integer",
      "description": "How many results to return."
    }
  },
  "required": [
    "query"
  ]
}
```

### Registry

- `ALL_TOOLS` — [`tools.py:313-323`](tools.py#L313) — a list of nine **instantiated singletons**.
- `TOOL_REGISTRY` — [`tools.py:325`](tools.py#L325) — `{t.name: t for t in ALL_TOOLS}`.
- `get_tools(read_only=False)` — [`tools.py:328-333`](tools.py#L328) — returns `TOOL_REGISTRY.copy()`, or filtered to `is_read_only` tools when `read_only=True`.

The filtered form is what implements plan mode: `juju.py:42` calls `get_tools(read_only=plan)`. Tools are not *refused* in plan mode — they are absent from the schema list sent to the model. In plan mode the model sees five tools (`read_file`, `grep`, `todo_write`, `web_fetch`, `web_search`) and cannot see `write_file`, `edit_file`, `bash`, or `task`.

Because the registry holds singletons, **every agent — parent and any sub-agent — shares the same tool objects.** Verified: `TOOL_REGISTRY['todo_write'] is subagent_tools['todo_write']` → `True`.

---

## 3. Message / transcript structure

**The object is a plain Python `list` of `dict`.** There is no Message class, no dataclass, no wrapper. It is created at [`juju.py:26`](juju.py#L26):

```python
messages = [{"role": "system", "content": SYSTEM}]
```

It is passed by reference into `agent.run(messages, tools, ask)` (`juju.py:43`) and **mutated in place** — `run` appends to the caller's list, and `compact` reassigns its contents via slice assignment (`messages[:] = ...` at `agent.py:74`).

Four message shapes occur:

| Role | Shape | Written at |
|---|---|---|
| `system` | `{"role", "content"}` | `juju.py:26`, and `juju.py:37` for plan-mode notices |
| `user` | `{"role", "content"}` | `juju.py:41` |
| `assistant` | `{"role", "content"}` or `{"role", "content", "tool_calls"}` | `agent.py:143`, `agent.py:146-148` |
| `tool` | `{"role", "tool_call_id", "content"}` | `agent.py:157-159` |

`tool_calls` entries are dicts of the form `{"id", "type": "function", "function": {"name", "arguments"}}`, built at `agent.py:129-132`.

**Persistence: none.** The transcript exists only in process memory. Grep for `json.dump|pickle|sqlite|.jsonl` returns no hits outside `.venv`. There is no save, no load, no resume. Exiting the process — including via Ctrl-C — discards the conversation permanently.

---

## 4. System prompt

Built at **import time** in [`tools.py:18-29`](tools.py#L18), as a module-level f-string. It is interpolated once when `tools` is first imported, so `cwd`, `os`, and the file listing are frozen at process start.

`agent.py:6` does `from tools import MODEL, SYSTEM`, which is why `agent.SYSTEM` also resolves (used by `SpawnAgentTool.execute` at `tools.py:247`).

An optional `JUJU.md` in the current working directory is appended under a `Project instructions` heading (`tools.py:27-29`).

Full text as built in this repo (no `JUJU.md` present):

```text
You are juju, a terminal coding agent. Be concise. Prefer tools over guessing.
Use todo_write to plan any task that takes more than a couple of steps, and update it as you go.

Environment:
cwd: /path/to/project
os: macOS-26.5.1-arm64-arm-64bit
files: .env, .git, .gitignore, .venv, README.md, RLM_MIGRATION_NOTES.md, agent.py, juju.egg-info, juju.py, pyproject.toml, tools.py
```

Template source, before interpolation:

```python
SYSTEM = f"""You are juju, a terminal coding agent. Be concise. Prefer tools over guessing.
Use todo_write to plan any task that takes more than a couple of steps, and update it as you go.

Environment:
cwd: {os.getcwd()}
os: {platform.platform()}
files: {', '.join(sorted(os.listdir('.')))}
"""

if os.path.exists("JUJU.md"):
    with open("JUJU.md") as f:
        SYSTEM += "\nProject instructions:\n" + f.read()
```

Two further system messages are appended at runtime by the REPL when plan mode toggles — `PLAN_ON` and `PLAN_OFF`, [`agent.py:82-86`](agent.py#L82), inserted at `juju.py:37`.

---

## 5. Token accounting

Tokens are counted; **cost is not tracked anywhere**.

- **Storage:** a single module-level global, `context_tokens = 0` — [`agent.py:34`](agent.py#L34). Declared `global` in both `run` (`agent.py:92`) and `compact` (`agent.py:58`).
- **Source:** exact counts from the provider, not an estimate. `stream_options={"include_usage": True}` (`agent.py:104`) makes OpenRouter emit a final usage chunk; `agent.py:115-116` reads `chunk.usage.total_tokens` into `context_tokens`.
- **Guard:** that usage chunk carries an empty `choices` array, so `agent.py:117-118` skips it (`if not chunk.choices: continue`) before indexing `chunk.choices[0]`.
- **Thresholds:** `CONTEXT_WINDOW` (default `262144`, env `DEV_CONTEXT_WINDOW`) and `RESERVE_TOKENS` (default `20000`, env `DEV_RESERVE_TOKENS`) — [`agent.py:14-15`](agent.py#L14).
- **Trigger:** `agent.py:95` — `context_tokens > CONTEXT_WINDOW - RESERVE_TOKENS`, evaluated before every model call. With defaults, that is 242,144 tokens.

**Compaction** — `compact(messages)` at [`agent.py:56-79`](agent.py#L56):

1. `last_turn_start` (`agent.py:48-53`) scans backwards for the most recent `user` message; that index is the boundary of the in-flight turn.
2. If the boundary is `<= 1` there are no completed turns; it prints `[nothing to compact yet]` and returns without calling the model.
3. `render` (`agent.py:37-45`) flattens `messages[:cut]` to `role: content` lines plus `tool_call: name(args)` lines. **`role: tool` results are included as content, but `tool_call_id` linkage is dropped.**
4. One **non-streaming** model call with the `SUMMARIZER` prompt (`agent.py:17-32`), sent as a single `system` message with the conversation interpolated.
5. `messages[:] = [messages[0], {"role": "user", "content": "Summary of the conversation so far:\n\n" + summary}] + messages[cut:]` — original system message, the summary as a **user** message, then the untouched current turn.
6. `context_tokens` is reset to `response.usage.completion_tokens` — the summary length, not a re-count of the new list.

Both compaction paths use this same function: auto (`agent.py:96`) and manual `/compact` (`juju.py:31-33`).

The counter is a module global, so **a sub-agent's usage overwrites the parent's count** (section 6).

---

## 6. Subagents

Yes. `SpawnAgentTool`, exposed to the model as `task` — [`tools.py:221-251`](tools.py#L221).

`execute` (`tools.py:243-251`) does:

```python
def execute(self, args: dict) -> str:
    import agent

    messages = [
        {"role": "system", "content": agent.SYSTEM},
        {"role": "user", "content": args["prompt"]},
    ]
    tools = self._get_tools()
    return agent.run(messages, tools, lambda tool, args: True)
```

Observed properties:

- **It is the same `agent.run` function.** A sub-agent is a recursive call into the identical loop, not a separate implementation.
- **Fresh two-message list**, built locally. It is not connected to the parent's `messages`; the parent sees only the returned final string, delivered as the `role: tool` content.
- **The `import agent` is inside the function**, not at module top. `agent.py:6` imports from `tools`, so a top-level import here would be circular.
- **Auto-approve.** The `approve` callback is `lambda tool, args: True`, so nothing inside a sub-agent prompts. The human approves *spawning* (`task` is `is_read_only = False`); everything the child then does — writes, edits, shell commands — runs ungated.
- **Recursion guard.** `_get_tools` (`tools.py:237-241`) filters `task` out of `TOOL_REGISTRY` by name, so a sub-agent cannot spawn a grandchild. Depth is capped at exactly 1. The result is cached on `self._tools` after first use.
- **Shared mutable tool instances.** The filtered dict holds the *same objects* as the parent's registry. A sub-agent calling `todo_write` overwrites the parent's `self.todos`.
- **Shared token counter.** `agent.context_tokens` is a module global, so sub-agent usage overwrites the parent's value, and a compaction inside a sub-agent resets the parent's counter.
- **Sequential.** Measured: three sub-agents each doing 1s of work take 3.0s.
- **Output is not namespaced.** The child streams to the same stdout with the same `[tool_name]` formatting; nothing distinguishes parent output from child output on screen.

---

## 7. Session state

**There is none.**

- No session ID is generated anywhere.
- No session directory is created. No `mkdir`, no temp dir, no state path exists in the code.
- No artifact directory. Tools write wherever the model specifies a path — `write_file` and `edit_file` take an arbitrary `path`, and `bash` runs arbitrary commands — all relative to the process CWD.
- The only durable state the agent produces is whatever those tools write directly to the user's filesystem.
- The only *read* of persistent state is `JUJU.md` at import (`tools.py:27`), and `.env` via `load_dotenv()` (`tools.py:11`).

The nearest thing to session state is in-memory and lost on exit: the `messages` list (`juju.py:26`), the `plan` boolean (`juju.py:25`), `agent.context_tokens` (`agent.py:34`), and `TodoWriteTool.todos` (`tools.py:211`).

Configuration comes from environment variables, read at import: `OPENROUTER_API_KEY` (`agent.py:11`), `FIRECRAWL_KEY` (`tools.py:14`), `DEV_MODEL` (`tools.py:16`), `DEV_CONTEXT_WINDOW` / `DEV_RESERVE_TOKENS` (`agent.py:14-15`).

---

## 8. Cancellation

**There is no cancellation handling.** Grep for `KeyboardInterrupt|signal|SIGINT|atexit|finally` returns zero hits outside `.venv`.

Ctrl-C mid-tool-call terminates the process. Verified empirically: with a `bash` tool running `sleep 30`, SIGINT produced exit code `-2` and this traceback:

```
tool started
Traceback (most recent call last):
  File "<string>", line 4, in <module>
  File "tools.py", line 176, in execute
    result = subprocess.run(
  ...
KeyboardInterrupt
```

Consequences, in order:

1. `KeyboardInterrupt` inherits from `BaseException`, **not** `Exception` — so the `try/except Exception` wrappers inside every tool do not catch it. It propagates straight through `execute`.
2. It propagates through `agent.run`, whose only `try/except` (`agent.py:106`) wraps just the model call and also catches only `Exception`.
3. It propagates through `main()` and kills the interpreter.
4. The `role: tool` result for the in-flight call is **never appended**, so the transcript is left with an `assistant` message carrying `tool_calls` and no matching `tool` responses. That list would be rejected by the API if resent — but it is moot, because:
5. Nothing is persisted, so the entire conversation is lost.

There is no way to interrupt a single tool call, or a single model response, and keep the session. The same applies to Ctrl-C at the `input()` prompt (`juju.py:29`) and during a sub-agent run.

The `bash` tool sets no `subprocess.run` timeout, so a command that never returns blocks the agent with no recovery other than killing the process.

---

## Migration risks

### (a) Replacing all tools with a single tool

1. **Nine hardcoded call sites define the tool set.** `ALL_TOOLS` (`tools.py:313-323`) is a literal list of instantiated classes. `TOOL_REGISTRY` and `get_tools` derive from it. Collapsing to one tool means every consumer of the nine-name assumption changes.

2. **`is_read_only` is load-bearing in two independent places.** It drives the approval gate (`juju.py:9`, via `agent.py:153`) and plan mode's filtering (`tools.py:330-332`). A single tool has one `is_read_only` value, so both mechanisms lose their granularity: the gate can no longer distinguish "read a file" from "rm -rf", and plan mode can no longer produce a read-only subset by filtering. Plan mode's enforcement currently *depends* on removing tools from the schema list, which a single-tool design cannot do.

3. **Dispatch is by name with no fallback.** `tools[call["function"]["name"]]` (`agent.py:150`) raises `KeyError` on any unrecognized name. During migration the model will keep emitting old tool names — from its own priors and from any summary text produced by compaction, which embeds `tool_call: read_file(...)` lines (`agent.py:44`) back into the transcript.

4. **Tool identity is baked into the system prompt.** `tools.py:19` names `todo_write` explicitly. That line contradicts a single-tool world and is interpolated at import.

5. **`TodoWriteTool` holds mutable instance state** (`self.todos`, `tools.py:211`) that survives across calls and is shared between parent and sub-agents. A stateless single tool has nowhere to put it.

6. **Result contract is `str` only.** `Tool.execute` returns a string (`tools.py:41`), and `agent.py:158` puts it straight into `content`. Any richer return (structured output, artifacts, streams) requires changing the abstract signature and every call site.

7. **Schemas are class attributes, not data.** Each `parameters` dict is a literal on a class (`tools.py:69-75`, `87-95`, etc.). A single tool with a dynamic schema has no existing mechanism to build one at runtime — `to_openai()` (`tools.py:44-53`) reads static attributes.

### (b) Letting a tool call recursively start another full agent run

1. **The recursion guard is a hard name filter, not a depth counter.** `_get_tools` removes `"task"` by literal string (`tools.py:240`). Depth is capped at exactly 1. There is no depth parameter anywhere in `run`'s signature (`agent.py:89`), so arbitrary nesting cannot currently be expressed, and re-enabling `task` for children would allow unbounded recursion with no limit to check against.

2. **`context_tokens` is a single module-level global** (`agent.py:34`). Every nested `run` reads and writes the same variable. With recursion this becomes incoherent: a child's usage triggers the parent's auto-compaction, and a child's `compact` resets the parent's counter to the child's summary length (`agent.py:78`). Correct nesting requires per-run token state, which means changing `run`'s signature or returning usage alongside the reply.

3. **Compaction mutates the caller's list in place** via `messages[:] =` (`agent.py:74`). Any nested run holding a reference to a shared list would have its history rewritten underneath it. Currently safe only because sub-agents build their own list (`tools.py:246-249`).

4. **Tool instances are singletons shared across all depths** (`tools.py:313-325`, confirmed by identity check). Any tool holding state — `TodoWriteTool.todos`, and `SpawnAgentTool._tools` itself — is shared by every agent in the tree. This is currently safe *only* because execution is sequential (`agent.py:149`); recursion plus any concurrency turns it into a race.

5. **`run` returns a bare `str`** (`agent.py:144`). A recursive call surfaces only final text — no token usage, no structured result, no error channel, no indication the child failed. `agent.py:108` returns `""` on an API error, which is indistinguishable from a child that legitimately produced no text.

6. **Approval does not propagate.** `SpawnAgentTool` hardcodes `lambda tool, args: True` (`tools.py:251`), discarding the parent's `approve` callback. Under recursion, one approval at depth 0 silently authorizes every action at every depth below it.

7. **stdout is global and unnamespaced.** `run` prints directly (`agent.py:124`, `152`). Nested runs interleave into one stream with no depth marker or prefix.

8. **The lazy `import agent` inside `execute`** (`tools.py:244`) exists to dodge the `agent` ↔ `tools` circular import (`agent.py:6` imports from `tools`). Any restructuring that makes recursion a first-class concept has to resolve that cycle rather than work around it.

9. **No cancellation or depth-aware cleanup.** Per section 8, Ctrl-C at any depth kills the whole process and discards everything. There is no mechanism to abort one child run and return control to its parent.

