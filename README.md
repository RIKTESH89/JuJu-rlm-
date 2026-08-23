# scrivo

An RLM terminal coding agent.

Most coding agents give the model a menu of tools — read a file, write a file, run a command
— and let it pick one per turn. scrivo gives it a Python kernel instead. The model writes code,
the kernel keeps its state between turns, and everything else follows from that: files are
`Path.read_text()`, shell is a `%%bash` cell, and a subagent is a function call.

```
+----------------------------------------------------+
| scrivo                                               |
| poolside/laguna-s-2.1:free                         |
| cwd: /path/to/your/project                         |
| /plan  /compact  ctrl-c to quit                    |
+----------------------------------------------------+

>
```

## What makes it different

**One tool.** `ipython`, running a persistent IPython kernel in its own virtualenv. Variables,
imports, and functions survive across turns, so the agent builds up state instead of redoing
work.

**Context lives in variables, not in the transcript.** The agent can hold a 5000-line file in
a variable and print only the three lines it needs. Output is truncated before it reaches the
model — but a traceback is never cut, because that is the part that explains the failure.

**Recursive subagents.** From inside a cell:

```python
answer = await rlm("Review auth.py for security issues", name="auth-reviewer")
```

That starts a *complete* child agent — its own kernel, its own context window, its own
session directory — and blocks until it returns its final answer as a string. Depth is
capped (default 1), so children cannot spawn grandchildren unless you raise the limit.

**Cost is tracked, not lost.** A child's tokens are billed to the parent session but never
added to the parent's context size. Conflating those two makes an agent look like it is about
to overflow its window when it is not.

**Nothing leaks.** Kernels, sockets, and child sessions are torn down on clean exit, Ctrl-C,
and unhandled exceptions alike.

## Other features

- **Permission gate** — anything that changes your machine shows you the exact call and waits
  for `y`. Read-only calls run immediately.
- **Compaction** — `/compact`, or automatically when the context window fills. Only completed
  turns are summarized; the turn in flight is left intact.
- **Project instructions** — drop a `SCRIVO.md` in your working directory and its contents
  become standing instructions.
- **Streaming** — replies print token by token.

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Create a `.env` in the project root:

```
OPENROUTER_API_KEY='your-openrouter-key'
DEV_MODEL='poolside/laguna-s-2.1:free'
```

Get a key at [openrouter.ai/keys](https://openrouter.ai/keys). `DEV_MODEL` is optional.

Then either activate the venv, or put `scrivo` on your PATH once:

```bash
ln -sf "$PWD/.venv/bin/scrivo" /opt/homebrew/bin/scrivo
```

## Usage

Run it from the directory you want to work in — that becomes its working context.

```bash
scrivo
```

Type at the `>` prompt. `/plan` toggles plan mode, `/compact` summarizes the conversation,
Ctrl-C quits.

The first `ipython` call builds a kernel virtualenv at `~/.myagent/kernel-venv` and boots a
kernel — one or two seconds, once per session. Sessions that never run code never pay it.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | — | required |
| `DEV_MODEL` | `poolside/laguna-s-2.1:free` | which model to run |
| `RLM_MAX_DEPTH` | `1` | how deep `rlm()` may recurse |
| `DEV_CONTEXT_WINDOW` | `262144` | context size used for auto-compaction |
| `DEV_RESERVE_TOKENS` | `20000` | headroom kept before compacting |
| `MYAGENT_KERNEL_PYTHON` | — | use this interpreter instead of building a venv |

## Tests

```bash
.venv/bin/python -m pytest
```

The end-to-end suite (`tests/test_rlm_e2e.py`) drives the real agent against a fixture
repository and makes real model calls — it is slow and consumes API quota. Everything else
runs offline in well under a minute.

## Notes

- The agent runs shell commands and edits files. Read what it asks before approving.
- Subagents run with auto-approval. Approving a cell that calls `rlm()` approves everything
  that child then does.
