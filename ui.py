"""Everything scrivo draws on the terminal.

Nothing else in the codebase writes to stdout. Keeping it here means one place
decides colour, width, and how deep a child agent is indented.
"""

import contextlib
import re
import time

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console(soft_wrap=False)

# How much of a cell's output to show on screen. Unrelated to the 8000-char
# budget in tools.shape_output, which is what the *model* sees.
MAX_RESULT_LINES = 15
INDENT_PER_DEPTH = 2

SEPARATOR = re.compile(r"[-_=]{20,}")

ACCENT = "cyan"
MUTED = "grey58"


def _indent(renderable, depth: int):
    return Padding(renderable, (0, 0, 0, depth * INDENT_PER_DEPTH))


def banner(name: str, model: str, cwd: str) -> None:
    body = Text()
    body.append(f"{model}\n", style=ACCENT)
    body.append(f"{cwd}\n", style=MUTED)
    body.append("/plan", style="bold")
    body.append("  toggle plan mode     ", style=MUTED)
    body.append("/compact", style="bold")
    body.append("  summarize history\n", style=MUTED)
    body.append("ctrl-c", style="bold")
    body.append("  quit", style=MUTED)

    console.print()
    console.print(Panel(body, title=f"[bold]{name}[/bold]", border_style=ACCENT,
                        padding=(1, 2)))


def prompt(plan: bool) -> str:
    """Read a line from the user. readline gives arrow keys and history free."""
    with contextlib.suppress(ImportError):
        import readline  # noqa: F401

    console.print()
    label = "(plan) ›" if plan else "›"
    style = "bold yellow" if plan else f"bold {ACCENT}"
    return console.input(f"[{style}]{label}[/] ")


@contextlib.contextmanager
def thinking(label: str = "thinking"):
    """A spinner for the dead air before the first token arrives."""
    if not console.is_terminal:
        yield lambda: None
        return

    status = console.status(f"[{MUTED}]{label}…[/]", spinner="dots")
    status.start()
    stopped = []

    def stop():
        if not stopped:
            stopped.append(True)
            status.stop()

    try:
        yield stop
    finally:
        stop()


def assistant_text(chunk: str) -> None:
    console.print(chunk, end="", markup=False, highlight=False)


def assistant_done() -> None:
    console.print()


def tool_call(name: str, args: dict, depth: int = 0) -> None:
    """The cell itself, syntax highlighted."""
    code = args.get("code")
    if code is None:
        body = Text(repr(args), style=MUTED)
        title = name
    else:
        first = code.lstrip().split("\n", 1)[0]
        language = "bash" if first.startswith("%%bash") else "python"
        body = Syntax(code.strip("\n"), language, theme="ansi_dark",
                      background_color="default", word_wrap=True)
        title = f"{name} · {language}" if language == "bash" else name

    console.print()
    console.print(_indent(
        Panel(body, title=f"[{ACCENT}]{title}[/]", title_align="left",
              border_style=MUTED, padding=(0, 1)),
        depth,
    ))


def tool_result(text: str, is_error: bool = False, depth: int = 0) -> None:
    """What the cell produced. Capped, except for failures."""
    if not text.strip():
        return

    lines = [
        line for line in text.rstrip("\n").split("\n")
        # IPython rules off its tracebacks with long dashes; at terminal width
        # those wrap into a stray character on the next line.
        if not SEPARATOR.fullmatch(line.strip())
    ]
    if not lines:
        return
    style = "red" if is_error else MUTED

    # A traceback's value is in its tail, so never trim one.
    if not is_error and len(lines) > MAX_RESULT_LINES:
        hidden = len(lines) - MAX_RESULT_LINES
        shown = lines[:MAX_RESULT_LINES]
        body = Group(
            Text("\n".join(shown), style=style),
            Text(f"… +{hidden} more lines", style=f"italic {MUTED}"),
        )
    else:
        body = Text("\n".join(lines), style=style)

    console.print(_indent(body, depth + 1))


def approve(name: str, args: dict) -> bool:
    """The permission gate. Anything but y declines."""
    answer = console.input(f"  [bold yellow]run {name}?[/] [{MUTED}]\\[y/n][/] ")
    if not console.is_terminal:
        console.print()  # piped stdin does not echo, so keep the lines apart
    return answer.strip() == "y"


def notice(text: str, kind: str = "info") -> None:
    styles = {"info": MUTED, "warn": "yellow", "error": "red"}
    console.print()
    console.print(Text(text, style=styles.get(kind, MUTED)))


def child_start(record: dict) -> None:
    """Begin a live line for a running child agent."""
    if not console.is_terminal:
        return
    status = console.status(_child_label(record), spinner="dots")
    status.start()
    record["_status"] = status


def child_tick(record: dict) -> None:
    """Refresh a running child's elapsed time."""
    status = record.get("_status")
    if status is not None:
        status.update(_child_label(record))


def child_finish(record: dict) -> None:
    """Replace the live line with a final one."""
    status = record.pop("_status", None)
    if status is not None:
        status.stop()

    seconds = record.get("duration_ms", 0) / 1000
    state = record.get("status", "done")
    colour = {"completed": "green", "failed": "red",
              "cancelled": "yellow"}.get(state, MUTED)
    console.print(
        Text.assemble(
            ("  ", ""),
            (record["name"], ACCENT),
            (f"  {seconds:.1f}s  ", MUTED),
            (state, colour),
        )
    )


def _child_label(record: dict) -> str:
    elapsed = time.time() - record["started_at"]
    return (
        f"[{ACCENT}]{record['name']}[/] "
        f"[{MUTED}]{record['model']}  {elapsed:.0f}s[/]"
    )
