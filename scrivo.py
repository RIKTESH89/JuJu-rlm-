import os

import ui
from agent import PLAN_OFF, PLAN_ON, compact, run
from tools import MODEL, SYSTEM, get_tools, shutdown_root


def ask(tool, args) -> bool:
    """Prompt user for approval on non-read-only tools."""
    return tool.is_read_only or ui.approve(tool.name, args)


def main() -> None:
    ui.banner("scrivo", MODEL, os.getcwd())
    plan = False
    messages = [{"role": "system", "content": SYSTEM}]

    try:
        repl(messages, plan)
    finally:
        # Ctrl-C, a crash, or a clean exit all land here.
        shutdown_root()


def repl(messages, plan: bool) -> None:
    while True:
        line = ui.prompt(plan)

        if line == "/compact":
            compact(messages)
            continue

        if line == "/plan":
            plan = not plan
            messages.append({"role": "system", "content": PLAN_ON if plan else PLAN_OFF})
            ui.notice("plan mode on" if plan else "plan mode off", "warn")
            continue

        messages.append({"role": "user", "content": line})
        tools = get_tools(read_only=plan)
        run(messages, tools, ask)


if __name__ == "__main__":
    main()
