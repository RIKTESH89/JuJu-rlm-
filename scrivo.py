import os

from agent import PLAN_OFF, PLAN_ON, compact, run
from tools import MODEL, SYSTEM, get_tools, shutdown_root


def ask(tool, args) -> bool:
    """Prompt user for approval on non-read-only tools."""
    return tool.is_read_only or input("[y/n] ") == "y"


def banner() -> None:
    """Print a formatted banner with model and environment info."""
    width = 52
    print(f"+{'-' * width}+")
    print(f"| scrivo{' ' * (width - 7)}|")
    print(f"| {MODEL}".ljust(width) + " |")
    print(f"| cwd: {os.getcwd()}".ljust(width) + " |")
    print(f"| /plan  /compact  ctrl-c to quit".ljust(width) + " |")
    print(f"+{'-' * width}+")


def main() -> None:
    banner()
    plan = False
    messages = [{"role": "system", "content": SYSTEM}]

    try:
        repl(messages, plan)
    finally:
        # Ctrl-C, a crash, or a clean exit all land here.
        shutdown_root()


def repl(messages, plan: bool) -> None:
    while True:
        line = input("\n(plan) > " if plan else "\n> ")

        if line == "/compact":
            compact(messages)
            continue

        if line == "/plan":
            plan = not plan
            messages.append({"role": "system", "content": PLAN_ON if plan else PLAN_OFF})
            print("plan mode on" if plan else "plan mode off")
            continue

        messages.append({"role": "user", "content": line})
        tools = get_tools(read_only=plan)
        run(messages, tools, ask)


if __name__ == "__main__":
    main()
