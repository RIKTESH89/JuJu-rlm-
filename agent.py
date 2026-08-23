import json
import os
from typing import Optional

from openai import OpenAI

import ui
from tools import MODEL, SYSTEM

# API key should be provided via environment variable for security
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

CONTEXT_WINDOW = int(os.environ.get("DEV_CONTEXT_WINDOW", 262144))
RESERVE_TOKENS = int(os.environ.get("DEV_RESERVE_TOKENS", 20000))

SUMMARIZER = """You are a conversation summarizer. Create a comprehensive summary of this conversation that captures:

1. The main goals and objectives discussed
2. Key decisions made and their rationale
3. Important code changes, file modifications, or technical details
4. Current state of any ongoing work
5. Any blockers, issues, or open questions
6. Next steps that were planned or suggested

Be thorough but concise. The summary will replace the ENTIRE conversation history, so include all information needed to continue the work effectively.

Format the summary as structured markdown with clear sections.

<conversation>
{conversation}
</conversation>"""

class TokenState:
    """Two numbers that must never be conflated.

    `context` is how full the model's context window is right now — it drives
    compaction, and a child's tokens must NOT inflate it.
    `billed_input` / `billed_output` are what the session is charged for, which
    a child's tokens MUST inflate.
    """

    def __init__(self) -> None:
        self.context = 0
        self.billed_input = 0
        self.billed_output = 0

    @property
    def billed_total(self) -> int:
        return self.billed_input + self.billed_output


# Used by the interactive REPL and the root session.
DEFAULT_TOKENS = TokenState()


def render(messages: list[dict]) -> str:
    """Flatten the conversation into text for the summarizer."""
    lines = []
    for m in messages:
        lines.append(f"{m['role']}: {m.get('content') or ''}")
        for call in m.get("tool_calls", []):
            fn = call["function"]
            lines.append(f"tool_call: {fn['name']}({fn['arguments']})")
    return "\n".join(lines)


def last_turn_start(messages: list[dict]) -> int:
    """Index of the most recent user message — the current turn begins there."""
    for i in range(len(messages) - 1, 0, -1):
        if messages[i]["role"] == "user":
            return i
    return len(messages)


def compact(messages: list[dict], tokens: Optional[TokenState] = None) -> None:
    """Summarize completed turns in place, leaving the current turn untouched."""
    tokens = tokens or DEFAULT_TOKENS

    cut = last_turn_start(messages)
    if cut <= 1:
        ui.notice("nothing to compact yet")
        return

    before = tokens.context
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SUMMARIZER.format(conversation=render(messages[:cut]))}
        ],
    )
    summary = response.choices[0].message.content

    messages[:] = [
        messages[0],
        {"role": "user", "content": f"Summary of the conversation so far:\n\n{summary}"},
    ] + messages[cut:]
    tokens.context = response.usage.completion_tokens
    tokens.billed_input += response.usage.prompt_tokens
    tokens.billed_output += response.usage.completion_tokens
    ui.notice(f"compacted {before} → ~{tokens.context} tokens")


PLAN_ON = (
    "Plan mode is ON. You are read-only: research and propose a plan for approval. "
    "Do not modify anything."
)
PLAN_OFF = "Plan mode is OFF. You may modify things again."


def run(
    messages: list[dict],
    tools: dict,
    approve,
    model: Optional[str] = None,
    raise_on_error: bool = False,
    tokens: Optional[TokenState] = None,
    depth: int = 0,
) -> str:
    """Run the agent loop: stream LLM responses, execute tool calls as approved."""

    tokens = tokens or DEFAULT_TOKENS

    while True:
        if tokens.context > CONTEXT_WINDOW - RESERVE_TOKENS:
            compact(messages, tokens)

        try:
            stream = client.chat.completions.create(
                model=model or MODEL,
                messages=messages,
                tools=[t.to_openai() for t in tools.values()],
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as e:
            # Children need failures as exceptions so they can be reported as
            # "CHILD FAILED"; the interactive REPL keeps printing and carrying on.
            if raise_on_error:
                raise
            ui.notice(f"error calling the model: {e}", "error")
            return ""

        reply = ""
        tool_calls = []
        finish_reason = None

        spinner = ui.thinking()
        stop_spinner = spinner.__enter__()

        for chunk in stream:
            if chunk.usage:
                # total_tokens is the size of this call's context; prompt and
                # completion are what we are billed for.
                tokens.context = chunk.usage.total_tokens
                tokens.billed_input += chunk.usage.prompt_tokens
                tokens.billed_output += chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if choice.delta.content:
                stop_spinner()
                ui.assistant_text(choice.delta.content)
                reply += choice.delta.content

            for tc in choice.delta.tool_calls or []:
                while len(tool_calls) <= tc.index:
                    tool_calls.append(
                        {"id": "", "type": "function",
                         "function": {"name": "", "arguments": ""}}
                    )
                call = tool_calls[tc.index]
                if tc.id:
                    call["id"] = tc.id
                if tc.function.name:
                    call["function"]["name"] = tc.function.name
                if tc.function.arguments:
                    call["function"]["arguments"] += tc.function.arguments

        stop_spinner()
        spinner.__exit__(None, None, None)

        if finish_reason != "tool_calls":
            ui.assistant_done()
            messages.append({"role": "assistant", "content": reply})
            return reply

        messages.append(
            {"role": "assistant", "content": reply, "tool_calls": tool_calls}
        )
        for call in tool_calls:
            name = call["function"]["name"]
            tool = tools.get(name)
            if tool is None:
                # Models reach for tools that no longer exist — rlm() especially,
                # which is Python, not a tool. Say so instead of dying.
                known = ", ".join(sorted(tools)) or "none"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": (
                        f"no such tool: {name}. Available tools: {known}. "
                        "rlm() is not a tool — call it as Python inside an ipython cell, "
                        "e.g. answer = await rlm(\"task\", name=\"short-name\")"
                    ),
                })
                continue

            tool.current_call_id = call["id"]
            args = json.loads(call["function"]["arguments"])
            ui.tool_call(tool.name, args, depth)
            if approve(tool, args):
                result = tool.execute(args)
            else:
                result = "user denied"
            ui.tool_result(result, getattr(tool, "last_error", False), depth)
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )
