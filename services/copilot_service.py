"""Copilot integration for the voice agent.

This module exposes a single safe entry point that the live voice agent can use
when a user asks for an autopilot-style task such as:

- "activate autopilot"
- "ask Copilot to write a program"
- "generate code for..."

The voice agent does not let Copilot mutate the workspace directly. Instead,
it returns a text answer that Gemini can speak back to the user.
"""

import asyncio
import re
from pathlib import Path
from typing import Any

from copilot import CopilotClient, PermissionRequest, PermissionRequestResult
from copilot.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject
from copilot.session_events import PermissionRequestShell, PermissionRequestWrite

AUTOPILOT_TRIGGER_PHRASES = (
    "activate autopilot",
    "use copilot",
    "ask copilot",
    "ask it to write",
    "write a program",
    "write a script",
    "write code",
    "generate code",
    "create a script",
    "autopilot",
)

COPILOT_TASK_DECLARATION = {
    "function_declarations": [
        {
            "name": "run_copilot_task",
            "description": (
                "Route an autopilot-style request to GitHub Copilot for text-only "
                "code generation or task execution guidance. Use this when the user "
                "asks to activate autopilot, ask Copilot to write code, generate a "
                "program, or create a script."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The full request to send to Copilot.",
                    }
                },
                "required": ["task"],
            },
        }
    ]
}


def should_use_copilot(text: str) -> bool:
    """Return True when the user's request sounds like an autopilot/code task."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False

    return any(phrase in lowered for phrase in AUTOPILOT_TRIGGER_PHRASES)


_FILENAME_LINE_RE = re.compile(r"^\s*FILENAME:\s*([a-zA-Z0-9_\-]+)\s*$", re.MULTILINE)

# Words that carry no topical meaning on their own — stripped before falling
# back to slugging the raw task text, so "write a program on stacks" reduces
# to "stacks" rather than "write_a_program_on".
_FILLER_WORDS = {
    "write", "a", "an", "the", "program", "script", "code", "function",
    "example", "that", "demonstrates", "demonstrate", "for", "instance",
    "please", "can", "you", "to", "of", "with", "on", "about", "generate",
    "create", "ask", "copilot", "activate", "autopilot", "one", "show",
    "me", "using", "is", "are", "it", "and", "or", "in", "implement",
}


def _parse_filename_and_code(raw_text: str) -> tuple[str | None, str]:
    """Pull the 'FILENAME: <slug>' line (if Copilot included one) out of the
    response, returning (slug_or_None, remaining_text_with_code)."""
    match = _FILENAME_LINE_RE.search(raw_text)
    if not match:
        return None, raw_text
    slug = match.group(1).lower()
    remaining = raw_text[: match.start()] + raw_text[match.end():]
    return slug, remaining


def _clean_response_text(text: str) -> str:
    """Trim common markdown fences and whitespace from Copilot replies."""
    if not text:
        return ""
    text = text.strip()
    match = re.search(r"```(?:\w+\n)?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _build_output_dir(base_dir: Path | None = None) -> Path:
    """Create and return the shared folder used for generated code files."""
    root = base_dir or Path(__file__).resolve().parent.parent
    output_dir = root / "copilot code outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _slugify_task_fallback(task: str, max_words: int = 4) -> str:
    """Fallback slug used only if Copilot didn't supply a FILENAME line —
    strips filler/instruction words so the topic noun survives, e.g.
    'write a program on stacks' -> 'stacks' rather than 'write_a_program_on'."""
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", task.lower()) if w not in _FILLER_WORDS]
    return "_".join(words[:max_words]) or "program"


def _build_output_filename(slug: str, output_dir: Path) -> str:
    """De-duplicate a topic slug into a filename, e.g. stacks.py, stacks_2.py
    if stacks.py already exists."""
    candidate = f"{slug}.py"
    if not (output_dir / candidate).exists():
        return candidate
    index = 2
    while (output_dir / f"{slug}_{index}.py").exists():
        index += 1
    return f"{slug}_{index}.py"


def write_generated_code(code: str, filename_slug: str, base_dir: Path | None = None) -> Path:
    """Write generated code into the output folder and return the created path."""
    output_dir = _build_output_dir(base_dir)
    filename = _build_output_filename(filename_slug, output_dir)
    file_path = output_dir / filename
    file_path.write_text(code.strip() + "\n", encoding="utf-8")
    return file_path


def _text_only_permissions(
    request: PermissionRequest, invocation: dict[str, Any]
) -> PermissionRequestResult:
    """Approve harmless actions but reject shell/file writes."""
    match request:
        case PermissionRequestShell():
            return PermissionDecisionReject(
                feedback="This workflow only returns text; shell access is disabled."
            )
        case PermissionRequestWrite():
            return PermissionDecisionReject(
                feedback="This workflow only returns text; file writes are disabled."
            )
        case _:
            return PermissionDecisionApproveOnce()


async def run_copilot_task(task: str, model: str = "gpt-5") -> dict[str, Any]:
    """Ask Copilot for a text answer suitable for the voice agent to speak."""
    if not task or not task.strip():
        return {"status": "error", "message": "No task provided."}

    final_text = ""
    done = asyncio.Event()

    def on_event(event) -> None:
        nonlocal final_text
        if event.type.value == "assistant.message":
            final_text = getattr(event.data, "content", "") or ""
        elif event.type.value == "session.idle":
            done.set()

    try:
        async with CopilotClient() as client:
            async with await client.create_session(
                model=model,
                on_permission_request=_text_only_permissions,
            ) as session:
                session.on(on_event)
                await session.send(
                    "You are a coding assistant. First, on a line by itself, output "
                    "exactly: FILENAME: <short_topic_slug> where <short_topic_slug> "
                    "is 1-3 lowercase words joined by underscores naming the core "
                    "topic (e.g. 'stacks', 'solid_single_responsibility', "
                    "'binary_search'), not the instruction itself. Then give a "
                    "concise answer in plain text. If code is needed, put it in a "
                    "single fenced code block. Do not perform file writes or shell "
                    f"actions. User request: {task}"
                )
                try:
                    await asyncio.wait_for(done.wait(), timeout=60)
                except asyncio.TimeoutError:
                    return {
                        "status": "error",
                        "message": "Copilot did not return a response in time.",
                    }
    except Exception as exc:  # pragma: no cover - defensive path
        return {"status": "error", "message": f"Copilot request failed: {exc}"}

    filename_slug, remaining_text = _parse_filename_and_code(final_text)
    cleaned = _clean_response_text(remaining_text)
    if not cleaned:
        return {"status": "error", "message": "Copilot returned an empty response."}

    if not filename_slug:
        filename_slug = _slugify_task_fallback(task)

    output_file = write_generated_code(cleaned, filename_slug)
    return {
        "status": "ok",
        "answer": cleaned,
        "source": "copilot",
        "output_file": str(output_file),
    }