"""
Example: exposing GitHub Copilot (via the official Copilot SDK) as a
"generate_code" tool that your own agent can call.

Auth: relies on the local `copilot login` session you already set up — no
token is read from the environment. That's deliberate; forcing a token
overrides use_logged_in_user and was the first bug we hit.
"""
import asyncio
import re
from pathlib import Path

from copilot import CopilotClient, PermissionRequest, PermissionRequestResult
from copilot.rpc import PermissionDecisionApproveOnce, PermissionDecisionReject
from copilot.session_events import PermissionRequestShell, PermissionRequestWrite


def _text_only_permissions(
    request: PermissionRequest, invocation: dict
) -> PermissionRequestResult:
    """
    We only want a text answer back — no shell commands, no file writes.
    Anything else (read, memory, etc.) is approved since it's harmless and
    may be part of how the agent normally composes a reply.
    """
    match request:
        case PermissionRequestShell():
            return PermissionDecisionReject(feedback="This tool only returns text; no shell access.")
        case PermissionRequestWrite():
            return PermissionDecisionReject(feedback="This tool only returns text; no file writes.")
        case _:
            return PermissionDecisionApproveOnce()


async def generate_code_with_copilot(prompt: str, model: str = "gpt-5") -> str:
    """
    Ask Copilot's agent to write code for `prompt` and return its response
    text (typically a fenced code block plus a short explanation).
    """
    final_text = ""
    done = asyncio.Event()

    def on_event(event) -> None:
        nonlocal final_text
        if event.type.value == "assistant.message":
            final_text = event.data.content
        elif event.type.value == "session.idle":
            done.set()

    async with CopilotClient() as client:
        async with await client.create_session(
            model=model,
            on_permission_request=_text_only_permissions,
        ) as session:
            session.on(on_event)
            await session.send(
                "Write code only, with no actions taken on disk — just "
                f"reply with the code in a single fenced code block for: {prompt}"
            )
            try:
                await asyncio.wait_for(done.wait(), timeout=60)
            except asyncio.TimeoutError:
                raise RuntimeError("Copilot did not return a response within 60s.")

    if not final_text:
        raise RuntimeError(
            "Copilot returned no assistant message even though the session went idle. "
            "Run copilot_debug.py again with this same permission handler to isolate it."
        )

    return final_text


def extract_code_block(response_text: str) -> str:
    """Pull the first fenced code block out of Copilot's reply, if present.
    Falls back to the raw text if Copilot didn't fence it."""
    match = re.search(r"```(?:\w+\n)?(.*?)```", response_text, re.DOTALL)
    return match.group(1).strip() if match else response_text.strip()


async def write_code_to_file(prompt: str, filepath, model: str = "gpt-5"):
    """
    Generate code for `prompt` via Copilot and write it to `filepath`
    (creating parent directories if needed). This writes the file ourselves
    rather than letting Copilot's own write tool do it — keeps full control
    over exactly where the file lands and what ends up in it.
    """
    filepath = Path(filepath)
    response = await generate_code_with_copilot(
        f"{prompt} (this will be saved as {filepath.name})", model=model
    )
    code = extract_code_block(response)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(code, encoding="utf-8")
    return filepath


async def main() -> None:
    written_path = await write_code_to_file(
        "a Python function that checks if a string is a palindrome, ignoring case and spaces",
        Path(__file__).with_name("palindrome.py"),
    )
    print(f"Wrote: {written_path}")
    print(written_path.read_text())


if __name__ == "__main__":
    asyncio.run(main())