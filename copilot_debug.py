"""
Debug version — temporarily logs everything and removes the two suspects
(forced PAT auth, blanket permission denial) so we can see what's actually
happening. Once this works, we tighten it back up.
"""
import asyncio

from copilot import CopilotClient
from copilot.session import PermissionHandler


async def debug_run() -> None:
    # No github_token= here on purpose: this lets the SDK fall back to
    # use_logged_in_user=True, i.e. whatever account the bundled CLI is
    # already logged into (the same one your IDE extension uses).
    async with CopilotClient() as client:
        # Cheap auth sanity check before we even try a chat — if this
        # throws or returns nothing, auth is the problem, full stop.
        try:
            models = await client.list_models()
            print("Authenticated OK. Available models:", models)
        except Exception as exc:
            print("AUTH FAILED before even creating a session:", repr(exc))
            return

        async with await client.create_session(
            model="gpt-5",
            on_permission_request=PermissionHandler.approve_all,  # wide open, for now
        ) as session:
            done = asyncio.Event()

            def on_event(event) -> None:
                # Log EVERY event type so nothing is invisible.
                print(f"[event] {event.type.value} -> {getattr(event, 'data', None)}")
                if event.type.value == "session.idle":
                    done.set()

            session.on(on_event)
            await session.send("What is 2+2?")  # trivial prompt, isolates the real bug
            await asyncio.wait_for(done.wait(), timeout=30)


if __name__ == "__main__":
    asyncio.run(debug_run())