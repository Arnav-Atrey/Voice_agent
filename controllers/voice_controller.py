"""
Voice controller — orchestrates the live audio loop:

  mic  -> send_audio_task -----> Gemini Live session
  Gemini Live session -> receive_task -> speaker_queue -> audio_output_task -> speakers
                                  |
                                  +--> history_service (persists transcript to Mongo)

All console/transcript printing lives here too, replacing the old tkinter UI.

Copilot dispatch is fire-and-forget: handle_tool_call() answers Gemini's
function call immediately with an "in_progress" status, then runs the real
Copilot session as a background asyncio.Task. When that finishes, the result
is re-injected into the live session as a fresh user turn via
session.send_client_content(), so Gemini can speak it without the receive
loop — which also drives audio playback — ever blocking on Copilot.
"""
import asyncio

import sounddevice as sd
from google import genai
from google.genai import types as genai_types

from config.settings import (
    CHANNELS,
    CHUNK_SIZE,
    GEMINI_LIVE_MODEL,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    require_gemini_api_key,
)
from models.conversation import ConversationSession
from services import history_service
from services.copilot_service import run_copilot_task
from services.gemini_service import build_live_config
from services.search_service import run_web_search

# Background Copilot tasks must be kept referenced somewhere, or asyncio is
# free to garbage-collect them mid-flight (a well-known asyncio footgun —
# a task with no surviving reference can be silently cancelled). One set per
# process is fine here since this is a single-session CLI agent.
_background_tasks: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Audio I/O tasks
# ---------------------------------------------------------------------------

async def audio_input_task(mic_queue: asyncio.Queue) -> None:
    """Capture 16 kHz 16-bit PCM from the microphone and push to mic_queue."""
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[mic] {status}", flush=True)
        loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(indata))

    with sd.RawInputStream(
        samplerate=INPUT_SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    ):
        while True:
            await asyncio.sleep(0.1)  # keep the context-manager alive


async def audio_output_task(
    speaker_queue: asyncio.Queue,
    interrupt_event: asyncio.Event,
) -> None:
    """Play 24 kHz PCM chunks from speaker_queue; drain on interruption."""
    stream = sd.RawOutputStream(
        samplerate=OUTPUT_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    stream.start()
    try:
        while True:
            chunk = await speaker_queue.get()
            if interrupt_event.is_set():
                while not speaker_queue.empty():
                    try:
                        speaker_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                interrupt_event.clear()
                continue
            stream.write(chunk)
    finally:
        stream.stop()
        stream.close()


# ---------------------------------------------------------------------------
# Send / receive tasks
# ---------------------------------------------------------------------------

async def send_audio_task(session, mic_queue: asyncio.Queue) -> None:
    """Forward every mic chunk to the Live session continuously (including
    silence — the server's VAD needs an uninterrupted stream)."""
    while True:
        chunk = await mic_queue.get()
        await session.send_realtime_input(
            audio=genai_types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
        )


async def _run_copilot_in_background(
    session, task: str, db_session: ConversationSession
) -> None:
    """
    The actual Copilot call, run outside the receive loop. On completion,
    injects the result back into the live session so Gemini picks it up and
    speaks it on its own schedule.

    Note: gemini-3.1-flash-live-preview does not yet support asynchronous
    function calling (Google's own docs confirm this), so the model will not
    generate anything further until *some* tool response is sent — that's
    why handle_tool_call() answers with "in_progress" immediately, rather
    than waiting for this function. This is also why the result is injected
    via send_realtime_input(text=...) instead of send_client_content():
    Google explicitly warns against mixing send_client_content with an
    active send_realtime_input audio stream ("will lead to unpredictable
    behavior and race conditions") — staying on the realtime channel avoids
    that entirely.

    If you ever switch GEMINI_LIVE_MODEL to gemini-2.5-flash-live-preview,
    that model *does* support NON_BLOCKING function declarations with a
    scheduling param (INTERRUPT/WHEN_IDLE/SILENT) — the officially
    sanctioned version of this pattern. Worth migrating to if/when you
    move models.
    """
    try:
        result = await run_copilot_task(task)
    except Exception as exc:
        result = {"status": "error", "message": str(exc)}

    if result.get("status") == "ok":
        print(f"[copilot background] done -> {result.get('output_file', 'n/a')}", flush=True)
    else:
        print(f"[copilot background] failed -> {result.get('message', 'unknown error')}", flush=True)
    await history_service.add_message(db_session, "tool", f"run_copilot_task({task}) -> {result}")

    if result.get("status") == "ok":
        injected_text = (
            "[System note, not from the user: a background Copilot task just "
            "finished. Summarize this for the user in 1-2 short spoken "
            "sentences, mentioning the file was saved if relevant, next time "
            "it's natural to speak.]\n"
            f"Task: {task}\n"
            f"Copilot's answer: {result.get('answer', '')}\n"
            f"Saved to: {result.get('output_file', 'n/a')}"
        )
    else:
        injected_text = (
            "[System note, not from the user: the background Copilot task "
            "failed. Briefly tell the user it didn't work and why, in 1 "
            "short spoken sentence, next time it's natural to speak.]\n"
            f"Task: {task}\n"
            f"Error: {result.get('message', 'unknown error')}"
        )

    try:
        await session.send_realtime_input(text=injected_text)
    except Exception as exc:
        # The Live session may already be closed (user hung up before
        # Copilot finished) — this is expected sometimes, not a real error.
        print(f"[copilot background] could not re-inject result: {exc}", flush=True)


async def handle_tool_call(session, tool_call, db_session: ConversationSession) -> None:
    """Run each requested function, log it, and send results back.

    run_copilot_task is dispatched fire-and-forget: Gemini gets an
    "in_progress" function response immediately (function calls must be
    answered to keep the session's turn state consistent), and the real
    work continues in a background task that re-injects its result later.
    Everything else (web_search) still runs and responds synchronously,
    since those calls are fast enough not to matter.
    """
    responses = []
    for fc in tool_call.function_calls or []:
        print(f"[tool] {fc.name}({dict(fc.args)})", flush=True)

        if fc.name == "web_search":
            query = (fc.args or {}).get("query", "")
            try:
                result = await run_web_search(query)
            except Exception as exc:
                result = {"error": str(exc)}
            await history_service.add_message(
                db_session, "tool", f"web_search({query}) -> {result}"
            )

        elif fc.name == "run_copilot_task":
            task = (fc.args or {}).get("task", "")
            result = {
                "status": "in_progress",
                "message": (
                    "Copilot is working on this in the background. Tell the user "
                    "you've kicked it off and you'll let them know when it's ready "
                    "— do not wait or go silent."
                ),
            }
            bg_task = asyncio.create_task(_run_copilot_in_background(session, task, db_session))
            _background_tasks.add(bg_task)
            bg_task.add_done_callback(_background_tasks.discard)

        else:
            result = {"error": f"Unknown tool: {fc.name}"}

        print(f"       -> {result}", flush=True)
        responses.append({"id": fc.id, "name": fc.name, "response": result})

    await session.send_tool_response(function_responses=responses)


async def receive_task(
    session,
    speaker_queue: asyncio.Queue,
    interrupt_event: asyncio.Event,
    db_session: ConversationSession,
) -> None:
    """Demultiplex all response types from Gemini Live, print transcript,
    and persist completed agent turns to Mongo."""
    printed_prefix = False
    pending_agent_text = ""

    while True:
        async for response in session.receive():

            if response.tool_call:
                await handle_tool_call(session, response.tool_call, db_session)
                continue

            sc = response.server_content
            if sc is None:
                continue

            if sc.interrupted:
                interrupt_event.set()
                while not speaker_queue.empty():
                    try:
                        speaker_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                if printed_prefix:
                    print(flush=True)
                printed_prefix = False
                pending_agent_text = ""
                print("[interrupted by user]", flush=True)
                continue

            if sc.model_turn:
                for part in sc.model_turn.parts or []:
                    if part.inline_data and part.inline_data.data:
                        await speaker_queue.put(part.inline_data.data)

            if sc.output_transcription and sc.output_transcription.text:
                text = sc.output_transcription.text.replace("\n", " ")
                if not printed_prefix:
                    print("Agent: ", end="", flush=True)
                    printed_prefix = True
                print(text, end="", flush=True)
                pending_agent_text += (" " if pending_agent_text else "") + text.strip()

            if sc.input_transcription and sc.input_transcription.text:
                text = sc.input_transcription.text.replace("\n", " ").strip()
                if text:
                    await history_service.add_message(db_session, "user", text)

            if sc.turn_complete and printed_prefix:
                print(flush=True)
                printed_prefix = False
                await history_service.add_message(db_session, "agent", pending_agent_text.strip())
                pending_agent_text = ""


# ---------------------------------------------------------------------------
# Top-level run loop
# ---------------------------------------------------------------------------

async def run() -> None:
    """Start a DB-backed session, connect to Gemini Live with prior-session
    context folded into the prompt, and run the audio loop until exit."""
    api_key = require_gemini_api_key()
    client = genai.Client(api_key=api_key)

    db_session = await history_service.start_session()
    history_context = await history_service.build_context_summary()
    config = build_live_config(history_context)

    mic_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    speaker_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    interrupt_event = asyncio.Event()

    print("CS Study Agent ready. Speak now. (Ctrl-C to quit.)", flush=True)
    print("Try: 'Explain binary search', 'Quiz me on OOP', 'What is a trie?'", flush=True)
    print("-" * 60, flush=True)

    try:
        async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
            await asyncio.gather(
                audio_input_task(mic_queue),
                audio_output_task(speaker_queue, interrupt_event),
                send_audio_task(session, mic_queue),
                receive_task(session, speaker_queue, interrupt_event, db_session),
            )
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    except Exception as exc:
        print(f"Session error: {exc}", flush=True)
    finally:
        # Give any in-flight Copilot background tasks a moment to finish
        # logging to history before we close the DB session out from
        # under them; don't hang forever if one stalls.
        if _background_tasks:
            await asyncio.wait(_background_tasks, timeout=5)
        await history_service.end_session(db_session)