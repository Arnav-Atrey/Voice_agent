"""
Voice controller — orchestrates the live audio loop:

  mic  -> send_audio_task -----> Gemini Live session
  Gemini Live session -> receive_task -> speaker_queue -> audio_output_task -> speakers
                                  |
                                  +--> history_service (persists transcript to Mongo)

All console/transcript printing lives here too, replacing the old tkinter UI.
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
from services.gemini_service import build_live_config
from services.search_service import run_web_search


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


async def handle_tool_call(session, tool_call, db_session: ConversationSession) -> None:
    """Run each requested function, log it, and send results back."""
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
        await history_service.end_session(db_session)