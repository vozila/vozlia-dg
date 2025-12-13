import os
import json
import time
import base64
import asyncio
import logging
import signal
import re
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import httpx
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse

from vozlia_obs import ObsHub, mount_routes  # <- your obs hub/routes




# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")

# -------------------------
# App
# -------------------------
app = FastAPI()

# -------------------------
# Routers
# -------------------------

from assistant_route import router as assistant_router
app.include_router(assistant_router)
# -------------------------
# Observability hub + routes
# -------------------------
obs = ObsHub(logger)
mount_routes(app, obs)

# -------------------------
# Shared HTTP clients (performance)
# -------------------------
openai_client: httpx.AsyncClient | None = None
eleven_client: httpx.AsyncClient | None = None
router_client: httpx.AsyncClient | None = None

# -------------------------
# Per-call state (session memory)
# streamSid -> {"topic": str|None, "last_user": str, "last_bot": str}
# -------------------------
CALL_STATE: dict[str, dict] = {}

# -------------------------
# Env / Config
# -------------------------
PORT = int(os.getenv("PORT", "10000"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://vozlia-dg.onrender.com").rstrip("/")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

FSM_ROUTER_URL = os.getenv("FSM_ROUTER_URL", f"{PUBLIC_BASE_URL}/assistant/route").rstrip("/")

TWILIO_SAMPLE_RATE = 8000
TWILIO_CODEC = "mulaw"

DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2-phonecall")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "en-US")

FINAL_UTTERANCE_COOLDOWN_MS = int(os.getenv("FINAL_UTTERANCE_COOLDOWN_MS", "450"))
MIN_FINAL_CHARS = int(os.getenv("MIN_FINAL_CHARS", "2"))

# Keep TTS outputs short enough to avoid long “dead air” stretches
MAX_SPOKEN_CHARS = int(os.getenv("MAX_SPOKEN_CHARS", "700"))
ACK_BEFORE_THINKING = os.getenv("ACK_BEFORE_THINKING", "0") == "1"

# Twilio expects 20ms frames for 8k mulaw: 160 bytes
TWILIO_FRAME_BYTES = 160
TWILIO_FRAME_SECONDS = 0.02

logger.info(f"PUBLIC_BASE_URL={PUBLIC_BASE_URL}")
logger.info(f"FSM_ROUTER_URL={FSM_ROUTER_URL}")


# -------------------------
# µ-law RMS (for “is it alive / clipping / static?”)
# -------------------------
MU_LAW_DECODE_TABLE = []
for i in range(256):
    mu = ~i & 0xFF
    sign = (mu & 0x80)
    exponent = (mu >> 4) & 0x07
    mantissa = mu & 0x0F
    magnitude = ((mantissa << 1) + 1) << (exponent + 2)
    sample = magnitude - 132
    MU_LAW_DECODE_TABLE.append(-sample if sign else sample)


def ulaw_rms(ulaw_bytes: bytes) -> float:
    if not ulaw_bytes:
        return 0.0
    acc = 0.0
    for b in ulaw_bytes:
        s = MU_LAW_DECODE_TABLE[b]
        acc += s * s
    return math.sqrt(acc / len(ulaw_bytes))


# -------------------------
# Lifecycle + SIGTERM
# -------------------------
@app.on_event("startup")
async def startup():
    global openai_client, eleven_client, router_client
    logger.info("APP STARTUP")
    openai_client = httpx.AsyncClient(timeout=25.0)
    eleven_client = httpx.AsyncClient(timeout=45.0)
    router_client = httpx.AsyncClient(timeout=25.0)


@app.on_event("shutdown")
async def shutdown():
    global openai_client, eleven_client, router_client
    logger.warning("APP SHUTDOWN (process terminated)")
    if openai_client:
        await openai_client.aclose()
    if eleven_client:
        await eleven_client.aclose()
    if router_client:
        await router_client.aclose()


def _handle_sigterm(*_):
    logger.warning("SIGTERM received")


signal.signal(signal.SIGTERM, _handle_sigterm)


# -------------------------
# Basic endpoints
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "vozlia-dg", "pipeline": "twilio->deepgram->router->elevenlabs->twilio"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# -------------------------
# Twilio inbound (TwiML)
# -------------------------
@app.post("/twilio/inbound-dg")
async def twilio_inbound(request: Request):
    base = str(request.base_url).rstrip("/")
    stream_url = base.replace("https://", "wss://") + "/twilio/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Hi, you’ve reached Vozlia. One moment while I connect.</Say>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>
"""
    logger.info(f"TwiML Stream URL => {stream_url}")
    return PlainTextResponse(twiml, media_type="application/xml")


# -------------------------
# Helpers
# -------------------------
def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))


def _bytes_to_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("utf-8")


def deepgram_ws_url() -> str:
    params = (
        f"model={DEEPGRAM_MODEL}"
        f"&language={DEEPGRAM_LANGUAGE}"
        f"&encoding=mulaw"
        f"&sample_rate={TWILIO_SAMPLE_RATE}"
        f"&punctuate=true"
        f"&interim_results=true"
        f"&endpointing=300"
        f"&smart_format=true"
    )
    return f"wss://api.deepgram.com/v1/listen?{params}"


async def twilio_clear(twilio_ws: WebSocket, stream_sid: str):
    try:
        await twilio_ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
    except Exception:
        logger.exception("Failed to send Twilio clear")


def _trim_for_voice(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "Okay."
    if len(t) <= MAX_SPOKEN_CHARS:
        return t
    cut = t[:MAX_SPOKEN_CHARS]
    last = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if last > 200:
        return cut[: last + 1].strip() + " If you want, I can keep going."
    return cut.strip() + "… If you want, I can keep going."


async def stream_ulaw_audio_to_twilio(
    twilio_ws: WebSocket,
    stream_sid: str,
    ulaw_audio: bytes,
    cancel_event: asyncio.Event,
):
    """
    Pace at real-time (20ms frames). This avoids “dead air” or weird playback artifacts.
    """
    idx = 0
    total = len(ulaw_audio)

    last_send_ts = 0.0
    seq = 0

    while idx < total:
        if cancel_event.is_set():
            return

        chunk = ulaw_audio[idx: idx + TWILIO_FRAME_BYTES]
        idx += TWILIO_FRAME_BYTES
        seq += 1

        now = time.time()
        dt_ms = (now - last_send_ts) * 1000.0 if last_send_ts else 0.0
        last_send_ts = now

        # OBS: outbound chunk
        await obs.emit(
            stream_sid,
            {
                "type": "audio_out",
                "seq": seq,
                "bytes": len(chunk),
                "dt_ms": round(dt_ms, 2),
                "rms": round(ulaw_rms(chunk), 2),
            },
        )

        msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": _bytes_to_b64(chunk)}}
        await twilio_ws.send_text(json.dumps(msg))

        await asyncio.sleep(TWILIO_FRAME_SECONDS)


async def elevenlabs_tts_ulaw_bytes(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=ulaw_8000"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/ulaw",
    }
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.75},
    }

    if eleven_client is None:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, headers=headers, json=body)
    else:
        r = await eleven_client.post(url, headers=headers, json=body)

    r.raise_for_status()
    logger.info(f"ElevenLabs OK: bytes={len(r.content)}")
    return r.content


async def call_fsm_router(text: str, meta: Optional[Dict[str, Any]] = None) -> str:
    payload = {"text": text, "meta": meta or {}}
    try:
        if router_client is None:
            async with httpx.AsyncClient(timeout=25.0) as client:
                r = await client.post(FSM_ROUTER_URL, json=payload)
        else:
            r = await router_client.post(FSM_ROUTER_URL, json=payload)

        r.raise_for_status()
        data = r.json()

        for key in ("spoken_reply", "reply", "text", "message"):
            if isinstance(data, dict) and isinstance(data.get(key), str) and data[key].strip():
                return data[key].strip()

        if isinstance(data, str) and data.strip():
            return data.strip()

        return "Okay."
    except Exception as e:
        logger.exception(f"FSM router call failed: {e}")
        return "I’m having trouble reaching my brain service right now. Please try again."


# -------------------------
# Twilio stream WebSocket
# -------------------------
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio media stream connected")

    stream_sid: Optional[str] = None

    # Deepgram
    dg_ws = None
    dg_task: Optional[asyncio.Task] = None

    # Speech tasks
    speak_task: Optional[asyncio.Task] = None
    speak_cancel = asyncio.Event()
    speak_lock = asyncio.Lock()
    assistant_speaking = False

    # Debounce finals
    last_final_ts = 0.0
    last_final_text = ""

    # OBS inbound pacing stats
    in_last_ts = 0.0
    in_seq = 0

    async def cancel_speaking():
        nonlocal speak_task, assistant_speaking
        if speak_task and not speak_task.done():
            speak_cancel.set()
            if stream_sid:
                await twilio_clear(websocket, stream_sid)
            try:
                speak_task.cancel()
            except Exception:
                pass
        assistant_speaking = False
        speak_cancel.clear()

    async def speak(text: str):
        """
        Serialize speech to avoid overlapping audio (which often sounds like static).
        """
        nonlocal speak_task, assistant_speaking
        if not stream_sid:
            logger.warning("speak() called before stream_sid; skipping")
            return

        async with speak_lock:
            await cancel_speaking()

            assistant_speaking = True
            speak_cancel.clear()
            spoken = _trim_for_voice(text)

            async def _run():
                nonlocal assistant_speaking
                t0 = time.time()
                try:
                    tts0 = time.time()
                    ulaw = await elevenlabs_tts_ulaw_bytes(spoken)
                    tts_ms = (time.time() - tts0) * 1000.0

                    speak0 = time.time()
                    await stream_ulaw_audio_to_twilio(websocket, stream_sid, ulaw, speak_cancel)
                    speak_ms = (time.time() - speak0) * 1000.0

                    # OBS: stage timing (router_ms may be filled by caller)
                    await obs.emit(
                        stream_sid,
                        {
                            "type": "stage",
                            "tts_ms": round(tts_ms, 2),
                            "speak_ms": round(speak_ms, 2),
                            "total_ms": round((time.time() - t0) * 1000.0, 2),
                        },
                    )

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.exception(f"ElevenLabs/Twilio speak failed: {e}")
                finally:
                    assistant_speaking = False

            speak_task = asyncio.create_task(_run())

    async def deepgram_reader():
        nonlocal last_final_ts, last_final_text, assistant_speaking
        try:
            async for msg in dg_ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                ch = data.get("channel") or {}
                alts = ch.get("alternatives") or []
                if not alts:
                    continue

                transcript = (alts[0].get("transcript") or "").strip()
                if not transcript:
                    continue

                is_final = bool(data.get("is_final"))

                # barge-in: cancel speaking on interim if assistant is talking
                if assistant_speaking and not is_final:
                    await cancel_speaking()

                if not is_final:
                    continue

                now = time.time()
                if len(transcript) < MIN_FINAL_CHARS:
                    continue

                if transcript == last_final_text and (now - last_final_ts) < 1.0:
                    continue

                if (now - last_final_ts) * 1000 < FINAL_UTTERANCE_COOLDOWN_MS:
                    if transcript.lower() in last_final_text.lower():
                        continue

                last_final_text = transcript
                last_final_ts = now

                logger.info(f"Deepgram FINAL: {transcript}")

                if not stream_sid:
                    continue

                if ACK_BEFORE_THINKING:
                    try:
                        await speak("Okay.")
                    except Exception:
                        pass

                # Router timing
                state = CALL_STATE.get(stream_sid) or {"topic": None, "last_user": "", "last_bot": ""}
                state["last_user"] = transcript
                CALL_STATE[stream_sid] = state

                r0 = time.time()
                reply = await call_fsm_router(transcript, meta={"stream_sid": stream_sid, "state": state})
                router_ms = (time.time() - r0) * 1000.0

                # OBS: router stage
                await obs.emit(
                    stream_sid,
                    {"type": "stage", "router_ms": round(router_ms, 2)},
                )

                logger.info(f"Router reply: {reply}")

                state["last_bot"] = reply
                CALL_STATE[stream_sid] = state

                await speak(reply)

        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"Deepgram reader error: {e}")

    try:
        if not DEEPGRAM_API_KEY:
            logger.error("Missing DEEPGRAM_API_KEY; cannot transcribe")
        else:
            logger.info("Connecting to Deepgram realtime WebSocket...")
            dg_ws = await websockets.connect(
                deepgram_ws_url(),
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                ping_interval=10,
                ping_timeout=20,
                close_timeout=5,
                max_size=2**23,
            )
            logger.info("Connected to Deepgram realtime.")
            dg_task = asyncio.create_task(deepgram_reader())

        while True:
            raw = await websocket.receive_text()
            evt = json.loads(raw)
            etype = evt.get("event")

            if etype == "connected":
                logger.info("Twilio stream event: connected")
                continue

            if etype == "start":
                stream_sid = evt.get("start", {}).get("streamSid")
                logger.info(f"Twilio stream event: start (streamSid={stream_sid})")

                if stream_sid:
                    CALL_STATE[stream_sid] = {"topic": None, "last_user": "", "last_bot": ""}
                    obs.ensure(stream_sid)
                    await obs.emit(stream_sid, {"type": "call_start"})

                try:
                    await speak("I’m connected. How can I help you today?")
                except Exception:
                    logger.exception("Initial ElevenLabs greeting failed")
                continue

            if etype == "media":
                if not stream_sid:
                    continue

                payload_b64 = evt.get("media", {}).get("payload")
                if not payload_b64:
                    continue

                audio_bytes = _b64_to_bytes(payload_b64)

                # OBS: inbound chunk stats
                in_seq += 1
                now = time.time()
                dt_ms = (now - in_last_ts) * 1000.0 if in_last_ts else 0.0
                in_last_ts = now

                await obs.emit(
                    stream_sid,
                    {
                        "type": "audio_in",
                        "seq": in_seq,
                        "bytes": len(audio_bytes),
                        "dt_ms": round(dt_ms, 2),
                        "rms": round(ulaw_rms(audio_bytes), 2),
                    },
                )

                # forward to Deepgram
                if dg_ws is not None:
                    try:
                        await dg_ws.send(audio_bytes)
                    except Exception:
                        logger.exception("Failed sending audio to Deepgram")
                continue

            if etype == "stop":
                logger.info("Twilio stream event: stop")
                break

    except Exception as e:
        logger.exception(f"Twilio stream handler error: {e}")

    finally:
        # Cancel speech
        try:
            await cancel_speaking()
        except Exception:
            pass

        # Deepgram cleanup
        if dg_task:
            dg_task.cancel()
            try:
                await dg_task
            except Exception:
                pass

        if dg_ws:
            try:
                await dg_ws.close()
            except Exception:
                pass

        # OBS: mark end (so "latest" can fall back to completed)
        if stream_sid:
            try:
                await obs.emit(stream_sid, {"type": "call_end"})
                # Your current vozlia_obs.py has mark_ended(). If not, comment this out.
                if hasattr(obs, "mark_ended"):
                    obs.mark_ended(stream_sid)
            except Exception:
                pass

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("Twilio media stream handler completed")
