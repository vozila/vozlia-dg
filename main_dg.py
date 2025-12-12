import os
import json
import time
import base64
import asyncio
import logging
import signal
import re
from typing import Optional, Dict, Any, Tuple

import httpx
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse

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
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://vozlia-backend.onrender.com").rstrip("/")

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

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Optional: allow disabling reasoning if it causes "reasoning-only" outputs
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower()
# Values: "", "low", "medium", "high"

# Voice / streaming safeguards
# Twilio expects 20ms frames for 8k mulaw: 160 bytes
TWILIO_FRAME_BYTES = 160
TWILIO_FRAME_SECONDS = 0.02

# Keep TTS outputs short enough to avoid long “dead air” stretches
MAX_SPOKEN_CHARS = int(os.getenv("MAX_SPOKEN_CHARS", "700"))  # ~45–60s depending on voice
ACK_BEFORE_THINKING = os.getenv("ACK_BEFORE_THINKING", "0") == "1"  # optional

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
    return {"ok": True, "service": "vozlia-backend", "pipeline": "twilio->deepgram->openai->elevenlabs"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -------------------------
# Twilio inbound (TwiML)
# -------------------------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    form = await request.form()
    from_num = form.get("From")
    to_num = form.get("To")
    call_sid = form.get("CallSid")
    logger.info(f"Incoming call: From={from_num}, To={to_num}, CallSid={call_sid}")

    stream_url = f"{PUBLIC_BASE_URL.replace('https://', 'wss://')}/twilio/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Hi, you’ve reached Vozlia. One moment while I connect.</Say>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>
"""
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
    # Try to cut at a sentence boundary
    cut = t[:MAX_SPOKEN_CHARS]
    m = re.search(r"(.+?[.!?])\s", cut[::-1])  # reverse search is messy; do forward instead
    # Simpler: find last sentence end in cut
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
    CRITICAL: pace the stream. Twilio media expects near-realtime audio.
    Sending a whole TTS blob as fast as possible often makes the call sound dead or drop.
    """
    idx = 0
    total = len(ulaw_audio)
    while idx < total:
        if cancel_event.is_set():
            return
        chunk = ulaw_audio[idx: idx + TWILIO_FRAME_BYTES]
        idx += TWILIO_FRAME_BYTES
        msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": _bytes_to_b64(chunk)}}
        await twilio_ws.send_text(json.dumps(msg))
        # pace at 20ms per frame
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
    logger.info(
        f"ElevenLabs OK: content-type={r.headers.get('content-type')} bytes={len(r.content)} first10={r.content[:10]}"
    )
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
# Topic inference (generic, light)
# -------------------------
_TOPIC_PREFIXES = [
    "tell me about ",
    "learn about ",
    "what is ",
    "what are ",
    "explain ",
    "help me understand ",
]

def _infer_topic(text: str) -> Optional[str]:
    if not text:
        return None
    low = text.lower().strip()
    for p in _TOPIC_PREFIXES:
        if low.startswith(p) and len(text) > len(p) + 2:
            return text[len(p):].strip(" .?!")
    return None

def _normalize_short(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^\w\s']", "", t).strip()
    return t

def _is_short_followup(text: str) -> bool:
    # One to three words like: "breeds", "history", "price", "steps", "habitat", "three"
    t = _normalize_short(text)
    if not t:
        return False
    return len(t.split()) <= 3

# -------------------------
# OpenAI helpers
# -------------------------
def _extract_openai_text(data: dict) -> str:
    # Best-case: Responses API provides output_text directly
    ot = data.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    # Next: scan output items for message content
    out = data.get("output")
    if isinstance(out, list):
        parts = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    # Last: some server tiers put text in "text"
    t = data.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    if isinstance(t, dict):
        for k in ("value", "text", "content"):
            v = t.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    return ""

def _output_sample_types(data: dict) -> list:
    out = data.get("output")
    types = []
    if isinstance(out, list):
        for item in out[:3]:
            if isinstance(item, dict):
                types.append(
                    {
                        "item_type": item.get("type"),
                        "content_types": [c.get("type") for c in (item.get("content") or []) if isinstance(c, dict)],
                    }
                )
    return types

async def _openai_post(payload: dict) -> Tuple[int, dict, str]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    if openai_client is None:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
    else:
        r = await openai_client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)

    status = r.status_code
    txt = r.text or ""
    data = {}
    try:
        data = r.json()
    except Exception:
        data = {}
    if status >= 400:
        # Raise after we capture details
        r.raise_for_status()
    return status, data, txt

async def openai_reply(user_text: str, state: Optional[Dict[str, Any]] = None) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI isn’t configured yet. Set OPENAI_API_KEY in Render."

    st = state or {}
    topic = st.get("topic") or "unknown"
    last_user = st.get("last_user") or ""
    last_bot = st.get("last_bot") or ""

    # If user says a short follow-up, nudge the model to treat it as a follow-up
    followup = _is_short_followup(user_text)

    system = (
        "You are Vozlia, a calm, confident AI voice assistant.\n"
        "PHONE STYLE:\n"
        "- Answer immediately with a helpful response (1–3 sentences).\n"
        "- If broad, give a quick overview + offer up to 3 options.\n"
        "- If the user gives a short follow-up (like 'breeds', 'history', 'steps', 'habitat', 'three'),\n"
        "  interpret it in the context of the CURRENT_TOPIC and continue.\n"
        "- Do NOT say: 'Tell me what you want to know' or 'What would you like to do next?'\n"
        "- Keep responses short enough for voice; avoid long essays.\n"
    )

    context = (
        f"CURRENT_TOPIC: {topic}\n"
        f"LAST_USER: {last_user}\n"
        f"LAST_ASSISTANT: {last_bot}\n"
        f"IS_SHORT_FOLLOWUP: {str(followup).lower()}\n"
    )

    payload: dict = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "system", "content": context},
            {"role": "user", "content": user_text},
        ],
        "max_output_tokens": 220,
    }

    # Optional reasoning. If it causes "reasoning-only" outputs, set OPENAI_REASONING_EFFORT="" in Render.
    if OPENAI_REASONING_EFFORT in ("low", "medium", "high"):
        payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}

    try:
        # First attempt
        _, data, _raw = await _openai_post(payload)
        text_out = _extract_openai_text(data)
        if text_out:
            return _trim_for_voice(text_out)

        logger.warning(f"OpenAI parse failed. output_sample_types={_output_sample_types(data)}")

        # Retry once: disable reasoning + force plain text
        payload2 = dict(payload)
        payload2.pop("reasoning", None)
        payload2["input"] = [
            {"role": "system", "content": system + "\nReturn plain text only."},
            {"role": "system", "content": context},
            {"role": "user", "content": user_text},
        ]
        _, data2, _raw2 = await _openai_post(payload2)
        text_out2 = _extract_openai_text(data2)
        if text_out2:
            return _trim_for_voice(text_out2)

        logger.warning(f"OpenAI parse failed again. output_sample_types={_output_sample_types(data2)}")
        return "I heard you — can you say that one more time in a single sentence?"

    except httpx.HTTPStatusError as e:
        logger.exception(f"OpenAI HTTP error: {e}")
        return "I had trouble reaching my brain service. Please try again."
    except Exception as e:
        logger.exception(f"OpenAI reply failed: {e}")
        return "I had trouble reaching my brain service. Please try again."

# -------------------------
# Router endpoint (topic-agnostic)
# -------------------------
@app.post("/assistant/route")
async def assistant_route(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    meta = body.get("meta") or {}

    state = (meta.get("state") if isinstance(meta, dict) else None) or {"topic": None, "last_user": "", "last_bot": ""}

    if not text:
        return {"spoken_reply": "I didn’t catch that. Can you say it again?", "intent": "general", "actions": [], "state": state}

    # Infer/update topic from explicit "tell me about X" style asks
    inferred = _infer_topic(text)
    if inferred:
        state["topic"] = inferred

    # Also allow topic updates if user says "switch topic to X" or "new topic: X"
    low = text.lower().strip()
    m = re.search(r"(?:new topic|switch topic|topic is|topic:)\s*(.+)$", low)
    if m:
        cand = m.group(1).strip(" .?!")
        if cand:
            state["topic"] = cand

    reply = await openai_reply(text, state=state)

    state["last_user"] = text
    state["last_bot"] = reply

    return {"spoken_reply": reply, "intent": "general", "actions": [], "state": state}

# -------------------------
# Twilio stream WebSocket
# -------------------------
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio media stream connected")

    stream_sid: Optional[str] = None

    dg_ws = None
    dg_task: Optional[asyncio.Task] = None

    speak_task: Optional[asyncio.Task] = None
    speak_cancel = asyncio.Event()
    speak_lock = asyncio.Lock()

    last_final_ts = 0.0
    last_final_text = ""

    assistant_speaking = False

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
        Serialize speech to avoid overlapping tasks and audio glitches.
        Also trims overly-long text to reduce call “dead air”.
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
                try:
                    ulaw = await elevenlabs_tts_ulaw_bytes(spoken)
                    await stream_ulaw_audio_to_twilio(websocket, stream_sid, ulaw, speak_cancel)
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

                # barge-in: if user speaks while assistant talking (interim is enough)
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

                # Optional: small acknowledgement BEFORE long thinking
                if ACK_BEFORE_THINKING:
                    try:
                        await speak("Okay.")
                    except Exception:
                        pass

                state = CALL_STATE.get(stream_sid) or {"topic": None, "last_user": "", "last_bot": ""}
                state["last_user"] = transcript
                CALL_STATE[stream_sid] = state

                reply = await call_fsm_router(transcript, meta={"stream_sid": stream_sid, "state": state})
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

                try:
                    await speak("I’m connected. How can I help you today?")
                except Exception:
                    logger.exception("Initial ElevenLabs greeting failed")
                continue

            if etype == "media":
                if dg_ws is not None:
                    payload_b64 = evt.get("media", {}).get("payload")
                    if payload_b64:
                        try:
                            audio_bytes = _b64_to_bytes(payload_b64)
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
        try:
            await cancel_speaking()
        except Exception:
            pass

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

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("Twilio media stream handler completed")
