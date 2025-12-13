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
MAX_SPOKEN_CHARS = int(os.getenv("MAX_SPOKEN_CHARS", "700"))
ACK_BEFORE_THINKING = os.getenv("ACK_BEFORE_THINKING", "0") == "1"

logger.info(f"PUBLIC_BASE_URL={PUBLIC_BASE_URL}")
logger.info(f"FSM_ROUTER_URL={FSM_ROUTER_URL}")

# -------------------------
# Audio probe logging (toggle)
# -------------------------
AUDIO_DEBUG = os.getenv("AUDIO_DEBUG", "0") == "1"
AUDIO_DEBUG_SAMPLE_EVERY_N = int(os.getenv("AUDIO_DEBUG_SAMPLE_EVERY_N", "25"))

# µ-law decode table (fast-ish) for basic energy checks
# Source: standard G.711 µ-law expansion
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
    # Returns RMS in PCM-ish units (not dBFS), just for “is it alive / clipping / garbage?”
    if not ulaw_bytes:
        return 0.0
    acc = 0.0
    for b in ulaw_bytes:
        s = MU_LAW_DECODE_TABLE[b]
        acc += s * s
    return math.sqrt(acc / len(ulaw_bytes))


@dataclass
class StreamProbe:
    stream_sid: str
    dir: str  # "in" or "out"
    frame_count: int = 0
    bytes_total: int = 0
    last_ts: float = 0.0
    last_log_frame: int = 0

    def log_frame(self, logger, raw_bytes: bytes, meta: dict | None = None):
        if not AUDIO_DEBUG:
            return
        self.frame_count += 1
        self.bytes_total += len(raw_bytes)

        now = time.time()
        dt_ms = (now - self.last_ts) * 1000.0 if self.last_ts else 0.0
        self.last_ts = now

        # sample logs every N frames
        if (self.frame_count - self.last_log_frame) < AUDIO_DEBUG_SAMPLE_EVERY_N:
            return
        self.last_log_frame = self.frame_count

        rms = ulaw_rms(raw_bytes)
        # Typical Twilio 20ms @ 8kHz µ-law = 160 bytes per frame
        logger.info(
            "[AUDIO_PROBE] %s streamSid=%s frame=%d bytes=%d dt_ms=%.1f rms=%.1f meta=%s",
            self.dir, self.stream_sid, self.frame_count, len(raw_bytes), dt_ms, rms,
            json.dumps(meta or {}, ensure_ascii=False),
        )


def sniff_audio_header(b: bytes) -> str:
    if len(b) >= 4 and b[:4] == b"RIFF":
        return "wav/riff"
    if len(b) >= 3 and b[:3] == b"ID3":
        return "mp3/id3"
    if len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0:
        return "mp3/frame"
    return "unknown/raw"


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
    return {"ok": True, "service": "vozlia-dg", "pipeline": "twilio->deepgram->router->openai->elevenlabs->twilio"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# -------------------------
# Twilio inbound (TwiML)
# -------------------------
@app.post("/twilio/inbound-dg")
async def twilio_inbound(request: Request):
    base = str(request.base_url).rstrip("/")  # e.g. https://vozlia-dg.onrender.com
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
    out_probe: Optional[StreamProbe] = None,
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

        if out_probe:
            out_probe.log_frame(logger, chunk, meta={"stage": "to_twilio"})

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

    ctype = r.headers.get("content-type", "")
    head = sniff_audio_header(r.content[:16])
    logger.info(
        f"ElevenLabs OK: content-type={ctype} bytes={len(r.content)} header={head} first10={r.content[:10]}"
    )

    # If we ever see mp3/wav headers here, something is wrong and will sound like static to Twilio.
    if head.startswith("mp3") or head.startswith("wav"):
        logger.warning(
            f"[AUDIO_WARN] ElevenLabs returned {head} but Twilio expects ulaw_8000. This can cause static."
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
    t = _normalize_short(text)
    if not t:
        return False
    return len(t.split()) <= 3


# -------------------------
# OpenAI helpers
# -------------------------
def _extract_openai_text(data: dict) -> str:
    ot = data.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

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
        r.raise_for_status()
    return status, data, txt


async def openai_reply(user_text: str, state: Optional[Dict[str, Any]] = None) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI isn’t configured yet. Set OPENAI_API_KEY in Render."

    st = state or {}
    topic = st.get("topic") or "unknown"
    last_user = st.get("last_user") or ""
    last_bot = st.get("last_bot") or ""

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

    if OPENAI_REASONING_EFFORT in ("low", "medium", "high"):
        payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}

    try:
        _, data, _raw = await _openai_post(payload)
        text_out = _extract_openai_text(data)
        if text_out:
            return _trim_for_voice(text_out)

        logger.warning(f"OpenAI parse failed. output_sample_types={_output_sample_types(data)}")

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

    inferred = _infer_topic(text)
    if inferred:
        state["topic"] = inferred

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

    # Audio probes (created once we have streamSid)
    in_probe: Optional[StreamProbe] = None
    out_probe: Optional[StreamProbe] = None

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

                    # One-time sanity log: does the first outbound chunk look like ulaw frames?
                    if AUDIO_DEBUG and ulaw:
                        logger.info(
                            "[AUDIO_PROBE] out_blob streamSid=%s bytes=%d first_chunk_type=%s",
                            stream_sid, len(ulaw), sniff_audio_header(ulaw[:16]),
                        )

                    await stre
