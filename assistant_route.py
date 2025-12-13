# assistant_route.py
import os, re
from typing import Optional, Dict, Any, Tuple

import httpx
from fastapi import APIRouter, Request

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low").strip().lower()
MAX_SPOKEN_CHARS = int(os.getenv("MAX_SPOKEN_CHARS", "700"))

_TOPIC_PREFIXES = [
    "tell me about ",
    "learn about ",
    "what is ",
    "what are ",
    "explain ",
    "help me understand ",
]

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
    return bool(t) and len(t.split()) <= 3

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

    return ""

async def _openai_post(payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

async def openai_reply(user_text: str, state: Dict[str, Any]) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI isn’t configured yet."

    topic = state.get("topic") or "unknown"
    last_user = state.get("last_user") or ""
    last_bot = state.get("last_bot") or ""
    followup = _is_short_followup(user_text)

    system = (
        "You are Vozlia, a calm, confident AI voice assistant.\n"
        "PHONE STYLE:\n"
        "- Answer immediately with a helpful response (1–3 sentences).\n"
        "- If broad, give a quick overview + offer up to 3 options.\n"
        "- If the user gives a short follow-up (like 'breeds', 'history', 'steps', 'habitat', 'three'),\n"
        "  interpret it in the context of the CURRENT_TOPIC and continue.\n"
        "- Keep responses short enough for voice.\n"
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

    data = await _openai_post(payload)
    txt = _extract_openai_text(data)
    if txt:
        return _trim_for_voice(txt)

    # Retry once without reasoning
    payload.pop("reasoning", None)
    data2 = await _openai_post(payload)
    txt2 = _extract_openai_text(data2)
    if txt2:
        return _trim_for_voice(txt2)

    return "I heard you — can you say that one more time in a single sentence?"

@router.post("/assistant/route")
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
