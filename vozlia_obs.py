# vozlia_obs.py
import os
import json
import time
import csv
import io
from collections import deque
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, PlainTextResponse


OBS_ENABLED = os.getenv("OBS_ENABLED", "1") == "1"
OBS_LOG_JSON = os.getenv("OBS_LOG_JSON", "0") == "1"
OBS_MAX_EVENTS_PER_CALL = int(os.getenv("OBS_MAX_EVENTS_PER_CALL", "5000"))
OBS_BACKFILL = int(os.getenv("OBS_BACKFILL", "300"))


class ObsHub:
    def __init__(self, logger):
        self.logger = logger
        self.events = {}   # streamSid -> deque[dict]
        self.subs = {}     # streamSid -> set(WebSocket)
        self.started = {}  # streamSid -> started_ts (epoch seconds)

    def ensure(self, sid: str):
        if not sid:
            return
        if sid not in self.events:
            self.events[sid] = deque(maxlen=OBS_MAX_EVENTS_PER_CALL)
            self.subs[sid] = set()
            self.started[sid] = time.time()

    def latest_sid(self) -> Optional[str]:
        if not self.started:
            return None
        # newest by started_ts
        return max(self.started.items(), key=lambda kv: kv[1])[0]

    def list_calls(self):
        items = sorted(self.started.items(), key=lambda x: x[1], reverse=True)
        return [{"streamSid": sid, "started_ts": ts} for sid, ts in items]

    def last_events(self, sid: str, limit: int = 2000):
        if sid not in self.events:
            return []
        evs = list(self.events[sid])
        return evs if limit <= 0 else evs[-limit:]

    async def emit(self, sid: str, event: dict):
        if not OBS_ENABLED or not sid:
            return
        self.ensure(sid)
        ev = {"ts": time.time(), "streamSid": sid, **event}
        self.events[sid].append(ev)

        if OBS_LOG_JSON:
            try:
                self.logger.info("[OBS_JSON] " + json.dumps(ev, ensure_ascii=False))
            except Exception:
                pass

        dead = []
        for ws in list(self.subs.get(sid, set())):
            try:
                await ws.send_text(json.dumps(ev, ensure_ascii=False))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subs.get(sid, set()).discard(ws)

    def _csv_stream(self, sid: str):
        # Stream CSV without building a huge in-memory string
        fieldnames = [
            "ts", "t_rel_s", "streamSid", "type",
            "dt_ms", "bytes", "rms", "seq",
            "router_ms", "tts_ms", "speak_ms", "total_ms",
        ]
        started_ts = self.started.get(sid) or 0.0

        def gen():
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate(0)

            for ev in self.last_events(sid, limit=0):
                ts = float(ev.get("ts") or 0.0)
                row = {
                    "ts": ts,
                    "t_rel_s": (ts - started_ts) if started_ts else None,
                    "streamSid": ev.get("streamSid"),
                    "type": ev.get("type"),
                    "dt_ms": ev.get("dt_ms"),
                    "bytes": ev.get("bytes"),
                    "rms": ev.get("rms"),
                    "seq": ev.get("seq"),
                    "router_ms": ev.get("router_ms"),
                    "tts_ms": ev.get("tts_ms"),
                    "speak_ms": ev.get("speak_ms"),
                    "total_ms": ev.get("total_ms"),
                }
                writer.writerow(row)
                yield buf.getvalue()
                buf.seek(0)
                buf.truncate(0)

        return gen()

    def dashboard_html(self, sid: str) -> str:
        started_ts = float(self.started.get(sid) or 0.0)

        # IMPORTANT: NOT an f-string. Avoids breaking on JS/CSS braces and prevents 500s.
        html = """<!doctype html>
<html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Vozlia Obs __SID__</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
  body { font-family: system-ui; margin:16px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .card { border:1px solid #ddd; border-radius:12px; padding:12px; flex:1; min-width:340px; }
  #log { height:220px; overflow:auto; background:#0b0b0b; color:#d7d7d7; padding:10px; border-radius:10px;
         font-family: ui-monospace, Menlo, monospace; font-size:12px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#f2f2f2; margin-right:8px; }
  .actions a { margin-right:12px; }
</style>
</head><body>

<h2>Flow B Observability: <span class="pill">__SID__</span></h2>
<div class="actions">
  <a href="/obs/latest">Latest dashboard</a>
  <a href="/obs/latest.csv">Download latest CSV</a>
  <a href="/obs/csv/__SID__">Download this CSV</a>
  <a href="/obs/calls">List calls</a>
</div>

<div class="row">
  <div class="card"><h3>Inbound (Twilio → Render)</h3><div id="in_dt"></div><div id="in_bytes"></div><div id="in_rms"></div></div>
  <div class="card"><h3>Outbound (Render → Twilio)</h3><div id="out_dt"></div><div id="out_bytes"></div><div id="out_rms"></div></div>
</div>

<div class="row" style="margin-top:12px;">
  <div class="card"><h3>Stage timings</h3><div id="stages"></div></div>
  <div class="card"><h3>Event log</h3><div id="log"></div></div>
</div>

<script>
const sid = "__SID__";
const startedTs = __STARTED_TS__; // epoch seconds

const wsProto = (location.protocol === "https:") ? "wss" : "ws";
const wsUrl = `${wsProto}://${location.host}/ws/obs/${sid}`;

// --- chart config ---
const N = 1200;              // points kept per trace
const FLUSH_MS = 250;        // throttle plotting

// x-axis = seconds since call start
function relTs(ts) {
  if (!startedTs) return ts;
  return (ts - startedTs);
}

function initPlot(divId, title) {
  Plotly.newPlot(divId, [
    {x:[], y:[], type:"scatter", mode:"lines", line:{shape:"spline"}}
  ], {
    title,
    margin:{t:36,l:50,r:10,b:40},
    xaxis:{title:"seconds since call start"},
    yaxis:{automargin:true}
  }, {displayModeBar:false});
}

function initStages() {
  Plotly.newPlot("stages", [
    {x:[], y:[], type:"scatter", mode:"lines", name:"router_ms", line:{shape:"spline"}},
    {x:[], y:[], type:"scatter", mode:"lines", name:"tts_ms",    line:{shape:"spline"}},
    {x:[], y:[], type:"scatter", mode:"lines", name:"speak_ms",  line:{shape:"spline"}},
  ], {
    title:"Timings (ms)",
    margin:{t:36,l:50,r:10,b:40},
    xaxis:{title:"seconds since call start"},
    yaxis:{title:"ms", automargin:true}
  }, {displayModeBar:false});
}

initPlot("in_dt", "Inbound dt_ms");
initPlot("in_bytes", "Inbound bytes");
initPlot("in_rms", "Inbound rms");
initPlot("out_dt", "Outbound dt_ms");
initPlot("out_bytes", "Outbound bytes");
initPlot("out_rms", "Outbound rms");
initStages();

function logLine(o) {
  const el = document.getElementById("log");
  el.textContent += JSON.stringify(o) + "\\n";
  el.scrollTop = el.scrollHeight;
  if (el.textContent.length > 200000) el.textContent = el.textContent.slice(-160000);
}

// --- buffered updates for smoothness ---
let qInDt=[], qInBytes=[], qInRms=[];
let qOutDt=[], qOutBytes=[], qOutRms=[];
let qStage=[];

function flush() {
  if (qInDt.length) {
    Plotly.extendTraces("in_dt",    {x:[qInDt.map(p=>p[0])],    y:[qInDt.map(p=>p[1])]},    [0], N);
    Plotly.extendTraces("in_bytes", {x:[qInBytes.map(p=>p[0])], y:[qInBytes.map(p=>p[1])]}, [0], N);
    Plotly.extendTraces("in_rms",   {x:[qInRms.map(p=>p[0])],   y:[qInRms.map(p=>p[1])]},   [0], N);
    qInDt=[]; qInBytes=[]; qInRms=[];
  }
  if (qOutDt.length) {
    Plotly.extendTraces("out_dt",    {x:[qOutDt.map(p=>p[0])],    y:[qOutDt.map(p=>p[1])]},    [0], N);
    Plotly.extendTraces("out_bytes", {x:[qOutBytes.map(p=>p[0])], y:[qOutBytes.map(p=>p[1])]}, [0], N);
    Plotly.extendTraces("out_rms",   {x:[qOutRms.map(p=>p[0])],   y:[qOutRms.map(p=>p[1])]},   [0], N);
    qOutDt=[]; qOutBytes=[]; qOutRms=[];
  }
  if (qStage.length) {
    const xs = qStage.map(p=>p[0]);
    Plotly.extendTraces("stages", {
      x:[xs, xs, xs],
      y:[ qStage.map(p=>p[1]), qStage.map(p=>p[2]), qStage.map(p=>p[3]) ]
    }, [0,1,2], N);
    qStage=[];
  }
}
setInterval(flush, FLUSH_MS);

const ws = new WebSocket(wsUrl);
ws.onopen  = () => logLine({info:"ws connected", wsUrl});
ws.onclose = () => logLine({info:"ws closed"});
ws.onerror = (e) => logLine({error:"ws error", e:String(e)});

ws.onmessage = (m) => {
  let ev=null; try { ev = JSON.parse(m.data); } catch { return; }
  logLine(ev);

  const x = relTs(ev.ts);

  if (ev.type === "audio_in") {
    qInDt.push([x, ev.dt_ms]);
    qInBytes.push([x, ev.bytes]);
    qInRms.push([x, ev.rms]);
  }
  if (ev.type === "audio_out") {
    qOutDt.push([x, ev.dt_ms]);
    qOutBytes.push([x, ev.bytes]);
    qOutRms.push([x, ev.rms]);
  }
  if (ev.type === "stage") {
    qStage.push([x, ev.router_ms ?? null, ev.tts_ms ?? null, ev.speak_ms ?? null]);
  }
};
</script>

</body></html>
"""

        # safe placeholder replacement
        html = html.replace("__SID__", sid)
        html = html.replace("__STARTED_TS__", str(started_ts))
        return html


async def ws_handler(hub: ObsHub, websocket: WebSocket, sid: str):
    await websocket.accept()
    hub.ensure(sid)
    hub.subs[sid].add(websocket)

    # backfill
    try:
        for ev in hub.last_events(sid, limit=OBS_BACKFILL):
            await websocket.send_text(json.dumps(ev, ensure_ascii=False))
    except Exception:
        pass

    try:
        while True:
            # keepalive from browser if any
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.subs.get(sid, set()).discard(websocket)


def mount_routes(app, hub: ObsHub):
    @app.get("/obs")
    def _obs_root():
        sid = hub.latest_sid()
        if not sid:
            return PlainTextResponse("No calls yet.", status_code=200)
        return RedirectResponse(url=f"/obs/call/{sid}", status_code=302)

    @app.get("/obs/latest")
    def _obs_latest():
        sid = hub.latest_sid()
        if not sid:
            return PlainTextResponse("No calls yet.", status_code=200)
        return HTMLResponse(hub.dashboard_html(sid))

    @app.get("/obs/latest.csv")
    def _obs_latest_csv():
        sid = hub.latest_sid()
        if not sid:
            return PlainTextResponse("No calls yet.", status_code=200)
        return StreamingResponse(
            hub._csv_stream(sid),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="vozlia_obs_latest_{sid}.csv"'},
        )

    @app.get("/obs/csv/{sid}")
    def _obs_csv(sid: str):
        if sid not in hub.started:
            return PlainTextResponse("Unknown streamSid.", status_code=404)
        return StreamingResponse(
            hub._csv_stream(sid),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="vozlia_obs_{sid}.csv"'},
        )

    @app.get("/obs/calls")
    def _calls():
        return {"ok": True, "active_calls": hub.list_calls(), "count": len(hub.started)}

    @app.get("/obs/events/{sid}")
    def _events(sid: str, limit: int = 2000):
        return {"ok": True, "streamSid": sid, "events": hub.last_events(sid, limit=limit)}

    @app.get("/obs/call/{sid}")
    def _dash(sid: str):
        if sid not in hub.started:
            return PlainTextResponse("Unknown streamSid.", status_code=404)
        return HTMLResponse(hub.dashboard_html(sid))

    @app.websocket("/ws/obs/{sid}")
    async def _ws(websocket: WebSocket, sid: str):
        await ws_handler(hub, websocket, sid)
