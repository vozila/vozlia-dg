# vozlia_obs.py
import os, json, time
from collections import deque
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

OBS_ENABLED = os.getenv("OBS_ENABLED", "1") == "1"
OBS_LOG_JSON = os.getenv("OBS_LOG_JSON", "0") == "1"
OBS_MAX_EVENTS_PER_CALL = int(os.getenv("OBS_MAX_EVENTS_PER_CALL", "5000"))
OBS_BACKFILL = int(os.getenv("OBS_BACKFILL", "300"))

class ObsHub:
    def __init__(self, logger):
        self.logger = logger
        self.events = {}   # streamSid -> deque
        self.subs = {}     # streamSid -> set(WebSocket)
        self.started = {}  # streamSid -> started_ts

    def ensure(self, sid: str):
        if not sid:
            return
        if sid not in self.events:
            self.events[sid] = deque(maxlen=OBS_MAX_EVENTS_PER_CALL)
            self.subs[sid] = set()
            self.started[sid] = time.time()

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

    def dashboard_html(self, sid: str) -> str:
        return f"""<!doctype html>
<html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Vozlia Obs {sid}</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
 body {{ font-family: system-ui; margin:16px; }}
 .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
 .card {{ border:1px solid #ddd; border-radius:12px; padding:12px; flex:1; min-width:340px; }}
 #log {{ height:220px; overflow:auto; background:#0b0b0b; color:#d7d7d7; padding:10px; border-radius:10px;
        font-family: ui-monospace, Menlo, monospace; font-size:12px; }}
 .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#f2f2f2; margin-right:8px; }}
</style>
</head><body>
<h2>Flow B Observability: <span class="pill">{sid}</span></h2>

<div class="row">
  <div class="card"><h3>Inbound (Twilio → Render)</h3><div id="in_dt"></div><div id="in_bytes"></div><div id="in_rms"></div></div>
  <div class="card"><h3>Outbound (Render → Twilio)</h3><div id="out_dt"></div><div id="out_bytes"></div><div id="out_rms"></div></div>
</div>

<div class="row" style="margin-top:12px;">
  <div class="card"><h3>Stage timings</h3><div id="stages"></div></div>
  <div class="card"><h3>Event log</h3><div id="log"></div></div>
</div>

<script>
const sid = "{sid}";
const wsProto = (location.protocol === "https:") ? "wss" : "ws";
const wsUrl = `${{wsProto}}://${{location.host}}/ws/obs/${{sid}}`;

const N=600;
function S(){{return {{x:[],y:[]}}}}
const inDt=S(), inBytes=S(), inRms=S();
const outDt=S(), outBytes=S(), outRms=S();
const stage={{x:[], router:[], tts:[], speak:[]}};
function push(s,x,y){{s.x.push(x); s.y.push(y); if(s.x.length>N){{s.x.shift(); s.y.shift();}}}}
function pushStage(x,r,t,s){{stage.x.push(x); stage.router.push(r??null); stage.tts.push(t??null); stage.speak.push(s??null);
  if(stage.x.length>N){{stage.x.shift(); stage.router.shift(); stage.tts.shift(); stage.speak.shift();}}}}
function plot(div,title,s){{Plotly.react(div,[{{x:s.x,y:s.y,type:"scatter",mode:"lines"}}],
  {{title,margin:{{t:36,l:40,r:10,b:30}},xaxis:{{title:"ts"}},yaxis:{{automargin:true}}}},{{displayModeBar:false}});}}
function plotStages(){{
  Plotly.react("stages",[
    {{x:stage.x,y:stage.router,type:"scatter",mode:"lines",name:"router_ms"}},
    {{x:stage.x,y:stage.tts,type:"scatter",mode:"lines",name:"tts_ms"}},
    {{x:stage.x,y:stage.speak,type:"scatter",mode:"lines",name:"speak_ms"}},
  ],{{title:"Timings (ms)",margin:{{t:36,l:40,r:10,b:30}},xaxis:{{title:"ts"}},yaxis:{{title:"ms",automargin:true}}}},
  {{displayModeBar:false}});
}}
function logLine(o){{const el=document.getElementById("log"); el.textContent+=JSON.stringify(o)+"\\n"; el.scrollTop=el.scrollHeight;
  if(el.textContent.length>200000) el.textContent=el.textContent.slice(-160000);
}}

plot("in_dt","in dt_ms",inDt); plot("in_bytes","in bytes",inBytes); plot("in_rms","in rms",inRms);
plot("out_dt","out dt_ms",outDt); plot("out_bytes","out bytes",outBytes); plot("out_rms","out rms",outRms);
plotStages();

const ws=new WebSocket(wsUrl);
ws.onopen=()=>logLine({{info:"ws connected", wsUrl}});
ws.onclose=()=>logLine({{info:"ws closed"}});
ws.onerror=(e)=>logLine({{error:"ws error", e:String(e)}});

ws.onmessage=(m)=>{{
  let ev=null; try{{ev=JSON.parse(m.data)}}catch{{return}}
  logLine(ev);
  const ts=ev.ts;
  if(ev.type==="audio_in"){{push(inDt,ts,ev.dt_ms); push(inBytes,ts,ev.bytes); push(inRms,ts,ev.rms);
    plot("in_dt","in dt_ms",inDt); plot("in_bytes","in bytes",inBytes); plot("in_rms","in rms",inRms);}}
  if(ev.type==="audio_out"){{push(outDt,ts,ev.dt_ms); push(outBytes,ts,ev.bytes); push(outRms,ts,ev.rms);
    plot("out_dt","out dt_ms",outDt); plot("out_bytes","out bytes",outBytes); plot("out_rms","out rms",outRms);}}
  if(ev.type==="stage"){{pushStage(ts,ev.router_ms,ev.tts_ms,ev.speak_ms); plotStages();}}
}};
</script>
</body></html>"""

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
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        hub.subs.get(sid, set()).discard(websocket)

def mount_routes(app, hub: ObsHub):
    @app.get("/obs/calls")
    def _calls():
        return {"ok": True, "active_calls": hub.list_calls(), "count": len(hub.started)}

    @app.get("/obs/events/{sid}")
    def _events(sid: str, limit: int = 2000):
        return {"ok": True, "streamSid": sid, "events": hub.last_events(sid, limit=limit)}

    @app.get("/obs/call/{sid}")
    def _dash(sid: str):
        return HTMLResponse(hub.dashboard_html(sid))

    @app.websocket("/ws/obs/{sid}")
    async def _ws(websocket: WebSocket, sid: str):
        await ws_handler(hub, websocket, sid)
