"""Model adapters, persistent cache and request budget (OpenRouter / Ollama / mock)."""
import os, json, time, hashlib, asyncio, random
import httpx
from dotenv import load_dotenv
from cube import MOVES, DESC, DIST, apply

load_dotenv()
KEY = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Instructor limits are the defaults (65 total; Claude 5, others 20). Override in .env only if you were allowed more.
CACHE_FILE, TOTAL_CAP = "cache.json", int(os.getenv("TOTAL_CAP", "65"))

# slot: (label, kind, OpenRouter id, default Ollama model, max live OpenRouter calls)
SLOTS = {
    "jev":  ("Jev 1.13", "decision", "typesafe/jev-1.13", "gemma3", 20),
    "luna": ("GPT-6 Luna Decisions", "decision", "openai/gpt-6-luna-decisions", "qwen3", 20),
    "clef": ("Cloudflare Clef", "decision", "cloudflare/clef", "clef", 20),
    "opus": ("Claude Opus 5.5", "llm", "anthropic/claude-opus-5.5", "deepseek-r1", 5),
}
for _s in SLOTS:  # optional per-model override, e.g. MAX_CALLS_OPUS=10
    SLOTS[_s] = SLOTS[_s][:4] + (int(os.getenv(f"MAX_CALLS_{_s.upper()}", SLOTS[_s][4])),)
LAYOUT = "54 facelets, faces in order U,R,F,D,L,B, 9 per face, row-major (Kociemba)"
INSTR = "Choose the single next move most likely to progress toward the solved cube."
SCHEMA = {"type": "object", "properties": {"move": {"type": "string", "enum": MOVES}},
          "required": ["move"], "additionalProperties": False}

_lock = asyncio.Lock()
DB = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {"responses": {}, "calls": {}}
def _save(): json.dump(DB, open(CACHE_FILE, "w"))

async def _reserve(slot):
    """Count a real OpenRouter request BEFORE sending it (failures/retries count too). Never reset."""
    async with _lock:
        c = DB["calls"]
        if c.get("total", 0) >= TOTAL_CAP or c.get(slot, 0) >= SLOTS[slot][4]:
            return False
        c["total"] = c.get("total", 0) + 1; c[slot] = c.get(slot, 0) + 1; _save()
        return True

def _body(kind, mode, model, state, recent):
    st = {"layout": LAYOUT, "cube_facelets": state, "recent_moves": recent[-5:]}
    if mode == "openrouter" and kind == "decision":
        return {"model": model, "state": st, "questions": {"next_move": {
            "type": "choice", "instructions": INSTR, "criteria": DESC}}}
    msg = [{"role": "user", "content": f"{INSTR}\nState: {json.dumps(st)}\nAllowed moves: {json.dumps(DESC)}\n"
            'Answer with JSON only: {"move": "<one allowed move>"}'}]
    if mode == "ollama":
        return {"model": model, "stream": False, "format": SCHEMA, "messages": msg}
    return {"model": model, "max_tokens": 4000, "usage": {"include": True}, "messages": msg,
            "response_format": {"type": "json_schema", "json_schema": {"name": "move", "strict": True, "schema": SCHEMA}}}

def _parse(kind, mode, raw, out):
    try:
        if mode == "mock":
            out["move"] = raw["move"]
        elif mode == "openrouter" and kind == "decision":
            a = raw["answers"]["next_move"]
            out["move"], out["conf"] = a["choice"], a.get("confidence")
            out["cost"] = raw.get("usage", {}).get("cost", 0) or 0
        else:
            txt = raw["message"]["content"] if mode == "ollama" else raw["choices"][0]["message"]["content"]
            out["move"] = json.loads(txt)["move"]
            out["cost"] = (raw.get("usage", {}).get("cost", 0) or 0) if mode == "openrouter" else 0
    except Exception as e:
        out["error"] = f"malformed response: {e}"
    if out["move"] not in MOVES:           # log and count; never silently convert
        out["error"] = out["error"] or f"invalid move: {out['move']!r}"
        out["move"] = None
    if out["cached"]:
        out["cost"] = 0

async def _mock(slot, state):
    p, lat = {"jev": (.7, .05), "luna": (.85, .1), "clef": (.5, .03), "opus": (.95, 1.5)}[slot]
    await asyncio.sleep(lat)
    best = [m for m in MOVES if DIST.get(apply(state, m), 99) < DIST.get(state, 99)]
    return {"move": random.choice(best) if best and random.random() < p else random.choice(MOVES)}

async def ask(slot, mode, state, recent):
    """One model decision. No retries (same policy for all models). Returns move, latency, cached, cost, conf, error."""
    _, kind, orid, olname, _ = SLOTS[slot]
    model = orid if mode != "ollama" else os.getenv(f"OLLAMA_{slot.upper()}", olname)
    body = _body(kind, mode, model, state, recent)
    key = hashlib.sha256(json.dumps([mode, slot, body], sort_keys=True).encode()).hexdigest()
    out = {"move": None, "conf": None, "cost": 0.0, "cached": False, "error": None, "latency": 0.0}
    t0 = time.perf_counter()
    if mode != "mock" and key in DB["responses"]:
        raw, out["cached"] = DB["responses"][key], True
    elif mode == "mock":
        raw = await _mock(slot, state)
    else:
        if mode == "openrouter" and not await _reserve(slot):
            out["error"] = "call cap reached"; return out
        url = (("https://openrouter.ai/api/alpha/decisions" if kind == "decision"
                else "https://openrouter.ai/api/v1/chat/completions") if mode == "openrouter" else OLLAMA + "/api/chat")
        hdr = {"Authorization": f"Bearer {KEY}"} if mode == "openrouter" else {}
        _msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [LIVE {mode}] {slot} model={model} -> {url} calls_so_far={DB['calls']}"
        print(_msg, flush=True); open("requests.log", "a").write(_msg + "\n")
        try:
            async with httpx.AsyncClient(timeout=120 if kind == "llm" else 60) as c:
                r = await c.post(url, headers=hdr, json=body)
            r.raise_for_status(); raw = r.json()
        except Exception as e:
            body_txt = getattr(getattr(e, "response", None), "text", "")[:300]
            out["error"] = f"request failed: {e}"; out["latency"] = time.perf_counter() - t0
            open("requests.log", "a").write(f"ERROR {slot}: {e} | BODY: {body_txt}\n"); return out
        async with _lock:
            DB["responses"][key] = raw; _save()
    out["latency"] = time.perf_counter() - t0
    _parse(kind, mode, raw, out)
    if out["error"] and not out["cached"] and mode != "mock":
        async with _lock:
            DB["responses"].pop(key, None); _save()
    return out
