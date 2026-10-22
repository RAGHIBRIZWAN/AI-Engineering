import asyncio, json, os, time
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from cube import SOLVED, apply, scramble
import llm

MAX_MOVES, OPT, RES = 20, 3, "results.json"
app, RUN, _TASKS = FastAPI(), {"status": "idle", "models": {}}, []
avg = lambda x: sum(x) / len(x) if x else None

class RunReq(BaseModel):
    mode: str = "mock"      # mock | ollama | openrouter
    seed: int = 1

def _save_result(rec):
    rows = json.load(open(RES)) if os.path.exists(RES) else []
    if rec["calls"] == 0 and any(r["mode"] == rec["mode"] and r["seed"] == rec["seed"] and r["slot"] == rec["slot"] for r in rows):
        return  # pure cache replay of an already-recorded run: keep the original live record
    rows.append(rec); json.dump(rows, open(RES, "w"), indent=1)

async def play(slot, start, mode, seed):
    m = RUN["models"][slot]
    cap = min(MAX_MOVES, llm.SLOTS[slot][4]) if mode == "openrouter" else MAX_MOVES  # Opus: 5 calls max
    m["t0"], state = time.time(), start
    while state != SOLVED and m["steps"] < cap:
        m["status"] = "thinking"
        r = await llm.ask(slot, mode, state, m["moves"])
        if r["error"] == "call cap reached":
            m["reason"] = "call cap reached"; break
        m["steps"] += 1; m["calls"] += not r["cached"]; m["hits"] += r["cached"]; m["cost"] += r["cost"]
        m["lat_cached" if r["cached"] else "lat_live"].append(r["latency"])
        if r["conf"] is not None: m["conf"].append(r["conf"])
        if r["move"]:
            state = apply(state, r["move"]); m["moves"].append(r["move"]); m["state"] = state; m["bad"] = 0
        else:
            m["invalid"] += 1; m["errors"].append(r["error"]); m["bad"] = m.get("bad", 0) + 1
            if mode == "openrouter" and m["bad"] >= 2:
                m["reason"] = "stopped after 2 errors in a row: " + str(r["error"])[:80]; break
    solved = state == SOLVED
    m.update(status="solved" if solved else "unsolved", end=time.time(),
             reason="solved" if solved else (m["reason"] or "ran out of moves"))
    _save_result({"mode": mode, "seed": seed, "slot": slot, "solved": solved, "moves": len(m["moves"]),
                  "gap": len(m["moves"]) - OPT if solved else None, "elapsed": m["end"] - m["t0"],
                  "lat_live": m["lat_live"], "lat_cached": m["lat_cached"], "calls": m["calls"],
                  "hits": m["hits"], "cost": m["cost"], "invalid": m["invalid"], "avg_conf": avg(m["conf"])})

async def runall(start, mode, seed):
    if mode == "ollama":  # one model at a time, so they don't slow each other down
        for s in llm.SLOTS:
            await asyncio.gather(play(s, start, mode, seed), return_exceptions=True)
    else:
        await asyncio.gather(*(play(s, start, mode, seed) for s in llm.SLOTS), return_exceptions=True)
    RUN["status"] = "done"

@app.post("/api/run")
async def run(req: RunReq):
    if RUN["status"] == "running": raise HTTPException(409, "A run is already in progress")
    if req.mode not in ("mock", "ollama", "openrouter"): raise HTTPException(400, "Unknown mode")
    if req.mode == "openrouter" and not llm.KEY: raise HTTPException(400, "OPENROUTER_API_KEY is missing in .env")
    start = scramble(req.seed)
    RUN.update(status="running", mode=req.mode, seed=req.seed, start=start, models={
        s: dict(label=v[0], kind=v[1], state=start, moves=[], steps=0, calls=0, hits=0, cost=0.0, invalid=0,
                errors=[], lat_live=[], lat_cached=[], conf=[], status="waiting", reason="", t0=None, end=None)
        for s, v in llm.SLOTS.items()})
    _TASKS.append(asyncio.create_task(runall(start, req.mode, req.seed)))
    return {"ok": True}

@app.get("/api/state")
def state(): return RUN

@app.get("/api/budget")
def budget(): return {"calls": llm.DB["calls"], "total_cap": llm.TOTAL_CAP, "caps": {s: v[4] for s, v in llm.SLOTS.items()}}

@app.get("/api/config")
def config(): return {"name": os.getenv("STUDENT_NAME", "YOUR NAME"), "roll": os.getenv("STUDENT_ROLL", "YOUR ROLL NO")}

@app.get("/api/stats")
def stats(mode: str = "mock"):
    rows = json.load(open(RES)) if os.path.exists(RES) else []
    out = []
    for slot, v in llm.SLOTS.items():
        rs = [r for r in rows if r["mode"] == mode and r["slot"] == slot]
        if not rs: continue
        sv = [r for r in rs if r["solved"]]
        out.append({"label": v[0], "runs": len(rs), "success": 100 * len(sv) / len(rs),
                    "lat_live": avg([x for r in rs for x in r["lat_live"]]),
                    "lat_cached": avg([x for r in rs for x in r["lat_cached"]]),
                    "total_time": avg([r["elapsed"] for r in rs]), "moves": avg([r["moves"] for r in rs]),
                    "gap": avg([r["gap"] for r in sv]), "calls": sum(r["calls"] for r in rs),
                    "hits": sum(r["hits"] for r in rs), "cost": sum(r["cost"] for r in rs),
                    "invalid": sum(r["invalid"] for r in rs)})
    return out

app.mount("/", StaticFiles(directory="static", html=True))
