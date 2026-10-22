import json, sys
from statistics import mean, median

mode = sys.argv[1] if len(sys.argv) > 1 else "openrouter"
rows = [r for r in json.load(open("results.json")) if r["mode"] == mode]
if not rows:
    sys.exit(f"No results for mode '{mode}'")
LABEL = {"jev": "Jev 1.13", "luna": "GPT-6 Luna Decisions", "clef": "Cloudflare Clef", "opus": "Claude Opus 5.5"}
f = lambda x, d=2: "-" if x is None else f"{x:.{d}f}"
st = {}
for s in LABEL:
    rs = [r for r in rows if r["slot"] == s]
    if not rs:
        continue
    live = [x for r in rs for x in r["lat_live"]]
    cac = [x for r in rs for x in r["lat_cached"]]
    sv = [r for r in rs if r["solved"]]
    cf = [r["avg_conf"] for r in rs if r["avg_conf"] is not None]
    st[s] = dict(n=len(rs), succ=100 * len(sv) / len(rs), lat=mean(live) if live else None,
                 med=median(live) if live else None, cac=mean(cac) if cac else None,
                 time=mean(r["elapsed"] for r in rs), moves=mean(r["moves"] for r in rs),
                 gap=mean(r["gap"] for r in sv) if sv else None, calls=sum(r["calls"] for r in rs),
                 hits=sum(r["hits"] for r in rs), bad=sum(r["invalid"] for r in rs),
                 cost=sum(r["cost"] for r in rs), conf=mean(cf) if cf else None)

out = [f"## Results ({mode} mode)", f"Puzzles: {sorted({r['seed'] for r in rows})} (seeds), each exactly 3 moves from solved, optimum = 3, move limit = 20.", "",
       "| Model | Runs | Success % | Avg latency/move (s) | Median latency (s) | Avg cached (s) | Avg total time (s) | Avg moves | Optimal gap | API calls | Cache hits | Invalid | Cost ($) | Avg confidence |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s, v in st.items():
    out.append(f"| {LABEL[s]} | {v['n']} | {f(v['succ'], 0)} | {f(v['lat'], 3)} | {f(v['med'], 3)} | {f(v['cac'], 4)} | {f(v['time'])} | {f(v['moves'], 1)} | {f(v['gap'], 1)} | {v['calls']} | {v['hits']} | {v['bad']} | {v['cost']:.5f} | {f(v['conf'], 2)} |")

def best(key, low=True):
    c = {s: v[key] for s, v in st.items() if v[key] is not None}
    if not c:
        return "n/a"
    t = min(c.values()) if low else max(c.values())
    return ", ".join(LABEL[s] for s, x in c.items() if x == t) + f" ({f(t, 3)})"

out += ["", "### Data for the five questions",
        f"1. Fastest decisions (lowest avg live latency): {best('lat')}",
        f"2. Highest solve rate: {best('succ', False)} %",
        f"3. Shortest solutions (lowest optimal gap, 0 = optimal): {best('gap')}",
        f"4. Fastest model: {best('lat')} | Best quality: {best('gap')}. Compare the two lines to answer.",
        "5. Fast model vs frontier LLM vs hybrid: argue from the table above (latency, success, gap, cost)."]
text = "\n".join(out)
open(f"report_{mode}.md", "w", encoding="utf-8").write(text)
print(text)
