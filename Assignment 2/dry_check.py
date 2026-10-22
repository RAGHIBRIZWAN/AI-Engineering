import json, llm
from cube import scramble
s = scramble(1)
for slot, v in llm.SLOTS.items():
    body = llm._body(v[1], "openrouter", v[2], s, [])
    print("=====", slot, "|", v[2], "|", "decisions" if v[1] == "decision" else "chat")
    txt = json.dumps(body)
    print(txt[:700] + ("..." if len(txt) > 700 else ""))
    print("top-level keys:", list(body.keys()))
    print("criteria count:", len(body.get("questions", {}).get("next_move", {}).get("criteria", {})))
print("calls used so far:", llm.DB["calls"])
