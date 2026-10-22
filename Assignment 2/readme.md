# Cube Arena: fast decision models vs a frontier LLM

**Student:** Raghib Rizwan Rabani  |  **Roll No:** 23k-0012  |  **Course:** AI Engineering, Assignment 02

## Demo recording

**[Watch the screen recording: Assignment 2 AI.mp4](Assignment%202%20AI.mp4)**

The recording shows the real OpenRouter run on seed 50 in the Rubik's Cube Race interface, with my name and roll number pinned at the top of the screen throughout. The four models race on the same cube. At the end all four are **NOT SOLVED**:

| Model | Final state shown on screen |
|---|---|
| Jev 1.13 | RAN OUT OF MOVES: 20 moves, 0:20.93 |
| GPT-6 Luna Decisions | CALL CAP REACHED: 0 moves (its request limit was already used up) |
| Cloudflare Clef | RAN OUT OF MOVES: 20 moves, 0:45.60 |
| Claude Opus 5.5 | CALL CAP REACHED: 5 moves, 1:25.62 |

Only solved runs get a rank medal in the interface, so none of the four has a medal in this run. The same numbers appear in the tables below.

---

## What was built

An AI-powered Rubik's Cube game and benchmark. Every model gets the **same** starting cube, the **same** state/action contract and the same move definitions, and picks one move per turn until the cube is solved or the move limit is reached.

- **Cube engine** (`cube.py`): 54-facelet state, 18 moves (U, R, F, D, L, B with `'` and `2`), solved check. Facelet permutations are generated from 3D geometry. `python test_cube.py` verifies it (e.g. `(R U R' U')` x 6 returns to solved).
- **Exact optimum by BFS** (`cube.py`): breadth-first search from the solved cube gives the exact shortest distance. Every start state is exactly **3 moves** from solved (BFS-verified), so the optimal solution length is 3.
- **Model adapters** (`llm.py`): Decisions API (`POST /api/alpha/decisions`) for Jev, GPT-6 Luna Decisions and Cloudflare Clef, with one typed `choice` question over the 18 moves (same wording for all three); chat completions with constrained JSON output for Claude Opus 5.5. Responses are validated; malformed or invalid output is counted, never silently converted into a move.
- **Backend and game UI** (`main.py`, `static/index.html`): FastAPI backend and an HTML/JS interface titled "Rubik's Cube Race". Four cards in a 2x2 grid, one per model, each with a real 3D cube drawn from the current 54-facelet state, a timer, a move counter, a status line (thinking / deciding, SOLVED, NOT SOLVED), a rank medal for solved runs, and a result strip (for example "RAN OUT OF MOVES" or "CALL CAP REACHED"). Below the cards: a **live table** computed from the race on screen (status, rank, moves, optimal gap, live and cached latency, calls, cache hits, invalid moves, confidence, cost and whether answers were live or replayed from the cache) and a **saved-results table** aggregated from `results.json`. Name and roll number are pinned in a banner at the top.
- **Budget and cache safeguards** (rules from the instructor's email): default limits from the instructor's email (20 moves max per model, Claude 5 requests, 65 in total) enforced in the backend and overridable in `.env` (see Honest notes); every real request counted *before* it is sent (failures included) and persisted in `cache.json` (never reset); every request checks the persistent cache first; live latency and cached latency are tracked separately; every real request is logged to `requests.log`; no retries (same policy for all models); loops are bounded; a model stops after 2 errors in a row.
- **Modes**: `mock` (no API, for debugging), `ollama` (local, free), `openrouter` (real, limited).

## Models

| Model | Category | OpenRouter ID | Interface |
|---|---|---|---|
| Jev 1.13 | System-1 decision model | `typesafe/jev-1.13` | Decisions API |
| GPT-6 Luna Decisions | System-1 decision model | `openai/gpt-6-luna-decisions` | Decisions API |
| Cloudflare Clef | System-1 decision model | `cloudflare/clef` | Decisions API |
| Claude Opus 5.5 | Frontier LLM | `anthropic/claude-opus-5.5` | Chat completions |

## Results: OpenRouter (real run, 2 puzzles: seeds 100 and 50)

Both puzzles start exactly 3 moves from solved (BFS-verified), so the optimum is 3. Move limit = 20 (Claude Opus: limited by its request budget, see notes). Source: `results.json`, produced by `python report.py openrouter` (`report_openrouter.md`).

| Model | Runs | Solved | Success % | Avg latency / move (s) | Median (s) | Avg total time (s) | Avg moves | Optimal gap | Live calls | Cache hits | Invalid | Cost ($) | Avg confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | 2 | 0/2 | 0 | 1.881 | 1.506 | 28.22 | 20.0 | n/a | 30 | 10 | 0 | 0.00101 | 0.32 |
| GPT-6 Luna Decisions | 1 (*) | 0/1 | 0 | 5.628 (*) | 5.628 | 11.26 | 2.0 | n/a | 2 | 2 | 2 | 0.00000 | 0.17 |
| Cloudflare Clef | 2 | 0/2 | 0 | 2.403 | 2.268 | 26.43 | 19.5 | n/a | 22 | 17 | 0 | 0.00235 | 0.06 |
| Claude Opus 5.5 | 2 | 1/2 | 50 | 15.106 | 14.532 | 60.42 | 4.0 | **0** | 8 | 0 | 0 | 0.14700 | n/a |

Optimal gap = model moves minus exact shortest moves (3), defined only for solved runs.

Per puzzle:

| Model | Seed 100 | Seed 50 |
|---|---|---|
| Jev 1.13 | not solved, 20 moves, 2.088 s/move (17 live, 3 cache) | not solved, 20 moves, 1.610 s/move (13 live, 7 cache), 20.93 s |
| GPT-6 Luna Decisions | not solved, 2 moves, then refused (HTTP 502) | **not run**: its request limit was already used up by refusals |
| Cloudflare Clef | not solved, 19 moves, 3.631 s/move (2 live, 17 cache) (**) | not solved, 20 moves, 2.280 s/move (20 live), 45.60 s |
| Claude Opus 5.5 | **solved in 3 moves = optimum**, 11.740 s/move, 35.22 s | not solved after 5 moves (request limit reached), 17.125 s/move, 85.62 s |

(*) Luna was evaluated on seed 100 only, and its numbers come from two *refused* calls, so they are not comparable with the others. Seed 50 was not run for Luna.
(**) Clef's seed-100 latency comes from only 2 live calls; 17 of its 19 moves were replayed from the cache of an earlier session of the same run.

### Real OpenRouter requests used (`cache.json` / `requests.log`)

Total **103**: Jev 33, GPT-6 Luna 20, Clef 40, Opus 10. Total recorded cost of the finished runs is about **$0.15**, almost all of it Claude Opus ($0.147). The original budget rules were 65 requests in total and 5 for Claude; the seed-50 run was done with raised limits (see "Honest notes").

## Results: Ollama (local baseline, 5 puzzles, seeds 3 to 7)

**Important:** in Ollama mode all four slots ran the **same** small model, `gemma3:1b`. This was a pipeline and baseline test (free, no API limits), not a comparison of the four models. Models were run one at a time so they did not slow each other down.

| Slot (all `gemma3:1b`) | Runs | Success % | Avg latency / move (s) | Median (s) | Avg total time (s) | Avg moves | Local requests | Cache hits | Invalid |
|---|---|---|---|---|---|---|---|---|---|
| Jev slot | 5 | 0 | 0.644 | 0.629 | 12.36 | 20.0 | 96 | 4 | 0 |
| Luna slot | 5 | 0 | 0.625 | 0.624 | 11.13 | 20.0 | 89 | 11 | 0 |
| Clef slot | 5 | 0 | 0.630 | 0.629 | 8.69 | 20.0 | 69 | 31 | 0 |
| Opus slot | 5 | 0 | 0.648 | 0.637 | 10.62 | 20.0 | 82 | 18 | 0 |

Mock-mode runs in `results.json` were only used to debug the app and are not part of any result.

## Answers to the five questions (two real puzzles, so tentative)

**1. Which model is fastest in making decisions?**
Jev: 1.88 s average live latency per move (median 1.51 s) over 30 live calls on two puzzles, followed by Clef at 2.40 s (22 live calls). Luna's 5.6 s comes from two refused calls only and cannot be ranked. Claude Opus is clearly the slowest at 15.1 s per move, about 8x slower than Jev.

**2. Which model achieved the highest solve rate?**
Claude Opus 5.5: 1 of 2 (50%). Jev and Clef solved 0 of 2, and Luna 0 of 1. Two puzzles are evidence, not a statistic.

**3. Which model found the shortest solutions, and how close to the optimum?**
Only Opus solved a cube: on seed 100 in **3 moves, the exact optimum (gap 0)**. On seed 50 it did not solve it within its 5 allowed requests. For the others the gap is undefined because they never solved a cube. Jev and Clef kept repeating or undoing moves (for example `U U'`) over 20 turns, which shows they pick a plausible-looking move each turn but do not plan ahead.

**4. Did the fastest model also achieve the best solution quality?**
No. Jev was the fastest per move but never solved the cube. Opus was about 8x slower per move and was the only model that solved one. Cost differs sharply: Opus about $0.147 against about $0.003 for all decision models together (roughly 40x).

**5. Fastest decision model, frontier LLM, or a hybrid in a real application?**
A **hybrid**, with honest limits on the evidence. The frontier LLM was the only one that solved a multi-step cube, but it also failed the second puzzle and is slow and expensive, so it should not be trusted blindly either. Decision models are about 8x faster per move, far cheaper, and expose a confidence score. Their average confidence was low on tasks they failed (0.32 Jev, 0.17 Luna, 0.06 Clef), which suggests a routing design: use the fast model first and escalate to the frontier LLM when confidence is low or progress stalls. Decision models fit fast, high-volume single-step choices (classification, routing) better than multi-step planning. Two puzzles are too few to prove the hybrid, so this is a design recommendation to test, not a conclusion.

## Honest notes and limitations

- **Two real puzzles only.** The results show behaviour on two cubes, not a statistically valid ranking. The Ollama runs (5 puzzles) used a single small model.
- **Raised request limits for the final run.** The original budget rules were 65 requests in total and 5 for Claude. After the first experiment (seed 100) those limits were used up, so the seed-50 run was done with limits raised through `.env` (`TOTAL_CAP=120`, `MAX_CALLS_JEV=40`, `MAX_CALLS_CLEF=40`, `MAX_CALLS_OPUS=10`, `MAX_CALLS_LUNA=20`). The real total used was 103 requests. Counters are persistent and were never reset.
- **Claude Opus on seed 50 was cut off.** It had only 5 requests left, so it made 5 moves, did not solve the cube and stopped with "call cap reached". I cannot say whether more moves would have solved it.
- **GPT-6 Luna refused.** After two valid moves on seed 100, Luna returned HTTP 502 `OpenAI refused to answer question "next_move"` in several separate sessions at the same state (see `requests.log`). Refusal is a documented outcome of OpenAI's Decisions API, surfaced by OpenRouter as a 502. Retrying did not help, and changing the wording would have broken the "same question for all models" rule, so Luna is recorded as not solved. Its 20 requests were 2 answered moves and the rest refusals, which used up its limit, so it was not run on seed 50.
- **Clef lost one call on seed 100.** One of its calls was lost when I stopped an earlier session with Ctrl+C before the answer was saved, so it made 19 moves there.
- **Opus lost one request to a bug of mine.** The first Opus call used `max_tokens=300`; the model spent all of them reasoning and returned `finish_reason: length` with empty content. I raised the limit to 4000 and stopped bad responses from being cached.
- **Cache hits.** Some runs were done in several sessions of the same puzzle using the persistent cache, and a model that repeats a state sends an identical request. Replayed answers are counted as cache hits, their latency is tracked separately, and they are not counted as new API calls. This is why some wall-clock totals (Clef and Jev on seed 100) understate real time.
- **Excluded attempts.** Several runs on other seeds (1, 2, 5, 10, 101) and repeated Luna sessions were blocked by the request limits or refusals and made 0 moves. They are not results. They are kept in `results_incomplete_attempts.json` for transparency and are not in any table.
- Model names in the assignment screenshots (for example "GPT-6 Sol", "Laya") are mock-ups; the IDs in the table above are the ones used.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env           # then fill in your own values
uvicorn main:app
```

Open http://127.0.0.1:8000, choose a mode (Mock, Ollama, OpenRouter) and a seed, and press **Start race**. The OpenRouter mode asks for confirmation because it makes real, limited requests.

`.env` (never committed):

```
OPENROUTER_API_KEY=your_key_here
STUDENT_NAME=Your Name
STUDENT_ROLL=Your Roll Number
# optional, only if you are allowed more requests (defaults 65 / 20 / 20 / 20 / 5):
# TOTAL_CAP=120
# MAX_CALLS_JEV=40
# MAX_CALLS_LUNA=20
# MAX_CALLS_CLEF=40
# MAX_CALLS_OPUS=10
OLLAMA_URL=http://localhost:11434
OLLAMA_JEV=gemma3:1b
OLLAMA_LUNA=gemma3:1b
OLLAMA_CLEF=gemma3:1b
OLLAMA_OPUS=gemma3:1b
```

Useful commands:

```bash
python test_cube.py              # engine tests + BFS counts (1, 18, 243, 3240)
python dry_check.py              # prints the request bodies without sending anything
python report.py openrouter      # statistics table (also: ollama, mock)
```

## Repository layout

| Path | Purpose |
|---|---|
| `cube.py` | Cube engine, 18 moves, BFS optimum, scramble generator |
| `llm.py` | Model adapters, persistent cache, request budget, logging |
| `main.py` | FastAPI backend and game runner |
| `static/index.html` | Cube Arena interface and statistics dashboard |
| `report.py` | Builds the statistics tables from `results.json` |
| `test_cube.py`, `dry_check.py` | Engine tests and request dry-run |
| `results_incomplete_attempts.json` | Blocked or empty attempts (other seeds, repeated Luna refusals), excluded from all tables |
| `cache.json`, `results.json`, `requests.log` | Persistent cache and request counters, per-run results, log of every real request |
| `report_openrouter.md`, `report_ollama.md` | Generated statistics |
| `Assignment 2 AI.mp4` | Screen recording of the real OpenRouter run (seed 50) |
| `.env.example`, `requirements.txt` | Setup |
