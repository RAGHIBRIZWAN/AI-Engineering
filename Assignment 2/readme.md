# Cube Arena: fast decision models vs a frontier LLM

**Student:** Raghib Rizwan Rabani  |  **Roll No:** 23k-0012  |  **Course:** AI Engineering

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
- **Modes**: `mock` (no API, for debugging) and `openrouter` (real, limited).

## Models

| Model | Category | OpenRouter ID | Interface |
|---|---|---|---|
| Jev 1.13 | System-1 decision model | `typesafe/jev-1.13` | Decisions API |
| GPT-6 Luna Decisions | System-1 decision model | `openai/gpt-6-luna-decisions` | Decisions API |
| Cloudflare Clef | System-1 decision model | `cloudflare/clef` | Decisions API |
| Claude Opus 5.5 | Frontier LLM | `anthropic/claude-opus-5.5` | Chat completions |

## Results: OpenRouter (real run, 1 puzzle, seed 50)

The start state is exactly 3 moves from solved (BFS-verified), so the optimum is 3. Move limit = 20 (Claude Opus: limited by its request budget, see notes). Source: `results.json`, produced by `python report.py openrouter` (`report_openrouter.md`).

| Model | Solved | Success % | Avg latency / move (s) | Median (s) | Total time (s) | Total moves | Optimal gap | Live calls | Cache hits | Invalid | Cost ($) | Avg confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | 0/1 | 0 | 1.610 | 1.680 | 20.93 | 20 (limit) | n/a | 13 | 7 | 0 | 0.00044 | 0.34 |
| Cloudflare Clef | 0/1 | 0 | 2.280 | 2.125 | 45.60 | 20 (limit) | n/a | 20 | 0 | 0 | 0.00216 | 0.05 |
| Claude Opus 5.5 | 0/1 | 0 | 17.125 | 14.944 | 85.62 | 5 (request limit) | n/a | 5 | 0 | 0 | 0.11217 | n/a |
| GPT-6 Luna Decisions | not run | n/a | n/a | n/a | n/a | 0 | n/a | 0 | 0 | 0 | 0.00000 | n/a |

Optimal gap = model moves minus exact shortest moves (3), defined only for solved runs; no model solved this cube, so it is undefined. GPT-6 Luna could not be run on this puzzle: its request limit was already used up by earlier refusals (HTTP 502, see notes). Final states on screen: Jev and Clef "RAN OUT OF MOVES", Luna and Opus "CALL CAP REACHED".

### Real OpenRouter requests

The seed-50 run itself used 38 live requests: Jev 13, Clef 20, Opus 5 (about **$0.115**, almost all of it Claude Opus). The persistent counters in `cache.json` / `requests.log` cover the whole project, including earlier development and an earlier run that is not reported here (see notes): **103** requests in total (Jev 33, GPT-6 Luna 20, Clef 40, Opus 10).

## Answers to the five questions (one real puzzle, so tentative)

**1. Which model is fastest in making decisions?**
Jev: 1.61 s average live latency per move (median 1.68 s, 13 live calls), followed by Clef at 2.28 s (20 live calls). Claude Opus is clearly the slowest at 17.1 s per move, about 10x slower than Jev. GPT-6 Luna was not run on this puzzle, so it cannot be ranked.

**2. Which model achieved the highest solve rate?**
None. All three models that ran (Jev, Clef, Opus) solved 0 of 1, so the solve rate does not separate them on this puzzle. One puzzle is evidence, not a statistic.

**3. Which model found the shortest solutions, and how close to the optimum?**
No model solved the cube, so the optimal gap is undefined for all of them (the optimum is 3 moves). Jev and Clef used all 20 moves and kept repeating or undoing moves (for example `U U'`), which shows they pick a plausible-looking move each turn but do not plan ahead. Opus made 5 moves and was then stopped by its request limit; I cannot say whether more moves would have solved it.

**4. Did the fastest model also achieve the best solution quality?**
On this puzzle quality did not differ: nobody solved the cube. Jev was the fastest but unsolved. Opus was about 10x slower per move and about 40x more expensive ($0.112 against about $0.003 for Jev and Clef together), and it was also unsolved. So here the extra time and cost of the frontier LLM bought nothing measurable.

**5. Fastest decision model, frontier LLM, or a hybrid in a real application?**
A **hybrid**, as a design recommendation, not a conclusion from this data. One failed puzzle does not show that the frontier LLM is better at multi-step planning, only that it is slow and costly. Decision models are about 10x faster per move, far cheaper, and expose a confidence score. Their average confidence was low on this failed task (0.34 Jev, 0.05 Clef), which suggests a routing design: use the fast model first and escalate to the frontier LLM when confidence is low or progress stalls (for example the same moves repeating). Decision models fit fast, high-volume single-step choices (classification, routing) better than multi-step planning. More puzzles are needed to test this.

## Honest notes and limitations

- **One real puzzle reported.** The tables show behaviour on one cube (seed 50), not a statistically valid ranking.
- **An earlier run exists but is not reported.** Before the seed-50 run, a first real experiment was made on a different puzzle (seed 100). It is not part of the tables or answers above. Its data is still in the raw files (`results.json`, `cache.json`, `requests.log`). It is also why the total request count (103) is higher than the 38 requests of the seed-50 run.
- **Raised request limits.** The original budget rules were 65 requests in total and 5 for Claude. Those limits were used up before the seed-50 run, so it was done with limits raised through `.env` (`TOTAL_CAP=120`, `MAX_CALLS_JEV=40`, `MAX_CALLS_CLEF=40`, `MAX_CALLS_OPUS=10`, `MAX_CALLS_LUNA=20`). Counters are persistent and were never reset.
- **Claude Opus was cut off.** It had only 5 requests left, so it made 5 moves, did not solve the cube and stopped with "call cap reached".
- **GPT-6 Luna refused in earlier sessions.** It returned HTTP 502 `OpenAI refused to answer question "next_move"` after two valid moves, in several separate sessions (see `requests.log`). Refusal is a documented outcome of OpenAI's Decisions API, surfaced by OpenRouter as a 502. Retrying did not help, and changing the wording would have broken the "same question for all models" rule. Its limit was used up by these refusals, so it was not run on seed 50.
- **Opus lost one request to a bug of mine during development.** An early Opus call used `max_tokens=300`; the model spent all of them reasoning and returned `finish_reason: length` with empty content. I raised the limit to 4000 and stopped bad responses from being cached.
- **Cache hits.** Jev's 7 cache hits on seed 50 are repeated identical states (it kept cycling), which send an identical request. Replayed answers are counted as cache hits, their latency is tracked separately, and they are not counted as new API calls.
- **Excluded attempts.** Several runs on other seeds (1, 2, 5, 10, 101) were blocked by the request limits or refusals and made 0 moves. They are not results. They are kept in `results_incomplete_attempts.json` for transparency and are not in any table.
- Model names in the assignment screenshots (for example "GPT-6 Sol", "Laya") are mock-ups; the IDs in the table above are the ones used.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env           # then fill in your own values
uvicorn main:app
```

Open http://127.0.0.1:8000, choose a mode (Mock or OpenRouter) and a seed, and press **Start race**. The OpenRouter mode asks for confirmation because it makes real, limited requests.

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
```

Useful commands:

```bash
python test_cube.py              # engine tests + BFS counts (1, 18, 243, 3240)
python dry_check.py              # prints the request bodies without sending anything
python report.py openrouter      # statistics table (also: mock)
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
| `report_config.json` | Seeds excluded from the generated reports and the saved-results table |
| `results_incomplete_attempts.json` | Blocked or empty attempts (other seeds, repeated Luna refusals), excluded from all tables |
| `cache.json`, `results.json`, `requests.log` | Persistent cache and request counters, per-run results, log of every real request |
| `report_openrouter.md` | Generated statistics |
| `Assignment 2 AI.mp4` | Screen recording of the real OpenRouter run (seed 50) |
| `.env.example`, `requirements.txt` | Setup |
