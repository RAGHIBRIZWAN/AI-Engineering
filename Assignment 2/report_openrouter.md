## Results (openrouter mode)
Puzzles: [50] (seeds), each exactly 3 moves from solved, optimum = 3, move limit = 20.

| Model | Runs | Success % | Avg latency/move (s) | Median latency (s) | Avg cached (s) | Avg total time (s) | Avg moves | Optimal gap | API calls | Cache hits | Invalid | Cost ($) | Avg confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | 1 | 0 | 1.610 | 1.680 | 0.0000 | 20.93 | 20.0 | - | 13 | 7 | 0 | 0.00044 | 0.34 |
| Cloudflare Clef | 1 | 0 | 2.280 | 2.125 | - | 45.60 | 20.0 | - | 20 | 0 | 0 | 0.00216 | 0.05 |
| Claude Opus 5.5 | 1 | 0 | 17.125 | 14.944 | - | 85.62 | 5.0 | - | 5 | 0 | 0 | 0.11217 | - |

### Data for the five questions
1. Fastest decisions (lowest avg live latency): Jev 1.13 (1.610)
2. Highest solve rate: Jev 1.13, Cloudflare Clef, Claude Opus 5.5 (0.000) %
3. Shortest solutions (lowest optimal gap, 0 = optimal): n/a
4. Fastest model: Jev 1.13 (1.610) | Best quality: n/a. Compare the two lines to answer.
5. Fast model vs frontier LLM vs hybrid: argue from the table above (latency, success, gap, cost).