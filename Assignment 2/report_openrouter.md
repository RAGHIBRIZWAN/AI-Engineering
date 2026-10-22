## Results (openrouter mode)
Puzzles: [50, 100] (seeds), each exactly 3 moves from solved, optimum = 3, move limit = 20.

| Model | Runs | Success % | Avg latency/move (s) | Median latency (s) | Avg cached (s) | Avg total time (s) | Avg moves | Optimal gap | API calls | Cache hits | Invalid | Cost ($) | Avg confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev 1.13 | 2 | 0 | 1.881 | 1.506 | 0.0000 | 28.22 | 20.0 | - | 30 | 10 | 0 | 0.00101 | 0.32 |
| GPT-6 Luna Decisions | 1 | 0 | 5.628 | 5.628 | 0.0000 | 11.26 | 2.0 | - | 2 | 2 | 2 | 0.00000 | 0.17 |
| Cloudflare Clef | 2 | 0 | 2.403 | 2.268 | 0.0000 | 26.43 | 19.5 | - | 22 | 17 | 0 | 0.00235 | 0.06 |
| Claude Opus 5.5 | 2 | 50 | 15.106 | 14.532 | - | 60.42 | 4.0 | 0.0 | 8 | 0 | 0 | 0.14700 | - |

### Data for the five questions
1. Fastest decisions (lowest avg live latency): Jev 1.13 (1.881)
2. Highest solve rate: Claude Opus 5.5 (50.000) %
3. Shortest solutions (lowest optimal gap, 0 = optimal): Claude Opus 5.5 (0.000)
4. Fastest model: Jev 1.13 (1.881) | Best quality: Claude Opus 5.5 (0.000). Compare the two lines to answer.
5. Fast model vs frontier LLM vs hybrid: argue from the table above (latency, success, gap, cost).