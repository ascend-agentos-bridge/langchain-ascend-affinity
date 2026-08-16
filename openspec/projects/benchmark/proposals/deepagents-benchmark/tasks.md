# Tasks: deepagents Affinity Benchmark

- [x] T1. Library: add `_stream` (SSE) + `affinity_stats` counters to
      `AscendAffinityChatModel` per REQ-1.
- [x] T2. Unit tests for streaming (content, tool-call deltas, affinity
      fields on stream, `[DONE]` handling) and counters.
- [x] T3. `benchmark/tasks.py`: financial task set + deterministic tools.
- [x] T4. `benchmark/agents.py`: baseline / affinity deepagents factories.
- [x] T5. `benchmark/run_benchmark.py`: setup, engine probe, dual-agent
      execution, TTFT recorder, correctness scoring, md+json report.
- [x] T6. Bilingual benchmark READMEs + root README Benchmark section +
      `.gitignore` for `benchmark/reports/`.
- [x] T7. Quality gate green (pylint 10.00 incl. benchmark/, coverage ≥ 90%).
- [x] T8. Real-engine smoke run (user-provided endpoint) producing a report.
