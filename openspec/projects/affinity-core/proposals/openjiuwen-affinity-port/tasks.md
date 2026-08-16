# Tasks: openJiuwen Affinity Port

- [x] T1. Remove obsolete machinery: `langchain_ascend/callbacks/`,
      `langchain_ascend/backends/`, old `AscendChatLLM` exports; prune stale
      tests (`test_compute_affinity.py`, `test_backend_*.py`,
      `test_mock_hardware` fixtures, pydantic-fields test for old fields).
- [x] T2. Implement `AscendAffinityChatModel` in `llms/chat_ascend.py` per
      design D1–D6 (salt injection, session resolution, prefix diff, partial
      release, failure containment, sync + async).
- [x] T3. Update `langchain_ascend/__init__.py` exports to REQ-1 surface.
- [x] T4. Rewrite unit tests: tracker suite (kept), model request-contract
      suite (REQ-2), scheduling-fidelity suite (REQ-3), release-transport
      suite (REQ-4, incl. async + failure paths).
- [x] T5. Rewrite root READMEs (EN + zh-CN) per REQ-5: three-framework Quick
      Start, mechanism section, installation; keep bilingual sync and
      language-switch line.
- [x] T6. Run `python scripts/quality_gate.py` until pylint 10.00 + coverage
      ≥ 90%.

> Amendment (2026-08-16): the `example/` directory (mock engine +
> verification harness) was removed after review — simulated TTFT numbers
> are self-fulfilling and add nothing over the unit-test contract suites.
> Real benefit is measured on real Ascend engines only.
