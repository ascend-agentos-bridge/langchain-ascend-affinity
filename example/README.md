# Verification Example: openJiuwen Affinity Port

A hardware-free harness that proves the ported affinity mechanism works:
[verify_affinity.py](verify_affinity.py) replays the *same* deterministic
two-user, three-turn dialogue schedule twice against a mock Ascend engine
([mock_engine.py](mock_engine.py)):

- **plain phase** — `AscendAffinityChatModel(enable_affinity=False)`: no
  `cache_salt`, both users share one anonymous KV-cache bucket, interleaved
  turns keep diverging on it (5 of 6 requests pay a partial/full recompute).
- **affinity phase** — the same model salt-bound per session
  (`bind(session_id=...)`): each user gets an isolated cache bucket (4 of 6
  requests warm), and the mid-session history rewrite triggers the ported
  prefix-diff scheduler to issue exactly one `POST /release_kv_cache`.

## Run

```bash
pip install -r example/requirements.txt
python example/verify_affinity.py          # PASS + comparison table

# options
python example/verify_affinity.py --port 8001
MOCK_TTFT_COLD_MS=500 python example/verify_affinity.py   # exaggerate the gap
```

## Reading the output

```text
metric                        plain    affinity
requests                          6           6
cold starts                       1           2
warm hits                         0           4
partial recomputes                5           0
kv releases                       0           1
avg TTFT (ms)                 189.6        96.7
salt buckets                anonymous  user-A,user-B
answers identical across phases: yes
```

- **warm hits / partial recomputes**: salt binding isolates sessions so pure
  appends hit the cache instead of recomputing a stale prefix.
- **kv releases**: the scheduler detected the rewritten history message and
  told the engine to drop only the stale suffix (agent-core compatible
  payload), keeping the valid prefix resident.
- **answers identical**: the engine and schedule are deterministic, so both
  phases serve byte-equal answers — the comparison is apples-to-apples.

The script exits non-zero if any invariant breaks (no release fired, no warm
gain, TTFT not clearly lower, answers diverged), so it doubles as a smoke test.

## Mock engine

`mock_engine.py` exposes `/v1/chat/completions`, `/release_kv_cache` and
`/metrics`, binds KV blocks to `cache_salt`, and prices TTFT by cache
temperature (warm / partial·scaled-by-stale-fraction / cold). Tunables via
env vars: `MOCK_TTFT_WARM_MS` (20), `MOCK_TTFT_COLD_MS` (250),
`MOCK_KV_SLOTS` (4), `MOCK_ENGINE_PORT` (8000). See the module docstring.
