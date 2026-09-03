---
name: hif4-version
description: Create, record, and rank flat sequential HiF4 algorithm versions. Use when allocating a solution/vN_method directory or updating evaluated version metadata; do not use for tree or branch management.
---

# HiF4 Flat Versions

Store every algorithm in `solution/vN_<method>/`, where `N` is the next global integer and `<method>` is a concise lowercase summary. Versions are a flat sequence; do not infer generation or ancestry from `N`.

`.agent/versions.json` is the authoritative ledger. `based_on` records the comparison or implementation reference only. It does not create a parent-child edge and never limits which version may be studied next.

Do not select a reference from score alone. `.agent/knowledge/pareto.json`
maintains non-dominated versions over total score, Linear MSE, Attention MSE
and runtime; choose the version whose strength matches the proposed focus.
Consult the bounded positive/negative records in
`.agent/knowledge/experiments.json` before allocating a repeated idea.

A new version must be a structural quantization algorithm with a falsifiable hypothesis. Keep two or three theory-backed hyperparameter alternatives inside that version under `trials/`; never allocate a version for a pure threshold, alpha, gain, factor, or candidate-count change.

Use `.agent/runner.py` during asynchronous work because it exclusively allocates global version numbers and writes shared state. For paused manual maintenance, use `scripts/versions.py`:

```powershell
python .agent/skills/hif4-version/scripts/versions.py create `
  --based-on v0_softmax_aware_qk `
  --method discrete_attention_search `
  --focus attention `
  --hypothesis "直接在合法 HiF4 候选中最小化校准 Attention 输出误差"

python .agent/skills/hif4-version/scripts/versions.py record `
  --name v1_discrete_attention_search `
  --linear-mse 0.0025 --attention-mse 0.00032 --score 4.95

python .agent/skills/hif4-version/scripts/versions.py list
python .agent/skills/hif4-version/scripts/versions.py queue
```

The queue lists every valid evaluated version by the current 1:5 weighted score. Choose `based_on` according to algorithm fit as well as score; for example, the Linear-MSE leader may be a better implementation reference for a Linear-only strategy than the global score leader.

Always preserve fixed NVFP4 decoding: consecutive groups of 16 E2M1 values multiply the supplied E4M3 scale, then restore shape.
