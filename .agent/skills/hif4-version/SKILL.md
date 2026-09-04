---
name: hif4-version
description: Create, record, and rank flat sequential HiF4 algorithm versions. Use when allocating a solution/vN_method directory or updating evaluated version metadata; do not use for tree or branch management.
---

# HiF4 Flat Versions

Store every algorithm in `solution/vN_<method>/`, where `N` is the next global integer and `<method>` is a concise lowercase summary. Versions are a flat sequence; do not infer generation or ancestry from `N`.

`.agent/versions.json` is the authoritative ledger. `based_on` records the comparison or implementation reference only. It does not create a parent-child edge and never limits which version may be studied next.

Every queued task also has a `search_mode` and a decomposed `root_cause`. `explore` means testing a new mechanism, cross-family cause, or scratch design; `exploit` means structurally deepening a measured positive result or Pareto component advantage. Runner maintains one global six-slot portfolio targeting four `explore` and two `exploit` tasks, counting already running tasks. This is a resource mix, not ancestry and not a per-version child quota.

Do not select a reference from score alone. `.agent/knowledge/pareto.json`
maintains non-dominated versions over total score, Linear MSE, Attention MSE
and runtime; choose the version whose strength matches the proposed focus.
Consult the bounded positive/negative records in
`.agent/knowledge/experiments.json` before allocating a repeated idea.

A new version must be a structural quantization algorithm with a falsifiable hypothesis. Keep two or three theory-backed hyperparameter alternatives inside that version under `trials/`; never allocate a version for a pure threshold, alpha, gain, factor, or candidate-count change.

Autonomous follow-up directions are accepted only from post-evaluation report feedback (initial seeds and explicit user-enqueued research questions are the only entry-point exceptions). Each fully evaluated version returns exactly two `explore` and one `exploit` proposals after full metrics and diagnostics are available. Implementation output must not predeclare successors. Runner may select any already fully evaluated `based_on`; an unsuccessful current version does not force its proposals to inherit its code.

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

The queue dynamically combines source quality, evidence, novelty, uncertainty, family history, exploration/stagnation rewards, cost and failure penalties; it does not simply copy a source version's score. Within each search role it covers distinct focus values and algorithm families before filling by priority. Choose `based_on` according to algorithm fit as well as score; for example, the Linear-MSE leader may be a better implementation reference for a Linear-only strategy than the global score leader.

Only a complete 50 Linear + 250 Attention result may update the ledger or satisfy the automatic target. The target is a full score of `20000`; compact self-check and screening scores never count. Once reached, Runner stops new dispatch while allowing already running atomic tasks to finish.

Recovery preserves version identity. When a timed-out workspace already contains substantial implementation artifacts, `recover` resumes at `implementation_finalize`; when full evaluation is checkpointed, it resumes report/registration without spending another full evaluation.

Always preserve fixed NVFP4 decoding: consecutive groups of 16 E2M1 values multiply the supplied E4M3 scale, then restore shape.
