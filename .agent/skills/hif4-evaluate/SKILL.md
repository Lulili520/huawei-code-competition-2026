---
name: hif4-evaluate
description: Validate and score one or more Huawei HiF4 competition solution.py versions with the project's official-format self-check and local Linear/Attention MSE scorer. Use when testing, comparing, accepting, or rejecting solution versions.
---

# HiF4 Solution Evaluation

Use this skill's maintained script at `.agent/skills/hif4-evaluate/scripts/evaluate.py`; do not recreate scoring logic in temporary scripts.

## Run

From the repository root, run:

```powershell
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/skills/hif4-evaluate/scripts/evaluate.py solution/v0_hessian_repair/solution.py solution/<candidate>/solution.py --datasets-dir datasets/combined --linear-groups 10 --attention-groups 50 --fidelity-label formal
```

For local-agent orchestration, request machine-readable evidence:

```powershell
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/skills/hif4-evaluate/scripts/evaluate.py solution/<candidate>/solution.py --datasets-dir datasets/combined --linear-groups 10 --attention-groups 50 --fidelity-label formal --json-output .agent/runtime/runs/<run>/evaluation.json
```

Runner invokes this absolute environment Python directly after checking its `--version`; formal evaluation must not use `conda run`, whose wrapper can fail independently of the algorithm.

The evaluator first runs the official `self_check.py` on its compact 10-case
suite to validate interfaces and shapes. This self-check never produces a
version score. Runner-managed versions use a two-fidelity cascade when there
are three internal configurations:

1. evaluate every config on the same deterministic, evenly spaced 10 Linear +
   50 Attention screening subset;
2. promote the top two screening configs;
3. evaluate the promoted configs on all 50 Linear + 250 Attention cases;
4. register only a full-300 result. One or two configs go directly to full
   evaluation.

All configs in one stage are passed to one evaluator process so the 3.7 GB
dataset is mapped once rather than once per config.

Candidate-independent CPU reference outputs and standard-encoder MSE values
are cached per dataset group under `.agent/runtime/reference-cache/`. The key
binds the dataset manifest, PyTorch version, and cache algorithm version.
Candidate quantization and all requested cases still execute on every run.
Use `--no-reference-cache` only for parity diagnostics; a cache is acceptable
only after score, Linear MSE, and Attention MSE match exactly.

- Linear Output MSE;
- Attention Output MSE;
- final local score: average each category, weight Linear : Attention as 1:5,
  then multiply by 300 and 100 percentage points. With 50 Linear and 250
  Attention cases this is exactly the sum of all 300 per-case percentage gains.

Every formal local result uses the immutable unified dataset at
`datasets/combined` and observes exactly 50 Linear and 250 Attention cases.
Reports and the flat registry must record these observed counts from evaluator
JSON. Partial results and F60 screening cannot rank a version and are not kept
in the public score ledger.
The original official mini sample remains reference material and an interface
compatibility source; it is not the current ranking dataset.

Only `.agent/runner.py` may run and register formal local-agent evaluations. Worker Agents may prepare at most three configurations but must not write formal metrics themselves.

After a successful full evaluation, Runner writes an evaluation summary and checkpoint before invoking the report Agent. If report generation or registration later fails, `recover` reuses that checkpoint and does not repeat the expensive 300-case evaluation. A screening or manual partial diagnostic is never a valid checkpoint for version registration.

The JSON output also records per-case standard/player MSE and percentage-point
gain, score distribution (minimum, P10, median, P90, maximum and negative case
count), reference-cache hit/build time, calibration/dynamic/output phase
timings, and selected group indices. Algorithms with internal
candidate search may expose a read-only `hif4_get_diagnostics()` function; its
JSON-serializable counters are captured under `implementation_diagnostics`.

For a non-registering screening diagnostic, use:

```powershell
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/skills/hif4-evaluate/scripts/evaluate.py solution/<version>/solution.py --datasets-dir datasets/combined --linear-groups 2 --attention-groups 10
```

Use `--skip-self-check` only for repeated profiling after the same unchanged candidate has already passed the official check in the current experiment.

## Accepting a result

A valid experiment must satisfy all of the following:

1. The candidate is stored at `solution/<version>/solution.py`.
2. Official output-format checks pass with no failed case.
3. The evaluator exits with code 0 and prints both MSE values and the final score.
4. Compare versions by final score first; use Linear and Attention MSE, negative-case count and runtime to identify which path and trade-off caused the change.

The local standard HiF4 encoder is a reproducible approximation because the platform does not publish its internal standard encoder or hidden `MSE_STD`. Never present the local score as the official platform score.

`20000` is a reference target on the actually executed 300-case score scale, not an automatic stop. Runner continues until the user explicitly pauses or ends it. Compact self-check, 60-case screening, manual diagnostics and estimated platform scores never count as formal results.

After execution, remove generated `__pycache__` directories from `solution/`; do not retain copied mini-sample data or temporary evaluators in the repository.

When the version belongs to `.agent/versions.json`, use `$hif4-version` to record the displayed MSE and score after updating its report.
