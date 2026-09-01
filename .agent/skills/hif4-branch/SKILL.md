---
name: hif4-branch
description: Create, track, score, and visualize branching HiF4 solution experiments without overwriting parent versions. Use when starting a new optimization direction, recording an evaluated version, or choosing among multiple branches.
---

# HiF4 Version Tree

Keep solution directories flat at `solution/<version>/`. Store parent-child relationships and experiment results in `.agent/version-tree.json`. A parent may have at most three direct algorithm children.

The formal version unit is a structural quantization algorithm. Pure alpha, gain, threshold, coefficient, factor, or candidate-count changes belong to at most three internal configurations of one algorithm version and must not create branches.

## Version names

Name every new branch as `v<number>_<short_change_name>`. The suffix must be a brief description of this branch's actual modification, using lowercase English words joined by underscores.

`v0_hessian_repair` is the existing root-version exception. New children start at `v1_<short_change_name>`; the creation script intentionally rejects creation of additional `v0_*` versions.

Good examples:

- `v2_k_hessian`: change the K-side Hessian;
- `v2_linear_order`: change Linear quantization order;
- `v2_scale_search`: change the scale candidate search;
- `v3_v_outlier`: add V outlier protection.

Avoid vague suffixes such as `new`, `better`, `test`, `final`, or a person's name. Sibling branches may share the same numeric generation, because the suffix distinguishes their hypotheses. The implementation file inside each directory remains `solution.py`; do not rename it to `v2_xxx.py`, because the competition submission contract requires `solution.py`.

For asynchronous local-agent work, use `.agent/runner.py`; it is the only writer for queue and tree state. `tree.py` remains a manual inspection and recovery helper.

## Create a branch

Choose one primary hypothesis and one focus: `linear`, `attention`, `format`, or `combined`.

```powershell
python .agent/skills/hif4-branch/scripts/tree.py create `
  --parent v0_hessian_repair `
  --name v2_linear_order `
  --focus linear `
  --hypothesis "改变 Hessian 量化顺序可以降低 Linear Output MSE"
```

This copies only the parent's `solution.py`, creates a pre-implementation `policy.md`, and registers the edge. `report.md` is generated later by `$hif4-report` after evaluation. Never overwrite an existing version or reuse a name for a different hypothesis.

Each parent may have at most three direct children. Once all three slots are used, continue from a suitable child rather than adding a fourth sibling.

## Record evaluated results

After `$hif4-evaluate` succeeds, record the exact displayed values:

```powershell
python .agent/skills/hif4-branch/scripts/tree.py record `
  --name v2_linear_order `
  --linear-mse 0.0024 `
  --attention-mse 0.00034 `
  --score 6.02 `
  --status evaluated
```

Allowed statuses are `draft`, `evaluated`, `promising`, and `rejected`; the root may use `baseline`. A rejected branch remains in the tree so later agents do not repeat the same experiment.

## Inspect the tree

```powershell
python .agent/skills/hif4-branch/scripts/tree.py show
```

Show the optimization priority queue:

```powershell
python .agent/skills/hif4-branch/scripts/tree.py queue
```

The queue contains evaluated nodes with free child slots, ordered by final score from high to low. Use it as the default scheduling order, while still choosing a direction-specific parent when a lower-scoring node has a uniquely better Linear or Attention result.

Select a parent according to the intended direction, not only global score. For example, a child with the best Linear MSE may be the right parent for another Linear experiment even if its total score is not yet highest.
