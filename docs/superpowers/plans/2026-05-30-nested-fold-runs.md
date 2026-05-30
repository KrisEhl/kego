# Nested Fold Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group multi-fold cluster submissions under a single MLflow parent run so folds log `val_rmse` instead of `fold0_val_rmse`, and `kego ls` shows one row per experiment (not N rows per fold).

**Architecture:** When `kego run --target cluster --folds 0,1,2,3,4` is called, create one parent MLflow run tagged `kego_is_parent: true`, then create N child runs each tagged `mlflow.parentRunId`. Each fold job resumes its pre-created child run (existing path — no changes to `runner.py` or `cluster.py`). `kego ls` hides children by default; `--children` reveals them with a FOLD column. Single-fold and local runs are unchanged.

**Tech Stack:** Python 3.10+, MLflow Python SDK (`MlflowClient`), pytest

---

## File Map

| File | Change |
|---|---|
| `kego/cli/commands/run.py` | Refactor `_pre_create_runs` to create parent + children for multi-fold |
| `kego/cli/commands/ls.py` | Hide child runs by default; add `--children` flag; add FOLD column |
| `tests/cli/test_commands_ls.py` | Tests for child filtering and `--children` display |
| `tests/cli/test_e2e.py` | Update e2e golden-path test (now produces parent + child runs) |

`runner.py`, `cluster.py`, and `local.py` require **no changes** — cluster folds resume pre-created child runs as before; local always runs single fold.

---

## Task 1: Parent + child run pre-creation in `run.py`

**Files:**
- Modify: `kego/cli/commands/run.py` — replace `_pre_create_runs`

### What changes

`_pre_create_runs` currently creates N sibling runs (one per fold). Change it to:

- **When `len(folds) > 1`**: create 1 parent run, then N child runs each tagged with `mlflow.parentRunId = <parent_run_id>`.
- **When `len(folds) == 1`**: behave exactly as before (one run, no parent wrapper).

Parent run tags:
- `kego_id`, `mlflow.runName`, `kego_target`, `kego_debug`, `kego_primary_metric`
- `kego_is_parent: "true"`
- `kego_fold_count: str(len(folds))`

Child run tags: same as current, plus `mlflow.parentRunId: <parent_run_id>`.

Return value stays `dict[int, str]` (fold → child run_id) — callers unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/cli/test_e2e.py`:

```python
def test_run_multifold_creates_parent_and_children(tmp_path: Path, repo_root: Path) -> None:
    """Multi-fold cluster submission creates one parent + N child MLflow runs."""
    import mlflow

    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch_env = {
        "MLFLOW_TRACKING_URI": mlflow_uri,
        "PYTHONPATH": str(repo_root),
    }
    # We call _pre_create_runs directly (no real cluster needed)
    import os
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri

    from kego.cli.commands.run import _pre_create_runs
    from kego.cli.config import ClusterConfig, CompetitionConfig, KegoConfig
    from pathlib import Path as P

    config = KegoConfig(
        cluster=ClusterConfig(ray_address="http://x:8265", mlflow_uri=mlflow_uri),
        competition=CompetitionConfig(
            slug="test-comp", kaggle_user="u", enable_gpu=False,
            submit_file="s.csv", pattern="script", inference_notebook="t.py",
            checkpoint_dir="out", primary_metric="rmse",
        ),
        repo_root=P("/repo"),
        competition_dir=None,
    )

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("test-comp")

    fold_run_ids = _pre_create_runs(config, "test-comp", "my-run", "abc123", {}, [0, 1, 2])

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    all_runs = client.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("test-comp").experiment_id]
    )

    parent_runs = [r for r in all_runs if r.data.tags.get("kego_is_parent") == "true"]
    child_runs = [r for r in all_runs if r.data.tags.get("mlflow.parentRunId")]

    assert len(parent_runs) == 1
    assert len(child_runs) == 3
    assert parent_runs[0].data.tags["kego_fold_count"] == "3"
    parent_id = parent_runs[0].info.run_id
    for child in child_runs:
        assert child.data.tags["mlflow.parentRunId"] == parent_id
    # Return value maps folds to child run IDs (not parent)
    assert set(fold_run_ids.keys()) == {0, 1, 2}
    assert set(fold_run_ids.values()) == {r.info.run_id for r in child_runs}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
uv run pytest tests/cli/test_e2e.py::test_run_multifold_creates_parent_and_children -v
```
Expected: FAIL (current `_pre_create_runs` creates 3 siblings, no parent)

- [ ] **Step 3: Implement**

Replace `_pre_create_runs` in `kego/cli/commands/run.py`:

```python
def _pre_create_runs(
    config: cfg_module.KegoConfig,
    experiment_name: str,
    run_name: str,
    experiment_id: str,
    cli_params: dict[str, str],
    folds: list[int],
) -> dict[int, str]:
    """Create MLflow runs for a multi-fold submission.

    Single fold: creates one run (unchanged behaviour).
    Multi-fold: creates one parent run + one child run per fold.
    Returns fold → child_run_id (callers unchanged).
    Falls back gracefully if MLflow is unreachable.
    """
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.cluster.mlflow_uri
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    run_ids: dict[int, str] = {}
    try:
        from mlflow.entities import Param
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        exp = mlflow.set_experiment(experiment_name)
        primary_metric = config.competition.primary_metric if config.competition else ""
        multi = len(folds) > 1

        parent_run_id: str | None = None
        if multi:
            parent_tags = {
                "kego_id": experiment_id,
                "kego_target": "cluster",
                "kego_debug": "false",
                "mlflow.runName": run_name,
                "kego_primary_metric": primary_metric,
                "kego_is_parent": "true",
                "kego_fold_count": str(len(folds)),
            }
            parent = client.create_run(
                experiment_id=exp.experiment_id,
                run_name=run_name,
                tags=parent_tags,
            )
            parent_run_id = parent.info.run_id
            if cli_params:
                client.log_batch(
                    parent_run_id,
                    params=[Param(k, str(v)) for k, v in cli_params.items()],
                )

        for fold in folds:
            fold_params = {**cli_params, "fold": str(fold)}
            tags = {
                "kego_id": experiment_id,
                "kego_target": "cluster",
                "kego_debug": "false",
                "mlflow.runName": run_name,
                "kego_primary_metric": primary_metric,
            }
            if parent_run_id:
                tags["mlflow.parentRunId"] = parent_run_id
            run = client.create_run(
                experiment_id=exp.experiment_id,
                run_name=f"{run_name} fold={fold}",
                tags=tags,
            )
            client.log_batch(
                run.info.run_id,
                params=[Param(k, str(v)) for k, v in fold_params.items()],
            )
            run_ids[fold] = run.info.run_id
    except Exception as e:
        print(
            f"  Warning: could not pre-create MLflow runs ({e})"
            " — jobs won't appear in kego ls until they start",
            flush=True,
        )

    return run_ids
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
uv run pytest tests/cli/test_e2e.py::test_run_multifold_creates_parent_and_children -v
```
Expected: PASS

- [ ] **Step 5: Run full suite to check for regressions**

```bash
uv run pytest tests/ -q
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add kego/cli/commands/run.py tests/cli/test_e2e.py
git commit -m "feat(runs): create parent + child runs for multi-fold cluster submissions"
```

---

## Task 2: `kego ls` — hide children by default, add `--children` + FOLD column

**Files:**
- Modify: `kego/cli/commands/ls.py`
- Modify: `tests/cli/test_commands_ls.py`

### What changes

1. Add `--children` flag to `add_parser`.
2. In `_ls`, after fetching runs: if `not args.show_children`, filter out rows where `tags.mlflow.parentRunId` is set.
3. In `format_table`, add optional `show_fold: bool = False`. When True, insert a `FOLD` column (6 chars) between NAME and COMPETITION. Fold value comes from:
   - Child run: `params.fold`
   - Parent run: `N folds` (from `tags.kego_fold_count`)
   - Regular run: `-`

- [ ] **Step 1: Write the failing tests**

Add to `tests/cli/test_commands_ls.py`:

```python
def test_ls_hides_child_runs_by_default(mlflow_db, capsys):
    """Child runs (mlflow.parentRunId set) are hidden unless --children is passed."""
    from mlflow.tracking import MlflowClient
    import secrets

    client = MlflowClient()
    exp = mlflow.set_experiment("test-exp")

    # Parent run
    parent = client.create_run(
        exp.experiment_id,
        run_name="parent-run",
        tags={"kego_id": secrets.token_hex(3), "kego_target": "cluster",
              "kego_debug": "false", "kego_is_parent": "true", "kego_fold_count": "2"},
    )
    client.set_terminated(parent.info.run_id, status="RUNNING")

    # Child runs
    for fold in range(2):
        child = client.create_run(
            exp.experiment_id,
            run_name=f"parent-run fold={fold}",
            tags={"kego_id": secrets.token_hex(3), "kego_target": "cluster",
                  "kego_debug": "false", "mlflow.parentRunId": parent.info.run_id},
        )
        client.set_terminated(child.info.run_id, status="FINISHED")

    _ls(_make_ls_args(), [])
    out = capsys.readouterr().out
    assert "parent-run" in out
    assert "fold=0" not in out
    assert "fold=1" not in out


def test_ls_children_flag_shows_child_runs(mlflow_db, capsys):
    """--children shows child runs alongside parent runs."""
    from mlflow.tracking import MlflowClient
    import secrets

    client = MlflowClient()
    exp = mlflow.set_experiment("test-exp")

    parent = client.create_run(
        exp.experiment_id, run_name="parent-run",
        tags={"kego_id": secrets.token_hex(3), "kego_target": "cluster",
              "kego_debug": "false", "kego_is_parent": "true", "kego_fold_count": "2"},
    )
    client.set_terminated(parent.info.run_id, status="RUNNING")

    for fold in range(2):
        child = client.create_run(
            exp.experiment_id,
            run_name=f"parent-run fold={fold}",
            tags={"kego_id": secrets.token_hex(3), "kego_target": "cluster",
                  "kego_debug": "false", "mlflow.parentRunId": parent.info.run_id},
        )
        client.log_param(child.info.run_id, "fold", str(fold))
        client.set_terminated(child.info.run_id, status="FINISHED")

    _ls(_make_ls_args(show_children=True), [])
    out = capsys.readouterr().out
    # Both parent and children visible
    assert "parent-run" in out
    assert out.count("parent-run") >= 3  # parent row + 2 child rows
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/cli/test_commands_ls.py::test_ls_hides_child_runs_by_default tests/cli/test_commands_ls.py::test_ls_children_flag_shows_child_runs -v
```
Expected: both FAIL (`show_children` attribute not found)

- [ ] **Step 3: Add `--children` to `_make_ls_args` in tests**

In `tests/cli/test_commands_ls.py`, update `_make_ls_args`:

```python
def _make_ls_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        name=None,
        status=None,
        target=None,
        competition=None,
        since=None,
        limit=50,
        show_all=False,
        show_metric_name=False,
        show_children=False,   # add this line
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
```

- [ ] **Step 4: Add `--children` to `add_parser` in `ls.py`**

In `kego/cli/commands/ls.py`, inside `add_parser`, after the `--metric-name` argument:

```python
p.add_argument(
    "--children",
    action="store_true",
    dest="show_children",
    help="Show child fold runs (hidden by default; requires parent run created with --folds)",
)
```

- [ ] **Step 5: Add child-run filtering to `_ls`**

In `kego/cli/commands/ls.py`, in `_ls`, after the debug-run filter block:

```python
# Filter debug runs in Python — MLflow filter doesn't handle missing tags correctly
if not args.show_all and "tags.kego_debug" in runs.columns:
    runs = runs[runs["tags.kego_debug"] != "true"]

# Hide child fold runs unless --children is requested
if not args.show_children and "tags.mlflow.parentRunId" in runs.columns:
    runs = runs[
        runs["tags.mlflow.parentRunId"].isna()
        | (runs["tags.mlflow.parentRunId"] == "")
    ]
```

- [ ] **Step 6: Add FOLD column to `format_table`**

Update the `format_table` signature and body in `kego/cli/commands/ls.py`:

```python
def format_table(
    runs: pd.DataFrame,
    primary_metric: str,
    exp_names: dict[str, str] | None = None,
    show_metric_name: bool = False,
    show_fold: bool = False,
) -> list[str]:
    """Format experiment runs into a table. Returns list of lines."""
    import pandas as pd

    if runs.empty:
        return ["No experiments found."]

    fallback_metric = _resolve_metric(runs, primary_metric)

    fold_col = f" {'FOLD':<6}" if show_fold else ""
    metric_name_col = f" {'METRIC_NAME':<10}" if show_metric_name else ""
    header = (
        f"{'ID':<8} {'NAME':<26}{fold_col} {'COMPETITION':<20} {'TARGET':<8}"
        f" {'METRIC':>8}{metric_name_col} {'STATUS':<10} {'AGO'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for _, row in runs.iterrows():
        exp_id = str(row.get("tags.kego_id", "?"))[:6]
        name = str(row.get("tags.mlflow.runName", "?"))[:26]
        mlflow_exp_id = str(row.get("experiment_id", ""))
        competition = (exp_names or {}).get(mlflow_exp_id, "?")[:20]
        target = str(row.get("tags.kego_target", "local"))[:8]
        metric_name = str(row.get("tags.kego_primary_metric") or fallback_metric)
        metric = _metric_str(row, fallback_metric)
        status = str(row.get("status", "?"))[:10]
        start = row.get("start_time")
        ago = _ago(start) if start is not None and pd.notna(start) else "?"
        metric_name_cell = f" {metric_name:<10}" if show_metric_name else ""

        fold_cell = ""
        if show_fold:
            parent_run_id = row.get("tags.mlflow.parentRunId", "")
            fold_count = row.get("tags.kego_fold_count", "")
            fold_param = row.get("params.fold", "")
            if parent_run_id:
                fold_val = f"f{fold_param}" if fold_param else "?"
            elif fold_count:
                fold_val = f"{fold_count}×"
            else:
                fold_val = "-"
            fold_cell = f" {fold_val:<6}"

        lines.append(
            f"{exp_id:<8} {name:<26}{fold_cell} {competition:<20} {target:<8}"
            f" {metric:>8}{metric_name_cell} {status:<10} {ago}"
        )

    return lines
```

- [ ] **Step 7: Pass `show_fold` to `format_table` in `_ls`**

In `_ls`, update the `format_table` call:

```python
table_lines = format_table(
    runs, primary_metric, exp_names, args.show_metric_name,
    show_fold=args.show_children,
)
```

- [ ] **Step 8: Run the new tests**

```bash
uv run pytest tests/cli/test_commands_ls.py::test_ls_hides_child_runs_by_default tests/cli/test_commands_ls.py::test_ls_children_flag_shows_child_runs -v
```
Expected: both PASS

- [ ] **Step 9: Run full suite**

```bash
uv run pytest tests/ -q
```
Expected: all pass

- [ ] **Step 10: Commit**

```bash
git add kego/cli/commands/ls.py tests/cli/test_commands_ls.py
git commit -m "feat(ls): hide child fold runs by default; add --children flag with FOLD column"
```

---

## Task 3: Push and verify

- [ ] **Step 1: Push**

```bash
git push
```

- [ ] **Step 2: Manual smoke test**

Submit a real multi-fold job and verify the output:

```bash
# Should show 1 parent row (not 5 fold rows)
uv run kego ls --competition <slug>

# Should show parent + 5 child rows with FOLD column
uv run kego ls --competition <slug> --children
```

Expected `kego ls` output (default):
```
ID       NAME                       COMPETITION          TARGET     METRIC   STATUS     AGO
------------------------------------------------------------------------
abc123   my-run                     birdclef-2026        cluster       —     RUNNING    2m
```

Expected `kego ls --children` output:
```
ID       NAME                       FOLD   COMPETITION          TARGET     METRIC   STATUS     AGO
------------------------------------------------------------------------------------------
abc123   my-run                     5×     birdclef-2026        cluster       —     RUNNING    2m
abc123   my-run fold=0              f0     birdclef-2026        cluster  0.9120    FINISHED    2m
abc123   my-run fold=1              f1     birdclef-2026        cluster  0.9140    FINISHED    2m
...
```

---

## Self-Review

**Spec coverage:**
- ✅ Folds grouped under parent run — Task 1
- ✅ Each fold logs without fold prefix (child runs are independent runs — scripts log `val_rmse` not `fold0_val_rmse`) — inherent in nested run structure
- ✅ `kego ls` shows one row per experiment — Task 2 (child filter)
- ✅ `kego ls --children` shows fold rows — Task 2 (--children flag)
- ✅ Single-fold and local runs unchanged — Task 1 (`multi = len(folds) > 1` guard)
- ✅ `kego logs` unchanged — child runs still have `ray_submission_id`; parent silently skipped by existing `continue` guard

**Placeholder scan:** None found — all steps have complete code.

**Type consistency:** `format_table` gains `show_fold: bool = False` (default False — all existing callers unaffected). `_make_ls_args` gains `show_children=False`.
