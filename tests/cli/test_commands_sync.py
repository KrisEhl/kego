import argparse

import mlflow

from kego.cli.commands.sync import _sync


def _make_sync_args(source: str, dest: str, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(source=source, dest=dest, dry_run=dry_run)


def _seed_source(uri: str) -> str:
    """Create one finished run with params + a metric history. Returns its run_id."""
    from mlflow.tracking import MlflowClient

    c = MlflowClient(tracking_uri=uri)
    exp_id = c.create_experiment("offline-comp")
    run = c.create_run(exp_id, run_name="offline-run", tags={"kego_id": "abc123", "kego_target": "local"})
    rid = run.info.run_id
    c.log_param(rid, "model", "xgboost")
    for step, val in enumerate([0.9, 0.7, 0.55]):  # a 3-point learning curve
        c.log_metric(rid, "val_rmse", val, step=step)
    c.set_terminated(rid, status="FINISHED")
    return rid


def _dest_runs(uri: str):
    from mlflow.tracking import MlflowClient

    c = MlflowClient(tracking_uri=uri)
    out = []
    for e in c.search_experiments():
        out.extend(c.search_runs([e.experiment_id]))
    return c, out


def test_sync_copies_run_with_params_and_metric_history(tmp_path, capsys):
    src = f"sqlite:///{tmp_path}/offline.db"
    dst = f"sqlite:///{tmp_path}/cluster.db"
    src_rid = _seed_source(src)

    rc = _sync(_make_sync_args(src, dst), [])
    assert rc == 0

    client, runs = _dest_runs(dst)
    copied = [r for r in runs if r.data.tags.get("kego_synced_from") == src_rid]
    assert len(copied) == 1
    r = copied[0]
    assert r.data.params["model"] == "xgboost"
    assert r.info.status == "FINISHED"
    # full metric history preserved (the learning curve), not just the last point
    history = client.get_metric_history(r.info.run_id, "val_rmse")
    assert [m.value for m in sorted(history, key=lambda m: m.step)] == [0.9, 0.7, 0.55]

    mlflow.set_tracking_uri("")  # reset global state


def test_sync_is_idempotent(tmp_path):
    src = f"sqlite:///{tmp_path}/offline.db"
    dst = f"sqlite:///{tmp_path}/cluster.db"
    _seed_source(src)

    _sync(_make_sync_args(src, dst), [])
    _sync(_make_sync_args(src, dst), [])  # second run must not duplicate

    _, runs = _dest_runs(dst)
    assert len([r for r in runs if r.data.tags.get("kego_synced_from")]) == 1

    mlflow.set_tracking_uri("")


def test_sync_dry_run_writes_nothing(tmp_path, capsys):
    src = f"sqlite:///{tmp_path}/offline.db"
    dst = f"sqlite:///{tmp_path}/cluster.db"
    _seed_source(src)

    _sync(_make_sync_args(src, dst, dry_run=True), [])
    out = capsys.readouterr().out
    assert "Would copy 1" in out

    _, runs = _dest_runs(dst)
    assert runs == []

    mlflow.set_tracking_uri("")
