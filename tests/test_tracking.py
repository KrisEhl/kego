import pytest

from kego.tracking import Tracker, resolve_tracking_uri


def _up(*ok):
    return lambda uri: uri in ok


def test_uses_explicit_when_reachable():
    uri = resolve_tracking_uri(
        explicit="http://central:5000",
        hub="http://hub:5000",
        reachable=_up("http://central:5000", "http://hub:5000"),
        offline="/home/x/off.db",
    )
    assert uri == "http://central:5000"


def test_falls_back_to_hub_when_no_explicit():
    uri = resolve_tracking_uri(
        explicit=None, hub="http://hub:5000", reachable=_up("http://hub:5000"), offline="/home/x/off.db"
    )
    assert uri == "http://hub:5000"


def test_explicit_down_falls_through_to_hub():
    uri = resolve_tracking_uri(
        explicit="http://central:5000",
        hub="http://hub:5000",
        reachable=_up("http://hub:5000"),
        offline="/home/x/off.db",
    )
    assert uri == "http://hub:5000"


def test_offline_sqlite_when_nothing_reachable():
    uri = resolve_tracking_uri(
        explicit="http://central:5000",
        hub="http://hub:5000",
        reachable=_up(),
        offline="/home/x/off.db",
    )
    assert uri == "sqlite:////home/x/off.db"


def test_tracker_logs_metric_and_tags(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    with Tracker.open(uri, experiment="exp1", run_name="r1", tags={"machine": "m5"}) as t:
        t.log_metric("gauntlet_avg", 68.5, step=1)
        t.set_tags({"git_sha": "abc123"})

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("exp1")
    runs = mlflow.search_runs([exp.experiment_id])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["metrics.gauntlet_avg"] == 68.5
    assert row["tags.machine"] == "m5"
    assert row["tags.git_sha"] == "abc123"


def test_tracker_noop_never_crashes():
    t = Tracker.noop()
    t.log_metric("x", 1.0, step=0)
    t.set_tags({"a": "b"})
    t.close()  # must not raise


def test_tracker_open_failure_returns_safe_noop():
    # unwritable sqlite path -> open fails -> safe no-op, never raises
    t = Tracker.open("sqlite:////nonexistent-dir-xyz-42/ml.db", experiment="exp", run_name="r")
    t.log_metric("x", 1.0)
    t.set_tags({"k": "v"})
    t.close()


def test_default_tracking_uri_prefers_env(monkeypatch):
    from kego.tracking import default_tracking_uri

    monkeypatch.setenv("KEGO_MLFLOW", "http://central:5000")
    assert default_tracking_uri(fleet_path="/nonexistent.toml") == "http://central:5000"


def test_default_tracking_uri_uses_fleet_hub(monkeypatch, tmp_path):
    from kego.tracking import default_tracking_uri

    monkeypatch.delenv("KEGO_MLFLOW", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    fp = tmp_path / "fleet.toml"
    fp.write_text('[hub]\nname = "omarchyd"\nmlflow = "http://hub:5000"\n')
    assert default_tracking_uri(fleet_path=fp) == "http://hub:5000"


def test_default_tracking_uri_offline_fallback(monkeypatch, tmp_path):
    from kego.tracking import default_tracking_uri

    monkeypatch.delenv("KEGO_MLFLOW", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = default_tracking_uri(fleet_path=tmp_path / "nope.toml")
    assert uri.startswith("sqlite:///")
