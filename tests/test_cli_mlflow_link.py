def test_mlflow_run_link_builds_hash_routed_url():
    from kego.pipeline.cli import _mlflow_run_link

    link = _mlflow_run_link("http://omarchyd:5000", "50", "15389de22ea44e0ca1b96982f6e70fd3")
    assert link == "http://omarchyd:5000/#/experiments/50/runs/15389de22ea44e0ca1b96982f6e70fd3"


def test_mlflow_run_link_strips_trailing_slash():
    from kego.pipeline.cli import _mlflow_run_link

    link = _mlflow_run_link("http://omarchyd:5000/", "50", "abc123")
    assert link == "http://omarchyd:5000/#/experiments/50/runs/abc123"


def test_mlflow_run_link_returns_dash_for_non_http_uri():
    from kego.pipeline.cli import _mlflow_run_link

    assert _mlflow_run_link("sqlite:///offline.db", "50", "abc123") == "-"


def test_format_mlflow_links_lists_each_unique_run_once():
    from kego.pipeline.cli import _format_mlflow_links

    rows = [
        {"version": "76", "mlflow_link": "http://omarchyd:5000/#/experiments/50/runs/abc"},
        {"version": "75", "mlflow_link": "http://omarchyd:5000/#/experiments/50/runs/abc"},
        {"version": "53", "mlflow_link": "http://omarchyd:5000/#/experiments/50/runs/def"},
    ]
    out = _format_mlflow_links(rows)
    assert "v76,75  http://omarchyd:5000/#/experiments/50/runs/abc" in out
    assert "v53  http://omarchyd:5000/#/experiments/50/runs/def" in out


def test_format_mlflow_links_skips_rows_without_a_link():
    from kego.pipeline.cli import _format_mlflow_links

    rows = [{"version": "1", "mlflow": "-"}, {"version": "2", "mlflow_link": "-"}]
    assert _format_mlflow_links(rows) == ""


def test_format_mlflow_links_returns_empty_when_no_rows():
    from kego.pipeline.cli import _format_mlflow_links

    assert _format_mlflow_links([]) == ""
