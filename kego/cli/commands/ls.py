from __future__ import annotations

import argparse
import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _ago(start: datetime.datetime) -> str:
    delta = datetime.datetime.now(tz=datetime.timezone.utc) - start
    hours = int(delta.total_seconds() // 3600)
    if hours >= 24:
        return start.to_pydatetime().astimezone().strftime("%Y-%m-%d %H:%M")
    if hours > 0:
        return f"{hours}h"
    return f"{int(delta.total_seconds() // 60)}m"


_SKIP_METRICS = {"epoch", "loss", "train_loss", "val_loss", "lr", "learning_rate"}
_STATUSES = ["running", "finished", "failed", "killed"]
_CHILD_STATUS_PRIORITY = ["RUNNING", "SCHEDULED", "FAILED", "KILLED", "FINISHED"]


def _local_resources() -> str:
    """One-line CPU/MEM summary for the local machine."""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return f"CPU {cpu:.0f}%  MEM {mem.used / 1e9:.0f}/{mem.total / 1e9:.0f} GB"
    except Exception:
        return "unavailable"


def _cluster_resources(ray_address: str) -> str:
    """One-line GPU/CPU summary fetched from the Ray dashboard."""
    import json
    import urllib.request

    url = ray_address.rstrip("/") + "/nodes?view=summary"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except Exception:
        return "offline"

    parts: list[str] = []
    for node in data.get("data", {}).get("summary", []):
        host = node.get("hostname", node.get("ip", "?"))
        cpu = node.get("cpu", "?")
        node_strs = [f"CPU {cpu}%"]
        for g in node.get("gpus", []):
            name = g.get("name", "GPU").replace("NVIDIA GeForce ", "")
            util = g.get("utilizationGpu", "?")
            used = g.get("memoryUsed", 0) / 1024
            total = g.get("memoryTotal", 0) / 1024
            node_strs.append(f"{name} {util}% {used:.0f}/{total:.0f} GB")
        parts.append(f"{host}: {' · '.join(node_strs)}")

    return "  |  ".join(parts) if parts else "no nodes"


def _parse_since(value: str) -> int:
    """Parse a duration string to a UTC millisecond cutoff timestamp.

    Accepted formats: 30m, 2h, 7d
    """
    units = {"m": "minutes", "h": "hours", "d": "days"}
    unit = value[-1]
    if unit not in units or not value[:-1].isdigit():
        raise argparse.ArgumentTypeError(f"Invalid --since value '{value}'. Use e.g. 30m, 2h, 7d.")
    delta = datetime.timedelta(**{units[unit]: int(value[:-1])})
    cutoff = datetime.datetime.now(tz=datetime.timezone.utc) - delta
    return int(cutoff.timestamp() * 1000)


def _resolve_metric(runs: pd.DataFrame, primary_metric: str) -> str:
    """Return primary_metric if it has data, otherwise the first non-bookkeeping metric."""
    preferred = f"metrics.{primary_metric}"
    if preferred in runs.columns and runs[preferred].notna().any():
        return primary_metric
    for col in runs.columns:
        name = col[len("metrics.") :]
        if col.startswith("metrics.") and name not in _SKIP_METRICS and runs[col].notna().any():
            return name
    return primary_metric


def _metric_str(row: pd.Series, fallback_metric: str, prefer_fallback: bool = False) -> str:
    """Return formatted metric value for a single run."""
    import pandas as pd

    # Unscoped views can mix competitions, so keep per-run tags there. A scoped
    # competition view uses the current competition config to handle corrected metrics.
    metric_name = fallback_metric if prefer_fallback else str(row.get("tags.kego_primary_metric") or fallback_metric)
    if not metric_name:
        metric_name = fallback_metric
    val = row.get(f"metrics.{metric_name}")
    return f"{val:.4f}" if val is not None and pd.notna(val) else "—"


def _derive_parent_statuses(runs: pd.DataFrame) -> pd.DataFrame:
    """Set displayed parent status from child run statuses."""
    import pandas as pd

    required = {"run_id", "status", "tags.kego_is_parent", "tags.mlflow.parentRunId"}
    if runs.empty or not required.issubset(runs.columns):
        return runs

    runs = runs.copy()
    for parent_id, children in runs.groupby("tags.mlflow.parentRunId", dropna=True):
        if not parent_id or pd.isna(parent_id):
            continue
        statuses = {str(status).upper() for status in children["status"].dropna()}
        if not statuses:
            continue
        derived = next((status for status in _CHILD_STATUS_PRIORITY if status in statuses), None)
        if derived is None:
            continue
        parent_mask = (runs["run_id"] == parent_id) & (runs["tags.kego_is_parent"].fillna("") == "true")
        runs.loc[parent_mask, "status"] = derived

    return runs


def format_table(
    runs: pd.DataFrame,
    primary_metric: str,
    exp_names: dict[str, str] | None = None,
    show_metric_name: bool = False,
    show_fold: bool = False,
    prefer_primary_metric: bool = False,
    ray_status: dict[str, str] | None = None,
) -> list[str]:
    """Format experiment runs into a table. Returns list of lines."""
    import pandas as pd

    if runs.empty:
        return ["No experiments found."]

    fallback_metric = _resolve_metric(runs, primary_metric)

    fold_col = f" {'FOLD':<6}" if show_fold else ""
    metric_name_col = f" {'METRIC_NAME':<10}" if show_metric_name else ""
    ray_col = f" {'RAY':<9}" if ray_status is not None else ""
    header = (
        f"{'ID':<8} {'NAME':<26}{fold_col} {'COMPETITION':<20} {'TARGET':<8}"
        f" {'METRIC':>8}{metric_name_col} {'STATUS':<10}{ray_col} {'AGO'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for _, row in runs.iterrows():
        exp_id = str(row.get("tags.kego_id", "?"))[:6]
        name = str(row.get("tags.mlflow.runName", "?"))[:26]
        mlflow_exp_id = str(row.get("experiment_id", ""))
        competition = (exp_names or {}).get(mlflow_exp_id, "?")[:20]
        target = str(row.get("tags.kego_target", "local"))[:8]
        metric_name = (
            fallback_metric if prefer_primary_metric else str(row.get("tags.kego_primary_metric") or fallback_metric)
        )
        metric = _metric_str(row, fallback_metric, prefer_fallback=prefer_primary_metric)
        status = str(row.get("status", "?"))[:10]
        start = row.get("start_time")
        ago = _ago(start) if start is not None and pd.notna(start) else "?"
        metric_name_cell = f" {metric_name:<10}" if show_metric_name else ""
        fold_cell = ""
        is_child = False
        if show_fold:
            parent_run_id = row.get("tags.mlflow.parentRunId", "")
            fold_count = row.get("tags.kego_fold_count", "")
            fold_param = row.get("params.fold", "")
            is_child = pd.notna(parent_run_id) and bool(parent_run_id)
            if is_child:
                fold_val = f"f{fold_param}" if (pd.notna(fold_param) and fold_param) else "?"
            elif pd.notna(fold_count) and fold_count:
                fold_val = f"{fold_count}×"  # noqa: RUF001
            else:
                fold_val = "-"
            fold_cell = f" {fold_val:<6}"

        # Indent child rows: blank the ID and prefix the name with └─
        if is_child:
            row_id = " " * 6
            row_name = ("└─ " + str(row.get("tags.mlflow.runName", "?"))[:23])[:26]
        else:
            row_id = exp_id
            row_name = name

        ray_cell = ""
        if ray_status is not None:
            sub = row.get("tags.ray_submission_id")
            ray_state = ray_status.get(sub, "-") if (pd.notna(sub) and sub) else "-"
            ray_cell = f" {ray_state:<9}"

        lines.append(
            f"{row_id:<8} {row_name:<26}{fold_cell} {competition:<20} {target:<8}"
            f" {metric:>8}{metric_name_cell} {status:<10}{ray_cell} {ago}"
        )

    return lines


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = subparsers.add_parser("ls", help="List and compare experiments")
    p.add_argument("--name", metavar="SUBSTR", help="Filter by run name prefix")
    p.add_argument(
        "--status",
        choices=_STATUSES,
        metavar="STATUS",
        help=f"Filter by run status: {', '.join(_STATUSES)}",
    )
    p.add_argument(
        "--target",
        choices=["local", "cluster"],
        metavar="TARGET",
        help="Filter by compute target: local, cluster",
    )
    p.add_argument(
        "--competition",
        metavar="SLUG",
        help="Filter by competition slug (e.g. birdclef-2026)",
    )
    p.add_argument(
        "--since",
        metavar="DURATION",
        help="Only show runs started within this window, e.g. 30m, 2h, 7d",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Max runs to show (default: 50)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Include debug runs",
    )
    p.add_argument(
        "--metric-name",
        action="store_true",
        dest="show_metric_name",
        help="Show metric name column",
    )
    p.add_argument(
        "--ray",
        action="store_true",
        dest="show_ray",
        help="Add a RAY column with each run's Ray job state (combines ls + status)",
    )
    p.set_defaults(func=_ls)


def _ls(args: argparse.Namespace, extra_args: list[str]) -> int:
    import logging
    import os

    import mlflow

    from kego.cli import config as cfg_module

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)

    repo_root = cfg_module.find_repo_root()
    competition_dir = cfg_module.find_competition_dir_by_slug(args.competition, repo_root) if args.competition else None
    config = cfg_module.load_config(repo_root=repo_root, competition_dir=competition_dir)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config.cluster.mlflow_uri
    mlflow.set_tracking_uri(tracking_uri)

    filter_parts: list[str] = []
    if args.name:
        filter_parts.append(f"tags.`mlflow.runName` LIKE '{args.name}%'")
    if args.status:
        filter_parts.append(f"attributes.status = '{args.status.upper()}'")
    if args.target:
        filter_parts.append(f"tags.kego_target = '{args.target}'")
    if args.since:
        try:
            filter_parts.append(f"attributes.start_time > {_parse_since(args.since)}")
        except argparse.ArgumentTypeError as e:
            print(str(e))
            return 1

    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Scope experiment lookup when --competition is given to avoid fetching all
        if args.competition:
            scoped = client.search_experiments(filter_string=f"name = '{args.competition}'")
            if not scoped:
                print(f"No experiments found for competition: {args.competition}")
                return 0
            exp_names = {e.experiment_id: e.name for e in scoped}
            experiment_ids: list[str] | None = list(exp_names)
        else:
            all_exps = client.search_experiments()
            exp_names = {e.experiment_id: e.name for e in all_exps}
            experiment_ids = None

        search_kwargs: dict = {
            "filter_string": filter_string,
            "order_by": ["start_time DESC"],
            "max_results": args.limit,
        }
        if experiment_ids is not None:
            search_kwargs["experiment_ids"] = experiment_ids
        else:
            search_kwargs["search_all_experiments"] = True

        runs = mlflow.search_runs(**search_kwargs)
    except Exception:
        print(
            f"Cannot reach MLflow at {tracking_uri} — is the cluster online?\n"
            "  Start cluster : uv run kego cluster start\n"
            "  Test locally  : MLFLOW_TRACKING_URI=sqlite:///local.db uv run kego ls"
        )
        return 1

    import pandas as pd

    # Filter debug runs in Python — MLflow filter doesn't handle missing tags correctly
    if not args.show_all and "tags.kego_debug" in runs.columns:
        runs = runs[runs["tags.kego_debug"] != "true"]

    # When child runs appear without their parent (e.g. due to --status filter),
    # fetch the missing parent runs so the nested structure stays intact.
    if "tags.mlflow.parentRunId" in runs.columns:
        child_parent_ids = runs["tags.mlflow.parentRunId"].dropna().pipe(lambda s: s[s != ""]).unique().tolist()
        present_ids = set(runs["run_id"].tolist()) if "run_id" in runs.columns else set()
        missing = [pid for pid in child_parent_ids if pid not in present_ids]
        if missing:
            parent_rows = []
            for pid in missing:
                try:
                    pr = client.get_run(pid)
                    row: dict = {
                        "run_id": pr.info.run_id,
                        "experiment_id": pr.info.experiment_id,
                        "status": pr.info.status,
                        "start_time": pd.Timestamp(pr.info.start_time, unit="ms", tz="UTC"),
                    }
                    row.update({f"tags.{k}": v for k, v in pr.data.tags.items()})
                    row.update({f"params.{k}": v for k, v in pr.data.params.items()})
                    row.update({f"metrics.{k}": v for k, v in pr.data.metrics.items()})
                    parent_rows.append(row)
                except Exception as exc:
                    logging.getLogger(__name__).debug("Could not fetch parent run %s: %s", pid, exc)
            if parent_rows:
                runs = pd.concat([runs, pd.DataFrame(parent_rows)], ignore_index=True)

    runs = _derive_parent_statuses(runs)

    # Re-sort so each parent row appears immediately before its children.
    # Groups are ordered by parent start_time DESC; within a group parent comes first.
    if "tags.kego_is_parent" in runs.columns and "tags.mlflow.parentRunId" in runs.columns:
        parent_times = dict(
            zip(
                runs.loc[runs["tags.kego_is_parent"].fillna("") == "true", "run_id"],
                runs.loc[runs["tags.kego_is_parent"].fillna("") == "true", "start_time"],
            )
        )
        runs["_group_time"] = runs.apply(
            lambda r: (
                parent_times.get(r.get("tags.mlflow.parentRunId", ""), r["start_time"])
                if (pd.notna(r.get("tags.mlflow.parentRunId", "")) and r.get("tags.mlflow.parentRunId", ""))
                else r["start_time"]
            ),
            axis=1,
        )
        runs["_is_parent"] = (runs["tags.kego_is_parent"].fillna("") == "true").astype(int)
        runs = runs.sort_values(["_group_time", "_is_parent"], ascending=[False, False], ignore_index=True).drop(
            columns=["_group_time", "_is_parent"]
        )

    primary_metric = "metric"
    if config.competition:
        primary_metric = config.competition.primary_metric

    # Auto-show FOLD column when the result contains any nested runs
    has_nested = ("tags.mlflow.parentRunId" in runs.columns and runs["tags.mlflow.parentRunId"].notna().any()) or (
        "tags.kego_is_parent" in runs.columns and runs["tags.kego_is_parent"].notna().any()
    )

    ray_status = None
    if args.show_ray:
        from kego.cli import ray

        ray_status = ray.job_statuses(config.cluster.ray_address)

    table_lines = format_table(
        runs,
        primary_metric,
        exp_names,
        args.show_metric_name,
        show_fold=has_nested,
        prefer_primary_metric=bool(args.competition and config.competition),
        ray_status=ray_status,
    )
    for line in table_lines:
        print(line)

    sep_width = len(table_lines[0]) if table_lines else 80
    print("-" * sep_width)
    local = _local_resources()
    cluster = _cluster_resources(config.cluster.ray_address)
    print(f"  local  {local}  │  cluster  {cluster}")

    return 0
