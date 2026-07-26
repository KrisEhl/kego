"""``kego`` command-line entry point.

Mirrors the usage sketched in the repo README::

    kego run   --config catboostv1 [--fast] [--params models.0.params.lr=0.01]
    kego tune  --config catboostv1 --tune catboost --trials 50
    kego ensemble --from-experiment expA expB
    kego submit   --from-ensemble my_best --message "cat+lgbm"

``--config`` names a YAML under the configs dir; ``--params`` are OmegaConf
dotlist overrides applied on top. Wire as a console script in pyproject:
``[project.scripts] kego = "kego.pipeline.cli:main"``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
import tomllib
from pathlib import Path
from urllib.parse import quote, urlparse

from kego.cli.commands.cancel import cancel_run
from kego.cli.commands.league import dispatch_league, merge_leagues, run_league_locally, show_league_matrix
from kego.cli.commands.logs import run_logs
from kego.cli.commands.train_agent import dispatch_train_agent, train_agent_overrides
from kego.fleet import repo_root as find_repo_root
from kego.pipeline.battle import run_battle_benchmark
from kego.pipeline.config import ModelConfig, PipelineConfig, load_config
from kego.pipeline.executor import get_executor
from kego.pipeline.runner import Pipeline, RunOutcome
from kego.pipeline.tune import HPSpace
from kego.tracking import (
    archive_version,
    default_tracking_uri,
    filter_worse_versions,
    format_leaderboard,
    leaderboard,
    pending_registrations,
    sync_outbox,
    unarchive_version,
)
from kego.tracking.prune import active_model_versions

# mlflow is an optional dependency (see pyproject.toml `[project.optional-dependencies].ml`)
# and is never imported at module top level anywhere in kego — every use below is a local
# `from mlflow.tracking import MlflowClient` so plain CLI usage doesn't hard-require it.


def finite_nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kego")
    sub = parser.add_subparsers(dest="command", required=True)

    common_base = argparse.ArgumentParser(add_help=False)
    common_base.add_argument("--config", help="config name/path (YAML)")
    common_base.add_argument(
        "--executor", choices=["serial", "ray"], default="serial", help="execution backend (default: %(default)s)"
    )
    common_base.add_argument("--force", action="store_true", help="ignore cache; retrain everything")
    common_base.add_argument("--tag", default="", help="experiment tag")
    common_base.add_argument("--task", help="task name (competition slug)")

    common = argparse.ArgumentParser(add_help=False, parents=[common_base])
    common.add_argument("--params", nargs="*", default=[], help="OmegaConf config overrides")

    run = sub.add_parser("run", parents=[common], help="train grid (cached) + ensemble")
    run.add_argument("--fast", action="store_true")
    run.add_argument("--no-ensemble", dest="do_ensemble", action="store_false")
    run.add_argument("--submit", action="store_true")
    run.add_argument("--model", help="ad-hoc model name to run (e.g. catboost)")
    run.add_argument("--hp-tune", action="store_true", help="enable hyperparameter tuning")
    run.add_argument("--hp-params", nargs="*", default=[], help="hyperparameter search space specs")

    ens = sub.add_parser("ensemble", parents=[common], help="re-ensemble stored preds")
    ens.add_argument("--from-experiment", nargs="+")
    ens.add_argument("--from-ensemble")

    tune = sub.add_parser("tune", parents=[common], help="Optuna HP tuning")
    tune.add_argument("--tune", nargs="+", metavar="MODEL")
    tune.add_argument("--trials", type=int, default=50)

    submit = sub.add_parser("submit", parents=[common], help="submit to Kaggle")
    submit.add_argument("version", nargs="?", help="registry model version for simulation-agent submissions")
    submit.add_argument("--from-ensemble")
    submit.add_argument("--message", default="")

    sub.add_parser("status", parents=[common], help="check current training runs")

    cancel = sub.add_parser("cancel", parents=[common], help="kill an active local, Ray, or fleet run")
    cancel.add_argument(
        "target_or_run_id",
        nargs="?",
        help="fleet machine name, 'ray', or MLflow run ID prefix; omit to kill the local run "
        "(falling back to a lone active Ray job)",
    )
    cancel.add_argument("job_id", nargs="?", help="Ray job ID to stop (only used when target_or_run_id is 'ray')")

    cache = sub.add_parser("cache", parents=[common], help="manage cache")
    cache.add_argument("action", choices=["status", "prune"], help="cache action")

    sub.add_parser("submissions", parents=[common], help="list submissions")

    sync = sub.add_parser("sync", help="replay queued checkpoint registrations from the outbox to the hub")
    sync.add_argument("--uri", help="target MLflow URI (default: resolved hub from fleet.toml)")
    sync.add_argument("--list", action="store_true", help="show pending outbox entries without syncing")

    league = sub.add_parser("league", parents=[common], help="run a task-specific model league")
    league.add_argument(
        "--games", type=int, default=200, help="games per pairing in the round-robin (default: %(default)s)"
    )
    league.add_argument("--search-count", type=int, default=10, help="MCTS simulations per move (default: %(default)s)")
    league.add_argument("--workers", type=int, help="parallel game workers (default: CPU count - 2)")
    league.add_argument("--debug", action="store_true", help="show per-game engine output (default: suppressed)")
    league.add_argument(
        "--cache-dir",
        help="cache directory for downloaded registry checkpoints "
        "(default: competitions/<task>/outputs/cached_registry)",
    )
    league.add_argument(
        "--write-ratings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write Elo ratings back to the model registry (default: enabled)",
    )
    league.add_argument(
        "--include-local-mcts",
        action="store_true",
        help="also enter the local outputs/mcts.pth checkpoint as a participant",
    )
    league.add_argument(
        "--partial-save-every",
        type=int,
        default=5000,
        help="save partial league artifacts to MLflow every N games; 0 disables (default: %(default)s)",
    )
    league.add_argument(
        "--stall-timeout",
        type=finite_nonnegative_float,
        default=300,
        help="fail if no game completes for N seconds; 0 disables (default: %(default)s)",
    )
    league.add_argument("--target", help="fleet machine name to dispatch to (rsync + SSH-launch); omit locally")
    league_sub = league.add_subparsers(dest="league_cmd")

    league_matrix = league_sub.add_parser("matrix", parents=[common], help="show a saved task league matrix")
    league_matrix.add_argument("--run-id", help="MLflow run id to show; defaults to latest league run")

    league_merge = league_sub.add_parser("merge", parents=[common], help="merge saved league matrices")
    league_merge.add_argument(
        "--run-ids",
        nargs="*",
        default=[],
        help="MLflow league run ids to merge; defaults to finished unmerged league runs",
    )
    league_merge.add_argument(
        "--latest",
        type=int,
        help="limit automatic selection to the N latest finished unmerged league runs",
    )
    league_merge.add_argument(
        "--write-ratings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write merged Elo ratings back to the model registry (default: enabled)",
    )

    battle = sub.add_parser("battle", parents=[common], help="battle local pokemon agents against each other")
    battle.add_argument("--agent1", help="path to first agent python file")
    battle.add_argument("--agent2", help="path to second agent python file")
    battle.add_argument("--games", type=int, help="number of games to play")
    battle.add_argument("--deck1", help="optional path to deck CSV for agent 1")
    battle.add_argument("--deck2", help="optional path to deck CSV for agent 2")

    train_parser = sub.add_parser(
        "train-agent", parents=[common_base], help="run task-specific simulation agent/policy training"
    )
    train_parser.add_argument(
        "--agent", required=True, help="agent implementation to train (directory under agents/, e.g. mcts)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="target total training iterations; automatically resumes a compatible shorter run",
    )
    train_parser.add_argument("--output", help="path to save the trained model/weights")
    train_parser.add_argument("--init-checkpoint", help="warm-start from a local .pth path or registry:<version>")
    train_parser.add_argument(
        "--num-workers",
        type=int,
        help="parallel rollout/eval workers (default: [train] num_workers in the competition's "
        "kego.toml, else CPU count - 2, capped at the per-phase game count)",
    )
    train_parser.add_argument(
        "--variant", required=True, help="name of the variant configuration under configs/variants/"
    )
    train_parser.add_argument(
        "--target", help="fleet machine name to dispatch to (rsync + SSH-launch); omit to run locally"
    )

    models = sub.add_parser("models", parents=[common], help="show the model-registry Elo standings for a task")
    models.add_argument("--sort-by", default="elo", help="metric tag to rank agents by")
    models.add_argument(
        "--breakdown",
        "-b",
        action="store_true",
        help="show per-opponent win-rate columns (wr_*) instead of the metadata columns",
    )
    models.add_argument(
        "--color", action=argparse.BooleanOptionalAction, default=None, help="color Elo by rating uncertainty"
    )
    models.add_argument(
        "--mlflow",
        action="store_true",
        help="show a direct link to each registry version's originating MLflow run",
    )
    models.add_argument(
        "--extended",
        "-e",
        action="store_true",
        help="show extended metadata columns (machine, git_sha)",
    )
    models_sub = models.add_subparsers(dest="models_cmd")

    models_prune = models_sub.add_parser(
        "prune", parents=[common], help="retire model versions from standings and league runs"
    )
    models_prune.add_argument(
        "versions",
        nargs="*",
        metavar="VERSION",
        help="version numbers to retire — the 'version' column of `kego models` "
        "(e.g. `kego models prune 31 34`); omit when using --drop-worse",
    )
    models_prune.add_argument(
        "--drop-worse",
        action="store_true",
        help="retire statistically worse variants of the same model architecture",
    )
    models_prune.add_argument(
        "--drop-worse-min-games",
        type=int,
        default=20,
        help="minimum games required for a version to be considered rated and subject to drop-worse",
    )
    models_prune.add_argument(
        "--drop-worse-k",
        type=float,
        default=1.96,
        help="confidence multiplier for drop-worse (default %(default)s for a 95%% confidence interval)",
    )

    models_unprune = models_sub.add_parser("unprune", parents=[common], help="restore retired model versions")
    models_unprune.add_argument(
        "versions",
        nargs="+",
        metavar="VERSION",
        help="version numbers to restore — the 'version' column of `kego models` (pruned versions "
        "are hidden from standings; find them in the MLflow model registry under archived versions)",
    )

    logs = sub.add_parser("logs", parents=[common], help="view or tail the logs of an active/past fleet run")
    logs.add_argument(
        "target_or_run_id",
        nargs="?",
        help="fleet machine name or MLflow run ID prefix; if omitted, defaults to local latest log",
    )
    logs.add_argument(
        "run_id",
        nargs="?",
        help="optional run ID prefix (only needed if target_or_run_id is specified as a machine name)",
    )
    logs.add_argument(
        "--tail",
        "-f",
        action="store_true",
        help="tail the log file continuously (like tail -f)",
    )
    logs.add_argument(
        "--lines",
        "-n",
        type=int,
        default=100,
        help="number of lines to print from the end of the log (default: %(default)s)",
    )

    return parser


def detect_task() -> str:
    curr = Path.cwd()
    for parent in [curr, *curr.parents]:
        toml_path = parent / "kego.toml"
        if toml_path.exists():
            try:
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                if "competition" in data:
                    comp = data["competition"]
                    if isinstance(comp, dict):
                        return comp.get("slug") or comp.get("name") or "default"
                    elif isinstance(comp, list):
                        for item in comp:
                            if isinstance(item, dict):
                                if "name" in item:
                                    return item["name"]
                                if "slug" in item:
                                    return item["slug"]
            except Exception:  # noqa: S110
                pass
    return "default"


_train_agent_overrides = train_agent_overrides
_dispatch_train_agent = dispatch_train_agent
_dispatch_league = dispatch_league
_run_logs = run_logs
_run_league_locally = run_league_locally


_show_league_matrix = show_league_matrix


def _load_submission_module(task_name: str):
    script = find_repo_root() / "competitions" / task_name / "submit_leader.py"
    spec = importlib.util.spec_from_file_location(f"{task_name}_submit", script)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load submission helper at {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_merge_leagues = merge_leagues


def _rule_agent_rows(task_name: str) -> list[dict]:
    if task_name != "pokemon-tcg-ai-battle":
        return []
    return [
        {
            "agent": name,
            "version": "rule",
            "type": "rule",
            "elo": str(elo),
            "elo_rd": "30.0",
            "games": "-",
            "deck": deck,
            "rating_status": "anchor",
        }
        for name, elo, deck in [
            ("Mega Abomasnow ex", 1650.0, "abomasnow"),
            ("Dragapult ex", 1520.0, "dragapult"),
            ("Mega Lucario ex", 1450.0, "lucario"),
            ("Zacian ex", 1350.0, "zacian"),
            ("Random", 1200.0, "sample"),
        ]
    ]


def _registry_agent_name(row: dict) -> str:
    if row.get("variant"):
        version = row.get("version", "-")
        epoch = row.get("epoch")
        suffix = f" @{epoch}" if epoch and str(epoch) != "-" else ""
        return f"v{version} {row['variant']}{suffix}"
    if row.get("agent_name"):
        return str(row["agent_name"])

    version = row.get("version", "-")
    deck = row.get("deck")
    model_args = row.get("model_args")
    epoch = row.get("epoch")

    if deck or model_args:
        parts = [f"v{version}", f"mcts-{deck or 'unknown'}"]
        if model_args:
            parts.append(str(model_args).strip("()").replace(", ", "/"))
        if epoch:
            parts.append(f"@{epoch}")
        return " ".join(parts)

    return f"Registry v{version}"


def _numeric_row_value(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def _mlflow_run_link(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Build an MLflow UI link when the tracking URI is browser-accessible."""
    if urlparse(tracking_uri).scheme not in {"http", "https"}:
        return "-"
    return (
        f"{tracking_uri.rstrip('/')}/#/experiments/{quote(str(experiment_id), safe='')}"
        f"/runs/{quote(str(run_id), safe='')}"
    )


def _format_mlflow_links(rows: list[dict]) -> str:
    """A ``v<versions>  <full url>`` line per unique training run, printed after (not
    inside) the leaderboard table: the table row is already wide, and a full run URL
    appended to it wraps mid-URL in most terminals — copying a wrapped line turns the
    wrap into a literal space, corrupting the URL. One URL per line avoids that."""
    by_link: dict[str, list[str]] = {}
    for row in rows:
        link = row.get("mlflow_link")
        if link and link != "-":
            by_link.setdefault(link, []).append(str(row.get("version", "-")))
    if not by_link:
        return ""
    lines = ["", "MLflow run links:"]
    lines.extend(f"  v{','.join(versions)}  {link}" for link, versions in by_link.items())
    return "\n".join(lines)


def _use_color(force: bool | None = None) -> bool:
    if force is not None:
        return force
    return "NO_COLOR" not in os.environ and (
        os.environ.get("FORCE_COLOR") not in (None, "", "0") or sys.stdout.isatty()
    )


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")

    parser = build_parser()
    args = parser.parse_args(argv)

    task_name = getattr(args, "task", None) or detect_task()

    if args.command == "logs":
        return _run_logs(args.target_or_run_id, args.run_id, args.tail, args.lines, task_name)

    if args.command == "cancel":
        return cancel_run(args.target_or_run_id, args.job_id, task_name)

    if args.command == "sync":
        pending = pending_registrations()
        if args.list:
            if not pending:
                print("Outbox empty — nothing queued.")
                return 0
            for entry_dir in pending:
                meta = json.loads((entry_dir / "entry.json").read_text())
                print(
                    f"{entry_dir.name}  model={meta['name']}  checkpoint={meta['checkpoint']}  "
                    f"queued={meta['queued_at']}"
                )
            return 0
        if not pending:
            print("Outbox empty — nothing to sync.")
            return 0
        synced, errors = sync_outbox(uri=getattr(args, "uri", None))
        for entry, model, version in synced:
            print(f"Synced {entry} -> {model} v{version}")
        for err in errors:
            print(f"Failed: {err}")
        return 1 if errors else 0

    if args.command == "models":
        from mlflow.tracking import MlflowClient

        uri = default_tracking_uri()
        models_cmd = getattr(args, "models_cmd", None)

        if models_cmd == "prune":
            client = MlflowClient(tracking_uri=uri)
            versions = list(getattr(args, "versions", []) or [])
            drop_worse = getattr(args, "drop_worse", False)
            if bool(versions) == bool(drop_worse):
                print("Specify either version numbers or --drop-worse (not both, not neither).")
                return 1
            to_archive: list[str] = []
            if drop_worse:
                active = active_model_versions(client, task_name)
                _, dropped = filter_worse_versions(
                    active,
                    min_games=args.drop_worse_min_games,
                    k=args.drop_worse_k,
                )
                to_archive = [str(v.version) for v in dropped]
                if not to_archive:
                    print(f"No worse variants to prune for '{task_name}'.")
                    return 0
                print(f"Drop-worse selected {len(to_archive)} version(s): {', '.join(f'v{v}' for v in to_archive)}")
            else:
                to_archive = [str(v) for v in versions]

            for version in to_archive:
                try:
                    archive_version(client, task_name, version, dropped=drop_worse)
                    print(f"Pruned version {version} of model '{task_name}'.")
                except Exception as e:
                    print(f"Error pruning version {version}: {e}")
                    return 1
            print("Pruned versions are excluded from standings and league runs.")
            return 0

        if models_cmd == "unprune":
            client = MlflowClient(tracking_uri=uri)
            for version in args.versions:
                try:
                    unarchive_version(client, task_name, str(version))
                    print(f"Unpruned version {version} of model '{task_name}'.")
                except Exception as e:
                    print(f"Error unpruning version {version}: {e}")
                    return 1
            print("Restored versions are included back on standings and league runs.")
            return 0

        rows = leaderboard(uri, task_name, sort_by=args.sort_by)
        submission_stats = Pipeline(PipelineConfig(task=task_name)).model_submission_stats()
        mlflow_client = MlflowClient(tracking_uri=uri) if args.mlflow else None
        use_color = _use_color(args.color)
        for row in rows:
            row.setdefault("trained", row.get("completed_iterations") or row.get("epoch") or "-")
            row.setdefault("agent", _registry_agent_name(row))
            row.setdefault("type", "registry")
            row.update(submission_stats.get(str(row.get("version")), {"submitted": "-", "public_rank": "-"}))

            # Combine submitted + public_rank into "kaggle"
            submitted = row.get("submitted", "-")
            public_rank = row.get("public_rank", "-")
            if submitted == "-" or submitted == "no":
                row["kaggle"] = "-"
            elif public_rank == "-":
                row["kaggle"] = submitted
            else:
                row["kaggle"] = f"{submitted} #{public_rank}"

            # Combine epoch + trained into "iters"
            epoch = row.get("epoch")
            trained = row.get("trained")
            if str(epoch) == str(trained) or epoch == "-" or trained == "-":
                row["iters"] = str(trained) if trained else "-"
            else:
                row["iters"] = f"{trained}/{epoch}"

            training_run_id = row.get("training_run_id") or row.get("run_id")
            if mlflow_client and training_run_id:
                try:
                    run = mlflow_client.get_run(training_run_id)
                    # Older registry versions only reference their artifact-registration
                    # run. Do not present that empty run as the metrics-bearing trainer.
                    if not row.get("training_run_id") and not run.data.metrics:
                        row["mlflow"] = "-"
                    else:
                        link = _mlflow_run_link(uri, run.info.experiment_id, training_run_id)
                        row["mlflow"] = training_run_id[:12] if link != "-" else "-"
                        row["mlflow_link"] = link
                except Exception:
                    row["mlflow"] = "-"
        rows = [*rows, *_rule_agent_rows(task_name)]
        rows = sorted(rows, key=lambda row: _numeric_row_value(row, args.sort_by), reverse=True)
        seen: set[str] = set()
        if args.breakdown:
            wr_cols = sorted({k for r in rows for k in r if k.startswith("wr_")})
            base = ["agent", "type", args.sort_by, "gauntlet_avg", *wr_cols]
            if args.extended:
                base.extend(["machine", "git_sha"])
            base.extend(["created", "version"])
        else:
            base = [
                "agent",
                "type",
                args.sort_by,
                "elo_rd",
                "games",
                "gauntlet_avg",
                "kaggle",
                "deck",
                "iters",
            ]
            if args.extended:
                base.extend(["machine", "git_sha"])
            base.extend(["created", "version"])
        cols = [c for c in base if not (c in seen or seen.add(c))]
        if args.mlflow:
            cols.append("mlflow")
        print(f"{task_name} — {len(rows)} agents · tracking {uri}")
        if "elo" in cols and use_color:
            print("elo color: green <= +/-1 RD, yellow <= +/-5 RD, red > +/-5 RD")
        print(format_leaderboard(rows, cols, max_widths={"git_sha": 8}, color_elo=use_color))
        if args.mlflow:
            links = _format_mlflow_links(rows)
            if links:
                print(links)
        return 0

    if args.command == "league":
        league_cmd = getattr(args, "league_cmd", None)
        if league_cmd == "matrix":
            return _show_league_matrix(task_name, getattr(args, "run_id", None))
        if league_cmd == "merge":
            return _merge_leagues(task_name, args.run_ids, args.write_ratings, latest=args.latest)
        target = getattr(args, "target", None)
        if target:
            return _dispatch_league(task_name, target, args)
        return _run_league_locally(task_name, args)

    if args.command == "run" and not args.config and not getattr(args, "model", None):
        parser.error("Either --config or --model must be specified.")

    if args.config:
        try:
            config = load_config(args.config, getattr(args, "params", []), task_name=task_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            return 1
    else:
        config = PipelineConfig(task=task_name)

    if getattr(args, "model", None):
        model_params = {}
        if args.params:
            for p in args.params:
                if ":" in p:
                    k, v = p.split(":", 1)
                elif "=" in p:
                    k, v = p.split("=", 1)
                else:
                    k, v = p, "true"

                try:
                    if v.lower() == "true":
                        val = True
                    elif v.lower() == "false":
                        val = False
                    elif "." in v:
                        val = float(v)
                    else:
                        val = int(v)
                except ValueError:
                    val = v
                model_params[k] = val

        hp_space_dict = {}
        if getattr(args, "hp_params", None):
            for hp_spec in args.hp_params:
                hp = HPSpace.parse(hp_spec)
                hp_space_dict[hp.name] = {
                    "type": hp.type,
                    "low": hp.low,
                    "high": hp.high,
                    "log": hp.log,
                    "choices": hp.choices,
                }

        model_config = ModelConfig(
            name=args.model,
            hyper_params=model_params,
            hp_space=hp_space_dict,
        )
        config = dataclasses.replace(config, models=[model_config])

    if getattr(args, "hp_tune", False):
        config = dataclasses.replace(config, tune=True)

    if getattr(args, "do_ensemble", None) is not None:
        config = dataclasses.replace(config, do_ensemble=args.do_ensemble)
    if getattr(args, "force", False):
        config = dataclasses.replace(config, force=args.force)

    executor_kind = getattr(args, "executor", "serial")
    executor = get_executor(executor_kind)

    pipeline = Pipeline(config, executor=executor)

    if args.command == "run":
        try:
            outcome = pipeline.run()
            if getattr(args, "submit", False):
                pipeline.submit(outcome, message=getattr(args, "tag", ""))
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "tune":
        models_to_tune = args.tune if getattr(args, "tune", None) else [m.name for m in config.models]
        try:
            pipeline.tune(models=models_to_tune)
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "ensemble":
        try:
            pipeline.ensemble(
                experiments=args.from_experiment,
                ensemble_tag=args.tag,
            )
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "submit":
        message = args.message
        if task_name == "pokemon-tcg-ai-battle":
            prepared = _load_submission_module(task_name).prepare_submission(args.version)
            message = message or prepared["message"]
        elif args.version:
            print(f"Error: registry version arguments are not supported for task {task_name!r}.")
            return 1
        outcome = RunOutcome(predictions=[])
        try:
            pipeline.submit(outcome, message=message)
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        return 0
    elif args.command == "status":
        pipeline.status()
        return 0
    elif args.command == "cache":
        pipeline.cache(args.action)
        return 0
    elif args.command == "submissions":
        pipeline.submissions()
        return 0
    elif args.command == "battle":
        # Resolve battle parameters: CLI arguments take precedence over config file values
        agent1 = getattr(args, "agent1", None)
        agent2 = getattr(args, "agent2", None)
        games = getattr(args, "games", None)
        deck1 = getattr(args, "deck1", None)
        deck2 = getattr(args, "deck2", None)

        if args.config:
            battle_cfg = config.battle
            if agent1 is None:
                agent1 = battle_cfg.agent1
            if agent2 is None:
                agent2 = battle_cfg.agent2
            if games is None:
                games = battle_cfg.games
            if deck1 is None:
                deck1 = battle_cfg.deck1
            if deck2 is None:
                deck2 = battle_cfg.deck2

        # Defaults
        if games is None:
            games = 10

        if not agent1 or not agent2:
            parser.error(
                "Both --agent1 and --agent2 must be specified (either via CLI arguments or in the --config file)."
            )

        try:
            run_battle_benchmark(
                agent1_path=agent1,
                agent2_path=agent2,
                num_games=games,
                deck1_path=deck1,
                deck2_path=deck2,
            )
        except Exception as e:
            print(f"Error running battle benchmark: {e}")
            return 1
        return 0

    elif args.command == "train-agent":
        epochs = getattr(args, "epochs", None)
        output = getattr(args, "output", None)
        overrides = _train_agent_overrides(args)
        target = getattr(args, "target", None)
        if target and target not in ("local", "cluster"):
            return _dispatch_train_agent(task_name, target, epochs, output, overrides)
        try:
            pipeline.train_agent(epochs=epochs, output_path=output, **overrides)
        except NotImplementedError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error during training: {e}")
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
