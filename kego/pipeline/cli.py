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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kego")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="config name/path (YAML)")
    common.add_argument("--params", nargs="*", default=[], help="overrides or parameters")
    common.add_argument("--executor", choices=["serial", "ray"], default="serial")
    common.add_argument("--force", action="store_true", help="ignore cache; retrain everything")
    common.add_argument("--tag", default="", help="experiment tag")
    common.add_argument("--task", help="task name (competition slug)")

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
    submit.add_argument("--from-ensemble")
    submit.add_argument("--message", default="")

    sub.add_parser("status", parents=[common], help="check current training runs")

    cache = sub.add_parser("cache", parents=[common], help="manage cache")
    cache.add_argument("action", choices=["status", "prune"], help="cache action")

    sub.add_parser("submissions", parents=[common], help="list submissions")

    battle = sub.add_parser("battle", parents=[common], help="battle local pokemon agents against each other")
    battle.add_argument("--agent1", help="path to first agent python file")
    battle.add_argument("--agent2", help="path to second agent python file")
    battle.add_argument("--games", type=int, help="number of games to play")
    battle.add_argument("--deck1", help="optional path to deck CSV for agent 1")
    battle.add_argument("--deck2", help="optional path to deck CSV for agent 2")

    train_parser = sub.add_parser(
        "train-agent", parents=[common], help="run task-specific simulation agent/policy training"
    )
    train_parser.add_argument("--epochs", type=int, help="number of training epochs or iterations")
    train_parser.add_argument("--output", help="path to save the trained model/weights")
    train_parser.add_argument("--init-checkpoint", help="warm-start from a local .pth path or registry:<version>")
    train_parser.add_argument("--deck-file", help="training deck CSV path relative to the competition directory")
    train_parser.add_argument("--self-play-games", type=int, help="self-play games per training iteration")
    train_parser.add_argument("--search-count", type=int, help="MCTS search count target for self-play")
    train_parser.add_argument("--train-steps", type=int, help="optimizer steps per training iteration")
    train_parser.add_argument("--num-workers", type=int, help="parallel rollout/eval workers")
    train_parser.add_argument(
        "--target", help="fleet machine name to dispatch to (rsync + SSH-launch); omit to run locally"
    )

    models = sub.add_parser("models", parents=[common], help="show the model-registry leaderboard for a task")
    models.add_argument("--sort-by", default="elo", help="metric tag to rank agents by")
    models.add_argument(
        "--breakdown",
        "-b",
        action="store_true",
        help="show per-opponent win-rate columns (wr_*) instead of the metadata columns",
    )

    return parser


def detect_task() -> str:
    from pathlib import Path

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

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


def _train_agent_overrides(args) -> dict:
    keys = ["init_checkpoint", "deck_file", "self_play_games", "search_count", "train_steps", "num_workers"]
    return {k: getattr(args, k, None) for k in keys if getattr(args, k, None) is not None}


def _dispatch_train_agent(task_name: str, target: str, epochs: int | None, output: str | None, overrides: dict) -> int:
    """Ship the local tree to fleet machine ``target`` and SSH-launch training there (§5.4)."""
    from pathlib import Path

    from kego.dispatch import DEFAULT_EXCLUDES, dispatch, other_competition_excludes
    from kego.fleet import git_sha, load_fleet, machine_name
    from kego.tracking import create_run, default_tracking_uri

    repo_root = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), Path.cwd())
    fleet_path = repo_root / "fleet.toml"
    if not fleet_path.exists():
        print(f"No fleet.toml at {fleet_path}; cannot dispatch --target {target}.")
        return 1
    try:
        machine = load_fleet(fleet_path).machine(target)
    except KeyError as e:
        print(f"Error: {e}")
        return 1

    uri = default_tracking_uri(fleet_path)
    tags = {
        "machine": machine.name,
        "task": task_name,
        "git_sha": git_sha(repo_root),
        "dispatched_from": machine_name(),
        "target": target,
    }
    run_id = create_run(uri, experiment=task_name, run_name=f"{machine.name}-{task_name}", tags=tags)
    if not run_id:
        print(
            f"Could not create an MLflow run at {uri} (hub unreachable). Offline dispatch is a later phase; aborting."
        )
        return 1

    cmd_args = ["train-agent", "--task", task_name]
    if epochs:
        cmd_args += ["--epochs", str(epochs)]
    if output:
        cmd_args += ["--output", output]
    for key, flag in [
        ("init_checkpoint", "--init-checkpoint"),
        ("deck_file", "--deck-file"),
        ("self_play_games", "--self-play-games"),
        ("search_count", "--search-count"),
        ("train_steps", "--train-steps"),
        ("num_workers", "--num-workers"),
    ]:
        if key in overrides:
            cmd_args += [flag, str(overrides[key])]

    excludes = DEFAULT_EXCLUDES + other_competition_excludes(repo_root, keep=task_name)
    print(f"Dispatching {task_name} to {machine.name} ({machine.ssh}) — run {run_id}")
    try:
        dispatch(machine, cmd_args, run_id=run_id, local_dir=str(repo_root), excludes=excludes)
    except Exception as e:
        print(f"Dispatch failed: {e}")
        return 1
    print(f"Launched on {machine.name}. Track metrics in MLflow at: {uri}")
    print(f"To view remote logs, run:  ssh {machine.ssh} 'cat ~/.kego/logs/{run_id}.log'")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    import dataclasses

    from kego.pipeline.config import ModelConfig, PipelineConfig, load_config
    from kego.pipeline.executor import get_executor
    from kego.pipeline.runner import Pipeline

    task_name = getattr(args, "task", None) or detect_task()

    if args.command == "models":
        from kego.tracking import default_tracking_uri, format_leaderboard, leaderboard

        uri = default_tracking_uri()
        rows = leaderboard(uri, task_name, sort_by=args.sort_by)
        seen: set[str] = set()
        if args.breakdown:
            wr_cols = sorted({k for r in rows for k in r if k.startswith("wr_")})
            base = [args.sort_by, "gauntlet_avg", *wr_cols, "version"]
        else:
            base = [args.sort_by, "elo_rd", "games", "gauntlet_avg", "deck", "epoch", "machine", "git_sha", "version"]
        cols = [c for c in base if not (c in seen or seen.add(c))]
        print(f"{task_name} — {len(rows)} agents · tracking {uri}")
        print(format_leaderboard(rows, cols))
        return 0

    if args.command == "run" and not args.config and not getattr(args, "model", None):
        parser.error("Either --config or --model must be specified.")

    if args.config:
        config = load_config(args.config, args.params, task_name=task_name)
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
            from kego.pipeline.tune import HPSpace

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
        from kego.pipeline.runner import RunOutcome

        outcome = RunOutcome(predictions=[])
        try:
            pipeline.submit(outcome, message=args.message)
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
            from kego.pipeline.battle import run_battle_benchmark

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
