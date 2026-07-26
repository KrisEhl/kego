from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch

comp_dir = Path(__file__).parent.resolve()
repo_root = comp_dir.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(comp_dir))

from kego.fleet import git_sha, machine_name
from kego.pipeline.battle import load_agent, load_deck, locate_cg_dir
from kego.timing import DEFAULT, Timings, timed, timer
from kego.tracking import (
    Tracker,
    default_tracking_uri,
    register_checkpoint_or_queue,
    resolve_training_resume,
)

# NOTE: MLflow stays locally imported where used so telemetry remains optional.

cg_parent = locate_cg_dir()
sys.path.insert(0, str(cg_parent))

from agents.mcts import (
    MAX_ACTION_COMBINATIONS,
    MODEL_ARGS,
    RESULT_DRAW,
    Child,
    Node,
    PolicyValueNet,
    SparseVector,
    build_children,
    create_node,
    encode_actions,
    encode_state,
    ensure_tensors,
    enumerate_action_combinations,
    infer_opponent_hidden_cards,
    model_args_from_state_dict,
    select_child,
)

from cg.api import (
    SearchState,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start


class LearnSample:
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec: SparseVector):
        self.value = value
        self.policy = policy
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


class LearnInput:
    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)


@dataclass(frozen=True, slots=True)
class PendingLeaf:
    node: Node
    path: tuple[Node, ...]
    actions: list[list[int]]
    encoder: SparseVector
    decoder: SparseVector
    player_index: int
    root_player_index: int
    virtual_loss: float

    def add_virtual_loss(self) -> None:
        for node in self.path:
            node.visit += 1
            node.total -= self.virtual_loss

    def resolve(self, value: float, policy: list[float]) -> None:
        for node in self.path:
            node.visit -= 1
            node.total += self.virtual_loss
        resolved_value = -value if self.player_index != self.root_player_index else value
        self.node.value = resolved_value
        self.node.backprop(resolved_value)
        build_children(self.node, self.actions, policy)


@dataclass(frozen=True, slots=True)
class CompletedLeaf:
    node: Node


class OpponentKind(StrEnum):
    RANDOM = "random"
    RULE = "rule"
    SELF = "self"


@dataclass(frozen=True, slots=True)
class GauntletPayload:
    model_state: dict
    opponent_kind: OpponentKind
    opponent_agent_path: str | None
    opponent_deck_path: str | None
    agent_deck: list[int]
    game_indices: list[int]
    seed: int
    search_count: int
    best_state: dict | None


@dataclass(frozen=True, slots=True)
class OpponentSpec:
    kind: OpponentKind
    name: str
    agent_path: str | None = None
    deck_path: str | None = None

    @classmethod
    def parse(cls, value: str) -> OpponentSpec:
        if value in (OpponentKind.RANDOM, OpponentKind.SELF):
            return cls(OpponentKind(value), value)
        if value.startswith("rule:"):
            name = value.split(":", 1)[1]
            return cls(
                OpponentKind.RULE,
                name,
                str(comp_dir / "agents" / f"{name}.py"),
                str(comp_dir / "decks" / f"{name}.csv"),
            )
        raise ValueError(f"unknown opponent specification: {value!r}")

    def worker_payload(
        self,
        model_state: dict,
        agent_deck: list[int],
        game_indices: list[int],
        seed: int,
        search_count: int,
        best_state: dict | None,
    ) -> GauntletPayload:
        return GauntletPayload(
            model_state,
            self.kind,
            self.agent_path,
            self.deck_path,
            agent_deck,
            game_indices,
            seed,
            search_count,
            best_state,
        )


def _max_action_combinations() -> int:
    return int(os.environ.get("MCTS_MAX_ACTION_COMBINATIONS", str(MAX_ACTION_COMBINATIONS)))


def eval_nn_train(sv_enc: SparseVector, sv_dec: SparseVector, model: PolicyValueNet) -> tuple[float, list[float]]:
    device = next(model.parameters()).device
    with timer("nn_eval"):
        value, policy = model(
            torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
            torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
            torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
            torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
            torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
        )
        return (value.tolist()[0][0], policy.tolist()[0])


def eval_nn_batch(
    svs: list[tuple[SparseVector, SparseVector]], model: PolicyValueNet
) -> list[tuple[float, list[float]]]:
    """Evaluate B (encoder, decoder) inputs in a single forward; return per-item (value, policy).

    Each decoder is padded to the largest action count in this batch so the model
    can reshape it into a dense ``(batch, actions)`` tensor. Returned policies are
    sliced back to each item's real action count.
    """
    device = next(model.parameters()).device
    n_actions = [len(sv_dec.offset) for _, sv_dec in svs]
    padded_actions = max(n_actions)
    enc, dec = LearnInput(), LearnInput()
    for sv_enc, sv_dec in svs:
        enc.add(sv_enc)
        dec.add(sv_dec)
        for _ in range(padded_actions - len(sv_dec.offset)):
            dec.offset.append(len(dec.index))
    b = len(svs)
    with timer("nn_eval"):
        out_enc, out_dec = model(
            torch.tensor(enc.index, dtype=torch.int32, device=device),
            torch.tensor(enc.value, dtype=torch.float32, device=device),
            torch.tensor(enc.offset, dtype=torch.int32, device=device),
            torch.tensor(dec.index, dtype=torch.int32, device=device),
            torch.tensor(dec.value, dtype=torch.float32, device=device),
            torch.tensor(dec.offset, dtype=torch.int32, device=device),
        )
        values = out_enc.view(b, -1).tolist()
        policies = out_dec.view(b, -1).tolist()
    return [(values[i][0], policies[i][: n_actions[i]]) for i in range(b)]


def create_node_train(
    parent: Node | None, search_state: SearchState, your_index: int, your_deck: list[int], model: PolicyValueNet
) -> tuple[Node, LearnSample | None]:
    """Thin training wrapper around the shared `create_node`: evaluates with
    `eval_nn_train` (timed) instead of inference's `eval_nn`, and captures the raw
    (pre-sign-flip) value/policy as a `LearnSample` via a closure cell.
    """
    sample_cell: list[LearnSample | None] = [None]

    def evaluate(obs, actions):
        sv_enc = encode_state(obs, your_deck)
        sv_dec = encode_actions(obs, actions)
        value, policy = eval_nn_train(sv_enc, sv_dec, model)
        sample_cell[0] = LearnSample(value, policy, sv_enc, sv_dec)
        return value, policy

    node = create_node(
        parent,
        search_state,
        your_index,
        evaluate,
        max_action_combinations=_max_action_combinations(),
    )
    return node, sample_cell[0]


def _materialize_leaf(
    parent: Node,
    child: Child,
    path: tuple[Node, ...],
    your_index: int,
    your_deck: list[int],
    virtual_loss: float,
) -> PendingLeaf | CompletedLeaf:
    with timer("engine"):
        search_state = search_step(parent.state.searchId, child.select)
    node = Node(parent, search_state)
    child.node = node
    leaf_path = (*path, node)
    state = search_state.observation.current
    if state.result >= 0:
        node.value = 0.0 if state.result == RESULT_DRAW else (1.0 if state.result == your_index else -1.0)
        node.backprop(node.value)
        return CompletedLeaf(node)
    actions = enumerate_action_combinations(
        search_state.observation.select.maxCount,
        len(search_state.observation.select.option),
        cap=_max_action_combinations(),
    )
    leaf = PendingLeaf(
        node,
        leaf_path,
        actions,
        encode_state(search_state.observation, your_deck),
        encode_actions(search_state.observation, actions),
        state.yourIndex,
        your_index,
        virtual_loss,
    )
    leaf.add_virtual_loss()
    return leaf


def _leaf_batch_wave(root: Node, n_leaves: int, your_index: int, your_deck: list[int], model: PolicyValueNet) -> None:
    virtual_loss = 1.0
    completed = 0
    while completed < n_leaves:
        pending: list[PendingLeaf] = []
        for _ in range(n_leaves - completed):
            current = root
            path = (root,)
            while current.children:
                child = select_child(current, your_index)
                if child is None:
                    break
                if child.node is None:
                    leaf = _materialize_leaf(current, child, path, your_index, your_deck, virtual_loss)
                    if isinstance(leaf, PendingLeaf):
                        pending.append(leaf)
                    else:
                        completed += 1
                    break
                current = child.node
                path = (*path, current)
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    completed += 1
                    break
            if pending and not current.children:
                break
        if not pending:
            if completed < n_leaves and not root.children:
                raise RuntimeError("batched MCTS search exhausted a non-terminal tree before using its budget")
            continue
        encoded_leaves = [(leaf.encoder, leaf.decoder) for leaf in pending]
        for leaf, (value, policy) in zip(pending, eval_nn_batch(encoded_leaves, model), strict=True):
            leaf.resolve(value, policy)
            completed += 1


@timed("mcts_move")
def mcts_train_agent(
    obs_dict: dict, your_deck: list[int], model: PolicyValueNet, search_count=10
) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    opponent = infer_opponent_hidden_cards(state, your_index, random)

    with timer("engine"):
        search_state = search_begin(
            obs,
            your_deck=random.sample(your_deck, state.players[your_index].deckCount),
            your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
            opponent_deck=opponent.deck,
            opponent_prize=opponent.prize,
            opponent_hand=opponent.hand,
            opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
        )

    root, sample = create_node_train(None, search_state, your_index, your_deck, model)

    do_batched = os.environ.get("MCTS_BATCHED") == "1" and bool(root.children)
    if do_batched:
        _leaf_batch_wave(root, search_count, your_index, your_deck, model)

    for _ in range(0 if do_batched else search_count):
        current = root
        while True:
            next_node = select_child(current, your_index)
            if next_node is None:
                break
            if next_node.node is None:
                with timer("engine"):
                    search_state = search_step(current.state.searchId, next_node.select)
                next_node.node, _ = create_node_train(current, search_state, your_index, your_deck, model)
                break
            else:
                current = next_node.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

    max_child = None
    max_visit = -1
    min_value = 10.0
    for child in root.children:
        if child.node is not None:
            if max_visit < child.node.visit:
                max_child = child
                max_visit = child.node.visit
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v

    sample.value = root.total / root.visit
    for i in range(len(root.children)):
        child = root.children[i]
        v = sample.value
        if child.node is None:
            v = min_value - v - 0.03
        else:
            v = child.node.total / child.node.visit - v
        sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    return max_child.select, sample


def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


def _worker_init():
    # NOTE: Per-worker torch pools oversubscribe CPU cores during batch-1 inference.
    torch.set_num_threads(1)


def _transport_state(model: PolicyValueNet) -> dict:
    """State dict as numpy arrays. Tensors sent through the worker pool use torch's
    fd-based shared memory (one fd per tensor), which exhausts file descriptors over
    many pool.map calls ('Too many open files'). numpy pickles to plain bytes instead.
    """
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _build_cpu_model(state_dict) -> PolicyValueNet:
    model = PolicyValueNet(*model_args_from_state_dict(state_dict))
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
    model.eval()
    return model


def _parse_model_args(raw) -> tuple[int, int, int, int, int] | None:
    if raw in (None, ""):
        return None
    if isinstance(raw, tuple):
        vals = raw
    elif isinstance(raw, list):
        vals = tuple(raw)
    else:
        vals = tuple(int(part.strip()) for part in str(raw).strip("()").split(",") if part.strip())
    if len(vals) != 5:
        raise ValueError(f"model_args must have 5 integers, got {vals!r}")
    return tuple(int(v) for v in vals)


def _resolve_init_checkpoint(spec: str, task: str, comp_dir: Path) -> Path:
    """Resolve a warm-start checkpoint from a local path or `registry:<version>`."""
    if spec.startswith("registry:"):
        version = spec.split(":", 1)[1]
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=default_tracking_uri())
        model_version = client.get_model_version(task, version)
        cache_dir = comp_dir / "outputs" / "init_checkpoints" / f"registry_v{version}"
        downloaded = Path(client.download_artifacts(model_version.run_id, "checkpoint", dst_path=str(cache_dir)))
        wanted = model_version.tags.get("checkpoint_filename")
        candidates = [downloaded] if downloaded.suffix == ".pth" else sorted(downloaded.rglob("*.pth"))
        if wanted:
            candidates = [p for p in candidates if p.name == wanted]
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoint found in registry:{version} artifact {downloaded}")
        return candidates[0]

    path = Path(spec).expanduser()
    if not path.is_absolute():
        path = comp_dir / path
    if not path.exists():
        raise FileNotFoundError(f"init_checkpoint not found: {path}")
    return path


_AGENT_CACHE: dict = {}


def _opp_agent_fn(opp_file: str):
    """Load (and cache per worker process) a rule agent's move function."""
    if opp_file not in _AGENT_CACHE:
        _AGENT_CACHE[opp_file] = load_agent(opp_file)
    return _AGENT_CACHE[opp_file].agent


def _split_counts(total: int, parts: int) -> list[int]:
    """Split ``total`` items into at most ``parts`` near-equal positive chunks."""
    parts = max(1, min(parts, total))
    base, rem = divmod(total, parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _collect_game(mcts_deck, model, search_count, opp_move, opp_deck, mcts_index) -> list[LearnSample]:
    """Play one game and return MCTS-side training samples (bootstrapped value targets).

    ``opp_move is None`` -> self-play (both seats are MCTS, both sides collected).
    Otherwise MCTS plays seat ``mcts_index`` vs ``opp_move`` (only the MCTS side is
    collected) — this is how the agent learns to beat the rule-agent pool.
    """
    self_play = opp_move is None
    if self_play:
        deck0 = deck1 = mcts_deck
    else:
        deck0, deck1 = (mcts_deck, opp_deck) if mcts_index == 0 else (opp_deck, mcts_deck)
    obs, _ = battle_start(deck0, deck1)
    samples: list[list[LearnSample]] = [[], []]
    while obs["current"]["result"] < 0:
        cur = obs["current"]["yourIndex"]
        if self_play or cur == mcts_index:
            selected, sample = mcts_train_agent(obs, mcts_deck, model, search_count)
            samples[cur].append(sample)
        else:
            selected = opp_move(obs)
        obs = battle_select(selected)
    battle_finish()

    result = obs["current"]["result"]
    out: list[LearnSample] = []
    for i in (0, 1) if self_play else (mcts_index,):
        LAMBDA = 0.9
        value = 1.0 if i == result else -1.0
        for sample in reversed(samples[i]):
            label = (value + sample.value) * 0.5
            value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
            sample.value = label
            out.append(sample)
    return out


def _play_vs_game(mcts_deck, model, opp_deck, opp_move, your_index: int, search_count: int) -> int:
    """One game: MCTS(model, mcts_deck) at ``your_index`` vs ``opp_move`` at the other
    seat (each side gets its own deck). Returns 0=win, 1=loss, 2=draw for the MCTS side.
    """
    deck0, deck1 = (mcts_deck, opp_deck) if your_index == 0 else (opp_deck, mcts_deck)
    obs, _ = battle_start(deck0, deck1)
    while obs["current"]["result"] < 0:
        if obs["current"]["yourIndex"] == your_index:
            selected, _ = mcts_train_agent(obs, mcts_deck, model, search_count)
        else:
            selected = opp_move(obs)
        obs = battle_select(selected)
    battle_finish()
    res_val = obs["current"]["result"]
    if res_val == 2:
        return 2
    return 0 if res_val == your_index else 1


def _selfplay_worker(payload):
    """Worker entry: play ``n_games`` games on a fresh CPU model, cycling through the
    opponent pool (None = self-play, else a (kind, file, deck) spec). Returns
    ``(samples, component_timings)`` so the parent can merge per-worker timing.
    """
    state_dict, n_games, agent_deck, seed, search_count, opp_pool = payload
    random.seed(seed)
    DEFAULT.reset()
    model = _build_cpu_model(state_dict)
    out: list[LearnSample] = []
    with torch.no_grad():
        for g in range(n_games):
            spec = opp_pool[(seed + g) % len(opp_pool)]
            if spec is None:
                out.extend(_collect_game(agent_deck, model, search_count, None, agent_deck, 0))
            else:
                kind, opp_file, opp_deck_path = spec
                opp_move = random_agent if kind == "random" else _opp_agent_fn(opp_file)
                opp_deck = agent_deck if kind == "random" else load_deck(opp_deck_path)
                out.extend(_collect_game(agent_deck, model, search_count, opp_move, opp_deck, g % 2))
    return out, DEFAULT.as_dict()


def _gauntlet_worker(payload: GauntletPayload):
    """Play eval games of the current MCTS model vs ONE opponent; return ([W,L,D], timings).

    opponent kinds: 'random' (random_agent), 'rule' (a loaded agent module's agent fn),
    'self' (MCTS driven by the best checkpoint so far).
    """
    random.seed(payload.seed)
    DEFAULT.reset()
    model = _build_cpu_model(payload.model_state)

    if payload.opponent_kind is OpponentKind.RANDOM:
        opp_move, opp_deck = random_agent, payload.agent_deck
    elif payload.opponent_kind is OpponentKind.RULE:
        assert payload.opponent_agent_path is not None and payload.opponent_deck_path is not None
        opp_move = _opp_agent_fn(payload.opponent_agent_path)
        opp_deck = load_deck(payload.opponent_deck_path)
    elif payload.opponent_kind is OpponentKind.SELF:
        assert payload.best_state is not None
        best_model = _build_cpu_model(payload.best_state)

        def opp_move(o):
            return mcts_train_agent(o, payload.agent_deck, best_model, payload.search_count)[0]

        opp_deck = payload.agent_deck
    else:
        raise ValueError(f"unknown opponent kind: {payload.opponent_kind}")

    res = [0, 0, 0]
    with torch.no_grad():
        for game_index in payload.game_indices:
            result = _play_vs_game(
                payload.agent_deck,
                model,
                opp_deck,
                opp_move,
                game_index % 2,
                payload.search_count,
            )
            res[result] += 1
    return res, DEFAULT.as_dict()


@dataclass(frozen=True, slots=True)
class TrackingMetadata:
    task: str
    variant: str | None
    iterations: int
    eval_games: int
    self_play_games: int
    eval_every: int
    search_count: int
    batched: bool
    num_workers: int
    eval_opponents: tuple[str, ...]
    selfplay_opponents: tuple[str, ...]
    replay_buffer_size: int
    train_steps: int
    model_args: tuple[int, int, int, int, int]
    deck_path: Path
    deck_file: str
    init_checkpoint: str | None
    init_checkpoint_path: Path | None
    start_iteration: int
    config_fingerprint: str | None
    output_path: str
    resume_version: int | None

    def tags(self) -> dict[str, str]:
        model = self.model_args
        agent_name = self.variant or (
            f"mcts-{self.deck_path.stem}-d{model[0]}-h{model[1]}-ff{model[2]}-enc{model[3]}-dec{model[4]}"
        )
        tags = {
            "agent_name": agent_name,
            "machine": machine_name(),
            "git_sha": git_sha(Path(__file__).resolve().parents[2]),
            "task": self.task,
            "deck": self.deck_path.stem,
            "search_count": str(self.search_count),
            "self_play_games": str(self.self_play_games),
            "batched": str(self.batched),
            "model_args": str(self.model_args),
        }
        if self.variant:
            tags["variant"] = self.variant
        if continued_from := (
            f"registry:{self.resume_version}" if self.resume_version is not None else self.init_checkpoint
        ):
            tags["continued_from"] = continued_from
        if self.config_fingerprint:
            tags["training_fingerprint"] = self.config_fingerprint
        return tags

    def params(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "iterations": self.iterations,
            "eval_games": self.eval_games,
            "self_play_games": self.self_play_games,
            "eval_every": self.eval_every,
            "search_count": self.search_count,
            "batched": self.batched,
            "num_workers": self.num_workers,
            "eval_opponents": ",".join(self.eval_opponents),
            "selfplay_opponents": ",".join(self.selfplay_opponents),
            "replay_buffer_size": self.replay_buffer_size,
            "train_steps": self.train_steps,
            "model_args": self.model_args,
            "deck": self.deck_path.stem,
            "deck_file": self.deck_file,
            "init_checkpoint": self.init_checkpoint or "",
            "init_checkpoint_resolved": str(self.init_checkpoint_path) if self.init_checkpoint_path else "",
            "start_iteration": self.start_iteration,
            "training_fingerprint": self.config_fingerprint or "",
            "output_path": self.output_path,
        }


def _open_tracker(metadata: TrackingMetadata, uri: str, output_file: Path) -> tuple[Tracker, dict[str, str]]:
    tags = metadata.tags()
    tracker = Tracker.open(
        uri,
        experiment=metadata.task,
        run_id=os.environ.get("KEGO_MLFLOW_RUN_ID"),
        run_name=f"{tags['machine']}-{output_file.stem}",
        tags=tags,
    )
    if tracker.run_id:
        tags["training_run_id"] = tracker.run_id
    tracker.set_tags(tags)
    tracker.log_params(metadata.params())
    return tracker, tags


def run_training_loop(
    iterations=3,
    eval_games=5,
    self_play_games=10,
    output_path="outputs/mcts_model.pth",
    num_workers: int | None = None,
    eval_every: int = 1,
    search_count: int = 10,
    batched: bool = False,
    eval_opponents: list[str] | None = None,
    selfplay_opponents: list[str] | None = None,
    replay_buffer_size: int = 100000,
    train_steps: int = 100,
    deck_file: str | None = None,
    init_checkpoint: str | None = None,
    model_args=None,
    variant: str | None = None,
    config_fingerprint: str | None = None,
    features: dict | None = None,
):
    if eval_opponents is None:
        eval_opponents = ["random", "rule:zacian", "rule:abomasnow", "rule:dragapult", "rule:lucario", "self"]
    if selfplay_opponents is None:
        selfplay_opponents = ["self"]

    def _resolve_pool(specs):
        pool = []
        for spec in specs:
            if spec == "self":
                pool.append(None)
            elif spec == "random":
                pool.append(("random", None, None))
            elif spec.startswith("rule:"):
                nm = spec.split(":", 1)[1]
                pool.append(("rule", str(comp_dir / "agents" / f"{nm}.py"), str(comp_dir / "decks" / f"{nm}.csv")))
        return pool or [None]

    opp_pool = _resolve_pool(selfplay_opponents)

    # NOTE: Worker processes inherit these MCTS settings from the environment.
    os.environ["MCTS_BATCHED"] = "1" if batched else "0"
    features = features or {}
    max_action_combinations = int(features.get("max_action_combinations", MAX_ACTION_COMBINATIONS))
    if max_action_combinations <= 0:
        raise ValueError("features.max_action_combinations must be greater than zero")
    os.environ["MCTS_MAX_ACTION_COMBINATIONS"] = str(max_action_combinations)

    if deck_file is None:
        deck_file = "decks/abomasnow.csv"
    deck_path = comp_dir / deck_file
    agent_deck = load_deck(str(deck_path))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 2)
    num_workers = max(1, min(num_workers, max(self_play_games, eval_games, 1)))
    print(f"Starting MCTS self-play training on {device} | rollout workers: {num_workers}...")

    _uri = default_tracking_uri()
    _task = "pokemon-tcg-ai-battle"
    resume = None
    resume_payload = None
    if config_fingerprint and not init_checkpoint:
        try:
            # NOTE: Isolated destinations prevent concurrent MLflow artifact collisions.
            with tempfile.TemporaryDirectory(prefix="kego-training-resume-") as resume_cache:
                resume = resolve_training_resume(
                    _uri,
                    _task,
                    config_fingerprint,
                    target_iterations=iterations,
                    cache_dir=resume_cache,
                )
                if resume:
                    resume_payload = torch.load(resume.path, map_location=device, weights_only=False)
        except Exception as exc:
            print(f"Automatic resume unavailable; starting fresh ({exc})", flush=True)
            resume = None
            resume_payload = None

        if resume_payload:
            required = {
                "model_state",
                "optimizer_state",
                "scheduler_state",
                "replay",
                "python_rng_state",
                "torch_rng_state",
            }
            if resume_payload.get("format_version") != 1:
                print(
                    f"Automatic resume unavailable; starting fresh (unsupported training-state format {resume_payload.get('format_version')!r})",
                    flush=True,
                )
                resume = None
                resume_payload = None
            elif resume_payload.get("training_fingerprint") != config_fingerprint:
                print(
                    "Automatic resume unavailable; starting fresh (training-state fingerprint does not match its registry metadata)",
                    flush=True,
                )
                resume = None
                resume_payload = None
            elif int(resume_payload.get("completed_iterations", -1)) != resume.completed_iterations:
                print(
                    "Automatic resume unavailable; starting fresh (training-state iteration does not match its registry metadata)",
                    flush=True,
                )
                resume = None
                resume_payload = None
            elif missing := required - resume_payload.keys():
                print(
                    f"Automatic resume unavailable; starting fresh (training state is missing required fields: {sorted(missing)})",
                    flush=True,
                )
                resume = None
                resume_payload = None

    init_checkpoint_path = None
    init_state_dict = None
    actual_model_args = _parse_model_args(model_args) or MODEL_ARGS
    if resume_payload:
        actual_model_args = model_args_from_state_dict(resume_payload["model_state"])
    elif init_checkpoint:
        init_checkpoint_path = _resolve_init_checkpoint(init_checkpoint, "pokemon-tcg-ai-battle", comp_dir)
        init_state_dict = torch.load(init_checkpoint_path, map_location=device)
        actual_model_args = model_args_from_state_dict(init_state_dict)

    init_iterations = 0
    if init_checkpoint:
        if init_checkpoint.startswith("registry:"):
            ver_str = init_checkpoint.split(":", 1)[1]
            try:
                from mlflow.tracking import MlflowClient

                client = MlflowClient(tracking_uri=default_tracking_uri())
                mv = client.get_model_version("pokemon-tcg-ai-battle", ver_str)
                init_iterations = int(mv.tags.get("completed_iterations") or mv.tags.get("epoch") or 0)
            except Exception as e:
                print(f"  Note: Could not query completed_iterations for {init_checkpoint}: {e}", flush=True)
        if init_iterations == 0 and init_checkpoint_path:
            match = re.search(r"iter(\d+)", init_checkpoint_path.name)
            if match:
                init_iterations = int(match.group(1))
            elif isinstance(init_state_dict, dict) and "completed_iterations" in init_state_dict:
                init_iterations = int(init_state_dict["completed_iterations"])

    model = PolicyValueNet(*actual_model_args).to(device)
    if resume_payload:
        assert resume is not None
        model.load_state_dict(ensure_tensors(resume_payload["model_state"]))
        print(
            f"Resuming compatible registry:{resume.version} from iteration {resume.completed_iterations}/{iterations}",
            flush=True,
        )
    elif init_state_dict is not None:
        model.load_state_dict(ensure_tensors(init_state_dict))
        print(f"Warm-started model from {init_checkpoint_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # NOTE: A fixed horizon keeps resumed runs on the same learning-rate history.
    decay_horizon = 250
    min_lr_ratio = 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            min_lr_ratio
            + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(step, decay_horizon) / decay_horizon))
        ),
    )
    start_iteration = 0
    if resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        start_iteration = int(resume_payload["completed_iterations"])
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)

    replay: list[LearnSample] = list(resume_payload.get("replay", [])) if resume_payload else []

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    run_timings = Timings()

    tracking = TrackingMetadata(
        task=_task,
        variant=variant,
        iterations=iterations,
        eval_games=eval_games,
        self_play_games=self_play_games,
        eval_every=eval_every,
        search_count=search_count,
        batched=batched,
        num_workers=num_workers,
        eval_opponents=tuple(eval_opponents),
        selfplay_opponents=tuple(selfplay_opponents),
        replay_buffer_size=replay_buffer_size,
        train_steps=train_steps,
        model_args=actual_model_args,
        deck_path=deck_path,
        deck_file=deck_file,
        init_checkpoint=init_checkpoint,
        init_checkpoint_path=init_checkpoint_path,
        start_iteration=start_iteration,
        config_fingerprint=config_fingerprint,
        output_path=output_path,
        resume_version=resume.version if resume else None,
    )
    _track, _run_tags = _open_tracker(tracking, _uri, output_file)
    best_results: dict[str, float] = dict(resume_payload.get("best_results", {})) if resume_payload else {}
    best_score = float(resume_payload.get("best_score", -1.0)) if resume_payload else -1.0
    best_state = (
        resume_payload.get("best_state") if resume_payload else _transport_state(model) if init_checkpoint else None
    )
    best_iteration = resume_payload.get("best_iteration") if resume_payload else None
    if best_iteration is not None:
        best_iteration = int(best_iteration)
    if resume_payload:
        random.setstate(resume_payload["python_rng_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and resume_payload.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        torch.save(ensure_tensors(best_state or resume_payload["model_state"]), str(output_file))
    if start_iteration >= iterations:
        message = (
            f"Compatible checkpoint already reached target iteration {iterations}; no training needed."
            if resume
            else f"Target iteration {iterations} requires no training."
        )
        print(message, flush=True)
        _track.close()
        return

    last_completed_iteration = start_iteration

    def _save_training_state(completed_iterations: int, path: Path) -> None:
        torch.save(
            {
                "format_version": 1,
                "training_fingerprint": config_fingerprint,
                "completed_iterations": completed_iterations,
                "target_iterations": iterations,
                "model_state": _transport_state(model),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "replay": replay,
                "best_state": best_state,
                "best_iteration": best_iteration,
                "best_score": best_score,
                "best_results": best_results,
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )

    print("[5/7] Spawning worker pool...", flush=True)
    # NOTE: The game engine has global state, so concurrent games require processes.
    pool = None
    if num_workers > 1:
        print(f"  Configuring environment for {num_workers} workers...", flush=True)
        # NOTE: Spawned workers need explicit import paths and single-threaded torch.
        os.environ["PYTHONPATH"] = os.pathsep.join(
            p for p in [str(comp_dir), str(repo_root), os.environ.get("PYTHONPATH", "")] if p
        )
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        print(f"  Spawning {num_workers} worker processes (this may take 30-60 seconds)...", flush=True)
        _spawn_start = time.perf_counter()
        pool = mp.get_context("spawn").Pool(num_workers, initializer=_worker_init)
        _spawn_elapsed = time.perf_counter() - _spawn_start
        print(f"  ✓ Worker pool ready ({num_workers} processes, took {_spawn_elapsed:.1f}s)", flush=True)
    else:
        print("  Running single-threaded (no worker pool)", flush=True)

    def _run(worker, payloads):
        return pool.map(worker, payloads) if pool is not None else [worker(p) for p in payloads]

    print("[6/7] Configuring evaluation opponents...", flush=True)
    opponent_specs = [OpponentSpec.parse(spec) for spec in eval_opponents]
    print(
        f"  Configured {len(opponent_specs)} evaluation opponents: {[spec.name for spec in opponent_specs]}",
        flush=True,
    )

    def _gauntlet(cur_state, best_state, seed_base):
        """Eval the current model vs each opponent; return {name: win_rate%} and the
        rule-agent average (the metric the best checkpoint is gated on)."""
        results = {}
        for opponent in opponent_specs:
            if opponent.kind == "self" and best_state is None:
                continue
            payloads = [
                opponent.worker_payload(cur_state, agent_deck, chunk, seed_base + i, search_count, best_state)
                for i, chunk in enumerate(list(range(eval_games))[w::num_workers] for w in range(num_workers))
                if chunk
            ]
            with run_timings.timer("eval"):
                returns = _run(_gauntlet_worker, payloads)
            agg = [0, 0, 0]
            for triple, tdict in returns:
                for k in range(3):
                    agg[k] += triple[k]
                run_timings.merge(tdict)
            wr = (100 * agg[0]) // max(1, agg[0] + agg[1])
            results[opponent.name] = wr
            print(f"  vs {opponent.name:<10} {wr:3d}%  (W{agg[0]} L{agg[1]} D{agg[2]})")
        rule_wrs = [wr for nm, wr in results.items() if nm not in ("random", "self")]
        avg = sum(rule_wrs) / len(rule_wrs) if rule_wrs else float(results.get("random", 0))
        print(f"  gauntlet avg (rule agents): {avg:.1f}%")
        return results, avg

    training_completed_normally = False
    try:
        for iter_idx in range(start_iteration, iterations):
            current_epoch = init_iterations + iter_idx + 1
            total_epochs = init_iterations + iterations
            if init_iterations > 0:
                print(
                    f"\n--- Iteration {current_epoch}/{total_epochs} [iter {iter_idx + 1}/{iterations} warm-started from {init_checkpoint}] ---",
                    flush=True,
                )
            else:
                print(f"\n--- Iteration {iter_idx + 1}/{iterations} ---", flush=True)
            cpu_state = _transport_state(model)

            if eval_games > 0 and (iter_idx % eval_every == 0 or (iter_idx + 1) % 50 == 0):
                print("Evaluating gauntlet (random / rule agents / self)...")
                results, avg = _gauntlet(cpu_state, best_state, 7000 + iter_idx * 131)
                _track.log_metric("gauntlet_avg", avg, step=current_epoch)
                _track.log_metric("progress_pct", 100.0 * (iter_idx + 1) / iterations, step=current_epoch)
                for _name, _wr in results.items():
                    _track.log_metric(f"wr_{_name}", _wr, step=current_epoch)
                if avg > best_score:
                    best_score = avg
                    best_state = cpu_state
                    best_iteration = current_epoch
                    best_results = results
                    torch.save(model.state_dict(), str(output_file))
                    print(f"  -> new best (avg {avg:.1f}%), checkpointed to {output_file}")

            search_count_start = min(10, search_count)
            if iterations > 1:
                current_search_count = int(
                    search_count_start + (search_count - search_count_start) * (iter_idx / (iterations - 1))
                )
            else:
                current_search_count = search_count

            print(
                f"Collecting self-play training data ({self_play_games} games) with dynamic MCTS search_count={current_search_count}..."
            )
            payloads = [
                (cpu_state, c, agent_deck, iter_idx * 9973 + i, current_search_count, opp_pool)
                for i, c in enumerate(_split_counts(self_play_games, num_workers))
                if c
            ]
            with run_timings.timer("self_play"):
                returns = _run(_selfplay_worker, payloads)
            sample_list = []
            for samples, tdict in returns:
                sample_list.extend(samples)
                run_timings.merge(tdict)

            replay.extend(sample_list)
            if len(replay) > replay_buffer_size:
                replay = replay[-replay_buffer_size:]
            print(f"Training on buffer ({len(replay)} samples, {train_steps} steps)...")
            _train_start = time.perf_counter()
            model.train()
            BATCH_SIZE = min(128, len(replay))
            if BATCH_SIZE > 0:
                sum_enc = sum_dec = 0.0
                for _ in range(train_steps):
                    batch = random.sample(replay, BATCH_SIZE)
                    padded_actions = max(len(sample.policy) for sample in batch)
                    input_enc = LearnInput()
                    input_dec = LearnInput()
                    mask = []
                    label_enc = []
                    label_dec = []
                    for sample in batch:
                        input_enc.add(sample.sv_enc)
                        input_dec.add(sample.sv_dec)
                        label_enc.append(sample.value)
                        label_dec.extend(sample.policy)
                        for _ in range(len(sample.policy)):
                            mask.append(1.0)
                        for _ in range(padded_actions - len(sample.policy)):
                            mask.append(0.0)
                            label_dec.append(0.0)
                            input_dec.offset.append(len(input_dec.index))

                    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
                    label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)
                    label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(BATCH_SIZE, -1)

                    optimizer.zero_grad()
                    out_enc, out_dec = model(
                        torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                        torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                        torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                        torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                        torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                        torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
                    )

                    loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
                    loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
                    loss_dec = (loss_dec * mask_tensor).sum() / float(BATCH_SIZE)
                    loss = loss_enc + loss_dec

                    sum_enc += float(loss_enc.detach())
                    sum_dec += float(loss_dec.detach())
                    loss.backward()
                    optimizer.step()
                _track.log_metric("loss_value", sum_enc / train_steps, step=current_epoch)
                _track.log_metric("loss_policy", sum_dec / train_steps, step=current_epoch)
                _track.log_metric("lr", scheduler.get_last_lr()[0], step=current_epoch)
                print(
                    f"Training complete. avg loss: value={sum_enc / train_steps:.4f} "
                    f"policy={sum_dec / train_steps:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
                )
                scheduler.step()
            run_timings.add("train", time.perf_counter() - _train_start)
            run_timings.report(f"timings @ iter {iter_idx + 1}")
            last_completed_iteration = current_epoch

            checkpoint_interval = 50
            if (iter_idx + 1) % checkpoint_interval == 0:
                iter_output_file = output_file.parent / f"{output_file.stem}_iter{current_epoch}.pth"
                iter_state_file = output_file.parent / f"{output_file.stem}_iter{current_epoch}.train.pt"
                torch.save(model.state_dict(), str(iter_output_file))
                _save_training_state(current_epoch, iter_state_file)
                print(f"  -> Intermediate checkpoint saved to {iter_output_file}")
                iter_score = (
                    avg
                    if (eval_games > 0 and (iter_idx % eval_every == 0 or (iter_idx + 1) % checkpoint_interval == 0))
                    else best_score
                )
                _version = register_checkpoint_or_queue(
                    _uri,
                    _task,
                    str(iter_output_file),
                    tags={
                        **_run_tags,
                        "epoch": str(current_epoch),
                        "completed_iterations": str(current_epoch),
                        "deck": deck_path.stem,
                        "gauntlet_avg": round(iter_score, 2) if iter_score >= 0 else 0.0,
                    },
                    training_state_path=str(iter_state_file),
                )
                if _version:
                    print("  -> Registered intermediate checkpoint as registry model version.")
                else:
                    print("  (Registry unavailable, checkpoint queued to outbox — run `kego sync` later.)")

        if eval_games > 0:
            print("\nFinal gauntlet on trained model...")
            final_state = _transport_state(model)
            results, avg = _gauntlet(final_state, best_state, 999983)
            if avg > best_score:
                best_score = avg
                best_state = final_state
                best_iteration = last_completed_iteration
                best_results = results
                torch.save(model.state_dict(), str(output_file))
                print(f"  -> new best (avg {avg:.1f}%), checkpointed to {output_file}")
        training_completed_normally = True
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if training_completed_normally:
            if best_score < 0:
                torch.save(model.state_dict(), str(output_file))
                best_iteration = last_completed_iteration
            final_state_file = output_file.parent / f"{output_file.stem}_iter{last_completed_iteration}.train.pt"
            _save_training_state(last_completed_iteration, final_state_file)
            _version = register_checkpoint_or_queue(
                _uri,
                _task,
                str(output_file),
                tags={
                    **_run_tags,
                    "completed_iterations": str(last_completed_iteration),
                    "deck": deck_path.stem,
                    **({"epoch": str(best_iteration)} if best_iteration is not None else {}),
                    **({"gauntlet_avg": round(best_score, 2)} if best_score >= 0 else {}),
                    **({f"wr_{n}": round(w, 2) for n, w in best_results.items()} if best_score >= 0 else {}),
                },
                training_state_path=str(final_state_file),
            )
            if not _version:
                print("  (Registry unavailable, best checkpoint queued to outbox — run `kego sync` later.)")
        _track.close()

    run_timings.report("timings @ final")
    print(f"Done. Best rule-agent avg: {best_score:.1f}%. Model saved to {output_file}")


if __name__ == "__main__":
    run_training_loop(iterations=1, eval_games=2, self_play_games=2, output_path="outputs/mcts_model.pth")
