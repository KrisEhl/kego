import math
import os
import random
import sys
import time
from pathlib import Path

import torch

# Setup paths to import cg and agents
comp_dir = Path(__file__).parent.resolve()
repo_root = comp_dir.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(comp_dir))

from kego.pipeline.battle import load_agent, load_deck, locate_cg_dir
from kego.timing import DEFAULT, Timings, timed, timer

cg_parent = locate_cg_dir()
sys.path.insert(0, str(cg_parent))

from agents.mcts import (
    MODEL_ARGS,
    MyModel,
    SparseVector,
    get_decoder_input,
    get_encoder_input,
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


class Child:
    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob


class Node:
    def __init__(self, parent: "Node | None", state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def eval_nn_train(sv_enc: SparseVector, sv_dec: SparseVector, model: MyModel) -> tuple[float, list[float]]:
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


def _enumerate_actions(obs) -> list[list[int]]:
    """Enumerate up to 64 candidate action-index combinations for an observation."""
    actions = []
    indices = list(range(obs.select.maxCount))
    for _ in range(64):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < len(obs.select.option) - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions


def eval_nn_batch(svs: list[tuple[SparseVector, SparseVector]], model: MyModel) -> list[tuple[float, list[float]]]:
    """Evaluate B (encoder, decoder) inputs in a single forward; return per-item (value, policy).

    Mirrors the training-loop batching: each decoder is padded to 64 words so the
    model returns out_enc (B, 1) and out_dec (B, 64); the policy is sliced back to
    each item's real action count.
    """
    device = next(model.parameters()).device
    n_actions = [len(sv_dec.offset) for _, sv_dec in svs]
    enc, dec = LearnInput(), LearnInput()
    for sv_enc, sv_dec in svs:
        enc.add(sv_enc)
        dec.add(sv_dec)
        for _ in range(64 - len(sv_dec.offset)):
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


def _build_children(node: Node, actions: list[list[int]], policy: list[float]) -> None:
    """Attach softmax-weighted children to a node from a policy vector."""
    total_prob = 0.0
    for i in range(len(policy)):
        p = math.exp(policy[i] * 10.0)
        node.children.append(Child(actions[i], p))
        total_prob += p
    for c in node.children:
        c.prob /= total_prob


def create_node_train(
    parent: Node | None, search_state: SearchState, your_index: int, your_deck: list[int], model: MyModel
) -> tuple[Node, LearnSample | None]:
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current

    if state.result >= 0:
        if state.result == 2:
            node.value = 0.0
        elif state.result == your_index:
            node.value = 1.0
        else:
            node.value = -1.0
        node.backprop(node.value)
        sample = None
    else:
        actions = _enumerate_actions(obs)
        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn_train(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)
        _build_children(node, actions, policy)
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return node, sample


def _select_child(current: Node, your_index: int):
    """UCB-select the best child of ``current`` (None if it has none)."""
    best, chosen = -1e18, None
    c = 0.4 * math.sqrt(current.visit)
    flip = current.state.observation.current.yourIndex != your_index
    for child in current.children:
        if child.node is None:
            q = current.total / current.visit
            visit = 0
        else:
            q = child.node.total / child.node.visit
            visit = child.node.visit
        if flip:
            q = -q
        u = q + c * child.prob / (1 + visit)
        if u > best:
            best, chosen = u, child
    return chosen


def _leaf_batch_wave(root: Node, n_leaves: int, your_index: int, your_deck: list[int], model: MyModel) -> None:
    """Tree-parallel MCTS wave: select up to ``n_leaves`` leaves by UCB descent (with a
    virtual loss so selections diversify and go DEEP), materialize them, evaluate them all
    in ONE batched forward, then backprop with the virtual loss removed. Gives real search
    depth AND batched inference (a move needs ~2 forwards instead of ~n_leaves)."""
    VLOSS = 1.0
    pending = []  # (leaf_node, path, actions, sv_enc, sv_dec, leaf_state)
    for _ in range(n_leaves):
        current = root
        path = [root]
        while current.children:
            child = _select_child(current, your_index)
            if child is None:
                break
            if child.node is None:  # unexpanded -> this is our leaf to evaluate
                with timer("engine"):
                    search_state = search_step(current.state.searchId, child.select)
                node = Node(current, search_state)
                child.node = node
                path.append(node)
                leaf = search_state.observation.current
                if leaf.result >= 0:  # terminal: value known, no NN needed
                    node.value = 0.0 if leaf.result == 2 else (1.0 if leaf.result == your_index else -1.0)
                    node.backprop(node.value)
                else:
                    actions = _enumerate_actions(search_state.observation)
                    sv_enc = get_encoder_input(search_state.observation, your_deck)
                    sv_dec = get_decoder_input(search_state.observation, actions)
                    pending.append((node, path, actions, sv_enc, sv_dec, leaf))
                    for pn in path:  # virtual loss so the next selection avoids this path
                        pn.visit += 1
                        pn.total -= VLOSS
                break
            current = child.node  # descend into an already-expanded child
            path.append(current)
            if current.state.observation.current.result >= 0:
                current.backprop(current.value)
                break
    if pending:
        for (node, path, actions, _se, _sd, leaf), (value, policy) in zip(
            pending, eval_nn_batch([(p[3], p[4]) for p in pending], model)
        ):
            for pn in path:  # remove virtual loss before the real backprop
                pn.visit -= 1
                pn.total += VLOSS
            v = -value if leaf.yourIndex != your_index else value
            node.value = v
            node.backprop(v)
            _build_children(node, actions, policy)


@timed("mcts_move")
def mcts_train_agent(
    obs_dict: dict, your_deck: list[int], model: MyModel, search_count=10
) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active

    with timer("engine"):
        search_state = search_begin(
            obs,
            your_deck=random.sample(your_deck, state.players[your_index].deckCount),
            your_prize=random.sample(your_deck, len(state.players[your_index].prize)),
            opponent_deck=[1072] * state.players[1 - your_index].deckCount,
            opponent_prize=[1] * len(state.players[1 - your_index].prize),
            opponent_hand=[1] * state.players[1 - your_index].handCount,
            opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
        )

    root, sample = create_node_train(None, search_state, your_index, your_deck, model)

    do_batched = os.environ.get("MCTS_BATCHED") == "1" and bool(root.children)
    if do_batched:
        _leaf_batch_wave(root, search_count, your_index, your_deck, model)

    for _ in range(0 if do_batched else search_count):
        current = root
        while True:
            next_node = _select_child(current, your_index)
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

    # Both paths produce real visit counts now, so select the most-visited (robust) child.
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
    # Batch-1 inference gains nothing from intra-op threads; N workers each spawning
    # a full torch thread pool oversubscribes the cores and collapses throughput.
    torch.set_num_threads(1)


def _transport_state(model: MyModel) -> dict:
    """State dict as numpy arrays. Tensors sent through the worker pool use torch's
    fd-based shared memory (one fd per tensor), which exhausts file descriptors over
    many pool.map calls ('Too many open files'). numpy pickles to plain bytes instead.
    """
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _build_cpu_model(state_dict) -> MyModel:
    model = MyModel(*_model_args_from_state_dict(state_dict))
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict.items()})
    model.eval()
    return model


def _layer_count(state_dict, prefix: str, suffix: str) -> int:
    found = []
    for key in state_dict:
        if key.startswith(prefix) and key.endswith(suffix):
            try:
                found.append(int(key[len(prefix) :].split(".", 1)[0]))
            except ValueError:
                pass
    return max(found) + 1 if found else 0


def _model_args_from_state_dict(state_dict) -> tuple[int, int, int, int, int]:
    d_model = int(state_dict["encoder_bag.weight"].shape[1])
    d_feedforward = int(state_dict["encoder.layers.0.linear1.weight"].shape[0])
    num_heads = MODEL_ARGS[1] if d_model % MODEL_ARGS[1] == 0 else 4
    return (
        d_model,
        num_heads,
        d_feedforward,
        _layer_count(state_dict, "encoder.layers.", ".linear1.weight"),
        _layer_count(state_dict, "decoder.", ".fc1.weight"),
    )


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

        from kego.tracking import default_tracking_uri

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


_AGENT_CACHE: dict = {}  # per-worker cache of loaded rule-agent modules


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
    state_dict, n_games, sample_deck, seed, search_count, opp_pool = payload
    random.seed(seed)
    DEFAULT.reset()
    model = _build_cpu_model(state_dict)
    out: list[LearnSample] = []
    with torch.no_grad():
        for g in range(n_games):
            spec = opp_pool[(seed + g) % len(opp_pool)]
            if spec is None:
                out.extend(_collect_game(sample_deck, model, search_count, None, sample_deck, 0))
            else:
                kind, opp_file, opp_deck_path = spec
                opp_move = random_agent if kind == "random" else _opp_agent_fn(opp_file)
                opp_deck = sample_deck if kind == "random" else load_deck(opp_deck_path)
                out.extend(_collect_game(sample_deck, model, search_count, opp_move, opp_deck, g % 2))
    return out, DEFAULT.as_dict()


def _gauntlet_worker(payload):
    """Play eval games of the current MCTS model vs ONE opponent; return ([W,L,D], timings).

    opponent kinds: 'random' (random_agent), 'rule' (a loaded agent module's agent fn),
    'self' (MCTS driven by the best checkpoint so far).
    """
    cur_state, kind, opp_file, opp_deck_path, mcts_deck, game_indices, seed, search_count, best_state = payload
    random.seed(seed)
    DEFAULT.reset()
    model = _build_cpu_model(cur_state)

    if kind == "random":
        opp_move, opp_deck = random_agent, mcts_deck
    elif kind == "rule":
        opp_move, opp_deck = _opp_agent_fn(opp_file), load_deck(opp_deck_path)
    elif kind == "self":
        best_model = _build_cpu_model(best_state)

        def opp_move(o):
            return mcts_train_agent(o, mcts_deck, best_model, search_count)[0]

        opp_deck = mcts_deck
    else:
        raise ValueError(f"unknown opponent kind: {kind}")

    res = [0, 0, 0]
    with torch.no_grad():
        for gi in game_indices:
            res[_play_vs_game(mcts_deck, model, opp_deck, opp_move, gi % 2, search_count)] += 1
    return res, DEFAULT.as_dict()


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
):
    import multiprocessing as mp

    # Opponents to evaluate the trained agent against. "random" = floor, "rule:<name>"
    # = a heuristic meta-deck agent (agents/<name>.py + decks/<name>.csv), "self" = the
    # best checkpoint so far. The best-checkpoint gate uses the rule-agent average.
    if eval_opponents is None:
        eval_opponents = ["random", "rule:zacian", "rule:abomasnow", "rule:dragapult", "rule:lucario", "self"]
    # Opponents to GENERATE self-play data against. "self" = MCTS mirror (both seats
    # collected); "rule:<name>" = learn to beat that heuristic (MCTS side collected).
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

    # Toggle batched MCTS leaf evaluation; set before the worker pool so spawned
    # children (and the inline path) inherit it via the environment.
    os.environ["MCTS_BATCHED"] = "1" if batched else "0"

    if deck_file is None:
        deck_file = "decks/abomasnow.csv"
    deck_path = comp_dir / deck_file
    sample_deck = load_deck(str(deck_path))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if num_workers is None:
        num_workers = max(1, (os.cpu_count() or 2) - 2)
    # No point spawning more workers than games in a phase.
    num_workers = max(1, min(num_workers, max(self_play_games, eval_games, 1)))
    print(f"Starting MCTS self-play training on {device} | rollout workers: {num_workers}...")

    init_checkpoint_path = None
    init_state_dict = None
    actual_model_args = _parse_model_args(model_args) or MODEL_ARGS
    if init_checkpoint:
        init_checkpoint_path = _resolve_init_checkpoint(init_checkpoint, "pokemon-tcg-ai-battle", comp_dir)
        init_state_dict = torch.load(init_checkpoint_path, map_location=device)
        actual_model_args = _model_args_from_state_dict(init_state_dict)
    model = MyModel(*actual_model_args).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
        print(f"Warm-started model from {init_checkpoint_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Cosine LR decay over the run so late iterations settle instead of wandering.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, iterations), eta_min=3e-5)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)

    # Replay buffer: train on a sliding window of recent self-play instead of only the
    # latest iteration's games — larger, more diverse batches => lower-variance updates
    # and less forgetting (the main source of the win-rate wobble).
    replay: list[LearnSample] = []

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    run_timings = Timings()  # cumulative wall-clock per phase + merged worker components

    # Fleet tracking: log live metrics to the central MLflow (or an offline sqlite fallback)
    # and register the best checkpoint in the model registry. Crash-safe — a no-op if MLflow
    # is unreachable, so training never blocks on telemetry. See the fleet-fabric spec.
    from kego.fleet import git_sha, machine_name
    from kego.tracking import Tracker, default_tracking_uri, register_checkpoint

    _uri = default_tracking_uri()
    _task = "pokemon-tcg-ai-battle"
    _repo_root = Path(__file__).resolve().parents[2]
    _agent_name = (
        f"mcts-{deck_path.stem}-"
        f"d{actual_model_args[0]}-h{actual_model_args[1]}-ff{actual_model_args[2]}-"
        f"enc{actual_model_args[3]}-dec{actual_model_args[4]}"
    )
    _run_tags = {
        "agent_name": _agent_name,
        "machine": machine_name(),
        "git_sha": git_sha(_repo_root),
        "task": _task,
        "deck": deck_path.stem,
        "search_count": str(search_count),
        "self_play_games": str(self_play_games),
        "batched": str(batched),
        "model_args": str(actual_model_args),
    }
    if init_checkpoint:
        _run_tags["continued_from"] = init_checkpoint
    # Attach to the dispatcher's run when dispatched (KEGO_MLFLOW_RUN_ID injected over SSH),
    # else open a fresh run. set_tags after open so our tags land whether new or resumed.
    _run_id = os.environ.get("KEGO_MLFLOW_RUN_ID")
    _track = Tracker.open(
        _uri, experiment=_task, run_id=_run_id, run_name=f"{_run_tags['machine']}-{output_file.stem}", tags=_run_tags
    )
    _track.set_tags(_run_tags)
    _track.log_params(
        {
            "iterations": iterations,
            "eval_games": eval_games,
            "self_play_games": self_play_games,
            "eval_every": eval_every,
            "search_count": search_count,
            "batched": batched,
            "num_workers": num_workers,
            "eval_opponents": ",".join(eval_opponents),
            "selfplay_opponents": ",".join(selfplay_opponents),
            "replay_buffer_size": replay_buffer_size,
            "train_steps": train_steps,
            "model_args": actual_model_args,
            "deck": deck_path.stem,
            "deck_file": deck_file,
            "init_checkpoint": init_checkpoint or "",
            "init_checkpoint_resolved": str(init_checkpoint_path) if init_checkpoint_path else "",
            "output_path": output_path,
        }
    )
    best_results: dict[str, float] = {}  # per-opponent WRs at the best gauntlet_avg, for registry tags

    # Self-play/eval run on CPU-only worker processes: batch-1 inference is far cheaper
    # on CPU than via per-call GPU launches/syncs, and the games are independent. The
    # cg engine keeps global state, so each game needs its own process (not thread).
    # Training (batched) still runs on the GPU in this process.
    pool = None
    if num_workers > 1:
        # Spawned children must be able to import this module + agents/ + cg, and must
        # run single-threaded torch (see _worker_init) to avoid core oversubscription.
        os.environ["PYTHONPATH"] = os.pathsep.join(
            p for p in [str(comp_dir), str(repo_root), os.environ.get("PYTHONPATH", "")] if p
        )
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        pool = mp.get_context("spawn").Pool(num_workers, initializer=_worker_init)

    def _run(worker, payloads):
        return pool.map(worker, payloads) if pool is not None else [worker(p) for p in payloads]

    opponent_specs = []  # (kind, name, agent_file, deck_file)
    for spec in eval_opponents:
        if spec in ("random", "self"):
            opponent_specs.append((spec, spec, None, None))
        elif spec.startswith("rule:"):
            nm = spec.split(":", 1)[1]
            opponent_specs.append(
                ("rule", nm, str(comp_dir / "agents" / f"{nm}.py"), str(comp_dir / "decks" / f"{nm}.csv"))
            )

    def _gauntlet(cur_state, best_state, seed_base):
        """Eval the current model vs each opponent; return {name: win_rate%} and the
        rule-agent average (the metric the best checkpoint is gated on)."""
        results = {}
        for kind, name, opp_file, opp_deck_path in opponent_specs:
            if kind == "self" and best_state is None:
                continue
            payloads = [
                (cur_state, kind, opp_file, opp_deck_path, sample_deck, chunk, seed_base + i, search_count, best_state)
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
            results[name] = wr
            print(f"  vs {name:<10} {wr:3d}%  (W{agg[0]} L{agg[1]} D{agg[2]})")
        rule_wrs = [wr for nm, wr in results.items() if nm not in ("random", "self")]
        avg = sum(rule_wrs) / len(rule_wrs) if rule_wrs else float(results.get("random", 0))
        print(f"  gauntlet avg (rule agents): {avg:.1f}%")
        return results, avg

    best_score = -1.0  # best rule-agent average; output_file always holds this checkpoint
    best_state = _transport_state(model) if init_checkpoint else None

    try:
        for iter_idx in range(iterations):
            print(f"\n--- Iteration {iter_idx + 1}/{iterations} ---")
            cpu_state = _transport_state(model)

            # 1. Evaluation gauntlet (throttled by eval_every, forced at checkpoint intervals); keep the best by rule avg.
            if eval_games > 0 and (iter_idx % eval_every == 0 or (iter_idx + 1) % 50 == 0):
                print("Evaluating gauntlet (random / rule agents / self)...")
                results, avg = _gauntlet(cpu_state, best_state, 7000 + iter_idx * 131)
                _track.log_metric("gauntlet_avg", avg, step=iter_idx)
                _track.log_metric("progress_pct", 100.0 * (iter_idx + 1) / iterations, step=iter_idx)
                for _name, _wr in results.items():
                    _track.log_metric(f"wr_{_name}", _wr, step=iter_idx)
                if avg > best_score:
                    best_score = avg
                    best_state = cpu_state
                    best_results = results
                    torch.save(model.state_dict(), str(output_file))
                    print(f"  -> new best (avg {avg:.1f}%), checkpointed to {output_file}")

            # 2. Self-Play data collection (parallel across workers)
            # Dynamic MCTS Search Scheduling: ramp search_count from min(10, search_count) up to search_count
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
                (cpu_state, c, sample_deck, iter_idx * 9973 + i, current_search_count, opp_pool)
                for i, c in enumerate(_split_counts(self_play_games, num_workers))
                if c
            ]
            with run_timings.timer("self_play"):
                returns = _run(_selfplay_worker, payloads)
            sample_list = []
            for samples, tdict in returns:
                sample_list.extend(samples)
                run_timings.merge(tdict)

            # 3. Model updates / Training — replay buffer + random minibatches.
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
                        for _ in range(64 - len(sample.policy)):
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
                _track.log_metric("loss_value", sum_enc / train_steps, step=iter_idx)
                _track.log_metric("loss_policy", sum_dec / train_steps, step=iter_idx)
                _track.log_metric("lr", scheduler.get_last_lr()[0], step=iter_idx)
                print(
                    f"Training complete. avg loss: value={sum_enc / train_steps:.4f} "
                    f"policy={sum_dec / train_steps:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
                )
            scheduler.step()
            run_timings.add("train", time.perf_counter() - _train_start)
            run_timings.report(f"timings @ iter {iter_idx + 1}")

            # Save and register intermediate checkpoint every 50 iterations
            checkpoint_interval = 50
            if (iter_idx + 1) % checkpoint_interval == 0:
                iter_output_file = output_file.parent / f"{output_file.stem}_iter{iter_idx + 1}.pth"
                torch.save(model.state_dict(), str(iter_output_file))
                print(f"  -> Intermediate checkpoint saved to {iter_output_file}")
                try:
                    iter_score = (
                        avg
                        if (
                            eval_games > 0 and (iter_idx % eval_every == 0 or (iter_idx + 1) % checkpoint_interval == 0)
                        )
                        else best_score
                    )
                    register_checkpoint(
                        _uri,
                        _task,
                        str(iter_output_file),
                        tags={
                            **_run_tags,
                            "epoch": str(iter_idx + 1),
                            "deck": deck_path.stem,
                            "gauntlet_avg": round(iter_score, 2) if iter_score >= 0 else 0.0,
                        },
                    )
                    print("  -> Registered intermediate checkpoint as registry model version.")
                except Exception as exc:  # noqa: BLE001
                    print(f"  (Registry unavailable, intermediate checkpoint not registered: {exc})")

        # Final gauntlet on the fully-trained model (pool still open); keep it if best.
        if eval_games > 0:
            print("\nFinal gauntlet on trained model...")
            final_state = _transport_state(model)
            results, avg = _gauntlet(final_state, best_state, 999983)
            if avg > best_score:
                best_score = avg
                best_state = final_state
                best_results = results
                torch.save(model.state_dict(), str(output_file))
                print(f"  -> new best (avg {avg:.1f}%), checkpointed to {output_file}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        # Register the best checkpoint (best-checkpointing already wrote it to output_file).
        # Skipped when we never evaluated (best_score < 0) — nothing to rank. Best-effort:
        # a registry outage must not fail an otherwise-good training run.
        if best_score >= 0:
            try:
                register_checkpoint(
                    _uri,
                    _task,
                    str(output_file),
                    tags={
                        **_run_tags,
                        "deck": deck_path.stem,
                        "gauntlet_avg": round(best_score, 2),
                        **{f"wr_{n}": round(w, 2) for n, w in best_results.items()},
                    },
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  (registry unavailable, checkpoint not registered: {exc})")
        _track.close()

    # If we never evaluated (eval_games=0), fall back to saving the final model.
    if best_score < 0:
        torch.save(model.state_dict(), str(output_file))

    run_timings.report("timings @ final")
    print(f"Done. Best rule-agent avg: {best_score:.1f}%. Model saved to {output_file}")


if __name__ == "__main__":
    run_training_loop(iterations=1, eval_games=2, self_play_games=2, output_path="outputs/mcts_model.pth")
