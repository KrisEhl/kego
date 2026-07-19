"""Integration tests for the pokemon MCTS self-play (sequential + leaf-batched).

These need the competition's `cg` game engine (shipped in the competition data). If it
isn't importable (e.g. CI without the data), the whole module is skipped.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

COMP = Path(__file__).resolve().parents[1] / "competitions" / "pokemon-tcg-ai-battle"


@pytest.fixture(scope="module")
def tm():
    """Load competitions/.../train_agent.py the way task.py does (registers it in
    sys.modules). Skips if the cg engine / decks aren't present."""
    if not (COMP / "train_agent.py").exists():
        pytest.skip("pokemon competition not present")
    spec = importlib.util.spec_from_file_location("train_agent", str(COMP / "train_agent.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_agent"] = mod
    try:
        spec.loader.exec_module(mod)  # runs locate_cg_dir() + imports cg
    except Exception as e:  # cg engine unavailable, missing data, etc.
        pytest.skip(f"pokemon MCTS env unavailable: {e}")
    return mod


@pytest.fixture(scope="module")
def model(tm):
    torch.manual_seed(0)
    m = tm.PolicyValueNet(*tm.MODEL_ARGS)
    m.eval()
    return m


@pytest.fixture(scope="module")
def deck(tm):
    return tm.load_deck(str(COMP / "decks" / "abomasnow.csv"))


def _self_play(tm, model, deck, search_count):
    with torch.no_grad():
        return tm._collect_game(deck, model, search_count, None, deck, 0)


def test_sparse_vector_tracks_words_and_absolute_positions(tm):
    sv = tm.SparseVector()
    sv.word_start()
    sv.add(2, 0.5)
    sv.add_single(1)
    sv.add_pos(3)
    sv.word_start()
    sv.add(1, 2)

    assert sv.index == [2, 0, 5]
    assert sv.value == [0.5, 1.0, 2.0]
    assert sv.offset == [0, 2]


def test_sparse_vector_omits_zero_values_but_advances_single_position(tm):
    sv = tm.SparseVector()
    sv.add(3, 0)
    sv.add_single(False)
    sv.add(2, True)

    assert sv.pos == 1
    assert sv.index == [3]
    assert sv.value == [1.0]


def test_sparse_vector_preserves_empty_word_boundaries(tm):
    sv = tm.SparseVector()
    sv.word_start()
    sv.word_start()
    sv.add_single(1)
    sv.word_start()

    assert sv.offset == [0, 0, 1]


@pytest.mark.parametrize("batched", ["1", "0"])
def test_selfplay_game_completes_and_produces_samples(tm, model, deck, batched):
    """A full self-play game runs to completion and yields valid training samples,
    in both leaf-batched (MCTS_BATCHED=1) and sequential (=0) modes."""
    os.environ["MCTS_BATCHED"] = batched
    samples = _self_play(tm, model, deck, search_count=6)
    assert len(samples) > 0
    for s in samples:
        assert -1.0 <= s.value <= 1.0  # bootstrapped value target stays in range
        assert len(s.policy) == len(s.sv_dec.offset)  # one policy entry per candidate action


def test_mcts_move_returns_valid_selection(tm, model, deck):
    """mcts_train_agent returns a concrete move (list of selection indices) + a sample."""
    os.environ["MCTS_BATCHED"] = "1"
    from cg.game import battle_start

    obs, _ = battle_start(deck, deck)
    with torch.no_grad():
        selected, sample = tm.mcts_train_agent(obs, deck, model, 6)
    assert isinstance(selected, list) and len(selected) > 0
    assert all(isinstance(i, int) for i in selected)
    assert sample is not None


def test_eval_nn_batch_matches_individual(tm, model, deck):
    """The batched evaluator must return the same value/policy as evaluating each input
    individually — this is where offset/mask/reshape bugs would show up."""
    os.environ["MCTS_BATCHED"] = "0"
    samples = _self_play(tm, model, deck, search_count=6)
    svs = [(s.sv_enc, s.sv_dec) for s in samples[:8]]
    assert len(svs) >= 2  # need a real batch
    with torch.no_grad():
        batched = tm.eval_nn_batch(svs, model)
        for (v_b, p_b), (sv_enc, sv_dec) in zip(batched, svs):
            v_i, p_i = tm.eval_nn_train(sv_enc, sv_dec, model)
            assert abs(v_b - v_i) < 1e-4
            assert len(p_b) == len(p_i)
            assert all(abs(a - b) < 1e-4 for a, b in zip(p_b, p_i))


def test_leaf_batching_uses_fewer_forwards_per_move(tm, model, deck):
    """Leaf-batching should use fewer NN forwards than sequential search while still
    processing additional waves when a frontier is narrower than the search budget."""
    from kego.timing import DEFAULT

    ratios = {}
    for batched in ("1", "0"):
        os.environ["MCTS_BATCHED"] = batched
        DEFAULT.reset()
        _self_play(tm, model, deck, search_count=8)
        moves = DEFAULT.count.get("mcts_move", 0)
        nn = DEFAULT.count.get("nn_eval", 0)
        assert moves > 0
        ratios[batched] = nn / moves

    assert ratios["1"] < ratios["0"]  # batched does fewer forwards per move
    assert ratios["1"] < 5.0


def test_leaf_batching_uses_full_budget_without_leaking_virtual_loss(tm, model, deck, monkeypatch):
    os.environ["MCTS_BATCHED"] = "1"
    roots = []
    create_node_train = tm.create_node_train

    def capture_root(parent, *args, **kwargs):
        node, sample = create_node_train(parent, *args, **kwargs)
        if parent is None:
            roots.append(node)
        return node, sample

    monkeypatch.setattr(tm, "create_node_train", capture_root)
    from cg.game import battle_finish, battle_start

    obs, _ = battle_start(deck, deck)
    try:
        with torch.no_grad():
            tm.mcts_train_agent(obs, deck, model, search_count=70)
    finally:
        battle_finish()

    root = roots[0]
    assert root.visit == 71
    assert any(child.node and child.node.children for child in root.children)


def test_leaf_batching_restores_virtual_loss_exactly(tm, model, deck, monkeypatch):
    os.environ["MCTS_BATCHED"] = "1"
    roots = []
    initial_totals = []
    create_node_train = tm.create_node_train

    def capture_root(parent, *args, **kwargs):
        node, sample = create_node_train(parent, *args, **kwargs)
        if parent is None:
            roots.append(node)
            initial_totals.append(node.total)
        return node, sample

    def zero_batch(svs, _model):
        return [(0.0, [0.0] * len(sv_dec.offset)) for _, sv_dec in svs]

    monkeypatch.setattr(tm, "create_node_train", capture_root)
    monkeypatch.setattr(tm, "eval_nn_batch", zero_batch)
    from cg.game import battle_finish, battle_start

    obs, _ = battle_start(deck, deck)
    try:
        with torch.no_grad():
            tm.mcts_train_agent(obs, deck, model, search_count=1)
    finally:
        battle_finish()

    assert roots[0].visit == 2
    assert roots[0].total == pytest.approx(initial_totals[0])


def test_training_logs_and_registers_best(tm, deck, tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from kego.tracking import leaderboard

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    monkeypatch.setenv("KEGO_MLFLOW", uri)
    monkeypatch.setenv("KEGO_MACHINE", "test-box")

    tm.run_training_loop(
        iterations=1,
        eval_games=2,
        self_play_games=4,
        output_path=str(tmp_path / "m.pth"),
        num_workers=2,
        eval_every=1,
        search_count=4,
        batched=True,
        selfplay_opponents=["self"],
        eval_opponents=["random"],
        config_fingerprint="test-fingerprint",
    )

    board = leaderboard(uri, "pokemon-tcg-ai-battle", sort_by="gauntlet_avg")
    assert len(board) >= 1
    assert board[0]["machine"] == "test-box"
    assert "gauntlet_avg" in board[0]
    assert board[0]["git_sha"] != ""
    assert board[0]["training_fingerprint"] == "test-fingerprint"
    assert board[0]["training_run_id"] == board[0]["run_id"]
    assert board[0]["completed_iterations"] == "1"
    assert 0 <= int(board[0]["epoch"]) <= 1
    assert board[0]["training_state_filename"].endswith(".train.pt")
    state = torch.load(tmp_path / "m_iter1.train.pt", map_location="cpu", weights_only=False)
    assert state["best_iteration"] == int(board[0]["epoch"])


def test_training_uses_exact_compatible_registry_state_without_retraining(tm, tmp_path, monkeypatch, capsys):
    pytest.importorskip("mlflow")
    from kego.tracking import register_checkpoint

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    monkeypatch.setenv("KEGO_MLFLOW", uri)
    model = tm.PolicyValueNet(*tm.MODEL_ARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    weights = tmp_path / "weights.pth"
    state = tmp_path / "weights_iter1.train.pt"
    torch.save(model.state_dict(), weights)
    torch.save(
        {
            "format_version": 1,
            "training_fingerprint": "compatible",
            "completed_iterations": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "replay": [],
            "best_state": model.state_dict(),
            "best_score": 50.0,
            "best_results": {"random": 50},
            "python_rng_state": __import__("random").getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "target_iterations": 1,
        },
        state,
    )
    register_checkpoint(
        uri,
        "pokemon-tcg-ai-battle",
        str(weights),
        tags={"training_fingerprint": "compatible", "completed_iterations": 1},
        training_state_path=str(state),
    )
    output = tmp_path / "resumed.pth"

    tm.run_training_loop(
        iterations=1,
        eval_games=0,
        self_play_games=0,
        output_path=str(output),
        config_fingerprint="compatible",
    )

    assert output.exists()
    assert "already reached target iteration 1" in capsys.readouterr().out


def test_training_resumes_only_remaining_iterations(tm, tmp_path, monkeypatch, capsys):
    pytest.importorskip("mlflow")
    from kego.tracking import register_checkpoint

    uri = f"sqlite:///{tmp_path / 'ml.db'}"
    monkeypatch.setenv("KEGO_MLFLOW", uri)
    model = tm.PolicyValueNet(*tm.MODEL_ARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    weights = tmp_path / "weights.pth"
    state = tmp_path / "weights_iter1.train.pt"
    torch.save(model.state_dict(), weights)
    torch.save(
        {
            "format_version": 1,
            "training_fingerprint": "compatible",
            "completed_iterations": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "replay": [],
            "best_state": model.state_dict(),
            "best_score": 50.0,
            "best_results": {"random": 50},
            "python_rng_state": __import__("random").getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "target_iterations": 1,
        },
        state,
    )
    register_checkpoint(
        uri,
        "pokemon-tcg-ai-battle",
        str(weights),
        tags={"training_fingerprint": "compatible", "completed_iterations": 1},
        training_state_path=str(state),
    )

    tm.run_training_loop(
        iterations=2,
        eval_games=0,
        self_play_games=0,
        output_path=str(tmp_path / "resumed.pth"),
        config_fingerprint="compatible",
    )

    out = capsys.readouterr().out
    assert "--- Iteration 2/2 ---" in out
    assert "--- Iteration 1/2 ---" not in out
