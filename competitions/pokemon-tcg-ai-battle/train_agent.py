import math
import random
import sys
from pathlib import Path

import torch

# Setup paths to import cg and agents
comp_dir = Path(__file__).parent.resolve()
repo_root = comp_dir.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(comp_dir))

from kego.pipeline.battle import load_deck, locate_cg_dir

cg_parent = locate_cg_dir()
sys.path.insert(0, str(cg_parent))

from agents.mcts import (
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
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
    )
    return (value.tolist()[0][0], policy.tolist()[0])


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

        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn_train(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)

        total_prob = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            total_prob += p
        for c in node.children:
            c.prob /= total_prob
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return node, sample


def mcts_train_agent(
    obs_dict: dict, your_deck: list[int], model: MyModel, search_count=10
) -> tuple[list[int], LearnSample]:
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active

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

    for _ in range(search_count):
        current = root
        while True:
            value = -1e9
            c = 0.4 * math.sqrt(current.visit)
            for child in current.children:
                visit = 0
                if child.node is None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if value < v:
                    value = v
                    next_node = child

            if next_node.node is None:
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


def run_training_loop(iterations=3, eval_games=5, self_play_games=10, output_path="outputs/mcts_model.pth"):
    deck_path = comp_dir / "decks" / "abomasnow.csv"
    sample_deck = load_deck(str(deck_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting MCTS self-play training on {device}...")
    model = MyModel(128, 2, 256, 1, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    for iter_idx in range(iterations):
        print(f"\n--- Iteration {iter_idx + 1}/{iterations} ---")

        # 1. Evaluation against Random Agent
        if eval_games > 0:
            model.eval()
            print("Evaluating model against Random baseline...")
            results = [0, 0, 0]  # win, lose, draw
            with torch.no_grad():
                for game_idx in range(eval_games):
                    obs, start_data = battle_start(sample_deck, sample_deck)
                    your_index = game_idx % 2
                    while True:
                        if obs["current"]["result"] >= 0:
                            break
                        if obs["current"]["yourIndex"] == your_index:
                            selected, _ = mcts_train_agent(obs, sample_deck, model)
                        else:
                            selected = random_agent(obs)
                        obs = battle_select(selected)
                    battle_finish()

                    res_val = obs["current"]["result"]
                    if res_val == 2:
                        results[2] += 1
                    elif res_val == your_index:
                        results[0] += 1
                    else:
                        results[1] += 1
            win_rate = (100 * results[0]) // max(1, results[0] + results[1])
            print(f"Evaluation Win Rate: {win_rate}% (Wins: {results[0]}, Losses: {results[1]}, Draws: {results[2]})")

        # 2. Self-Play data collection
        sample_list = []
        model.eval()
        print(f"Collecting self-play training data ({self_play_games} games)...")
        with torch.no_grad():
            for game_idx in range(self_play_games):
                obs, _ = battle_start(sample_deck, sample_deck)
                samples = [[], []]
                while True:
                    if obs["current"]["result"] >= 0:
                        break
                    selected, sample = mcts_train_agent(obs, sample_deck, model)
                    samples[obs["current"]["yourIndex"]].append(sample)
                    obs = battle_select(selected)
                battle_finish()

                for i in range(2):
                    LAMBDA = 0.9
                    value = 1.0 if i == obs["current"]["result"] else -1.0
                    for sample in reversed(samples[i]):
                        label = (value + sample.value) * 0.5
                        value = value * LAMBDA + sample.value * (1.0 - LAMBDA)
                        sample.value = label
                        sample_list.append(sample)

        # 3. Model updates / Training
        print(f"Training on {len(sample_list)} collected samples...")
        model.train()
        random.shuffle(sample_list)
        BATCH_SIZE = min(128, len(sample_list))
        if BATCH_SIZE > 0:
            batch_count = len(sample_list) // BATCH_SIZE
            for batch_idx in range(batch_count):
                input_enc = LearnInput()
                input_dec = LearnInput()
                mask = []
                label_enc = []
                label_dec = []
                start = BATCH_SIZE * batch_idx
                for j in range(start, start + BATCH_SIZE):
                    sample = sample_list[j]
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

                loss.backward()
                optimizer.step()
            print("Training epoch complete.")

    torch.save(model.state_dict(), str(output_file))
    print(f"Model successfully trained and saved to {output_file}")


if __name__ == "__main__":
    run_training_loop(iterations=1, eval_games=2, self_play_games=2, output_path="outputs/mcts_model.pth")
