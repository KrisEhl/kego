import math
import os
import random

import torch

try:
    from agents.base import BaseAgent
    from agents.base import get_card as get_card_helper
except ImportError:
    from base_agent import BaseAgent
    from base_agent import get_card as get_card_helper
from cg.api import (
    Card,
    Observation,
    PlayerState,
    Pokemon,
    SearchState,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)

# Global metadata loading for the neural network features
try:
    all_card = all_card_data()
    card_count = max(all_card, key=lambda c: c.cardId).cardId + 1
    attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1
except Exception:
    card_count = 2000
    attack_count = 2000

num_words_encoder = 24
encoder_size = 22000
decoder_main_feature = 8
decoder_attack_offset = 14
decoder_card_offset = decoder_attack_offset + attack_count
decoder_size = decoder_card_offset + (1 + decoder_main_feature + 48) * card_count

# (d_model, num_heads, d_feedforward, n_encoder_layers, n_decoder_layers).
# Single source of truth: training (train_agent.py) and inference both build MyModel
# from this, so checkpoints always match. num_heads must divide d_model.
MODEL_ARGS = (192, 4, 384, 2, 2)


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class MyModel(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_feedforward: int, num_layers_encoder: int, num_layers_decoder: int
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(self, index_encoder, value_encoder, offset_encoder, index_decoder, value_decoder, offset_decoder):
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)


class SparseVector:
    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


def add_card(sv: SparseVector, card: Card | Pokemon | None):
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)


def add_cards(sv: SparseVector, cards: list[Card] | None, value: float):
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)


def add_pokemon(sv: SparseVector, poke: Pokemon | None):
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400.0)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60.0)
    sv.add_single(len(ps.discard) / 60.0)
    sv.add_single(ps.handCount / 8.0)
    sv.add_single(len(ps.bench) / 5.0)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)
    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if ps.active:
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()
    for id in your_deck:
        sv.add(id, 0.25)
    sv.add_pos(card_count)

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10.0)
    sv.add_single(state.firstPlayer == your_index)
    return sv


def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None):
    if card is not None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)


def decoder_card_id(sv: SparseVector, context: int, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)


def decoder_card(sv: SparseVector, context: int, card: Card | Pokemon | None):
    if card is not None:
        decoder_card_id(sv, context, card.id)


def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = int(obs.select.context) if hasattr(obs.select.context, "value") else int(obs.select.context)

    for action in actions:
        sv.word_start()
        if not action:
            sv.add(0, 1)
            continue

        for i in action:
            o = obs.select.option[i]
            o_type = int(o.type) if hasattr(o.type, "value") else int(o.type)
            # OptionType mapping:
            # END=14, YES=1, NO=2, SPECIAL_CONDITION=16, NUMBER=0, ATTACK=13, PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10, DISCARD=11, RETREAT=12
            if o_type == 14:
                sv.add(1, 1)
            elif o_type == 1:
                sv.add(2, 1)
            elif o_type == 2:
                sv.add(3, 1)
            elif o_type == 16:
                sv.add(4 + int(o.specialConditionType), 1)
            elif o_type == 0:
                sv.add(9 + min(o.number, 4), 1)
            elif o_type == 13:
                sv.add(decoder_attack_offset + o.attackId, 1)
            elif o_type == 7:
                decoder_main(sv, 0, ps.hand[o.index] if ps.hand else None)
            elif o_type == 8:
                decoder_main(sv, 1, get_card_helper(obs, int(o.area), o.index, your_index))
                decoder_main(sv, 2, get_card_helper(obs, int(o.inPlayArea), o.inPlayIndex, your_index))
            elif o_type == 9:
                decoder_main(sv, 3, get_card_helper(obs, int(o.area), o.index, your_index))
                decoder_main(sv, 4, get_card_helper(obs, int(o.inPlayArea), o.inPlayIndex, your_index))
            elif o_type == 10:
                decoder_main(sv, 5, get_card_helper(obs, int(o.area), o.index, your_index))
            elif o_type == 11:
                decoder_main(sv, 6, get_card_helper(obs, int(o.area), o.index, your_index))
            elif o_type == 12:
                decoder_main(sv, 7, ps.active[0] if ps.active else None)
            elif o_type == 3:  # CARD
                decoder_card(sv, context, get_card_helper(obs, int(o.area), o.index, o.playerIndex))
            elif o_type == 4:  # TOOL_CARD
                card = get_card_helper(obs, int(o.area), o.index, o.playerIndex)
                if card and hasattr(card, "tools") and o.toolIndex < len(card.tools):
                    decoder_card(sv, context, card.tools[o.toolIndex])
            elif o_type in (5, 6):  # ENERGY_CARD / ENERGY
                card = get_card_helper(obs, int(o.area), o.index, o.playerIndex)
                if card and hasattr(card, "energyCards") and o.energyIndex < len(card.energyCards):
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
            elif o_type == 15:  # SKILL
                decoder_card_id(sv, context, o.cardId)
    return sv


def eval_nn(sv_enc: SparseVector, sv_dec: SparseVector, model: MyModel) -> tuple[float, list[float]]:
    device = next(model.parameters()).device
    with torch.no_grad():
        value, policy = model(
            torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
            torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
            torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
            torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
            torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
        )
    return (value.tolist()[0][0], policy.tolist()[0])


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


def create_node(
    parent: Node | None, search_state: SearchState, your_index: int, your_deck: list[int], model: MyModel
) -> Node:
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
        value, policy = eval_nn(sv_enc, sv_dec, model)
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

    return node


class MCTSTransformerAgent(BaseAgent):
    def __init__(self, deck="abomasnow.csv", model_path=None, model_args=None):
        self.deck = self._load_deck(deck)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_args = tuple(model_args or MODEL_ARGS)
        self.model = MyModel(*self.model_args).to(self.device)
        self.model.eval()

        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"[MCTSTransformerAgent] model_path not found: {model_path}")
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[MCTSTransformerAgent] loaded weights from {model_path}", flush=True)
            except Exception as e:
                raise ValueError(f"[MCTSTransformerAgent] FAILED to load weights from {model_path}: {e}") from e

        # Inference search depth (override with MCTS_SEARCH_COUNT). More = stronger,
        # slower. Inference has no training-time budget, so it can go much deeper than
        # self-play's search_count.
        self.SEARCH_COUNT = int(os.environ.get("MCTS_SEARCH_COUNT", "10"))
        print(f"[MCTSTransformerAgent] SEARCH_COUNT={self.SEARCH_COUNT}", flush=True)

    def get_deck(self) -> list[int]:
        return self.deck

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self.get_deck()

        your_index = obs.current.yourIndex
        state = obs.current
        active = state.players[1 - your_index].active

        # Perform Search API initialization
        search_state = search_begin(
            obs,
            your_deck=random.sample(self.deck, state.players[your_index].deckCount),
            your_prize=random.sample(self.deck, len(state.players[your_index].prize)),
            opponent_deck=[1072] * state.players[1 - your_index].deckCount,  # Snorlax filler
            opponent_prize=[1] * len(state.players[1 - your_index].prize),
            opponent_hand=[1] * state.players[1 - your_index].handCount,
            opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
        )

        root = create_node(None, search_state, your_index, self.deck, self.model)

        for _ in range(self.SEARCH_COUNT):
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
                    v += c * child.prob / (1.0 + visit)
                    if value < v:
                        value = v
                        next_child = child

                if next_child.node is None:
                    s_state = search_step(current.state.searchId, next_child.select)
                    next_child.node = create_node(current, s_state, your_index, self.deck, self.model)
                    break
                else:
                    current = next_child.node
                    if current.state.observation.current.result >= 0:
                        current.backprop(current.value)
                        break

        # Select the child with the highest visit count
        max_child = None
        max_visit = -1
        for child in root.children:
            if child.node is not None:
                if max_visit < child.node.visit:
                    max_child = child
                    max_visit = child.node.visit

        search_end()
        if max_child is not None:
            return max_child.select
        return root.children[0].select if root.children else [0]


# Define agent instance wrapper
_agent_instance = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        # MCTS_MODEL_PATH / MCTS_DECK let callers (e.g. the tournament) plug in a
        # trained checkpoint + deck; unset => untrained weights on the default deck.
        _agent_instance = MCTSTransformerAgent(
            deck=os.environ.get("MCTS_DECK", "abomasnow.csv"),
            model_path=os.environ.get("MCTS_MODEL_PATH"),
        )
    return _agent_instance.act(obs_dict)
