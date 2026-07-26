"""Shared MCTS tree mechanics.

Selection is PUCT-style: ``q + 0.4*sqrt(N_parent)*prior/(1+n)``. Priors use
``exp(policy*10)`` and candidate combinations are capped at 64 to bound inference
latency on large multi-select prompts. Values use the root player's perspective:
+1 is a win for ``your_index`` and values are negated at opponent-to-move nodes.

Inference determinizes hidden information on every move: the agent samples its own
unknown deck and Prize cards, while unknown opponent cards use Snorlax card 1072 as
a filler. That determinization is prepared by ``agent.py`` before this module builds
the search tree.
"""

import math
from collections.abc import Callable

# The cg API exposes match results as raw integers: player 0, player 1, or draw.
import numpy as np
import torch

from cg.api import Observation, SearchState

from .encoding import SparseVector
from .model import PolicyValueNet

RESULT_DRAW = 2
EXPLORATION_C = 0.4
POLICY_TEMPERATURE = 10.0
MAX_ACTION_COMBINATIONS = 64


def evaluate_position(sv_enc: SparseVector, sv_dec: SparseVector, model: PolicyValueNet) -> tuple[float, list[float]]:
    device = next(model.parameters()).device

    if not hasattr(model, "_enc_index_buf"):
        model._enc_index_buf = torch.zeros(30000, dtype=torch.int32, device=device)
        model._enc_value_buf = torch.zeros(30000, dtype=torch.float32, device=device)
        model._enc_offset_buf = torch.zeros(100, dtype=torch.int32, device=device)
        model._dec_index_buf = torch.zeros(100000, dtype=torch.int32, device=device)
        model._dec_value_buf = torch.zeros(100000, dtype=torch.float32, device=device)
        model._dec_offset_buf = torch.zeros(1000, dtype=torch.int32, device=device)

    enc_idx_len = len(sv_enc.index)
    enc_val_len = len(sv_enc.value)
    enc_off_len = len(sv_enc.offset)
    dec_idx_len = len(sv_dec.index)
    dec_val_len = len(sv_dec.value)
    dec_off_len = len(sv_dec.offset)

    if enc_idx_len > model._enc_index_buf.shape[0]:
        new_size = max(enc_idx_len, model._enc_index_buf.shape[0] * 2)
        model._enc_index_buf = torch.zeros(new_size, dtype=torch.int32, device=device)
        model._enc_value_buf = torch.zeros(new_size, dtype=torch.float32, device=device)
    if enc_off_len > model._enc_offset_buf.shape[0]:
        new_size = max(enc_off_len, model._enc_offset_buf.shape[0] * 2)
        model._enc_offset_buf = torch.zeros(new_size, dtype=torch.int32, device=device)

    if dec_idx_len > model._dec_index_buf.shape[0]:
        new_size = max(dec_idx_len, model._dec_index_buf.shape[0] * 2)
        model._dec_index_buf = torch.zeros(new_size, dtype=torch.int32, device=device)
        model._dec_value_buf = torch.zeros(new_size, dtype=torch.float32, device=device)
    if dec_off_len > model._dec_offset_buf.shape[0]:
        new_size = max(dec_off_len, model._dec_offset_buf.shape[0] * 2)
        model._dec_offset_buf = torch.zeros(new_size, dtype=torch.int32, device=device)

    enc_idx_arr = np.array(sv_enc.index, dtype=np.int32)
    enc_val_arr = np.array(sv_enc.value, dtype=np.float32)
    enc_off_arr = np.array(sv_enc.offset, dtype=np.int32)
    dec_idx_arr = np.array(sv_dec.index, dtype=np.int32)
    dec_val_arr = np.array(sv_dec.value, dtype=np.float32)
    dec_off_arr = np.array(sv_dec.offset, dtype=np.int32)

    model._enc_index_buf[:enc_idx_len].copy_(torch.from_numpy(enc_idx_arr))
    model._enc_value_buf[:enc_val_len].copy_(torch.from_numpy(enc_val_arr))
    model._enc_offset_buf[:enc_off_len].copy_(torch.from_numpy(enc_off_arr))
    model._dec_index_buf[:dec_idx_len].copy_(torch.from_numpy(dec_idx_arr))
    model._dec_value_buf[:dec_val_len].copy_(torch.from_numpy(dec_val_arr))
    model._dec_offset_buf[:dec_off_len].copy_(torch.from_numpy(dec_off_arr))

    enc_idx = model._enc_index_buf[:enc_idx_len]
    enc_val = model._enc_value_buf[:enc_val_len]
    enc_off = model._enc_offset_buf[:enc_off_len]
    dec_idx = model._dec_index_buf[:dec_idx_len]
    dec_val = model._dec_value_buf[:dec_val_len]
    dec_off = model._dec_offset_buf[:dec_off_len]

    with torch.no_grad():
        value, policy = model(enc_idx, enc_val, enc_off, dec_idx, dec_val, dec_off)
    return (value.tolist()[0][0], policy.tolist()[0])


class Child:
    def __init__(self, select: list[int], prob: float) -> None:
        self.node = None
        self.select = select
        self.prob = prob


class Node:
    def __init__(self, parent: "Node | None", state: SearchState) -> None:
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    def backprop(self, value: float) -> None:
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def enumerate_action_combinations(
    max_count: int, num_options: int, cap: int = MAX_ACTION_COMBINATIONS
) -> list[list[int]]:
    """Enumerate index combinations of size `max_count` from `num_options` options.

    Combinations are generated in lexicographic order (each a sorted list of distinct
    indices in `range(num_options)`), and generation stops once `cap` combinations have
    been produced.
    """
    if max_count > num_options:
        max_count = num_options
    if max_count < 0 or cap <= 0 or (num_options == 0 and max_count > 0):
        return []
    actions = []
    indices = list(range(max_count))
    for _ in range(cap):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < num_options - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions


def build_children(
    node: Node, actions: list[list[int]], policy: list[float], policy_temperature: float = POLICY_TEMPERATURE
) -> None:
    """Attach softmax-weighted children to a node from a policy vector."""
    total_prob = 0.0
    for i in range(len(policy)):
        p = math.exp(policy[i] * policy_temperature)
        node.children.append(Child(actions[i], p))
        total_prob += p
    for c in node.children:
        c.prob /= total_prob


def select_child(current: Node, your_index: int, c_puct: float = EXPLORATION_C) -> Child | None:
    """UCB-select the best child of ``current`` (None if it has none)."""
    best, chosen = -1e18, None
    c = c_puct * math.sqrt(current.visit)
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


Evaluator = Callable[[Observation, list[list[int]]], tuple[float, list[float]]]


def create_node(
    parent: Node | None,
    search_state: SearchState,
    your_index: int,
    evaluate: Evaluator,
    c_puct: float = EXPLORATION_C,
    policy_temperature: float = POLICY_TEMPERATURE,
    max_action_combinations: int = MAX_ACTION_COMBINATIONS,
) -> Node:
    if max_action_combinations <= 0:
        raise ValueError("max_action_combinations must be greater than zero")
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current

    if state.result >= 0:
        if state.result == RESULT_DRAW:
            node.value = 0.0
        elif state.result == your_index:
            node.value = 1.0
        else:
            node.value = -1.0
        node.backprop(node.value)
    else:
        actions = enumerate_action_combinations(
            obs.select.maxCount,
            len(obs.select.option),
            cap=max_action_combinations,
        )
        value, policy = evaluate(obs, actions)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)
        build_children(node, actions, policy, policy_temperature=policy_temperature)
    return node
