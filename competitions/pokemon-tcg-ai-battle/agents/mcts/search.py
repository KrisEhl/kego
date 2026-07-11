import math
from collections.abc import Callable

import torch

from cg.api import Observation, SearchState

from .encoding import SparseVector
from .model import MyModel

# The cg API exposes match results as raw integers: player 0, player 1, or draw.
RESULT_DRAW = 2


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


def enumerate_action_combinations(max_count: int, num_options: int, cap: int = 64) -> list[list[int]]:
    """Enumerate index combinations of size `max_count` from `num_options` options.

    Combinations are generated in lexicographic order (each a sorted list of distinct
    indices in `range(num_options)`), and generation stops once `cap` combinations have
    been produced.
    """
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


def build_children(node: Node, actions: list[list[int]], policy: list[float]) -> None:
    """Attach softmax-weighted children to a node from a policy vector."""
    total_prob = 0.0
    for i in range(len(policy)):
        p = math.exp(policy[i] * 10.0)
        node.children.append(Child(actions[i], p))
        total_prob += p
    for c in node.children:
        c.prob /= total_prob


def select_child(current: Node, your_index: int):
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


Evaluator = Callable[[Observation, list[list[int]]], tuple[float, list[float]]]


def create_node(parent: Node | None, search_state: SearchState, your_index: int, evaluate: Evaluator) -> Node:
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
        actions = enumerate_action_combinations(obs.select.maxCount, len(obs.select.option))
        value, policy = evaluate(obs, actions)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)
        build_children(node, actions, policy)
    return node
