"""Sparse state and action encoding for the MCTS policy/value network.

The state encoder emits 24 words: two players times eight bench words (sharing
positions within each player's block), two active words, two player-state words,
then hand, own-deck, stadium, and misc words. Continuous scales are HP/400,
deck and discard counts/60, hand/8, bench/5, discarded cards weighted 0.25,
and attached Energy weighted 0.5. The misc word contains turn and first-player
features.

Action features start with 14 fixed slots, followed by the attack block at offset
14, then card blocks sized ``(1 + 8 roles + 48 contexts) * card_count``.
"""

try:
    from base_agent import get_card as get_card_helper
except ImportError:
    from agents.base import get_card as get_card_helper
from cg.api import Card, Observation, OptionType, PlayerState, Pokemon

from .model import CARD_COUNT, DECODER_ATTACK_OFFSET, DECODER_CARD_OFFSET, DECODER_MAIN_FEATURE


class SparseVector:
    def __init__(self) -> None:
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool) -> None:
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int) -> None:
        self.pos += pos

    def add_single(self, value: float | int | bool) -> None:
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self) -> None:
        self.offset.append(len(self.index))


def add_card(sv: SparseVector, card: Card | Pokemon | None) -> None:
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(CARD_COUNT)


def add_cards(sv: SparseVector, cards: list[Card] | None, value: float) -> None:
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(CARD_COUNT)


def add_pokemon(sv: SparseVector, poke: Pokemon | None) -> None:
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * CARD_COUNT)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400.0)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)


def add_player(sv: SparseVector, ps: PlayerState) -> None:
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


def encode_state(obs: Observation, your_deck: list[int]) -> SparseVector:
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
    sv.add_pos(CARD_COUNT)

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10.0)
    sv.add_single(state.firstPlayer == your_index)
    return sv


def add_role_card_feature(sv: SparseVector, feature_index: int, card: Card | Pokemon | None) -> None:
    if card is not None:
        sv.add(DECODER_CARD_OFFSET + feature_index * CARD_COUNT + card.id, 1)


def add_context_card_id(sv: SparseVector, context: int, card_id: int) -> None:
    sv.add(DECODER_CARD_OFFSET + (DECODER_MAIN_FEATURE + context) * CARD_COUNT + card_id, 1)


def add_context_card(sv: SparseVector, context: int, card: Card | Pokemon | None) -> None:
    if card is not None:
        add_context_card_id(sv, context, card.id)


def encode_actions(obs: Observation, actions: list[list[int]]) -> SparseVector:
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
            if o_type == OptionType.END:
                sv.add(1, 1)
            elif o_type == OptionType.YES:
                sv.add(2, 1)
            elif o_type == OptionType.NO:
                sv.add(3, 1)
            elif o_type == OptionType.SPECIAL_CONDITION:
                sv.add(4 + int(o.specialConditionType), 1)
            elif o_type == OptionType.NUMBER:
                sv.add(9 + min(o.number, 4), 1)
            elif o_type == OptionType.ATTACK:
                sv.add(DECODER_ATTACK_OFFSET + o.attackId, 1)
            elif o_type == OptionType.PLAY:
                add_role_card_feature(sv, 0, ps.hand[o.index] if ps.hand else None)
            elif o_type == OptionType.ATTACH:
                add_role_card_feature(sv, 1, get_card_helper(obs, int(o.area), o.index, your_index))
                add_role_card_feature(sv, 2, get_card_helper(obs, int(o.inPlayArea), o.inPlayIndex, your_index))
            elif o_type == OptionType.EVOLVE:
                add_role_card_feature(sv, 3, get_card_helper(obs, int(o.area), o.index, your_index))
                add_role_card_feature(sv, 4, get_card_helper(obs, int(o.inPlayArea), o.inPlayIndex, your_index))
            elif o_type == OptionType.ABILITY:
                add_role_card_feature(sv, 5, get_card_helper(obs, int(o.area), o.index, your_index))
            elif o_type == OptionType.DISCARD:
                add_role_card_feature(sv, 6, get_card_helper(obs, int(o.area), o.index, your_index))
            elif o_type == OptionType.RETREAT:
                add_role_card_feature(sv, 7, ps.active[0] if ps.active else None)
            elif o_type == OptionType.CARD:
                add_context_card(sv, context, get_card_helper(obs, int(o.area), o.index, o.playerIndex))
            elif o_type == OptionType.TOOL_CARD:
                card = get_card_helper(obs, int(o.area), o.index, o.playerIndex)
                if card and hasattr(card, "tools") and o.toolIndex < len(card.tools):
                    add_context_card(sv, context, card.tools[o.toolIndex])
            elif o_type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                card = get_card_helper(obs, int(o.area), o.index, o.playerIndex)
                if card and hasattr(card, "energyCards") and o.energyIndex < len(card.energyCards):
                    add_context_card(sv, context, card.energyCards[o.energyIndex])
            elif o_type == OptionType.SKILL:
                add_context_card_id(sv, context, o.cardId)
    return sv
