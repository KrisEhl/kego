import os
from abc import ABC, abstractmethod
from enum import IntEnum

from cg.api import Card, CardType, Observation, OptionType, Pokemon, all_card_data, to_observation_class


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    LOOKING = 12


def module_dir() -> str:
    module_file = globals().get("__file__")
    if module_file:
        return os.path.dirname(os.path.abspath(module_file))
    return "/kaggle_simulations/agent"


def get_int_value(v) -> int | None:
    """Helper to extract raw integer value from enum, tuple, string, or number."""
    if isinstance(v, int):
        return v
    if isinstance(v, tuple) and len(v) > 0:
        return get_int_value(v[0])
    if hasattr(v, "value"):
        return get_int_value(v.value)
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    return None


def get_card(obs: Observation, area_val: int, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    try:
        player_index = get_int_value(player_index)
        if player_index is None or player_index < 0 or player_index >= len(obs.current.players):
            return None
        ps = obs.current.players[player_index]
        area_val = get_int_value(area_val)
        index = get_int_value(index)
        if index is None or index < 0:
            return None

        if area_val == AreaType.DECK:
            return obs.select.deck[index] if obs.select and obs.select.deck and index < len(obs.select.deck) else None
        elif area_val == AreaType.HAND:
            return ps.hand[index] if ps.hand and index < len(ps.hand) else None
        elif area_val == AreaType.DISCARD:
            return ps.discard[index] if ps.discard and index < len(ps.discard) else None
        elif area_val == AreaType.ACTIVE:
            return ps.active[index] if ps.active and index < len(ps.active) else None
        elif area_val == AreaType.BENCH:
            return ps.bench[index] if ps.bench and index < len(ps.bench) else None
        elif area_val == AreaType.PRIZE:
            return ps.prize[index] if ps.prize and index < len(ps.prize) else None
        elif area_val == AreaType.STADIUM:
            return obs.current.stadium[index] if obs.current.stadium and index < len(obs.current.stadium) else None
        elif area_val == AreaType.LOOKING:
            return obs.current.looking[index] if obs.current.looking and index < len(obs.current.looking) else None
    except Exception:
        pass
    return None


class BaseAgent(ABC):
    @abstractmethod
    def act(self, obs_dict: dict) -> list[int]:
        """Determine the action list to return based on the observation dict."""
        pass

    @abstractmethod
    def get_deck(self) -> list[int]:
        """Return the 60 card IDs in the agent's deck."""
        pass

    def _load_deck(self, deck: list[int] | str) -> list[int]:
        if isinstance(deck, list):
            return deck

        base_dir = module_dir()
        parent_dir = os.path.dirname(base_dir)
        file_paths = [
            deck,
            os.path.join(base_dir, deck),
            os.path.join(parent_dir, deck),
            os.path.join(parent_dir, "decks", deck),
            f"/kaggle_simulations/agent/{deck}",
            f"competitions/pokemon-tcg-ai-battle/{deck}",
            f"competitions/pokemon-tcg-ai-battle/decks/{deck}",
            "deck.csv",
            os.path.join(base_dir, "deck.csv"),
            "/kaggle_simulations/agent/deck.csv",
        ]
        checked = []
        for path in file_paths:
            # Resolve relative paths relative to working directory or base path
            full_path = os.path.abspath(path)
            checked.append(full_path)
            if os.path.exists(path):
                try:
                    with open(path) as file:
                        csv = file.read().splitlines()
                    deck_list = []
                    for line in csv:
                        if line.strip() and not line.strip().startswith("#"):
                            try:
                                deck_list.append(int(line.strip()))
                            except ValueError:
                                continue
                        if len(deck_list) == 60:
                            return deck_list
                    if len(deck_list) >= 60:
                        return deck_list[:60]
                except Exception as e:
                    raise ValueError(f"Failed to read deck file at {path}: {e}") from e

        raise FileNotFoundError(
            f"Deck file '{deck}' not found in any of the checked paths:\n" + "\n".join(f"  - {p}" for p in checked)
        )


class RuleScoringAgent(BaseAgent):
    def __init__(self, deck: list[int] | str):
        self.deck = self._load_deck(deck)

    def get_deck(self) -> list[int]:
        return self.deck

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self.get_deck()

        sel = obs.select
        opts = sel.option
        sel_type_val = get_int_value(sel.type)
        sel_context_val = get_int_value(sel.context)

        # 1. Yes/No questions (e.g. coin flips or option confirmations)
        # YES_NO select type is 9
        if sel_type_val == 9:
            for i, o in enumerate(opts):
                # YES option type is 1
                if get_int_value(o.type) == 1:
                    return [i]
            return [0] if len(opts) > 0 else []

        # 2. Setup Active Pokémon
        # SETUP_ACTIVE_POKEMON select context is 1
        if sel_context_val == 1:
            return [0] if len(opts) > 0 else []

        # 3. Setup Bench Pokémon
        # SETUP_BENCH_POKEMON select context is 2
        if sel_context_val == 2:
            return list(range(min(sel.maxCount, len(opts))))

        # 4. Score all options
        scores = []
        for idx, o in enumerate(opts):
            try:
                score = self.score_option(o, sel_type_val, sel_context_val, obs, idx)
            except Exception:
                score = 0.0
            scores.append(score)

        output = []
        if len(scores) >= 1:
            # Sort in descending order of score
            sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            for i in range(min(sel.maxCount, len(sorted_scores))):
                # Select option if it is positive, or required by minCount, or not an optional benching step
                # SETUP_BENCH_POKEMON context is 2, TO_BENCH context is 5
                if sorted_scores[i][1] >= 0 or sel.minCount > i or (sel_context_val != 2 and sel_context_val != 5):
                    output.append(sorted_scores[i][0])

        return output

    def score_option(self, option, select_type: int, select_context: int, obs: Observation, option_index: int) -> float:
        opt_type_val = get_int_value(option.type)

        if opt_type_val == OptionType.NUMBER:
            return self.score_number(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.YES or opt_type_val == OptionType.NO:
            return self.score_yes_no(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.CARD:
            return self.score_card(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.TOOL_CARD:
            return self.score_tool_card(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.ENERGY_CARD:
            return self.score_energy_card(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.ENERGY:
            return self.score_energy(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.PLAY:
            return self.score_play(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.ATTACH:
            return self.score_attach(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.EVOLVE:
            return self.score_evolve(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.ABILITY:
            return self.score_ability(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.DISCARD:
            return self.score_discard(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.RETREAT:
            return self.score_retreat(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.ATTACK:
            return self.score_attack(option, select_context, obs, option_index)
        elif opt_type_val == OptionType.END:
            return self.score_end(option, select_context, obs, option_index)

        return 0.0

    # Handler hooks with baseline scores (subclasses will override these)
    def score_number(self, option, context: int, obs: Observation, option_index: int) -> float:
        return float(option.number) if option.number is not None else 0.0

    def score_yes_no(self, option, context: int, obs: Observation, option_index: int) -> float:
        opt_type_val = get_int_value(option.type)
        return 1.0 if opt_type_val == 1 else 0.0

    def score_card(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_tool_card(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_energy_card(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_energy(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_play(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_attach(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_evolve(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_ability(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_discard(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_retreat(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_attack(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0

    def score_end(self, option, context: int, obs: Observation, option_index: int) -> float:
        return 0.0


class MockCardData:
    def __init__(self, card_id: int) -> None:
        self.cardId = card_id
        self.megaEx = card_id in (723, 678)
        self.ex = card_id in (121, 140, 184, 1071, 336, 678)
        self.name = ""
        self.stage2 = card_id in (121, 723)
        self.stage1 = card_id in (120, 722, 674)
        self.cardType = CardType.POKEMON if card_id < 1000 else CardType.ITEM


card_table: dict[int, MockCardData] = {}
try:
    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
except Exception:
    pass


def get_card_data(card_id: int) -> MockCardData:
    if card_id in card_table:
        return card_table[card_id]
    return MockCardData(card_id)
