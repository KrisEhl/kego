from collections import defaultdict

try:
    from agents.base import RuleScoringAgent, get_card, get_int_value
except ImportError:
    from base_agent import RuleScoringAgent, get_card, get_int_value
from cg.api import Observation, Pokemon


class AbomasnowAgent(RuleScoringAgent):
    def __init__(self, deck="abomasnow.csv"):
        super().__init__(deck)
        self.Kyogre = 721
        self.Snover = 722
        self.Mega_Abomasnow_ex = 723
        self.Ultra_Ball = 1121
        self.Precious_Trolley = 1126
        self.Carmine = 1192
        self.Lillie_Determination = 1227
        self.Surfing_Beach = 1262
        self.Basic_Water_Energy = 3

    def score_option(self, option, select_type: int, select_context: int, obs: Observation, option_index: int) -> float:
        state = obs.current
        if state is None or state.players is None or len(state.players) <= 1:
            return 0.0
        my_index = state.yourIndex
        my_state = state.players[my_index]
        if my_state is None:
            return 0.0

        # Calculate counts
        field_counts = defaultdict(int)
        hand_counts = defaultdict(int)
        discard_counts = defaultdict(int)

        bench_attacker_index0 = -1  # Mega Abomasnow ex
        bench_attacker_index1 = -1  # Kyogre
        if my_state.bench is not None:
            for i, card in enumerate(my_state.bench):
                if card is not None:
                    field_counts[card.id] += 1
                    if card.id == self.Mega_Abomasnow_ex and len(card.energies) >= 2:
                        bench_attacker_index0 = i
                    elif card.id == self.Kyogre and len(card.energies) >= 1:
                        bench_attacker_index1 = i

        if my_state.hand is not None:
            for card in my_state.hand:
                if card is not None:
                    hand_counts[card.id] += 1

        if my_state.discard is not None:
            for card in my_state.discard:
                if card is not None:
                    discard_counts[card.id] += 1

        op_active_hp = 0
        op_state = state.players[1 - my_index]
        if op_state is not None and op_state.active is not None:
            for card in op_state.active:
                if card is not None:
                    op_active_hp = card.hp

        prefer_ky = op_active_hp <= 20 * discard_counts[self.Basic_Water_Energy]
        switch_index = -1
        if my_state.active is not None:
            for card in my_state.active:
                if card is None:
                    continue
                field_counts[card.id] += 1
                if card.id == self.Mega_Abomasnow_ex and len(card.energies) >= 2:
                    if prefer_ky and bench_attacker_index1 >= 0:
                        switch_index = bench_attacker_index1
                elif card.id == self.Kyogre and len(card.energies) >= 1:
                    if not prefer_ky and bench_attacker_index0 >= 0:
                        switch_index = bench_attacker_index0
                elif bench_attacker_index0 >= 0:
                    switch_index = bench_attacker_index0

        # Now score
        opt_type_val = get_int_value(option.type)
        score = 0.0

        if opt_type_val == 0:  # NUMBER
            score = float(option.number) if option.number is not None else 0.0
        elif opt_type_val == 1:  # YES
            score = 1.0
        elif opt_type_val == 3:  # CARD
            card_obj = get_card(obs, get_int_value(option.area), option.index, option.playerIndex)
            if card_obj is not None:
                energy_count = len(card_obj.energies) if isinstance(card_obj, Pokemon) else 0
                # SWITCH, TO_ACTIVE, SETUP_ACTIVE_POKEMON
                if select_context in (3, 4, 1):
                    score += energy_count * 2
                    if option.index == switch_index:
                        score += 100
                    if card_obj.id == self.Mega_Abomasnow_ex:
                        score += 20
                    elif card_obj.id == self.Kyogre:
                        score += 10
                # TO_BENCH, TO_HAND
                elif select_context in (5, 7):
                    if card_obj.id == self.Snover:
                        if field_counts[card_obj.id] >= 1:
                            score += 5
                        elif field_counts[self.Mega_Abomasnow_ex] >= 1:
                            score += 15
                        else:
                            score += 30
                    elif card_obj.id == self.Mega_Abomasnow_ex:
                        if field_counts[self.Snover] >= 1 and field_counts[card_obj.id] + hand_counts[card_obj.id] == 0:
                            score += 100
                        else:
                            score += 10
                    elif card_obj.id == self.Kyogre:
                        if field_counts[card_obj.id] >= 1:
                            score += 1
                        else:
                            score += 20
                # DISCARD
                elif select_context == 8:
                    if card_obj.id == self.Basic_Water_Energy:
                        score += 100
                    elif card_obj.id == self.Mega_Abomasnow_ex:
                        score += 10
                    elif card_obj.id == self.Carmine:
                        if hand_counts[self.Lillie_Determination] >= 1:
                            score += 30
                    elif card_obj.id == self.Lillie_Determination:
                        score -= 20

                    if hand_counts[card_obj.id] >= 2:
                        score += 500
                    hand_counts[card_obj.id] -= 1
        elif opt_type_val == 7:  # PLAY
            card_obj = get_card(obs, 2, option.index, my_index)  # Hand area is 2
            if card_obj is None:
                score = -100.0
            else:
                score = 10000.0
                if card_obj.id == self.Ultra_Ball:
                    if hand_counts[self.Basic_Water_Energy] >= 3 or (
                        my_state.handCount >= 4
                        and (
                            field_counts[self.Mega_Abomasnow_ex] + hand_counts[self.Mega_Abomasnow_ex] == 0
                            or field_counts[self.Mega_Abomasnow_ex] + field_counts[self.Snover] == 0
                            or field_counts[self.Kyogre] == 0
                        )
                    ):
                        score = 4000.0
                    else:
                        score = -1.0
                elif card_obj.id == self.Carmine:
                    if field_counts[self.Snover] >= 1 and hand_counts[self.Mega_Abomasnow_ex] >= 1:
                        score = -1.0
                    else:
                        score = 3000.0
                elif card_obj.id == self.Lillie_Determination:
                    if (
                        field_counts[self.Snover] >= 1
                        and field_counts[self.Mega_Abomasnow_ex] == 0
                        and hand_counts[self.Mega_Abomasnow_ex] >= 1
                    ):
                        score = -1.0
                    else:
                        score = 3100.0
        elif opt_type_val == 8:  # ATTACH
            pokemon = get_card(obs, get_int_value(option.inPlayArea), option.inPlayIndex, my_index)
            if pokemon is None:
                score = -100.0
            else:
                score = 5000.0
                energy_count = len(pokemon.energies)
                # Bench area is 5
                if energy_count == 0 and get_int_value(option.inPlayArea) == 5:
                    score += 1
                if pokemon.id == self.Snover:
                    score += 1
                    if energy_count == 1:
                        score -= 100
                    elif energy_count >= 2:
                        score -= 400
                    if bench_attacker_index0 >= 0:
                        score -= 300
                elif pokemon.id == self.Mega_Abomasnow_ex:
                    score += 10
                    if energy_count == 1:
                        score += 30
                    elif energy_count >= 2:
                        score -= 300
                    if bench_attacker_index0 >= 0:
                        score -= 200
                elif pokemon.id == self.Kyogre:
                    score += 5
                    if len(pokemon.energies) >= 1:
                        score -= 200
                    if bench_attacker_index1 >= 0:
                        score -= 200
                # Active area is 4
                if get_int_value(option.inPlayArea) == 4:
                    if bench_attacker_index0 >= 0 and bench_attacker_index1 >= 0 and energy_count <= 2:
                        score += 200
        elif opt_type_val == 9:  # EVOLVE
            pokemon = get_card(obs, get_int_value(option.inPlayArea), option.inPlayIndex, my_index)
            if pokemon is None:
                score = -100.0
            else:
                score = 10000.0 + len(pokemon.energies)
        elif opt_type_val == 10:  # ABILITY
            card_obj = get_card(obs, get_int_value(option.area), option.index, my_index)
            score = -1.0
            if card_obj is not None:
                if card_obj.id == self.Surfing_Beach and switch_index >= 0:
                    score = 2000.0

        elif opt_type_val == 12:  # RETREAT
            if switch_index >= 0:
                score = 1500.0
            else:
                score = -1.0
        elif opt_type_val == 13:  # ATTACK
            score = 1000.0
            if option.attackId == 1042:  # Riptide
                score += discard_counts[self.Basic_Water_Energy] * 20 - 90
            elif option.attackId == 1046:  # Hammer-lanche
                if op_active_hp <= 200:
                    score -= 100
                else:
                    score += 100
        elif opt_type_val == 14:  # END
            score = 0.0

        return score


# Define submission agent wrapper for Kaggle and pipeline execution
_agent_instance = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AbomasnowAgent()
    return _agent_instance.act(obs_dict)
