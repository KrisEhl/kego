from collections import defaultdict

try:
    from agents.base import RuleScoringAgent, get_card, get_int_value
except ImportError:
    from base_agent import RuleScoringAgent, get_card, get_int_value
from cg.api import CardType, Observation, Pokemon, all_card_data, to_observation_class


class MockCardData:
    def __init__(self, card_id):
        self.cardId = card_id
        self.megaEx = card_id in (723,)
        self.ex = card_id in (121, 140, 184, 1071, 336)
        self.name = ""
        self.stage2 = card_id in (121, 723)
        self.stage1 = card_id in (120, 722)
        self.cardType = CardType.POKEMON if card_id < 1000 else CardType.ITEM


card_table = {}
try:
    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
except Exception:
    pass


def get_card_data(card_id: int) -> MockCardData:
    if card_id in card_table:
        return card_table[card_id]
    return MockCardData(card_id)


class AttackPlan:
    def __init__(self):
        self.attack = 0
        self.counter = []


class DragapultAgent(RuleScoringAgent):
    def __init__(self, deck="dragapult.csv"):
        super().__init__(deck)

        # Card IDs used in deck heuristics
        self.Dreepy = 119
        self.Drakloak = 120
        self.Dragapult_ex = 121
        self.Fezandipiti_ex = 140
        self.Latias_ex = 184
        self.Budew = 235
        self.Meowth_ex = 1071
        self.Rare_Candy = 1079
        self.Unfair_Stamp = 1080
        self.Buddy_Buddy_Poffin = 1086
        self.Night_Stretcher = 1097
        self.Crushing_Hammer = 1120
        self.Ultra_Ball = 1121
        self.Poke_Pad = 1152
        self.Lucky_Helmet = 1156
        self.Boss_Orders = 1182
        self.Crispin = 1198
        self.Brock_Scouting = 1210
        self.Lillie_Determination = 1227
        self.Team_Rocket_Watchtower = 1256
        self.Basic_Fire_Energy = 2
        self.Basic_Psychic_Energy = 5

        self.UNNECESSARY = -10000000.0

        # State tracking
        self.can_switch = False
        self.can_attack = False
        self.can_main_attack = False
        self.use_support = 0
        self.bench_attacker = False
        self.pre_turn_log = []
        self.current_turn_log = []
        self.prize = []
        self.card_counts = defaultdict(int)
        self.serial_set = set()
        self.plan_a = AttackPlan()
        self.plan_b = AttackPlan()

    def no_damage_dex(self, card_id: int) -> bool:
        # Drednaw (158), Milotic ex (207), Sylveon (330), Crustle (345)
        return card_id in (158, 207, 330, 345)

    def no_damage_counter(self, pokemon: Pokemon) -> bool:
        # Poltchageist (28), Empoleon ex (199), Skeledirge (203), Milotic ex (207), Misty's Magikarp (362), Antique Cover Fossil (1136)
        if pokemon.id in (28, 199, 203, 207, 362, 1136):
            return True
        for card in pokemon.energyCards:
            # Mist Energy (11), Rock Fighting Energy (20)
            if card.id in (11, 20):
                return True
        return False

    def prize_count(self, pokemon: Pokemon, is_attack_damage: bool) -> int:
        data = get_card_data(pokemon.id)
        count = 3 if data.megaEx else 2 if data.ex else 1
        if is_attack_damage:
            for card in pokemon.energyCards:
                if card.id == 12:  # Legacy Energy
                    count -= 1
            for card in pokemon.tools:
                if card.id == 1172 and "Lillie" in getattr(data, "name", ""):  # Lillie’s Pearl
                    count -= 1
        return max(0, count)

    def pokemon_score(self, pokemon: Pokemon, is_attack_damage: bool) -> int:
        data = get_card_data(pokemon.id)
        score = self.prize_count(pokemon, is_attack_damage) * 1000
        score += len(pokemon.energies) * 150
        score += len(pokemon.tools) * 100
        if data.stage2:
            score += 250
        elif data.stage1:
            score += 130

        # Noctowl (173), Fan Rotom (174), Archaludon ex (190), Meowth ex (1071)
        if pokemon.id in (173, 174, 190, 1071):
            score -= 200
        if pokemon.id == 112 and len(pokemon.energies) >= 1:  # Munkidori
            score += 300
        score += pokemon.hp
        return score

    def add_card_count(self, card, my_index: int):
        if card is None:
            return
        if isinstance(card, Pokemon) or getattr(card, "playerIndex", -1) == my_index:
            if card.serial not in self.serial_set:
                self.card_counts[card.id] -= 1
                self.serial_set.add(card.serial)
        if isinstance(card, Pokemon):
            for c in card.energyCards:
                self.add_card_count(c, my_index)
            for c in card.tools:
                self.add_card_count(c, my_index)
            for c in card.preEvolution:
                self.add_card_count(c, my_index)

    def set_card_counts(self, obs: Observation, my_index: int):
        self.card_counts.clear()
        self.serial_set.clear()
        for id in self.deck:
            self.card_counts[id] += 1

        state = obs.current
        my_state = state.players[my_index]
        if my_state.hand:
            for card in my_state.hand:
                self.add_card_count(card, my_index)
        for card in my_state.discard:
            self.add_card_count(card, my_index)
        for card in my_state.bench:
            self.add_card_count(card, my_index)
        for card in my_state.active:
            self.add_card_count(card, my_index)
        for card in state.stadium:
            self.add_card_count(card, my_index)
        if state.looking is not None:
            for card in state.looking:
                self.add_card_count(card, my_index)
        if obs.select and obs.select.effect:
            self.add_card_count(obs.select.effect, my_index)

    def main_option_proc(self, obs: Observation, damage: int):
        state = obs.current
        select = obs.select
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]

        self.can_switch = False
        self.can_attack = False
        self.can_main_attack = False
        for o in select.option:
            o_type_val = get_int_value(o.type)
            if o_type_val == 12:  # RETREAT
                self.can_switch = True
            elif o_type_val == 13:  # ATTACK
                self.can_attack = True
                if o.attackId == 154:  # Phantom Dive
                    self.can_main_attack = True

        self.plan_a.attack = -1
        self.plan_b.attack = -1
        if not self.can_main_attack and not (self.bench_attacker and self.can_switch):
            return

        cards = []
        if op_state.active and op_state.active[0] is not None:
            cards.append(op_state.active[0])
        for pokemon in op_state.bench:
            cards.append(pokemon)

        if not cards:
            return

        counter_indices = []
        ci = [0]
        remain_damage = 60
        while ci:
            index = ci[-1]
            if index >= len(cards):
                ci.pop()
                if ci:
                    ci[-1] += 1
                continue
            hp = cards[index].hp
            if remain_damage >= hp:
                counter_indices.append(ci.copy())
                if index < len(cards) - 1:
                    remain_damage -= hp
                    ci.append(index + 1)
                    continue
            if index == len(cards) - 1:
                ci.pop()
                if ci:
                    remain_damage += cards[ci[-1]].hp
            if ci:
                ci[-1] += 1
        counter_indices.append([])

        remain_prize = len(my_state.prize)
        plan_score = 0
        for i, pokemon in enumerate(cards):
            base_prize_count = 0
            base_score = self.pokemon_score(pokemon, True)
            active_damage = 0 if self.no_damage_dex(pokemon.id) else damage
            if pokemon.hp <= active_damage:
                base_prize_count += self.prize_count(pokemon, True)
            else:
                base_score *= active_damage / pokemon.hp

            ci = []
            max_score = base_score
            if remain_prize <= base_prize_count:
                max_score = 50000
            else:
                for indices in counter_indices:
                    if i in indices:
                        continue
                    prize = base_prize_count
                    score = base_score
                    for idx in indices:
                        if idx < len(cards):
                            prize += self.prize_count(cards[idx], False)
                            score += self.pokemon_score(cards[idx], False)
                    if remain_prize <= prize:
                        score = 50000
                    else:
                        if prize >= 2:
                            if remain_prize <= 4:
                                score -= 1200
                        elif prize == 1:
                            score -= 300
                        else:
                            score += 1200
                    if max_score < score:
                        max_score = score
                        ci = indices
            if plan_score < max_score:
                plan_score = max_score
                self.plan_a.attack = i
                self.plan_a.counter = ci
            if i == 0:
                self.plan_b.attack = self.plan_a.attack
                self.plan_b.counter = self.plan_a.counter

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self.get_deck()

        state = obs.current
        select = obs.select
        my_index = state.yourIndex

        if state.turn == 0:
            self.prize.clear()
            self.pre_turn_log.clear()
            self.current_turn_log.clear()
        else:
            for log in obs.logs:
                self.current_turn_log.append(log)
                if get_int_value(log.type) == 3:  # LogType.TURN_END is 3
                    self.pre_turn_log = self.current_turn_log
                    self.current_turn_log = []

        # Run state tracking logic
        if select.deck is not None:
            self.set_card_counts(obs, my_index)
            for card in select.deck:
                self.card_counts[card.id] -= 1
            self.prize.clear()
            for id in self.card_counts:
                for _ in range(self.card_counts[id]):
                    self.prize.append(id)

        self.set_card_counts(obs, my_index)
        for id in self.prize:
            self.card_counts[id] -= 1

        return super().act(obs_dict)

    def score_option(self, option, select_type: int, select_context: int, obs: Observation, option_index: int) -> float:
        state = obs.current
        select = obs.select
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]

        pre_ko = False
        no_item = False
        for log in self.pre_turn_log:
            if get_int_value(log.type) == 15:  # LogType.ATTACK is 15
                if log.attackId == 323:  # Itchy Pollen
                    no_item = True
            elif get_int_value(log.type) == 6:  # LogType.MOVE_CARD is 6
                if (
                    log.playerIndex == my_index
                    and (get_int_value(log.fromArea) in (4, 5))  # BENCH or ACTIVE
                    and get_int_value(log.toArea) == 3
                ):  # DISCARD
                    pre_ko = True

        prize_diff = len(my_state.prize) - len(op_state.prize)

        field_counts = defaultdict(int)
        hand_counts = defaultdict(int)
        discard_counts = defaultdict(int)

        active_id = 0
        self.bench_attacker = False
        can_evolve_dreepy = False
        evolve_dreepy_count = 0
        can_evolve_drakloak = False
        damage = 200
        for card in my_state.active:
            if card is None:
                continue
            active_id = card.id
            field_counts[card.id] += 1
            if not card.appearThisTurn:
                if card.id == self.Dreepy:
                    can_evolve_dreepy = True
                    evolve_dreepy_count += 1
                elif card.id == self.Drakloak:
                    can_evolve_drakloak = True
        for card in my_state.bench:
            field_counts[card.id] += 1
            if not card.appearThisTurn:
                if card.id == self.Dreepy:
                    can_evolve_dreepy = True
                    evolve_dreepy_count += 1
                elif card.id == self.Drakloak:
                    can_evolve_drakloak = True
            if card.id == self.Dragapult_ex and len(card.energies) >= 2:
                self.bench_attacker = True

        main_pokemon_count = field_counts[self.Dreepy] + field_counts[self.Drakloak] + field_counts[self.Dragapult_ex]
        no_more_dex = field_counts[self.Dragapult_ex] * 2 >= len(op_state.prize)

        stadium_id = 0
        for card in state.stadium:
            stadium_id = card.id

        support_count = 0
        for card in my_state.hand:
            hand_counts[card.id] += 1
            if get_card_data(card.id).cardType == CardType.SUPPORTER and card.id != self.Boss_Orders:
                support_count += 1

        for card in my_state.discard:
            discard_counts[card.id] += 1

        def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
            energy_count = len(pokemon.energies)
            if get_card_data(attach_id).cardType == CardType.TOOL:
                score = 60000
                if active:
                    score += 1000
                return score

            if pokemon.id == self.Budew:
                return -1
            elif pokemon.id in (self.Meowth_ex, self.Fezandipiti_ex, self.Latias_ex):
                if active and not self.can_switch and not my_state.asleep and not my_state.paralyzed:
                    if self.bench_attacker or field_counts[self.Budew] >= 1:
                        return 22000
                    else:
                        return 18000
                else:
                    return -1
            if active and self.can_main_attack:
                return -1
            score = 20000
            if energy_count >= 2:
                if active and not self.can_switch and not my_state.asleep and not my_state.paralyzed:
                    score += 200
                else:
                    return -1
            elif energy_count == 1:
                if pokemon.energyCards and attach_id == pokemon.energyCards[0].id:
                    return -1
                if pokemon.id == self.Dragapult_ex:
                    score += 250
                elif pokemon.id == self.Dreepy:
                    score -= 150
                else:
                    score -= 200
                if active:
                    score += 200
            else:
                if active:
                    if self.bench_attacker:
                        score += 400
                else:
                    if pokemon.id == self.Dragapult_ex:
                        score += 150
                    elif pokemon.id == self.Dreepy:
                        score += 100
                    else:
                        score += 50
                    if self.bench_attacker:
                        score -= 200
            if no_more_dex and (pokemon.id in (self.Dreepy, self.Drakloak)):
                score -= 500
            return score

        def hand_score(id: int, ignore_count: bool):
            score = 0
            if id == self.Dreepy:
                if main_pokemon_count >= 3:
                    score = 1000
                else:
                    score = 18000
            elif id == self.Drakloak:
                if can_evolve_dreepy:
                    score = 20000
                else:
                    score = 3000
            elif id == self.Dragapult_ex:
                if no_more_dex:
                    score = int(self.UNNECESSARY)
                elif can_evolve_dreepy and hand_counts[self.Rare_Candy] >= 1 and not no_item:
                    score = 40000
                elif can_evolve_drakloak:
                    if field_counts[id] == 0:
                        score = 30000
                    elif field_counts[id] == 1:
                        score = 10000
                    else:
                        score = 50
                else:
                    if field_counts[id] >= 2:
                        score = 50
                    else:
                        score = 2000
            elif id == self.Fezandipiti_ex:
                if pre_ko:
                    score = 50000
                elif prize_diff <= -2:
                    score = 5
                elif len(op_state.prize) == 1:
                    score = int(self.UNNECESSARY)
            elif id == self.Latias_ex:
                if active_id in (self.Fezandipiti_ex, self.Meowth_ex, self.Dreepy):
                    if field_counts[self.Drakloak] + field_counts[self.Dragapult_ex] == 0:
                        score = 28000
                    else:
                        score = 15000
                else:
                    score = 10
            elif id == self.Budew:
                if field_counts[id] + field_counts[self.Drakloak] + field_counts[self.Dragapult_ex] >= 1:
                    score = int(self.UNNECESSARY)
                elif state.turn >= 2:
                    score = 30000
            elif id == self.Meowth_ex:
                if support_count > hand_counts[self.Boss_Orders] or stadium_id == self.Team_Rocket_Watchtower:
                    score = 5
                elif state.supporterPlayed:
                    score = 40
                else:
                    score = 35000
            elif id == self.Rare_Candy:
                if no_more_dex:
                    score = int(self.UNNECESSARY)
                elif can_evolve_dreepy and hand_counts[self.Dragapult_ex] >= 1:
                    score = 40000
            elif id == self.Unfair_Stamp:
                if pre_ko:
                    score = 80000
                elif len(op_state.prize) == 1:
                    score = int(self.UNNECESSARY)
                else:
                    score = 80
            elif id == self.Buddy_Buddy_Poffin:
                count = self.card_counts[self.Dreepy]
                if count == 0:
                    score = int(self.UNNECESSARY)
                else:
                    if state.turn <= 2 and field_counts[self.Budew] == 0 and self.card_counts[self.Budew] >= 1:
                        count += 1
                    if count >= 2:
                        score = 35000
            elif id == self.Night_Stretcher:
                for disc_id in discard_counts:
                    if discard_counts[disc_id] >= 1:
                        card_type = get_card_data(disc_id).cardType
                        if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                            score = max(score, hand_score(disc_id, ignore_count))
            elif id == self.Crushing_Hammer:
                score = 20
            elif id == self.Ultra_Ball:
                if main_pokemon_count <= 2 or field_counts[self.Dreepy] >= 1:
                    score = 70
                else:
                    score = 5
            elif id == self.Poke_Pad:
                score = max(hand_score(self.Dreepy, ignore_count), hand_score(self.Drakloak, ignore_count))
            elif id == self.Lucky_Helmet:
                score = 15
            elif id == self.Boss_Orders:
                if self.plan_a.attack > 0:
                    score = 60000
            elif id == self.Crispin:
                if not ignore_count or support_count == 0:
                    if (
                        self.card_counts[self.Basic_Fire_Energy] == 0
                        or self.card_counts[self.Basic_Psychic_Energy] == 0
                    ):
                        score = 10
                    if not self.can_main_attack and not self.bench_attacker and field_counts[self.Dragapult_ex] >= 1:
                        score = 55000
                    else:
                        score = 25000
            elif id == self.Brock_Scouting:
                if not ignore_count or support_count == 0:
                    if state.turn == 2 and field_counts[self.Budew] + field_counts[self.Latias_ex] == 0:
                        score = 50000
                    else:
                        score = 30000
            elif id == self.Lillie_Determination:
                if not ignore_count or support_count == 0:
                    score = 45000
            elif id == self.Team_Rocket_Watchtower:
                if stadium_id != 0 and stadium_id != self.Team_Rocket_Watchtower:
                    score = 4000
            elif id in (self.Basic_Fire_Energy, self.Basic_Psychic_Energy):
                if self.can_main_attack and (
                    len(op_state.prize) <= 2 or (self.bench_attacker and len(op_state.prize) <= 4)
                ):
                    score = int(self.UNNECESSARY)
                else:
                    max_score = -10000
                    for pokemon in my_state.active:
                        if pokemon is not None:
                            max_score = max(max_score, attach_score(id, pokemon, True))
                    for pokemon in my_state.bench:
                        max_score = max(max_score, attach_score(id, pokemon, False))
                    score = max_score - 5000
                    if self.can_main_attack or self.bench_attacker:
                        score /= 10

            if not ignore_count and hand_counts[id] > 0:
                if id == self.Drakloak and hand_counts[id] < evolve_dreepy_count:
                    score -= 10
                elif id == self.Dreepy:
                    score -= 100
                else:
                    score -= 100000
            return score

        if select_type == 0:  # MAIN
            self.main_option_proc(obs, damage)
            self.use_support = 0
            if not state.supporterPlayed:
                support_score = 0
                for o in select.option:
                    if get_int_value(o.type) == 7:  # PLAY
                        card = get_card(obs, 2, o.index, state.yourIndex)
                        if get_card_data(card.id).cardType == CardType.SUPPORTER:
                            s = hand_score(card.id, True)
                            if support_score < s:
                                support_score = s
                                self.use_support = card.id

        hand_scores = []
        negative_hand_count = 0
        for card in my_state.hand:
            s = hand_score(card.id, False)
            hand_scores.append(s)
            if s < 0:
                negative_hand_count += 1
            hand_counts[card.id] += 1
            if get_card_data(card.id).cardType == CardType.SUPPORTER and card.id != self.Boss_Orders:
                support_count += 1

        no_draw = my_state.deckCount <= 8
        do_switch = not self.can_main_attack and (
            self.bench_attacker or (active_id != self.Budew and field_counts[self.Budew] >= 1 and state.turn >= 2)
        )
        effect_card_id = 0 if select.effect is None else select.effect.id
        context_card_id = 0 if select.contextCard is None else select.contextCard.id

        opt_type_val = get_int_value(option.type)
        score = 0.0

        if opt_type_val == 0:  # NUMBER
            score = float(option.number) if option.number is not None else 0.0
        elif opt_type_val == 1 or opt_type_val == 2:  # YES / NO
            if select_context == 41:  # IS_FIRST is 41
                score = -1.0
            else:
                score = 1.0
        elif opt_type_val == 3:  # CARD
            card = get_card(obs, get_int_value(option.area), option.index, option.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                hp = card.hp if isinstance(card, Pokemon) else 0
                # SWITCH (3), TO_ACTIVE (4), SETUP_ACTIVE_POKEMON (1)
                if select_context in (3, 4, 1):
                    if option.playerIndex == my_index:
                        if card.id == self.Dreepy:
                            score += 10000
                        elif card.id == self.Drakloak:
                            if energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == self.Dragapult_ex:
                            score += 50000
                        elif card.id == self.Budew:
                            if select_context != 3:
                                score += 100000
                            elif not self.bench_attacker:
                                score += 30000
                        elif card.id == self.Fezandipiti_ex:
                            score -= 1000
                        elif card.id == self.Meowth_ex:
                            score -= 2000
                    else:
                        if self.plan_a.attack == option.index + 1:
                            score += 100000
                    score += energy_count * 1000
                    score += hp
                # SETUP_BENCH_POKEMON (2)
                elif select_context == 2:
                    if my_index == state.firstPlayer or card.id != self.Dreepy:
                        score = -1.0
                # TO_BENCH (5), TO_HAND (7)
                elif select_context in (5, 7):
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == self.Crispin:
                        score = 100000.0 - hand_score(card.id, True)
                # DISCARD (8)
                elif select_context == 8:
                    hand_counts[card.id] -= 1
                    if get_card_data(card.id).cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                # DAMAGE_COUNTER (13), DAMAGE_COUNTER_ANY (14)
                elif select_context in (13, 14):
                    if hp > 0:
                        score = 100000.0 - 10.0 * hp + self.pokemon_score(card, False)
                        if select_context == 13:
                            if 210 <= hp <= 230:
                                score += 20000.0 + hp * 20.0
                                if get_int_value(option.area) == 4:  # ACTIVE is 4
                                    score += 10000.0
                            elif 40 <= hp <= 90:
                                score += 10000.0 + hp * 20.0
                            elif hp <= 30:
                                score += -10000.0 + hp * 20.0
                            if card.id in (133, 351):
                                score += 30000.0
                        else:
                            idx = option.index + 1
                            if idx in self.plan_b.counter:
                                score += 100000.0
                            else:
                                remain_damage = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain_damage:
                                    score += 30000.0
                                elif 20 <= hp <= 60 + remain_damage:
                                    score += 10000.0
                                elif hp == 10:
                                    score -= 100000.0
                            if self.no_damage_counter(card):
                                score = -1.0
                # ATTACH_FROM (21)
                elif select_context == 21:
                    score = attach_score(context_card_id, card, get_int_value(option.area) == 4)
                    if card.id == self.Dragapult_ex:
                        score += 200
        elif opt_type_val in (4, 5, 6):  # TOOL_CARD, ENERGY_CARD, ENERGY
            # Discard energy (Retreat or Crushing Hammer)
            if option.playerIndex != my_index:
                score = 20.0 if get_int_value(option.area) == 5 else 10.0
                card = get_card(obs, get_int_value(option.area), option.index, option.playerIndex)
                if card is not None and get_card_data(card.id).cardType == CardType.SPECIAL_ENERGY:
                    score += 1.0
        elif opt_type_val == 7:  # PLAY
            card = get_card(obs, 2, option.index, my_index)
            card_score = hand_scores[option.index]
            if card.id == self.Dreepy:
                score = 51000.0
            elif card.id == self.Fezandipiti_ex:
                score = 53000.0 if card_score > 0 else -1.0
            elif card.id == self.Latias_ex:
                score = 51000.0 if active_id not in (self.Drakloak, self.Dragapult_ex) else -1.0
            elif card.id == self.Budew:
                score = 52000.0 if field_counts[self.Budew] == 0 and field_counts[self.Dragapult_ex] == 0 else -1.0
            elif card.id == self.Meowth_ex:
                if state.supporterPlayed or stadium_id == self.Team_Rocket_Watchtower:
                    score = -1.0
                elif support_count == 0:
                    score = 50000.0
                elif support_count == hand_counts[self.Boss_Orders] and self.plan_a.attack > 0:
                    score = 50000.0
                else:
                    score = -1.0
            elif card.id == self.Rare_Candy:
                score = -1.0 if no_more_dex else 75000.0
            elif card.id == self.Unfair_Stamp:
                score = 15000.0
            elif card.id == self.Night_Stretcher:
                score = 42000.0 if card_score >= 18000.0 else -1.0
            elif card.id == self.Crushing_Hammer:
                score = 40000.0
            elif card.id == self.Boss_Orders:
                score = 35000.0 if card.id == self.use_support else -1.0
            elif card.id == self.Lillie_Determination:
                score = 14000.0 if card.id == self.use_support else -1.0
            elif card.id == self.Team_Rocket_Watchtower:
                score = 80000.0 if stadium_id > 0 or state.turn == 1 else -1.0
            elif no_draw:
                score = -1.0
            elif card.id == self.Buddy_Buddy_Poffin:
                score = 46000.0 if self.card_counts[self.Dreepy] > 0 else -1.0
            elif card.id == self.Ultra_Ball:
                score = 44000.0 if negative_hand_count >= 2 else -1.0
            elif card.id == self.Poke_Pad:
                score = 45000.0 if self.card_counts[self.Dreepy] + self.card_counts[self.Drakloak] > 0 else -1.0
            elif card.id in (self.Crispin, self.Brock_Scouting):
                score = 35000.0 if card.id == self.use_support else -1.0
        elif opt_type_val == 8:  # ATTACH
            card = get_card(obs, get_int_value(option.area), option.index, my_index)
            pokemon = get_card(obs, get_int_value(option.inPlayArea), option.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, get_int_value(option.inPlayArea) == 4)
        elif opt_type_val == 9:  # EVOLVE
            pokemon = get_card(obs, get_int_value(option.inPlayArea), option.inPlayIndex, my_index)
            score = len(pokemon.energies)
            if pokemon.id == self.Dreepy:
                score += 30000.0
            elif field_counts[self.Dragapult_ex] >= 2 or (
                field_counts[self.Dragapult_ex] == 1 and len(op_state.prize) <= 2
            ):
                score = -1.0
            else:
                score += 70000.0
        elif opt_type_val == 10:  # ABILITY
            card = get_card(obs, get_int_value(option.area), option.index, my_index)
            if no_draw:
                score = -1.0
            elif card.id == 1267:  # Lumiose City (1267)
                score = 1.0
            else:
                score = 40000.0
        elif opt_type_val == 12:  # RETREAT
            score = 10000.0 if do_switch else -1.0
        elif opt_type_val == 13:  # ATTACK
            score = float(option.attackId) if option.attackId is not None else 0.0
        elif opt_type_val == 14:  # END
            score = 0.0

        return score


# Define submission agent wrapper for Kaggle and pipeline execution
_agent_instance = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = DragapultAgent()
    return _agent_instance.act(obs_dict)
