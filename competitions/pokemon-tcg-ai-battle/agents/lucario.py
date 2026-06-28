import time
from collections import defaultdict

try:
    from agents.base import BaseAgent, get_int_value
    from agents.base import get_card as get_card_helper
except ImportError:
    from base_agent import BaseAgent, get_int_value
    from base_agent import get_card as get_card_helper
from cg.api import (
    Card,
    CardType,
    Observation,
    Pokemon,
    all_card_data,
    to_observation_class,
)

# Search API imports
_SEARCH_OK = False
try:
    from cg.api import search_begin, search_release, search_step

    _SEARCH_OK = True
except Exception:
    _SEARCH_OK = False


class MockCardData:
    def __init__(self, card_id):
        self.cardId = card_id
        self.megaEx = card_id in (723, 678)
        self.ex = card_id in (121, 140, 184, 1071, 336, 678)
        self.name = ""
        self.stage2 = card_id in (121, 723)
        self.stage1 = card_id in (120, 722, 674)
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


class C:
    MAKUHITA = 673
    HARIYAMA = 674
    LUNATONE = 675
    SOLROCK = 676
    RIOLU = 677
    MEGA_LUCARIO_EX = 678

    BASIC_FIGHTING_ENERGY = 6
    DUSK_BALL = 1102
    SWITCH = 1123
    PREMIUM_POWER_PRO = 1141
    FIGHTING_GONG = 1142
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BOSS_ORDERS = 1182
    CARMINE = 1192
    LILLIE_DETERMINATION = 1227
    GRAVITY_MOUNTAIN = 1252

    LUMIOSE_CITY = 1267
    LILLIES_PEARL = 1172
    LEGACY_ENERGY = 12
    CRUSTLE = 345


class AttackPlan:
    def __init__(
        self,
        attacker: int = -1,
        target: int = -1,
        attack_index: int = -1,
        remain_hp: int = -1,
        needs_energy: bool = False,
    ):
        self.attacker = attacker
        self.target = target
        self.attack_index = attack_index
        self.remain_hp = remain_hp
        self.needs_energy = needs_energy


def prize_count(pokemon: Pokemon) -> int:
    data = get_card_data(pokemon.id)
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == C.LEGACY_ENERGY:
            count -= 1
    for card in pokemon.tools:
        if card.id == C.LILLIES_PEARL and "Lillie" in getattr(data, "name", ""):
            count -= 1
    return max(0, count)


def target_score(pokemon: Pokemon) -> int:
    data = get_card_data(pokemon.id)
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    if pokemon.id in (144, 322, 323, 337):
        score -= 200
    if pokemon.id == 112 and len(pokemon.energies) >= 1:
        score += 300
    score += pokemon.hp
    return score


class LucarioPolicy:
    def __init__(self, obs: Observation, plan: AttackPlan, ability_used: bool):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = get_int_value(self.select.context)
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.my_prizes_left = len(self.me.prize)
        self.plan = plan
        self.ability_used = ability_used

        self.field_counts = defaultdict(int)
        self.hand_counts = defaultdict(int)
        self.discard_counts = defaultdict(int)
        self.has_ready_lucario_line = False
        self.has_ready_hariyama_line = False
        self.can_switch = False
        self.can_gust = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0

        self._count_cards()
        self._scan_main_options()

    def choose(self) -> list[int]:
        if not self.select.option or self.select.maxCount == 0:
            return []

        scores = [self._score_option(option) for option in self.select.option]
        ranked = [i for i, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]
        return ranked

    def _count_cards(self) -> None:
        for pokemon in self.me.active + self.me.bench:
            if pokemon is None:
                continue
            self.field_counts[pokemon.id] += 1
            if pokemon.id in (C.MAKUHITA, C.HARIYAMA) and len(pokemon.energies) >= 3:
                self.has_ready_hariyama_line = True
            if pokemon.id in (C.RIOLU, C.MEGA_LUCARIO_EX) and len(pokemon.energies) >= 2:
                self.has_ready_lucario_line = True

        if self.me.hand:
            for card in self.me.hand:
                self.hand_counts[card.id] += 1
        for card in self.me.discard:
            self.discard_counts[card.id] += 1

    def _scan_main_options(self) -> None:
        if self.context != 0:  # SelectContext.MAIN is 0
            return
        for option in self.select.option:
            o_type_val = get_int_value(option.type)
            if o_type_val == 7:  # PLAY
                card = get_card_helper(self.obs, 2, option.index, self.my_index)  # HAND is 2
                if card is not None:
                    if card.id == C.SWITCH:
                        self.can_switch = True
                    elif card.id == C.BOSS_ORDERS:
                        self.can_gust = True
            elif o_type_val == 9:  # EVOLVE
                card = get_card_helper(self.obs, 2, option.index, self.my_index)
                if card is not None and card.id == C.HARIYAMA:
                    self.can_gust = True
            elif o_type_val == 12:  # RETREAT
                self.can_switch = True
            elif o_type_val == 13:  # ATTACK
                self.can_attack = True
                if option.attackId == 983:  # MEGA_BRAVE is 983
                    self.can_use_mega_brave = True

    def _my_board(self) -> list[Pokemon | None]:
        return self.me.active + self.me.bench

    def _opponent_board(self) -> list[Pokemon | None]:
        return self.opponent.active + self.opponent.bench

    def _can_evolve_board_index(self, board_index: int) -> bool:
        for option in self.select.option:
            if get_int_value(option.type) != 9:  # EVOLVE
                continue
            target_index = option.inPlayIndex
            if get_int_value(option.inPlayArea) == 5:  # BENCH
                target_index += 1
            if target_index == board_index:
                return True
        return False

    def _base_attack(self, pokemon: Pokemon, attack_index: int) -> tuple[int, int, int] | None:
        energy_required = 0
        base_damage = 0
        base_score = 0

        if pokemon.id == C.MEGA_LUCARIO_EX:
            if attack_index == 0:
                energy_required = 1
                base_damage = 130
                base_score += 60 * min(3, self.discard_counts[C.BASIC_FIGHTING_ENERGY])
            else:
                energy_required = 2
                base_damage = 270
            if self.my_prizes_left in (2, 3):
                base_score -= 500
        elif pokemon.id == C.HARIYAMA:
            if attack_index == 0:
                energy_required = 3
                base_damage = 210
            else:
                return None
        elif pokemon.id == C.SOLROCK and self.field_counts[C.LUNATONE] >= 1:
            if attack_index == 0:
                energy_required = 1
                base_damage = 70
            else:
                return None
        else:
            return None

        if base_damage <= 0:
            return None
        return energy_required, base_damage, base_score

    def _base_attack_after_evolution(self, pokemon: Pokemon, board_index: int, attack_index: int):
        if pokemon.id == C.MAKUHITA and attack_index == 0 and self._can_evolve_board_index(board_index):
            return 3, 210, -100
        return self._base_attack(pokemon, attack_index)

    def _energy_target_score(self, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000 + (10 if active else 0)

        if pokemon.id in (C.MAKUHITA, C.HARIYAMA):
            score += 1 if pokemon.id == C.HARIYAMA else 0
            score += 100 if energy_count < 3 else 0
            score -= 50 if self.has_ready_hariyama_line else 0
        elif pokemon.id == C.LUNATONE:
            score -= 100
        elif pokemon.id == C.SOLROCK:
            score += 20 if energy_count < 1 else -100
        elif pokemon.id in (C.RIOLU, C.MEGA_LUCARIO_EX):
            score += 1 if pokemon.id == C.MEGA_LUCARIO_EX else 0
            score += 100 if energy_count < 2 else 0
            score -= 50 if self.has_ready_lucario_line else 0
        return score

    def _score_option(self, option) -> float:
        opt_type_val = get_int_value(option.type)
        if opt_type_val == 0:  # NUMBER
            return float(option.number) if option.number is not None else 0.0
        if opt_type_val == 1:  # YES
            return 100.0 if self.context == 41 else 1.0  # IS_FIRST is 41
        if opt_type_val == 2:  # NO
            return 0.0
        if opt_type_val == 3:  # CARD
            return self._score_card_choice(option)
        if opt_type_val == 7:  # PLAY
            return self._score_play(option)
        if opt_type_val == 8:  # ATTACH
            return self._score_attach(option)
        if opt_type_val == 9:  # EVOLVE
            return self._score_evolve(option)
        if opt_type_val == 10:  # ABILITY
            return self._score_ability(option)
        if opt_type_val == 12:  # RETREAT
            return 2000.0 if self.plan.attacker >= 1 else -1.0
        if opt_type_val == 13:  # ATTACK
            return 1100.0 if (option.attackId == 983) == (self.plan.attack_index == 1) else 1000.0
        return 0.0

    def _score_card_choice(self, option) -> float:
        card = get_card_helper(self.obs, get_int_value(option.area), option.index, option.playerIndex)
        if card is None:
            return 0.0

        # SWITCH (3), TO_ACTIVE (4)
        if self.context in (3, 4):
            return self._score_active_choice(option, card)
        if self.context == 1:  # SETUP_ACTIVE_POKEMON
            return self._score_setup_active(card)
        if self.context == 7:  # TO_HAND
            return self._score_to_hand(card)
        if self.context == 21 and isinstance(card, Pokemon):  # ATTACH_FROM
            return float(self._energy_target_score(card, get_int_value(option.area) == 4))
        return 0.0

    def _score_active_choice(self, option, card: Pokemon | Card) -> float:
        if not isinstance(card, Pokemon):
            return 0.0

        if option.playerIndex != self.my_index:
            return 100.0 if option.index == self.plan.target - 1 else 0.0

        score = len(card.energies) * 2.0
        if option.index == self.plan.attacker - 1:
            score += 100.0
        if card.id == C.MEGA_LUCARIO_EX:
            score += 8.0 if self.my_prizes_left in (2, 3) else 20.0
        elif card.id == C.HARIYAMA and len(card.energies) >= 2:
            score += 15.0
        elif card.id == C.MAKUHITA and len(card.energies) >= 2:
            score += 10.0
        elif card.id == C.SOLROCK:
            score += 5.0
        elif card.id == C.RIOLU:
            score += 4.0
        return score

    def _score_setup_active(self, card: Pokemon | Card) -> int:
        if card.id == C.SOLROCK:
            return 2 if self.state.firstPlayer == self.my_index else 4
        if card.id == C.RIOLU:
            return 3
        if card.id == C.MAKUHITA:
            return 1
        return 0

    def _score_to_hand(self, card: Pokemon | Card) -> float:
        score = 200.0 - self.hand_counts[card.id] * 100.0
        if card.id == C.MAKUHITA:
            score += -10.0 if self.field_counts[card.id] >= 1 else 10.0
        elif card.id == C.HARIYAMA:
            score += 20.0 if self.field_counts[C.MAKUHITA] >= 1 else -20.0
        elif card.id == C.LUNATONE:
            score += -250.0 if self.field_counts[card.id] >= 1 else 60.0
        elif card.id == C.SOLROCK:
            score += -250.0 if self.field_counts[card.id] >= 1 else 50.0
        elif card.id == C.RIOLU:
            lucario_line = self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX]
            score += -150.0 if lucario_line >= 2 else -3.0 if lucario_line >= 1 else 40.0
        elif card.id == C.MEGA_LUCARIO_EX:
            score += 40.0 if self.field_counts[C.RIOLU] >= 1 else -15.0
        elif card.id == C.BASIC_FIGHTING_ENERGY:
            score += 30.0 if not self.ability_used or not self.state.energyAttached else -1.0
        return score

    def _score_play(self, option) -> float:
        card = get_card_helper(self.obs, 2, option.index, self.my_index)
        if card is None:
            return 0.0
        data = get_card_data(card.id)
        if get_int_value(data.cardType) == 0:  # POKEMON is 0
            return self._score_play_pokemon(card)
        return self._score_play_trainer(card)

    def _score_play_pokemon(self, card: Card) -> float:
        score = 20000.0
        if card.id in (C.LUNATONE, C.SOLROCK) and self.field_counts[card.id] >= 1:
            return -1.0
        if card.id == C.RIOLU and self.field_counts[C.RIOLU] + self.field_counts[C.MEGA_LUCARIO_EX] >= 2:
            return -1.0
        return score

    def _score_play_trainer(self, card: Card) -> float:
        if card.id == C.SWITCH:
            return 6000.0 if self.plan.attacker > 0 else -1.0
        if card.id == C.PREMIUM_POWER_PRO:
            if self.state.supporterPlayed and self.plan.remain_hp <= 0:
                return -1.0
            if not self.can_attack:
                can_bridge_draw = (
                    not self.state.supporterPlayed
                    and self.hand_counts[C.CARMINE] > 0
                    and self.hand_counts[C.LILLIE_DETERMINATION] == 0
                    and not self._low_deck()
                )
                return 3050.0 if can_bridge_draw else -1.0
            return 5000.0
        if card.id == C.BOSS_ORDERS:
            return 3200.0 if self.plan.target >= 1 else -1.0
        if card.id == C.CARMINE:
            return -1.0 if self._low_deck() else 3000.0
        if card.id == C.LILLIE_DETERMINATION:
            return -1.0 if self._low_deck() else 3100.0
        if card.id == C.GRAVITY_MOUNTAIN:
            return self._score_gravity_mountain()
        return 10000.0

    def _score_gravity_mountain(self) -> float:
        opponent_has_stage2 = any(
            pokemon is not None and get_card_data(pokemon.id).stage2 for pokemon in self._opponent_board()
        )
        if opponent_has_stage2:
            return 3500.0
        return 1200.0 if self.stadium_id else -1.0

    def _low_deck(self) -> bool:
        return self.me.deckCount <= 8

    def _score_attach(self, option) -> float:
        card = get_card_helper(self.obs, 2, option.index, self.my_index)
        pokemon = get_card_helper(self.obs, get_int_value(option.inPlayArea), option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon) or card is None:
            return 0.0

        if card.id == C.HERO_CAPE:
            score = 7000.0
            if pokemon.id == C.RIOLU:
                score += 100.0
            elif pokemon.id == C.MEGA_LUCARIO_EX:
                score += 200.0
            return score

        score = float(self._energy_target_score(pokemon, get_int_value(option.inPlayArea) == 4))
        board_index = option.inPlayIndex if get_int_value(option.inPlayArea) == 4 else option.inPlayIndex + 1
        if board_index == self.plan.attacker and self.plan.needs_energy:
            score += 200.0
        return score

    def _score_evolve(self, option) -> float:
        pokemon = get_card_helper(self.obs, get_int_value(option.inPlayArea), option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return 0.0
        if pokemon.id == C.MAKUHITA and self.plan.target == 0:
            return -1.0
        return 9000.0 + len(pokemon.energies)

    def _score_ability(self, option) -> float:
        card = get_card_helper(self.obs, get_int_value(option.area), option.index, self.my_index)
        if card is None:
            return 0.0
        if card.id == C.LUMIOSE_CITY:
            return 1.0
        if card.id == C.LUNATONE and self._low_deck():
            return -1.0
        return 30000.0


class LucarioAgent(BaseAgent):
    def __init__(self, deck="lucario.csv"):
        self.deck = self._load_deck(deck)
        self.pre_turn = -1
        self.ability_used = False
        self.plan = AttackPlan()

    def get_deck(self) -> list[int]:
        return self.deck

    def _plan_attack(self, obs: Observation) -> None:
        state = obs.current
        select = obs.select
        my_index = state.yourIndex
        my_state = state.players[my_index]
        op_state = state.players[1 - my_index]

        self.can_switch = False
        self.can_attack = False
        self.can_use_mega_brave = False
        self.can_gust = False

        for option in select.option:
            o_type_val = get_int_value(option.type)
            if o_type_val == 7:  # PLAY
                card = get_card_helper(obs, 2, option.index, my_index)
                if card is not None:
                    if card.id == C.SWITCH:
                        self.can_switch = True
                    elif card.id == C.BOSS_ORDERS:
                        self.can_gust = True
            elif o_type_val == 9:  # EVOLVE
                card = get_card_helper(obs, 2, option.index, my_index)
                if card is not None and card.id == C.HARIYAMA:
                    self.can_gust = True
            elif o_type_val == 12:  # RETREAT
                self.can_switch = True
            elif o_type_val == 13:  # ATTACK
                self.can_attack = True
                if option.attackId == 983:
                    self.can_use_mega_brave = True

        best_score = -1
        self.plan = AttackPlan()

        if state.turn < 2:
            return

        my_board = my_state.active + my_state.bench
        op_board = op_state.active + op_state.bench

        # Differentiate logic depending on options
        policy_helper = LucarioPolicy(obs, self.plan, self.ability_used)

        for attacker_index, my_pokemon in enumerate(my_board):
            if my_pokemon is None:
                continue
            if attacker_index != 0 and not self.can_switch:
                break

            for attack_index in range(2):
                attack = policy_helper._base_attack_after_evolution(my_pokemon, attacker_index, attack_index)
                if attack is None:
                    continue
                energy_required, base_damage, base_score = attack

                energy_count = len(my_pokemon.energies)
                if attack_index == 1 and attacker_index == 0 and energy_count >= 2 and not self.can_use_mega_brave:
                    break

                needs_energy = False
                if energy_count < energy_required:
                    if policy_helper.hand_counts[C.BASIC_FIGHTING_ENERGY] >= 1 and not state.energyAttached:
                        energy_count += 1
                        needs_energy = energy_count >= energy_required
                    if not needs_energy:
                        continue

                for target_index, op_pokemon in enumerate(op_board):
                    if op_pokemon is None:
                        continue
                    if target_index != 0 and not self.can_gust:
                        break

                    # Crustle wall counterplay
                    my_data = get_card_data(my_pokemon.id)
                    crustle_immune = op_pokemon.id == C.CRUSTLE and (my_data.ex or my_data.megaEx)
                    damage = 0 if crustle_immune else base_damage

                    score = target_score(op_pokemon)
                    prize = prize_count(op_pokemon) if op_pokemon.hp <= damage else 0
                    if prize == 0:
                        if op_pokemon.hp > 0:
                            score *= damage / op_pokemon.hp
                        else:
                            score = 0
                    if len(op_state.prize) <= prize:
                        score = 50000

                    if crustle_immune:
                        score = -10000

                    score += base_score
                    score += 220 if attacker_index == 0 else 0
                    score += 300 if target_index == 0 else 0
                    score += energy_count

                    if score > best_score:
                        best_score = score
                        self.plan = AttackPlan(
                            attacker=attacker_index,
                            target=target_index,
                            attack_index=attack_index,
                            remain_hp=op_pokemon.hp - damage,
                            needs_energy=needs_energy,
                        )

    def _evaluate_state(self, obs: Observation) -> float:
        st = obs.current
        if st is None:
            return 0.0
        me = st.players[st.yourIndex]
        op = st.players[1 - st.yourIndex]

        val = 0.0
        val += (len(op.prize) - len(me.prize)) * 10000.0
        for p in (me.active[0] if me.active else None, *me.bench):
            if p is None:
                continue
            val += len(p.energies) * 120.0
            if p.id == C.MEGA_LUCARIO_EX:
                val += 400.0
            if p.id == C.HARIYAMA:
                val += 200.0
        if me.active and me.active[0] is not None:
            val += me.active[0].hp * 1.0
        if op.active and op.active[0] is not None:
            val -= op.active[0].hp * 1.5
        val += me.handCount * 5.0
        return val

    def _legal_fallback(self, select) -> list[int]:
        n = len(select.option)
        k = max(1, select.minCount) if n else 0
        k = min(k, n)
        return list(range(k))

    def _search_plan(self, obs_dict: dict, obs: Observation) -> list[int] | None:
        if not (_SEARCH_OK and USE_SEARCH):
            return None
        select = obs.select
        if select is None or get_int_value(select.context) != 0:  # MAIN is 0
            return None

        t0 = time.time()
        sbi = getattr(obs, "search_begin_input", None) or obs_dict.get("search_begin_input")
        if sbi is None:
            return None

        policy = LucarioPolicy(obs, self.plan, self.ability_used)
        base_order = policy.choose()
        candidates = base_order[:SEARCH_MAX_CANDIDATES]

        best_idx, best_val = None, float("-inf")
        for first in candidates:
            if time.time() - t0 > SEARCH_TIME_BUDGET:
                break
            sid = None
            try:
                res = search_begin(
                    obs,
                    your_deck=random.sample(self.deck, obs.current.players[obs.current.yourIndex].deckCount),
                    your_prize=random.sample(self.deck, len(obs.current.players[obs.current.yourIndex].prize)),
                    opponent_deck=[1072] * obs.current.players[1 - obs.current.yourIndex].deckCount,
                    opponent_prize=[1] * len(obs.current.players[1 - obs.current.yourIndex].prize),
                    opponent_hand=[1] * obs.current.players[1 - obs.current.yourIndex].handCount,
                    opponent_active=[1072]
                    if len(obs.current.players[1 - obs.current.yourIndex].active) > 0
                    and obs.current.players[1 - obs.current.yourIndex].active[0] is None
                    else [],
                )
                if getattr(res, "error", 0) != 0 or res.state is None:
                    return None
                sid = res.state.searchId
                cur = res.state.observation

                sel = [first]
                steps = 0
                while steps < 40:
                    ar = search_step(sid, sel)
                    if getattr(ar, "error", 0) != 0 or ar.state is None:
                        break
                    cur = ar.state.observation
                    if cur.select is None or cur.current is None:
                        break
                    if cur.current.result is not None and cur.current.result != -1:
                        break
                    if cur.current.yourIndex != obs.current.yourIndex:
                        break
                    if get_int_value(cur.select.context) != 0:
                        sub = LucarioPolicy(cur, self.plan, self.ability_used).choose()
                        sel = sub[: max(1, cur.select.minCount)]
                        steps += 1
                        continue
                    nxt = LucarioPolicy(cur, self.plan, self.ability_used).choose()
                    sel = [nxt[0]]
                    steps += 1
                    if get_int_value(cur.select.option[nxt[0]].type) == 14:  # END
                        ar = search_step(sid, sel)
                        if ar.state is not None:
                            cur = ar.state.observation
                        break

                val = self._evaluate_state(cur)
                if val > best_val:
                    best_val, best_idx = val, first
            except Exception:
                return None
            finally:
                try:
                    if sid is not None:
                        search_release(sid)
                except Exception:
                    pass

        if best_idx is None:
            return None
        rest = [i for i in base_order if i != best_idx]
        return [best_idx] + rest

    def act(self, obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return self.get_deck()

        if self.pre_turn != obs.current.turn:
            self.pre_turn = obs.current.turn
            self.ability_used = False
            self.plan = AttackPlan()

        select = obs.select
        try:
            self._plan_attack(obs)
            ordered = None
            if USE_SEARCH:
                ordered = self._search_plan(obs_dict, obs)
            if ordered is None:
                policy = LucarioPolicy(obs, self.plan, self.ability_used)
                ordered = policy.choose()

            n = len(select.option)
            ordered = [i for i in ordered if 0 <= i < n]
            if not ordered:
                return self._legal_fallback(select)

            k = min(select.maxCount, n)
            k = max(k, min(max(1, select.minCount), n))

            # Record if ability was used from the first action chosen
            if ordered and get_int_value(select.option[ordered[0]].type) == 10:  # ABILITY
                card = get_card_helper(
                    obs,
                    get_int_value(select.option[ordered[0]].area),
                    select.option[ordered[0]].index,
                    obs.current.yourIndex,
                )
                if card is not None and card.id == C.LUNATONE:
                    self.ability_used = True

            return ordered[:k]
        except Exception:
            return self._legal_fallback(select)


# Define submission agent wrapper for Kaggle and pipeline execution
_agent_instance = None


def agent(obs_dict: dict) -> list[int]:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LucarioAgent()
    return _agent_instance.act(obs_dict)
