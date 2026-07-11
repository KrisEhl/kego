# Deck lists

Deck CSV files contain one simulator card ID per line. Keep experimental decks separate rather than overwriting a baseline that has existing training or league results.

Every CSV under `decks/` is covered automatically by `tests/test_pokemon_decks.py`. The tests enforce 60 cards, known simulator IDs, copy and ACE SPEC limits, at least one Basic Pokémon, and acceptance by the battle engine.

## Dragapult baseline

File: [`decks/dragapult.csv`](decks/dragapult.csv)

The saved baseline is a streamlined Dragapult control list. It uses Crushing Hammer, two Team Rocket's Watchtower, Budew, and Lucky Helmet to disrupt the opponent while developing Dragapult ex. It has 16 Pokémon, 36 Trainers, and 8 Energy.

## Dragapult–Blaziken

File: [`decks/dragapult_blaziken.csv`](decks/dragapult_blaziken.csv)

This list is adapted from [Lucas Portal's tournament deck list on Limitless TCG](https://play.limitlesstcg.com/tournament/6a19c0ff56b11a587a942cd3/player/lucasportal/decklist). The original `Special Red Card CRI 82` is not present in the competition simulator, so it is replaced with a third Rare Candy. All other cards match the referenced list by card identity; the simulator represents them with its own card IDs rather than physical set and collector numbers.

### Composition

| Category | Cards |
|---|---|
| Pokémon (23) | 4 Dreepy, 4 Drakloak, 2 Dragapult ex, 2 Torchic, 1 Combusken, 2 Blaziken ex, 2 Budew, 2 Munkidori, 1 Chi-Yu, 1 Fezandipiti ex, 1 Lillie's Clefairy ex, 1 Meowth ex |
| Trainers (29) | 4 Lillie's Determination, 3 Boss's Orders, 2 Crispin, 1 Dawn, 4 Buddy-Buddy Poffin, 4 Ultra Ball, 3 Poké Pad, 3 Rare Candy, 2 Night Stretcher, 1 Unfair Stamp, 1 Team Rocket's Watchtower, 1 Risky Ruins |
| Energy (8) | 3 Psychic Energy, 3 Fire Energy, 2 Darkness Energy |

### Comparison with the baseline

Only 37 of 60 card slots are unchanged. The adapted list removes Latias ex and the baseline's Crushing Hammer, Lucky Helmet, and Brock's Scouting package. It adds the Blaziken ex evolution line, Munkidori, Chi-Yu, Lillie's Clefairy ex, Dawn, Risky Ruins, and Darkness Energy.

The baseline is simpler and more control-oriented. The adapted deck has a larger action space: Blaziken ex accelerates Basic Energy from the discard pile, Darkness Energy enables Munkidori to move damage counters, and Risky Ruins supplies additional damage-counter pressure. This makes it potentially stronger but more demanding for a rule agent or learned policy to pilot.

## Meta references

[Limitless TCG](https://play.limitlesstcg.com/decks) is the preferred source for current meta deck lists. Filters should be checked for the applicable Standard rotation and set snapshot, then every selected card must be verified against the simulator registry.
