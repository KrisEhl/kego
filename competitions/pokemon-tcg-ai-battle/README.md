## Plan
- [~] Track 1 - Search/data efficiency ablation: deeper MCTS targets with fewer optimizer steps.
- [ ] Track 2 - League evaluation: compare registered checkpoints with enough games to distinguish Elo from saturated gauntlet score.
- [ ] Track 3 - Self-play diversity ablation: increase generated games per iteration before changing model architecture.
- [ ] Track 4 - Architecture ablation: test smaller/faster or deeper encoder only after search/data runs are ranked.

## Results log
| Run | Model | Win rate / Elo | Notes |
|---|---|---:|---|
| Registry v1 | MCTS Transformer `(256, 4, 512, 2, 2)` | Elo 1746.2, gauntlet_avg 90.2 | Best known registry agent as of 2026-07-04; trained on omarchyd, git `e7bd6e8`. |
| Registry v2 | MCTS Transformer `(256, 4, 512, 2, 2)` | Elo 1657.0, gauntlet_avg 90.2 | Same saturated gauntlet score as v1 but lower league Elo; use league games to rank future runs. |
| Registry v3 | MCTS Transformer variant | Elo 1155.2, gauntlet_avg 45.0 | Weak Windows run; useful as a regression marker only. |
| search40_train100 | MCTS Transformer `(256, 4, 512, 2, 2)` | TBD | In progress: `search_count=40`, `train_steps=100`, `self_play_games=48`, batched MCTS. Tests whether deeper targets beat extra gradient steps. |

## Dead ends
| Approach | Result | Why it failed |
|---|---|---|
| Duplicate long training with same config | TBD | Expected low information gain: current registry already has two runs at gauntlet_avg 90.2 but different Elo. |
