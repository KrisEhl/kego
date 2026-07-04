## Plan
- [~] Track 1 - Search/data efficiency ablation: deeper MCTS targets with fewer optimizer steps.
- [ ] Track 2 - League evaluation: compare registered checkpoints with enough games to distinguish Elo from saturated gauntlet score.
- [~] Track 3 - Self-play diversity ablation: increase generated games per iteration before changing model architecture.
- [~] Track 4 - Architecture ablation: test smaller/faster or deeper encoder only after search/data runs are ranked.

## Results log
| Run | Model | Win rate / Elo | Notes |
|---|---|---:|---|
| Registry v1 | MCTS Transformer `(256, 4, 512, 2, 2)` | Elo 1746.2, gauntlet_avg 90.2 | Best known registry agent as of 2026-07-04; trained on omarchyd, git `e7bd6e8`. |
| Registry v2 | MCTS Transformer `(256, 4, 512, 2, 2)` | Elo 1657.0, gauntlet_avg 90.2 | Same saturated gauntlet score as v1 but lower league Elo; use league games to rank future runs. |
| Registry v3 | MCTS Transformer variant | Elo 1155.2, gauntlet_avg 45.0 | Weak Windows run; useful as a regression marker only. |
| Registry v4 | MCTS Transformer duplicate baseline-ish seed | gauntlet_avg 62.25 | Finished DESKTOP seed-variance run; registered without league Elo yet. |
| search40_train100 | MCTS Transformer `(256, 4, 512, 2, 2)` | TBD | In progress: `search_count=40`, `train_steps=100`, `self_play_games=48`, batched MCTS. Tests whether deeper targets beat extra gradient steps. |
| selfplay96_train100 | MCTS Transformer `(256, 4, 512, 2, 2)` | TBD | In progress: `search_count=25`, `train_steps=100`, `self_play_games=96`, batched MCTS. Tests whether more diverse targets beat extra gradient steps. |
| small192_selfplay48_train100 | MCTS Transformer `(192, 4, 384, 2, 2)` | TBD | In progress: `search_count=25`, `train_steps=100`, `self_play_games=48`, batched MCTS. Tests whether a smaller/faster model is sufficient. |
| small192_dragapult_train100 | MCTS Transformer `(192, 4, 384, 2, 2)` on Dragapult deck | TBD | In progress: CPU probe on `mn-exjk9p93n75h`, `search_count=25`, `train_steps=100`, `self_play_games=48`, 100 epochs. Tests deck sensitivity. |

## Dead ends
| Approach | Result | Why it failed |
|---|---|---|
| Duplicate long training with same config on omarchyl | Killed at progress_pct 45.5, gauntlet_avg 54.25 | Low information gain versus using the machine for a controlled ablation; near-finished DESKTOP run kept for seed-variance signal. |
