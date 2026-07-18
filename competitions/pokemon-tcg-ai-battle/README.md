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
| Pre-2026-07-04 `submit-leader` packaging | Kaggle LB below hardcoded agents despite strong local league | Submitted `main.py` did not auto-load packaged `mcts.pth`; kernel also only searched the competition input path, missing attached weight datasets. |

## Operational notes
- `--epochs` is the target total iteration count. Training automatically resumes the highest registry checkpoint with the same effective variant and source/data fingerprint, restoring model, optimizer, LR scheduler, replay buffer, RNG, best score, and iteration offset. For example, requesting 300 after a compatible 250-iteration run executes iterations 251–300.
- Setting `[train].init_checkpoint` or passing `--init-checkpoint` remains an explicit weight-only warm start and disables automatic selection for that run.
- Before spending a Kaggle submission, build `submission.tar.gz`, extract it locally, confirm `main.py` loads packaged `mcts.pth` with no `MCTS_MODEL_PATH`, and run smoke games against the rule agents.

## Decks

See [`DECKLISTS.md`](DECKLISTS.md) for saved deck compositions, provenance, simulator substitutions, and comparisons. The original Dragapult baseline remains in `decks/dragapult.csv`; the Limitless-derived Dragapult–Blaziken list is saved separately in `decks/dragapult_blaziken.csv`.

## MCTS architecture

The MCTS implementation lives in the `agents/mcts/` package:

- `model.py` defines the sparse transformer policy/value network and infers architecture dimensions from saved checkpoints.
- `encoding.py` converts observations and legal actions into sparse feature vectors.
- `opponent.py` detects known archetypes from revealed cards and samples plausible hidden hand, Prize, and deck cards for search determinization.
- `search.py` owns shared tree nodes, action enumeration, policy priors, node evaluation, and PUCT selection.
- `agent.py` handles hidden-information determinization, checkpoint/device configuration, and the Kaggle `agent()` entry point.
- `__init__.py` exposes the public API and compatibility aliases used by older training scripts.

Training imports the shared model, encoders, and tree mechanics rather than maintaining parallel implementations. Submission packaging includes the complete package behind a small `main.py` shim. Existing checkpoints remain compatible because model attribute names and feature layouts are pinned by golden tests.

Inference and training also share the same opponent determinization. It recognizes the saved Abomasnow, Dragapult, Dragapult–Blaziken, Zacian, and Lucario lists; revealed cards are subtracted before the unseen zones are sampled. Unknown or tied archetypes are sampled rather than treated as certain.
