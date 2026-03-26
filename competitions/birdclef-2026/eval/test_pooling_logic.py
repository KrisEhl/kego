"""
Test class-conditional pooling and persistence penalty logic in isolation.

Verifies:
  1. Event classes (birds): max pooling across overlapping windows per slot
  2. Texture classes (frogs/insects): mean pooling (unchanged)
  3. Persistence penalty: isolated spikes in texture classes are suppressed

Run:
    python competitions/birdclef-2026/test_pooling_logic.py
"""

import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

N_SPECIES = 6
# Indices 0-2 = event (birds), 3-5 = texture (frogs/insects)
is_event = np.array([True, True, True, False, False, False])
is_texture = ~is_event


def aggregate_slots(window_preds, positions, n_slots, clip_duration, sr):
    """Class-conditional overlap aggregation.

    Event classes  → max  pooling across overlapping windows per slot.
    Texture classes → mean pooling across overlapping windows per slot.
    """
    n_species = window_preds.shape[1]
    slot_sum = np.zeros((n_slots, n_species), dtype=np.float32)
    slot_max = np.full((n_slots, n_species), -np.inf, dtype=np.float32)
    slot_counts = np.zeros(n_slots, dtype=np.int32)

    for w_idx, pos in enumerate(positions):
        w_start = pos / sr
        w_end = w_start + clip_duration
        for slot in range(n_slots):
            s_start = slot * clip_duration
            s_end = s_start + clip_duration
            if w_start < s_end and w_end > s_start:
                slot_sum[slot] += window_preds[w_idx]
                slot_max[slot] = np.maximum(slot_max[slot], window_preds[w_idx])
                slot_counts[slot] += 1

    slot_counts = np.maximum(slot_counts, 1)
    slot_mean = slot_sum / slot_counts[:, np.newaxis]
    # Replace -inf (no coverage) with 0
    slot_max = np.where(slot_max == -np.inf, 0.0, slot_max)

    return np.where(is_event, slot_max, slot_mean)


def apply_persistence_penalty(preds, is_texture, threshold=0.30, strength=0.40):
    """Suppress isolated spikes in texture (chorusing) classes.

    A spike at time t is penalised if it exceeds the mean of its neighbours
    by more than `threshold`. The excess above threshold is pulled back by
    `strength` toward the neighbour mean.

    Event classes are untouched (bursty calls should not be penalised).
    """
    if len(preds) < 3:
        return preds

    out = preds.copy()
    nbr_mean = np.empty_like(preds)
    nbr_mean[0] = preds[1]
    nbr_mean[-1] = preds[-2]
    nbr_mean[1:-1] = 0.5 * (preds[:-2] + preds[2:])

    spike = np.maximum(preds - nbr_mean, 0.0)  # positive only
    excess = np.maximum(spike - threshold, 0.0)
    out[:, is_texture] -= strength * excess[:, is_texture]
    return np.clip(out, 0.0, 1.0)


# ── test 1: class-conditional pooling ─────────────────────────────────────────
print("=" * 60)
print("TEST 1: class-conditional pooling")
print("=" * 60)

# Simulate a bird call that occupies window 1 (out of 3 overlapping windows
# for slot 0). Mean pooling would dilute the call; max pooling should
# preserve the peak.
#
# Slot 0 covered by windows 0, 1, 2:
#   window 0: event=0.1, texture=0.4
#   window 1: event=0.9, texture=0.5   ← bird call peaks here
#   window 2: event=0.1, texture=0.4

SR = 32000
CLIP_DURATION = 5.0
STRIDE = 2.5
clip_samples = int(SR * CLIP_DURATION)
stride_samples = int(SR * STRIDE)
# 15s audio → n_slots=3; positions = 0, 2.5s, 5s, 7.5s, 10s, 12.5s
audio_len = int(SR * 15)
positions = list(range(0, audio_len - clip_samples + 1, stride_samples))
n_slots = audio_len // clip_samples  # = 3

# window predictions: shape (n_windows, n_species)
# For simplicity, slot 0 windows = 0,1,2; slot 1 = 2,3,4; slot 2 = 4,5,6
n_windows = len(positions)
window_preds = np.full((n_windows, N_SPECIES), 0.1, dtype=np.float32)
# Bird call in window 1 (pos=2.5s, overlaps slot 0)
window_preds[1, :3] = 0.9  # event spike
window_preds[1, 3:] = 0.5  # texture also elevated (but not an isolated spike here)

slot_out = aggregate_slots(window_preds, positions, n_slots, CLIP_DURATION, SR)

print("\nSlot 0 results (bird call in window 1 of 3 overlapping windows):")
print(f"  Event species (max pool):    {slot_out[0, :3].round(3)}")
print(f"  Texture species (mean pool): {slot_out[0, 3:].round(3)}")

# Expected: event ≈ 0.9 (max preserved), texture ≈ mean(0.1, 0.5, 0.1) ≈ 0.233
mean_event = np.mean([0.1, 0.9, 0.1])
print(f"\n  Old mean pool would give event: {mean_event:.3f} (diluted)")
print(f"  Max pool gives event:           {slot_out[0, 0]:.3f} (peak preserved) ✓")

assert slot_out[0, 0] >= 0.85, f"Max pool failed: {slot_out[0, 0]}"
assert slot_out[0, 3] < 0.35, f"Mean pool failed: {slot_out[0, 3]}"
print("  PASS")


# ── test 2: persistence penalty ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: persistence penalty")
print("=" * 60)

# Case A: isolated spike in texture class — should be suppressed
# Case B: sustained high score in texture class — should be preserved
# Case C: isolated spike in event class (bird call) — should NOT be suppressed

T = 7
preds = np.full((T, N_SPECIES), 0.05, dtype=np.float32)

# Case A: texture spike at t=3, low at t=2 and t=4
preds[3, 4] = 0.75  # texture species 4: isolated spike

# Case B: sustained frog chorus at t=2,3,4,5
preds[2:6, 5] = 0.80  # texture species 5: persistent → should survive

# Case C: isolated bird call at t=3
preds[3, 1] = 0.85  # event species 1: isolated spike → must NOT be penalised

after = apply_persistence_penalty(preds.copy(), is_texture)

print("\nCase A — isolated texture spike (sp 4, t=3):")
print(f"  Before: {preds[:, 4].round(3)}")
print(f"  After:  {after[:, 4].round(3)}")
assert after[3, 4] < preds[3, 4] - 0.1, "Isolated texture spike should be penalised"
print("  PASS: spike suppressed")

print("\nCase B — sustained texture signal (sp 5, t=2..5):")
print(f"  Before: {preds[:, 5].round(3)}")
print(f"  After:  {after[:, 5].round(3)}")
assert after[3, 5] > 0.70, "Sustained chorus should be mostly preserved"
print("  PASS: sustained signal preserved")

print("\nCase C — isolated bird call (sp 1, t=3):")
print(f"  Before: {preds[:, 1].round(3)}")
print(f"  After:  {after[:, 1].round(3)}")
assert after[3, 1] == preds[3, 1], "Event class isolated spike must NOT be penalised"
print("  PASS: bird call left intact")


# ── test 3: combined pipeline ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: combined pipeline — slot aggregation → smoothing → penalty")
print("=" * 60)


def smooth_class_conditional(
    preds, is_event, is_texture, alpha_texture=0.35, alpha_event=0.15
):
    """Current smoothing: texture avg-neighbour, event local-max."""
    if len(preds) < 2:
        return preds
    T = len(preds)
    smooth = preds.copy()
    # Texture: average-neighbour
    smooth[1:-1][:, is_texture] = (
        (1 - 2 * alpha_texture) * preds[1:-1][:, is_texture]
        + alpha_texture * preds[:-2][:, is_texture]
        + alpha_texture * preds[2:][:, is_texture]
    )
    smooth[0][is_texture] = (1 - alpha_texture) * preds[0][
        is_texture
    ] + alpha_texture * preds[1][is_texture]
    smooth[-1][is_texture] = (1 - alpha_texture) * preds[-1][
        is_texture
    ] + alpha_texture * preds[-2][is_texture]
    # Event: local-max propagation
    local_max = preds.copy()
    local_max[1:-1] = np.maximum(preds[1:-1], np.maximum(preds[:-2], preds[2:]))
    local_max[0] = np.maximum(preds[0], preds[1])
    local_max[-1] = np.maximum(preds[-1], preds[-2])
    smooth[:, is_event] = (1 - alpha_event) * preds[
        :, is_event
    ] + alpha_event * local_max[:, is_event]
    return smooth


# Simulate 12-slot soundscape with one frog noise burst and one real bird call
T = 12
slots = np.full((T, N_SPECIES), 0.03, dtype=np.float32)
slots[5, 4] = 0.72  # texture false positive: isolated noise burst
slots[3:7, 5] = 0.78  # real frog chorus: sustained
slots[8, 0] = 0.81  # real bird call: brief but genuine

# Correct order: penalty BEFORE smoothing.
# Reason: avg-neighbour smoothing propagates spike to t±1, erasing the
# "isolated" signature — penalty must fire on the raw slots first.
penalized = apply_persistence_penalty(slots.copy(), is_texture)
final = smooth_class_conditional(penalized, is_event, is_texture)

# Compare: old pipeline (smooth only) vs new (penalty → smooth)
old = smooth_class_conditional(slots.copy(), is_event, is_texture)

print("\nFrog noise burst (sp 4, t=5):")
print(
    f"  raw={slots[5, 4]:.2f}  old(smooth only)={old[5, 4]:.2f}  new(penalty→smooth)={final[5, 4]:.2f}"
)

print("\nFrog chorus onset (sp 5, t=2, neighbour t=1=0.03 / t=3=0.78):")
print(f"  raw={slots[2, 5]:.2f}  old={old[2, 5]:.2f}  new={final[2, 5]:.2f}")

print("\nFrog chorus core (sp 5, t=4, both neighbours 0.78):")
print(f"  raw={slots[4, 5]:.2f}  old={old[4, 5]:.2f}  new={final[4, 5]:.2f}")

print("\nBird call (sp 0, t=8, isolated):")
print(f"  raw={slots[8, 0]:.2f}  old={old[8, 0]:.2f}  new={final[8, 0]:.2f}")

assert final[5, 4] < old[5, 4], (
    "Penalty→smooth should reduce noise burst more than smooth alone"
)
assert final[4, 5] > 0.60, "Real chorus core should survive"
assert final[8, 0] == old[8, 0], "Bird call must not be affected"
print("\nAll assertions passed. ✓")
