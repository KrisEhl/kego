"""Precompute mel spectrograms for all training files and save as float16 .npy.

Reads train_audio/*.ogg → computes log-mel spec (up to MAX_DURATION seconds) →
saves to specs_cache/ mirroring the audio directory structure.

All BirdCLEF 2026 files are already at 32kHz, so no resampling is needed.
Capped at MAX_DURATION to keep disk usage manageable (~17GB for 30s cap).

Usage:
    python precompute_specs.py              # all files, 8 workers
    python precompute_specs.py --workers 4  # fewer workers
    python precompute_specs.py --check      # verify cache completeness
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from train import DATA, FMAX, FMIN, HOP_LENGTH, N_FFT, N_MELS, SR

MAX_DURATION = 30.0  # seconds; longer files are truncated
CACHE_DIR = DATA / "specs_cache"


def compute_and_save(args: tuple) -> tuple[str, bool, str]:
    """Worker: load audio, compute mel spec, save as float16 npy."""
    filename, audio_dir, cache_dir = args
    src = audio_dir / filename
    stem = Path(filename).stem
    subdir = Path(filename).parent
    dst = cache_dir / subdir / f"{stem}.npy"

    if dst.exists():
        return filename, True, "cached"

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        info = sf.info(src)
        max_frames = int(MAX_DURATION * info.samplerate)
        frames_to_read = min(info.frames, max_frames)

        y, file_sr = sf.read(
            src, frames=frames_to_read, dtype="float32", always_2d=False
        )
        if y.ndim == 2:
            y = y.mean(axis=1)

        # Resample only if not 32kHz (should not happen for BirdCLEF 2026)
        if file_sr != SR:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=SR)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            fmin=FMIN,
            fmax=FMAX,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float16)
        np.save(dst, log_mel)
        return filename, True, "ok"
    except Exception as e:
        return filename, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report cache completeness, don't compute",
    )
    args = parser.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    audio_dir = DATA / "train_audio"
    filenames = train["filename"].tolist()

    if args.check:
        cached = sum(
            1
            for f in filenames
            if (CACHE_DIR / Path(f).parent / f"{Path(f).stem}.npy").exists()
        )
        print(f"Cache: {cached}/{len(filenames)} files ({cached / len(filenames):.1%})")
        if cached < len(filenames):
            missing = [
                f
                for f in filenames
                if not (CACHE_DIR / Path(f).parent / f"{Path(f).stem}.npy").exists()
            ]
            print(f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(f, audio_dir, CACHE_DIR) for f in filenames]

    print(f"Computing specs for {len(tasks)} files with {args.workers} workers...")
    print(f"Output: {CACHE_DIR}")

    ok = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compute_and_save, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), unit="file") as pbar:
            for fut in as_completed(futures):
                _, success, msg = fut.result()
                if not success:
                    errors += 1
                    tqdm.write(f"ERROR {futures[fut]}: {msg}")
                elif msg == "cached":
                    skipped += 1
                else:
                    ok += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok, skip=skipped, err=errors)

    total_gb = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.npy")) / 1024**3
    print(f"\nDone. ok={ok}, skipped={skipped}, errors={errors}")
    print(f"Cache size: {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
