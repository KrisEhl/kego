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
from train import (
    CACHE_DIR_BASELINE,
    CACHE_DIR_BASELINE_HTK,
    CACHE_DIR_HGNETV2,
    DATA,
    FMAX,
    FMIN,
    FMIN_HTK,
    HOP_LENGTH,
    HOP_LENGTH_HGNETV2,
    N_FFT,
    N_FFT_BASELINE,
    N_FFT_HGNETV2,
    N_MELS,
    N_MELS_BASELINE,
    N_MELS_HGNETV2,
    SOUNDSCAPE_CACHE_DIR_BASELINE_HTK,
    SOUNDSCAPE_CACHE_DIR_HGNETV2,
    SR,
)

SOUNDSCAPE_CACHE_DIR = DATA / "specs_cache_soundscape_224"

MAX_DURATION = 30.0  # seconds; longer XC clips are truncated (soundscapes: no cap)
CACHE_DIR = DATA / "specs_cache"


def compute_and_save(args: tuple) -> tuple[str, bool, str]:
    """Worker: load audio, compute mel spec, save as float16 npy."""
    (
        filename,
        audio_dir,
        cache_dir,
        n_mels,
        n_fft,
        hop_length,
        cap_duration,
        fmin,
        htk,
    ) = args
    src = audio_dir / filename
    stem = Path(filename).stem
    subdir = Path(filename).parent
    dst = cache_dir / subdir / f"{stem}.npy"

    if dst.exists():
        return filename, True, "cached"

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        info = sf.info(src)
        if cap_duration is not None:
            max_frames = int(cap_duration * info.samplerate)
            frames_to_read = min(info.frames, max_frames)
        else:
            frames_to_read = info.frames

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
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            fmin=fmin,
            fmax=FMAX,
            htk=htk,
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
        "--baseline",
        action="store_true",
        help="Compute 224-mel specs (n_fft=2048) into specs_cache_224/ for baseline model",
    )
    parser.add_argument(
        "--soundscapes",
        action="store_true",
        help="Compute 224-mel specs for train_soundscapes/ into specs_cache_soundscape_224/. "
        "Full 60s per file (no cap). Used by PerchDataset for fast stage-1 pretraining.",
    )
    parser.add_argument(
        "--htk",
        action="store_true",
        help="Use HTK mel scale (fmin=0, htk=True). Only valid with --baseline or --soundscapes. "
        "Saves to specs_cache_224_htk/ or specs_cache_soundscape_224_htk/.",
    )
    parser.add_argument(
        "--hgnetv2",
        action="store_true",
        help="Compute 256-mel specs (n_fft=2048, hop=625, fmin=20, slaney) into specs_cache_hgnetv2/. "
        "Use with --soundscapes to also cache train_soundscapes/.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report cache completeness, don't compute",
    )
    args = parser.parse_args()

    if args.hgnetv2:
        fmin = FMIN  # 20 Hz, slaney norm
        htk = False
        hop_length = HOP_LENGTH_HGNETV2
        n_mels = N_MELS_HGNETV2
        n_fft = N_FFT_HGNETV2
        if args.soundscapes:
            cache_dir = SOUNDSCAPE_CACHE_DIR_HGNETV2
            audio_dir = DATA / "train_soundscapes"
            filenames = [p.name for p in sorted(audio_dir.glob("*.ogg"))]
            cap_duration = None
        else:
            cache_dir = CACHE_DIR_HGNETV2
            audio_dir = DATA / "train_audio"
            train = pd.read_csv(DATA / "train.csv")
            filenames = train["filename"].tolist()
            cap_duration = MAX_DURATION
    elif args.soundscapes:
        fmin = FMIN_HTK if args.htk else FMIN
        htk = args.htk
        hop_length = HOP_LENGTH
        if args.htk:
            cache_dir = SOUNDSCAPE_CACHE_DIR_BASELINE_HTK
        else:
            cache_dir = SOUNDSCAPE_CACHE_DIR
        n_mels = N_MELS_BASELINE
        n_fft = N_FFT_BASELINE
        audio_dir = DATA / "train_soundscapes"
        filenames = [p.name for p in sorted(audio_dir.glob("*.ogg"))]
        cap_duration = None  # soundscapes are full 60s — no cap
    elif args.baseline:
        fmin = FMIN_HTK if args.htk else FMIN
        htk = args.htk
        hop_length = HOP_LENGTH
        cache_dir = CACHE_DIR_BASELINE_HTK if args.htk else CACHE_DIR_BASELINE
        n_mels = N_MELS_BASELINE
        n_fft = N_FFT_BASELINE
        audio_dir = DATA / "train_audio"
        train = pd.read_csv(DATA / "train.csv")
        filenames = train["filename"].tolist()
        cap_duration = MAX_DURATION
    else:
        fmin = FMIN
        htk = False
        hop_length = HOP_LENGTH
        cache_dir = CACHE_DIR
        n_mels = N_MELS
        n_fft = N_FFT
        audio_dir = DATA / "train_audio"
        train = pd.read_csv(DATA / "train.csv")
        filenames = train["filename"].tolist()
        cap_duration = MAX_DURATION

    if args.check:
        cached = sum(
            1
            for f in filenames
            if (cache_dir / Path(f).parent / f"{Path(f).stem}.npy").exists()
        )
        print(f"Cache: {cached}/{len(filenames)} files ({cached / len(filenames):.1%})")
        if cached < len(filenames):
            missing = [
                f
                for f in filenames
                if not (cache_dir / Path(f).parent / f"{Path(f).stem}.npy").exists()
            ]
            print(f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (f, audio_dir, cache_dir, n_mels, n_fft, hop_length, cap_duration, fmin, htk)
        for f in filenames
    ]

    print(f"Computing specs for {len(tasks)} files with {args.workers} workers...")
    print(f"n_mels={n_mels}, n_fft={n_fft}, hop={hop_length}, fmin={fmin}, htk={htk}")
    print(f"Output: {cache_dir}")

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

    total_gb = sum(f.stat().st_size for f in cache_dir.rglob("*.npy")) / 1024**3
    print(f"\nDone. ok={ok}, skipped={skipped}, errors={errors}")
    print(f"Cache size: {total_gb:.1f} GB")


if __name__ == "__main__":
    main()
