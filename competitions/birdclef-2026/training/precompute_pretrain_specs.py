"""Precompute HTK mel spectrograms for BirdCLEF 2021-2024 training audio.

Saves float16 .npy files to specs_cache_pretrain/ inside each year's directory,
mirroring the audio subdirectory structure. Uses identical mel config to v7
(n_mels=224, n_fft=2048, hop=512, fmin=0, htk=True, fmax=16k, sr=32k).

Usage (run on cluster):
    KEGO_PATH_DATA=/home/kristian/projects/kego/data \\
    uv run python competitions/birdclef-2026/training/precompute_pretrain_specs.py \\
        --years 2021 2022 2023 2024 --workers 8

    # Check completeness only:
    ... --check
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

SR = 32000
N_MELS = 224
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = 16000
MAX_DURATION = 30.0  # cap long XC recordings

DATA_ROOT = Path(os.getenv("KEGO_PATH_DATA", "data")) / "birdclef"
CACHE_NAME = "specs_cache_pretrain"


def _audio_dir(year_dir: Path, year: int) -> Path:
    return year_dir / ("train_short_audio" if year == 2021 else "train_audio")


def _collect_files(year: int) -> list[tuple[Path, Path]]:
    """Return list of (audio_path, cache_path) for all .ogg files in the year."""
    year_dir = DATA_ROOT / f"birdclef-{year}"
    audio_dir = _audio_dir(year_dir, year)
    cache_dir = year_dir / CACHE_NAME

    pairs = []
    for ogg in sorted(audio_dir.rglob("*.ogg")):
        rel = ogg.relative_to(audio_dir)
        cache_path = cache_dir / rel.parent / f"{ogg.stem}.npy"
        pairs.append((ogg, cache_path))
    return pairs


def _worker(args: tuple) -> tuple[str, bool, str]:
    audio_path, cache_path = args
    if cache_path.exists():
        return str(audio_path), True, "cached"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        info = sf.info(audio_path)
        max_frames = int(MAX_DURATION * info.samplerate)
        frames_to_read = min(info.frames, max_frames)
        y, file_sr = sf.read(
            audio_path, frames=frames_to_read, dtype="float32", always_2d=False
        )
        if y.ndim == 2:
            y = y.mean(axis=1)
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
            htk=True,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float16)
        np.save(cache_path, log_mel)
        return str(audio_path), True, "ok"
    except Exception as e:
        return str(audio_path), False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024]
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--check", action="store_true", help="Report completeness only, no compute"
    )
    args = parser.parse_args()

    for year in args.years:
        pairs = _collect_files(year)
        cached = sum(1 for _, cp in pairs if cp.exists())
        print(
            f"\n{year}: {len(pairs)} files, {cached} cached, {len(pairs) - cached} pending",
            flush=True,
        )

        if args.check or len(pairs) == cached:
            continue

        pending = [(ap, cp) for ap, cp in pairs if not cp.exists()]
        errors = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker, p): p for p in pending}
            with tqdm(total=len(pending), desc=f"{year}", unit="file") as bar:
                for fut in as_completed(futs):
                    _, ok, _ = fut.result()
                    if not ok:
                        errors += 1
                    bar.update(1)

        total = _collect_files(year)
        done = sum(1 for _, cp in total if cp.exists())
        print(
            f"{year}: complete — {done}/{len(total)} cached, {errors} errors",
            flush=True,
        )


if __name__ == "__main__":
    main()
