"""Local steps of the 19th training workflow: prepare uploads, upload, verify and predict.

19th runs are started and followed through its MCP server (prep_run, submit_run,
get_results). This script does everything around those calls that touches files:

  1. prepare       Write the upload CSV for each training size (id column dropped) and a
                   manifest with its byte count, SHA-256 and the prediction declaration to
                   pass to prep_run.
  2. upload        PUT one CSV to the signed upload URL that prep_run returned.
  3. predict       Verify a downloaded model bundle against the checksum get_results
                   reported, check its prediction code is the reviewed version, run it on
                   the holdout and write benchmark/<dataset>/predictions/
                   pred_<size>_<mode>_<variant>.csv.
  4. predict-test  Same checks, then predict the competition's test.csv and write a Kaggle
                   submission to benchmark/<dataset>/submissions/.

Then score with benchmark_services.py (score / compare / summary).

Usage:
  uv run python nineteenth_workflow.py --dataset fraud prepare
  uv run python nineteenth_workflow.py --dataset fraud upload --size 5000 --url '<signed upload url>'
  uv run python nineteenth_workflow.py --dataset fraud predict --size 5000 --mode high --variant fast \\
      --bundle model_bundle.zip --expected-sha256 <bundle sha256 from get_results>
  uv run python nineteenth_workflow.py --dataset fraud predict-test --size 599960 \\
      --bundle model_bundle.zip --expected-sha256 <sha256>
"""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_services import (
    DATASETS,
    DEFAULT_MODE,
    DEFAULT_VARIANT,
    MODEL_VARIANTS,
    TRAINING_MODES,
    DatasetConfig,
    load_holdout,
)

POSITIVE_LABEL_VALUE = 1

# SHA-256 over the bundle's *.py and requirements.txt (sorted relative path + bytes). This is
# the oracle_numpy.v1 runtime (tfp-runtime-0.12) reviewed on 2026-11-26: standard library and
# numpy only, no network, subprocess or eval. A bundle with other code is refused until reviewed.
REVIEWED_BUNDLE_CODE_SHA256 = {"825c4a65bb5ecea9aa8ea2e7c7767a166b595c30c90881613db5005f9fa80c57"}


def uploads_directory(config: DatasetConfig) -> Path:
    return config.samples_directory / "nineteenth"


def manifest_path(config: DatasetConfig) -> Path:
    return uploads_directory(config) / "manifest.json"


def models_directory(config: DatasetConfig) -> Path:
    return config.directory / "models"


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_code_sha256(bundle_directory: Path) -> str:
    """Hash every executable part of a bundle so a changed runtime is detected."""
    digest = hashlib.sha256()
    code_files = sorted(
        path
        for path in bundle_directory.rglob("*")
        if path.is_file() and (path.suffix == ".py" or path.name == "requirements.txt")
    )
    for path in code_files:
        digest.update(path.relative_to(bundle_directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prediction_declaration(config: DatasetConfig) -> dict:
    """The prep_run `prediction` object for every run on this dataset."""
    return {
        "task": "binary",
        "target_column": config.target,
        "output_kind": "event_probability",
        "positive_label": POSITIVE_LABEL_VALUE,
        "input_columns": config.features,
    }


def prepare(config: DatasetConfig) -> None:
    """Write upload CSVs (no id column) and the manifest for prep_run."""
    directory = uploads_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"prediction": prediction_declaration(config), "files": {}}
    for size in config.sample_sizes:
        sample = pd.read_csv(config.samples_directory / f"train_{size}.csv")
        upload_path = directory / f"train_{size}.csv"
        sample[[*config.features, config.target]].to_csv(upload_path, index=False)
        manifest["files"][str(size)] = {
            "path": upload_path.relative_to(config.directory).as_posix(),
            "format": "csv",
            "content_length": upload_path.stat().st_size,
            "content_digest": f"sha256:{file_sha256(upload_path)}",
            "idempotency_key": f"{config.name}-benchmark-train-{size}-v1",
        }
        print(f"  train_{size}.csv  {upload_path.stat().st_size:>10d} bytes  sha256:{file_sha256(upload_path)}")
    manifest_path(config).write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {manifest_path(config)}")


def upload(config: DatasetConfig, size: int, url: str) -> None:
    """PUT one prepared CSV to a signed upload URL from prep_run."""
    if not url.startswith("https://files.19th.com/"):
        raise ValueError("Upload URL must be a files.19th.com signed URL returned by prep_run")
    upload_path = uploads_directory(config) / f"train_{size}.csv"
    body = upload_path.read_bytes()
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Goog-Content-Length-Range": f"{len(body)},{len(body)}",
        },
    )
    with urllib.request.urlopen(request) as response:
        print(f"  train_{size}.csv uploaded: HTTP {response.status}")


def unpack_verified_bundle(
    config: DatasetConfig, size: int, mode: str, variant: str, bundle_zip: Path, expected_sha256: str
) -> Path:
    """Check the bundle checksum and code version, then unpack it under benchmark/<dataset>/models."""
    actual_sha256 = file_sha256(bundle_zip)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Bundle checksum {actual_sha256} does not match get_results' {expected_sha256}")
    bundle_directory = models_directory(config) / f"train_{size}_{mode}_{variant}"
    bundle_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_zip) as archive:
        for member in archive.namelist():
            target = (bundle_directory / member).resolve()
            if bundle_directory.resolve() not in target.parents and target != bundle_directory.resolve():
                raise ValueError(f"Bundle entry escapes its folder: {member}")
        archive.extractall(bundle_directory)
    code_sha256 = bundle_code_sha256(bundle_directory)
    if code_sha256 not in REVIEWED_BUNDLE_CODE_SHA256:
        raise ValueError(
            f"Bundle prediction code {code_sha256} is not a reviewed version. Read "
            f"{bundle_directory}/predict.py and runtime/ before adding its hash to REVIEWED_BUNDLE_CODE_SHA256."
        )
    return bundle_directory


def run_bundle(config: DatasetConfig, bundle_directory: Path, frame: pd.DataFrame, name: str) -> np.ndarray:
    """Run the bundle's own predict.py on rows and return event probabilities in row order."""
    rows = frame[config.features].copy()
    for column in rows.columns:
        if rows[column].dtype == bool:
            rows[column] = rows[column].astype(int)
    rows_path = bundle_directory / f"{name}_rows.json"
    rows_path.write_text(rows.to_json(orient="records"))
    # -I isolates the interpreter from this project's modules and environment.
    completed = subprocess.run(
        [sys.executable, "-I", str(bundle_directory / "predict.py"), "--format", "npz", str(rows_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    predictions = json.loads(completed.stdout)["predictions"]
    if len(predictions) != len(frame):
        raise ValueError(f"Bundle returned {len(predictions)} predictions for {len(frame)} rows")
    return np.array([row["event_probability"] for row in predictions], dtype=float)


def predict(config: DatasetConfig, size: int, mode: str, variant: str, bundle_zip: Path, expected_sha256: str) -> None:
    """Predict the holdout and save benchmark/<dataset>/predictions/pred_<size>_<mode>_<variant>.csv."""
    bundle_directory = unpack_verified_bundle(config, size, mode, variant, bundle_zip, expected_sha256)
    holdout = load_holdout(config)
    probabilities = run_bundle(config, bundle_directory, holdout, "holdout")
    config.predictions_directory.mkdir(parents=True, exist_ok=True)
    output_path = config.service_predictions_path(size, mode, variant)
    pd.DataFrame({"id": holdout["id"], "probability": probabilities}).to_csv(output_path, index=False)
    print(f"  Saved {output_path}")


def predict_test(
    config: DatasetConfig, size: int, mode: str, variant: str, bundle_zip: Path, expected_sha256: str
) -> None:
    """Predict the competition test set and write a Kaggle submission file."""
    bundle_directory = unpack_verified_bundle(config, size, mode, variant, bundle_zip, expected_sha256)
    test = pd.read_csv(config.data_directory / "test.csv")
    probabilities = run_bundle(config, bundle_directory, test, "test")
    submissions_directory = config.directory / "submissions"
    submissions_directory.mkdir(parents=True, exist_ok=True)
    output_path = submissions_directory / f"submission_19th_{size}_{mode}_{variant}.csv"
    pd.DataFrame({"id": test["id"], config.target: probabilities}).to_csv(output_path, index=False)
    print(f"  Saved {output_path} ({len(test)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    upload_parser = subparsers.add_parser("upload")
    upload_parser.add_argument("--size", type=int, required=True)
    upload_parser.add_argument("--url", required=True)
    for command in ["predict", "predict-test"]:
        predict_parser = subparsers.add_parser(command)
        predict_parser.add_argument("--size", type=int, required=True)
        predict_parser.add_argument("--bundle", type=Path, required=True)
        predict_parser.add_argument("--expected-sha256", required=True)
        predict_parser.add_argument("--mode", choices=TRAINING_MODES, default=DEFAULT_MODE)
        predict_parser.add_argument("--variant", choices=MODEL_VARIANTS, default=DEFAULT_VARIANT)
    arguments = parser.parse_args()
    config = DATASETS[arguments.dataset]
    if arguments.command in ["predict", "predict-test"]:
        bundle_arguments = (
            arguments.size,
            arguments.mode,
            arguments.variant,
            arguments.bundle,
            arguments.expected_sha256,
        )

    if arguments.command == "prepare":
        prepare(config)
    elif arguments.command == "upload":
        upload(config, arguments.size, arguments.url)
    elif arguments.command == "predict":
        predict(config, *bundle_arguments)
    elif arguments.command == "predict-test":
        predict_test(config, *bundle_arguments)


if __name__ == "__main__":
    main()
