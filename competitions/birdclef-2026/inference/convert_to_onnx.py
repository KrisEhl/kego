"""Convert soundscape-v7 PyTorch checkpoints to ONNX format.

onnxruntime is pre-installed on Kaggle CPU notebooks — no wheel bundling needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model definition (must match train.py exactly) ───────────────────────────


class GEMFreqPool(nn.Module):
    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        return torch.sigmoid((att * cls).sum(dim=-1))


class BirdModelBaseline(nn.Module):
    def __init__(
        self,
        backbone: str,
        n_classes: int,
        pretrained: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 64, 128)
            feat_dim = self.encoder(_dummy).shape[1]
        self.gem_pool = GEMFreqPool(p_init=3.0)
        self.head = AttentionSEDHead(feat_dim, n_classes, dropout)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.gem_pool(feat)
        return self.head(feat)


# ── Conversion ───────────────────────────────────────────────────────────────

CKPT_DIR = Path("/tmp/birdclef-v7-onnx")
OUT_DIR = Path("/tmp/birdclef-v7-onnx")
OUT_DIR.mkdir(exist_ok=True)

ckpt_paths = sorted(CKPT_DIR.glob("soundscape-v7_fold*.pt"))
if not ckpt_paths:
    print("ERROR: no soundscape-v7 checkpoints found")
    sys.exit(1)

print(f"Found {len(ckpt_paths)} checkpoints")

for ckpt_path in ckpt_paths:
    print(f"\nConverting {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt.get("backbone", "tf_efficientnet_b0.ns_jft_in1k")
    n_mels = ckpt.get("n_mels", 224)
    n_fft = ckpt.get("n_fft", 2048)
    hop = ckpt.get("hop_length", 512)
    fmin = ckpt.get("fmin", 0)
    htk = ckpt.get("htk", True)
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))

    print(f"  backbone={backbone}, n_mels={n_mels}, hop={hop}, fmin={fmin}, htk={htk}")
    print(f"  epoch={epoch}, val_loss={val_loss:.4f}")

    n_classes = ckpt["model"]["head.cls_conv.bias"].shape[0]
    time_frames = 32000 * 5 // hop + 1
    example = torch.zeros(1, 3, n_mels, time_frames)

    model = BirdModelBaseline(backbone=backbone, n_classes=n_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Verify PyTorch output shape
    with torch.no_grad():
        out_pt = model(example)
    print(
        f"  PyTorch output: {out_pt.shape}, range [{out_pt.min():.3f}, {out_pt.max():.3f}]"
    )

    stem = ckpt_path.stem  # e.g. soundscape-v7_fold0
    out_onnx = OUT_DIR / f"{stem}.onnx"

    torch.onnx.export(
        model,
        example,
        str(out_onnx),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"  Saved: {out_onnx}  ({out_onnx.stat().st_size / 1e6:.1f} MB)")

    # Save mel params sidecar
    meta = {
        "backbone": backbone,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop,
        "fmin": fmin,
        "htk": htk,
        "norm_type": "minmax",
        "n_classes": n_classes,
        "epoch": epoch if epoch != "?" else -1,
        "val_loss": float(val_loss),
    }
    meta_path = OUT_DIR / f"{stem}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Verify ONNX output matches PyTorch
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(out_onnx), sess_opts, providers=["CPUExecutionProvider"]
    )
    out_onnx_val = session.run(None, {"input": example.numpy()})[0]
    max_diff = np.abs(out_pt.numpy() - out_onnx_val).max()
    print(
        f"  Max diff PyTorch vs ONNX: {max_diff:.2e} {'✓' if max_diff < 1e-4 else 'WARNING'}"
    )

# Write dataset-metadata.json
meta_ds = {
    "title": "BirdCLEF 2026 Soundscape-v7 ONNX",
    "id": "aldisued/birdclef2026-soundscape-v7-onnx",
    "licenses": [{"name": "CC0-1.0"}],
}
with open(OUT_DIR / "dataset-metadata.json", "w") as f:
    json.dump(meta_ds, f, indent=2)

print(f"\nAll done. Files in {OUT_DIR}:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name:50s} {p.stat().st_size / 1e6:.1f} MB")
