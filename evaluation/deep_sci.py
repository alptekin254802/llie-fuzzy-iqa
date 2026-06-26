"""
Step 2b - SCI (Self-Calibrated Illumination, CVPR 2022) end-to-end.

Place at: evaluation/deep_sci.py        Run: python -m evaluation.deep_sci

Second modern deep enhancer (Retinex self-calibration paradigm, complements
Zero-DCE's curve estimation). Same one-script pattern as deep_zerodce.py: runs SCI
on the LOL low-light images, saves enhanced outputs, and computes the FULL column
set so it drops straight into the existing analysis.

Output: results/deep_sci_all.csv   (method = "sci")
        results/enhanced_deep/sci/<filename>.png

WEIGHTS (bundled in the official repo, tiny):
  From https://github.com/vis-opt-group/SCI  ->  CVPR/weights/medium.pt
  place it at:  models/sci_medium.pt   (under the project root)
  (If outputs look too dark/bright when you eyeball them, try easy.pt or
   difficult.pt from the same folder and update WEIGHT_NAME below.)

Architecture (EnhanceNetwork) and input convention (RGB [0,1], no resize) are
reproduced verbatim from the official model.py so the weights load exactly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import pyiqa
from evaluation.metrics import compute_psnr, compute_ssim, compute_entropy

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
LOW_DIR = BASE / "preprocessed" / "low"
HIGH_DIR = BASE / "preprocessed" / "high"
CLASSICAL_CSV = BASE / "results" / "enhancement_metrics.csv"
WEIGHTS = BASE / "models" / "sci_medium.pt"
SAVE_DIR = BASE / "results" / "enhanced_deep" / "sci"
OUTPUT_CSV = BASE / "results" / "deep_sci_all.csv"
METHOD = "sci"
NR_METRICS = ["niqe", "brisque", "musiq", "maniqa", "clipiqa"]


# ---------- SCI EnhanceNetwork (verbatim from official model.py) -------------
class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super().__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.in_conv = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size, stride=1, padding=padding),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(self.conv)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, 3, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        illu = fea + x
        illu = torch.clamp(illu, 0.0001, 1)
        return illu


def load_sci(weights: Path, device):
    net = EnhanceNetwork(layers=1, channels=3).to(device).eval()
    sd = torch.load(str(weights), map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    if any(k.startswith("enhance.") for k in sd):
        sd = {k[len("enhance."):]: v for k, v in sd.items() if k.startswith("enhance.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing}")
    return net


# ----------------------------- helpers (same as Zero-DCE script) -------------
def resolve(directory: Path, fname: str) -> Path:
    p = directory / fname
    if p.exists():
        return p
    stem = Path(fname).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        if (directory / f"{stem}{ext}").exists():
            return directory / f"{stem}{ext}"
    raise FileNotFoundError(f"{fname} not in {directory}")


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_tensor(rgb01: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1).unsqueeze(0).float().clamp(0, 1).to(device)


def nr_features(rgb01: np.ndarray) -> dict:
    rgb8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY).astype(np.float64)
    R, G, B = rgb8[..., 0].astype(np.float64), rgb8[..., 1].astype(np.float64), rgb8[..., 2].astype(np.float64)
    rg, yb = R - G, 0.5 * (R + G) - B
    Kmask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray, cv2.CV_64F, Kmask, borderType=cv2.BORDER_REPLICATE)
    H, W = gray.shape
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "colorfulness": float((np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)) / 255.0),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var() / (255.0 ** 2)),
        "noise": float(np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6.0 * max(W - 2, 1) * max(H - 2, 1)) / 255.0),
    }


def main() -> None:
    if not WEIGHTS.exists():
        sys.exit(f"SCI weights not found at {WEIGHTS}\n"
                 f"Download CVPR/weights/medium.pt from https://github.com/vis-opt-group/SCI "
                 f"and place it there as sci_medium.pt.")
    if not CLASSICAL_CSV.exists():
        sys.exit(f"Not found: {CLASSICAL_CSV}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    net = load_sci(WEIGHTS, device)
    print("SCI weights loaded.")

    lpips_m = pyiqa.create_metric("lpips", device=device)
    nr_models = {}
    for name in NR_METRICS:
        try:
            nr_models[name] = pyiqa.create_metric(name, device=device)
        except Exception as e:
            print(f"  [skip] {name}: {e}")

    filenames = sorted(pd.read_csv(CLASSICAL_CSV)["filename"].unique())
    n = len(filenames)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Images: {n}")

    rows = []
    for i, fname in enumerate(filenames, 1):
        low = load_rgb(resolve(LOW_DIR, fname))
        high = load_rgb(resolve(HIGH_DIR, fname))
        x = to_tensor(low, device)
        with torch.no_grad():
            illu = net(x)
            enh_t = torch.clamp(x / illu, 0, 1)
        enh = enh_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

        cv2.imwrite(str(SAVE_DIR / f"{Path(fname).stem}.png"),
                    cv2.cvtColor((enh * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        rec = {"filename": fname, "method": METHOD,
               "psnr": compute_psnr(enh, high),
               "ssim": compute_ssim(enh, high),
               "entropy": compute_entropy(enh)}
        high_t = to_tensor(high, device)
        with torch.no_grad():
            try:
                rec["lpips"] = float(lpips_m(enh_t, high_t).item())
            except Exception:
                rec["lpips"] = np.nan
            for name, m in nr_models.items():
                try:
                    rec[name] = float(m(enh_t).item())
                except Exception:
                    rec[name] = np.nan
        rec.update(nr_features(enh))
        rows.append(rec)
        if i % 25 == 0 or i == n:
            print(f"  {i}/{n} images done")

    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({len(out)} rows)")
    print(f"Enhanced images: {SAVE_DIR}")
    print("\nSCI means:")
    for c in ["psnr", "ssim", "lpips", "niqe", "musiq", "maniqa", "brightness", "contrast", "entropy", "sharpness"]:
        if c in out:
            print(f"  {c:12s}: {out[c].mean():.4g}")
    print("\n[Next: concatenate classical + zerodce + sci and re-run the full analysis.]")


if __name__ == "__main__":
    main()
