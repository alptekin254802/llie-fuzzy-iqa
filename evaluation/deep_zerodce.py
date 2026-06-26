"""
Step 2a - Zero-DCE (deep low-light enhancement) end-to-end.

Place at: evaluation/deep_zerodce.py     Run: python -m evaluation.deep_zerodce

Runs the Zero-DCE curve-estimation network on the LOL low-light images, saves the
enhanced outputs (so you can eyeball them), and computes the FULL column set used
elsewhere so the deep method drops straight into the existing analysis:
  psnr, ssim, entropy            (vs the paired reference, via evaluation.metrics)
  lpips, niqe, brisque           (pyiqa)
  musiq, maniqa, clipiqa         (pyiqa, modern NR-IQA)
  brightness, contrast, colorfulness, sharpness, noise   (no-reference features)

Output: results/deep_zerodce_all.csv   (method = "zerodce")
        results/enhanced_deep/zerodce/<filename>.png  (saved enhanced images)

WEIGHTS (one-time download, ~few hundred KB):
  Get "Epoch99.pth" from the official Zero-DCE repo
    https://github.com/Li-Chongyi/Zero-DCE  ->  Zero-DCE_code/snapshots/Epoch99.pth
  and place it at:  models/zerodce_Epoch99.pth   (under the project root)
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
CLASSICAL_CSV = BASE / "results" / "enhancement_metrics.csv"   # for the filename list
WEIGHTS = BASE / "models" / "zerodce_Epoch99.pth"
SAVE_DIR = BASE / "results" / "enhanced_deep" / "zerodce"
OUTPUT_CSV = BASE / "results" / "deep_zerodce_all.csv"
METHOD = "zerodce"
NR_METRICS = ["niqe", "brisque", "musiq", "maniqa", "clipiqa"]


# ----------------------------- Zero-DCE (DCE-Net) -----------------------------
class DCENet(nn.Module):
    def __init__(self, nf: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r = torch.split(x_r, 3, dim=1)
        for i in range(8):
            x = x + r[i] * (torch.pow(x, 2) - x)
        return x


# ----------------------------- helpers ---------------------------------------
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
    """Same no-reference features as evaluation/nr_features.py (for consistency)."""
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
        sys.exit(f"Zero-DCE weights not found at {WEIGHTS}\n"
                 f"Download Epoch99.pth from https://github.com/Li-Chongyi/Zero-DCE "
                 f"(Zero-DCE_code/snapshots/Epoch99.pth) and place it there.")
    if not CLASSICAL_CSV.exists():
        sys.exit(f"Not found: {CLASSICAL_CSV}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    net = DCENet().to(device).eval()
    state = torch.load(str(WEIGHTS), map_location=device)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state)
    print("Zero-DCE weights loaded.")

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
        with torch.no_grad():
            enh_t = net(to_tensor(low, device)).clamp(0, 1)
        enh = enh_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

        # save enhanced image for visual inspection
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

    # quick look: how does Zero-DCE compare on the key metrics (means)?
    print("\nZero-DCE means:")
    for c in ["psnr", "ssim", "lpips", "niqe", "musiq", "maniqa", "brightness", "contrast", "entropy", "sharpness"]:
        if c in out:
            print(f"  {c:12s}: {out[c].mean():.4g}")
    print("\n[Next: RetinexNet, then we concatenate classical + deep and re-run the analysis.]")


if __name__ == "__main__":
    main()
