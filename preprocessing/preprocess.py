"""Reproduce the LOL preprocessing: RGB, bilinear 256 x 256, float32 / 255."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def preprocess_image(path, size=(256, 256)):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.clip(cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0, 0, 1)


def preprocess_dataset(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Input and output directories must be separate trees")
    pairs = {part: sorted(p for p in (root / part).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
             for part in ("low", "high")}
    if not pairs["low"] or [p.name for p in pairs["low"]] != [p.name for p in pairs["high"]]:
        raise ValueError("Expected nonempty low/high directories with identical image filenames")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    counts = {}
    for part, paths in pairs.items():
        (output / part).mkdir(parents=True, exist_ok=True)
        for path in paths:
            rgb = np.clip(preprocess_image(path) * 255.0, 0, 255).astype(np.uint8)
            if not cv2.imwrite(str(output / part / path.name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"Cannot save {path.name}")
        counts[part] = len(paths)
    manifest = {"resolution": [256, 256], "interpolation": "INTER_LINEAR", "counts": counts,
                "filenames": [p.name for p in pairs["low"]]}
    (output / "preprocessing.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lol-root", type=Path, required=True, help="Directory containing LOL low/ and high/")
    ap.add_argument("--output", type=Path, default=Path("preprocessed"))
    args = ap.parse_args()
    print(json.dumps(preprocess_dataset(args.lol_root, args.output), indent=2))


if __name__ == "__main__":
    main()
