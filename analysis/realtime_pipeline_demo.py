"""
realtime_pipeline_demo.py
=========================
Live capture -> enhance -> assess pipeline for the reference-free fuzzy NR-IQA
method. Measures sustained fps and per-stage latency, and emits the per-frame
perceptual score.

IMPORTANT - which assessor produces the SCORE
---------------------------------------------
  --assessor real         (DEFAULT) uses the calibrated reference-free fuzzy
                          system (the LPIPS-DE-calibrated rule base shown in
                          Fig. 5). This is the ONLY correct choice when the
                          per-frame SCORE is reported/plotted (Fig. 6b): the
                          scores must come from the calibrated consequents.
  --assessor placeholder  a structurally identical but PLACEHOLDER fuzzy
                          (random consequents). Use ONLY for pure timing tests
                          when the calibrated module is unavailable; its SCORES
                          are NOT meaningful and must never be plotted/reported.

Timing (fps, latency) is representative either way because the cost depends on the
computation structure, not on the consequent values. Scores are only correct with
--assessor real.

Sources:   --source 0 | video.mp4 | frames/ | synth
Enhancer:  --enhancer gamma | clahe | none | zerodce  (+ --zerodce-weights, --device)

Examples
--------
  # paper headline (calibrated scores + real timing, Zero-DCE on GPU):
  python analysis/realtime_pipeline_demo.py --source dark.mp4 --enhancer zerodce \
        --device cuda --assessor real --seconds 20 --plot

  # pure timing sanity check, no calibrated module needed:
  python analysis/realtime_pipeline_demo.py --source synth --assessor placeholder --frames 600

Outputs: prints a summary and writes results/pipeline_log.csv. With --plot it
writes fig6_pipeline_score.{png,pdf,svg} and fig6_pipeline_fps.{png,pdf,svg} to
results/figures/.
"""
from __future__ import annotations
import argparse, glob, os, sys, time, platform
from pathlib import Path
import numpy as np

_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
_DEFAULT_ZERODCE_WEIGHTS = _BASE / "models" / "zerodce_Epoch99.pth"

try:
    import cv2
except ImportError as e:
    raise SystemExit("OpenCV required: pip install opencv-python") from e


# --------------------------------------------------------------------------- #
#  Feature extraction (same definitions as the paper)
# --------------------------------------------------------------------------- #
def _gray_u8(img):
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(np.uint8)

def extract_features(img):
    g = _gray_u8(img)
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    p = hist / max(hist.sum(), 1.0); p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    contrast = float(g.astype(np.float32).std())
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return entropy, contrast, sharp


# --------------------------------------------------------------------------- #
#  Assessor 1 - REAL calibrated reference-free fuzzy 
# --------------------------------------------------------------------------- #
class RealAssessor:
    """
    Wraps the calibrated reference-free fuzzy system so the per-frame score is
    the genuine LPIPS-DE-calibrated value that matches Fig. 5.
    """
    def __init__(self):
        try:
            from analysis.reference_free_fuzzy import score_features as _sf
            self._score = _sf
        except Exception as ex:
            raise SystemExit(
                "RealAssessor could not import the calibrated scorer.\n"
                "analysis.reference_free_fuzzy.score_features must expose the\n"
                "calibrated 27-rule reference-free fuzzy score used in Fig. 5.\n"
                f"Import error: {ex}\n"
                "For a pure timing test you may use --assessor placeholder instead.")

    def infer(self, e, c, s) -> float:
        return float(self._score(e, c, s))


# --------------------------------------------------------------------------- #
#  Assessor 2 - PLACEHOLDER (timing only; scores NOT meaningful)
# --------------------------------------------------------------------------- #
def trapmf(x, a, b, c, d):
    x = np.asarray(x, float)
    l = np.where(b > a, (x - a) / np.maximum(b - a, 1e-12), 1.0)
    r = np.where(d > c, (d - x) / np.maximum(d - c, 1e-12), 1.0)
    return np.clip(np.minimum(np.minimum(l, 1.0), r), 0.0, 1.0)

class PlaceholderFIS:
    """
    Structurally identical to the paper's FIS (3x3 trapezoidal MFs, 27 rules,
    min-implication, max-aggregation, centroid defuzzification) so TIMING is
    representative. The breakpoints and 27 consequents are PLACEHOLDERS (random),
    so the SCORES it returns are NOT the paper's scores and must never be plotted
    or reported. Use only for --assessor placeholder timing tests.
    """
    def __init__(self, res=101):
        ranges = {"entropy": (0, 8), "contrast": (0, 128), "sharpness": (0, 2000)}
        self.mf = {}
        for n, (lo, hi) in ranges.items():
            s = hi - lo
            q = [lo + s * p for p in (0, .2, .4, .5, .6, .8, 1)]
            self.mf[n] = [(q[0], q[0], q[1], q[2]),
                          (q[1], q[2], q[4], q[5]),
                          (q[4], q[5], q[6], q[6])]
        self.u = np.linspace(0, 100, res)
        outs = [(0, 0, 20, 40), (20, 40, 50, 65), (50, 65, 75, 90), (75, 90, 100, 100)]
        self.out_mf = np.stack([trapmf(self.u, *o) for o in outs])
        self.rule_out = np.random.default_rng(42).integers(0, 4, size=27)  # PLACEHOLDER

    def infer(self, e, c, s):
        me = np.array([trapmf(e, *m) for m in self.mf["entropy"]])
        mc = np.array([trapmf(c, *m) for m in self.mf["contrast"]])
        ms = np.array([trapmf(s, *m) for m in self.mf["sharpness"]])
        fire = np.minimum.reduce(np.broadcast_arrays(
            me[:, None, None], mc[None, :, None], ms[None, None, :])).ravel()
        agg = np.zeros_like(self.u)
        for r in range(27):
            agg = np.maximum(agg, np.minimum(fire[r], self.out_mf[self.rule_out[r]]))
        d = agg.sum()
        return float((self.u * agg).sum() / d) if d > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
#  Enhancers (pluggable)
# --------------------------------------------------------------------------- #
def enhance_gamma(img, gamma=0.5):
    lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(img, lut)

def enhance_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def enhance_none(img):
    return img

class ZeroDCEEnhancer:
    """Load DCE-Net once; per-frame RGB uint8 -> tensor -> enhance -> RGB uint8."""
    def __init__(self, weights_path, device="auto"):
        try:
            import torch
        except ImportError as e:
            raise SystemExit("Zero-DCE requires PyTorch: pip install torch") from e
        from evaluation.deep_zerodce import DCENet
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise SystemExit(
                f"Zero-DCE weights not found at {weights_path}\n"
                "Download Epoch99.pth from https://github.com/Li-Chongyi/Zero-DCE "
                "(Zero-DCE_code/snapshots/Epoch99.pth) and place it there.")
        self._torch = torch; self._device = device
        net = DCENet().to(device).eval()
        state = torch.load(str(weights_path), map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net.load_state_dict(state); self._net = net
        print(f"Zero-DCE loaded: {weights_path}  device={device}")

    def __call__(self, img):
        t = (self._torch.from_numpy(np.ascontiguousarray(img))
             .permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self._device))
        with self._torch.no_grad():
            out = self._net(t).clamp(0.0, 1.0)
        rgb = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

ENHANCERS = {"gamma": enhance_gamma, "clahe": enhance_clahe, "none": enhance_none}


# --------------------------------------------------------------------------- #
#  Frame source
# --------------------------------------------------------------------------- #
class Source:
    def __init__(self, spec, size, synth_n=2000):
        self.size = size; self.cap = None; self.frames = None; self.i = 0
        if spec == "synth":
            self.kind = "synth"; self.n = synth_n; self.rng = np.random.default_rng(0)
        elif os.path.isdir(spec):
            self.kind = "folder"; ps = []
            for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                ps += glob.glob(os.path.join(spec, e))
            self.frames = sorted(ps)
            if not self.frames: raise SystemExit(f"No images in {spec}")
        else:
            self.kind = "cv"
            src = int(spec) if spec.isdigit() else spec
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened(): raise SystemExit(f"Cannot open source {spec}")

    def read(self):
        if self.kind == "synth":
            if self.i >= self.n: return None
            self.i += 1
            return self.rng.integers(0, 90, size=(*self.size, 3), dtype=np.uint8)
        if self.kind == "folder":
            if self.i >= len(self.frames): return None
            im = cv2.imread(self.frames[self.i]); self.i += 1
            return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), self.size[::-1])
        ok, im = self.cap.read()
        if not ok: return None
        return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), self.size[::-1])

    def release(self):
        if self.cap is not None: self.cap.release()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--enhancer", default="gamma", choices=list(ENHANCERS) + ["zerodce"])
    ap.add_argument("--assessor", default="real", choices=["real", "placeholder"],
                    help="real = calibrated fuzzy scorer (correct scores, Fig.5); "
                         "placeholder = timing-only, scores NOT meaningful")
    ap.add_argument("--zerodce-weights", default=str(_DEFAULT_ZERODCE_WEIGHTS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--size", default="256x256")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    try: cv2.setNumThreads(1)
    except Exception: pass

    h, w = (int(x) for x in args.size.lower().split("x"))
    src = Source(args.source, (h, w))
    enh = ZeroDCEEnhancer(args.zerodce_weights, args.device) if args.enhancer == "zerodce" \
        else ENHANCERS[args.enhancer]
    assessor = RealAssessor() if args.assessor == "real" else PlaceholderFIS()
    if args.assessor == "placeholder":
        print("WARNING: --assessor placeholder -> SCORES ARE NOT MEANINGFUL "
              "(timing only). Do not plot/report these scores.")

    f0 = src.read()
    if f0 is None: raise SystemExit("source produced no frames")
    for _ in range(10):
        e, c, s = extract_features(enh(f0)); assessor.infer(e, c, s)

    t_cap, t_enh, t_ass, t_tot, scores, stamps = [], [], [], [], [], []
    t_start = time.perf_counter(); n = 0
    while True:
        if args.seconds is not None:
            if time.perf_counter() - t_start >= args.seconds: break
        elif n >= args.frames: break
        c0 = time.perf_counter()
        frame = src.read()
        if frame is None: break
        c1 = time.perf_counter()
        eimg = enh(frame)
        c2 = time.perf_counter()
        e, c, s = extract_features(eimg); score = assessor.infer(e, c, s)
        c3 = time.perf_counter()
        t_cap.append((c1 - c0) * 1e3); t_enh.append((c2 - c1) * 1e3)
        t_ass.append((c3 - c2) * 1e3); t_tot.append((c3 - c0) * 1e3)
        scores.append(score); stamps.append(c3 - t_start); n += 1
        if args.show:
            disp = cv2.cvtColor(eimg, cv2.COLOR_RGB2BGR)
            cv2.putText(disp, f"Q={score:5.1f}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("capture->enhance->assess", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    src.release()
    if args.show: cv2.destroyAllWindows()

    import statistics as st
    def med(a): return st.median(a) if a else 0.0
    sustained = n / (sum(t_tot) / 1e3) if sum(t_tot) > 0 else 0.0
    print("=" * 60)
    print("Real-time capture -> enhance -> assess pipeline")
    print("=" * 60)
    print(f"Machine  : {platform.processor() or platform.machine()}")
    print(f"Source   : {args.source}  Enhancer: {args.enhancer}  Assessor: {args.assessor}  Size: {h}x{w}")
    print(f"Frames   : {n}")
    print("-" * 60)
    print(f"capture  : {med(t_cap):7.3f} ms")
    print(f"enhance  : {med(t_enh):7.3f} ms")
    print(f"assess   : {med(t_ass):7.3f} ms   <-- proposed method")
    print(f"total    : {med(t_tot):7.3f} ms")
    print("-" * 60)
    print(f">> sustained pipeline fps : {sustained:6.1f}")
    print(f">> assess median ms       : {med(t_ass):6.3f}")
    if t_tot:
        print(f">> min instantaneous fps  : {1000.0 / max(t_tot):6.1f}")

    import csv
    out_dir = _BASE / "results"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline_log.csv"
    with open(log_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["frame", "t_s", "cap_ms", "enh_ms", "assess_ms", "total_ms", "score"])
        for i in range(n):
            wtr.writerow([i, f"{stamps[i]:.4f}", f"{t_cap[i]:.4f}", f"{t_enh[i]:.4f}",
                          f"{t_ass[i]:.4f}", f"{t_tot[i]:.4f}", f"{scores[i]:.4f}"])
    print(f"\nSaved: {log_path}")

    if args.plot:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if args.assessor == "real":
                plt.figure(figsize=(6, 2.6)); plt.plot(stamps, scores, lw=1)
                plt.xlabel("time (s)"); plt.ylabel("perceptual score")
                plt.title(f"Per-frame quality (zerodce, {sustained:.0f} fps)")
                plt.tight_layout()
                for ext in ("png", "pdf", "svg"):
                    plt.savefig(fig_dir / f"fig6_pipeline_score.{ext}", dpi=150 if ext == "png" else None)
                plt.close()
            else:
                print("[plot] score plot skipped: placeholder scores are not meaningful")
            inst = [1000.0 / t if t > 0 else 0 for t in t_tot]
            plt.figure(figsize=(6, 2.6)); plt.plot(stamps, inst, lw=1)
            plt.axhline(30, ls="--", c="r", lw=0.8, label="30 fps")
            plt.xlabel("time (s)"); plt.ylabel("instantaneous fps"); plt.legend()
            plt.tight_layout()
            for ext in ("png", "pdf", "svg"):
                plt.savefig(fig_dir / f"fig6_pipeline_fps.{ext}", dpi=150 if ext == "png" else None)
            plt.close()
            print(f"Saved figures to {fig_dir}.")
        except Exception as ex:
            print(f"[plot] skipped: {ex}")


if __name__ == "__main__":
    main()
