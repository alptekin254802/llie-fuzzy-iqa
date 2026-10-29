# Lightweight Real-Time Reference-Free Quality Assessment of Low-Light Image Enhancement

Computational companion to **Lightweight Real-Time Reference-Free Quality
Assessment of Low-Light Image Enhancement via Calibrated Fuzzy Inference**,
under review at *Journal of Real-Time Image Processing*.

The main NR-free assessor maps entropy, normalized contrast and normalized
Laplacian variance to an LPIPS-calibrated score. It is a zero-order Sugeno model
with 27 singleton consequents, minimum rule firing and a normalized weighted
average. Calibration uses references offline; scoring uses only the enhanced
image. Agreement with LPIPS is distinct from agreement with human observers.

NR-mono is a separate variant: product rule firing, a Ruspini membership
partition and ordered consequents fitted by constrained least squares. Its
response is nondecreasing in the oriented features; the Laplacian feature is
sign-reversed. This response constraint does not imply perceptual validity.

## Setup

Use Python 3.11 in a virtual environment. The pinned core dependencies in
`requirements-reproduction.txt` support fitting, plotting and the tests. Run the
commands below from the repository root. Frozen inference needs NumPy and
OpenCV; the fitting and validation commands also use SciPy and pandas. Deep
metrics require the listed PyTorch and pyiqa dependencies and their weights.

```bash
git clone https://github.com/alptekin254802/llie-fuzzy-iqa.git
cd llie-fuzzy-iqa
python -m venv .venv
# Activate: .venv\Scripts\Activate.ps1 (PowerShell), or source .venv/bin/activate (bash)
python -m pip install -r requirements-reproduction.txt
python -B -m unittest discover -s tests -v
```

## Score an enhanced image

```python
import cv2
from evaluation.three_features import extract
from evaluation.frozen_scorer import FrozenScorer

model = FrozenScorer("nr_free")  # or "nr_mono"; initialize once
bgr = cv2.imread("enhanced.png")
if bgr is None:
    raise FileNotFoundError("enhanced.png")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
score = model(extract(rgb))
```

For an optional NR-free display label, use `quality_label(score)` from
`evaluation.quality_labels`. The maximum-membership band is chosen from Poor
(0, 0, 20, 40), Fair (20, 40, 50, 65), Good (50, 65, 75, 90), and Excellent
(75, 90, 100, 100); exact ties select the lower band. These are display categories,
not validated human quality judgments. They do not alter the numeric score and
are not applicable directly to the raw negative-LPIPS output of NR-mono.

`models/nr_free.json` and `models/nr_mono.json` contain the full-precision
deployment fits, feature orientations, membership breakpoints and rule order.
The CSV field `sharpness` means Laplacian variance divided by 255 squared.
It is an image statistic, not a direct perceptual sharpness measurement.
NR-free returns its calibrated 0–100 score. NR-mono returns the fitted negative-LPIPS
scale; its rule figures use an affine 0–100 display scale. Deployment models use all calibration scenes; held-out evaluation uses separate
fold fits. NR-free's reported primary protocol uses global membership
breakpoints and orientations, with train-fold consequents. NR-free reaches 0.753 under this protocol and 0.754 when all preprocessing
is fitted within training folds. NR-mono's fold evaluation uses
train-fold breakpoints and the LOL feature orientations. The main NR-mono fold mean is 0.751; full-precision parameters, predictions and near-tie checks are recorded in [the validation results](results/monotone_validation.json).

The additional Q-Align comparison reports fp16 accuracy and 4-bit GPU timing
(1.18 s/image on a GTX 1060). Archived fp16 predictions, paired scene-bootstrap
results and portable `run.py` / `analyze.py` scripts have a separate
[reproduction guide](docs/QALIGN.md); the original 4-bit scores remain the timing source.

The [webcam search and timing guide](realtime/README.md) covers live processing,
an offline selftest and recomputation of the archived frame timings.

## Data and preprocessing

Download LOL from the [authors' project](https://daooshee.github.io/BMVC2018website/).
The calibration split contains 485 paired scenes under `our485/low` and
`our485/high`. Recreate the stored PNG inputs with:

```bash
python -m preprocessing.preprocess --lol-root /path/to/our485 --output preprocessed
```

This uses RGB conversion, **bilinear** resizing to 256 x 256, float32 division
by 255, and uint8 serialization. Enhancement follows resizing. Human-rated
enhanced images use **area** resizing to the same operating resolution.
The preprocessing command requires matching paired filenames and an empty
output directory. Feature and perceptual-metric extraction entry points are
under `evaluation/`; classical enhancement operators are under
`preprocessing/enhancement/`.

RLIE images and MOS labels are available from the
[RLIE authors](https://github.com/CQUPT-HuBo90/RLIE).
Use `python -m analysis.rlie_human_mos --rlie_root /path/to/RLIE`.
Its feature cache is resolution-specific (`rlie_features_area256.csv`).
Six-feature MOS calibration is run with
`python -m analysis.rlie_enriched --rlie_root /path/to/RLIE`.
Unpaired cross-dataset scripts describe the DICM, LIME, MEF, NPE and VV layouts
in `evaluation/cross_dataset.py`.

## Reproduce reported computations

Included derived CSVs support the main calibration and figure commands without
raw-image downloads. Model fitting uses scene-disjoint five-fold splits.

| Computation | Command / output |
|---|---|
| Primary figures and fold-mean table | `python -m analysis.make_figures`; `results/figures_numbers.csv` |
| NR-mono fold validation and near-tie sensitivity | `python -m analysis.validate_monotone`; `results/monotone_validation.json` |
| NR-free and NR-mono calibration | `python -m analysis.reference_free_fuzzy --output calibration_run` |
| Within-method figure | `analysis.make_figures`: classical out-of-fold predictions; Zero-DCE and SCI scored by the unchanged classical deployment fit |
| NR-mono rule heatmap | `analysis.make_figures`: frozen product-model consequents, affinely displayed on a relative 0–100 scale |
| Scene-clustered paired statistics | `python -m analysis.significance_tests`; `results/significance_tests.json` |
| Fusion / method-disjoint / strict preprocessing controls | `python -m analysis.reviewer_experiments` |
| Feature preprocessing sensitivity | `python -m analysis.preprocessing_sensitivity` |
| Supplementary membership functions | `python -m analysis.make_extra_figures` |
| Pipeline panels from the same recorded run | `python -m analysis.plot_pipeline` |

The significance analysis compares fixed out-of-fold predictions with eight
metrics/baselines. It resamples 485 scenes with replacement (2,000 draws),
retains all outputs of each selected scene, and reports percentile confidence
intervals, null-centered bootstrap p-values and Holm adjustment. Pooled SRCC
and mean fold SRCC are separate estimands. The intervals condition on the
fitted models; the bootstrap does not refit them.

The [final qualitative-figure guide](docs/FIGURES.md) reproduces the plate-rack example and the rule-selected LOL/SQUARE-LOL panels.

## Additional validation studies

[Study commands, data layouts and numerical conventions](docs/STUDIES.md) cover
the regional, optimization, native-resolution, synthetic closed-loop, human-rated
and additional-metric experiments. Derived recorded summaries are under
`results/studies/`. Use a separate configured workspace to reproduce them.

[VQA² and VITAL results](results/lmm_extra/) include the released-model scores and paired comparisons.
[Exploratory real-camera check](realtime/camera_exploratory/) provides the protocol and anonymized derived tables.

## Timing and video demonstration

```bash
python -m analysis.benchmark_runtime --images /path/to/our485/low --sessions 3 --deep
python -m analysis.realtime_analysis
python -m analysis.realtime_pipeline_demo --source video.mp4 --enhancer zerodce --device cuda --frames 1200 --plot
```

Timing measures the actual frozen singleton scorers. It records PNG/JPEG reads,
color conversion, resize, feature extraction, inference, complete in-memory
paths and disk-inclusive paths separately, with raw samples in JSON. Three
fresh processes form three sessions. Model loading and warmup are excluded.
CLIP-IQA+ and MUSIQ are evaluated on CPU with one thread when `--deep` is set.
Input sizes from 256 x 256 to 1920 x 1080 are controlled resamplings of the same
source images. Reads use a warm filesystem cache. For comparable measurements,
use the same power profile and an otherwise idle machine; each run records its
environment. New measurements are written to `results/timing_run.json`;
the recorded campaign is `results/timing_benchmark.json`.

Latency and speed ratios are machine- and protocol-specific. The 95th percentile
is not a worst-case guarantee. A single score, capture-enhance-assess throughput,
and a search over multiple enhancement candidates measure different workloads.
The video score trace is a model output, without framewise human-quality labels.

For Zero-DCE, obtain `Epoch99.pth` from the
[Zero-DCE authors](https://github.com/Li-Chongyi/Zero-DCE) and use
`--zerodce-weights /path/to/Epoch99.pth`.

## Additional comparisons

The [training-method coverage control](results/studies/method_coverage/README.md) tests human-calibrated transfer with two to six training methods.

The [calibration-source overlap control](results/studies/targeted_overlap/README.md) removes shared calibration scenes and compares the effect with 20 random-removal fits.

The [exploratory unseen-enhancement check](results/studies/unseen_operators/README.md) includes eight additional methods, per-scene records and reproducible confidence intervals.

The [RichIQA guide](docs/RICHIQA.md) covers the released author checkpoint,
LOL and human-dataset scores, CPU/GPU timing and paired scene-bootstrap contrasts.

The four additional LOL scorers have both negative-LPIPS and negative-NIQE
correlations in `results/studies/extra_metrics_lol_anchors.json`. Reproduce them
from archived scores with `python -m analysis.complete_lol_correlations`. This
uses the existing five scene folds and population SD, negating raw BPRI/BMPRI
scores. It does not run inference or fit any models.

## Citation and license

Please cite the manuscript title and repository while the paper is under review.
The code license is in `LICENSE`; retained SQUARE-LOL scores carry the attribution and MIT permission in [NOTICE](NOTICE). LIEQ and RLIE image-level labels are obtained separately; [local label joining and reproduction](results/lmm_extra/README.md#human-labels-and-reproduction) are documented. Image datasets and pretrained weights are
obtained separately under the terms of their respective providers.

Full protocols, implementation details and supplementary caveats are collected in [Technical notes](docs/TECHNICAL_NOTES.md).
