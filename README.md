# Lightweight Real-Time Reference-Free Quality Assessment of Low-Light Image Enhancement

Code and computed result tables for the paper:

> "Lightweight Real-Time Reference-Free Quality Assessment of Low-Light Image Enhancement via Calibrated Fuzzy Inference"  
> Journal of Real-Time Image Processing (Springer), under review.

The method is a lightweight, interpretable, reference-free image quality assessor
for low-light image enhancement (LLIE). It uses three no-reference features of
the enhanced image, entropy, contrast, and a sharpness indicator, and maps them to
a perceptual quality score with a calibrated 27-rule Mamdani fuzzy inference
system. The rule consequents are calibrated offline against LPIPS; inference uses
only the enhanced image.

## Repository Layout

```text
llie-fuzzy-nr-iqa/
|-- analysis/                  # calibration, validation, figures, runtime, realtime scripts
|-- evaluation/                # optional raw-image metric/feature extraction scripts
|-- preprocessing/             # classical LLIE operators used by extractors
|-- results/                   # computed CSV tables and summaries
|   `-- figures/               # paper-ready PDF/SVG figures
|-- models/                    # optional pretrained weights, not tracked
|-- requirements.txt
|-- LICENSE        # MIT
└-- README.md
```

The `results/` CSV files are included so the paper tables and figures can be
reproduced without downloading image datasets or rerunning slow perceptual metric
extraction. Raw images, demo videos, enhanced-image caches, and model weights are
not included.

## Installation

```bash
git clone https://github.com/alptekin254802/llie-fuzzy-nr-iqa.git
cd llie-fuzzy-nr-iqa
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.11. Commands below are run from the repository root.

## Datasets (only needed to regenerate the CSVs from raw images)

The provided CSVs already contain the per-image features and metrics, so most
results reproduce with **no downloads**. To rebuild the CSVs from raw images, obtain
the datasets below from their original sources. Please cite the corresponding papers
if you use them.

**Paired benchmark (calibration + main study)**

- **LOL** — Wei, C., Wang, W., Yang, W., Liu, J. "Deep Retinex Decomposition for
  Low-Light Enhancement." *BMVC* (2018). arXiv:1808.04560.
  Project page: https://daooshee.github.io/BMVC2018website/

**Unpaired real-world test sets (cross-dataset transfer, Sec. 4.8 / Table S1)**

- **DICM** — Lee, C., Lee, C., Kim, C.-S. "Contrast enhancement based on layered
  difference representation of 2D histograms." *IEEE TIP* 22(12), 5372–5384 (2013).
  https://doi.org/10.1109/TIP.2013.2284059
- **LIME** — Guo, X., Li, Y., Ling, H. "LIME: Low-Light Image Enhancement via
  Illumination Map Estimation." *IEEE TIP* 26(2), 982–993 (2017).
  https://doi.org/10.1109/TIP.2016.2639450
- **MEF** — Ma, K., Zeng, K., Wang, Z. "Perceptual Quality Assessment for
  Multi-Exposure Image Fusion." *IEEE TIP* 24(11), 3345–3356 (2015).
  https://doi.org/10.1109/TIP.2015.2442920
- **NPE** — Wang, S., Zheng, J., Hu, H.-M., Li, B. "Naturalness Preserved Enhancement
  Algorithm for Non-Uniform Illumination Images." *IEEE TIP* 22(9), 3538–3548 (2013).
  https://doi.org/10.1109/TIP.2013.2261309
- **VV** — Vonikakis, V., Kouskouridas, R., Gasteratos, A. "On the evaluation of
  illumination compensation algorithms." *Multimedia Tools and Applications* 77(8),
  9211–9231 (2018). https://doi.org/10.1007/s11042-017-4783-x
  Dataset: https://sites.google.com/site/vonikakis/datasets

**Real-world human-MOS benchmark (Sec. 4.9 / Table 3)**

- **RLIE** — Li, C., Hu, B., Chen, T., Li, L., He, L., Gao, X. "Low-Light Image
  Enhancement Quality Assessment: A Real-World Dataset and an Objective Method."
  *ACM MM* (2025). https://doi.org/10.1145/3746027.3758296
  Repository: https://github.com/CQUPT-HuBo90/RLIE
  The RLIE human-opinion label files (`bt_scores.csv`, `normalized_scores.csv`) belong
  to that dataset and are **not** redistributed here; download them from the RLIE repo.

## Fast Reproduction From Included CSVs

These commands use the CSVs already in `results/`.

| Step | Command | Produces |
|---|---|---|
| 1. Main paper figures and main table numbers | `python -m analysis.make_figures` | `results/figures/fig2_correlation_matrix.{pdf,svg}`, `fig3_main_alignment.{pdf,svg}`, `fig4_generalization.{pdf,svg}`, `fig4b_within_method.{pdf,svg}`, `fig5_rule_heatmap.{pdf,svg}`, `results/figures_numbers.csv`, `results/rule_table_mono.csv` |
| 2. Supplementary membership and disagreement figures | `python -m analysis.make_extra_figures` | `results/figures/figS_membership_functions.{pdf,svg}` plus supplementary diagnostic figures |
| 3. Qualitative cases | `python -m analysis.make_qualitative_figure` | `results/figures/fig6_qualitative.{pdf,svg}` |
| 4. Hand-tuned fuzzy vs perceptual anchors | `python -m analysis.validate_against_perceptual` | `results/perceptual_validation_summary.csv`, `results/figures/alignment_with_lpips.png`, `results/figures/imagelevel_correlation_matrix.png` |
| 5. Reference-based calibrated fuzzy baseline | `python -m analysis.optimize_fuzzy` | `results/perceptual_optimization_summary.csv`, `results/optimized_rule_table_{free,mono}.csv`, `results/figures/optimized_alignment.png` |
| 6. Main reference-free calibrated fuzzy model | `python -m analysis.reference_free_fuzzy` | `results/reference_free_summary.csv`, `results/nr_fuzzy_scores.csv`, `results/nr_rule_table_{free,mono}.csv`, `results/figures/reference_free_alignment.png` |
| 7. Significance tests | `python -m analysis.significance_tests` | `results/significance_tests.csv` |
| 8. Leakage, learned-fusion, algorithm-disjoint, seed-stability checks | `python -m analysis.reviewer_experiments` | `results/reviewer_experiments.csv` |
| 9. Broadened calibration pool stress test | `python -m analysis.tier3_broaden` | `results/tier3_broaden.csv` |

## Human-Opinion Validation

These require the external RLIE dataset path.

| Step | Command | Produces |
|---|---|---|
| RLIE human-MOS validation | `python -m analysis.rlie_human_mos --rlie_root /path/to/RLIE` | `results/rlie_features.csv`, `results/rlie_human_mos.csv` |
| Enriched 3-feature vs 6-feature test | `python -m analysis.rlie_enriched --rlie_root /path/to/RLIE` | `results/rlie_features6.csv`, `results/enriched_feature_test.csv` |

## Runtime And Real-Time Pipeline

| Step | Command | Produces |
|---|---|---|
| Assessor runtime sweep | `python -m analysis.benchmark_runtime --sweep` | `results/runtime_results.csv` |
| Runtime sweep plus deep NR-IQA baselines | `python -m analysis.benchmark_runtime --sweep --deep` | `results/runtime_results.csv`, `results/runtime_deep.csv` |
| Synthetic realtime sanity check | `python -m analysis.realtime_pipeline_demo --source synth --enhancer gamma --assessor real --frames 600 --plot` | `results/pipeline_log.csv`, `results/figures/fig6_pipeline_fps.{png,pdf,svg}`, `results/figures/fig6_pipeline_score.{png,pdf,svg}` |
| Video + Zero-DCE realtime demo | `python -m analysis.realtime_pipeline_demo --source path/to/video.mp4 --enhancer zerodce --device cuda --assessor real --seconds 20 --plot` | same realtime log and Fig. 6 pipeline figures |
| Summarize realtime numbers | `python -m analysis.realtime_analysis` | prints latency, frame-budget occupancy, resolution sweep, sustained FPS, and per-stage medians from `results/runtime_results.csv` and `results/pipeline_log.csv` |

`--assessor real` uses `analysis.reference_free_fuzzy.score_features`, the
calibrated 27-rule reference-free scorer. `--assessor placeholder` is included
only for timing sanity checks when the calibrated scorer is unavailable; its
scores should not be reported.

For the Zero-DCE realtime demo, place the pretrained weights here:

```text
models/zerodce_Epoch99.pth
```

The weights are not redistributed in this repository.

## Rebuilding Input CSVs From Raw Images

This path is slower and requires the original image datasets.

| Output CSV | Command |
|---|---|
| `results/nr_features.csv` | `python -m evaluation.nr_features` |
| `results/perceptual_metrics.csv` | `python -m evaluation.perceptual_metrics` |
| `results/modern_nriqa.csv` | `python -m evaluation.modern_nriqa` |
| `results/deep_zerodce_all.csv` | `python -m evaluation.deep_zerodce` |
| `results/deep_sci_all.csv` | `python -m evaluation.deep_sci` |
| `results/cross_dataset.csv` | `python -m evaluation.cross_dataset` |

The extraction scripts expect the raw/preprocessed images in the paths described
in each script header. Deep enhancer scripts also require their corresponding
pretrained weights under `models/`.

## Included Results

Important included CSVs:

- `results/enhancement_metrics.csv`
- `results/perceptual_metrics.csv`
- `results/modern_nriqa.csv`
- `results/nr_features.csv`
- `results/fuzzy_enhancement_results.csv`
- `results/nr_fuzzy_scores.csv`
- `results/reference_free_summary.csv`
- `results/significance_tests.csv`
- `results/reviewer_experiments.csv`
- `results/runtime_results.csv`
- `results/pipeline_log.csv`

Important included figures:

- `results/figures/fig2_correlation_matrix.pdf`
- `results/figures/fig3_main_alignment.pdf`
- `results/figures/fig4_generalization.pdf`
- `results/figures/fig4b_within_method.pdf`
- `results/figures/fig5_rule_heatmap.pdf`
- `results/figures/fig6_qualitative.pdf`
- `results/figures/fig6_pipeline_fps.pdf`
- `results/figures/fig6_pipeline_score.pdf`
- `results/figures/figS_membership_functions.pdf`

## Citation

The citation will be updated after acceptance. For now, please cite the paper
title and repository if you use the code or computed tables.
