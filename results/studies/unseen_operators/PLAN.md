# Exploratory unseen-enhancement protocol

Recorded on 28 October 2026. The initial plan preceded feature extraction, LPIPS evaluation and the SSR reproduction check. The four-classical analysis and brightness/noise comparisons were added after inspection of the initial five-classical results and are explicitly post-hoc. Original records are retained privately; their hashes are in `protocol_sources.json`.

## Inputs and fixed analysis

The same 485 LOL calibration scenes are used, with the existing bilinear 256 x 256 RGB inputs. Enhancement follows resizing; outputs are clipped to [0,1] as float32. Features use `evaluation.nr_features.nr_features`; LPIPS-Alex 0.1 uses the unchanged `to_tensor` adapter, eval mode, batch 1 and no additional crop or resize. This assesses unseen enhancement methods on known scenes, not unseen scenes or human quality.

The initial five-method pool contains CLAHE, gamma, log, SSR-50 and histogram equalisation (2,425 rows). The later four-method pool excludes SSR-50 (1,940 rows). Feature signs and q20/q40/q60/q80 membership breakpoints are fitted within each pool. Both use the existing minimum t-norm, 27-rule objective and differential evolution with population factor 8, 40 generations, seed 42, tolerance 0.001, no polish, one worker and deferred updating. Numerical fitting uses one BLAS thread. The separately frozen released NR-free model is also scored; its results match the five-method fit in this experiment.

The SSR-50 check compares 485 recreated feature rows to the original rows (absolute tolerance 1e-12) and LPIPS to the original targets (1e-6). Four-method calibration must reproduce the archived SSR-50 correlation. Missing, nonfinite or mismatched rows stop the run; no scene is dropped and tolerances are not adjusted.

## Fixed enhancement settings

| Method | Settings |
|---|---|
| SSR-15 / SSR-80 / SSR-250 | Existing RGB log-surround implementation; sigma 15, 80 or 250; epsilon 1e-8; one global min-max normalisation |
| MSR | Equal average of the three log-surround responses before the same global normalisation |
| MSRCR | Pinned upstream defaults: sigma [15,80,250], alpha 125, beta 46, G 5, b 25, low/high clips 0.01/0.99; upstream uint8 processing, then float32 /255 |
| AGCWD | HSV V channel; 256-bin histogram, weighting alpha 0.5, cumulative weighted histogram and gamma LUT; uint8 truncation; preserve H/S; identity for a uniform histogram and keep zero intensity at zero |
| Zero-DCE++ | Official Epoch99 checkpoint, scale factor 1, eight curve iterations; CPU float32 without cropping |
| RUAS-MIT | Official MIT5K checkpoint, final enhanced output, unchanged network settings; CPU float32; LOL checkpoint excluded |

AGCWD follows Huang et al., DOI [10.1109/TIP.2012.2226047](https://doi.org/10.1109/TIP.2012.2226047). MSRCR follows the pinned implementation of the multiscale Retinex colour-restoration method; its display settings are implementation defaults, not a unique reconstruction of the original paper.

## Summaries and uncertainty

For each method, compute across-scene SRCC with negative LPIPS, mean LPIPS and the percentage outside the five-method training feature ranges. The single-feature directions are entropy, contrast and negative Laplacian energy, fixed from the original five-method calibration. The post-hoc brightness and negative-noise columns retain their stated directions, without test-dependent reversal. These two features are never added to the fuzzy model.

Intervals use 2,000 bootstrap draws of 485 scenes with replacement, `numpy.random.default_rng(42)`, the same indices for each method. Ranks are recomputed for each draw; constant predictions contribute zero. Percentile endpoints are 2.5% and 97.5%; fitted parameters remain fixed. The four-method results reuse the existing features, LPIPS targets and bootstrap draws; no enhancement was rerun for that analysis. Historical held-out rows in `results.csv` are labelled separately and receive no new intervals.

## Skipped methods and execution environment

LIME was skipped because the official MATLAB archive download returned a Google Drive sign-in page; no approximate replacement was implemented. EnlightenGAN was excluded because its reported training mixture contains LOL images. The source evidence is the [LIME project](https://sites.google.com/view/xjguo/lime) and [EnlightenGAN paper, Section IV-A](https://arxiv.org/html/1906.06972#S4.SS1).

Recorded versions: NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, OpenCV 4.13.0.92, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118 and pyiqa 0.1.15.post2. LPIPS and deep enhancement run on CPU with four PyTorch threads; fitting uses one BLAS thread. Seeds are fixed at 42.
