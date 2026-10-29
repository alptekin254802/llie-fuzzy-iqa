# Technical notes
Full supplementary protocols and implementation details. This archive follows the S-section order of the supplement immediately after the live-timing and exploratory-camera integration. Table and figure identifiers below are stable source labels; current numbering is supplied by the supplement. Numeric macros were expanded automatically from its frozen TeX files.
The original-release archive is [Zenodo](https://doi.org/10.5281/zenodo.20938906); the revision-specific material is maintained in this repository.

## Contents

| Supplement | Topics |
|---|---|
| S1 | [Implementation and preprocessing](#s1-implementation-and-preprocessing) |
| S2 | [Unpaired images and expanded enhancement pools](#s2-unpaired-images-and-expanded-enhancement-pools) |
| S3 | [Monotonicity and rule support](#s3-monotonicity-and-rule-support) |
| S4 | [Objective anchors and within-method evaluation](#s4-objective-anchors-and-within-method-evaluation) |
| S5 | [Scene-level uncertainty](#s5-scene-level-uncertainty) |
| S6 | [Capacity and calibration-target controls](#s6-capacity-and-calibration-target-controls) |
| S7 | [Timing and video demonstration](#s7-timing-and-video-demonstration) |
| S8 | [Additional metrics and their directions](#s8-additional-metrics-and-their-directions) |
| S9 | [Differential-evolution sensitivity](#s9-differential-evolution-sensitivity) |
| S10 | [Regional features and Retinex diagnosis](#s10-regional-features-and-retinex-diagnosis); [Exploratory unseen-enhancement check](#exploratory-unseen-enhancement-check) |
| S11 | [Native-resolution processing](#s11-native-resolution-processing) |
| S12 | [Synthetic closed-loop selection and live timing](#s12-synthetic-closed-loop-selection-and-live-timing); [Exploratory real-camera check](#exploratory-real-camera-check); [Complete exploratory camera protocol](#complete-exploratory-camera-protocol); [Earlier live timing sessions](#earlier-live-timing-sessions) |
| S13 | [Human transfer and target preferences](#s13-human-transfer-and-target-preferences); [Source-overlap record correction](#source-overlap-record-correction); [Targeted calibration-source overlap control](#targeted-calibration-source-overlap-control) |
| S14 | [Human recalibration and RLIE correction](#s14-human-recalibration-and-rlie-correction); [Training-method coverage control](#training-method-coverage-control) |
| S15 | [Additional learned metrics](#s15-additional-learned-metrics); [Q-Align](#q-align); [RichIQA](#richiqa); [Scope recorded on 19 October 2026 (historical)](#scope-recorded-on-19-october-2026-historical); [VQA² and VITAL: multimodal evaluation protocol](#multimodal-evaluation-protocol) |

## S1. Implementation and preprocessing


<a id="s1-p01"></a>

The main LOL analysis comprises 485 scenes and five classical enhancements per scene. Bilinear resizing (OpenCV `INTER_LINEAR`) produces paired $`256\times256`$ LOL inputs before enhancement. Human and unpaired external datasets use area interpolation; the separate preprocessing-sensitivity and timing controls specify their own interpolation. For float RGB outputs, features use `clip(255*RGB,0,255).astype(uint8)`, i.e. truncation rather than rounding to nearest. OpenCV converts the resulting RGB image to grayscale using BT.601 coefficients. Entropy uses 256 histogram bins and logarithm base two. Contrast is the population standard deviation divided by 255. Laplacian energy is the population variance of `cv2.Laplacian(gray, CV_64F)` divided by $`255^2`$. The default aperture argument is `ksize=1`: its effective kernel is the $`3\times3`$ cross stencil
``` math
\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix},
```
with `BORDER_REFLECT_101`. This is not the different kernel obtained by explicitly setting `ksize=3`. It is also not the replicated-border operation used by the separate noise feature in the six-feature analysis.

<a id="s1-p02"></a>

CLAHE operates on the Lab lightness channel, with clip limit 2 and an $`8\times8`$ tile grid. Gamma correction uses exponent $`1/1.5=2/3`$. The logarithmic operator applies $`\log(1+I)`$ and divides by its image maximum plus $`10^{-8}`$. SSR applies a Gaussian with standard deviation 50 pixels to each RGB channel, subtracts logarithms with $`10^{-8}`$ stabilization, then shifts and scales the resulting image by its minimum and maximum. Histogram equalization operates separately on the three RGB channels. Outputs are clipped to $`[0,1]`$.

<a id="s1-p03"></a>

The hand-tuned FR comparator takes PSNR, SSIM and entropy. Its low/medium/high trapezoids are respectively $`(0,0,15,25)`$, $`(15,25,30,40)`$, $`(30,40,50,50)`$ for PSNR; $`(0,0,.3,.5)`$, $`(.3,.5,.7,.85)`$, $`(.7,.85,1,1)`$ for SSIM; and $`(0,0,2,4)`$, $`(2,4,5,6.5)`$, $`(5,6.5,8,8)`$ for entropy. The four output sets use $`(0,0,20,40)`$, $`(20,40,50,65)`$, $`(50,65,75,90)`$ and $`(75,90,100,100)`$ on a 1,000-point universe. Rules use minimum activation, maximum aggregation and centroid defuzzification, with a fixed fallback for uncovered combinations. These descriptive sets are not validated human-quality categories. The separate FR-calibrated comparator uses PSNR, SSIM and entropy with the quantile memberships below, minimum firing and 27 constant consequents optimized against negative LPIPS by the same DE settings as NR-free. Its orientations and breakpoints are estimated globally, and consequents are fitted within the scene-training folds.

<a id="s1-p04"></a>

For an oriented feature, let $`a=q_{20}`$, $`b=q_{40}`$, $`c=q_{60}`$ and $`d=q_{80}`$. The low membership is one below $`a`$, decreases linearly to zero on $`[a,b]`$, and remains zero. The medium membership rises on $`[a,b]`$, equals one on $`[b,c]`$, and decreases on $`[c,d]`$. The high membership rises on $`[c,d]`$ and remains one above $`d`$. Thus the three memberships form a partition of unity, including outside the observed feature range. The stored minimum and maximum delimit the plotted range but do not truncate the outer shoulders.

<a id="s1-p05"></a>

**Figure caption (supfig:mf).** Deployment-model memberships. The oriented Laplacian-energy coordinate is the negative of the measured energy. Its concentrated transitions reflect a skewed feature distribution; membership values do not constitute probability estimates or a guarantee of equal sample counts.

<a id="s1-p06"></a>

**Table caption (sup:breakpoints).** Deployment-model transition breakpoints; the main paper gives the same values. Electronic model files retain full precision.

<a id="s1-p07"></a>

**Table caption (sup:nested).** Preprocessing estimated globally versus entirely within each training fold. Rule consequents are always training-fold fitted. Values are held-out LPIPS SRCC. The small difference is a sensitivity finding, not evidence that global supervised orientation is leakage-free.

<a id="s1-p08"></a>

The following preprocessing control uses area interpolation and re-extracts features and recalibrates the model for each variation. Its own reference run is 0.757, slightly different from the primary 0.753 analysis; deltas must be interpreted relative to this control’s reference, not substituted into the main table. Changes in histogram resolution and luminance convention are smaller than those from aperture or image size.

<a id="s1-p09"></a>

**Table caption (sup:pretable).** Preprocessing-control results. The “default Laplacian” entry uses the cross stencil defined above.

## S2. Unpaired images and expanded enhancement pools


<a id="s2-p01"></a>

The BIMEF collection provides 128 source scenes across DICM, LIME, MEF, NPE and VV. Each receives the five classical enhancements, giving 640 outputs. The all-LOL NR-free model is applied without fitting to these images. No reference or human opinion is available here; the correlations below are agreements between objective metrics. The datasets and their source references are identified in the main paper’s data statement.

<a id="s2-p02"></a>

**Table caption (sup:cross).** Unpaired transfer, using the same all-LOL deployment parameters. Counts are enhanced outputs, not independent scenes.

<a id="s2-p03"></a>

The operator-withholding analyses retain the same 485 source scenes across training and test, but exclude the test operator from calibration. They test transfer to an unseen operator, not simultaneous independence of scenes and operators. The deep-operator test has two different roles. Zero-DCE and SCI can be evaluated as new inputs to the five-operator model; their correlations across scenes are modest. In the expanded-pool experiment, a new model is calibrated on all seven methods, giving mean LPIPS SRCC 0.722 (PSNR 0.678; SSIM 0.708). This refitting does not demonstrate unchanged-model transfer.

<a id="s2-p04"></a>

**Table caption (sup:disjoint).** Withheld-operator SRCC. For a classical operator, fitting excludes that operator; for Zero-DCE and SCI, the classical calibration pool is used. Each test contains 485 scenes.

<a id="s2-p05"></a>

**Table caption (sup:broaden).** Selected withheld-operator controls with a broader calibration pool. Other operators were not evaluated in this particular comparison. More methods do not consistently rescue transfer.

## S3. Monotonicity and rule support


<a id="s3-p01"></a>

For product firing, $`w_{ijk}=\mu_i(x)\mu_j(y)\mu_k(z)`$ and $`\sum_{ijk}w_{ijk}=1`$. Fix $`y,z`$, and define $`c_i=\sum_{jk}\mu_j(y)\mu_k(z)\theta_{ijk}`$. The adjacent inequalities $`\theta_{i+1,j,k}\geq\theta_{i,j,k}`$ imply $`c_{i+1}\geq c_i`$, since the multiplying weights are nonnegative. On a transition interval in $`x`$, only neighbouring memberships vary, so
``` math
Q=(1-t)c_i+t c_{i+1},\qquad 0\leq t\leq1.
```
This is nondecreasing in $`x`$; on a plateau or an outer shoulder it is constant. The argument applies to each axis. The numerical denominator stabilization in the product implementation is a constant factor because the unmodified weights sum to one; it does not change this conclusion.

<a id="s3-p02"></a>

The 54 inequalities consist of two adjacent comparisons for each of the nine combinations of the other coordinates, repeated over three axes. SLSQP minimizes squared error to negative LPIPS subject to these linear inequalities. This differs from adding a finite penalty to a rank objective: a finite penalty can still allow violations. Normalized minimum firing also has an input-dependent denominator, so ordered consequents alone do not establish this proof.

<a id="s3-p03"></a>

An ordered minimum-inference control, despite satisfying its consequent constraints, has decreasing adjacent responses on a $`31^3`$ grid: 26 along entropy and 595 along contrast, with a maximum drop of 0.0317. NR-mono has no decreases above $`10^{-9}`$ tolerance; the largest numerical decrease is $`6.8\times10^{-14}`$. This grid check is a numerical diagnostic. The analytical argument, rather than the finite grid, supplies the monotonicity guarantee.

<a id="s3-p04"></a>

The canonical single-thread NR-mono analysis uses Python 3.11, NumPy 2.4.4 and SciPy 1.17.1. Fold mean and SD are 0.751 and 0.020; within-scene SRCC is 0.632. Its training-fold membership fit and deployment fit are distinct. Rank calculations use full-precision predictions: near-equal predictions can change their ordering after rounding. Rounding predictions to nine decimal places changes the fold mean to 0.752, which is reported only as a numerical sensitivity check.

<a id="s3-p05"></a>

**Figure caption (supfig:free).** Unconstrained NR-free consequents on their actual $`0`$–$`100`$ scale. These values need not be ordered. Stars identify rules without calibration activation. The main-paper NR-mono heatmap, in contrast, uses an affine display scale.

<a id="s3-p06"></a>

Rules 19 and 21 (one-based lexicographic order) have zero activation in the LOL calibration sample. These correspond to high entropy, low contrast, and low or high oriented energy. Their unconstrained values are not empirically identified by that sample. Constraints can make their values orderly without supplying missing perceptual evidence. The rule-support perturbation analysis changes SQUARE-LOL within-scene correlation by less than 0.002; that local sensitivity does not justify extrapolation to other image distributions. The complete deployment parameters are in the repository files `models/nr_free.json` and `models/nr_mono.json`; the same JSON files accompany the source package. Use each model with its own inference operator.

## S4. Objective anchors and within-method evaluation


<a id="s4-p01"></a>

**Table caption (sup:allanchors).** All original objective-anchor correlations, including BRISQUE. Negative distances are oriented higher-is-better. Values are fold mean $`\pm`$ SD. Model names identify the individual features, hand-tuned reference-dependent fusion, calibrated FR fusion and the two NR variants.

<a id="s4-p02"></a>

**Figure caption (supfig:correlation).** Descriptive correlation matrix for the classical enhancement pool. NR-free uses the all-data deployment fit here, not held-out predictions; inferential comparisons use the held-out tables. The hand-tuned baseline’s strong association with SSIM does not establish human validity.

<a id="s4-p03"></a>

**Table caption (sup:within).** Correlation across scenes within each classical method, using held-out NR-free predictions. These values replace an in-sample within-method calculation. They must not be confused with within-scene ranking of different methods.

<a id="s4-p04"></a>

**Figure caption (supfig:within).** Within-method agreement. The five classical operators use scene-held-out predictions with the operator represented in training. Zero-DCE and SCI are unseen operators scored by the all-classical deployment fit. This is different from withholding a classical operator during calibration.

## S5. Scene-level uncertainty


<a id="s5-p01"></a>

For a paired contrast, each bootstrap sample draws scenes with replacement and keeps all enhanced versions and both scorers’ predictions. Fold-mean statistics retain the original fold identity; pooled statistics concatenate OOF predictions. The reported difference is NR-free minus comparator. Percentile intervals use 2,000 draws and seed 20261018. The null-centered two-sided $`p`$-value uses a plus-one finite-sample correction, so its smallest attainable value is $`1/2001\approx0.0005`$, not an extremely small parametric tail probability. Holm correction is applied separately to the eight contrasts for each target/statistic. Intervals below are marginal, not simultaneous intervals. Predictions are not refitted within bootstrap samples.

<a id="s5-p02"></a>

**Table caption (sup:bootstrap).** Paired NR-free-minus-comparator differences in SRCC for the negative LPIPS and NIQE targets. Fold-mean and pooled out-of-fold (OOF) statistics are distinct. All original estimates and 95% intervals are retained; $`p_H`$ denotes the Holm-adjusted $`p`$-value.

<a id="s5-p03"></a>

The training-method-mean baseline is fitted only on training scenes. Its fold-mean, pooled and within-scene correlations are 0.565, 0.531 and 0.808. NR-free’s corresponding values are 0.753, 0.730 and 0.634. The NR-free advantage in fold mean is 0.188 (95% CI $`[0.154,0.214]`$); its within-scene disadvantage is $`-0.174`$ ($`[-0.201,-0.149]`$). The two findings are compatible: ranking images across scenes and choosing among methods within a scene are different tasks.

## S6. Capacity and calibration-target controls


<a id="s6-p01"></a>

The same three input features are compared under several response functions. Least-squares fuzzy inference changes the consequent objective while retaining the membership representation. The additive membership model removes three-way rule interactions. Raw, clipped and quantile-transformed linear models test simpler mappings, and monotone decision stumps supply a different piecewise model. The LOL column below belongs to this control’s training-only preprocessing protocol, not the primary global-preprocessing analysis.

<a id="s6-p02"></a>

**Table caption (sup:capacitytable).** LOL proxy fit and zero-shot within-scene human agreement. All human columns use the corresponding all-LOL fit without human labels.

<a id="s6-p03"></a>

The SQUARE-LOL difference between raw linear and NR-free is $`-0.086`$ (95% CI $`[-0.107,-0.065]`$), whereas on RLIE it is $`+0.144`$ ($`[0.095,0.191]`$). Least-squares fuzzy, additive membership and monotone additive models are close to NR-free on SQUARE-LOL. These controls associate transfer behaviour with the response representation but do not establish a unique causal mechanism or a universal fuzzy advantage.

<a id="s6-p04"></a>

A 36-dimensional BRISQUE-feature SVR trained against LOL negative LPIPS obtains fold-mean SRCC 0.885 (SD 0.019), above all three-feature models. Its human transfer is nevertheless weak (Table `sup:proxybrisque`). High proxy fit is therefore not sufficient for transfer to human rankings. This is a newly LPIPS-trained feature regressor, distinct from the released BRISQUE metric used as a comparator.

<a id="s6-p05"></a>

**Table caption (sup:proxybrisque).** Human transfer of the LOL-LPIPS-trained BRISQUE-feature SVR. No human labels enter these fits.

## S7. Timing and video demonstration


<a id="s7-p01"></a>

The controlled benchmark uses an AMD Ryzen 7 1700, Python 3.11 and OpenCV, with one thread for OpenCV, PyTorch and numerical libraries. The Windows high-performance profile is held fixed. There are three separate sessions with ten warm-up calls. Each ordinary operation has 100 timed calls per session over a 50-image set (300 calls pooled); native feature scoring and each deep model have 50 calls per session (150 pooled). Session medians and pooled 95th percentiles characterize repeated latency under these conditions. They are not worst-case execution-time bounds. PNG and JPEG reading use a warm filesystem cache. The source images are controlled resamplings for the resolution timing study; the separate native-quality study uses the actual native images.

<a id="s7-p02"></a>

**Table caption (sup:extratime).** Summary of single-image timings. The controlled rows pool three CPU sessions; memory to score includes area interpolation to 256 square when needed, and PNG to score also includes reading. Native rows do not resize. Additional metric rows use the separate protocols below. Full session medians and the resolution scan remain in the repository’s `results/timing_benchmark.json`.

<a id="s7-p03"></a>

BPRI/BMPRI use MATLAB R2025b, 50 images and three warm-ups; StairIQA CPU and GTX 1060 results are measured separately. RichIQA uses 100 fixed images in each of three CPU sessions and all 7,865 images on the GTX 1060. Its times include PNG decoding and the official preprocessing/scoring call. Q-Align uses 5,440 in-memory area-interpolated inputs on the GTX 1060; file reading and the external resize are excluded. These separate measurements do not share every benchmark condition. Both learned-metric runs synchronize CUDA, exclude warm-ups and retain all subsequent calls, including slow calls. Section `sup:learned` gives implementation details.

<a id="s7-p04"></a>

The demonstration processes a prerecorded video with Zero-DCE on the GTX 1060 and CPU feature extraction/scoring. It retains the original demo’s bilinear resize to $`256\times256`$, consistent with the primary LOL pipeline; the separate human-data and external-data evaluations use area interpolation. The recorded run uses the AMD Ryzen Balanced power profile and is separate from the controlled CPU benchmark. After one warm-up frame, 1,199 timed frames give about 78 frames/s, computed as frame count divided by summed read/prepare–enhance–assess time; display and output writing are excluded. This result should not be interchanged with older runs from different sessions. The corrected scores use the same units as calibration; earlier mixed normalization compressed the score distribution. No human labels were collected for this clip. The correlation between frame changes and score changes is about 0.26, which does not validate quality tracking.

## S8. Additional metrics and their directions


<a id="s8-p01"></a>

BPRI and BMPRI use their released MATLAB models without LLIE retraining. StairIQA uses the mixed-database ResNet-50 checkpoint with the KonIQ head, resize to 384, five $`320\times320`$ crops, ImageNet normalization and averaging over crops. MATLAB receives 8-bit RGB; floating-point enhancement results are quantized by truncation to match the main feature pipeline. Main-paper tables report their LOL-LPIPS and human-ranking results, including negative results.

<a id="s8-p02"></a>

The direction check uses the ten lexicographically first normal-exposure LOL references, resized with area interpolation. Each clean image receives five levels of Gaussian noise ($`\sigma=5,10,20,30,50`$ in 8-bit units), blur ($`\sigma=0.5,1,2,3,5`$ pixels), and JPEG compression (qualities 90, 70, 50, 30, 10). One standard-normal field per scene is scaled for the noise levels; values are rounded to the nearest 8-bit intensity. Including the clean baseline gives 160 images and 320 metric evaluations. Spearman correlation is computed between distortion severity and raw score within each scene, then averaged.

<a id="s8-p03"></a>

**Table caption (sup:direction).** Mean raw-score correlation with increasing degradation severity. Every scene has a positive correlation and a larger endpoint score for both metrics and all three distortion types.

<a id="s8-p04"></a>

These results support lower-is-better orientation for both metrics. They do not show that either metric is monotone for every change: BPRI can respond in the opposite direction at the mildest noise level. The check establishes score direction, not validity for all LLIE artefacts.

<a id="s8-p05"></a>

RichIQA and the representative LMM evaluator Q-Align are evaluated without LLIE fitting in Section `sup:learned`. Other instruction/reasoning methods are discussed with model-specific scope and access information; no superiority over untested models is claimed.

## S9. Differential-evolution sensitivity


<a id="s9-p01"></a>

The sensitivity plan fixes population factors 8, 15 and 30, maximum generations 40, 100 and 300, and seeds 0, 1 and 2. Each of the 27 combinations uses five scene-disjoint folds with training-only preprocessing. “Population factor” multiplies the 27-dimensional consequent space; it is not the population size itself. The original 8/40 setting was selected as a practical computational budget. The present comparison asks whether its proxy agreement is unusually dependent on that choice; it is not used to choose a best-performing replacement on the test data.

<a id="s9-p02"></a>

**Table caption (sup:detable).** Population/generation summaries over seeds 0, 1 and 2. Min, median and max are computed from the full-precision results of the three runs in each row. All 27 configurations remain in `results/studies/de_sensitivity.json`.

<a id="s9-p03"></a>

Across these runs, LOL fold means range from 0.754 to 0.768, a spread of 0.014, below the plan’s 0.02 operational threshold. SQUARE-LOL within-scene transfer ranges from 0.346 to 0.402. Thus the proxy result is relatively stable in this grid, while the human result varies more. This does not prove convergence or global optimality. Least-squares calibration also obtains similar proxy agreement, as shown in Section `sup:capacity`.

<a id="s9-p04"></a>

The previous finite monotonicity penalty used weights zero and two. Zero recovered the unconstrained objective; two was a heuristic compromise that could still violate rule order. It is not retained as a justified optimum. The revised NR-mono formulation instead enforces 54 hard constraints and uses product firing. This change removes the need for a penalty-weight sensitivity claim.

## S10. Regional features and Retinex diagnosis


<a id="s10-p01"></a>

The plan defined three comparable three-input models before results were inspected. The global baseline uses entropy, contrast and Laplacian energy. The tile variant divides the image into a $`4\times4`$ grid and uses the 10th percentile of tile entropy, median tile contrast and 90th percentile of tile Laplacian energy. The intensity-region variant retains global entropy and contrast, replacing global energy by the variance of Laplacian responses, divided by $`255^2`$, over pixels with normalized grayscale intensity $`Y/255`$ at most $`2/3`$. Dark ($`<1/3`$), middle ($`1/3`$–$`2/3`$) and bright ($`>2/3`$) descriptors use this normalized intensity and are recorded separately; an empty masked region falls back to global energy. The tested variant uses the dark-plus-middle mask, not a feature search over these results.

<a id="s10-p02"></a>

The improvement criterion was an SSR correlation gain of at least 0.10 *or* a positive lower confidence bound for SQUARE-LOL within-scene improvement, together with a LOL fold-mean loss below 0.02. This is an operational criterion, not a claim of general human improvement. Training-only preprocessing is refitted in every fold.

<a id="s10-p03"></a>

**Table caption (sup:regionaltable).** Regional-feature controls. LOL is fold-mean LPIPS SRCC, SSR is operator-held-out SRCC, and human columns are mean within-scene SRCC.

<a id="s10-p04"></a>

The intensity-region variant meets the SSR branch of the criterion: gain 0.109, with a descriptive follow-up interval $`[0.064,0.157]`$. Its SQUARE-LOL change is $`-0.014`$, interval $`[-0.028,0.001]`$. The tile variant lowers both LOL and human performance. Regional information partly addresses this particular SSR failure, but the remaining SSR correlation is negative and a human-ranking benefit is not established. The additional intervals are descriptive follow-up; they are not a new independent confirmation.

<a id="s10-p05"></a>

When SSR is withheld, only 2.7% of its images exceed the other four operators’ observed range in any feature. Nonetheless, the model substantially overrates SSR relative to its negative-LPIPS target and misranks SSR outputs across scenes. This is compatible with conditional distribution shift within overlapping feature ranges. Grayscale global statistics cannot distinguish all colour, local-contrast and halo configurations that produce the same triplet.

<a id="s10-p06"></a>

Diagnostic comparisons use mean Lab chroma displacement, pixelwise chroma distance, brightness changes, edge overshoot/undershoot and high-pass mismatch to the reference. SSR shows large chroma and edge changes. Associations with LPIPS identify plausible missing cues, but are not controlled interventions separating colour shift, halos and other artefacts. We therefore do not state that any one of them causes the LPIPS penalty or that the metric is necessarily wrong. The two cited stereoscopic scene-component methods motivate localization; this test is an independent 2D feature control, not a claimed reimplementation of either stereoscopic method.

### Exploratory unseen-enhancement check

An exploratory analysis added during revision evaluated eight additional enhancement variants and methods on the same 485 LOL scenes used for calibration. The initial plan compared a five-classical-method fit and the frozen released NR-free model; their results matched. Excluding SSR-50 to form a four-classical-method fit was added after inspecting these results, as were brightness and negative-noise comparisons. These analyses are marked post-hoc in the result columns and protocol. No new images or LPIPS targets were computed for the four-method analysis. The experiment tests unseen methods on known scenes, not unseen scenes.

The [protocol and records](../results/studies/unseen_operators/README.md) retain settings, per-scene predictions, confidence intervals and source hashes. LIME was not evaluated because its official MATLAB code requires a sign-in to download. EnlightenGAN was excluded because its reported training mixture included LOL images. Upstream Retinex, Zero-DCE++ and RUAS code and weights are obtained separately under their respective terms; download locations and pinned hashes are documented. The reproduction command checks the archived values without rerunning enhancement or calibration.

## S11. Native-resolution processing


<a id="s11-p01"></a>

This comparison changes the complete pipeline: enhancement, feature extraction and LPIPS are all calculated at native LOL resolution, then compared with a complete $`256\times256`$ pipeline. Training-only preprocessing and calibration are repeated within folds. Human transfer uses the corresponding all-LOL fit and computes features at the matching resolution. It does not mix native features with a resized LPIPS target. These controls have their own fold fits and must not replace the primary main-paper result.

<a id="s11-p02"></a>

**Table caption (sup:nativetable).** Native and resized pipelines. Human columns report within-scene SRCC. Native LOL inputs are $`600\times400`$.

<a id="s11-p03"></a>

For NR-free, the native-minus-resized difference in LOL within-scene agreement is $`-0.081`$ (95% scene-bootstrap CI $`[-0.110,-0.053]`$). This interval concerns LOL and LPIPS, not a human-opinion dataset. Human agreement decreases for NR-free in all three datasets. NR-mono also declines on SQUARE-LOL and RLIE but improves on LIEQ, from a slightly negative correlation to a small positive one.

<a id="s11-p04"></a>

These results give no advantage to native processing for the tested primary scorer. They do not show that downsampling is information-preserving: small halos, fine noise and local detail can disappear at $`256\times256`$. The comparison evaluates whether the present three-feature assessor makes productive use of that additional native information, not whether another model or a native-resolution human viewing experiment would do so.

## S12. Synthetic closed-loop selection and live timing


<a id="s12-p01"></a>

The plan selects 20 scenes from each of the five LOL folds without replacement (random seed 20261018), giving 100 scenes and 400 frames. Each frame starts from the $`256\times256`$ low-light image. Exposure multipliers $`[1,0.25,0.5,0.125]`$ are applied in linearized sRGB and converted back to clipped sRGB; the normal-light reference is unchanged. Every frame has nine gamma exponents from 0.2 to 1.0 in increments of 0.1. The default is $`2/3`$, evaluated separately. For each evaluation fold, the constant gamma minimizes mean LPIPS over the other 80 selected scenes; all folds choose 0.3. NR-free and NR-mono are calibrated on the original enhancement outputs from all 388 scenes in the other four LOL folds. Thus both procedures exclude the test fold, but their training sets and objectives differ. Entropy, brightness, NIQE and pretrained CLIP-IQA+ select the same candidates without test references. The brightness heuristic prefers a mean grayscale intensity close to 0.45. A reference-based LPIPS oracle provides a comparator that cannot be deployed without references.

<a id="s12-p02"></a>

The primary tie rule regards scores within $`10^{-9}`$ of the maximum as tied, preferring continuity with the preceding gamma, then proximity to the default, then the lower gamma. Uniform selection among tied maxima is reported separately. Evaluation first aggregates the four frames within a scene, then uses scene-bootstrap contrasts; frames from the same scene are not treated as independent observations. These sequences contain exposure changes but no motion or temporal human-quality labels.

<a id="s12-p03"></a>

**Table caption (sup:looptable).** Closed-loop quality, historical-run latency and tie diagnostics. Lower LPIPS and higher PSNR/SSIM are better; PSNR is in dB. Times sum generation and scoring over nine candidates, with fixed settings avoiding search. Uniform LPIPS averages eligible tied outputs; frames counts ties. The two NR-free-minus-comparator contrasts use scene-bootstrap 95% intervals and unadjusted two-sided $`p`$-values; negative differences favour NR-free.

<a id="s12-p04"></a>

The search experiment uses the Ryzen 7 1700 CPU for generation, features and fuzzy scoring, with one numerical thread, and a GTX 1060 for CLIP-IQA+, NIQE and reference LPIPS. GPU timings are synchronized and include the measured input transfer. Consequently, the CLIP-IQA+ search entry is not comparable to its single-score CPU latency in Section `sup:timing`. Each per-frame timing is a sum of separately timed candidate-generation and selector components, not a wall-clock measurement of the full controller. Image reading, synthetic exposure construction and selection bookkeeping are excluded. Entropy and brightness share the measured feature block rather than optimized stand-alone implementations. These historical timings belong to one search campaign, separate from the three-session CPU benchmark and the live measurement below.

<a id="s12-p05"></a>

NR-free and NR-mono improve substantially over the default but do not beat the training-selected fixed gamma. Brightness and entropy have better PSNR and SSIM. This target dependence is expected to matter when a selector has been calibrated to LPIPS. Holding out scenes makes the LPIPS test a legitimate generalization test, but it remains evidence about LPIPS, not an independent human preference test. Single-score latency, the historical candidate-component sums and the live processing measurement below have different timing boundaries. The synthetic timing repeat retained floating-point inputs and the original float64 exponentiation. In the same-session, single-thread comparison on the same 400 synthetic frames, the median sum of candidate-generation and scoring components decreased from 82.25 ms to 68.00 ms without changing features, scores or gamma selections. These repeat timings are separate from the historical run in Table `sup:looptable`.

<a id="s12-p06"></a>

Windows PnP identifies the webcam as USB Video Device (VID 0C45, PID 6301), using Microsoft `usbvideo.sys` (`usbvideo.inf`, version 10.0.19041.6033). DirectShow exposure uses nominal $`\log_2`$ seconds; setting $`-5`$ corresponds to about $`1/32`$ s. The view was a static desk scene under room illumination.

<a id="s12-p07"></a>

For the live measurement, the camera supplied 8-bit $`640\times480`$ frames through DirectShow with exposure, gain, white balance and focus locked. Lookup tables preserved the original reciprocal exponent, clipping and float32-to-RGB8 conversion. At the start of each of five separate sessions, a 100-frame bit-equality selftest compared candidate RGB8 images, all three features, both NR scores and selected gammas against the original floating-point path; every selftest passed. Processing time includes bilinear resizing to $`256\times256`$, generation and scoring of nine candidates, selection and full-resolution LUT correction, but excludes capture waiting and display; display was disabled. The three sessions at exposure $`-5`$ each contained 2,000 frames, with the first 50 excluded as warmup, leaving 5,850 frames; the extra sessions at $`-3`$ and $`-4`$ each retained 450 of 500 frames after the same warmup. All numerical libraries and OpenCV used one thread on the Ryzen 7 1700 CPU.

<a id="s12-p08"></a>

In the earlier sessions at exposure $`-5`$, the camera delivered frames at a mean interval of 39.97 ms and the loop ran at 25.02 fps, despite the requested 30 fps setting. Processing had a median of 9.26 ms and never exceeded 19.13 ms, within the nominal 33.3 ms period of a 30 fps camera.

<a id="s12-p09"></a>

Three additional sessions used DirectShow, MJPG, $`640\times480`$, exposure $`-6`$ and the same locked controls. MJPG was confirmed by readback in every session. Each session passed the 100-frame bit-equality selftest and retained 1,950 of 2,000 frames after the fixed 50-frame warmup. Processing excluded capture and display and used the same nine-candidate kernel. Across 5,850 retained frames, processing had a median of 8.58 ms, p95 of 8.77 ms, p99 of 9.32 ms and maximum of 22.53 ms. The camera interval had a median of 32.00 ms and mean of 33.84 ms; 4.75% of intervals exceeded 48 ms. The measured loop reached 29.55 fps with zero misses under the same-frame host-interval definition. All processing times were below the nominal 33.3 ms budget. Camera intervals are host capture-completion differences, not sensor timestamps.

<a id="s12-p10"></a>

Applying MJPG after the size and frame-rate requests prevented the observed YUY2 fallback; in the separate capture-only control at exposure $`-6`$, the mean delivery interval decreased from about 39.95 to 33.80 ms. The earlier YUY2 format was inferred from a later reproduction of the original setting order, rather than recorded during those sessions. Exposure also changed in the new formal sessions, so their difference cannot be attributed solely to the format.

<a id="s12-p11"></a>

**Table caption (sup:liveTiming).** Live processing latency and camera delivery. Capture waiting and display are excluded from processing; $`n`$ excludes 50 warmup frames per session. Miss counts processing times exceeding the same-frame host interval. Asterisk: earlier YUY2 format inferred by reproduction. n/a: interval-tail summaries unavailable for the two extra sessions.

<a id="s12-p12"></a>

#### Exploratory real-camera check.

With the same webcam, locked controls and uncompressed YUY2 frames, each of 21 scenes in one room was captured at a reference exposure (mean of 16 frames before and after) and at 4 shorter exposures, 5 frames each; scenes failing fixed quality checks were rejected, and one duplicate view was removed before analysis. Candidates, selectors and tie rules match Table `sup:looptable`; values are scene means with 2,000-draw scene-bootstrap intervals, without hypothesis tests (Table `sup:camloop`). These frames were less dark than the synthetic sequences (mean luminance 0.32 to 0.10); NR-free was within 0.006 LPIPS of the default gamma and 0.046 below the fixed gamma, and entropy and brightness rules gave similar LPIPS. The check covers one camera and one room.

<a id="s12-p13"></a>

**Table caption (sup:camloop).** Exploratory real-camera selection: scene means with descriptive 95% scene-bootstrap intervals in brackets. Lower LPIPS and higher PSNR/SSIM are better; PSNR is in dB. Uniform LPIPS averages tied outputs and Frames counts ties, as in Table `sup:looptable`. n/a: selector latency was not measured in this check.

### Complete exploratory camera protocol

The final exploratory amendment supersedes the original confirmatory hypothesis and the earlier transport/exposure plan. The check uses no hypothesis tests or p-values. All dry runs are excluded. Two consecutive records showed the same view; the first had an incorrect room label and was excluded before analysis, retaining the later correctly labelled record. The exclusion decision and its ledger hash remain in the private audit record; raw metadata and rejected records remain unchanged and private. The resulting analysis contains 21 distinct scenes from one room. Images and the original room/direction labels are not published.

The [original plan](../realtime/camera_exploratory/PLAN.json), [Amendment 1](../realtime/camera_exploratory/PLAN_AMENDMENT_1.json) and [Amendment 3](../realtime/camera_exploratory/PLAN_AMENDMENT_3.json) record acquisition, locked controls, exposure and quality checks, candidate arithmetic, selectors and scoring; Amendment 3 defines the final exploratory design.

### Earlier live timing sessions

These earlier sessions are retained for the measurement history. Processing excludes capture waiting and display; warmup frames are excluded. The earlier YUY2 format was inferred by reproduction (asterisk). Interval-tail summaries were unavailable for the extra sessions (n/a).

| Session / exposure | n | Median processing (ms) | p95 | p99 | Max. | Median interval (ms) | Mean interval (ms) | >48 ms (%) | Miss (%) | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| Old 1 / $-5$ | 1,950 | 9.29 | 9.84 | 11.26 | 19.13 | 32.40 | 39.97 | 25.64 | 0.00 | 25.02  |
| Old 2 / $-5$ | 1,950 | 9.26 | 10.13 | 11.81 | 17.94 | 32.38 | 39.97 | 18.36 | 0.00 | 25.02  |
| Old 3 / $-5$ | 1,950 | 9.23 | 9.69 | 10.43 | 18.57 | 32.32 | 39.97 | 18.51 | 0.00 | 25.02  |
| Extra 1 / $-3$ | 450 | 9.52 | 10.66 | 11.28 | 18.95 | 128.01 | 130.34 | n/a | 0.00 | 7.67  |
| Extra 2 / $-4$ | 450 | 9.43 | 10.17 | 11.32 | 15.79 | 64.01 | 65.24 | n/a | 0.00 | 15.33  |
| Old pooled / $-5$* | 5,850 | 9.26 | 9.92 | 11.37 | 19.13 | 32.37 | 39.97 | 20.84 | 0.00 | 25.02  |

## S13. Human transfer and target preferences

### Source-overlap record correction

Normal-light matching classifies all 290 rated SQUARE-LOL scenes as 231 shared and 59 non-overlapping, including low00710 among the shared scenes. The [record correction and validation history](../results/studies/normal_light_overlap_classification/README.md#source-overlap-record-correction) preserve the earlier low-light records, the failed complete-agreement check and the image evidence for reclassification; fixed predictions were regrouped without fitting or inference.

<a id="s13-p01"></a>

All primary human-transfer scores use the same all-LOL deployment model and area interpolation. Across five calibration seeds, LIEQ within-scene SRCC ranged from $`-0.016`$ to $`0.095`$ (seed 42: $`0.095`$). SQUARE-LOL, LIEQ and RLIE contain ten alternatives per scene. Human scores are evaluated as provided by each dataset. Within-scene rank agreement is the principal enhancement-selection outcome; pooled SRCC is supplementary because between-scene differences can be large. RLIE has no paired normal-light reference; the released reference folder contains the low-light inputs. Scores computed against a low-light input would not be an FR assessment of fidelity to the desired normal-light image, so LPIPS and DISTS entries are marked n/a for RLIE.

<a id="s13-p02"></a>

**Table caption (sup:humanpooled).** Pooled zero-shot human SRCC for the principal metric comparators. These values complement, rather than replace, the within-scene results in the main paper. FR metrics are omitted for RLIE.

<a id="s13-p03"></a>

**Table caption (sup:humandetail).** Fuzzy-model transfer without human calibration. Top-1 averages uniformly over tied maxima; chance is the scene mean of the fraction of human-score maxima. Pairwise accuracy gives tied predictions half credit and excludes tied target pairs.

<a id="s13-p04"></a>

**Table caption (sup:overlap).** SQUARE-LOL source-scene overlap sensitivity. Sharing source content with LOL is disclosed even though the human-rated enhancement outputs and human labels were not calibration inputs. Added LPIPS-Alex and Q-Align rows use stored scores; RF per-image scores for this training protocol were unavailable.

<a id="s13-p05"></a>

**Table caption (sup:humanci).** Within-scene human-correlation differences, comparator minus NR-free, with paired scene-bootstrap 95% intervals. Positive differences favour the comparator. The pretrained-metric analysis uses 2,000 draws, seed 7; added BPRI/BMPRI/StairIQA comparisons use seed 20261018. These are marginal descriptive intervals.

<a id="s13-p06"></a>

The target itself can favour different methods from human observers. On SQUARE-LOL, LPIPS-Alex chooses the method labelled Semi in the dataset on 267 of 290 scenes, while human opinion scores choose it on 25. The LPIPS-Alex/LPIPS-VGG ordering also differs between SQUARE-LOL and LIEQ. These observations motivate direct human calibration and explicit target checks, rather than interpreting all neural distances as interchangeable perceptual truth.

<a id="s13-p07"></a>

Only 0.24%, 0.10% and 1.69% of SQUARE-LOL, LIEQ and RLIE images, respectively, lie outside the LOL range in at least one of the three features. Limited transfer therefore cannot be diagnosed merely from whether a feature exceeds its calibration minimum or maximum. The mapping from similar feature values to human preference can differ between domains.

## S14. Human recalibration and RLIE correction


<a id="s14-p01"></a>

Human recalibration uses five scene-disjoint folds and fits orientations, membership breakpoints and model parameters on training scenes only. The pooled-objective fuzzy model optimizes global training human scores rank correlation. The within-scene DE model instead optimizes the mean of scene-wise rank correlations. Its training implementation uses ordinal ranks to break ties; the reported evaluation uses average ranks, as elsewhere in this paper. This difference is part of the training objective, not permission to break evaluation ties arbitrarily.

<a id="s14-p02"></a>

The least-squares fuzzy control demeans the design and target within training scenes and adds ridge regularization $`0.001`$. Linear and BRISQUE-feature SVR controls also use scene-demeaned features for the within-scene objective. At inference, those two controls use the candidate set to calculate the scene mean; they are comparative ranking procedures, not independent single-image scorers. The RBF-SVR uses $`C=10`$ and the scale-based kernel parameter after standardization. The global BRISQUE-SVR fits undemeaned features and human scores. Consequently, pooled correlations of the within-scene models need not be high and do not measure the intended training objective.

<a id="s14-p03"></a>

**Table caption (sup:recal).** Human-recalibrated results. All values are held out by scene. These are not zero-shot results and are not mixed into the main zero-shot table. Method-mean is trained on human labels: not zero-shot; requires known methods.

<a id="s14-p04"></a>

To test operator dependence, the additional analysis withholds both scene content and groups of enhancement methods. Ten random five-versus-five partitions of the ten enhancement methods (seed 1) are crossed with five scene folds. Fitting uses training scenes and five training methods; evaluation uses held-out scenes and the other five methods. Table `sup:humanmethods` reports mean and SD over these 50 estimates. This control keeps the LOL-fitted feature orientations and membership functions fixed, fitting only the human-target model parameters on each training partition; it therefore differs from the scene-only recalibration above. A strong same-dataset scene-disjoint result can still fall substantially when enhancement methods are also new. More human labels can improve a target-specific fit, but do not eliminate method-distribution dependence.

<a id="s14-p05"></a>

**Table caption (sup:humanmethods).** Human evaluation with both scenes and methods withheld. The LOL-prior row uses no human labels. Within-scene correlation is computed over the unseen alternatives in each split; it is not directly interchangeable with the ten-method scene-disjoint average above.

<a id="s14-p06"></a>

The corrected RLIE three-feature comparison uses $`256\times256`$ area interpolation throughout. Zero-shot NR-free has pooled SRCC 0.106; human-score-calibrated fuzzy inference gives 0.249; the six-feature RBF-SVR gives 0.353. The six features are entropy, contrast, Laplacian energy, mean brightness, colourfulness and a simple noise estimate. Its gain belongs to a learned SVR fusion, not to fuzzy inference. The previous native-resolution values (0.058, 0.267 and 0.391) should not be mixed with the corrected protocol. Published IACA and MIIHDP RLIE results are supervised results under their own protocols; they do not form a matched comparison with a zero-shot LOL model.

<a id="s14-p07"></a>

The electronic `nr_free.json` and `nr_mono.json` files accompanying this supplement contain the complete full-precision deployment parameters. The manuscript uses rounded numerical displays only; the computed rankings and table values do not use rounded model coefficients.

## S15. Additional learned metrics


<a id="s15-p01"></a>

Q-Align and RichIQA use released weights without LLIE training, label calibration or checkpoint selection. Human inputs use the same RGB area-interpolated $`256\times256`$ images as the other metrics. SQUARE-LOL, LIEQ and RLIE contain 2900/1000/1540 images from 290/100/154 scenes. RichIQA additionally scores the 2,425 LOL outputs, giving 7,865 images. The LOL outputs retain the original bilinear-preprocessed enhancement pipeline; their subsequent area interpolation to the same 256-square size is an identity operation.

<a id="s15-p02"></a>

#### Q-Align.

We use `pyiqa` Q-Align with fp16 weights from `q-future/one-align`, revision `dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb` (Wu et al., ICML 2024; <https://proceedings.mlr.press/v235/wu24ah.html>). Its documented training mix is KonIQ, SPAQ, KADID, AVA and LSVQ. The standard quality prompt and probability-weighted score are retained; higher is better. The internal CLIP processor resizes/center-crops to $`448\times448`$ and normalizes the input. Image-level scores of the 4-bit and fp16 models agree closely (SRCC 0.992 to 0.997); tables report fp16 accuracy and 4-bit timing.

<a id="s15-p03"></a>

Q-Align uses batch size one, PyTorch 2.7.1+cu118, pyiqa 0.1.15.post2, Transformers 4.36.1, Accelerate 0.26.1 and bitsandbytes 0.49.2. Accuracy uses fp16 weights with CPU offloading; 4-bit timing keeps all parameters on CUDA. Timing includes the internal processor and scorer, plus observed Windows WDDM memory management. Concurrent desktop GPU activity occurred during part of RLIE scoring; those calls are retained.

<a id="s15-p04"></a>

#### RichIQA.

The unchanged official `inference.py` is from <https://github.com/YixuanGao98/RichIQA>, commit `bfcf34939937e8fd2a55179982d648c4dfb29c37`; the newly shared weights are from <https://huggingface.co/GYX98/RichIQA>, revision `af8ff09037533c265c8a57d981319b549b253a4b`. The required CvT-21-384 backbone comes from the README’s OneDrive model zoo. Both weight files have recorded SHA-256 hashes and the full checkpoint is loaded strictly. Internal preprocessing uses PIL bilinear resize to $`224\times224`$ and ImageNet normalization. The mean over five ordered MOS categories, 1–5, is higher-is-better, as established from the code and training scale before testing; the README describes mean quality but does not explicitly state direction.

<a id="s15-p05"></a>

RichIQA uses float32, batch one, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, timm 1.0.27 and Pillow 12.2.0. Torch, OpenCV and BLAS use one CPU thread. CPU timing passes the loaded CPU module to the same prediction function. The first 25 filenames per dataset form the 100-image CPU set; three warm-ups per dataset precede each of three sessions, with separate GPU warm-ups. Complete prediction-call timing includes PNG decoding, internal preprocessing, transfer, prediction and scalar conversion; external preparation and model loading are excluded. CPU/GPU scores differ by at most 0.00000142. Table `sup:extratime` gives CPU/GPU timing summaries.

<a id="s15-p06"></a>

**Table caption (sup:qalignsummary).** Additional learned-metric agreement. LOL targets negative LPIPS; human datasets target opinion scores. Within is mean within-scene SRCC, pooled is image-level SRCC, and pairs is within-scene pair accuracy. The LOL fold mean uses the main comparison’s five held-out scene folds.

<a id="s15-p07"></a>

Q-Align has higher within-scene correlation and pair accuracy than NR-free on all three human datasets, but lower pooled SQUARE-LOL correlation. RichIQA’s LOL fold-mean advantage and its dataset-dependent human transfer are reported alongside all contrasts below.

<a id="s15-p08"></a>

**Table caption (sup:qaligncontrasts).** Learned metric minus NR-free; positive values favour the learned metric. This direction is opposite to the NR-free-minus-comparator LOL contrasts in Section `sup:statistics`.

<a id="s15-p09"></a>

Both analyses use 2,000 whole-scene bootstrap draws, percentile 95% intervals and two-sided null-centered $`p`$-values with a plus-one correction, unadjusted for these additional comparisons. Seeds are 20261018 for Q-Align and 20261019 for RichIQA. LOL uses NR-free OOF predictions and resamples scenes within each fixed fold for the fold-mean contrast; human comparisons use the frozen deployment model. Fits remain fixed. Average ranks are used, constant predictions receive zero scene SRCC, target-tied pairs are skipped, and prediction ties receive half credit. Q-Align has 178 exact prediction ties among opinion-distinct pairs.

<a id="s15-p10"></a>

#### Scope recorded on 19 October 2026 (historical).

MINT-IQA assesses generated-image quality, authenticity and prompt correspondence. Its public inference expects a generation prompt, absent from these LLIE datasets, and `MINTIQA.pt`; the linked Hugging Face bundle lists backbone/pretraining files but not that task checkpoint. An advertised Baidu alternative was not retrieved ([MINT-IQA repository](https://github.com/IntMeGroup/MINT-IQA)). VQA$`^2`$ and VITAL support image and video scoring, with public code and checkpoint links ([VQA$`^2`$ repository](https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment); [VITAL repository](https://github.com/jzhws/VITAL-Series)). Within the computational budget, Q-Align supplies a representative scalar image-quality comparison rather than an exhaustive LMM benchmark. At the access checks on 19 October 2026, OmniVQA had only a short README and license ([OmniVQA repository](https://github.com/jzhws/Omni-VQA)); Refine-IQA released training code but listed testing code and S1/S2 weights as forthcoming ([Refine-IQA repository](https://github.com/jzhws/VisualQualityAssessment-RL-Trainer)). No quantitative claim is made about untested models.

The paragraph above records the earlier access check. The completed comparison additionally evaluates VQA² and VITAL; the final source-availability check is dated 23 October 2026 and is linked below.

<a id="multimodal-evaluation-protocol"></a>

### VQA² and VITAL: multimodal evaluation protocol

Model sources, official prompts, scoring, environments, precision diagnostics, pilot acceptance, timing and dated source checks are recorded in the [VQA² and VITAL protocol](../results/lmm_extra/README.md#multimodal-evaluation-protocol).

## Targeted calibration-source overlap control

The follow-up uses the verified normal-light source classification to remove the 231 shared LOL scenes before all orientation, membership and consequent estimation. The remaining 254 scenes provide 1,270 outputs from the same five enhancement methods. The positive control refits all 485 scenes and reproduces within-scene SRCC 0.380726 on all 290 SQUARE-LOL scenes and 0.390968 on its 59 non-overlapping scenes. Human scores are used only for evaluation. Existing area-interpolated feature records are reused; no image features or reference metrics are recomputed.

NR-free uses entropy, contrast and Laplacian energy, 27 minimum-t-norm singleton rules and raw negative LPIPS as target. Differential evolution maximises pooled training SRCC with population multiplier 8, 40 generations, seed 42, bounds [0,100], tolerance 0.001, no polish, deferred updating and one worker. Removing scenes precedes feature orientation and quantile membership estimation. The frozen parameters are applied once to all 2,900 test images; the 290-scene and 59-scene summaries use that same prediction vector. Within-scene SRCC is the equally weighted scene mean; constant predictions contribute zero, and constant human targets are omitted. Pooled SRCC uses all images in each test subset.

For the random control, NumPy default_rng with seeds 1 through 20 samples 231 scene identifiers without replacement from the sorted 485 LOL scene identifiers. Sampling does not use overlap labels; the DE seed remains 42 for every fit. Each fit retains 254 scenes. Reported random ranges are the observed minimum and maximum, not bootstrap intervals. Removing the shared scenes gives correlations within all four random-control ranges. This comparison describes sensitivity to calibration-scene selection and does not establish absence of every form of overfitting.

The [reproduction guide](../results/studies/targeted_overlap/README.md), archived numeric results and portable script document this control. The public script combines the recorded targeted and random-removal workflows; fitting, row order, numeric parsing and scoring are retained, while local paths and private audit bookkeeping are removed. Complete reproduction is checked against the archived numeric values and subset hashes, with runtimes excluded.

## Training-method coverage control

Supplement S14 additionally holds four test methods fixed within each of 20 method splits while increasing a nested training set from two to six methods, together with their images. Five stored scene folds keep training and test scenes separate. Frozen LOL orientations and membership functions, ridge-0.001 least-squares fuzzy fitting and scene-centred three-feature linear regression match the S14 controls. The unchanged LOL prior uses no human scores. A preparation check reproduces the original S14 mean and SD values to three decimals.

The primary contrast is fuzzy LS at six minus two training methods, separately for each dataset. Secondary contrasts compare each fitted model with the LOL prior at each coverage level. Two thousand paired scene-bootstrap draws retain all 20 observations of each scene, keep the original folds equally weighted, and hold model fits and method splits fixed. Percentile intervals are marginal and unadjusted. Method diversity and image count increase together. [Protocol, numerical summaries and reproduction](../results/studies/method_coverage/README.md) include all planned outcomes; the test has four alternatives per scene, unlike the five-method S14 comparison.
